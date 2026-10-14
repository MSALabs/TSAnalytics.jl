# ---------------------------------------------------------------------------
# Stage 11.4 -- threshold (SETAR) and smooth-transition (LSTAR)
# autoregression. handoff/stage 11/docs/stage-11.4-tar-star-handoff.md
#
# SETAR (Tong 1978; Tong & Lim 1980): conditional least squares, one
# dummy-expanded regression per candidate threshold, the minimum SSR wins.
# LSTAR (Teräsvirta 1994): the regime coefficients profiled out by least
# squares, (γ, c) found from a start grid and then BFGS. Hansen's (1999)
# bootstrap test of linearity. Reference R's tsDyn (GPL -- conventions
# re-derived from measured behaviour, no code transcribed); the SETAR point
# estimator also agrees with Python's aeon and with an independent base-R
# grid search, bit-exactly.
# ---------------------------------------------------------------------------

export fit_setar, SETARModel, select_setar, fit_lstar, LSTARModel, setar_test, ThresholdTest, tsdyn_ic

# --- the embedding ------------------------------------------------------------

# rows t = m+1..n; threshold variable z_t = y_{t-(d+1)} (thDelay d is
# zero-indexed on the lag, as tsDyn)
function _tar_embed(y::AbstractVector, m::Int, d::Int)
    n = length(y)
    rows = (m + 1):n
    Y = y[rows]
    lag(j) = y[rows .- j]
    z = lag(d + 1)
    return Y, lag, z, rows
end

# regime design: deterministic terms then the regime's lags
function _tar_regime_cols(include::Symbol, lags::AbstractVector{Int}, lag, nr::Int)
    cols = Vector{Float64}[]
    include in (:const, :both) && push!(cols, ones(nr))
    include in (:trend, :both) && push!(cols, collect(1.0:nr))
    for j in lags
        push!(cols, lag(j))
    end
    return isempty(cols) ? zeros(nr, 0) : reduce(hcat, cols)
end

_tar_detnames(include::Symbol) = include === :const ? ["const"] : include === :trend ? ["trend"] :
                                 include === :both ? ["const", "trend"] : String[]

# --- the candidate grid --------------------------------------------------------
# All sorted values of z INCLUDING TIES (tsDyn's makeTh; sort(unique(z))
# gives a different grid on tied data), trimmed at ceil(trim n) .. floor(n(1-trim)).
# grid = :full takes every in-range value; grid = :tsdyn reproduces tsDyn's
# seq(down, up, length.out = up - down) with its fractional indices truncated
# -- which silently skips values (one on log10(lynx)), so it is an option for
# reproducing tsDyn's selection tables, not the default.
function _tar_grid(z::AbstractVector, trim::Real; grid::Symbol=:full)
    0 < trim < 0.5 || throw(ArgumentError("trim must be in (0, 0.5), got $trim"))
    allTh = sort(z)
    ng = length(allTh)
    down = ceil(Int, trim * ng)
    up = floor(Int, ng * (1 - trim))
    up > down || throw(ArgumentError("too few observations for trim = $trim"))
    if grid === :full
        return unique(allTh[down:up])
    elseif grid === :tsdyn
        L = up - down
        idx = L == 1 ? [down] : [floor(Int, down + i * (up - down) / (L - 1) + 1e-9) for i in 0:(L - 1)]
        return unique(allTh[idx])
    end
    throw(ArgumentError("grid must be :full or :tsdyn, got :$grid"))
end

# --- SETAR -----------------------------------------------------------------------

"""
    SETARModel

A fitted SETAR model, from [`fit_setar`](@ref): `thresholds` (full
precision), `coef`/`coefnames` (`const.L, phiL.1, …, const.H, …`, with an
`M` regime for two thresholds), `ssr`, `nobs` (regression rows), standard
errors under the package's convention (`se`, residual variance
`SSR / (rows − k)`) and under `tsDyn`'s (`se_tsdyn`, `SSR / (n − p)`), the
SSR `profile` over the candidate thresholds, and the regime of each row.
"""
struct SETARModel <: TimeSeriesModel
    y::Vector{Float64}
    m::Int
    thDelay::Int
    thresholds::Vector{Float64}
    searched::Bool
    include::Symbol
    lags::Vector{Vector{Int}}
    coef::Vector{Float64}
    names::Vector{String}
    ssr::Float64
    nobs::Int
    se::Vector{Float64}
    se_tsdyn::Vector{Float64}
    regime::Vector{Int}
    fitted::Vector{Float64}
    resid::Vector{Float64}
    profile::Vector{Tuple{Float64,Float64}}   # (candidate, SSR)
end

# fit at given thresholds: the dummy-expanded least squares
function _setar_fit_at(y, m, d, ths::Vector{Float64}, include, lagsets)
    Y, lag, z, rows = _tar_embed(y, m, d)
    nr = length(Y)
    reg = ones(Int, nr)
    for (r, th) in enumerate(sort(ths))
        reg[z .> th] .= r + 1                          # low regime is z <= th, inclusive
    end
    blocks = Matrix{Float64}[]
    for r in eachindex(lagsets)
        Xr = _tar_regime_cols(include, lagsets[r], lag, nr)
        push!(blocks, Xr .* (reg .== r))
    end
    X = reduce(hcat, blocks)
    any(c -> all(iszero, c), eachcol(X)) && return nothing
    F = qr(X)
    b = F \ Y
    e = Y .- X * b
    return (; b, e, ssr=dot(e, e), X, reg, nr)
end

function _setar_names(include, lagsets)
    labs = length(lagsets) == 2 ? ["L", "H"] : ["L", "M", "H"]
    nm = String[]
    for (r, l) in enumerate(lagsets)
        for dn in _tar_detnames(include)
            push!(nm, "$dn.$(labs[r])")
        end
        for j in l
            push!(nm, "phi$(labs[r]).$j")
        end
    end
    return nm
end

# the SSR at every candidate. Sorted once by z; the cross-products of the two
# regimes are carried incrementally, each crossing row moved from the high
# block to the low, so a candidate costs O(k^2) plus two small Cholesky solves
# rather than a fresh factorisation. The chosen threshold is refitted from
# scratch, so the reported fit carries none of the accumulated rounding.
function _setar_profile(y, m, d, cands, include, lagsets; fixed=nothing)
    Y, lag, z, rows = _tar_embed(y, m, d)
    nr = length(Y)
    XL = _tar_regime_cols(include, lagsets[1], lag, nr)
    XH = _tar_regime_cols(include, lagsets[end], lag, nr)
    if fixed !== nothing || length(lagsets) == 3
        # two-threshold conditional searches: a direct refit per candidate
        return [(c, (f = _setar_fit_at(y, m, d, vcat(fixed, c), include, lagsets);
                     f === nothing ? Inf : f.ssr)) for c in cands]
    end
    o = sortperm(z)
    kL, kH = size(XL, 2), size(XH, 2)
    AL = zeros(kL, kL); bL = zeros(kL)
    AH = XH' * XH; bH = XH' * Y
    yy = dot(Y, Y)
    out = Tuple{Float64,Float64}[]
    pos = 0
    for c in sort(cands)
        while pos < nr && z[o[pos + 1]] <= c
            pos += 1
            i = o[pos]
            xl = view(XL, i, :); xh = view(XH, i, :)
            BLAS.ger!(1.0, collect(xl), collect(xl), AL); bL .+= xl .* Y[i]
            BLAS.ger!(-1.0, collect(xh), collect(xh), AH); bH .-= xh .* Y[i]
        end
        CL = cholesky(Symmetric(AL); check=false)
        CH = cholesky(Symmetric(AH); check=false)
        if !(issuccess(CL) && issuccess(CH))
            push!(out, (c, Inf))
            continue
        end
        ssr = yy - dot(bL, CL \ bL) - dot(bH, CH \ bH)
        push!(out, (c, ssr))
    end
    return out
end

"""
    fit_setar(y, m; thDelay=0, nthresh=1, threshold=nothing, trim=0.15,
              include=:const, mL=m, mH=m, mM=m, ML=nothing, MH=nothing,
              MM=nothing, grid=:full, max_iter=2) -> SETARModel

Self-exciting threshold autoregression, by conditional least squares:
```
y_t = c_L + Σ φ_{L,j} y_{t−j} + ε_t   if z_t ≤ γ
      c_H + Σ φ_{H,j} y_{t−j} + ε_t   if z_t >  γ,     z_t = y_{t−(thDelay+1)}
```
The low regime is `z ≤ γ`, inclusive. For each candidate threshold one
dummy-expanded regression is fitted and the smallest sum of squares wins;
the candidates are the sorted values of `z`, **ties included**, trimmed at
`trim` (0.15). `nthresh = 2` adds a middle regime by **sequential** search
(the first threshold, then the second conditional on it, then the first
again -- `max_iter` alternations), not a joint grid.

`thDelay` counts from zero, as R's `tsDyn`: `thDelay = 0` uses `y_{t−1}`.
`mL`, `mH`, `mM` give each regime its own order; `ML`, `MH`, `MM` give it an
explicit (possibly non-contiguous) set of lags. `include` is `:const`
(one intercept per regime), `:none`, `:trend` or `:both`.

`threshold` fits at given threshold(s) instead of searching -- pass it at
full precision: a threshold typed back from a 12-digit print can move a
boundary observation into the other regime and change the fit.

**Standard errors**: `se` uses the residual variance `SSR / (rows − k)`.
`tsDyn` divides by the full series length less the slope count,
`SSR / (n − p)` -- including observations the regression never used -- and
reports `mse` and AIC on `n` as well; those are kept in `se_tsdyn` (and
`tsdyn_ic`) so its numbers can be reproduced exactly. **The threshold has no
standard error**: its asymptotic distribution is non-standard (Hansen
2000), and a Gaussian one would mislead. `tsDyn` zero-pads it.

`grid = :tsdyn` reproduces `tsDyn`'s candidate grid exactly, including
its fractional-index truncation, which skips values -- useful only to
reproduce its selection tables. `tsDyn` uses `trim` defaults of 0.15
(`setar`), 0.1 (`setarTest`) and 0.1 (`lstar`'s start grid); this package
uses 0.15 throughout.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = log10.(readdlm(TSAnalytics.LYNX, ','; skipstart=1)[:, 2]);

julia> m = fit_setar(y, 2; thDelay=1, warn=false);

julia> m.thresholds
1-element Vector{Float64}:
 3.3100557377508912

julia> round(m.ssr, digits=8)
4.34819128

julia> [count(==(r), m.regime) for r in 1:2]
2-element Vector{Int64}:
 78
 34
```
"""
function fit_setar(y, m::Integer; thDelay::Integer=0, nthresh::Integer=1, threshold=nothing,
                   trim::Real=0.15, include::Symbol=:const, mL::Integer=m, mH::Integer=m,
                   mM::Integer=m, ML=nothing, MH=nothing, MM=nothing, grid::Symbol=:full,
                   max_iter::Integer=2, warn::Bool=true)
    yv = Vector{Float64}(tsvalues(y))
    any(isnan, yv) && throw(ArgumentError("fit_setar: y contains NaN"))
    m >= 1 || throw(ArgumentError("fit_setar: m must be >= 1"))
    0 <= thDelay < m || throw(ArgumentError(
        "fit_setar: thDelay must satisfy 0 <= thDelay < m (= $m); thDelay = d uses y_{t-(d+1)}"))
    nthresh in (1, 2) || throw(ArgumentError("fit_setar: nthresh must be 1 or 2"))
    include in (:const, :none, :trend, :both) || throw(ArgumentError(
        "fit_setar: include must be :const, :none, :trend or :both"))
    lagL = ML === nothing ? collect(1:mL) : collect(Int, ML)
    lagH = MH === nothing ? collect(1:mH) : collect(Int, MH)
    lagM = MM === nothing ? collect(1:mM) : collect(Int, MM)
    for l in (lagL, lagH, lagM)
        all(1 .<= l .<= m) || throw(ArgumentError("fit_setar: every lag must be in 1:m (= 1:$m)"))
    end
    lagsets = nthresh == 1 ? [lagL, lagH] : [lagL, lagM, lagH]
    Y, lag, z, rows = _tar_embed(yv, m, Int(thDelay))
    cands = _tar_grid(z, trim; grid=grid)
    profile = Tuple{Float64,Float64}[]
    if threshold !== nothing
        ths = sort(threshold isa Real ? [Float64(threshold)] : Float64.(collect(threshold)))
        length(ths) == nthresh || throw(ArgumentError(
            "fit_setar: give $nthresh threshold(s) for nthresh = $nthresh"))
        searched = false
    elseif nthresh == 1
        profile = _setar_profile(yv, m, Int(thDelay), cands, include, lagsets)
        ths = [profile[argmin(last.(profile))][1]]
        searched = true
    else
        # sequential: the best single threshold, then the second conditional
        # on it, then the first again conditional on the second
        p1 = _setar_profile(yv, m, Int(thDelay), cands, include, [lagL, lagH])
        th1 = p1[argmin(last.(p1))][1]
        nmin = ceil(Int, trim * length(z))
        okpair(a, b) = (lo = min(a, b); hi = max(a, b);
                        count(<=(lo), z) >= nmin && count(>(hi), z) >= nmin &&
                        count(x -> lo < x <= hi, z) >= nmin)
        th2 = NaN
        for _ in 1:max_iter
            c2 = [c for c in cands if c != th1 && okpair(th1, c)]
            isempty(c2) && throw(ArgumentError("fit_setar: no admissible second threshold"))
            p2 = _setar_profile(yv, m, Int(thDelay), c2, include, lagsets; fixed=th1)
            th2new = p2[argmin(last.(p2))][1]
            c1 = [c for c in cands if c != th2new && okpair(th2new, c)]
            p1b = _setar_profile(yv, m, Int(thDelay), c1, include, lagsets; fixed=th2new)
            th1new = p1b[argmin(last.(p1b))][1]
            done = th1new == th1 && th2new == th2
            th1, th2 = th1new, th2new
            profile = p1b
            done && break
        end
        ths = sort([th1, th2])
        searched = true
    end
    f = _setar_fit_at(yv, m, Int(thDelay), ths, include, lagsets)
    f === nothing && throw(ArgumentError(
        "fit_setar: a regime is empty at threshold(s) $(ths) -- too few observations per regime"))
    k = length(f.b)
    n = length(yv)
    XtXi = inv(Symmetric(f.X' * f.X))
    se = sqrt.(f.ssr / (f.nr - k) .* diag(XtXi))
    se_t = sqrt.(f.ssr / (n - k) .* diag(XtXi))
    names = _setar_names(include, lagsets)
    model = SETARModel(yv, Int(m), Int(thDelay), ths, searched, include, lagsets, f.b, names,
                       f.ssr, f.nr, se, se_t, f.reg, Y .- f.e, f.e, profile)
    warn && _setar_root_warning(model)
    return model
end

# the regimes' characteristic roots; a root on or inside the unit circle
# means that regime, on its own, is not stationary (tsDyn warns likewise)
function _setar_regime_roots(m::SETARModel)
    labs = length(m.lags) == 2 ? ["low", "high"] : ["low", "middle", "high"]
    out = Dict{String,Vector{Float64}}()
    for (r, l) in enumerate(m.lags)
        ϕ = zeros(maximum(l; init=0))
        lab = length(m.lags) == 2 ? ["L", "H"][r] : ["L", "M", "H"][r]
        for j in l
            ϕ[j] = m.coef[findfirst(==("phi$lab.$j"), m.names)]
        end
        isempty(ϕ) && continue
        # roots of 1 - ϕ1 z - … - ϕp z^p = inverse companion eigenvalues
        ev = eigvals(_companion([fill(c, 1, 1) for c in ϕ]))
        out[labs[r]] = sort(1 ./ abs.(ev))
    end
    return out
end

function _setar_root_warning(m::SETARModel)
    for (lab, rts) in _setar_regime_roots(m)
        minimum(rts) < 1 && @warn "fit_setar: possible unit root in the $lab regime (smallest " *
                                  "characteristic root modulus $(round(minimum(rts); digits=6)))"
    end
    return nothing
end

StatsAPI.coef(m::SETARModel) = copy(m.coef)
StatsAPI.coefnames(m::SETARModel) = copy(m.names)
StatsAPI.nobs(m::SETARModel) = m.nobs
StatsAPI.residuals(m::SETARModel) = copy(m.resid)
StatsAPI.fitted(m::SETARModel) = copy(m.fitted)
StatsAPI.stderror(m::SETARModel) = copy(m.se)
StatsAPI.dof(m::SETARModel) = length(m.coef) + (m.searched ? length(m.thresholds) : 0)
_setar_sigma2(m::SETARModel) = m.ssr / m.nobs
StatsAPI.loglikelihood(m::SETARModel) = -m.nobs / 2 * (log(2pi) + log(_setar_sigma2(m)) + 1)
StatsAPI.aic(m::SETARModel) = m.nobs * log(_setar_sigma2(m)) + 2 * StatsAPI.dof(m)
StatsAPI.bic(m::SETARModel) = m.nobs * log(_setar_sigma2(m)) + log(m.nobs) * StatsAPI.dof(m)

"""
    tsdyn_ic(m::SETARModel) -> (mse, aic, bic)

`tsDyn`'s figures, reproduced from the same SSR: everything on the **full
series length** `n`, not the `rows` the regression used --
`mse = SSR/n`, `AIC = n log(mse) + 2k`, `BIC = n log(mse) + log(n) k`, with
`k` counting a searched threshold. `aic` and `bic` on the
model use the regression rows instead.
"""
function tsdyn_ic(m::SETARModel)
    n = length(m.y)
    mse = m.ssr / n
    k = StatsAPI.dof(m)
    return (mse=mse, aic=n * log(mse) + 2k, bic=n * log(mse) + log(n) * k)
end

"""
    predict(m::SETARModel, h) -> Vector

The deterministic skeleton iterated `h` steps: each step's regime chosen
by the forecast so far, no noise (as `tsDyn`'s `predict`). A nonlinear
model's forecast distribution is not Gaussian, so no interval is given.
"""
function StatsAPI.predict(m::SETARModel, h::Integer)
    h >= 1 || throw(ArgumentError("predict(::SETARModel): h must be >= 1"))
    y = copy(m.y)
    labs = length(m.lags) == 2 ? ["L", "H"] : ["L", "M", "H"]
    for _ in 1:h
        n = length(y)
        z = y[n - m.thDelay]
        r = 1 + count(<(z), m.thresholds)
        lab = labs[r]
        v = 0.0
        for (dn, val) in (("const", 1.0), ("trend", Float64(m.nobs + n - length(m.y) + 1)))
            idx = findfirst(==("$dn.$lab"), m.names)
            idx === nothing || (v += m.coef[idx] * val)
        end
        for j in m.lags[r]
            v += m.coef[findfirst(==("phi$lab.$j"), m.names)] * y[n + 1 - j]
        end
        push!(y, v)
    end
    return y[(end - h + 1):end]
end

function Base.show(io::IO, m::SETARModel)
    nt = length(m.thresholds)
    println(io, "SETAR model, $(nt + 1) regimes, m = $(m.m), thDelay = $(m.thDelay) ",
            "(z_t = y_{t-$(m.thDelay + 1)}), rows = $(m.nobs)")
    # full precision: a threshold must round-trip
    println(io, "  threshold$(nt > 1 ? "s" : ""): ", join(repr.(m.thresholds), ", "),
            m.searched ? " (searched)" : " (given)")
    for (nm, b, s) in zip(m.names, m.coef, m.se)
        println(io, "  ", rpad(nm, 10), lpad(string(round(b; sigdigits=10)), 18),
                "   se ", round(s; sigdigits=6))
    end
    counts = [count(==(r), m.regime) for r in 1:(nt + 1)]
    println(io, "  regime sizes: ", join(counts, " / "))
    print(io, "  SSR = ", round(m.ssr; sigdigits=12), "   AIC = ", round(StatsAPI.aic(m); digits=4),
          "  (no standard error for the threshold: its distribution is non-standard)")
end

# --- order and delay selection ----------------------------------------------------

"""
    select_setar(y, m; thDelay=0:m-1, mL=1:m, mH=1:m, trim=0.15,
                 criterion=:pooled_aic, grid=:full) -> NamedTuple

A grid search over threshold × delay × the two regimes' orders, as R's
`selectSETAR`. `criterion` is `:ssr`, `:aic` (on the full series length, as
`tsDyn`) or `:pooled_aic` (`tsDyn`'s default: the two regimes fitted
separately, each scored by its full Gaussian AIC with the variance counted,
on a sample embedded at `max(thDelay + 1, mL, mH)` -- so its rows depend on
the orders being compared). Returns the best `(thDelay, mL, mH, th, value)` and the whole
table, best first.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = log10.(readdlm(TSAnalytics.LYNX, ','; skipstart=1)[:, 2]);

julia> s = select_setar(y, 2; thDelay=0:1);

julia> (s.best.thDelay, s.best.mL, s.best.mH, s.best.th)
(1, 2, 2, 3.3100557377508912)
```
"""
function select_setar(y, m::Integer; thDelay=0:(m - 1), mL=1:m, mH=1:m, trim::Real=0.15,
                      criterion::Symbol=:pooled_aic, grid::Symbol=:full)
    criterion in (:ssr, :aic, :pooled_aic) || throw(ArgumentError(
        "select_setar: criterion must be :ssr, :aic or :pooled_aic"))
    yv = Vector{Float64}(tsvalues(y))
    n = length(yv)
    rows = NamedTuple{(:thDelay, :mL, :mH, :th, :value),Tuple{Int,Int,Int,Float64,Float64}}[]
    for d in thDelay
        0 <= d < m || throw(ArgumentError("select_setar: thDelay values must be in 0:(m-1)"))
        _, _, z, _ = _tar_embed(yv, Int(m), Int(d))
        cands = _tar_grid(z, trim; grid=grid)
        for a in mL, b in mH
            for c in cands
                f = _setar_fit_at(yv, Int(m), Int(d), [c], :const, [collect(1:a), collect(1:b)])
                f === nothing && continue
                val = if criterion === :ssr
                    f.ssr
                elseif criterion === :aic
                    n * log(f.ssr / n) + 2 * (length(f.b) + 1)
                else
                    _setar_pooled_aic(yv, Int(d), Int(a), Int(b), c)
                end
                push!(rows, (thDelay=Int(d), mL=Int(a), mH=Int(b), th=c, value=val))
            end
        end
    end
    sort!(rows; by=r -> r.value)
    return (best=rows[1], table=rows)
end

# tsDyn's pooled AIC, re-derived from its measured behaviour: each regime
# fitted separately by least squares on a sample embedded at
# max(thDelay + 1, mL, mH) -- not at m, so the rows depend on the orders --
# and scored by the full Gaussian AIC with the variance counted as a
# parameter, n_r (log 2π + log(SSR_r/n_r) + 1) + 2 (k_r + 1).
function _setar_pooled_aic(y, d, a, b, th)
    meff = max(d + 1, a, b)
    Y, lag, z, _ = _tar_embed(y, meff, d)
    tot = 0.0
    for (sel, k) in ((z .<= th, a), (z .> th, b))
        nr = count(sel)
        nr > k + 1 || return Inf
        X = hcat(ones(nr), [lag(j)[sel] for j in 1:k]...)
        e = Y[sel] .- X * (X \ Y[sel])
        S = dot(e, e)
        tot += nr * (log(2pi) + log(S / nr) + 1) + 2 * (k + 2)
    end
    return tot
end

# --- LSTAR -------------------------------------------------------------------------

"""
    LSTARModel

A fitted LSTAR model, from [`fit_lstar`](@ref): `phiL` and `phiH` (the
**high** coefficients are increments: the high regime's levels are
`phiL + phiH`), `gamma`, `th`, `ssr`, the transition `G` per row, and
standard errors (`tsDyn`'s convention). `gamma_at_bound` records a slope
that ran to the start grid's maximum.
"""
struct LSTARModel <: TimeSeriesModel
    y::Vector{Float64}
    m::Int
    thDelay::Int
    phiL::Vector{Float64}
    phiH::Vector{Float64}
    gamma::Float64
    th::Float64
    ssr::Float64
    nobs::Int
    se::Vector{Float64}
    G::Vector{Float64}
    converged::Bool
    gamma_at_bound::Bool
    names::Vector{String}
end

_lstar_G(z, γ, c) = 1 ./ (1 .+ exp.(-γ .* (z .- c)))

# profile: given (γ, c), the regime coefficients by least squares on
# [X, X .* G] -- the ADDITIVE form tsDyn fits, ŷ = X'φL + (X'φH) G, not
# the convex (1 - G) φL + G φH
function _lstar_profile(Y, X, z, γ, c)
    G = _lstar_G(z, γ, c)
    D = hcat(X, X .* G)
    b = D \ Y
    e = Y .- D * b
    return dot(e, e), b, G
end

# the SSR as a function of every parameter, and tsDyn's standard errors
# from its Hessian: vcov = 2 SSR / n · H⁻¹, n the full series length
function _lstar_ssr_full(Y, X, z, θ)
    k = size(X, 2)
    G = _lstar_G(z, θ[end - 1], θ[end])
    e = Y .- X * θ[1:k] .- (X * θ[(k + 1):(2k)]) .* G
    return dot(e, e)
end

function _lstar_se(Y, X, z, θ, n)
    H = ForwardDiff.hessian(v -> _lstar_ssr_full(Y, X, z, v), θ)
    C = cholesky(Symmetric(H); check=false)
    issuccess(C) || return fill(NaN, length(θ))
    return sqrt.(max.(diag(2 * _lstar_ssr_full(Y, X, z, θ) / n .* inv(C)), 0.0))
end

"""
    fit_lstar(y, m; thDelay=0, trim=0.15, ngamma=40, nth=200, gamma_max=100.0) -> LSTARModel

Logistic smooth-transition autoregression,
```
ŷ_t = X_t' φ_L + (X_t' φ_H) · G(z_t),   G(z) = 1 / (1 + exp(−γ (z − c)))
```
with `X_t = [1, y_{t−1}, …, y_{t−m}]` and `z_t = y_{t−(thDelay+1)}`.

**The form is additive**: `φ_H` are increments, so the high regime's
coefficients are `φ_L + φ_H`. (The convex form `(1 − G) φ_L + G φ_H` is a
different model -- on log10(lynx) its SSR at R's estimates is 67 times
worse.) **γ multiplies the raw `z − c`**, not `(z − c)/sd(z)` as in much of
the literature, so it is not comparable with a standardised slope.

The regime coefficients are profiled out by least squares; `(γ, c)` start
from the best of an `ngamma × nth` grid (γ in `1:gamma_max`, `c` over the
trimmed range of `z`), then BFGS. The grid is not optional: the likelihood
is nearly flat in γ, and without it the optimiser stays wherever it began.
A γ at or past `gamma_max` is reported (`gamma_at_bound`) -- the transition
is then effectively abrupt, and [`fit_setar`](@ref) is the better model.

Standard errors are `tsDyn`'s: `vcov = 2 SSR / n · H⁻¹`, `H` the Hessian of
the SSR in all parameters, `n` the full series length. They are fragile
for γ (its profile is nearly flat).

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = log10.(readdlm(TSAnalytics.LYNX, ','; skipstart=1)[:, 2]);

julia> m = fit_lstar(y, 2; thDelay=1);

julia> round(m.ssr, digits=5)
4.33764

julia> m.gamma_at_bound
false
```
"""
function fit_lstar(y, m::Integer; thDelay::Integer=0, trim::Real=0.15, ngamma::Integer=40,
                   nth::Integer=200, gamma_max::Real=100.0)
    yv = Vector{Float64}(tsvalues(y))
    0 <= thDelay < m || throw(ArgumentError("fit_lstar: thDelay must satisfy 0 <= thDelay < m"))
    0 < trim < 0.5 || throw(ArgumentError("fit_lstar: trim must be in (0, 0.5)"))
    Y, lag, z, rows = _tar_embed(yv, Int(m), Int(thDelay))
    nr = length(Y)
    X = hcat(ones(nr), [lag(j) for j in 1:m]...)
    zs = sort(z)
    lo, hi = zs[ceil(Int, trim * nr)], zs[floor(Int, (1 - trim) * nr)]
    γs = range(1.0, gamma_max; length=ngamma)
    cs = range(lo, hi; length=nth)
    grid = [(γ, c) for γ in γs, c in cs]
    ssrs = Vector{Float64}(undef, length(grid))
    Threads.@threads for i in eachindex(grid)
        ssrs[i] = _lstar_profile(Y, X, z, grid[i]...)[1]
    end
    γ0, c0 = grid[argmin(ssrs)]
    # a saturated transition (G constant over the sample) makes [X, X G]
    # rank-deficient: score it as infeasible so the line search backs off
    obj(v) = try
        _lstar_profile(Y, X, z, exp(v[1]), v[2])[1]
    catch err
        err isa SingularException || rethrow()
        oftype(v[1], Inf)
    end
    res = _optimize(obj, [log(γ0), c0]; method=:bfgs, autodiff=true, g_tol=1e-10, iterations=2000)
    γ, c = exp(res.minimizer[1]), res.minimizer[2]
    ssr, b, G = _lstar_profile(Y, X, z, γ, c)
    k = m + 1
    θ = vcat(b, γ, c)
    se = _lstar_se(Y, X, z, θ, length(yv))
    names = vcat("const.L", ["phiL.$j" for j in 1:m], "const.H", ["phiH.$j" for j in 1:m], "gamma", "th")
    atb = γ >= gamma_max * (1 - 1e-6)
    atb && @warn "fit_lstar: gamma = $(round(γ; sigdigits=6)) is at or beyond the start grid's maximum " *
                 "($gamma_max): the transition is effectively abrupt -- consider fit_setar"
    return LSTARModel(yv, Int(m), Int(thDelay), b[1:k], b[(k + 1):(2k)], γ, c, ssr, nr, se, G,
                      res.converged, atb, names)
end

StatsAPI.coef(m::LSTARModel) = vcat(m.phiL, m.phiH, m.gamma, m.th)
StatsAPI.coefnames(m::LSTARModel) = copy(m.names)
StatsAPI.stderror(m::LSTARModel) = copy(m.se)
StatsAPI.nobs(m::LSTARModel) = m.nobs
StatsAPI.dof(m::LSTARModel) = 2 * (m.m + 1) + 2

function Base.show(io::IO, m::LSTARModel)
    println(io, "LSTAR model, m = $(m.m), thDelay = $(m.thDelay), rows = $(m.nobs) ",
            "(additive form: the high regime's coefficients are phiL + phiH)")
    for (nm, b, s) in zip(m.names, StatsAPI.coef(m), m.se)
        println(io, "  ", rpad(nm, 10), lpad(string(round(b; sigdigits=10)), 18),
                "   se ", round(s; sigdigits=6))
    end
    print(io, "  SSR = ", round(m.ssr; sigdigits=12), m.converged ? "" : "   (optimiser did NOT converge)",
          m.gamma_at_bound ? "   gamma at the grid maximum: effectively abrupt" : "")
end

# --- Hansen's (1999) test -----------------------------------------------------------

"""
    ThresholdTest <: HypothesisTest

Hansen's (1999) tests of linearity against SETAR, from
[`setar_test`](@ref): the sums of squares of the linear AR and the one- and
two-threshold SETARs, the statistics `F_ij = n (S_i − S_j) / S_j` with
`n = rows`, and bootstrap p-values and critical values.
"""
struct ThresholdTest <: HypothesisTest
    ssr::Vector{Float64}                 # AR, SETAR(1), SETAR(2)
    F::Vector{Float64}                   # 1vs2, 1vs3, 2vs3
    pvalues::Vector{Float64}
    critical::Matrix{Float64}            # 3 x 4: 0.90, 0.95, 0.975, 0.99
    nboot::Int
    nobs::Int
end

"""
    setar_test(y, m; thDelay=0:m-1, trim=0.15, nboot=1000, rng=Random.default_rng()) -> ThresholdTest

Hansen's (1999) bootstrap tests: linear AR(`m`) against one and two
thresholds, and one threshold against two. `F_ij = n (S_i − S_j)/S_j`, `n`
the regression rows, each SETAR searched over `thDelay`. The null
distribution is non-standard, so the p-values are by bootstrap only:
residuals resampled under the null model, every statistic recomputed.
`tsDyn`'s default of 10 replications is far too few to be stable; the
default here is 1000.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = log10.(readdlm(TSAnalytics.LYNX, ','; skipstart=1)[:, 2]);

julia> t = setar_test(y, 2; thDelay=0:1, nboot=10);

julia> round.(t.F, digits=4)
3-element Vector{Float64}:
 36.9468
 46.5898
  7.251
```
"""
function setar_test(y, m::Integer; thDelay=0:(m - 1), trim::Real=0.15, nboot::Integer=1000,
                    rng=Random.default_rng())
    yv = Vector{Float64}(tsvalues(y))
    nboot >= 1 || throw(ArgumentError("setar_test: nboot must be >= 1"))
    stats(v) = begin
        Y, lag, _, _ = _tar_embed(v, Int(m), 0)
        X = hcat(ones(length(Y)), [lag(j) for j in 1:m]...)
        b = X \ Y
        S1 = sum(abs2, Y .- X * b)
        S2 = minimum(fit_setar(v, m; thDelay=d, trim=trim, warn=false).ssr for d in thDelay)
        S3 = minimum(fit_setar(v, m; thDelay=d, nthresh=2, trim=trim, warn=false).ssr for d in thDelay)
        nr = length(Y)
        (S=[S1, S2, S3], F=[nr * (S1 - S2) / S2, nr * (S1 - S3) / S3, nr * (S2 - S3) / S3], b=b, nr=nr)
    end
    obs = stats(yv)
    Y, lag, _, _ = _tar_embed(yv, Int(m), 0)
    X = hcat(ones(length(Y)), [lag(j) for j in 1:m]...)
    e = Y .- X * obs.b
    e .-= mean(e)
    Fb = zeros(nboot, 3)
    for b in 1:nboot
        ys = copy(yv[1:m])
        for _ in 1:(length(yv) - m)
            t = length(ys)
            push!(ys, obs.b[1] + sum(obs.b[j + 1] * ys[t + 1 - j] for j in 1:m) + e[rand(rng, 1:length(e))])
        end
        Fb[b, :] = stats(ys).F
    end
    pv = [mean(Fb[:, i] .>= obs.F[i]) for i in 1:3]
    q = [0.90, 0.95, 0.975, 0.99]
    crit = [quantile(Fb[:, i], qq) for i in 1:3, qq in q]
    return ThresholdTest(obs.S, obs.F, pv, crit, Int(nboot), obs.nr)
end

statistic(t::ThresholdTest) = t.F
pvalue(t::ThresholdTest) = t.pvalues

function Base.show(io::IO, t::ThresholdTest)
    println(io, "Hansen (1999) threshold tests, bootstrap with $(t.nboot) replications, rows = $(t.nobs)")
    println(io, "  SSR: AR = ", round(t.ssr[1]; sigdigits=10), ", SETAR(1) = ", round(t.ssr[2]; sigdigits=10),
            ", SETAR(2) = ", round(t.ssr[3]; sigdigits=10))
    for (i, lab) in enumerate(("1 vs 2", "1 vs 3", "2 vs 3"))
        println(io, "  ", lab, ": F = ", round(t.F[i]; digits=4), ", bootstrap p = ", round(t.pvalues[i]; digits=4))
    end
    print(io, "  (no asymptotic p-value: the null distribution is non-standard)")
end
