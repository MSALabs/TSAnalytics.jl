export fit_tbats, TBATSModel

"""
    TBATSModel

A fitted TBATS model, returned by [`fit_tbats`](@ref).

## Fields

- `lambda`: the Box-Cox parameter, or `nothing` when no transform was used.
- `alpha`, `beta`, `phi`: level, trend and damping parameters. `beta` and
  `phi` are `nothing` when the component is absent.
- `gamma1`, `gamma2`: one seasonal smoothing pair per seasonal period.
- `ar`, `ma`: the ARMA coefficients of the error process, possibly empty.
- `periods`, `k`: the seasonal periods and the number of harmonics used for
  each. `periods` may be non-integer.
- `seed_states`: the fitted initial state, laid out as
  `[l, b?, s_1..s_k1, s*_1..s*_k1, ..., d_1..d_p, e_1..e_q]` — R's own
  ordering, so its `seed.states` can be fed straight in.
- `fitted`, `resid`, `sse`: one-step fitted values **on the original scale**,
  the residuals **on the transformed scale** (which is what the likelihood is
  built from and what R's `errors` holds), and their sum of squares.
- `sigma2`, `loglik`, `aic`, `bic`, `nparams`, `nobs`, `converged`.
"""
struct TBATSModel
    lambda::Union{Nothing,Float64}
    alpha::Float64
    beta::Union{Nothing,Float64}
    phi::Union{Nothing,Float64}
    gamma1::Vector{Float64}
    gamma2::Vector{Float64}
    ar::Vector{Float64}
    ma::Vector{Float64}
    periods::Vector{Float64}
    k::Vector{Int}
    seed_states::Vector{Float64}
    fitted::Vector{Float64}
    resid::Vector{Float64}
    sse::Float64
    sigma2::Float64
    loglik::Float64
    aic::Float64
    bic::Float64
    nparams::Int
    nobs::Int
    converged::Bool
end

function Base.show(io::IO, m::TBATSModel)
    print(io, "TBATS(")
    print(io, m.lambda === nothing ? "-" : string(round(m.lambda, digits=4)), ", {")
    print(io, join(string.(Int.(round.(m.periods, digits=0))), ","), "}, {")
    print(io, join(string.(m.k), ","), "}, ")
    print(io, length(m.ar), ",", length(m.ma), ")")
    println(io, "  n=", m.nobs, m.converged ? "" : "  (NOT CONVERGED)")
    print(io, "  alpha = ", round(m.alpha, digits=6))
    m.beta !== nothing && print(io, "   beta = ", round(m.beta, digits=6))
    m.phi !== nothing && print(io, "   phi = ", round(m.phi, digits=6))
    println(io)
    if !isempty(m.gamma1)
        println(io, "  gamma1 = ", round.(m.gamma1, digits=8),
                    "   gamma2 = ", round.(m.gamma2, digits=8))
    end
    isempty(m.ar) || println(io, "  ar = ", round.(m.ar, digits=6))
    isempty(m.ma) || println(io, "  ma = ", round.(m.ma, digits=6))
    print(io, "  SSE = ", round(m.sse, digits=4),
              "   sigma2 = ", round(m.sigma2, digits=5),
              "   AIC = ", round(m.aic, digits=4))
end

"""
    _tbats_nstate(trend, k, p, q) -> Int

The length of the TBATS state vector: a level, a trend when present, two
states per harmonic per seasonal period, and the ARMA error buffers.
"""
_tbats_nstate(trend::Bool, k::AbstractVector{<:Integer}, p::Integer, q::Integer) =
    1 + (trend ? 1 : 0) + 2 * sum(k) + p + q

"""
    _tbats_recursion!(fitted, resid, y, x0, alpha, beta, phi, gamma1, gamma2,
                      ar, ma, periods, k, trend) -> sse

The TBATS recursion of De Livera, Hyndman & Snyder (2011), run forward on
the (already Box-Cox transformed) series `y` from the initial state `x0`.

    yhat_t = l + phi*b + sum_i sum_j s_{i,j} + sum ar_i d_{t-i} + sum ma_i e_{t-i}
    e_t    = y_t - yhat_t
    d_t    = y_t - (l + phi*b + sum_i sum_j s_{i,j})
    l      <- l + phi*b + alpha*d_t
    b      <- phi*b + beta*d_t
    s_{i,j}  <-  cos(lam)*s_{i,j} + sin(lam)*sstar_{i,j} + gamma1_i*d_t
    sstar_{i,j} <- -sin(lam)*s_{i,j} + cos(lam)*sstar_{i,j} + gamma2_i*d_t

with `lam = 2*pi*j/periods[i]`. **Only the `s` components enter the
measurement**, not the `sstar` ones; the pair exists so that each harmonic
rotates rather than merely decaying, which is what lets a non-integer
period work at all.

The state updates use `d_t`, the ARMA-filtered error, not the innovation
`e_t` — they coincide only when there is no ARMA component. Verified
against R at R's own parameters and seed states: the recursion reproduces
its `fitted[1]`, its SSE (`672.212547` against `672.2125`), its
`variance`, its `likelihood` (`n*log(SSE)`) and its `AIC` exactly.

`x0` is laid out as `[l, b?, s..., sstar..., d_1..d_p, e_1..e_q]`, where
the `s` block holds every harmonic of period 1, then every harmonic of
period 2, and so on, with the matching `sstar` block after it — R's own
`seed.states` ordering, confirmed on a `k = 5` monthly fit where
`l + b + sum(first 5 seeds)` reproduces its `fitted[1]` to the digit.
"""
function _tbats_recursion!(fitted::AbstractVector, resid::AbstractVector,
                           y::AbstractVector, x0::AbstractVector,
                           alpha, beta, phi, gamma1::AbstractVector,
                           gamma2::AbstractVector, ar::AbstractVector,
                           ma::AbstractVector, periods::AbstractVector,
                           k::AbstractVector{<:Integer}, trend::Bool)
    n = length(y)
    T = promote_type(eltype(x0), eltype(y), typeof(alpha))
    K = sum(k)
    p, q = length(ar), length(ma)

    i = 1
    l = T(x0[i]); i += 1
    b = trend ? (v = T(x0[i]); i += 1; v) : zero(T)
    s = T[x0[i + j - 1] for j in 1:K]; i += K
    sst = T[x0[i + j - 1] for j in 1:K]; i += K
    dbuf = T[x0[i + j - 1] for j in 1:p]; i += p
    ebuf = T[x0[i + j - 1] for j in 1:q]

    # the rotation angles, flattened in the same order as the s block
    cosl = Vector{T}(undef, K)
    sinl = Vector{T}(undef, K)
    gi = Vector{Int}(undef, K)
    c = 1
    for (idx, m) in enumerate(periods)
        for j in 1:k[idx]
            lam = 2 * pi * j / m
            cosl[c] = cos(lam)
            sinl[c] = sin(lam)
            gi[c] = idx
            c += 1
        end
    end

    sse = zero(T)
    @inbounds for t in 1:n
        base = l + phi * b
        for j in 1:K
            base += s[j]
        end
        arma = zero(T)
        for j in 1:p
            arma += ar[j] * dbuf[j]
        end
        for j in 1:q
            arma += ma[j] * ebuf[j]
        end
        yh = base + arma
        e = y[t] - yh
        d = e + arma                 # d_t = y_t - base
        fitted[t] = yh
        resid[t] = e
        sse += e^2

        lnew = l + phi * b + alpha * d
        bnew = trend ? phi * b + beta * d : zero(T)
        for j in 1:K
            sj, ssj = s[j], sst[j]
            s[j] = cosl[j] * sj + sinl[j] * ssj + gamma1[gi[j]] * d
            sst[j] = -sinl[j] * sj + cosl[j] * ssj + gamma2[gi[j]] * d
        end
        # shift the ARMA buffers: index 1 is the most recent
        for j in p:-1:2
            dbuf[j] = dbuf[j - 1]
        end
        p >= 1 && (dbuf[1] = d)
        for j in q:-1:2
            ebuf[j] = ebuf[j - 1]
        end
        q >= 1 && (ebuf[1] = e)
        l, b = lnew, bnew
    end
    return sse
end

"""
    _tbats_seed(y, alpha, beta, phi, gamma1, gamma2, ar, ma, periods, k, trend)
        -> Vector{Float64}

The initial state that minimises the sum of squared residuals, given the
smoothing parameters — obtained exactly, by least squares, not numerically.

This is what makes TBATS fittable at all. The state vector has
`2*sum(k) + 2 + p + q` entries, which is 12 even for a single monthly
period with five harmonics, and handing all of them to a derivative-free
optimiser alongside the smoothing parameters would be hopeless. But for
**fixed** smoothing parameters the recursion is affine in the initial
state:

    x_t = (F - g*w') x_{t-1} + g*y_t      so      e(x0) = e(0) + D*x0

where column `j` of `D` is the residual path produced by starting at the
`j`th unit vector with `y = 0`. The least-squares solution is then
`x0 = -(D'D) \\ (D' * e(0))`, costing `nstate + 1` recursion runs.

De Livera, Hyndman & Snyder (2011) do the same thing, and R's `tbats`
reports only the smoothing parameters in `parameters\$vect` while still
counting the seed states in its AIC — four and twelve respectively on a
monthly `AirPassengers` fit, giving the sixteen its AIC implies.
"""
function _tbats_seed(y::AbstractVector, alpha, beta, phi,
                     gamma1::AbstractVector, gamma2::AbstractVector,
                     ar::AbstractVector, ma::AbstractVector,
                     periods::AbstractVector, k::AbstractVector{<:Integer},
                     trend::Bool)
    n = length(y)
    ns = _tbats_nstate(trend, k, length(ar), length(ma))
    f = Vector{Float64}(undef, n)
    e0 = Vector{Float64}(undef, n)
    _tbats_recursion!(f, e0, y, zeros(ns), alpha, beta, phi, gamma1, gamma2,
                      ar, ma, periods, k, trend)
    D = Matrix{Float64}(undef, n, ns)
    zy = zeros(n)
    unit = zeros(ns)
    col = Vector{Float64}(undef, n)
    for j in 1:ns
        fill!(unit, 0.0)
        unit[j] = 1.0
        _tbats_recursion!(f, col, zy, unit, alpha, beta, phi, gamma1, gamma2,
                          ar, ma, periods, k, trend)
        @views D[:, j] .= col
    end
    # minimise ||e0 + D*x0||^2
    x0 = try
        -(D \ e0)
    catch
        zeros(ns)
    end
    return all(isfinite, x0) ? x0 : zeros(ns)
end

"""
    _tbats_matrices(alpha, beta, phi, gamma1, gamma2, periods, k, trend)
        -> (F, g, w)

The TBATS state space matrices, for the forecastability check below.

    x_t = F*x_{t-1} + g*eps_t        y_t = w'*x_{t-1} + eps_t

`F` is block diagonal: `[1 phi; 0 phi]` for the level and trend, then a
`[cos lam  sin lam; -sin lam  cos lam]` rotation per harmonic. `w` picks
the level, the damped trend and the `s` components but **not** the `sstar`
ones, and `g` carries `alpha`, `beta` and the two seasonal smoothing
parameters per harmonic.
"""
function _tbats_matrices(alpha, beta, phi, gamma1::AbstractVector,
                          gamma2::AbstractVector, periods::AbstractVector,
                          k::AbstractVector{<:Integer}, trend::Bool)
    K = sum(k)
    nt = trend ? 2 : 1
    ns = nt + 2K
    F = zeros(ns, ns)
    w = zeros(ns)
    g = zeros(ns)
    F[1, 1] = 1.0
    w[1] = 1.0
    g[1] = alpha
    if trend
        F[1, 2] = phi
        F[2, 2] = phi
        w[2] = phi
        g[2] = beta
    end
    c = 1
    for (idx, per) in enumerate(periods)
        for j in 1:k[idx]
            lam = 2 * pi * j / per
            is = nt + c
            iss = nt + K + c
            F[is, is] = cos(lam)
            F[is, iss] = sin(lam)
            F[iss, is] = -sin(lam)
            F[iss, iss] = cos(lam)
            w[is] = 1.0
            g[is] = gamma1[idx]
            g[iss] = gamma2[idx]
            c += 1
        end
    end
    return F, g, w
end

"""
    TBATS_STABILITY_MARGIN

How far inside the unit circle the spectral radius of `F - g*w'` must sit
for a TBATS fit to be accepted: `1e-5`.

A plain `< 1` is not enough in practice. Left to itself the optimiser rides
the boundary, because a nearly non-stationary seasonal component fits the
sample better: on the bundled quarterly fixture it reaches a radius of
`0.9999999459` and an SSE of `657.39` against R's `672.21`, which is a
lower in-sample error bought with a seasonal that barely mean-reverts and
so forecasts worse, not better.

`1e-5` is chosen to keep the references' own optima comfortably inside —
R's quarterly fit has radius `0.9999280`, a margin of `7.2e-5` — while
excluding the boundary itself. It is a loose guard, not a statistical
constraint: a radius of `1 - 1e-5` still implies a seasonal component with
a half-life of tens of thousands of periods, so nothing anyone would call
mean-reverting is being ruled out.
"""
const TBATS_STABILITY_MARGIN = 1e-5

"""
    _tbats_admissible(alpha, beta, phi, gamma1, gamma2, periods, k, trend) -> Bool

Whether the parameters give a **forecastable** TBATS model: every
eigenvalue of `D = F - g*w'` strictly inside the unit circle.

This is the real parameter region, and it is not a box. De Livera, Hyndman
& Snyder (2011) derive it, and R enforces the same condition; without it
the optimiser happily runs away. Measured on the bundled quarterly
fixture: dropping the condition and bounding the parameters individually
instead reaches `gamma1 = -0.477`, `gamma2 = 0.446` and an SSE of
`297.10` against R's `672.21` -- less than half, and not a better model
but a divergent one, whose seasonal component chases the noise and whose
forecasts are worthless. With the condition in place the fit lands beside
R's.

The spectral radius is what matters rather than the individual parameters,
so there is no useful box to substitute; this is one of the places where
`LinearAlgebra` earns its keep over a hand-rolled check.
"""
function _tbats_admissible(alpha, beta, phi, gamma1::AbstractVector,
                            gamma2::AbstractVector, periods::AbstractVector,
                            k::AbstractVector{<:Integer}, trend::Bool)
    F, g, w = _tbats_matrices(alpha, beta, phi, gamma1, gamma2, periods, k, trend)
    D = F - g * transpose(w)
    ev = try
        eigvals(D)
    catch
        return false
    end
    return all(isfinite, ev) && maximum(abs, ev) < 1 - TBATS_STABILITY_MARGIN
end

"""
    _tbats_kmax(m) -> Int

The largest usable number of harmonics for a seasonal period `m`.

`floor(m/2)` harmonics span an arbitrary seasonal pattern of integer
period `m`, so that looks like the ceiling — but for an **even integer**
`m` the last one, `j = m/2`, has `lam = pi`, which makes
`[cos lam sin lam; -sin lam cos lam]` equal `-I`. The pair then stops
rotating: `s` and `sstar` evolve independently, and since only `s` enters
the measurement, `sstar` becomes **completely unidentified**. So the
ceiling is `m/2 - 1` there.

This is not a cosmetic restriction. Admitting the Nyquist harmonic leaves
a state the data cannot inform, and the optimiser gets lost in it: on the
bundled quarterly fixture, `k = 2` for `m = 4` reaches an SSE of `1216.2`
where `k = 1` reaches `657.4` — a strictly richer model fitting almost
twice as badly, which is the signature of an unidentified direction
rather than of a worse model.

The rule reproduces R's own choices exactly: `forecast::tbats` selects
`k = 1` for quarterly data and `k = 5` for monthly, which are
`4/2 - 1` and `12/2 - 1`.

A non-integer period never hits `lam = pi` — that needs `j = m/2` with `j`
an integer — so `floor(m/2)` stands there.
"""
function _tbats_kmax(m::Real)
    if isinteger(m) && iseven(Int(m))
        return max(1, Int(m) ÷ 2 - 1)
    end
    return max(1, Int(floor(m / 2)))
end

"""
    _tbats_default_k(periods, n) -> Vector{Int}

The default harmonic count per seasonal period: [`_tbats_kmax`](@ref),
capped so the state stays well short of the sample size.

R searches **downward** from its own maximum by AIC, which
[`fit_tbats`](@ref) does not do — pass `k` to choose. On `AirPassengers`
R's search lands on exactly this default, `k = 5`.
"""
function _tbats_default_k(periods::AbstractVector, n::Integer)
    out = Int[]
    for m in periods
        ki = _tbats_kmax(m)
        ki = min(ki, max(1, (n - 4) ÷ 4))
        push!(out, ki)
    end
    return out
end

"""
    fit_tbats(y, periods; k=nothing, lambda=nothing, trend=true, damped=false,
              arma=(0, 0), optimizer_method=:nelder_mead) -> TBATSModel

Fit a **TBATS** model — Trigonometric seasonality, Box-Cox transform, ARMA
errors, Trend and Seasonal components (De Livera, Hyndman & Snyder 2011).

TBATS exists to handle the seasonality that [`fit_ets`](@ref) cannot:

- **More than one seasonal period at once** — pass `periods=[7, 365.25]` for
  daily data with a weekly and an annual cycle.
- **Non-integer periods.** `365.25` is not a number of states you can hold;
  it is an angle you can rotate by, which is what the trigonometric
  representation buys.
- **Long periods cheaply.** A weekly cycle in half-hourly data has period
  `336`; a seasonal-state model needs 336 states, while `k = 4` harmonics
  need 8.

Each seasonal period contributes `k[i]` harmonic pairs, rotating by
`2*pi*j/periods[i]` per step. `k=nothing` uses the largest identifiable
count, which is `floor(m/2)` except for an **even integer** period, where
the `j = m/2` harmonic rotates by `pi`, stops mixing its pair, and leaves
one state the data cannot inform. That rule reproduces R's own selections:
`k = 1` for quarterly, `k = 5` for monthly.

## Arguments

- `periods`: one seasonal period, or several. `Float64` is fine.
- `k`: harmonics per period. A vector, or a single integer for all.
- `lambda`: Box-Cox parameter. `nothing` for none, a number to pin it, or
  `:auto` to choose it by [`guerrero_lambda`](@ref) before fitting.
- `trend`, `damped`: as in [`fit_ets`](@ref). `damped=true` implies a trend.
- `arma`: accepted for signature compatibility, but only `(0, 0)` is
  implemented; see below.

## The parameter region is not a box

TBATS is only forecastable when every eigenvalue of `D = F - g*w'` lies
strictly inside the unit circle, and that condition is what the optimiser
is constrained by here. It is not optional window dressing: bounding the
parameters individually instead, on the bundled quarterly fixture, reaches
`gamma1 = -0.477` and an SSE of `297.10` against R's `672.21` -- a fit
less than half the size that is divergent rather than better, with a
seasonal component chasing the noise.

The region is also **one-sided and not symmetric in the seasonal
parameters**. For a quarterly single-harmonic model every `gamma2 < 0` is
forecastable and every `gamma2 > 0` is not, whatever the magnitude, so
there is no small-gamma neighbourhood to start from on the wrong side.
Fitted TBATS models also sit very close to the boundary by nature — the
trigonometric seasonal is nearly a random walk — with R's own quarterly
fit at a spectral radius of `0.99993`. Being near the boundary is normal;
being outside it is not, and `converged` is `false` if the fit ends up
there.

## What this does not do

- **No ARMA errors.** The `A` in TBATS. The recursion handles them, but
  the forecastability condition above is only derived here for the
  no-ARMA state, and an unconstrained ARMA fit would reintroduce exactly
  the divergence that condition prevents. Passing a non-zero order errors
  rather than quietly ignoring it.
- **No automatic model selection.** R's `tbats()` searches over Box-Cox
  on/off, trend on/off, damping on/off, ARMA orders and the harmonic
  counts, by AIC. This fits exactly the model you ask for. Comparing a few
  candidates by `aic` is a short loop and is shown in the manual; a proper
  `auto_tbats` is a separate piece of work, as `auto_arima` was.

## Verification

The recursion, the likelihood and the AIC are checked against R at R's own
parameters and seed states, which tests the model rather than the
optimiser: on the bundled quarterly fixture it reproduces R's `fitted[1]`,
its SSE (`672.212547` against R's `672.2125`), its `variance`, its
`likelihood` — R reports `n*log(SSE)` — and its `AIC` exactly. The seed
state layout is confirmed separately on a `k = 5` monthly fit.

`loglik` here is the full Gaussian log-likelihood, as everywhere else in
this package, so it is comparable with [`fit_ets`](@ref)'s and
[`fit_arima`](@ref)'s. **R's `tbats` reports `n*log(SSE)` and calls it
`likelihood`**, which is neither a log-likelihood nor comparable with
anything else; its `AIC` is that plus `2*nparams`. Those differ from this
package's by a constant, exactly as documented for [`fit_ets`](@ref).

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                                 "verification", "ets", "ets_y.csv")));

julia> m = fit_tbats(y, 4; k=1);

julia> m.k
1-element Vector{Int64}:
 1

julia> length(m.seed_states)       # l, b, s_1, sstar_1
4
```

See also [`fit_ets`](@ref), [`fourier_terms`](@ref), [`forecast`](@ref).
"""
function fit_tbats(y, periods; k=nothing, lambda=nothing, trend::Bool=true,
                   damped::Bool=false, arma::Tuple{Integer,Integer}=(0, 0),
                   optimizer_method::Symbol=:nelder_mead)
    yv = Float64.(collect(tsvalues(y)))
    n = length(yv)
    n >= 5 || throw(ArgumentError("fit_tbats: need at least 5 observations, got $n"))
    any(isnan, yv) && throw(ArgumentError("fit_tbats: series contains NaN"))

    per = Float64.(collect(periods isa Number ? (periods,) : periods))
    isempty(per) && throw(ArgumentError("fit_tbats: give at least one seasonal period"))
    all(>(1), per) || throw(ArgumentError(
        "fit_tbats: every seasonal period must be > 1, got $per"))
    p, q = Int(arma[1]), Int(arma[2])
    (p >= 0 && q >= 0) || throw(ArgumentError("fit_tbats: arma orders must be >= 0"))
    # The recursion handles ARMA errors, but the forecastability condition
    # that keeps the optimiser honest is only built for the no-ARMA state
    # here, and shipping an unconstrained ARMA fit would be shipping the
    # divergent fits that condition exists to prevent.
    (p == 0 && q == 0) || throw(ArgumentError(
        "fit_tbats: ARMA errors are not implemented yet -- the forecastability " *
        "condition on the eigenvalues of F - g*w' is only derived here for the " *
        "no-ARMA state, and without it the optimiser diverges. Use arma=(0, 0), " *
        "or fit_arima to the residuals and treat the two stages separately."))

    kk = if k === nothing
        _tbats_default_k(per, n)
    elseif k isa Number
        fill(Int(k), length(per))
    else
        Int.(collect(k))
    end
    length(kk) == length(per) || throw(DimensionMismatch(
        "fit_tbats: k has $(length(kk)) entries for $(length(per)) periods"))
    all(>=(1), kk) || throw(ArgumentError("fit_tbats: every k must be >= 1"))
    for (i, m) in enumerate(per)
        kmax = _tbats_kmax(m)
        kk[i] <= kmax || throw(ArgumentError(
            "fit_tbats: k[$i] = $(kk[i]) exceeds the usable maximum $kmax for " *
            "period $(per[i]). Harmonics past floor(m/2) alias lower ones, and " *
            "for an even integer period the j = m/2 harmonic has lam = pi, " *
            "which leaves its sstar state unidentified."))
    end
    has_t = trend || damped
    ns = _tbats_nstate(has_t, kk, p, q)
    ns + 2 < n || throw(ArgumentError(
        "fit_tbats: the state has $ns entries for $n observations — reduce k"))

    # --- Box-Cox -----------------------------------------------------------
    lam = if lambda === nothing
        nothing
    elseif lambda === :auto
        all(>(0), yv) || throw(ArgumentError(
            "fit_tbats: lambda=:auto needs a strictly positive series"))
        guerrero_lambda(yv, Int(round(maximum(per))))
    else
        Float64(lambda)
    end
    z = lam === nothing ? yv : boxcox(yv, lam)

    # --- parameter packing --------------------------------------------------
    nper = length(per)
    # [alpha, beta?, phi?, gamma1..., gamma2..., ar..., ma...]
    function unpack(v)
        i = 1
        alpha = v[i]; i += 1
        beta = has_t ? (w = v[i]; i += 1; w) : 0.0
        phi = damped ? (w = v[i]; i += 1; w) : 1.0
        g1 = v[i:(i + nper - 1)]; i += nper
        g2 = v[i:(i + nper - 1)]; i += nper
        arc = v[i:(i + p - 1)]; i += p
        mac = v[i:(i + q - 1)]
        return alpha, beta, phi, collect(g1), collect(g2), collect(arc), collect(mac)
    end
    function inregion(alpha, beta, phi, g1, g2, arc, mac)
        # cheap screens first, then the real condition
        0 < alpha < 1.5 || return false
        has_t && !(-0.5 < beta < 1.5) && return false
        damped && !(0.8 <= phi <= 1.0) && return false
        all(g -> abs(g) < 1.0, g1) || return false
        all(g -> abs(g) < 1.0, g2) || return false
        return _tbats_admissible(alpha, beta, phi, g1, g2, per, kk, has_t)
    end

    fitted = Vector{Float64}(undef, n)
    resid = similar(fitted)
    function objective(v)
        alpha, beta, phi, g1, g2, arc, mac = unpack(v)
        inregion(alpha, beta, phi, g1, g2, arc, mac) || return 1e10
        x0 = _tbats_seed(z, alpha, beta, phi, g1, g2, arc, mac, per, kk, has_t)
        sse = _tbats_recursion!(fitted, resid, z, x0, alpha, beta, phi, g1, g2,
                                arc, mac, per, kk, has_t)
        isfinite(sse) ? sse : 1e10
    end

    # A deterministic multi-start, for the same reason fit_ets has one: with a
    # single start a RICHER TBATS can land on a worse optimum than a simpler
    # one, which is the optimiser getting stuck rather than the model being
    # worse. Measured before this was added: k = 2 on a period-7 fit reached
    # an SSE of 5168 where k = 1 reached 4534.
    #
    # Every start must be ADMISSIBLE, which is not a detail. The region is
    # one-sided in gamma2 -- for a quarterly single-harmonic model, gamma2 < 0
    # is forecastable and gamma2 > 0 is not, whatever the magnitude -- so a
    # start with the wrong sign makes the objective 1e10 everywhere the
    # simplex can reach, and the fit silently returns the starting values.
    # That is exactly what happened on a two-period fit before this guard
    # existed: alpha, beta and both gammas came back as the start point, to
    # the digit, with converged = true.
    START_GRID = [(0.1, 0.01, -1e-4, -1e-4, 0.98),
                  (0.02, 0.001, -1e-3, -1e-3, 0.95),
                  (0.4, 0.05, -1e-2, -1e-2, 0.90),
                  (0.1, 0.01, 1e-3, -1e-3, 0.97)]
    starts = Vector{Vector{Float64}}()
    for (a0, b0v, g10, g20, p0) in START_GRID
        v = Float64[a0]
        has_t && push!(v, b0v)
        damped && push!(v, p0)
        append!(v, fill(g10, nper))
        append!(v, fill(g20, nper))
        append!(v, fill(0.1, p))
        append!(v, fill(0.1, q))
        objective(v) < 1e10 && push!(starts, v)
    end
    if isempty(starts)
        # fall back to a scan over gamma magnitudes before giving up, so a
        # period whose admissible shell is narrow still gets fitted
        for g in (-1e-6, -1e-5, -3e-4, -3e-3, -3e-2)
            v = Float64[0.05]
            has_t && push!(v, 0.005)
            damped && push!(v, 0.97)
            append!(v, fill(g, nper))
            append!(v, fill(g, nper))
            append!(v, fill(0.1, p))
            append!(v, fill(0.1, q))
            if objective(v) < 1e10
                push!(starts, v)
                break
            end
        end
    end
    isempty(starts) && throw(ArgumentError(
        "fit_tbats: no forecastable starting point found for periods $per with " *
        "k = $kk. Every candidate had a spectral radius of F - g*w' at or above " *
        "one, so no fit was attempted rather than one being reported from the " *
        "starting values. Try fewer harmonics."))

    attempts = [try
                    _optimize(objective, x; method=optimizer_method, autodiff=false)
                catch e
                    e isa ArgumentError ? nothing : rethrow()
                end for x in starts]
    valid = [r for r in attempts if r !== nothing]
    isempty(valid) && throw(ArgumentError(
        "fit_tbats: the optimizer failed from every starting point"))
    best = valid[argmin([objective(r.minimizer) for r in valid])]
    for _ in 1:6
        cur = objective(best.minimizer)
        nxt = try
            _optimize(objective, best.minimizer; method=optimizer_method, autodiff=false)
        catch e
            e isa ArgumentError ? nothing : rethrow()
        end
        nxt === nothing && break
        objective(nxt.minimizer) <= cur && (best = nxt)
        cur - objective(best.minimizer) <= 1e-10 * max(abs(cur), 1.0) && break
    end

    alpha, beta, phi, g1, g2, arc, mac = unpack(best.minimizer)
    x0 = _tbats_seed(z, alpha, beta, phi, g1, g2, arc, mac, per, kk, has_t)
    sse = _tbats_recursion!(fitted, resid, z, x0, alpha, beta, phi, g1, g2,
                            arc, mac, per, kk, has_t)

    # nparams: the smoothing parameters, the seed states, Box-Cox if fitted,
    # and sigma2 -- R counts the first two and not the last two
    nsmooth = 1 + (has_t ? 1 : 0) + (damped ? 1 : 0) + 2 * nper + p + q
    nparams = nsmooth + ns + (lam === nothing ? 0 : 1) + 1
    sigma2 = sse / n
    loglik = -n / 2 * (log(2pi) + log(sigma2) + 1)
    aic = -2 * loglik + 2 * nparams
    bic = -2 * loglik + nparams * log(n)
    fit_orig = lam === nothing ? copy(fitted) : boxcox_inv(fitted, lam)

    # `converged` also requires the final point to be forecastable: the
    # objective is 1e10 outside the region, so an optimiser that never found
    # its way in would otherwise report success at its starting values.
    admissible = inregion(alpha, beta, phi, g1, g2, arc, mac)
    return TBATSModel(lam, alpha, has_t ? beta : nothing, damped ? phi : nothing,
                      g1, g2, arc, mac, per, kk, x0, fit_orig, copy(resid),
                      sse, sigma2, loglik, aic, bic, nparams, n,
                      best.converged && isfinite(sse) && admissible)
end

"""
    forecast(m::TBATSModel, horizon; level=[80.0, 95.0]) -> Forecast
    predict(m::TBATSModel, horizon; level=[80.0, 95.0]) -> Forecast

Forecast `horizon` steps ahead from a fitted [`TBATSModel`](@ref).

TBATS is a linear Gaussian state space model, so the point forecast is the
state propagated with zero innovations and the variance is the usual
`sigma2 * (1 + sum(psi[1:h-1].^2))` — the same arithmetic
[`fit_ets`](@ref)'s additive-error models use, with the shock weights
obtained by propagating a unit innovation's state perturbation rather than
by powering a transition matrix by hand.

When a Box-Cox transform was applied the point forecast and both interval
bounds are back-transformed. **The back-transformed point forecast is a
median, not a mean** — exponentiating does not commute with taking an
expectation — which is the same caveat [`boxcox_inv`](@ref) carries and
which [Chapter 7](../introduction/07-transformations.md) works through.
The intervals are unaffected by that distinction, since a quantile does
transform.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                                 "verification", "ets", "ets_y.csv")));

julia> f = forecast(fit_tbats(y, 4; k=1), 8);

julia> f.horizon
8

julia> issorted(f.se)
true
```

See also [`fit_tbats`](@ref), [`accuracy`](@ref).
"""
function StatsAPI.predict(model::TBATSModel, horizon::Integer;
                          level::Vector{<:Real}=[80.0, 95.0])
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    isempty(level) && throw(ArgumentError("level must be non-empty"))
    all(0 .< level .< 100) || throw(ArgumentError("level entries must be in (0, 100)"))

    # the state after the last observation: re-run the fitted recursion and
    # keep its final state, rather than storing it twice
    xN = _tbats_final_state(model)
    point_z = _tbats_propagate(xN, model, horizon)
    # psi weights: the same propagation from a unit innovation's perturbation
    gvec = _tbats_shock(model)
    psi = _tbats_propagate(gvec, model, horizon)
    se_z = [sqrt(model.sigma2 * (1 + sum(abs2, view(psi, 1:(h - 1))))) for h in 1:horizon]

    zs = [_confidence_z(1 - l / 100) for l in level]
    lo_z = reduce(hcat, [point_z .- zi .* se_z for zi in zs])
    hi_z = reduce(hcat, [point_z .+ zi .* se_z for zi in zs])

    lam = model.lambda
    point = lam === nothing ? point_z : boxcox_inv(point_z, lam)
    lower = lam === nothing ? lo_z : reshape(boxcox_inv(vec(lo_z), lam), size(lo_z))
    upper = lam === nothing ? hi_z : reshape(boxcox_inv(vec(hi_z), lam), size(hi_z))
    se = lam === nothing ? se_z : (upper[:, end] .- lower[:, end]) ./ (2 * zs[end])
    name = string("TBATS{", join(string.(model.k), ","), "}")
    return Forecast(point, se, Float64.(level), lower, upper, horizon, name)
end

forecast(model::TBATSModel, horizon::Integer; level::Vector{<:Real}=[80.0, 95.0]) =
    StatsAPI.predict(model, horizon; level=level)

"`_tbats_final_state(m)` -- the state after the last observation, from
re-running the fitted recursion. Cheaper to recompute than to store, and it
cannot drift out of step with `seed_states` this way."
function _tbats_final_state(m::TBATSModel)
    has_t = m.beta !== nothing
    beta = has_t ? m.beta : 0.0
    phi = m.phi === nothing ? 1.0 : m.phi
    K = sum(m.k)
    p, q = length(m.ar), length(m.ma)
    # the recursion does not expose its final state, so step it here
    x = copy(m.seed_states)
    cosl, sinl, gi = _tbats_angles(m)
    i0 = 1 + (has_t ? 1 : 0)
    for t in 1:m.nobs
        l = x[1]
        b = has_t ? x[2] : 0.0
        base = l + phi * b
        for j in 1:K
            base += x[i0 + j]
        end
        arma = 0.0
        for j in 1:p
            arma += m.ar[j] * x[i0 + 2K + j]
        end
        for j in 1:q
            arma += m.ma[j] * x[i0 + 2K + p + j]
        end
        e = m.resid[t]
        d = e + arma
        xn = copy(x)
        xn[1] = l + phi * b + m.alpha * d
        has_t && (xn[2] = phi * b + beta * d)
        for j in 1:K
            sj = x[i0 + j]
            ssj = x[i0 + K + j]
            xn[i0 + j] = cosl[j] * sj + sinl[j] * ssj + m.gamma1[gi[j]] * d
            xn[i0 + K + j] = -sinl[j] * sj + cosl[j] * ssj + m.gamma2[gi[j]] * d
        end
        for j in p:-1:2
            xn[i0 + 2K + j] = x[i0 + 2K + j - 1]
        end
        p >= 1 && (xn[i0 + 2K + 1] = d)
        for j in q:-1:2
            xn[i0 + 2K + p + j] = x[i0 + 2K + p + j - 1]
        end
        q >= 1 && (xn[i0 + 2K + p + 1] = e)
        x = xn
    end
    return x
end

"`_tbats_angles(m)` -- the per-harmonic cosines, sines and owning period
index, flattened in the state vector's own order."
function _tbats_angles(m::TBATSModel)
    K = sum(m.k)
    cosl = Vector{Float64}(undef, K)
    sinl = Vector{Float64}(undef, K)
    gi = Vector{Int}(undef, K)
    c = 1
    for (idx, per) in enumerate(m.periods)
        for j in 1:m.k[idx]
            lam = 2 * pi * j / per
            cosl[c] = cos(lam); sinl[c] = sin(lam); gi[c] = idx
            c += 1
        end
    end
    return cosl, sinl, gi
end

"`_tbats_shock(m)` -- the state perturbation a unit innovation produces,
which propagated forward gives the psi weights."
function _tbats_shock(m::TBATSModel)
    has_t = m.beta !== nothing
    K = sum(m.k)
    p, q = length(m.ar), length(m.ma)
    g = zeros(_tbats_nstate(has_t, m.k, p, q))
    i0 = 1 + (has_t ? 1 : 0)
    g[1] = m.alpha
    has_t && (g[2] = m.beta)
    _, _, gi = _tbats_angles(m)
    for j in 1:K
        g[i0 + j] = m.gamma1[gi[j]]
        g[i0 + K + j] = m.gamma2[gi[j]]
    end
    p >= 1 && (g[i0 + 2K + 1] = 1.0)
    q >= 1 && (g[i0 + 2K + p + 1] = 1.0)
    return g
end

"`_tbats_propagate(x, m, h)` -- the measurement path produced by evolving
the state `x` forward with zero innovations."
function _tbats_propagate(x0::AbstractVector, m::TBATSModel, horizon::Integer)
    has_t = m.beta !== nothing
    beta = has_t ? m.beta : 0.0
    phi = m.phi === nothing ? 1.0 : m.phi
    K = sum(m.k)
    p, q = length(m.ar), length(m.ma)
    cosl, sinl, gi = _tbats_angles(m)
    i0 = 1 + (has_t ? 1 : 0)
    x = Float64.(collect(x0))
    out = Vector{Float64}(undef, horizon)
    for h in 1:horizon
        l = x[1]
        b = has_t ? x[2] : 0.0
        base = l + phi * b
        for j in 1:K
            base += x[i0 + j]
        end
        arma = 0.0
        for j in 1:p
            arma += m.ar[j] * x[i0 + 2K + j]
        end
        for j in 1:q
            arma += m.ma[j] * x[i0 + 2K + p + j]
        end
        out[h] = base + arma
        d = arma                   # zero innovation
        xn = copy(x)
        xn[1] = l + phi * b
        has_t && (xn[2] = phi * b)
        for j in 1:K
            sj = x[i0 + j]
            ssj = x[i0 + K + j]
            xn[i0 + j] = cosl[j] * sj + sinl[j] * ssj + m.gamma1[gi[j]] * d
            xn[i0 + K + j] = -sinl[j] * sj + cosl[j] * ssj + m.gamma2[gi[j]] * d
        end
        for j in p:-1:2
            xn[i0 + 2K + j] = x[i0 + 2K + j - 1]
        end
        p >= 1 && (xn[i0 + 2K + 1] = d)
        for j in q:-1:2
            xn[i0 + 2K + p + j] = x[i0 + 2K + p + j - 1]
        end
        q >= 1 && (xn[i0 + 2K + p + 1] = 0.0)
        x = xn
    end
    return out
end

StatsAPI.coef(m::TBATSModel) = vcat(m.alpha,
    m.beta === nothing ? Float64[] : [m.beta],
    m.phi === nothing ? Float64[] : [m.phi],
    m.gamma1, m.gamma2, m.ar, m.ma)
StatsAPI.residuals(m::TBATSModel) = m.resid
StatsAPI.fitted(m::TBATSModel) = m.fitted
StatsAPI.nobs(m::TBATSModel) = m.nobs
StatsAPI.loglikelihood(m::TBATSModel) = m.loglik
StatsAPI.aic(m::TBATSModel) = m.aic
StatsAPI.bic(m::TBATSModel) = m.bic
