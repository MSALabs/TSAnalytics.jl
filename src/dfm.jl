# ---------------------------------------------------------------------------
# Stage 11.2 -- Dynamic factor models.
# handoff/stage 11/docs/stage-11.2-dynamic-factor-handoff.md
#
#   y_t = Λ f_t + ε_t,                ε_t ~ N(0, H), H diagonal
#   f_t = A_1 f_{t-1} + … + A_p f_{t-p} + η_t,   η_t ~ N(0, I)
#
# Direct maximum likelihood on the sequential multivariate filter (11.0a),
# with the factor VAR kept stationary by the Ansley-Kohn reparameterisation
# (11.0d) and a stationary prior recomputed at every evaluation (11.0c's
# `_stationary_cov`). Primary reference statsmodels `DynamicFactor`
# (Q = I, stationary initialisation); R's `dfms` an independent check.
# ---------------------------------------------------------------------------

export fit_dfm, DFMModel, factors, rotate_factors, factor_r2, series_r2, bai_ng_ic

"""
    DFMModel

A fitted dynamic factor model, from [`fit_dfm`](@ref): `Lambda` (`N × r`
loadings), `A` (the factor VAR's `p` coefficient matrices), `H` (the `N`
idiosyncratic variances), the smoothed factors and their standard errors
([`factors`](@ref)), `loglik`, `converged`, and `normalisation` -- `:lower`
for the fitted identification, `:rotated` after [`rotate_factors`](@ref).
"""
struct DFMModel <: StateSpaceModel
    Y::Matrix{Float64}
    names::Vector{Symbol}
    r::Int
    p::Int
    Lambda::Matrix{Float64}
    A::Vector{Matrix{Float64}}
    H::Vector{Float64}
    loglik::Float64
    nobs::Int
    converged::Bool
    iterations::Int
    estimated::Bool
    normalisation::Symbol
    factors::Matrix{Float64}
    factors_se::Matrix{Float64}
end

# --- the parameter vector -------------------------------------------------
# θ = [Λ's free entries, column by column] ++ [p r² unconstrained VAR
#      values] ++ [log H]. The leading block's diagonal is FREE during the
# search and made positive afterwards (flipping a factor's sign, with its
# loadings and VAR row and column, is an exact symmetry). Holding it positive
# by exp() instead traps the optimiser at a zero diagonal from bad starts.

_dfm_free(N, r) = [(i, j) for j in 1:r for i in 1:N if i >= j]

function _dfm_unpack(θ::AbstractVector, N, r, p)
    S = eltype(θ)
    fr = _dfm_free(N, r)
    Λ = zeros(S, N, r)
    for (k, (i, j)) in enumerate(fr)
        Λ[i, j] = θ[k]
    end
    o = length(fr)
    A = _constrain_stationary(θ[(o + 1):(o + p * r^2)], r, p)
    h = exp.(θ[(o + p * r^2 + 1):end])
    return Λ, A, h
end

function _dfm_pack(Λ, A, h)
    N, r = size(Λ)
    fr = _dfm_free(N, r)
    θΛ = [Λ[i, j] for (i, j) in fr]
    return vcat(θΛ, _unconstrain_stationary(A), log.(h))
end

function _dfm_system(Λ::AbstractMatrix, A::AbstractVector, h::AbstractVector)
    N, r = size(Λ)
    p = length(A)
    S = promote_type(eltype(Λ), eltype(A[1]), eltype(h), Float64)
    m = r * p
    T = zeros(S, m, m)
    for j in 1:p
        T[1:r, ((j - 1) * r + 1):(j * r)] = A[j]
    end
    for j in 2:p
        T[((j - 1) * r + 1):(j * r), ((j - 2) * r + 1):((j - 1) * r)] = Matrix{S}(I, r, r)
    end
    Z = hcat(Matrix{S}(Λ), zeros(S, N, m - r))
    R = vcat(Matrix{S}(I, r, r), zeros(S, m - r, r))
    Q = Matrix{S}(I, r, r)
    Hm = Matrix{S}(Diagonal(h))
    return T, Z, R, Q, Hm
end

# the log-likelihood under the stationary prior, recomputed from (Λ, A, h)
function _dfm_loglik(Y, Λ, A, h; strict::Bool=false)
    T, Z, R, Q, Hm = _dfm_system(Λ, A, h)
    P0, ok = _stationary_cov(T, R * R'; maxiter=200)
    ok || return eltype(T)(-Inf)
    ssm = TimeVaryingSSM{eltype(T)}([T], [Z], [R], [Q], [Hm], size(T, 1))
    return _kf_mv(ssm, Y, zeros(eltype(T), size(T, 1)), P0, strict)[1]
end

# PCA on the (caller-standardised) data, loadings by OLS, a VAR on the
# principal components, idiosyncratic variances from the residuals -- then
# moved into the identification: Q = I and Λ's leading block lower-triangular
# with a positive diagonal.
function _dfm_start(Y, r, p)
    n, N = size(Y)
    Yf = copy(Y)
    for j in 1:N                                   # fill NaN with 0 for the PCA only
        for t in 1:n
            isnan(Yf[t, j]) && (Yf[t, j] = 0.0)
        end
    end
    C = Symmetric(Yf' * Yf ./ n)
    ev = eigen(C)
    V = ev.vectors[:, sortperm(ev.values; rev=true)[1:r]]
    F = Yf * V
    Λ = (F \ Yf)'
    # a VAR(p) on the components, by least squares
    Zl = hcat([F[(p - j + 1):(n - j), :] for j in 1:p]...)
    Fy = F[(p + 1):end, :]
    B = Zl \ Fy
    E = Fy .- Zl * B
    Σ = Symmetric(E' * E ./ size(E, 1)) + 1e-8I
    A = [Matrix(B[((j - 1) * r + 1):(j * r), :]') for j in 1:p]
    # f̃ = L⁻¹ f has unit innovation variance: Λ̃ = Λ L, Ã = L⁻¹ A L
    L = cholesky(Σ).L
    Λ = Λ * L
    A = [L \ (Aj * L) for Aj in A]
    # rotate so the leading r x r block is lower-triangular, positive diagonal:
    # Λ[1:r,:] = Lb Qb (LQ), so Λ Qb' has a lower-triangular lead
    Fq = qr(Matrix(Λ[1:r, :]'))
    Qb = Matrix(Fq.Q)'
    Lb = Λ[1:r, :] * Qb'
    D = Diagonal(sign.(diag(Lb)) .+ (diag(Lb) .== 0))
    G = Qb' * D                                      # orthogonal
    Λ = Λ * G
    A = [G' * Aj * G for Aj in A]
    for j in 1:r
        Λ[j, j] = max(Λ[j, j], 1e-3)
        for i in 1:(j - 1)
            Λ[i, j] = 0.0
        end
    end
    # keep the start inside the stationary region
    ρ = maximum(abs, eigvals(_companion(A)))
    ρ >= 0.98 && (A = [Aj .* (0.95 / ρ)^j for (j, Aj) in enumerate(A)])
    h = vec([max(var(filter(!isnan, Y[:, i] .- (F * Λ[i, :]))), 1e-3) for i in 1:N])
    return Λ, A, h
end

"""
    fit_dfm(Y, r; p=1, start=nothing, params=nothing) -> DFMModel

Fit a **dynamic factor model** by maximum likelihood: the `N` columns of
`Y` (observations × series) driven by `r < N` common factors following a
stationary VAR(`p`),
```
y_t = Λ f_t + ε_t,    ε_t ~ N(0, H), H diagonal
f_t = A_1 f_{t-1} + … + A_p f_{t-p} + η_t,    η_t ~ N(0, I)
```

**`Y` is used as given.** The likelihood is not scale-invariant, so the
model does not standardise inside the fit; pass [`standardize`](@ref)`(Y)`
when the series are on different scales (the references do not agree on
this either). Missing values (`NaN`) are allowed and skipped.

**The model is identified explicitly**, which neither reference does: the
factor innovation variance is `I` (as statsmodels' `DynamicFactor`), and the
leading `r × r` block of `Λ` is lower-triangular with a positive diagonal,
removing the remaining rotation and the sign flips. The fit is therefore
unique -- `r(r−1)/2` fewer parameters than `DynamicFactor`, with the same
maximised likelihood, because the removed directions are flat. R's `dfms`
leaves the factor covariance unrestricted, pinned only by its PCA start.
[`rotate_factors`](@ref) gives any other rotation without refitting.

**The prior is the factor process's stationary distribution**, recomputed
from the parameters at every evaluation -- `DynamicFactor`'s convention.
`dfms` estimates the initial state instead (and statsmodels'
`DynamicFactorMQ` does so by default), which moves the reported likelihood
by about 0.9 on the test fixtures; compare like with like.

Estimated by direct maximum likelihood (L-BFGS, automatic derivatives),
with the factor VAR kept stationary by an Ansley-Kohn reparameterisation.
EM is not provided: at the sizes this package targets, direct ML converges
in tens of iterations where EM takes thousands.

`r >= N`, `N < 2` and `p < 1` are refused (`p = 0` would not mean "static
factors" in either reference). `start` (a `(Λ, A, H)` tuple) overrides the
PCA starting values; `params` evaluates the model at given `(Λ, A, H)`
without optimising.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(TSAnalytics.US_CHANGE, ','; skipstart=1);

julia> m = fit_dfm(standardize(Y), 1);

julia> round(m.loglik, digits=2)
-1143.52

julia> round.(vec(factor_r2(m)), digits=2)
5-element Vector{Float64}:
 0.43
 0.1
 0.9
 0.01
 0.84
```
"""
function fit_dfm(Y, r::Integer; p::Integer=1, start=nothing, params=nothing,
                 iterations::Integer=2000)
    M = Matrix{Float64}(tsmatrix(Y))
    n, N = size(M)
    nms = tsnames(Y)
    names = nms === nothing ? [Symbol("y", i) for i in 1:N] : Symbol.(nms)
    N >= 2 || throw(ArgumentError("fit_dfm: a factor model needs at least 2 series, got $N"))
    1 <= r < N || throw(ArgumentError(
        "fit_dfm: need 1 <= r < N (the number of series, $N); got r = $r. With as many " *
        "factors as series the idiosyncratic variances are not identified -- statsmodels " *
        "refuses this too"))
    p >= 1 || throw(ArgumentError(
        "fit_dfm: the factor VAR order p must be >= 1, got $p. (statsmodels silently turns " *
        "p = 0 into white-noise factors, dfms refuses it; static factor analysis is a " *
        "different model.)"))
    any(c -> count(!isnan, c) < 2, eachcol(M)) && throw(ArgumentError(
        "fit_dfm: every series needs at least 2 observed values"))
    if params !== nothing
        Λ, A, h = params
        A isa AbstractMatrix && (A = [A])
        Λ, A, h = Matrix{Float64}(Λ), Matrix{Float64}.(A), Vector{Float64}(h)
        (size(Λ) == (N, r) && length(A) == p && all(size(a) == (r, r) for a in A) &&
         length(h) == N) || throw(ArgumentError("fit_dfm: params do not match N = $N, r = $r, p = $p"))
        ll = _dfm_loglik(M, Λ, A, h; strict=true)
        converged, iters, estimated = true, 0, false
    else
        Λ0, A0, h0 = start === nothing ? _dfm_start(M, r, p) : start
        A0 isa AbstractMatrix && (A0 = [A0])
        θ0 = _dfm_pack(Matrix{Float64}(Λ0), Matrix{Float64}.(A0), Vector{Float64}(h0))
        obj(θ) = (ll = _dfm_loglik(M, _dfm_unpack(θ, N, r, p)...); isfinite(ll) ? -ll : eltype(θ)(1e15))
        res = _optimize(obj, θ0; method=:lbfgs, autodiff=true, g_tol=1e-8, iterations=Int(iterations))
        Λ, A, h = _dfm_unpack(res.minimizer, N, r, p)
        # the sign normalisation: a positive leading diagonal
        D = Diagonal([Λ[j, j] < 0 ? -1.0 : 1.0 for j in 1:r])
        Λ = Λ * D
        A = [D * Aj * D for Aj in A]
        ll = -res.minimum
        converged, iters, estimated = res.converged, res.iterations, true
    end
    F, Fse = _dfm_smooth(M, Λ, A, h)
    return DFMModel(M, names, r, p, Λ, A, h, ll, n, converged, iters, estimated, :lower, F, Fse)
end

function _dfm_smooth(Y, Λ, A, h)
    T, Z, R, Q, Hm = _dfm_system(Λ, A, h)
    r = size(Λ, 2)
    P0, _ = _stationary_cov(T, R * R'; maxiter=200)
    ssm = TimeVaryingSSM{Float64}([T], [Z], [R], [Q], [Hm], size(T, 1))
    al, V = kalman_smoother(ssm, Y, zeros(size(T, 1)), P0)
    F = Matrix(al[1:r, :]')
    Fse = reduce(vcat, [sqrt.(max.(diag(Vt)[1:r], 0.0))' for Vt in V])
    return F, Fse
end

"""
    factors(m::DFMModel) -> (F, se)

The smoothed factors, **observations × factors** (statsmodels returns them
the other way round), with their standard errors.
"""
factors(m::DFMModel) = (copy(m.factors), copy(m.factors_se))

"""
    rotate_factors(m::DFMModel, G) -> DFMModel

The same fitted model in another rotation: `Λ G`, `f̃ = G' f`, `Ã = G' A G`
for an orthogonal `G` (a reflection is allowed). The likelihood is
unchanged; the result records `normalisation = :rotated`.
"""
function rotate_factors(m::DFMModel, G::AbstractMatrix)
    size(G) == (m.r, m.r) || throw(ArgumentError("rotate_factors: G must be $(m.r) × $(m.r)"))
    G'G ≈ I || throw(ArgumentError("rotate_factors: G must be orthogonal (G'G = I)"))
    Λ = m.Lambda * G
    A = [G' * Aj * G for Aj in m.A]
    F, Fse = _dfm_smooth(m.Y, Λ, A, m.H)
    ll = _dfm_loglik(m.Y, Λ, A, m.H; strict=true)
    return DFMModel(m.Y, m.names, m.r, m.p, Λ, A, copy(m.H), ll, m.nobs, m.converged,
                    m.iterations, m.estimated, :rotated, F, Fse)
end

"""
    factor_r2(m::DFMModel) -> Matrix   # N × r
    series_r2(m::DFMModel) -> Vector   # N

`factor_r2[i, j]` is the R² of series `i` regressed on smoothed factor `j`
alone, with a constant -- statsmodels' `coefficients_of_determination`.
`series_r2[i]` is the share of series `i`'s variance explained by all the
factors together, `1 − var(y − Λ f) / var(y)` -- R's `dfms` `summary()\$R2`.
Different quantities: the first splits the explanation by factor and does
not add up across factors unless they are uncorrelated.
"""
function factor_r2(m::DFMModel)
    N = size(m.Y, 2)
    out = zeros(N, m.r)
    for i in 1:N, j in 1:m.r
        ok = .!isnan.(m.Y[:, i])
        y = m.Y[ok, i]; f = m.factors[ok, j]
        out[i, j] = cor(y, f)^2
    end
    return out
end

"""
    series_r2(m::DFMModel) -> Vector   # N

The share of each series' variance explained by all the factors together,
`1 − var(y − Λ f) / var(y)` -- R's `dfms` `summary()\$R2`. See
[`factor_r2`](@ref) for the per-factor split, a different quantity.
"""
function series_r2(m::DFMModel)
    N = size(m.Y, 2)
    fitted = m.factors * m.Lambda'
    return [begin
                ok = .!isnan.(m.Y[:, i])
                1 - var(m.Y[ok, i] .- fitted[ok, i]) / var(m.Y[ok, i])
            end for i in 1:N]
end

"""
    bai_ng_ic(Y, rmax) -> NamedTuple (r, IC1, IC2, IC3)

Bai & Ng's (2002) information criteria for the number of factors, from
principal components of `Y` (pass [`standardize`](@ref)`(Y)` if the series
differ in scale): with `V(k)` the mean squared residual after `k`
components,
```
IC1 = ln V(k) + k (N+T)/(NT) ln(NT/(N+T))
IC2 = ln V(k) + k (N+T)/(NT) ln(min(N,T))
IC3 = ln V(k) + k ln(min(N,T)) / min(N,T)
```
**Not a reliable selector at small `N`.** The criteria are consistent as both
`N` and `T` grow; with a handful of series they often keep decreasing past
the true number (on a six-series fixture built with two factors they do).
"""
function bai_ng_ic(Y, rmax::Integer)
    M = Matrix{Float64}(tsmatrix(Y))
    any(isnan, M) && throw(ArgumentError("bai_ng_ic: Y contains NaN"))
    T, N = size(M)
    1 <= rmax < min(N, T) || throw(ArgumentError("bai_ng_ic: need 1 <= rmax < min(N, T)"))
    ev = eigen(Symmetric(M' * M))
    V = ev.vectors[:, sortperm(ev.values; rev=true)]
    NT, NpT, Cm = N * T, N + T, min(N, T)
    ic1, ic2, ic3 = zeros(rmax), zeros(rmax), zeros(rmax)
    for k in 1:rmax
        Vk = V[:, 1:k]
        E = M .- M * Vk * Vk'
        lv = log(sum(abs2, E) / NT)
        ic1[k] = lv + k * NpT / NT * log(NT / NpT)
        ic2[k] = lv + k * NpT / NT * log(Cm)
        ic3[k] = lv + k * log(Cm) / Cm
    end
    return (r=collect(1:rmax), IC1=ic1, IC2=ic2, IC3=ic3)
end

# --- StatsAPI -----------------------------------------------------------------

_dfm_nparams(m::DFMModel) = length(_dfm_free(size(m.Y, 2), m.r)) + m.p * m.r^2 + size(m.Y, 2)

StatsAPI.loglikelihood(m::DFMModel) = m.loglik
StatsAPI.nobs(m::DFMModel) = m.nobs
StatsAPI.dof(m::DFMModel) = _dfm_nparams(m)
StatsAPI.aic(m::DFMModel) = -2m.loglik + 2_dfm_nparams(m)
StatsAPI.bic(m::DFMModel) = -2m.loglik + log(m.nobs) * _dfm_nparams(m)

function Base.show(io::IO, m::DFMModel)
    n, N = size(m.Y)
    println(io, "Dynamic factor model: $N series, $(m.r) factor$(m.r > 1 ? "s" : ""), ",
            "factor VAR($(m.p)), n = $n")
    println(io, "  identification: ", m.normalisation === :lower ?
            "Q = I, leading loadings block lower-triangular with positive diagonal" :
            "Q = I, rotated (rotate_factors)")
    println(io, "  log-likelihood = ", round(m.loglik; digits=4), "   AIC = ",
            round(StatsAPI.aic(m); digits=4), "   BIC = ", round(StatsAPI.bic(m); digits=4),
            "   parameters = ", _dfm_nparams(m))
    print(io, m.estimated ? (m.converged ? "  optimiser converged" :
                             "  optimiser did NOT converge -- treat the estimates with care") :
                           "  evaluated at given parameters (not estimated)")
end

# --- forecasting --------------------------------------------------------------

"""
    forecast(m::DFMModel, h; level=[80.0, 95.0]) -> VARForecast

Forecast every series `h` steps ahead: the last smoothed factor state
propagated by the factor VAR, mapped through `Λ`, with the factor and
idiosyncratic variances. Returns a [`VARForecast`](@ref) (one column per
series). `predict` is the same function.
"""
function forecast(m::DFMModel, horizon::Integer; level=[80.0, 95.0])
    horizon >= 1 || throw(ArgumentError("forecast(::DFMModel): horizon must be >= 1"))
    T, Z, R, Q, Hm = _dfm_system(m.Lambda, m.A, m.H)
    P0, _ = _stationary_cov(T, R * R'; maxiter=200)
    ssm = TimeVaryingSSM{Float64}([T], [Z], [R], [Q], [Hm], size(T, 1))
    al, V = kalman_smoother(ssm, m.Y, zeros(size(T, 1)), P0)
    n, N = size(m.Y)
    a = T * al[:, n]
    P = T * V[n] * T' + R * R'
    point = zeros(horizon, N); se = zeros(horizon, N)
    for h in 1:horizon
        point[h, :] = Z * a
        se[h, :] = sqrt.(max.(diag(Z * P * Z') .+ m.H, 0.0))
        a = T * a
        P = T * P * T' + R * R'
    end
    lv = Float64.(level)
    zq = [_confidence_z(1 - l / 100) for l in lv]
    lower = cat([point .- z .* se for z in zq]...; dims=3)
    upper = cat([point .+ z .* se for z in zq]...; dims=3)
    return VARForecast(point, se, lv, lower, upper, copy(m.names), Int(horizon),
                       "DFM($(m.r) factor$(m.r > 1 ? "s" : ""), VAR($(m.p)))")
end

StatsAPI.predict(m::DFMModel, horizon::Integer; level=[80.0, 95.0]) =
    forecast(m, horizon; level=level)
