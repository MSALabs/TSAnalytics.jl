export fit_sarima, SarimaModel

"""
    SarimaModel <: UnivariateModel

Result of [`fit_sarima`](@ref): seasonal ARIMA(p,d,q)(P,D,Q)_s.

- `phi`/`theta`: regular AR/MA coefficients (length `p`/`q`).
- `Phi`/`Theta`: seasonal AR/MA coefficients (length `P`/`Q`).
- `mean`: fitted mean, or `nothing` if not estimated (see
  [`fit_sarima`](@ref) for when it's silently dropped, matching R).
- `se`: standard errors, in `[phi; theta; Phi; Theta; mean?]` order
  (matching this project's own already-verified optimizer parameter
  layout).
- `order`/`seasonal_order`: `(p,d,q)` / `(P,D,Q,s)`.
- `method`/`se_type`/`converged`: same meaning as [`ArmaModel`](@ref).
"""
struct SarimaModel <: UnivariateModel
    phi::Vector{Float64}
    theta::Vector{Float64}
    Phi::Vector{Float64}
    Theta::Vector{Float64}
    mean::Union{Nothing,Float64}
    se::Vector{Float64}
    loglik::Float64
    sigma2::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    order::Tuple{Int,Int,Int}
    seasonal_order::Tuple{Int,Int,Int,Int}
    method::Symbol
    se_type::Symbol
    converged::Bool
end

StatsAPI.loglikelihood(m::SarimaModel) = m.loglik
StatsAPI.aic(m::SarimaModel) = m.aic
StatsAPI.bic(m::SarimaModel) = m.bic
StatsAPI.coef(m::SarimaModel) = vcat(m.phi, m.theta, m.Phi, m.Theta)

"""
    StatsAPI.nobs(m::SarimaModel) -> Int

`n - d - D*s`, extending [`ArimaModel`](@ref)'s `n - d` convention to
seasonal differencing -- matches R's `stats::arima()` exactly (verified
directly on a real SARIMA(1,0,0)(1,1,0)₁₂ fit: `96 - 0 - 1*12 = 84`).
**Not** Python's `statsmodels SARIMAX`, whose `nobs` is the full `n`
(`96` on the same series) for the same diffuse-state-augmentation
reason as the non-seasonal case.
"""
StatsAPI.nobs(m::SarimaModel) = m.nobs

"_sarima_unpack(params, yv, p, q, P, Q, s, include_mean) -> (ar, ma, yc) --
shared by the natural-parametrization objective and per-observation
contributions: extracts the four raw coefficient blocks (already in the
*natural*, untransformed space -- no `partrans` here, that only happens
in `fit_sarima`'s own optimizer objective), combines them via
`combined_ar_ma` into the state-space `ar`/`ma`, and demeans `yv`."
function _sarima_unpack(params::AbstractVector, yv::Vector{Float64},
                         p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer, include_mean::Bool)
    T = eltype(params)
    phi   = p > 0 ? params[1:p] : T[]
    theta = q > 0 ? params[(p + 1):(p + q)] : T[]
    Phi   = P > 0 ? params[(p + q + 1):(p + q + P)] : T[]
    Theta = Q > 0 ? params[(p + q + P + 1):(p + q + P + Q)] : T[]
    mu = include_mean ? params[end] : zero(T)
    ar, ma = combined_ar_ma(phi, Phi, theta, Theta, s)
    return ar, ma, yv .- mu
end

"_sarima_natural_objective(params, yv, p, q, P, Q, s, include_mean) --
negative log-likelihood as a function of the *natural* (untransformed)
phi/theta/Phi/Theta/mean, for `_hessian_se`/`_opg_se`
(Stage 6.7 reuses these generic Stage 6.5 helpers directly, not a
seasonal-specific reimplementation)."
function _sarima_natural_objective(params::AbstractVector, yv::Vector{Float64},
                                    p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer, include_mean::Bool)
    T = eltype(params)
    ar, ma, yc = _sarima_unpack(params, yv, p, q, P, Q, s, include_mean)
    ssm = build_statespace(ar, ma)
    loglik, sigma2, v, F, converged = kalman_filter(ssm, yc)
    return converged ? -loglik : T(1e10)
end

"_sarima_loglik_contributions(params, yv, p, q, P, Q, s, include_mean) --
per-observation log-likelihood terms, for the OPG standard-error
estimator -- see `_arma_loglik_contributions` (Stage 6.5) for the
identical concentrated-Gaussian-density formula, here evaluated on the
seasonal-combined state-space model."
function _sarima_loglik_contributions(params::AbstractVector, yv::Vector{Float64},
                                       p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer, include_mean::Bool)
    T = eltype(params)
    ar, ma, yc = _sarima_unpack(params, yv, p, q, P, Q, s, include_mean)
    ssm = build_statespace(ar, ma)
    loglik, sigma2, v, F, converged = kalman_filter(ssm, yc)
    n = length(yv)
    contribs = Vector{T}(undef, n)
    if !converged
        fill!(contribs, T(-1e10) / n)
        return contribs
    end
    for t in 1:n
        contribs[t] = -0.5 * (log(2π) + log(sigma2) + log(F[t]) + v[t]^2 / (sigma2 * F[t]))
    end
    return contribs
end

"""
    _sarima_css_start_values(yc, p, q, P, Q, s) -> Vector{Float64}

Seasonal analogue of `_css_start_values` (Stage 6.5): the same
CSS recursion (`_css_objective`), but `unpack` runs `partrans` on all
four raw blocks and then `combined_ar_ma` to get the trial `(ar, ma)`
each iteration -- the CSS warm-start for `:css_ml` needs the same
seasonal composition the full ML objective does, not a plain-ARMA
approximation of it.
"""
function _sarima_css_start_values(yc::Vector{Float64}, p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer)
    p + q + P + Q == 0 && return Float64[]
    function unpack(raw)
        T = eltype(raw)
        phi   = p > 0 ? partrans(raw[1:p]) : T[]
        theta = q > 0 ? partrans(raw[(p + 1):(p + q)]) : T[]
        Phi   = P > 0 ? partrans(raw[(p + q + 1):(p + q + P)]) : T[]
        Theta = Q > 0 ? partrans(raw[(p + q + P + 1):(p + q + P + Q)]) : T[]
        return combined_ar_ma(phi, Phi, theta, Theta, s)
    end
    res = _optimize(raw -> _css_objective(raw, yc, unpack), zeros(p + q + P + Q))
    return res.minimizer
end

"""
    fit_sarima(y, order::Tuple{Int,Int,Int}, seasonal_order::Tuple{Int,Int,Int,Int};
               include_mean=true, method=:ml, se_type=:hessian,
               optimizer_method=:lbfgs, start_params=nothing) -> SarimaModel

Fit seasonal ARIMA(p,d,q)(P,D,Q)_s: seasonally- and regularly-differences
`y` (Stage 1.1's `diff`, seasonal difference applied first, matching its
own documented `diff(diff(y, s), 1)` composition), then fits the
resulting stationary series by maximum likelihood -- regular and
seasonal AR/MA polynomials are multiplied together every optimizer
iteration via `combined_ar_ma` (already dual-verified for 20
seasonal cases on its own) before handing off to the same
`build_statespace`/`kalman_filter` core [`fit_arma`](@ref) (Stage 6.5)
uses. Parameter vector ordering is `[phi; theta; Phi; Theta]`, this
project's own already-empirically-verified optimizer layout (see
`handoff/stage-6.7-sarima-handoff.md`).

Verified end-to-end against real R `stats::arima(seasonal=...)` and
real Python `statsmodels.tsa.statespace.sarimax.SARIMAX` on a genuine
SARIMA(1,0,0)(1,1,0)₁₂ series, and independently extended (beyond the
handoff's own flagged gap) with a dedicated case exercising all four
polynomial blocks (`p,q,P,Q > 0`) to validate the parameter-unpacking
arithmetic the single-seasonal-term case can't exercise.

**Not a thin wrapper over [`fit_arima`](@ref)** (unlike how 6.6 wrapped
6.5) -- `combined_ar_ma` has to run *inside* the optimizer's loop, since
the combined polynomial changes every iteration as the four blocks are
searched independently, so this stage needs its own objective function.
What *is* reused directly: `partrans`, `build_statespace`,
`kalman_filter`, `_optimize`, `_hessian_se`/`_opg_se`, and the CSS
recursion (`_css_objective`) -- every actual numerical primitive, just
recomposed for four parameter blocks instead of two.

- `include_mean` is silently forced to `false` whenever `d > 0` **or**
  `D > 0` (confirmed directly against real R: `include.mean=TRUE` with
  `D=1, d=0` gives bit-identical coefficients to `include.mean=FALSE` --
  the same finding as [`fit_arima`](@ref)'s `d>0` case, now confirmed to
  extend to seasonal differencing too, not just assumed).
- `nobs = n - d - D*s`, matching R exactly -- see
  `StatsAPI.nobs`'s own docstring below.
- All other keywords match [`fit_arma`](@ref)'s exactly (same reasoning
  as [`fit_arima`](@ref): this stage's job is seasonal composition, not
  new fitting options).

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "sarima", "sarima_shared.csv"), ','; skipstart=1));

julia> m = fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); include_mean=false);

julia> m.converged
true

julia> round(m.phi[1], digits=4)
0.3424

julia> round(m.Phi[1], digits=4)
0.3243

julia> m.nobs  # n - D*s = 96 - 1*12
84
```
"""
function fit_sarima(y, order::Tuple{Int,Int,Int}, seasonal_order::Tuple{Int,Int,Int,Int};
                     include_mean::Bool=true,
                     method::Symbol=:ml,
                     se_type::Symbol=:hessian,
                     optimizer_method::Symbol=:lbfgs,
                     start_params::Union{Nothing,Vector{Float64}}=nothing)
    method in (:ml, :css_ml) || throw(ArgumentError("method must be :ml or :css_ml"))
    se_type in (:hessian, :opg) || throw(ArgumentError("se_type must be :hessian or :opg"))
    p, d, q = order
    P, D, Q, s = seasonal_order
    p >= 0 && d >= 0 && q >= 0 || throw(ArgumentError("order must be non-negative: got $order"))
    P >= 0 && D >= 0 && Q >= 0 || throw(ArgumentError("seasonal_order must be non-negative: got $seasonal_order"))
    s >= 1 || throw(ArgumentError("seasonal period s must be >= 1, got $s"))
    (P > 0 || D > 0 || Q > 0) && s < 2 &&
        throw(ArgumentError("seasonal_order needs period s >= 2 when P, D, or Q > 0 (got s=$s)"))

    yv = Float64.(collect(tsvalues(y)))
    yd = yv
    D > 0 && (yd = diff(yd, s; differences=D))
    d > 0 && (yd = diff(yd, 1; differences=d))
    n = length(yd)

    include_mean_effective = (d > 0 || D > 0) ? false : include_mean
    nparam = p + q + P + Q + (include_mean_effective ? 1 : 0)
    n > nparam || throw(ArgumentError("fit_sarima: not enough observations ($n after differencing) " *
                                       "for order $order, seasonal_order $seasonal_order" *
                                       (include_mean_effective ? " with a mean" : "")))
    mu0 = include_mean_effective ? sum(yd) / n : 0.0

    x0 = if start_params !== nothing
        length(start_params) == nparam ||
            throw(ArgumentError("start_params must have length p+q+P+Q" *
                                 (include_mean_effective ? "+1" : "") * " = $nparam"))
        start_params
    elseif method == :css_ml
        css = _sarima_css_start_values(yd .- mu0, p, q, P, Q, s)
        include_mean_effective ? vcat(css, mu0) : css
    else
        include_mean_effective ? vcat(zeros(p + q + P + Q), mu0) : zeros(p + q + P + Q)
    end

    function objective(raw::AbstractVector)
        T = eltype(raw)
        phi   = p > 0 ? partrans(raw[1:p]) : T[]
        theta = q > 0 ? partrans(raw[(p + 1):(p + q)]) : T[]
        Phi   = P > 0 ? partrans(raw[(p + q + 1):(p + q + P)]) : T[]
        Theta = Q > 0 ? partrans(raw[(p + q + P + 1):(p + q + P + Q)]) : T[]
        mu = include_mean_effective ? raw[end] : zero(T)
        ar, ma = combined_ar_ma(phi, Phi, theta, Theta, s)
        ssm = build_statespace(ar, ma)
        loglik, sigma2, v, F, converged = kalman_filter(ssm, yd .- mu)
        return converged ? -loglik : T(1e10)
    end

    # order (0,·,0) seasonal_order (0,·,0) with include_mean effectively
    # false: no free parameters at all (pure white noise, only sigma2
    # estimated directly from the data) -- _optimize (Optim.jl's LBFGS)
    # throws BoundsError on a zero-length x0, same case already guarded
    # in fit_arma; evaluated directly here rather than routed through
    # the optimizer.
    result = isempty(x0) ? (minimizer=Float64[], converged=true) :
                            _optimize(objective, x0; method=optimizer_method)
    n0 = result.minimizer
    phi_hat   = p > 0 ? partrans(n0[1:p]) : Float64[]
    theta_hat = q > 0 ? partrans(n0[(p + 1):(p + q)]) : Float64[]
    Phi_hat   = P > 0 ? partrans(n0[(p + q + 1):(p + q + P)]) : Float64[]
    Theta_hat = Q > 0 ? partrans(n0[(p + q + P + 1):(p + q + P + Q)]) : Float64[]
    mu_hat = include_mean_effective ? n0[end] : nothing
    yc_hat = include_mean_effective ? yd .- mu_hat : yd
    ar, ma = combined_ar_ma(phi_hat, Phi_hat, theta_hat, Theta_hat, s)
    ssm = build_statespace(ar, ma)
    loglik, sigma2, = kalman_filter(ssm, yc_hat)

    params_hat = include_mean_effective ? vcat(phi_hat, theta_hat, Phi_hat, Theta_hat, mu_hat) :
                                            vcat(phi_hat, theta_hat, Phi_hat, Theta_hat)
    se = se_type == :hessian ?
         _hessian_se(params -> _sarima_natural_objective(params, yd, p, q, P, Q, s, include_mean_effective), params_hat) :
         _opg_se(params -> _sarima_loglik_contributions(params, yd, p, q, P, Q, s, include_mean_effective), params_hat)

    k = nparam + 1  # +1 for sigma2, matching R's actual AIC/BIC exactly (Stage 6.5's finding)
    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + k * log(n)

    return SarimaModel(phi_hat, theta_hat, Phi_hat, Theta_hat, mu_hat, se,
                        loglik, sigma2, aic, bic, n, order, seasonal_order,
                        method, se_type, result.converged)
end

function Base.show(io::IO, m::SarimaModel)
    p, q, P, Q = length(m.phi), length(m.theta), length(m.Phi), length(m.Theta)
    _, d, _ = m.order
    _, D, _, s = m.seasonal_order
    println(io, "ARIMA(", p, ",", d, ",", q, ")(", P, ",", D, ",", Q, ")[", s, "]",
            m.mean !== nothing ? " with mean" : "", ", n=", m.nobs,
            " (", m.method, ", se: ", m.se_type, ")")
    names = vcat(["ar$i" for i in 1:p], ["ma$i" for i in 1:q],
                 ["sar$i" for i in 1:P], ["sma$i" for i in 1:Q],
                 m.mean !== nothing ? ["mean"] : String[])
    coefs = vcat(m.phi, m.theta, m.Phi, m.Theta, m.mean !== nothing ? [m.mean] : Float64[])
    z = coefs ./ m.se
    pval = _chisq_ccdf.(z .^ 2, 1)
    ct = StatsBase.CoefTable(hcat(coefs, m.se, z, pval), ["Coef.", "Std. Error", "z", "Pr(>|z|)"], names)
    println(io, ct)
    print(io, "Log-likelihood: ", round(m.loglik, digits=2),
          "   AIC: ", round(m.aic, digits=2), "   BIC: ", round(m.bic, digits=2))
    m.converged || print(io, "\nWARNING: optimizer did not converge")
end
