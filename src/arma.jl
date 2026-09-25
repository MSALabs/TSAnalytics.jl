export fit_arma, ArmaModel

"""
    ArmaModel <: UnivariateModel

Result of [`fit_arma`](@ref): a non-seasonal ARMA(p,q) model fit by
maximum likelihood via the `GaussianSSM` Kalman filter engine.

- `ar`/`ma`: fitted AR/MA coefficients (length `p`/`q`).
- `mean`: fitted mean, or `nothing` if `include_mean=false`. When
  present, it was estimated *jointly* with `ar`/`ma` (matching R's
  actual `include.mean=TRUE` behavior), not by pre-demeaning the series
  with the sample mean and fitting ARMA on the residual.
- `se`: standard errors, in `[ar; ma; mean]` order (matching `coef`'s
  own ordering) -- see [`fit_arma`](@ref) for the Hessian/OPG/robust choice.
- `order`: `(p, q)`.
- `method`: `:ml` or `:css_ml` -- which starting-value strategy was used.
- `se_type`: `:hessian`, `:opg`, or `:robust` -- which standard-error
  convention. `:robust` is the Huber-White/QMLE sandwich built from the
  other two (see `_robust_se`); it is *not* uniformly wider than either.
- `converged`: whether the optimizer reported convergence.
"""
struct ArmaModel <: UnivariateModel
    ar::Vector{Float64}
    ma::Vector{Float64}
    mean::Union{Nothing,Float64}
    se::Vector{Float64}
    loglik::Float64
    sigma2::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    order::Tuple{Int,Int}
    method::Symbol
    se_type::Symbol
    converged::Bool
end

"_arma_demeaned(params, yv, p, q, include_mean) -> (phi, theta, yc), all in
`eltype(params)` -- shared helper extracting the natural AR/MA
coefficients and demeaned series from a `[phi; theta; mean?]` parameter
vector, used identically by the natural-parametrization objective, the
per-observation contributions, and the CSS objective below."
function _arma_unpack(params::AbstractVector, yv::Vector{Float64}, p::Integer, q::Integer, include_mean::Bool)
    T = eltype(params)
    phi = p > 0 ? params[1:p] : T[]
    theta = q > 0 ? params[(p + 1):(p + q)] : T[]
    mu = include_mean ? params[end] : zero(T)
    yc = yv .- mu
    return phi, theta, yc
end

"_arma_natural_objective(params, yv, p, q, include_mean) -- negative
log-likelihood as a function of the *natural* (untransformed) AR/MA
coefficients and mean directly, no `partrans`/Monahan reparametrization.
Used only for the Hessian/OPG standard-error computation (Stage 6
handoff §4.5): R evaluates the Hessian at the optimum with the
transform switched off, not on the transformed objective `_optimize`
actually searched over -- taking the Hessian of the transformed
objective gives standard errors that look plausible and are quietly
wrong. `_optimize` itself is never called with this objective; only
`ForwardDiff.hessian`/`ForwardDiff.jacobian` evaluate it, at a single
fixed point (the fitted coefficients)."
function _arma_natural_objective(params::AbstractVector, yv::Vector{Float64}, p::Integer, q::Integer, include_mean::Bool)
    T = eltype(params)
    phi, theta, yc = _arma_unpack(params, yv, p, q, include_mean)
    ssm = build_statespace(phi, theta)
    loglik, sigma2, v, F, converged = kalman_filter(ssm, yc)
    return converged ? -loglik : T(1e10)
end

"_arma_loglik_contributions(params, yv, p, q, include_mean) --
per-observation log-likelihood terms (the concentrated Gaussian density
at each t), summing to the same total `-_arma_natural_objective`
returns. Needed for the OPG standard-error estimator, which requires
the per-observation score (gradient of each individual contribution),
not just the total gradient."
function _arma_loglik_contributions(params::AbstractVector, yv::Vector{Float64}, p::Integer, q::Integer, include_mean::Bool)
    T = eltype(params)
    phi, theta, yc = _arma_unpack(params, yv, p, q, include_mean)
    ssm = build_statespace(phi, theta)
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
    _hessian_se(natural_objective, params_hat) -> Vector{Float64}

R-style asymptotic standard errors: `sqrt(diag(inv(H)))` where `H` is
the Hessian of `natural_objective` (the negative log-likelihood as a
function of the *natural*, untransformed parameters -- no `partrans`/
Monahan reparametrization) at `params_hat`, via `ForwardDiff.hessian` --
exact automatic differentiation, not a numerical-differencing
approximation. Generic over `natural_objective` (a single-argument
closure) so both [`fit_arma`](@ref) (`_arma_natural_objective`) and
`fit_sarima` (Stage 6.7, its own seasonal natural-objective) share this
one implementation rather than duplicating the Hessian/error-handling
logic per model type.

Returns `NaN` entries (not a thrown error) if `H` is numerically
singular -- this genuinely happens at an invertibility-boundary optimum
(e.g. an optimizer converging to `|ma| ~ 1` from a poor starting point,
which the ML surface can have as a real local optimum, matching R's own
documented starting-value sensitivity) rather than something to paper
over with a misleadingly small pseudo-inverse-based number.
"""
function _hessian_se(natural_objective, params_hat::Vector{Float64})
    H = ForwardDiff.hessian(natural_objective, params_hat)
    vc = try
        inv(H)
    catch e
        e isa Union{LinearAlgebra.SingularException,LinearAlgebra.LAPACKException} ||
            rethrow()
        return fill(NaN, length(params_hat))
    end
    return sqrt.(max.(diag(vc), 0.0))
end

"""
    _opg_se(loglik_contributions, params_hat) -> Vector{Float64}

Python-`statsmodels`-style outer-product-of-gradients standard errors:
`sqrt(diag(inv(J'J)))` where `J` is the `n x k` Jacobian of
`loglik_contributions` (per-observation log-likelihood terms, as a
function of the natural parameters) at `params_hat`, via
`ForwardDiff.jacobian`. Generic over `loglik_contributions`, same
reasoning as `_hessian_se`. Genuinely different numbers from
`_hessian_se` -- both asymptotically valid estimators of the
same quantity, not a bug in either (see [`fit_arma`](@ref)'s docstring).
"""
function _opg_se(loglik_contributions, params_hat::Vector{Float64})
    J = ForwardDiff.jacobian(loglik_contributions, params_hat)
    vc = try
        inv(J' * J)
    catch e
        e isa Union{LinearAlgebra.SingularException,LinearAlgebra.LAPACKException} ||
            rethrow()
        return fill(NaN, length(params_hat))
    end
    return sqrt.(max.(diag(vc), 0.0))
end

"""
    _robust_se(natural_objective, loglik_contributions, params_hat) -> Vector{Float64}

Huber-White/QMLE "sandwich" standard errors: `sqrt(diag(H⁻¹ (J'J) H⁻¹))`,
assembled from the *same* two quantities [`_hessian_se`](@ref) and
[`_opg_se`](@ref) already compute -- `H` the Hessian of the negative
log-likelihood (the "bread") and `J'J` the outer product of the
per-observation score contributions (the "meat"). No new numerics: the
two existing estimators are the two halves of this one.

Valid under quasi-maximum-likelihood, i.e. when the conditional mean and
dynamics are correctly specified but the innovation distribution is not
necessarily Gaussian. Where the information-matrix equality holds
(correct specification, Gaussian errors) `H ≈ J'J` and the sandwich
collapses to `_hessian_se`'s own answer -- asserted directly in the test
suite rather than assumed.

**Not uniformly larger than the classical alternatives.** Verified
against real `rugarch` output (`handoff/robust-se-and-diagnostics-handoff.md`):
on its own reference GARCH fit the robust/classical ratio runs
`0.827`/`1.169`/`1.073` across the three parameters -- smaller for one of
them. Any test asserting `robust >= classical` would be wrong.

Returns `NaN` entries under the same singularity conditions as the other
two, rather than papering over a degenerate optimum with a pseudo-inverse.
"""
function _robust_se(natural_objective, loglik_contributions, params_hat::Vector{Float64})
    H = ForwardDiff.hessian(natural_objective, params_hat)
    J = ForwardDiff.jacobian(loglik_contributions, params_hat)
    vc = try
        Hinv = inv(H)
        Hinv * (J' * J) * Hinv
    catch e
        e isa Union{LinearAlgebra.SingularException,LinearAlgebra.LAPACKException} ||
            rethrow()
        return fill(NaN, length(params_hat))
    end
    return sqrt.(max.(diag(vc), 0.0))
end

"""
    _css_objective(raw, yc, unpack) -> scalar

Generic conditional-sum-of-squares objective: `unpack(raw)` maps the
current trial raw (Monahan/`partrans`-space) parameter vector to
`(ar, ma)` -- already-`partrans`-transformed, already-combined AR/MA
coefficient vectors -- and this function runs the standard textbook CSS
recursion against `yc` using those: `e_t = y_t - sum(ar_i*y_{t-i}) -
sum(ma_j*e_{t-j})`, treating presample `y`/`e` as zero, `css =
sum(e_t^2)`. Cheap (no Kalman filter/Lyapunov solve) relative to the
full ML objective. Shared by `_css_start_values` (plain ARMA,
`unpack` is the identity after `partrans`) and `fit_sarima`'s (Stage
6.7) seasonal warm-start (`unpack` additionally runs `combined_ar_ma`),
which differ only in how `raw` maps to `(ar, ma)`, not in the recursion
itself.
"""
function _css_objective(raw::AbstractVector, yc::Vector{Float64}, unpack)
    T = eltype(raw)
    ar, ma = unpack(raw)
    p, q = length(ar), length(ma)
    n = length(yc)
    e = zeros(T, n)
    css = zero(T)
    for t in 1:n
        pred = zero(T)
        for i in 1:p
            pred += ar[i] * (t - i >= 1 ? yc[t - i] : zero(T))
        end
        for j in 1:q
            pred += ma[j] * (t - j >= 1 ? e[t - j] : zero(T))
        end
        e[t] = yc[t] - pred
        css += e[t]^2
    end
    return css
end

"""
    _css_start_values(yc, p, q) -> Vector{Float64}

Conditional-sum-of-squares starting values for the `:css_ml` method's
AR/MA block -- see `_css_objective` for the recursion itself.
`yc` is already demeaned (using the sample mean as `:css_ml`'s starting
guess for the mean itself, refined only in the subsequent full-ML
stage, matching R's actual two-stage behavior). Returns a raw
(Monahan/`partrans`-space) parameter vector of length `p+q`, suitable as
part of `_optimize`'s starting point for the ML stage (see
`handoff/stage-6.5-arma-mle-handoff.md` §1, §3).
"""
function _css_start_values(yc::Vector{Float64}, p::Integer, q::Integer)
    p + q == 0 && return Float64[]  # nothing to warm-start (order (0,0))
    unpack(raw) = (p > 0 ? partrans(raw[1:p]) : eltype(raw)[],
                   q > 0 ? partrans(raw[(p + 1):(p + q)]) : eltype(raw)[])
    res = _optimize(raw -> _css_objective(raw, yc, unpack), zeros(p + q))
    return res.minimizer
end

"""
    fit_arma(y, order; include_mean=true, method=:ml, se_type=:hessian,
             optimizer_method=:lbfgs, start_params=nothing) -> ArmaModel

Fit a non-seasonal ARMA(p,q) model by maximum likelihood, via the
`GaussianSSM` Kalman filter (`build_statespace`/`kalman_filter`, dual-
verified against R and Python internally) wired to Stage 4.1's
`_optimize` and Stage 4.2's `partrans` Monahan reparametrization.
Verified end-to-end against real R `stats::arima()` and Python
`statsmodels.tsa.arima.model.ARIMA` on the same generated ARMA(1,1)
series (both `include_mean=false`, and -- independently, beyond the
handoff's own ground truth -- `include_mean=true` on a mean-shifted
version of the same series) -- see
`handoff/stage-6.5-arma-mle-handoff.md` §3 -- coefficients and
log-likelihood match both references to several digits.

`order = (p, q)`: AR and MA orders, either of which may be `0`
(including `order = (0, 0)`, a pure white-noise-plus-optional-mean
model -- no AR/MA parameters to estimate, matches R's
`arima(order=c(0,0,0))` exactly, verified directly). No `d`
(differencing, Stage 6.6 wraps this function with it), no seasonal terms
(Stage 6.7), no exogenous regressors (Stage 7) -- this stage's scope is
deliberately narrower than R's/Python's full model.

- `include_mean=true` (default): the mean is a free parameter, estimated
  *jointly* with the AR/MA coefficients (matching R's actual
  `include.mean=TRUE` behavior) -- **not** a one-time sample-mean
  subtraction followed by fitting ARMA on the residual, which is a
  meaningfully different (and wrong) point estimate on finite samples,
  confirmed by direct comparison against real R on a mean-shifted series.
- `method=:ml` (default): optimizes from a zero start in the
  Monahan-transformed space (mean starts from the sample mean), matching
  R's `method="ML"`. `method=:css_ml`: derives AR/MA starting values by
  conditional sum of squares first (`TSAnalytics._css_start_values`),
  then refines by the same full-ML objective -- matches the *spirit* of
  R's actual default (`"CSS-ML"`), which is meaningfully faster (~4x,
  measured) from better starting values, not a different final answer on
  well-behaved data.
- `se_type=:hessian` (default): R-style asymptotic standard errors from
  the Hessian of the log-likelihood in the *natural* parametrization,
  evaluated at the optimum (**not** the Hessian of the
  `partrans`-transformed objective actually searched over -- that gives
  standard errors that look plausible and are quietly wrong; see
  `handoff/stage-6-arima-handoff.md` §4.5). `se_type=:opg`: Python-style
  outer-product-of-gradients standard errors. **These are genuinely
  different numbers** (confirmed against real R and Python: `0.088`
  Hessian-based vs. `0.081` OPG-based, for the same fitted ARMA(1,1)
  coefficient) -- both are asymptotically valid estimators of the same
  quantity, not a bug in either; pick the one matching whichever
  reference you're validating against.

  `se_type=:robust`: the Huber-White/QMLE "sandwich",
  `sqrt(diag(H⁻¹ (J'J) H⁻¹))` -- assembled from the exact same two
  quantities the other two options use, so it costs one extra matrix
  product rather than any new numerics. Valid under quasi-maximum
  likelihood: correct mean/dynamics, not-necessarily-Gaussian
  innovations. **Neither R's `stats::arima` nor Python's `statsmodels`
  offers this for an ARIMA model at all** -- `sandwich::vcovHC` cannot
  even consume an `arima` object (it has no `terms` component), so
  there is no cross-language reference to validate against directly;
  it is instead verified by exact reduction, since for a pure linear
  model the same sandwich *is* White's HC0, matched to `1e-7` against
  real `sandwich::vcovHC(type="HC0")` via [`arx`](@ref). **It is not
  uniformly wider than the classical alternatives** -- on real
  `rugarch` output the robust/classical ratio runs `0.827`/`1.169`/
  `1.073` across three parameters. `NaN` entries in `se` (rather
  than a crash) mean the Hessian/outer-product matrix was numerically
  singular at the fitted point -- this genuinely happens at an
  invertibility-boundary local optimum (e.g. `_optimize` converging to
  `|ma| ~ 1` from a poor starting point, confirmed to occur with a plain
  zero start on some series), not something this function papers over
  with a misleadingly small number.
- `start_params`: explicit override (raw, Monahan/`partrans`-space
  vector, length `p+q` or `p+q+1` with a mean), bypassing `method`'s own
  starting-value choice.

`aic`/`bic` count `sigma2` as an estimated parameter in addition to the
AR/MA/mean coefficients (`k = p + q + (include_mean ? 1 : 0) + 1`),
matching R's actual `AIC()`/`BIC()` output exactly -- confirmed by
direct execution; a formula omitting the `+1` for `sigma2` reproduces
neither reference.

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "arma", "arma11_fit.csv"), ','; skipstart=1));

julia> m = fit_arma(y, (1, 1); include_mean=false);

julia> m.converged
true

julia> round(m.ar[1], digits=4)
0.5466
```
"""
function fit_arma(y, order::Tuple{Int,Int};
                   include_mean::Bool=true,
                   method::Symbol=:ml,
                   se_type::Symbol=:hessian,
                   optimizer_method::Symbol=:lbfgs,
                   start_params::Union{Nothing,Vector{Float64}}=nothing)
    method in (:ml, :css_ml) || throw(ArgumentError("method must be :ml or :css_ml"))
    se_type in (:hessian, :opg, :robust) ||
        throw(ArgumentError("se_type must be :hessian, :opg, or :robust"))
    p, q = order
    p >= 0 && q >= 0 || throw(ArgumentError("order must be non-negative: got ($p, $q)"))

    yv = Float64.(collect(tsvalues(y)))
    n = length(yv)
    nparam = p + q + (include_mean ? 1 : 0)
    n > nparam || throw(ArgumentError("fit_arma: not enough observations ($n) for order $order" *
                                       (include_mean ? " with a mean" : "")))
    mu0 = include_mean ? sum(yv) / n : 0.0

    x0 = if start_params !== nothing
        length(start_params) == nparam ||
            throw(ArgumentError("start_params must have length p+q$(include_mean ? "+1" : "") = $nparam"))
        start_params
    elseif method == :css_ml
        css = _css_start_values(yv .- mu0, p, q)
        include_mean ? vcat(css, mu0) : css
    else
        include_mean ? vcat(zeros(p + q), mu0) : zeros(p + q)
    end

    function objective(raw::AbstractVector)
        T = eltype(raw)
        phi = p > 0 ? partrans(raw[1:p]) : T[]
        theta = q > 0 ? partrans(raw[(p + 1):(p + q)]) : T[]
        mu = include_mean ? raw[end] : zero(T)
        ssm = build_statespace(phi, theta)
        loglik, sigma2, v, F, converged = kalman_filter(ssm, yv .- mu)
        return converged ? -loglik : T(1e10)
    end

    # order (0,0) with include_mean=false: no free parameters at all (a pure
    # white-noise model -- only sigma2 is estimated, directly from the
    # data, matching R's arima(order=c(0,0,0), include.mean=FALSE)) --
    # _optimize (Optim.jl's LBFGS) throws BoundsError on a zero-length x0,
    # confirmed by direct testing, so this case is evaluated directly
    # rather than routed through the optimizer.
    result = isempty(x0) ? (minimizer=Float64[], converged=true) :
                            _optimize(objective, x0; method=optimizer_method)
    phi_hat = p > 0 ? partrans(result.minimizer[1:p]) : Float64[]
    theta_hat = q > 0 ? partrans(result.minimizer[(p + 1):(p + q)]) : Float64[]
    mu_hat = include_mean ? result.minimizer[end] : nothing
    yc_hat = include_mean ? yv .- mu_hat : yv
    ssm = build_statespace(phi_hat, theta_hat)
    loglik, sigma2, = kalman_filter(ssm, yc_hat)

    params_hat = include_mean ? vcat(phi_hat, theta_hat, mu_hat) : vcat(phi_hat, theta_hat)
    natobj = params -> _arma_natural_objective(params, yv, p, q, include_mean)
    llcontrib = params -> _arma_loglik_contributions(params, yv, p, q, include_mean)
    se = se_type == :hessian ? _hessian_se(natobj, params_hat) :
         se_type == :opg     ? _opg_se(llcontrib, params_hat) :
                               _robust_se(natobj, llcontrib, params_hat)

    k = nparam + 1  # +1 for sigma2, matching R's actual AIC/BIC exactly
    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + k * log(n)

    return ArmaModel(phi_hat, theta_hat, mu_hat, se,
                      loglik, sigma2, aic, bic, n, order, method, se_type, result.converged)
end

function Base.show(io::IO, m::ArmaModel)
    p, q = m.order
    names = vcat(["ar$i" for i in 1:p], ["ma$i" for i in 1:q],
                 m.mean !== nothing ? ["mean"] : String[])
    coefs = vcat(m.ar, m.ma, m.mean !== nothing ? [m.mean] : Float64[])
    z = coefs ./ m.se
    pval = _chisq_ccdf.(z .^ 2, 1)
    ct = StatsBase.CoefTable(hcat(coefs, m.se, z, pval), ["Coef.", "Std. Error", "z", "Pr(>|z|)"], names)
    println(io, "ARMA(", p, ",", q, ")", m.mean !== nothing ? " with mean" : "",
            ", n=", m.nobs, " (", m.method, ", se: ", m.se_type, ")")
    println(io)
    println(io, ct)
    print(io, "Log-likelihood: ", round(m.loglik, digits=2),
          "   AIC: ", round(m.aic, digits=2), "   BIC: ", round(m.bic, digits=2))
    m.converged || print(io, "\nWARNING: optimizer did not converge")
end
