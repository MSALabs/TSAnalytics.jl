export auto_arimax

_nobs_model(m::ArimaxModel) = m.nobs
_nobs_model(m::SarimaxModel) = m.nobs
_nparam_model(m::ArimaxModel) = length(m.beta) + length(m.arma.ar) + length(m.arma.ma)
_nparam_model(m::SarimaxModel) = length(m.beta) + length(m.arma.phi) + length(m.arma.theta) +
                                  length(m.arma.Phi) + length(m.arma.Theta)

"_try_fit_x -- like `_try_fit` (Stage 6.8) but fits `fit_arimax`/
`fit_sarimax` (`model=:mle`, per section 3 of the handoff -- `auto_arimax`
searches the fixed-coefficient model only, never `:tvss`, since comparing
`:tvss` candidates by AICc across a search grid would repeat the exact
model-vs-model comparison error Stage 8.3 flagged as incorrect) with
`exog` threaded through. Same infeasible-candidate-skipping behavior:
`ArgumentError` (too few observations after differencing, etc.) is
caught and treated as \"this candidate can't be fit\", not a search
failure, matching R/`pmdarima`'s own behavior."
function _try_fit_x(yv, p, d, q, P, D, Q, s, seasonal, exog, include_mean, method, se_type, optimizer_method)
    try
        return seasonal ?
               fit_sarimax(yv, (p, d, q), (P, D, Q, s), exog; include_mean=include_mean, model=:mle, method=method,
                           se_type=se_type, optimizer_method=optimizer_method) :
               fit_arimax(yv, (p, d, q), exog; include_mean=include_mean, model=:mle, method=method,
                          se_type=se_type, optimizer_method=optimizer_method)
    catch e
        e isa ArgumentError || rethrow()
        return nothing
    end
end

"_ols_residuals(y, exog) -> Vector{Float64} -- ordinary least squares of
`y` on `exog` *with* an intercept, regardless of this function's own
`include_mean` keyword (a separate, preliminary step for differencing-
order detection only, not tied to the final fitted model's own mean
treatment). Matches `pmdarima`'s actual source exactly (confirmed
directly, not from documentation): `LinearRegression().fit(X, y)` uses
scikit-learn's default `fit_intercept=True`."
function _ols_residuals(y::Vector{Float64}, exog::Matrix{Float64})
    Xd = hcat(exog, ones(length(y)))
    beta_ols = Xd \ y
    return y .- Xd * beta_ols
end

"""
    auto_arimax(y, exog; d=nothing, D=nothing, max_p=5, max_q=5, max_P=2, max_Q=2,
                max_order=5, max_d=2, max_D=1, seasonal=false, m=1,
                information_criterion=:aicc, alpha=0.05, stepwise=true,
                parallel=true, include_mean=true, method=:ml, se_type=:hessian,
                optimizer_method=:lbfgs, trace=false) -> Union{ArimaxModel,SarimaxModel}

Automatic order selection for [`fit_arimax`](@ref)/[`fit_sarimax`](@ref)
(Stage 8.3), extending [`auto_arima`](@ref) (Stage 6.8) to the case where
`exog` is present. Shares `auto_arima`'s own search algorithms exactly
(`_stepwise_search`/`_exhaustive_search`, generalized during this stage
to take a `fitfn(p,q,P,Q)` closure instead of being hardcoded to
`fit_arima`/`fit_sarima`) -- this function contributes only the
exog-aware differencing-order detection and the `model=:mle`-only
fitting closure, not a new search algorithm.

**Searches `model=:mle` only** -- confirmed directly from `pmdarima`'s
own full signature that `time_varying_regression` is not a first-class
`auto_arima` parameter at all (only reachable via a generic
`sarimax_kwargs={}` passthrough), so this project adopts the same scope.
Searching `model=:tvss` candidates too would also be conceptually
murkier: per [`fit_arimax`](@ref)'s own finding, `:mle` and `:tvss` are
different *models*, not different computations of the same estimate, so
comparing their information-criterion values across a search grid would
repeat that exact comparison error.

**`d`/`D` must be detected on the residuals of `y` regressed on `exog`,
never on raw `y`** -- confirmed directly from `pmdarima`'s actual source
(not documentation): `xx = y - LinearRegression().fit(X, y).predict(X)`,
then the unit-root tests run on `xx`. This guards against the classic
spurious-regression failure mode where a trending regressor makes `y`
*look* non-stationary even though it's stationary once the regressor's
effect is removed. **This guard is real but not perfect**, verified
directly: on a genuine cointegration setup (`x` a random walk, `y`
stationary *given* `x`'s effect removed -- Stock 1987's superconsistency
setting), real `pmdarima` itself selects `d=1`, not the naively-expected
`d=0` -- finite-sample differencing-order detection in a near-
cointegrated case remains genuinely hard, the residual-based correction
protects against the *worse* failure mode (testing raw `y` directly)
without solving differencing detection perfectly. `seasonal=true` still
requires `D` passed explicitly -- inherits `auto_arima`'s own honest
limitation (no Canova-Hansen/OCSB seasonal unit-root test exists in this
project yet).

**AICc's parameter count includes `k_exog`** -- confirmed directly by
inspecting a real fitted parameter vector (`[beta, ar1, ma1, sigma2]`,
4 entries for an ARIMA(1,1,1)+1-exog model): each exogenous regressor
(and the intercept column, when `include_mean=true` survives -- see
[`fit_arimax`](@ref)'s own `d>0`/`D>0` forced-off convention) is a real
estimated parameter and counts toward the penalty exactly like `beta`'s
own `length` already reflects, no separate accounting needed.

`include_mean`/`method`/`se_type`/`optimizer_method`/`stepwise`/
`parallel`/`trace` all carry [`auto_arima`](@ref)'s own exact meaning --
see its docstring. No `test` keyword (unlike the handoff's own initially
proposed signature) -- `auto_arima` itself has none either (always
`kpss_test`), and this function's own stated design goal is matching
`auto_arima`'s signature shape exactly, not introducing a new knob
`auto_arima` doesn't have.

Verified against real Python `pmdarima.auto_arima` on the cointegration-
style case above -- see
`test/verification/autoarimax/auto-arimax-ground-truth-transcript.txt`.

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> d = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "autoarimax", "auto_arimax_shared.csv"), ','; skipstart=1);

julia> y, x = d[:, 1], reshape(d[:, 2], :, 1);

julia> m = auto_arimax(y, x; max_p=3, max_q=3, max_d=2);

julia> m isa ArimaxModel
true

julia> m.d   # matches real pmdarima's own selection on this cointegration-style series
1
```
"""
function auto_arimax(y, exog; d::Union{Nothing,Integer}=nothing, D::Union{Nothing,Integer}=nothing,
                      max_p::Integer=5, max_q::Integer=5, max_P::Integer=2, max_Q::Integer=2,
                      max_order::Integer=5, max_d::Integer=2, max_D::Integer=1,
                      seasonal::Bool=false, m::Integer=1,
                      information_criterion::Symbol=:aicc, alpha::Real=0.05,
                      stepwise::Bool=true, parallel::Bool=true,
                      include_mean::Bool=true, method::Symbol=:ml, se_type::Symbol=:hessian,
                      optimizer_method::Symbol=:lbfgs, trace::Bool=false)
    information_criterion in (:aic, :aicc, :bic) ||
        throw(ArgumentError("information_criterion must be :aic, :aicc, or :bic"))
    max_p >= 0 && max_q >= 0 && max_P >= 0 && max_Q >= 0 ||
        throw(ArgumentError("max_p/max_q/max_P/max_Q must be non-negative"))
    max_order >= 0 || throw(ArgumentError("max_order must be non-negative"))
    max_d >= 0 && max_D >= 0 || throw(ArgumentError("max_d/max_D must be non-negative"))
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0, 1)"))

    Dsel = 0
    s = 1
    if seasonal
        m >= 2 || throw(ArgumentError("seasonal=true requires m >= 2, got m=$m"))
        D === nothing && throw(ArgumentError(
            "auto_arimax: automatic seasonal-differencing-order (D) detection is not yet " *
            "implemented -- no Canova-Hansen/OCSB seasonal unit-root test exists in this " *
            "project yet, the same limitation as auto_arima (Stage 6.8). Pass D explicitly " *
            "for seasonal=true."))
        D >= 0 || throw(ArgumentError("D must be non-negative"))
        D <= max_D || throw(ArgumentError("D=$D exceeds max_D=$max_D"))
        Dsel = D
        s = m
    end

    yv = Float64.(collect(tsvalues(y)))
    n0 = length(yv)
    Xmat = exog isa AbstractMatrix ? Float64.(exog) : reshape(Float64.(collect(exog)), :, 1)
    size(Xmat, 1) == n0 ||
        throw(ArgumentError("exog must have the same number of rows as y (got $(size(Xmat, 1)) vs $n0)"))

    yseas = Dsel > 0 ? diff(yv, s; differences=Dsel) : yv
    Xseas = Dsel > 0 ? reduce(hcat, [diff(Xmat[:, j], s; differences=Dsel) for j in 1:size(Xmat, 2)]) : Xmat

    dsel = d === nothing ? _select_d(_ols_residuals(yseas, Xseas), max_d, alpha) : Int(d)
    dsel >= 0 || throw(ArgumentError("d must be non-negative"))
    dsel <= max_d || throw(ArgumentError("d=$dsel exceeds max_d=$max_d"))

    fitfn(p, q, P, Q) = _try_fit_x(yv, p, dsel, q, P, Dsel, Q, s, seasonal, Xmat, include_mean, method, se_type,
                                    optimizer_method)

    return stepwise ?
           _stepwise_search(fitfn, seasonal, max_p, max_q, max_P, max_Q, max_order, information_criterion, trace) :
           _exhaustive_search(fitfn, seasonal, max_p, max_q, max_P, max_Q, max_order, information_criterion,
                               parallel, trace)
end
