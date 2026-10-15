# ---------------------------------------------------------------------------
# Lag-feature regression -- "the ML reduction" (Stage 12.3,
# handoff/stage 12/docs/stage-12.3-lag-feature-regression-handoff.md).
#
# Not a second lag-matrix builder: the design is arx's own `_arx_design`, and
# the recursive forecast is forecast(::ARXModel)'s own row recursion. What is
# added is the one thing arx cannot do -- any regressor, injected as a pair
# of functions -- the direct strategy, and lagged rolling features. With OLS
# injected (the default) the recursive strategy IS arx + forecast, which four
# references (mlforecast, sktime, forecastML, and arx itself) agree on to
# 2e-15.
# ---------------------------------------------------------------------------

export lag_regression, LagRegressionModel, RollingFeature, rolling_feature

"""
    RollingFeature

A summary of `y` over a window that ends `lag` periods before the target,
built by [`rolling_feature`](@ref) for [`lag_regression`](@ref).
"""
struct RollingFeature
    stat::Symbol
    lag::Int
    window::Int
end

const _ROLLING_STATS = (:mean, :std, :min, :max, :expanding_mean)

"""
    rolling_feature(stat, lag, window=0) -> RollingFeature

A rolling summary of `y` as a regression feature: `stat` (`:mean`, `:std`,
`:min`, `:max`) of `y` over the `window` values ending at `t − lag`, or
`:expanding_mean`, the mean of everything up to `t − lag` (`window` must
then be 0). **`lag` must be at least 1** -- a window ending at `t` contains
the value being predicted. That is refused here, at construction, as
`mlforecast` does; `sktime`'s `WindowSummarizer` accepts it, and its
"lag 0" mean feature *is* `y_t` (an in-sample R² of exactly 1, no warning).

Under the recursive strategy a rolling feature is **recomputed from the
forecasts** at each step, as both Python references do: at step 4 of a
3-wide rolling mean of lag 1, the mean is of three forecasts.
"""
function rolling_feature(stat::Symbol, lag::Integer, window::Integer=0)
    stat in _ROLLING_STATS || throw(ArgumentError(
        "rolling_feature: stat must be one of $(join(_ROLLING_STATS, ", ")), got :$stat"))
    lag >= 1 || throw(ArgumentError(
        "rolling_feature: lag must be >= 1 -- a window ending at the target's own period " *
        "contains the value being predicted, so it would leak the target (lag = $lag)"))
    if stat === :expanding_mean
        window == 0 || throw(ArgumentError("rolling_feature: an expanding mean takes no window"))
    else
        window >= (stat === :std ? 2 : 1) || throw(ArgumentError(
            "rolling_feature: window must be >= $(stat === :std ? 2 : 1) for :$stat"))
    end
    return RollingFeature(stat, Int(lag), Int(window))
end

_rf_name(f::RollingFeature, shift) = f.stat === :expanding_mean ? "exp_mean(L$(f.lag + shift))" :
                                     "roll_$(f.stat)(L$(f.lag + shift),w$(f.window))"
_rf_reach(f::RollingFeature) = f.stat === :expanding_mean ? f.lag : f.lag + f.window - 1

function _rf_eval(f::RollingFeature, shift)
    L = f.lag + shift
    w = f.window
    stat = f.stat
    return (yv, t) -> begin
        local v = stat === :expanding_mean ? view(yv, 1:(t - L)) : view(yv, (t - L - w + 1):(t - L))
        stat === :mean ? mean(v) : stat === :std ? std(v) : stat === :min ? minimum(v) :
        stat === :max ? maximum(v) : mean(v)
    end
end

# OLS, the default regressor: the arx estimator, so the reduction can be
# checked against arx exactly
_lagreg_ols_fit(X, y) = X \ y
_lagreg_ols_predict(b, X) = X * b

"""
    LagRegressionModel

A fitted lag-feature regression, from [`lag_regression`](@ref): the
`strategy`, one fitted model per horizon for `:direct` (one in all for
`:recursive`), each with its column `names`, and the series and regressors
needed to forecast. `models` holds whatever the injected `fit` returned --
for the OLS default, the coefficient vector.
"""
struct LagRegressionModel
    strategy::Symbol
    models::Vector{Any}
    names::Vector{Vector{String}}
    nobs::Vector{Int}
    fit_fn::Any
    predict_fn::Any
    y::Vector{Float64}
    lags::Vector{Int}
    trend::Symbol
    seasonal::Bool
    period::Union{Nothing,Int}
    exog::Union{Nothing,Matrix{Float64}}
    past_exog::Union{Nothing,Matrix{Float64}}
    past_lags::Vector{Int}
    features::Vector{RollingFeature}
    horizon::Int
end

"""
    lag_regression(y, lags; fit=(X, y) -> X \\ y, predict=(model, X) -> X * model,
                   strategy=:recursive, horizon=nothing, trend=:c, seasonal=false,
                   period=nothing, exog=nothing, past_exog=nothing, past_lags=1,
                   features=RollingFeature[], same_rows=false) -> LagRegressionModel

Forecasting as regression on lagged features -- the "ML reduction" of
`mlforecast`, `sktime`'s `make_reduction` and R's `forecastML` -- with
**any regressor**, injected as two functions:

- `fit(X, y) -> model`, given the design matrix and the target;
- `predict(model, X) -> vector`, one prediction per row of `X`.

That is `forecastML`'s contract (a function, not an estimator object); the
Python references require an object with `fit`, `predict` and `get_params`,
the last only so they can clone it. The default is ordinary least squares,
and then **`strategy = :recursive` is exactly [`arx`](@ref) with
[`forecast`](@ref)** -- the same design, built by the same code, so the two
agree to rounding.

The features are those of `arx`: deterministic terms (`trend`,
`seasonal`/`period`), the lags of `y` (`lags`, an integer or any subset),
known-future regressors `exog` (contemporaneous; their future values are
required to forecast), past-only regressors `past_exog` at `past_lags`,
and lagged rolling summaries from [`rolling_feature`](@ref). Lag 0 is
refused everywhere.

**Strategies.**
- `:recursive`: one model for one step ahead, iterated, each forecast fed
  back as the next step's lag (and rolling features recomputed from it).
- `:direct` (needs `horizon = H`): a separate model for each `h = 1…H`,
  each on the features available at the forecast origin -- lag `ℓ` for
  step `h` is `y_{t−ℓ−h+1}` relative to its target. This is
  **origin-anchored**, the convention of `sktime`: the same feature set at
  every horizon, so coefficients are comparable across `h`. `forecastML`
  anchors to the target instead and drops lags below `h`; the two agree
  only at `h = H`. Each horizon uses every row it can (`sktime`'s
  `windows_identical=False`), so the `h = 1` model is the recursive model;
  `same_rows = true` restricts every horizon to the forecast origins at
  which all `H` targets exist -- the earliest `n − reach − H + 1` of them
  (`sktime`'s `make_reduction` default, `windows_identical=True`).

Two other strategies are deliberately not offered. **DirRec** (each
horizon also given the earlier horizons' predictions) and
**multi-output** (all horizons in one model) are identical to `:direct`
under least squares, by orthogonality (measured `4.4e-16` and `2.2e-16`);
they differ only for a non-linear regressor, and add nothing a user
cannot build from `fit`.

[`forecast`](@ref) returns point forecasts only: an injected regressor
carries no error model, and no reference offers intervals for a reduction
either. For intervals from the linear model, use `arx`, whose analytic
standard errors none of the references have.

```jldoctest
julia> using TSAnalytics, DelimitedFiles, LinearAlgebra

julia> y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1));

julia> r = lag_regression(y, 2);

julia> forecast(r, 3).point ≈ forecast(arx(y, 2), 3).point
true

julia> ridge(λ) = ((X, y) -> (X'X + λ * I) \\ (X'y));

julia> forecast(lag_regression(y, 2; fit=ridge(0.0)), 3).point ≈ forecast(r, 3).point
true
```
"""
function lag_regression(y, lags::Union{Integer,AbstractVector{<:Integer}};
                        fit=_lagreg_ols_fit, predict=_lagreg_ols_predict,
                        strategy::Symbol=:recursive, horizon::Union{Nothing,Integer}=nothing,
                        trend::Symbol=:c, seasonal::Bool=false,
                        period::Union{Nothing,Integer}=nothing, exog=nothing, past_exog=nothing,
                        past_lags::Union{Integer,AbstractVector{<:Integer}}=1,
                        features::AbstractVector{RollingFeature}=RollingFeature[],
                        same_rows::Bool=false)
    fit isa Function || throw(ArgumentError(
        "lag_regression: `fit` must be a function (X, y) -> model, not a fitted object " *
        "($(typeof(fit))); it is refitted for every horizon"))
    predict isa Function || throw(ArgumentError(
        "lag_regression: `predict` must be a function (model, X) -> vector"))
    strategy in (:recursive, :direct) || throw(ArgumentError(
        "lag_regression: strategy must be :recursive or :direct (DirRec and multi-output " *
        "equal :direct under least squares; see the docstring)"))
    trend in (:n, :c, :t, :ct) || throw(ArgumentError("lag_regression: trend must be :n, :c, :t or :ct"))
    seasonal && (period === nothing || period < 2) &&
        throw(ArgumentError("lag_regression: seasonal=true requires period >= 2"))
    H = if strategy === :direct
        horizon === nothing && throw(ArgumentError("lag_regression: strategy=:direct needs `horizon`"))
        horizon >= 1 || throw(ArgumentError("lag_regression: horizon must be >= 1"))
        Int(horizon)
    else
        horizon === nothing || throw(ArgumentError(
            "lag_regression: `horizon` is for strategy=:direct; a recursive model forecasts any horizon"))
        1
    end

    yv = collect(Float64, tsvalues(y))
    n = length(yv)
    any(!isfinite, yv) && throw(ArgumentError("lag_regression: missing/non-finite values in y"))
    lagset = _arx_lagset(lags, "lags")
    ex = _arx_regressors(exog, n, "exog")
    pex = _arx_regressors(past_exog, n, "past_exog")
    plags = pex === nothing ? Int[] : _arx_lagset(past_lags, "past_lags")
    reach = max(maximum(lagset), isempty(plags) ? 0 : maximum(plags),
                isempty(features) ? 0 : maximum(_rf_reach, features))

    models = Any[]
    allnames = Vector{String}[]
    nobs = Int[]
    for h in 1:H
        shift = strategy === :direct ? h - 1 : 0
        hb = reach + shift
        # same_rows: only the origins at which every horizon's target exists --
        # the earliest ones, so horizon h's targets run (reach+h):(n-H+h)
        last = same_rows ? n - H + h : n
        last > hb || throw(ArgumentError(
            "lag_regression: not enough observations ($n) for the lags at horizon $h"))
        extra = [(_rf_name(f, shift), _rf_eval(f, shift)) for f in features]
        X, target, names = _arx_design(yv, lagset, hb, trend, seasonal, period, ex, pex, plags;
                                       shift=shift, extra=extra, last=last)
        if fit === _lagreg_ols_fit
            rank(X) == size(X, 2) || throw(ArgumentError(
                "lag_regression: design matrix is singular/collinear at horizon $h -- " *
                "check for redundant features"))
        end
        model = fit(X, target)
        p = predict(model, X)
        (p isa AbstractVector && length(p) == size(X, 1)) || throw(ArgumentError(
            "lag_regression: `predict(model, X)` must return a vector with one value per " *
            "row of X ($(size(X, 1))); got $(typeof(p))" *
            (p isa AbstractArray ? " of size $(size(p))" : "")))
        push!(models, model); push!(allnames, names); push!(nobs, size(X, 1))
    end
    return LagRegressionModel(strategy, models, allnames, nobs, fit, predict, yv, lagset, trend,
                              seasonal, seasonal ? Int(period) : nothing, ex, pex, plags,
                              collect(features), strategy === :direct ? H : 0)
end

function _lagreg_extra(m::LagRegressionModel, shift)
    isempty(m.features) && return nothing
    return Dict(_rf_name(f, shift) => _rf_eval(f, shift) for f in m.features)
end

"""
    forecast(m::LagRegressionModel, horizon; newexog=nothing, newpast=nothing, history=nothing) -> Forecast

Point forecasts from a [`lag_regression`](@ref) model (no intervals: the
`Forecast` has no levels). Known-future regressors need `newexog` for every
step; past-only ones need `newpast` beyond their smallest lag under the
recursive strategy, and never under the direct one. A direct model
forecasts at most its fitted `horizon`. `history` forecasts from the end
of another stretch of the same series with the fitted coefficients
unchanged -- a model fitted earlier, applied to newer data (the
`tscv(…; refit=false)` case); it is supported without regressors only.
"""
function forecast(m::LagRegressionModel, horizon::Integer; newexog=nothing, newpast=nothing,
                  history=nothing)
    horizon >= 1 || throw(ArgumentError("forecast: horizon must be >= 1"))
    if history !== nothing
        (m.exog === nothing && m.past_exog === nothing) || throw(ArgumentError(
            "forecast: `history` is supported for models without regressors"))
        hv = collect(Float64, tsvalues(history))
        reach = max(maximum(m.lags), isempty(m.features) ? 0 : maximum(_rf_reach, m.features)) +
                (m.strategy === :direct ? m.horizon - 1 : 0)
        length(hv) > reach || throw(ArgumentError("forecast: `history` is too short for the lags"))
        m = LagRegressionModel(m.strategy, m.models, m.names, m.nobs, m.fit_fn, m.predict_fn, hv,
                               m.lags, m.trend, m.seasonal, m.period, nothing, nothing, m.past_lags,
                               m.features, m.horizon)
    end
    m.strategy === :direct && horizon > m.horizon && throw(ArgumentError(
        "forecast: this direct model was fitted for horizons 1:$(m.horizon); refit with " *
        "horizon >= $horizon"))
    n = length(m.y)
    minp = isempty(m.past_lags) ? 0 : minimum(m.past_lags)
    xext = _arx_future_regressors(m.exog, newexog, horizon, 0, "known-future regressors", "newexog")
    # a direct model's past-only lags are shifted past the horizon: no future values needed
    pext = _arx_future_regressors(m.past_exog, newpast, horizon,
                                  m.strategy === :direct ? horizon : minp,
                                  "past-only regressors", "newpast")
    point = Vector{Float64}(undef, horizon)
    if m.strategy === :recursive
        yext = copy(m.y)
        extra = _lagreg_extra(m, 0)
        for h in 1:horizon
            row = _arx_forecast_row(m.names[1], n + h, yext, xext, pext, extra)
            point[h] = only(m.predict_fn(m.models[1], reshape(row, 1, :)))
            push!(yext, point[h])
        end
    else
        for h in 1:horizon
            row = _arx_forecast_row(m.names[h], n + h, m.y, xext, pext, _lagreg_extra(m, h - 1))
            point[h] = only(m.predict_fn(m.models[h], reshape(row, 1, :)))
        end
    end
    name = string("lag regression (", m.strategy, ", lags ", join(m.lags, ","), ")")
    return Forecast(point, fill(NaN, horizon), Float64[], zeros(horizon, 0), zeros(horizon, 0),
                    horizon, name)
end

StatsAPI.predict(m::LagRegressionModel, horizon::Integer; kwargs...) = forecast(m, horizon; kwargs...)
StatsAPI.nobs(m::LagRegressionModel) = m.nobs[1]

function Base.show(io::IO, m::LagRegressionModel)
    println(io, "Lag-feature regression, strategy = ", m.strategy,
            m.strategy === :direct ? ", horizons 1:$(m.horizon)" : "",
            m.fit_fn === _lagreg_ols_fit ? ", least squares" : ", injected regressor")
    print(io, "  features: ", join(m.names[1], ", "), "   rows: ",
          m.strategy === :direct ? join(m.nobs, "/") : m.nobs[1])
end
