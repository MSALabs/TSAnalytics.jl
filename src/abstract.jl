"""
    TimeSeriesModel

Root abstract type for every fitted model in TSAnalytics (ARIMA, SARIMAX,
ETS, UnobservedComponents, VAR, ...). Concrete subtypes participate in the
StatsAPI contract: `fit`, `coef`, `vcov`, `residuals`, `predict`,
`loglikelihood`, `aic`, `bic`, `nobs`.

This mirrors GLM.jl's separation of concerns: a model type holds the fitted
state, and behaviour is added via small, composable methods rather than a
single monolithic struct.
"""
abstract type TimeSeriesModel end

"""
    StateSpaceModel <: TimeSeriesModel

Root abstract type for any model expressed in linear Gaussian state-space
form (ARIMA/SARIMAX, UnobservedComponents, linear ETS, ...), so that they
all share one well-tested Kalman filter/smoother rather than each
reimplementing filtering.

The engine is built and public: [`build_statespace`](@ref) produces the
[`GaussianSSM`](@ref) that `fit_arma`/`fit_arima`/`fit_sarima` filter
through, and [`kalman_filter`](@ref)/[`kalman_smoother`](@ref) are
defined for both it and [`TimeVaryingSSM`](@ref). What is *not* built is
a uniform `convert(GaussianSSM, ::StateSpaceModel)` method -- each fit
function constructs its own representation directly, since the mapping
differs per model family. See
[Manual: State Space and the Kalman Filter](../manual/07-state-space-and-kalman.md).
"""
abstract type StateSpaceModel <: TimeSeriesModel end

"""
    UnivariateModel <: TimeSeriesModel

Marker for models with a single observed series (as opposed to VAR/VECM,
which are inherently multivariate).
"""
abstract type UnivariateModel <: TimeSeriesModel end

"""
    HypothesisTest

Root abstract type for statistical tests in TSAnalytics (unit-root tests,
whiteness tests, seasonality tests, ...). Every concrete test supports
`statistic`, `pvalue`, and a `show` method that prints a compact summary
table, following the convention used by HypothesisTests.jl.
"""
abstract type HypothesisTest end

"""
    statistic(t::HypothesisTest) -> Real

The test statistic computed for `t`.
"""
statistic(t::HypothesisTest) = t.statistic

"""
    pvalue(t::HypothesisTest) -> Real

The p-value associated with `t`'s test statistic.
"""
pvalue(t::HypothesisTest) = t.pvalue
