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
form (ARIMA/SARIMAX, UnobservedComponents, linear ETS, ...). Subtypes are
expected to be convertible to a `GaussianSSM` representation (planned,
not yet implemented -- see Stage 6 of `development-sequence.md`) so that
they can all share a single, well-tested Kalman filter/smoother
implementation instead of each reimplementing filtering.
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
