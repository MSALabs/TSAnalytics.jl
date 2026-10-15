# Datasets are not exported (avoids polluting `using TSAnalytics` with
# generic names like `NILE` in the caller's namespace) -- access as
# `TSAnalytics.NILE`, etc. Same convention StateSpaceModels.jl uses for
# the same reason.
#
# These are validation fixtures, not part of the modeling API: implement
# a function, compute the reference number in R/Python on one of these
# series, hardcode it into a test, iterate until it matches. See
# data/README.md for provenance, licensing, and which dataset fits which
# development stage.

"""
    TSAnalytics.NILE

Absolute path to `nile.csv`: annual flow of the Nile at Aswan
(10^8 m^3), 1871-1970. The canonical local-level-model / unit-root-test
example (Durbin & Koopman (2012), ch. 2).
"""
const NILE = joinpath(dirname(@__DIR__), "test_data", "nile.csv")

"""
    TSAnalytics.LYNX

Absolute path to `lynx.csv`: annual Canadian lynx trappings in the
Mackenzie River district, 1821-1934 (R's `datasets::lynx`, n = 114). The
canonical threshold-autoregression example, usually taken in `log10`
(Tong 1990). Not the 91-point "Lynx" of [`dataset`](@ref)'s catalogue.
"""
const LYNX = joinpath(dirname(@__DIR__), "test_data", "lynx.csv")

"""
    TSAnalytics.HAMILTON_GNP

Absolute path to `hamilton_gnp.csv`: quarterly growth of US real GNP
(100 times the change in its log), 1951Q1-1984Q3, n = 135 -- the series
Hamilton (1989) fitted the first Markov-switching autoregression to, as
distributed with statsmodels.
"""
const HAMILTON_GNP = joinpath(dirname(@__DIR__), "test_data", "hamilton_gnp.csv")

"""
    TSAnalytics.CEREAL

Absolute path to `montgomery_cereal.csv`: weekly sales of a cereal brand,
104 weeks, a rival product launched in week 88 -- Montgomery, Jennings &
Kulahci (2008), Table 6.12 (printed p. 335), transcribed from the page and
checked twice. Their intervention-analysis example (Example 6.4).
"""
const CEREAL = joinpath(dirname(@__DIR__), "test_data", "montgomery_cereal.csv")

"""
    TSAnalytics.FORECAST_ERRORS

Absolute path to `montgomery_forecast_errors.csv`: 50 one-step-ahead
forecast errors, Montgomery, Jennings & Kulahci (2008), Table 2.3 (printed
p. 54), transcribed from the page. They reproduce every control limit the
book publishes for them (Figures 2.33-2.35).
"""
const FORECAST_ERRORS = joinpath(dirname(@__DIR__), "test_data", "montgomery_forecast_errors.csv")

"""
    TSAnalytics.AIR_PASSENGERS

Absolute path to `airpassengers.csv`: monthly US airline passenger
totals, 1949-1960. The canonical multiplicative-seasonality / SARIMA
example (Box & Jenkins, 1976).
"""
const AIR_PASSENGERS = joinpath(dirname(@__DIR__), "test_data", "airpassengers.csv")

"""
    TSAnalytics.SUNSPOTS_YEAR

Absolute path to `sunspot_year.csv`: yearly sunspot numbers, 1700-1988.
Classic nonlinear/AR benchmark series (H. Tong, 1996).
"""
const SUNSPOTS_YEAR = joinpath(dirname(@__DIR__), "test_data", "sunspot_year.csv")

"""
    TSAnalytics.US_CHANGE

Absolute path to `uschange.csv`: quarterly percentage changes in US
consumption, income, production, savings, and unemployment, 1960-2016.
Multivariate/VAR and regression-with-ARIMA-errors benchmark (Hyndman &
Athanasopoulos, *Forecasting: Principles and Practice*).
"""
const US_CHANGE = joinpath(dirname(@__DIR__), "test_data", "uschange.csv")

"""
    TSAnalytics.VEHICLE_FATALITIES

Absolute path to `vehicle_fatalities.csv`: annual road traffic
fatalities in Norway and Finland, 1970-2003. Structural time series /
local linear trend example (Commandeur & Koopman, 2007, ch. 3).
"""
const VEHICLE_FATALITIES = joinpath(dirname(@__DIR__), "test_data", "vehicle_fatalities.csv")

"""
    TSAnalytics.INTERNET

Absolute path to `internet.csv`: number of users logged onto an
internet server, per minute, 100 observations. Short, non-seasonal
ARMA example (Durbin & Koopman, 2012, ch. 9).
"""
const INTERNET = joinpath(dirname(@__DIR__), "test_data", "internet.csv")

"""
    TSAnalytics.MONTHLY

Absolute path to `monthly.csv`: synthetic n=48, period-12 (even) series
with a linear trend and sinusoidal seasonality, additive-appropriate.
Generated for Stage 3.1 dual-verification against real R and Python
output (`numpy.random.seed(42)`, `t=arange(48)`,
`(100+0.5t) + 10*sin(2*pi*t/12) + N(0,1)`) -- see
`handoff/stage-3.1-classical-decompose-handoff.md` for the exact script
and `handoff/verification/stage-3.1-verification-transcript.txt` for
full-precision reference output.
"""
const MONTHLY = joinpath(dirname(@__DIR__), "test_data", "monthly.csv")

"""
    TSAnalytics.PERIOD7

Absolute path to `period7.csv`: synthetic n=42, period-7 (odd) series
with a linear trend and sinusoidal seasonality, additive-appropriate.
Generated for Stage 3.1 (continues the same RNG stream as `monthly.csv`
-- see that constant's docstring and the Stage 3.1 handoff for the exact
script).
"""
const PERIOD7 = joinpath(dirname(@__DIR__), "test_data", "period7.csv")

"""
    TSAnalytics.MONTHLY_OUTLIER

Absolute path to `monthly_outlier.csv`: `monthly.csv` with two outliers
injected (0-indexed `y[10] += 30`, `y[30] -= 25`) to actually engage
`stl_decompose`'s `robust=true`/`outer` reweighting -- `monthly.csv`
itself has no points extreme enough to drive any robustness weight below
1. Generated for Stage 3.2's outer-loop verification -- see
`handoff/stage-3.2-stl-handoff.md` section 3(f) and
`stage-3.2-transcript.txt` for the exact reference numbers this produces.
"""
const MONTHLY_OUTLIER = joinpath(dirname(@__DIR__), "test_data", "monthly_outlier.csv")

"""
    TSAnalytics.MULT_MONTHLY

Absolute path to `mult_monthly.csv`: synthetic n=48, period-12 series,
multiplicative-appropriate (strictly positive, trend and seasonality
combine multiplicatively). Generated for Stage 3.1 (continues the same
RNG stream as `monthly.csv`/`period7.csv` -- see the Stage 3.1 handoff
for the exact script).
"""
const MULT_MONTHLY = joinpath(dirname(@__DIR__), "test_data", "mult_monthly.csv")

"""
    TSAnalytics.AR2_ARX

Absolute path to `ar2_arx.csv`: synthetic n=100 AR(2) series
(`numpy.random.seed(0)`, `y[t] = 0.5*y[t-1] + 0.2*y[t-2] + e[t]`, `e ~
N(0,1)`). Generated for Stage 5.1's `arx` verification -- every
coefficient, standard error, `sigma2`, log-likelihood, AIC, and BIC
tested against it are independently re-verified against real
`statsmodels.tsa.ar_model.AutoReg`, not transcribed from the handoff
alone. See also [`TSAnalytics.ARX_EXOG_X1`](@ref)/
[`TSAnalytics.ARX_EXOG_Y2`](@ref) for the paired exogenous-regressor case.
"""
const AR2_ARX = joinpath(dirname(@__DIR__), "test_data", "ar2_arx.csv")

"""
    TSAnalytics.ARX_EXOG_X1

Absolute path to `arx_exog_x1.csv`: n=100 exogenous regressor
(`numpy.random.RandomState(1).randn(100)`), paired with
[`TSAnalytics.AR2_ARX`](@ref) for Stage 5.1's `arx(...; exog=...)`
verification.
"""
const ARX_EXOG_X1 = joinpath(dirname(@__DIR__), "test_data", "arx_exog_x1.csv")

"""
    TSAnalytics.ARX_EXOG_Y2

Absolute path to `arx_exog_y2.csv`: `AR2_ARX .+ 0.5 .* ARX_EXOG_X1` --
the dependent series for Stage 5.1's `arx(...; exog=...)` verification
(so the exogenous regressor genuinely explains part of the outcome, not
just being along for the ride).
"""
const ARX_EXOG_Y2 = joinpath(dirname(@__DIR__), "test_data", "arx_exog_y2.csv")

"""
    TSAnalytics.HOURLY_MSTL

Absolute path to `hourly_mstl.csv`: synthetic n=500 hourly series with
two seasonal periods (daily, 24; weekly, 168) plus a quadratic trend,
additive-appropriate. Generated for Stage 3.3's `mstl_decompose`
verification (`numpy.random.seed(0)`, `t=arange(1,501)`,
`(0.0001*t^2+100) + 5*sin(2*pi*t/24) + 10*sin(2*pi*t/(24*7)) + N(0,1)`) --
a scaled-down version of `statsmodels.tsa.seasonal.MSTL`'s own docstring
example, kept small enough to verify exactly and run quickly in tests.
"""
const HOURLY_MSTL = joinpath(dirname(@__DIR__), "test_data", "hourly_mstl.csv")
