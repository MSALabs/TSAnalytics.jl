# API Reference

## Container-agnostic interface

```@docs
tsvalues
tsindex
```

## Differencing

```@docs
diff
diffinv
tsdiff
tsundiff
```

## Filters & smoothing

```@docs
convolution_filter
recursive_filter
moving_average
```

## Decomposition

```@docs
classical_decompose
ClassicalDecomposition
stl_decompose
STLDecomposition
mstl_decompose
MSTLDecomposition
```

## Descriptive statistics

```@docs
acf
pacf
ACFResult
```

## Spectral analysis

```@docs
periodogram
spectral_density
PeriodogramResult
```

## Transforms

```@docs
boxcox
boxcox_inv
guerrero_lambda
boxcox_profile_plot
```

## Unit root tests

```@docs
adf_test
kpss_test
pp_test
ADFTest
KPSSTest
PPTest
adf_pvalue_response_surface
adf_critical_values_response_surface
pp_rho_pvalue
kpss_hobijn_autolag
```

## Diagnostics

```@docs
ljungbox_test
qs_test
jarque_bera_test
durbin_watson_test
arch_lm_test
dk_heteroskedasticity_test
durbin_watson_pvalue_exact
LjungBoxTest
QSTest
JarqueBeraTest
DurbinWatsonTest
ARCHLMTest
DKHeteroTest
```

## Residual diagnostic plot

```@docs
diagnostic_plot
DiagnosticPlotResult
```

## Seasonal subseries plot

```@docs
seasonal_subseries_plot
```

## Univariate models

```@docs
arx
ARXModel
forecast
Forecast
fit_arma
ArmaModel
fit_arima
ArimaModel
fit_sarima
SarimaModel
auto_arima
fit_arimax
ArimaxModel
fit_sarimax
SarimaxModel
auto_arimax
fit_autoreg_garch
AutoregGarchModel
```

## Volatility models

```@docs
fit_garch
fit_garch_multi
GarchModel
forecast_volatility
VolatilityForecast
```

## Realized volatility

```@docs
realized_variance
bipower_variation
jump_test
realized_semivariance
realized_measures
JumpTest
```

## Accuracy metrics

```@docs
mae
rmse
mape
smape
mase
accuracy
```

## Cross-validation

```@docs
expanding_window_split
sliding_window_split
tscv
```

## Exponential smoothing

```@docs
holt_winters
ExponentialSmoothingModel
```

## Model-fitting infrastructure

```@docs
OptimResult
partrans
invpartrans
```

## Abstract type hierarchy

```@docs
TimeSeriesModel
StateSpaceModel
UnivariateModel
HypothesisTest
statistic
pvalue
```
