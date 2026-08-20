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

## Unit root tests

```@docs
adf_test
kpss_test
pp_test
ADFTest
KPSSTest
PPTest
```

## Diagnostics

```@docs
ljungbox_test
qs_test
jarque_bera_test
durbin_watson_test
LjungBoxTest
QSTest
JarqueBeraTest
DurbinWatsonTest
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
