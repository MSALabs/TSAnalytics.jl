# API: ARIMAX and Regression with ARIMA Errors

```@docs
arx
ARXModel
autoreg
AutoRegModel
fit_arimax
ArimaxModel
fit_sarimax
SarimaxModel
auto_arimax
fourier_terms
```

## Lag-feature regression

[`arx`](@ref) is already the recursive reduction with least squares;
[`lag_regression`](@ref) injects any regressor, adds the direct strategy
and lagged rolling features, and builds its design with `arx`'s own code.

```@docs
lag_regression
LagRegressionModel
rolling_feature
RollingFeature
```
