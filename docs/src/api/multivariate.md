# API: Multivariate

Vector time series. See the manual chapter
[Vector Autoregression](../manual/11-vector-autoregression.md) for the
narrative, including the places where the two references disagree with
each other.

## Vector autoregression

```@docs
fit_var
VARModel
var_select
VARSelection
```

## Stability

```@docs
roots
is_stable
companion
```

## Granger causality

```@docs
granger_test
instantaneous_causality
GrangerTest
InstantaneousCausalityTest
```

## Forecasting

```@docs
VARForecast
```
