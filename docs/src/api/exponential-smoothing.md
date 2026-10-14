# API: Exponential Smoothing, ETS, Theta and TBATS

A fitted ETS model pulled apart into the states it estimates — the
display R's `plot.ets` draws and `statsmodels` has no equivalent for. The
worked explanation is in
[Manual: Exponential Smoothing, ETS, Theta and TBATS](../manual/10-exponential-smoothing.md).

```@example apiets
using TSAnalytics, Plots
iip = dataset("iip_india")
m = fit_ets(iip.value, 12; error = :add, trend = :add, seasonal = :add)
plot(state_decomposition(m); size = (700, 600))
```

## Classical exponential smoothing

```@docs
holt_winters
ExponentialSmoothingModel
```

## ETS

```@docs
fit_ets
auto_ets
ETSModel
notation
```

## Theta

```@docs
fit_theta
ThetaModel
```

## TBATS

```@docs
fit_tbats
TBATSModel
```
