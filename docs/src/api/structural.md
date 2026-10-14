# API: Structural VAR

A Blanchard-Quah decomposition of GNP growth and unemployment: the
supply shock is the one allowed a permanent effect on output, the demand
shock is not. The worked explanation is in
[Manual: Vector Autoregression](../manual/11-vector-autoregression.md#Structural-VARs).

```@example apisvar
using TSAnalytics, Plots
e = dataset("econ5")
Y = (gnp_growth = diff(log.(e.gnp)), unemp = e.unemp[2:end])
plot(irf(blanchard_quah(fit_var(Y, 2)), 20); size = (750, 600))
```

```@docs
fit_svar
blanchard_quah
SVARModel
```
