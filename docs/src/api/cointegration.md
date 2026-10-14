# API: Cointegration and VECM

The estimated cointegrating relation between log GNP and log consumption
in `econ5`, with its autocorrelation — the display that shows whether an
"equilibrium" actually behaves like one, and one no Python library draws.
The worked explanation is in
[Manual: Cointegration and VECM](../manual/13-cointegration-and-vecm.md).

```@example apicoint
using TSAnalytics, Plots
e = dataset("econ5")
Y = (gnp = log.(e.gnp), consum = log.(e.consum))
plot(cointegration_plot(fit_vecm(Y; lags = 2, rank = 1)); size = (750, 300))
```

## Residual-based tests

```@docs
coint_test
CointTest
phillips_ouliaris_test
PhillipsOuliarisTest
reject
critical_values
```

## Johansen

```@docs
johansen_test
JohansenTest
coint_rank
normalize_beta
```

## VECM

```@docs
fit_vecm
VECMModel
var_rep
```
