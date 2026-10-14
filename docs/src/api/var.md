# API: Vector Autoregression

A fitted VAR's impulse responses, with analytic ±1.96 standard-error
bands — the output most of this page exists to produce. The worked
explanation is in
[Manual: Vector Autoregression](../manual/11-vector-autoregression.md).

```@example apivar
using TSAnalytics, Plots
e = dataset("econ5")
m = fit_var((unemp = e.unemp, gnp = log.(e.gnp), consum = log.(e.consum)), 2)
plot(irf(m, 12; stderr = true); size = (750, 600))
```

## Fitting

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

## Forecasting

```@docs
VARForecast
```

## Granger causality

```@docs
granger_test
instantaneous_causality
GrangerTest
InstantaneousCausalityTest
```

## Impulse responses and variance decomposition

```@docs
irf
IRFResult
ma_rep
orth_ma_rep
irf_stderr
cum_effect_stderr
lr_effects
lr_effect_stderr
coef_alpha
fevd
FEVDResult
```

## Residual diagnostics

`arch_lm_test` gains a method for a fitted VAR — the multivariate
ARCH-LM test — and its docstring is listed with the univariate method on
[API: Diagnostics](diagnostics.md).

```@docs
var_diagnostics
portmanteau_test
PortmanteauTest
normality_test
MultivariateNormalityTest
bg_test
BreuschGodfreyTest
es_test
EdgertonShukurTest
MultivariateARCHLMTest
```
