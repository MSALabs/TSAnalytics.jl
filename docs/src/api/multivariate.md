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

`arch_lm_test` gains a method for a fitted VAR; its docstring is listed
with the univariate method.

## Cointegration

```@docs
coint_test
CointTest
phillips_ouliaris_test
PhillipsOuliarisTest
reject
johansen_test
JohansenTest
coint_rank
normalize_beta
critical_values
```

## VECM

```@docs
fit_vecm
VECMModel
var_rep
```

## Structural VAR

```@docs
fit_svar
blanchard_quah
SVARModel
```

## Forecasting

```@docs
VARForecast
```
