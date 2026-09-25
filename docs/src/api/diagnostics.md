# API: Diagnostics

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

## Portmanteau, normality, and structure tests

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

## Asymmetry and parameter stability

Both of these exist in R's `rugarch` and had no equivalent here; both
were implemented by reading that package's own source rather than the
originating papers, because the two disagree on construction in ways
that change the numbers — see each docstring.

```@docs
sign_bias_test
nyblom_test
SignBiasTest
NyblomTest
```
