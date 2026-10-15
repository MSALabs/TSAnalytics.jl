# API: Threshold Models

Autoregressions whose coefficients depend on which side of a threshold a
lagged value falls (Tong 1990): the self-exciting threshold AR (SETAR),
with an abrupt switch, and the logistic smooth-transition AR (LSTAR), with
a gradual one. `thDelay = d` means the threshold variable is
`y_{t−(d+1)}`, and the low regime is `z ≤ th`, inclusive — both as in R's
`tsDyn`, so its numbers can be checked directly. The task-oriented guide is
[Manual: Threshold and Smooth-Transition Models](../manual/20-threshold-models.md).

```@example tarapi
using TSAnalytics, Plots, DelimitedFiles
y = log10.(readdlm(TSAnalytics.LYNX, ',', skipstart = 1)[:, 2])   # Canadian lynx
m = fit_setar(y, 2; thDelay = 1, warn = false)
plot(threshold_regime_plot(m); size = (700, 320))
```

Each year coloured by the regime its value two years earlier put it in.

```@example tarapi
plot(threshold_profile(m); size = (600, 320))
```

The sum of squares at every candidate threshold; the search takes the
minimum, at `3.3100557377508912` — the same value `tsDyn` finds, to every
digit.

## SETAR

```@docs
fit_setar
SETARModel
select_setar
tsdyn_ic
```

## LSTAR

```@docs
fit_lstar
LSTARModel
```

## Testing for a threshold

The charts are built by [`threshold_regime_plot`](@ref),
[`threshold_phase_plot`](@ref), [`threshold_profile`](@ref) and
[`transition_plot`](@ref), in [API: Plotting](plotting.md); a
[`ThresholdTest`](@ref) plots its bootstrap distributions directly.

```@docs
setar_test
ThresholdTest
```
