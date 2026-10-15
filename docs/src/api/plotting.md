# API: Plotting

`diagnostic_plot` dispatches on what it is given: residuals plus an
ARMA-family model produce the standard four-panel display, while a
fitted [`GarchModel`](@ref) produces the six variance-model panels
instead. Both return their computed values as data, so the numbers are
usable with no plotting backend loaded.

```@docs
diagnostic_plot
DiagnosticPlotResult
GarchDiagnosticPlotResult
seasonal_subseries_plot
boxcox_profile_plot
```

## State decomposition, roots, and multivariate results

Each builds the data its chart draws, as a plain struct, and `plot(x)`
draws it with any Plots.jl backend loaded. A [`Forecast`](@ref),
[`VARForecast`](@ref), [`VolatilityForecast`](@ref), [`IRFResult`](@ref)
and [`FEVDResult`](@ref) need no builder: `plot` takes them directly,
with `history = y` for the forecasts.

```@docs
state_decomposition
StateDecomposition
arma_roots
ARMARootsResult
ts_display
TSDisplayResult
seasonal_plot
lag_plot
roots_plot
RootsPlotResult
VARDiagnosticPlotResult
cointegration_plot
CointegrationPlotResult
```

## Unobserved components, factors, regimes and thresholds

The Stage 11 models plot the same way. `state_decomposition` and
`diagnostic_plot` take an unobserved-components model; the rest are
builders for one chart each. `plot(t)` draws a [`ThresholdTest`](@ref)'s
bootstrap distributions directly. No confidence set for a threshold is
drawn: it is not computed, and a shaded region with no stated coverage
would mislead. The pairwise regime-transition probabilities get no plot,
for want of one natural rendering.

```@docs
factor_paths
FactorPathResult
loadings_plot
LoadingsResult
factor_r2_plot
FactorR2Result
factor_selection_plot
FactorSelectionResult
factor_residual_plot
FactorResidualResult
regime_plot
RegimeProbabilityResult
threshold_regime_plot
ThresholdRegimeResult
threshold_phase_plot
ThresholdPhaseResult
threshold_profile
ThresholdProfileResult
transition_plot
TransitionFunctionResult
```

## Change points, intermittent demand and lag regression

The Stage 12 results plot directly: a [`DemandIntervals`](@ref) and an
[`IntermittentForecast`](@ref) as stems, an [`ADICV2Result`](@ref) as a
quadrant, a [`ChangePointResult`](@ref), [`BreakpointsResult`](@ref),
[`FStatsResult`](@ref) and [`EFPResult`](@ref) each with its chart (see
those pages). Two builders remain, both for lag regression.

```@docs
lag_feature_map
LagFeatureMap
cv_window_layout
CVWindowLayout
```

Which lags each horizon's model uses. A direct model anchored at the
forecast origin drifts one lag per step:

```@example lagapi
using TSAnalytics, Plots
y = sin.(1:60) .+ (1:60) ./ 10
m = lag_regression(y, [1, 2, 4]; strategy = :direct, horizon = 4)
plot(lag_feature_map(m); size = (600, 320))
```

Blocked cross-validation trains on both sides of each held-out block:

```@example lagapi
plot(cv_window_layout(blocked_window_split(24; window_length = 6); kind = "Blocked CV"); size = (650, 300))
```
