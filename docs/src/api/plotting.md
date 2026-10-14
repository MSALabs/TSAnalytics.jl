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
