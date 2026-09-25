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
