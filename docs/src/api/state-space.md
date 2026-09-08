# API: State Space and the Kalman Filter

The `GaussianSSM`/`TimeVaryingSSM` Kalman filter engine that
`fit_arma`/`fit_arima`/`fit_sarima`/`fit_arimax`/`fit_sarimax` share (see
[Manual: State Space and the Kalman Filter](../manual/07-state-space-and-kalman.md))
is currently internal, not part of the exported public API — there is no
separate low-level state-space API beyond what those fitting functions
already expose. The [`StateSpaceModel`](@ref) abstract type itself is
documented in [API: Primitives](primitives.md), alongside the rest of the
package's type hierarchy.
