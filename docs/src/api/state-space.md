# API: State Space and the Kalman Filter

The `GaussianSSM`/`TimeVaryingSSM` Kalman filter engine that
`fit_arma`/`fit_arima`/`fit_sarima`/`fit_arimax`/`fit_sarimax` share (see
[Manual: State Space and the Kalman Filter](../manual/07-state-space-and-kalman.md),
and [Part VI](../introduction/29-the-state-space-form.md) of the
Introduction) is exported as public API. The
[`StateSpaceModel`](@ref) abstract type itself is documented in
[API: Primitives](primitives.md), alongside the rest of the package's
type hierarchy.

## `GaussianSSM` — the ARMA companion form

Time-invariant, stationary-only, ARMA-companion-form (`Z` is implicitly
`e1`, `H` is implicitly `0`): the form `fit_arma`/`fit_arima`/
`fit_sarima` build on internally.

```@docs
GaussianSSM
build_statespace
stationary_cov
combined_ar_ma
```

## `TimeVaryingSSM` — the general Durbin & Koopman form

Genuinely arbitrary per-period `Z_t`/`T_t`/`R_t`/`Q_t`/`H_t`: the form
regression-with-drifting-coefficients ([`fit_arimax`](@ref)'s
`model=:tvss`) and diffuse initialisation build on.

```@docs
TimeVaryingSSM
to_time_varying
kalman_filter_diffuse
```

## Filtering and smoothing

`kalman_filter`/`kalman_smoother` are each defined for both forms
above — the `GaussianSSM` method and the `TimeVaryingSSM` method are
documented together here.

```@docs
kalman_filter
kalman_smoother
```
