# API: State Space and the Kalman Filter

The `GaussianSSM`/`TimeVaryingSSM` Kalman filter engine that
`fit_arma`/`fit_arima`/`fit_sarima`/`fit_arimax`/`fit_sarimax` share (see
[Manual: State Space and the Kalman Filter](../manual/07-state-space-and-kalman.md),
and [Part VII](../introduction/33-the-state-space-form.md) of the
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
documented together here. The `TimeVaryingSSM` methods also take a
**matrix** of observations, one column per series, which is what
[`fit_dfm`](@ref) runs on: each period's observations are processed **one
series at a time**, as a sequence of scalar updates with the transition
applied once per period, so no `N × N` matrix is formed or inverted and a
missing value is simply skipped. That requires a diagonal observation
covariance `H`; a dense one is refused.

The sequential treatment is Durbin & Koopman's (2001), in their section on
treating multivariate series as univariate. None of the six textbooks this
project checks against states it — Hamilton and Shumway & Stoffer both form
and invert the full innovation covariance — so it is cited to Durbin &
Koopman directly, and the recursion is verified against `statsmodels` and
`dfms` (to `1.6e-12` and `1.4e-11` on a five-series fixture) rather than
against a printed page.

```@docs
kalman_filter
kalman_smoother
```
