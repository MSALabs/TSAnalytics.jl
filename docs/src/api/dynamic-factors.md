# API: Dynamic Factor Models

Many series driven by a few common factors, estimated by maximum likelihood
on the sequential multivariate Kalman filter, with an explicit
identification so the fit is unique. Standardise first with
[`standardize`](@ref) when the series differ in scale -- the model does not
do it for you.

## Fitting

```@docs
fit_dfm
DFMModel
standardize
```

## Factors and rotation

```@docs
factors
rotate_factors
```

## How much the factors explain

```@docs
factor_r2
series_r2
bai_ng_ic
```
