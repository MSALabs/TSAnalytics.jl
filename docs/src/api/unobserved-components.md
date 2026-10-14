# API: Unobserved Components

Structural time-series models: a series split by a fitted model into a
level, a slope, seasonal, cycle and autoregressive components and an
irregular, each with its own variance and recovered with a standard error.
The models run on the [`TimeVaryingSSM`](@ref) filter and smoother of
[API: State Space](state-space.md).

## Fitting

```@docs
fit_uc
UCModel
```

## Named models

```@docs
uc_local_level
uc_local_linear_trend
uc_random_walk
uc_smooth_trend
uc_bsm
```

## Components

```@docs
uc_decomposition
UCDecomposition
```
