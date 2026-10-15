# API: Unobserved Components

A series split by a fitted model into a level, a slope, seasonal, cycle and
autoregressive components and an irregular, each with its own variance and
recovered with a standard error. The models run on the
[`TimeVaryingSSM`](@ref) filter and smoother of
[API: State Space](state-space.md); the task-oriented guide is
[Manual: Unobserved Components](../manual/17-unobserved-components.md).

These are also called *structural time series models* in the literature
(Tsay 2010, §11.3.5; Harvey 1989). That is unrelated to the *structural VAR*
of [API: Structural VAR](structural.md), which identifies shocks in a
multivariate system. The clash of names is unfortunate and predates both.

```@example ucapi
using TSAnalytics, Plots
y = log.(dataset("jj").value)                  # quarterly J&J earnings per share
m = fit_uc(y; level = :stochastic, seasonal = 4)
plot(state_decomposition(m); size = (700, 520))
```

The level and the seasonal carry 95% bands for the smoothed component;
the irregular, a residual, does not. This is the model of Tsay's Example
11.3, and its two standard deviations agree with his published
S-PLUS/SsfPack estimates to about `1e-7`.

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

[`state_decomposition`](@ref) and `diagnostic_plot(m)` also take a
`UCModel`; both are documented in [API: Plotting](plotting.md).

```@docs
uc_decomposition
UCDecomposition
```
