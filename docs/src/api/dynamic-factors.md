# API: Dynamic Factor Models

Many series driven by a few common factors that follow a VAR, estimated by
maximum likelihood on the sequential multivariate Kalman filter, with an
explicit identification so the fit is unique. Standardise first with
[`standardize`](@ref) when the series differ in scale — the model does not
do it for you. The task-oriented guide is
[Manual: Dynamic Factor Models](../manual/18-dynamic-factor-models.md).

```@example dfmapi
using TSAnalytics, Plots
e = dataset("econ5")
g(x) = 100 .* diff(log.(x))                    # quarterly growth, per cent
Z = standardize(hcat(diff(e.unemp), g(e.gnp), g(e.consum), g(e.govinv), g(e.prinv)))
Y = (unemp = Z[:, 1], gnp = Z[:, 2], consum = Z[:, 3], govinv = Z[:, 4], prinv = Z[:, 5])
m = fit_dfm(Y, 1)
plot(factor_paths(m); size = (700, 300))
```

One factor behind five quarterly US series — unemployment, GNP,
consumption, government and private investment, all in changes — with its
95% band.

```@example dfmapi
plot(loadings_plot(m); size = (420, 380))
```

The loadings: which series move with the factor, and in which direction.
The identification makes the first series' loading positive, so this
factor rises with unemployment — and GNP and private investment load on it
negatively. Its sign is a convention; the pattern is not.

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

The charts above and the variance-decomposition and selection charts are
built by [`factor_paths`](@ref), [`loadings_plot`](@ref),
[`factor_r2_plot`](@ref), [`factor_selection_plot`](@ref) and
[`factor_residual_plot`](@ref), documented in [API: Plotting](plotting.md).

```@docs
factor_r2
series_r2
bai_ng_ic
```
