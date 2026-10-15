# API: Regime Switching

Models whose parameters change with an unobserved regime that follows a
Markov chain (Hamilton 1989). The transition matrix is **left-stochastic**:
`P[i, j] = Pr(S_t = i | S_{t−1} = j)`, so its **columns** sum to one — the
convention of Hamilton and `statsmodels`; Shumway & Stoffer write its
transpose. The task-oriented guide is
[Manual: Regime Switching](../manual/19-regime-switching.md).

```@example msapi
using TSAnalytics, Plots, DelimitedFiles
gnp = Float64.(readdlm(TSAnalytics.HAMILTON_GNP, ',', skipstart = 1)[:, 2])
m = canonicalise(fit_ms_autoregression(gnp, 4))
plot(regime_plot(m); mode = :shaded, regime = 1, size = (700, 320))
```

Hamilton's model of US GNP growth: the quarters where the smoothed
probability of the low-growth regime exceeds one half are shaded.

## Fitting

```@docs
fit_ms_regression
fit_ms_autoregression
MSModel
```

## Regimes

The chart is built by [`regime_plot`](@ref), and `diagnostic_plot(m;
regime)` gives per-regime residual diagnostics; both are in
[API: Plotting](plotting.md).

```@docs
regime_probabilities
expected_durations
canonicalise
```
