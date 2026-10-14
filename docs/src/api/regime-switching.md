# API: Regime Switching

Models whose parameters change with an unobserved regime that follows a
Markov chain (Hamilton 1989). The transition matrix is **left-stochastic**:
`P[i, j] = Pr(S_t = i | S_{t−1} = j)`, so its columns sum to one.

## Fitting

```@docs
fit_ms_regression
fit_ms_autoregression
MSModel
```

## Regimes

```@docs
regime_probabilities
expected_durations
canonicalise
```
