# Forecasting and Accuracy

Producing a forecast is the easy half. The half that decides whether
anyone should act on it is measuring how good it is — against a
benchmark, on data the model has not seen.

```@example fc
using TSAnalytics, Statistics

y = dataset("cardox").value
train, test = y[1:240], y[241:252]   # fit on 20 years, hold out one
nothing # hide
```

## Forecast

```@example fc
m = fit_sarima(train, (1,1,1), (0,1,1,12))
f = forecast(m, 12)

println("model     : ", f.model_name)
println("horizon   : ", f.horizon)
println("levels    : ", f.levels)
println("point[1:3]: ", round.(f.point[1:3], digits=3))
```

**`forecast(m, h)` is the signature on every forecastable type** —
`ArmaModel`, `ArimaModel`, `SarimaModel` and `ARXModel` all retain the
series they were fitted to. `SarimaModel` also accepts
`forecast(m, y, h)` if you want to state which series you mean.

`StatsAPI.predict(m, h; level=level)` is exactly equivalent — the two are
mechanical aliases and neither is deprecated. `forecast` exists for
people arriving from R's `forecast()`; `predict` for people arriving
from the Julia statistics ecosystem.

Models with exogenous regressors take the future regressor values
instead of a horizon alone:

```julia
forecast(m::ArimaxModel,  newexog, horizon)
forecast(m::SarimaxModel, newexog, horizon)
```

`newexog` has one row per step and one column per regressor you
originally passed — the intercept column is reconstructed internally,
not expected from you. See [ARIMAX and
Regression](08-arimax-and-regression.md).

### Intervals

```@example fc
println("lower/upper are ", size(f.lower), " — one column per level")
println("95% at h=1  : [", round(f.lower[1,2], digits=3), ", ", round(f.upper[1,2], digits=3), "]")
println("width h=1   : ", round(f.upper[1,2] - f.lower[1,2], digits=3))
println("width h=12  : ", round(f.upper[12,2] - f.lower[12,2], digits=3))
```

`level` is given in **percentages** — `[80.0, 95.0]` by default,
matching R's convention rather than Python's single `alpha`. The
interval roughly doubles in width from one month ahead to twelve, which
is the useful part of the output: it is the model telling you how far
out it stops being worth listening to.

!!! warning "Intervals assume the parameters are known"
    Standard errors come from the Box-Jenkins psi-weight propagation
    with coefficients treated as fixed. **Parameter estimation
    uncertainty is not included**, so real intervals are slightly too
    narrow — more so on short series.

    This matches what R and `statsmodels` actually do, not what they
    ideally would. It is a shared limitation rather than a difference,
    but it is worth knowing before you quote a 95 % interval to
    somebody.

## Measure it

```@example fc
acc = accuracy(test, f.point, train; sp=12)
println(map(x -> round(x, digits=5), acc))
```

[`accuracy`](@ref) returns MAE, RMSE and MAPE always, and MASE when you
pass the training series. `sp` is the seasonal period used for MASE's
denominator — **pass it for seasonal data**, or MASE silently scales
against a one-step naive benchmark and flatters a seasonal model.

| Measure | Units | Watch out for |
|---|---|---|
| `mae` | Same as the data | — |
| `rmse` | Same as the data | Punishes large errors harder; the one to optimise if big misses are costly |
| `mape` | Percent | Explodes near zero, undefined at zero, asymmetric between over- and under-forecasting |
| `mase` | Dimensionless | `1.0` means "no better than the naive benchmark" |

MASE is the one to quote across series, because it is scale-free and
has a meaningful zero point.

## Against a benchmark

A number on its own says nothing. Four benchmark forecasters come with
the package:

```@example fc
for (lab, fn) in (("naive",          (yt,h) -> naive(yt, h)),
                   ("seasonal_naive", (yt,h) -> seasonal_naive(yt, h, 12)),
                   ("drift",          (yt,h) -> drift(yt, h)),
                   ("mean_forecast",  (yt,h) -> mean_forecast(yt, h)))
    a = accuracy(test, fn(train, 12).point, train; sp=12)
    println(rpad(lab, 15), " MASE = ", round(a.mase, digits=4))
end
```

The SARIMA scored `0.387`. Seasonal naive — "same month last year" —
scores `1.426`, so the model is roughly **3.7 times better** than
repeating last year, and that is the claim worth making.

Note `mean_forecast` at `11.9`. On a trending series the sample mean is
a catastrophic forecast, which is exactly why it belongs in the table:
a benchmark you beat by a mile tells you nothing, and one you barely
beat tells you a great deal.

| Benchmark | Predicts | Right comparison for |
|---|---|---|
| [`naive`](@ref) | The last value | Anything near a random walk |
| [`seasonal_naive`](@ref) | The value one period ago | Seasonal data |
| [`drift`](@ref) | Last value plus average change | Trending data |
| [`mean_forecast`](@ref) | The sample mean | Stationary data only |

All four return a [`Forecast`](@ref), with intervals, so they drop into
anything that consumes one.

## One split is one sample

A single train/test split measures how the model did on one particular
twelve months. [`tscv`](@ref) runs a rolling origin instead — refit,
forecast, step forward, repeat:

```@example fc
e_sarima = tscv(train, (yt, h) -> forecast(fit_sarima(yt, (0,1,1), (0,1,1,12)), h).point;
                h=1, initial=200)
e_snaive = tscv(train, (yt, h) -> seasonal_naive(yt, h, 12); h=1, initial=200)

println("origins : ", size(e_sarima, 1))
println("SARIMA  RMSE : ", round(sqrt(mean(e_sarima .^ 2)), digits=5))
println("snaive  RMSE : ", round(sqrt(mean(e_snaive .^ 2)), digits=5))
```

Forty origins rather than one, and the ranking holds by a factor of
four and a half. That is a claim you can defend.

`fit_forecast_fn` is called as `(train_data, hmax)` and may return
either a plain vector or a `Forecast` — so the benchmark forecasters
slot in unchanged, as above.

The returned matrix holds **errors** (`actual - forecast`), one row per
origin, one column per requested horizon, matching R's `tsCV`
convention exactly. R additionally pads its result to full series
length with `NA`s for time-index alignment; this does not, because a
fold that can never be computed carries no information.

### Several horizons at once

```@example fc
eh = tscv(train, (yt, h) -> seasonal_naive(yt, h, 12); h=[1, 6, 12], initial=200)
println("size ", size(eh))
println("RMSE per horizon: ", round.(sqrt.(mean(eh .^ 2; dims=1))[:], digits=4))
```

Error grows with horizon — `0.98`, `1.21`, `1.53`. Quoting a
one-step-ahead accuracy for a model you intend to run twelve steps out
overstates it by half.

`window=nothing` (the default) expands the training set each fold,
matching R's `window=NULL`. Pass an `Integer` for a fixed-size sliding
window, which is what you want if you believe the relationship itself
is changing. `initial` skips that many observations before the first
origin, and it is usually what keeps a cross-validation from taking all
afternoon.

## Backing out a transformation

If you forecast a transformed series, reversing it takes care — the
naive back-transform gives the **median** of the forecast distribution,
not the mean. [`boxcox_inv`](@ref) does the plain inverse;
[Chapter 7](../introduction/07-transformations.md) has the bias
correction and when it is worth applying.

## What does not forecast yet

`model=:tvss` fits but does not forecast — there `beta` is a latent
time-varying state rather than a fixed coefficient, so a forecast needs
its projected path and a second variance term. `model=:mle` (the
default) forecasts normally. GARCH models forecast **variance**, not
level, through
[`forecast_volatility`](@ref); see
[GARCH and Volatility](06-garch-and-volatility.md).

## See also

- [Was It Any Good?](../getting-started/03-was-it-any-good.md) — diagnostics before accuracy
- [Chapters 22–23](../introduction/22-forecasting.md) — what a forecast interval actually claims, and how to evaluate one honestly
- [API: Forecasting](../api/forecasting.md)
