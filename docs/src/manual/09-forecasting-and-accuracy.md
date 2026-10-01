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

## Scoring the interval, not just the point

Everything above scores the **point** forecast. The interval is a
separate claim, and until you score it separately it is unfalsifiable —
you can see that an interval is wide, but not whether it was wide in the
right places.

```@example fc
ia = interval_accuracy(test, f)
println("levels   : ", ia.level)
println("coverage : ", round.(ia.coverage, digits=4))
println("winkler  : ", round.(ia.winkler, digits=4))
println("crps     : ", round(ia.crps, digits=6))
```

Coverage `0.83` and `0.92` against nominal `0.80` and `0.95` — close,
on twelve points, which is as much as twelve points can tell you.

| Measure | Scores | Lower is better |
|---|---|---|
| [`interval_coverage`](@ref) | How often the actual fell inside | No — compare to nominal |
| [`winkler_score`](@ref) | Width, plus a steep penalty for missing | Yes |
| [`crps_normal`](@ref) | The whole predictive distribution, in data units | Yes |
| [`pinball_loss`](@ref) | A single quantile | Yes |

### Coverage alone will mislead you

```@example fc
bad = interval_accuracy(test, seasonal_naive(train, 12, 12))
println("model  : coverage ", round(ia.coverage[2], digits=3),
        "   winkler ", round(ia.winkler[2], digits=3))
println("snaive : coverage ", round(bad.coverage[2], digits=3),
        "   winkler ", round(bad.winkler[2], digits=3))
```

**The benchmark has better 95 % coverage than the model and is far
worse.** It achieves `1.00` by being wide enough to contain anything,
and the Winkler score says so — it charges for width as well as for
misses, which is exactly the trade coverage cannot see.

That is why [`interval_accuracy`](@ref) reports them together, and why a
coverage figure quoted on its own is not evidence of a good interval.

### CRPS

[`crps_normal`](@ref) scores the whole predictive distribution rather
than a point or an interval, in the units of the data. It reduces to the
absolute error as the distribution collapses to a point, which makes it
directly comparable with MAE:

```@example fc
println("crps : ", round(ia.crps, digits=5))
println("mae  : ", round(mae(test, f.point), digits=5))
println("crps of a degenerate forecast: ",
        round(crps_normal([3.0], 2.5, 1e-8), digits=6), "   |3 - 2.5| = 0.5")
```

For a simulated predictive distribution — EGARCH, where
[`forecast_volatility`](@ref) has no closed form — use
[`crps_ensemble`](@ref) on the draws instead. Scoring simulated paths
with a Gaussian rule would assume away the reason they were simulated.

!!! note "These are checked against their definitions, not another package"
    Neither R's `scoringRules` nor Python's `properscoring` is reachable
    in this project's environment, and R's `forecast` has no Winkler,
    pinball or CRPS function. So `crps_normal`'s closed form is verified
    against **numerical integration of the CRPS definition** instead,
    agreeing to `5e-13`; `pinball_loss` against the identity that it is
    half the MAE at the median; `winkler_score` against hand
    computation.

    Arguably a stronger check than reproducing another implementation,
    but a different one — worth knowing which you have. See
    [Appendix B](../introduction/B-verification.md).

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
not the mean. Box-Cox is non-linear, so it does not commute with taking
an expectation; what survives it is the quantile.

```@example fc
ylog = log.(train)
flog = forecast(fit_sarima(ylog, (0,1,1), (0,1,1,12)), 12)
med = boxcox_inv(flog.point, 0.0)                     # median
mu  = boxcox_inv(flog.point, 0.0; fvar=flog.se .^ 2)  # mean
println("median h=1 : ", round(med[1], digits=4), "   h=12: ", round(med[12], digits=4))
println("mean   h=1 : ", round(mu[1],  digits=4), "   h=12: ", round(mu[12],  digits=4))
println("gap grows with horizon: ", round(mu[12]-med[12], digits=5),
        " vs ", round(mu[1]-med[1], digits=5))
```

`fvar` is the forecast variance on the **transformed** scale, so
`flog.se .^ 2`. The gap widens with the horizon because the forecast
variance does — the correction matters most exactly where forecasts are
least certain.

**Which one you want is a choice.** Correct when the forecasts will be
**summed or aggregated**, since medians do not add, or when feeding
something that assumes an expectation. Leave it alone when you want the
value the series is equally likely to fall above or below.
[Chapter 7](../introduction/07-transformations.md) works it through.

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
