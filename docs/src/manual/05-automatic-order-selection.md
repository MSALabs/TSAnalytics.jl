# Automatic Order Selection

[`auto_arima`](@ref) searches `(p, q, P, Q)` for you and picks the fit
with the best information criterion. It is the Hyndman-Khandakar (2008)
procedure, built entirely on top of [`fit_arima`](@ref) and
[`fit_sarima`](@ref) — it contributes no fitting mathematics of its own,
only the search.

That distinction matters when a result surprises you: anything
`auto_arima` returns, you can reproduce by calling `fit_sarima`
yourself with the order it chose.

```@example auto
using TSAnalytics

y = dataset("cardox").value[1:240]   # monthly CO₂, 20 years
nothing # hide
```

## The simplest call

```@example auto
m = auto_arima(y)
println("d selected: ", m.d, "   p: ", length(m.arma.ar), "   q: ", length(m.arma.ma))
println("AIC: ", round(m.arma.aic, digits=3))
```

Non-seasonal by default. The return type follows what was searched — an
[`ArimaModel`](@ref) here, a [`SarimaModel`](@ref) once `seasonal=true`
— so reach for `m.arma.aic` in one case and `m.aic` in the other.

## Seasonal search

```@example auto
ms = auto_arima(y; seasonal=true, m=12, D=1, max_p=3, max_q=3)
println(ms.order, ms.seasonal_order, "   AIC: ", round(ms.aic, digits=3))
```

Three keywords rather than one. `seasonal=true` turns the seasonal
search on, `m=12` says what a season is, and **`D` must be passed
explicitly** — which is the one place this differs from both references.

!!! warning "`D` cannot be auto-detected"
    R's `auto.arima` detects `D` with the Canova-Hansen test and
    `pmdarima` uses OCSB. Neither seasonal unit-root test exists in this
    package yet, and defaulting `D` to an unverified guess would be
    worse than asking. Pass it explicitly: `1` for a series with a
    seasonal pattern that persists, `0` for one where it does not.

    A seasonal plot or a [`qs_test`](@ref) on the once-differenced
    series will usually settle it in a few seconds.

Ordinary `d`, by contrast, *is* detected — by repeated
[`kpss_test`](@ref), differencing while the null of stationarity is
rejected, up to `max_d`. Both references do the same:

```@example auto
using Random
Random.seed!(11)
for (lab, s) in (("white noise", randn(200)),
                  ("random walk", cumsum(randn(200))),
                  ("I(2)",        cumsum(cumsum(randn(200)))))
    println(rpad(lab, 12), " d = ", auto_arima(s; max_p=2, max_q=2).d)
end
```

## Watch it search

```@example auto
auto_arima(y; max_p=1, max_q=1, trace=true)
nothing # hide
```

`trace=true` prints each candidate and its criterion as the search
visits it. It is the fastest way to answer "why did it not pick the
model I expected" — usually the answer is that the model you expected
was never tried, because the hill-climb stopped improving before it got
there.

## The criterion decides the answer

`information_criterion` defaults to `:aicc`, matching R's
`auto.arima`. `pmdarima` defaults to `:aic`. On this series the choice
is not cosmetic:

| `information_criterion` | Selected |
|---|---|
| `:aicc` (default) | ARIMA(1,1,1)(0,1,1)[12] |
| `:aic` | ARIMA(1,1,1)(0,1,1)[12] |
| `:bic` | ARIMA(0,1,1)(0,1,1)[12] |

BIC's heavier penalty drops the AR term and returns the airline model.
Neither is wrong — they are answering different questions, and
comparing this package against `pmdarima` without first aligning the
criterion compares two of them at once. [Beyond the
Defaults](../getting-started/04-beyond-defaults.md) works through the
same comparison on the fitted coefficients.

## Stepwise or exhaustive

`stepwise=true` (the default) is the actual Hyndman-Khandakar greedy
hill-climb: fit four base models, then repeatedly try ±1 moves on
`p`, `q`, `P` and `Q`, taking the single best-improving move each round
until none improves.

`stepwise=false` is a full grid over every `(p,q,P,Q)` with
`p+q+P+Q ≤ max_order`. It is thorough and it is slow — on the seasonal
search above, roughly forty times slower, and it lands on a *different*
model, ARIMA(0,1,3)(1,1,1)[12], with a better AIC than the stepwise
answer found.

That is the honest trade: the hill-climb can stop at a local optimum.
Use it while exploring and run the grid once before committing, rather
than paying for the grid on every call.

| | `stepwise=true` | `stepwise=false` |
|---|---|---|
| Models fitted | Tens | Hundreds |
| Finds the global optimum | Not guaranteed | Within `max_order`, yes |
| `parallel` keyword | Silently ignored | Threads the grid |

`parallel` is a no-op under `stepwise=true` because a sequential
hill-climb has nothing to parallelise — each round depends on the last.
Both references scope their own `parallel=`/`n_jobs=` the same way.

## What is not searched

**`include_mean` is passed straight through**, not searched. Both
references toggle a constant on and off as part of the procedure; this
version does not. It is a deliberate first-version simplification, not
an oversight — and remember `fit_arima`/`fit_sarima` force the mean off
anyway whenever `d > 0` or `D > 0`, which covers most of the cases
where it would have mattered.

Candidates that cannot be fit — too few observations left after
differencing, say — are skipped silently, matching both references. You
get an error only if *no* candidate anywhere in the search could be fit.

## Bounding the search

| Keyword | Default | Bounds |
|---|---|---|
| `max_p`, `max_q` | `5` | Non-seasonal orders |
| `max_P`, `max_Q` | `2` | Seasonal orders |
| `max_order` | `5` | `p+q+P+Q`, in grid mode |
| `max_d`, `max_D` | `2`, `1` | How far differencing may go |

Lowering `max_p`/`max_q` is the first thing to reach for when a search
is taking too long, and it costs less than you would think — a seasonal
model rarely needs more than two or three non-seasonal terms once the
seasonal part is doing its job.

## After the search

`auto_arima` returns an ordinary fitted model, so everything else
applies to it unchanged:

```@example auto
using StatsAPI: residuals
r = residuals(ms, y)
println("Ljung-Box(24) p = ", round(ljungbox_test(r, 24; fitdf=3).pvalue, digits=4))
```

Selecting an order is not the same as validating one. The criterion
ranked this model above the others it tried; it says nothing about
whether *any* of them left white-noise residuals. Run the diagnostics.

## See also

- [Fitting ARMA Models](04-fitting-arma-models.md) — what the search calls underneath
- [Diagnostics](02-diagnostics.md) — checking the model it picked
- [Chapter 21](../introduction/21-choosing-an-order.md) — why automatic selection works at all, and when it does not
- [API: ARMA Models](../api/arma-models.md)
