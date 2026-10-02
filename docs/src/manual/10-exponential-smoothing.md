# Exponential Smoothing, ETS and Theta

An ARIMA model says what a series is. An exponential smoothing model
says how to update a belief about it: keep a running estimate of the
level, and maybe of the slope and the seasonal pattern, and revise each
one by a fraction of every new forecast error. The fractions —
`alpha`, `beta`, `gamma` — are the parameters.

This package has three entry points to that family. Two are the
classical recursions and the full likelihood-based ETS; the third,
[`fit_theta`](@ref), looks like a different method entirely and turns
out to be simple exponential smoothing with a drift term — it has its
own section at the end.

The difference between the first two is worth getting straight before
you pick one.

| | [`holt_winters`](@ref) | [`fit_ets`](@ref) |
|---|---|---|
| Form | Classical recursions | Innovations state space |
| Reports | `sse` | `sse`, `loglik`, `aic`, `aicc`, `bic` |
| Initial states | Heuristic, fixed | Optimised jointly |
| Damped trend | No | Yes (`phi`) |
| Model selection | — | [`auto_ets`](@ref) by AICc |
| Reference | R's `HoltWinters` | R's `forecast::ets` |

If you want the textbook recursions with parameters you choose or have
optimised against SSE, `holt_winters` is the smaller, more direct tool.
If you want a likelihood — and therefore information criteria, and
therefore automatic selection — you want `fit_ets`. If you want the
competition-winning default with almost nothing to choose, skip to
[The Theta method](#The-Theta-method).

## The taxonomy, and which part of it is here

ETS models are labelled `ETS(E,T,S)` for the error, trend and seasonal
components, each additive, multiplicative or absent. Thirty
combinations exist. **Six are implemented**, and they are exactly the
ones that are linear:

| `trend` | `seasonal` | Model | Also known as |
|---|---|---|---|
| `:none` | `:none` | ETS(A,N,N) | Simple exponential smoothing |
| `:add` | `:none` | ETS(A,A,N) | Holt's linear |
| `:damped` | `:none` | ETS(A,Ad,N) | Damped Holt |
| `:none` | `:add` | ETS(A,N,A) | Seasonal, no trend |
| `:add` | `:add` | ETS(A,A,A) | Additive Holt-Winters |
| `:damped` | `:add` | ETS(A,Ad,A) | Damped additive Holt-Winters |

The other twenty-four involve a multiplicative error or a
multiplicative seasonal. Those are not linear Gaussian state space
models: their exact prediction intervals do not have a closed form and
have to be simulated. They are **refused by name** rather than
approximated silently —

```@example ets
using TSAnalytics
try
    fit_ets(randn(40), 4; seasonal=:mul)
catch e
    println(e.msg)
end
```

The practical workaround is older than the taxonomy: take logs. A
series whose seasonal swing grows with its level is additive on the log
scale, and that is what the rest of this page does.

## Fitting one

```@example ets
using TSAnalytics, Statistics

y = log.(dataset("jj").value)   # quarterly J&J earnings per share, 1960-80
m = fit_ets(y, 4; trend=:add, seasonal=:add)
```

The period is a positional argument, defaulting to `1`. Pass it
whenever `seasonal=:add`; the series must cover at least two full
periods.

```@example ets
println("alpha = ", round(m.alpha, digits=6))
println("beta  = ", round(m.beta,  digits=8))
println("gamma = ", round(m.gamma, digits=6))
println("sse   = ", round(m.sse,   digits=6))
```

`beta` has gone to its lower bound. That is not a failure: `log(jj)`
grows at an almost exactly constant rate, so the slope needs no
updating and the model keeps the one it started with. The seasonal
pattern, by contrast, gets a large `gamma` — it moves.

The fitted components are available separately, which is most of the
reason to prefer an ETS over an ARIMA when you have to explain the
model to somebody:

```@example ets
println("level[1:3]    : ", round.(m.level[1:3], digits=4))
println("trend[1:3]    : ", round.(m.trend_component[1:3], digits=6))
println("seasonal[1:4] : ", round.(m.seasonal_component[1:4], digits=4))
```

### Pinning parameters

`fixed` takes a `NamedTuple` and removes those parameters from the
optimisation rather than merely starting from them:

```@example ets
f = fit_ets(y, 4; trend=:add, seasonal=:add, fixed=(alpha=0.3, beta=0.05))
println("alpha = ", f.alpha, "  nparams = ", f.nparams,
        "  (free fit had ", m.nparams, ")")
```

### The admissible region

`constraint=:traditional` (the default) is R's region: each parameter in
`(0,1)`, with `beta <= alpha` and `gamma <= 1 - alpha`.
`constraint=:admissible` instead asks only that the underlying
state-space system be stable, which is strictly weaker — so it can only
ever fit at least as well, at the cost of parameters that are harder to
interpret.

```@example ets
t = fit_ets(y, 4; trend=:add, seasonal=:add, constraint=:traditional)
a = fit_ets(y, 4; trend=:add, seasonal=:add, constraint=:admissible)
println("traditional sse = ", round(t.sse, digits=6))
println("admissible  sse = ", round(a.sse, digits=6))
```

Identical, here — this series' optimum is comfortably *inside* the
traditional region, so widening the region finds nothing new. That is
the usual outcome. `:admissible` earns its place on series where the
traditional constraints bind, which you can detect by a parameter
sitting exactly on `alpha`, on `1 - alpha`, or on `0`.

## Letting it choose

[`auto_ets`](@ref) fits all six and returns the lowest **AICc** —
Hyndman's recommendation for finite samples, and what R's `ets`
selects by.

```@example ets
best = auto_ets(y, 4)
println(notation(best))
```

```@example ets
for (tr, se) in ((:none,:none), (:add,:none), (:damped,:none),
                 (:none,:add),  (:add,:add),  (:damped,:add))
    local fi = fit_ets(y, 4; trend=tr, seasonal=se)
    println(rpad(notation(fi), 12), "  sse = ", rpad(round(fi.sse, digits=4), 8),
            "  aicc = ", round(fi.aicc, digits=2))
end
```

Note that the ranking is **not** the SSE ranking. ETS(A,Ad,N) fits
better than ETS(A,A,N) — `1.8923` against `1.9220` — and still loses on
AICc, because the damping parameter is not worth what it buys. On 84
observations that correction is doing real work.

With more than one thread available the six fits run in parallel. The
parallel and serial paths select the *identical* model, not merely an
equally good one — asserted in the test suite rather than assumed.

```@example ets
println(notation(auto_ets(y, 4; parallel=false)) == notation(best))
```

Pass `seasonal=false` to restrict the search to the three non-seasonal
candidates. It is also applied automatically when the series is shorter
than two periods.

## Forecasting

```@example ets
fc = forecast(best, 8)
println("point : ", round.(fc.point[1:4], digits=4))
println("levels: ", fc.levels)
println("se    : ", round.(fc.se[1:4], digits=4))
```

`predict(m, h)` is the same function. Because this series was logged,
undo the transform to read the forecast — and remember that
exponentiating a mean forecast gives a **median** on the original scale:

```@example ets
println("earnings per share: ", round.(exp.(fc.point[1:4]), digits=3))
```

[Chapter 7](../introduction/07-transformations.md) covers when to
correct for that and when not to.

The point forecast is the state propagated with zero innovations, which
for a linear model is exact. The consequence worth knowing is what
`phi` does to the long run:

```@example ets
u = forecast(fit_ets(y, 4; trend=:add,    seasonal=:add), 60)
d = forecast(fit_ets(y, 4; trend=:damped, seasonal=:add), 60)
# compare points one full year apart, so the seasonal term cancels
println("undamped, year 14 -> 15: ", round(u.point[60] - u.point[56], digits=6))
println("damped,   year 14 -> 15: ", round(d.point[60] - d.point[56], digits=6))
```

An undamped trend extrapolates in a straight line forever. A damped one
converges to `l_n + phi/(1-phi) * b_n`. Over a long horizon that is
usually the difference between a forecast somebody will believe and one
they will not.

## Where the numbers differ from R and `statsmodels`

All three agree on what the model *is*. They report it differently, in
three ways that will otherwise look like bugs.

### The log-likelihood differs from R's by a constant

`loglik` here is the full Gaussian log-likelihood,
`-n/2 * (log(2pi) + log(sse/n) + 1)`, matching `statsmodels` and —
more to the point — matching every other model in this package, so an
ETS `aic` is comparable with a [`fit_arima`](@ref) one.

**R's `ets` reports `-(n/2)*log(sse)`.** This package's figure exceeds
R's by exactly `n/2 * (log(n) - log(2pi) - 1)`:

```@example ets
n = length(y)
println("constant on n = ", n, ": ", round(n/2 * (log(n) - log(2pi) - 1), digits=6))
```

So R's `loglik`, `aic` and `aicc` are not comparable with this
package's without accounting for it — and note the constant depends on
`n`, so it does not even cancel when comparing two fits to series of
different lengths. The *rankings* are unaffected, which is
why `auto_ets` and R's `ets` still select the same model.

### `sigma2` is `sse/n`, where R uses `sse/(n - np)`

The maximum-likelihood estimate, consistent with the likelihood it is
built from. R's degrees-of-freedom-adjusted version is larger, so **R's
prediction intervals are about 3.5% wider** on a 120-point fit.

Substituting R's `sigma2` into this package's own interval arithmetic
reproduces R's 95% bound to all eight printed decimals — the point
forecast, the `psi` weights and the interval construction are
identical, and the whole gap is the denominator. Neither choice is
wrong; the same `n`-versus-`n-k` decision is documented for
[`arx`](@ref).

### `statsmodels`' damped fits can stop at `phi = 0.98`

`0.98` is the upper bound on `phi` in R, in `statsmodels` and here —
`0.8 <= phi <= 0.98`, adopted deliberately. Hitting it is not by itself
a problem: on `log(jj)` R's damped fit returns `phi = 0.97995483`,
i.e. the bound, and so does this one.

On a simulated series where R finds an *interior* optimum, though,
`statsmodels` still returns `0.98` with a materially worse SSE —
`5044.65` and `420.31` against R's `5002.51` and `412.36`. For the
damped models R is the better reference, and that is the one this
package targets.

## The reduction onto `holt_winters`

ETS(A,A,A) *is* classical additive Holt-Winters, and the two
implementations here agree to `1e-13`. But you cannot check that by
passing the same three numbers to both, for two reasons that are easy
to miss:

1. **The parameterisations differ.** `holt_winters` is classical,
   `fit_ets` is innovations. The map is `beta = alpha * beta_star` and
   `gamma = gamma_star * (1 - alpha)`.
2. **The windows differ.** For a seasonal model `holt_winters` starts
   its recursion at `t = m+1`, scoring `n - m` observations; `fit_ets`
   scores all `n`.

Match both and they coincide. Since `holt_winters` is independently
verified against R's `HoltWinters`, that makes this a real check rather
than two of this package's own guesses agreeing — the test suite
asserts it, including that the *unmapped* parameters do **not** agree.

Similarly, `trend=:damped` with `phi` pinned to `1.0` is bit-identical
to `trend=:add`:

```@example ets
a = fit_ets(y, 4; trend=:damped, fixed=(alpha=0.3, beta=0.1, phi=1.0), initial=:heuristic)
b = fit_ets(y, 4; trend=:add,    fixed=(alpha=0.3, beta=0.1),          initial=:heuristic)
println("sse difference: ", abs(a.sse - b.sse))
```

## The Theta method

[`fit_theta`](@ref) is the method that won the M3 competition
(Assimakopoulos & Nikolopoulos 2000). Its original description — rescale
the series' local curvature by a factor `theta`, fit a line to each of
two rescalings, forecast both and average — sounds like something quite
unlike exponential smoothing. Hyndman & Billah (2003) showed it is not:
it reduces **exactly** to simple exponential smoothing plus half the
regression slope as drift. That equivalence is what is implemented here,
because it is exact rather than an approximation.

```@example theta
using TSAnalytics, Statistics

y = dataset("jj").value       # the levels this time, not the logs
t = fit_theta(y, 4)
```

Three numbers describe the whole model: `alpha` from the SES fit, `b0`
the regression slope, and the drift the forecast applies, `b0/2`.

```@example theta
println("alpha      = ", round(t.alpha, digits=6))
println("drift/step = ", round(t.b0 / 2, digits=6))
println("level      = ", round(t.level, digits=4))
```

### Seasonality is handled by adjustment, not by a component

Unlike [`fit_ets`](@ref), Theta has no seasonal state. A seasonal series
is **divided** by a classical multiplicative seasonal figure first, SES
and the drift are fitted to what remains, and the forecast is multiplied
back. Whether to do that at all is decided by a test on the lag-`period`
autocorrelation:

```@example theta
println("statistic = ", round(t.seasonal_statistic, digits=4),
        "  vs threshold ", round(TSAnalytics._confidence_z(0.10), digits=4))
println("adjusted  = ", t.deseasonalized)
```

!!! warning "That test detects autocorrelation, not seasonality"
    It compares `|r_m|` against Bartlett's standard error, so **any**
    series with strong autocorrelation at lag `m` trips it. The Nile is
    annual data with no seasonality whatever, and declared quarterly its
    statistic is `1.69` against the `1.64` threshold — a false positive.
    Declared half-yearly it is `3.14`. A pure linear trend reaches `3.50`
    at period 4.

    Both references gate on the same quantity, so this is inherited
    rather than chosen, but pass `deseasonalize=false` when you know
    better. `seasonal_test=false` forces the adjustment the other way.

Because the adjustment is multiplicative, the series has to be strictly
positive — so the log transform that [`fit_ets`](@ref) needed for this
same series is not available here, and is not needed:

```@example theta
try
    fit_theta(log.(y), 4)
catch e
    println(e.msg)
end
```

### Forecasting

```@example theta
f = forecast(t, 8)
println("point : ", round.(f.point[1:4], digits=4))
println("se    : ", round.(f.se[1:4], digits=4))
```

The drift is **never damped**, so the forecast is a straight line
extended indefinitely, with the seasonal pattern repeating on top. There
is no `phi` and no mechanism to flatten a long horizon — which is the
method's main structural limitation, and shows up immediately on a
series that does not grow linearly:

```@example theta
train, test = y[1:72], y[73:84]
rmse(a, p) = accuracy(a, p).rmse
println("Theta             : ", round(rmse(test, forecast(fit_theta(train, 4), 12).point), digits=4))
println("ETS on logs       : ", round(rmse(test, exp.(forecast(auto_ets(log.(train), 4), 12).point)), digits=4))
println("seasonal naive    : ", round(rmse(test, repeat(train[end-3:end], 3)), digits=4))
```

Theta beats the seasonal naive benchmark and loses clearly to ETS here.
That is not a surprise and not a defect: J&J earnings grow roughly
*exponentially*, ETS can model that on the log scale, and Theta's drift
is additive and linear. On the M3 data — mostly short series without
that kind of curvature — the ranking goes the other way, which is how it
won.

### Two sums of squares, because R reports two

```@example theta
println("sse          = ", round(t.sse, digits=4), "   (y - fitted, original scale)")
println("sse_adjusted = ", round(t.sse_adjusted, digits=4), "   (the SES fit's own)")
```

R's `thetaf` returns `fitted` on the original scale but `residuals` on
the deseasonalised one, so on a seasonal series its `y - fitted` and its
`residuals` are two different vectors — 9% apart in sum of squares on
the verification fixture. Both are reported here under distinct names,
and `resid` is always `y - fitted`. If you have been computing an
in-sample RMSE from R's `residuals` on a seasonal Theta fit, it is on the
deseasonalised scale.

The prediction intervals differ from both references in several further
ways, three of which are defects in the references; they are enumerated
in [Coming from R or Python](12-coming-from-r-python.md).

## What is not here

- **Multiplicative error or seasonality** — the other twenty-four
  forms. Take logs, or use [`fit_sarima`](@ref).
- **Prediction intervals that account for parameter uncertainty.** As
  everywhere else in this package, and in R and `statsmodels`, the
  fitted parameters are treated as known, so the intervals are slightly
  too narrow.
- **TBATS.** It sits behind the multiplicative forms in the queue,
  because it needs them; see
  [The Frontier](../introduction/41-the-frontier.md).
- **A damped Theta.** The drift is linear and unbounded, as in both
  references. Use a damped ETS when a long horizon has to flatten.

## See also

- [Fitting ARMA Models](04-fitting-arma-models.md) — the other way to model a seasonal series
- [Forecasting and Accuracy](09-forecasting-and-accuracy.md) — evaluating what comes out of `forecast`
- [Coming from R or Python](12-coming-from-r-python.md) — every documented divergence in one place
- [API: ARMA Models](../api/arma-models.md) — `fit_ets`, `auto_ets`, `fit_theta`, `holt_winters`
