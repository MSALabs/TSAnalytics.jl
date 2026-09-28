# Your First Model

Fifteen lines, start to forecast. This page uses `@example` blocks
rather than the `jldoctest` blocks the rest of this section uses,
because the point here is partly the pictures.

## A series

```@example first-model
using TSAnalytics, Plots

d = dataset("cardox")
y = d.value[1:240]
dates = d.date[1:240]
println("Mauna Loa CO₂, ", dates[1], " to ", dates[end], "  (n = ", length(y), ")")
println("range: ", round(minimum(y), digits=2), " to ", round(maximum(y), digits=2), " ppm")
```

Monthly atmospheric CO₂ measured at Mauna Loa — the first twenty years
of it, which is enough to fit on and short enough to keep this page
quick. It ships with the package, so there is nothing to download.

## Look at it first

```@example first-model
plot(dates, y; legend=false, xlabel="", ylabel="ppm",
     title="Atmospheric CO₂, Mauna Loa")
```

Two things are visible immediately and both matter for what follows. The
series **trends** upward, steeply and without pausing. And it has a
**sawtooth** riding on that trend — an annual cycle, as the northern
hemisphere's plants take up carbon each summer and release it each
winter.

Neither is subtle here, which is the point of looking first. A model
that ignores the trend will forecast a flat line through a rising
series; one that ignores the cycle will be wrong in a different
direction every month of the year.

## Fit something

```@example first-model
m = auto_arima(y; seasonal=true, m=12, D=1)
```

That is the whole fit. [`auto_arima`](@ref) searched a space of
candidate models and returned the one that scored best, and printing it
gives the standard coefficient table.

`m=12` says the seasonal period is twelve months. `D=1` says to take one
seasonal difference — and **you have to pass that explicitly**. The
package does not yet have a seasonal unit-root test to decide it for
you, which is a real gap rather than an oversight; both R and Python use
one (Canova-Hansen, OCSB) and this package has neither yet.
[Beyond the Defaults](04-beyond-defaults.md) says more about what to do
meanwhile.

## What came back

```@example first-model
println("selected order          : ", m.order)
println("selected seasonal order : ", m.seasonal_order)
println("AIC                     : ", round(m.aic, digits=2))
println("BIC                     : ", round(m.bic, digits=2))
println("observations used       : ", m.nobs)
```

ARIMA(1,1,1)(0,1,1)[12]. Read that as: one autoregressive term, one
ordinary difference, one moving-average term, plus a seasonal
moving-average term at lag 12 and one seasonal difference.

If that shape looks familiar, it should — ARIMA(0,1,1)(0,1,1)[12] is
the *airline model*, the single most-fitted seasonal specification in
the field. The search arrived one AR term away from it, unaided, on a
series nobody told it was seasonal beyond the period.

Note `nobs = 227`, not `240`. Thirteen observations are consumed by the
differencing — one ordinary, twelve seasonal — and this package reports
the count it actually computed a likelihood on. **`statsmodels` would
report `240` here.** Neither is wrong; they are different conventions,
and since every information criterion is built from `nobs`, AIC is not
comparable across the two without knowing which you have. This is the
most common cross-language surprise in the whole package, and
[Coming from R or Python](../manual/11-coming-from-r-python.md)
collects the rest.

```@example first-model
using StatsAPI: coef
println("coefficients: ", round.(coef(m), digits=4))
```

## Forecast

```@example first-model
f = forecast(m, y, 12)
println("next 3 months : ", round.(f.point[1:3], digits=2))
println("95% interval, first month: [", round(f.lower[1,2], digits=2), ", ",
        round(f.upper[1,2], digits=2), "]")
```

```@example first-model
plot(1:length(y), y; label="observed", xlabel="month index", ylabel="ppm",
     title="twelve months ahead, with 80% and 95% intervals")
plot!(length(y)+1:length(y)+12, f.point; ribbon=(f.point .- f.lower[:,2], f.upper[:,2] .- f.point),
      label="forecast", linewidth=2)
```

The forecast continues both features the plot showed: the climb and the
sawtooth. The interval widens with horizon, which is the model saying
what it genuinely does and does not know — a one-month-ahead statement
is worth more than a twelve-month-ahead one, and the picture should show
that rather than hide it.

## What just happened

Six steps ran, none of which you had to ask for:

1. **The differencing order was chosen by a test**, not by eye — a KPSS
   stationarity test on the series, repeated until it stops rejecting.
2. **The seasonal difference was the one thing you supplied** (`D=1`),
   for the reason given above.
3. **Candidate orders were searched stepwise**, not exhaustively — a
   small number of neighbouring models are tried and the search walks
   downhill, which is what makes this finish in seconds rather than
   minutes.
4. **Models were scored on AICc**, the small-sample-corrected
   information criterion. R's `auto.arima` defaults to the same thing;
   Python's `pmdarima` defaults to plain AIC, so the two can select
   different models on identical data.
5. **Each candidate was fitted by maximum likelihood through a Kalman
   filter.** Every ARMA, ARIMA and SARIMA fit in this package runs the
   same state-space recursion underneath — the topic of Part VI of the
   Introduction, and genuinely the same code.
6. **The intervals came from the state covariance** the filter was
   already carrying, not from a separate calculation bolted on
   afterwards.

## Next

The fit above has not been checked. It reported a model and a forecast,
and nothing so far asked whether the model *fits* — whether what it
left behind looks like noise, or like structure it failed to capture.

That question is [Was It Any Good?](03-was-it-any-good.md), and it is
the more important half.
