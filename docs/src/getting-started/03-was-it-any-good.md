# Was It Any Good?

[Your First Model](02-first-model.md) produced a forecast without ever
asking whether the model deserved to be believed. This page asks.

There are two separate moments for it. **Before** fitting, you check
whether the series is the shape your model assumes. **After** fitting,
you check whether what the model left behind looks like noise — because
if it does not, there is structure still on the table.

## Before: is it stationary?

Almost every model in this package assumes a stationary series, and
almost no interesting series is one. Two tests answer this, and you
should run **both**:

```jldoctest getting-started
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(500));  # a random walk

julia> adf_test(y).pvalue > 0.10     # cannot reject a unit root
true

julia> kpss_test(y).pvalue <= 0.05   # rejects stationarity
true
```

They are not redundant, because **their null hypotheses are opposites**.
ADF's null is "there *is* a unit root"; KPSS's null is "the series *is*
stationary". Running one alone leaves you unable to distinguish "the
data says stationary" from "the data says nothing much".

Together there are four outcomes, and all four occur in practice:

| ADF | KPSS | Read it as |
|---|---|---|
| rejects | does not reject | **Stationary.** Both agree. Model it as is. |
| does not reject | rejects | **A unit root.** Both agree. Difference it. |
| does not reject | does not reject | **Not enough data to tell.** Neither test can resolve it — common on short or highly persistent series. |
| rejects | rejects | **Neither description fits.** Often a trend or a structural break rather than a unit root. |

The random walk above is the second row, which is the answer you would
hope for on a series that genuinely is one.

## Differencing, and checking it worked

```jldoctest getting-started
julia> dy = diff(y);

julia> adf_test(dy).pvalue < 0.05    # now rejects the unit root
true

julia> kpss_test(dy).pvalue > 0.05   # and no longer rejects stationarity
true
```

First row of the table: both tests now agree the differenced series is
stationary. That is the signal to stop differencing. **Differencing
again from here would be a mistake** — over-differencing inflates the
variance and plants a negative spike at lag 1 in the autocorrelation,
which is the single most common way a beginner damages a series while
trying to help it.

## Is there anything left to model?

```jldoctest getting-started
julia> length(acf(dy, 0:5).values)
6

julia> ljungbox_test(dy, 10).pvalue > 0.05   # looks like white noise
true
```

[`ljungbox_test`](@ref) pools autocorrelation across many lags into one
number. A large p-value here says there is no *linear* structure left
worth modelling — which for a differenced random walk is exactly right,
since a random walk's increments are noise by construction.

!!! warning "A large p-value is not proof of a good model"
    Every test on this page can only fail to reject. `p = 0.82` does not
    mean the residuals *are* noise; it means this particular test, at
    this sample size, could not show otherwise. A clean diagnostic panel
    means *not obviously wrong*, which is a genuinely useful thing to
    know and a considerably weaker claim than it looks.

## After: checking a fitted model

Now the real case — the model from the previous page:

```jldoctest getting-started
julia> using StatsAPI: residuals

julia> co2 = dataset("cardox").value[1:240];

julia> m = fit_sarima(co2, (1,1,1), (0,1,1,12));

julia> r = residuals(m, co2);

julia> length(r) == m.nobs    # 227: differencing consumed 13 observations
true
```

`residuals(m, co2)` takes the series explicitly, because a
`SarimaModel` does not retain the data it was fitted to. Note the
length: `227`, not `240`. R pads its own residuals back to the full
length; this package returns only what it actually computed.

Four questions, four tests:

```jldoctest getting-started
julia> ljungbox_test(r, 24; fitdf=3).pvalue > 0.05   # no leftover autocorrelation
true

julia> qs_test(r, 12).pvalue > 0.05                   # no leftover seasonality
true

julia> jarque_bera_test(r).pvalue > 0.05              # roughly normal
true

julia> arch_lm_test(r, 12).pvalue > 0.05              # no volatility clustering
true
```

All four pass, which is an unusually clean result and not what most
real series give you.

**`fitdf=3` matters.** Three parameters were estimated, so three
degrees of freedom are spent, and a Ljung-Box test that ignores this
reports a p-value that is too large — it will tell you the residuals
are fine slightly more often than it should. Pass the number of fitted
ARMA terms.

**`qs_test` is the one people skip.** It targets the seasonal lags
specifically, rather than spreading its attention across every lag the
way a portmanteau test does. On monthly data it is the test most likely
to catch a real failure, because the seasonal lag is where a
twelve-month model most often falls short.

## All of it at once

```jldoctest getting-started
julia> d = diagnostic_plot(r; fitdf=3, ppq=3, period=12);

julia> d.nlag
36

julia> minimum(d.ljungbox_pvalues) > 0.05
true
```

[`diagnostic_plot`](@ref) assembles the standard four-panel display —
standardized residuals, their ACF, a normal Q-Q plot, and Ljung-Box
p-values across a *range* of lags rather than a single one. Call
`plot(d)` with `Plots` loaded to see it.

**Pass `period=12`.** Without it the panel's lag count defaults to
`20`, which never reaches the seasonal lag at all; with it, the count
becomes `36`. A panel that stops at lag 20 on monthly data can show a
clean bill of health on a model with obvious residual seasonality,
simply because it never looked far enough.

The object holds every plotted value, so the panel is usable with no
plotting backend loaded.

## When something fails

| Symptom | Likely cause | Where to go |
|---|---|---|
| Ljung-Box rejects | Orders too low | Raise `p`/`q`, or let [`auto_arima`](@ref) search |
| `qs_test` rejects | Seasonality not absorbed | Add a seasonal term, or check `D` |
| Jarque-Bera rejects | Fat tails or outliers | Often harmless for forecasts; matters for intervals |
| ARCH-LM rejects | Variance is not constant | The mean model may be fine — see [GARCH](../manual/06-garch-and-volatility.md) |

The last row is the one worth internalising: **an ARCH-LM rejection is
not a failure of the ARIMA model.** The mean can be modelled correctly
while the variance is not, and the fix is a different model layered on
top rather than a different order.

## Next

[Beyond the Defaults](04-beyond-defaults.md) covers the handful of
settings you will actually want to change once the defaults stop being
enough.
