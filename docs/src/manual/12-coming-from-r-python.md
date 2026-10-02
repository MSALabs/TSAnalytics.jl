# Coming from R or Python

Two tables to find the function you want, and then the part that
actually matters: **where this package deliberately returns a different
number than the one you are used to.**

That list is not an apology. Some entries are places this package made
a different choice on purpose and would make it again; others are
arithmetic conventions where both answers are correct and you simply
need to know which you have. The failure mode this page exists to
prevent is comparing two numbers across languages, seeing a mismatch,
and concluding something is broken.

## From R

| R | TSAnalytics |
|---|---|
| `arima(y, order=c(p,d,q))` | [`fit_arima`](@ref)`(y, (p,d,q))` |
| `arima(y, order=, seasonal=)` | [`fit_sarima`](@ref)`(y, (p,d,q), (P,D,Q,s))` |
| `arima(y, xreg=X)` | [`fit_arimax`](@ref)`(y, (p,d,q), X)` |
| `forecast::auto.arima(y)` | [`auto_arima`](@ref)`(y)` |
| `forecast::nsdiffs(y)` | [`nsdiffs`](@ref)`(y, m)` |
| `forecast::ocsb.test(y)` | [`ocsb_test`](@ref) |
| `forecast::forecast(m, h=12)` | [`forecast`](@ref)`(m, 12)` |
| `forecast::accuracy(f, test)` | [`accuracy`](@ref) |
| `stl(y, s.window=)` | [`stl_decompose`](@ref)`(y, period)` |
| `decompose(y)` | [`classical_decompose`](@ref)`(y, period)` |
| `HoltWinters(y)` | [`holt_winters`](@ref) |
| `forecast::ets(y)` | [`auto_ets`](@ref) |
| `forecast::ets(y, model="AAA")` | [`fit_ets`](@ref)`(y, m; trend=:add, seasonal=:add)` |
| `forecast::ets(y, model="MAM")` | [`fit_ets`](@ref)`(y, m; error=:mul, trend=:add, seasonal=:mul)` |
| `forecast::thetaf(y)` | [`fit_theta`](@ref) |
| `acf(y)` / `pacf(y)` | [`acf`](@ref) / [`pacf`](@ref) |
| `Box.test(y, type="Ljung-Box")` | [`ljungbox_test`](@ref) |
| `tseries::adf.test` / `urca::ur.df` | [`adf_test`](@ref) |
| `tseries::kpss.test` | [`kpss_test`](@ref) |
| `tseries::pp.test` | [`pp_test`](@ref) |
| `tseries::jarque.bera.test` | [`jarque_bera_test`](@ref) |
| `lmtest::dwtest` | [`durbin_watson_test`](@ref) |
| `FinTS::ArchTest` | [`arch_lm_test`](@ref) |
| `rugarch::ugarchfit` | [`fit_garch`](@ref) |
| `rugarch::signbias` | [`sign_bias_test`](@ref) |
| `rugarch::nyblom` | [`nyblom_test`](@ref) |
| `stats::filter` | [`convolution_filter`](@ref) / [`recursive_filter`](@ref) |
| `forecast::BoxCox.lambda` | [`guerrero_lambda`](@ref) |
| `spec.pgram` | [`periodogram`](@ref) |

## From Python

| `statsmodels` / `pmdarima` / `arch` | TSAnalytics |
|---|---|
| `tsa.arima.ARIMA(y, order=)` | [`fit_arima`](@ref) |
| `tsa.statespace.SARIMAX(y, order=, seasonal_order=)` | [`fit_sarima`](@ref) / [`fit_sarimax`](@ref) |
| `pmdarima.auto_arima(y)` | [`auto_arima`](@ref) |
| `pmdarima.arima.nsdiffs(y, m)` | [`nsdiffs`](@ref)`(y, m; test=:ocsb)` |
| `pmdarima.arima.OCSBTest` | [`ocsb_test`](@ref) |
| `tsa.ar_model.AutoReg` | [`arx`](@ref) |
| `tsa.seasonal.seasonal_decompose` | [`classical_decompose`](@ref) |
| `tsa.seasonal.STL` | [`stl_decompose`](@ref) |
| `tsa.seasonal.MSTL` | [`mstl_decompose`](@ref) |
| `tsa.stattools.acf` / `pacf` | [`acf`](@ref) / [`pacf`](@ref) |
| `tsa.stattools.adfuller` | [`adf_test`](@ref) |
| `tsa.stattools.kpss` | [`kpss_test`](@ref) |
| `stats.diagnostic.acorr_ljungbox` | [`ljungbox_test`](@ref) |
| `stats.diagnostic.het_arch` | [`arch_lm_test`](@ref) |
| `stats.stattools.durbin_watson` | [`durbin_watson_test`](@ref) |
| `arch.arch_model` | [`fit_garch`](@ref) |
| `tsa.holtwinters.ExponentialSmoothing` | [`holt_winters`](@ref) |
| `tsa.exponential_smoothing.ETSModel` | [`fit_ets`](@ref) |
| `tsa.forecasting.theta.ThetaModel` | [`fit_theta`](@ref) |

## Where the defaults differ

Same computation, different switch position. Every one of these is
adjustable — you just have to know to adjust it.

| Setting | This package | R | Python |
|---|---|---|---|
| `kpss_test` lag rule | `nlags=:short` | `:short` | `:auto` (Hobijn) |
| `auto_arima` criterion | `:aicc` | `:aicc` | `:aic` (`pmdarima`) |
| `ljungbox_test` statistic | Ljung-Box | **Box-Pierce** (`Box.test` default) | Ljung-Box |
| ARMA standard errors | `se_type=:hessian` | Hessian | OPG (`statsmodels`) |
| GARCH covariance | `cov_type=:robust` | both reported (`rugarch`) | robust (`arch`) |

Two of these change *conclusions* rather than just digits:

**`auto_arima`'s criterion can select a different model.** On the CO₂
series in [Beyond the Defaults](../getting-started/04-beyond-defaults.md),
`:aicc` returns ARIMA(1,1,1)(0,1,1)[12] and `:bic` returns
ARIMA(0,1,1)(0,1,1)[12]. Comparing this package against `pmdarima`
without aligning the criterion compares two different questions.

**`ljungbox_test` deliberately does not match R's default.** R's
`Box.test` defaults to Box-Pierce; this package defaults to Ljung-Box,
which has better finite-sample properties — a case where matching R
would mean shipping the weaker statistic for compatibility's sake. Pass
`boxpierce=true` for R's default behaviour.

## Where the results differ deliberately

These are not settings. They are different definitions of the reported
quantity, and knowing which convention you have is the whole job.

### `nobs`, and therefore every information criterion

This package reports `n - d - D*s` — the number of observations it
actually computed a likelihood on. `statsmodels` reports the full `n`,
because it uses diffuse state augmentation and genuinely keeps them
all. R's `arima` reports `n - d`, agreeing with this package.

**AIC and BIC are built from `nobs`, so they are not comparable across
this package and `statsmodels` without accounting for it.** This is the
single most common cross-language surprise here. Neither convention is
wrong; they answer different questions about what an "observation" is
once you have differenced.

### Residual length

`residuals(m)` returns `n - d - D*s` values. R pads its own back to
the full length of the input. Same residuals, different framing of what
the leading observations mean.

### `ljungbox_test` with a vector of lags

Passing `lags=[5, 10]` here sums **exactly those two lags** into one
statistic. Python's `acorr_ljungbox(y, lags=[5,10])` instead returns
**two cumulative statistics**, one through lag 5 and one through lag 10.

A Python user expecting per-lag rows will silently get a different
number. Pass an `Integer` for the cumulative-through-`h` behaviour both
packages share.

### The GARCH variance recursion's first value

`fit_garch` seeds the variance recursion with an exponentially-weighted
backcast of the first 75 squared residuals (decay `0.94`), matching
Python's `arch` exactly. R's `rugarch` instead assigns the whole-sample
mean of squared residuals directly.

On a 1,260-point series this was worth `1.05` log-likelihood units, and
the diagnosis is verified directly against `rugarch` 1.5.6: its default
seeds `sigma2[1] = 1.9016916513`, exactly `mean(e²)`, and its
log-likelihood at this package's fitted parameters is `-2079.60906763`
against the `-2078.55638045` reported here. The formulas are otherwise
identical; the entire gap is one number.

**Align it with `rec.init=0.94`**, which asks `rugarch` for this
package's own EWMA convention and closes the gap from `1.046` to
`0.014` log-likelihood units, with coefficients to `1.5e-04`. Two
traps: a `rec.init` value at or above `1` is read as a **count of
observations** rather than a variance, so passing `1.9016` silently
means "use one data point"; and `ugarchfilter` ignores the argument, so
alignment works only at fit time.

### `sign_bias_test` follows `rugarch`, not the paper

Engle & Ng's paper describes three separate regressions. `rugarch` fits
**one** joint four-regressor regression and reads three t-values off
it. The two give different numbers; this package matches `rugarch`, and
agrees with it to all eight printed digits.

`nyblom_test` is the same situation — `rugarch` uses the outer product
`G'G` where the paper says Hessian.

### The seasonal differencing test, and which default

`nsdiffs` defaults to **`test=:seas`**, R's own default: decompose with
STL, measure the share of non-trend variation the seasonal component
carries, difference if it exceeds `0.64`. `pmdarima` defaults to
**OCSB**, a regression-based unit-root test. Pass `test=:ocsb` for that.

**The two can reach different answers.** On `log(AirPassengers)`,
seasonal strength is `0.96` so `:seas` says difference, while OCSB's
statistic is `-1.951` against a critical value of `-1.803` so `:ocsb`
says do not. R reaches both of those too, so this is a real
disagreement between accepted tests rather than an implementation
difference.

`ocsb_test` follows **R's `forecast::ocsb.test`**, not `pmdarima`'s
`OCSBTest`, which departs from R in three ways: it adds a constant to
the auxiliary regression, it does not lag the `Z4`/`Z5` regressors, and
its lag-selection index is off by one. The statistics differ by up to
`1.25` in measured cases, so `pmdarima` figures are not a cross-check.
Full numbers in [`ocsb_test`](@ref)'s own docstring.

Neither HEGY nor Canova-Hansen is implemented. Both are reachable in R
only through the separate `uroot` package, and neither is its default.

### `partrans` parameterisation

The optimiser searches a transformed parameter space (Monahan's
reparameterisation) so stationarity is enforced structurally rather
than by constraint. Standard errors are computed on the **natural**
parameters, matching R — taking the Hessian of the transformed
objective would give plausible-looking numbers that are quietly wrong.

### `fit_ets`'s log-likelihood differs from R's by a constant

[`fit_ets`](@ref) reports the full Gaussian log-likelihood,
`-n/2 * (log(2pi) + log(sse/n) + 1)`, matching `statsmodels` and every
other model in this package — so an ETS `aic` is comparable with a
[`fit_arima`](@ref) one. **R's `ets` reports `-(n/2)*log(sse)`**, which
this one exceeds by exactly `n/2 * (log(n) - log(2pi) - 1)`:
`116.976881` on a 120-point series, `66.903469` on an 84-point one,
verified against `statsmodels` to four decimals.

Account for it before comparing R's `loglik`, `aic` or `aicc` with this
package's — and note it depends on `n`, so it does not cancel between
series of different lengths. Model *rankings* are unaffected, which is why
[`auto_ets`](@ref) and R's `ets` still select the same model.

### `fit_ets`'s `sigma2`, and intervals 3.5% narrower than R's

`sigma2` is `sse/n` here, the ML estimate the likelihood is built from.
**R's `ets` uses `sse/(n - np)`.** Substituting R's value into this
package's own interval arithmetic reproduces R's 95% bound to all eight
printed decimals, so the entire difference is the denominator — the
point forecast, the `psi` weights and the interval construction are
identical. The same `n`-versus-`n-k` choice is documented for
[`arx`](@ref).

### `statsmodels`' damped ETS fits can stop at the `phi` bound

`0.8 <= phi <= 0.98` in R, in `statsmodels` and here. Hitting `0.98` is
not itself wrong — on `log(dataset("jj").value)` R returns
`phi = 0.97995483` for ETS(A,Ad,A), and so does this. But on a
simulated series where R finds an interior optimum, `statsmodels` still
returns `0.98` with a materially worse SSE (`420.31` against R's
`412.36`). For the damped models this package targets R.

### R's no-trend seasonal ETS intervals are too wide at h = m, 2m, ...

For ETS(A,N,A) and ETS(M,N,A), R's `forecast::ets` applies the seasonal
variance increment one period early — effectively counting `floor(h/m)`
elapsed seasonal shocks where the correct count is `floor((h-1)/m)`. The
seasonal state used at horizon `h = m` is `s_n`, fixed by an in-sample
innovation and therefore known at the forecast origin; R treats it as
random.

Settled by simulation rather than by argument. Four million paths from a
fitted ETS(A,N,A) give `2.5808*sigma2` at `h = 4` where R reports
`3.0536*sigma2`; six million paths from R's *own* fitted ETS(M,N,A)
state agree with this package to **0.09% at every horizon** against R
being `+25.6%` too wide at `h = 4` and `+13.4%` at `h = 8`.

R's **trended** seasonal models agree with this package exactly, so this
is one branch of R's variance code rather than a general disagreement.

### `fit_ets`'s multiplicative-error `sse` is in relative units

With `error=:mul` the innovation is relative, `eps = (y-yhat)/yhat`, so
`resid` and `sse` are too — matching what R's `residuals()` returns for
an M-error model, and three orders of magnitude smaller than the
additive-error figure on the same series. `aic`/`aicc`/`bic` remain
comparable across error types because the likelihood carries the
Jacobian `sum(log|yhat|)`; R's decomposes the same way, verified to
`1e-13`.

### R's ETS(M,Ad,M) forecast contradicts R's own fitted model

R accumulates the damped trend as `(1 + phi + ... + phi^(h-1))` in its
ETS(M,Ad,M) **forecast**, while its own one-step recursion is
`(l + phi*b)*s` and so implies `(phi + ... + phi^h)`. R's non-seasonal
damped models use the standard accumulation — `ETS(A,Ad,N)` matches
`l + sum(phi^(1:h))*b` exactly — and so does R's in-sample fitted value
for ETS(M,Ad,M) itself. Only its forecast differs.

Eight million simulated paths of the model's own recursion agree with
this package at every horizon; R's mean is off by `0.04` rising to
`0.27` over eight steps and its standard error by `0.02%` rising to
`0.8%`.

For the two **undamped** class-3 forms, ETS(M,N,M) and ETS(M,A,M), this
package and R agree to `6e-9` on the mean and `0.0000%` on the standard
error.

[`auto_ets`](@ref) searches the same fifteen models R does. Both exclude
multiplicative trend by default, and both refuse ETS(A,·,M) outright.

### `fit_theta`'s intervals differ from both references, in four ways

The point forecasts agree with R's `thetaf` and `statsmodels`'
`ThetaModel` to `2e-5`. The intervals do not, and the reasons are worth
separating because three of the four are defects in the references
rather than choices.

1. **`statsmodels`' `sigma2` ignores its own deseasonalisation.** It fits
   a `SARIMAX(0,1,1)` with drift to the *original* series while the point
   forecast is built on the deseasonalised one. On `AirPassengers` that is
   `993.23` against `104.71` for the same estimator computed consistently.
   The giveaway is that its `sigma2` is the identical number
   (`49.7153734174`) whether the bundled fixture is declared quarterly or
   non-seasonal. `use_mle=True` avoids this path.

2. **`statsmodels`' variance growth factor has the square misplaced.** It
   documents the variance as following from the model's IMA(1,1) structure
   and computes `sigma2*(1 + (h-1)*(1 + (alpha-1)^2))`; the IMA(1,1)
   result is `sigma2*(1 + (h-1)*(1 + (alpha-1))^2)`, which simplifies to
   `sigma2*(1 + (h-1)*alpha^2)` — R's formula, and the standard SES one.
   The two agree only at `alpha = 1`. At `alpha = 0.68` its intervals
   widen 2.4 times too fast. This package uses `alpha^2`.

3. **Neither reference scales the standard error by the seasonal factor.**
   With `y = x*S` and the error additive on `x`, `var(y) = S^2 var(x)`, so
   the interval scales with the point forecast. On `AirPassengers`, where
   the factors run `0.80` to `1.23`, R's intervals are about 23% too
   narrow at the seasonal peak. This package scales them; dividing the
   scaling back out reproduces R's bound.

4. **`sigma2` is `sse/n` here against R's `sse/(n-2)`** — the same
   ML-versus-adjusted choice documented above for [`fit_ets`](@ref).

Undo (3) and (4) together, with `alpha` pinned to R's value, and R's 95%
bounds come back to `3e-9`.

### R's `thetaf` returns `fitted` and `residuals` on different scales

Its `fitted` is reseasonalised; its `residuals` are the underlying SES
model's, on the deseasonalised scale. So on a seasonal series
`y - fitted` and `residuals` are different vectors — on the bundled
fixture their sums of squares are `1209.29841334` and `1315.21211028`,
9% apart. An in-sample RMSE taken from R's `residuals` is therefore on
the deseasonalised scale even though the fitted values it looks paired
with are not.

[`fit_theta`](@ref) reports both, as `sse` and `sse_adjusted`, and
`resid` is always `y - fitted`.

### The Theta seasonality test detects autocorrelation, not seasonality

Both references gate deseasonalisation on `|r_m|` against a standard
error, so any series with strong autocorrelation at lag `m` trips it.
The Nile is annual and has no seasonality whatever, yet declared
quarterly its statistic is `1.69` against a `1.64` threshold, and
declared half-yearly, `3.14`. A pure linear trend reaches `3.50` at
period 4. Pass `deseasonalize=false` when you know better.

`statsmodels`' version of the test is additionally **liberal**: it uses
`1 + sum(r^2)` where Bartlett's large-lag variance has
`1 + 2*sum(r^2)`, so it declares seasonality more often than R does.

## What is missing here that you may be looking for

| Missing | Note |
|---|---|
| HEGY / Canova-Hansen | [`nsdiffs`](@ref) covers R's default seasonal-strength heuristic and OCSB. HEGY and CH need R's separate `uroot` package and are not built here. |
| A Burg PACF via R's `pacf()` | **R's `pacf()` silently ignores its `method` argument** — all four options return the same numbers. R's real Burg estimator is `ar.burg()`, and `pacf(method=:burg)` here matches it (and `statsmodels`' `pacf_burg`) to `4e-11`. |
| `forecast` for `model=:tvss` | `model=:mle` forecasts; `:tvss` does not — `beta` is a latent state there, so it needs a projected path and a second variance term. |
| Robust SE for ARIMA in R | Not a gap here — `sandwich::vcovHC` **cannot consume an `arima` object at all**. `se_type=:robust` has no R counterpart to compare against. |

## Further reading

The Introduction carries eighteen `disagreement` boxes, each working
through one of these as the investigation it actually was — what R
returned, what Python returned, what the source said, and which one
this package follows. [Chapter 18](../introduction/18-fitting-arma.md)
compares seven estimates of a single standard error across three
languages; [Appendix B](../introduction/B-verification.md) collects the
verification standard itself.
