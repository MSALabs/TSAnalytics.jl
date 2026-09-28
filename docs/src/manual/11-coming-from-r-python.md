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
| `forecast::forecast(m, h=12)` | [`forecast`](@ref)`(m, y, 12)` |
| `forecast::accuracy(f, test)` | [`accuracy`](@ref) |
| `stl(y, s.window=)` | [`stl_decompose`](@ref)`(y, period)` |
| `decompose(y)` | [`classical_decompose`](@ref)`(y, period)` |
| `HoltWinters(y)` | [`holt_winters`](@ref) |
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

`residuals(m, y)` returns `n - d - D*s` values. R pads its own back to
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

On a 1,260-point series this was worth `1.05` log-likelihood units.
Holding parameters fixed and changing only that seed reproduces
`rugarch`'s likelihood to `1.58e-09`, so the formulas are otherwise
identical — the entire gap is one number. `rugarch` figures therefore
cannot be used as targets for anything likelihood-based without
aligning the seed first.

### `sign_bias_test` follows `rugarch`, not the paper

Engle & Ng's paper describes three separate regressions. `rugarch` fits
**one** joint four-regressor regression and reads three t-values off
it. The two give different numbers; this package matches `rugarch`, and
agrees with it to all eight printed digits.

`nyblom_test` is the same situation — `rugarch` uses the outer product
`G'G` where the paper says Hessian.

### `partrans` parameterisation

The optimiser searches a transformed parameter space (Monahan's
reparameterisation) so stationarity is enforced structurally rather
than by constraint. Standard errors are computed on the **natural**
parameters, matching R — taking the Hessian of the transformed
objective would give plausible-looking numbers that are quietly wrong.

## What is missing here that you may be looking for

| Missing | Note |
|---|---|
| Seasonal unit-root test | So `D` must be passed explicitly to `auto_arima`. R uses Canova-Hansen, `pmdarima` uses OCSB. |
| `pacf(method=:burg)` | `:yw`, `:ywm`, `:ols` only; the error message names `:burg` explicitly. |
| `dist=:t` for GARCH | Normal innovations only. |
| `forecast` for `ArimaxModel`/`SarimaxModel` | They fit, but have no forecast method yet. |
| Robust SE for ARIMA in R | Not a gap here — `sandwich::vcovHC` **cannot consume an `arima` object at all**. `se_type=:robust` has no R counterpart to compare against. |

## Further reading

The Introduction carries eighteen `disagreement` boxes, each working
through one of these as the investigation it actually was — what R
returned, what Python returned, what the source said, and which one
this package follows. [Chapter 18](../introduction/18-fitting-arma.md)
compares seven estimates of a single standard error across three
languages; [Appendix B](../introduction/B-verification.md) collects the
verification standard itself.
