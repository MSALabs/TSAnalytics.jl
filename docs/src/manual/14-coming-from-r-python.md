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
| `vars::VAR(Y, p)` | [`fit_var`](@ref) |
| `vars::VARselect` | [`var_select`](@ref) |
| `vars::roots` | [`roots`](@ref) — same convention |
| `vars::causality(…, cause=)` | [`granger_test`](@ref)`(m; cause=)` |
| `lmtest::grangertest` | [`granger_test`](@ref)`(y, x; lags=)` — a different test |
| `vars::irf(m, n.ahead=h)` | [`irf`](@ref)`(m, h)` — horizon first, `[response, impulse]` |
| `vars::fevd(m, n.ahead=h)` | [`fevd`](@ref)`(m, h)` — `decomp[variable, horizon, shock]` |
| `vars::serial.test(type="PT.asymptotic")` | [`portmanteau_test`](@ref)`(m; lags)` (`adjusted=true` for `"PT.adjusted"`) |
| `vars::serial.test(type="BG")` / `"ES"` | [`bg_test`](@ref) / [`es_test`](@ref) |
| `vars::normality.test` | [`normality_test`](@ref) |
| `vars::arch.test` | [`arch_lm_test`](@ref)`(m; lags)` |
| `urca::ca.jo(Y, ecdet="none", K=k+1)` | [`johansen_test`](@ref)`(Y; lags=k, deterministic=:const)` — different critical values, see below |
| `urca::ca.po(Y, type="Pz")` | [`phillips_ouliaris_test`](@ref) — `Pz` with a constant or trend differs, see below |
| `tseries::po.test` | not shipped — a third Phillips-Ouliaris normalization |
| `urca::cajorls(ca.jo(…), r)` | [`fit_vecm`](@ref)`(Y; lags, rank=r)` — lag coefficients differ by `Π`, see below |
| `vars::vec2var(j, r)` | [`var_rep`](@ref)`(fit_vecm(…))` |
| `tsDyn::VECM(Y, lag, r, estim="2OLS")` | [`fit_vecm`](@ref)`(…; method=:twostep)` |
| `vars::SVAR(m, Amat=…, Bmat=…)` | [`fit_svar`](@ref)`(m; A=…, B=…)` — `missing` where R has `NA` |
| `vars::BQ(m)` | [`blanchard_quah`](@ref)`(m)` |
| `HoltWinters(y)` | [`holt_winters`](@ref) |
| `forecast::ets(y)` | [`auto_ets`](@ref) |
| `forecast::ets(y, model="AAA")` | [`fit_ets`](@ref)`(y, m; trend=:add, seasonal=:add)` |
| `forecast::ets(y, model="MAM")` | [`fit_ets`](@ref)`(y, m; error=:mul, trend=:add, seasonal=:mul)` |
| `forecast::thetaf(y)` | [`fit_theta`](@ref) |
| `forecast::tbats(y)` | [`fit_tbats`](@ref) (no automatic model search) |
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
| `VAR(Y).fit(p)` | [`fit_var`](@ref) |
| `VAR(Y).select_order(L)` | [`var_select`](@ref) |
| `.roots` | `1 ./ `[`roots`](@ref)`(m)` — reciprocals, see below |
| `.test_causality` | [`granger_test`](@ref)`(m; cause=)` |
| `.test_inst_causality` | [`instantaneous_causality`](@ref) |
| `.irf(h)` / `.fevd(h)` | [`irf`](@ref) / [`fevd`](@ref) — the same numbers, different layout |
| `IRAnalysis.stderr(orth=…)` | [`irf_stderr`](@ref)`(m, h; orth=…)` |
| `.test_whiteness(nlags, adjusted=…)` | [`portmanteau_test`](@ref)`(m; lags, adjusted=…)` |
| `.test_normality()` | [`normality_test`](@ref) |
| `tsa.stattools.coint(y, X, trend=…)` | [`coint_test`](@ref)`(hcat(y, X); trend=…)` |
| `vecm.coint_johansen(Y, det_order, k_ar_diff)` | [`johansen_test`](@ref)`(Y; lags=k_ar_diff)` |
| `vecm.VECM(Y, k_ar_diff, coint_rank, deterministic="co")` | [`fit_vecm`](@ref)`(Y; lags=k_ar_diff, rank=coint_rank)` |
| `SVAR(Y, svar_type="A", A=…)` | [`fit_svar`](@ref)`(fit_var(Y, p); A=…)` — `missing` where statsmodels has `"E"` |
| `tsa.holtwinters.ExponentialSmoothing` | [`holt_winters`](@ref) |
| `tsa.exponential_smoothing.ETSModel` | [`fit_ets`](@ref) |
| `tsa.forecasting.theta.ThetaModel` | [`fit_theta`](@ref) |
| `tbats.TBATS` (the `tbats` package) | [`fit_tbats`](@ref) |

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

### R's seasonal TBATS prediction intervals are far too narrow

This is the largest of the R divergences documented here, and the one most
likely to matter in practice. For a **seasonal** TBATS model R's
`forecast::tbats` uses `c_j = alpha` for every shock weight, where the
correct weight is `w'F^(j-1)g`. At `h = 8` on the bundled quarterly
fixture R's standard error is `2.640`, against `7.31` from this package,
from Python's `tbats` package, and from three million simulated paths of
the model's own recursion. R's interval is **2.8 times too narrow**, and
the gap grows with the horizon.

R's **non-seasonal** `bats` intervals are correct — they match
`sqrt(1 + sum((alpha + j*beta)^2))` exactly — which is how the problem was
localised to the seasonal branch rather than to the variance code in
general.

The point forecasts agree: this package reproduces R's to `5e-7` and
Python's to `2e-3` at R's own parameters and seed states, and reproduces
R's SSE, `variance`, `likelihood` and `AIC` exactly.

### `fit_tbats` estimates its seed states by least squares, unlike both references

Given the smoothing parameters the TBATS recursion is affine in the
initial state, so the SSE-minimising seed can be had exactly from a
regression. Both references use a heuristic seed instead: at **identical**
smoothing parameters this package reaches an SSE of `660.96` where they
report `672.21`. Fits therefore do not reproduce theirs exactly, and are
better in sample rather than merely different.

Relatedly, `loglik` is the full Gaussian log-likelihood here, comparable
with [`fit_ets`](@ref)'s and [`fit_arima`](@ref)'s. **R's `tbats` reports
`n*log(SSE)` and calls it `likelihood`**, which is not a log-likelihood at
all; its `AIC` is that plus `2*nparams`.

### `roots` means opposite things in the two references

`vars::roots()` returns the **moduli of the companion-matrix
eigenvalues** — stable if all `< 1`. `statsmodels`' `.roots` returns the
**roots of the characteristic polynomial** — stable if all `|·| > 1`.
They are exact reciprocals, verified on five fixtures to between
`1.1e-16` and `9.3e-15`.

[`roots`](@ref) follows R. Getting this wrong silently *inverts* a
stability test, which is why it is called out here as well as in the
docstring. For `statsmodels`' convention, take `1 ./ roots(m)`.

### `deterministic=:trend` is not `statsmodels`' `trend="ct"`

R's `type="trend"` is a trend with **no constant**; `statsmodels`'
`trend="ct"` is constant **and** trend. On the bundled fixture they
differ by `6.15` in log-likelihood. [`fit_var`](@ref) spans both
vocabularies with five values and is spelled like neither reference, for
exactly this reason: `:none`, `:const`, `:trend` (R's), `:both` (`"ct"`),
`:quadratic` (`"ctt"`). R has no equivalent of the last; `statsmodels`
has none of the third.

### Order selection holds the sample fixed; a fitted model does not

Both `VARselect` and `select_order` estimate every candidate on the same
`T = n − lag_max` observations. A fitted model uses its own `n − p`. Both
numbers are right and both references report both, so
[`var_select`](@ref) and `aic(fit_var(…))` differ by construction — on
the bundled fixture, `-0.6050354269` against `-0.5819716232`.

Relatedly, `statsmodels`' `select_order(L)` returns `L+1` rows, for lags
`0…L`, where `VARselect` returns `L`. **Except with `trend="n"`**, where
a lag-0 model has no regressors at all and `statsmodels` silently drops
that row — so the off-by-one is conditional, and code that applies it
unconditionally reads the wrong lag.

### `vars::causality` reports p = 0 for any moderately strong rejection

It computes `1 - pf(...)` and `1 - pchisq(...)` instead of the upper
tail, so its p-values underflow to exactly zero in ordinary use. On
`climhyd`, an `F` of `13.36` on `(10, 2634)` degrees of freedom is
reported as `0` where the true tail is `3.66e-23`; a χ² of `184.7` on 5
df as `0` against `5.23e-38`.

This package computes the upper tail directly, so **its p-values are not
comparable with `vars` in the tail** — they agree with `statsmodels`,
which calls a survival function and gets it right.

### `statsmodels` cannot test instantaneous causality between two blocks

`test_inst_causality` raises `IndexError` whenever **both** sets have
more than one member: it indexes a mask with two fancy index arrays,
which only broadcast when one has length 1. Verified failing at (2,4),
(3,3) and (4,2) on `K = 6`, and working at (1, K−1) and (K−1, 1). It
always errors rather than returning a wrong answer, and the behaviour is
identical on 0.14.1 and 0.15.0.

For a general block `vars` is therefore the only reference, and
[`instantaneous_causality`](@ref) agrees with it there.

### Both references accept VAR inputs that cannot produce a fit

`n = 8`, `K = 2`, `p = 5` — eleven parameters per equation from three
observations — returns without complaint from both. [`fit_var`](@ref)
refuses that, along with `p < 1`, `NaN` input, `K = 1` and an
unrecognised `deterministic`, each by name and naming the fix.

### Johansen's lag count and deterministic names are traps in `urca`

Two of `ca.jo`'s arguments read as if they mean something they do not.
`K` is the order of the **levels** VAR, so `K = 2` is `lags = 1` here —
`statsmodels`' `k_ar_diff` counts differenced lags, as this package does.
And `ecdet = "none"` does **not** mean no deterministic terms: it still
fits a constant, unrestricted, outside the cointegrating relation. It is
`deterministic = :const`, which is `statsmodels`' `det_order = 0`. The
full mapping is in [Cointegration](12-cointegration.md).

### Johansen critical values with an unrestricted constant

The statistics agree with `urca` and `statsmodels` to `1e-10`; the
critical values for the common case do not agree with `urca`. Its
`ecdet = "none"` — an unrestricted constant, despite the name — reads
Osterwald-Lenum's Table **1.1\***, the case where the data's constant
lies only in the cointegrating relation (5% trace `8.18, 17.95, 31.52`).
[`johansen_test`](@ref) reads Table **1**, the case the model actually
fits (`3.76, 15.41, 29.68`), which `statsmodels`' independently simulated
table matches. `urca`'s `ecdet = "const"` and `"trend"` use the matching
tables (1\* and 2\*) and agree here exactly.

### `urca`'s Phillips-Ouliaris `Pz` with a constant or trend

`ca.po` builds the moment matrix from the raw levels where Phillips &
Ouliaris (1990) demean or detrend them first. Its statistic then changes
when a constant is added to the data, and under the null it rejected
0.25% (constant) and 0% (trend) of the time at the nominal 5% in 4000
simulations. [`phillips_ouliaris_test`](@ref) follows the paper; `Pu`,
and `Pz` with no deterministic terms, agree with `ca.po` to `1e-10`.

### An Engle-Granger statistic with automatic lag selection

`statsmodels`' `coint` and an Engle-Granger built from R's `lm` and
`urca::ur.df` agree exactly at a fixed lag and pick different lags when
each searches. [`coint_test`](@ref) uses [`adf_test`](@ref)'s search,
which is `statsmodels`'; pass `lags` with `autolag = nothing` to
reproduce either.

### `cajorls`' short-run coefficients

`urca::ca.jo` defaults to `spec = "longrun"`, which puts the levels
term at lag `K` rather than lag 1. The adjustment coefficients and `β`
are unchanged, but the lag coefficients become `Γ*_i = Γ_i + Π`.
[`fit_vecm`](@ref) reports the lag-1 form, as `statsmodels`, `tsDyn` and
`vars::vec2var` do; the test suite checks the identity.

### `statsmodels`' SVAR log-likelihood is the reduced form's

Whatever the restrictions, `SVARResults.llf` is the unrestricted VAR's
log-likelihood, so an over-identified model looks just-identified. The
usual two-variable example, `A = [1 0; E 1]` with `B = I`, is
over-identified — it also fixes both structural variances at one — and
`vars` rejects it (`LR = 121.6` on 2 df). [`fit_svar`](@ref) reports the
structural likelihood and the LR test.

### Long-run effects of an unstable VAR

Both references compute `(I - Σ A_j)⁻¹` regardless of stability.
[`lr_effects`](@ref) refuses when a companion root is on or outside the
unit circle — `econ5`'s VAR(2) is a real example — while
[`irf`](@ref) at a finite horizon keeps working.

### `vars`' diagnostic p-values

All ten of `vars`' residual-diagnostic p-values are `1 − CDF`, like its
Granger test, and print `0` for strong rejections that have representable
tails. Compare statistics with `vars`, never p-values.

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
