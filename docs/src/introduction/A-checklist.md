# Appendix A: Diagnostic Checklist

No narrative. This is the page to keep open while you work.

Each row gives the question, the function that answers it, the value
you want, and what to do when you do not get it. How to *call* each
function is in [Manual: Diagnostics](../manual/02-diagnostics.md); what
each one *means* is in Part II. This page is the order.

---

## Before fitting anything

| Check | Call | Want | If not |
|---|---|---|---|
| Is the variance stable? | Plot it. Then [`guerrero_lambda`](@ref) | λ near `1` | λ near `0` → take logs, or [`boxcox`](@ref) with the returned λ |
| Is there a trend or seasonality you can see? | `plot(y)`, [`acf`](@ref) | — | Slow ACF decay means differencing, not a high AR order |
| Is it stationary? | [`adf_test`](@ref) **and** [`kpss_test`](@ref) | ADF rejects, KPSS does not | See the four-way table below |
| Is there a seasonal period you did not know about? | [`periodogram`](@ref) | A clear peak | No peak: the seasonality may be moving — [Chapter 36](36-calendar-effects.md) |
| Are there outliers? | Plot it. [`stl_decompose`](@ref)`(y, m; robust=true)`, then `.weights` | No zero weights | Weight `0` means the fit refused to explain that point — decide whether it is real |

### The two stationarity tests together

Run **both**. Their nulls are opposites, and either alone cannot tell
"the data says stationary" from "the data says nothing".

| ADF | KPSS | Read as | Do |
|---|---|---|---|
| rejects | does not reject | Stationary | Fit |
| does not reject | rejects | Unit root | [`diff`](@ref) once, re-test |
| does not reject | does not reject | Underpowered | More data, or decide on other grounds |
| rejects | rejects | Neither fits | Suspect a trend or a structural break |

ADF has low power against a persistent alternative, so a
non-rejection there is weak evidence. Weight the KPSS result
accordingly.

---

## Choosing an order

| Check | Call | Want | If not |
|---|---|---|---|
| How many differences? | [`kpss_test`](@ref) repeatedly, or let [`auto_arima`](@ref) do it | `d ≤ 2` | `d = 3` almost always means a transformation was skipped |
| Seasonal differences? | [`nsdiffs`](@ref), then look at the ACF at the seasonal lag | `D ∈ {0, 1}` | `auto_arima` detects it if you leave `D` out. Check it anyway — the two available tests can disagree |
| Which `(p,q)`? | [`pacf`](@ref) and [`acf`](@ref), or `auto_arima` | A readable cutoff | Use `method=:ols` for `pacf` on a near-unit-root series; `:yw` is badly biased there |
| Did the search find the best model? | `auto_arima(...; trace=true)` | The model you expected appears in the trace | If it never appears, the hill-climb stopped early — try `stepwise=false` |

Comparing information criteria across models requires the **same
differencing**. AIC and BIC are built from the likelihood of the
differenced series, and `d = 1` and `d = 2` are likelihoods of
different data.

---

## After fitting

Run these in order. Each one can invalidate the ones after it.

| # | Check | Call | Want | If not |
|---|---|---|---|---|
| 1 | Did it converge? | `m.converged` | `true` | Reduce the order, try `method=:css_ml`, or difference once more |
| 2 | Are the standard errors finite? | `m.se` | All finite | `NaN` means a singular Hessian — usually an MA term on the invertibility boundary |
| 3 | Is every coefficient distinguishable from zero? | `println(m)` | \|z\| > 2 | Drop the term and refit; compare AICc |
| 4 | Is there autocorrelation left? | [`ljungbox_test`](@ref)`(r, 20; fitdf=k)` | p > 0.05 | The mean model is incomplete — read the residual ACF for where |
| 5 | At the seasonal lag specifically? | [`qs_test`](@ref)`(r, m)` | p > 0.05 | Portmanteau tests dilute this across 24 lags and miss it |
| 6 | Is the variance constant? | [`arch_lm_test`](@ref)`(r, 12)` | p > 0.05 | [GARCH](../manual/06-garch-and-volatility.md) |
| 7 | Is it a shift rather than clustering? | [`dk_heteroskedasticity_test`](@ref) | p > 0.05 | A one-off variance break is not a GARCH problem |
| 8 | Are the residuals normal? | [`jarque_bera_test`](@ref) | p > 0.05 | Point forecasts survive this; **intervals do not** |
| 9 | All at once | [`diagnostic_plot`](@ref)`(r; period=m)` | Four clean panels | **Pass `period`** or the default 20 lags never reaches the seasonal one |

**Pass `fitdf`** to `ljungbox_test` on model residuals — the number of
ARMA parameters estimated. Omitting it makes the test too forgiving,
and it is the single most common way a bad model passes.

---

## After fitting a variance model

| Check | Call | Want | If not |
|---|---|---|---|
| Is the clustering gone? | [`arch_lm_test`](@ref) on `resid ./ sqrt.(sigma2)` | p > 0.05 | Raise the order, or the mean model is the problem |
| Is persistence below 1? | `alpha + beta` | `< 1` | At `≥ 1` the unconditional variance does not exist |
| Is there asymmetry left? | [`sign_bias_test`](@ref) | Joint p > 0.05 | `model=:gjr` or `model=:egarch` |
| Did the parameters stay constant? | [`nyblom_test`](@ref) | Below the 5 % critical value | A structural break — consider splitting the sample |

---

## Before quoting a forecast

| Check | Call | Want | If not |
|---|---|---|---|
| Does it beat the obvious benchmark? | [`accuracy`](@ref)`(test, f.point, train; sp=m)` | `mase < 1` | `mase ≥ 1` means [`seasonal_naive`](@ref) would have done as well |
| Is that one lucky split? | [`tscv`](@ref) | The ranking holds across origins | One split is one sample |
| Is the accuracy quoted at the horizon you will use? | `tscv(...; h=[1, 6, 12])` | — | One-step accuracy overstates twelve-step accuracy, often by half |
| Did you pass `sp`? | — | `sp = m` for seasonal data | Without it, MASE scales against a one-step benchmark and flatters the model |

Remember that prediction intervals here treat the fitted coefficients
as **known**, so they are slightly too narrow. R and `statsmodels` do
the same; it is a shared limitation, not a divergence.

---

## When numbers disagree with R or Python

Before assuming a bug, check these five, in this order. Each has caught
a real false alarm.

1. **`nobs`.** This package reports `n - d - D*s`, R reports `n - d`,
   `statsmodels` reports the full `n`. **AIC and BIC are built from it**,
   so they are not comparable across all three.
2. **The information criterion.** `auto_arima` defaults to `:aicc`
   like R; `pmdarima` defaults to `:aic`. Different criteria select
   different models.
3. **The test statistic.** `ljungbox_test` defaults to Ljung-Box;
   R's `Box.test` defaults to **Box-Pierce**.
4. **The standard-error convention.** `:hessian` here and in R,
   OPG in `statsmodels`.
5. **The GARCH variance seed.** `rugarch` uses the whole-sample mean of
   squared residuals; this package and `arch` use a 0.94-decay backcast.
   Worth `1.05` log-likelihood units on a 1,260-point series.

[Coming from R or Python](../manual/11-coming-from-r-python.md) has
every one of these in full, with the numbers.

---

## See also

- [Manual: Diagnostics](../manual/02-diagnostics.md) — how to call everything above
- [Was It Any Good?](../getting-started/03-was-it-any-good.md) — the same sequence as a worked example
- [Appendix B](B-verification.md) — the standard every number in this book was held to
