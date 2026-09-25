# Handoff: Chapter 34 — Regression with ARIMA Errors

`docs/src/introduction/34-regression-with-arima-errors.md`. Target 12–13 pages.

Part VII opens. Everything in Parts IV to VI treated a series as
arriving on its own. This chapter brings in outside information — and
its disagreement box is the one place in the book where the standard
citation and the standard implementation do not match.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

### R and Python both do the simple thing, and the textbook citation says otherwise

The usual reference for regression inside a state space model is de
Jong (1991), on diffuse filtering for regression coefficients. That is
what the roadmap for this package originally assumed both references
did.

**Neither does.** Confirmed from R's own `stats::arima` source:

```r
x <- x - xreg %*% par[narma + (1L:ncxreg)]
```

The regression coefficients live in `par` — the *same* parameter vector
as `phi` and `theta`. At every likelihood evaluation `X·beta` is
subtracted from the series first, and the ordinary ARMA likelihood is
computed on what remains. No state augmentation. No diffuse
initialisation. The state space is untouched.

Python's defaults, confirmed from `SARIMAX`'s signature:

```
mle_regression           default = True
use_exact_diffuse        default = False
time_varying_regression  default = False
```

Same approach — coefficients as ordinary MLE parameters.

**Confirmed numerically**, AR(1) plus one regressor, `n = 150`:

```
R      ar1 = 0.476774167   x = 1.894806442   loglik = -211.1256527
Python ar1 = 0.47677706    x = 1.89480149    llf    = -211.1256527433344
```

Essentially exact agreement, both from default settings.

So the diffuse machinery of chapter 33 — which this book spent a whole
chapter on — is **not** what either reference uses for this problem. It
is real, it is correctly implemented, and it belongs to chapter 35's
drifting-coefficient case and to chapter 41's unobserved components
models. It does not belong here.

That is worth saying plainly. It is unusual for a book to report that
its own reference chapter turned out not to be needed for the obvious
application.

---

## 2. The chapter, beat by beat

### Beat 1 — A relationship worth using (about 1.5 pages)

**Chart 1 — two series that genuinely move together, plotted on twin
axes.**

Use `cmort` and `tempr` — mortality and temperature, Shumway &
Stoffer's own pair, both bundled.

*Reading:* colder weeks have higher mortality. That is a real
relationship with a physical mechanism, and no univariate model in
Parts IV to VI could use it. An ARIMA model of mortality alone would
have to infer next week's mortality from mortality history, while
ignoring a thermometer.

**Chart 2 — an ordinary least squares fit of one on the other, with its
residuals and their ACF.**

*Reading:* the fit looks reasonable and the residuals are heavily
autocorrelated. Chapter 8 explained exactly what that means — the
standard errors are wrong, the t-statistics are inflated, and the
relationship may be real but the reported precision is not.

Report the actual Durbin-Watson statistic or residual ACF values.

### Beat 2 — Two wrong ways round (about 2 pages)

**Chart 3 — the residuals from chart 2 fitted with an ARMA, in a second
step.**

*Reading:* the obvious fix. Regress first, model the residuals second.
It is better than ignoring the problem and it is still wrong, because
the regression coefficients were estimated under an assumption of
independence that has now been contradicted. Fixing the second stage
does not repair the first.

**Chart 4 — differencing both series and regressing the differences.**

*Reading:* the other obvious fix, and it throws away the level
relationship entirely. If mortality and temperature are related in
levels, a regression of changes on changes answers a different
question. Sometimes that is the question you want; usually it is not.

Both approaches are common in practice and both are compromises. The
right answer is to estimate everything at once.

### Beat 3 — Estimate it all together (about 2.5 pages)

**Chart 5 — the joint fit: regression coefficient and ARMA parameters
estimated simultaneously.**

*Reading:* the model says the series equals `X·beta` plus an error term
that follows an ARMA process. The likelihood is computed by subtracting
`X·beta` first and running the ARMA likelihood on the remainder — which
is exactly chapter 18's likelihood, with two extra parameters in the
search.

Report the fitted coefficient and compare it against the OLS estimate
from chart 2. They will differ, and more importantly the standard
errors will differ substantially.

**Chart 6 — the regression coefficient's confidence interval from OLS
and from the joint fit, side by side.**

*Reading:* the OLS interval is too narrow, usually by a lot. This is
chapter 8's spurious-regression problem in a milder form — autocorrelated
errors inflate apparent precision — and the joint fit is what fixes it
properly.

### Beat 4 — Differencing the regressors too (about 2 pages)

**Chart 7 — a trending response and a trending regressor, fitted with
`d = 1` applied to both and to only one.**

*Reading:* if the response is differenced, the regressors must be
differenced identically. Otherwise the model is relating changes in one
thing to levels of another, which is almost never what anyone means.

This is a real and easy mistake, and most software handles it silently
— which is convenient and worth knowing about, because a user who
differences the response by hand and passes undifferenced regressors
gets a model that runs and means nothing.

State what this package does.

### Beat 5 — The citation and the code (about 3 pages)

The `disagreement` box, and it has an unusual shape.

**Chart 8 — the same data fitted by R and by Python, with default
settings, coefficients and log-likelihoods reported.**

*Reading:* use the verified numbers. They agree to five decimals.

Then the box. The standard citation for regression in state space
models is de Jong (1991), on diffuse filtering. Both R and Python cite
that literature. **Neither uses it by default.**

Show R's actual source line. Show Python's three defaults. Both fold
the coefficients into the same optimiser as the ARMA parameters — the
simplest possible approach, and the one a reader would guess before
being told about diffuse filtering at all.

The diffuse machinery is genuinely there in Python, behind
`time_varying_regression=True` and `use_exact_diffuse=True`, and it
exists for a different problem — coefficients that *move*, which is
chapter 35.

The lesson worth drawing: **a citation describes what is possible, not
necessarily what is running.** Reading the source settled a question
that reading the documentation would not have.

And a note for this book specifically: chapter 33's diffuse
initialisation is not wasted. It is needed for chapter 35, and for
chapter 41's unobserved components models — where the initial states
genuinely have no prior. Just not here.

### Beat 6 — Where this leaves you (half a page)

You can bring outside information into a model and get honest standard
errors for it.

The coefficient has been one number throughout. Chapter 32 showed that
sometimes it should not be.

Chapter 35.

No recap.

---

## 3. The `india` box

Placed in beat 1, after chart 1.

The natural Indian example for this chapter is monthly industrial
production against a policy rate, or agricultural output against
rainfall. The rainfall case is the cleaner one — the mechanism is
physical rather than behavioural, the data exists, and the relationship
is strong enough to survive a short sample.

It also sets up a limitation. Rainfall affects output with a lag that
depends on the crop cycle, and a contemporaneous regressor cannot
represent that. Lagged regressors are the fix and this package does not
provide a dedicated distributed-lag utility — a gap worth naming
honestly.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **fpp3 ch. 10** | The framing — "regression with ARIMA errors" is fpp3's name for this and it is the clearest one. Also the insistence on differencing regressors identically | The `fable` syntax |
| **Shumway & Stoffer** | `cmort`/`tempr`/`part` as the running example, and their treatment of lagged regression | — |
| **Hamilton ch. 8** | Why autocorrelated errors invalidate OLS inference, done properly | The GLS theory |
| **Montgomery, Jennings & Kulahci** | The transfer-function tradition, which is this idea from the engineering side and richer on lag structure | The full transfer-function apparatus |
| **Tsay** | Regression with time series errors in a financial setting | — |
| **de Jong (1991)** | Cite it — and then report that neither reference implementation uses it here. That contrast is the chapter's most interesting half-page | — |

**On examples:** `cmort`, `tempr`, `part` bundled and canonical.
`global_economy` for a macro pair.

---

## 5. Voice

**Beat 5's finding should be reported without triumph.** Both packages
made a sensible engineering choice; the citation describes a more
general framework that happens not to be needed for the common case.
Nobody is wrong. What is interesting is that reading the code and
reading the citation give different impressions.

**Do not let chapter 33 look wasted.** Say explicitly where diffuse
initialisation *is* needed — chapters 35 and 41 — so a reader does not
conclude they read that chapter for nothing.

Avoid, beyond earlier lists:

- Calling this "ARIMAX" without noting that the term is used
  inconsistently, sometimes for this model and sometimes for a
  different one with lagged dependent variables.
- Deriving the GLS connection.

---

## 6. Checklist

- [ ] Eight charts through `@example ch34`
- [ ] OLS residual autocorrelation shown, with real numbers
- [ ] Both two-step compromises shown and criticised fairly
- [ ] Joint fit compared against OLS, **standard errors** contrasted
- [ ] Regressor differencing shown, package behaviour stated
- [ ] `disagreement` box shows **R's actual source line** and Python's
      three defaults
- [ ] The verified five-decimal agreement reported
- [ ] Chapter 33's relevance preserved — say where diffuse *is* needed
- [ ] `india` box on rainfall, with the lagged-regressor gap named
- [ ] Ends by opening chapter 35, no recap
- [ ] 12–13 pages, British-Indian spelling

---

## 7. What to do

1. Re-run the R/Python comparison with a stated seed so the numbers
   reproduce from the text.
2. Confirm this package's regressor-differencing behaviour before
   writing beat 4.
3. Check whether a distributed-lag helper exists before naming it as a
   gap in the `india` box.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
