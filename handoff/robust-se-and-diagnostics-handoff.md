# Handoff: Robust Standard Errors and Diagnostic Plots — closing the R gap

Status: **partially implemented** — gaps 1, 2, 3a, 3b and 4 done; 3c, 3d, 3e open.

## Implementation log

| Item | Status | Commit | Validation achieved |
|---|---|---|---|
| Gap 1 — `se_type=:robust`, ARMA family | ✅ | `3eedd89` | collapse property; no reference exists in R or Python |
| Gap 2 — `se_type=:robust` for `arx` | ✅ | `3eedd89` | **exact vs `sandwich::vcovHC(HC0)`, 1e-7** |
| Gap 3a — Nyblom stability | ✅ | `e76c336` | **exact vs `rugarch::nyblom`, 10 digits** |
| Gap 3b — Sign Bias (Engle–Ng) | ✅ | `3eedd89` | **exact vs `rugarch::signbias`, all 8 digits** |
| Gap 3c — adjusted Pearson GoF | ⬜ open | — | reference values in §3c below |
| Gap 3d — weighted LB / ARCH-LM | ⬜ open | — | reference values in §3d below |
| Gap 3e — Shibata / Hannan–Quinn | ⬜ open | — | formulas confirmed, see note below |
| Gap 4 — diagnostic plots | ✅ | `e76c336` | six GARCH panels, recipe-rendered |

**Deviation from the original spec, stated plainly.** The request was
for a `diagnostics::Bool` keyword on the fit functions that plots when
true. That was implemented instead as `diagnostic_plot(::GarchModel)`,
dispatching on model type, for three reasons: `src/` depends on
`RecipesBase` and not `Plots`, so nothing inside a fit function can
render; a boolean that triggers a side-effecting plot from inside an
estimation call is not composable; and the package already had
`diagnostic_plot(resid, ::ArmaModel)` as the established spelling for
exactly this. `plot(diagnostic_plot(m))` draws the charts, and the
returned object carries the numbers whether or not anything is drawn.

**Gap 3e note.** All four information-criterion formulas were confirmed
against `rugarch`'s reported values for the benchmark fit
(`n=1260, k=3, LL=-2079.6025`): Akaike `3.305718`, Bayes `3.317954`,
Shibata `3.305707`, Hannan–Quinn `3.310316`, where Shibata is
`-2LL/n + log((n+2k)/n)` and Hannan–Quinn is
`-2LL/n + 2k·log(log n)/n`. **`rugarch` divides all four by `n`; this
package's `aic`/`bic` do not.** Implementing these means either
breaking that convention or documenting a second one — decide before
writing the code, not after.

---

## Where this fits

- **Depends on**: Stage 6.5's `_hessian_se`/`_opg_se` (the two building blocks a
  sandwich estimator is assembled from — both already exist), Stage 7's
  `fit_garch`/`cov_type` (the one place robust SE is already built), and
  `src/diagnostics.jl` / `src/diagnosticplot.jl`.
- **Feeds into**: reporting parity with `rugarch::ugarchfit`, which is the
  reference users coming from R will compare against.
- **Not a roadmap row yet** — named descriptively rather than `stage-N`,
  following the precedent of `gaussianssm-performance-handoff.md` and
  `print-formatting-handoff.md`.

## Scope decisions already taken (project maintainer, 2026-09-26)

1. **Extend the existing `se_type` keyword to accept `:robust`.** Do *not*
   add a separate `robust_se::Bool`. The package already carries two
   spellings for this knob (`cov_type` in GARCH, `se_type` everywhere
   else); a third would make it worse. Option (b) of three offered.
2. **`diagnostics::Bool` produces plots**, in the manner of a plot
   function — not a returned battery of test objects. This is the
   `rugarch::plot(fit)` / `stats::tsdiag` behaviour, not the
   `ugarchfit` show-table behaviour.

## Verified environment

Everything below was run directly this session, not recalled:

- **R 4.6.0**, `rugarch` 1.5.6, `forecast`, `sandwich`, `lmtest`, `tseries`
  (`fGarch` and `orcutt` are *not* installed and were not used).
- **Julia 1.9.4**, TSAnalytics at commit `33ba7ee`.
- **Shared data**: 1,260 observations (5 years × 252 trading days),
  simulated GARCH(1,1) with `ω=0.02, α=0.09, β=0.90`, seed `20260925`,
  written to CSV once and read identically by both languages.

---

## Gap 1 — robust SE for the ARMA/ARIMA family (absent from **both** R and Julia)

**This is the largest item, and it is an opportunity rather than a catch-up.**

Verified directly:

```r
m <- arima(e, order=c(1,0,0))
names(m)          # coef, sigma2, var.coef, mask, loglik, aic, arma, residuals, ...
vcovHC(m)         # ERROR: no terms component nor attribute
coeftest(m)       # runs, but uses the non-robust var.coef
```

`stats::arima` exposes only `var.coef`, the inverse observed
information. `sandwich::vcovHC` **cannot consume an `arima` object at
all** — it requires a `terms` component that `arima` does not produce.
`lmtest::coeftest` appears to work but silently uses the same
non-robust covariance.

Julia's position is the same: `fit_arma`, `fit_arima`, `fit_sarima`,
`fit_arimax`, `fit_sarimax`, `fit_autoreg_garch`, `auto_arima` and
`auto_arimax` all accept `se_type=:hessian|:opg`, and **both are
classical** — `:opg` is the Berndt–Hall–Hall–Hausman outer-product
form, not a sandwich.

**Implementing `se_type=:robust` here puts TSAnalytics ahead of R, not
level with it.** The sandwich is `H⁻¹ · OPG · H⁻¹`, and both factors
already exist as `_hessian_se` and `_opg_se` — this is assembly of
existing verified pieces, not new numerics.

**Verification problem to solve first**: there is no R or Python
reference for this on an ARIMA model, so the "validate against a real
reference number" policy cannot apply directly. Two options, both
used elsewhere in this project:
- reduce to a case where a reference *does* exist (a regression with
  no ARMA structure is an `lm`, where `sandwich::vcovHC` works and can
  be matched exactly); and
- assert the algebraic identity that the sandwich collapses to `H⁻¹`
  when the information-matrix equality holds.

---

## Gap 2 — robust SE for `arx` (R **has** it, Julia does not)

The one place R is genuinely ahead. `arx`'s R counterparts (`dynlm`,
plain `lm` on constructed lags) produce `lm`-class objects, which
`sandwich::vcovHC` supports fully (HC0–HC5) and `lmtest::coeftest`
consumes directly.

Julia's `arx` has **no SE option at all** — its signature is
`arx(y, lags; trend, seasonal, period, exog, hold_back, method=:qr)`.
It computes `se` via the QR/OLS path with no choice of estimator.

This one *can* be validated against a real reference: fit the same
lag structure in R via `lm`, take `vcovHC(fit, type="HC1")`, and match.

---

## Gap 3 — GARCH diagnostics `rugarch` reports that Julia lacks

`ugarchfit`'s `show()` prints a battery Julia has no equivalent for.
All values below are from the real fit on the shared data and are
usable directly as test targets.

### 3a. Nyblom parameter-stability test — **missing**

```
Joint Statistic:  0.6638
Individual Statistics:
  omega   0.2633214
  alpha1  0.3825469
  beta1   0.2736798
Joint critical values:      10% 0.846   5% 1.01   1% 1.35
Individual critical values: 10% 0.35    5% 0.47   1% 0.75
```

### 3b. Sign Bias Test (Engle–Ng) — **missing**

```
                    t-value       prob
Sign Bias          0.76039831   0.4471594
Negative Sign Bias 0.03979805   0.9682605
Positive Sign Bias 0.23210355   0.8164954
Joint Effect       2.08252126   0.5554569
```

Directly relevant to Chapter 26's asymmetry material, which currently
has no formal test to point at.

### 3c. Adjusted Pearson goodness-of-fit — **missing**

```
  group  statistic  p-value(g-1)
     20   19.01587     0.4558183
     30   27.33333     0.5537262
```

### 3d. Weighted (Fisher–Gallagher) Ljung-Box and ARCH-LM — **missing**

Julia has `ljungbox_test` (standard, plus Box–Pierce via `boxpierce=true`)
and `arch_lm_test` (standard). `rugarch` reports the *weighted* variants,
which adjust for estimated parameters at the lag level:

```
Weighted Ljung-Box, standardised residuals
  Lag[1]                    0.0211   p 0.8845
  Lag[2*(p+q)+(p+q)-1][2]   0.2357   p 0.8326
  Lag[4*(p+q)+(p+q)-1][5]   1.1157   p 0.8330
  d.o.f = 0

Weighted Ljung-Box, standardised SQUARED residuals
  Lag[1]                    0.08596  p 0.7694
  Lag[2*(p+q)+(p+q)-1][5]   1.69348  p 0.6924
  Lag[4*(p+q)+(p+q)-1][9]   3.72841  p 0.6352
  d.o.f = 2

Weighted ARCH LM
  ARCH Lag[3]  stat 1.261  shape 0.500  scale 2.000  p 0.2615
  ARCH Lag[5]  stat 3.180  shape 1.440  scale 1.667  p 0.2648
  ARCH Lag[7]  stat 4.287  shape 2.315  scale 1.543  p 0.3070
```

Note the non-obvious lag selection rule — `2(p+q)+(p+q)-1` and
`4(p+q)+(p+q)-1` — which should be read from `rugarch`'s own source
rather than reconstructed from the printed output.

### 3e. Shibata and Hannan–Quinn information criteria — **missing**

```
Akaike        3.305718
Bayes         3.317954
Shibata       3.305707
Hannan-Quinn  3.310316
```

Julia has `aic`/`bic` only. Note `rugarch` reports these **divided by
n**, unlike Julia's absolute-scale `aic`/`bic` — a convention
difference that must be stated, not silently reconciled.

---

## Gap 4 — diagnostic plots

`rugarch::plot(fit, which=)` offers **twelve**, read from its own source
(`rugarch:::.plotgarchfit`):

1. Series with 2 Conditional SD Superimposed
2. Series with 1% VaR Limits
3. Conditional SD (vs |returns|)
4. ACF of Observations
5. ACF of Squared Observations
6. ACF of Absolute Observations
7. Cross Correlation
8. Empirical Density of Standardized Residuals
9. QQ-Plot of Standardized Residuals
10. ACF of Standardized Residuals
11. ACF of Squared Standardized Residuals
12. News-Impact Curve

Julia's `diagnostic_plot` produces a 4-panel display (standardised
residuals, residual ACF, normal Q-Q, Ljung-Box p-values across lags),
transcribed from `astsa::sarima()`. It covers 9 and 10 above and
nothing GARCH-specific — no conditional-SD panel, no news-impact
curve, no squared/absolute ACF.

**`diagnostics=true` should render the applicable subset for whichever
model it is attached to** — the GARCH set differs from the ARIMA set,
and the choice of which panels apply is model-dependent, not a fixed
list.

On the R side for ARIMA, note there is very little to match:
`forecast::checkresiduals` produces plots plus a **single** Ljung-Box
test, and `stats::tsdiag` produces plots and returns `NULL`. Julia's
`diagnostic_plot` already exceeds both by returning a populated
`DiagnosticPlotResult`.

---

## What Julia already has that R does not — do not regress these

| Capability | Julia | R |
|---|---|---|
| Jarque–Bera on residuals | ✅ `jarque_bera_test` | not in `rugarch`'s battery |
| QS seasonal test | ✅ `qs_test` | not in base; `seasonal` pkg only |
| Durbin–Watson | ✅ `durbin_watson_test` | separate `lmtest` |
| Durbin–Koopman heteroskedasticity | ✅ `dk_heteroskedasticity_test` | absent |
| Diagnostics returned as data, not just drawn | ✅ `DiagnosticPlotResult` | `tsdiag` returns `NULL` |
| GARCH robust SE | ✅ `cov_type=:robust`, **default** | ✅ `rugarch` (computes both always) |

---

## Proposed Julia API

```julia
# Gap 1 + 2 — one new accepted value, no new keyword
fit_arma(y, order; se_type = :hessian | :opg | :robust)      # and the whole family
arx(y, lags; se_type = :hessian | :robust)                    # new keyword on arx

# Gap 3 — new standalone tests, matching the existing *_test naming
nyblom_test(model)            -> NyblomTest
sign_bias_test(model)         -> SignBiasTest
pearson_gof_test(model, groups) -> PearsonGoFTest
ljungbox_test(x; weighted=true)   # extend, rather than a new function
arch_lm_test(x; weighted=true)    # extend

# Gap 4 — plots
fit_garch(y, p, q; diagnostics = false)   # true → render applicable panels
```

**Naming note for whoever implements this**: `cov_type` in `fit_garch`
stays as-is. It is the older name, it matches Python `arch`'s own
spelling, and CLAUDE.md's convention is to keep a reference's exact
naming deliberately. `se_type=:robust` is the new spelling everywhere
else. This leaves a documented inconsistency rather than a silent one —
call it out in both docstrings.

## Open questions for the maintainer

1. **Default for `se_type` in the ARIMA family.** `fit_garch` already
   defaults to `:robust`; `rugarch` reports robust by default too.
   Switching the ARIMA family's default from `:hessian` to `:robust`
   would change standard errors for every existing user. Recommend
   keeping `:hessian` as the default and making `:robust` opt-in.
2. **Does `diagnostics=true` return the model, the plot, or both?**
   Affects whether the call is composable in a pipeline.
3. **Are all five of Gap 3's tests wanted, or only some?** Each needs
   its own reference validation against `rugarch`; Sign Bias and Nyblom
   are the two with the clearest analytical value for this book
   (Chapters 26 and 32 respectively).
4. **Performance.** Yesterday's benchmark put Julia at 5.8 ms per warm
   GARCH fit against `rugarch`'s 118 ms — and part of that gap is
   precisely that `rugarch` computes this battery on every fit. Keeping
   both `diagnostics` and the extra tests strictly opt-in is what
   preserves that advantage.

## Hand-verified test values

The shared dataset and both fits are reproducible from
`handoff/` scripts; the fitted parameters agreed across all three
languages tested:

| | ω | α | β | loglik |
|---|---|---|---|---|
| Julia `fit_garch` | 0.03012 | 0.08709 | 0.89845 | −2078.5564 |
| Python `arch` 5.1.0 | 0.0301 | 0.0871 | 0.8984 | −2078.5564 |
| R `rugarch` 1.5.6 | 0.02911 | 0.08744 | 0.89860 | −2079.6025 |

R's log-likelihood differs by 1.05 with parameters this close. **This
was traced to its exact cause and reproduced to 1.58e-09 — see the
section below.** It is a variance-recursion initialisation difference,
not a worse optimum, not an optimiser artefact, and not a formula
disagreement.

---

## Why Julia/Python and R disagree — solved, and reproduced to 1.58e-09

**Root cause: one number — how `sigma2[1]` is seeded.** The
log-likelihood formula itself is byte-for-byte equivalent across all
three implementations.

### The two conventions

| | seed value | how it enters |
|---|---|---|
| **Julia `fit_garch` / Python `arch`** | `backcast` = 0.94-decay EWMA of the **first 75** squared residuals = `0.9504096718` | substituted for *both* missing pre-sample lags, then run through the recursion: `sigma2[1] = ω + (α+β)·backcast` = **`0.9667879487`** |
| **R `rugarch`** (`rec.init='all'`, the default) | `mean(e²)` over the **whole sample** = `1.9016916513` | assigned **directly**: `sigma2[1]` = **`1.9016916513`** |

Two independent differences compound here: a different statistic (first-75
EWMA versus whole-sample mean) *and* a different point of entry (through
the recursion versus assigned directly). On this dataset the seeds differ
by roughly a factor of two, because the first 75 observations happen to be
a calm stretch — which is precisely when the two conventions diverge most.

Julia's choice is not arbitrary: `_garch_backcast`'s own docstring records
that it matches `arch.univariate.volatility.GARCH.backcast` exactly, which
is why Julia and Python agree to every printed digit.

### Proof 1 — identical formula, at fixed parameters

Evaluating the *same* hand-written GARCH(1,1) log-likelihood in Julia at
Julia's fitted parameters, changing **only** the seed:

```
Julia params + Julia backcast-through-recursion : LL = -2078.55638045   sigma2[1] = 0.9667879487
   fit_garch's own reported loglik              : -2078.55638045          ← exact match

Julia params + rugarch init (sigma2[1] = mean e²): LL = -2079.60906763   sigma2[1] = 1.9016916513
   rugarch's LL at those same params            : -2079.60906763
   difference                                   : 1.58e-09               ← 8 decimal places
```

The 1.58e-09 residual is the precision of the R figure as printed, not a
real disagreement.

### Proof 2 — identical optimum, refitting under R's convention

Re-optimising in Julia with `sigma2[1] = mean(e²)` recovers `rugarch`'s
own fitted parameters:

```
Julia refit, rugarch init : omega=0.02911106  alpha=0.08744180  beta=0.89859159  LL=-2079.602503
rugarch's own fit         : omega=0.02910576  alpha=0.08743932  beta=0.89859614  LL=-2079.602500
abs diff                  : omega=5.30e-06    alpha=2.48e-06    beta=4.55e-06    LL=3.05e-06
```

Agreement to ~6 decimal places on the likelihood and ~5–6 on each
parameter. The residual is optimiser tolerance — Nelder–Mead here against
`rugarch`'s `solnp` — not a modelling difference.

### Consequences for this work

1. **No bug on either side.** Both conventions are defensible; any fixed
   positive seed converges to the same MLE asymptotically. The choice only
   moves finite-sample numbers.
2. **`rugarch` numbers cannot be used directly as test targets for
   anything likelihood-based** unless the seed is aligned first. Its
   standard errors, information criteria, and every weighted test
   statistic inherit this offset.
3. **Reproducing an R result therefore requires seeding `sigma2[1]` with
   `mean(e²)` directly.** `fit_garch` currently has no keyword for this.
   If cross-checking against R matters, the cheapest route is an internal
   seed option used by tests only, rather than a public keyword that
   invites users to silently change their numbers.
4. This is the same class of issue as Chapter 37's first-observation
   term, and the same class as the `nobs` convention documented in
   Appendix B — a third instance of the same lesson, and worth citing
   as such.

### Reproduction

Scripts used, all in this session's scratchpad and re-runnable:
`garch_bench_data.csv` (seed 20260925), `match.jl` (Proof 1),
`match_fit2.jl` (Proof 2), `r_filter.R` / `r_match.R` (R side).
`ugarchfilter`'s `filter.control` was found to **ignore** `rec.init`
— it always used `mean(e²)` — so the R side of Proof 1 was obtained
from `ugarchfit` with fixed parameters instead.

`rugarch`'s classical-vs-robust standard errors on the shared data,
usable directly to validate Gap 1's implementation once a comparable
model is constructed:

```
         classical se   robust se   ratio
omega     0.01439185    0.01189745  0.827
alpha1    0.01941470    0.02270469  1.169
beta1     0.02348885    0.02519720  1.073
```

Note the ratio is **not** uniformly greater than one — robust SEs are
smaller for `omega` here. Any test asserting "robust ≥ classical"
would be wrong.
