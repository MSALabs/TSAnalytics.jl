# Handoff: Chapter 37 — Autoregressive Errors with Changing Variance

`docs/src/introduction/37-autoregressive-errors-with-changing-variance.md`.
Target 10–11 pages.

The last chapter of Part VII, and the one that joins Part V to Part VII.
A regression with autocorrelated errors whose variance also moves — and
the chapter's best material is a single omitted term that cost 1.038
log-likelihood units.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

### The first observation needs its own variance, and omitting it is expensive

Building the combined likelihood by hand — regression, AR-structured
errors, GARCH-structured innovation variance — and checking it against
the already-verified plain fit by degenerating the GARCH part to a
constant:

**First attempt**, dropping the first observation as a conditioning
shortcut:

```
combined likelihood (degenerate):  -210.087648
target (chapter 34's verified fit): -211.125653
difference:                           1.038005
```

Off by 1.038 — far too large to be numerical noise, and the sign said
the shortcut was *adding* likelihood by discarding a term.

**Corrected**, treating the AR error's first observation as drawn from
its own stationary distribution, `σ²/(1−φ²)`:

```
combined likelihood (degenerate):  -211.125653
target:                             -211.125653
difference:                           7.68e-09
```

Machine precision. **One term, correctly handled, worth 1.038
log-likelihood units** — which on any information criterion is the
difference between two models being distinguishable and not.

### No single reference implements this combination

Checked directly. Python's `arch.univariate.ARX` combines `Y`'s **own
lags** with regressors and a pluggable volatility process — which is a
different model from "regression, then AR-structured residual". SAS's
`PROC AUTOREG` is the closest single reference and it is not runnable
here.

So this chapter's verification comes from **reduction to an
already-trusted case**, which is the same technique used in chapters 15
and 32 and in this package's own development. Say so.

---

## 2. The chapter, beat by beat

### Beat 1 — Two problems in one series (about 1.5 pages)

**Chart 1 — a regression's residuals, with their ACF and their squared
ACF. Three panels.**

*Reading:* the residuals are autocorrelated — chapter 34's problem. And
their squares are autocorrelated too — chapter 24's problem. Both at
once, in one series, which is the normal state of financial and much
macroeconomic data rather than a contrived case.

Fixing one and ignoring the other leaves the model wrong in a way the
diagnostics will catch. Chapter 12's panel would fail on two counts.

### Beat 2 — Fixing one at a time (about 2 pages)

**Chart 2 — the model with AR errors and constant variance, and its
standardised residuals.**

*Reading:* chapter 34's model. The correlation is gone from the
residuals. The variance clustering is untouched and clearly visible.

**Chart 3 — the model with GARCH errors and no AR structure.**

*Reading:* chapter 25's model applied to the regression residuals. The
variance clustering is handled; the residual correlation is not.

Neither is adequate and both are commonly done, usually in two separate
steps with the output of one fed into the other. That two-step approach
has the same flaw chapter 34 identified — the first stage was estimated
under an assumption the second stage contradicts.

### Beat 3 — Both at once (about 2.5 pages)

**Chart 4 — the combined model's structure drawn: regression, AR
filter, GARCH variance, in sequence.**

*Reading:* the construction is a chain. Subtract `X·beta` to get the
regression residual. Apply the AR filter to get the innovation. Model
that innovation's variance with GARCH. Each stage is something the
reader has already met; the novelty is estimating all three jointly.

Write the likelihood out — it is short, and it makes the chain
concrete:

```
nu_t = y_t − x_t·beta                    (regression residual)
e_t  = nu_t − phi·nu_{t−1}               (AR innovation)
h_t  = omega + alpha·e_{t−1}² + beta_g·h_{t−1}
loglik = −0.5 Σ [ log(2π h_t) + e_t²/h_t ]
```

**Chart 5 — the fitted model's conditional variance, with the
standardised residuals above it.**

*Reading:* both problems addressed. The standardised residuals should
now pass chapter 10's portmanteau test and chapter 11's ARCH-LM test.
Show both passing, and note that this is the first model in the book
that needed two different diagnostic families to validate.

### Beat 4 — The term that was nearly dropped (about 2.5 pages)

The `disagreement` box, and it is unusual — a disagreement with a
naive implementation rather than with another package.

**Chart 6 — the likelihood computed two ways across a range of
parameter values: with the first observation handled properly, and
with it dropped.**

*Reading:* the two curves are offset. Use the verified numbers — the
shortcut version gave `-210.087648` against a correct `-211.125653`, a
gap of 1.038.

Then the explanation. An AR(1) error process does not start from
nowhere. Its first value is itself drawn from the process's stationary
distribution, with variance `σ²/(1−φ²)` — larger than `σ²`, because a
persistent process wanders. Conditioning on it, or dropping it, throws
that term away and the likelihood is wrong by a constant.

**Chart 7 — the correction's size against φ.**

*Reading:* the gap grows as the AR coefficient approaches one, because
`1/(1−φ²)` diverges. For a weakly autocorrelated series the shortcut
barely matters. For the persistent series that Indian and most
macroeconomic data actually produce, it matters a great deal.

That is the chapter's practical lesson: an approximation whose error
depends on a parameter you are estimating is dangerous, because you
cannot know in advance whether it is safe.

**Chart 8 — the reduction test.**

*Reading:* set the GARCH parameters to zero so the variance is
constant, and the combined likelihood must equal chapter 34's plain fit
exactly. It does — to `7.68e-09`, which is machine precision.

This is the verification, and it is worth naming the technique: with no
reference implementation to compare against, correctness is established
by checking that the general model collapses exactly onto a special
case that *is* verified. Chapter 15 used this. Chapter 32 used it. It
is a transferable habit.

### Beat 5 — What it buys (about 1.5 pages)

**Chart 9 — prediction intervals from the constant-variance model and
the combined model, over a period containing a turbulent stretch.**

*Reading:* chapter 24 made this point about GARCH alone; here it
applies to a regression model. The constant-variance intervals are too
wide when calm and too narrow when turbulent. The combined model's
track the conditions.

Note the cost honestly: more parameters, a harder optimisation, and
convergence that is not guaranteed. For a series where the variance is
genuinely stable, none of this is worth it, and chapter 11's tests are
how you find out before committing.

### Beat 6 — Where this leaves you (half a page)

Part VII is finished. Outside information — measured regressors,
drifting relationships, calendars — can be brought into a model, with
errors that are both correlated and heteroskedastic.

Part VIII turns to what national statistical offices do with all of
this, which is a different problem with a different standard of
evidence.

No recap. One sentence marking the end of a Part.

---

## 3. The `india` box

Placed in beat 4, after chart 7.

Chart 7's finding lands directly on Indian data. Monthly Indian
inflation and industrial growth series typically show AR coefficients
between 0.7 and 0.95 — chapter 17's `india` box established this — and
the first-observation correction grows with exactly that coefficient.

At `φ = 0.9` the stationary variance is more than five times `σ²`, so
the omitted term is substantial. An implementation that takes the
conditioning shortcut will be systematically wrong on precisely the
persistent series that Indian macro work is built on, and the error will
not announce itself.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **SAS `PROC AUTOREG` documentation** | The single closest reference for this exact model shape. Worth citing precisely because no open-source package covers it cleanly | SAS syntax |
| **Tsay ch. 3** | Regression with GARCH errors in a financial setting | Option pricing |
| **Hamilton chs. 8, 21** | Both halves — GLS-type corrections for autocorrelated errors, and the ARCH likelihood — which is what the combination joins | The asymptotics |
| **Shumway & Stoffer** | Regression with correlated errors, and the exact-likelihood treatment of the first observation | — |
| **fpp3** | Nothing; this combination is outside its scope | — |
| **Montgomery, Jennings & Kulahci** | The industrial case — process variance changes as equipment ages, alongside autocorrelated measurement | Control charts |

**On examples:** the verification used chapter 34's dataset, which
keeps the reduction test exact. `nyse` or `gafa_stock` with a regressor
for a real financial case.

---

## 5. Voice

**Beat 4 is the chapter and it is a small story well told.** A shortcut
that seemed harmless, a 1.038 discrepancy that could not be explained
away, one term restored, and machine-precision agreement. It is short,
concrete, and it demonstrates the book's own working method better than
any assertion about rigour would.

**Name the reduction technique explicitly.** It has now appeared three
times and readers should recognise it as a tool they can use, not as
something this book happens to do.

**Be honest that no reference implements this.** It is unusual and it
changes the verification standard, and saying so is better than
implying a cross-check that did not happen.

Avoid, beyond earlier lists:

- Presenting the combined model as generally necessary. It is
  necessary when both problems are present, which chapter 11's tests
  determine.
- Deriving the stationary variance of an AR(1). State it and show what
  omitting it costs.

---

## 6. Checklist

- [ ] Nine charts through `@example ch37`
- [ ] Both problems shown present simultaneously in one series
- [ ] Each single fix shown leaving the other problem untouched
- [ ] The combined likelihood written out
- [ ] Standardised residuals shown passing **both** diagnostic families
- [ ] **The 1.038 discrepancy and its resolution told as the box**
- [ ] The correction's size shown growing with φ
- [ ] **Reduction test shown reaching 7.68e-09**
- [ ] The reduction technique named, with its earlier uses
- [ ] The absence of a reference implementation stated
- [ ] `india` box connecting φ ≈ 0.9 to the size of the correction
- [ ] Ends by opening Part VIII, no recap
- [ ] 10–11 pages, British-Indian spelling

---

## 7. What to do

1. Re-run both likelihood computations — shortcut and corrected — with
   a stated seed, so the 1.038 and the 7.68e-09 on the page come from
   that run.
2. Generate chart 7 by sweeping φ and recomputing the correction; the
   growth with φ is asserted here and should be measured.
3. Confirm what `fit_autoreg_garch` in this package actually does about
   the first observation before writing beat 4. If it takes the
   shortcut, the chapter has found a bug and should say so.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
