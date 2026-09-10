# Handoff: Chapter 25 — ARCH and GARCH

`docs/src/introduction/25-arch-and-garch.md`. Target 12–13 pages.

Chapter 24 established that the variance moves and is predictable. This
chapter models it, using an idea that is genuinely simple and whose
notation is genuinely confusing — and the confusion is the disagreement
box.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Two [U] claims, both re-verified

### The order convention: confirmed, and the reason for the confusion is now clear

Read from `arch` 8.0.0's own source:

```python
GARCH(p: int = 1, o: int = 0, q: int = 1, power: float = 2.0)
```

So in this package **`p` is the ARCH order, `q` is the GARCH order, and
`o` is a separate slot for asymmetric terms**. ARCH comes first.

That matches R's `garchOrder = c(p, q)` convention. The confusion in
circulation comes from a different source — some texts and wrappers
write GARCH(p, q) with `p` as the *GARCH* order, which is the exact
reverse. Both conventions exist in print.

**The practical consequence is that GARCH(1,1) is safe and everything
else is not.** With `p = q = 1` the two conventions coincide and nobody
notices. A GARCH(2,1) means two different models depending on whose
notation you are reading, and since GARCH(1,1) is what almost everyone
fits, the ambiguity survives unexamined.

### The covariance default: confirmed

Fitting GARCH(1,1) to a simulated series, `n = 1500`:

```
params:      omega 0.090584   alpha[1] 0.090088   beta[1] 0.870850
se (default) 0.029439         0.016731            0.023705
se (classic) 0.031199         0.017134            0.025569
```

`arch`'s default is the robust sandwich estimator, not the classical
one, and the standard errors differ by roughly 6% on `omega`. This is
the same family of problem as chapter 18's four-way standard-error
disagreement, and referring back to it is worth a sentence — the reader
has met this before and should recognise it as a pattern rather than a
one-off.

---

## 2. The chapter, beat by beat

### Beat 1 — Model the squares (about 1.5 pages)

**Chart 1 — chapter 24's squared-return ACF, once more.**

*Reading:* squared returns are autocorrelated, which means squared
returns can be predicted from past squared returns. That sentence is
the whole idea. An autoregression on squared returns is exactly what
chapter 17 built, applied to a different quantity.

Engle's contribution in 1982 was not the mechanism, which was familiar.
It was noticing that the variance was the thing to model, at a time
when variance was assumed to be a nuisance parameter.

### Beat 2 — ARCH (about 2 pages)

**Chart 2 — a simulated ARCH(1) series with its conditional variance
plotted beneath it. Two panels.**

*Reading:* the variance panel spikes immediately after every large
return and decays back. That is the model: today's variance is a
constant plus a multiple of yesterday's squared shock. Big move
yesterday, big expected move today.

**Chart 3 — ARCH(1), ARCH(4) and ARCH(12) fitted to the same real
series, with their conditional variances overlaid.**

*Reading:* ARCH(1) is too jumpy — the variance drops back almost
immediately and real volatility does not behave that way. Adding lags
smooths it, and by ARCH(12) the fit is reasonable and twelve parameters
have been spent on it.

This is chapter 17's parsimony problem in a new setting, and it has the
same solution.

### Beat 3 — GARCH (about 2.5 pages)

**Chart 4 — GARCH(1,1) fitted to the same series, its conditional
variance overlaid on ARCH(12)'s.**

*Reading:* nearly the same variance path, from three parameters instead
of twelve. The GARCH term feeds yesterday's *variance* back in, which
gives an infinite memory with geometrically declining weights — exactly
what ARCH(12) was approximating with twelve separate coefficients.

Report the actual fitted parameters. `omega = 0.090584`,
`alpha = 0.090088`, `beta = 0.870850` from the verified run, or your
own.

**Chart 5 — persistence. The variance forecast decaying back to its
long-run level for three different `alpha + beta` values.**

*Reading:* `alpha + beta` governs how long a shock's effect lasts. At
0.96 — which is the verified fit above, and entirely typical — a
volatility shock takes months to fade. As the sum approaches one the
decay becomes arbitrarily slow, and at one the variance never reverts
at all.

The parallel with chapter 8's unit root is exact and worth drawing
explicitly. This is the same near-unit-root problem in a different
place, and it has the same consequence: the model is estimable but the
long-run behaviour it implies is barely identified from the data.

Note that `omega > 0` is required for the variance to stay positive,
and that `alpha, beta ≥ 0` are needed for the same reason. Chapter 26
shows a model that escapes these constraints.

### Beat 4 — Fitting one (about 2 pages)

**Chart 6 — the conditional variance from a fitted GARCH(1,1) on
`nyse`, with the returns above it.**

*Reading:* walk through the fit end to end. The likelihood is
maximised numerically; the constraints from beat 3 must hold; and
convergence is not guaranteed, which is why multi-start fitting exists.

**Chart 7 — standardised residuals and their squared ACF.**

*Reading:* this is the diagnostic that matters. Divide each return by
its fitted conditional standard deviation. If the model captured the
variance dynamics, the standardised residuals should have no remaining
structure in their squares — chapter 11's ARCH-LM test applied to the
standardised residuals should now pass.

Show it passing, and note that this is the volatility equivalent of
chapter 10's residual check. The tools transfer.

### Beat 5 — Two conventions and one default (about 2.5 pages)

**Chart 8 — the same data fitted as "GARCH(2,1)" under both order
conventions, with the two conditional variances overlaid.**

*Reading:* they are different models and the chart shows two different
variance paths. Neither is wrong; they answer to different notation.

Then the `disagreement` box using section 1. The structure worth using:
`arch`'s signature is `GARCH(p, o, q)` with `p` the ARCH order,
matching R's `garchOrder = c(p, q)`. Some texts reverse it. GARCH(1,1)
is immune because the two coincide there, which is why the ambiguity
has survived — almost nobody fits anything else.

Then the second part: `arch` defaults to robust standard errors, not
classical, with a roughly 6% difference on this fit. Refer back to
chapter 18 explicitly. **This is the second time the same class of
problem has appeared, and naming it as a pattern is more useful than
treating it as a new surprise.**

State this package's own conventions for both.

### Beat 6 — Where this leaves you (half a page)

You can model a variance that moves, with three parameters, and check
whether it worked.

The model treats a large fall and a large rise identically. Anyone who
has watched a market fall knows that is not how volatility behaves.

Chapter 26.

No recap.

---

## 3. The `india` box

Placed in beat 3, after chart 5.

Fitted `alpha + beta` for Indian equity indices typically lands in the
0.95 to 0.99 range — high persistence, meaning volatility shocks decay
slowly, and close enough to one that the long-run variance the model
implies is poorly determined.

That has a practical edge. A GARCH model fitted across a period
containing a structural break will report very high persistence,
because the break looks like an extremely slow-decaying shock. High
estimated persistence is therefore weak evidence of genuine persistence
and reasonable evidence that chapter 24's variance-shift test should
have been run first.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Tsay ch. 3** | The primary treatment. Model specification, the stylised facts, and the practical fitting guidance | The distributional variants |
| **Engle (1982), Bollerslev (1986)** | Cite both properly; the ARCH-to-GARCH step is one of the cleaner ideas in econometrics | Derivations |
| **Hamilton ch. 21** | Why the likelihood takes the form it does and what the constraints buy | The asymptotics |
| **Shumway & Stoffer ch. 5** | GARCH fitted to `nyse`, and standardised residuals as the diagnostic | — |
| **Montgomery, Jennings & Kulahci** | Little; volatility modelling is outside their tradition, which is worth noticing | — |
| **fpp3** | Nothing; fpp3 does not cover GARCH | — |

**On examples:** `nyse`, `sp500w`, `djia`, `gafa_stock` bundled. The
simulated ARCH series should be labelled as simulated.

---

## 5. Voice

**The idea is simple and the notation is not.** Beat 1 should land the
idea in two sentences before any Greek letters appear. Readers who
understand "autoregression on the squares" will follow everything else.

**The near-unit-root parallel in beat 3 is worth drawing carefully.**
It connects Part V back to Part II and shows the reader that the same
structural problem recurs across the subject.

Avoid, beyond earlier lists:

- Deriving the GARCH likelihood.
- Listing the GARCH variants. There are dozens; chapter 26 covers the
  two that matter.
- Calling `alpha + beta` "the persistence" without saying persistence
  of what.

---

## 6. Checklist

- [ ] Eight charts through `@example ch25`
- [ ] The idea stated before the notation
- [ ] ARCH(1) shown being too jumpy; ARCH(12) shown being expensive
- [ ] GARCH(1,1) shown matching ARCH(12) with three parameters
- [ ] Persistence chart, with the near-unit-root parallel drawn
- [ ] Standardised-residual diagnostic shown passing
- [ ] `disagreement` box covers **both** the order convention and the
      covariance default, with verified numbers
- [ ] Chapter 18 referenced explicitly as the same class of problem
- [ ] Package's own conventions stated for both
- [ ] `india` box on persistence and structural breaks
- [ ] Ends by opening chapter 26, no recap
- [ ] 12–13 pages, British-Indian spelling

---

## 7. What to do

1. Re-run the GARCH fit and both covariance types during drafting, with
   a stated seed.
2. Confirm this package's parameter-order convention and its default
   covariance estimator before writing beat 5.
3. For chart 8, fit both order conventions explicitly rather than
   describing the difference.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
