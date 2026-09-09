# Handoff: Chapter 10 — Testing the Residuals

`docs/src/introduction/10-testing-the-residuals.md`. Target 11–12 pages.

Chapters 8 and 9 tested the data before modelling. This chapter tests
what is left after — and residuals are where a model tells you, if you
ask, whether it worked.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. A [U] claim, re-verified — partially

`all-chapters-handoff.md` tagged this chapter's disagreement **[U]**:
*the exact Durbin-Watson null distribution requires the design matrix,
not just the residuals, which is why most implementations quietly offer
only an approximation.*

**The statsmodels half is confirmed.** Verified this session:

```
statsmodels.stats.stattools.durbin_watson(resids, axis=0)
  -> 1.816583
```

It takes **only the residuals**, and returns **only the statistic —
no p-value at all**. That is the substance of the claim: unable to
compute an exact null distribution without the design matrix,
statsmodels declines to give a probability rather than give a wrong
one. The same pattern as Phillips-Perron's `ρ` in chapter 9, and worth
naming as a recurring principle.

**The R half remains unverified.** The claim that `lmtest::dwtest`
computes an exact p-value via Farebrother's algorithm because it
receives a model object could not be checked — CRAN has been
unreachable throughout this project. **Write the statsmodels half as
verified and the R half as reported but unconfirmed**, or retry CRAN
and upgrade it.

### Ljung-Box against Box-Pierce, verified

```
lag   lb_stat  lb_pvalue   bp_stat  bp_pvalue
 1     0.2768     0.5988    0.2667     0.6056
 2     0.6610     0.7185    0.6322     0.7290
 3     0.7841     0.8533    0.7477     0.8619
 4     1.1560     0.8853    1.0924     0.8955
 5     1.4848     0.9148    1.3932     0.9251
```

Ljung-Box is consistently larger than Box-Pierce and the gap widens
with lag — 3.8% at lag 1, 6.6% at lag 5. Ljung-Box applies a
finite-sample weighting that Box-Pierce omits, which matters most
exactly where sample sizes are small and the correction was designed to
help. Box-Pierce is essentially never the right choice today; it
survives in software for historical reasons.

---

## 2. The chapter, beat by beat

### Beat 1 — What a model leaves behind (about 1 page)

**Chart 1 — a fitted series and its residuals, two panels.**

*Reading:* the fit looks good. The residuals look like noise. That
impression is the thing this chapter exists to interrogate, because
"looks like noise" is precisely the judgement chapter 8 showed the eye
to be bad at.

If the model has captured everything systematic, what remains should be
unpredictable from its own past. That is a testable claim, and it is
the only claim residual diagnostics can actually check.

### Beat 2 — Look at the correlogram (about 2 pages)

The obvious first move, and it is the right one.

**Chart 2 — the ACF of those residuals.**

*Reading:* if a bar pokes outside the band, something remains that the
model did not capture, and its lag tells you what. A spike at lag 12 on
monthly residuals means seasonality is unmodelled. A spike at lag 1
means the short-run dynamics are wrong.

Refer back to chapter 5's band controversy in one sentence, because it
applies here with force — a residual spike that is significant under
one convention and not the other will change whether a reader accepts
their model.

**Chart 3 — residual ACFs from three models on the same series: one
under-specified, one adequate, one over-specified.**

*Reading:* the under-specified model leaves visible structure. The
adequate model leaves bars inside the band. The over-specified model
also leaves bars inside the band — residual diagnostics **cannot
detect over-fitting**. They test whether you have captured enough, not
whether you have used too much. That is what information criteria in
chapter 21 are for, and a reader who thinks a clean correlogram
vindicates their model has learned only half of what this chart shows.

### Beat 3 — One test instead of twenty (about 2 pages)

Looking at each lag separately means twenty chances to see something
spurious. The portmanteau tests combine them.

**Chart 4 — Ljung-Box and Box-Pierce statistics across lags on the same
residuals.**

*Reading:* use the verified table from section 1. The two curves track
each other, Ljung-Box always slightly above, and the gap widens with
lag because of the finite-sample weighting. On the eighty-point series
verified here the difference never changes a conclusion, but on a
shorter series it can, and Ljung-Box is the one to use.

**Chart 5 — the effect of the `fitdf` correction.**

*Reading, and this is the beat's real content:* residuals from a fitted
model are not raw data. Estimating `p + q` parameters uses up degrees
of freedom, and testing residuals as though it had not makes the test
too lenient — it will pass models it should fail. Show the p-value with
and without the correction on the same residuals. The uncorrected
version is systematically more forgiving, which is the opposite of what
a diagnostic should be.

State plainly what this package's default does and whether the caller
must supply `fitdf` themselves.

### Beat 4 — Two more questions to ask (about 2.5 pages)

**Chart 6 — the QS test on residuals with residual seasonality.**

*Reading:* a portmanteau test spreads its attention across all lags. If
you specifically suspect seasonality, a test aimed at the seasonal lags
has more power against that alternative. Show a case where Ljung-Box
passes and QS does not — that is the whole argument for having both.

**Chart 7 — Durbin-Watson on the same residuals.**

*Reading:* the oldest of these tests and the narrowest. It looks at
lag-1 autocorrelation only, and it comes from the regression tradition
rather than the time series one, which is why it remains standard in
econometrics papers and nearly absent from forecasting practice. Its
statistic runs from 0 to 4 with 2 meaning no autocorrelation.

Then the `disagreement` box from section 1 — statsmodels returns the
statistic and no p-value, verified; R is reported to compute an exact
one from the design matrix, unverified here. Note that this package
offers an approximation and names it as one.

The recurring principle deserves stating: **three times now — PP's
`ρ`, KPSS's clipped p-values, and Durbin-Watson — the honest
implementations refuse to produce a number they cannot compute
properly.** That is a pattern worth trusting, and its absence in a
package is worth noticing.

### Beat 5 — When the residuals pass and the model is still wrong (about 1.5 pages)

**Chart 8 — residuals that pass Ljung-Box comfortably, plotted as a
time series.**

Choose a series with volatility clustering.

*Reading:* the correlogram is clean. The Ljung-Box p-value is
comfortable. And the plot shows unmistakable structure — quiet
stretches and violent ones, alternating. Nothing tested in this chapter
detects it, because every test here asks about correlation in the
residuals and this is structure in their *variance*.

That is chapter 11.

### Beat 6 — Where this leaves you (half a page)

You can check whether a model captured the systematic movement in a
series, using a combined test rather than twenty separate looks, with
the degrees of freedom accounted for.

You cannot yet check anything about the residuals' distribution or
their variance, and chart 8 showed both matter.

No recap.

---

## 3. The `india` box

Placed in beat 4, after chart 6.

Residual seasonality is the diagnostic that matters most for Indian
monthly data, and it is the one most likely to fail. A model that
handles a fixed twelve-month cycle will still leave structure behind
when the actual cycle moves — Diwali in October one year and November
the next. The residual ACF shows a smear around lag 12 rather than a
clean spike, which is harder to spot and easy to dismiss as noise.

The QS test, aimed at the seasonal frequencies, catches this more
reliably than a general portmanteau test does.

One paragraph, forward-referencing chapter 36.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **fpp3 §5.4** | The framing — residual diagnostics as a mandatory step, not an optional one, and the insistence on looking at the plot as well as the test | The `fable` syntax |
| **Shumway & Stoffer** | The `fitdf` correction handled correctly, which many treatments skip | — |
| **Ljung and Box via Hamilton** | Why the finite-sample weighting exists | The derivation |
| **Montgomery, Jennings & Kulahci** | Durbin-Watson's regression-tradition context, which explains why it persists | Control charts |
| **Tsay** | That clean residuals with clustered variance is the normal state of financial data — the beat 5 setup | GARCH itself; that is Part V |
| **Cowpertwait & Metcalfe** | Patience with the idea that residuals are data too | R specifics |

**On examples:** any fitted model on a bundled series. `nyse` or
`gafa_stock` for beat 5's volatility clustering.

---

## 5. Voice

**The over-fitting point in beat 2 is easy to lose and worth
protecting.** Readers routinely treat a clean correlogram as proof the
model is right. It is evidence the model is not obviously wrong, which
is a much weaker claim.

**Beat 5 should feel like a trapdoor opening.** The chapter spends
eight pages building confidence in a set of tests and then shows a
series that passes all of them and is plainly not white noise. That
transition into Part V's territory is the chapter's best moment.

Avoid, beyond earlier lists:

- Presenting Box-Pierce as a live alternative. Explain why it exists
  and that Ljung-Box supersedes it.
- "The residuals should be white noise" without saying what would make
  them otherwise.

---

## 6. Checklist

- [ ] Eight charts through `@example ch10`
- [ ] Under-, correctly- and over-specified residual ACFs shown
      together, with the over-fitting blind spot stated
- [ ] Ljung-Box versus Box-Pierce table from a live run
- [ ] `fitdf` correction demonstrated with both p-values shown
- [ ] QS shown catching something Ljung-Box misses
- [ ] `disagreement` box separates the verified statsmodels half from
      the unverified R half
- [ ] The refuse-to-guess principle named, with all three instances
- [ ] Beat 5's passing-but-wrong residuals present
- [ ] `india` box on moving-festival residual seasonality
- [ ] Ends by opening chapter 11, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 7. What to do

1. **Retry CRAN once.** If `lmtest` installs, verify the Durbin-Watson
   exact-p-value claim and upgrade the box from partially verified to
   verified. If it does not, say so in the chapter.
2. Check whether this package's Ljung-Box requires `fitdf` from the
   caller or infers it from a fitted model. Beat 3 depends on the
   answer.
3. Construct beat 3's three models on the same real series so the
   comparison is like-for-like.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
