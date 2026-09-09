# Handoff: Chapter 11 — Testing Distribution and Variance

`docs/src/introduction/11-testing-distribution-and-variance.md`. Target 11–12 pages.

Chapter 10 ended with residuals that passed every correlation test and
were obviously not noise. This chapter picks that up, and by its end
the reader will have the diagnostic that motivates the whole of Part V.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

Both tests in this chapter were constructed and executed during this
project, on cases where the answer was known by construction, and both
were checked in **both** directions — a test that fires on the positive
case and stays quiet on the negative one is worth far more than one
checked only where it should fire.

### ARCH-LM (Engle's test for conditional heteroskedasticity)

```
ARCH effects present:  statistic = 25.345   p = 4.29e-05   -> rejects, correctly
white noise:           statistic =  2.977   p = 0.562      -> does not reject, correctly
```

Constructed as: regress squared residuals on their own lags, take
`(n − lags) · R²`, compare against χ² with `lags` degrees of freedom.

### The Durbin & Koopman variance-ratio test

```
heteroskedastic residuals:  statistic = 13.059   p ≈ 4.44e-16  -> rejects, correctly
homoskedastic residuals:    statistic =  1.036   p = 0.901     -> does not reject, correctly
```

Constructed as: split standardised residuals into thirds, take the
ratio of the sum of squares of the last third to the first, compare
against an F distribution.

Note how interpretable the statistic is — **1.036 on homoskedastic data
is almost exactly 1**, which is what a ratio of two equal variances
should be. That transparency is worth pointing out; most test
statistics do not wear their meaning so plainly.

---

## 2. The chapter, beat by beat

### Beat 1 — The residuals from last chapter (about 1 page)

**Chart 1 — the chapter 10 residuals again, with their Ljung-Box
p-value printed on the plot.**

Deliberate continuity. Same data, same comfortable p-value, same
obvious structure.

*Reading:* every test in chapter 10 asked one question — is there
correlation left in these residuals? The answer was no, and the answer
was correct. The question was incomplete. Quiet stretches followed by
violent ones is structure, it is predictable, and it is invisible to
any test of correlation, because the *sign* of the next residual really
is unpredictable. It is the *size* that is not.

### Beat 2 — Look at the squares (about 2 pages)

The obvious move, and it works immediately.

**Chart 2 — the ACF of the residuals beside the ACF of the squared
residuals. Two panels.**

*Reading, and this is the chapter's pivot:* the first panel is clean.
The second is not — the squared residuals are strongly autocorrelated,
often out to many lags. A large residual is followed by another large
one, regardless of sign. Squaring discards the sign and leaves the
magnitude, and the magnitude is exactly what has memory.

This single pair of panels contains the entire idea of conditional
heteroskedasticity, and it arrives before any formula.

### Beat 3 — Testing it properly (about 2 pages)

**Chart 3 — ARCH-LM on two series: one with genuine ARCH effects, one
white noise.**

*Reading:* use the verified numbers from section 1 — `25.345` against
`p = 4.29e-05` on the ARCH series, `2.977` against `p = 0.562` on white
noise. Both directions, which is what makes the test trustworthy rather
than merely enthusiastic.

Explain the construction, because it is unusually transparent: regress
the squared residuals on their own lags and ask whether that regression
explains anything. If squared residuals predict squared residuals,
volatility has memory. The test is little more than chart 2 turned into
a number.

**Chart 4 — ARCH-LM statistic against the number of lags tested.**

*Reading:* the choice of lag count matters and there is no canonical
answer. Too few and slow-moving volatility is missed; too many and
power dissipates. Four is conventional for monthly data, and
conventional is not the same as correct.

### Beat 4 — Variance that shifts rather than clusters (about 2 pages)

A different failure with a different remedy.

**Chart 5 — a series whose variance is genuinely higher in its second
half.**

*Reading:* this is not clustering. There is no alternation between calm
and turbulent; there is one level of variability, then another. ARCH-LM
may or may not fire on it, because the structure is not a
lag-by-lag dependence.

**Chart 6 — the variance-ratio test on this series and on a
homoskedastic one.**

*Reading:* the verified numbers from section 1. And the point worth
making — the statistic is a ratio of variances, so `1.036` means the
last third is 3.6% more variable than the first, and `13.059` means it
is thirteen times more variable. The number tells you the size of the
problem, not merely that there is one.

**Chart 7 — rolling standard deviation of both series.**

*Reading:* the visual counterpart. Clustering shows as a rolling
standard deviation that oscillates; a variance shift shows as one that
steps. Two different pictures, two different tests, two different
remedies — a GARCH model for the first, an intervention or a
transformation for the second.

### Beat 5 — Is it normal, and does it matter? (about 2 pages)

**Chart 8 — a Q-Q plot of the residuals.**

*Reading:* the tails depart from the line, usually more than expected.
Financial residuals almost always do. A Q-Q plot is more informative
than any normality test because it shows *where* the departure is —
tails, centre, or one side — and that determines whether it matters.

**Chart 9 — Jarque-Bera on residuals from three series.**

*Reading:* the test combines skewness and excess kurtosis into one
statistic. Its practical weakness is that on a long series it rejects
almost always, because real data is never exactly normal and a large
`n` will find that out. A rejection on ten thousand observations tells
you very little.

Say plainly what non-normality actually costs. Point forecasts are
usually fine. **Prediction intervals are not** — they are computed from
normal quantiles, and if the tails are heavier than normal the
intervals are too narrow and the coverage is worse than advertised.
That is the honest answer to "does it matter", and it is more useful
than a rule about p-values.

### Beat 6 — Where this leaves you (half a page)

Three questions can now be asked of a set of residuals: is there
correlation left, is the variance stable, and is the distribution
roughly normal.

Asking them one at a time is tedious and easy to skip. Chapter 12
assembles them into a single picture that answers all three at once,
and that picture is what you will actually use.

Further out: chart 2 showed volatility with memory, and nothing in this
book yet models it. That is Part V, and ARCH-LM is the test that sends
you there.

No recap.

---

## 3. The `india` box

Placed in beat 4, after chart 7.

Indian equity and currency series show variance shifts at identifiable
policy moments, not merely clustering. The 1991 liberalisation, the
2016 demonetisation announcement, and the introduction of currency
futures each mark points where the variability of the relevant series
changed level and stayed changed.

That distinction matters for the choice of remedy. A GARCH model treats
volatility as something that wanders and reverts. A one-off regime
change is better handled as an intervention — a known date, a level
shift in variance — and fitting GARCH to it will produce a model that
persistently over-predicts volatility in the calm period and
under-predicts in the turbulent one.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Tsay** | The whole framing — the squared-residual ACF as the diagnostic that opens volatility modelling. `FinTS`'s `ArchTest` is the reference for this chapter | GARCH itself |
| **Durbin & Koopman** | The variance-ratio test, which is theirs and appears in almost no other introductory treatment | State space context |
| **fpp3 §5.4** | The insistence that a histogram and a Q-Q plot are part of residual checking, not optional extras | — |
| **Shumway & Stoffer** | Q-Q plots as standard practice | — |
| **Montgomery, Jennings & Kulahci** | Variance shifts as a distinct, industrially common phenomenon needing a different fix from clustering | Control charts, though the connection is close |
| **Hamilton ch. 21** | Why ARCH-LM takes the form it does | The derivation |

**On examples:** `nyse`, `sp500w`, `gafa_stock`, `djia` all bundled and
suitable for the volatility material.

---

## 5. Voice

**Chart 2 is the chapter and should be treated that way.** Two panels,
one clean and one not, and the entire concept of conditional
heteroskedasticity sitting there before a formula appears. Do not rush
past it.

**Answer "does non-normality matter" honestly.** The useful answer is
about prediction intervals, not about p-values, and most treatments
never say so.

Avoid, beyond earlier lists:

- Introducing GARCH. It is four chapters away and the temptation here
  is strong.
- Treating normality as a requirement. It is not; it is an assumption
  behind specific outputs, and naming which ones is the point.

---

## 6. Checklist

- [ ] Nine charts through `@example ch11`
- [ ] Opens on chapter 10's residuals, same data, deliberate continuity
- [ ] Chart 2 (residual versus squared-residual ACF) present and
      given room
- [ ] ARCH-LM shown in **both** directions with the verified numbers
- [ ] Variance-ratio test shown in both directions, and its statistic
      interpreted as a ratio
- [ ] Clustering versus variance shift distinguished, with different
      remedies named
- [ ] Non-normality's cost stated as prediction-interval coverage
- [ ] Jarque-Bera's large-`n` weakness stated
- [ ] `india` box on policy-driven variance shifts
- [ ] Ends by opening chapters 12 and Part V, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 7. What to do

1. Re-run both tests during drafting. The section 1 numbers are real
   but the chapter's should come from its own run, and both directions
   must be shown.
2. Choose beat 1's residual series so that chart 2's contrast is
   genuinely striking. A GARCH-like simulated series or `nyse` returns
   will both work; try both and use the clearer one.
3. Check whether this package's `ARCHLMTest` and `DKHeteroTest` are
   reachable as functions and what those functions are called — the
   types are exported but I did not confirm the constructors.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
