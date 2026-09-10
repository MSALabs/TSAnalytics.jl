# Handoff: Chapter 27 — Forecasting Volatility

`docs/src/introduction/27-forecasting-volatility.md`. Target 11–12 pages.

Chapters 25 and 26 described variance. This chapter projects it
forward — and discovers that one of the three models cannot be
projected forward analytically at all, which is a structural fact
rather than an implementation gap.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. A [U] claim, re-verified with the exact error

Recorded claim: *"EGARCH has no analytic multi-step forecast;
simulation is structurally required."*

**Confirmed exactly.** Both models fitted to the same simulated series,
`n = 1500`, forecasting five steps:

```
GARCH   analytic    -> [1.80022  1.82048  1.83995  1.85866  1.87664]
GARCH   simulation  -> [1.80022  1.82352  1.83767  1.84550  1.86315]
EGARCH  analytic    -> ValueError: Analytic forecasts not available for horizon > 1
EGARCH  simulation  -> [1.69620  1.72856  1.75350  1.78779  1.82897]
```

Two findings, not one.

**EGARCH raises an error rather than returning an approximation.**
Quote the message verbatim — it is a good example of a library refusing
to produce a number it cannot compute properly, which is now the fourth
instance of that pattern in this book (after PP's `ρ`, KPSS's clipped
p-values and Durbin-Watson's missing p-value). Name it as a pattern.

**GARCH's analytic and simulated forecasts agree closely.** Identical at
`h = 1` (1.80022 both), and within about 0.2% out to `h = 5`. That
agreement is a genuine validation of the simulation engine — if
simulation reproduces the analytic answer where both exist, it can be
trusted where only simulation exists.

---

## 2. The chapter, beat by beat

### Beat 1 — Tomorrow is easy (about 1 page)

**Chart 1 — a fitted GARCH conditional variance, with the one-step
forecast marked at the end.**

*Reading:* the one-step forecast requires no work at all. Today's
squared shock and today's variance are both known, so tomorrow's
variance falls straight out of the recursion. There is no uncertainty
about the forecast itself — only about the shock that will realise
against it.

That is why one-step volatility forecasting is nearly trivial and why
the interesting questions start at step two.

### Beat 2 — Further out (about 2.5 pages)

**Chart 2 — the multi-step variance forecast from GARCH, with the
long-run variance marked as a horizontal line.**

*Reading:* the forecast converges towards the unconditional variance,
`omega/(1 − alpha − beta)`. From below if current volatility is calm,
from above if turbulent. The rate of convergence is governed by
`alpha + beta`, which chapter 25 established is typically around 0.96
— so convergence takes months, not days.

Report the fitted long-run variance and the half-life of a shock. The
half-life is `log(0.5)/log(alpha + beta)` and it is a much more
intuitive number than the coefficients themselves — for `alpha + beta =
0.96` it is about 17 periods.

**Chart 3 — variance forecasts from three different starting points:
after a calm period, after a turbulent one, and from the long-run level.**

*Reading:* all three converge to the same place at the same rate,
starting from different heights. That is mean reversion in variance,
and it is the single most useful thing GARCH tells you — a turbulent
period will calm down, predictably, and you can say roughly how fast.

### Beat 3 — When the formula runs out (about 2.5 pages)

**Chart 4 — the attempt to produce an analytic EGARCH forecast, with
the error.**

*Reading:* show the actual `ValueError` text. Then explain why it is
not a missing feature.

The GARCH recursion is linear in the variance, so taking expectations
step by step works — the expectation of a sum is the sum of the
expectations. EGARCH's recursion is linear in the *logarithm* of the
variance, and the expectation of an exponential is not the exponential
of the expectation. The step that makes GARCH's multi-step forecast
easy simply does not exist for EGARCH.

**That is structural.** No amount of implementation effort produces a
closed-form EGARCH multi-step forecast, and a library that offered one
would be offering an approximation without saying so.

Name the pattern: this is the fourth time in the book that a
well-built implementation has declined to return a number rather than
return a wrong one. It is a good habit and worth recognising.

### Beat 4 — Simulate instead (about 2.5 pages)

**Chart 5 — a fan of simulated volatility paths, with their mean
overlaid.**

*Reading:* draw shocks, run the recursion forward, repeat many times,
average. The mechanism is obvious once stated and it works for any
model whose recursion can be run forward, which is all of them.

**Chart 6 — the simulated mean against the analytic answer for GARCH,
at increasing numbers of paths.**

*Reading, and this is the beat's payload:* use the verified numbers.
At `h = 1` they agree exactly. Across five steps they agree to within a
fraction of a percent. As the number of paths rises the simulated
answer converges on the analytic one.

**This is how you validate a simulation engine** — run it where the
answer is known, confirm it agrees, then trust it where the answer is
not known. Say that explicitly, because it is a transferable technique
and this book has used it before (chapter 15's reduction tests, the
`Q_beta = 0` case in the package's own development).

**Chart 7 — the distribution of simulated variance at `h = 20`, not
just its mean.**

*Reading:* simulation gives the whole distribution, which the analytic
formula does not. The variance forecast has its own uncertainty, and it
is skewed — the upside is much longer than the downside, because
variance is bounded below by zero and not above.

That skew is genuinely useful for risk work and is invisible in a point
forecast.

### Beat 5 — Choosing a method (about 1.5 pages)

**Chart 8 — timing: analytic against simulation at several path
counts.**

*Reading:* analytic is essentially free; simulation costs
proportionally to the number of paths. Report the actual timings. Then
the sensible default — use analytic where it exists and is enough, and
simulation where it does not or where the full distribution is wanted.

State what this package does. A `method = :auto` that picks analytic
for GARCH and GJR and simulation for EGARCH is a reasonable design and
worth naming as such if that is what exists.

**`julia` box.** Simulation across paths is embarrassingly parallel —
each path is independent — and threading it is a natural fit. Note that
the random number generator needs care under threading if results are to
be reproducible. Half a page.

### Beat 6 — Where this leaves you (half a page)

You can forecast a variance forward, by formula where one exists and by
simulation where it does not, and you know how to check that the
simulation is right.

Everything so far has estimated volatility from daily data. If
intraday data is available, volatility can be *measured* rather than
modelled — which is a different approach with different strengths.

Chapter 28.

No recap.

---

## 3. The `india` box

Placed in beat 2, after chart 3.

Mean reversion in variance has a direct use for Indian markets. NSE
volatility spikes around budget announcements, election results and
monetary policy decisions, and these are largely scheduled events with
known dates.

A GARCH forecast made before such an event will not anticipate it —
the model has no calendar and no knowledge that Thursday is a policy
day. It will, however, correctly predict the decay afterwards, and the
half-life from beat 2 tells you roughly how long elevated volatility
should persist.

Anticipating the spike itself needs an exogenous regressor, which is
the same answer chapter 36 gives for festivals.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Tsay ch. 3** | Multi-step variance forecasting and the mean-reversion result; the primary source here | Option-pricing applications |
| **Andersen et al. via Tsay** | Volatility forecast evaluation, which is genuinely hard because the target is unobserved — a good forward reference to chapter 28 | The full literature |
| **Hamilton ch. 21** | Why the linear recursion permits step-by-step expectations, which is exactly what EGARCH lacks | The algebra |
| **Shumway & Stoffer** | Simulation-based forecasting as a general technique | — |
| **fpp3 §5.5** | Bootstrapped intervals as the same idea in a different setting — worth one sentence connecting them | The `fable` syntax |
| **Montgomery, Jennings & Kulahci** | Nothing directly | — |

**On examples:** `nyse`, `sp500w`, `djia` bundled. The forecast
comparison is on simulated data with a stated seed so it reproduces.

---

## 5. Voice

**Beat 3's explanation must be genuine.** "EGARCH cannot do this" is
unsatisfying; "the expectation of an exponential is not the exponential
of the expectation" is a real reason and a reader can check it. Give
the reason.

**Beat 4's validation technique is the transferable lesson.** Run the
new method where the answer is known, then trust it where it is not.
This book has used it repeatedly and naming it here makes the pattern
visible.

Avoid, beyond earlier lists:

- Deriving the multi-step GARCH recursion.
- Presenting simulation as inferior. It is slower and more general, and
  it gives the whole distribution.
- Value-at-Risk. It is the obvious application and it is a rabbit hole.

---

## 6. Checklist

- [ ] Eight charts through `@example ch27`
- [ ] Long-run variance and shock half-life both reported as numbers
- [ ] Convergence shown from three different starting points
- [ ] **The EGARCH error quoted verbatim**
- [ ] The structural reason given, not just the fact
- [ ] The refuse-to-guess pattern named, with all four instances
- [ ] **Simulation validated against analytic where both exist**, with
      the verified agreement
- [ ] Full simulated distribution shown, with its skew noted
- [ ] Timing comparison reported
- [ ] `julia` box on parallel simulation and RNG reproducibility
- [ ] `india` box on scheduled-event volatility
- [ ] Ends by opening chapter 28, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 7. What to do

1. Re-run the analytic-versus-simulation comparison with a stated seed
   so the table reproduces from the text.
2. Confirm whether this package has a `method = :auto` and what it
   selects for each model family.
3. Time both methods and report real numbers rather than describing the
   trade-off.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
