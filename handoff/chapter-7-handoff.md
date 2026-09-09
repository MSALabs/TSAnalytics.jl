# Handoff: Chapter 7 — Transformations

`docs/src/introduction/07-transformations.md`. Target 11–12 pages.

The last chapter of Part I, and the one that closes a loop opened in
chapter 1. `jj` was the first real series the reader met, and the thing
that made it interesting — seasonal swings growing with the level — has
been left unresolved for six chapters. This chapter resolves it.

Same requirements as chapters 1–6. Reading template from chapter 1.

---

## 1. An honest verification gap

This is the one chapter in Part I whose central method **could not be
verified against its reference implementation**. Guerrero's automatic
lambda selection is what fpp3 recommends and what `forecast::BoxCox.lambda`
implements, and CRAN has been unreachable throughout this project.

What is available as a comparison point is scipy's maximum-likelihood
lambda, which is a *different* method answering a *related* question.
Verified this session on a deliberately multiplicative series —
`y = (10 + 0.4t)·exp(0.25·ε)`, n = 120:

```
scipy boxcox_normmax(method='mle')  ->  lambda = 0.422509
variance raw     = 295.60
variance of logs =   0.2940
```

Two things worth noting. The log transform reduces variance by a factor
of about a thousand on a series built to need it. And MLE selects
**0.42, not 0**, even though the series was constructed multiplicatively
and log is the theoretically right answer.

That gap is not a bug. MLE and Guerrero optimise different criteria —
MLE maximises normality of the residuals, Guerrero minimises the
coefficient of variation of subseries variability — and on the same
data they routinely disagree. **Write that honestly**, and state in the
chapter that this package's Guerrero implementation has not been
cross-checked against R's. If CRAN becomes reachable, this is the first
thing worth revisiting.

---

## 2. The package API

Confirmed exported: **`boxcox`**, **`boxcox_inv`**, **`guerrero_lambda`**,
**`boxcox_profile_plot`**.

`boxcox_profile_plot` returns data rather than a plot — `(lambdas,
loglik)` — so the chapter draws it. That is useful here, because the
profile curve is a chart in its own right.

---

## 3. The chapter, beat by beat

### Beat 1 — The problem left open in chapter 1 (about 1 page)

**Chart 1 — `jj`, plotted again.**

Return to it deliberately. The reader has seen this series before and
will recognise it, which is the point.

*Reading:* the seasonal swings in 1960 are a few cents. By 1980 they
are close to a dollar. The *pattern* has not changed — the same
quarters are high and low throughout — but its size scales with the
level. Chapter 1 named this and moved on. Every technique since has
quietly assumed a series whose variability stays put, and this one does
not.

### Beat 2 — Subtracting does not help (about 1.5 pages)

The obvious attempt: if the problem is that the series grows, remove
the growth.

**Chart 2 — `jj` detrended, and its seasonal swings measured
year by year. Two panels: the detrended series, and a plot of
within-year range against year.**

*Reading:* detrending removes the level and leaves the problem
untouched. The second panel makes it unmistakable — the within-year
range climbs steadily whatever you do to the mean. The difficulty is
not additive and cannot be fixed by subtraction. It is multiplicative,
and multiplication is undone by taking logs.

### Beat 3 — The log, and the family it belongs to (about 2 pages)

**Chart 3 — `jj` and log `jj`, two panels.**

*Reading:* the swings become roughly constant width and the growth
becomes roughly linear. One operation fixed both problems at once,
which is not a coincidence — exponential growth with proportional
seasonality is exactly what logs linearise.

**Chart 4 — the Box-Cox family. The same series at
λ = −1, −0.5, 0, 0.5, 1. Five panels.**

*Reading:* λ = 1 is the raw series, λ = 0 is the log, and the others
interpolate and extrapolate around them. The family is continuous, so
the question stops being "should I take logs?" and becomes "how far
along this scale should I go?" — which is a question with a numerical
answer rather than a yes or no.

State the definition here, after the pictures. Note that λ = 0 is
defined as the log by a limiting argument rather than by substitution,
and that this is exactly the boundary case a truthiness check gets
wrong — forward-reference chapter 16, where a falsy-zero bug is the
disagreement box.

### Beat 4 — Choosing λ (about 3 pages)

Two methods, both worth showing, because they disagree.

**Chart 5 — the profile likelihood curve from `boxcox_profile_plot`.**

*Reading:* log-likelihood against λ, with a clear maximum. The
classical approach is to read the peak off this curve by eye, which is
what `MASS::boxcox` in R is designed for — it is primarily a plotting
function, and that says something about how the method was meant to be
used.

The curve is usually flat near its top. That flatness matters: it means
λ = 0.42 and λ = 0 are often nearly equally supported, and the reader
should be told to prefer a round number when the curve does not care.
A log transform that a colleague can interpret beats a 0.4225 power
that nobody can.

**Chart 6 — Guerrero's criterion across λ.**

*Reading:* Guerrero splits the series into subseries by seasonal
period, computes the coefficient of variation of their rescaled
variabilities, and minimises it. The criterion is explicitly about
making seasonal variability *consistent*, which is a different goal
from making residuals normal. The two curves need not peak in the same
place, and on the verified example in section 1 they do not — MLE gives
0.42 on a series where 0 is theoretically right.

Show both curves together if it can be done cleanly.

**The `disagreement` box goes here**, and it is honest about its own
limits: MLE and Guerrero are different criteria and disagree by design;
this package implements Guerrero following fpp3; and that
implementation has *not* been checked against `forecast`'s because CRAN
was unreachable. Saying so is better than implying a verification that
did not happen.

### Beat 5 — Getting back (about 2 pages)

**Chart 7 — a forecast on the log scale, back-transformed two ways:
naively, and with a bias correction. Three panels or one overlay.**

*Reading, and this catches almost everyone:* exponentiating the mean of
the logs does not give the mean of the original series. It gives
something closer to the median, and it is systematically too low. On a
skewed series the gap can be substantial.

Whether that is a problem depends on what you asked for. If you want a
median forecast, the naive back-transformation is correct. If you want a
mean — and most people saying "forecast" mean the mean — a bias
correction is required. State plainly which one this package's
`boxcox_inv` performs, and check it rather than assuming.

**Chart 8 — `varve` raw and transformed.**

*Reading:* a second case, and a cleaner one than `jj` because there is
no seasonality to distract from the variance problem. Glacial sediment
thickness with variability that scales with the level, and Shumway &
Stoffer's own standard example for this exact point.

### Beat 6 — Where this leaves you (half a page)

Part I is finished. The reader can difference a series, filter it,
measure its autocorrelation, find its rhythms and stabilise its
variance.

What they cannot do is decide, on any principled basis, whether a given
series actually needs any of it. Every decision in Part I has been made
by eye.

Part II replaces the eye with tests.

No recap. This is the end of a Part as well as a chapter, and one
sentence acknowledging that is enough.

---

## 4. The `india` box

Placed in beat 3, after chart 4.

Indian macroeconomic series are frequently reported as index numbers
with a base year set to 100 — the IIP, the WPI, the CPI. An index is
already a ratio, so proportional growth is baked into how it is
constructed, and such series very often want a log transform. It is
worth checking rather than assuming, but the prior is different from
that for a series measured in physical units.

There is also a practical wrinkle. A rebased index has its base reset
to 100 partway through, which is a level shift, and taking logs does
not remove it — chapter 3's rebasing spike survives the transformation
intact.

One paragraph.

---

## 5. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **fpp3 §3.1** | The placement — transformations belong with decomposition, early, not as an advanced topic. And Guerrero as the recommended automatic method | The `fable` syntax |
| **Shumway & Stoffer** | `varve` as the clean example, and the honest treatment of back-transformation bias | The Box-Cox theory |
| **Tsay** | That the log return *is* a transformation of exactly this kind, applied so routinely in finance that nobody calls it one | Distributional detail |
| **Montgomery, Jennings & Kulahci** | The variance-stabilisation framing from an industrial quality-control tradition, where this idea is older than time series analysis | Control charts |
| **Hamilton** | Little. He treats transformations as a preliminary rather than a subject, which is itself worth a sentence | — |
| **Cowpertwait & Metcalfe** | Patience with why logs work on multiplicative structure | R specifics |

**On examples:** `jj`, `varve`, `gafa_stock` and `nyse` all bundled.
The return transformation can be shown on `nyse` rather than Tsay's
CRSP series.

---

## 6. Voice

**Do not present λ selection as solved.** Two respected methods
disagree, the likelihood surface is often flat, and the practical
advice — round to something interpretable — is not what an
optimisation-minded reader expects. That honesty is more useful than a
clean procedure.

**The back-transformation bias deserves its space.** It is the single
most common practical error involving transformations and it produces
forecasts that are quietly, consistently too low.

Avoid, beyond earlier lists:

- Deriving the limiting argument for λ = 0. State it, do not prove it.
- "Simply take logs."
- Treating Box-Cox as a normalising transformation. It stabilises
  variance; normality is a hoped-for side effect, and conflating the
  two is what makes the MLE-versus-Guerrero disagreement confusing.

---

## 7. Checklist

- [ ] Eight charts through `@example ch7`
- [ ] Opens by returning to `jj` from chapter 1
- [ ] Detrending shown failing before logs are introduced
- [ ] Box-Cox family shown across five λ values
- [ ] Both profile-likelihood and Guerrero criterion curves shown
- [ ] `disagreement` box states plainly that Guerrero was **not**
      cross-verified against R, and why
- [ ] Back-transformation bias demonstrated, with `boxcox_inv`'s actual
      behaviour checked and stated
- [ ] Flat-likelihood advice — prefer a round λ — included
- [ ] `india` box on index numbers and rebasing
- [ ] Ends by opening Part II, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 8. What to do

1. **Check what `boxcox_inv` actually does about bias correction**
   before writing beat 5. If it does not correct, say so; if it does,
   say which correction. This is the chapter's most practically
   important sentence and it must describe real behaviour.
2. Run `guerrero_lambda` and the profile-likelihood peak on the same
   series and report both numbers, whatever they are. If they agree
   closely, that is a finding too — but do not assume the section 1
   disagreement transfers to a different series.
3. Retry CRAN once. If `forecast` installs, cross-check
   `guerrero_lambda` against `BoxCox.lambda(method="guerrero")` and
   upgrade the chapter's verification status accordingly.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 7.
