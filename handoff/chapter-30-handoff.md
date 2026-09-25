# Handoff: Chapter 30 — The Kalman Filter

`docs/src/introduction/30-the-kalman-filter.md`. Target 12–13 pages.

The chapter the whole of Part VI exists for. Chapter 29 built the
notation. This chapter gives it an engine — and then tells the reader
they have been running that engine since chapter 18 without knowing it.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. The reveal, and the evidence for it

**Verified this session.** The module's own include order:

```
line 15:  include("statespace/gaussianssm.jl")
line 16:  include("statespace/timevaryingssm.jl")
line 17:  include("statespace/diffuseinit.jl")
line 20:  include("differencing.jl")
line 21:  include("filters.jl")
...
          arma.jl, arima.jl, sarima.jl load later still
```

The state-space engine loads **before differencing, before filters,
before anything in Parts I to V**. Not by accident and not by
alphabetical order — because ARMA estimation is built on top of it, and
Julia requires the dependency to exist first.

That is the chapter's payload and it is checkable by any reader who
opens the source. **The book's structure mirrors the software's
architecture**, and this is where the reader finds that out.

The likelihood in chapter 18 — described there functionally, as
something computed by running the model forward through the data and
collecting one-step-ahead forecast errors — was this. Chapter 18 was
deliberately written to describe the mechanism without naming it.

---

## 2. The blocking decision, restated

Chapter 29's section 1 applies here with more force: **nothing is
exported from `src/statespace/`**. If the export decision has not been
made, make it before drafting. This chapter either teaches an API or
explains an architecture, and the two read completely differently.

---

## 3. The chapter, beat by beat

### Beat 1 — Guessing where you are (about 1.5 pages)

**Chart 1 — a hidden state and its noisy observations, both drawn.**

Simulate a local level so the true state is known and can be plotted.

*Reading:* the observations scatter around the truth. The problem is to
recover the line from the dots, and the difficulty is that the line
itself is moving — so neither averaging everything nor trusting the
latest point is right.

Averaging everything ignores that the level has moved. Trusting the
latest observation ignores that it is noisy. The answer has to be
somewhere between, and *where* between should depend on how noisy the
observations are relative to how fast the level moves.

That is the whole idea, stated before any machinery.

### Beat 2 — Predict, then correct (about 2.5 pages)

**Chart 2 — one step, drawn in three panels: the prediction from
yesterday, the new observation, and the corrected estimate sitting
between them.**

*Reading:* the prediction comes from the transition equation — where
the state should have moved to. The observation arrives. The corrected
estimate is a weighted average of the two, and the weight is the whole
content of the filter.

**Chart 3 — the same step with three different noise ratios.**

*Reading:* when observations are precise the correction lands near the
observation. When they are noisy it stays near the prediction. The
weight — the Kalman gain — is computed from the two uncertainties, so
the filter is not choosing a compromise arbitrarily; it is choosing the
one that minimises the variance of the result.

Report the actual gain values for the three cases. The number becomes
intuitive once seen alongside the picture.

**Chart 4 — the full recursion running across a series, with the
filtered estimate and its uncertainty band.**

*Reading:* the band contracts at the start as the filter learns, then
settles. For a time-invariant model the gain converges to a constant
and the filter becomes a fixed exponentially-weighted rule — which is
worth noting, because it connects to chapter 4's recursive filters and
to Holt-Winters.

### Beat 3 — Where the likelihood comes from (about 2.5 pages)

**Chart 5 — the one-step-ahead forecast errors from the recursion, with
their variances.**

*Reading:* the filter produces, at every step, a prediction and its
uncertainty. The difference between prediction and observation is a
forecast error, and its distribution is known. Multiply those densities
together across the whole series and you have the likelihood.

This is the prediction error decomposition, and it is the reason the
state space form is not merely a tidy notation. **It turns any model
written in the form into a model you can fit**, using one routine.

**Chart 6 — the likelihood surface for a local level model, over the
two variance parameters.**

*Reading:* a surface with a maximum, computed entirely by running the
recursion at each candidate parameter pair. The optimiser from chapter
18 does the rest. Nothing new is needed.

### Beat 4 — The reveal (about 2.5 pages)

**Chart 7 — an AR(2) fitted twice: once with `fit_arma`, once by
writing it in state space form and running the general filter. The
coefficients and log-likelihood from both, side by side.**

*Reading:* they match. Not approximately — to whatever precision the
optimiser delivers, and the chapter should report the actual maximum
absolute difference.

Then state it plainly. Every ARMA fit in chapter 18, every ARIMA fit in
chapter 19, every seasonal model in chapter 20 and every automatic
selection in chapter 21 ran this recursion. The likelihood described
there was computed by this filter. The reader has been using it for
twelve chapters.

**Chart 8 — the include order, shown as the actual source lines.**

*Reading:* use the verified listing from section 1. `gaussianssm.jl` at
line 15, `arma.jl` far below it, because the dependency runs that way.
The architecture is not an interpretation — it is visible in the file.

This is the moment the book's structure justifies itself, and it should
be allowed to land without ornament. One or two sentences, then move
on. Overselling it would spoil it.

### Beat 5 — What else it gives you (about 1.5 pages)

*Reading, with a small chart if useful:* three things fall out for
free once a model is in this form.

Missing observations — chapter 29 showed this; the filter simply skips
the update step and keeps predicting.

Forecasting — running the transition equation forward with no
observations to correct against *is* the forecast, and the growing
uncertainty band comes out automatically rather than from a separate
formula.

And time-varying parameters, which chapter 32 takes up.

Note the honest cost: the filter is a sequential recursion, so it
cannot be vectorised across time. Every step depends on the one before.
For a long series that is a genuine performance constraint and it is
why this package's earlier attempts at optimising the inner loop were
worth the effort.

**`julia` box** on that: the recursion is a hot loop, allocation inside
it matters, and views rather than copies for the matrix slices make a
measurable difference. Half a page, concrete, drawn from the package's
own optimisation history if the details are to hand.

### Beat 6 — Where this leaves you (half a page)

You can estimate the current state given everything observed up to now,
and fit any model written in the form.

"Up to now" is doing work in that sentence. Once the whole series has
been seen, the estimate of the state at time 10 should improve — there
are forty more observations that say something about it, and the filter
ignored all of them.

Chapter 31.

No recap.

---

## 4. The `india` box

Placed in beat 5, after the missing-data point.

The 2020 lockdown left genuine holes in Indian monthly indicators, and
several state-level series have administrative gaps of a few months.
The filter's treatment of these is not a workaround — the uncertainty
band widens through the gap and narrows when observations resume, which
is an honest representation of what is known.

Compare that with the common alternative of interpolating first. An
interpolated series looks complete, and every subsequent calculation
treats the invented values as data. The filter's version looks
uncertain because it is.

One paragraph.

---

## 5. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Durbin & Koopman ch. 4** | The definitive treatment of the recursion and the prediction error decomposition | The full derivation |
| **Hamilton ch. 13** | The ARMA-in-state-space result that makes beat 4's reveal possible, and why the likelihood follows | The asymptotics |
| **Shumway & Stoffer ch. 6** | The filter presented with worked examples and plots rather than as pure algebra | The EM algorithm |
| **Kalman (1960)** | Cite it. The paper is from control engineering, not statistics, which is worth one sentence — this tool arrived in time series from outside | — |
| **Cowpertwait & Metcalfe** | The predict-correct intuition explained patiently | R specifics |
| **Tsay** | Little directly | — |

**On examples:** simulated local level for charts 1–4 so the true state
is known; a simulated AR(2) for the beat 4 comparison so the answer can
be checked; `gtemp_land` or `GNP23` for a real fit.

---

## 6. Voice

**The reveal must be understated.** It is a genuinely satisfying moment
and the temptation is to make a meal of it. Two sentences and the
source listing do more than a page of build-up. The reader will feel it
without being told to.

**Do not derive the gain.** Chart 3 makes it intuitive. The algebra
belongs in the Manual or a reference, and deriving it here costs three
pages and loses people.

**Beat 1's framing before machinery is essential.** A reader who
understands "somewhere between the prediction and the observation, and
where depends on which you trust" can follow everything else. One who
meets the matrix recursion first will not.

Avoid, beyond earlier lists:

- Control-theory language. The tool came from there; the book is not
  about that.
- Calling it optimal without saying in what sense — minimum variance
  among linear estimators, under the model's assumptions.
- Any suggestion that the reader should have guessed the reveal.

---

## 7. Checklist

- [ ] Eight charts through `@example ch30`
- [ ] Idea stated before machinery, in beat 1
- [ ] Predict-correct shown in three panels
- [ ] Gain shown varying with the noise ratio, with real values
- [ ] Prediction error decomposition connected to the likelihood
- [ ] **AR(2) fitted both ways, difference reported**
- [ ] **Include order shown as actual source lines**
- [ ] Reveal kept to two sentences plus evidence
- [ ] Sequential-recursion cost stated honestly
- [ ] `julia` box on the hot loop, half a page
- [ ] `india` box on lockdown gaps versus interpolation
- [ ] Ends by opening chapter 31, no recap
- [ ] 12–13 pages, British-Indian spelling

---

## 8. What to do

1. **Confirm the export decision from chapter 29's section 1.**
2. **Run the beat 4 comparison.** Fit an AR(2) with `fit_arma` and via
   the general state-space route, and report the actual difference. If
   they do not match, that is a finding and the chapter changes
   completely — so do this early.
3. Re-check the include line numbers against current source before
   quoting them; files move.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 7.
