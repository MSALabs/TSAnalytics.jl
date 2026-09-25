# Handoff: Chapter 32 — Time-Varying Systems

`docs/src/introduction/32-time-varying-systems.md`. Target 10–11 pages.

Every matrix so far has stayed put. This chapter lets them move, which
sounds like a small generalisation and is not — it is what makes
regression coefficients drift, seasonal patterns evolve, and structural
change representable inside a model rather than around it.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. A [U] claim, re-verified

Recorded claim: *statsmodels stores time-invariant matrices as 2D and
time-varying ones as 3D, with time as the last axis.*

**Confirmed.** Same model, before and after assigning a time-varying
design:

```
time-INVARIANT design shape:  (1, 2)
time-VARYING  design shape:  (1, 2, 40)      -> (k_endog, k_states, nobs)
```

Time is the **last** axis, not the first. That convention is worth
stating explicitly because it is the opposite of what most people
assume — a series of matrices reads naturally as "matrix number `t`",
which would put time first.

The chapter should also note this package's own convention, which
differs by design: a `Vector{Matrix}` rather than a 3-D array, with
length 1 meaning "broadcast this across all `t`". That is the more
idiomatic Julia representation and it makes the invariant case a
special case of the general one rather than a separate code path.

**Confirm both before writing** — the statsmodels half is verified this
session, the package half is recorded from earlier work and has not
been re-checked against current source.

---

## 2. The chapter, beat by beat

### Beat 1 — A relationship that does not hold still (about 1.5 pages)

**Chart 1 — a scatter of `y` against `x` for a series where the
relationship changes partway through, with one fitted line over
everything.**

*Reading:* the single line fits neither half. Its slope is a compromise
between two genuinely different regimes, and the residuals show it —
systematically positive in one stretch, negative in another.

Chapter 34 will fit regressions with ARIMA errors and will assume the
coefficient is one number. Sometimes it is not, and the failure is not
subtle.

**Chart 2 — the same data with a rolling-window regression slope
plotted over time.**

*Reading:* the crude diagnostic. Fit the regression on a moving window
and watch the coefficient move. It works, it is ad hoc, and it has all
the problems chapter 4 identified — the window length is arbitrary,
the estimate lags, and there is no uncertainty attached to the path.

A model would do better, and the state space form already has the
machinery.

### Beat 2 — Put the coefficient in the state (about 2.5 pages)

**Chart 3 — the construction drawn: the regressor entering through a
time-varying `Z`, with the coefficient sitting in the state vector.**

*Reading:* the trick is small and the consequences are large. Make the
coefficient a state rather than a parameter. Its transition is the
identity — it persists — and giving it a process variance lets it
drift.

`Z` now has to change with `t`, because the regressor's value changes
with `t`. That is the whole reason time-varying matrices are needed,
and it arrives as a consequence rather than as an abstraction.

**Chart 4 — the smoothed coefficient path with its uncertainty band,
over the rolling-window estimate from chart 2.**

*Reading:* the model's path is smoother, comes with an interval, and
uses chapter 31's backward pass so it is not lagging. And crucially,
**the amount of drift is estimated rather than assumed** — the process
variance is a parameter, and if the data says the coefficient is
constant, the fitted variance goes to zero and the path is flat.

That is the argument against the rolling window in one sentence: the
window imposes a rate of change; the model estimates one.

### Beat 3 — The special case is the general case (about 2 pages)

**Chart 5 — the same model fitted with the process variance fixed at
zero, beside an ordinary fixed-coefficient regression.**

*Reading:* they coincide. Exactly, to whatever precision the optimiser
gives, and the chapter should report the actual maximum difference.

**This is the correctness test that matters for the whole chapter** —
if a time-varying implementation does not reduce exactly to the
time-invariant one when the variation is switched off, something is
wrong. It is the same reduction-test discipline used elsewhere in this
project.

**Chart 6 — three fitted process variances on three series: one
genuinely constant, one slowly drifting, one with a sharp break.**

*Reading:* the fitted variance is near zero for the first, moderate for
the second, and for the third the model does badly — a single drift
variance cannot represent a sudden jump, so it either over-smooths the
break or over-fits the flat stretches either side.

Say that honestly. Smooth drift and abrupt change are different
phenomena and this model handles one of them.

### Beat 4 — Two ways to store a moving matrix (about 2 pages)

**Chart 7 — no chart needed here, or a small schematic of the two
layouts.**

*Reading:* the `disagreement` box, using section 1.

statsmodels uses a 3-D array with time last — `(k_endog, k_states,
nobs)`. That is efficient for the numerical inner loop and it means the
invariant and varying cases have different array ranks, so the code
must branch on dimensionality.

This package uses a `Vector{Matrix}` with length 1 meaning "broadcast".
That makes the invariant case a one-element vector rather than a
different type, so the same code path serves both, and it is the more
natural Julia representation.

Neither is wrong. The 3-D layout is closer to the mathematics as
usually written; the vector layout is closer to how the code wants to
be organised. State the trade-off and disclose which this package uses,
because anyone porting a model between them needs to know.

Note the practical trap: with time on the last axis, a reader
constructing a design matrix by hand will naturally build it with time
first and get a silent shape error — or worse, a valid-looking array
that means something else.

### Beat 5 — What else moves (about 1.5 pages)

**Chart 8 — a seasonal component with a time-varying transition, so the
seasonal pattern evolves.**

*Reading:* the coefficient case was the easy one. Any of the matrices
can vary. A time-varying transition lets a seasonal pattern change
shape — which is what chapter 15's STL did by smoothing across years,
now done inside a model with uncertainty attached.

Note the honest limit: this project's own work verified the
time-varying machinery carefully for a varying `Z` and **did not
verify varying `T`, `R`, `Q` or `H` individually against a reference**.
That gap was flagged when the feature was built and, so far as this
handoff knows, remains open. Say so, or verify it and remove the
caveat.

### Beat 6 — Where this leaves you (half a page)

Matrices can move, and the amount they move is estimated rather than
assumed.

One thing has been quietly assumed throughout Part VI. Every filter
needed a starting point — an initial state and its uncertainty — and
every example so far has had one available. For a stationary model the
starting distribution is the process's own stationary distribution.

For a local level there is no such thing. A random walk has no
stationary distribution, so there is nothing to start from.

Chapter 33.

No recap.

---

## 3. The `india` box

Placed in beat 3, after chart 6.

The relationship between Indian monetary policy and lending rates has
changed repeatedly — through the base rate regime, then MCLR, then
external benchmarking. A fixed-coefficient regression across the whole
period estimates an average of several regimes and describes none.

A drifting coefficient handles the gradual parts of that transition
reasonably. It handles the regulatory switch dates badly, for exactly
the reason chart 6 shows — the changes were announced and abrupt, not
gradual, and a smooth drift model will smear them across neighbouring
months.

Knowing which kind of change you have is the point of chart 6, and for
Indian policy data the answer is usually "both, at different times".

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Durbin & Koopman ch. 3** | The general time-varying formulation, which is theirs and which most other treatments simplify away | The full generality |
| **Harvey via Durbin & Koopman** | Time-varying parameter models as a named class with real applications | The structural taxonomy |
| **Hamilton ch. 13** | The formal statement, and the connection to random-coefficient regression | The algebra |
| **Shumway & Stoffer ch. 6** | Worked examples with a varying design matrix | — |
| **Cowpertwait & Metcalfe** | Rolling regression as the naive precursor, explained patiently | R specifics |
| **Tsay** | Time-varying beta in asset pricing, which is the best-known application of this idea outside statistics | Finance theory |

**On examples:** a simulated regression with a known drifting
coefficient for beats 1–3, so the truth is visible; `global_economy`
for a real two-variable case if one works cleanly.

---

## 5. Voice

**Beat 3's reduction test is the chapter's spine.** A generalisation
that does not collapse back to the special case is not a
generalisation, it is a different model. Report the actual difference,
not an assurance.

**Be honest about the unverified matrices in beat 5.** Only the varying
design case was checked against a reference when this was built. Saying
so costs a sentence and buys credibility that the rest of the book has
earned.

**Do not oversell drifting coefficients.** They are genuinely useful
and they cannot represent a break. Chart 6 exists to say so.

Avoid, beyond earlier lists:

- Calling this "the general case" without noting what it still cannot
  do.
- Presenting the storage question as a Julia-versus-Python argument. It
  is a genuine design trade-off with reasons on both sides.

---

## 6. Checklist

- [ ] Eight charts through `@example ch32`
- [ ] Rolling regression shown as the naive precursor, with its problems
- [ ] Coefficient-as-state construction drawn before the algebra
- [ ] Drift variance shown being **estimated**, including going to zero
- [ ] **Exact reduction to the fixed case demonstrated**, with the real
      maximum difference reported
- [ ] Smooth drift versus abrupt break distinguished, with the model
      failing honestly on the latter
- [ ] `disagreement` box on 3-D-time-last versus `Vector{Matrix}`, with
      the verified shapes
- [ ] The construction trap noted — time last, not first
- [ ] **The unverified `T`/`R`/`Q`/`H` gap stated**, or closed
- [ ] `india` box on policy regime changes
- [ ] Ends by opening chapter 33, no recap
- [ ] 10–11 pages, British-Indian spelling

---

## 7. What to do

1. Confirm this package's storage convention against current source
   before writing beat 4. The `Vector{Matrix}` claim is from earlier
   work.
2. **Run the reduction test.** Fit with the drift variance fixed at
   zero and compare against a fixed-coefficient fit. Report the number.
3. Decide whether to verify the varying `T`/`R`/`Q`/`H` cases now or
   disclose the gap. Either is acceptable; silence is not.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
