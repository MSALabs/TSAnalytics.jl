# Handoff: Chapter 35 — Coefficients That Drift

`docs/src/introduction/35-coefficients-that-drift.md`. Target 11–12 pages.

Chapter 34 estimated one coefficient for the whole sample. Chapter 32
built the machinery to let it move. This chapter joins them — and
discovers that the two options are not two ways of computing the same
thing, which is a distinction with real consequences.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

### The two options are different **models**, not different computations

Running Python's `time_varying_regression=True, mle_regression=False,
use_exact_diffuse=True` on the same data as chapter 34's fixed-
coefficient fit:

```
param_names:  ['ar.L1', 'var.x1', 'sigma2']
params:        0.475909212,  0.000109371580,  0.979298992
llf:          -213.74793442697535
```

**`beta` is not in the parameter list.** Only its *process variance*
(`var.x1`) is estimated. The coefficient itself is a latent state,
recovered from the filtered path, and it genuinely moves:

```
filtered beta, first 5:  2.000  1.778  1.458  1.925  2.133
filtered beta, last 5:   1.871  1.875  1.864  1.869  1.875
```

Against chapter 34's fixed fit — one number, `1.8948`, and
`loglik = -211.1257`.

### The reduction case converges in estimates but not in likelihood

Constraining the process variance to exactly zero, so the coefficient
cannot drift:

```
ar.L1 = 0.47465807   var.x1 = 0.0   sigma2 = 0.98227306
llf   = -213.76718527012807
filtered beta converges to ~1.895 by the series end
```

**Point estimates converge** — final filtered beta `≈1.895` against the
fixed fit's `1.8948`; AR coefficient `0.4747` against `0.4768`.

**Likelihoods do not** — `-213.767` against `-211.126`, a gap of 2.64
that does not close.

The reason is chapter 33's: the drifting model carries a diffuse phase
that the fixed model never has, so the two likelihoods are computed
over different effective samples and are not on a common scale.

**Comparing them is a category error**, and the chapter should say so
in exactly those terms.

---

## 2. The chapter, beat by beat

### Beat 1 — One number for thirty years (about 1.5 pages)

**Chart 1 — a regression fitted over a long sample, with its residuals
split into the first half and the second half.**

*Reading:* the residuals are systematically positive in one stretch and
negative in another. The single fitted coefficient is a compromise
between two regimes and describes neither well.

Chapter 32 showed this pattern with a rolling window. Chapter 34 fitted
exactly this kind of model and assumed the coefficient was fixed. The
two chapters have been on a collision course and this is the collision.

### Beat 2 — Let it move (about 2.5 pages)

**Chart 2 — the construction: the coefficient placed in the state
vector, the regressor entering through a time-varying `Z`.**

*Reading:* chapter 32 built this. Here it acquires a purpose. The
coefficient persists — its transition is the identity — and a process
variance governs how much it is allowed to wander from one period to
the next.

**Chart 3 — the filtered and smoothed coefficient paths, with
uncertainty bands, over the fixed-coefficient estimate as a horizontal
line.**

*Reading:* use the verified path. The coefficient starts near 2.0,
drops to 1.458, recovers, and settles near 1.875. The fixed estimate
of 1.8948 is a horizontal line through the middle of that, and it is
not wrong so much as incomplete — it is roughly where the coefficient
ended up, and it says nothing about how it got there.

The band is wide early and narrows. That is chapter 33's diffuse phase
visible in a picture: at the start the model knows nothing about the
coefficient, and the data has to tell it.

### Beat 3 — How much drift is there really? (about 2 pages)

**Chart 4 — three fits with the process variance fixed at three
values, from zero to large.**

*Reading:* at zero the path is flat. Large, and it chases every
observation. In between it drifts.

**Chart 5 — the estimated process variance and its likelihood profile.**

*Reading:* the point is that the drift rate is **estimated, not
assumed**. `var.x1 = 0.000109` on the verified fit — small, but not
zero. The rolling window of chapter 32 imposed a rate through the
window length; this model asks the data.

Note honestly that the profile is often flat near zero, so distinguishing
"a little drift" from "no drift" is hard on short samples. That is the
same identification problem chapters 9 and 25 met from other
directions.

### Beat 4 — Two models, not two methods (about 3 pages)

The `disagreement` box, and it is the chapter's centre.

**Chart 6 — the two fits side by side: fixed coefficient and drifting
coefficient, with their likelihoods printed.**

*Reading:* use the verified numbers. `-211.126` against `-213.748`. A
reader who has met information criteria in chapter 21 will immediately
want to compare them, and that instinct is wrong here.

**Chart 7 — the reduction case: process variance forced to zero, its
coefficient path, and its likelihood.**

*Reading:* the path flattens and converges to `≈1.895`, which matches
the fixed model's `1.8948`. The AR coefficients match too. **The point
estimates agree.**

The likelihoods do not — `-213.767` against `-211.126`. Forcing the
drift to zero did not make the two models the same object.

Then the explanation. The drifting model treats the coefficient as a
diffuse state with no prior, so it carries a diffuse phase. The fixed
model estimates the coefficient as an ordinary parameter and has no
diffuse phase at all. Their likelihoods are computed over different
effective samples.

**So they cannot be compared by likelihood, AIC, or any criterion built
from likelihood.** That rules out the obvious way of choosing between
them, and the chapter should say what to do instead: compare them
out-of-sample, using chapter 23's cross-validation, which does not care
how the likelihood was normalised.

Note that this package names the choice `model = :mle` and
`model = :tvss` rather than a boolean flag, precisely because they are
different models. A boolean would suggest two settings of one thing.

### Beat 5 — When drift is the wrong story (about 1.5 pages)

**Chart 8 — a coefficient that changes abruptly, fitted with a drift
model.**

*Reading:* the drift model smears the break across neighbouring
periods, because a random walk cannot jump. It over-smooths the change
and over-fits the flat stretches either side.

Chapter 32 made this point about the machinery; here it has
consequences for a real modelling decision. Gradual change and abrupt
change need different treatments, and the choice is not a tuning
parameter.

The proper treatment of a known break is an intervention term — a dummy
regressor at a known date — which is ordinary chapter 34 material and
needs none of this chapter's machinery.

### Beat 6 — Where this leaves you (half a page)

A coefficient can be fixed or drifting, the drift rate is estimated,
and the two options cannot be compared by likelihood.

Every regressor so far has been another measured series. Some of the
most useful ones are not measured at all — they are constructed from a
calendar.

Chapter 36.

No recap.

---

## 3. The `india` box

Placed in beat 5, after chart 8.

The pass-through from the RBI's policy rate to bank lending rates has
changed repeatedly — through the base rate regime, MCLR, and external
benchmarking. Some of that change was gradual as banks repriced their
books; some was abrupt, on announced switch dates.

A drifting-coefficient model handles the first well and the second
badly. The honest approach for Indian policy data is usually a
combination — a drift term for the gradual repricing and explicit
intervention dummies at the regulatory switch dates.

That combination is straightforward to specify and almost nobody does
it, because the two mechanisms live in different chapters of most
textbooks.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Durbin & Koopman** | The time-varying parameter formulation and the diffuse treatment of the coefficient | The derivations |
| **Harvey via Durbin & Koopman** | Time-varying parameter models as a named class | The structural taxonomy |
| **Tsay** | Time-varying beta in asset pricing — the best-known application of this idea anywhere | Finance theory |
| **Hamilton ch. 13** | The random-coefficient regression connection | The algebra |
| **fpp3** | Little; fpp3 does not cover drifting coefficients, which is worth noticing | — |
| **Montgomery, Jennings & Kulahci** | Adaptive parameter estimation in process control, the same idea from another tradition | Control charts |

**On examples:** the simulated regression from section 1, with a stated
seed. `global_economy` or a policy-rate pair for a real case.

---

## 5. Voice

**The category-error point must be stated plainly and more than once.**
A reader who has spent chapter 21 learning to compare models by AIC
will reach for it here, and the whole box exists to stop them. Say it
in beat 4, and say it again in the summary sentence at the end.

**Do not present drifting coefficients as more sophisticated.** They
answer a different question and they fail on abrupt change. The fixed
model is right whenever the coefficient is fixed, which is often.

Avoid, beyond earlier lists:

- Calling `:tvss` "the general case". It is a different model, not a
  superset.
- Suggesting the drift variance can be reliably estimated on short
  samples. Beat 3 says otherwise.

---

## 6. Checklist

- [ ] Eight charts through `@example ch35`
- [ ] Fixed-coefficient residuals shown failing across a split sample
- [ ] Filtered **and** smoothed coefficient paths shown with bands
- [ ] The verified drifting path used — 2.000 → 1.458 → 1.875
- [ ] Drift variance shown being estimated, with the flat-profile
      caveat
- [ ] **Reduction case shown**: estimates converge, likelihoods do not
- [ ] The verified likelihood gap (−211.126 versus −213.767) reported
- [ ] **Category error stated explicitly**, with cross-validation named
      as the alternative
- [ ] The `model = :mle` / `:tvss` naming rationale explained
- [ ] Abrupt change shown defeating the drift model
- [ ] `india` box on policy pass-through, gradual and abrupt together
- [ ] Ends by opening chapter 36, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 7. What to do

1. Re-run both fits and the reduction case during drafting, with a
   stated seed, so all three likelihoods on the page come from that run.
2. Confirm this package's `Q_beta` or equivalent argument for fixing
   the drift variance at zero — beat 4's reduction case needs it.
3. Verify that the filtered coefficient path is actually exposed on the
   result object; chart 3 depends on it.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
