# Handoff: Chapter 33 — Diffuse Initialisation

`docs/src/introduction/33-diffuse-initialisation.md`. Target 11–12 pages.

The last chapter of Part VI, and the one with the strongest verified
disagreement box in the whole book — a case where the obvious intuition
is not merely imprecise but points the wrong way.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

Test system: a local level (unit root) plus a regression coefficient
with no prior information at all — both genuinely diffuse, and the two
canonical reasons this chapter exists, combined in one model. `n = 50`.

### The naive comparison points the wrong way

Comparing total log-likelihoods across all observations:

```
EXACT diffuse:            -13.845557
approximate, κ = 1e4:     -23.057337     (diff  -9.212)
approximate, κ = 1e6:     -27.661082     (diff -13.816)
approximate, κ = 1e8:     -32.266237     (diff -18.421)
approximate, κ = 1e10:    -36.871403     (diff -23.026)
approximate, κ = 1e12:    -41.477153     (diff -27.632)
```

Read at face value, this says the approximation gets **worse** as κ
grows — the exact opposite of the standard intuition that a large κ
approximates "no information" well.

**The comparison is wrong, not the method.** During the diffuse phase
each observation's likelihood contribution carries a `−0.5·log(F_t)`
term, and `F_t` is inflated by the huge initial variance. As
`κ → ∞`, those specific terms diverge. Confirmed directly: the first
two per-observation values grow steadily more negative with κ, while
every observation after the diffuse phase is essentially unchanged.

### The correct comparison, excluding the diffuse-affected observations

With `d = 2` diffuse states, dropping the first two contributions:

```
EXACT total (excl. first 2):  -11.239210

κ = 1e4    diff =  0.000026
κ = 1e6    diff =  0.00000026     <- R's actual default
κ = 1e8    diff =  0.00000010
κ = 1e10   diff =  0.00000506     <- getting worse
κ = 1e12   diff = -0.00057471     <- markedly worse
```

**There is a sweet spot.** Agreement improves to about `1e8` and then
degrades — floating-point cancellation from an excessively large
initial variance. Bigger is not safer, and the failure is on both
sides.

### The diffuse phase is bounded

```
n =   50:  nobs_diffuse = 2
n =  500:  nobs_diffuse = 2
n = 5000:  nobs_diffuse = 2
```

Constant across a hundredfold increase in sample size, equal to the
number of diffuse states. `O(d)`, not `O(n)`.

---

## 2. The chapter, beat by beat

### Beat 1 — Where does the filter start? (about 1.5 pages)

**Chart 1 — a stationary AR(1) filtered from three different starting
guesses, all three paths converging.**

*Reading:* for a stationary model the starting point barely matters.
The process has its own stationary distribution, the filter is
initialised from it, and even a poor guess washes out within a few
observations. Every example in chapters 29 to 32 quietly relied on
this.

**Chart 2 — a local level filtered from three different starting
guesses.**

*Reading:* they do not converge in the same way, and more importantly
there is no principled starting distribution to use. A random walk has
no stationary distribution — its variance grows without bound — so
there is nothing to draw an initial state from. The question "what do
you believe before seeing any data" has no sensible answer, because the
honest answer is nothing at all.

That is the problem, and it is not exotic. Every local level, every
local linear trend, every regression coefficient estimated inside the
state, and — as chapter 41 will show — every unobserved components
model has it.

### Beat 2 — Just make it enormous (about 2 pages)

The obvious attempt, and it is what R actually does.

**Chart 3 — the same local level initialised with variance 1e2, 1e6 and
1e12, the three filtered paths overlaid.**

*Reading:* a very large initial variance means "I know nothing", and it
works — the first observation essentially determines the state, because
the prior is so vague it contributes nothing. R's `stats::arima` uses
`kappa = 1e6` as its documented default and has done for decades.

It is a pragmatic and effective approximation. The question is how
good, and the honest answer requires care.

### Beat 3 — The comparison that misleads (about 3 pages)

The chapter's centrepiece and its `disagreement` box.

**Chart 4 — total log-likelihood against κ, with the exact value marked
as a horizontal line.**

*Reading:* use the first verified table. The curve moves steadily away
from the exact value as κ rises. Anyone reading this chart would
conclude that the approximation degrades and that a modest κ is safest.

That conclusion is wrong, and finding out why is worth the space.

**Chart 5 — the per-observation likelihood contributions for two κ
values, with the diffuse-phase observations highlighted.**

*Reading:* only the first two contributions move. Everything from the
third observation onwards is essentially identical across κ. The
divergence in chart 4 is entirely concentrated in the diffuse phase,
where the `−0.5·log(F_t)` term is being inflated by construction.

Those observations are not comparable between the two methods. The
exact method handles them by a different route and the approximate one
by an enormous variance, and the resulting numbers are on different
scales.

**Chart 6 — the corrected comparison: log-likelihood excluding the
first `d` observations, against κ.**

*Reading:* use the second verified table. The picture inverts. At
R's default of `1e6` the agreement is to `2.6e-7` — excellent. The best
agreement is near `1e8`. And past `1e10` it degrades again, reaching
`-5.7e-4` at `1e12`.

**Bigger is not safer.** There is a sweet spot, the failure modes are on
both sides, and neither is announced.

Then state what the box is actually about. This is not two packages
disagreeing. It is a case where **the obvious way to check whether an
approximation is good gives the wrong answer**, and only a correct
comparison reveals that the approximation is in fact very good in the
right range. The methodological lesson generalises well past this
chapter.

### Beat 4 — Doing it exactly (about 2 pages)

**Chart 7 — the exact diffuse recursion's uncertainty over the first
few observations, beside the approximate one.**

*Reading:* the exact method treats the unknown states as genuinely
unknown rather than as very-uncertain, and resolves them as
observations arrive. After `d` observations the diffuse part is fully
determined and the filter becomes ordinary.

**Chart 8 — `nobs_diffuse` against sample size.**

*Reading:* use the third verified table. Flat at 2 across `n = 50` to
`n = 5000`. The diffuse phase costs `O(d)`, not `O(n)`, so exact
initialisation is cheap regardless of series length — the special
handling applies to a handful of observations and then stops.

That matters for the implementation: a short prefix loop followed by
the ordinary recursion, rather than a branch inside every iteration.

Note the practical payoff — the exact method needs no tuning parameter
at all. No κ to choose, no sweet spot to find, no failure modes on
either side.

### Beat 5 — What it is for (about 1.5 pages)

**Chart 9 — a regression coefficient estimated inside the state, with
and without diffuse initialisation.**

*Reading:* the coefficient is a fixed unknown with no prior — the exact
situation this chapter solves. Chapter 35 will meet it again, and
chapter 41's unobserved components models are built entirely from
diffuse components.

Note the finding recorded elsewhere in this project, if it can be
verified: ETS turns out **not** to need diffuse initialisation, because
its initial states are estimated as ordinary parameters, whereas
unobserved components models genuinely do. Same-looking models,
opposite answers to the same question — and worth a sentence because it
shows the question is not automatic.

### Beat 6 — Where this leaves you (half a page)

Part VI is finished. The state space form, the filter, the smoother,
time-varying matrices, and a principled start.

All of it has treated the series as arriving alone. Nothing has brought
in outside information — a regressor, a calendar, a known event.

Part VII does.

No recap. One sentence marking the end of a Part.

---

## 3. The `india` box

Placed in beat 5, after chart 9.

Indian quarterly series in their current base are short — a few dozen
observations. The diffuse phase costs `d` of them, and for a local
linear trend with a seasonal component `d` can reach seven or eight.

On a fifty-point series that is a meaningful fraction, and it is the
same compounding problem chapters 9, 19 and 21 each identified from a
different direction: short series make unit-root tests uninformative,
make the AICc convention matter, make order selection unreliable, and
now cost a visible share of the sample to initialisation.

None of these is fixable by better software.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Durbin & Koopman ch. 5** | The definitive treatment of exact diffuse initialisation. This chapter is theirs more than anyone's | The full derivation |
| **Ansley & Kohn (1985), Koopman (1997)** | Cite properly; the exact method is theirs | — |
| **Hamilton ch. 13** | The approximate approach and why it was the standard for so long | — |
| **de Jong (1991)** | Cited widely for diffuse regression — and worth noting that this project verified R and Python do **not** actually use state augmentation for regression coefficients, which is a genuine gap between the standard citation and the standard implementation | — |
| **Shumway & Stoffer** | Initialisation treated practically rather than theoretically | — |
| **fpp3** | Nothing | — |

**On examples:** the local-level-plus-regression system from section 1,
simulated with a stated seed so the tables reproduce.

---

## 5. Voice

**Beat 3 is the best disagreement box in the book and should be told as
an investigation.** A chart that says one thing, a correction that
inverts it, and a sweet spot with failure modes on both sides. Told
flatly it is three tables; told properly it is the chapter.

**Do not present exact initialisation as obviously better.** The
approximation at `1e6` agrees to `2.6e-7`, which is excellent for most
purposes. The exact method's real advantage is that it has no knob to
get wrong — that is a genuine argument and a more modest one than
"more accurate".

Avoid, beyond earlier lists:

- Deriving the exact recursion.
- Calling the approximate method "wrong". It is very good in the right
  range and the chapter's own numbers say so.
- Any suggestion that the reader should have anticipated the inversion
  in beat 3.

---

## 6. Checklist

- [ ] Nine charts through `@example ch33`
- [ ] Stationary case shown converging from any start, before the
      problem is introduced
- [ ] R's `kappa = 1e6` default named
- [ ] **The misleading comparison shown first**, with the verified
      totals
- [ ] Per-observation breakdown showing only the diffuse phase moves
- [ ] **The corrected comparison shown inverting the conclusion**, with
      the sweet spot and both failure modes
- [ ] `nobs_diffuse` flat across three sample sizes
- [ ] The `O(d)` implementation consequence stated
- [ ] ETS-versus-UCM contrast noted if verifiable
- [ ] `india` box compounding the short-series thread from chapters 9,
      19 and 21
- [ ] Ends by opening Part VII, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 7. What to do

1. Re-run all three verified tables during drafting with a stated seed,
   so every number on the page comes from that run.
2. Confirm whether this package exposes a κ-style approximate option at
   all. Earlier design work recommended **not** exposing one, precisely
   because of the sweet-spot problem — if that recommendation was
   followed, say so and why.
3. Verify the ETS-versus-UCM contrast before including it.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
