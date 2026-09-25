# Handoff: Chapter 29 — The State Space Form

`docs/src/introduction/29-the-state-space-form.md`. Target 10–11 pages.

Part VI opens, and its placement is the book's most deliberate
structural decision. Durbin & Koopman start here. Hamilton treats it as
a late specialist topic. This book puts it after the reader has already
fitted dozens of ARMA models — because every one of those fits was
running this machinery, and chapter 30 says so.

**Do not spoil that.** This chapter builds the apparatus; chapter 30
reveals what it has been powering.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. A decision that blocks this Part

Verified this session: **nothing is exported from `src/statespace/`.**
`GaussianSSM`, `TimeVaryingSSM` and `kalman_filter` are entirely
internal — none appears among the package's 92 public symbols.

A book chapter teaching an unexported API is awkward at best. Two
honest options:

**Export them.** They are genuinely useful — anyone building a custom
model wants a general Kalman filter — and Part VI then reads as a
tutorial like every other Part.

**Write Part VI as architecture rather than tutorial.** Explain what
the engine does and why it matters, show the internal code, and be
explicit that the reader cannot call it directly today.

**Decide before drafting chapter 29**, because the chapter's voice
depends on it. The first option is better if the maintainers are
willing; the second is honest if they are not. What would be wrong is
writing tutorial prose for functions the reader cannot reach.

This handoff assumes the first. If the second is chosen, the beats
survive but every `julia>` example becomes a code listing.

---

## 2. The chapter, beat by beat

### Beat 1 — Four models, four sets of machinery (about 1.5 pages)

**Chart 1 — four series with four fitted models: an AR(2), a local
level, a regression with time-varying coefficients, and a seasonal
decomposition. Four panels.**

*Reading:* four different problems, and so far in this book four
entirely separate pieces of apparatus. Each has its own estimation
routine, its own forecasting rule, its own diagnostics.

That is how the subject is usually taught and it is not how it has to
be. All four can be written in one notation, and once they are, one
estimation routine and one forecasting rule serve all of them.

The claim should sound implausible when first made. It is the chapter's
job to make it obvious by the end.

### Beat 2 — Two equations (about 2 pages)

Introduce the form. An observation equation connecting what you see to
a hidden state, and a transition equation describing how the state
moves.

**Chart 2 — the structure drawn as a diagram: hidden states evolving
along the bottom, observations dropping out of them, noise entering at
both levels.**

*Reading:* the picture is worth more than the equations here. Two
sources of randomness — one in how the state moves, one in how it is
observed — and a state that is never seen directly. Everything else is
bookkeeping about matrix shapes.

Name the matrices, briefly: `Z` maps state to observation, `T` moves
the state forward, `R` and `Q` govern the state noise, `H` the
observation noise. Do not dwell; they become concrete in beat 3.

### Beat 3 — Writing familiar models down (about 3 pages)

The chapter's substance. Three worked translations.

**Chart 3 — a local level model: the observed series with its hidden
level drawn beneath.**

*Reading:* the simplest non-trivial case. The state is one number, the
level, and it takes a random walk. The observation is the level plus
noise. Two variances to estimate. Show the matrices explicitly — they
are all scalars, which makes this the right first example.

**Chart 4 — an AR(2), written in state space form.**

*Reading:* the state is two numbers, holding the current and previous
values. `T` contains the AR coefficients. `Z` picks off the first
element. There is no observation noise at all — `H` is zero — because
an AR process *is* its own state.

This is the translation that matters most, because it is the one
chapter 30 exploits. Show it carefully.

**Chart 5 — a local linear trend: level and slope as two states.**

*Reading:* level and slope both evolve, the slope feeding into the
level. Setting the slope's variance to zero gives a deterministic
trend; letting it move gives a trend that bends. One model, two
behaviours, controlled by a variance rather than a structural choice.

**Chart 6 — the same series fitted as a local level and as a local
linear trend, with both hidden states shown.**

*Reading:* the extra state buys the ability to extrapolate a direction.
The cost is a variance parameter and a more uncertain forecast.

### Beat 4 — Why bother (about 2 pages)

**Chart 7 — a model that would be awkward any other way: a local level
with a missing stretch in the middle of the series.**

*Reading:* the state space form handles missing observations without
special-casing. The state keeps evolving; there is simply no
observation to update against, so uncertainty grows through the gap and
contracts when data resumes. Show the widening band across the gap.

Chapter 2 discussed missing data as a nuisance. Here it is barely a
complication, and that is a genuine argument for the framework rather
than an aesthetic one.

**Chart 8 — combining components: a local level plus a seasonal plus a
regression term, in one model.**

*Reading:* components stack. Two models written separately become one
model by concatenating their states and block-diagonalising their
transitions. Nothing about the estimation routine changes.

That composability is the framework's real payoff and it is why chapter
41's unobserved components models are possible at all.

### Beat 5 — The cost of generality (about 1 page)

*Reading, no chart or one small one:* the general form is more verbose
than the special cases. Writing an AR(2) as four matrices is more work
than writing it as two coefficients, and for a reader who only ever
fits AR(2) models the abstraction earns nothing.

It earns its keep when models start combining, when data goes missing,
when parameters need to vary — and when one estimation routine has to
serve all of them. Say this honestly rather than presenting the
framework as unambiguously superior.

### Beat 6 — Where this leaves you (half a page)

You can write down a state space model. You cannot yet do anything with
one — no likelihood, no estimation, no forecast.

What is needed is a way to work out, at each point in time, what the
hidden state probably is given everything observed so far.

Chapter 30.

No recap.

---

## 3. The `india` box

Placed in beat 4, after chart 7.

Indian macroeconomic series have genuine gaps. Monthly IIP was
disrupted during the 2020 lockdown; several state-level series have
missing months from administrative changes; older series have periods
where collection methodology changed and the data was withdrawn.

Every method in Parts I to V requires either a complete series or an
ad-hoc patch. Interpolating first and modelling afterwards treats a
guess as data, and the model has no way to know which observations were
real. The state space form handles the gap natively and the resulting
uncertainty band shows honestly where the information is thin.

For Indian data this is not a minor convenience.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Durbin & Koopman ch. 3** | The canonical treatment. The two-equation form, the matrix conventions, and the insistence that one framework serves many models | Their decision to open with it; this book earns it differently |
| **Hamilton ch. 13** | The AR-to-state-space translation done carefully, and why `H = 0` for a pure AR | The full generality |
| **Shumway & Stoffer ch. 6** | Worked component models and the composability argument | The EM algorithm; not needed here |
| **Harvey via Durbin & Koopman** | Structural time series as the motivating application — level, trend, seasonal as separate interpretable states | The full structural taxonomy; chapter 41 |
| **fpp3** | Nothing; fpp3 does not use state space form explicitly, though ETS is one underneath | — |
| **Montgomery, Jennings & Kulahci** | Little | — |

**On examples:** `gtemp_land` or `GNP23` for the local level and local
linear trend; a simulated AR(2) for chart 4 so the translation can be
checked against known coefficients.

---

## 5. Voice

**Do not name the Kalman filter.** The reveal in chapter 30 depends on
it, and this chapter can describe everything it needs without the word.
If a sentence seems to require it, rewrite the sentence.

**Beat 5 must be genuine.** The framework has real costs and a chapter
that presents it as free will lose readers who have just written four
matrices to express two coefficients.

**Matrices need pictures.** Chart 2's diagram does more than the
algebra. Readers who see the structure will tolerate the notation.

Avoid, beyond earlier lists:

- Full matrix algebra for the general case. Show the three worked
  examples; the general form can stay in the Manual.
- The word "elegant". Show it and let the reader decide.
- Any forward reference that gives away chapter 30's reveal.

---

## 6. Checklist

- [ ] Eight charts through `@example ch29`
- [ ] Structure shown as a diagram before any matrices
- [ ] **Three worked translations** — local level, AR(2), local linear
      trend — with matrices shown explicitly
- [ ] AR(2)'s `H = 0` noted and explained
- [ ] Missing data handled natively, with the widening band shown
- [ ] Composability demonstrated by stacking components
- [ ] Beat 5 states the cost honestly
- [ ] **Kalman filter not named anywhere**
- [ ] `india` box on real gaps in Indian series
- [ ] Ends by opening chapter 30, no recap
- [ ] 10–11 pages, British-Indian spelling

---

## 7. What to do

1. **Settle the export question in section 1 before writing a word.**
   It determines whether this is a tutorial or an architecture
   explanation.
2. Verify the AR(2) translation numerically — fit an AR(2) directly and
   through the state space form, and confirm the coefficients match.
   That check belongs in the chapter.
3. Render every chart; honest CI note if the environment cannot.
4. Check against section 6.
