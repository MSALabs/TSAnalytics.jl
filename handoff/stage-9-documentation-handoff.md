# Handoff: Documentation for Stage 9 — API reference, manual, and the book

**Status: implemented (2026-10-14)** — see `development-sequence.md`, "Documentation for Stages 9 and 10", for what was built and where it departs from this document. *Originally:* pending. Unlike the Stage 10 documentation handoff, the code
here is **already built and already partly documented** — so this is a
restructuring and a gap-filling job, not a from-scratch one. Read §1
before planning anything; what is missing is not what you would guess.

For a fresh Claude Code session with no prior context. Companion to
`stage-10-documentation-handoff.md` (same conventions, same build
constraints) and `plot-recipes-handoff.md`, which is a **hard
dependency** for the charts — see §2.4.

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.

---

## 1. The three gaps, measured

Stage 9 — `holt_winters`, `fit_ets`/`auto_ets` (15 of 30 forms),
`fit_theta`, `fit_tbats` — landed on 2026-10-02. Its documentation is in
an unusual state: thorough in one place, structurally misfiled in
another, and entirely absent from a third.

### 1.1 API — the entries exist, on the wrong page

`[S]` There is **no exponential-smoothing API page**. All ten Stage 9
entries sit under an `## Exponential smoothing` heading inside
**`api/arma-models.md`**, a 36-line page titled *"API: ARMA Models"*:

```markdown
## Exponential smoothing

```@docs
holt_winters
ExponentialSmoothingModel
fit_ets
auto_ets
ETSModel
notation
fit_theta
ThetaModel
fit_tbats
TBATSModel
```
```

ETS, Theta and TBATS are not ARMA models. `[V]` The page is a catch-all
that grew: its three sections are "Univariate models" (the ARMA family),
"Exponential smoothing" (Stage 9), and "Model-fitting infrastructure"
(`OptimResult`, `partrans`, `invpartrans`).

**The good news**: because the entries exist, `checkdocs=:exports` is
already satisfied. Nothing is broken — this is a move, not a rescue, and
it can be done in one commit with no risk.

### 1.2 Manual — the longest chapter in the book, with no pictures

`[V]` `manual/10-exponential-smoothing.md` is **687 lines with 30
`@example` blocks and zero `plot(` calls**.

It is the longest manual chapter by a wide margin — `[V]` the others run
104 to 344 lines — and it is the only one covering a stage whose R
counterpart has the richest visual vocabulary in the whole subject. `[V]`
`forecast` 8.21.1 ships `plot.ets`, `autoplot.ets`, `plot.bats`,
`plot.tbats`, `plot.forecast`, `checkresiduals`; `[V]` statsmodels'
`ETSResults` has **no plot method at all**.

So the single clearest documentation gap in the package is: the chapter
that should be the most visual has no figures.

`[S]` Its ten sections, for reference when placing figures:

```
The taxonomy, and which part of it is here      (line 37)
Fitting one                                     (76)
Letting it choose                               (144)
Forecasting                                     (181)
Where the numbers differ from R and statsmodels  (218)
The reduction onto holt_winters                 (270)
Multiplicative errors                           (298)
The Theta method                                (448)
TBATS: seasonality that ETS cannot represent    (561)
What is not here                                (671)
```

### 1.3 The book — exponential smoothing is not taught at all

`[V]` **No Introduction chapter teaches it.** Searching all 41 chapters
plus appendices for "exponential smoothing", "Holt-Winters", "Holt",
"Theta method" or "TBATS" as a *method being taught* returns nothing.
The only substantive mentions are:

- `introduction/30-the-kalman-filter.md:102` — one line observing that
  Holt-Winters *"turn[s] out to be [a] special case"* of the filter.
- `introduction/41-the-frontier.md` — which `[V]` **already describes
  Stage 9 correctly**, opening its ETS section with *"This section used
  to say the whole ETS taxonomy was missing. Fifteen of the thirty forms
  now exist"*.
- `introduction/B-verification.md:150` — an admonition noting *"This
  section used to list ETS as single-verified"*.

That last pair is worth noting: **chapter 41 and appendix B were already
updated when Stage 9 landed.** This is the opposite of the Stage 10
situation, where chapter 41 currently tells readers the multivariate
track does not exist and must be rewritten. Here, no correction is
needed — only addition.

Part IV, "Models That Remember" (chapters 17–23), teaches ARMA → ARIMA →
seasonal ARIMA → order selection → forecasting → honest evaluation. A
reader finishing it can fit an ARIMA and cannot fit an ETS, which is the
other half of what practitioners actually use.

---

## 2. Constraints

### 2.1 The same three build constraints

`[S]` From `docs/make.jl`: `checkdocs=:exports` and `doctest=true`, plus
the recorded HTML size threshold that caused `api.md` to be split into
`api/*.md`. `stage-10-documentation-handoff.md` §2 sets them out in
full; they apply unchanged. Two Stage-9-specific notes:

- **`checkdocs` is already satisfied** (§1.1), so the API restructure
  cannot break the build — but moving an entry out of one `@docs` block
  and into another must be done as a move, not a copy, or Documenter
  errors on the duplicate.
- **No `jldoctest` on optimizer output.** Stage 9 is full of it — `[U]`
  the stage notes record `auto_ets` needing a 4-point multi-start grid
  plus a bounded polish loop, and ETS smoothing parameters are
  optimiser output. Use plain `@example` blocks.

### 2.2 `Plots` is already available

`[V]` `docs/Project.toml` already lists `Plots` with `compat = "1"`, and
`[V]` there are zero `savefig` calls anywhere — Documenter captures the
last value of an `@example` block. Charts need no new infrastructure.

### 2.3 Indian data is **not** a problem here

`[V]` This is the welcome difference from Stage 10. Exponential smoothing
is univariate, and `dataset("iip_india")` — 180 monthly observations of
the MoSPI Index of Industrial Production, General Index — is exactly the
right shape: seasonal, trending, Indian, and already bundled with its
provenance recorded.

`[V]` It is already used in two book chapters
(`05-autocorrelation.md:213`, `12-the-diagnostic-panel.md:196`), so the
pattern is established. **No data decision is needed and nothing has to
be sourced.** Build the Indian thread of this Part on `iip_india`
throughout.

### 2.4 Hard dependency: the plot recipes

`[V]` `Forecast`, `ETSModel`, `TBATSModel` and `ThetaModel` have **zero
mentions** in `src/recipes.jl`. There is no forecast plot and no state
decomposition in the package.

**So `plot-recipes-handoff.md` §4.1 (`Forecast`) and §4.3
(`StateDecomposition`) must land before the figures in this document can
be written.** Both are Stage-9-only work needing nothing from Stage 10,
and §4.1 is the single highest-value recipe in the package — `[S]`
`Forecast` is what eleven model families return.

Order: recipes, then these figures. Writing the prose first and the
figures later is fine; writing figure code against a recipe that does
not exist is not.

---

## 3. API reference

### 3.1 One new page, one page trimmed

**Create `api/exponential-smoothing.md`**, and **move** — not copy — the
ten entries out of `api/arma-models.md`:

```markdown
# API: Exponential Smoothing, ETS and Theta

A fitted ETS model decomposed into the states it estimates — the
display R's `plot.ets` produces and that `statsmodels` has no
equivalent for. The worked explanation is in
[Manual: Exponential Smoothing, ETS and Theta](../manual/10-exponential-smoothing.md).

```@example apiets
using TSAnalytics, Plots
iip = dataset("iip_india")
m = fit_ets(iip.value, 12; error=:add, trend=:additive, seasonal=:additive)
plot(state_decomposition(m))
```

## Classical exponential smoothing

```@docs
holt_winters
ExponentialSmoothingModel
```

## ETS

```@docs
fit_ets
auto_ets
ETSModel
notation
```

## Theta

```@docs
fit_theta
ThetaModel
```

## TBATS

```@docs
fit_tbats
TBATSModel
```
```

`api/arma-models.md` then keeps its two genuine sections — the ARMA
family and the model-fitting infrastructure — and drops to about 22
lines, which is in line with the other API pages.

`[S]` `notation` currently sits in that block; check what it documents
before moving it. If it is the general ETS(error,trend,season) notation
helper it belongs here; if it is broader, it belongs in
`api/primitives.md`.

### 3.2 Register it, and fix the two inbound references

- `make.jl`: add `"api/exponential-smoothing.md"` to the
  `"API Reference"` array, after `api/arma-models.md`.
- `[V]` `manual/10-exponential-smoothing.md`'s own "See also" currently
  ends with *"[API: ARMA Models](../api/arma-models.md) — `fit_ets`,
  `auto_ets`, `fit_theta`, `fit_tbats`, `holt_winters`"*. Point it at
  the new page.
- `[V]` Search for any other inbound reference before committing; that
  line is the one this session found, not necessarily the only one.

### 3.3 One chart, at the top

Same policy as `stage-10-documentation-handoff.md` §3.3 and for the same
reason — the size threshold is a recorded, previously-hit problem.
**One figure per API page**, immediately after the title, as
orientation.

For this page it is the state decomposition, because that is the display
the page's functions exist to make possible and the one Python cannot
produce at all.

---

## 4. Manual

**No new chapter.** `manual/10-exponential-smoothing.md` already covers
the material thoroughly at 687 lines. The work is **figures**, and a
little restructuring to make room for them.

### 4.1 Eight to twelve figures, placed by section

`[V]` The manual averages 1.5 `plot(` calls per page against the book's
9.2; this chapter has 0. Target **10**, placed where the prose already
makes the argument:

| § | Figure | Why there |
|---|---|---|
| Fitting one | 1. `plot(state_decomposition(m))` on `iip_india` — observed, level, slope, season | The chapter's defining image; `[V]` R-only |
| Fitting one | 2. Fitted vs observed, with residuals beneath | What "it fits" means concretely |
| Letting it choose | 3. AICc across the 15 candidate forms, as a horizontal bar chart with the selected one marked | `auto_ets`'s decision made visible |
| Letting it choose | 4. The chosen model's decomposition against a deliberately wrong one (`seasonal=:none` on a seasonal series) | Why selection matters |
| Forecasting | 5. `plot(forecast(m, 24); history = iip.value)` — the fan chart | The output most readers want |
| Forecasting | 6. The same forecast from a damped-trend fit, overlaid | `[U]` the damped forecast converges to `l_n + φ/(1−φ)·b_n` rather than extrapolating a line — the clearest visual argument for damping |
| Where the numbers differ | 7. This package's 95% interval against R's on the same fit | `[U]` the stage notes record R's `sigma2 = sse/(n−np)` making its intervals ~3.5% wider; show it |
| Multiplicative errors | 8. An additive-error and a multiplicative-error fit on the same series, residuals side by side | Why the error type is not cosmetic: the residual scale changes with the level |
| Multiplicative errors | 9. Simulated interval coverage for a form whose exact interval has no closed form | `[U]` the 15 refused forms are refused *because* intervals must be simulated; show what that looks like |
| The Theta method | 10. Theta's forecast against SES and against the drift line it is built from | `[U]` Theta **is** SES plus half the regression slope — one figure makes the identity obvious |
| TBATS | 11. A two-seasonal-period series with `plot(state_decomposition(m))` showing both seasonal panels | `[V]` the panel-per-period structure `plot.bats` produces |
| TBATS | 12. Trigonometric seasonal terms for `k = 1` and `k = 5` harmonics | Why `k` harmonics beat `m` seasonal states |

Figures 1, 5, 6 and 11 are the ones that **require** §2.4's recipes.
The rest can be drawn from plain vectors if the recipes slip.

`[S]` Use one named `@example` session per chapter
(`@example ets`), as `manual/11-plotting.md` does with `@example plots`,
so `iip_india` loads once.

### 4.2 Carry the CI-rendering admonition, or say it was checked

`[S]` `manual/11-plotting.md` opens with a warning that its `@example`
blocks render on CI rather than having been eyeballed locally, naming
the cause (a broken `Qt6`/`GR` on the authoring machine) and telling the
next person to fix the code rather than delete the warning.

If this session cannot render locally, add the same admonition in the
same voice. If it can, say so and look at all ten figures before
committing. Either way the reader learns which it was.

### 4.3 While you are in there

- `[V]` The chapter is 687 lines because Stage 9 landed in two halves
  (the six linear forms, then the nine multiplicative ones). The
  `## Multiplicative errors` section at line 298 runs to 448 — nearly a
  quarter of the chapter. Consider whether it reads as one argument or
  as two appended ones; **do not restructure it blindly**, read it
  first.
- `[V]` Update the "See also" API link per §3.2.

### 4.4 `manual/12-coming-from-r-python.md`

`[V]` 395 lines. Stage 9 produced the project's richest set of
divergences, and `[U]` its stage notes record them: R's `sigma2`
denominator, the `statsmodels`-vs-R log-likelihood concentration
constant, `statsmodels`' damped fits sitting at the `φ = 0.98` bound,
`thetaf` returning `fitted` and `residuals` on different scales, and
three outright bugs in R's `forecast`.

Check whether that page already carries them. If it does, add the
figure from §4.1 row 7 as the visual version. If it does not, that is a
larger gap than anything else in this document and should be its own
task.

---

## 5. The book

### 5.1 A new Part, four chapters

The proposal:

```
Part IV  -- Models That Remember          chapters 17-23   (unchanged)
Part V   -- Models That Forget            chapters 24-27   <- NEW
Part VI  -- Models That Get Turbulent     chapters 28-32   (was Part V,  24-28)
Part VII -- The Machinery Underneath      chapters 33-37   (was VI,  29-33)
Part VIII-- Bringing in Outside Information  38-41         (was VII, 34-37)
Part IX  -- What Statistical Agencies Do     42-44         (was VIII,38-40)
Part X   -- The Frontier                     45            (was IX,  41)
```

**"Models That Forget"** is the title to use: exponential smoothing is
the family that weights the recent past and discounts the rest, and it
sits as the deliberate counterpart to Part IV's "Models That Remember".
`[S]` That pairing is the kind of structural rhyme
`handoff/chapter-writing-guide.md` §4 asks for — the book reading as a
story rather than a reference.

| Ch | Title | Covers |
|---|---|---|
| 24 | Weighting the Recent Past | simple exponential smoothing, Holt's trend, Holt-Winters; the classical recursions and what α, β, γ do |
| 25 | A Taxonomy of Thirty | the ETS(error, trend, season) framework, the innovations state-space form, the likelihood, AICc selection, damped trend |
| 26 | When the Error is Multiplicative | multiplicative errors and seasonals, the Jacobian that makes AICs comparable, why fifteen forms are refused, and the two R bugs found by simulation |
| 27 | Theta, and Seasonality ETS Cannot Represent | Theta as SES plus half the drift; TBATS's trigonometric seasonality, multiple and non-integer periods; the third R bug |

### 5.2 Placement, and the alternative

**Why Part V and not later.** Chapter 27's Theta section rests on the
identity that Theta *is* SES plus half the regression slope — a result
the reader can only appreciate with both forecasting families fresh.
Putting exponential smoothing immediately after ARIMA sets the two
side by side, which is also how a practitioner meets them.

**The alternative, stated honestly**: place it after the state-space
Part, since ETS's innovations form *is* a state-space model and the
chapter could then lean on machinery the reader already has. `[U]` But
the Stage 9 work established that `fit_ets` needs **nothing** from the
state-space layer — the innovations form optimises the initial states
as free parameters, so no diffuse initialisation is required. The
dependency is conceptual, not actual, and making the reader wait four
Parts for the second-most-used forecasting family is the worse trade.

### 5.3 Renumbering cost: measured, and small

`[V]` Inserting four chapters before the current chapter 24 renumbers
24–41 → 28–45. The cost:

- **7 inbound cross-references** across the entire docs tree to
  chapters 24–41 — concentrated in
  `36-calendar-effects.md` (6 inbound) and a handful of singles to
  chapters 29, 32, 34, 38, 39, 41.
- **18 lines** in `make.jl`.
- 18 file renames.

That is cheap for correct reading order. `[V]` Compare the opposite call
in `stage-10-documentation-handoff.md` §4.1, where the manual chapters
were **appended** rather than inserted because
`manual/11-plotting.md` and `12-coming-from-r-python.md` carry 14
inbound references. Same method, different answer, because the numbers
differ.

Do the renames and the `make.jl` edit as **one commit with no content
changes**, so the diff is reviewable, then add the new chapters in a
second.

### 5.4 Chapter 24 — the full specification

**File**: `introduction/24-weighting-the-recent-past.md`
**Length**: 6–8 pages
**Voice**: per `handoff/chapter-writing-guide.md` §3 — British-Indian
spelling, Indian institutions cited alongside the usual ones, dates as
*14 March 2024*, measured and unhurried prose.
**Structure**: the guide's six beats, in order.

**1. The question.** A reader has monthly IIP and wants next month's
figure. Chapter 22 gave them an ARIMA forecast. But the ARIMA treats an
observation from five years ago as evidence of the same kind as last
month's, differing only through the estimated coefficients. Is that
right? Open with the intuition every practitioner has: recent data
should count for more.

**2. The obvious attempt.** A moving average of the last `k`
observations. Show why it is unsatisfying: it weights the last `k`
equally and everything before them at zero, and `k` has to be chosen
with no principle. **Figure 1**: IIP with 3-month and 12-month moving
averages overlaid — `[S]` the package's `ma` from Stage 1.2 already does
this and chapter 04 already introduced it, so this is a callback, not
new machinery.

**3. The idea.** Weights that decay geometrically: `ŷ_{t+1} = αy_t +
(1−α)ŷ_t`. Show the recursion unrolled into its weight sequence, and
that the weights sum to one. **Figure 2**: the weight profile for
α = 0.1, 0.3 and 0.8, as decaying bars — the single most clarifying
picture in the chapter, and nothing in the reference packages draws it.

**4. Making it work.** `[V]` `dataset("iip_india")` — 180 monthly
observations, MoSPI. Fit simple exponential smoothing, then Holt's
trend, then Holt-Winters with the seasonal term. Show the SSE falling
and say plainly that a falling in-sample SSE is not by itself evidence
the model is better — `[S]` chapter 23, "Evaluating Honestly", has
already taught the reader this, so lean on it.

**Figure 3**: the three fits overlaid on the data.
**Figure 4**: `plot(state_decomposition(m))` for the Holt-Winters fit —
level, slope and seasonal pulled apart. **Figure 5**:
`plot(forecast(m, 24); history = iip.value)`.

**Detailed explanation after each example**, which the guide requires:
for Figure 4, what the level panel is actually showing and why it is
smoother than the data; for Figure 5, what the interval does and does
not account for.

**5. The complication.** Two, and the second is the chapter's hinge.
First, α, β and γ have to come from somewhere — chosen by eye, or by
minimising SSE, and those give different answers. Second: minimising SSE
is not maximising a likelihood, so there is no AIC, no standard errors,
and no principled way to choose between Holt's and Holt-Winters. That is
exactly the gap chapter 25 fills, and naming it here is what makes the
next chapter feel inevitable rather than arbitrary.

**6. Where this leaves you.** The reader can fit the classical
recursions and read their output, knows why recent data should weigh
more, and knows that the framework for choosing among them is coming.

**Figures: 5 minimum, 7 preferred.** `[V]` The book's own average is 9.2
`plot(` calls per chapter.

### 5.5 Chapters 25–27: one handoff each

`[U]` This is the pattern the user set for all 41 existing chapters
("one handoff for one chapter"), and it is what kept them consistent.
§5.1 maps them. Three notes for whoever writes those handoffs:

- **Chapter 26 carries two of the project's three `forecast` bug
  findings** — `[U]` R's no-trend seasonal interval variance applying
  the seasonal increment one period early, and its ETS(M,Ad,M) forecast
  contradicting its own fitted recursion. Both were settled by
  simulating the model rather than by argument. The writing guide's
  "the complication" beat is where they belong, and they are the
  book's strongest evidence that the verification discipline earns its
  keep.
- **Chapter 27 carries the third** — the seasonal TBATS prediction
  intervals being 2.8× too narrow.
- **All three need the honest framing**: these are bugs in a package
  that is excellent and that this project leans on heavily. The tone is
  "here is what careful checking turns up", not a victory lap.

### 5.6 What does *not* need changing

`[V]` Unlike Stage 10, chapter 41 and appendix B were **already updated
when Stage 9 landed**:

- `41-the-frontier.md`'s ETS section opens *"This section used to say
  the whole ETS taxonomy was missing. Fifteen of the thirty forms now
  exist"* and describes the multiplicative work accurately.
- `B-verification.md:150` carries *"This section used to list ETS as
  single-verified"*.

Both will need a light pass once chapters 24–27 exist — to point at them
rather than at the manual — but neither contains anything false. Check,
do not assume.

---

## 6. Order of work

1. **`plot-recipes-handoff.md` §4.1 and §4.3** — the `Forecast` and
   `StateDecomposition` recipes. Nothing visual here works without them
   (§2.4).
2. **The API move** (§3). One commit, no risk, immediately tidier.
3. **The ten manual figures** (§4.1). Highest value per hour in this
   document: 687 lines of good prose currently carry no pictures.
4. **The renumbering commit** (§5.3), content unchanged.
5. **Chapter 24** (§5.4), as the template — and look at it before
   writing 25–27, the way the writing guide says to start with chapter
   15 rather than chapter 1.
6. **Chapters 25–27**, one handoff each.
7. **The light pass** over chapter 41 and appendix B (§5.6), and
   `manual/12`'s divergence list (§4.4).

Steps 2 and 3 are independent of everything else and can go first.

---

## 7. Checklist

1. **Re-establish Julia** and a clean docs build before writing
   anything.
2. Confirm §2.4's recipes exist before writing figure code.
3. Move, do not copy, the ten `@docs` entries (§2.1) — a duplicate
   entry errors the build.
4. Check what `notation` documents before relocating it (§3.1).
5. One figure per API page, at the top; size-threshold note in the
   commit message (§3.3).
6. Ten figures in `manual/10`, one named `@example` session, Indian data
   (`iip_india`) throughout.
7. The CI-rendering admonition, or an explicit statement that the
   figures were checked locally (§4.2).
8. Renumbering as its own commit: 18 file renames, 18 `make.jl` lines,
   7 cross-references.
9. `make.jl`: the new API page, Part V inserted, Parts V–IX renamed to
   VI–X.
10. Verify §5.6 rather than assuming — read chapter 41's ETS section and
    `B-verification.md:150` before editing either.
11. Update `development-sequence.md`'s documentation section, and
    `[U]` fix the two known home-page inaccuracies while in there (the
    stale `residuals`/`forecast` sentence, and the dataset count saying
    90 where `data/` holds 88).

---

## 8. What this does not cover

- Chapters 25–27 in detail (§5.5 maps them; each needs its own handoff).
- Stage 10's documentation — `stage-10-documentation-handoff.md`.
- The plot recipes themselves — `plot-recipes-handoff.md`.
- A PDF or print build of the book.
- `[U]` Whether the remaining fifteen ETS forms (twelve
  multiplicative-trend, three ETS(A,·,M)) should be documented beyond
  being refused by name. They are refused for stated reasons that R also
  acts on; the manual's "What is not here" section already covers it,
  and chapter 26 should too, but a reader wanting multiplicative trend
  needs to be told plainly that no mainstream package offers it by
  default.
