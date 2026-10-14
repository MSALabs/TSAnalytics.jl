# Handoff: Documentation for Stage 10 — API reference, manual, and the book

**Status: implemented (2026-10-14)** — see `development-sequence.md`, "Documentation for Stages 9 and 10", for what was built and where it departs from this document. *Originally:* pending. Lands after the Stage 10 code; §2's build constraints
mean the API pages are **not optional** — the docs build fails without
them.

For a fresh Claude Code session with no prior context. Read
`stage-10-overall-handoff.md` for the stage's own conventions and
`plot-recipes-handoff.md` for the charts this document asks for; the
documentation conventions themselves are all here.

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every claim about the current docs is `[V]` or `[S]`.

---

## 1. What exists today, counted

`[V]` The docs tree, and where charts actually are:

| Section | pages | `@example` blocks | `plot(` calls |
|---|---|---|---|
| `getting-started/` | 5 | 7 | 5 |
| `manual/` | 12 | 145 | **18, across only 5 of 12 pages** |
| `introduction/` (the book) | 41 chapters + appendices A/B/C | 368 | **379** |
| `api/` | 9 | **0** | **0** |

`[V]` `plot(` appears in `manual/11-plotting.md` (8), `01-primitives.md`
(5), `03-decomposition.md` (2), `06-garch-and-volatility.md` (2),
`02-diagnostics.md` (1) — and nowhere else in the manual.

`[V]` `Plots` is **already** a `docs/Project.toml` dependency with
`compat = "1"`, and there are **zero** `savefig` calls: Documenter
captures the last value of an `@example` block and embeds the image. So
charts need no new infrastructure anywhere — only `@example` blocks.

`[S]` API pages are thin by design: 10–74 lines each, grouped `@docs`
blocks under headings, no prose. `api/decomposition.md` is 10 lines and
is the one acknowledged stub.

---

## 2. Three build constraints, and one is the reason this document exists

`[S]` From `docs/make.jl`:

```julia
doctest=true,
checkdocs=:exports,  # build fails if an exported name has no docstring
```

**1. `checkdocs=:exports` makes the API pages mandatory.** `[V]`
Counting the public names the ten Stage 10 handoffs propose — functions
and types declared in their `##  Proposed Julia API` sections, excluding
`forecast`/`predict`/`Forecast`, which already exist — gives **34 new
functions and 16 new types, about 50 new public names**:

```
arch_lm_test  bg_test  blanchard_quah  coef_alpha  coint_rank  coint_test
companion  critical_values  cum_effect_stderr  es_test  fevd  fit_svar
fit_var  fit_vecm  granger_test  instantaneous_causality  irf  irf_stderr
is_stable  johansen_test  lr_effect_stderr  lr_effects  ma_rep
normality_test  normalize_beta  orth_ma_rep  phillips_ouliaris_test
portmanteau_test  roots  tsmatrix  tsnames  var_diagnostics  var_rep
var_select

ArchLMTest  BreuschGodfreyTest  CointTest  EdgertonShukurTest  FEVDResult
GrangerTest  IRFResult  InstantaneousCausalityTest  JohansenTest
MultivariateNormalityTest  PhillipsOuliarisTest  PortmanteauTest
SVARModel  VARModel  VARSelection  VECMModel
```

Every one needs a docstring **and** an entry in a `@docs` block, or
`makedocs` errors. Re-derive the list from the shipped exports rather
than trusting this one — it was parsed from handoff prose, and a few
names will have changed during implementation.

**2. `doctest=true` in CI.** Any `jldoctest` in a Stage 10 docstring
must match byte-for-byte on CI. For numerically optimized output — SVAR
coefficients, VECM β — **do not write a `jldoctest`**; use a plain
`@example` block, or round hard inside the doctest. `[S]` The make.jl
comment says the same thing from the other side: *"leave as default
(true) in CI so a docstring example silently drifting out of date fails
the build."*

**3. The HTML size threshold.** `[S]` From `make.jl`:

> `size_threshold` workaround removed -- no longer needed once `api.md`
> was split into `docs/src/api/*.md` (see
> `handoff/docs-restructure-skeleton-handoff.md`); if this comes back,
> something regressed.

That is a direct warning against loading the API pages with figures.
§3.3 settles how many charts an API page gets, and the answer is "one".

---

## 3. API reference

### 3.1 Three new pages, three existing pages extended

| File | Contents |
|---|---|
| **`api/var.md`** (new) | `fit_var`, `var_select`, `VARModel`, `VARSelection`, `VARForecast`, `roots`, `is_stable`, `companion`, `ma_rep`, `orth_ma_rep`, `irf`, `IRFResult`, `irf_stderr`, `cum_effect_stderr`, `lr_effects`, `lr_effect_stderr`, `fevd`, `FEVDResult`, `coef_alpha` |
| **`api/cointegration.md`** (new) | `coint_test`, `CointTest`, `phillips_ouliaris_test`, `PhillipsOuliarisTest`, `johansen_test`, `JohansenTest`, `coint_rank`, `normalize_beta`, `critical_values`, `fit_vecm`, `VECMModel`, `var_rep` |
| **`api/structural.md`** (new) | `fit_svar`, `SVARModel`, `blanchard_quah` |
| `api/diagnostics.md` (extend) | `portmanteau_test`, `PortmanteauTest`, `normality_test`, `MultivariateNormalityTest`, `arch_lm_test`, `ArchLMTest`, `bg_test`, `BreuschGodfreyTest`, `es_test`, `EdgertonShukurTest`, `var_diagnostics` — under a new `## Multivariate` heading, after the existing univariate ones |
| `api/primitives.md` (extend) | `tsmatrix`, `tsnames`, beside `tsvalues` |
| `api/plotting.md` (extend) | every new recipe from `plot-recipes-handoff.md` |

`[S]` Follow `api/forecasting.md`'s shape exactly — a `#` title, then
`##` headings each wrapping one `@docs` block, no prose between them:

```markdown
# API: Vector Autoregression

## Fitting

```@docs
fit_var
VARModel
var_select
VARSelection
```

## Stability

```@docs
roots
is_stable
companion
```
```

Add the three pages to `make.jl`'s `"API Reference"` array, after
`api/arimax.md` and before `api/forecasting.md`, so the order tracks the
manual's.

### 3.2 `api/decomposition.md` — fix the stub while you are here

`[S]` It is 10 lines and the only acknowledged stub in the docs. Stage
10 should not add a page while leaving it. Filling it is a 20-minute job
and it is on the project's open-items list.

### 3.3 Charts on API pages: exactly one per page, at the top

This is the one place where the request and the build constraint pull
against each other, so decide it once here.

API pages are lookup surfaces — a reader arriving at `api/var.md` wants
a signature, and `[V]` no API page in the package has ever carried a
figure. Against that, §2's size threshold is a recorded, previously-hit
problem.

**The resolution: one figure per new API page, immediately after the
title, before the first `@docs` block** — the page's single most
representative output, as an orientation device rather than a tutorial.
Everything else goes in the manual.

```markdown
# API: Vector Autoregression

A fitted VAR's impulse responses — the output most of this page exists
to produce. The worked explanation is in
[Manual: Vector Autoregression](../manual/13-vector-autoregression.md).

```@example apivar
using TSAnalytics, Plots
e = dataset("econ5")
m = fit_var(hcat(e.unemp, e.gnp, e.consum), 2; deterministic=:const)
plot(irf(m, 12; stderr=true))
```

## Fitting
...
```

One figure per page, three pages, three figures. Per page:

| Page | the one figure |
|---|---|
| `api/var.md` | the IRF grid with analytic bands |
| `api/cointegration.md` | the cointegrating relations with their ACF (`coint_relations`) — `[V]` the plot **no Python library has** |
| `api/structural.md` | a Blanchard-Quah structural IRF |

If the build then warns about page size, the figure comes out of the API
page and stays in the manual. Say that in the commit message so the next
person knows the order of precedence.

---

## 4. Manual

### 4.1 Four new chapters, appended as 13–16

```
manual/13-vector-autoregression.md
manual/14-causality-and-impulse-responses.md
manual/15-cointegration-and-vecm.md
manual/16-structural-var.md
```

**Append; do not renumber.** `[V]` `manual/11-plotting.md` and
`12-coming-from-r-python.md` are referenced **14 times across 12 files**
(including `index.md`, both appendices, `getting-started/02` and `/05`,
and five other manual chapters). Inserting the new chapters at 11–14 and
pushing those two to 15–16 means editing all fourteen references for a
cosmetic gain. The sidebar shows titles, not numbers.

### 4.2 Length and shape

`[V]` The manual's existing range is 104–687 lines, median ~226.
`manual/10-exponential-smoothing.md` is the outlier at 687 because
Stage 9 landed in two halves. Target **250–350 lines** per new chapter —
`08-arimax-and-regression.md` (323) and `09-forecasting-and-accuracy.md`
(344) are the right models.

`[S]` The house pattern per section: a short prose statement of what the
function is for, a named `@example` block, then the honest note about
what it assumes or where it disagrees with R/Python. The manual is where
the convention traps from the stage charter §5 belong — a user hitting
the `roots` reciprocal or the `ecdet="none"` naming needs it here, not
in a handoff.

### 4.3 Charts: this is the section that fixes the manual's gap

`[V]` The manual averages 1.5 `plot(` calls per page against the book's
9.2. The four new chapters should carry **6–10 figures each**, which is
both what the user asked for and what brings the manual into line.

Per chapter, the figures that earn their place:

**13 — Vector Autoregression**
1. The three `econ5` series, stacked (`plot` on the matrix)
2. `var_select` criteria across lags — four lines, the minimum marked
3. Fitted vs observed, per equation
4. `roots` on the complex plane with the unit circle
5. `plot(forecast(m, 12))` — the `VARForecast` fan chart
6. The same forecast on a near-unit-root fixture, to show the fan widening
7. `var_diagnostics` panel (forward-reference to chapter 14 or 16)

**14 — Causality and Impulse Responses**
1. The IRF grid, orthogonalized
2. The same grid non-orthogonalized, side by side — **the ordering
   lesson**, and `[V]` the one where permuting columns changes the
   orthogonalized answer completely while leaving the other unchanged
3. Cumulative IRF
4. IRF with analytic bands and with bootstrap bands, labelled, so the
   reader sees they differ
5. FEVD stacked bars
6. FEVD at a long horizon, showing the shares settle
7. A Granger-causality summary table rendered as a heatmap of p-values

**15 — Cointegration and VECM**
1. Two cointegrated series on one axis, their spread beneath
2. The same for two independent random walks — **the null being true
   looks different**
3. The Engle-Granger residual with its ADF statistic annotated
4. The Johansen trace statistic against its critical values, per rank
5. `coint_relations` — the relations and their ACF
6. A VECM forecast, with the levels reverting to the relation
7. `var_rep`'s roots, showing the unit root a VECM has by construction

**16 — Structural VAR**
1. Reduced-form vs structural IRF for the same VAR
2. A Blanchard-Quah decomposition, both shocks
3. The long-run impact matrix as a labelled heatmap, its upper triangle
   visibly zero
4. Structural FEVD
5. The A-model's estimated `A` as an annotated matrix

`[S]` Use named `@example` blocks (`@example var`, `@example svar`) so
each chapter shares one session and the data loads once — the pattern
`manual/11-plotting.md` already uses with `@example plots`.

### 4.4 Carry forward the CI-rendering admonition

`[S]` `manual/11-plotting.md` opens with a warning that its `@example`
blocks render on CI rather than having been eyeballed locally, naming
the specific cause (a broken `Qt6`/`GR` on the authoring machine) and
instructing the next person to fix the code rather than delete the
warning.

**If the session writing these chapters cannot render locally, add the
same admonition, in the same voice.** If it *can* render, say so
instead, and look at every figure before committing. Either way the
reader learns which it was.

### 4.5 Extend `manual/12-coming-from-r-python.md`

`[V]` 395 lines, and Stage 10 gives it the richest material in the
package. A new `## Multivariate` section should carry the mapping table
and, in particular, the five traps that silently produce wrong answers:

| TSAnalytics | R | Python | the trap |
|---|---|---|---|
| `roots(m) < 1` | `vars::roots` | `.roots`, stable if `> 1` | `[V]` exact reciprocals |
| `lags` (differenced) | `urca`'s `K` = `lags + 1` | `k_ar_diff` | off by one |
| `deterministic=:const` | `ecdet="none"` | `det_order=0` | `[V]` `"none"` still has a constant |
| `var_select` IC | `VARselect` | `select_order`, returns lag **0** too | `[V]` off-by-one and a fixed sample |
| `granger_test(m; cause)` | `vars::causality` | `.test_causality` | `[V]` agrees; `lmtest::grangertest` is a **different** test |

`[S]` That page's existing style is a table plus a worked
"here is the same thing in both" example — follow it.

---

## 5. The book

### 5.1 Chapter 41 must be rewritten, and that is not optional

`[V]` `introduction/41-the-frontier.md` currently tells the reader, at
lines 32–47, that the package has **no VAR, no cointegration testing, no
vector error-correction model**, explains why a VAR is a different kind
of object, and at lines 198–199 says *"VAR is the biggest single
addition, opening the whole multivariate track behind it."*

Once Stage 10 lands, the book's final chapter is **factually wrong about
its own package**. Rewriting it is part of shipping Stage 10, not a
follow-up.

What the rewrite keeps: the chapter's job, which is an honest map of what
exists and what does not. What it loses: the multivariate section. What
it gains: the genuinely remaining frontier — `[V]` from the handoffs'
own out-of-scope sections — multiplicative-trend ETS, Sims-Zha IRF
bands, restricted-β/α likelihood-ratio tests, Gregory-Hansen,
threshold VECM, sign-restricted SVAR, structural-break and CUSUM tests,
Toda-Yamamoto, frequency-domain causality, generalized (Pesaran-Shin)
impulse responses, and unobserved components / dynamic factor models
(Stage 11).

### 5.2 Part structure

`[V]` Current: Part IX is "The Frontier" with chapter 41 alone, followed
by appendices A/B/C.

Proposed:

```
Part IX -- Series That Move Together        <- new, chapters 42-48
Part X  -- The Frontier                     <- chapter 41 renumbered to 49
Appendices                                   <- unchanged
```

`[V]` Renumbering 41 → 49 is cheap: `41-the-frontier.md` is referenced
only **twice**, in `introduction/40-adjusting-an-indian-series.md` and
`getting-started/05-where-next.md`. Do it, so numeric order matches
reading order — the opposite call from §4.1, and for the opposite
reason.

### 5.3 Part IX: seven chapters

`[S]` Parts IV (7 chapters) and V (5) are the precedent; the writing
guide sets 6–8 pages per chapter, so Stage 10's material does not fit in
one. The mapping:

| Ch | Title | Covers | Stage item |
|---|---|---|---|
| 42 | Two Series at Once | why a VAR, cross-correlation, the parameter cost, lag choice | 10.1 |
| 43 | Fitting a Vector Autoregression | estimation, order selection, stability, diagnostics | 10.1, 10.8 |
| 44 | Does One Series Help Predict Another? | Granger causality, the system/pairwise distinction, what "causality" does not mean | 10.2 |
| 45 | Tracing a Shock | IRF, the ordering problem, FEVD | 10.3 |
| 46 | Series That Cannot Drift Apart | the cointegration idea, spurious regression, Engle-Granger, Phillips-Ouliaris | 10.4a, 10.4b |
| 47 | How Many Relations? | Johansen, rank selection, VECM | 10.5, 10.6 |
| 48 | Shocks With Names | structural identification, A/B models, Blanchard-Quah, and identification as an assumption | 10.7 |

**This document specifies chapter 42 in full, as the template.** The
remaining six get their own handoffs, one per chapter — `[U]` which is
the pattern the user set for all 41 existing chapters ("one handoff for
one chapter"), and it is what kept them consistent.

### 5.4 Chapter 42 — the full specification

**File**: `introduction/42-two-series-at-once.md`
**Length**: 6–8 pages
**Voice**: per `handoff/chapter-writing-guide.md` §3 — British-Indian
spelling, Indian institutions cited alongside the usual ones, dates as
*14 March 2024*, measured and unhurried prose.
**Structure**: the guide's six beats, in order.

**1. The question.** A reader has IIP and a price index, or Lake
Shasta's inflow and rainfall, and can already model each one. What does
modelling them *together* buy? Open with the honest answer: sometimes
nothing, and the chapter will show when.

**2. The obvious attempt.** Fit two univariate ARIMAs and regress one
residual on the other. Show why this is not enough: it imposes a
direction, it cannot represent feedback, and it throws away the
contemporaneous covariance. `[V]` Use `dataset("climhyd")` — 454 × 6,
GPL-3, Shumway & Stoffer's own cross-correlation example — and the
package's existing `ccf`.

**Figure 1**: the two series. **Figure 2**: their cross-correlation
(`ccf`, `[S]` which returns `ACFResult` tagged `:ccf`, so the existing
recipe draws it), with the lead-lag asymmetry visible.

**Detailed explanation after the example** — the guide requires this and
it is the chapter's substance: what the CCF peak at a negative lag
means, why it is not symmetric, and why "x leads y" is a statement about
prediction, not mechanism.

**3. The idea.** Each series regressed on the lagged history of *all*
of them. State the parameter cost plainly: `k²p + k` coefficients
against a univariate model's `p + 1`. `[V]` At `k = 5, p = 2` —
`econ5`'s dimensions — that is 55 against 3.

**4. Making it work.** `[V]` `dataset("econ5")` → 161 rows, 5 columns
(`unemp`, `gnp`, `consum`, `govinv`, `prinv`), astsa, GPL-3, Shumway &
Stoffer — the canonical VAR example from one of the six books, already
bundled. Fit a small VAR, show the coefficient table, show
`var_select`'s criteria disagreeing.

**Figure 3**: the five series, stacked. **Figure 4**: `var_select`'s
four criteria across lags, with each minimum marked — `[V]` on `econ5`
they select **5, 2, 1, 5** for AIC, HQ, SC, FPE, which is a real
disagreement and the honest teaching moment.

**5. The complication.** Two of them. First, `k²p` grows fast and 161
quarterly observations do not support `p = 5` at `k = 5` — that is 130
parameters. Second, and this is the beat the chapter builds to: both
`gnp` and `consum` look non-stationary, and a VAR in levels on
non-stationary series is the spurious-regression trap chapter 46 is
about. End by naming where that goes.

**6. Where this leaves you.** The reader can fit and read a VAR, choose
its order, and knows that the ordering question (chapter 45) and the
stationarity question (chapter 46) are still open.

**Figures: 5 minimum, 7 preferred.** The guide's requirement is "as many
charts as possible"; the measured target is the book's own average,
`[V]` 9.2 `plot(` calls per chapter.

### 5.5 The Indian-data problem, and a decision needed

`[V]` **There is no Indian multivariate series in the bundle.**
`dataset("iip_india")` is 180 rows × 2 columns (`date`, `value`) —
univariate, MoSPI, "provenance as supplied by the project maintainer".
Every bundled multivariate dataset is American or Australian: `econ5`
and `climhyd` from astsa, `aus_production` and `global_economy` from
tsibbledata.

That is a problem for a Part about series moving together, in a book
whose stated requirement is Indian data throughout rather than in one
token chapter.

**This needs your decision, and it needs data.** The options:

1. **Supply 2–4 Indian series** that plausibly move together, at a
   common frequency, long enough for a VAR — IIP general index with
   CPI and the repo rate would carry chapters 42–48 and give a real
   cointegration example. `[S]` The provenance convention already
   exists: the catalogue's MoSPI entry cites the ministry and states
   the data was supplied by the maintainer rather than independently
   re-downloaded. Follow it exactly.
2. **Build an Indian two-series example from IIP plus one more series**,
   and use `econ5` for the larger-`k` mechanics. Weaker, but it keeps
   the Indian thread alive.
3. **Say so in the text.** `[S]` The writing guide §6 already requires
   this when a canonical example cannot be reproduced; the same honesty
   applies to a missing Indian one.

**Do not invent or scrape the data.** A fabricated Indian series in a
book that stakes its credibility on verification would be the worst
possible failure, and it would be undetectable to a reader.

My recommendation is (1), and the concrete ask is: which series, at what
frequency, and over what span — then the CSVs with their source, and
`data/inventory.tsv` gets the rows.

---

## 6. Order of work

1. **Docstrings as the code lands**, not afterwards. `[V]`
   `checkdocs=:exports` means an undocumented export breaks the build,
   so every Stage 10 PR carries its own docstrings.
2. **The three API pages**, as each item lands. Plus
   `api/decomposition.md`'s stub (§3.2).
3. **The four manual chapters**, after the matching items, with the
   plot recipes from `plot-recipes-handoff.md` in place — `[V]` without
   them there is nothing to call, and the manual is where most of the
   charts go.
4. **Chapter 41's rewrite** (§5.1), which can happen as soon as the code
   is merged and does not wait for the new chapters.
5. **Chapter 42**, per §5.4, as the template — and look at it before
   writing 43–48, the way the writing guide says to start with chapter
   15 rather than chapter 1.
6. **Chapters 43–48**, one handoff each.
7. **`manual/12-coming-from-r-python.md`**'s multivariate section (§4.5).

---

## 7. Checklist

1. **Re-establish Julia** and a docs build before writing anything;
   `[V]` `make.jl` runs doctests and `checkdocs`, so a broken build is
   the first thing you will see and the last thing you want to debug
   alongside new prose.
2. Re-derive §2's name list from the **shipped** exports, not from this
   document.
3. No `jldoctest` on optimizer output (§2 constraint 2).
4. One figure per API page, at the top, and the size-threshold note in
   the commit message (§3.3).
5. Named `@example` blocks per manual chapter; 6–10 figures each.
6. The CI-rendering admonition, or an explicit statement that the
   figures were checked locally (§4.4).
7. `make.jl`: three new API pages, four new manual pages, Part IX
   renamed and Part X added, `41-the-frontier.md` → `49-the-frontier.md`
   with its two references updated.
8. Chapter 41's rewrite is **in scope**, not deferred (§5.1).
9. Bring the Indian-data decision (§5.5) to the user **before** writing
   chapter 42, since it changes which dataset the chapter is built on.
10. Update `development-sequence.md`'s documentation section and
    `docs/src/index.md`'s feature list — `[U]` the home page also has
    two known inaccuracies on the project's open-items list (a stale
    `residuals`/`forecast` sentence and a dataset count saying 90 where
    `data/` holds 88); fix them in the same pass.

---

## 8. What this does not cover

- The SeasonalAdjustment.jl docs, which are a separate site.
- Stage 11 and 12 content.
- A PDF or print build of the book. `[S]` The book currently exists only
  as the Documenter site's "Introduction to Time Series Analysis"
  section; whether it also becomes a typeset ~300-page document is an
  open question this document does not answer.
- Chapters 43–48 in detail. §5.3 maps them; each needs its own handoff.
