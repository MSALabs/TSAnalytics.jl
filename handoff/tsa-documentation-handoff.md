# Handoff: Completing the TSAnalytics.jl Documentation

## Status: COMPLETE for its own scope (2026-09-28) — but see "Documentation debt" at the end

Every page listed below is written, and every `@example` block in them
executes. The documentation has no remaining stubs.

**Stage 9 (2026-10-02/03) added four model families that this document
predates**, and the audit at the end of it records exactly what is and is
not up to date. Read that section before treating the documentation as
current.

Decisions taken, for the record:

- **§2.7, the state-space export question** — resolved in favour of
  exporting the engine. `GaussianSSM`, `TimeVaryingSSM`,
  `build_statespace`, `stationary_cov`, `combined_ar_ma`,
  `kalman_filter`, `kalman_smoother`, `to_time_varying` and
  `kalman_filter_diffuse` are public API, documented in
  `docs/src/api/state-space.md` and used directly in Manual 07.
- **§3.1, the TSA/SA boundary** — approved as proposed. Chapters 38–40
  are a bridge, not a treatment; 39 and 40's code blocks are written
  against SeasonalAdjustment.jl's real API but marked not-executed,
  since pulling the Census binary into this docs build would be the
  wrong trade. Chapter 36 turned out to cover the Diwali material in
  full, so 40 narrowed to "what it takes to *publish* an adjusted
  series", as §3.1 anticipated it might.
- **§2.x, the divergence list** — `manual/11-coming-from-r-python.md`
  is its single canonical home. Other pages link to it rather than
  restating it.

Three real bugs were found by writing the documentation, each fixed
with a regression test:

1. `stl_decompose(robust=true)` threw `DimensionMismatch` on a large
   outlier — a zero-weight Loess window produced a `NaN` that
   `_stl_lowpass`'s `filter(!isnan, ...)` then swallowed.
2. `fit_arma(y, (1,0); method=:css_ml)` returned `ar = 1.0` with
   `loglik = -Inf` while reporting `converged = true` — the CSS warm
   start walked onto a gradient plateau in the unbounded transformed
   space.
3. `test/runtests.jl` imported `Statistics: mean` but not `std`, so a
   `test_sarima.jl` assertion passed in isolation and errored under the
   full suite.

The original plan follows, unchanged.

---

Eighteen pages to write, two to expand. The conceptual half is done —
Introduction chapters 1–37 are 68,368 words — and what remains is the
onboarding and task-oriented layer.

**Read first:** `docs/src/introduction/15-stl.md` and
`docs/src/manual/10-plotting.md`. They are the two best examples of the house
voice and the only written pages in their respective sections.

---

## 0. Conventions — match these exactly

These are observed from the written chapters, not invented here. A new page that
breaks them will read as a different book.

### 0.1 Executable examples, not pre-generated figures

Chapters use Documenter `@example` blocks with a **named block per chapter**.
Documenter runs them at build time and embeds the generated plots.

````markdown
```@example ch15
using TSAnalytics, Plots, Random
Random.seed!(3)
...
plot(contaminated; title="...", legend=false)
```
````

There are **zero image references** across all 37 written chapters. Everything
is generated. Do not introduce committed `.png`/`.svg` files.

Getting Started uses `jldoctest` with a named block instead, because its claims
are assertions rather than pictures:

````markdown
```jldoctest getting-started
julia> adf_test(y).pvalue > 0.10
true
```
````

**Rule: Introduction and Manual → `@example`. Getting Started → `jldoctest`.**

### 0.2 The two house admonitions

| Admonition | Uses | Purpose |
|---|---|---|
| `!!! disagreement "When Implementations Disagree"` | 18 | R and Python give different numbers for the same nominal computation. Told as the investigation it actually was, numbered steps, with the resolution. |
| `!!! julia` | 12 | Something that is specifically a Julia-side consideration or advantage |

Both have CSS classes in `docs/src/assets/custom.css` (`.admonition.is-disagreement`,
`.admonition.is-julia`). `!!! warning` is used sparingly (twice).

**`!!! disagreement` is the package's signature device.** The verification work
in `development-sequence.md` is full of unused material for it — the Guerrero
`period=1` subseries bug, the STL `k > n` additive-vs-multiplicative width
adjustment, `AutoReg`'s `n` vs `n-k` standard-error denominator, Python's
`lmbda=0.0` truthiness quirk. Mine it rather than inventing examples.

### 0.3 Narrative continuity

Chapters open by referring back:

> Chapter 14 ended with three specific complaints against classical
> decomposition: a seasonal pattern frozen for the life of the series, no
> defence against a single bad observation, and silence at both ends.

Every new Introduction chapter must connect to the one before it. The Manual
does not — its pages are entered from search.

### 0.4 Data

Written chapters use synthetic series built inline with `Random.seed!`, plus the
bundled benchmark series the README names: **AirPassengers, Nile, sunspots**.

84 datasets ship via `dataset()`. Prefer a bundled one over an inline synthetic
series where the point is real behaviour rather than a controlled demonstration.

### 0.5 Cross-references

- Functions: `` [`acf`](@ref) ``
- Pages: `` [Plotting](../manual/10-plotting.md) `` — relative paths
- `checkdocs = :exports` and `doctest = true` are both live in CI. A broken
  `@ref` or a stale doctest output fails the build.

---

## 1. Getting Started — 3 pages to write, 2 to expand

Five pages, about 1,500 words each. This is the section a first-time user reads
and it is currently the weakest part of the docs.

### 1.1 `01-installation.md` — expand from 4 lines

Currently just `] add TSAnalytics`.

| Section | Content |
|---|---|
| Installing | The `Pkg.add` line, unchanged |
| Checking it works | A three-line session proving the package loads and runs |
| What you do not need | **The container-agnostic point belongs here, not buried in ch. 3.** No TSFrames, no TimeSeries, no DataFrames dependency — a plain `Vector` works, and so does anything satisfying `tsvalues`/`tsindex`. This is design principle 0 and it is the first thing that distinguishes the package. |
| Optional companions | `Plots.jl` for the recipes; nothing else required |

Target ~600 words. It is an installation page; it does not need more.

### 1.2 `02-first-model.md` — write

**The most important page in the section.** A reader's entire impression forms
here.

| Section | Content |
|---|---|
| A series | Load a bundled one via `dataset()` — AirPassengers is the obvious choice and the whole package is validated against it |
| Look at it first | `plot`, and one sentence on what is visible |
| Fit something | `auto_arima` — it is the package's headline convenience and it finds the airline model unaided |
| What came back | The model object, `coef`, `aic`, the selected order |
| Forecast | `forecast(m, 12)` and `plot` of the result |
| What just happened | Six sentences: differencing order chosen by KPSS, orders by stepwise search on AICc, fitting via the Kalman filter, intervals from the state covariance |

Target ~1,800 words. End by pointing at chapter 3.

### 1.3 `03-was-it-any-good.md` — expand from 193 words

Already has three good `jldoctest` sections: pre-model stationarity checks,
autocorrelation, and `diagnostic_plot` on residuals.

What is missing:

- **It never fits a model.** The residual section uses `randn(...)` as a stand-in.
  Once chapter 2 exists, use its actual fitted model's residuals.
- **No interpretation.** `adf_test(y).pvalue > 0.10` is asserted as `true` but the
  reader is not told what to conclude or what a failure would look like.
- The container-agnostic paragraph at the top belongs in chapter 1 (§1.1).

Target ~1,500 words after expansion.

### 1.4 `04-beyond-defaults.md` — write

The three or four settings a user will actually change first.

| Section | Content |
|---|---|
| When the automatic order is wrong | `fit_sarima` with an explicit order; why you might override |
| Seasonal data | `auto_arima(y; seasonal=true, D=1)` — **and the honest note that `D` must be passed explicitly**, because no seasonal unit-root test exists yet. Better said here plainly than discovered as a surprise. |
| Transformations | `boxcox`/`guerrero_lambda`, and when a log helps |
| Information criterion | `:aicc` default vs `:aic`/`:bic`, and why AICc for finite samples |

Target ~1,600 words.

### 1.5 `05-where-next.md` — expand from 140 words

Three good design notes already. Add: a short map of what lives in the Manual
versus the Introduction versus the API reference, so the reader knows where to
go for which kind of question.

Target ~800 words.

---

## 2. Manual — 10 pages to write

Task-oriented, entered from search, no narrative arc. `10-plotting.md` is the
model: short intro, `@example` per task, cross-reference and stop.

**Every section heading should be answerable as "How do I …".**

**Every page must not** restate a keyword table (the API reference has it),
explain a concept (the Introduction has it), or read as a tutorial (Getting
Started has it).

Target 1,200–2,000 words each.

### 2.1 `01-primitives.md`

`diff`/`diffinv`, `convolution_filter`/`recursive_filter`/`moving_average`,
`acf`/`pacf`, `periodogram`/`spectral_density`, `boxcox`/`guerrero_lambda`.

Tasks: difference a series and undo it; apply a moving average; get ACF with
confidence bands; choose between `pacf`'s three methods; compute a periodogram
and smooth it; find a Box-Cox lambda.

**Include the `:burg` gap** — `pacf` has three methods and not the fourth, and a
reader comparing against R will look for it.

### 2.2 `02-diagnostics.md`

All nine tests: `adf_test`, `kpss_test`, `pp_test`, `ljungbox_test`, `qs_test`,
`jarque_bera_test`, `durbin_watson_test`, `arch_lm_test`,
`dk_heteroskedasticity_test`.

Tasks: test for a unit root; test for stationarity (and why you run both); test
residuals for autocorrelation; test for seasonality in residuals; test
normality; test for ARCH effects.

**Two things worth their own sections.** Running ADF and KPSS together and
reading the four possible outcome combinations — that is how they are actually
used. And `ljungbox_test`'s vector-of-lags behaviour, which **differs from
Python's**: `acorr_ljungbox(y, lags=[5,10])` gives two cumulative statistics,
while this package sums exactly those lags. Documented in the docstring; it
belongs here too, because it is the kind of difference that silently produces a
wrong number.

### 2.3 `03-decomposition.md`

`classical_decompose`, `stl_decompose`, `mstl_decompose`.

Tasks: decompose a monthly series; choose additive vs multiplicative; handle an
outlier (`robust=true`); decompose with two seasonal periods; get the robustness
weights back; when to use which of the three.

**Mention the `parallel` keyword** on `stl_decompose` — it exists, it is on by
default, and nothing else tells a user.

### 2.4 `04-fitting-arma-models.md`

`fit_arma`, `fit_arima`, `fit_sarima`.

Tasks: fit a known order; include a mean (**and why it is silently forced off
when `d > 0` or `D > 0`** — matching R, and confusing without explanation); get
standard errors; read the `CoefTable`; what `converged = false` means.

**The `nobs` convention needs a section.** This package reports `n - d - D*s`,
matching R; `statsmodels` reports the full `n` because it uses diffuse state
augmentation. A reader comparing AIC across the two will otherwise conclude
something is broken.

### 2.5 `05-automatic-order-selection.md`

`auto_arima`, `auto_arimax`.

Tasks: let the package choose; constrain the search; stepwise vs exhaustive;
choose the information criterion; **pass `D` explicitly for seasonal data**.

The `D` limitation needs a clear, honest section — not a footnote. Say what is
missing (a seasonal unit-root test), what both references use (Canova-Hansen,
OCSB), and what to do meanwhile.

### 2.6 `06-garch-and-volatility.md`

The GARCH family plus `realizedvol.jl`.

Tasks: fit a GARCH; fit GJR/EGARCH for asymmetry; forecast volatility; compute
realized measures; **`dist = :t` is not implemented** — say so here, since a
reader will look for it.

### 2.7 `07-state-space-and-kalman.md`

**Blocked on a decision.** `build_statespace`, `kalman_filter`,
`kalman_smoother`, `combined_ar_ma` and `stationary_cov` are **unexported** —
`development-sequence.md` says "unexported until Stage 8 settles the shape", and
Stage 8 has shipped.

Three options:

1. Export them, document them normally
2. Keep them private, and make this page about what the engine does for you —
   which models share it, what diffuse initialisation means, how to tell whether
   a fit converged — without a public API to call
3. Drop the page

**Recommend option 2 for now**, with the page honestly stating the engine is
internal. Option 1 is a real API commitment that should be decided on its own
merits, not because a documentation page needs filling.

### 2.8 `08-arimax-and-regression.md`

`arx`, `arimax`, `auto_arimax`.

Tasks: fit AR-X with exogenous regressors; regression with ARIMA errors; choose
`model = :mle` vs `:tvss`; forecast with future exogenous values (and the error
when they are absent).

**The `:mle` vs `:tvss` finding belongs here** — Stage 8.3 established that both
R and Python use joint MLE for regression coefficients rather than diffuse-state
augmentation, and that the two likelihoods are not comparable. A
`!!! disagreement` box is the right form.

### 2.9 `09-forecasting-and-accuracy.md`

`forecast`, `predict`, `Forecast`, the accuracy metrics, `tscv`, the four
benchmarks.

Tasks: forecast with intervals; change the interval level; compare against
`naive`/`seasonal_naive`/`drift`/`mean_forecast`; score with
`mae`/`rmse`/`mape`/`mase`; run rolling-origin cross-validation.

**Carry the sMAPE warning across.** `smape`'s docstring has a `!!! warning`
quoting Hyndman's own recommendation against using it. A user browsing the
metric list should meet that here too.

### 2.10 `11-coming-from-r-python.md`

The migration page.

| Section | Content |
|---|---|
| R translation table | `arima`/`auto.arima`/`stl`/`decompose`/`HoltWinters`/`Box.test`/`adf.test` → equivalents |
| Python translation table | `statsmodels` and `pmdarima` equivalents |
| Where defaults differ | KPSS `nlags` (`:short` = R, `:auto` = Python); `auto_arima` IC (`:aicc` = R, `:aic` = Python); `ljungbox_test` defaulting to Ljung-Box where R's `Box.test` defaults to Box-Pierce |
| Where results differ deliberately | `nobs` convention; `partrans` parameterisation; `mstl` `lambda=0`; `ljungbox_test` vector-lag semantics |

**This page is the natural home for the divergence list**, and it is scattered
across docstrings today. Collect it once.

---

## 3. Introduction — 5 pages

### 3.1 The boundary decision, first

Chapters **38 (Official Seasonal Adjustment)**, **39 (X-13 from Julia)** and
**40 (Adjusting an Indian Series)** cover ground that SeasonalAdjustment.jl's own
planned 20-chapter Introduction treats in full.

**Decide before writing.** Recommended split:

> TSA's 38–40 are a **bridge**, not a treatment. Three short chapters
> (~1,200 words each, TSA's own shortest length) that establish *why there is a
> whole other program and a whole other package*, give one worked example, and
> hand off. SA's Introduction carries the real treatment.

Without this, chapters 10–12 of SA's book and 38–40 of TSA's say the same things
twice, in two places, maintained separately.

**Check first:** chapter 36 is `calendar-effects` and is already written at 2,367
words. It may already cover Diwali, in which case chapter 40's scope shrinks
further.

### 3.2 `38-official-seasonal-adjustment.md`

**Connects to:** Part III (chapters 13–16, decomposition) and chapter 36
(calendar effects).

The argument: STL and classical decomposition are how *you* decompose a series.
Statistical offices do something else, for reasons that are institutional as much
as statistical — reproducibility, revision policy, published diagnostics,
calendar effects modelled rather than smoothed. Introduce X-11's lineage briefly
and name X-13ARIMA-SEATS.

Ends by naming SeasonalAdjustment.jl. ~1,200 words, no deep treatment.

### 3.3 `39-x13-from-julia.md`

What the sibling package is and one worked example: load a series, `x13`, plot,
check one diagnostic. Then the handoff to SA's own documentation.

**Do not duplicate SA's Getting Started.** This is a pointer chapter with a
demonstration, ~1,200 words.

### 3.4 `40-adjusting-an-indian-series.md`

**Connects to:** chapter 36, calendar effects.

Diwali moves between October and November, so no fixed monthly seasonal factor
can absorb it — a concrete limitation of everything Part III taught. This is the
chapter that motivates the whole SA package from inside TSA's own narrative.

~1,400 words. Ends pointing at SA's Moving Holidays chapter.

### 3.5 `41-the-frontier.md`

The closing chapter. What the package does not do yet and where the field goes
next: VAR/VECM and multivariate methods, unobserved components, Markov switching,
seasonal unit-root testing, ETS. Honest about the roadmap rather than
aspirational.

~1,500 words.

### 3.6 `A-checklist.md`

One or two pages, pinnable, no narrative. What to check, which function, what
value to want, what to do when it fails. Draws on Part II (chapters 8–12).

Should cross-link to `02-diagnostics.md` in the Manual rather than repeat it —
the checklist is *what to check in order*, the Manual page is *how to call it*.

---

## 4. Testing and CI

Both gates are already live and must stay passing:

- `checkdocs = :exports` — a broken `@ref` fails the build
- `doctest = true` — a stale `jldoctest` output fails the build

New `@example` blocks execute on CI (Ubuntu). `manual/10-plotting.md` carries a
`!!! warning` explaining that its blocks were written on a machine with a broken
GR/Qt6 install and were first rendered on CI rather than locally. **If you hit
the same problem, add the same kind of note rather than deleting the examples.**

`JULIA_NUM_THREADS=4` is set in CI specifically so the threaded paths
(`stl_decompose`, `auto_arima` exhaustive) are exercised. Examples that use them
run threaded; do not assume serial output ordering.

---

## 5. Order

| # | Pages | Why |
|---|---|---|
| 1 | Getting Started 01, 02, 03, 04, 05 | Weakest section, highest traffic, and chapter 3 cannot be finished until chapter 2 exists |
| 2 | Manual 01, 02, 03 | Primitives, diagnostics, decomposition — the surface most users touch |
| 3 | Manual 04, 05, 09 | Fitting, order selection, forecasting — the core workflow |
| 4 | Manual 11 | Migration; benefits from 01–09 existing to link into |
| 5 | Manual 06, 08 | GARCH and ARIMAX |
| 6 | Manual 07 | **After the export decision in §2.7** |
| 7 | Introduction 41, A-checklist | Independent of the SA boundary |
| 8 | Introduction 38, 39, 40 | **After the boundary decision in §3.1** |

Items 1–5 are unblocked today. Only two pages and one appendix wait on decisions.

---

## 6. Open questions

1. **Export the state-space engine?** (§2.7) Blocks one Manual page and is a real
   API commitment. Stage 8 has shipped, so the stated reason for keeping it
   private has expired.
2. **TSA/SA seasonal-adjustment boundary?** (§3.1) Blocks three chapters in TSA
   and affects three in SA.
3. **Does chapter 36 already cover Diwali?** Changes chapter 40's scope.
4. **Does `dataset()` include an airline series?** 84 datasets ship;
   `02-first-model.md` assumes AirPassengers is reachable. Check before writing.
5. **Should the divergence list live in `11-coming-from-r-python.md` only**, or
   also as a standalone reference? It is currently scattered across docstrings
   and `development-sequence.md`. One canonical home is better than three.


---

# Documentation debt, audited 2026-10-03

**Resolved 2026-10-14.** All five gaps below are closed — Gap 1 by a new Part V (chapters 24–27), Gaps 2–5 in manual 05/09 and Appendices B/C — together with the Stage 10 documentation. See `development-sequence.md`, "Documentation for Stages 9 and 10".

Stage 9 shipped `fit_ets`/`auto_ets` (all fifteen models in R's default
space), `fit_theta` and `fit_tbats`. The audit below was run against the
tree at `4bb66f3`. **Nothing here is a stub or a wrong number** — it is
prose that has gone stale and coverage that was never written.

## Verified complete, and mechanically enforced

- **API reference: 133 of 133 exported names** appear in an `@docs`
  block. Not an assertion — `docs/make.jl` sets `checkdocs=:exports`, so
  a missing entry fails the build, and the build passes. `fit_ets`,
  `auto_ets`, `ETSModel`, `notation`, `fit_theta`, `ThetaModel`,
  `fit_tbats` and `TBATSModel` are all on `api/arma-models.md`.
- **`docs/src/manual/10-exponential-smoothing.md`** is new and in the
  nav at position 10, retitled "Exponential Smoothing, ETS, Theta and
  TBATS". It covers `holt_winters` → ETS → Theta → TBATS, with every
  `@example` executed at build time. Plotting and the R/Python page were
  renumbered to 11 and 12 to make room, and all inbound links updated.
- **`docs/src/manual/12-coming-from-r-python.md`** holds every divergence
  found in Stage 9, which keeps the §2.x decision above intact: that page
  is still the single canonical home for the divergence list.
- `introduction/41-the-frontier.md`, `docs/src/index.md`'s status note,
  and `development-sequence.md` are all current.

## Gap 1 — the book has no exponential-smoothing chapter (needs a decision)

The Introduction is 41 chapters and exponential smoothing appears only in
passing: chapters 30 and 33 mention it to contrast with the Kalman
filter, 41 lists it, and `C-further-reading.md` cites fpp3. There is no
chapter on it.

**This predates Stage 9.** `holt_winters` shipped at Stage 5.5 and never
got one either. But ETS, Theta and TBATS now make it a four-model hole in
the narrative, and the book is the one part of the documentation a reader
goes through in order.

It needs a structural decision before anything is written, which is why
it is listed first and was not simply fixed:

- **One chapter**, after chapter 23, closing Part IV. Cheap: no
  renumbering, no `make.jl` churn.
- **A short Part** (smoothing → the ETS taxonomy → Theta and TBATS)
  slotted after chapter 23. Better pedagogy, but renumbers chapters
  24–41 and every cross-reference to them — roughly 40 files.

Either way the material already exists in Manual 10 and can be adapted
rather than written cold.

## Gap 2 — `manual/09-forecasting-and-accuracy.md:28` lists the wrong types

It reads:

> **`forecast(m, h)` is the signature on every forecastable type** —
> `ArmaModel`, `ArimaModel`, `SarimaModel` and `ARXModel` all retain the
> series they were fitted to.

Three types are missing: **`ETSModel`, `ThetaModel`, `TBATSModel`**.
Checked against the source rather than guessed — those three have
`forecast(model, horizon)` methods, and `ExponentialSmoothingModel`
(`holt_winters`) and `AutoRegModel` correctly do **not** appear because
neither forecasts. `ArimaxModel`/`SarimaxModel` are already handled
further down that page via their `newexog` signature.

This is the page a reader lands on for forecasting, so it is the gap most
likely to mislead.

## Gap 3 — `manual/05-automatic-order-selection.md` covers only `auto_arima`

Zero occurrences of "ets", case-insensitive. The chapter is titled
"Automatic Order Selection" and `auto_ets` now runs a fifteen-model AICc
search with an `additive_only` switch and a parallel path that is
asserted to select the identical model. Either a section here or a clear
cross-reference to Manual 10.

## Gap 4 — `C-further-reading.md` cites none of Stage 9's primary sources

The file has a single `# Further Reading` heading and no subsections. Three
works are now primary sources for shipped code and none is listed:

- **Hyndman, Koehler, Ord & Snyder (2008)**, *Forecasting with
  Exponential Smoothing* — the ETS taxonomy, the parameter regions, and
  the class 1/2/3 variance results `fit_ets` implements.
- **Assimakopoulos & Nikolopoulos (2000)** — the Theta method; and
  **Hyndman & Billah (2003)**, whose SES-plus-drift equivalence is what
  `fit_theta` actually implements.
- **De Livera, Hyndman & Snyder (2011)** — TBATS.

## Gap 5 — Appendix B describes only two kinds of disagreement

**Correcting an earlier reading of my own here**: the "Eighteen of these
appear across the Introduction" count is *not* stale. It counts
Introduction chapters carrying a `disagreement` box, and Stage 9's
findings were written up in the Manual, not in the book. The count
stands.

The actual gap is that `B-verification.md` frames verification as two
cases — the references disagree with each other, or they share a
limitation this package shares too — and Stage 9 needed a third:

> **A reference is simply wrong, and the matter was settled by
> simulating the model's own recursion rather than by consulting a
> second reference.**

That happened three times, and it is the method that did the decisive
work, so the appendix that explains the verification standard should
describe it:

1. **R's ETS(A,N,A)/(M,N,A) intervals** apply the seasonal variance
   increment one period early — `floor(h/m)` elapsed seasonal shocks
   where the correct count is `floor((h-1)/m)`. 4,000,000 paths give
   `2.5808*sigma2` at `h = 4` against R's `3.0536*sigma2`; 6,000,000
   from R's own fitted ETS(M,N,A) state agree with this package to 0.09%
   at every horizon against R being +25.6% too wide at `h = 4`.
2. **R's ETS(M,Ad,M) forecast contradicts R's own fitted model** —
   it accumulates the damped trend as `(1 + phi + ... + phi^(h-1))` where
   its own one-step recursion `(l + phi*b)*s` implies
   `(phi + ... + phi^h)`. 8,000,000 paths side with this package.
3. **R's seasonal TBATS intervals are 2.8x too narrow** — `c_j = alpha`
   for every shock weight where the correct weight is `w'F^(j-1)g`. At
   `h = 8` R gives `2.640` against `7.31` from this package, from
   Python's `tbats`, and from 3,000,000 paths.

In all three cases R's *neighbouring* branch is correct — non-seasonal
`bats`, trended ETS, non-seasonal damped ETS — which is how each was
localised rather than written off as "R is wrong". That nuance belongs in
the appendix too, because it is what makes the claims credible.

## Also worth recording, not documentation gaps

- **`getting-started/05-where-next.md` has no route to Manual 10.** Its
  routing table predates the chapter. One row.
- **`development-sequence.md` rows 0.1–0.2 and 6.1–6.4 carry no
  `built` marker** although they are built — the state-space engine is in
  `src/statespace/`. Since that file doubles as the changelog, this is a
  gap in it. Flagged twice and still unfixed; it is a two-minute edit.
- **Two deliberate gaps in Stage 9 itself**, already documented in the
  manual and the error messages, listed here so they are not mistaken
  for documentation debt: no `auto_tbats` (R's AIC search over
  Box-Cox/trend/damping/ARMA/harmonics), and no ARMA errors in TBATS —
  the recursion handles them but the forecastability condition is only
  derived for the no-ARMA state, so a non-zero order errors rather than
  fitting unconstrained.

# Documentation debt added by Stage 10, audited 2026-10-13

Stage 10 shipped with its **manual** coverage complete — chapter 11
(VAR, IRF/FEVD, structural VARs, residual diagnostics), a new chapter 12
(cointegration and the VECM), and the R/Python translation chapter
renumbered to 14 with every Stage 10 divergence. What it did **not**
touch is the Introduction (the book) and its apparatus:

1. **The book has no multivariate part.** Chapters 1–41 are univariate.
   VAR, impulse responses, Granger causality and cointegration are a
   natural Part (the frontier chapter 41 should point at it), and the
   spurious-regression story is the strongest motivating example the book
   could have for chapter 03's differencing argument.
2. **`C-further-reading` lacks every Stage 10 primary source:**
   Lütkepohl (2005); Johansen (1988, 1991) and Johansen & Juselius (1990);
   Osterwald-Lenum (1992); Engle & Granger (1987); MacKinnon (1994, 2010);
   Phillips & Ouliaris (1990); Blanchard & Quah (1989); Amisano &
   Giannini (1997); Kilian (1998) on bootstrap IRF bands; Edgerton &
   Shukur (1999).
3. **Appendix B (verification) has no Stage 10 entries**, and Stage 10
   added several to its "reference wrong" category: `urca::ca.po`'s `Pz`
   with a constant or trend (not level-invariant, ~0% size), `urca`'s
   Johansen `ecdet="none"` using Osterwald-Lenum's Table 1.1\*,
   `statsmodels`' SVAR `llf` being the reduced form's, and `vars`'
   package-wide `1 − CDF` p-values. The `_std_normal_cdf` tail bug
   (this package's own, fixed in 10.4a) belongs in the "we were wrong"
   column alongside them.
4. **`getting-started/05-where-next.md`** has no route to chapters 11
   or 12.
