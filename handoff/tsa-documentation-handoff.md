# Handoff: Completing the TSAnalytics.jl Documentation

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
