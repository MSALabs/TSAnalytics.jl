# Stage 11 — Plot recipes

**Prerequisite reading:** `stage-11-overall-handoff.md`, and the existing
`handoff/plot-recipes-handoff.md` from the Stages 9–10 plotting work, whose
conventions this document extends rather than restates.

**Eighteen recipes already exist** in `src/recipes.jl` (`ACFResult`,
`PeriodogramResult`, three decompositions, two diagnostic-plot types, `Forecast`,
`VARForecast`, `VolatilityForecast`, `StateDecomposition`, `ARMARootsResult`,
`TSDisplayResult`, `IRFResult`, `FEVDResult`, `RootsPlotResult`,
`VARDiagnosticPlotResult`, `CointegrationPlotResult`). [V] grep at HEAD
`ad0010e`. Stage 11 adds **thirteen**.

---

## 1. The established pattern — follow it exactly

Three rules, all already in force, all load-bearing.

### 1.1 A result type in `src/plotdata.jl`, a recipe in `src/recipes.jl`

Data extraction and rendering are separate files, and the data is useful on its
own. `state_decomposition(m::ETSModel) -> StateDecomposition` is the template:
the function computes, the struct carries, the recipe draws.

### 1.2 `RecipesBase` stays a soft dependency

The header comment in `src/recipes.jl` [S] explains the reasoning and it has not
changed: `plot(result)` works for any caller with a backend loaded, and the
package takes on no real plotting dependency.

### 1.3 Never write a recipe on a bare `NamedTuple`

Also from that header: a `NamedTuple` recipe "would hijack plotting for *any*
unrelated NamedTuple in a session with TSAnalytics loaded". `PeriodogramResult`
exists specifically so a recipe could be attached without that hazard.

**This has a direct consequence for Stage 11:** every plottable Stage 11 output
needs a named struct, even where a tuple would be the natural return. Budget for
thirteen structs, not thirteen recipes on loose data.

### 1.4 The refusal precedent

`seasonal_subseries_plot` returns plain data with **no recipe**, and its
docstring says why: there is "no one natural rendering … a recipe would just be
encoding one opinionated choice among several equally valid ones." [S]

Stage 11 has one case in the same position (§6.4) and it gets the same treatment.

### 1.5 Decimation

`_decimate_local_max` exists for long series [S]. Stage 11's only candidates for
it are 11.4's phase plots on long simulated series (fixture C is 1000 points) and
11.2's residual boxplots. Reuse it; do not write a second one.

---

## 2. Stage 11.1 — Unobserved Components

### 2.1 `UCDecomposition` → component panels **with bands**

This is the headline Stage 11 plot and the one the references handle worst.

**What the references do:**

| reference | what it draws |
|---|---|
| statsmodels `plot_components` | **216 lines** [S]; panels for level, trend, seasonal, freq_seasonal, cycle, autoregressive, regression — with confidence bands |
| R `plot.StructTS` | the three BSM states only |
| R `plot.ets` (for contrast) | **at most four panels**, using `s1` alone for the seasonal [U, from the Stage 9 work] |

**Decision: reuse `StateDecomposition` for the rendering, and extend it with
optional standard errors.**

`StateDecomposition` already exists for ETS and TBATS and already draws one panel
per present state (`src/plotdata.jl:32`) [S]. A UC decomposition is the same
shape. Adding an optional `se` field — `Union{Nothing,Vector{Float64}}` per
component — means:

- the existing ETS and TBATS recipes pass `nothing` and are **completely
  unaffected**, which must be asserted by a test that their rendered output is
  unchanged;
- the UC recipe gets bands for free;
- a future ETS variant that can produce state standard errors gets them without
  another struct.

```julia
state_decomposition(m::UCModel) -> StateDecomposition
```

so the verb is the same across model types, which is the whole point of the
existing design.

**Band convention:** ±1.96·se, drawn as a ribbon, **labelled as a 95% interval
for the smoothed component**, not as a forecast interval. The distinction matters
and the label is where it gets made.

**The `irregular` panel gets no band** — it is a residual, and a band around it
would invite the wrong reading.

### 2.2 Fitted versus observed

Covered by the existing `Forecast` recipe's `history=` keyword [U] for the
forecast case. For the in-sample case, `StateDecomposition`'s first panel should
show `y` with the fitted values overlaid, which is what both references do.

### 2.3 What not to build

No separate "UC diagnostic plot". `DiagnosticPlotResult` already exists and takes
residuals; `diagnostic_plot(m::UCModel)` should dispatch to it. One recipe, not
two.

---

## 3. Stage 11.2 — Dynamic factor models

Five recipes. **Two of them exist in neither reference**, which is the
interesting part.

### 3.1 `FactorPathResult` — factor paths with bands

**Neither reference ships this.** [V] statsmodels gives you
`factors.smoothed` and `factors.smoothed_cov` and no plot; `dfms`' `plot.dfm`
draws paths with no bands.

We have `smoothed_cov` from the smoother, so the band is the obvious use of data
already computed. One panel per factor, ribbon at ±1.96·se.

```julia
factor_paths(m::DynamicFactorModel) -> FactorPathResult
```

**Orientation trap:** `res.factors.smoothed` in statsmodels is
**(k_factors, nobs)** with numpy input — transposed relative to the data [V].
Ours returns observations × factors, and a test asserts the orientation
explicitly (11.2 §6.7). Mention it in the docstring, because anyone porting a
script from statsmodels will hit it.

### 3.2 `FactorPathResult` with the series overlaid

`dfms`' `plot.dfm(type="joint")` draws all standardised series in grey with the
factor paths overlaid [V, exercised]. Worth having as a keyword on the same
recipe (`overlay_series=true`) rather than a second type — it is the same data
rendered two ways.

`plot.dfm(type="individual")` is one stacked panel per factor, which is §3.1's
default. So one struct covers all three of `dfms`' modes.

### 3.3 `LoadingsResult` — the loadings heatmap

**Neither reference has one.** [V] It is the natural way to read which series
load on which factor, it is cheap, and with the §3.2 normalisation of the 11.2
handoff (lower-triangular Λ) it is actually interpretable rather than being an
artefact of an arbitrary rotation.

N×r heatmap, diverging colour scale centred at zero, series names on one axis.

### 3.4 `FactorR2Result` — the variance decomposition

Both references have something, and they are **different quantities** (11.2 §7):

- statsmodels `coefficients_of_determination` → per-factor R², N×r, with
  `plot_coefficients_of_determination` (62 lines) [S]
- `dfms` `summary(m)$R2` → total across all factors, N-vector

Export both (`factor_r2`, `series_r2`) and draw the per-factor version as a
grouped bar chart, with the total as an annotation or a second series. Fixture A's
per-series R² is `[0.91086263, 0.67706417, 0.61379053, 0.50857762, 0.82011444]`
[V] — a good doctest figure.

### 3.5 `FactorSelectionResult` — scree and Bai–Ng IC

`dfms` has `screeplot` and `plot.ICr` [V, exercised]. Two panels: eigenvalues of
the correlation matrix, and IC1/IC2/IC3 against `r`.

**With a caveat drawn on the plot, not just in the docs.** Measured on fixture B,
the criteria **do not select the true r = 2** — they keep decreasing through
r = 4 [V]. A plot that invites the reader to take the minimum would be
misleading, so the recipe should mark the true-r case in the doctest and the
docstring must say the criteria are consistent only asymptotically in both N and
T, and N = 6 is not asymptotic.

### 3.6 Residual boxplot by series

`dfms`' `plot.dfm(type="residual")` [V, exercised]. Cheap, useful for spotting a
series the factors do not explain. Fold it into `DiagnosticPlotResult` if that
type can take a matrix; otherwise a small `FactorResidualResult`.

---

## 4. Stage 11.3 — Markov-switching

**statsmodels has no plotting methods at all** — `grep "def plot"
tsa/regime_switching/*.py` returns zero hits [V]. **MSwM has four S4 methods,
all of which ran successfully** [V] `s11/ms/plots_mswm.R`. So the panel designs
come from MSwM and the data from statsmodels, which is a clean division.

### 4.1 `RegimeProbabilityResult` — the two panels worth copying

```julia
regime_probabilities(m::MarkovSwitchingModel) -> RegimeProbabilityResult
```

carrying filtered, smoothed and predicted marginal probabilities (each `nobs × k`),
the fitted values, the series, and the expected durations.

Two renderings, selected by keyword:

| mode | source | draws |
|---|---|---|
| `:probabilities` (default) | `MSwM::plotProb(which=1)` | one panel per regime, with **smoothed and filtered overlaid** — the overlay is the point, since the gap between them is where the smoother adds information |
| `:shaded` | `MSwM::plotProb(which≥2)` | the series with the periods in each regime **shaded** by smoothed probability |

The shaded version is the one that appears in every published
Markov-switching figure, and it is the one a book chapter wants.

**Shape trap:** MSwM's `smoProb` has **`nobs+1` rows** (row 1 is the t=0 smoothed
distribution) while `filtProb` has `nobs` [V]. Ours are both `nobs`. Note it in
the docstring for anyone comparing.

**Structural assertion worth a test:** the last filtered and last smoothed value
must coincide — measured 0.92771603 for both on the canonical fixture [V].

### 4.2 Expected durations are **not** a plot

`[4.07604793, 10.42589261]` on the canonical fit [V]. Two numbers. They go in
`show` and in the summary table, not on a chart. MSwM computes them **not at
all** (`grep -i duration` → 0 hits) [V]; statsmodels exposes them but prints no
duration table. Ours prints them.

### 4.3 Per-regime diagnostics

`MSwM::plotDiag` draws residuals versus fitted, a Normal Q-Q, and ACF/PACF of
residuals **and of squared residuals**, pooled or per regime [V].

`DiagnosticPlotResult` already covers the pooled case. Add a `regime=` keyword to
`diagnostic_plot(m::MarkovSwitchingModel)` that selects the regime and weights
the residuals by the smoothed probabilities. **Do not add a new type** — this is
a dispatch, not a new plot.

### 4.4 `MSwM::plotReg` — skip it

Response and regressors against one regime's probabilities. It is a scatter of
things already visible in §4.1's shaded panel, and it needs the exogenous matrix
plumbed through for little gain. Note the omission in the docs.

---

## 5. Stage 11.4 — Threshold models

Six recipes, and two of them exist in neither reference.

### 5.1 `ThresholdRegimeResult` — the regime-switching plot

`plot.setar`'s central panel: the series over time, points or segments coloured by
regime, with `abline(h = th)` for the threshold [S]. The canonical SETAR figure.

Carries the series, the regime assignment per observation, the threshold(s), and
the fitted values.

**Boundary convention on the plot:** the low regime is `z ≤ γ` inclusive, and on
the lynx fixture **exactly one observation sits on the threshold**
(`regime[63] == 1`) [V]. A plot that drew it ambiguously would be wrong; assert
the colour assignment of that observation in a test.

### 5.2 `ThresholdPhaseResult` — the phase plot

`y_t` against `y_{t−j}`, points coloured by regime, fitted curve overlaid dashed,
"low"/"high" legend [S] `plot.setar`, 75 lines.

**`tsDyn` does not implement this for `nthresh=2`** — it prints a title saying so
[S]. Ours should draw it, with one colour per regime; there is no reason three
regimes cannot be coloured.

For long series use `_decimate_local_max` [S].

`autopairs`/`autotriples` (bivariate and trivariate phase plots with
persp/image/contour modes) [S] are a generalisation of this and are **out of
scope** — they are a nonlinear-dynamics exploration tool, not a model plot.

### 5.3 `ThresholdProfileResult` — the SSR profile

`plot.selectSETAR` draws the criterion against the candidate threshold [S], and
it `stop`s unless `criterion="SSR"` — an arbitrary restriction, since the profile
is meaningful for any criterion. **Ours does not restrict it.**

The grid search already computes this (11.4 §6.2), so storing it costs one
vector. It is also what makes §5.4 possible.

### 5.4 Threshold confidence set — **neither reference has it**

The LR profile with a critical-value cutoff, shading the region of thresholds not
rejected. Hansen's (1997, 2000) confidence set.

**This is gated on the 11.4 §5.2 scope decision.** If the LR-profile confidence
set is built, this plot is nearly free given §5.3's stored profile. If it is not,
**do not draw a half-version** — a shaded region with no stated coverage would be
worse than no plot. The honest-refusal principle applies to charts too.

### 5.5 `TransitionFunctionResult` — **neither reference has it**

`G(z_t; γ, γ_th)` against `z` (and optionally against `t`). `plot.lstar` only
colours points by `G` quantile [S]; plotting `G` itself is the direct way to see
how abrupt the transition is.

Measured on the lynx fixture, `G` ranges over `[3.40372245713e-09,
0.996447296207]` with mean 0.273310796644 [V] — so the curve very nearly
saturates at both ends, which is exactly the thing the plot should make visible
and which motivates the γ→∞ reduction test (11.4 §7.7.1).

**Annotate the plot when γ is at the start-grid bound.** γ pinned at 100 means
the transition is effectively abrupt (11.4 §6.3, measured on two of three
fixtures), and a reader looking at a step function should be told the parameter
is at a boundary rather than estimated.

### 5.6 `ThresholdTestResult` — the bootstrap density

`plot.Hansen99Test` draws the bootstrap density of each F statistic, the observed
statistic as a dashed red vertical, and a χ² reference curve **for y-axis scaling
only** [S], 73 lines.

**Do not draw the χ² curve.** It is not the null distribution — the test has no
asymptotic p-value, it is bootstrap-only (11.4 §5.4) — and a reference curve on
the plot invites exactly the misreading that `tsDyn`'s own code comment guards
against. Draw the bootstrap density, the observed statistic, and the
critical-value verticals at 0.90/0.95/0.975/0.99.

Measured on lynx with `nboot=200, seed=1234`: observed `1vs2 = 36.94677181537`
against critical values `15.0806675937 / 16.7602990397 / 18.4110807325 /
18.8319866334` [V] — a figure worth using in the doctest, with the caveat that
the critical values are seed-dependent and not a portable expectation.

### 5.7 `plot.nlar`'s eight-panel display — already covered

Series, residuals, ACF and PACF of each, plus average mutual information of each
[S]. `TSDisplayResult` and `DiagnosticPlotResult` between them already draw six
of the eight; average mutual information does not exist in the package and is a
nonlinear-dependence diagnostic in its own right.

**Dispatch `ts_display` and `diagnostic_plot` on the threshold model types and
stop there.** Do not reproduce the eight-panel layout; note the AMI omission.

---

## 6. Summary of new types and recipes

| # | Type | Item | File | Reference analogue |
|---|---|---|---|---|
| 1 | *(extend)* `StateDecomposition` with optional `se` | 11.1 | `plotdata.jl` | `plot_components`, 216 lines |
| 2 | `FactorPathResult` | 11.2 | `plotdata.jl` | **bands: neither reference** |
| 3 | `LoadingsResult` | 11.2 | `plotdata.jl` | **neither reference** |
| 4 | `FactorR2Result` | 11.2 | `plotdata.jl` | both, different quantities |
| 5 | `FactorSelectionResult` | 11.2 | `plotdata.jl` | `screeplot`, `plot.ICr` |
| 6 | `FactorResidualResult` | 11.2 | `plotdata.jl` | `plot.dfm(type="residual")` |
| 7 | `RegimeProbabilityResult` | 11.3 | `plotdata.jl` | `MSwM::plotProb` |
| 8 | `ThresholdRegimeResult` | 11.4 | `plotdata.jl` | `plot.setar` |
| 9 | `ThresholdPhaseResult` | 11.4 | `plotdata.jl` | `plot.setar` phase panel |
| 10 | `ThresholdProfileResult` | 11.4 | `plotdata.jl` | `plot.selectSETAR` |
| 11 | `TransitionFunctionResult` | 11.4 | `plotdata.jl` | **neither reference** |
| 12 | `ThresholdTestResult` | 11.4 | `plotdata.jl` | `plot.Hansen99Test` |
| 13 | *(gated)* threshold confidence set | 11.4 | — | **neither reference**; see §5.4 |

**Dispatches added to existing types, no new struct:**
`state_decomposition(::UCModel)`, `diagnostic_plot(::UCModel)`,
`diagnostic_plot(::MarkovSwitchingModel; regime)`,
`diagnostic_plot(::SETARModel)`, `ts_display` on the new model types,
`forecast`/`predict` recipes via the existing `Forecast`.

**Four of the thirteen do not exist in either R or Python**: factor paths with
bands, the loadings heatmap, the transition function, and the threshold
confidence set. That is the honest answer to "all functionalities from related R
and Python packages including charts and graphs" — we match them and then exceed
them in four places, each for a stated reason.

### 6.4 The one refusal

**No recipe for the pairwise regime-transition probabilities** from 11.3's Kim
smoother. They are a `nobs × k × k` array and there is no single natural
rendering — a heatmap per period, a small-multiple of `k²` time series, or a
transition-count summary are all defensible and none is obviously right. Follow
`seasonal_subseries_plot`'s precedent: return the data, document the choice, and
let the caller pick. [S] for the precedent

---

## 7. Testing the recipes

The existing plotting work established how, and it does not change: **test the
data, not the rendering.**

| what | how |
|---|---|
| the extraction functions | assert the returned struct's fields against the fixture figures in the sub-stage handoffs |
| panel count and ordering | `length(d.names)` and its contents, per specification |
| band arithmetic | ±1.96·se computed from the smoother's `V`, asserted against a hand computation at three periods |
| orientation | §3.1 — observations × factors, asserted explicitly |
| regime assignment | §5.1 — including the boundary observation at `regime[63]` |
| **no regression in the existing recipes** | `StateDecomposition` with `se = nothing` renders identically for ETS and TBATS — compare the recipe's emitted series to a stored reference |
| `RecipesBase` stays soft | a test that the package loads and `state_decomposition` works with **no** plotting backend present |

**Do not** test image output. Nothing in the existing suite does, and the
reference plot methods were themselves only read, not rendered — `tsDyn`'s
appearance is [S], not [V], and the handoff says so.

**One rendering check is worth having**, once, manually: build each of the
thirteen with a backend loaded and look at them before the item is called done.
Record in `development-sequence.md` that this was done, since it cannot be
automated.

---

## 8. Documentation obligations

Each new recipe needs an `@example` block in the API reference and at least one
figure in the manual, per the Stage 9 and 10 documentation pattern.

**And the Documenter trap from the Stage 9/10 documentation work applies
directly:** a block that prints and then plots renders **only the plot** — the
printed output is silently discarded. The fix is to split the block, keeping the
plotting in a following block within the same named `@example` session so no
state is lost. The scan after that fix found **236 of 838 blocks ending in a
plot, none containing a print statement** [V, from the Stage 9/10 audit].

**Stage 11 adds thirteen plots and will add dozens of `@example` blocks. Every
one that prints a number and then draws a chart must be split at authoring
time**, not fixed in a later sweep. Put this in the documentation handoff as a
checklist item, and consider a CI grep: any `@example` block containing both a
`print`/`println` and a trailing plot call is a build failure waiting to happen.

---

## 9. Budget

| Item | Types | Recipes | Estimated lines |
|---|---|---|---|
| 11.1 | 1 extension | 1 modified | 80–120 |
| 11.2 | 5 | 5 | 320–420 |
| 11.3 | 1 | 1 (two modes) + 1 dispatch | 150–200 |
| 11.4 | 5 (+1 gated) | 5 (+1 gated) | 350–450 |
| **total** | **12 (+1)** | **12 (+1)** | **900–1190** |

Comparable to the reference plotting code it replaces — `tsDyn`'s plot methods
total 240 lines across its classes, MSwM's 240, statsmodels'
`plot_components` + `plot_coefficients_of_determination` 278, `dfms`'
`plot.dfm` + screeplots ~160 — against our ~1000 for a superset with four plots
none of them has.
