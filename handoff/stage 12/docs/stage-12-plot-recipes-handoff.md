# Stage 12 — Plot recipes

**Prerequisite reading:** `stage-12-overall-handoff.md`, and the existing plot-recipes handoffs from Stages 9–11, whose conventions this extends rather than restates.

**Twenty-nine recipes exist** in `src/recipes.jl` at HEAD `2eccb82` [V]. Stage 12 adds **eight**, and five of the eight exist in **no reference**.

---

## 1. The established pattern — unchanged

1. **A result type in `src/plotdata.jl`, a recipe in `src/recipes.jl`.** Data extraction and rendering stay separate, and the data is useful alone.
2. **`RecipesBase` stays a soft dependency** — see the header comment in `src/recipes.jl` [S].
3. **Never a recipe on a bare `NamedTuple`** — it would hijack plotting for any unrelated NamedTuple in a session with TSAnalytics loaded. Every plottable output gets a named struct.
4. **The refusal precedent** — `seasonal_subseries_plot` returns data with no recipe because there is "no one natural rendering". Stage 12 has one such case (§5).
5. **Reuse `_decimate_local_max`** for long series; do not write a second one.

---

## 2. Stage 12.1 — intermittent demand

**Neither reference has a single intermittent-specific plot.** Exhaustive check: no `plot.croston`, no `autoplot.croston` in `forecast`'s 360 namespace objects; nothing in `statsforecast`. [V]

And the one thing that *does* render is rendering it wrongly. `autoplot` on a croston object completes without error — the rendered PNG was inspected — and draws a **connected line plot**: sloped segments between zeros that imply fractional demand on days with none. [V]

### 2.1 `DemandIntervals` → the decomposition panel

```julia
demand_intervals(m::IntermittentModel) -> DemandIntervals
```

carrying the non-zero demands, the inter-arrival intervals, their smoothed levels, and the positions. Three stacked panels: the series as a **stem/spike plot**, the demand sequence with its level, the interval sequence with its level.

**The stem plot is the point.** A spike mark is the correct default for a mostly-zero series and neither reference provides one. The components *are* reachable in R (`f$model$demand$x`, `$period$x`, 22 each on F1) — nothing draws them. [V]

### 2.2 `ADICV2Result` → the classification quadrant

Average demand interval on one axis, squared coefficient of variation on the other, with the **1.32** and **0.49** cutoffs drawn as a crosshair and the series plotted as a point.

**Neither target reference implements the classifier at all** (charter §5; zero hits for `1.32`, `0.49`, `lumpy`, `erratic`, `ADI`, `cv2`). The GPL `tsintermittent` draws the quadrant (`idclass(outplot="detail")`, with `polygon(c(1,1.32,1.32,1), c(0,0,0.49,0.49))` at line 188) — **pattern reference only, nothing transcribed.**

**Two honesty requirements on this chart.** `tsintermittent` implements a **binary** SBA-versus-Croston decision; the four-quadrant labels *smooth / intermittent / erratic / lumpy* appear nowhere in its source. If we draw them, **the naming is ours** and the docstring says so. And the chart should mark where the fixture sits: F3 lands at ADI 1.3187, just inside the boundary.

### 2.3 Inter-arrival histogram

A histogram of the intervals, with the fitted level marked. Trivial given `DemandIntervals`, and absent from both references.

**Fold it into `DemandIntervals`' recipe as a keyword** rather than adding a type — it is the same data rendered differently.

---

## 3. Stage 12.2 — change-point and breaks

### 3.1 What the references draw [S]

| reference | method | draws |
|---|---|---|
| `changepoint` | `plot.cpt` | the series with vertical changepoint lines |
| | `plot.cpt.range` | the CROPS path; `diagnostic=TRUE` gives breaks-versus-penalty |
| `strucchange` | `plot.efp` (96 lines) | the fluctuation process with its boundary at level α |
| | `plot.Fstats` (32) | the F-statistic path against time, with a boundary; `pval=TRUE` switches to p-values |
| | `plot.breakpointsfull` → `plot.summary.breakpointsfull` | **not the series** — a dual-axis BIC-and-RSS-versus-m curve |
| | `lines.breakpoints`, `lines.confint.breakpoints` | overlays onto an existing plot |
| `ruptures` | `display` (102 lines) | one panel per feature, regimes as alternating `axvspan` shading |

### 3.2 `ChangePointResult` → the segmented series — **a composite none of them has**

One panel: the series, the **fitted segment means as a step function**, the breakpoints as verticals, and — where available — their **confidence intervals as horizontal bands at the break**.

`strucchange` has the CI data and the `lines` primitive but **no composite plot**; `changepoint` draws breaks without segment means; `ruptures` shades regimes but has nothing on the inference side at all. [V/S]

This is the chart a reader actually wants and it exists nowhere. It is also cheap: `break_confint` already returns the bounds as exact integers.

### 3.3 `BreakpointsResult` → the selection curve

BIC and RSS against the number of breaks, matching `plot.summary.breakpointsfull`. On `fx0` the BIC minimum at m = 2 is **838.563795** [V] — a good doctest figure, and the plot that justifies the selected m.

### 3.4 `FStatsResult` → the statistic path

The F-statistic against candidate break date, with the trimming bounds marked.

**Draw no boundary line unless we have a defensible critical value.** Charter §6 defers the supF p-values; a boundary drawn from a simulated critical value must be **labelled as simulated**, with the replication count in the legend. `plot.Fstats`' own boundary comes from the GPL tables.

### 3.5 `EFPResult` → the fluctuation process

The process with its boundary — and here the boundary **is** defensible, because the CUSUM family's p-values are closed-form (charter §6).

**Worth drawing well, because it carries the item's most instructive result**: on the variance-change fixture, Score-CUSUM detects what OLS-CUSUM misses (p = **0.00038111** against **0.503186**) [V]. Two processes, one chart, visibly different.

### 3.6 The penalty path

`changepoint`'s `cpt.range` CROPS plot is the only elbow plot in any reference. Worth having — number of breakpoints against penalty — as a keyword on `ChangePointResult` rather than a type.

---

## 4. Stage 12.3 — lag-feature regression

**The spread here is extreme**: `forecastML` is **41% ggplot2** (2163 lines across 8 methods), `mlforecast` ships **zero** plotting, `sktime` has `plot_lags`, `plot_windows`, `plot_series`. [V]

**Nobody plots feature importance** — grepped both Python references for `feature_importance` co-occurring with `def plot`: zero hits. And it would be pointless for us: OLS has no importances.

Two recipes, both for things `forecastML` and `sktime` *both* bother to plot:

### 4.1 `LagFeatureMap` → which lags feed which horizon

A grid of horizon against lag, marked where the feature is used. For the recursive strategy it is a single row; for `direct` it shows the anchoring — which is exactly the §2.2 decision in the 12.3 handoff made visible, and the clearest way to explain origin-anchored versus target-anchored to a reader.

### 4.2 `CVWindowLayout` → the fold structure

Train and test blocks per fold, as `plot.windows` and `plot_windows` both draw. Covers `expanding_window_split`, `sliding_window_split`, and the new blocked CV — and the blocked case is the one worth seeing, since training happens on **both sides** of the held-out block.

**Nothing else from 12.3.** No predicted-versus-actual faceted by horizon (that is `forecastML`'s single largest function at 545 lines, and `plot(::Forecast)` already covers the useful part), no hyperparameter paths, no importances.

---

## 5. The one refusal

**No recipe for the full penalty-versus-segmentation surface** from CROPS. It is a set of segmentations indexed by penalty, and there is no single natural rendering — a breakpoint-count elbow, a per-penalty segment plot, or a heatmap of breakpoint positions against penalty are all defensible and none is obviously right. §3.6 provides the elbow as a keyword; the rest returns as data.

Follow `seasonal_subseries_plot`'s precedent: return it, document the choice, let the caller pick. [S] for the precedent.

---

## 6. Summary

| # | Type | Item | Reference analogue |
|---|---|---|---|
| 1 | `DemandIntervals` (+ histogram keyword) | 12.1 | **neither reference** |
| 2 | `ADICV2Result` | 12.1 | **neither target reference** (GPL `tsintermittent` only) |
| 3 | `ChangePointResult` (+ penalty-path keyword) | 12.2 | partial — **the composite exists nowhere** |
| 4 | `BreakpointsResult` | 12.2 | `plot.summary.breakpointsfull` |
| 5 | `FStatsResult` | 12.2 | `plot.Fstats` |
| 6 | `EFPResult` | 12.2 | `plot.efp` |
| 7 | `LagFeatureMap` | 12.3 | `plot.lagged_df` / `plot_lags` |
| 8 | `CVWindowLayout` | 12.3 | `plot.windows` / `plot_windows` |

**Five of the eight have no equivalent in any reference**: the stem-plot decomposition, the ADI/CV² quadrant, the inter-arrival histogram, the composite segmented series with confidence bands, and the penalty path outside `changepoint`.

**Dispatches, no new struct:** `diagnostic_plot(::IntermittentModel)`, `diagnostic_plot(::BreakpointsResult)`, `plot(::IntermittentForecast)` via the stem mark.

---

## 7. Testing

Unchanged: **test the data, not the rendering.**

| what | how |
|---|---|
| extraction functions | assert fields against the sub-stage handoff figures |
| the stem mark | assert the recipe emits a spike/stem series type, not a line — this is the one *rendering* assertion worth making, because the wrong default is what both references do |
| ADI/CV² placement | F3 at (1.3187, 0.2545), inside the boundary |
| segment means | step function matches `ChangePointResult`'s segment parameters |
| confidence bands | the exact integers from `break_confint` |
| boundary labelling | a simulated critical value is labelled as such |
| no regression | the 29 existing recipes render unchanged |
| `RecipesBase` stays soft | the package loads and every extraction function works with no backend |

**One manual rendering pass** before the item is called done, recorded in `development-sequence.md` since it cannot be automated. The intermittent stem plot in particular needs eyes on it — the failure mode is a chart that looks fine and means something false.

---

## 8. Documentation obligation

Every new recipe needs an `@example` block in the API reference and at least one manual figure.

**The `@example` guard now exists and will catch the trap** — `docs/check_example_blocks.jl`, wired into `make.jl:8`, parses each block with Julia's own parser and fails the build when a block prints and then ends in a displayed value [V]. It is stricter and more correct than the grep the Stage 11 handoff asked for. Stage 12 inherits the protection; nothing further is needed beyond writing blocks that pass it.

---

## 9. Budget

| Item | Types | Recipes | Lines |
|---|---|---|---|
| 12.1 | 2 | 2 (+1 keyword) | 180–240 |
| 12.2 | 4 | 4 (+1 keyword) | 300–400 |
| 12.3 | 2 | 2 | 140–190 |
| **total** | **8** | **8** | **620–830** |

Against the references: `forecastML` 2163 lines, `strucchange`'s plot methods ~160, `changepoint`'s plot code inside a 1204-line class file, `ruptures` 102, `mlforecast` **0**.
