# Handoff: plots for Stages 9 and 10

**Status: superseded (2026-10-14)** by `handoff/stage 10/plot-recipes-handoff.md`, which the build was reconciled to; kept for the record.

Six charts for the exponential-smoothing (Stage 9) and multivariate
(Stage 10) results, which today have no figures at all except 10.3's IRF
grid and FEVD shares. Requested by the user on 2026-10-14 after a scoping
answer; the scope below is what was proposed and approved.

## 1. Architecture — follow what exists

`src/recipes.jl`'s header states the policy and this item keeps it:
`RecipesBase` recipes only, **no plotting dependency**, so `plot(x)` works
for any caller with a Plots.jl backend loaded. No new `[deps]`.

Where a chart needs data the result object does not hold, follow
`diagnostic_plot`'s existing pattern: a small exported function builds a
plain result struct, and the recipe draws that struct. The struct is the
tested artefact; the recipe is thin.

- `Forecast` and `VARForecast` carry **no history** — a fan chart needs
  it — so `forecast_plot(y, f)` takes the series.
- `ETSModel` carries its state paths but **not the data**; components
  plots take `y` too, for the observed panel.
- `TBATSModel` stores only `seed_states`; the state path is recomputed by
  the fitted recursion (refactor `_tbats_final_state` so it and the new
  path function cannot drift apart). Its components are on the **Box-Cox
  scale** when `lambda` is set — say so in the panel title.
- `ThetaModel` stores no states (it is SES on a deseasonalized series
  plus a drift): no components plot; its fan chart is covered by
  `Forecast`.

## 2. The six charts

| # | Function → result | Recipe draws |
|---|---|---|
| 1 | `forecast_plot(y, f::Forecast; history)` → `ForecastPlotResult` | history line, then one shaded band per level (widest lightest) and the point path, on a continuous time axis `1:n`, `n+1:n+h` |
| 2 | `forecast_plot(Y, f::VARForecast; history)` → `VARForecastPlotResult` | the same, one panel per variable |
| 3 | `components_plot(y, m)` for `ETSModel`, `ExponentialSmoothingModel`, `TBATSModel` → `ComponentsPlotResult` | stacked panels: observed, level, slope/trend (if any), seasonal (if any; TBATS one panel per period) |
| 4 | `roots_plot(m)` for `VARModel`, `VECMModel` → `RootsPlotResult` | companion eigenvalues in the complex plane, unit circle, equal aspect |
| 5 | `diagnostic_plot(m::VARModel; lags)` → `VARDiagnosticPlotResult` | `K × 2` grid: each equation's residuals over time, and their ACF with ±1.96/√T bands |
| 6 | `cointegration_plot(m::VECMModel)`, `cointegration_plot(Y, t::JohansenTest; rank)` → `CointegrationPlotResult` | one panel per relation: `β'Ỹ_t` over time with its mean — stationary when the relation is real |

Deliberately not plotted: test statistics (Granger, Johansen trace,
Phillips-Ouliaris) — a bar against a critical value adds nothing to the
printed table.

## 3. Definitions worth pinning

- **History window.** `history = nothing` shows all of `y`; an integer
  shows the last `history` observations. Time indices stay the series'
  own positions, so a forecast always starts at `n + 1`.
- **Bands.** One per level in `f.levels`; drawn widest first so narrower
  bands sit on top.
- **ETS seasonal state.** `seasonal_component` is the seasonal state path
  in the model's own scale — additive (deviation) or multiplicative
  (ratio); the panel title says which.
- **Roots.** Complex eigenvalues of the companion matrix, not the moduli
  `roots` returns; the moduli are `abs.` of what is plotted. For a VECM,
  the levels VAR's companion (`var_rep`), so the `K − r` unit roots sit
  on the circle by construction.
- **Cointegrating relation.** `β'Ỹ_{t}` with `β` identity-normalized and
  `Ỹ` augmented by the restricted deterministic term when there is one —
  exactly the error-correction term the VECM regresses on.
- **VAR ACF bands** ±1.96/√T, matching `acf`'s own convention.

## 4. Verification

No backend is needed to test a recipe (`RecipesBase.apply_recipe`), as
`test/test_recipes.jl` already does. For each chart:

1. the result struct's data are right — reconstructed independently in
   the test (e.g. ETS components reproduce `fitted`; TBATS state path's
   last row equals `_tbats_final_state`; the cointegrating relation equals
   the VECM's own error-correction regressor; roots' moduli equal
   `roots(m)`);
2. the recipe emits the expected number of series per panel;
3. refusals: mismatched `y`/model length, `history` < 1, a `Forecast`
   with no levels, etc.

The docs build renders every chart through Plots/GR (already in
`docs/Project.toml`), in the manual chapters that use them — 10, 11, 12
and 13 (plotting) — so a broken recipe fails the docs gate.

## 5. Checklist

1. `src/plotdata.jl` — the builders and result structs (after `svar.jl`).
2. Recipes in `src/recipes.jl`, house style.
3. `test/test_plots_stage9_10.jl`.
4. Docs: API entries; figures in manual 10, 11, 12; a section each in 13.
5. Roadmap row; full suite and docs build; commit.
