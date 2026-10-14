# Handoff: Plot recipes for Stage 9 and Stage 10

**Status: implemented ✅ (2026-10-14)** — see `development-sequence.md`'s "Plot recipes" section for the deviations (TBATS panels, FEVD stacking, `history` handling, `cointegration_plot`). Cross-cutting, not a numbered stage — like the
textbook-dataset catalogue and the documentation restructure.

For a fresh Claude Code session with no prior context. Read
`stage-10-overall-handoff.md` §7 for the stage-wide performance policy;
everything else needed is here.

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every reference claim below is `[V]` or `[S]` against the installed
packages (`forecast` 8.21.1, `vars` 1.6.1, `urca` 1.3.3, `tsDyn`
11.0.5.2 on R 4.3.3; `statsmodels` 0.15.0). **Julia was not installed
in the sandbox that produced this document** — re-verify any
package-internal claim before building on it.

---

## 1. Why this document exists

Two reasons, and the second is the bigger one.

**The package has nine chart-producing model families and seven
recipes.** `[S]` `src/recipes.jl` covers `ACFResult`,
`PeriodogramResult`, `ClassicalDecomposition`, `STLDecomposition`,
`MSTLDecomposition`, `DiagnosticPlotResult`, `GarchDiagnosticPlotResult`
— and `[V]` grepping `recipes.jl` for each remaining plottable type
returns **zero** mentions of `Forecast`, `ETSModel`, `TBATSModel`,
`ThetaModel`, `VolatilityForecast` and `ArimaModel`. **There is no
forecast plot at all**, which is the single most-used chart in the whole
subject.

**The book requires charts throughout.** The repeated instruction across
all 41 chapter handoffs is *"add charts and graphs as much as
possible"*. Satisfying that chapter by chapter produces ad-hoc plotting
code in `@example` blocks; satisfying it with recipes means
`plot(result)` works in the manual, in the book, and for any user, from
one implementation.

- **Depends on:** Stage 9 (built), Stage 10.1/10.3/10.8 for the
  multivariate half. The Stage 9 recipes can be built **today**.
- **Feeds into:** every Introduction chapter and every manual chapter.

---

## 2. The four patterns already established in this package

`[S]` This is the spine of the document. The package has already made
this decision four different ways, each deliberately, and new plots must
pick one rather than invent a fifth.

### Pattern A — a recipe directly on an existing result type

When there is **one** natural rendering. `[S]` `ACFResult` (one recipe
serving `acf`, `pacf` **and** `ccf` through its `kind` field — R needs
`ggAcf`/`ggPacf`/`ggCcf`, three functions, for the same job) and
`PeriodogramResult`.

### Pattern B — a multi-panel recipe via `layout`

When the panels are **already fields** of the result. `[S]`
`STLDecomposition`:

```julia
@recipe function f(d::STLDecomposition)
    layout --> (4, 1)
    legend --> false
    link --> :x
    t = 1:length(d.observed)
    @series begin
        subplot := 1
        ylabel := "observed"
        t, d.observed
    end
    # ... trend, seasonal, remainder
end
```

Note `link --> :x` — shared x-axis across stacked panels. Every new
stacked-panel recipe uses it.

### Pattern C — a computing function, a panel struct, then a recipe

When the panels need **computation** that should be testable without a
plotting backend. `[S]` `diagnostic_plot(resid; …) -> DiagnosticPlotResult`
holds `std_resid`, `acf_lags`, `acf`, `qq_theoretical`, `qq_sample`,
`lb_lags`, `ljungbox_pvalues`, `nlag`, `ppq` — and the recipe renders
them at `layout --> (2, 2)`. `[S]` `test/test_diagnosticplot.jl` is 143
lines testing the *numbers*; `test/test_recipes.jl` then only has to
check that the recipe emits six series.

**This is the pattern for anything where the panel content is derived,
not stored** — and most of what follows is.

### Pattern D — a data-returning function and *no* recipe

When there is **no single natural rendering**, the package deliberately
refuses to encode one. `[S]` `seasonal_subseries_plot` returns a plain
NamedTuple, and its docstring says why in as many words:

> deliberately a plain NamedTuple, not a dedicated result type/recipe:
> unlike `periodogram` (a single fixed display, `freq` vs `spec`), this
> one has no one natural rendering (panels? overlaid lines? which axis
> is which?), so a recipe would just be encoding one opinionated choice
> among several equally valid ones.

**Respect this.** §4.6 proposes two more Pattern-D functions and
explicitly does *not* propose recipes for them.

### The soft-dependency rule

`[S]` `src/recipes.jl`'s header: `RecipesBase` only, never `Plots.jl` or
`Makie.jl`. Nothing below changes that, and no new dependency is needed
for any of it.

---

## 3. What the references draw, verified

The inventory, with the panel structures read from source rather than
assumed. Corrections to what I previously reported are marked.

### 3.1 Stage 9

| Plot | R | Python |
|---|---|---|
| ETS state decomposition | `plot.ets` / `autoplot.ets` | **none** |
| BATS/TBATS state decomposition | `plot.bats` / `plot.tbats` | **none** |
| Forecast fan chart | `plot.forecast` / `autoplot.forecast` | `ThetaModelResults.plot_predict`; `tsaplots.plot_predict` |
| Residual check panel | `checkresiduals` | `plot_diagnostics` — state-space results **only**, not ETS |
| Series + ACF/PACF | `ggtsdisplay` | — |
| Seasonal / subseries | `ggseasonplot`, `ggsubseriesplot`, `ggmonthplot`, `monthplot` | `month_plot`, `quarter_plot`, `seasonal_plot`, `seasonal_diagnostic_plot` |
| Lag plot | `gglagplot`, `gglagchull` | — |
| ACF/PACF/CCF | `ggAcf`, `ggPacf`, `ggCcf`, `ggtaperedacf`, `ggtaperedpacf` | `plot_acf`, `plot_pacf`, `plot_ccf`, `plot_pccf`, `plot_accf_grid` |
| Inverse AR/MA roots | `plot.Arima` / `autoplot.Arima`, `plot.ar` | — |

`[V]` `ETSResults`' MRO is `ETSResultsWrapper → ResultsWrapper →
object`, so it never inherits `plot_diagnostics`; `dir()` finds **no**
method containing "plot" on either `ETSResults` or
`HoltWintersResults`. `[V]` And `plot_predict` needs a pandas index — on
a plain ndarray it raises `AttributeError: 'numpy.ndarray' object has no
attribute 'index'`.

**A correction to what I told you earlier.** I said `plot.ets` draws one
panel per state — 14 for a monthly ETS(A,A,A). `[S]` Reading
`forecast:::plot.ets`, it draws **at most four**:

```r
plot(cbind(observed = y,
           level  = x$states[, 1],
           slope  = x$states[, "b"],
           season = x$states[, "s1"]),      # s1 ONLY, not s1..s12
     main = paste("Decomposition by", x$method, "method"))
```

| ETS form | panels |
|---|---|
| (·,N,N) | observed, level → **2** |
| (·,A,N) or (·,Ad,N) | observed, level, slope → **3** |
| (·,N,A) | observed, level, season → **3** |
| (·,A,A) or (·,Ad,A) | observed, level, slope, season → **4** |

The object has 14 state columns; the plot shows the *current* seasonal
state `s1` as one series. Match that.

`[S]` `plot.bats`/`plot.tbats` generalize it: `observed`, `level`,
optional `slope`, then **one `season` panel per seasonal period**, taking
the first harmonic state of each period block. Title *"Decomposition by
BATS model"*. So TBATS on periods (7, 365.25) gives 5 panels.

`[S]` `checkresiduals`' Ljung-Box lag rule, worth matching because it is
not obvious:

```r
lag <- ifelse(freq > 1, 2 * freq, 10)
lag <- min(lag, round(length(residuals) / 5))
lag <- max(df + 3, lag)
```

### 3.2 Stage 10

| Plot | R (`vars` / `urca` / `tsDyn`) | Python |
|---|---|---|
| Fitted VAR panel | `plot.varest` — per equation: fit+observed, residuals, resid ACF, resid PACF | `VARResults.plot` (data only) |
| Residual check | `plot.varcheck` — per equation: residuals, histogram+density, ACF, PACF | — |
| Residual autocorrelation grid | inside the above | `plot_acorr(nlags, resid, linewidth)` — K×K grid |
| Sample autocorrelation grid | — | `plot_sample_acorr` |
| **IRF** | `plot.varirf` — `[V]` one page per impulse × K response panels, bootstrap percentile band | `IRAnalysis.plot(...)` — **five** band types: `asym`, `mc`, `sz1`, `sz2`, `sz3` |
| Cumulative IRF | `irf(cumulative=TRUE)` → `plot.varirf` | `IRAnalysis.plot_cum_effects` |
| **FEVD** | `plot.varfevd` — stacked bars per variable | `FEVD.plot(periods, figsize)` |
| Forecast | `plot.varprd` | `VARResults.plot_forecast(steps, alpha, plot_stderr)` |
| **Fan chart** | `vars::fanchart` | — |
| **Stability / CUSUM** | `plot.varstabil` — `[V]` an OLS-CUSUM `efp` object per equation | — |
| Simulated paths | — | `VARResults.plotsim` |
| VECM | `plot.vec2var` | `VECMResults.plot_data`, `plot_forecast` |
| SVAR | via `irf()`/`fevd()` → `varirf`/`varfevd` | `SVARResults` inherits the VARResults set |
| **Cointegration** | `urca` S4 `plot` for `ca.jo`, `ca.po`, `ur.df`, `ur.pp`, `ur.kpss`, `ur.ers`, `ur.sp`, `ur.za`; `plotres` | **none** |
| Threshold / nonlinear | `tsDyn`: `plot_ECT`, `plot.TVAR`, `plot.GIRF_df` | — |

Two asymmetries to exploit: **statsmodels' IRF plot is the better one**
(five band types against one), and **cointegration plotting is R-only**.

---

## 4. Stage 9 recipes — buildable today

### 4.1 `Forecast` — the missing centrepiece (Pattern A)

`[S]` `src/forecast.jl:15`:

```julia
struct Forecast
    point::Vector{Float64}
    se::Vector{Float64}
    levels::Vector{Float64}
    lower::Matrix{Float64}      # horizon × levels
    upper::Matrix{Float64}
    horizon::Int
    model_name::String
end
```

Every field the plot needs is already there. The fan chart is nested
ribbons, widest level outermost, so the shading reads as a gradient the
way `plot.forecast` and `fanchart` both do.

```julia
"""
    @recipe f(f::Forecast)

Fan chart of a [`Forecast`](@ref): the point forecast as a line with one
shaded ribbon per prediction-interval level, widest outermost. Matches
the standard display of R's `forecast:::plot.forecast` and
`vars::fanchart`, which Python has no model-level equivalent of (its
`plot_predict` requires a pandas index and draws a single band).

Pass `history` to draw the in-sample series to the left of the forecast
origin, which is what makes a fan chart readable:

    plot(fc; history = y)
"""
@recipe function f(fc::Forecast; history = nothing)
    legend --> :topleft
    xlabel --> "Horizon"
    ylabel --> ""
    title  --> fc.model_name

    n0 = history === nothing ? 0 : length(tsvalues(history))
    hx = (n0 + 1):(n0 + fc.horizon)

    # widest level first so narrower ribbons draw on top
    order = sortperm(fc.levels; rev = true)
    for (k, j) in enumerate(order)
        @series begin
            seriestype  := :line
            fillrange   := fc.upper[:, j]
            fillalpha   := 0.12 + 0.10 * (k - 1)
            linealpha   := 0
            color       := :steelblue
            label       := string(Int(round(fc.levels[j])), "%")
            hx, fc.lower[:, j]
        end
    end
    if history !== nothing
        @series begin
            seriestype := :line
            color      := :black
            label      := "observed"
            1:n0, tsvalues(history)
        end
    end
    seriestype --> :line
    color      --> :steelblue
    linewidth  --> 2
    label      --> "forecast"
    hx, fc.point
end
```

`[V]` Series count for the test: `length(levels) + (history === nothing ? 0 : 1) + 1`.
With the default two levels and no history, **3**.

This one recipe covers **every** forecasting model in the package —
`[S]` `Forecast` is what `fit_arima`, `fit_sarima`, `fit_ets`,
`fit_theta`, `fit_tbats`, `holt_winters`, `naive`, `seasonal_naive`,
`drift`, `mean_forecast` and the benchmark forecasts all return. Build
it first; it is the largest gap closed by the least code.

### 4.2 `VolatilityForecast` (Pattern A)

The same shape for GARCH: the conditional-variance (or σ) path with its
interval. Keep the axis label as σ rather than σ² unless the field says
otherwise — check `src/garchforecast.jl` for which the struct stores and
label it honestly.

### 4.3 ETS and TBATS state decomposition (Pattern C)

`[S]` This is the signature Stage 9 chart and it is **R-only**. The
panels are derived (the states are inside the fitted model, not exposed
as plot-ready fields), so Pattern C applies: a function computes the
panel data, a struct holds it, a recipe renders it.

```julia
export state_decomposition, StateDecomposition

struct StateDecomposition
    observed::Vector{Float64}
    level::Vector{Float64}
    slope::Union{Nothing,Vector{Float64}}
    seasonal::Vector{Vector{Float64}}     # one per seasonal period
    periods::Vector{Int}
    model_name::String
end

state_decomposition(m::ETSModel)   -> StateDecomposition
state_decomposition(m::TBATSModel) -> StateDecomposition
```

Panel count, matching §3.1 exactly: `2 + (slope !== nothing) + length(seasonal)`.

```julia
@recipe function f(d::StateDecomposition)
    npan = 2 + (d.slope !== nothing) + length(d.seasonal)
    layout --> (npan, 1)
    legend --> false
    link   --> :x
    title  --> "Decomposition by " * d.model_name
    t = 1:length(d.observed)
    k = 1
    @series begin; subplot := k; ylabel := "observed"; t, d.observed; end; k += 1
    @series begin; subplot := k; ylabel := "level";    t, d.level;    end; k += 1
    if d.slope !== nothing
        @series begin; subplot := k; ylabel := "slope"; t, d.slope; end; k += 1
    end
    for (i, s) in enumerate(d.seasonal)
        lab = length(d.seasonal) == 1 ? "season" : "season $(d.periods[i])"
        @series begin; subplot := k; ylabel := lab; t, s; end; k += 1
    end
end
```

**Match R's choice of seasonal series: the current seasonal state
(`s1`), one line per period — not all `m` seasonal states.** `[S]` §3.1
has the source. An implementation that plots all twelve monthly states
produces a plot nobody else draws and that no reference can check.

### 4.4 Inverse AR/MA roots (Pattern A)

`[S]` `plot.Arima`/`autoplot.Arima` draw the **inverse** roots of the AR
and MA polynomials inside the unit circle — inside means stationary and
invertible, which is the whole point of the display. Python has no
equivalent.

```julia
export arma_roots, ARMARootsResult

struct ARMARootsResult
    ar_inverse::Vector{ComplexF64}
    ma_inverse::Vector{ComplexF64}
    model_name::String
end
arma_roots(m)  # ArimaModel, SarimaModel, ArmaModel, ArimaxModel, SarimaxModel
```

The recipe draws the unit circle, then the AR and MA inverse roots as
two scatter series. Series count **3** (circle + AR + MA), or 2 when one
polynomial is empty — make the empty case emit no series rather than an
empty one, and test both.

`[S]` The package already has the stationarity/invertibility machinery
(`_ar_stationary` in `src/autoreg.jl`, the Monahan transform in
`src/monahan.jl`), so `arma_roots` is extraction, not new mathematics.
A test can assert the obvious invariant: `is_stationary(m)` iff all
`abs.(ar_inverse) .< 1`.

### 4.5 `ts_display` (Pattern C)

`[S]` R's `ggtsdisplay` is the standard first-look figure: the series,
with ACF and PACF beneath. The package has every piece
(`acf`, `pacf`, both returning `ACFResult`) but no combined figure.

```julia
ts_display(x; lags=nothing, period=1) -> TSDisplayResult
```

holding the series plus two `ACFResult`s, with a `layout --> @layout [a{0.5h}; b c]`
recipe. Series count **1 + 3 + 3 = 7** (each `ACFResult` recipe emits
three, per `[S]` `test/test_recipes.jl`'s own assertion).

### 4.6 Two things that should **not** get recipes (Pattern D)

`[S]` `seasonal_subseries_plot` already set the precedent, and its
reasoning applies verbatim to:

- **`seasonal_plot(x, period)`** — R's `ggseasonplot` overlays cycles as
  lines; its polar variant draws the same data on a circle; Python's
  `seasonal_plot` does something different again. Three equally valid
  renderings, so return `(values, position, cycle)` and let the caller
  choose.
- **`lag_plot(x, lags)`** — `gglagplot` draws `x_t` against `x_{t-k}` in
  a panel grid; `gglagchull` draws convex hulls of the same data. Return
  the paired vectors.

**Say so in each docstring, in the same words
`seasonal_subseries_plot` uses.** A future session will otherwise "fix"
the missing recipe.

---

## 5. Stage 10 recipes

These depend on the Stage 10 items. Each item handoff already asks for
its recipes; this section is the shared specification so they come out
consistent.

### 5.1 IRF (Pattern A, on 10.3's `IRFResult`)

`[S]` `stage-10.3-irf-fevd-handoff.md` §8 defines `IRFResult` with
`orth`, `nonorth`, `lower`, `upper`, `stderr_orth`, `stderr_nonorth`,
`horizons`, `names`, `method`.

`[V]` R draws **one page per impulse × K response panels**. A single
Julia recipe should instead draw the **K × K grid in one figure** —
`layout --> (K, K)` — which is what `IRAnalysis.plot` does and which
reads better than R's page-per-impulse paging. Row = response,
column = impulse, labelled from `names`, with the band as a ribbon when
`method !== :none`.

**Label the band honestly from `method`.** `[S]` The result type records
whether the band came from the analytic path or the bootstrap; a plot
that says "95%" without saying which is doing the reader a disservice,
and the two differ.

Series count: `K²` lines + `K²` ribbons when a band is present, plus
`K²` zero reference lines — so `3K²` with bands, `2K²` without. Pin it
in the test.

### 5.2 FEVD (Pattern A, on 10.3's `FEVDResult`)

`[V]` Both references draw **stacked bars**, one panel per variable,
horizon on the x-axis, shock composition stacked to 1. Match that:
`layout --> (K, 1)`, `seriestype := :bar`, `bar_position := :stack`.

`[V]` Assert in the test that each panel's bars sum to 1 — the
`FEVDResult` invariant 10.3 §7.7 already checks numerically.

### 5.3 `VARForecast` (Pattern A, on 10.1's type)

`[S]` `stage-10.1-var-handoff.md` §6.3 adds `VARForecast` rather than
generalizing `Forecast`. Its recipe is §4.1's fan chart at
`layout --> (K, 1)` — **write §4.1 so the per-variable loop is a shared
helper**, not a copy.

### 5.4 VAR residual panel (Pattern C, with 10.8)

`[V]` R has two overlapping figures — `plot.varest` (fit+observed,
residuals, resid ACF, resid PACF per equation) and `plot.varcheck`
(residuals, histogram+density, ACF, PACF per equation). statsmodels has
only the K×K `plot_acorr` grid.

Build **one** panel type, following `DiagnosticPlotResult`'s precedent
exactly: `var_diagnostic_plot(m::VARModel) -> VARDiagnosticPlotResult`
holding per-equation standardized residuals, ACF, Q-Q and Ljung-Box
p-values. `layout --> (K, 4)`. `[S]` This mirrors
`src/diagnosticplot.jl` closely enough that it should be written by
reading that file first, not from this sketch.

### 5.5 Cointegration (Pattern C, with 10.5)

`[V]` **R-only, no Python counterpart.** `urca`'s `ca.jo` plot draws the
cointegrating relations (`β'y_t` for each `r`) and their
autocorrelation — the visual check that the relation is actually
stationary, which is the whole claim a Johansen test makes.

```julia
coint_relations(t::JohansenTest, r::Integer) -> CointRelationResult
```

holding the `r` relation series and an `ACFResult` for each, at
`layout --> (r, 2)`. This is a real differentiation opportunity: it is
the plot that makes the test interpretable and Python users have nothing
like it.

### 5.6 Stability / CUSUM — defer, and say why

`[V]` `vars::plot.varstabil` wraps `strucchange`'s `efp` objects;
statsmodels has nothing. An OLS-CUSUM path is not hard, but it is a
**test** (with its own critical values) wearing a plot's clothing, and
the package has no structural-break machinery at all.

**Do not build half of it.** Note it in the docstring of the VAR
diagnostic panel as not implemented, and leave it as a candidate row of
its own alongside the restricted-β tests 10.6 defers.

---

## 6. Testing

`[S]` The established approach, from `test/test_recipes.jl`, needs no
plotting backend:

```julia
rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), obj)
@test !isempty(rec)
@test length(rec) == <expected series count>
```

`[S]` And the existing counts, which the new tests should sit beside:
`ACFResult` → 3, each decomposition → 4, two-period `mstl` → 5,
`DiagnosticPlotResult` → 6.

Two rules from the existing split:

1. **Panel *numbers* are tested in the computing function's own test
   file**, not the recipe's. `[S]` `test/test_diagnosticplot.jl` is 143
   lines of numeric assertions; `test_recipes.jl` only counts series.
   Follow that: `test_statedecomp.jl` asserts the state values against
   the fitted model, `test_recipes.jl` asserts the panel count.
2. **Every count is asserted, never just non-emptiness.** A recipe that
   silently drops the confidence band still passes `!isempty`.

Expected counts for everything proposed here:

| Recipe | series |
|---|---|
| `Forecast`, 2 levels, no history | 3 |
| `Forecast`, 2 levels, with history | 4 |
| `StateDecomposition`, ETS(A,A,A) | 4 |
| `StateDecomposition`, ETS(A,N,N) | 2 |
| `StateDecomposition`, TBATS 2 periods | 5 |
| `ARMARootsResult`, AR and MA both non-empty | 3 |
| `TSDisplayResult` | 7 |
| `IRFResult`, K=2, with bands | 12 |
| `IRFResult`, K=2, no bands | 8 |
| `FEVDResult`, K=2 | 2 (one stacked-bar series per panel) |
| `VARForecast`, K=2, 2 levels | 6 |
| `VARDiagnosticPlotResult`, K=2 | 12 |

Re-derive each from the written recipe rather than trusting this table —
it is a specification, not a measurement.

---

## 7. Performance

Per the stage charter §7, plus one specific to plotting.

`[S]` `_decimate_local_max(x, y, maxpoints)` already exists in
`src/recipes.jl`, and its docstring explains why it is not a stride
subsample: a narrow spectral peak can fall between strided indices and
vanish, while local-max retention always keeps the largest point in each
bin, so the global peak survives exactly.

**Reuse it** for any series that can be long — the `Forecast` history
line, the `StateDecomposition` panels on daily data, the
`CointRelationResult` series. Do **not** decimate an IRF or FEVD: those
have `h+1` points by construction, and `h` is small.

And nothing in a recipe should compute. `[S]` That is what Pattern C is
for: `diagnostic_plot` does the work once, the recipe only arranges it.
A recipe that calls `acf` internally recomputes on every redraw.

---

## 8. Build order

1. **`Forecast`** (§4.1). Largest gap, least code, covers eleven
   model families at once.
2. **`StateDecomposition`** (§4.3) — the signature Stage 9 chart, and
   R-only. Match the panel rules in §3.1 exactly.
3. **`ts_display`** (§4.5) and **`arma_roots`** (§4.4) — both small,
   both used in early book chapters.
4. **Pattern-D functions** (§4.6), with the refusal documented.
5. `VolatilityForecast` (§4.2).
6. Then, as each Stage 10 item lands: `IRFResult` and `FEVDResult` with
   10.3, `VARForecast` with 10.1, the VAR diagnostic panel with 10.8,
   the cointegration relations with 10.5.

---

## 9. Checklist

1. **Re-establish Julia** and re-verify every `[S]` claim before coding.
2. **Read `src/recipes.jl` end to end first**, and
   `src/diagnosticplot.jl` before writing any Pattern-C type. The house
   style is specific and this document only sketches it.
3. For each new recipe: docstring with the `@recipe f(…)` signature
   line, `-->` for defaults and `:=` for forced attributes, and a note
   naming the R or Python plot it matches.
4. For each Pattern-D function: the refusal, in
   `seasonal_subseries_plot`'s own words.
5. Tests in two places per §6 — numbers in the computing function's
   file, series counts in `test_recipes.jl`.
6. No new dependency. `RecipesBase` only.
7. Add the plots to the manual chapters as `@example` blocks — `[S]`
   the docs already use them, and a recipe is only useful once something
   calls it.
8. Update `development-sequence.md`: add a cross-cutting
   "Plot recipes" section next to the textbook-dataset catalogue, with
   the §8 order.
9. **Correct the record**: an earlier note in this project said
   `plot.ets` draws one panel per state (14 for monthly ETS(A,A,A)).
   `[S]` It draws at most four, using `s1` alone. §3.1 has the source.

---

## 10. What this does not cover

- Interactive or web output. `RecipesBase` is backend-agnostic by
  design and that is the whole point.
- `vars::fanchart`'s exact shading breaks. `[V]` It takes its own
  `colors` and `cis` arguments; §4.1's nested-ribbon fan is the same
  idea with the package's own defaults, not a reproduction.
- Sims-Zha IRF bands. `[V]` statsmodels has three variants (~410 lines);
  `vars` has none. `stage-10.3-irf-fevd-handoff.md` §2 already scopes
  them out, so the IRF recipe plots whichever band the result carries
  and labels it.
- Structural-break / CUSUM plots (§5.6).
