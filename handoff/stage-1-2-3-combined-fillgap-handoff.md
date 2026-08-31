# Handoff: Stages 1-3 Fill-Gap — Functionality + Cross-Cutting Visualization Layer

For a fresh Claude Code session picking this up with no prior context.
This combines three stages' worth of gap-finding against all six
reference books (Hamilton, fpp3, Durbin & Koopman, Tsay, Ladiray &
Quenneville, Shumway & Stoffer) into one document, structured in four
independently-completable parts. **Do these roughly in order (A, B,
D depend on nothing in this doc but each other's underlying result
types already existing; C is a documented non-finding, skip it) but
don't treat this as one monolithic task** — mark each part complete
separately in `development-sequence.md`.

---

## Part A — Stage 1 functionality: periodogram (1.5), Box-Cox (1.6)

Full detail already in `stage-1-fillgap-handoff.md` — summarized here,
that document is the authoritative source for exact coefficients/tests.

**1.5 Periodogram/spectral density**: corroborated by Hamilton and
Shumway & Stoffer (`astsa::mvspec`, tied to the book's own Chapter 4).
Verified directly against base R's `spec.pgram` — real defaults
(`taper=0.1`, `detrend=TRUE`), real test case (peak frequency `1/12`
exactly recovered on a known-periodicity series). `astsa::mvspec`
deliberately overrides the taper default to `0`, forcing "conscious
tapering" — recommend matching that, documented explicitly.

```julia
periodogram(x; taper=0.0, detrend=true, demean=false, pad=0) -> (freq, spec)
spectral_density(x, spans; kwargs...) -> (freq, spec, df, bandwidth)
```

**1.6 Box-Cox transform + Guerrero's method**: corroborated by fpp3
(placed alongside decomposition, not a later chapter). Formula and
Guerrero's algorithm documented from fpp3's own text; **not
independently executed this session** (CRAN unreachable) — flagged
honestly, unlike 1.5's execution-backed verification.

```julia
boxcox(x, lambda) -> Vector{Float64}
boxcox_inv(y, lambda) -> Vector{Float64}
guerrero_lambda(x, period=1; bounds=(-1.0, 2.0)) -> Float64
```

---

## Part B — Stage 2 functionality: 4 existing-test improvements + 2 new tests

Full detail already in `stage-2-fillgap-handoff.md` — summarized here.

| # | Item | Verification depth |
|---|---|---|
| 1 | ADF response-surface p-values (replaces linear interpolation) | **Full** — real MacKinnon coefficients extracted directly from `statsmodels.tsa.adfvalues` source |
| 2 | PP `:rho` p-value (currently `NaN`) | **Full** — same source file, the Z-statistic tables |
| 3 | KPSS `:auto` (Hobijn et al. 1998) | **Full** — real, short algorithm found and read directly (`JimVaranelli/KPSS-autolag`), with real fixture test cases |
| 4 | Durbin-Watson `:exact` (Farebrother's AS 153 / "Pan's procedure") | **Citation-verified only** — exact paper and algorithm name confirmed, but no working source inspected; two honest implementation paths given in the source handoff |
| 5 | ARCH-LM test (new) | **Full** — real formula, executed in R this session on both positive and negative cases |
| 6 | D&K heteroskedasticity variance-ratio test (new) | **Full** — same, executed and verified both directions |

---

## Part C — Stage 3 functionality: no gap found

Checked directly: Durbin & Koopman's structural/unobserved-components
decomposition (trend + seasonal + **cycle** + irregular, confirmed via
`statsmodels.tsa.statespace.structural.UnobservedComponents`) is
genuinely different from classical/STL/MSTL — but it belongs to a
**later, separate roadmap stage** (structural time series, comparable
to R's `bsts`), not Stage 3's filter/smoother-based scope. Hamilton's
Wold decomposition is theoretical, not a practical algorithm.
Ladiray & Quenneville is SeasonalAdjustment.jl's territory. Tsay and
Shumway & Stoffer add nothing beyond what fpp3's own pass already
covered (Box-Cox, now in Part A). **Nothing to implement here** — this
section exists so the "no gap" finding is on record, not silently
absent.

---

## Part D — Cross-cutting visualization layer (new, full design)

**The finding, corroborated three independent ways**: R base's
`tsdiag()`, `astsa::sarima()` (actual R source read directly), and
Python's `SARIMAXResults.plot_diagnostics()` all produce the *same*
4-panel residual diagnostic display — standardized residuals over time,
ACF of residuals, a normal Q-Q plot, and **Ljung-Box p-values plotted
across a range of lags** (not a single number). This single plot
integrates Stage 1 (ACF) and Stage 2 (Ljung-Box, normality) into one
standard, universal display — the highest-value item in this part.

### Design: `RecipesBase.jl` recipes, not a hard `Plots.jl` dependency

Matches the idiomatic Julia pattern (how `Distributions.jl`,
`StatsPlots.jl`-adjacent packages, and R's own generic `plot()`/
`autoplot()` dispatch work) — define `@recipe` methods for existing
result types so `plot(result)` works naturally for any user with a
plotting backend loaded, **without** this project taking on `Plots.jl`
or `Makie.jl` as a real dependency.

```julia
# Stage 1
@recipe function f(r::ACFResult) ... end            # bar/stem plot + confidence bands, already the natural companion to Stage 1.3
@recipe function f(r::PeriodogramResult) ... end     # log-scale spectral plot + CI cross, from Part A's 1.5
boxcox_profile_plot(x, lambdas=range(-1,2,length=50)) -> RecipesBase output  # log-likelihood vs lambda, the classic by-eye Box-Cox selection view

# Stage 2 -- the headline item
diagnostic_plot(resid::Vector{Float64}; fitdf::Int=0, lags::Union{Nothing,Int}=nothing)
    # 4-panel: standardized residuals, ACF of residuals, Q-Q plot,
    # Ljung-Box p-values across lags with a 0.05 reference line --
    # matches astsa::sarima()'s confirmed real structure exactly
@recipe function f(r::ADFTest) ... end   # thin wrapper if useful; the 4-panel above is the real value, not per-test plots

# Stage 3
@recipe function f(d::ClassicalDecomposition) ... end   # stacked original/trend/seasonal/remainder, shared x-axis
@recipe function f(d::STLDecomposition) ... end
@recipe function f(d::MSTLDecomposition) ... end
seasonal_subseries_plot(x, period) -> RecipesBase output  # fpp3's own signature visual
```

### The `diagnostic_plot` lag-count logic, taken directly from `astsa::sarima`'s real source

Confirmed from the actual R code, not reconstructed from a
description — worth matching exactly since it's a real, tested
convention, not an arbitrary choice:
```r
nlag = ifelse(S<7, 20, 3*S); nlag = min(nlag, 52)
ppq = p+q+P+Q - (fixed params) + abs(fitdf)
if (nlag < ppq + 8) nlag = ppq + 8
```
i.e., the number of lags shown scales with seasonal period `S` (20 for
non-seasonal/short-period series, `3*S` for longer), capped at 52, and
padded to be at least `ppq+8` beyond the number of estimated ARMA
parameters — not a fixed default like `lags=10`.

### Comprehensive test matrix

```julia
using Test

@testset "diagnostic_plot -- lag-count logic matches astsa's real convention" begin
    # S<7 -> 20 (capped/padded per the formula above); S>=7 -> 3*S, capped at 52
    @test _diagnostic_nlag(period=4, ppq=2, fitdf=0) == 20
    @test _diagnostic_nlag(period=12, ppq=2, fitdf=0) == 36
    @test _diagnostic_nlag(period=12, ppq=30, fitdf=0) >= 38  # padding rule triggers
end

@testset "diagnostic_plot -- Ljung-Box p-values computed at every lag shown, not just one" begin
    resid = randn(200)
    result = diagnostic_plot(resid)  # or the underlying data-producing function, pre-recipe
    @test length(result.ljungbox_pvalues) == result.nlag - result.ppq
end

@testset "decomposition recipes -- all three Stage 3 result types dispatch correctly" begin
    y = 100 .+ 10 .* sin.(2π .* (1:96) ./ 12) .+ randn(96)
    for d in (classical_decompose(y, 12), stl_decompose(y, 12), mstl_decompose(y, [12]))
        rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), d)
        @test !isempty(rec)  # confirms a recipe actually exists and produces series, not a silent no-op
    end
end

@testset "periodogram recipe -- reuses Part A's real verified case" begin
    # same period-12 series from Part A; confirm the recipe's plotted
    # frequency axis peaks where periodogram() itself already verified it does
end

@testset "boxcox_profile_plot -- peaks near the lambda guerrero_lambda independently selects" begin
    x = abs.(randn(96)) .+ 5.0
    guerrero_lam = guerrero_lambda(x, 12)
    profile = boxcox_profile_plot(x)
    peak_lam = profile.lambdas[argmax(profile.loglik)]
    @test isapprox(peak_lam, guerrero_lam; atol=0.3)  # two different selection methods, same rough answer expected
end
```

---

## What to do with this

1. **Part A and B first** — reuse the existing, more detailed handoffs
   directly; this document doesn't replace them, it summarizes and
   contextualizes.
2. **Part C requires no implementation** — just make sure
   `development-sequence.md` records the "checked, no gap" finding so
   it isn't re-investigated later.
3. **Part D**: implement the `RecipesBase.jl` recipes for existing
   Stage 1/3 result types first (cheap, mechanical, low risk), then
   `diagnostic_plot` (the real, valuable, more involved piece — get the
   lag-count formula exactly right, it's a real tested convention, not
   a guess).
4. Add `RecipesBase` as a genuine (lightweight) dependency; do **not**
   add `Plots.jl`/`Makie.jl` themselves — that decision was deliberate,
   matching the idiomatic Julia ecosystem pattern.
5. Update `development-sequence.md`: mark 1.5, 1.6, 2.8, 2.9, the four
   Stage 2 improvements, and a new visualization-layer entry (perhaps
   `1.7`/cross-cutting, given it spans three stages) — and record Part
   C's non-finding explicitly, not by omission.
