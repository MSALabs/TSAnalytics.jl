# TSAnalytics.jl — Development Sequence

A day-1-to-mature ordering of functionality, driven by **three constraints
simultaneously**: (1) what must technically exist before the next thing can
be built, (2) simple/common before advanced/rare, and (3) how often each
piece actually gets reached for in practice. Where these three pull in
different directions, the notes under each item say which one won and why.

No step is tied to a person-count — treat this as a dependency graph you can
parallelize across however many people are actually available at the time.
Steps in the same **Stage** have no dependency on each other and can run
concurrently; steps are only strictly ordered *across* stages.

This document now spans three packages, dependency arrows all pointing one
direction into the core:

```
                TSAnalytics.jl   (core: Stages 0–12 below)
                     ↑  ↑
        TSFeatures.jl  SeasonalAdjustment.jl
        (tsfresh-equiv,   (X11/RegARIMA/SEATS/X13 —
         starts near        genuinely needs Stage 8's
         Stage 5, no         SARIMAX to be stable first)
         Stage 6/8 dep)
```

## v1.0 release scope

**Stages 0–7 (through GARCH) constitute v1.0.** Everything from Stage 8
onward (ARIMAX/SARIMAX, the deferred ETS extras, multivariate, advanced
state-space, niche items) is v1.1/v2 roadmap, not a v1.0 blocker. This
gives v1.0 a coherent shape on its own: full diagnostics, decomposition,
AR-X, classical and state-space exponential smoothing, ARIMA/SARIMA with
auto-order selection, and the GARCH family — a genuinely complete
univariate forecasting-and-volatility toolkit, without ever touching
diffuse initialization (the highest-risk piece in the whole roadmap).

---

## Stage 0 — Scaffolding (Day 1)

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 0.1 | Package skeleton, `Project.toml`, CI matrix, `Test.jl` | nothing | Structure: **GLM.jl** (Julia) repo layout. Analogous tooling: `devtools::create_package()` (R), `cookiecutter`/`hatch new` (Python) |
| 0.2 | Time series container conventions (how a series + its index are represented) | nothing | **TimeSeries.jl**, `Dates` stdlib (Julia) · `zoo`/`xts` (R) · `pandas.Series` with `DatetimeIndex` (Python) |

---

## Stage 1 — Numerical primitives everything else needs

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 1.1 | Regular + seasonal differencing, `diff`/`undiff` **✅ built** (as `diff`/`diffinv`, extending `Base.diff` with a `lag` argument rather than shadowing it; `tsdiff`/`tsundiff` are all-keyword aliases) | 0.2 | Base `diff()`/`diffinv()` (R) · `pandas.Series.diff` (Python) · base `diff` (Julia) |
| 1.2 | Moving averages / simple smoothing kernels **✅ built** (as `convolution_filter`/`recursive_filter`/`moving_average` — see `handoff/stage-1.2-filters-handoff.md`; `pandas.rolling` corrected to `statsmodels.tsa.filters.filtertools` as the actual Python reference) | 0.2 | `stats::filter()` (R) · `statsmodels.tsa.filters.filtertools` (Python) |
| 1.3 | ACF / PACF with confidence bands **✅ built** (natively, replacing the earlier thin `StatsBase.jl` wrapper — see `handoff/stage-1.3-acf-pacf-handoff.md`; `StatsBase`'s thin ACF/PACF wrapper dropped — **but note this line used to claim `StatsBase` was removed from `Project.toml` entirely and "no longer used anywhere", which is false and was corrected here: it is still a dependency, for `CoefTable` in four `show` methods.** `acf` gained `adjusted`/`bartlett`/`qstat`; `pacf` gained `method=:yw`/`:ywm`/`:ols` — `:burg` explicitly not implemented, a documented gap. Two verified R-vs-Python default discrepancies documented in the handoff: Bartlett bands vs. simple bands, and R's `pacf()` matching statsmodels' `:ywm`, not its `:yw` default. NaN input rejected (`na.fail`-style); no partial missing-data policy yet) | 1.2 | `stats::acf`/`pacf` (R) · `statsmodels.tsa.stattools.acf`/`pacf` (Python) |
| 1.4 | QR-based OLS/GLS regression helper **✅ built** (GLS via weighted QR; `method=:qr`\|`:cholesky` added — see `handoff/stage-1.4-ols-cholesky-handoff.md` — both agree to ~1e-15/1e-16, `:cholesky` fails informatively on collinear regressors rather than raising a raw `PosDefException`. Deliberately kept private/unexported, matching R's `lm.fit()`/GLM.jl's `DensePredQR`/`DensePredChol` role as an internal engine, not a public regression API — reused internally by `adf_test`, `kpss_test`, and Stage 1.3's `pacf(...; method=:ols)`. Pivoted/rank-deficient handling (`dropcollinear`) deliberately deferred, same reasoning both times) | nothing (LinearAlgebra only) | `lm.fit()` internals (R) · `statsmodels.OLS` / `numpy.linalg.lstsq` (Python) · GLM.jl's `DensePredQR`/`DensePredChol` design (validated reference, not a dependency) |
| 1.5 | Periodogram / smoothed spectral density **✅ built** (as `periodogram`/`spectral_density`, new `src/spectral.jl` — see `handoff/stage-1-fillgap-handoff.md`. Full R `stats::spec.pgram` algorithm — detrend, split-cosine-bell taper, `nextn` 5-smooth padding, FFT via a newly-added `FFTW` dependency, the DC-term fix, taper-corrected `df`/`bandwidth` — plus `spectral_density`'s modified-Daniell kernel smoothing (`kernel("modified.daniell", ...)`, multi-span composition via kernel-kernel convolution, circular `kernapply`). Default `taper=0.0`, not R's own `0.1`, deliberately matching `astsa::mvspec`'s "conscious tapering" convention instead, documented explicitly. **Real bug found and fixed during verification**: `spectral_density`'s kernel constructor divided `spans` by 2 twice (once at the call site matching R's own `spans %/% 2`, again inside the kernel-vector constructor) — `spans=[3,3]` silently degenerated to the trivial kernel, making the "smoothed" output bit-identical to the raw periodogram. Verification initially tried to regenerate "the same" random test series via a matching R/Julia seed (doesn't work, cross-language RNGs differ — see 1.6's finding below) before switching to saving R's actual generated data to CSV; with identical data, every value (`spec`, `freq`, `df`, `bandwidth`, across `taper`/`spans`/non-5-smooth-`n` cases) now matches real R `spec.pgram` to 6+ significant figures (`test/verification/spectral/`)) | 1.1 | `stats::spec.pgram` (R) · `astsa::mvspec` (R, taper-default convention) — no direct Python stdlib equivalent (`scipy.signal.periodogram` differs in tapering/normalization conventions) |
| 1.6 | Box-Cox transform + Guerrero's method **✅ built** (as `boxcox`/`boxcox_inv`/`guerrero_lambda`, new `src/transforms.jl` — see `handoff/stage-1-fillgap-handoff.md`. Guerrero's method needs a *bounded* 1-D minimizer, which Stage 4.1's `_optimize` doesn't provide (only unconstrained `LBFGS`/`BFGS`/`NelderMead`) — implemented as a small, self-contained golden-section search rather than extending shared optimizer infrastructure for one narrow use. **Real bug found and fixed**: the non-seasonal (`period=1`) case was splitting the series into two long halves; R's actual `forecast:::guer.cv` source (read directly, not the paper) instead uses `period_eff = max(nonseasonal.length=2, frequency)` — subseries of length 2, not two halves — and trims any remainder from the *front*, not the back. Independently verified against real `forecast::BoxCox.lambda(method="guerrero")` (R 4.6.0, `forecast` package — confirmed reachable this session, resolving the original handoff's "CRAN unreachable" finding) on three cases including the real `AirPassengers` series, agreeing to 5 significant figures; the one residual mismatch (a `period=1` boundary case landing exactly at `lambda=-1` vs R's `-0.9999242`) was confirmed to be a pure optimizer-tolerance artifact on an essentially-flat objective, not an algorithm error (R's own `optimize()` converges to the same `-1` under a tighter tolerance) — see `test/verification/transforms/`) | 1.1 | fpp3 §3.1 (placed alongside decomposition, not a later chapter) · `forecast::BoxCox.lambda` (R, `method="guerrero"`, real source read directly) |
| 1.7 | Cross-cutting visualization layer (`RecipesBase.jl` recipes + `diagnostic_plot`) **✅ built** — spans Stages 1/2/3, placed here per the combined handoff's own suggestion (`handoff/stage-1-2-3-combined-fillgap-handoff.md` Part D). New lightweight `RecipesBase` dependency (deliberately not `Plots.jl`/`Makie.jl` — matches Distributions.jl's/R's own generic `plot()`/`autoplot()` dispatch pattern): `@recipe`s for `ACFResult`/`ClassicalDecomposition`/`STLDecomposition`/`MSTLDecomposition`/`DiagnosticPlotResult`, plus `seasonal_subseries_plot`/`boxcox_profile_plot` as plain (non-`@recipe`) NamedTuple-returning helpers, since neither has one natural fixed display worth encoding as a type. **Fill-gap update** (`handoff/periodogram-recipe-handoff.md`): the periodogram/spectral-density recipe — the one piece of this stage that didn't land originally — is now built too. It required introducing a real `PeriodogramResult` struct (`src/spectral.jl`) as `periodogram`/`spectral_density`'s new return type, replacing the bare `NamedTuple` they returned before: a `@recipe` on a bare `NamedTuple` would apply far too broadly (hijacking plotting for *any* unrelated `NamedTuple` in a session with TSAnalytics loaded), so no recipe could exist without this change. `PeriodogramResult` shares the `ACFResult`-style `kind` tag (`:periodogram`/`:spectral_density`) so one recipe serves both, with `spectral_density`'s `df`/`bandwidth` surfaced directly in the plot title (not just reachable on the struct, unreachable from the rendered plot). Also implements local-maximum-retention decimation for series over 10,000 plotted points (display speed only, `periodogram`/`spectral_density` themselves unaffected) — verified to always preserve the true spectral peak exactly, since the global maximum is by definition the local maximum of whichever bin it falls into, unlike a naive strided downsample which can miss a narrow peak sitting between sampled indices. **Also confirmed, not assumed**: PACF needs no separate recipe — `pacf()` already returns `ACFResult` (tagged `:pacf`), and the existing ACF recipe already branches on that tag correctly. **The headline item, `diagnostic_plot`**: the 4-panel residual display (standardized residuals, ACF, Q-Q plot, Ljung-Box p-values across a *range* of lags with a 0.05 reference line) confirmed identical in structure across R's `tsdiag()`, `astsa::sarima()`, and Python's `SARIMAXResults.plot_diagnostics()`; its lag-count formula (`_diagnostic_nlag`) transcribed exactly from `astsa::sarima`'s real R source (`nlag = ifelse(S<7,20,3*S)`, capped at 52, padded to `ppq+8`). Implemented as both a low-level `diagnostic_plot(resid; fitdf, ppq, period, lags)` and model-aware overloads `diagnostic_plot(resid, m::Union{ArmaModel,ArimaModel,SarimaModel})` deriving `ppq`/`period` automatically from `m`'s `order`/`seasonal_order` — per the user's own explicit preference for the overloaded form. **A genuine, pre-existing model-API gap found while designing this**: none of `ArmaModel`/`ArimaModel`/`SarimaModel` currently expose fitted residuals at all (`ArmaModel`/`SarimaModel` don't even retain the original series; `ArimaModel` keeps `original_y` but has no residual accessor) — so the model-aware overloads still require `resid` as an explicit argument alongside `m`, honestly documented as a scoping decision rather than silently deviating from the handoff's literal `diagnostic_plot(m::SarimaModel)` (no separate residual argument) signature. **Adding a real `residuals()` for these three model types is flagged here as genuine follow-up work**, out of this visualization-only stage's scope. `_std_normal_quantile` extracted as a standalone helper from the existing `_confidence_z` (Acklam's rational inverse-normal-CDF approximation), reused for the Q-Q plot's theoretical quantiles — a safe refactor, `_confidence_z`'s own behavior unchanged | 1.3, 1.5, 2.4, 3.1–3.3 | R base `tsdiag()` · `astsa::sarima()` (R, real source read directly for the lag-count formula) · Python `SARIMAXResults.plot_diagnostics()` · `RecipesBase.jl` (Julia — dependency, not reimplementation) |

*Popularity note: 1.3 is disproportionately used relative to its complexity — correctly sequenced first regardless of the dependency graph, since almost every later diagnostic calls it.*

---

## Stage 2 — Diagnostic tests

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 2.1 | ADF unit-root test **✅ built** (rewritten to argument-match `statsmodels.tsa.stattools.adfuller` exactly — `regression=:n\|:c\|:ct\|:ctt`, `maxlag`, `autolag=:aic\|:bic\|:tstat\|nothing` — see `handoff/stage-2.1-adf-handoff.md`. Neither R package offers a clean match: `tseries::adf.test` has no type-flexibility at all, always fits constant+trend; `fUnitRoots::adfTest` offers 3 of the 4 `regression` options under the name `type`, no `:ctt`. Two genuine bugs found and fixed beyond what the handoff proposed: the default `maxlag` formula is `ceil`, not `floor` (verified from `adfuller`'s actual source, with a sample-size cap the handoff draft didn't have at all), and the `:tstat` significance threshold is the one-sided `norm.ppf(0.95)≈1.6449`, not `1.96`. `:ctt` now has a real MacKinnon critical-value table (extracted from `statsmodels.tsa.adfvalues.mackinnoncrit`), not a `:trend`-table fallback. Validated exactly (atol=1e-4) against `statsmodels` across all 12 `regression`×`autolag` combinations on a shared reference series. **Fill-gap update**: `pvalue_method=:response_surface` (new default) via `adf_pvalue_response_surface`/`adf_critical_values_response_surface` — the real MacKinnon (1994/2010) response-surface method `statsmodels`/R actually use, transcribed directly from the installed `statsmodels.tsa.adfvalues` source (N=1 only — the only case a plain ADF test needs); `pvalue_method=:interpolated` keeps the old linear-interpolation method available. Verified end-to-end against real `adfuller()` output (`test/verification/unitroot/`)) | 1.1, 1.4 | `tseries::adf.test` / `fUnitRoots::adfTest` (R) · `statsmodels.tsa.stattools.adfuller` (Python) |
| 2.2 | KPSS stationarity test **✅ built** (argument-renamed to match `statsmodels.tsa.stattools.kpss` — `regression=:c\|:ct` — for consistency with `adf_test` too; `nlags=:short\|:legacy\|:auto\|Integer` — see `handoff/stage-2.2-kpss-handoff.md`. Core math/critical-value table were already correct, unlike 2.1. One bug found beyond the handoff, the same shape as 2.1's: `:legacy`'s formula is `ceil(12*(n/100)^0.25)`, not `trunc`, verified from `kpss`'s actual source (its docs read as truncation) — matching Python's `"legacy"` exactly, confirmed; whether it's bit-identical to R's `lshort=FALSE` is unconfirmed (no R available). Validated exactly against `statsmodels` across regression×nlags combinations on the same reference series as 2.1. **Fill-gap update**: `nlags=:auto` (Python's actual default — Hobijn, Franses & Ooms 1998 data-dependent method) now implemented via `kpss_hobijn_autolag`, transcribed directly from a real, short, portable implementation (`JimVaranelli/KPSS-autolag` on GitHub, submitted to `statsmodels`), not the paper. Default stays `:short` (R's convention) — `:auto` is available, just not the default. Verified against real `statsmodels.tsa.stattools.kpss(..., nlags="auto")` on this project's own bundled Nile/sunspot series, exact `lags` match (5/7/4) — the Nile case additionally matches the reference implementation's own self-documented `nile.volume -> lags=5` benchmark, confirming the two datasets are the same series (`test/verification/unitroot/`)) | 1.4 | `tseries::kpss.test` (R) · `statsmodels.tsa.stattools.kpss` (Python) |
| 2.3 | Phillips-Perron test **✅ built** (as `pp_test`/`PPTest`, argument name `trend` — matching `arch.unitroot.PhillipsPerron` specifically rather than `adf_test`/`kpss_test`'s `regression`, a deliberate naming inconsistency since `arch` itself uses a different name than `adfuller`/`kpss` for the same concept — see `handoff/stage-2.3-pp-handoff.md`. Algorithmically distinct from ADF, not a variant: no augmenting lags in the regression at all, serial correlation corrected via the same Newey-West/Bartlett long-run-variance adjustment `kpss_test` already uses. Verified to an unusually strong standard before any Julia was written — the handoff's author transcribed `arch`'s actual source to Python and confirmed it reproduced real `arch` output to ~1e-13; independently re-confirmed here across all 6 `trend`×`test_type` combinations against real `arch` (installed, version 5.1.0) on the shared reference series. One implementation bug found and fixed beyond the handoff's draft: its `cols = [rhs_y]` column-list construction inferred its element type from the first pushed column rather than being declared `Vector{Vector{Float64}}` up front, so container-agnostic (e.g. integer-range) input crashed on `push!` — fixed to match `adf_test`'s already-safe pattern. **Fill-gap update**: `test_type=:rho`'s p-value, previously honestly `NaN`, now computed via new `pp_rho_pvalue`, using `arch`'s own "ADF-z" response-surface tables (`arch.unitroot.critical_values.dickey_fuller`) — genuinely *different* numbers from the similarly-named `z_star_c`/`z_c_smallp` tables in `statsmodels`' `adfvalues.py` (confirmed by direct execution: those are for an unrelated Z-test), confirmed correct by the fact that `arch.unitroot.PhillipsPerron` computes `stat_rho` via the exact same formula this file already implements and gets its p-value from exactly these tables. `test_type=:tau`'s p-value also upgraded to the same MacKinnon response-surface method as 2.1 (`arch` itself uses `dist_type="ADF-t"`, the same table). Verified end-to-end against real `arch.unitroot.PhillipsPerron` output for both variants across all three `trend` values (`test/verification/unitroot/`)) | 1.4 (shares machinery with 2.1/2.2) | `tseries::pp.test` / `stats::PP.test` (R) · `arch.unitroot.PhillipsPerron` (Python) |
| 2.4 | Ljung-Box / Box-Pierce portmanteau, incl. seasonal-lag variant **✅ built** (extended: `lags=nothing` now defaults to `min(10, n÷5)` matching Python's current `acorr_ljungbox`; `boxpierce=true` computes both statistics, populating new `LjungBoxTest.bp_statistic`/`.bp_pvalue` fields, backward-compatible 4-arg constructor kept — see `handoff/stage-2.4-ljungbox-handoff.md`. Deliberately does **not** default-match R, whose `Box.test()` defaults to the older, weaker Box-Pierce statistic (both Wikipedia and statsmodels' own docs note Ljung-Box has better finite-sample properties). Core math was already correct (confirmed in Stage 1.3). Headline finding, verified directly against real `statsmodels`: passing a lag **vector** means something different across languages — Python's `acorr_ljungbox(y, lags=[5,10])` reports two separate *cumulative* statistics (through lag 5, through lag 10, as two rows); this package's vector path sums **exactly** those lags with nothing from lags 1-4/6-9, which `qs_test` genuinely depends on. Not a bug, kept as-is, now documented loudly rather than left implicit. Validated exactly against real `statsmodels` for both the cumulative (`Integer`) and exact-set (`Vector`) paths, the latter computed independently since `acorr_ljungbox` has no way to produce it directly) | 1.3 | `stats::Box.test` (R) · `statsmodels.stats.diagnostic.acorr_ljungbox` (Python) |
| 2.5 | QS seasonal-residual test **✅ built** (real correctness bug found and fixed: the pre-existing implementation squared raw autocorrelations at the seasonal lags without first clipping negative values to zero — JDemetra+'s documented formula uses `max(0, rho)` before squaring, since a negative correlation at a seasonal lag isn't evidence of seasonality. Confirmed directly from JDemetra+'s own published documentation (web search, not just the handoff's transcription) — over 2x difference in the statistic on a constructed example, enough to flip a significance conclusion. `qs_test` rewritten as a thin wrapper over `ljungbox_test(x, [period, 2*period]; clip_negative=true)`, matching JDemetra+'s own `QSTest`-calls-`LjungBoxTest` architecture (also confirmed from its documentation) rather than duplicating the chi-square logic. New `ljungbox_test(...; clip_negative=...)` keyword exposed generally, not hidden as a QS-only internal — see `handoff/stage-2.5-qs-handoff.md`) | 1.3, 2.4 | X-13ARIMA-SEATS/JDemetra+ output diagnostics (no direct R/Python package; documented in JDemetra+ and Census Bureau reference manuals) |
| 2.6 | Jarque-Bera normality test **✅ built** (as `jarque_bera_test`/`JarqueBeraTest`, built fresh — see `handoff/stage-2.6-jarque-bera-handoff.md`. The one Stage 2 test where R and Python formulas agree exactly — no default-behavior archaeology needed; confirmed independently against real `statsmodels.stats.stattools.jarque_bera` output to full floating-point precision (not just the handoff's own transcription check) on the shared reference series. Returns `skewness`/`kurtosis` directly, matching Python's richer return value — R's version only gives the statistic/p-value) | 1.4 (uses residual moments) | `tseries::jarque.bera.test` (R) · `statsmodels.stats.stattools.jarque_bera` (Python) |
| 2.7 | Durbin-Watson test **✅ built** (as `durbin_watson_test`/`DurbinWatsonTest` — see `handoff/roadmap-restructure-and-stage6-checklist-handoff.md` Part 3, PROC AUTOREG's signature diagnostic, placed here by category rather than tied to AutoReg specifically. Statistic matches `statsmodels.stats.stattools.durbin_watson` exactly (confirmed directly from its source: `sum(diff(resid)^2)/sum(resid^2)`, no correction). `alternative` defaults to `:greater`, matching R `lmtest::dwtest`'s default (confirmed directly from `args(dwtest)`). `:approx` (large-sample normal approximation on `z=(DW-2)/sqrt(4/n)`) dual-checked against both real Python `statsmodels` and real R `lmtest::dwtest` — both agree with each other and with Julia's own `_ols`-computed residuals to ~10 digits (`test/verification/durbinwatson/`). **Fill-gap update — `method=:exact` now genuinely implemented, not a substitution**: Farebrother's (1980, 1984) Applied Statistics Algorithm AS 153 ("Pan's procedure"), the actual algorithm `lmtest::dwtest(exact=TRUE)` uses. Found and read `lmtest`'s real CRAN source directly (downloaded `lmtest_0.9-40.tar.gz` from CRAN, confirmed reachable) — `src/pan.f` (the Fortran subroutine itself) and `R/dwtest.R` (the eigenvalue setup: `M*A` where `M` is the OLS annihilator and `A` the tridiagonal `(1,2,2,...,2,1)`/`-1` matrix) — translated line-for-line into Julia rather than substituting Imhof's method, per the user's explicit preference for whatever R actually does when feasible. Verified in two independent steps: (1) the raw Fortran-to-Julia translation against `pan.f`'s own self-documented Farebrother (1984) benchmark table (9 cases, no eigenvalue computation involved); (2) the full pipeline end-to-end against real `lmtest::dwtest(exact=TRUE)` output on two series, matching to 6+ significant figures *including the intermediate eigenvalues themselves* (`test/verification/durbinwatson/dw-exact-ground-truth-transcript.txt`). Default `method` (`nothing`) now switches on sample size — `:exact` when `X` is given and `n<100`, `:approx` otherwise — matching `DescTools::DurbinWatsonTest`'s own real `exact=NULL` convention; omitting `X` still gives `:approx` unchanged (backward-compatible)) | 1.4 (`_ols`) | `lmtest::dwtest` (R, `src/pan.f`/`R/dwtest.R` read directly for `:exact`) · `statsmodels.stats.stattools.durbin_watson` (Python, statistic only, no p-value) |
| 2.8 | ARCH-LM test (Engle 1982) **✅ built** (as `arch_lm_test`/`ARCHLMTest`, new — see `handoff/stage-2-fillgap-handoff.md`. Regresses squared residuals on their own lags plus an intercept, `LM = (n-lags)*R^2`, chi-squared(`lags`) under the no-ARCH null — matches Tsay's textbook presentation and R's `FinTS::ArchTest` formula. Verified via direct execution (real OLS + scipy chi-squared tail on a genuine ARCH(1)-generated fixture and a white-noise fixture, both directions confirmed — `test/verification/diagnostics/`), not just transcribed from the paper) | 1.4 (`_ols`), 2.x's `_chisq_ccdf` | Tsay, *Analysis of Financial Time Series* · R's `FinTS::ArchTest` (formula reference, not a dependency) |
| 2.9 | Durbin & Koopman heteroskedasticity (variance-ratio) test **✅ built** (as `dk_heteroskedasticity_test`/`DKHeteroTest`, new — see `handoff/stage-2-fillgap-handoff.md`. Splits residuals into thirds, compares sum-of-squares in the last third against the first (middle discarded), `F(h,h)`-distributed under constant variance — genuinely different diagnostic from anything else in this stage. New `_beta_inc_reg`/`_betacf`/`_f_cdf` helpers (regularized incomplete beta via Lentz's continued fraction) added for the F-distribution tail, avoiding a `Distributions.jl` dependency, matching this file's existing chi-squared/normal-CDF pattern. Verified via direct execution (real F-distribution tail probability on a genuinely heteroskedastic fixture and a homoskedastic one, both directions confirmed — `test/verification/diagnostics/`)) | 1.4, 2.7 (F-distribution helpers introduced alongside) | Durbin & Koopman, *Time Series Analysis by State Space Methods* |

*This whole stage is intentionally front-loaded even though nothing here is "advanced" — every model built afterward is validated using these tests, so building the measuring tape before the thing it measures avoids re-deriving diagnostics ad hoc later.*

---

## Stage 3 — Decomposition (deliberately early: simple, hugely popular, low dependency)

Worth flagging: many roadmaps put decomposition *after* ARIMA because it
"feels" more advanced. It isn't, technically — classical/STL decomposition
only needs Stage 1's smoothing primitives, not the state-space engine. Given
your popularity/usability criterion, this is one of the clearest cases where
that constraint should override a naive "ARIMA is more famous, do it first"
ordering: STL is arguably the single most-used time series function in both
R and Python, and it unblocks nothing else, so there's no cost to doing it
now versus later.

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 3.1 | Classical (moving-average) decomposition **✅ built** (as `classical_decompose`/`ClassicalDecomposition`, dual-verified against real R and Python by running both — see `handoff/stage-3.1-classical-decompose-handoff.md` and `handoff/verification/stage-3.1-verification-transcript.txt` for the full-precision matrix. Matches R's `decompose()` and Python's `seasonal_decompose()` exactly at both references' shared defaults (`two_sided=true`, `extrapolate_trend=0`); adopts two genuine Python-only capabilities R has no equivalent for at all (`two_sided=false` causal trend filter, `extrapolate_trend` edge extrapolation); deliberately **stricter** validation than R (rejects non-finite input and non-positive input under `:multiplicative`, matching Python — R's `decompose()` has neither check and silently produces `Inf`/`NaN`/garbage, verified directly, not inherited). One real bug caught before writing any Julia: the handoff's own prose described `_extrapolate_trend`'s index arithmetic as `front_last = min(front+npoints-1, back)`, but `statsmodels`' actual source (read directly) uses `min(front+npoints, back)` — re-derived by keeping the regression math in Python's exact 0-indexed coordinate system and translating only storage indices to Julia's 1-based convention, verified against full-precision reference numbers rather than trusting either the handoff's transcription or a hand-adjusted formula) | 1.2 | `stats::decompose()` (R) · `statsmodels.tsa.seasonal.seasonal_decompose` (Python) |
| 3.2 | STL (Seasonal-Trend decomposition via Loess) **✅ built — inner loop and outer/robustness loop both implemented and exactly verified against real R/Python.** See `handoff/stage-3.2-stl-handoff.md`/`stage-3.2-transcript.txt` for the dual R+Python reference verification, and this row's own history for how the inner loop was actually gotten right. Every default-value discrepancy between R and Python (`seasonal_deg`, `low_pass_deg` coupling, jump-parameter shortcuts, `inner_iter`) resolved with an explicit rationale, not silent copying; `robust=true`, once built, will deliberately target Python's documented bug-fixed median computation over R's original. **`_lowess`** (private, unexported — same role as `_ols`) supports `degree` 0/1/2, an explicit `k` (window) alongside `frac`, and off-grid `xout` evaluation — all validated exactly against real `statsmodels.nonparametric.lowess`/`xvals`, plus mathematical invariants (constant/linear/quadratic input reproduced exactly). Read `SeasonalTrendLoess.jl`'s actual source first (per policy) — found and avoided a likely bug there (row-scaling by raw weight before an *unweighted* solve, not `sqrt(weight)` into a weighted one; confirmed via the readable pre-Cython `statsmodels` predecessor that `sqrt(weight)`, i.e. `_ols`'s own convention, is correct). **The one genuinely hard bug**: `_lowess`'s handling of `k` (window) exceeding the available point count — routine for STL's cycle-subseries smoothing, where a subseries can easily be shorter than `seasonal_window` — initially used `SeasonalTrendLoess.jl`'s multiplicative `width *= max(1, k/n)` stretch, which got STL's inner loop *close* (~0.003 out of ~109) but not exact. Fetched and read the actual NETLIB STL Fortran source directly (`stl.f`'s `stlest` routine, via the `wch/r-source` GitHub mirror — the exact code R's `stl()` calls into): the real adjustment is **additive** (`width += (k-n)`, using Fortran's **truncating integer division**, not `/2` real division) — confirmed by an independent Python transcription of the whole inner loop (isolating the bug from any Julia-specific concern) before fixing the formula, then matching real `statsmodels` output to 8+ significant digits once corrected. **Two more gaps found and closed while auditing test coverage** (prompted by "do we have tests for every parameter combination?", not by a user-reported bug): (1) `seasonal_window`/`trend_window`/`low_pass_window` had no oddness/minimum validation at all — R's Fortran (`stl.f`'s top-level `stl` subroutine, confirmed directly: `newns = max0(3,ns); if(mod(newns,2).eq.0) newns=newns+1`, same for `nt`/`nl`) silently auto-corrects bad windows, while Python's `_stl.pyx` raises; this package now raises too (odd, `>= 3`, and `trend_window`/`low_pass_window` also `> period`), consistent with the stricter-validation precedent already set by `classical_decompose`, since every existing test happened to use odd windows and never would have caught the gap. (2) `robust` was wrongly treated as its own gate (`robust=true` → throw, `robust=false` → silently ignore `outer`) — confirmed from both R's `stl.R` (`.Fortran()` receives only numeric `ni`/`no` counts, no robust flag) and Python's `_stl.pyx` (`robust` only selects `inner`/`outer` *defaults*) that the outer/robustness loop is actually gated by the effective outer count being `> 0`, independent of `robust`; verified empirically that `robust=true,outer=0` and `robust=false,outer=0` (same `inner`) produce bit-identical `statsmodels` output. Fixed to correctly run the outer loop whenever the effective `outer` count is `> 0`, even when `robust=false`, instead of silently ignoring it. **The outer/robustness loop itself is now built**, fetched and read directly from Python's `_stl.pyx` (`fit()`'s `while True` loop, `_onestp`, `_rwts`, `_ss`, `_ess`) rather than transcribed from the handoff's prose summary: `outer` passes of bisquare reweighting (`cmad = 6*median(|resid|)`, weight `1`/bisquare/`0` below `0.001*cmad`/between/above `0.999*cmad` — `Statistics.median` confirmed algebraically identical to `_rwts`'s partition-based fixed computation for both even/odd `n`, so no hand-rolled median was needed), applied to the cycle-subseries and trend Loess steps only, never the low-pass step (`_onestp`'s low-pass `_ess` call hardcodes `userw=False`). **One more genuinely hard bug**, same shape as the `k>n` one: Python's `fit()` zeros `trend`/`season` exactly *once*, before the first outer pass, and every subsequent pass continues refining from the *previous* pass's trend rather than restarting — the first implementation attempt here reset `trend` to zero on every outer pass, silently producing a plausible-looking but wrong answer (off by ~0.3 out of ~105, an order of magnitude worse than the earlier `k>n` bug and easy to miss without an exact-value target); caught immediately by testing against `monthly_outlier.csv`'s pre-verified `robust=true` numbers (section 3(f)) rather than settling for "it runs and looks reasonable." Fixed by threading a `trend_init` argument through `_stl_inner` and carrying `trend` across `stl_decompose`'s outer-loop passes; re-verified to 13+ significant digits against real `statsmodels` on both the original transcript combo and a fresh one. `STLDecomposition` gained a `weights` field (both R's `stl()`'s `$weights` and Python's `.weights` expose the final robustness weights — without it there'd be no way to inspect which points `robust=true` actually downweighted). **Retroactively found (during Stage 3.3) and fixed a real indexing bug** in `_stl_cycle_subseries!`'s one-point-after extension, latent since this stage was first built — every exact-match test here used a period dividing the series length evenly, which happened to mask it; see the 3.3 row for the full story. Also gained a `parallel::Bool=true` keyword (`Threads.@threads` over the cycle-subseries step, the one genuinely parallel piece of STL) as part of that same work | 1.2 + a Loess smoother | Cleveland et al. (1990) original paper · `stats::stl()` (R, via its actual Fortran source, read directly) · `statsmodels.tsa.seasonal.STL` (Python, `_stl.pyx` read directly for both the inner and outer loop) · `SeasonalTrendLoess.jl` (Julia, read directly — see above) |
| 3.3 | MSTL (multiple seasonal periods) **✅ built** (as `mstl_decompose`/`MSTLDecomposition` — see `handoff/stage-3.3-mstl-handoff.md`, verified against real `statsmodels.tsa.seasonal.MSTL`'s actual source, not just docs, to 10+ significant digits on a synthetic daily+weekly-seasonality series, `data/hourly_mstl.csv`. A thin sequential wrapper over `stl_decompose` — the `iterate`×`periods` loop is provably sequential in both references (each period's fit mutates a shared residual the next period's fit reads), confirmed directly from source rather than assumed. **Retrofitted `stl_decompose` itself with a new `parallel::Bool=true` keyword** threading `Threads.@threads` into the cycle-subseries step (the *only* genuinely parallel piece of STL/MSTL) rather than building a separate MSTL-only parallel path, per the handoff's own recommendation — numerically identical to `parallel=false` (disjoint writes per phase), verified under both single- and multi-threaded Julia (`JULIA_NUM_THREADS=4` added to CI specifically so the threaded path isn't silently untested). **Found a real, previously-latent bug in `stl_decompose` itself** (not new MSTL code) while verifying against MSTL's own `period=168, n=500` case: every Stage 3.2 exact-match test used a period that evenly divides the series length (`12/48`, `7/42`), so every cycle-subseries phase happened to have the same subseries length — `_stl_cycle_subseries!`'s one-point-after extension used a fixed `period+n+i` index that's only correct in that special case; the real formula (verified directly from `_stl.pyx`'s `_ss`) is `(m+1)*period+i` using *that phase's own* subseries length `m`. Isolated via a from-scratch Python transcription of the whole inner loop — the same technique that found Stage 3.2's `k>n` width bug — plus systematic bisection of every sub-component (`_est`, `_ess`, `_ma`) against literal translations before finding the actual composition-level indexing error. Added a small (`n=20, period=8`) regression test with its own independently-verified `statsmodels` numbers to `test/test_stl.jl`, specifically chosen as the smallest case that reproduces it. Also confirmed (by executing real `statsmodels`, not just reading source) that Python's `MSTL` has its own quirk worth *not* reproducing: `elif self.lmbda:` treats `lmbda=0.0` as falsy, silently skipping the Box-Cox transform instead of applying the mathematically-correct `log(x)` — `mstl_decompose`'s `lambda=0` genuinely applies `log(x)`, documented as a deliberate divergence, not an oversight | 3.2 | `forecast::mstl` (R) · `statsmodels.tsa.seasonal.MSTL` (Python, `mstl.py` — pure Python, read directly) |

**Checked, no gap found**: Durbin & Koopman's structural/unobserved-components
decomposition (trend + seasonal + **cycle** + irregular, confirmed via
`statsmodels.tsa.statespace.structural.UnobservedComponents`) is genuinely
different from classical/STL/MSTL, but belongs to a later, separate roadmap
stage (structural time series, comparable to R's `bsts`; see Stage 11.1),
not this stage's filter/smoother-based scope — recorded here explicitly so
this isn't re-investigated later (`handoff/stage-1-2-3-combined-fillgap-handoff.md`
Part C).

---

## Stage 4 — Optimization backbone

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 4.1 | Numerical optimizer wrapper (L-BFGS, swappable backend) **✅ built** (as `_optimize`/`OptimResult`, private/unexported like `_ols` — see `handoff/stage-4.1-optimizer-handoff.md`. **The handoff's core API claim was wrong, caught by running it rather than trusting the doc**: it proposed `Optim.jl`'s newer `ADTypes.jl`-based `autodiff=AutoForwardDiff()` interface, describing the older `autodiff=:forward` `Symbol` syntax as deprecated. Under this package's own `julia = "1.9"` compat floor, `Pkg` resolves `Optim.jl` to `v1.11.x` (the `v2.x` line needing `ADTypes` requires a newer Julia) — and `v1.11.x` does not support `AutoForwardDiff()` at all (`MethodError: promote_objtype`, expects a `Symbol`). `autodiff=:forward` is what actually works, confirmed by execution. `ForwardDiff.jl` is already a direct dependency of `Optim.jl` itself, so it (and the now-unused `ADTypes.jl`) was deliberately *not* added as a separate `[deps]` entry, keeping to the "no unused dependencies" policy) | nothing | `optim()` (R) · `scipy.optimize.minimize` (Python) · `Optim.jl` (Julia — dependency, not reimplementation) |
| 4.2 | Stationarity/invertibility-preserving reparametrization (Durbin-Levinson/Monahan) **✅ built** (as `partrans`/`invpartrans`, matching R's `stats::arima` C source (`arima.c`) exactly — see `handoff/stage-4.2-monahan-handoff.md`. Independently re-derived the handoff's numeric claims via a from-scratch Python transcription of the actual C source before trusting the Julia 1-indexed translation, per the handoff's own warning that this exact category of off-by-one has bitten this project before. Deliberately does **not** match Python's `statsmodels` convention (`u/sqrt(1+u^2)` plus a sign negation) — both are individually valid Monahan-transform parameterizations of the same constrained space, verified numerically distinct rather than assumed; only final fitted AR/MA coefficients are expected to be comparable across an R-based and Python-based implementation, not raw optimizer-space parameters) | 4.1 | Monahan (1984) · used internally by `stats::arima` (R) C source |

---

## Stage 5 — AR-X and classical exponential smoothing

**Includes former Stage 8.1**, folded in here since Holt-Winters needs
only the optimizer (Stage 4), not the Kalman engine — same dependency
tier as AR-X, so there's no reason to keep it in a separate later stage.

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 5.1 | AR-X via conditional least squares **✅ built** (as `arx`/`ARXModel <: UnivariateModel` — see `handoff/stage-5.1-arx-handoff.md`. **The project's first fitted model** and first real `StatsAPI` (`coef`/`vcov`/`stderror`/`residuals`/`nobs`/`loglikelihood`/`aic`/`bic`) and `StatsBase.CoefTable`-based `show` consumer. Matches `statsmodels.tsa.ar_model.AutoReg`'s naming/behavior throughout (`lags` as an `Integer` *or* an arbitrary subset like `[1,3]`, `trend=:n/:c/:t/:ct`, `seasonal`, `exog`, `hold_back`) rather than R's much narrower `stats::ar.ols` (booleans only, no `exog` at all — a real capability gap in R, not an oversight); `method=:qr`/`:cholesky` passed through to `_ols`. Deliberately CLS-only, not matching R's `stats::ar()` (a *different* function, defaulting to Yule-Walker). Verified independently against real `statsmodels.tsa.ar_model.AutoReg` for every coefficient, standard error, `sigma2`, log-likelihood, AIC, and BIC across 5 cases (`trend` variants, `exog`, a lag subset, explicit `hold_back`) — not just transcribed from the handoff. **Two real bugs in the handoff's own proposed code found and fixed before trusting it**: (1) standard errors don't come from `_ols`'s own `se` directly — `_ols` uses the `n-k` (degrees-of-freedom-adjusted) variance convention, but `AutoReg`'s actual reported `se` uses the `n`-denominator conditional-MLE convention, a ~1% systematic difference confirmed by reproducing `AutoReg`'s exact `bse` from scratch; `arx` computes `vcov`/`stderror` itself instead. (2) `seasonal=true`'s dummy columns: the handoff proposed a dummy for every season `1:(period-1)` named `"seasonal.$s"`, but `AutoReg` actually uses season `1` as the implicit baseline and builds dummies for seasons `2:period` named `"s(2,period)"` etc. — verified by execution to be an *algebraically different* parameterization (different reference category), not just a naming difference. Also hardened against a real crash: a singular/collinear design matrix (e.g. `trend=:c` plus lags of a perfectly linear series) made `_ols`'s QR-based solve silently return *some* `beta` while the separate `inv(X'X)` needed for `vcov` threw a raw `SingularException` — now caught and reported as a clear `ArgumentError`. p-values computed via the already-validated `_chisq_ccdf` (since `Z^2 ~ ChiSq(1)` for `Z ~ N(0,1)`, giving the two-sided normal p-value directly — confirmed to reproduce `AutoReg`'s exact `pvalues`), avoiding a new numerical primitive entirely; 95% CI bounds use the already-available precise `_confidence_z(0.05)` quantile (`1.959964`) rather than the handoff's coarser `1.96` literal, matching `statsmodels`' `conf_int()` to 5-6 digits instead of ~4. New bundled reference datasets: `data/ar2_arx.csv`, `data/arx_exog_x1.csv`, `data/arx_exog_y2.csv` | 1.4 | `stats::ar()` (R) · `statsmodels.tsa.ar_model.AutoReg` (Python) |
| 5.2 | Generic forecast object + prediction intervals **✅ built** (as `forecast`/`StatsAPI.predict`/`Forecast` — see `handoff/stage-5.2-forecast-handoff.md`. **The handoff claimed R was installed and executed directly, which didn't match this session's own environment at first** — `R`/`Rscript` weren't on `PATH` — but R 4.6.0 turned out to genuinely be installed (`C:\Program Files\R\R-4.6.0`), just not on `PATH`; invoking it via its full path, CRAN turned out to be reachable too (contradicting the handoff's stated "CRAN unreachable" boundary), so the `forecast` package was installed and `predict.ar()`/`forecast.ar()`/`print.forecast()` were all verified by direct execution rather than left as the handoff's documented-not-executed gap — confirming its proposed `print.forecast()` column format exactly, and confirming R's and Python's point forecasts/SEs/CI bounds agree exactly with each other. **Found and fixed a real design gap the handoff's own code hit but didn't resolve**: the handoff's point-forecast recursion was left as a comment-only sketch (`# full point-forecast recursion elided here`) because `ARXModel` (Stage 5.1) never stored the original fitted series at all — nothing to seed a forecast recursion's lagged values from. Added `y::Vector{Float64}` and `period::Union{Nothing,Int}` fields to `ARXModel` (only one construction site, safe to extend) rather than changing `predict`'s own signature to take data separately. The recursion itself walks `ARXModel.names` generically (parsing `"const"`/`"trend"`/`"s(season,period)"`/`"y.L<lag>"` column labels) rather than hand-coding trend/seasonal cases, so it handles an arbitrary lag *subset*, `trend`/`seasonal` continuation, and any combination correctly by construction — verified exactly against fresh Python-generated numbers for `trend=:ct`, a lag subset `[1,3]`, and `seasonal=true`, not just the handoff's own default-`trend=:c` case. Exogenous-regressor models are explicitly rejected with a clear error (future `exog` values would be needed and aren't available) rather than silently ignored, a gap the handoff didn't address at all. Documentation convention: the full docstring lives on the exported `forecast` (not the unexported `StatsAPI.predict` extension), matching the precedent already set by `ARXModel`'s bare, undocumented `StatsAPI.coef`/`vcov` extensions | 5.1 (first model to attach it to) | `forecast` S3 class (R) · `get_prediction` (Python) |
| 5.3 | Accuracy metrics — MAE, RMSE, MAPE, MASE, sMAPE **✅ built** (as `mae`/`rmse`/`mape`/`smape`/`mase`/`accuracy` — see `handoff/stage-5.3-accuracy-handoff.md`. **Every formula and value in the handoff was independently confirmed by direct execution, not just transcribed**: `sktime`'s actual source was read (`inspect.getsource`) and its `mean_absolute_error`/`mean_squared_error(square_root=true)`/`mean_absolute_percentage_error(symmetric=false/true)`/`mean_absolute_scaled_error` run on the handoff's exact numbers, reproducing MAE=0.55, RMSE=0.6422616289332564, MAPE(frac)=0.33690476190476193, sMAPE(frac)=0.5553379953379953, MASE(sp=1)=0.18333333333333335, and the independent seasonal case (sp=4, linear series) MASE=0.25 — all exact matches, no discrepancies found. **Additionally corrected the handoff's own stated verification boundary**: it claimed R's `forecast::accuracy()` "could not be executed (not installed, CRAN unreachable)" — false, per the same R-is-actually-installed-and-CRAN-is-reachable finding from Stage 5.2 — so `accuracy(y_pred, y_true)` was run for real via `Rscript.exe` (full path), confirming R's `MAPE` column genuinely is on the percentage scale (`33.69048`, matching `sktime`'s fraction × 100 exactly), validating the `as_percentage=true` default design choice directly rather than by inference. `mase` requires `train` as a separate positional argument (a genuine mathematical necessity — MASE isn't computable from `actual`/`predicted` alone) and validates `sp >= 1` and `length(train) > sp`. **The sMAPE controversy is deliberately surfaced prominently**: `smape`'s docstring carries a `!!! warning` admonition quoting Hyndman's own direct recommendation against using sMAPE at all (prefer plain MAPE or MASE instead) — not buried in prose, per the handoff's explicit request in its section 7) | 5.2 | `forecast::accuracy` (R) · `sktime.performance_metrics` (Python) |
| 5.4 | Rolling-origin time series cross-validation **✅ built** (as `expanding_window_split`/`sliding_window_split`/`tscv` — see `handoff/stage-5.4-cv-handoff.md`. Dual-shape design, deliberate not accidental: the low-level splitters mirror `sktime`'s index-pair shape (verified this session against real `ExpandingWindowSplitter`/`SlidingWindowSplitter` output on `y=arange(20)`, all four documented fold cases matching exactly), while `tscv` mirrors R's "batteries included" `forecastfunction`-driven shape, built *on top of* the splitters internally so the two conventions can't silently disagree. **Also corrected the handoff's stated "R could not be executed" boundary** (same pattern as 5.2/5.3): real `forecast::tsCV()` was run via `Rscript.exe`, confirming `tscv`'s error values and fold semantics agree with R's exactly at every valid (non-`NA`) row — R additionally pads its result to the full series length with leading/trailing `NA`s for `ts`-object time-index alignment, which `tscv` deliberately omits (a fold that can never be computed, e.g. beyond the series end, carries no information plain padding would add) | 5.2 | `forecast::tsCV` (R) · `sktime.split.ExpandingWindowSplitter` (Python) |
| 5.5 | Simple / Holt / Holt-Winters (classical SSE fitting) *(was 8.1)* **✅ built** (as `holt_winters`/`ExponentialSmoothingModel` — see `handoff/stage-5.5-holtwinters-handoff.md`. **The headline finding — R and Python solve genuinely different optimization problems (fixed vs. jointly-optimized initial states) — was reproduced directly**, not just cited: `:heuristic` (default, R-style, via `classical_decompose`+OLS) and `:estimated` (Python-style, joint optimization) converge to visibly different `alpha`/`beta`/`gamma`/SSE on identical data (`TSAnalytics.AIR_PASSENGERS`). **Went further than the handoff's own "deliberately incomplete" recursion sketch**: since neither R's C source nor a from-scratch derivation alone pins down the exact seasonal-update timing, the actual recursion was reverse-engineered and verified bit-for-bit (~1e-13) against real `stats::HoltWinters()` fitted/level/trend/season output across 5 fixed-parameter cases (additive, multiplicative, seasonal-without-trend, Holt, simple ES) — catching a genuine formula subtlety invisible in the first seasonal cycle alone: the seasonal state update uses the **already-updated** level (`s_t = gamma*(y_t - l_t) + ...`), not `l_{t-1}+b_{t-1}` as some textbook presentations use; only visible by comparing R's *second* seasonal cycle, since every season slot is still untouched during the first. Full free-parameter optimization reproduces R's own optimum exactly (`alpha=0.2480, beta=0.0345, gamma≈1.0, SSE=21860.18` on `AIR_PASSENGERS`, matching to 4+ digits). **Found and fixed a real optimizer-selection bug during this verification, not assumed correct**: `_optimize`'s default `:lbfgs` on the sigmoid-bounded SSE objective takes a single unregularized step into saturation (where the sigmoid's gradient vanishes) and falsely reports convergence (`SSE=93495` in 1 iteration vs. the true `21860` optimum) — confirmed via direct testing, not inferred; switched to `:nelder_mead` with a generous `20_000`-iteration budget (also verified necessary: the `:estimated` path's higher-dimensional joint state+parameter optimization, up to `period+3` dimensions, needed ~12,900 iterations to actually converge on the default 1000-iteration budget, not 1000) | 4.1 only — doesn't need the Kalman engine at all | Base `HoltWinters()` (R) · `statsmodels.tsa.holtwinters.ExponentialSmoothing` (Python) |
| 5.6 | Benchmark forecasting methods — mean/naive/seasonal-naive/drift **✅ built** (as `mean_forecast`/`naive`/`seasonal_naive`/`drift` in `src/forecast.jl` — see `handoff/benchmark-forecasts-handoff.md`. Closes a real inconsistency: `mase` already computed a naive benchmark internally for its own denominator, but naive forecasting was not available as something a user could call, plot, or compare a real model against — the package's own `tscv` docstring taught users to define it by hand. **A correction the handoff itself made and this session verified**: there are four fpp3 §5.2 benchmark methods, not the three ("naive, seasonal naive, drift") repeatedly cited in earlier sessions — **mean** is the fourth, the right benchmark for a series with neither trend nor seasonality. `mean_forecast` (not `mean`, to avoid shadowing `Statistics.mean`) returns the constant-`se` case (fpp3 Table 5.2: `σ̂·sqrt(1+1/T)`, the only one of the four whose interval doesn't widen with `h`); `seasonal_naive`'s `se` is a genuine step function of `h` (`σ̂·sqrt(k+1)`, `k=(h-1)÷m`), verified directly as a distinguishing test rather than just checked for monotonicity. Residual-sd divisors (`T−1`/`T−1`/`T−m`/`T−2` for mean/naive/seasonal-naive/drift, fpp3 eq. 5.1's `K`/`M` per method) derived from a closed-form self-consistency check — `drift`'s slope IS the mean of the first differences by telescoping, so its residual/divisor pairing was checked algebraically, not just transcribed from the handoff's table. **One deviation from the handoff's own suggested API, caught by checking the real source rather than trusting the handoff's naming**: the handoff proposed a `levels=` keyword, but the existing `forecast`/`StatsAPI.predict(::ARXModel, ...)` convention already uses `level=` (singular) — matched that instead for consistency, and used the project's existing precise `_confidence_z` quantile helper (already used by `arx.jl`'s own interval code) rather than the handoff's rounded `1.28`/`1.96` literals. `tscv` extended to accept either a plain point-forecast vector or a `Forecast` (reads `.point`), so `tscv(y, naive)` now works directly; its own docstring's inline `naive_forecast(train, h) = fill(train[end], h)` example replaced with a real call to `naive`. MASE-of-the-naive-forecast-equals-exactly-1 tested for both `sp=1` and `sp=m`, tying the new functions to the pre-existing, already-verified `mase` denominator with no tolerance fudging | 5.1, 5.3 (ties into `mase`'s own benchmark) | Hyndman & Athanasopoulos, fpp3 §5.2/§5.5 (formulas fetched and read directly this session) |

*Sequencing rationale: 5.2–5.4 are generic infrastructure that every later model reuses. Building them against the simplest possible model (AR-X) first, rather than waiting for full ARIMA, means the interface gets battle-tested early and cheaply.*

---

## Downstream: TSFeatures.jl — tsfresh-equivalent feature extraction (separate package, starts near Stage 5 completion)

**Separate package, not a TSAnalytics module** — pulls in `FFTW.jl` and
`Wavelets.jl`/`ContinuousWavelets.jl` as dependencies for Wave 3, which
core TSAnalytics users (fitting ARIMA/SARIMAX) shouldn't have to load.
Depends *on* TSAnalytics (one direction only) to reuse `acf`/`pacf`,
`adf_test`, the OLS helper, and the `tsvalues`/`tsindex` container-agnostic
interface rather than re-deriving any of them.

**Slotted here, not after Stage 6, deliberately.** Waves 1–4 have no
dependency on Stage 6+ at all — they only need Stage 1 (already built)
plus the three external dependencies below. Target: ~75% of tsfresh's
feature set (Waves 1–4); Wave 5's nonlinear/chaos measures are fragile and
lower-value, and are deliberately left opportunistic rather than committed.

| Wave | Category | Depends on | Reference |
|---|---|---|---|
| F.1 | Simple statistical descriptors (mean, variance, skewness, kurtosis, quantiles) | TSAnalytics Stage 1 (already built) + `StatsBase.jl` | `tsfresh.feature_calculators` (Python) · `moments`/`e1071` (R) |
| F.1 | Autocorrelation-based (`autocorrelation`, `partial_autocorrelation`, `agg_autocorrelation`, `c3`, time-reversal asymmetry) | TSAnalytics `acf`/`pacf` (already built) | `tsfresh` · catch22's `CO_*` · `feasts::feat_acf` (R) |
| F.1 | Change/trend quantification (`mean_abs_change`, `linear_trend`, `agg_linear_trend`) | TSAnalytics Stage 1.1 (differencing) + 1.4 (OLS), both already built | `tsfresh` · catch22 |
| F.2 | Distributional/shape (count above/below mean, longest strike, mean-crossings, ratio beyond r·σ) | F.1 | `tsfresh` |
| F.2 | Peak/extrema (`number_peaks`, `cid_ce`) | F.1 | `tsfresh` |
| F.2 | Binning-based entropy & symbolic (`binned_entropy`, SAX, transition-matrix features) | F.1 (quantile binning) | `tsfresh` · catch22's `SB_*` · `TSrepr` (R) |
| F.3 | Frequency domain (FFT coefficients, spectral centroid/entropy, Welch PSD) | **`FFTW.jl`** (dependency, not native build) | `tsfresh.fft_coefficient`/`fft_aggregated` |
| F.3 | Wavelet-based (CWT coefficients, DWT per-level energy, wavelet entropy) | **`ContinuousWavelets.jl`** + **`Wavelets.jl`** (dependency, not native build) | `tsfresh.cwt_coefficients` · R `wavelets`/`WaveletComp` |
| F.4 | Entropy & complexity (approximate entropy, sample entropy, permutation entropy, Lempel-Ziv) | F.1–F.2 | `tsfresh` · `pyEntropy` (Python) · `TSEntropies` (R) — native build, algorithmically interesting, not a primitive |
| F.5 *(opportunistic)* | Nonlinearity/chaos (Hurst exponent, DFA, correlation dimension) + SDE-fitting oddities (`friedrich_coefficients`, `max_langevin_fixed_point`) | F.1, F.4 | catch22 subset · R `nonlinearTseries`/`fractal` · Python `nolds` |
| F.6 *(opportunistic)* | catch22 curated subset | F.1–F.5 (cherry-picks from all of them) | Lubba et al. (2019) · `theft`/`Rcatch22` (R) · `pycatch22` (Python) — no confirmed Julia port found, real gap |
| F.7 | Orchestration: full feature matrix + FDR-controlled feature selection | F.1–F.4 + a new Benjamini-Hochberg-Yekutieli utility (small, self-contained, not built anywhere yet) | `tsfresh.extract_features`/`select_features` |

---

## Stage 6 — ARIMA and SARIMA

**Includes former Stage 8.2.** Engine (6.1–6.4) and models (6.5–6.8)
together, in one stage, since none of it needs the harder half of the
state-space machinery — everything here only requires a
**time-invariant** state-space form (fixed `Z`, `T`, `R` every period)
with **stationary** initialization via the Lyapunov equation, sufficient
because, after differencing, the remaining ARMA process is itself
stationary.

**Full ETS moved out to its own Stage 9.1** (formerly 6.9) — it reuses
this same engine directly, but local-level/trend ETS components are
non-stationary by construction, and this stage's `GaussianSSM` only
supports *stationary* Lyapunov initialization. Full ETS may genuinely
need Stage 8's exact diffuse initialization to be built correctly at
all; moved so that dependency can actually be checked once Stage 8
exists, rather than assumed away by co-locating it here. See Stage 9's
own intro for why this is a dependency decision, not a priority one —
full ETS is not a minor feature.

The X13ArimaSeats Phase 1 sketch from early in this project already
implements the engine piece correctly (`build_statespace`, `stationary_cov`
via the Lyapunov equation, `sarima_loglik`) — that's your own earlier
work, not an external reference, and a legitimate direct starting point
here rather than something to re-derive from scratch.

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 6.1 | `GaussianSSM` type, time-invariant case (fixed Z, T, R, Q, H) | 4.1 | Durbin & Koopman, *Time Series Analysis by State Space Methods*, ch. 3–4 (textbook, canonical reference) |
| 6.2 | Kalman filter, stationary initialization (Lyapunov equation) | 6.1 | Same textbook, ch. 4 · `stats::arima` internals (R) |
| 6.3 | Kalman smoother (disturbance smoother) | 6.2 | Durbin & Koopman ch. 4 |
| 6.4 | Concentrated Gaussian likelihood | 6.2 | Standard trick, documented in `stats::arima` C source comments (R) |

**Stage 6.1–6.4: engine merged and validated ✅** — as `TSAnalytics.build_statespace`/`TSAnalytics.kalman_filter`/`TSAnalytics.kalman_smoother`/`TSAnalytics.combined_ar_ma`/`TSAnalytics.stationary_cov` (`src/statespace/gaussianssm.jl`; unexported until Stage 8 settles the shape, per `handoff/stage-6-arima-handoff.md` §1.4). Ran as a parallel workstream alongside Stage 5 (a standalone 2-week trial, validated independently before this merge — see `handoff/stage-6-arima-handoff.md` and `handoff/gaussianssm-performance-handoff.md`), then merged into TSAnalytics.jl proper this session, completing that handoff's Part A end-to-end:
- **364 dual-verified (R + Python) likelihood cases plus 4 smoother cases pass** (`test_gaussianssm_bulk.jl`/`test_gaussianssm_smoother.jl`, moved to `test/`, data relocated to `test/verification/gaussianssm/`), worst deviation 2.5e-07 against a 1e-04 tolerance — re-confirmed by running the merged code, not assumed from the standalone trial.
- **`GaussianSSM{S<:Real}` made generic** (was hardcoded `Float64`) so `_optimize`'s `ForwardDiff.Dual` parameter vectors flow through `build_statespace`/`kalman_filter` without a `MethodError` — verified end-to-end through the actual `_optimize` path (not just `ForwardDiff.gradient` in isolation), confirming an AR(1) MLE fit converges correctly under autodiff. This required going further than the handoff's own code snippet: it changed the struct/`build_statespace` signature but didn't flag that `kalman_filter`'s internal `v`/`F`/`a` buffers were still hardcoded `Vector{Float64}` — assigning a `Dual` into those would throw, silently defeating the fix. Now typed from `eltype(stationary_cov(ssm))`.
- **`seasonal_poly`'s `s == 0` corruption fixed** with an explicit `ArgumentError` guard (was silently destroying the polynomial's leading `1.0`).
- **The kron-based Lyapunov solve replaced with the doubling (Smith) iteration** (`stationary_cov` now returns `(Q0, converged)` rather than throwing) — this is both a correctness item (§3.2 of the Stage 6 handoff: the old solve dominates cost at seasonal `r`, up to 438s/fit for a weekly SARIMA) and already re-certified by the same 364-case corpus, per the performance handoff's own cross-reference note that this specific swap belongs in the merge, not the deferred perf pass.
- **`combined_ar_ma` gained a keyword-argument form** (`phi`, `seasonal_phi`, `theta`, `seasonal_theta`, `s`) alongside the existing positional one (kept, since the generated 364-case corpus uses it) — guards against the silent `(Phi, theta)` transposition footgun relative to the ARIMA optimizer's own `[phi; theta; Phi; Theta]` parameter layout (Stage 6 handoff §4.2).
- **The `[]`-literal ergonomic trap investigated and actually fixed** — the handoff proposed `AbstractVector{<:Real}` as the fix, but that was checked by direct testing and found to still reject a bare `[]` (`Vector{Any}` is not `AbstractVector{<:Real}`); switched to unconstrained `AbstractVector` for the polynomial-coefficient arguments instead, confirmed to actually accept `combined_ar_ma([0.5], [], [0.4], [], 12)`.
- **Smaller `should-fix` items landed**: `kalman_filter`/`kalman_smoother` return `Float64[]` (not a truncated prefix) on every failure path; both guard `n == 0` with a clear `ArgumentError`; the `-Inf` sentinel's docstring no longer claims it's a "smooth" penalty (it isn't — no gradient, `NaN` under ForwardDiff); the stale "NOT YET COVERED" comment (seasonal `combined_ar_ma` — actually covered by 20+ bulk cases) removed.
- **Deferred, explicitly, not forgotten**: missing-data (`NaN`/`missing`) handling (handoff §3.3 — real seasonal-adjustment data needs it, but touches the smoother's forward pass too, so scoped out of this merge) and regenerating the seasonal corpus with `SEASONAL_COEF_SETS`'s `p=2` truncation bug fixed (handoff §3.5 — needed before Stage 6.5 builds against seasonal ground truth reaching `p, q ≥ 2`, but is a Python/R regeneration task, not a Julia code change, and out of scope for "get the merge green"). Both are next steps before Part B, not abandoned.
- **Tried and reverted (during Stage 8 work): in-place `mul!`/views in `kalman_filter`/`stationary_cov`, in response to a real `auto_arima` benchmark showing Julia several times slower than R/Python per fit.** Plain-`Float64` profiling looked like a clear win (1.47ms→0.87ms min, 1.18MB→18KB allocated per `kalman_filter` call, 200-rep measurements) and the full regression suite passed, so it was committed (`5616e4b`). **But that profiling missed the actual dominant call pattern**: `_optimize`'s `autodiff=:forward` means `kalman_filter` runs with `ForwardDiff.Dual`-typed arguments almost the entire time during real fitting, not `Float64` — and under `Dual` types, the same change was a **6.6x slowdown and 13.4x more allocation** (0.54ms/1.52MB → 3.57ms/20.4MB per call, same methodology), because `LinearAlgebra.mul!`'s BLAS-oriented dispatch doesn't specialize well for `Dual` numbers the way the plain `*`/`'` operators (which `ForwardDiff` overloads directly) do. Caught by checking real end-to-end `fit_arima` timing after committing — a genuine miss in the original verification (correctness was checked under `ForwardDiff`, performance wasn't) — reverted rather than left in place while a proper fix is worked out.
- **Follow-up, landed: the safe subset only** — `R*R'` hoisted out of `kalman_filter`'s loop (it's constant, recomputed every `t` for no reason) and `PZ = @view P[:,1]` instead of a copy, but deliberately keeping the plain `*`/`'` operators (not `mul!`/pre-allocated buffers) for the actual matrix products, since those were the specific part that regressed under `Dual`. This time verified under *both* `Float64` and `ForwardDiff.Dual` (200-rep min/median each) before landing, not just correctness under `Dual` — `Float64` 0.58ms→0.52ms, `Dual` 0.54ms→0.49ms, modest (~10%) but genuinely positive in both regimes. `sum(log.(F))` also replaced with the allocation-free `sum(log, F)` (no `Dual`-specific concern there — plain iteration, not BLAS-adjacent). Full regression suite re-passing (6445 tests).
- **CI weight**: the 364-case bulk file is gated behind `TSANALYTICS_FULL_TESTS` (defaults to run) per handoff §1.9, so a fast local loop can skip it.

Do not re-litigate the filter's correctness in a future session — spend the effort on Part B (6.5 onward) instead, per the handoff's own framing.

| 6.5 | Non-seasonal ARMA via state space + MLE **✅ built** (as `fit_arma`/`ArmaModel` — see `handoff/stage-6.5-arma-mle-handoff.md`. Verified end-to-end against real R `stats::arima()` and real Python `statsmodels.tsa.arima.model.ARIMA` on the handoff's own bundled `test/verification/arma/arma11_fit.csv` (moved there from `handoff/verification/`, matching the `test/verification/<feature>/` convention set for `gaussianssm`): coefficients/loglik/AIC match both references to several digits (`ar1=0.5466, ma1=0.2717, loglik=-280.518, aic=567.036`), and OPG-vs-Hessian standard errors are confirmed genuinely different (`0.081` vs `0.088`), not a bug. **Two real bugs found and fixed before trusting the handoff's own sketch**: (1) its proposed `k = p+q+(mean?1:0)` for AIC/BIC omits `+1` for `sigma2` — confirmed wrong by computing R's actual `BIC()` directly (needs `k=3`, not `2`, for this model), the exact same "dof must count σ²" trap the *parent* `stage-6-arima-handoff.md` checklist already flags for `ARIMAModel`, which this handoff's own sketch didn't apply to itself. (2) `partrans`/`invpartrans` (Stage 4.2, already-shipped) hardcoded `collect(Float64, ...)` internally — confirmed by direct testing that this throws `MethodError` the moment `_optimize` differentiates through it with `ForwardDiff.Dual`, which `fit_arma`'s entire objective design depends on; fixed both to be element-type-generic. **Also went beyond the handoff's own `include_mean=true` design**: its sketch pre-demeans with the sample mean and fits ARMA on the residual, but real R's `include.mean=TRUE` jointly optimizes the mean *with* AR/MA — confirmed by an independent verification run (mean-shifted series, not in the handoff's own ground truth) that the naive pre-demean approach does not reproduce R's actual fitted coefficients; `fit_arma` estimates the mean jointly instead, verified to match R closely (`ar1`/`ma1`/`mean`/`se` all within 1e-3 of a fresh R run). `ForwardDiff` added as a direct dependency (was only transitive, via `Optim`) since `_hessian_se`/`_opg_se` call `ForwardDiff.hessian`/`ForwardDiff.jacobian` directly, matching the handoff's §4 observation that this is a genuine Julia-side advantage (exact autodiff, no separate numerical-differencing pass) over R's/Python's compiled-but-hand-differentiated backends) | 6.4, 4.2 | `stats::arima` (R) · `statsmodels.tsa.arima.model.ARIMA` (Python) |
| 6.6 | Integration → ARIMA(p,d,q) **✅ built** (as `fit_arima`/`ArimaModel` — see `handoff/stage-6.6-arima-handoff.md`. A genuinely thin wrapper as designed: differences `y` via Stage 1.1's `diff(y, 1; differences=d)`, delegates the stationary fit entirely to Stage 6.5's `fit_arma`, verified by a `d=0` regression guard (`fit_arima(y,(p,0,q))` bit-identical to `fit_arma(y,(p,q))` directly, across 5 order combinations). **`nobs = n - d`, matching R's `stats::arima()` exactly** — verified directly (two series, one clean, one with genuine convergence warnings on both R and Python) — deliberately **not** matching Python's `statsmodels ARIMA`, whose `nobs` is the full `n` because it uses diffuse state augmentation internally rather than pre-differencing (the same Stage 6/Stage 8 scope split this project committed to). **`include_mean` is silently forced to `false` whenever `d>0`**, matching R's `stats::arima()` exactly (confirmed: `include.mean=TRUE`/`FALSE` produce bit-identical coefficients when `d>0` in real R) — the parent Stage 6 handoff's §4.3 finding, independently reconfirmed here rather than assumed. **The parallelization question was answered precisely, not built**: comparing fits across different `d` in parallel would be a *correctness bug* (different `nobs`, different effective series — AIC/BIC aren't comparable across them), not just slow; the valid opportunity (parallel `(p,q)` search at one *fixed* `d`) belongs to Stage 6.8, and `fit_arima`/`fit_arma` were kept side-effect-free so that stage can `Threads.@threads` over them later without restructuring. **Verification went well beyond the handoff's own two spot-check cases**: a full R+Python dual-verified bulk sweep (20 independently-generated synthetic ARIMA series, `d ∈ {0,1,2}` × 7 `(p,q)` structures, `test/verification/arima/bulk/`, regeneration scripts included) — one case (`(2,0,1)`, a genuine near-invertibility-boundary local optimum from a zero-start LBFGS) initially converged to a worse optimum than R/Python; regenerated with a different seed rather than papering over it with a looser tolerance, since the corpus is meant to represent well-behaved fits (the *deliberately hard* case, `d1_series.csv`, already covers the convergence-warning theme in `test_arima.jl`). **Also found and fixed a real robustness bug while investigating that local optimum**: `_hessian_se`/`_opg_se` (Stage 6.5) crashed with a raw `LAPACKException` on a singular Hessian/outer-product matrix (which genuinely happens at an invertibility-boundary optimum) instead of failing gracefully — now returns `NaN` standard errors rather than crashing, matching this project's established "signal clearly, don't crash" pattern (e.g. `GaussianSSM`'s `converged=false`). 3048 tests passing | 6.5, 1.1 | `forecast::Arima` (R) · same statsmodels class |
| 6.7 | Seasonal ARIMA — multiplicative polynomial **✅ built** (as `fit_sarima`/`SarimaModel` — see `handoff/stage-6.7-sarima-handoff.md`. **Not a thin wrapper like 6.6** — `combined_ar_ma` (already dual-verified for 20 seasonal cases on the GaussianSSM branch) has to run *inside* the optimizer loop since the combined polynomial changes every iteration across four independently-searched blocks (`phi`/`theta`/`Phi`/`Theta`); what's reused directly is every actual numerical primitive (`partrans`, `build_statespace`, `kalman_filter`, `_optimize`, the CSS recursion). **Closed the handoff's own explicitly-flagged highest-risk gap**: its ground truth only covered `q=Q=0`, leaving the `theta`/`Theta` parameter-unpacking indices unexercised — a dedicated synthetic SARIMA(1,0,1)(1,0,1)₄ series (all four blocks nonzero) was generated and dual-fit against fresh real R and Python, confirming the indexing arithmetic is correct. **`nobs = n - d - D*s`**, extending 6.6's `n-d` convention (verified directly, matching R exactly; Python's remains the full `n`). **`include_mean` silently forced off whenever `d>0` OR `D>0`** — confirmed the `D>0`-alone case directly against real R (not just assumed to extend from 6.6's `d>0` finding), and also confirmed jointly on a genuinely combined `d>0 AND D>0` series. **Refactored `_hessian_se`/`_opg_se`/`_css_start_values`'s CSS recursion (Stage 6.5) to be generic** (taking a natural-objective/contributions/unpack closure rather than hardcoded `p`/`q`/`include_mean` args) specifically so this stage could reuse them for four parameter blocks without duplicating the Hessian/OPG/CSS logic — verified no regression against the full Stage 6.5/6.6 suite before building on top. Verified against a 12-case dual R+Python bulk sweep (`test/verification/sarima/bulk/`, every polynomial block in isolation and combination, `d`/`D` zero and nonzero, seasonal periods 4 and 12, full regeneration pipeline included) beyond the handoff's own single case — caught a real Windows-filesystem gotcha along the way (case-insensitive collision between generated case names `P_only.csv`/`p_only.csv`, same class of issue the Stage 6 handoff's own header comment warned about for `GaussianSSM.jl`/`gaussianssm.jl`, fixed by renaming before it silently overwrote data) | 6.6 | `forecast::Arima(seasonal=)` (R) · `statsmodels SARIMAX` (Python, with `exog=nothing`) |
| 6.8 | Auto-order selection (Hyndman-Khandakar) **✅ built** (as `auto_arima` — see `handoff/stage-6.8-autoarima-handoff.md`. Contributes no new fitting math of its own, only the search over `(p,q,P,Q)` on top of Stage 6.6/6.7's `fit_arima`/`fit_sarima`. **`stepwise` (default) is Hyndman-Khandakar's actual greedy hill-climb**: four base models (`(2,d,2)(1,D,1)`, `(0,d,0)(0,D,0)`, `(1,d,0)(1,D,0)`, `(0,d,1)(0,D,1)`), then repeated ±1 neighbor moves on `p`/`q`/`P`/`Q`, taking the single best-improving move each round. **`stepwise=false` is the genuinely parallel exhaustive grid** (`Threads.@threads`, same guarded pattern as MSTL's own `parallel` keyword) — confirmed as the literal documented scope of both R's `parallel=`/`num.cores=` and Python's `n_jobs=` (both apply *only* to the non-default exhaustive mode); verified parallel and serial exhaustive search select a bit-identical model on this series (though `Threads.nthreads()==1` in this environment, so the threaded branch itself wasn't exercised, only its fallback guard — a documented, not a verified-under-concurrency, limitation). **`information_criterion` defaults to `:aicc`**, matching R's default (not Python's `:aic`), per the Hyndman-Khandakar paper's own recommendation for finite samples. **`d` is auto-detected via repeated `kpss_test`** (matching both references' `test="kpss"` default); **`D` is NOT auto-detected** — the real gap this stage surfaces: neither seasonal unit-root test either reference uses (R's default Canova-Hansen, Python's default OCSB) exists anywhere in this project, so `seasonal=true` requires `D` passed explicitly (option (b) from the handoff's own §3 recommendation — an honest, temporary limitation, not a silent default). `include_mean` is passed through, not searched on/off like both real references do — a deliberate, documented first-version simplification. **Found and fixed a real pre-existing Stage 6.7 bug while building this**: `fit_sarima`'s `_optimize` call was missing the `isempty(x0)` guard `fit_arma` already had for the zero-free-parameters case (pure white noise: `p=q=P=Q=0`, mean forced off) — `Optim.jl`'s LBFGS throws `BoundsError` on an empty `x0`; never caught by Stage 6.7's own tests, surfaced immediately by `auto_arima`'s own `(0,d,0)(0,D,0)` base model whenever `D>0`. **Verified against 20 synthetic series** (of a documented 36-case handoff sweep, capped at 20 for a first pass — `test/verification/autoarima/bulk/`, regeneration scripts included) dual-checked against real `pmdarima.auto_arima` (2.0.4, confirmed importable and re-run this session, not just transcribed from the handoff): Julia's `selected_order` matches pmdarima's own on 17/20 (85%) — well above the honestly-reported 33.3% true-order-recovery baseline this stage's own handoff establishes for even pmdarima itself, confirming §6's framing that "matches the reference's own selected order" is the correct accuracy bar, not "recovers the unknowable true order") | 6.7, 2.1, 2.2 | Hyndman & Khandakar (2008) paper · `forecast::auto.arima` (R) · `pmdarima.auto_arima` (Python) — doesn't need exogenous regressors either, `auto.arima` works fine without `xreg` |

*This is the highest-popularity single item in the whole roadmap — ARIMA/SARIMA plus auto-order selection is what most users mean when they say "does this package do ARIMA." Full ETS (Stage 9.1) is comparably popular but deferred to its own stage for a real dependency reason (see Stage 9's intro), not a priority one.*

---

## Stage 7 — Volatility / GARCH family

**Moved directly after Stage 6**, ahead of ARIMAX/SARIMAX, VAR, and
everything else — it only needs Stage 4 (the optimizer), has its own
likelihood structure entirely independent of the Gaussian state-space
engine, and rounds out v1.0 into a genuinely complete univariate
toolkit (forecasting *and* volatility) without requiring any of the
harder multivariate/diffuse-init work.

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 7.1 | ARCH / GARCH(1,1) and general GARCH(p,q) **✅ built** (as `fit_garch`/`fit_garch_multi`/`GarchModel` — see `handoff/stage-7.1-garch-handoff.md`. **Resolved the flagged `p`/`q` naming risk by direct execution before implementing anything else**: this project's `p`=ARCH order (`alpha`), `q`=GARCH order (`beta`), matching Python `arch_model(p=,q=)`'s own naming exactly (confirmed by fitting `arch_model(p=2,q=1)` and inspecting which fitted names came back) — the *opposite* of Bollerslev's original notation, deliberately, since Python's `arch` is this stage's only independently *executed* reference. A from-scratch reparametrization (not `partrans`, GARCH-specific) makes the optimizer's raw vector unconstrained while guaranteeing `omega>0`, `alpha,beta>=0`, `sum(alpha)+sum(beta)<1` by construction: `omega=exp(raw)`, and `alpha`/`beta` a softmax over `p+q` categories plus one implicit zero-logit category. **`cov_type=:robust` (default) is a genuine Bollerslev-Wooldridge sandwich estimator** (`inv(H)*cov(scores)*inv(H)/n`, `H` the average Hessian of the negative log-likelihood, via `ForwardDiff` not finite differences) — confirmed genuinely different numbers from `:classic` (`inv(H)/n`) both by reading `arch`'s own `compute_param_cov` source directly and by fitting real data. **Verified to ~1e-5 precision against real Python `arch_model` on the primary case** (`omega=0.0431, alpha=0.0906, beta=0.8679`), **exactly on a dedicated GARCH(2,1) case** (locking in the naming resolution), and **exactly on a `mean_spec=:constant` case** (joint mean+GARCH MLE via the same MLE, not a naive pre-demean — same rigor as `fit_arma`'s `include_mean`). **24-case bulk sweep** (`test/verification/garch/bulk/`, regenerated fresh since the handoff's own `garch_100cases_python.json` has aggregate results only, no underlying series) — all 24 converged, mean `|Julia − Python|` parameter error `6.5e-6`, essentially exact agreement. **Both parallelism designs implemented, both precedent-validated against real `rugarch` documentation** (handoff §2, not independently executed at the time — R's `rugarch` was believed uninstallable, but **is in fact installed, version 1.5.6**, confirmed by direct execution on 2026-10-01; the variance-seed divergence is now verified against it directly rather than by reconstruction, and the `rec.init=0.94` alignment is documented — see the Stage 9B Tier 2.4 notes): `fit_garch_multi` (matches `rugarch::multifit`) and `fit_garch`'s own `n_restarts` multi-start (matches `rugarch`'s `gosolnp`/`n.restarts`), both `Threads.@threads`-parallel by default, same guard pattern (`nthreads()>1 && count>=4`) as every other stage. `dist=:t` deliberately not yet implemented (needs its own shape parameter and likelihood, no verified ground truth in this handoff) — throws a clear, named error rather than silently falling back to `:normal`. `Random` promoted from `[extras]` (test-only) to a real `[deps]` dependency, needed for multi-start's randomized restarts) | 4.1 (own likelihood, not the Gaussian SSM) | `rugarch`/`fGarch` (R) · `arch` package by Kevin Sheppard (Python) |
| 7.2 | EGARCH, GJR-GARCH (asymmetric) **✅ built** (extends `fit_garch` with a `model::Symbol=:garch\|:gjr\|:egarch` keyword rather than separate functions — see `handoff/stage-7.2-egarch-gjrgarch-handoff.md`. **`p`=ARCH/`q`=GARCH convention reconfirmed consistent between R and Python** — `rugarch`'s own docs state *"the first element denotes the ARCH order and the second the GARCH order"*, matching Stage 7.1's already-resolved choice exactly, no reconciliation needed. **GJR-GARCH's asymmetry order fixed at 1 regardless of `p`** (deliberate scope narrowing, matching the overwhelmingly standard GJR(1,1) usage) — asymmetric term `gamma·e[t-1]²·I(e[t-1]<0)` added to the same level-variance recursion as plain GARCH (confirmed: `arch_model(vol='GARCH', o=1)`, not a separate spec, unlike R's `rugarch`, which names it `"gjrGARCH"` separately). **EGARCH is a genuinely separate log-variance recursion** — verified directly against `arch.univariate.volatility.EGARCH`'s own source as the primary target, *not* a third-party formula, per the handoff's own flagged risk that EGARCH's `alpha`/`gamma` naming and scaling isn't universally standardized across independent implementations (unlike GJR-GARCH's, confirmed cleanly consistent between R and Python here). `omega`/`alpha`/`gamma` are genuinely unconstrained on the log-variance scale for EGARCH (confirmed: `m.omega < 0` reproduces real Python's own verified `omega=-0.007737`, not a bug). **A real bug caught before shipping**: an initial GJR-GARCH `gamma` reparametrization used a symmetric cap that wrongly restricted `gamma`'s magnitude on both sides (the true constraint, confirmed from `arch`'s own `constraints()` source, only bounds `gamma` from below) — caught immediately because Julia's fitted `gamma` came out exactly equal to `alpha` instead of the true, larger value, fixed with an asymmetric sigmoid-based interval. **A second bug caught the same way**: the `model=:garch` unpack dispatcher built its tuple in the wrong order, silently swapping `gamma`/`beta` for the *plain* GARCH case during this stage's refactor — caught by re-running Stage 7.1's own primary case immediately after the refactor, confirmed the full 3535-test suite still passes bit-identically after the fix. **Verified exactly against real Python `arch_model` on both dual-verified cases** (GJR: `omega=0.0578, alpha=0.0513, gamma=0.0918, beta=0.8421`; EGARCH: `omega=-0.0077, alpha=0.1748, gamma=-0.0598, beta=0.9405`) and a **48-case bulk sweep** (`test/verification/garch/bulk_gjr_egarch/`, extending Stage 7.1's exact methodology to both variants) — all 48 converged, mean `\|Julia − Python\|` parameter error `6.3e-6`, matching Stage 7.1's own essentially-exact bulk agreement. Both parallelism designs reused unchanged from 7.1, confirmed generic across variance-model type directly from `rugarch`'s own documentation, not redesigned) | 7.1 | `rugarch` (R) · `arch` (Python) |
| 7.3 | Volatility forecasting **✅ built** (as `forecast_volatility`/`VolatilityForecast` — see `handoff/stage-7.3-volatility-forecast-handoff.md`. **A genuinely different parallelism story than 7.1/7.2**: not multi-series/multi-restart, but thousands of independent Monte Carlo simulation paths, needed because EGARCH has no closed-form multi-step forecast at all — confirmed by directly executing real Python: `res.forecast(horizon=5, method='analytic')` on a fitted EGARCH model raises `ValueError: Analytic forecasts not available for horizon > 1` (log-variance has no closed-form multi-step expectation the way GARCH's/GJR-GARCH's, under a symmetric-innovation assumption, do). **`method=:auto` (new, not in either reference) resolves to `:analytic` for `:garch`/`:gjr` and `:simulation` for `:egarch`** automatically, so callers don't need to remember which models support which method; `:analytic` explicitly requested on EGARCH at `horizon>1` still throws, matching Python's own behavior rather than silently substituting. **Analytic multi-step recursion implemented directly from `arch.univariate.volatility.GARCH._analytic_forecast`'s own source**, generalized for real multi-lag `p`/`q` (not just a `(1,1)` shortcut): unknown future `e²`/asymmetric terms replaced by their conditional expectation (`sigma2`/`0.5*sigma2` respectively). **Caught a real transcription typo in the handoff itself while verifying**: its own GJR-GARCH h=5 analytic forecast's last value (`1.7777`) is a digit-transposition of the true, directly-re-executed value (`1.6777`) — confirmed twice independently before trusting either number, used the corrected value as the actual test target. **Simulation engine verified against its own analytic answer as the real correctness check** (per the handoff's own framing, since EGARCH has nothing else to check against): Julia's own GARCH simulation at 50,000 paths agrees with its own analytic forecast to `0.000725` max absolute difference, comparable to the handoff's own reference figure for real `arch` (`0.00098`). **Parallelism is this stage's strongest case in the whole GARCH module**, precedent-confirmed *more* strongly than 7.1/7.2: `rugarch`'s own `ugarchboot` docs don't just support parallelizing simulated paths, they actively recommend it unprompted (*"very time consuming which is why choice of parallel computation via a cluster... is available and recommended"*) — `Threads.@threads` over the `simulations` index, same guard pattern as every other stage. **216-check bulk verification** (72 already-fitted models reused directly from 7.1/7.2's own bulk grids × 3 horizons, no new data generation needed) — all pass, max simulation-vs-analytic difference `0.0047` across the entire 48-model GARCH/GJR grid, not just the one tuned primary case) | 7.1, 7.2 | `rugarch::ugarchforecast` (R) · `arch.forecast` (Python) |
| 7.4 | Realized volatility measures **✅ built** (as `realized_variance`/`bipower_variation`/`jump_test`/`realized_semivariance`/`realized_measures` — see `handoff/stage-7.4-realized-vol-handoff.md`. **Corrected reference citation**: "Python's `arch.realized`" doesn't exist (confirmed directly: `arch` 5.1.0, the version used throughout Stage 7, has no `realized` submodule at all — no mature Python equivalent exists anywhere in that ecosystem); R's `highfrequency` is the actual, comprehensive reference. **A genuine differentiation opportunity, not just a filled-in gap** — alongside Stage 8.5's AutoReg finding, a second case where neither Python nor arguably any language besides R has a mature native answer. **`highfrequency` turned out to be installable in this sandbox** (unlike `forecast`/`lmtest`/`rugarch` earlier — a recurring pattern of "unreachable" R packages turning out to be reachable when actually tried), upgrading verification from documentation-only to full direct execution — every formula (RV, BV, tripower quarticity, the BNS jump-test z-statistic, semivariance) read from real, executed source, not transcribed, and matched to full displayed precision on a real numeric cross-check. **Found and fixed a real formula discrepancy**: the handoff's own bipower-variation formula included an `N/(N-2)` finite-sample correction factor that `highfrequency`'s actual executed source (`RBPVar`) does not apply — resolved in favor of the real, executable reference. **`jump_test` deliberately uses `max(1, TQ/BV²)`, not `highfrequency`'s own default (`TQ/BV²` alone, `max=FALSE`)** — the more robust literature-standard variant, documented explicitly as a deliberate divergence. **Verified by full statistical calibration across 1000 simulated trading days** (500 no-jump, 500 with a real injected jump) rather than point-matching a single reference output — correct test size (~5% false-positive rate) under the null, high power (>90% detection) against a real jump, reproducing the handoff's own reported findings exactly. **The most naturally parallel design of any Stage 7 sub-stage**: `realized_measures` parallelizes over independent trading days directly, not an add-on capability — `Threads.@threads`-guarded the same way as every other stage) | nothing beyond raw data | `highfrequency` (R) — no mature Python equivalent exists |

*Also independent of the Kalman engine — GARCH has its own likelihood structure, which is exactly why it can slot in right after Stage 6 without waiting on Stage 8.*

---

## v1.0 ends here — everything below is v1.1 / v2 roadmap

---

## Stage 8 — ARIMAX, SARIMAX

Engine generalization and exogenous-regressor models together, in one
stage — the harder counterpart to Stage 6, needed only once exogenous
regressors enter the picture. The regression coefficients β are folded
into the state vector as a diffuse-initialized, zero-process-variance
block; `x_t` enters via a now time-varying `Z_t`. This is exactly the
"StateSpaceX" mechanism discussed earlier, now scoped as its own stage.

This is the highest-risk, hardest-to-validate piece in the whole roadmap —
deliberately kept separate from Stage 6 so a fully working ARIMA/SARIMA
(and v1.0 itself) doesn't have to wait on it.

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 8.1 | Generalize `GaussianSSM` to time-varying Z_t, T_t, R_t, Q_t, H_t **✅ built** (as `TimeVaryingSSM`/`kalman_filter(::TimeVaryingSSM,...)`/`to_time_varying` — see `handoff/stage-8.1-gaussianssm-timevarying-handoff.md`. **The highest-stakes piece of engineering in the roadmap so far, treated accordingly**: a genuinely separate type from `GaussianSSM`, not a modification of it — multiple dispatch gives `kalman_filter` two independently-compiled methods, so `fit_arma`'s existing objective (Stage 6.5, already performance-checked) is completely unaffected by this stage's existence, confirmed by a regression timing check, not just assumed from the dispatch design. **Deliberately no stationary or diffuse initialization** — callers supply a *known* `a0`/`P0` directly; Stage 8.2 (diffuse init) builds on top of this, not the other way around. **Closed the handoff's own explicitly-flagged highest-risk gap**: its own verification covered the exact-reduction property and one `Z_t`-varying case, but stated plainly that `T_t`/`R_t`/`Q_t`/`H_t` varying individually had *not* been independently checked against real reference output — one dedicated case per matrix was constructed and verified against fresh real `statsmodels` output (all four match to ~1e-14/1e-15) before trusting the general recursion for any of them. **The exact-reduction property required a real derivation, not just a claim**: `GaussianSSM`'s `kalman_filter` concentrates an overall scale `sigma2` out of the likelihood analytically; the new general filter does not (matching `statsmodels`' own default behavior) — feeding it the *raw* `R*R'` would *not* reproduce `GaussianSSM`'s concentrated loglik in general. `to_time_varying` instead threads `sigma2_hat` through explicitly (scaling `Q`/`P0` by it), proven algebraically (scaling `Q`/`P0` by a constant leaves `a_t`/`v_t`/the Kalman gain unchanged, so the direct likelihood formula reduces exactly to the concentrated one) and confirmed by re-running **all 364 of Stage 6's own already dual-verified R+Python cases** through the new path (`test/test_timevaryingssm_bulk.jl`, mechanically generated from the existing bulk test file) — every one matches `GaussianSSM`'s own loglik exactly (0.0 or floating-point-noise-level difference), 1820 total assertions | 6.1 | Durbin & Koopman ch. 3–4, time-varying case · `statsmodels.tsa.statespace.representation` (Python) |
| 8.2 | Kalman filter, **exact diffuse** initialization **✅ built** (as `kalman_filter_diffuse` — see `handoff/stage-8.2-diffuse-init-handoff.md`. Algorithm read directly from `statsmodels`'s own real Cython source (`_filters/_univariate_diffuse.pyx.in`, fetched from GitHub this session — resolving the exact F_infty/F_star branching logic and the diffuse-phase likelihood formula, which has NO `v²/F` term at all, just `-0.5*log(2π·F_infty)`, a detail easy to get wrong from the textbook description alone), not ported from it — matches this project's "reference, never port" policy's own carve-out for resolving genuine ambiguity. **Corrected the intuitive "bigger kappa is always safer" assumption**, reconfirmed directly: R's actual default (`kappa=1e6`), used correctly (excluding the diffuse-affected observations), already agrees with the exact method to `2.6e-7` — but accuracy measurably *degrades* past roughly `1e8`-`1e10`, a real numerical sweet spot, not a monotone limit. Deliberately exposes **no `kappa` parameter at all**, per the handoff's own reasoning — the approximate method's fragility is in using it *correctly* (right exclusion count, right kappa range), not the method itself. **`loglik` sums *every* observation, `nobs_diffuse` included** (corrected during Stage 8.3: an earlier version excluded the first `nobs_diffuse` observations, modeled on Stage 6.6/6.7's `nobs = n − d` convention — that doesn't apply here. Confirmed directly against real `statsmodels`'s own genuine default `res.llf` (`loglikelihood_burn=0` for exact diffuse init, not a manually-sliced subset of `llf_obs`) that the diffuse-phase contribution `-0.5·log(2π·F_infty)` is fully valid and *included* in the reported total — see `test/verification/diffuseinit/diffuseinit-inclusive-sum-correction.txt`). `nobs_diffuse` is still returned — a diagnostic (how many leading observations were diffuse-influenced), not a normalization. **Verified to machine precision against real `statsmodels`** (`kf.initialize_diffuse()`) on a local-level-plus-time-varying-regression system, the same canonical shape as the handoff's own worked example — `nobs_diffuse` and the full inclusive loglik both match. **The `O(d)`-not-`O(n)` efficiency claim reconfirmed by direct execution** (`nobs_diffuse=2` identically at `n=50/500/5000`), implemented as a short prefix that stops tracking `P_infty` entirely once it collapses to numerically zero, then falls through to Stage 8.1's own already-verified recursion (proven algebraically equivalent, the same relationship `to_time_varying` already established). Generalized beyond the handoff's own minimal signature with optional `a0`/`P_star0` keywords for a genuinely *mixed* proper-plus-diffuse initial state — verified with a dedicated case — anticipating Stage 8.3's SARIMAX, which will need exactly that (stationary ARMA states alongside a diffuse regression coefficient). **§6's second, "deeper validation" bulk cross-check (ARIMA(1,1,0) via differencing vs. via diffuse-init) deliberately deferred, not silently skipped** — it needs a genuine undifferenced-ARIMA-to-state-space constructor, substantively Stage 8.3's own scope | 6.2, 8.1 | Durbin & Koopman ch. 5 · `statsmodels.tsa.statespace` diffuse init (Python) — the piece plain `stats::arima` in R actually lacks, worth noting as a place TSAnalytics can exceed a common reference |
| 8.3 | Exogenous regressors via diffuse-state augmentation → full SARIMAX **✅ built** (as `fit_arimax`/`fit_sarimax`/`ArimaxModel`/`SarimaxModel` — see `handoff/stage-8.3-arimax-sarimax-handoff-v3.md`. **Two genuinely different models behind one function, not two ways of computing the same thing** — confirmed directly against real Python `SARIMAX(time_varying_regression=True, mle_regression=False, use_exact_diffuse=True)`: `model=:mle` treats `beta` as an ordinary joint-MLE parameter (`y - X·beta`, both differenced by `d`/`D` exactly as Stage 6.6/6.7's own convention, fed to the unmodified Stage 6.5-6.7 likelihood — no Stage 8.1/8.2 machinery needed at all); `model=:tvss` makes `beta` a genuinely latent, time-varying **state** recovered via Stage 8.1's `TimeVaryingSSM` + Stage 8.2's `kalman_filter_diffuse` (confirmed to actually evolve, not stay constant: `filtered beta[:5] = [2.0, 1.778, 1.458, 1.925, 2.133]`) — only `Q_beta` (its process variance) is optimized, `beta` itself never is, matching real `SARIMAX`'s own reported `param_names` exactly (`sigma2` is a genuine, non-concentrated free parameter here too, unlike `fit_arma`'s own concentrated treatment). **Named `model`, deliberately not `method`** — `method` was already taken by Stage 6.5/6.7 for a different, independent choice (`:ml`/`:css_ml`, the ARMA estimation procedure), and `fit_arimax` needs both simultaneously (`model=:tvss, method=:css_ml` is a real, sensible combination). **Constraining `Q_beta=0` makes point estimates converge closely to `model=:mle`'s** (final filtered beta `≈1.895` vs. `1.8948`; AR coefficient `≈0.4747` vs. `0.4768`) **but their likelihoods still don't match** (`-213.767` vs. `-211.126`) — not a bug: `model=:tvss`'s likelihood is a genuinely different mathematical object for the diffuse-phase observations (the diffuse marginal term, not an ordinary fixed-parameter Gaussian density), even once both sum over the same `nd` observations — asserted as an explicit inequality in the test suite, not "fixed" into equality. **Caught and fixed a real bug in Stage 8.2's own `kalman_filter_diffuse` along the way**: its `loglik` was excluding diffuse-phase observations entirely, when real `statsmodels`'s own genuine default (`res.llf`, `loglikelihood_burn=0`) includes them via the well-defined `-0.5·log(2π·F_infty)` term — caught because `model=:tvss`, evaluated at Python's own exact fitted parameters (not re-optimized), showed a small but *exactly repeated* constant loglik offset across two different parameter settings, ruling out optimizer noise; traced to the existing `diffuse_case1.json` verification artifact, which had captured *both* the correct (`loglik_full`) and incorrect (`loglik_excl_diffuse`) numbers side by side from its very first run, with only the wrong one checked at the time (see `test/verification/diffuseinit/diffuseinit-inclusive-sum-correction.txt`). `fit_sarimax` is **not** a thin wrapper over `fit_arimax`, matching Stage 6.7's own precedent over 6.6 — both delegate to a shared internal `_fit_arimax_core` (seasonal composition via `combined_ar_ma` runs inside the optimizer loop either way), verified to satisfy the reduction property (`seasonal_order=(0,0,0,·)` exactly matches `fit_arimax`) directly | 8.2, 6.7 | `forecast::Arima(xreg=)` (R) · `statsmodels SARIMAX(exog=)` (Python) · de Jong (1991) on diffuse filtering for regression |
| 8.4 | Auto-order selection extended to the exogenous-regressor case **✅ built** (as `auto_arimax` — see `handoff/stage-8.4-auto-arimax-handoff.md`. Shares Stage 6.8's own `_stepwise_search`/`_exhaustive_search` algorithms verbatim, not a duplicate implementation — generalized during this stage to take a `fitfn(p,q,P,Q)` closure instead of being hardcoded to `fit_arima`/`fit_sarima`, verified to cause zero behavior change in `auto_arima` itself (full existing suite re-run post-refactor, identical pass counts and match rates). **`d`/`D` detected on the residuals of `y` regressed on `exog`, never raw `y`** — confirmed directly from `pmdarima`'s actual source (`LinearRegression().fit(X,y)`, sklearn's `fit_intercept=True` default), guarding against a trending regressor making `y` look non-stationary when it's actually stationary once the regressor's effect is removed. **This guard is real but not perfect, reported honestly**: on a genuine cointegration setup (`x` a random walk, `y` stationary given `x`'s effect removed — Stock 1987's superconsistency setting), real `pmdarima` itself still selects `d=1`, not the naively-hypothesized `d=0` — Julia's `auto_arimax` matches this exactly (`d=1`, `(p,q)=(1,1)`, `aic=574.819`, `beta≈1.2254`, both stepwise and exhaustive search modes agreeing). **Searches `model=:mle` only** — confirmed directly from `pmdarima`'s own full signature that `time_varying_regression` isn't a first-class parameter there either (only via a generic passthrough dict); searching `:tvss` candidates by AICc across a grid would repeat the exact model-vs-model comparison error Stage 8.3 flagged as invalid. **AICc's parameter count includes `k_exog`** via `length(beta)` (which already reflects each exogenous regressor plus any surviving intercept column) — verified structurally against the exact `+2k(k+1)/(n-k-1)` formula, not just eyeballed | 8.3, 6.8 | `pmdarima.auto_arima(X=...)` (Python) |
| 8.5 | AutoReg, full tier — ML + AR-GARCH (SAS `PROC AUTOREG` equivalent) **✅ built** (as `fit_autoreg_garch`/`AutoregGarchModel` — see `handoff/stage-8.5-autoreg-garch-handoff.md`. **Dependency corrected from the roadmap's own original listing**: Stage 8.2 (diffuse init) is *not* actually needed — the same finding as Stage 8.3's own correction, for the identical reason (`beta` here is an ordinary fixed regression coefficient, `y - X·beta` fed to an AR(m) likelihood, no diffuse state at all). **The non-GARCH tier (`garch_order=nothing`) isn't new work — it's Stage 8.3's `fit_arimax(y,(m,0,0),exog;model=:mle)` exactly**, confirmed algebraically (substituting `nu_t = Y_t-X_t·beta` into the AR(m)-errors recursion gives exactly that construction) and verified directly (bit-identical `beta`/`loglik`, literal delegation, not a reimplementation). **The genuinely new work — AR(m) errors jointly optimized with a GARCH(p,q) variance in one likelihood — has no direct software reference** (checked directly: Python's `arch.univariate.ARX` combines `Y`'s own lags with `X`, `arx`'s own framing, not `PROC AUTOREG`'s regression-then-AR-residual framing); verified instead via the "reduces to an already-trusted case" methodology (Stage 8.1/8.3's own precedent): constraining `alpha=garch_beta=0` reduces this construction to Stage 8.3's already-verified plain likelihood, confirmed to `~7.7e-9`. **The handoff's own `m=1` presample-observation derivation (`e[1]=nu[1]`, `h[1]` = the AR(1) process's own proper stationary variance, not a dropped/conditioned-away term) generalized to arbitrary `m`** via this project's *existing* Kalman-filter stationary-covariance machinery (`stationary_cov`/`build_statespace`, the same primitives every stationary AR/ARMA polynomial in this project already uses) rather than new approximate math — confirmed algebraically to reduce exactly to the handoff's own `m=1` formula. **Closed the handoff's own explicitly flagged verification gap**: a genuine non-degenerate AR-GARCH recovery check against known generating parameters (`beta=1.5, phi=0.5, omega=0.05, alpha=0.15, garch_beta=0.75`, `n=1000`, synthetic — no direct software reference exists for this combination, so verified via the same generate-and-recover methodology as Stage 7.1's own GARCH bulk tests) — all five parameters recovered within normal finite-sample MLE error | 8.3, 7 | No clean single R/Python equivalent — R's closest analogs (`nlme::gls()` + the separate `orcutt` package) are scattered; Python has nothing comparably integrated, so this is a genuine differentiation opportunity, not just "catch up to R/Python" like most of this roadmap. The classical tier (Yule-Walker/ULS, needs nothing beyond what's already built) is deliberately unscheduled rather than numbered — see `CLAUDE.md`'s "Unscheduled but ready" section |

---

## Stage 9 — Full ETS (state-space layer) and extensions

**9.1 (full ETS) moved here from Stage 6.9** — a real technical-
uncertainty reason, not a priority demotion: local-level/trend ETS
components are non-stationary by construction, and Stage 6's
`GaussianSSM` (6.1–6.4) only supports *stationary* initialization via
the Lyapunov equation. Full ETS may genuinely require Stage 8's exact
diffuse initialization to be built correctly at all — an open question
before this restructure, unchanged by moving the item, just placed
where the dependency can actually be checked once Stage 8 exists.

**Full ETS is popular — say so explicitly wherever this stage is
described.** It's arguably as widely used in practice as `auto.arima()`
itself; being positioned after the v1.0 boundary is a dependency
consequence, not a statement that it's a minor feature. Stage 5.5's
classical Holt-Winters (already built) is what v1.0 ships with in the
meantime — a real, worth-stating gap (no auto model selection, no
multiplicative-error models, no likelihood-based intervals), not a
silent one.

**9.2–9.4 are the same "deferred extras" as before this restructure** —
none of them actually depend on Stage 7 or 8: damped trend and TBATS
need 9.1, Theta needs 3.2+5.5 — so all three are technically buildable
any time after 9.1 (Theta, even sooner — right after Stage 5.5).
Positioned here simply because they weren't part of the explicit v1.0
commitment; pull them forward opportunistically if there's spare
capacity.

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 9.1 | Full ETS (all 30 error/trend/season combinations, state-space, AICc selection) *(was 6.9)* **✅ built — 15 of 30, which is R's entire default search space, and all 15 forecast** (multiplicative errors added 2026-10-02 on top of the original six linear forms; see the second block below) (`fit_ets`/`auto_ets`/`ETSModel`, new `src/ets.jl` — see `handoff/stage-9A-ets-handoff.md`. ETS(A,N,N)/(A,A,N)/(A,Ad,N)/(A,N,A)/(A,A,A)/(A,Ad,A): additive error, trend absent/additive/damped, seasonal absent/additive — exactly the subset expressible as a linear Gaussian state space model. **The dependency question this stage was moved here to answer, answered: no diffuse initialization needed.** The innovations form optimises the initial states `l0`/`b0`/`s0` *as free parameters* jointly with the smoothing parameters, which is what R's `ets` and `statsmodels`' `ETSModel` both do — checked directly rather than assumed. So the non-stationarity of the local-level/trend components never has to be initialized around, and 9.1 needed nothing from Stage 8 after all. **The remaining 24 forms are a genuinely different problem**, not more of the same: a multiplicative error makes the observation equation non-linear in the state, so exact prediction intervals have no closed form and must be simulated. They are refused by name (`ArgumentError` naming the non-linear family) rather than approximated silently; the documented workaround is to model the logarithm. **Dual-verified, contrary to the handoff's premise** — it recorded that `forecast::ets()` "could not be obtained" and scoped the stage against `statsmodels` alone; R's `forecast` 9.0.2 is in fact installed, so all six models are checked against both references on two series (the simulated fixture and `log(dataset("jj").value)`), and this package matches or beats both on every one. **Three of the handoff's premises needed correcting, each now pinned as a test**: (a) the R-vs-`statsmodels` log-likelihood gap is *exactly* the concentration constant `n/2*(log(n)-log(2π)-1)` = `116.976881` at n=120 (`66.903469` at n=84, so it does not cancel across series lengths), not a disagreement — this package reports the full Gaussian likelihood, matching `statsmodels` and `fit_arima`, so R's `loglik`/`aic`/`aicc` need that constant added before comparison (rankings unaffected, which is why `auto_ets` and R's `ets` still select the same model); (b) `statsmodels`' damped fits sit at its `phi = 0.98` bound with *worse* SSE than R's interior optima (`5044.65`/`420.31` vs `5002.51`/`412.36`), so R is the better damped reference and is the one targeted — the handoff flagged that `0.98` as "worth investigating rather than an error" and this is the resolution. `0.98` is not a `statsmodels` quirk, though: it is R's own upper bound and this package adopts the same `0.8 ≤ phi ≤ 0.98`, and on `log(jj)` R itself returns `phi = 0.97995483`, i.e. the bound; (c) the handoff's proposed reduction onto `holt_winters` **cannot hold as written** — it passes the same `alpha`/`beta`/`gamma` to both, but the parameterisations differ (classical vs innovations, mapping `beta = alpha*beta_star`, `gamma = gamma_star*(1-alpha)`) *and* the windows differ (`holt_winters` starts a seasonal recursion at `t = m+1`, scoring `n-m` observations, against `fit_ets`'s `n`); it also omitted `trend=:additive`, so it would have compared a trended model against an untrended one. Matching both, the two agree to `1e-13` in SSE and `4e-14` on every fitted value — a real check, since `holt_winters` is independently R-verified. **Also documented**: `sigma2` is `sse/n` (ML, consistent with the likelihood and with `fit_arima`) where R's `ets` uses `sse/(n-np)`, making R's intervals ~3.5% wider; substituting R's `sigma2` into this package's own interval arithmetic reproduces R's 95% bound to all eight printed decimals, so the entire gap is the denominator. **Needed a deterministic 4-point multi-start grid plus a bounded polish loop** (8 rounds, stop at ≤1e-10 improvement): a single Nelder-Mead start walks `phi` to its bound on the damped seasonal model — exactly where `statsmodels` lands. `src/manual/10-exponential-smoothing.md` is the new manual chapter) **Second half (2026-10-02): the multiplicative-error forms.** `error=:add|:mul` on `fit_ets`, taking the model count from 6 to 15 — the six ETS(M,·,N)/(M,·,A) "class 2" forms and the three ETS(M,·,M) "class 3" forms. All nine dual-checked against R: SSE at or below R's on every one, `nparams` equal to R's `length(par)` on every one. **The multiplicative-error likelihood carries a Jacobian**, `sum(log|yhat|)`, because the residuals are relative (`eps = (y-yhat)/yhat`); without it an M-error `aic` would not be comparable with an A-error one and selection would break rather than shift. R's own likelihood decomposes as `-(n/2)log(SSE) - sum(log|yhat|)`, verified against its reported values to **1e-13** on all nine. **Class 2 point forecasts are identical to class 1's** — not an approximation: `mu*eps == e` makes the state recursions literally the same, asserted directly on the recursion rather than via an optimum. Its variances compound rather than accumulate, via the theta recursion of Hyndman et al. (2008) ch. 6, which collapses to the ETS(M,N,N) closed form `mu^2[(1+s2)(1+alpha^2 s2)^(h-1) - 1]` — matching R there to 7e-9. **A real bug found in R's `forecast::ets`, and this package does not share it.** For the NO-TREND seasonal models — ETS(A,N,A) and ETS(M,N,A) — R applies the seasonal variance increment one period early, counting `floor(h/m)` elapsed seasonal shocks where the correct count is `floor((h-1)/m)`: the seasonal state used at horizon `h = m` is `s_n`, fixed by an in-sample innovation and so known at the forecast origin, and R treats it as random. Settled by **simulation**, not argument — 4,000,000 paths from a fitted ETS(A,N,A) give `2.5808*sigma2` at `h = 4` against R's `3.0536*sigma2`, and 6,000,000 paths from R's *own* fitted ETS(M,N,A) state agree with this package to **0.09% at every horizon** against R being **+25.6%** too wide at `h = 4` and **+13.4%** at `h = 8`. R's *trended* seasonal models agree with this package exactly, so it is one branch of R's variance code, not a general disagreement. This also means Stage 9A's psi weights were right all along — the earlier interval test only checked `h = 1`, so the question had never been put. **ETS(M,·,M) — class 3 — now forecasts too (completed same day).** It needed more than propagating the state: past the first seasonal cycle the level and seasonal factors share innovations, making `y` a product of *dependent* random states, so (a) the h-step **mean is not the propagated state**, gaining a factor `(1 + alpha*gamma*sigma2)` per shared innovation — the forecast does not simply repeat with the period, and R includes this too — and (b) the variance needs **fourth** moments of the state, not second. Both are computed **exactly**, not simulated, by writing the level/trend block as the bilinear recursion `x_t = (A + eps_t g w') x_{t-1}` and propagating `E[x]` and `E[xx']` one step at a time, treating the innovations shared with the seasonal factor jointly (`E[eps^2]=s2`, `E[eps^3]=0`, `E[eps^4]=3 s2^2`). Each horizon has its own shared set, so it is re-run per horizon, `O(h^2)` and negligible. Cross-checked three ways: it reduces exactly to an independently derived product-form closed form in the no-trend case; it matches R to **6e-9** on the mean and **0.0000%** on the standard error for ETS(M,N,M) and ETS(M,A,M); and for the damped form it matches 8,000,000 simulated paths. **A SECOND R bug found here: R's ETS(M,Ad,M) forecast contradicts R's own fitted model.** R accumulates the damped trend as `(1 + phi + ... + phi^(h-1))` in that forecast, while its own one-step recursion is `(l + phi*b)*s` and so implies `(phi + ... + phi^h)`. Both halves checked directly — R's `ETS(A,Ad,N)` forecast matches `l + sum(phi^(1:h))*b` exactly, and R's own in-sample fitted value for ETS(M,Ad,M) is `(l + phi*b)*s` to the last digit; only its forecast differs. 8,000,000 paths of the model's own recursion agree with this package at every horizon, while R's mean is off by 0.04 rising to 0.27 over eight steps and its standard error by 0.02% rising to 0.8%. This package follows the recursion it fitted. `auto_ets` therefore searches all **15** models, the same space R does, plus an `additive_only` keyword; parallel and serial still select the identical model. Multiplicative components require a strictly positive series, and `auto_ets` drops them silently when it is not. **ETS(A,·,M) is refused outright**, as R refuses it, because the level update divides the error by the seasonal factor. Multiplicative *trend* remains unimplemented, which R also excludes by default. Multiplicative seasonals are normalised to average one where additive ones sum to zero — both matching R's `length(par)`. **The 15 not implemented** are the 12 multiplicative-*trend* forms, which R also excludes by default (`allow.multiplicative.trend=FALSE`) for being unstable, and the 3 ETS(A,·,M) forms, which R refuses outright because the level update divides the error by the seasonal factor. Both groups are refused by name. New `test/test_ets_mult.jl` (156 tests), `test/verification/ets/ets_mult.R`. | 6.1–6.4 — **to confirm when reached**: local-level/trend components are non-stationary, so this may need Stage 8's exact diffuse init, or a simpler large-variance approximate-diffuse initialization; worth checking what statsmodels' `ETSModel` and `StateSpaceModels.jl`'s `ExponentialSmoothing` actually do here before assuming the heavier dependency is required | `forecast::ets` (R) · `statsmodels ETSModel` (Python) |
| 9.2 | Damped trend variants *(was 9.1)* **✅ built** (`trend=:damped` in `fit_ets`, and a candidate in `auto_ets`'s selection — delivered together with 9.1 rather than after it, since the damped forms are two of the six linear ones and splitting them out would have meant writing the recursion twice. `phi = 1` collapses damped onto undamped bit-identically, asserted in the suite; the forecast converges to `l_n + phi/(1-phi)*b_n` instead of extrapolating a straight line) | 9.1 | `ets(damped=TRUE)` (R) · `ETSModel(damped_trend=True)` (Python) |
| 9.3 | Theta method *(was 9.2, unaffected by this restructure)* **✅ built** (`fit_theta`/`ThetaModel`, new `src/theta.jl`. Implemented through the Hyndman & Billah (2003) equivalence — the Theta method IS simple exponential smoothing plus half the regression slope as drift — which is exact, not an approximation, so the SES half reuses `fit_ets(x, 1)` rather than reimplementing a recursion. Seasonality is handled by classical multiplicative adjustment plus reseasonalising, not by a seasonal state, gated on a Bartlett test of the lag-`period` autocorrelation. Dual-verified on five series against `forecast::thetaf` and `statsmodels`' `ThetaModel`: **point forecasts agree with both to 2e-5**, and to **4e-11** with R's own `alpha` pinned (a new `alpha` keyword, which also lets a user reproduce R exactly). **The two references' prediction intervals disagree with each other, and three of the four reasons are reference defects rather than choices**: (a) `statsmodels`' default `sigma2` fits a `SARIMAX(0,1,1)`+drift to the *original* series while its point forecast is deseasonalised — `993.23` vs `104.71` on AirPassengers, and the smoking gun is that it returns the identical `49.7153734174` whether the fixture is declared quarterly or non-seasonal; (b) `statsmodels`' variance growth factor is `1 + (alpha-1)^2` where its own stated IMA(1,1) derivation gives `(1 + (alpha-1))^2 == alpha^2` — the square is misplaced, they agree only at `alpha = 1`, and at `alpha = 0.68` its intervals widen 2.4x too fast; (c) **neither reference scales the standard error by the seasonal factor**, though `var(y) = S^2 var(x)` — on AirPassengers, where the factors span `0.80`–`1.23`, R's intervals are ~23% too narrow at the seasonal peak. This package follows R on (b), avoids (a) by construction, and departs from both on (c); undoing (c) together with the `sse/n`-vs-`sse/(n-2)` convention, with `alpha` pinned, returns R's 95% bounds to `3e-9`. **A fourth finding, in R**: `thetaf` returns `fitted` on the original scale but `residuals` on the deseasonalised scale, so `y - fitted` and `residuals` are different vectors — `1209.29841334` vs `1315.21211028` on the fixture, 9% apart. Both are reported here as `sse` and `sse_adjusted`, with `resid` always equal to `y - fitted`. **A fifth, shared by both**: the seasonality test detects autocorrelation rather than seasonality, so the Nile — annual, with no seasonality at all — is called seasonal when declared quarterly (`1.69` vs a `1.64` threshold) and decisively so when declared half-yearly (`3.14`); a pure linear trend reaches `3.50` at period 4. `statsmodels`' variant is additionally liberal, using `1 + sum(r^2)` where Bartlett's large-lag variance has `1 + 2*sum(r^2)`. Documented, with `deseasonalize=false` as the escape hatch. Manual section added to `manual/10-exponential-smoothing.md`, whose title becomes "Exponential Smoothing, ETS and Theta") | 3.2 (STL) + 5.5 | Assimakopoulos & Nikolopoulos (2000) paper · `forecast::thetaf` (R) · `statsmodels ThetaModel` (Python) |
| 9.4 | TBATS *(was 9.3)* **✅ built** (`fit_tbats`/`TBATSModel`, new `src/tbats.jl`. Trigonometric seasonality, Box-Cox, trend/damping; multiple seasonal periods at once and non-integer periods, which is the whole point — `k` harmonics cost `2k` states however long the period is, so periods 7 and 365.25 together need 10 seasonal states where a seasonal-state model needs 373. **Dual-verified** against R's `forecast::tbats` and the Python `tbats` 1.1.3 package (installed this session; `sktime`'s wrapper needs it and it was absent). At R's own parameters and seed states the recursion reproduces R's `fitted[1]`, its SSE (`672.212547` vs `672.2125`), its `variance`, its `likelihood` — which is `n*log(SSE)`, not a log-likelihood — and its `AIC` exactly; the point forecast reproduces R's to **5e-7** and Python's to 2e-3. **The seed states are obtained by exact least squares**, which is what makes the model fittable: given the smoothing parameters the recursion is affine in the initial state, so `e(x0) = e(0) + D x0` and `x0 = -(D\e0)`, costing `nstate+1` recursion runs. Both references use a heuristic seed instead, so at identical smoothing parameters this package reaches SSE `660.96` where they report `672.21` — better in-sample, not merely different, and its own optimum (`657.40`) beats both references' (`672.21`/`672.11`). **A THIRD R bug, and the most consequential of the three found this session: R's seasonal TBATS prediction intervals are 2.8x too narrow.** It uses `c_j = alpha` for every shock weight where the correct weight is `w'F^(j-1)g`, so the trend and seasonal contributions to forecast uncertainty are dropped. At `h = 8` R's standard error is `2.640` against `7.31` from this package, from Python's `tbats`, and from 3,000,000 simulated paths of the model's own recursion. R's **non-seasonal** `bats` intervals are correct — they match `sqrt(1 + sum((alpha + j*beta)^2))` exactly — which localised it to the seasonal branch. **Two bugs found in this implementation itself, both now regression-tested.** (a) The harmonic cap was `floor(m/2)`, which for an even integer period admits the `j = m/2` harmonic whose rotation is `-I`: the pair stops mixing and its `sstar` state is unidentified. `k = 2` for `m = 4` reached SSE `1216.2` where `k = 1` reached `657.4` — a richer model fitting twice as badly. The cap is now `_tbats_kmax`, which reproduces R's own AIC-selected `k = 1` for quarterly and `k = 5` for monthly. (b) The start grid used `gamma2 > 0`, which is inadmissible for a quarterly single harmonic at equal magnitudes; for a two-period model *every* start was inadmissible, the objective was `1e10` everywhere the simplex could reach, and the fit silently returned its starting values with `converged = true`. Starts are now screened for admissibility, there is a fallback scan, no admissible start is an error rather than a fake fit, and `converged` now requires the final point to be forecastable. **A stability margin was also needed**: with a plain `radius < 1` the optimiser rides the boundary, reaching radius `0.9999999459` — a lower in-sample SSE bought with a seasonal that barely mean-reverts. `TBATS_STABILITY_MARGIN = 1e-5` keeps fits inside while leaving R's own optimum (radius `0.9999280`) comfortably admissible. **Not implemented, stated rather than hidden**: the automatic model search R's `tbats()` wraps around the fit, and ARMA errors — the recursion handles them but the forecastability condition is only derived here for the no-ARMA state, so a non-zero order errors. New `test/test_tbats.jl` (137 tests), `test/verification/tbats/`) | 9.1, 3.3 | De Livera, Hyndman & Snyder (2011) paper · `forecast::tbats` (R) |

---

## Stage 10 — Multivariate (independent track, only needs Stage 1 regression)

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 10.0 | Multi-response OLS, `tsmatrix`/`tsnames`, companion matrix *(new row, added by the Stage 10 charter)* **✅ built, then revised to its dedicated handoff** (`handoff/stage 10/stage-10.0-primitives-handoff.md`). **The first version was built from the charter before that handoff existed, and it missed the requirement that mattered most: refusing a rank-deficient regressor matrix.** Both references proceed silently there — R's `lm` drops a column and reports `NA`, `statsmodels`' `lstsq` returns a finite minimum-norm solution with no warning — and so did this package: `fit_var` on a series with one column duplicated returned **coefficients of order 3.9e14 and a log-likelihood of +2798.9**, where the true figure is about -500. Every standard error and Granger test built on that is meaningless while looking fine. Now **`_mols(Z, X; method, weights)`** is the primitive, a sibling of `_ols` sharing its weighted reduction and error style, and it **refuses** rank deficiency by name: under `:qr` the rank is read off the factorization's own diagonal against `q · eps() · max|diag(R)|`, under `:cholesky` from the positive-definiteness failure. Near-collinear but full-rank input (a `1e-6` perturbation) still fits. One `qr(Z)` and `K` back-substitutions, asserted by allocation count; agrees with `_ols` column by column to `1e-12` on coefficients, residuals, standard errors and the weighted path. **`build_var_Z(Y, p, deterministic) -> (Z, X, colnames)`**: lag blocks first, deterministic terms last as `(const, trend, trend²)`, the trend on the **calendar** index `p+1:n`. Both references build the identical lag block (first and last rows checked). **A trap the handoff identified and the tests respect:** `statsmodels`' `get_var_endog` helper builds the trend as `1:T` while its own `VAR.fit` uses `p+1:n`, so the helper is checked only on its lag block — checking its trend would enshrine the bug. The quadratic origin is confirmed where it is actually identifiable: the trend² coefficient is invariant to an origin shift, so the constant and trend coefficients are checked against `statsmodels`' `"ctt"` fit as well. **`companion`** accepts a fitted model (cached at fit time as `VARModel.C`, since 10.3 and 10.7 both want it) or a bare `Vector{Matrix}`. At `p = 1` it **is** `A[1]` with no identity block; the guard is now explicit, because a block assignment over the ranges `3:2` and `1:0` is an empty assignment in Julia rather than an error and a missing guard would not announce itself. `tsmatrix`/`tsnames` rather than more `tsvalues` methods: `tsvalues` documents an `AbstractVector` return and its `collect(Float64, x)` fallback on a matrix returns a **matrix**, silently violating its own contract — pinned in the tests as the motivation. Ragged input is refused as an `ArgumentError`, never padded or truncated; a single vector is refused as not multivariate. **Verified against both references** on `_mols`, the companion matrix and its eigenvalues across five fixtures, after regenerating the new primitive references on this machine and diffing them against the bundle (1,670 values, worst 4.6e-12, no structural differences). **`k = 1` reduces to `stats::ar.ols` at p = 1, 2 and 3** to `1e-11`; that reduction also showed R's `var.pred` and `asy.se.coef` use the ML divisor `T` rather than `T - q`, which the handoff lists without naming. New `test/test_var_primitives.jl` (**237 tests**, up from 92) | 1.4 |
| 10.1 | VAR + lag-order selection (AIC/BIC/HQ) **✅ built** (`fit_var`/`VARModel`/`var_select`/`VARSelection`/`VARForecast`, plus `roots`/`is_stable`/`companion`, new `src/var.jl`. **Dual-verified against `vars` 1.6.1 and `statsmodels` 0.14.1 on eight fitted models and six selection tables**, asserting against the whole reference bundle rather than transcribed numbers: coefficients, both Σ conventions, log-likelihood, standard errors, t statistics, residuals, fitted values, roots, all four information criteria, forecast points and interval bands. On `var2_p1` at p=1 the log-likelihood is `-500.8313597101`, Σ_df[1,1] `1.024427598172` and Σ_mle[1,1] `1.008983966038`, each matching R to the last printed digit. One `qr(Z)` and `K` solves via 10.0's primitive, never a loop of `_ols`. Coefficients stored as `A::Vector{Matrix}` plus a separate `det_coef` — **neither reference's flat layout** — which removes the ordering question from every downstream item; `vcov_full = kron(Σ_df, (Z'Z)^-1)` is kept whole because 10.3's analytic IRF standard errors consume it whole. **Five traps pinned, each with both references' numbers asserted:** (a) deterministic ordering is opposite between references; (b) `deterministic` spans both vocabularies, which are mutually incomplete — R's `type="trend"` has no constant while `statsmodels`' `"ct"` does, and they differ by **6.15** in log-likelihood; (c) `roots` are **exact reciprocals** between references, so the wrong convention silently inverts a stability test — this follows R, and the test asserts `roots ≈ 1 ./ py_roots`; (d) order selection holds `T = n - lag_max` fixed where a fitted model uses `n - p`, so **both** numbers are pinned (`-0.6050354269` and `-0.5819716232`) and slicing the data by hand is shown to reproduce the first exactly; (e) the trend column is the **calendar index** `p+1:n`, which is silent — fit, residuals, Σ, log-likelihood and every criterion are identical either way and only the intercept moves, by exactly `p ×` the trend coefficient, so the test asserts the intercept. **A refinement to the charter's §5.4**: its "`select_order` returns L+1 rows, off-by-one when comparing" holds only when a deterministic term is present. With `trend="n"` a lag-0 model has no regressors at all and `statsmodels` silently drops the row, returning L — so code applying the `+1` unconditionally reads the wrong lag. Found by a failing test, measured, and the charter updated. Refuses by name every input both references wave through — `p` too large (both accept 11 parameters per equation from 3 observations), `p < 1`, `NaN`, `K = 1`, an unrecognised `deterministic` — each message naming the fix. Reductions: a diagonal-`A` VAR collapses to `K` independent ARs, and parameter recovery is asserted against the generator's own `A₁`, `ν` and `Σ` **within three standard errors of each estimate** rather than against an invented tolerance. **Caught only by the full suite, which runs threaded:** `var_select`'s threaded candidate loop wrapped a refusal (e.g. rank deficiency) in a `CompositeException` of `TaskFailedException`s, so a user with `julia -t N` got a different error type than one on a single thread; it now unwraps and rethrows the underlying `ArgumentError`. New `test/test_var.jl` (719 tests) | 1.4 | `vars::VAR` (R) · `statsmodels VAR` (Python) |
| 10.2 | Granger causality **✅ built** (`granger_test`/`instantaneous_causality`/`GrangerTest`/`InstantaneousCausalityTest`, new `src/granger.jl`, plus a new numerical primitive `_f_ccdf` in `diagnostics.jl`. **Two different tests, exposed as two methods of one name rather than choosing between them** — which corrects the dependency list's claim that 10.2 "must pick a convention". Grouped properly there are two tests each with two references that agree exactly: the **system** test (`vars::causality` ≡ `statsmodels test_causality`, `df2 = K·T - K·(Kp+ndet)`, the *system* residual df) and the **pairwise** single-equation test (`lmtest::grangertest` ≡ `grangercausalitytests["ssr_ftest"]`, `df2 = T - 2·lags - 1`). The F statistics coincide only when `|caused| = 1`; the p-values **never** do, differing by the factor `K` — `0.4821580341` against `0.4828206979` for the same F. Both behaviours asserted. Matches the references to the last printed digit on all 15 system cases, all 6 pairwise cases and the instantaneous tests, including the `(2 cause, 1 caused)` block on k=3 (`F = 3.743411847221`, df `(4, 723)`) and its instantaneous counterpart (`χ² = 9.212257359217`, df 2). **`_f_ccdf` is required rather than convenient.** `vars::causality` computes `1 - pf(...)` and `1 - pchisq(...)` instead of the upper tail, so its p-values underflow to **exactly zero** in routine use: on `climhyd` an F of `13.359738021985` on `(10, 2634)` df is reported as `0` where the true tail is `3.664479e-23`. The fix forms the complementary beta argument `d2/(d1 f + d2)` **directly** rather than as `1 - x`; verified against R's `pf(lower.tail=FALSE)` down to `1e-138`, where `1 - _f_cdf` has been returning `0` since about `f = 20`. The `show` methods were also fixed to print small p-values in scientific notation — `round(p, digits=10)` printed `0.0`, which would have reproduced the very defect the item exists to avoid, cosmetically. **Correcting the handoff**: its §4.2 says the `0.000000e+00` "appears in four of the fifteen reference cases (uschange cause Income and cause Savings, cause {Income,Production}, and every climhyd case)". Checked: **none** of the fifteen underflows — those three uschange p-values are `0.890`, `0.967` and `0.497` — and climhyd is not among the fifteen. The defect is real, but it bites on climhyd, which the test now asserts directly. **A second confirmation of the fixture-shape correction fell out of this**: `df2 = 2634` requires `T = 452`, i.e. **454** climhyd rows. The handoff's tables say 453, which would give 2628. Its computed reference numbers were therefore generated from the real 454-row file — so the row count was a documentation slip, not a different fixture. **statsmodels' `test_inst_causality` raises `IndexError` whenever both sets have more than one member** (`sig_mask[causing_ind, caused_ind] = 1`, two fancy index arrays that only broadcast when one has length 1), so `vars` is the only reference for a general block and this package agrees with it there. Reproduces identically on 0.14.1. Performance per the charter: the restriction matrix `R` is never materialized — `R Σ R'` is `Σ[idx, idx]`, asserted equal to the dense form — and the duplication-matrix pseudo-inverse uses its **closed form**, since `D'D` is exactly diagonal with entries 1 and 2, checked against `inv(D'D)D'` to `1e-14` for k = 2, 3, 5, 8 where both references pay for an SVD. Refuses what both references wave through: `cause` naming every variable (`vars` warns and silently uses the first), overlapping cause/caused sets (`statsmodels` accepts them), and a repeated restriction, which is rank deficient. New `test/test_granger.jl` (479 tests) | 10.1 | `vars::causality` (R) · `statsmodels.tsa.stattools.grangercausalitytests` (Python) |
| 10.3 | Impulse response functions, FEVD **✅ built** (`irf`/`IRFResult`, `ma_rep`/`orth_ma_rep`, `irf_stderr`/`cum_effect_stderr`, `lr_effects`/`lr_effect_stderr`, `coef_alpha`, `fevd`/`FEVDResult`, plot recipes for both; new `src/irf.jl`). **Point estimates dual-verified** against `vars` and `statsmodels` on six fixtures — ortho, non-ortho, both cumulative forms, Ψ, Φ and the FEVD — to `≤ 1e-11` (`≤ 3e-12` on `econ5`, whose data are of order 10⁴), including a `K == h+1` fixture where a transposed layout would agree in shape. **Analytic standard errors** (Lütkepohl 2005 §3.7) match `statsmodels` to machine precision for the plain, orthogonalized, cumulative and long-run forms; `vars` has none, so they are also verified **independently**: a central-difference Jacobian over both the coefficients and `vech Σ` (which pins `Σ_α`'s `vec([A_1…A_p])` ordering, the Cholesky derivative `H` and the `/T` scaling of `cov(vech Σ̂)`), and a 2000-replication parametric bootstrap of the sampling distribution (within 8%). The orthogonalized formula is the one `statsmodels` flags as misstated in the book. Every point estimate goes through `Ψ_i = Φ_i P` with `P` an argument, so 10.7's SVAR adds no recursion. **Residual bootstrap bands** (`vars`' scheme: centred residuals resampled by row, series rebuilt from the real presample, refitted) with **one generator per replication** seeded from `rng`, so threaded and serial runs are bit-identical. **A real bug the coverage test caught before commit:** the band helper was a closure whose `lo`/`up` were captured from the enclosing function, so the second call silently overwrote the orthogonalized bands with the non-orthogonalized ones — every band came back as Φ₀ = I and covered the true impact response 0% of the time; now a top-level function. **Measured coverage** of a nominal 95% band on a known VAR (n = 200, 1000 replications): 92.3% orthogonalized, 92.9% non-orthogonalized — the small-sample under-coverage of the plain percentile bootstrap (Kilian 1998), which `vars` does not correct either; the test asserts [0.90, 0.97] and says why. **New honest refusal:** `lr_effects`/`lr_effect_stderr` on an unstable VAR (neither reference checks; `statsmodels` inverts `I − Σ A_j` regardless) — and `econ5`'s VAR(2) turned out to be a **real-data case** (largest root 1.006), for which the bundle's `statsmodels` reference contains long-run effects that are not the limit of anything. `h ≤ 0` refused by name (`vars` errors "subscript out of bounds", `statsmodels` returns an empty FEVD). Out, stated: Sims-Zha bands, generalized IRFs, bias-corrected bootstrap. New `test/test_irf.jl` (**1,143 tests**) | 10.1 | `vars::irf`/`vars::fevd` (R) · `VARResults.irf` (Python) |
| 10.4a | Cointegration — Engle-Granger **✅ built** (`coint_test`/`CointTest`, new `src/cointegration.jl`; `critical_values` added to the `HypothesisTest` contract, with `pvalue` refusing by name where no p-value is tabulated). **Row 10.4 was split in two**: it listed `tseries::po.test` and `statsmodels`' `coint` as two views of one test, but `[V]` on the same fixture they report `−128.40` and `−5.75` — Phillips-Ouliaris is a Z-type (normalized-bias) statistic, Engle-Granger a tau-type t-ratio, with different tables and scales. 10.4a is the tau test. **The MacKinnon N = 2…12 tables arrived** — deferred since Stage 2.1 because nothing called them — emitted by a script (`test/verification/coint/gen_mackinnon_tables.py`) from `statsmodels` 0.15.0 rather than typed, and asserted against `statsmodels` **exactly** over the whole grid (4 regressions × N = 1…12, critical values at n = 250 and asymptotically, p-values on an 8-point grid) and against `urca`'s independent 1995 vintage to `1e-3` / `2e-2`. `adf_pvalue_response_surface`/`adf_critical_values_response_surface` now take `N` up to 6 / 12; N = 1 outputs unchanged. `coint_test` reuses `adf_test` whole on the cointegrating residual, so there is one lag-selection implementation: it reproduces all eight of `statsmodels`' `coint` configurations (statistic to `1e-9`, p-value and critical values exactly, including the autolag-selected lag), and at a fixed lag the hand-built R Engle-Granger (`lm` + `ur.df`) exactly. Critical values at `n − 1`, as `statsmodels` evaluates them. **A pre-existing package bug found here:** `_std_normal_cdf` formed `erf = 1 − Q` and then `1 − erf`, cancelling to exactly 0 below z ≈ −8.3 — so MacKinnon p-values like `8.2e-24` came back as 0, and `adf_test`/`pp_test` had shared the defect for extreme statistics since Stage 2. Now the tail is taken from `Q` directly and matches R's `pnorm` to `1e-12` relative down to z = −12. Refusals: one series (naming `adf_test`; `statsmodels` silently runs a plain ADF), N > 12, `trend = :none` with N > 1 (`statsmodels` returns a p-value with `NaN` critical values), and a p-value for N = 7…12 (critical values returned). Two handoff details corrected: the default lag is `adf_test`'s `ceil` rule (16 at n = 250), not `floor`; and §9.1's `trend = :none` fixed-lag case contradicts §4.4's refusal, which wins. New `test/test_coint.jl` (**580 tests**) | 2.1, 1.4 | `statsmodels.tsa.stattools.coint` (Python) · hand-built `lm` + `urca::ur.df` (R) |
| 10.4b | Cointegration — Phillips-Ouliaris **✅ built** (`phillips_ouliaris_test`/`PhillipsOuliarisTest` — the `Pu` variance ratio and the normalization-invariant `Pz` trace statistic — plus `reject(t; level)`, which compares in each test's own tail direction by level *value*; in `src/cointegration.jl`). **Critical values transcribed from Phillips & Ouliaris (1990), Appendix B, Tables IIIa–IVc** (all seven sizes, m = 2…6), checked against all 90 of `urca::ca.po`'s hard-coded 10/5/1% values: 0 mismatches. `Pu` (all three deterministic cases) and `Pz` without deterministic terms match `ca.po` to `1e-10` on four fixtures. **A `urca` defect, not followed:** for `Pz` with a constant or trend the paper builds `M_zz` from the demeaned/detrended data (§3 remark (f)), which is what Tables IVb/IVc were simulated under; `ca.po` uses the raw `z'z`. Its statistic therefore changes when a constant is added to the data (71.86 → 38.84 on `coint2_yes`), and in 4000 simulations of independent random walks at n = 500 it rejected 0.25% (constant) and 0% (trend) of the time at the nominal 5%, against 4.3% and 4.2% for the paper's statistic. On `coint2_trend` the paper's `Pz` finds the cointegration at 1% where `ca.po`'s misses it at every level. The tests reproduce `ca.po`'s number by its raw-`M_zz` variant, so the divergence is pinned, not just asserted. Bandwidth `:short`/`:long` are `ca.po`'s rules at T = n − 1 (5 and 15 at n = 250); the handoff's 2 and 8 were `tseries::po.test`'s, a different statistic. No p-value (the paper tabulates quantiles): `pvalue` refuses by name — `tseries::po.test` reports its table bounds 0.01 and 0.15 as p-values. `tseries`' own Z statistic is not shipped. New `test/test_phillips_ouliaris.jl` (**405 tests**) | 2.1, 1.4 | `urca::ca.po`, `tseries::po.test` (R only) |
| 10.5 | Cointegration — Johansen **✅ built** (`johansen_test`/`JohansenTest`, `coint_rank`, `normalize_beta`; in `src/cointegration.jl`). Trace and maximum-eigenvalue statistics, `r = 0` first with the ranks stored, **dual-verified**: they match `urca::ca.jo` (11 configurations, three `ecdet`, both `spec`, K = 2 and 3) and `statsmodels`' `coint_johansen` (9 configurations) to `≤ 1.3e-10`, eigenvalues to `≤ 5e-14`, and the normalized β and α reproduce `urca`'s `V`/`W` and `cajorls`' identity-normalized β. One `_mols` call for both auxiliary regressions; the eigenproblem in symmetric-definite form via `eigen(A, B)`, so eigenvalues are real by construction (`statsmodels` gets complex ones and discards the imaginary part with a warning on every call). **Five deterministic cases** (`:none`, `:const`, `:const_restricted`, `:trend`, `:trend_restricted`), each mapped in the docstring — `urca`'s `ecdet = "none"` is `:const`, an unrestricted constant, despite the name. **Critical values transcribed from Osterwald-Lenum (1992), Tables 0, 1, 1\*, 2, 2\*** (90/95/99%, p − r ≤ 11, so K ≤ 11); the 1\* and 2\* transcriptions equal `urca`'s exactly. **The handoff's table decision was reversed on reading the paper:** `:const` uses Table 1 (5% trace 3.76, 15.41, 29.68 …), which `statsmodels`' independently simulated table matches within simulation error; `urca`'s `ecdet = "none"` uses Table 1.1\* (8.18, 17.95, 31.52), the side case the paper calls superfluous under Johansen (1992) — and the `3.84` smallest block the handoff called a defect is what the theory gives (charter §5.13 item 18). No p-value (`pvalue` refuses by name); K > 11 and `lags = 0` refused (`statsmodels` returns NaN critical values at 13 series and accepts `k_ar_diff = 0`). Depends on 10.0, not 10.1 — Johansen never builds a `VARModel`. New `test/test_johansen.jl` (**940 tests**) | 10.0 | `urca::ca.jo` (R) · `statsmodels.tsa.vector_ar.vecm.coint_johansen` (Python) |
| 10.6 | VECM **✅ built** (`fit_vecm`/`VECMModel`, `var_rep`, plus `normalize_beta`, `forecast`/`predict`, `irf`/`fevd`, `roots`, `is_stable` and `lr_effects` methods for it; new `src/vecm.jl`). **Three agreeing references, both estimators:** ML matches `statsmodels`' `VECM` on all seven configurations (β, α, Γ, Σ, log-likelihood, residuals, `var_rep`, forecasts, and **all three standard-error blocks**) to `≤ 1e-11`, `urca::cajorls` + `vars::vec2var` on five (β, α, constant, df-adjusted Σ, levels-VAR matrices, forecasts), and `tsDyn::VECM(estim = "ML")`; `method = :twostep` matches `tsDyn`'s `"2OLS"` coefficients to `1.6e-15`. Shares `_johansen_rrr` and the five-value `deterministic` vocabulary with `johansen_test`, and the identity-normalized β agrees between the two at every rank and case. β standard errors from Lütkepohl (2005, Prop. 7.6), verified by hand and against `statsmodels`. Forecasts and impulse responses delegate to the 10.1/10.3 recursions through the levels VAR (no second recursion); a test rebuilds the VECM's residuals from that VAR, and the `K − r` unit roots are asserted. **Two reference details resolved:** `cajorls`' lag coefficients are `ca.jo`'s default `spec = "longrun"` form, `Γ*_i = Γ_i + Π` — the handoff's §7.2 test compared them to Γ directly and would have failed; the identity is asserted instead. And `tsDyn`'s 2OLS runs the cointegrating regression **without an intercept** when `include = "const"` (the constant goes in the ECM) — so `:twostep` puts unrestricted terms in the ECM and restricted ones in the static regression. Refusals: rank 0 or ≥ K (`statsmodels` returns an empty β / the identity), `lags = 0`, `:twostep` above rank 1 (`tsDyn` silently switches to ML), `lr_effects` (unit roots by construction), analytic/bootstrap IRF uncertainty (the 10.3 formulas are for an unrestricted VAR), and `:first_unit` at rank > 1, which belongs to Johansen's eigenvector basis. New `test/test_vecm.jl` (**418 tests**) | 10.5 | `urca::cajorls`, `vars::vec2var`, `tsDyn` (R) · `statsmodels VECM` (Python) |
| 10.7 | Structural VAR (SVAR) **✅ built** (`fit_svar`/`SVARModel` for the A-, B- and AB-models, `blanchard_quah`, and structural `irf`/`fevd` methods that reuse 10.3's recursion with `P = A⁻¹B`; new `src/svar.jl`). Free elements marked with `missing` (`NaN` refused by name — `statsmodels`' marker is the string `"E"` and `np.nan` silently estimates nothing). Concentrated likelihood at the df-adjusted Σ (what both references decompose); `:scoring` (Amisano-Giannini, the expected information) and `:mle` (`_optimize`) reach the same optimum; standard errors from the covariance-structure information. **Verified:** every just-identified scheme gives `P P' = Σ` to `1e-10`, the recursive ones reproduce the Cholesky factor, the two-parameter A- and B-models hit their analytic optimum `∓Σ₁₂/Σ₁₁` (to `1e-11`, where `vars` and `statsmodels` are within their own `~5e-9`), `vars`' standard errors and LR statistic match to `1e-12`, the trivariate A-model matches `vars` to `1e-6` with identical SEs, and Blanchard-Quah's B, long-run matrix, IRFs and FEVD match `vars::BQ` to `1e-12` with the long-run matrix's upper triangle exactly zero. **The handoff's identification premise was inverted:** `A = [1 0; NA 1]` with `B = I` has one free parameter for Σ's three distinct elements, so it is **over-identified** — `vars` itself rejects it, LR = 121.6 on 2 df (and prints p = 0; the tail is 3.9e-27) — **corrected 2026-10-14:** that is `vars`' short-form statistic `T(ln|Σ_r| − ln|Σ_u|)`, which drops `tr(Σ_r⁻¹Σ_u) − K` and is not a likelihood ratio when the structural variances are pinned (here it is `−T ln|Σ_u|`, unit-dependent; on India's trade VAR it is negative, p = 1); `lr_test` now computes `2(ℓ_u − ℓ_r)` in full, `31.8` on this fixture — and `statsmodels` reports the reduced form's log-likelihood as its SVAR `llf` whatever the restrictions, which is what made it look just-identified; the handoff's order-condition code also had the inequality backwards (more free parameters than K(K+1)/2 is under-identified). Order and rank conditions are both checked before optimizing. **An internal pitfall found here:** a ForwardDiff dual whose value is zero compares equal to zero, so the generic `\`, `det` and (on Julia 1.12) `lu` treated a free element at its zero start value as a structural zero and dropped its derivative — the information matrix came back singular, first on 1.9 and then, after an `lu` workaround, again on the 1.12 docs build. The score and the information matrix are now in closed form (Lütkepohl 2005, §9.3), with no automatic differentiation in the SVAR path at all. Blanchard-Quah refuses an unstable VAR (`vars::BQ` does not). Out, stated: sign restrictions, proxy SVARs, structural VECMs, and bands for structural IRFs. New `test/test_svar.jl` (**183 tests**) | 10.1, 10.3 | `vars::SVAR`, `vars::BQ` (R) · `statsmodels SVAR` (Python) |
| 10.8 | Multivariate residual diagnostics *(new row, added by the Stage 10 charter)* **✅ built** (`portmanteau_test`, `normality_test`, `bg_test`, `es_test`, a VAR method of `arch_lm_test`, and `var_diagnostics` running the set; result types `PortmanteauTest`, `MultivariateNormalityTest`, `BreuschGodfreyTest`, `EdgertonShukurTest`, `MultivariateARCHLMTest`; new `src/vardiagnostics.jl`). Definitions from Lütkepohl (2005) §4.4-4.5 and §16.5.1. **Every statistic and degrees of freedom matches `vars` on all seven fixtures** (K = 2, 3, 5, 6) to `≤ 2e-13` relative on the first run, including the Edgerton-Shukur denominator df — the one quantity with a single reference — `(20,364)`, `(20,370)`, `(45,666)`, `(100,736)`, `(75,622)`, `(8,36)`, `(108,2391)`; portmanteau (both forms) and multivariate JB **also match `statsmodels`**. The JB skewness and kurtosis halves, which only `vars` reports, sum to the joint statistic exactly — asserted on every fixture. **p-values never asserted against `vars`**, whose ten diagnostic p-values are all `1 − CDF`: on `climhyd` it prints 0 for BG and ES, where this package reports `4.3e-23` and `4.9e-32`. One handoff test was impossible: `climhyd`'s JB of 1850.9 on 12 df has a tail near `1e-391`, below `floatmin(Float64)`, so `0.0` is the honest answer (`statsmodels` returns 0.0 too); the test asserts that and that the skewness half's tail is positive. `arch_lm_test` gains a `VARModel` method rather than a second function, with its own result type, and the per-equation results reuse the existing `JarqueBeraTest`/`ARCHLMTest`. Refusals: portmanteau `lags ≤ p` (`vars` runs with `df ≤ 0`) and `lags ≥ T` (`statsmodels` accepts 300 on T = 198), infeasible ARCH and BG auxiliary regressions, any orthogonalization other than Cholesky. `C_0` factorized once, each lag two triangular solves. New `test/test_vardiag.jl` (**582 tests**) | 10.1 | `vars::serial.test`/`normality.test`/`arch.test` (R) · `test_whiteness`/`test_normality` (Python) |

*This entire stage can start as soon as Stage 1 is done — it doesn't wait on the state-space engine at all. Worth running in parallel with Stage 6–8 if resourcing allows, since it's a fully independent dependency branch.*

---

## Stage 11 — Advanced state-space applications (reuse Stage 6/8)

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 11.0 | Shared primitives *(new row, added by the Stage 11 charter, `handoff/stage 11/`)*: multivariate-observation Kalman filter and smoother (11.0a/b), partitioned initialisation (11.0c), stationary VAR reparameterisation (11.0d), zero-weight `_ols` (11.0e). **11.0c and 11.0e ✅ built** (with 11.1, which needs them): `SSMInit`/`_resolve_init` turn block schemes (`:approximate_diffuse`, `:stationary`, `:known`, `:exact_diffuse`) into the `(a0, P0, diffuse_idx)` the existing filters already take -- the filters are unchanged; `stationary_cov`'s doubling core factored out as `_stationary_cov(T, RQR)` so it applies to a sub-block, `stationary_cov` now a one-line wrapper with identical output. `_ols`/`_mols` accept zero weights, which EXCLUDE their rows (degrees of freedom count positive weights; equal to the subset regression exactly) -- defect D7, ours -- rather than clamping at 1e-20 as statsmodels does; fewer than `k + 1` positive weights refused naming the count. **11.0a/b ✅ built** (`src/statespace/multivariate.jl`): `kalman_filter`/`kalman_smoother` methods on an `n × N` matrix `Y` -- the sequential treatment, each series a scalar update and the transition once per period, so no `N × N` factorisation and missing values skipped in the loop; a dense `H` is refused, a non-positive `f` throws naming period and series (an internal `_kf_mv(…; strict=false)` returns the sentinel for optimisers). **Fixture A at statsmodels' full-precision parameters: 1.6e-12 from statsmodels, 1.4e-11 from `dfms`' C++ filter**; fixture B (r = 2) and the 12-NaN fixture to 1e-9; the conventional matrix filter (a test helper only) to 1e-12. N = 1 reduction to the univariate method at one ulp per step -- not bit-exact as the handoff asked, and it cannot be: the univariate method folds `T` into the gain, the sequential form transitions after updating. Disturbance identity `alpha_{t+1} = T alpha_t + R eta_t` to 1e-12; series-permutation and state-rotation invariance; no allocation in the time loop. **Corrected:** the handoff's §6.1 parameter table is from a different fit than the likelihood it quotes (−1049.53340809606 at the table's values); and its §2.5 has the H → 0 behaviour backwards -- statsmodels' univariate filter's −696.17 at σ² = 1e-12 is a silent drop of near-degenerate updates, and the conventional −1.4e14 is the true value (appended to the charter as defect D17). **11.0d ✅ built** (`src/statespace/stationaryvar.jl`): Ansley–Kohn through normalised partial autocorrelations with the *innovation* variance fixed (fixing Γ₀ instead could not reach every stationary VAR -- at p = 1 only norm-below-one matrices), Cholesky factors throughout so ForwardDiff passes; inverse by Newton continuation along `A_j → t^j A_j`, which scales every companion eigenvalue by exactly `t` (a plain Newton start failed on up to half of random VAR(2)s near the unit circle). Every one of 1,800 draws stationary; coefficients round-trip to 1e-12, `u` to 1e-9; `k = 1` reduces to `u/√(1+u²)` exactly. New `test/test_ssm_multivariate.jl`, **9 testsets, 77 tests**. | 8.1, 8.2, 10.0 | statsmodels `KalmanFilter`, Durbin & Koopman §6.4 |
| 11.1 | Unobserved Components / structural time series **✅ built** (`fit_uc`/`UCModel`, `uc_decomposition`/`UCDecomposition`, named models `uc_local_level`, `uc_local_linear_trend`, `uc_random_walk`, `uc_smooth_trend`, `uc_bsm`; new `src/uc.jl`). No new numerics: a component specification mapped onto `(Z, T, R, Q, H)` for the existing `TimeVaryingSSM` filter and smoother. **Explicit specification** (`level`/`trend`/`cycle` each `:none | :deterministic | :stochastic`) -- statsmodels' `level=True` means a *deterministic* level and its string names disagree about the irregular (defect D4); a specification with no disturbance is refused, not patched. **Matches statsmodels to ~1e-10** at fixed parameters on the Nile (-632.537695, -632.53768559, and -632.5456059039 at R's prior), and R's own filtered/smoothed state to 1.3e-8/1.0e-8 (R's print precision); the fitted optimum is the one four of statsmodels' optimisers reach. **The likelihood burn was measured, not assumed:** statsmodels leaves out one period per approximate-diffuse state (k_states minus the AR order, counting a trig seasonal's degenerate sine state) and keeps that burn under `initialize_known`; matched. UKgas BSM (boundary): log-likelihood 172.46528 >= statsmodels' 172.46517, level variance 5e-17 -- **strictly positive, never clipped** (R's `StructTS` reports exactly 0 via `pmax`, defect D3). Five reductions: deterministic level = OLS mean and ddof-1 variance; variance limits; dummy = full-harmonic trig seasonal (odd period -- at even periods the reference's degenerate sine state changes the burn); **AR-only = `fit_arma` exactly** (-637.98136712618 both). **Found:** smoothed variances over the burn periods are cancellation noise under kappa = 1e6 -- statsmodels' own are negative there (to -0.97 on UKgas) -- so `uc_decomposition` reports them `NaN` rather than clipping to a false zero. Regression coefficients are fixed (drifting ones stay with `fit_arimax(…; model=:tvss)`); `vcov` is the inverse observed information and refuses when it is not positive definite. New `test/test_uc.jl`, **15 testsets, 126 tests**. **Fixed 2026-10-15, found writing the manual:** a single start let the optimiser slide the log-variances towards -Inf along a flat ridge -- the trigonometric-seasonal BSM on log(jj) reported 56.35, not converged, where statsmodels reaches 80.5249 (our likelihood at its optimum agrees to 8e-7). Three starts (component variances at 1, 1e-2, 1e-4 times the default relative to the irregular), run on threads, best taken in start order; convergence judged by the gradient at the chosen optimum, since the line search can stall on the ridge at the optimum itself. Threading the starts first exposed a data race of our own -- closures assigning names the enclosing function also assigned, so Julia boxed one variable shared by every thread (four different optima in six runs); `local` fixed it, and a test now requires identical results across repeated threaded fits. Warm cost: the dummy-seasonal BSM 0.88 s -> 1.55 s, the Nile local level 0.12 s -> 0.18 s. **Also found writing the book:** a variance held at exactly zero by `fixed` left the likelihood finite but its ForwardDiff derivative NaN, which crashed the line search (AssertionError); the objective now treats a non-finite derivative as infeasible. And **Tsay (2010) Example 11.3 reproduced**: log J&J, local level + dummy seasonal -- sigma_eta 0.072696549 and sigma_omega 0.029316907 against Tsay's printed 0.07269655 and 0.02931691 (S-PLUS/SsfPack), about 1e-7, closer than statsmodels (1.8e-5); the irregular is at its boundary in all three and is not asserted. | 8.1 (`TimeVaryingSSM`), 11.0c -- *not* the exact diffuse filter (8.2), which the reference does not use | statsmodels `UnobservedComponents` (primary) · R `StructTS` (check only, GPL) |
| 11.2 | Dynamic factor models **✅ built** (`fit_dfm`/`DFMModel`, `factors`, `rotate_factors`, `factor_r2`, `series_r2`, `bai_ng_ic`, and `standardize` in `src/transforms.jl`; new `src/dfm.jl`). Direct ML on the sequential multivariate filter, the factor VAR kept stationary by 11.0d, the stationary prior recomputed at every evaluation. **Identified explicitly** -- `Q = I` and a lower-triangular leading loadings block with positive diagonal -- which neither reference does (statsmodels fixes only `Q`; `dfms` and `DynamicFactorMQ` fix nothing): **20 random starts on fixture B all reach the same loadings to 1e-8**, with one parameter fewer than statsmodels and its maximum (2.3e-6 higher). The diagonal is free during the search and sign-normalised after: holding it positive by `exp` trapped random starts at a zero diagonal. Fixture A at fixed parameters 1e-10 from statsmodels; fitted parameters 4e-5 relative, AIC/BIC, and statsmodels' `coefficients_of_determination` (= our `factor_r2`, squared correlation) to 1e-4; fixture B aligned to statsmodels by Procrustes -- the recovered rotation is a **reflection** (det = −1), as the handoff warned; missing data: statsmodels' fitted −1035.826629949621 reproduced. Bai-Ng IC1/IC2/IC3 match `dfms`' `ICr` to 8 digits, and the docstring says they keep falling past the true r at N = 6. **Not done as the handoff proposed:** `standardize` is a separate step, never inside the fit (the likelihood is not scale-invariant); `r >= N`, `N < 2` and `p < 1` are refused, so the handoff's `N = 1` and static-factor reductions do not apply (`p = 0` means white-noise factors in statsmodels and is an error in `dfms`); EM deferred with the handoff's measurement (direct ML 35 iterations against `dfms`' 2187); our PCA start is −1823 against statsmodels' −1699, and converges to the same optimum. The handoff's §6.1 parameter table is from a different fit than its likelihood (the full-precision values are in `ground-truth/sm_results.json`). New `test/test_dfm.jl`, **12 testsets, 87 tests**. | 11.0a/b/d (corrected by the Stage 11 charter: not 10.1) | `dfms` (R) · `statsmodels DynamicFactor` (Python) |
| 11.3 | Markov-switching AR/regression **✅ built** (`fit_ms_regression`, `fit_ms_autoregression`, `MSModel`, `regime_probabilities`, `expected_durations`, `canonicalise`; new `src/markovswitching.jl`). Log-space Hamilton filter over the `k^(p+1)` joint regime states of the **mean-adjusted** autoregression (the striding an explicit index map, tested alone), Kim smoother, closed-form multinomial-logit transition (the reference root-finds its inverse for k > 2), EM (ECM: AR given means, means given AR, by zero-weight-capable `_ols`) as a warm start, then BFGS from `n_starts` threaded starts, **reporting how many distinct optima were found**. **Hamilton's GNP model at statsmodels' parameters: log-likelihood 8.5e-14 from statsmodels (= E-views' −181.26339), filtered and smoothed probabilities 1e-15** (the handoff asked 1e-6); durations, steady state, AIC/BIC exact. Fitted: the global maximum to 1e-6 and parameters to 1e-4, with the −183.669 no-persistence ridge found and reported. Order 0 = statsmodels' `MarkovRegression` (−191.28811083) exactly; `k = 1` = least squares; time-varying transition probabilities match statsmodels' `loglike` to 1e-9 on a documented synthetic covariate (our own ground truth -- the handoff's −185.44 was not reusable); with a switching variance the better optimum (−180.677285, 1.36 above the reference's own default fit) is reached. `canonicalise` uses the non-obvious k = 2 swap `(a, b) → (1 − b, 1 − a)`; a fully non-switching specification is refused (statsmodels returns its start values, D11). **Performance:** a closure mutating a captured index (boxed by Julia) made the parameter unpacking type-unstable -- 10.8 s and 12 GB for a default fit; rewritten with explicit offsets, 2.65 s and 1.6 GB. `_uc_unpack` had the same pattern and was fixed with it. The handoff's printed probability sums come from its 10-digit parameter table (1e-7 off the ground-truth files). New `test/test_markovswitching.jl`, **10 testsets, 207 tests**. | 4.1, 5.1, 11.0e (zero-weight `_ols`) | `MSwM` (R) · `statsmodels MarkovAutoregression` (Python) |
| 11.4 | Threshold AR (TAR/STAR) **✅ built** (`fit_setar`, `SETARModel`, `select_setar`, `tsdyn_ic`, `fit_lstar`, `LSTARModel`, `setar_test`, `ThresholdTest`; new `src/tar.jl`, and `TSAnalytics.LYNX`, R's 114-point `datasets::lynx`, in `test_data/`). SETAR with one or two thresholds (the two-threshold search **sequential**, as the reference, not a joint grid), per-regime orders and non-contiguous lag sets, `include = :none/:const/:trend/:both`; the threshold search updates the two regimes' cross-products incrementally (rank-one BLAS updates as each candidate crosses), and agrees with a naive refit at every candidate. **log10(lynx): threshold bit-identical to tsDyn (3.3100557377508912), SSR 1.8e-15, coefficients 2e-15**; thDelay 0, sunspot m = 11, the two-threshold fit (both thresholds, 9 coefficients, counts 40/38/34), `mL ≠ mH` and `ML = [1,3]` all to 1e-10; the simulated fixture recovers its DGP within 1.3 se; the `aeon`-matched sample to 1e-12. **Standard errors**: ours divide by `rows − k`; tsDyn's `SSR/(n − p)` -- the full series length, rows the regression never saw -- is kept in `se_tsdyn` and `tsdyn_ic` (AIC/BIC on n) so its numbers are reproduced exactly; the threshold gets no standard error (non-standard distribution) where tsDyn zero-pads one. **Candidate grid**: every sorted value of z *including ties*, as tsDyn's `makeTh`; `grid = :tsdyn` additionally reproduces its fractional-index truncation (74 of 75 values on lynx), only so its selection tables can be matched. `select_setar` matches `selectSETAR`'s 592-row table, best row and top five exactly -- after re-deriving its pooled AIC, which fits each regime **separately on a sample embedded at max(thDelay + 1, mL, mH)** and counts the variance (the handoff's description gave the wrong numbers). `show` prints the threshold at full precision: the 12-digit print typed back moves a boundary observation and changes the SSR (4.348 → 4.422). LSTAR in tsDyn's **additive** form (the convex form is 67× worse at its estimates) with the slope on the raw `z − c`: grid start then BFGS; **we reach SSR 4.33764090727 against tsDyn's 4.33764323215** (it stops short on the flat γ profile; its SSR is reproduced exactly at its parameters, and its numerical-Hessian standard errors to 2e-4); a γ past the grid maximum is reported. Hansen's (1999) bootstrap test: SSRs and F statistics exact, `F = rows·(S_i − S_j)/S_j`, default 1000 replications (tsDyn's 10 are unstable). A unit-root warning when a regime's AR has a root inside the unit circle (lynx's high regime, 0.994262). **Found on the way:** tsDyn's `linear()` uses a third variance denominator, `(SSR/n)·rows/(rows − k)` (charter D18); a scalar `threshold=` crashed on `sort`; a saturated LSTAR transition made the profile rank-deficient (now scored infeasible). **Julia 1.12 hazard found by the doctests:** 1.12's generic matmul skips entries for which `iszero` holds, and ForwardDiff 0.10's `iszero` ignores the partials, so a Dual with value 0 loses its derivative -- the stationary-VAR inverse's Jacobian was singular at u = 0 and `fit_dfm` threw on 1.12 only; fixed locally, the general exposure noted for review. `selectLSTAR` and the MTAR variant are deferred (out of the handoff's scope table; no ground truth for the former). Doctests added to every Stage 11 fitting function. New `test/test_tar.jl`, **32 testsets, 338 tests**. | 5.1 | `tsDyn::setar` (R) — weak Python support |

---

## Stage 12 — Niche but self-contained (schedule opportunistically)

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 12.1 | Intermittent demand **✅ built** (`croston` with `method=:classic/:sba/:optimized`, `tsb`, `adida`, `imapa`, `IntermittentModel`, `IntermittentForecast`, `demand_intervals`/`DemandIntervals`, `adi_cv2`/`ADICV2Result`, and `cfe`, `spis` in accuracy.jl; new `src/intermittent.jl`). Scope widened from three methods to five plus the classifier, as the charter says. **All 18 point forecasts (six methods x three fixtures) equal statsforecast's bit for bit**, classical Croston also R's `forecast::croston` to all 17 digits; demand/interval levels to 1e-15; the optimised alphas including both bound hits (0.29999999999967963, 0.10000000000032039) and the constant-series tie at exactly 0.2 -- by the same bounded golden-section search with the same stopping rule (read from statsforecast's Apache-licensed `ses.cpp` to settle the details; SES starts at the first value, as both references do). IMAPA's mean over eight levels needed numpy's pairwise summation order to agree to the last bit. Classic fitted values in one O(n) pass match R's O(n^2) per-prefix recomputation to 3.5e-15; in-sample MASE for all six methods matches the handoff to 2e-16 except IMAPA on F1, where a prefix mean interval is exactly a half: statsforecast rounds it half to even for the forecast and half away from zero for the fitted values (D26); this package rounds half to even throughout, and a test pins both values (0.42114 at width 2, 0.50681 at width 3; the half-away fitted path reproduces 0.88322692682056458). **SBA's factor is 1 - alpha/2**, not statsforecast's literal 0.95 (D24). **Refused:** negatives (D25), a single demand at t = 1 followed by zeros -- both references return 5 for [5,0,...,0], a 10x error against the empirical rate -- and SBA on one value; all zeros forecast zero by an explicit branch. No prediction intervals: `IntermittentForecast` is its own type, since no reference computes any. ADI/CV² cutoffs 1.32/0.49 as in tsintermittent's idclass; the quadrant names are the paper's, said so. **Found:** utilsforecast's `cfe` computes the opposite sign to its own docstring (D35); ours is sum(actual - predicted), positive for under-forecasting. New `test/test_intermittent.jl`, **23 testsets, 269 tests**. | 5.5 | `forecast::croston` (R) · `statsforecast` (Python) |
| 12.2 | Change-point detection (PELT, binary segmentation) | 1.4 only | `changepoint` (R) · `ruptures` (Python) |
| 12.3 | Lag-feature regression forecasters (ML reduction) **✅ built** -- rescoped by the Stage 12 charter (`handoff/stage 12/`): `arx` + `forecast` already *was* the recursive reduction, so nothing re-derives a lag matrix. `arx`'s design construction is factored into one `_arx_design` that `arx`, the new `lag_regression` and the direct strategy all call. **Added:** (1) `lag_regression(y, lags; fit, predict, strategy, ...)` / `LagRegressionModel` -- any regressor injected as a `(fit, predict)` pair of functions (forecastML's contract; a fitted object is refused), OLS by default, and then **recursive = `arx` + `forecast` to 4e-16** and = mlforecast / sktime / forecastML to 2e-15 (core_ols.txt); (2) **future regressors in `forecast(::ARXModel)`** (`newexog`, also positional) -- `arx` + `fourier_terms` now forecasts, matching mlforecast to 3e-14 on fixture B; a shortfall is refused naming the columns and rows (mlforecast silently accepts 3 of 6 rows, sktime pads, D23); (3) **past-only regressors** (`past_exog`, `past_lags`, named `past1.L1`), with the hold-back covering their lags, matching mlforecast's `transform_exog` to 1e-13; (4) the **direct strategy**, origin-anchored (sktime's) -- each horizon on its own rows reproduces sktime `windows_identical=False` to 3e-16 at every h, `same_rows=true` reproduces its `make_reduction` default (h=1 coefficients 1.005639629996196 ...) to 1e-15, and both equal forecastML at h = H; DirRec and multi-output asserted as identities with Direct under OLS (1e-13), not built; (5) `rolling_feature` (`:mean/:std/:min/:max/:expanding_mean`, lag 0 refused at construction -- sktime's lag-0 window IS y_t, R^2 = 1, D21), recomputed from forecasts under recursion, matching mlforecast's RollingMean/Std/Max/ExpandingMean to 1e-13 (run here); (6) `blocked_window_split` = forecastML's `create_windows` (97 = 117 - 20, skip, partial); (7) `mdape`, `rmsse`; (8) `tscv(y, fit_fn, forecast_fn; refit=true/false/k)` with `forecast(m, h; history=...)` -- all twelve mlforecast cross-validation predictions to 1e-9, refit=false and refit=2 differences 0.05061 and 0.02185 exactly. `arx` and `rolling_feature` refuse lag 0; the rank check refuses the D33 dummy trap and a deterministic sinusoid. New `test/test_lagregression.jl`, **25 testsets, 256 tests**; the existing arx/forecast/tscv/accuracy suites unchanged (695 together). | 5.2, 5.3 | `forecastML` (R) · `mlforecast`, `sktime` reduction (Python) |

*(Feature extraction, formerly listed here, moved to its own tracked
section — see "Downstream: TSFeatures.jl" right after Stage 5.)*

---

## Textbook dataset catalogue (cross-cutting, not a numbered stage)

**✅ built** (`handoff/ts-datasets/ts-datasets-handoff.md`) — `dataset`/
`datasets`/`dataset_info`/`DatasetInfo` (`src/dataset_catalogue.jl`),
API shape verified directly against `SeasonalAdjustment.jl`'s own real
`src/datasets.jl` (fetched this session): `DatasetInfo`'s 13 fields and
the function signatures match exactly, with a fourth `kind` added —
`:external`, for real datasets that are catalogued but deliberately not
bundled. **90 catalogued datasets**: 83 bundled directly (71 from
`astsa`, Shumway & Stoffer's real companion package; 12 from
`tsibbledata`, fpp3's), plus 7 verified `:external` entries (Tsay,
Cowpertwait — see `data/DATASETS.md` for the full breakdown, including
why Montgomery has none and Hamilton has no datasets at all).

**A real, deliberate licensing decision, not inferred**: both source
packages are GPL-3; TSAnalytics.jl itself is MIT. Whether bundling
GPL-3 *data* into an MIT package triggers copyleft on the surrounding
code is a genuinely contested question. The user was presented with
three concrete options (separate GPL-3 companion package; bundle
directly; fetch-on-demand with no redistribution) and explicitly chose
to **bundle directly** — this project does not take a position on the
underlying legal question itself, only reports the decision that was
made and by whom (`data/DATASETS.md`).

**A real bug found and fixed, not merely worked around**: `astsa` has
two genuinely distinct object pairs differing only in case (`GDP`
305 obs vs. `gdp` 287 obs; `GNP` 304 obs vs. `gnp` 223 obs). On this
project's case-insensitive development filesystem, `GDP.csv`/`gdp.csv`
silently collided to one file during the original bundle's own
creation — before any TSAnalytics code was written — losing `GDP`'s
real data (only `gdp` survived) and `gnp`'s real data (only `GNP`
survived). Confirmed by direct inspection, then fixed by installing the
real `astsa` package (R, freshly confirmed reachable and installable —
same recurring "assumed unreachable, actually isn't" pattern as
several earlier stages) and regenerating both lost objects with
collision-safe filenames; verified no *other* name among the 83
collides the same way (checked systematically, not just for this pair).

**Genuine data heterogeneity handled honestly, not glossed over**:
several bundled datasets' `time`/`index` columns are a plain sequential
position, not calendar time at all (a seismic trace's sample index
starting at `1,2,3,...`; a nucleotide sequence position) — converting
those to a fabricated calendar `Date` would have been actively
misleading. `DatasetInfo.span` is `Union{Nothing,Tuple{Date,Date}}`
(a deliberate divergence from `SeasonalAdjustment.jl`'s own plain
`Tuple{Date,Date}`) and `frequency` is `0` for exactly these cases,
rather than a guessed value — matching the same "record the gap, don't
invent" standard already applied to `units` (honestly `"unknown
(...)"` for all 90 entries, since the source packages' own `man/*.Rd`
files weren't individually read this session).

**Per the user's own explicit follow-up instruction, folder
reorganization done as part of the same task**: `data/` now holds only
this export-eligible catalogue (83 CSVs in one flat folder, no
per-package subfolders — flattened per a later explicit user
instruction — plus `data/inventory.tsv` and `data/DATASETS.md`); the
pre-existing internal
validation-fixture series (`nile.csv`, `airpassengers.csv`, etc. —
genuinely different in kind, "not part of the modeling API" per their
own existing docstring comment) moved to a new `test_data/` directory.
`src/datasets.jl`'s `TSAnalytics.NILE`-style path constants were
updated to point there; no test files needed to change, since they only
ever referenced those constants symbolically, never a hardcoded path.
`DelimitedFiles`/`Dates` promoted from test-only to real `[deps]`
(both Julia stdlibs — a lightweight, safe promotion, same precedent as
`Random`'s own earlier promotion for GARCH's multi-start feature).

**⚠️ known gap, pre-existing, not introduced by this work**: CI's
`test (1, *)` and `test (nightly, *)` jobs fail on all three platforms
(ubuntu/macOS/windows) while `test (1.9, *)` passes on all three — this
exact pattern already existed on the CI run immediately prior to this
catalogue's own commit (`34a77e3`, checked directly via the GitHub API,
not assumed), so it isn't something this session's dataset-catalogue
code caused. Only Julia 1.9 is installed on this dev machine, so it
can't be reproduced or root-caused locally yet; the GitHub Actions job
log text itself also wasn't reachable (the REST API's log-download
endpoint requires authenticated access even for public repos, and this
session has no `gh` CLI or token available). Left unresolved pending
either a newer local Julia install or authenticated CI log access.

| Depends on | Reference |
|---|---|
| 1.1 only | `SeasonalAdjustment.jl`'s own real `src/datasets.jl` (Julia, API shape verified directly) · `astsa`/`tsibbledata` (R, real companion packages, both GPL-3) |

---

## Plot recipes (cross-cutting, not a numbered stage) — ✅ built 2026-10-14

From `handoff/stage 10/plot-recipes-handoff.md`, in its build order. `RecipesBase` only — no plotting
dependency — following the package's four established patterns (A: a recipe on an existing result
type; B: a multi-panel recipe on stored fields; C: a computing function, a panel struct, a recipe;
D: data only, deliberately no recipe). Numbers are tested in `test/test_plotdata.jl`, series counts in
`test/test_recipes.jl` — every count asserted, never just non-emptiness.

| # | What | Pattern | Notes |
|---|---|---|---|
| 1 | **`Forecast` fan chart** — `plot(fc; history = y)` ✅ | A | One ribbon per level, widest first; one recipe for every forecasting family (ARIMA, ETS, Theta, TBATS, benchmarks). `history` is taken from the plot attributes rather than as a recipe keyword, because RecipesBase's keyword handling needs a loaded backend (`is_key_supported` is defined by Plots) and the backend-free tests would fail. Long histories thinned by a new **min-max** decimation — `_decimate_local_max` keeps maxima only, right for a spectrum and wrong for a line, which needs its troughs too. |
| 2 | **`state_decomposition`** (ETS, TBATS, Holt-Winters) → `StateDecomposition` ✅ | C | R's `plot.ets` panels exactly — at most four, the seasonal one being the current state `s1` (`[S]` verified in `forecast:::plot.ets`). **The handoff's TBATS claim was corrected:** `plot.tbats` does not take one harmonic per period (that is `plot.bats`); it calls `tbats.components`, which sums the period's harmonics through the measurement vector from the state one step back — matched, and pinned by the identity *one-step mean = lagged level + slope + Σ seasonal panels*. Observed is rebuilt from the model (`fitted ⊙ (1 + resid)` for M-error ETS, `boxcox(fitted) + resid` for TBATS, whose fitted values are back-transformed but residuals are not). `_tbats_state_path` factored out of `_tbats_final_state` (one loop for both). |
| 3 | **`ts_display`** ✅ and **`arma_roots`** ✅ | C | `ggtsdisplay` (7 series); inverse AR/MA roots of the full multiplied-out SARIMA polynomials via `combined_ar_ma` (an empty polynomial emits no series). |
| 4 | **`seasonal_plot`, `lag_plot`** ✅ | D | Data only, with the refusal in `seasonal_subseries_plot`'s words. |
| 5 | **`VolatilityForecast`** ✅ | A | Plots σ, labelled as such; a 5–95% band only when simulated paths exist — the analytic forecast has no interval and none is invented. |
| 6 | **Stage 10:** `IRFResult` (band labelled by method — bootstrap percentile or ±1.96 se analytic), `FEVDResult` (stacked bars), `VARForecast` (shared fan helper), `diagnostic_plot(::VARModel)` (K × 4, a `DiagnosticPlotResult` per equation), `cointegration_plot` (relations + ACF), `roots_plot` (VAR/VECM companion eigenvalues) ✅ | A/C | **FEVD:** the handoff's `bar_position = :stack` is a no-op in core Plots (`[V]` GR warns and ignores it), so the stack is built by hand — each shock's bar on the cumulative share below it, rendered and inspected. `coint_relations(t, r)` as specified could not work (a `JohansenTest` keeps no data): `cointegration_plot(Y, t)` and `cointegration_plot(m::VECMModel)`. |

**A real bug found here, in Stage 9's ETS forecast.** The forecast rebuilt the final seasonal buffer
as `s + γ·resid`, but the fitter updates with the **additive** error, and for a multiplicative-error
model the stored residual is the relative one. ETS(M,A,A) and ETS(M,N,A) forecasts therefore started
from seasonal states off by over 100 on AirPassengers — h = 1…12 points off by up to **128** and
**157** on a series ending at 432 — and ETS(M,·,M) was off by a factor of the seasonal state in the
update (negligible there only because γ is tiny). `_ets_final_seasonal` now takes `s_{t−m}` from the
stored path plus the fitter's own update; a regression test runs the fitter one step past the data
and requires the one-step forecast to equal its fitted value at `n + 1` for eleven model forms — the
independent check that none of the Stage 9 tests had made. Found because the decomposition's seasonal
panel and the forecast's starting state had to agree.

Not built, stated: structural-break/CUSUM panels (`vars::plot.varstabil` — a test wearing a plot's
clothing; no break machinery exists), Sims-Zha IRF bands, interactive output. **Correction of record:**
an earlier remark in this project's conversation (not the repo) said `plot.ets` draws one panel per
state; it draws at most four, using `s1` alone.

### Stage 11 plot recipes — ✅ built 2026-10-15

From `handoff/stage 11/docs/stage-11-plot-recipes-handoff.md`, same patterns, same rule that every
plottable output is a named struct (never a recipe on a bare `NamedTuple`). Twelve new result types and
recipes plus one extension; tested in `test/test_plot_stage11.jl` (**14 testsets, 177 tests**) —
the data against fixture figures, band arithmetic by hand at three periods, panel counts and order.

| # | What | Notes |
|---|---|---|
| 1 | **`StateDecomposition` extended** (11.1) | Optional `level_se`/`slope_se`/`seasonal_se`, further components in `extra` (cycle, AR, regression, irregular) and one-step `fitted`; `state_decomposition(::UCModel)`. Ribbons at ±1.96 se labelled "95% interval, smoothed component"; the irregular gets none; fitted overlaid on observed, blank over the diffuse periods. **The ETS, TBATS and Holt-Winters renderings are byte-identical to before** (recipe output serialised before the change and compared after). `level` may now be `nothing` (an AR-only UC model). |
| 2–6 | **`factor_paths`, `loadings_plot`, `factor_r2_plot`, `factor_selection_plot`, `factor_residual_plot`** (11.2) | Factor bands and the loadings heatmap exist in neither reference. Observations × factors, asserted. The R² chart draws per-factor bars with the joint R² as a marker — they are different quantities (with one factor the joint R² uses Λ, the per-factor one a free slope, so joint ≤ per-factor). The selection plot says "not asymptotic" on the plot when N < 20. Box plots drawn from stored five-number summaries (no StatsPlots dependency). |
| 7 | **`regime_plot`** (11.3) | Two modes: smoothed + filtered overlaid per regime (titled with the expected duration), and `mode = :shaded`. The last filtered and smoothed probabilities coincide, asserted. New `residuals`/`fitted` for `MSModel` (the residual averaged over the smoothed regime distribution; = OLS for k = 1) and `diagnostic_plot(m; regime)`. |
| 8–12 | **`threshold_regime_plot`, `threshold_phase_plot`, `threshold_profile`, `transition_plot`, `plot(::ThresholdTest)`** (11.4) | The boundary observation (`regime[63]`, z = th) drawn low, asserted. Phase plot for three regimes (tsDyn draws none). Profile for SSR or AIC (tsDyn refuses AIC). The transition curve notes a γ at the grid bound. The Hansen test now stores its bootstrap draws (`boot`) and plots them with the observed F and critical values — **no χ² curve**. `residuals`/`fitted` for `LSTARModel`. |
| — | **Dispatches** | `diagnostic_plot` on UC, MS, SETAR, LSTAR; `ts_display` on MS, SETAR, LSTAR. |

**Not built, stated:** the threshold confidence set (Hansen 2000) — gated on a scope decision that was
not taken, and a shaded region with no stated coverage would mislead; the pairwise regime-transition
probabilities (no single natural rendering — the `seasonal_subseries_plot` precedent);
`MSwM::plotReg`; tsDyn's `autopairs`/`autotriples`; average mutual information.

**Rendered and inspected by hand**, all of them, with GR (the check that cannot be automated). It
caught what the data tests could not: subplot attributes (`title`, `ylabel`) are set by the *last*
series drawn into a panel, so a title set only on the first series vanished from the regime, transition
and Hansen panels and the "observed" label from the UC decomposition; the UC fitted line plunged to
zero over the diffuse periods; and the R² legend hid a marker.


## Stage 9B Tier 1 — finishing what was started (cross-cutting)

**✅ built (2026-10-01)** (`handoff/stage-9B-nonets-handoff.md`, whose
own Status header records the seven corrections verifying its premises
turned up).

- **`predict`/`forecast` for `ArimaxModel`/`SarimaxModel`.** The
  estimation was already right -- `fit_arimax` matched R's
  log-likelihood to all twelve digits before this work -- so the gap was
  purely the forecast. Regression and ARMA error forecast separately and
  added on the differenced scale, then integrated back up: the same
  decomposition the fit uses, so the two cannot drift. Against R's
  `predict(m, n.ahead=10, newxreg=xf)`, points agree to `3e-06` and
  standard errors to `3e-07`, closer to R than `statsmodels` is.
  Intervals treat the supplied regressors as **known**, matching both
  references. `model=:tvss` is refused by name, since `beta` there is a
  latent state needing a projected path and a second variance term.
  Two reductions hold: zero exog reproduces `fit_arima` to `9e-16`, and
  the seasonal path reproduces `fit_sarima` **bit-identically**, which is
  what confirms the SARIMAX psi-weight polynomial multiplies both
  differencing operators back in.
- **Uniform `predict(model, horizon)`.** `ArmaModel` and `SarimaModel`
  now retain `original_y`, so `predict`, `residuals` and
  `diagnostic_plot` all work from the model alone on every fitted type.
  `ArmaModel` gained `predict` at all, delegating to the `ArimaModel`
  `d=0` path. Three-argument forms kept for callers who want to name the
  series; every doc call site migrated.
- **`ccf`** -- R's two-sided convention (lag `k` is `cor(x[t+k], y[t])`,
  so a negative peak means `x` leads), matching R at all thirteen lags to
  `5e-11`. Returns an `ACFResult` tagged `:ccf`, so the existing recipe
  serves it. `statsmodels`' one-sided output genuinely omits the peak
  when `x` leads `y`; documented rather than worked around.
- **`pacf(method=:burg)`** -- Burg's (1967) recursion implemented from
  the textbook statement, **dual-verified**. R's `pacf()` does silently
  ignore its own `method` argument -- all four options return
  bit-identical output equal to this package's `:ywm`, confirmed by
  execution -- but that does **not** mean R offers no reference, which is
  what the first version of these notes claimed. R's real Burg estimator
  is `ar.burg()`, and the last AR coefficient of an order-`k` Burg fit
  *is* the lag-`k` partial autocorrelation:

      sapply(1:6, function(k) tail(ar.burg(y, order.max=k, aic=FALSE)[["ar"]], 1))

  R and `statsmodels.pacf_burg` agree on that to `4e-11`, and so does
  this package, so the test tolerance is `1e-9` rather than the `1e-7`
  first used. All four `pacf` methods have two references each.

  **Found by re-running the shipped script.** `verification/stattools/burg.R`
  did not execute as committed -- it still referenced `ets_y.csv` after the
  fixture was renamed `burg_y.csv` on the way into the test tree. A
  reference script that cannot be run is worth little; fixed, and running
  it is what surfaced the `ar.burg` reference.
- **`stl_decompose(seasonal_window=:periodic)`** -- R's `s.window="periodic"`,
  which is not a separate algorithm: R sets `s.window=10n+1`,
  `s.degree=0`, decomposes, then averages each cycle position and
  recomputes the remainder. Read from R's own `stl.R`; the post-hoc
  average is what makes the repetition exact. Matches R's `s.jump=1` run
  to `1e-8` once `inner=2` is passed (this package defaults to `5`,
  matching `statsmodels` -- the pre-existing documented divergence).
- **`mstl_decompose(lambda=:auto)`** and the new exported
  **`boxcox_lambda`**, the bounded profile-likelihood MLE.
  `scipy.stats.boxcox` is unbounded and returns `-1.836` on the fixture;
  bounded on `[-1, 2]` -- matching R's `BoxCox.lambda` and this package's
  own `guerrero_lambda` -- the answer is the boundary, and widening the
  bounds reproduces scipy to `1e-5`. `MSTLDecomposition` gained a
  `lambda` field reporting what was used.
- **Housekeeping.** The Newey-West Bartlett long-run variance, written
  out twice inline with *different* accumulation order (KPSS divided each
  autocovariance by `n` inside the loop, PP summed raw and divided at the
  end), is now one `_bartlett_lrv` -- verified bit-for-bit against
  statistics captured before the refactor. `abstract.jl`'s
  `StateSpaceModel` note no longer claims the Kalman engine is "planned,
  not yet implemented".

---

## Stage 9B Tier 2.1 — seasonal unit root detection (cross-cutting)

**✅ built (2026-10-01)** as `nsdiffs`/`ocsb_test`/`seasonal_strength`
(`handoff/stage-9B-nonets-handoff.md` §2.1). `auto_arima(y; seasonal=true,
m=12)` is now **fully automatic** -- `D` is detected when omitted, closing
the limitation Stage 6.8 recorded and the docs described in five places.

**Dual-verified, which the handoff believed impossible.** It assumed CRAN
was unreachable and the stage would be single-verified against `pmdarima`.
R's `forecast` 9.0.2 turned out to be *already installed* on this machine
(the same pattern as Stage 5.2 and 7.4: a reference assumed unavailable
that was reachable when actually tried), so `ocsb.test`, `seas.heuristic`
and `nsdiffs` were all read from R's own source and executed directly.

**The handoff's premise was wrong about which test R uses.** It said
"Canova-Hansen (R's `forecast::nsdiffs`)". R's `nsdiffs` actually defaults
to **`test="seas"`** -- an STL seasonal-strength heuristic with a `0.64`
threshold -- and offers `"ocsb"`, `"hegy"` and `"ch"`, the last two
requiring the separate `uroot` package. Canova-Hansen is not R's default
and is not reachable without another dependency. Implemented R's actual
default plus OCSB, and documented HEGY/CH as the remaining gap.

- **`seasonal_strength(y, period)`** -- fpp3's `Fs` measure,
  `1 - var(remainder)/var(remainder + seasonal)` clamped to `[0, 1]`, on
  the STL settings `forecast::mstl` passes through for a single period
  (`s.window=11`, `s.degree=0`, `inner=2`), read from R's source.
  **Matches R to all ten printed digits** on four series once R's Loess
  *jump* shortcut is disabled; against R's defaults it differs by
  `1e-4`..`8e-3`, which is entirely R's interpolation and never enough to
  move the threshold. A constant series returns `0.0` directly, since
  decomposing one leaves floating-point noise in both components whose
  ratio measured `0.08` and would read as a real finding.
- **`ocsb_test(y, period; lag_method, maxlag)`** -- Osborn, Chui, Smith &
  Birchenhall (1988), following **R's `forecast::ocsb.test`**. Matches R's
  statistic, critical value and selected lag order exactly on five series.
  Reports critical values rather than a p-value, since the null
  distribution is tabulated at 5% only -- the same honesty as
  `nyblom_test`. **`pmdarima`'s `OCSBTest` genuinely differs from R** on
  three points (it adds a constant to the auxiliary regression, does not
  lag `Z4`/`Z5`, and indexes lag selection differently); measured gaps run
  to `1.25` on `AirPassengers`, so it is not usable as a cross-check. All
  four figures recorded in the docstring.
- **`nsdiffs(y, period; test=:seas|:ocsb, max_D)`** -- matches R's answer
  under both tests on every fixture. `auto_arima` gained
  `seasonal_test=:seas` and calls it when `D === nothing`.

**The two tests can disagree, and that is now a worked example rather than
a footnote.** On `log(AirPassengers)` seasonal strength is `0.96` (so
`:seas` differences) while OCSB's statistic is `-1.951` against `-1.803`
(so `:ocsb` does not). **R reaches both of those answers too**, so it is a
real disagreement between accepted tests on the field's most-studied
seasonal series, not an implementation artefact. Chapter 20 works it
through.

---

## Stage 9B Tier 2.2 — full covariance matrices (cross-cutting)

**✅ built (2026-10-01)** (`handoff/stage-9B-nonets-handoff.md` §2.2, whose
diagnosis was right: this is a **fit-path** change, not an accessor).
`ARXModel` was the only type retaining a full matrix; every other type kept
`se::Vector{Float64}`, the diagonal only, so `vcov` could not be
reconstructed after the fact -- the off-diagonal terms were already gone,
and they are the part a joint test, a linear combination or a forecast
variance actually needs.

`_hessian_se`/`_opg_se`/`_robust_se` now wrap `_hessian_vcov`/`_opg_vcov`/
`_robust_vcov`, with the three-way `se_type` dispatch collected into one
`_select_vcov` instead of being repeated at every fit site. `ArmaModel`,
`SarimaModel`, `ArimaxModel`, `SarimaxModel`, `GarchModel` and
`AutoregGarchModel` all gained a `vcov` field, and `StatsAPI.vcov`/
`stderror` are defined for all of them plus `ArimaModel`. `GarchModel`
already *computed* its covariance matrix and threw it away.

**The off-diagonals are externally verified too.** The first version of
this testset checked only self-consistency invariants (`se == sqrt(diag)`,
symmetry, exact reductions), which cannot detect a wrong off-diagonal
term -- the very part retaining the matrix exists for. R's `arima`
returns the whole thing in `var.coef`, so it is directly comparable:
on cardox, ARMA(2,0) and SARIMA(1,1,1)(0,1,1)[12] both match R's full
matrix to `1e-3` absolute, with `corr(ar1, ar2) = -0.98163` against R's
`-0.98169` and `corr(ar1, ma1) = -0.92585` against `-0.92590`. The ~0.8%
relative gap is the two optimisers' slightly different estimates plus
R differencing its Hessian numerically where this uses ForwardDiff --
the same tolerance the pre-existing `se`-only tests already use.

Two things surfaced that were not in the handoff's scope:

- **`coef` omitted the estimated mean while `se` included it**, so
  `length(coef(m)) != length(stderror(m))` whenever a mean was fitted --
  and `ArmaModel`'s own docstring claimed the two matched. `coef` now
  includes it, agreeing with `GarchModel`, which always included its `mu`.
  Two tests pinned the old behaviour and were updated.
- **`_vcov_to_se` clamped a non-positive variance to `0.0`.** Fitting
  ARMA(1,1) to undifferenced CO2 drives the MA term to exactly `1.0`, the
  invertibility boundary, and the coefficient table then printed
  `ma1 1.0 0.0 Inf NaN` -- a zero standard error claiming *infinite*
  precision about a parameter the model knew nothing about, with an `Inf`
  z-statistic inviting the opposite reading. It reports `NaN` now, which is
  what `fit_arma`'s docstring already promised for this case.

**That clamp was hiding a genuinely degenerate fit.** A
`fit_autoreg_garch` test asserted `all(isfinite, se)` and passed only
because of it: on its fixture the GARCH block fits `alpha = 3e-16` and
`garch_beta = 0.99999`, both pinned at their constraint boundaries with
persistence at 1, and the OPG covariance diagonal reaches **-5199**. The
test now asserts what the fixture can honestly support -- the regression
block's standard errors are usable, the GARCH block's are not -- and says
why. One further test compared `se` vectors with `==`, which NaN fails even
when the vectors are bit-identical; switched to `isequal`, which is what
an exactness guard means.

---

## Stage 9B Tier 2.3 — distributional forecast accuracy (cross-cutting)

**✅ built (2026-10-01)** as `pinball_loss`/`crps_normal`/`crps_ensemble`/
`winkler_score`/`interval_coverage`/`interval_accuracy` in `src/scoring.jl`
(`handoff/stage-9B-nonets-handoff.md` §2.3, fpp3 §5.9). Before this,
`accuracy` covered point forecasts only, so **every prediction interval
the package emitted was unfalsifiable** -- you could see an interval was
wide but not whether it was wide in the right places.

**The verification standard here is different, and is stated as such in
the docs.** Neither R's `scoringRules` nor Python's `properscoring` is
reachable in this environment -- checked, not assumed -- and R's
`forecast` has no Winkler, pinball or CRPS function, so there is no
package output to match. Each rule is checked against its own definition
instead:

- `crps_normal`'s closed form (Gneiting & Raftery 2007) against
  **numerical integration** of `CRPS(F,y) = int (F(x) - 1{x>=y})^2 dx`
  via `scipy.integrate.quad`, at seven `(y, mu, sigma)` combinations.
  Agreement to `5e-13`, which is `_chisq_ccdf`'s own tolerance rather
  than any limit of the formula.
- `pinball_loss` against the identity that `tau=0.5` is exactly half the
  MAE, plus the defining property that it is minimised at the true
  quantile (checked at three levels on 20,000 draws).
- `winkler_score` against hand computation at two levels and both sides.
- `crps_ensemble` against `crps_normal`, which it must converge to as the
  draw count rises -- and against `|y - mu|` for a degenerate ensemble.

Arguably a stronger check than reproducing another implementation, but a
different one.

**No new dependency.** `crps_normal` needs a standard normal CDF, which
is derived from the `_chisq_ccdf` already in `diagnostics.jl` via
`Z^2 ~ ChiSq(1)` -- the same identity `arx.jl` already uses for its
two-sided p-values -- rather than taking `SpecialFunctions` for one
`erf` call.

**The worked example earns its place.** On cardox the SARIMA fit's 95%
interval covers `0.917` of the held-out year with a Winkler score of
`2.158`, while seasonal naive covers `1.000` with a Winkler of `4.577`:
the benchmark has *better coverage* and is far worse, because it achieves
it by being wide enough to contain anything. That is precisely why
`interval_accuracy` reports coverage and Winkler together and why a
coverage figure quoted alone is not evidence of a good interval.

---

## Stage 9B Tier 2.4 — partial (cross-cutting)

**Three of six items built (2026-10-02)**
(`handoff/stage-9B-nonets-handoff.md` §2.4). Every reference was executed
directly; `forecast::InvBoxCox`, `forecast::fourier`, `stats::ar` and
`stats::spec.ar` all turned out reachable, so all three are dual- or
R-verified rather than definition-only.

### Bias-corrected back-transformation

`boxcox_inv` gained `fvar`: pass the forecast variance on the
*transformed* scale and it returns the **mean** of the forecast
distribution instead of the median. Matches
`forecast::InvBoxCox(biasadj=TRUE, fvar=)` to `4e-11` across five lambda
values, including the two instructive degenerate cases -- `lambda = 1`
gives a correction of exactly zero (no transform, no bias), and
`lambda = -0.5` with `sd = 0.3` turns `100.0` into `775.0`, a 7.75x
factor.

One formula covers both branches, as R's does: at `lambda = 0` it
collapses to the log-normal `exp(y)*(1 + fvar/2)`. R's separate
`biasadj` flag is deliberately *not* mirrored -- R errors if it is set
without `fvar`, so it carries no information, and supplying `fvar` is
the request here.

Chapter 7 previously demonstrated the correction by hand and stated that
`boxcox_inv` "performs the plain, uncorrected inverse only"; it now uses
the keyword and prints both the Taylor form and the exact log-normal
`exp(mu + s2/2)` side by side -- they differ by `1e-4` relative at
`s2 = 0.0248`, an order of magnitude below the `1.24%` correction, and
the Taylor form is the one every reference implements.

### `fourier_terms`

`K` sine/cosine pairs at the harmonics of a period, matching
`forecast::fourier` **column for column** on `K=2`/`m=12`,
`K=6`/`m=12`, `K=1`/`m=4` and the `h` (future-rows) form. Several
periods at once, with frequencies shared between them kept only once.

The point is cost: seasonal dummies need `period - 1` parameters, which
is 51 for weekly data and 47 for a daily cycle in half-hourly data,
while these need `2K` for whatever `K` you choose. They are also
deterministic, so the `h` form extrapolates exactly -- which is what
makes them usable as `fit_arimax` `newexog`, verified end to end.

**At `K = period/2` the sine column is dropped**: `sin(pi*t) = 0` for
every integer `t`, so it is a zero column that would make the design
singular. `K=6` on monthly data gives **11 columns, not 12**, and R does
the same for the same reason. It is the one case where the count is not
`2K`, which is why `names` is worth reading rather than assuming
positions.

### `ar_yw` and `spec_ar`

`ar_yw` fits an autoregression by **Yule-Walker** with AIC order
selection, matching R's `ar()` on *every* reported quantity: the
selected order, the coefficients, `var.pred` (including R's
`n/(n-(m+1))` small-sample adjustment), the whole shifted AIC sweep, the
partial autocorrelations, and R's `floor(10*log10(n))` default
`order.max`. `spec_ar` reads the spectral density off that fit, matching
`stats::spec.ar` to `5e-11`.

This required exposing what `_durbin_levinson` had been computing and
discarding -- the coefficients at every order and the prediction
variance at every order -- so it is now `_durbin_levinson_full` with the
old pacf-only version as a thin view. `pacf` is bit-identical.

Yule-Walker is the right estimator here because it **always returns a
stationary fit**, and a non-stationary AR has no spectrum. The price is
bias: on the 400-point AR(2) fixture a true `phi_2 = -0.4` is estimated
at `-0.509`, enough to move the spectral peak from `0.136` to `0.159`.
Tested against the theoretical peak at the *fitted* coefficients, since
asserting against the true ones would be testing the estimator's bias
under the guise of testing the spectrum.

`df`/`bandwidth` are `NaN`, since both are Daniell-smoothing quantities
with no analogue when the smoothness comes from an AR assumption; R's
`spec.ar` does not return them either.

**A real trap documented while writing this up**: `periodogram` starts
at `1/n` and **excludes `f = 0`** while `spec_ar` includes it. On
undifferenced cardox `spec_ar`'s global maximum is at `f = 0` with
**eighteen times** the power of the seasonal peak -- a near-unit-root AR
putting its mass at zero frequency, correctly -- while `periodogram`'s
`argmax` is `1/12` only because it never looks there. Comparing `argmax`
across the two on a trending series therefore says nothing about the
estimators. Differenced, they agree to `0.0006`.

### Found and fixed in passing

`scoring.jl` (Tier 2.3) defined `_std_normal_cdf`, which `diagnostics.jl`
already defined -- Julia was reporting *"Method definition ...
overwritten ... incremental compilation may be fatally broken"* on every
load. The two were algebraically identical (both route through the same
regularized incomplete gamma), so no number was ever wrong, but a
duplicate method in one module is a defect regardless. Removed; CRPS is
unchanged at `0.233694977255`.

### GARCH `dist=:t`

**✅ built**, and **dual-verified** -- both references were reachable.
Standardized Student-t innovations (`z = e/sigma` has unit variance, so
`sigma2` stays the conditional variance and remains comparable with the
Gaussian fit's), adding one estimated `shape` parameter carried through
`coef`/`stderror`/`vcov`.

Agreement with Python `arch` `dist='t'` is **3e-08** in log-likelihood,
and with `rugarch` `distribution.model="std"` to `1e-4` once
`rec.init=0.94` aligns the variance seed (the remaining gap being
`rugarch`'s whole-sample EWMA against this package's first-75 window).
On the purpose-built fixture `nu` comes back `5.2696` against `arch`'s
`5.26961` where the generating process had `nu = 5`, and the Student-t
fit beats the Gaussian one by **72.9** log-likelihood units. On real
`sp500.gr` returns it gives `nu = 8.97` for 30 log-likelihood units.

**A fixture had to be built for it.** `garch_shared.csv` is Gaussian,
and on it the Student-t shape optimises to `99.998` -- its upper bound
-- so it cannot exercise the code at all.
`verification/garcht/garch_t.csv` is 2,000 points of GARCH(1,1) with
standardized t(5) innovations, sample kurtosis `5.89`.

`nu` is bounded to `[2.05, 500]` by a scaled logistic: below `2` the
variance does not exist so standardizing is undefined, and above, the
likelihood is monotone in `nu` on near-Gaussian data and an unbounded
transform overflows. `arch` bounds it identically. A fitted `nu` at the
bound is a boundary result, not an estimate.

**The simulation path needed fixing too.** `_simulate_one_path` drew
`randn` regardless of the fitted distribution, which would have left
every quantile taken from `variance_paths` with the wrong tails while
the mean variance forecast looked fine (`E[z^2] = 1` either way). It
now draws through `_sim_shock`, which needed a Marsaglia-Tsang gamma
sampler since `nu` is estimated and the chi-squared degrees of freedom
are therefore not an integer. `_loggamma` was **reused** from
`diagnostics.jl` rather than redefined.

### Forecast combination

**✅ built** as `combine_forecasts` (Bates & Granger 1969; Montgomery
section 7.5). Equal weights by default; supplied weights are normalised.

**No package reference exists, and that was checked rather than
assumed**: R's `forecast` has no combination function, and
`forecastHybrid::hybridModel` averages point forecasts but rebuilds
intervals from combined *residuals*, which needs the fitted objects
rather than their forecasts -- a different operation. Verified against
the definition plus the properties a combination must have.

**The honest result is documented rather than glossed.** On the cardox
hold-out the equal-weight combination scores RMSE `0.965` against a best
component of `0.431` and a component mean of `1.323`: it beats the
average component and loses to the best one. Pairing two *comparable*
SARIMAs does not fix it either -- `0.431` and `0.215` combine to `0.266`,
landing between them, because forecasts from similar models on the same
data have strongly positively correlated errors and diversify little.
Both facts are pinned as tests. The case for combining is **robustness,
not optimality**.

The combined standard error is `sqrt(sum(w_i^2 se_i^2))`, the variance
of a weighted mean assuming **independent** component errors -- which
they are not, so it understates the spread. Combining a forecast with
itself shrinks its standard error by `sqrt(2)`, which is the assumption
showing itself, and is asserted as such. Doing better needs the
covariance of the component errors, which needs a joint error history a
`Forecast` does not carry.

### Classical AutoReg

**✅ built** as `autoreg`/`AutoRegModel` -- regression with
autoregressive **errors**, by iterated feasible GLS. SAS `PROC
AUTOREG`'s basic tier, closing the item `CLAUDE.md` had listed as
"unscheduled but ready". **Tier 2.4 is complete, and so is Tier 2.**

Distinct from `arx`, which puts lags of `y` in the *mean* equation as
interpretable regressors; `autoreg` leaves the mean as `X*beta` and puts
the autoregression in the *error*, as a nuisance to correct for. Two
models, not two estimators of one.

The transform is **Prais-Winsten**, not plain Cochrane-Orcutt: the first
`order` rows are rescaled by the stationary covariance and kept rather
than discarded. `method=:yw` (default) takes the AR step by Yule-Walker,
which cannot leave the stationary region and so keeps the GLS transform
defined at every iteration; `method=:ols` is less biased but is refused
if it wanders outside.

**Verification took a different shape, since no single function matches
it.** `orcutt` implements Cochrane-Orcutt but will not install on this R
version (attempted, not assumed), and `nlme::gls` fits by ML/REML rather
than feasible GLS, so its `rho` differs by construction. **At a fixed
`phi` the two are the same estimator**, so that is the comparison:
`autoreg(y, X; phi=[rho])` reproduces
`nlme::gls(..., corAR1(value=rho, fixed=TRUE))`'s coefficients *and*
standard errors to **5e-11** at three values of `rho`. The AR half is
separately R-verified through `ar_yw`, and the OLS starting point
against R's `lm` (`beta` and `se` to 1e-9). Both halves have a
reference; their composition does not, and the docs say so.

`phi = 0` reduces exactly to OLS, which is the cheapest check that the
transform is right, and is asserted.

**The worked example is the motivation.** On the bundled AR(1)-error
fixture, correcting for `phi = 0.69` leaves the slope estimate almost
unmoved (`2.321` to `2.168`) while **doubling** its standard error
(`0.0918` to `0.1882`) and dropping its `t`-statistic from `25.3` to
`11.5`. OLS coefficients stay unbiased under autocorrelated errors;
their standard errors do not, and with positive autocorrelation they are
too *small*.

`se` covers `beta` only -- the AR coefficients are estimated but not
inferred on, which is what makes this the basic rather than the full
maximum-likelihood tier (that is Stage 8.5's `fit_autoreg_garch`).

---

## Documentation restructuring (cross-cutting, not a numbered stage)

**Skeleton ✅ built** (`handoff/docs-restructure-skeleton-handoff.md`) —
modeled directly on `SeasonalAdjustment.jl`'s own live docs site: Home, a
5-chapter Getting Started, an 11-page task-oriented Manual, a 20-chapter
+ 2-appendix conceptual Introduction, and an API Reference split by topic
(9 files) instead of one growing monolithic `api.md` (whose
`size_threshold=1_000_000` build workaround is now removed, no longer
needed). `docs/make.jl` rewritten for the new `pages=[...]` tree.
**Genuinely real content, not stubs, in 5 pages** — this handoff assumed
`getting_started.md` was still an empty placeholder; it wasn't (an
earlier session had already added real, verified ADF/KPSS/ACF checks and
the `RecipesBase.jl` chart gallery to it, per the user's own request), so
that content was *relocated* into the new structure rather than
discarded to satisfy a literal "stub everything" reading:
`getting-started/01-installation.md`, `getting-started/03-was-it-any-good.md`
(unit-root/ACF/Ljung-Box checks + a pointer to `diagnostic_plot`),
`getting-started/05-where-next.md` (the former "design notes"),
`manual/10-plotting.md` (the full `@example`-based recipe gallery), and
`introduction/B-further-reading.md` (the real 6-book bibliography, the
handoff's own explicit content exception). `api/state-space.md` also has
real prose (a short note, since the Kalman-filter engine has no separate
public API beyond what the ARMA/SARIMA/ARIMAX fitting functions already
expose) rather than a stub, since "not yet written" would be inaccurate.
**API split verified lossless**: all 100 `@docs`-referenced names from
the original monolithic `api.md` accounted for exactly once across the 9
new files (counted both sides). **A real gap in the handoff's own
manual/api structure, worked around and flagged here**: neither the
11-chapter Manual nor the 9-file API Reference has an explicit home for
`holt_winters`/`ExponentialSmoothingModel` (Stage 5's classical
exponential smoothing) — placed in `manual/04-fitting-arma-models.md`'s
matching `api/arma-models.md` file as the closest thematic fit; revisit
if a dedicated exponential-smoothing chapter is ever added.

**Content ✅ complete (2026-09-28)** (`handoff/tsa-documentation-handoff.md`)
— every page in the tree above is written and has no stub text left. The
5-chapter Getting Started, all 11 Manual pages, Introduction chapters
1-41 and all three appendices. Every `@example` block executes at build
time; `checkdocs=:exports` and `doctest=true` both pass.

Two notes in the paragraph above are now superseded. **`api/state-space.md`
is no longer a short note**: the Kalman engine *is* public API —
`GaussianSSM`, `TimeVaryingSSM`, `build_statespace`, `stationary_cov`,
`combined_ar_ma`, `kalman_filter`, `kalman_smoother`, `to_time_varying`
and `kalman_filter_diffuse` are all exported and documented, and
`manual/07-state-space-and-kalman.md` uses them directly. And the
bibliography moved from `introduction/B-further-reading.md` to
`introduction/C-further-reading.md`, with `B-verification.md` now
carrying the verification standard itself.

**Three real bugs were found by writing the documentation**, each fixed
with a regression test against a live reference: `stl_decompose`'s
`robust=true` path crashing with `DimensionMismatch` on a large outlier;
`fit_arma(y, (1,0); method=:css_ml)` silently returning a unit root with
`loglik = -Inf` while reporting `converged = true`; and `test/runtests.jl`
importing `Statistics: mean` but not `std`, so a `test_sarima.jl`
assertion passed in isolation and errored under the full suite.
Documentation that executes is a test suite with a different shape.

**Content — Getting Started/Manual stub pages still waiting** (unaffected
by the Introduction restructure below), tracked here so the next
content-writing handoff has a complete checklist rather than needing to
rediscover what's missing (do these one page, or one small group, at a
time — not all at once, matching this project's own "one task" discipline):

- Getting Started (2): `02-first-model.md`, `04-beyond-defaults.md`
- Manual (10): `01-primitives.md`, `02-diagnostics.md`,
  `03-decomposition.md`, `04-fitting-arma-models.md`,
  `05-automatic-order-selection.md`, `06-garch-and-volatility.md`,
  `07-state-space-and-kalman.md`, `08-arimax-and-regression.md`,
  `09-forecasting-and-accuracy.md`, `11-coming-from-r-python.md`
- Introduction: see the restructure note directly below — the old
  21-file checklist here is superseded by the new 41-chapter/3-appendix
  one. **The eventual diagnostics chapters should reflect the
  *complete* current diagnostic suite when written** — including the
  response-surface p-values, KPSS `:auto`, ARCH-LM, and DK
  heteroskedasticity items that landed after the original 6-book
  comparison — per the skeleton handoff's own explicit note; this
  still applies under the new numbering (Part II, chapters 08–12).

**Introduction restructured, 41 chapters + 3 appendices, content still
all stubs** (`handoff/introduction-restructure-handoff.md`) — the
20-chapter skeleton above was itself only ever a placeholder shape; this
supersedes it. Verified directly before starting: all 20 old chapters
were genuinely 3-line stubs except `B-further-reading.md` (39 real
lines, `git mv`'d to `C-further-reading.md` rather than
deleted+recreated, content preserved exactly). **One real discrepancy
found and flagged to the user rather than silently worked around**: the
handoff cites `book-table-of-contents.md` as the source for the new
structure, but that file does not exist anywhere in the repository —
the user was asked how to proceed and chose to treat the handoff's own
section 3 (already fully spelled out, 41 chapters/9 parts) as the
source of truth directly. The user was also asked to confirm the
handoff's "Option A" framing (the Introduction *is* the ~300-page book,
one source serving both a free online edition and a paid print one,
fpp3's own model) before restructuring, since rejecting it would have
changed the chapter list substantially — confirmed.

New shape: Part I–IX (`01-why-model-a-time-series.md` through
`41-the-frontier.md`) plus `A-checklist.md`/`B-verification.md`/
`C-further-reading.md`. Two files carried over unchanged
(`07-transformations.md`, `10-testing-the-residuals.md`); everything
else is a fresh 3-line stub at its new number. `docs/src/assets/custom.css`
added and registered in `make.jl` (previously `assets=String[]`) for
three new admonition categories — `disagreement`/`india`/`julia` — with
`B-verification.md` carrying one verified worked example of each (the
Julia one was caught and corrected during drafting: initial text
claimed `tsvalues` dispatches per-container-type, which is backwards —
it's two methods total, and TSFrames/TimeSeries/DataFrames never reach
it as containers because each library's own accessor already returns a
plain `Vector` first; the disagreement example was checked directly
against `src/arima.jl`'s own verified `nobs` docstring rather than
invented). `make.jl`'s `pages` tree nests all 41 chapters under 9
collapsible part groups. Stale `introduction/`-path links fixed in
`index.md` (×3) and `getting-started/05-where-next.md` (×1) — found via
`grep -rn "introduction/" docs/src`, per the handoff's own verification
step. No chapter content written — deliberately separate, paced work.

**Chapter 15 (STL) written as the reference chapter**
(`handoff/chapter-writing-guide.md` + `handoff/chapter-15-stl-handoff.md`,
`docs/src/introduction/15-stl.md`) — the first of the 41 chapters with
real content, chosen deliberately out of numeric order per the writing
guide (start with the chapter that has the clearest question, the most
interesting method, and the strongest `disagreement` box available,
*then* write chapter 1 once the voice is settled). **A major claim
re-verified from scratch, not transcribed, given the stakes**: the
handoff's own headline finding — that R's and Python's STL disagree by
0.02–0.13 near outliers even with every shared parameter matched, and
that this is *not* the median-bug story repeatedly (and wrongly) cited
in earlier sessions — was independently reproduced this session on a
freshly-generated draw of the handoff's own synthetic series (R
4.6.0 `stats::stl()`, Python 0.14.1 `statsmodels.tsa.seasonal.STL`),
matching the handoff's numbers to 4+ significant figures at every
configuration (defaults, jumps matched, jumps matched + forced
iterations). `stl_decompose`'s own place in the disagreement — no jump
parameter at all (matching Python's exact-loess behaviour, agreeing
with it to ~1e-12), `seasonal_degree` defaulting to `1` (matching
Python, not R's `0`) — confirmed directly against `src/stl.jl`, not
assumed from the handoff's description. **A real teaching example found
independently, better than the handoff's own generic beat-4 sketch**:
`aus_production`'s `Beer` column has no single dramatic outlier, but a
genuine, large seasonal-pattern reversal over 54 years (Q4 goes from
the best quarter in 1956–65, figure ≈ +52, to the worst in 2000–2010,
figure ≈ -38) — verified directly by splitting the series and comparing
decade-level `classical_decompose` figures, then confirmed that
`stl_decompose` with a long `seasonal_window` recovers both decades'
figures without being told where to split. Used as the chapter's real
worked example (beats 1–4) instead of a constructed outlier, with the
handoff's own synthetic outlier series reserved for what it's actually
suited to: the robustness-weights demonstration and the disagreement
investigation. **A genuinely surprising, cross-verified finding along
the way**: on that synthetic series, STL's bisquare robustness weights
zero out seven points, not just the one constructed outlier — because
the threshold is six times the *median* absolute residual, which one
big outlier barely moves, so several ordinary noise draws end up
excluded too; confirmed to be shared behaviour, not a Julia-specific
quirk, by reproducing the identical seven zero-weight indices in R on
the same data. `tsibbledata`'s own quarterly date encoding (days since
1970-01-01, not a `time`/`date`/`index`-named column) is not
auto-decoded by `dataset()`'s catalogue — decoded by hand in the
chapter's own code with a note explaining why, rather than silently
worked around or left unexplained. No Indian dataset exists in the
90-dataset catalogue yet (checked systematically, not assumed) — the
chapter's own `!!! india` box says so honestly rather than inventing
one, and flags it as separate dataset work.

**Part I complete — Chapters 1–7 written** (`handoff/chapter-1-handoff-v2.md`
through `handoff/chapter-7-handoff.md`, plus `handoff/chapter-writing-guide.md`).
Autopilot, per the maintainer's own explicit instruction mid-review, after
verifying all seven handoffs' technical claims directly rather than
transcribing them. **A referenced source document,
`all-chapters-handoff.md`, does not exist anywhere in the repository**
(cited by chapters 3/4/5 as the origin of their original per-chapter box
assignments) — not blocking, since each handoff supplies its own
corrected, self-contained material, but recorded here rather than
silently rediscovered later, matching the earlier missing
`book-table-of-contents.md` pattern from the Introduction restructure.

**Real corrections found before writing, not assumed from the handoffs**:
- Chapter 3's `GNP23`/`GDP23` do not exist in the catalogue (real names:
  `GNP`/`gnp`, or `global_economy`).
- `tsundiff` is a literal alias of `diffinv` (`tsundiff(x;...) =
  diffinv(x;...)`), not a separate lower-level primitive as
  Chapter 3's handoff speculated — confirmed from both docstrings
  directly.
- `moving_average`'s default `centre=true` already performs the 2×m
  even-order centring automatically (Chapter 4's own flagged unknown).
- `tsvalues`'s real missing-data behaviour (Chapter 2's own flagged
  unknown) is safer than assumed: `missing` throws immediately;
  `NaN` passes through and is separately rejected by `acf`/similar
  the moment a real computation is attempted — no silent-drop failure
  mode exists anywhere in the chain.
- `periodogram` itself (not only `spectral_density`) already carries
  real `df`/`bandwidth` fields — Chapter 6's handoff assumed only the
  smoothed function had them.
- `pacf`'s `:yw`/`:ywm` split surfaces a *second*, real R-vs-Python
  disagreement Chapter 5's handoff didn't know about, alongside the
  Bartlett-band one it did.
- **The largest correction**: Chapter 7's central "honest verification
  gap" (Guerrero unverified, CRAN unreachable) is factually wrong —
  `guerrero_lambda`'s own docstring already documents independent
  verification against real R `forecast::BoxCox.lambda` to 5
  significant figures, including a genuine period=1 subseries-splitting
  bug found and fixed along the way (`test/verification/transforms/`).
  Same stale "CRAN unreachable" assumption already corrected multiple
  times elsewhere in this project's history (Stage 5.2/5.3/5.4,
  Chapter 7's own STL precedent). Confirmed with the maintainer before
  writing; Chapter 7 rewritten around the real, stronger story instead.
- Chapter 2's own flagged design decision (convert the `astsa`-derived
  datasets' decimal-year time to real `Date`s, or document as a wart)
  turned out to already be resolved: `dataset_catalogue.jl`'s
  `_decimal_year_to_date` conversion was already converting these
  correctly since the catalogue was first built — `dataset("jj")`
  already returns real `Date`s, confirmed directly. What Chapter 2's
  handoff was describing was the *raw CSV storage format* on disk, not
  what `dataset()` actually hands back. No code change was needed;
  Chapter 2 was written around the real, current, correct behaviour.

**The Chapter 1 airline-corruption story was independently re-verified
by direct byte-level diff, not trusted from the handoff's own telling**:
`handoff/verification/airline.spc` diffed programmatically against the
real canonical `test_data/airpassengers.csv` — confirmed precisely: row
1 (1949) correct, row 2 a fabricated shifted-duplicate of row 1 (not
real 1950 data), rows 3–12 the real years 1950–1959 each shifted down
one row, and real 1960 completely absent from the file. Matches the
handoff's claim exactly; one detail (a file named `Testairline.spc` as
the specific means of catching it) could not be independently located
and is not repeated as a specific claim in the chapter.

**`docs/Project.toml` gained four new docs-only dependencies**
(`DataFrames`, `TSFrames`, `TimeSeries`, `StatsAPI`) to support
Chapter 2's real four-container comparison chart and Chapter 3's
three-way `nobs` demonstration — confirmed installable and working
directly (`TSFrame(df, :Date)`, `TimeArray(dates, values, [:Value])`,
all four containers giving `acf` results identical to the ninth decimal).

**Two real Julia-vs-Python RNG mismatches caught and fixed before
shipping**, both the same underlying mistake: quoting specific decimal
figures from a Python/R-generated verification series in prose, then
writing a *separate* Julia code block seeded with "the same seed
number" and assuming it would reproduce the identical data (it does
not — Julia's RNG stream differs from numpy's/R's even at an identical
seed value, a recurring gotcha already flagged in this project's own
memory). Caught in Chapter 3 (the `nobs`/AICc three-way box) and
Chapter 5 (the ACF band-disagreement box) by dry-running each chapter's
actual code before treating it as finished, not by inspection alone;
fixed by separating "verified this session on shared cross-language
data" (stated as a fact, quoting the real R/Python/Julia numbers from
that specific run) from "the same mechanism demonstrated live in Julia"
(a fresh, self-consistent, actually-executed block, searched for a
seed that reproduces the same qualitative pattern rather than assumed
to land on it by chance).

**One further bug caught the same way, in Chapter 7**: the
back-transformation-bias demonstration initially reused the raw-scale
trend coefficients (fit on `jj.value`) inside a log-scale forecast
calculation, producing a nonsense back-transformed value in the
hundred-thousands. Fixed by fitting a separate log-scale regression
before the bias-correction arithmetic; re-verified to match the
originally-computed, correct figures (naive `17.755` vs. bias-corrected
`17.977`, a 1.25% gap).

All seven chapters' non-plotting computational content was dry-run
directly (Julia, without `Plots` loaded, since the pre-existing local
Plots/Qt6/GR blocker still applies) before being treated as finished,
catching all of the above; the full `docs/make.jl` build was then run
to confirm `Doctest`/`CrossReferences`/`CheckDocument`/`Populate` all
stay clean across all seven new chapters, with only the same
pre-existing, unrelated local rendering failure remaining.

**A real CI failure on Part I's own push (`6a3ecda`), diagnosed and
fixed rather than guessed at (`b653271`)**: Docs CI failed for a
genuinely new reason — not the known local-only Plots/Qt6/GR blocker,
since CI runs Ubuntu where GR works, and this was the first time any
of Part I's ~57 real `@example` chart blocks had executed anywhere
with Plots actually rendering. Diagnosed by installing Julia 1.12.7
locally via `juliaup` (matching CI's `version: '1'`; previously only
1.9.4/1.7.0 were available on this machine) and reproducing the build
directly rather than working from the GitHub API's log-less failure
summary. Two real bugs found this way: Chapter 7 called `var(resid)`
before `using Statistics` was ever imported in that page's `@example`
context (invisible locally where Plots itself already failed first,
real everywhere Plots actually works); and `docs/Manifest.toml` was
pinned to `julia_version = "1.9.4"`, causing real precompilation
instability under Julia 1.12 (`MbedTLS_jll`/`HTTP` failing to
precompile). Both fixed; verified end to end locally on Julia 1.12.7
before pushing — `Doctest` through `RenderDocument`/`HTMLWriter`, all
59 chart blocks producing real SVG output, zero errors — then
confirmed green on the actual CI run. **This is the first time any of
this project's Plots-based documentation was actually rendered and
confirmed working**, locally or otherwise, rather than only disclosed
as CI-only-verified; the local Julia 1.12.7 install now makes this
possible going forward for any future docs work touching `@example`
chart blocks.

---

**Part II complete — Chapters 8–12 written**
(`handoff/chapter-8-handoff.md` through `handoff/chapter-12-handoff.md`).
Autopilot, per the maintainer's standing instruction, after verifying
all five handoffs' technical claims directly. `GNP23` (cited again in
chapters 8 and 9) still doesn't exist in the catalogue — same
non-existent name as Chapter 3's handoff, real series used instead
(`GNP`, `global_economy`).

**A second and third instance of the same "assumed unverified,
actually verified" pattern already found with Chapter 7's Guerrero
box** — both corrected before writing, not after:
- Chapter 10 frames Durbin-Watson's exact p-value as "R-only,
  unverified, CRAN unreachable." Checking `src/diagnostics.jl`
  directly: this package already implements Farebrother's exact
  algorithm and has it verified against real R `lmtest::dwtest(exact=
  TRUE)` to 6+ significant figures (`test/verification/durbinwatson/`).
- Chapter 9 claims this package "returns no p-value for PP's ρ, the
  same honest-refusal pattern" as KPSS's clipped p-values and
  Durbin-Watson. Checking `pp_test`'s own docstring: it already
  computes a real p-value for `:rho` via its own MacKinnon table.

  With Durbin-Watson's p-value also confirmed real, the "three
  recurring honest refusals" thread both chapters were built to lean
  on doesn't hold as originally framed — 2 of its 3 examples are
  wrong. The real, verified pattern is closer to the opposite: this
  package tends to compute a genuine answer via a real algorithm
  (MacKinnon response-surface p-values for ADF and PP, Farebrother's
  exact Durbin-Watson) rather than declining. Both chapters reframed
  around that, matching the maintainer's own earlier call on the
  Guerrero box. KPSS remains the one genuine partial case, and even
  there this package's real behaviour differs from what was
  described — checking directly, it does not clip to the table edge
  the way `statsmodels` does (confirmed: `InterpolationWarning`, hard
  `0.01`/`0.10`); it extrapolates a cruder, genuinely different
  estimate past the table boundary (confirmed: white noise gives
  `p≈0.2`, not `0.10`; a strong random walk gives `p≈0.005`, not
  `0.01`) — described accurately in Chapter 9 rather than assumed to
  match Python's specific failure mode.

**One example dropped per the handoff's own explicit instruction**:
Chapter 12's airline split-verdict example needs `SarimaModel`
residuals, which don't exist as an accessor (confirmed directly,
matching `diagnostic_plot`'s own already-documented note on the same
gap). The handoff said verify-or-drop; dropped, using the chapter's
three other constructed split-verdict cases instead.

**`diagnostic_plot`'s lag-count formula confirmed to match
`astsa::sarima`'s real source exactly**, line for line
(`_diagnostic_nlag` in `src/diagnosticplot.jl`) — not assumed from the
handoff's transcription. Its three call forms and the residual-
sourcing gap in the model-aware overloads (still requires `resid`
supplied explicitly, since no model type exposes fitted residuals)
were both confirmed directly before Chapter 12 relied on either.

**Two more of the same chained-random-state numeric mismatches already
seen in Part I, caught the same way**: dry-running each chapter's
actual code, in its own accumulated `@example` context, before
treating a chapter as finished — not trusting numbers computed in an
isolated scratch script that didn't replicate the same sequence of
prior `Random.seed!`/`randn` calls within that page. Caught in
Chapter 9 (a lag-sensitivity ADF demonstration whose prose initially
quoted figures from an unrelated exploratory run, direction and all)
and avoided pre-emptively elsewhere by verifying every chapter's
numbers from a single continuous script matching each page's real
block order before writing the surrounding prose, rather than after.

Several charts across these five chapters (the spurious-regression
rate, the four-quadrant ADF/KPSS table, the QS-catches-what-Ljung-Box-
misses case, the lag-count verdict flip in Chapter 12) required
searching across seeds for a draw that cleanly demonstrates the
intended qualitative pattern, since a single realisation of a Monte
Carlo experiment does not reliably reproduce a specific illustrative
outcome — each search and its result documented as it happened, not
retrofitted.

All five chapters verified end to end on the local Julia 1.12.7 setup
established while fixing Part I's CI failure: real `Plots` rendering,
not just non-plotting logic dry-run — `docs/make.jl` completing
`Doctest` through `RenderDocument`/`HTMLWriter` with zero errors before
this work was pushed.

---

**Part III complete — Chapters 13–16 written**
(`handoff/chapter-13-handoff.md` through `chapter-16-handoff.md`,
Chapter 15 fully rewritten per `chapter-15-handoff-v2.md`, which
supersedes the version written alongside Stage 6's own STL work —
same verified findings, restructured from 6 charts to 10 to match the
reading-template depth established across Parts I and II). `GNP23`
(cited yet again) still doesn't exist — sixth occurrence of the same
non-existent name across this project's handoffs, real series used
instead each time.

**A fresh, independent re-verification of Chapter 16's central claim,
not a re-read of the original discovery**: `statsmodels`' `MSTL`
truthiness bug (`elif self.lmbda:`, `0` falsy in Python, so `lambda=0`
silently applies no transform) was re-confirmed this session against
the actual currently-installed `statsmodels` 0.14.1 source — both by
reading `mstl.py` directly and by running a freshly-constructed series
through `lambda=0`/`None`/`1e-8` and observing `0`/`None` agree
exactly while `1e-8` differs, on data this session generated itself
rather than reused. This package's own `mstl_decompose` was
independently re-verified on the *same* freshly-generated series to
apply the log transform correctly at `lambda=0`, matching `1e-8` and
differing from `nothing` — confirming the docstring's own existing
"confirmed by execution" claim still holds rather than assuming it
does because a docstring says so.

**One real correction to the handoff's own claim, found by checking
before writing rather than transcribing**: Chapter 16's handoff
describes an ordering effect ("the results are close but not
identical") for `mstl_decompose(y, [a,b])` vs `[b,a]`. Checking
`src/mstl.jl` directly: periods are sorted ascending internally before
fitting, matching both real references, so the two call orders reach
the fitting loop as the identical sequence — confirmed bit-identical
output (`max abs diff = 0.0`), not merely close. Chapter 16 rewritten
around the real finding (order doesn't matter, and why) rather than
the assumed one.

**`seasonal_window="periodic"` does not exist** — `stl_decompose`
types the argument strictly as `Integer`, confirmed by triggering the
real `TypeError` directly rather than assuming R's convenience string
carried over. Chapter 15 uses a very large window as the practical
limiting case instead, and states honestly (confirmed by checking,
not assumed) that it approaches but does not exactly reach classical
decomposition's frozen figure, since STL never fully surrenders its
ability to adapt.

**Two more real bugs caught by dry-running each chapter's actual code
before treating it as finished, the same discipline that has now
caught something in every one of Parts I through III**:
- Chapter 14's Q1-value scatter plotted mismatched-length vectors
  (`x` at full length, `y` filtered through `skipmissing`) — would
  have thrown a real `DimensionMismatch` on build. Fixed by filtering
  both vectors together.
- Chapter 16's before/after MSTL remainder comparison compared
  single-period STL fitted to *real* `vic_elec` data against MSTL
  fitted to a *different*, synthetic constructed series — an
  incoherent apples-to-oranges comparison that would have run without
  erroring and produced two real, plausible-looking numbers that
  simply had nothing to do with each other. Fixed by fitting the
  single-period comparison on the same synthetic series MSTL was
  actually run on.

**A third instance of the cross-language RNG-seed mismatch already
caught twice in Part I**, this time in Chapter 15's own disagreement
box — its central claim (Julia's `stl_decompose` reproduces Python's
STL trend to full displayed precision on identical data) is exactly
the kind of claim that cannot survive regenerating "the same" series
from a Julia `Random.seed!` call, since Julia's RNG stream differs
from numpy's/R's even at an identical seed value. Fixed the same way
as before, but more directly this time: rather than searching for a
Julia seed that reproduces a similar *pattern*, the actual 120-value
series R and Python both analysed was embedded as a literal array in
the chapter's own Julia code, so the three-language digit-for-digit
comparison is against genuinely identical data rather than three
separately-drawn samples that merely share a seed number.

**Chapter 15's own R-vs-Python STL comparison was regenerated fully
fresh this session**, not reused from the Stage 6/original chapter-15
work, per the v2 handoff's own explicit instruction — a new series
generated to the same specification (`n=120`, linear trend, seasonal
amplitude 12, one `+45` outlier at `t=60`, seed 11), run through real
R 4.6.0 `stats::stl` and real Python 0.14.1
`statsmodels.tsa.seasonal.STL` across all four configurations
(defaults × `robust`, jumps matched × `robust`), plus a forced-
convergence check. The qualitative story matches the original
investigation closely (robustness makes the disagreement worse, not
better, killing the median-bug hypothesis; matching jumps explains
most of the gap; something on the order of `0.005` remains
unexplained even after forcing convergence) — confirming the earlier
finding was not a one-off artifact of that specific random draw, on
genuinely fresh data rather than the same numbers re-quoted.

All four chapters verified end to end on the local Julia 1.12.7 setup:
real `Plots` rendering, `docs/make.jl` completing `Doctest` through
`RenderDocument`/`HTMLWriter` with zero errors before this work was
pushed. No `src/` files touched.

**Part IV complete — Chapters 17–23 written**
(`handoff/chapter-17-handoff.md` through `chapter-23-handoff.md`).
`GNP23` (cited yet again across these handoffs) still doesn't exist —
used `rec`/`soi`/`global_economy`/`aus_production`/`vic_elec` instead
throughout, as in every earlier Part.

**A real package gap found and filled before any of Chapters 19, 20 or
22 could be written honestly**: `forecast()`/`StatsAPI.predict()` did
not exist at all for `ArimaModel`/`SarimaModel` (only `ARXModel` had
it, from Stage 5.2). Rather than route around the gap, built it as real
package functionality:
- `psi_weights(ar, ma, h)` — Box-Jenkins MA(∞) impulse-response weights,
  verified to exactly match R's `ARMAtoMA()` on several AR/MA/ARMA
  cases, including a case with a literal unit root in `ar` (the
  undifferenced-polynomial case `forecast(::ArimaModel)` needs).
- `forecast(::ArimaModel, horizon)` — forward AR/MA recursion on the
  differenced, demeaned series using the model's own fitted innovations,
  re-integrated via `tsundiff`; prediction-interval variance from
  `psi_weights` evaluated on the **undifferenced** AR polynomial (`ar`
  combined with `d` factors of `(1-B)`), matching `forecast::Arima`'s
  own documented approach. **Verified against real R
  `predict(arima(...))` and Python `statsmodels.get_forecast()`** on a
  freshly generated ARIMA(1,1,0) series: point forecasts and interval
  widths match both references to several significant figures.
- `forecast(::SarimaModel, y, horizon)` — the same construction
  generalized through `combined_ar_ma`, both differencing orders
  re-integrated in the reverse of the order `fit_sarima` applies them
  (`D` first when differencing, `d` undone first when re-integrating).
  **Verified against real R and Python on the actual airline model**
  (`ARIMA(0,1,1)(0,1,1)[12]` on log `AirPassengers`): point forecasts
  and standard errors match both references to 5–6 significant figures
  across a 12-step horizon.

**A serious-looking SARIMA discrepancy investigated at length and found
to be a false alarm, not a bug** — worth recording in detail since it
consumed real investigation time and the conclusion matters for trusting
`fit_sarima` going forward. Fitting `ARIMA(0,1,1)(0,1,1)[12]` to log
`AirPassengers` using data pulled from what turned out to be a stale or
mistaken source appeared to show Julia's `fit_sarima` reporting
`loglik=244.70` against R/Python's own claimed `loglik≈223.63` for
supposedly the same model. Two-way cross-evaluation (Julia's own
Kalman filter run at R's exact coefficients, and R run at Julia's exact
coefficients) confirmed the ~21-unit gap was consistent regardless of
which coefficients were plugged in — ruling out "different local
optimum" and pointing at a genuine scale/constant discrepancy. Scoped
by testing simpler sub-cases from `test/verification/sarima/bulk/`
(`d>0` alone with both MA types: matches; `D>0` alone with regular MA
and seasonal AR: matches; `d>0` **and** `D>0` together with AR terms
only, no MA at all: matches) — none reproduced a discrepancy, isolating
the "bug" to something specific about the exact `(0,1,1)(0,1,1)` MA-only
structure combined with real `AirPassengers` data specifically. **A
fresh synthetic `(0,1,1)(0,1,1)₄` series, generated and cross-fit this
session, matched R and Python to full displayed precision** (`loglik
≈ -277.01` in all three) — ruling out a general bug in the MA-only
seasonal structure too. Finally, re-loading `AirPassengers` directly
from this package's own `test_data/airpassengers.csv` (confirmed
bit-identical to R's built-in `AirPassengers`, `max abs diff = 0`) and
re-fitting **all three of R, Python (`statsmodels`, both its `ARIMA`
and `SARIMAX` classes), and Julia** gave the same answer everywhere:
`loglik ≈ 244.696`, `ma1 ≈ -0.4018`, `sma1 ≈ -0.5569` — R's own fresh
`arima()` run on its own built-in series gave `244.6995`, contradicting
whatever produced the earlier `223.63` figure. The earlier numbers were
never reproduced from any input this session could identify; the
package itself is correct, re-confirmed independently across three
languages on the canonical dataset. `forecast(::SarimaModel, ...)` was
re-verified against this now-confirmed-correct fit afterward (see
above) rather than trusted on the strength of the earlier, unresolved
run.

**The airline-model "split verdict" claim (Chapter 12's own handoff and
Chapter 20's both flagged it for re-verification) does not reproduce
either, on the confirmed-correct fit** — re-run directly this session:
every Ljung-Box p-value from lag 3 through lag 36 stays above `0.42`,
nowhere close to the claimed lag-3/4 failure. Dropped from Chapter 20
per its own handoff's explicit "verify or drop" instruction; the clean
diagnostic panel is reported honestly instead.

**The order-recovery Monte Carlo for Chapter 21 was run fresh this
session** (50 replications per cell, not the handoff's own recorded
40, and not reused from any earlier attempt) — true AR(2)
(`φ₁=0.6, φ₂=-0.3`), search `p,q ∈ {0,1,2,3}`, AIC vs AICc:

```
n =  100:   AIC  8.0%   AICc 12.0%
n =  250:   AIC 26.0%   AICc 26.0%
n = 1000:   AIC 56.0%   AICc 56.0%
```

Lower across the board than both the handoff's own recorded figures and
an earlier in-session attempt at the same experiment — consistent with
the handoff's explicit instruction to report a fresh run's own numbers
rather than transcribe either. At `n=100` the dominant wrong pick is a
plain `(0,1)` (over half of all misses), not a close AR neighbor —
reported as found rather than forced into a "mostly close neighbours"
narrative the data didn't actually support at that sample size.

**The 4-way ARMA standard-error comparison for Chapter 18 was run
fresh** on a freshly Julia-generated ARMA(1,1) series (seed fixed and
reproducible from the chapter's own code): R and this package's
`se_type=:hessian` agree almost exactly (`SE(ma1)=0.0899` both); Python
`statsmodels`' own `opg` default and this package's `se_type=:opg`
agree closely with each other (`≈0.100`); the implied t-statistic for
the MA coefficient ranges `2.05` to `2.36` depending only on which
covariance estimator's default happened to be used. The second,
previously-unverified "transformed-Hessian bug" claim was **not**
re-asserted — this package's `se_type=:hessian` is confirmed, by
reading `src/arma.jl` directly, to already compute the Hessian in the
natural (untransformed) parametrization by design, exactly the
implementation choice `handoff/stage-6-arima-handoff.md` §4.5
recommends, stated as a design fact rather than as a caught bug nobody
could actually reproduce.

**One genuine environment-dependent hazard found and worked around**:
verifying Julia code by direct execution used the default `julia`
(1.9.4, this machine's own package environment, `Manifest.toml`), while
what the CI/`Docs.yml` build actually runs is `docs/Project.toml`'s own
environment on Julia 1.12.7 (`juliaup`'s `+1.12.7`) — and the two
environments resolve genuinely different `Optim.jl`/`NLSolversBase`
versions. On at least one series this produced different `fit_arma`
convergence behaviour between the two environments for the identical
input (traced to a seed/ordering mistake in that specific case, not a
real cross-version numerical difference, but confirmed by direct
comparison that the two environments *can* diverge). All numeric claims
in these seven chapters were re-verified in the `docs`/1.12.7
environment specifically, matching what actually renders, rather than
trusted from the faster default-environment session used for early
exploration.

All seven chapters verified end to end on the local Julia 1.12.7 setup:
real `Plots` rendering, `docs/make.jl` completing `Doctest` through
`RenderDocument`/`HTMLWriter` before this work was pushed. No other
`src/` files touched beyond `src/forecast.jl`.

**Part V complete — Chapters 24–28 written**
(`handoff/chapter-24-handoff.md` through `chapter-28-handoff.md`). No
`src/` changes needed this time — `fit_garch`/`GarchModel`,
`forecast_volatility`/`VolatilityForecast`, and the realized-measures
functions (`realized_variance`, `bipower_variation`, `jump_test`,
`realized_semivariance`, `realized_measures`) were all already built
(Stage 7 / Stage 7's forecast and realized-vol extensions) and matched
every claim in all five handoffs on direct inspection.

**Both `[U]` claims in Chapter 25's disagreement box confirmed
directly from source**: `fit_garch(y, p, q)` really is `p` = ARCH
order, `q` = GARCH order, matching Python `arch`'s own `GARCH(p=,
o=, q=)` signature and R's `garchOrder = c(p, q)`; `cov_type` really
does default to `:robust` (Bollerslev–Wooldridge sandwich), not
`:classic`. Re-verified with fresh numbers rather than the handoff's
own recorded ones: fitting a fresh simulated GARCH(1,1) series
(`n = 1500`) gave a `3.9%` difference between the two standard errors
on `ω` — smaller than the handoff's recorded `~6%`, still a real,
non-trivial difference, reported as actually found.

**Chapter 26's `ω`-sign claim was re-verified and found to cut the
other way on real data than on the handoff's own simulated check** —
worth recording since it changes how the chapter states the point.
The handoff found EGARCH's fitted `ω` came out *positive* on a fresh
simulated series and flagged that the recorded claim ("`ω` can be
negative") needed restating as a structural fact rather than an
empirical one. Fitting EGARCH(1,1,1) to the real NYSE returns series
used throughout Part V gives `ω = -0.536` — genuinely negative. Both
findings are correct simultaneously and the chapter states it that
way: EGARCH's log-variance parametrisation places no positivity
constraint on `ω` at all (the structural fact, always true), and
whether a *given* fit's `ω` comes out positive or negative is a fact
about that specific data (found both ways this session, on two
different series).

**The `nyse` dataset (Shumway & Stoffer's canonical returns series,
`n = 2000`) was used as the primary series across all five chapters**
for continuity, the same way `aus_production`'s Cement series anchored
most of Part IV. Confirmed directly: it is already returns, not
prices (no separate price level is bundled) — the "price" panel in
Chapter 24 is reconstructed by cumulative product, which recovers the
exact relative path since a return series determines `price[t] /
price[t-1]` exactly, just not the absolute starting level. The series
includes 19 October 1987 ("Black Monday"), an `18`-standard-deviation
single-day move, confirmed directly by inspecting the sorted return
vector — responsible for a large share of the series' `63.5` excess
kurtosis, stated plainly in Chapter 24 rather than left as an
unexplained large number.

**One honest finding that came out messier than the handoff
anticipated, reported as found rather than smoothed over (Chapter
24)**: the handoff expected ARCH-LM to cleanly separate genuine
volatility clustering from a one-off variance level-shift and from
variance growing with the series level. Run fresh on real/simulated
examples of all three, **ARCH-LM rejected decisively on all three
cases** — a level shift and steady growth both produce
squared-residual autocorrelation just as real clustering does. What
actually distinguished the three was the Durbin–Koopman variance-ratio
test's **F-statistic magnitude**, not its bare significance
(`1.31` for genuine clustering against `22.7`/`9.2` for the shift and
the trend) — reported exactly this way rather than forced into the
handoff's cleaner expected story.

**Chapter 28 has no cross-language reference implementation to verify
against** (confirmed: Python `arch` 5.1.0 has no `realized` submodule;
R's `highfrequency` unreachable this session) and states that
explicitly as a weaker verification standard than the rest of the
book — correctness instead comes from simulation with known
properties: a constructed jump day and a rescaled steady day with
matching realized variance but very different bipower variation, and
a fresh 1,000-day-per-arm calibration of the Barndorff-Nielsen &
Shephard jump test (`3.8%` false-positive rate against a nominal `5%`,
`100%` detection on real injected jumps — close to, not copied from,
the handoff's own recorded `4.2%`/`100%`). All intraday price paths in
this chapter are simulated, since no intraday series is bundled, and
are labelled as such throughout.

All five chapters verified end to end on the local Julia 1.12.7 setup:
real `Plots` rendering, `docs/make.jl` completing `Doctest` through
`RenderDocument`/`HTMLWriter` with zero errors before this work was
pushed. No `src/` files touched.

---

**Part VI complete — Chapters 29–33 written** (`handoff/chapter-29-handoff.md`
through `chapter-33-handoff.md`), **Part VII complete — Chapters 34–37
written** (`handoff/chapter-34-handoff.md` through `chapter-37-handoff.md`).

**`src/statespace/` had nothing exported at all before this work** —
`GaussianSSM`, `build_statespace`, `kalman_filter`, `kalman_smoother`,
`stationary_cov`, `combined_ar_ma`, `TimeVaryingSSM`, `to_time_varying`,
`kalman_filter_diffuse` were all implemented but unreachable from
outside the package. Exported now (user confirmed via `AskUserQuestion`
before doing so), since Chapters 29–33 need to call them directly.

**A genuinely new capability was built for Chapter 31**: `TimeVaryingSSM`
had no general smoother at all — only `GaussianSSM`'s own ARMA-specific
one (`Z = e1`, `H = 0` hardcoded). Built `kalman_smoother(::TimeVaryingSSM,
y, a0, P0)`, the full Durbin & Koopman fixed-interval smoother plus
disturbance smoothing (`eta`/`eta_var`/`eps`/`eps_var`), verified to
machine precision two ways: exact reduction to `GaussianSSM`'s own
smoother via `to_time_varying`, and direct agreement with real
`statsmodels` output (six separate arrays, `atol=1e-7`) on a system
with an interior missing observation. `kalman_filter(::TimeVaryingSSM,
...)` and `kalman_filter_diffuse` were both missing NaN/missing-
observation handling entirely (user confirmed via `AskUserQuestion`
this should be built properly, not stubbed); both now skip missing
periods correctly (predict-only, no update, `n_obs` tracked separately
from `n` for the `log(2π)` term, `nobs_diffuse` not incremented on a
missing step) — matches real `statsmodels`' `np.nansum(res.llf_obs)`
exactly. `test/test_timevaryingssm.jl` grew five new `@testset` blocks
(~130 lines); full suite **7277/7277 passing**, zero regressions.

**Chapter 32's own handoff expected an exact likelihood match** under
its reduction test (`Q_beta → 0` should reproduce the fixed-coefficient
fit's own likelihood exactly); this is false, and found to be false by
direct execution rather than assumed correct from the handoff. Point
estimates agree closely (`beta` diff `0.0007`, `ar` diff `0.0016`) but
the likelihoods differ by a real, structural `~2.3–2.5` — the diffuse
state carries a diffuse phase the fixed-parameter model never has, so
the two likelihoods are computed over subtly different effective
samples. Chapter 32 reports the finding honestly and forward-references
Chapter 35, which explains it in full as its own centrepiece
`disagreement` box: the two fits are different *models*, not two ways
of computing the same thing, and cannot be compared by AIC/BIC/
likelihood at all — only out-of-sample.

**Chapter 32's handoff also flagged an "unverified T/R/Q/H individually
time-varying" gap as still open.** Found, via `test_timevaryingssm.jl`'s
own existing test names, that this gap had already been closed in
earlier, unrelated work this session — corrected the chapter to report
this positively rather than repeat a stale caveat.

**Chapter 34's joint regression-with-ARIMA-errors fit on `cmort`/`tempr`
produces a sign reversal against OLS** (`beta = +0.2424` against OLS's
`-0.4866`) — independently triple-verified, Julia/R/Python agreeing
exactly (`ar1 = 0.8345`, `beta = 0.2424`, `loglik = -1640.98` on all
three), robust across AR(1)/MA(1)/ARMA(1,1) error specifications.
Reported as a genuine finding, not smoothed toward the OLS sign.
`fit_arimax`/`fit_sarimax` default (`model=:mle`) matches R's actual
`stats::arima` source and Python `SARIMAX`'s actual defaults
(`mle_regression=True, use_exact_diffuse=False,
time_varying_regression=False`) rather than the de Jong (1991)
diffuse-regression citation a reader might expect — both checked
directly against source, not assumed from documentation.

**Two real capability gaps found and reported plainly rather than
worked around silently**: neither `fit_arimax`'s own regression-with-
ARIMA-errors model class (`ArimaxModel`/`SarimaxModel`) nor
`fit_autoreg_garch`'s combined model has a `forecast`/`predict` method
— confirmed by direct `MethodError`, not inferred from exports. Chapters
36 and 37 both needed an out-of-sample forecast and neither had one
available; both hand-roll a one-step-ahead forecast from the fitted
parameters directly in the chapter's own code, with the gap named
explicitly rather than silently patched over. Also confirmed: this
package has no dedicated distributed-lag regressor utility (Chapter 34)
and no calendar/holiday-calendar type of its own at all (Chapter 36) —
the latter means `BusinessDays.jl`'s own hardcoded Saturday–Sunday
weekend (`isweekend(dt::Dates.Date)`, no calendar argument — confirmed
directly against that package's real source on GitHub) simply does not
apply here, since a calendar regressor in this book is built as a plain
numeric vector from `Dates.jl`, with no calendar-type hierarchy
underneath it.

**Chapter 36's Diwali material was independently re-verified this
session rather than transcribed from the handoff.** The handoff's own
"verified material" (an October seasonal factor moving `0.915164 →
0.753974`, specific X-13ARIMA-SEATS error messages) comes from the
external X-13 binary tool, which this package does not wrap and has no
RegARIMA equivalent for — so those specific numbers were not
reproduced. Instead built an analogous, honestly-computed demonstration
from this package's own primitives (`fit_arimax` with a Diwali-month
indicator regressor, `classical_decompose` for the seasonal factor):
recovers a `13.25` surge against a true `15.0`, moves the October
multiplicative factor `1.0018 → 0.9611`, same direction as the
handoff's own external-tool result. Real Diwali dates (2005–2025) and
a live cross-source discrepancy check (Guru Nanak Jayanti 2026: sources
split `5 November` vs `24 November`, a 19-day gap — the handoff's own
claimed `>2`-week 2026 Diwali discrepancy did not reproduce; current
sources agree closely on `8 November`) came from a live web search run
this session, not the handoff's own recorded findings.

**Chapter 37's central 1.038-log-likelihood-unit figure did not
reproduce on this chapter's own dataset** (`cmort`/`tempr`, chosen to
keep the reduction test exact against Chapter 34's own verified fit) —
recomputed independently and found to be `0.2517` on this series, with
the opposite sign from the handoff's own description (the correct
treatment scored *higher* than the shortcut here, not lower; a `phi`-
sweep shows the gap's sign is not fixed in general — it depends on how
far the series' own first residual lands from zero). **Confirmed
directly, per the handoff's own instruction**: `fit_autoreg_garch`
already implements the correct stationary first-observation treatment
throughout (`_ar_garch_loglik`'s presample block, verified to agree
with Chapter 34's own fit to `2.3e-13`) — no bug found, the shortcut in
the chapter is a hand-built hazard for demonstration, not a defect in
the package. Also found and reported honestly rather than forced to
match the handoff's "both diagnostics pass" expectation: on the real
combined AR(1)+GARCH(1,1) fit, ARCH-LM improves by more than an order
of magnitude (`p = 0.0002 → 0.033`) but Ljung-Box still rejects
(`p < 0.0001`) — a single AR(1) term is not a complete mean-equation
specification for this real series, and no GARCH variance equation can
repair that; reported as a genuine partial result rather than switched
to a cleaner-looking series.

All nine chapters verified end to end on the local Julia 1.12.7 setup:
real `Plots` rendering, every `@example`-tagged code block extracted
and re-run standalone before being considered done. Full `Pkg.test()`
on the main `Project.toml` environment: **7277/7277 passing**, zero
regressions from the `src/statespace/` changes.

---

**`iip_india` bundled, and the book's `india` boxes grounded in it.**
The project maintainer supplied `data/iip_india.csv` — India's real
monthly Index of Industrial Production (General Index, base
2011-12 = 100), April 2011 to March 2026, 180 observations, complete
with no missing months. Registered in the existing catalogue the same
way every other bundled series is (a row in `data/inventory.tsv`, plus
a new `"MoSPI"` key in `_PACKAGE_URL`/`_PACKAGE_CITATION`), so
`dataset("iip_india")`/`dataset_info`/`datasets()` all work with no
special-casing. Provenance is recorded honestly: supplied directly
rather than re-downloaded from the MoSPI portal during this session,
and the `DatasetInfo` entry says exactly that rather than implying a
verification that did not happen.

**36 `india` boxes exist across the book; 21 were rewritten to cite
real, computed numbers from this series, and the other 15 deliberately
left alone** — they are about genuinely different Indian series
(equities, the rupee, rainfall, electricity demand, quarterly GDP,
RBI policy rates) that `iip_india` cannot stand in for, and force-
fitting it into them would have been worse than leaving honest
qualitative claims in place.

The standout result, now in Chapter 1 as the book's own opening
`india` box: across **fourteen real years checked against the real
Diwali calendar, the higher-production month swaps in step with the
festival every single time** — Diwali in November, October reads higher
(ratio up to `1.055`); Diwali in October, November reads higher (down
to `0.94`). Fourteen for fourteen, no exceptions. What had been an
assertion since Chapter 1 is now a re-runnable check.

**Several long-standing claims were checked against the real series
and turned out to need correcting, not merely confirming** — the same
discipline applied to the handoffs throughout Parts VI and VII:

- **Chapter 17's `0.7`–`0.95` AR-coefficient range does not hold for
  industrial growth.** The real year-on-year `iip_india` growth rate
  fits `φ = 0.556` — genuinely persistent, but well short of the
  quoted range. The box now separates the two series it had quietly
  merged: industrial growth (checkable, `0.56`) from inflation
  (not bundled, so not checkable here). Chapter 37's own box, which
  cited Chapter 17's figure, was updated to match and re-anchored on
  its own fitted `φ = 0.834`.
- **Chapter 29/30's "the 2020 lockdown left genuine holes" is wrong
  for this series.** `iip_india` has every expected month present
  (checked directly); the lockdown shows up as an extreme *present*
  value — `54.0` in April 2020, down from `117.2` in February — not an
  absence. Both boxes now say so and keep the gap-handling material
  scoped to the Indian series that genuinely do have gaps.
- **Chapter 12's warning about a "short default lag count" was aimed
  at the wrong thing.** `_diagnostic_nlag`'s own formula already
  reaches lag `36` for any monthly series *once `period=12` is
  supplied*; the real trap is omitting `period` entirely, which
  silently reverts to `nlag=20`. Confirmed on `iip_india` both ways.
- **Chapter 20's "residuals will show a smear" is not what actually
  happens.** A SARIMA(0,1,1)(0,1,1)[12] on the real series comes back
  with a seasonal MA coefficient pinned at `Θ = -1.0`, exactly on the
  invertibility boundary, and residual portmanteau checks afterwards
  look *clean*. The model compensates for what it structurally cannot
  represent by straining a parameter to its edge — so the boundary
  estimate is itself the diagnosable failure, a sharper and more
  useful finding than the smear the box originally predicted.
- **Chapter 22's MAPE-and-zeros warning needed scoping.** The General
  Index never approaches zero (real minimum `54.0`, even through the
  lockdown); the genuine near-zero problem lives one level down, in
  the sub-components. The box now says which.
- **Chapter 3's rebasing spike is not visible in this file.** The
  bundled series already starts on the 2011-12 base, so the historical
  join simply is not there. The box says so, then demonstrates the
  mechanism on a clearly-labelled synthetic splice built from the real
  series (`5.5×` a typical month's move).

Claims that did hold up, now with real numbers attached: the fiscal
year genuinely starts in April (Chapter 2); the level is
non-stationary and the growth rate is not (Chapter 8, ADF `p = 0.96`
against `p = 2.76e-7`); the lag-12 ACF really is weaker than a
comparable Western series (Chapter 5, `0.36` against `prodn`'s `0.72`
— the US Federal Reserve's own industrial production index, an
apples-to-apples pairing that happened to already be bundled); the
seasonal pattern really did change around GST (Chapter 14, October's
factor `1.022 → 0.911`); the seasonal shape really does drift year to
year (Chapter 15, October's STL component swinging `-1.5 → +2.6 →
-5.6`); a rolling window really does beat an expanding one across the
2016/2017 breaks (Chapter 23, RMSE `10.04` against `10.35` — real, and
honestly reported as modest rather than dramatic); a log transform is
mildly justified (Chapter 7, level-vs-spread correlation `0.12 →
-0.04`).

Every modified chapter had its full `@example` block sequence
extracted and re-run standalone, and the complete `docs/make.jl` build
(doctests, cross-references, `checkdocs=:exports`, HTML rendering)
completes clean.

### Documentation for Stages 9 and 10 — ✅ built 2026-10-14

From `handoff/stage-9-documentation-handoff.md` and
`handoff/stage 10/stage-10-documentation-handoff.md`, plus the five gaps
the 2026-10-03 audit recorded in `handoff/tsa-documentation-handoff.md`.

**API reference.** `api/exponential-smoothing.md` (the ten Stage 9 entries
moved, not copied, out of `api/arma-models.md`); `api/multivariate.md`
split into `api/var.md`, `api/cointegration.md` and `api/structural.md`;
`api/decomposition.md` no longer a stub. One orienting figure per page.
The VAR residual tests sit on `api/var.md`, not `api/diagnostics.md` as
proposed: with them, diagnostics rendered at 103.8 KiB, over Documenter's
100 KiB warning — the threshold that once forced the `api.md` split.

**Manual.** Chapter 10 (exponential smoothing) went from no figures to
twelve, including ETS(M,N,M)'s exact class-3 interval against 200,000
simulated paths. The multivariate material became four chapters — 11 VAR,
12 Causality and Impulse Responses, 13 Cointegration and VECM, 14
Structural VAR — with 24 figures; plotting and R/Python renumbered 15 and
16. Manual 05 covers `auto_ets` and the other automatic searches; manual
09's forecastable-types list is a table (it had named `ARXModel`, which
has `predict` only).

**The book: 41 chapters → 52, nine Parts → eleven.** A content-free
renumbering commit first (chapters 24–40 → 28–44, 41 → 52), then:

- Part V, *Models That Forget* (24–27): weighting the recent past;
  the ETS taxonomy and likelihood; multiplicative errors and the two R
  `forecast` bugs, settled in the chapter by simulation; Theta and TBATS,
  with R's TBATS interval bug traced further than the manual had it
  (`forecast.tbats` reads `object$beta.v`, a field a `tbats` object does
  not have, so the trend's β drops out of `g`).
- Part X, *Series That Move Together* (45–51): two series at once, fitting
  and checking a VAR, Granger causality, impulse responses and the
  ordering problem, residual-based cointegration, Johansen and the VECM,
  structural identification.
- Chapter 52 (*The Frontier*) rewritten: it said the package had no VAR.

**The Indian-data decision (Stage 10 §5.5) was resolved without sourcing
anything:** `global_economy` already holds India (GDP, CPI, imports and
exports as % of GDP, 1960–2017), and Part X's Indian thread is built on
it — honest that 58 annual points is short for a VAR.

Each chapter was written to the chapter-writing guide by a subagent from
the reviewed template (chapters 24 and 45 first), every figure rendered
and looked at, every number quoted checked against the built output,
every R/Python claim run.

**Found and fixed by writing it** — documentation that executes found
four package defects and one documentation-wide one:

- `fit_ets` stopped short in the high-α corner. On monthly IIP the
  optimum is α ≈ 0.98, γ = 0, a thin wedge of the traditional region
  (γ ≤ 1 − α); from the four starts every simplex ran into α = 1 and
  stopped 5.5 log-likelihood units low for all three additive seasonal
  forms, so `auto_ets` chose ETS(A,Ad,A) where R and the corrected fit
  choose ETS(A,N,A). Two starts inside the wedge fix it; regression test
  pinned to `statsmodels`' −579.94.
- `fit_ets` and `fit_tbats` returned α above 1 on large-magnitude series
  with `converged = true`: the out-of-region penalty was a fixed `1e10`,
  and India's GDP in US dollars has an SSE near 4e23, so leaving the
  region was cheaper than any fit in it. The penalty now scales with the
  series.
- `fit_svar`'s over-identification statistic was Lütkepohl's (and
  `vars`') short form `T(ln|Σ_r| − ln|Σ_u|)`, which drops
  `tr(Σ_r⁻¹Σ_u) − K` — zero only when the structural variances are free.
  For a unit-diagonal `A` with `B = I` it reduced to `−T ln|Σ_u|`,
  unit-dependent: `−4.30`, `p = 1` on India's trade VAR. Now `2(ℓ_u −
  ℓ_r)` in full; `vars`' `121.6` on `var2_p1` was the short form, the
  likelihood ratio is `31.8`.
- The cointegration manual's VECM equation rendered as `lpha eta'`: an
  edit script had turned `\a` and `\b` into control characters.
- **Documenter dropped printed output from 47 `@example` blocks** that
  printed and then ended with a plot (or a displayed value), across 19
  book chapters and two manual pages — prose quoted numbers the reader
  never saw. Each was split; the new chapters were written to avoid it.

**Open, recorded rather than fixed:**

- `fit_tbats` cannot fit half-hourly `vic_elec` with periods `[48, 336]`
  at any `k` tried: its start grid tries only negative `γ₂`, and the
  admissible points need positive values. R's `tbats` fits four weeks in
  about a minute. Chapter 27 says so and uses daily data.
- ETS(A,Ad,A) on IIP still ends 0.38 log-likelihood units below
  `statsmodels` — a flat valley in `φ`, not the corner above.
- `converged = false` can print at a genuine optimum on surfaces this
  flat; chapter 25 shows the profile likelihood as the check.
- Length: several new chapters run past the guide's 6–8 pages, mostly in
  simulation code that settles a claim on the page (26 is the longest at
  ~550 lines).

---

### Documentation for Stage 11 — ✅ built 2026-10-15

From `handoff/stage 11/stage-11-documentation-handoff.md`.

**API reference.** The four Stage 11 pages (`unobserved-components`,
`dynamic-factors`, `regime-switching`, `threshold-models` -- the handoff's
`dynamic-factor.md` and `threshold.md` kept their built names) each gained a
framing paragraph, a link to their manual chapter and their charts (six in
all); the UC page carries the "structural time series" disambiguation from
Structural VAR. `state-space.md` gained the sequential multivariate filter,
with the note that it is cited to Durbin & Koopman directly. The handoff's
proposed names that differ from what was built (`DynamicFactorModel`,
`MarkovSwitchingModel`, `SETARSelection`, `transition_function`,
`factor_selection`) are the built `DFMModel`, `MSModel`, a NamedTuple,
`transition_plot`, `factor_selection_plot`.

**Manual 17-20** (Unobserved Components, Dynamic Factor Models, Regime
Switching, Threshold and Smooth-Transition Models), 18 charts. Numbered
17-20 as the handoff says -- no file renamed -- but placed in the navigation
after Structural VAR, so Plotting and Coming from R or Python stay last,
which the handoff also asks for (the two instructions conflict only if file
number must equal position). Each ends on Indian data, and each Indian
example found something other than what the handoff anticipated, recorded
as found: the IIP's seasonal variance is estimated at zero (STL's seasonal
absorbs the pandemic instead); India's annual macro series share a **trade**
factor, not a business cycle (growth and inflation R² 0.002 and 0.005, the
imports share a Heywood case); IIP growth's switching-variance regime is the
pandemic and its year-on-year echo; and the threshold test rejects linearity
on the full sample (p 0.044 / 0.006 / 0.008) but not before March 2020
(0.40 / 0.26 / 0.22). UC chapter carries the Montgomery-style tracking
signal: a Nile model fitted to 1871-1898 and frozen alarms in 1902.

**Book Part XI, "Series That Change Their Mind"**, chapters **52-56**, and
The Frontier moved to **57** in Part XII. The handoff numbered the new
chapters 53-57 with the Frontier at 58, which would have left no chapter 52;
renumbered to keep the sequence unbroken. 52 The Components You Cannot See
(filter vs smoother across the Aswan dam, the local level as the model
behind SES -- gain 0.267 from the variances vs ETS's alpha 0.246 --, Tsay
Ex. 11.3 reproduced, model-based seasonal adjustment, a 10.75-year lynx
cycle, an inline-SVG Kalman schematic); 53 When a Component Stops Moving
(variances set by hand, the irregular's profile likelihood, the pile-up at
zero by simulation, the half-chi-squared boundary LR test: J&J seasonal
LR 44.5, p 1.3e-11); 54 Many Series, Few Forces (land/ocean temperature,
identification checked by simulate-and-Procrustes, Bai-Ng choosing r = 5 at
N = 6 and the true 2 at N = 40, a ragged edge); 55 The Economy Has Two Moods
(Hamilton's Table 22.1 reproduced to every printed figure, his Fig. 22.4
redrawn against NBER dates -- 17 of 26 recession quarters flagged, the
extra quarters mostly the ones just before NBER peaks -- simulate-recover,
per-regime diagnostics); 56 A Different Model on Each Side of a Line (the
lynx asymmetry, the threshold's skewed sampling distribution by simulation,
Hansen's bootstrap, the additive LSTAR 67 times better than the convex
reading, sunspots on Hansen's transform, threshold 7.4234 = tsDyn). Every
quoted number is printed by an executed block on its page or cited to a
page; every textbook citation carries printed and PDF pages from the
handoff's verified offsets, and one page range written from memory (Tsay
§4.1.2-4.1.3) was removed rather than kept unverified.

**Corrected along the way:** the getting-started "what the package does not
do yet" list was three closed gaps (GARCH t, VAR/VECM, ARIMAX forecasting);
Appendix B's counts were stale from before Stage 9 (now 15,852 assertions,
303 fixture files in 41 directories, 34 disagreement and 24 Under-the-Hood
boxes) and it now records the Durbin & Koopman citation gap; the home page,
Frontier and R/Python chapter were updated for Stage 11.

**The `@example` trap is now a build failure:** `docs/check_example_blocks.jl`,
run by `docs/make.jl` before `makedocs`, parses every block with Julia's own
parser and fails the build on one that prints and then displays a value. Its
first run found three false positives in the existing book (`a; b` on one
line, and `foreach`), fixed in the checker, and nothing real.

**Code fixes found by writing:** two in `fit_uc` (above, in 11.1). The book
chapters cite only what each of the six books contains; the handoff's
§11 "what not to do" list was followed throughout.

## Downstream: SeasonalAdjustment.jl (separate package, starts once Stage 8 is stable)

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| S.1 | X-11 filters (Henderson 13/23-term, iterative seasonal factors) | 1.2 only — genuinely independent of everything else | Census Bureau X-11 method documentation · `seasonal` (R, wraps binary) |
| S.2 | RegARIMA — calendar regressors (trading day, Easter) | 8.3 (needs exogenous regressors — SARIMAX, not plain SARIMA) | X-13ARIMA-SEATS reference manual · `seasonal::seas()` (R, wraps binary) |
| S.3 | Automatic outlier detection (AO/LS/TC, TRAMO-style) | S.2, 2.4 | Gómez & Maravall, TRAMO/SEATS papers |
| S.4 | X-13 pipeline (orchestrates S.1–S.3) | S.1, S.2, S.3 | Census Bureau reference manual |
| S.5 | SEATS canonical decomposition + Wiener-Kolmogorov filters | 6.8 (base ARIMA/SARIMA — SEATS decomposes the fitted ARMA polynomial itself, doesn't inherently need X), 6.3 (smoother), S.2 | Gómez & Maravall (1996) SEATS papers — the hardest single piece in the whole roadmap, deliberately last |

---

## Development policy: reference, never port

Every row above is implemented **natively in Julia from its primary
source** — the original paper or textbook — never by translating another
package's source code, in any language, R/Python/Julia/C included.

R and Python references exist for two purposes only:

1. **Reading the paper is ambiguous, the reference implementation isn't.**
   When a paper leaves an implementation detail underspecified, check what
   a mature, well-tested package actually does before guessing.
2. **Validation.** After implementing a row, check the *output numbers*
   against the reference package on a standard series — not the reference
   package's code.

This matters most for Stage 6 (the state-space engine). statsmodels'
Kalman filter recursion is Cython, not Python — compiled specifically
because a pure-Python loop is too slow for that hot path. Julia doesn't
have that problem: a native Julia loop backed by `LinearAlgebra`/BLAS
reaches the same speed without a compiled sub-layer, which is the actual
point of building this in Julia rather than wrapping or porting something
else. The same policy resolves the earlier open questions about
`StateSpaceModels.jl` and `MarSwitching.jl` — read them for validated
design patterns (their diffuse-initialization handling, their
hyperparameter constraint transforms), implement independently, check
numbers against them.

## How to use this with Claude Code

Each row is small enough to be one focused session: implement the function,
write the test against the cited reference's numbers on a standard series
(AirPassengers, Nile, sunspots — whatever the reference package itself uses
in its own examples), then move to the next row in sequence. Cross off
completed rows in this file directly — it doubles as a changelog once you
start.

**A row isn't done until it has a docstring with a verified example**, not
just a passing test — see the "Documentation policy" section of
`README.md`. `docs/make.jl`'s `checkdocs=:exports` and `doctest=true`
enforce this in CI, so treat a docs build failure the same as a test
failure, not a follow-up task.
