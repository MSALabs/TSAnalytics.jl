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
| 1.3 | ACF / PACF with confidence bands **✅ built** (natively, replacing the earlier thin `StatsBase.jl` wrapper — see `handoff/stage-1.3-acf-pacf-handoff.md`; `StatsBase` dropped from `Project.toml` entirely, no longer used anywhere. `acf` gained `adjusted`/`bartlett`/`qstat`; `pacf` gained `method=:yw`/`:ywm`/`:ols` — `:burg` explicitly not implemented, a documented gap. Two verified R-vs-Python default discrepancies documented in the handoff: Bartlett bands vs. simple bands, and R's `pacf()` matching statsmodels' `:ywm`, not its `:yw` default. NaN input rejected (`na.fail`-style); no partial missing-data policy yet) | 1.2 | `stats::acf`/`pacf` (R) · `statsmodels.tsa.stattools.acf`/`pacf` (Python) |
| 1.4 | QR-based OLS/GLS regression helper **✅ built** (GLS via weighted QR; `method=:qr`\|`:cholesky` added — see `handoff/stage-1.4-ols-cholesky-handoff.md` — both agree to ~1e-15/1e-16, `:cholesky` fails informatively on collinear regressors rather than raising a raw `PosDefException`. Deliberately kept private/unexported, matching R's `lm.fit()`/GLM.jl's `DensePredQR`/`DensePredChol` role as an internal engine, not a public regression API — reused internally by `adf_test`, `kpss_test`, and Stage 1.3's `pacf(...; method=:ols)`. Pivoted/rank-deficient handling (`dropcollinear`) deliberately deferred, same reasoning both times) | nothing (LinearAlgebra only) | `lm.fit()` internals (R) · `statsmodels.OLS` / `numpy.linalg.lstsq` (Python) · GLM.jl's `DensePredQR`/`DensePredChol` design (validated reference, not a dependency) |

*Popularity note: 1.3 is disproportionately used relative to its complexity — correctly sequenced first regardless of the dependency graph, since almost every later diagnostic calls it.*

---

## Stage 2 — Diagnostic tests

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 2.1 | ADF unit-root test **✅ built** (rewritten to argument-match `statsmodels.tsa.stattools.adfuller` exactly — `regression=:n\|:c\|:ct\|:ctt`, `maxlag`, `autolag=:aic\|:bic\|:tstat\|nothing` — see `handoff/stage-2.1-adf-handoff.md`. Neither R package offers a clean match: `tseries::adf.test` has no type-flexibility at all, always fits constant+trend; `fUnitRoots::adfTest` offers 3 of the 4 `regression` options under the name `type`, no `:ctt`. Two genuine bugs found and fixed beyond what the handoff proposed: the default `maxlag` formula is `ceil`, not `floor` (verified from `adfuller`'s actual source, with a sample-size cap the handoff draft didn't have at all), and the `:tstat` significance threshold is the one-sided `norm.ppf(0.95)≈1.6449`, not `1.96`. `:ctt` now has a real MacKinnon critical-value table (extracted from `statsmodels.tsa.adfvalues.mackinnoncrit`), not a `:trend`-table fallback. Validated exactly (atol=1e-4) against `statsmodels` across all 12 `regression`×`autolag` combinations on a shared reference series) | 1.1, 1.4 | `tseries::adf.test` / `fUnitRoots::adfTest` (R) · `statsmodels.tsa.stattools.adfuller` (Python) |
| 2.2 | KPSS stationarity test **✅ built** (argument-renamed to match `statsmodels.tsa.stattools.kpss` — `regression=:c\|:ct` — for consistency with `adf_test` too; `nlags=:short\|:legacy\|:auto\|Integer` — see `handoff/stage-2.2-kpss-handoff.md`. Core math/critical-value table were already correct, unlike 2.1. One bug found beyond the handoff, the same shape as 2.1's: `:legacy`'s formula is `ceil(12*(n/100)^0.25)`, not `trunc`, verified from `kpss`'s actual source (its docs read as truncation) — matching Python's `"legacy"` exactly, confirmed; whether it's bit-identical to R's `lshort=FALSE` is unconfirmed (no R available). `:auto` (Python's actual default — Hobijn et al. 1998 data-dependent method) explicitly not implemented, throws rather than silently substituting `:short`. Validated exactly against `statsmodels` across regression×nlags combinations on the same reference series as 2.1) | 1.4 | `tseries::kpss.test` (R) · `statsmodels.tsa.stattools.kpss` (Python) |
| 2.3 | Phillips-Perron test **✅ built** (as `pp_test`/`PPTest`, argument name `trend` — matching `arch.unitroot.PhillipsPerron` specifically rather than `adf_test`/`kpss_test`'s `regression`, a deliberate naming inconsistency since `arch` itself uses a different name than `adfuller`/`kpss` for the same concept — see `handoff/stage-2.3-pp-handoff.md`. Algorithmically distinct from ADF, not a variant: no augmenting lags in the regression at all, serial correlation corrected via the same Newey-West/Bartlett long-run-variance adjustment `kpss_test` already uses. Verified to an unusually strong standard before any Julia was written — the handoff's author transcribed `arch`'s actual source to Python and confirmed it reproduced real `arch` output to ~1e-13; independently re-confirmed here across all 6 `trend`×`test_type` combinations against real `arch` (installed, version 5.1.0) on the shared reference series. One implementation bug found and fixed beyond the handoff's draft: its `cols = [rhs_y]` column-list construction inferred its element type from the first pushed column rather than being declared `Vector{Vector{Float64}}` up front, so container-agnostic (e.g. integer-range) input crashed on `push!` — fixed to match `adf_test`'s already-safe pattern. `test_type=:rho`'s p-value is honestly `NaN` (its asymptotic null distribution isn't the tau/t-distribution family the interpolation table covers), a documented gap, not a wrong number) | 1.4 (shares machinery with 2.1/2.2) | `tseries::pp.test` / `stats::PP.test` (R) · `arch.unitroot.PhillipsPerron` (Python) |
| 2.4 | Ljung-Box / Box-Pierce portmanteau, incl. seasonal-lag variant **✅ built** (extended: `lags=nothing` now defaults to `min(10, n÷5)` matching Python's current `acorr_ljungbox`; `boxpierce=true` computes both statistics, populating new `LjungBoxTest.bp_statistic`/`.bp_pvalue` fields, backward-compatible 4-arg constructor kept — see `handoff/stage-2.4-ljungbox-handoff.md`. Deliberately does **not** default-match R, whose `Box.test()` defaults to the older, weaker Box-Pierce statistic (both Wikipedia and statsmodels' own docs note Ljung-Box has better finite-sample properties). Core math was already correct (confirmed in Stage 1.3). Headline finding, verified directly against real `statsmodels`: passing a lag **vector** means something different across languages — Python's `acorr_ljungbox(y, lags=[5,10])` reports two separate *cumulative* statistics (through lag 5, through lag 10, as two rows); this package's vector path sums **exactly** those lags with nothing from lags 1-4/6-9, which `qs_test` genuinely depends on. Not a bug, kept as-is, now documented loudly rather than left implicit. Validated exactly against real `statsmodels` for both the cumulative (`Integer`) and exact-set (`Vector`) paths, the latter computed independently since `acorr_ljungbox` has no way to produce it directly) | 1.3 | `stats::Box.test` (R) · `statsmodels.stats.diagnostic.acorr_ljungbox` (Python) |
| 2.5 | QS seasonal-residual test **✅ built** (real correctness bug found and fixed: the pre-existing implementation squared raw autocorrelations at the seasonal lags without first clipping negative values to zero — JDemetra+'s documented formula uses `max(0, rho)` before squaring, since a negative correlation at a seasonal lag isn't evidence of seasonality. Confirmed directly from JDemetra+'s own published documentation (web search, not just the handoff's transcription) — over 2x difference in the statistic on a constructed example, enough to flip a significance conclusion. `qs_test` rewritten as a thin wrapper over `ljungbox_test(x, [period, 2*period]; clip_negative=true)`, matching JDemetra+'s own `QSTest`-calls-`LjungBoxTest` architecture (also confirmed from its documentation) rather than duplicating the chi-square logic. New `ljungbox_test(...; clip_negative=...)` keyword exposed generally, not hidden as a QS-only internal — see `handoff/stage-2.5-qs-handoff.md`) | 1.3, 2.4 | X-13ARIMA-SEATS/JDemetra+ output diagnostics (no direct R/Python package; documented in JDemetra+ and Census Bureau reference manuals) |
| 2.6 | Jarque-Bera normality test **✅ built** (as `jarque_bera_test`/`JarqueBeraTest`, built fresh — see `handoff/stage-2.6-jarque-bera-handoff.md`. The one Stage 2 test where R and Python formulas agree exactly — no default-behavior archaeology needed; confirmed independently against real `statsmodels.stats.stattools.jarque_bera` output to full floating-point precision (not just the handoff's own transcription check) on the shared reference series. Returns `skewness`/`kurtosis` directly, matching Python's richer return value — R's version only gives the statistic/p-value) | 1.4 (uses residual moments) | `tseries::jarque.bera.test` (R) · `statsmodels.stats.stattools.jarque_bera` (Python) |

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

## Stage 6 — ARIMA, SARIMA, and full ETS

**Includes former Stage 8.2**, folded in here since full state-space ETS
reuses this same time-invariant engine (6.1–6.4) directly. Engine and
models together, in one stage, since none of it needs the harder half of
the state-space machinery — everything here only requires a
**time-invariant** state-space form (fixed `Z`, `T`, `R` every period)
with **stationary** initialization via the Lyapunov equation, sufficient
because, after differencing, the remaining ARMA process is itself
stationary.

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
| 6.5 | Non-seasonal ARMA via state space + MLE | 6.4, 4.2 | `stats::arima` (R) · `statsmodels.tsa.arima.model.ARIMA` (Python) |
| 6.6 | Integration → ARIMA(p,d,q) | 6.5, 1.1 | `forecast::Arima` (R) · same statsmodels class |
| 6.7 | Seasonal ARIMA — multiplicative polynomial | 6.6 | `forecast::Arima(seasonal=)` (R) · `statsmodels SARIMAX` (Python, with `exog=nothing`) |
| 6.8 | Auto-order selection (Hyndman-Khandakar) | 6.7, 2.1, 2.2 | Hyndman & Khandakar (2008) paper · `forecast::auto.arima` (R) · `pmdarima.auto_arima` (Python) — doesn't need exogenous regressors either, `auto.arima` works fine without `xreg` |
| 6.9 | Full ETS (all 30 error/trend/season combinations, state-space, AICc selection) *(was 8.2)* | 6.1–6.4 — **to confirm when reached**: local-level/trend components are non-stationary, so this may need Stage 8's exact diffuse init, or a simpler large-variance approximate-diffuse initialization; worth checking what statsmodels' `ETSModel` and `StateSpaceModels.jl`'s `ExponentialSmoothing` actually do here before assuming the heavier dependency is required | `forecast::ets` (R) · `statsmodels ETSModel` (Python) |

*This is the highest-popularity single item in the whole roadmap — ARIMA/SARIMA plus auto-order selection is what most users mean when they say "does this package do ARIMA." Stage 6 delivers all of it, plus full ETS, self-contained, without touching diffuse initialization at all — that piece is isolated entirely in Stage 8.*

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
| 7.1 | ARCH / GARCH(1,1) and general GARCH(p,q) | 4.1 (own likelihood, not the Gaussian SSM) | `rugarch`/`fGarch` (R) · `arch` package by Kevin Sheppard (Python) |
| 7.2 | EGARCH, GJR-GARCH (asymmetric) | 7.1 | `rugarch` (R) · `arch` (Python) |
| 7.3 | Volatility forecasting | 7.1 | `rugarch::ugarchforecast` (R) · `arch.forecast` (Python) |
| 7.4 | Realized volatility measures | nothing beyond raw data | `highfrequency` (R) · `arch.realized` (Python) |

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
| 8.1 | Generalize `GaussianSSM` to time-varying Z_t, T_t, R_t, Q_t, H_t | 6.1 | Durbin & Koopman ch. 3–4, time-varying case · `statsmodels.tsa.statespace.representation` (Python) |
| 8.2 | Kalman filter, **exact diffuse** initialization | 6.2, 8.1 | Durbin & Koopman ch. 5 · `statsmodels.tsa.statespace` diffuse init (Python) — the piece plain `stats::arima` in R actually lacks, worth noting as a place TSAnalytics can exceed a common reference |
| 8.3 | Exogenous regressors via diffuse-state augmentation → full SARIMAX | 8.2, 6.7 | `forecast::Arima(xreg=)` (R) · `statsmodels SARIMAX(exog=)` (Python) · de Jong (1991) on diffuse filtering for regression |
| 8.4 | Auto-order selection extended to the exogenous-regressor case | 8.3, 6.8 | `pmdarima.auto_arima(X=...)` (Python) |

---

## Stage 9 — Deferred ETS extras (not required for v1.0)

**Former Stage 8.3–8.5.** None of these actually depend on Stage 7 or 8 —
damped trend only needs 6.9, Theta needs 3.2+5.5, TBATS needs 6.9+3.3 — so
they're technically buildable any time after Stage 6. Positioned here,
after the v1.0 boundary, simply because they weren't part of the explicit
v1.0 commitment; pull them forward opportunistically if there's spare
capacity during Stage 6/7 work.

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 9.1 | Damped trend variants *(was 8.3)* | 6.9 | `ets(damped=TRUE)` (R) · `ETSModel(damped_trend=True)` (Python) |
| 9.2 | Theta method *(was 8.4)* | 3.2 (STL) + 5.5 | Assimakopoulos & Nikolopoulos (2000) paper · `forecast::thetaf` (R) · `statsmodels ThetaModel` (Python) |
| 9.3 | TBATS *(was 8.5)* | 6.9, 3.3 | De Livera, Hyndman & Snyder (2011) paper · `forecast::tbats` (R) |

---

## Stage 10 — Multivariate (independent track, only needs Stage 1 regression)

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 10.1 | VAR + lag-order selection (AIC/BIC/HQ) | 1.4 | `vars::VAR` (R) · `statsmodels VAR` (Python) |
| 10.2 | Granger causality | 10.1 | `vars::causality` (R) · `statsmodels.tsa.stattools.grangercausalitytests` (Python) |
| 10.3 | Impulse response functions, FEVD | 10.1 | `vars::irf`/`vars::fevd` (R) · `VARResults.irf` (Python) |
| 10.4 | Cointegration — Engle-Granger | 2.1, 1.4 | `tseries::po.test` (R) · `statsmodels.tsa.stattools.coint` (Python) |
| 10.5 | Cointegration — Johansen | 10.1 | `urca::ca.jo` (R) · `statsmodels.tsa.vector_ar.vecm.coint_johansen` (Python) |
| 10.6 | VECM | 10.5 | `tsDyn` (R) · `statsmodels VECM` (Python) |
| 10.7 | Structural VAR (SVAR) | 10.1 | `vars::SVAR` (R) — weak Python support, a real differentiation opportunity |

*This entire stage can start as soon as Stage 1 is done — it doesn't wait on the state-space engine at all. Worth running in parallel with Stage 6–8 if resourcing allows, since it's a fully independent dependency branch.*

---

## Stage 11 — Advanced state-space applications (reuse Stage 6/8)

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 11.1 | Unobserved Components / structural time series | Stage 8 — local level/trend components are non-stationary by construction, genuinely needs diffuse (or approximate-diffuse) initialization, unlike plain ARIMA | `stats::StructTS`, `bsts` (R) · `statsmodels UnobservedComponents` (Python) |
| 11.2 | Dynamic factor models | Stage 8, 10.1 (multivariate) | `dfms` (R) · `statsmodels DynamicFactor` (Python) |
| 11.3 | Markov-switching AR/regression | 4.1, 5.1 | `MSwM` (R) · `statsmodels MarkovAutoregression` (Python) |
| 11.4 | Threshold AR (TAR/STAR) | 5.1 | `tsDyn::setar` (R) — weak Python support |

---

## Stage 12 — Niche but self-contained (schedule opportunistically)

| # | Functionality | Depends on | Reference |
|---|---|---|---|
| 12.1 | Croston's method / SBA / TSB (intermittent demand) | 5.5 | `forecast::croston` (R) · `statsforecast` (Python) |
| 12.2 | Change-point detection (PELT, binary segmentation) | 1.4 only | `changepoint` (R) · `ruptures` (Python) |
| 12.3 | Lag-feature regression forecasters (ML reduction) | 5.2, 5.3 | `forecastML` (R) · `mlforecast`, `sktime` reduction (Python) |

*(Feature extraction, formerly listed here, moved to its own tracked
section — see "Downstream: TSFeatures.jl" right after Stage 5.)*

---

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
