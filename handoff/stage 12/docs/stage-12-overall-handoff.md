# Stage 12 — Niche but self-contained
## Overall handoff (the charter)

**Written:** 2026-10-15
**Repository HEAD at writing:** `2eccb82` ("Stage 11 documentation: API pages, manual 17-20, book Part XI")
**Scope:** Stage 12.1 through 12.3, plus plot recipes and documentation
**Read this before any sub-stage handoff.**

---

## 0. Tagging convention

Unchanged from Stages 10 and 11.

| Tag | Meaning |
|---|---|
| **[V]** | **Verified by execution in this sandbox.** A program was run, the numbers are its output, the script is named. |
| **[S]** | **Read from source.** File and line given. Not executed. |
| **[U]** | **Unverified / carried forward.** A claim to test, not a fact to build on. |

Untagged statements are *decisions*, binding regardless of evidence.

---

## 1. What Stage 12 is

The roadmap lists three items. The survey changes two of them, and one change is large enough that it reshapes the stage.

### Revised item list

| # | Item | Roadmap says | Change |
|---|---|---|---|
| 12.1 | Intermittent demand: Croston, SBA, TSB, ADIDA, IMAPA, ADI/CV² classification | "Croston's method / SBA / TSB", depends on 5.5 | **Scope widened** to five methods plus the classifier; `forecast` has **only** classical Croston [V] |
| 12.2 | **Change-point detection AND structural-break testing** | "Change-point detection (PELT, binary segmentation)", depends on "1.4 only" | **Merged** with the "structural-break and CUSUM tests" gap the book's Frontier chapter lists separately. Three references agree on one fixture [V] |
| 12.3 | Lag-feature regression (ML reduction) | "Lag-feature regression forecasters (ML reduction)", depends on 5.2, 5.3 | **Substantially rescoped — most of it already exists.** §1.2 |

### 1.1 The 12.2 merge

`development-sequence.md` lists change-point detection; `docs/src/introduction/57-the-frontier.md` separately lists "Structural-break and CUSUM tests — `nyblom_test` says the parameters moved; these say *when*". They are one item: `strucchange::breakpoints` is Bai–Perron dynamic programming, which is `changepoint`'s SegNeigh with a regression cost. Building them separately would mean writing the same dynamic program twice.

The merge is also what makes the item verifiable. On a 300-point fixture with true breaks at 100 and 200, all three references agree [V]:

| reference | result |
|---|---|
| `changepoint` 2.3, PELT at pen 10/20/50, BinSeg Q=2, default MBIC | `100, 200` |
| `ruptures` 1.1.10, `Pelt(l2)` pen 10/20/50, `Binseg(n_bkps=2)` | `100, 200, 300` |
| `strucchange` 1.5.3, `breakpoints(y ~ 1)` | `100, 200` |

and `strucchange` adds what the other two lack: break-date confidence intervals `[99, 101]` and `[199, 201]`, BIC selecting two breaks at 838.563795, and supF = 90.053285.

### 1.2 The 12.3 rescope — the most important finding in the stage

**`arx` plus `forecast(::ARXModel)` already is the recursive ML reduction.** Not approximately — numerically.

On an AR(2) fixture with ordinary least squares injected as the regressor, four independent implementations across two languages agree [V]:

| implementation | const | y.L1 | y.L2 |
|---|---|---|---|
| `arx`-style construction, reproduced independently in this session | 1.010348499228957 | 0.602697210185463 | −0.358379118553815 |
| `mlforecast` 1.1.0 + `LinearRegression` | 1.0103484992289584 | 0.60269721018546285 | −0.35837911855381488 |
| `sktime` 1.2.0 recursive + `LinearRegression` | 1.0103484992289582 | 0.60269721018546285 | −0.35837911855381466 |
| `forecastML` 0.9.0 + `lm` | 1.010348499228958 | 0.60269721018546274 | −0.35837911855381488 |

Recursive forecasts for h = 1…6 agree to **2e-15** across all four. [V] `s12/mlreduction/out/core_ols.txt`, and reproduced independently here.

The structural reason is visible in the source. `arx` builds the design with `hold_back = maximum(lags)`, supports arbitrary lag subsets, trend and seasonal dummies [S] `src/arx.jl:126-200`. And `_arx_forecast_row`'s own docstring describes reading lag terms "from `yext` (the original series extended in-place with each step's own forecast so far)" [S] `src/forecast.jl` — which is precisely the recursive strategy.

**So Stage 12.3 must not build a `ml_reduction` that re-derives a lag matrix.** It would duplicate `arx` to 2e-15 and create a second place for the design-matrix convention to drift. The 12.3 handoff rescopes the item to the four things genuinely missing.

### Difficulty, measured

| # | Item | Difficulty | Dual-verifiable? | Dominant cost |
|---|---|---|---|---|
| 12.1 | Intermittent demand | **Easy** | **yes, to 17 digits** [V] | the degenerate-input policy and the absent intervals, not the forecasting |
| 12.2 | Change-point + breaks | **Moderate, with one hard island** | **yes, three ways** [V] | penalty conventions; and the supF null distribution, which is a *licensing* problem |
| 12.3 | Lag-feature regression | **Low–Moderate** | **yes, to 2e-15** [V] | deciding what not to build |

---

## 2. Governing principles (inherited, non-negotiable)

Restated because every one earned its place.

1. **Run it, don't recall it.** This charter's own figures were re-verified by hand after the surveys: `changepoint`'s MBIC suboptimality (2.06672005), `ruptures`' jump default, `sktime`'s lag-0 leak, and the `arx` equivalence were each reproduced independently before being written down.
2. **Reference, never port.** §4 makes this a licensing requirement — though for the first time in four stages it is a *lighter* constraint, because the primary references are mostly permissive.
3. **The honest-refusal pattern.** Throw `ArgumentError` naming what is missing. Stage 12 has an unusually sharp instance: both intermittent-demand references agree on an answer that is a **10× error** (§5.2), and neither refuses.
4. **Reduction testing.** Stage 12's reductions are unusually strong — three of them are exact identities (§8.2).
5. **`predict` and `forecast` are exact aliases** (`src/forecast.jl:434`). [U]

---

## 3. Reference environment (reproducible)

| Component | Version | How obtained |
|---|---|---|
| R | 4.3.3 | pre-installed | [V] |
| `forecast` | 8.21.1 | pre-installed | [V] |
| `strucchange` | 1.5.3 | pre-installed | [V] |
| `changepoint` | 2.3 | clone `github.com/cran/changepoint` + `R CMD INSTALL` | [V] |
| `changepoint.np`, `ecp`, `wbs`, `fpop`, `Segmentor3IsBack` | — | same route, **all five installed** | [V] |
| `forecastML` | 0.9.0 | same route, `--no-byte-compile`; **no extra deps needed** | [V] |
| `statsforecast` | 2.1.1 | `pip install` | [V] |
| `ruptures` | 1.1.10 | `pip install` | [V] |
| `mlforecast` | 1.1.0 | `pip install` (pulls optuna, sqlalchemy, alembic) | [V] |
| `sktime` | 1.2.0 | `pip install` | [V] |
| `scikit-learn` | 1.9.1 | pre-installed | [V] |

**CRAN remains unreachable**; the GitHub mirror route works for every R package tried. PyPI is reachable.

### Which reference serves which item

| Item | Primary (transcribable) | Independent check | Agreement |
|---|---|---|---|
| 12.1 | `statsforecast` (Apache-2.0) | `forecast::croston` (GPL-3) | **17 significant digits, and on all 13 edge cases** [V] |
| 12.2 detection | **`ruptures` (BSD-2)** | `changepoint`, `fpop`, `wbs`, `Segmentor3IsBack` (GPL/LGPL) | **96/96 configurations identical** [V] |
| 12.2 inference | — *(none permissive)* | `strucchange` (GPL) | mechanism reproduced to diff = 0; **tables are GPL** — §6 |
| 12.3 | `forecastML` (MIT), `sktime` (BSD-3), `sklearn` (BSD-3), `mlforecast` (Apache-2.0) | — | **2e-15 across four implementations** [V] |

---

## 4. Licensing — lighter than any previous stage, with one exception

| Package | License | Permitted use |
|---|---|---|
| `ruptures` 1.1.10 | **BSD-2-Clause** [V] | **Numbers and structure transcribable with attribution** |
| `forecastML` 0.9.0 | **MIT + file LICENSE** [V] | transcribable |
| `sktime` 1.2.0, `scikit-learn` 1.9.1 | **BSD-3-Clause** [V] | transcribable |
| `statsforecast` 2.1.1 | **Apache-2.0** [V] | transcribable — **note this is Apache, not BSD/MIT**: it carries a NOTICE-preservation clause the others do not |
| `mlforecast` 1.1.0 | **Apache-2.0** [V] | transcribable, same caveat |
| `forecast` 8.21.1 | **GPL-3** [V] | independent check only |
| `changepoint` 2.3, `strucchange` 1.5.3, `changepoint.np`, `ecp`, `wbs`, `Segmentor3IsBack` | **GPL / GPL-2 / GPL ≥2** [V] | independent checks only |
| `fpop` | **LGPL ≥2.1** [V] | still copyleft — treat as an independent check |
| `tsintermittent`, `expsmooth` | **GPL ≥2** [V] | pattern reference only |

**This is the first stage where the primary references are mostly permissive**, which matters in two concrete ways: 12.2's detection side can follow `ruptures`' architecture directly rather than re-deriving it from behaviour, and 12.3's expectations come entirely from permissive sources.

**The one exception is the binding constraint of the stage.** §6.

### 4.1 One fixture carries encumbered data

The intermittent-demand survey built a fixture from `expsmooth::partx` (automobile part sales, n=51), which is **GPL ≥2 data** [V]. The *outputs* computed on it are ours; the *inputs* are not. **Do not ship it in `test/`.** Either keep it as a verification-only comparison or find a non-GPL intermittent series. The other three fixtures (F1–F3) are generated and fully clean.

---

## 5. Cross-cutting conventions — the traps, with evidence

### 5.1 The two intermittent-demand references agree to 17 digits, and the reason matters

`forecast::croston` and `statsforecast`'s `CrostonClassic` return **0.65909246562518586** on the shared fixture — identical to all 17 significant digits, on all four fixtures and all thirteen edge cases. [V]

This is not luck. Three things coincide [V/S]:

1. **Identical interval construction.** `_intervals(x) = np.diff(np.where(x != 0)[0] + 1, prepend=0)` and R's `diff(c(0, (1:length(x))[x>0]))` are the same expression.
2. **Identical initialisation.** R hardcodes `initial="simple"` inside `croston2`; `ses.cpp:40` sets `fitted[0] = x[0]`.
3. **Identical recursion**, both in float64 — statsforecast's Python-side float32 cast is lossless through the Eigen `VectorXd` boundary for small integer counts (verified: int64, float32 and float64 inputs all give the same 17 digits).

**Consequence for the test policy:** 12.1's tier-1 tolerance can be *exact* on point forecasts, which no previous stage could claim.

### 5.2 Both references agree on an answer that is a 10× error

The series `[5, 0, 0, 0, 0, 0, 0, 0, 0, 0]` — one arrival, at t=1 — returns **5** from both references. [V] Demand = `[5]`, intervals = `[1]`, so Croston returns 5/1. The empirical rate is 0.5.

**Neither reference refuses on any degenerate-but-non-negative input.** Both always return a number. This is the sharpest honest-refusal case in four stages: Croston cannot estimate an inter-arrival rate from a single arrival at t=1, and saying so is more useful than a number that is wrong by an order of magnitude.

### 5.3 The SBA debiasing factor is a hardcoded literal, not a formula

`statsforecast`'s `_croston_sba` is **5 code lines** and does `out["mean"] *= 0.95` [S] `models.py:5401`. It coincides with the literature's `1 − α/2` **only because `_croston_classic` hardcodes α = 0.1.** [V]

**If we expose `alpha`, we must compute `1 − α_interval/2`, not use 0.95.** A port that copies the constant produces a bias correction that silently stops being a bias correction the moment a user changes α.

### 5.4 `changepoint`'s own default is provably suboptimal

Re-verified by hand in this session. [V]

On a 0.5-sd shift at t=150 (`fxB_weak`), `cpt.mean(penalty="MBIC", method="PELT")` — **the package default** — returns τ = 187 with MBIC criterion **285.00735894**, where brute force over every candidate and `method="BinSeg"` both return τ = 139 at **282.94063889**. Suboptimality **2.06672005**, with no warning.

The cause is architectural: **MBIC is implemented as six extra cost functions, not as a penalty** [S] `cost_general_functions.c:58`, and the modified cost violates PELT's K = 0 pruning inequality. Measured: **143 of 190** random segment splits violate it under the MBIC cost, against **0 of 190** under plain RSS. [V]

Two consequences: **never use `changepoint`'s defaults as an oracle** (use `penalty="Manual"`, `method="SegNeigh"`, or `fpop`), and **our implementation must either refuse MBIC with PELT or assert the pruning condition per cost**.

### 5.5 `ruptures`' default `jump=5` silently returns wrong breakpoints

Re-verified by hand. [V] On a fixture with true breaks at **97 and 203** (deliberately off a multiple of 5):

| configuration | result |
|---|---|
| `Pelt(l2)` **at defaults** (jump=5), pen=10 | **`[95, 100, 200, 205]`** |
| `Pelt(l2)`, `jump=1, min_size=1`, pen=10 | `[84, 97, 203]` |

Two true breakpoints become four wrong ones, all multiples of five. **`ruptures` never documents that `jump > 1` makes PELT approximate** — the only text is "subsample (one every *jump* points)" [S] `pelt.py:25`.

**The shared fixture hid this entirely**, because 100 and 200 are both multiples of 5. **Every reference comparison must use off-grid breakpoints and pass `jump=1`.**

### 5.6 The penalty mapping needs no rescaling — which was worth checking

`changepoint`'s `mll_mean(x,x2,x3,n) = x2 - x*x/n` and `ruptures`' `CostL2.error = var·n_seg` are the same quantity (segment RSS) [S]. Measured over **8 fixtures × 6 penalties × 2 minimum segment lengths = 96 configurations**: identical **96/96**, provided `jump=1`. [V]

So `pen=10` in `ruptures` *is* `pen.value=10` in `changepoint`. The silent mismatch is `jump`, not the penalty scale. Corroborated independently: `fpop::Fpop(λ = 2 log n)` returns the identical 28 breakpoints on the variance fixture. [V]

### 5.7 The Gaussian mean-change cost is not scale-invariant, and nobody says so

`changepoint`'s `mll_mean` is a bare sum of squares, **not a log-likelihood** — σ² is implicitly assumed known and equal to 1 [S]. So a series with σ² = 4 is effectively penalised at `β/4` in standardised units, and `BIC = 2 log n` is **not scale-invariant**.

`strucchange` concentrates one global σ² out of the pooled RSS and minimises `n·log(Σ RSS_i) + log(n)(nreg+1)(m+1)` — a different objective with numerically the same per-break penalty (11.407565 at n=300, nreg=1). Verified to machine precision against `summary()`'s own table at every m. [V]

`ruptures`' `normal` cost instead estimates σ² **per segment**. Three references, three objectives.

### 5.8 The breakpoint index convention — the hazard is real, but not where expected

Verified on a 10-point deterministic step (five 0s then five 10s). [V]

| reference | output | meaning |
|---|---|---|
| `changepoint` `cpts()` | **5** | 1-based inclusive end of the preceding segment |
| `strucchange` `$breakpoints` | **5** | 1-based inclusive end |
| `ruptures` `predict()` | **[5, 10]** | **0-based exclusive** segment end, series end appended |
| `ecp::e.divisive` | **`1, 101, 201, 301`** | **1-based START of each segment** |

**There is no off-by-one among the three primary references** — a 0-based exclusive end always equals a 1-based inclusive end, so `changepoint.cpts == strucchange.breakpoints == ruptures.predict()[:-1]` numerically. But `ecp` genuinely differs, proving the hazard exists.

**The trap specific to Julia:** `ruptures`' documentation *describes* 100 as the start of the new regime, which in 1-based Julia is **101**. A port that reads the number as a 1-based start is off by one on every break.

### 5.9 Three incompatible defaults for the minimum segment length

[V] `changepoint` `cpt.mean` **1** (and 2 for `cpt.var`/`cpt.meanvar`, 3 for `cpt.reg`); `ruptures` **2**, floored up by each cost's own minimum (5 for `ar`, 3 for `clinear`); `strucchange` **`h = 0.15` → 45 at n = 300**.

This alone flips the answer on **three of eight fixtures** — on a fixture with 4-sd shifts at t=3 and t=297, `changepoint` and `ruptures` both return exactly `3, 297` while `strucchange` returns nothing at all, with supF p = 0.449. Not a bug: h = 45 makes a 3-long edge segment inadmissible. [V]

**A Julia API with one `min_segment_length` default will silently pick one reference's behaviour.** Document the default against all three.

### 5.10 The three ML-reduction references make three incompatible exogenous choices

[V] This is the silent trap of 12.3.

| | `mlforecast` | `sktime` | `forecastML` |
|---|---|---|---|
| default treatment | **contemporaneous** (known-future) | **lagged** (past-only), always | **lagged** (past-only) |
| other mode available | by hand only | **none** — contemporaneous X is impossible | `dynamic_features=` |
| future values at predict | `X_df` **required**, loud error if absent | no check — **silent garbage** | `predict_future=` function |

And `arx` takes exog **contemporaneously only** [S] `src/arx.jl:183`, while `forecast(::ARXModel)` **refuses to forecast at all** when exog is present — its docstring calls this "a real, deliberate gap, not an oversight" [S].

### 5.11 Reference defects found (continuing the charter list, D19 onwards)

Stage 11 ended at D18. Every one below found by execution.

| # | Reference | Defect | Evidence |
|---|---|---|---|
| **D19** | `changepoint` | **The package default (`penalty="MBIC"`, `method="PELT"`) returns a provably suboptimal segmentation**, by 2.06672005 on its own criterion, with no warning. MBIC is implemented as a cost, not a penalty, and violates PELT's pruning inequality in 143/190 measured splits. | [V] §5.4, re-verified here |
| **D20** | `ruptures` | **Default `jump=5` silently snaps breakpoints to a grid of five**, turning two true breaks into four wrong ones, and is never documented as making PELT approximate. | [V] §5.5, re-verified here |
| **D21** | `sktime` | **`WindowSummarizer(lag_feature={'mean':[[0,1]]})` produces a feature that IS `y_t`** — max \|feature − y\| = **0.000e+00**, in-sample R² = **1.0000000000** — accepted with no warning at any point, and the leak propagates into a fitted forecaster. `mlforecast` makes this structurally impossible (`lags must be positive integers`). | [V] re-verified here |
| **D22** | `sktime` | **`strategy="dirrec"` silently discards `X` entirely.** With-X and without-X results differ by **0.000e+00**. The source admits it: `# todo: logic for X below is broken. Escape X until fixed.` then `if X is not None: X = None`. No warning. | [V] / [S] `_reduce.py:1177,1221` |
| **D23** | `sktime` | **Recursive and direct strategies silently extrapolate missing future `X`** via `np.pad(..., "edge")`, giving garbage (max diff 0.9851 against the correct answer) with no warning. | [V] |
| **D24** | `statsforecast` | **SBA applies a hardcoded 0.95**, which is `1 − α/2` only because α is hardcoded at 0.1. Applied unconditionally, even to a single-observation series. | [V] / [S] `models.py:5401` |
| **D25** | `statsforecast` | **Internally inconsistent on negatives**: `_demand` uses `y > 0` but `_intervals`/`_probability` use `y != 0`, so a negative value counts as an arrival but contributes no demand. No input check at all; `forecast::croston` rejects negatives. | [V] |
| **D26** | `statsforecast` | **ADIDA/IMAPA aggregation level uses banker's rounding in the mean path and half-away-from-zero in the fitted path.** On a fixture where the mean interval is exactly 2.5, Python `round` gives 2 and C++ `std::round` gives 3 — a **20% difference** in the forecast turning on a tie-break. | [V] |
| **D27** | `strucchange` | **MOSUM p-values saturate at exactly 0.01 and CUSUM p-values underflow to exactly 0**, both via `approx(..., rule=2)` constant extrapolation. On a manufactured 50-sd break: OLS-MOSUM p = 0.01, OLS-CUSUM p = 0. Two failure modes in opposite directions, both reported as exact. | [V] |
| **D28** | `strucchange` | **supF with k > 40 silently caps k and returns an invalid p-value** in an `htest` object indistinguishable from a valid one. | [V] |
| **D29** | `changepoint` | **`pen.value` is silently discarded for named penalties.** `penalty="BIC", pen.value=1e6` gives BIC, no warning. | [V] |
| **D30** | `forecastML` | **`method="recursive"` returns an empty object with no error** — there is no `match.arg(method)`, so anything other than `"direct"`/`"multi_output"` falls through every branch. `method="banana"` also "succeeds". | [V] |
| **D31** | `forecastML` | **`lookback` validation checks the wrong bound** (`min(horizons)` where `max` is meant), so an invalid combination passes validation and crashes eight frames deep inside dplyr with no mention of lags or horizons. | [V] |
| **D32** | `forecast` | **`accuracy()`'s MASE denominator silently depends on whether the input was a `ts`.** On a bare numeric vector it uses mean absolute deviation about the mean, not a naive forecast: MASE 1.3400472874157467 against the correct 1.0885456588355449. | [V] |
| **D33** | `mlforecast` | **`date_features_as_dummies=True` produces a rank-deficient design** (12 month dummies with no baseline dropped plus an implicit intercept — rank 14 of 15, condition 1.686e+17) and sklearn returns a silent minimum-norm solution. `arx`'s explicit rank check would throw. | [V] |

| **D34** | `mlforecast` | *Appended 2026-10-15, building 12.3.* **`predict(h, X_df=…)` with fewer future rows than `h` silently forecasts anyway**: with 3 of 6 rows of `x_known`/`x_past` it returns six numbers, the last three differing from the full-X forecast (3.4255 against 4.9844 at h = 4) with no warning -- although it raises loudly when `X_df` is absent or missing a column. TSAnalytics refuses, naming the columns and the shortfall. | [V] `ground-truth/exog_py.txt` case (d) |

| **D35** | `utilsforecast` | *Appended 2026-10-15, building 12.1.* **`cfe`'s sign contradicts its own docstring.** The docstring says "Positive values mean under forecast"; the code computes `sum(model − target)`, so a total under-forecast of 2 returns **−2.0** (forecasts 1,1,1 against actuals 2,0,3). TSAnalytics' `cfe` is `sum(actual − predicted)`, positive for under-forecasting, as documented there. | [V] executed on utilsforecast as installed with statsforecast 2.1.1; [S] `losses.py:241` |

**D19, D20, D21 and D24 are the four that change what gets built.**

---

## 6. The one hard island: the supF null distribution is a licensing problem, not a mathematical one

`strucchange`'s `pvalue.Fstats` is **38 lines of code plus 2234 lines of embedded coefficient tables** [S]. It is Hansen (1997), *Approximate asymptotic p values for structural-change tests*, JBES 15, 60–67 — and it is **not** a critical-value table. It is a fitted chi-square approximation with non-integer degrees of freedom: look up a row block by `k` and a 25-point `tau` grid, form a polynomial in the statistic, and return `1 − pchisq(dummy, df = fitted_df)`.

The survey **reimplemented the mechanism from the formula plus the table and reproduced strucchange's p-values to diff = 0 on five fixtures** [V]. So the algorithm is fully characterised. The obstacle is that the 2234 lines of coefficients are GPL.

**Replacing them means simulating the limiting process over a 40 × 25 (k, tau) grid and refitting the approximation** — a genuine research-flavoured sub-project.

**The good news, and it is substantial:**

- **Break-date confidence intervals need no table.** `pargmaxV` is **27 lines of closed-form CDF** (Bai 1997), two branches in terms of Φ and exp [S]. Fully implementable from published mathematics.
- **The CUSUM family needs no tables.** Rec-CUSUM has a closed form; OLS-CUSUM, Score-CUSUM and RE use a Kolmogorov–Smirnov series. Only the **MOSUM** family needs tables. [S]

**Decision: scope the supF/aveF/expF p-values as a separate, optional item.** Ship 12.2 with detection, Bai–Perron dating, break-date confidence intervals and the CUSUM family — all fully implementable — and report the supF *statistic* with critical values from simulation, or not at all, rather than letting one table block the stage. The 12.2 handoff says exactly what to build and in what order.

---

## 7. Performance policy

### 7.1 The measured bar is low, and compiled Julia should clear it easily

At n = 2000, all returning the correct four breakpoints [V]:

| implementation | elapsed |
|---|---|
| `changepoint` PELT (C) | **0.002 s** |
| `ruptures` `KernelCPD(linear)` (C extension) | 0.003 s |
| `ruptures` `Pelt(l2)` at defaults (jump=5) | 0.212 s |
| `ruptures` `Pelt(l2)` jump=1 (exact) | **6.122 s** |
| `strucchange breakpoints` | **10.367 s** |
| `ruptures` `Dynp(l2)` | 53.350 s |

Exact PELT in C is **~3000×** faster than exact PELT in pure Python and **~5000×** faster than `strucchange`'s O(n²) pure-R triangular RSS. **Julia should land near 0.002 s**, and that is the number to record.

### 7.2 Where the work actually is

PELT's core is a scalar dynamic program over `cumsum` and `cumsum²`. The costs are closed forms on `(Σy, Σy², n)` — all twelve of `changepoint`'s are **111 lines of C total** [S]. So:

- Precompute the prefix sums once; every segment cost is then O(1).
- `@views` and `mul!` throughout the Bai–Perron triangular RSS, which is the one O(n²) piece.
- Pre-allocated buffers **typed on the parameter eltype**, not `Vector{Float64}`.
- `@code_warntype` clean, `@allocated` zero inside the loops, asserted.

### 7.3 Threading

`Threads.@threads` over the candidate grid in the Bai–Perron path and over bootstrap replications. **Heed the Stage 11 lesson**: threading `fit_uc`'s restarts exposed a closure data race where names assigned inside the loop body were also assigned by the enclosing function, so Julia boxed one variable shared across threads — four different optima in six runs. Declare `local`, and add a determinism test that requires identical results across repeated threaded runs.

### 7.4 What not to optimise

The intermittent-demand forecasts are **flat in h** — all 24 measured (fixture × model) point-forecast vectors are bit-identical across h = 1…6 [V]. There is no recursion to optimise. And for the fitted-value path, prefer the single-pass O(n) form: R recomputes the whole Croston from scratch on every prefix, which is O(n²) and the source of the only disagreement between the two references (§8.1).

---

## 8. Test policy

The tiered policy from Stage 11 stands, but Stage 12's references agree far more readily, so **tier 1 is tighter here than in any previous stage.**

### 8.1 The three tiers, per item

| Item | Tier 1 (what is determined) | Tolerance |
|---|---|---|
| **12.1 point forecasts** | the forecast at a given α | **exact** — both references agree to 17 digits on all fixtures and all 13 edge cases [V] |
| **12.1 fitted values** | the in-sample fitted path | **`rtol=1e-12`, not exact** — R recomputes per-prefix (O(n²)), statsforecast does one pass; mathematically identical, ~20 ulp of accumulated summation-order error [V] |
| **12.2 detection** | breakpoints at a given penalty and cost | **exact** — 96/96 configurations identical with `jump=1` [V] |
| **12.2 Bai–Perron BIC** | the RSS/BIC table | **machine precision** — verified diff = 0 at every m [V] |
| **12.2 break-date CIs** | `confint` bounds | **exact integers** — `[99,100,101]`, `[199,200,201]`, `[112,139,162]` [V] |
| **12.3 with OLS injected** | coefficients and recursive forecasts | **2e-15** across four implementations [V] |

**Tier 2** (fitted optima under a non-linear model) **does not arise in Stage 12**, because the no-dependency rule means OLS is the injected default and the whole pipeline is deterministic.

**Tier 3** (qualitative) covers the degenerate-input cases in §5.2 and the boundary behaviour in 12.1's edge-case table.

### 8.2 Mandatory reductions — three are exact identities

1. **PELT must equal SegNeigh/dynamic programming** for any cost satisfying the K = 0 pruning inequality. This is exactness checkable for free, and it is how D19 was found. Assert it, and **assert the pruning condition per cost** so MBIC cannot silently break it.
2. **The recursive reduction with OLS injected must equal `arx` + `forecast(::ARXModel)`** to 1e-14. §1.2 — four implementations already agree to 2e-15, and `arx` is the cheapest oracle we own.
3. **DirRec ≡ Direct and multi-output ≡ Direct, exactly, under OLS.** Measured 4.441e-16 and 2.220e-16 [V], by OLS orthogonality. **This is why neither strategy should be built** (§9.5) — they are untestable as distinct strategies with a linear model, and diverge only for a regressor this package will never ship (random forest: 0.309531).

Additionally: **Croston at α = 0.1 with one non-zero must reproduce the edge-case table exactly**, including the 10× error at `[5,0,…,0]`, as a pinned record of what the references do and what we refuse.

### 8.3 Coverage targets

Per sub-stage: **≥ 15 testsets, ≥ 250 assertions**, with every reference figure in the handoff appearing in at least one test. Stage 11's shortfall on 11.0 and 11.2 (77 and 87 runtime tests against 200 and 234 asked) is the thing to avoid repeating — and Stage 12's smaller scope makes it achievable.

### 8.4 Test the defects

Each of D19–D33 that affects a user-visible choice gets a test asserting **our** behaviour with the reference defect named in a comment. D19, D20, D21 and D24 are mandatory.

### 8.5 Two methodology rules the surveys learned the hard way

1. **Rank-check every fixture design.** A deterministic sinusoid satisfies `x_t = 2cos(ω)x_{t−1} − x_{t−2}`, so any number of its lags spans only two dimensions — measured rank 2/6. R's `lm` silently returns `NA` coefficients where `arx`'s rank check throws, and a survey nearly filed its own degenerate fixture as a reference bug. [V]
2. **Parameterise the leakage probe by strategy.** A uniform "target == `y_t`" assumption produces a false positive on multi-output reductions, where row `t` legitimately has `y_lag_1 = y[t]` and targets `y[t+1…t+6]`. [V]

---

## 9. What NOT to do

1. **Do not use `changepoint`'s defaults as an oracle.** D19 — `penalty="MBIC", method="PELT"` is suboptimal. Use `penalty="Manual"`, `method="SegNeigh"`, or `fpop`.
2. **Do not compare against `ruptures` without `jump=1`, and not on round-numbered breakpoints.** D20.
3. **Do not build a `ml_reduction` that re-derives the lag matrix.** §1.2 — it would duplicate `arx` to 2e-15.
4. **Do not build DirRec or multi-output.** §8.2 — provably identical to Direct under OLS.
5. **Do not copy SBA's hardcoded 0.95.** D24 — compute `1 − α_interval/2`.
6. **Do not permit a lag-0 feature.** D21 — `mlforecast` forbids it at construction and that is the right design.
7. **Do not transcribe `strucchange`'s coefficient tables.** §6.
8. **Do not ship the `partx` fixture.** §4.1 — GPL data.
9. **Do not let a degenerate input return a confident number.** §5.2.
10. **Do not assume the two "direct" reductions are the same.** `forecastML` anchors lags to the target and drops lags < h; `sktime` keeps a fixed origin-anchored window. They agree **only at `h = fh_max`** [V].
11. **Do not standardise, clamp, or silently drop rows inside a fitter.** Carried from Stage 11.

---

## 10. On finishing each item

1. Every figure in the sub-stage handoff appears in a test.
2. Testset and assertion counts recorded in `development-sequence.md`, with what was dual-verified and what was not.
3. Anything unverified written into the docstring the user reads, not only the handoff.
4. The defect list appended to, never rewritten.
5. A benchmark, run and recorded, for anything §7 claimed.

---

## Appendix: open items carried in from Stage 11

| Item | Status |
|---|---|
| **The Julia 1.12 + ForwardDiff 0.10 `iszero` hazard** — 1.12's generic matmul skips entries where `iszero` holds and ForwardDiff 0.10's `iszero` ignores partials, so a `Dual` with value 0 silently loses its derivative. Fixed at **one** site (`stationaryvar.jl:95`, a `1e-300` substitution); CI runs `version: '1'`; nothing tests the class | **Open, and the top-priority carry-in.** Stage 12 adds no new AD paths (OLS and dynamic programming are derivative-free), so this does not grow — but it does not shrink either |
| `development-sequence.md:1937` says 47 `@example` blocks; the diff of `127e04c` is 41 plus 2 manual pages | Open |
| `development-sequence.md:741` still opens Tier 2.4 with "Three of six items built" while line 880 says it is complete | Open |
| Julia is not installed in this sandbox | Every claim about the package is [S] or [U], never [V] |
