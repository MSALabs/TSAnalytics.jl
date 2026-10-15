# Stage 12.3 — Lag-feature regression ("the ML reduction")
## Rescoped: most of this already exists

**Prerequisite reading:** `stage-12-overall-handoff.md`, especially **§1.2**, which is the finding that reshapes this item.

**Difficulty: Low–Moderate for what is genuinely missing.** The item as written in the roadmap is largely already shipped under the names `arx`, `forecast(::ARXModel)`, `tscv`, `expanding_window_split`, `sliding_window_split`, `fourier_terms` and `accuracy`.

**The hardest part of this item is deciding what not to build.**

---

## 1. The finding, restated

**`arx` plus `forecast(::ARXModel)` already is the recursive ML reduction.** With ordinary least squares injected, four independent implementations across two languages agree on the coefficients to ~1e-15 and on the recursive forecasts to **2e-15** (charter §1.2, reproduced independently in this session).

The structural confirmation is in the source: `_arx_forecast_row`'s own docstring describes reading lag terms "from `yext` (the original series extended in-place with each step's own forecast so far)" [S] `src/forecast.jl` — the recursive strategy, written down.

### 1.1 What already exists, verified against the references rather than assumed

| Capability | Where | Evidence [V] |
|---|---|---|
| Lag design matrix, **including arbitrary lag subsets** | `arx` | `lags=[1,3]` versus `mlforecast` `lags=[1,3]`: coefficients **6.661e-16**, recursive forecasts **2.665e-15**. **`sktime` has no subset argument at all** |
| Trend and seasonal dummies | `arx(trend=, seasonal=, period=)` | versus `mlforecast` month dummies: same column space, fitted values **7.461e-14** |
| Contemporaneous (known-future) exog | `arx(exog=)` | versus `mlforecast`'s default: **1.110e-15, the identical model** |
| **The recursive strategy itself, with prediction intervals** | `forecast(::ARXModel)` | matches all three references to **2e-15**; and its analytic psi-weight standard errors are **strictly ahead of all three** |
| Rolling-origin CV with an **injected function** | `tscv(y, fit_forecast_fn; ...)` | already `forecastML`'s design |
| Expanding / sliding splitters | `expanding_window_split`, `sliding_window_split` | already verified to match `sktime` exactly |
| Accuracy metrics | `accuracy.jl` | `mae, rmse, mape, smape, mase` — missing only `mdape`, `rmsse` |
| Known-future seasonal features | `fourier_terms` | matches `forecast::fourier` column for column |
| Target transforms | `transforms.jl`, `differencing.jl` | ≈ `mlforecast`'s `Differences`, `LocalBoxCox` |
| Forecast combination | `combine_forecasts` | ≈ `forecastML::combine_forecasts` |

**Building a `ml_reduction` that re-derives the lag matrix would reproduce `arx` to 2e-15** — duplicate code, duplicate tests, and a second place for the design-matrix convention to drift.

---

## 2. What to build — four things, ranked by value per line

### 2.1 Inject the regressor — HIGH value, LOW cost, ~60–100 lines

`arx` is hard-wired to OLS. **This is the only part of the item that cannot be reached with what exists.**

The interface question was resolved by probing all three references with five candidate "models" [V]:

| reference | contract | bare callable accepted? |
|---|---|---|
| `mlforecast` | `fit`/`predict` **plus `get_params`**, because `fit_models` calls `sklearn.base.clone` | **no** |
| `sktime` `make_reduction` | identical — `clone` again | **no** |
| **`forecastML`** | **a function.** `model_function=f(data, ...) -> model` with the outcome in the first column, plus `prediction_function=f(model, data) -> 1-column data.frame`. **Rejects a fitted object** with a clear message | **yes** |

The Python references' real contract is **three methods, two of which exist only to support `clone`** — and `clone` exists only because they refit the same estimator spec across folds and horizons.

**Decision: take `(fit_fn, predict_fn)`, following `forecastML` and matching `tscv`'s existing injected-callable pattern.** Keep OLS as the default so every expectation stays reproducible and matchable to 1e-15. Anything read off the fitted model beyond `predict` (a `feature_importances_` equivalent) must be optional — OLS has none.

### 2.2 The `direct` strategy — HIGH value, LOW–MEDIUM cost, ~80–150 lines

Genuinely absent and genuinely different: **5.79e-02** from recursive on the AR(2) fixture, far outside any tolerance [V].

**But the two references disagree on the lag anchoring, and they agree only at `h = fh_max`:**

| | anchoring |
|---|---|
| `forecastML` | **target-anchored** — discards any lag < h (`lookback[lookback >= forecast_horizon]`) [S], so `horizons=1:6, lookback=1:6` gives h=1 seven columns and h=6 only two |
| `sktime` | **origin-anchored** — a fixed `window_length` of lags for every h |

They are the same reduction reparameterised (`forecastML`'s `lookback_control = {h, h+1}` ≡ `sktime`'s `window_length=2`), and the measured agreement runs from 2.68e-02 at h=1 down to **2.22e-16 at h=6** — exact only where the usable sample is pinned at both ends. [V]

**Decision: origin-anchored (sktime's).** The feature set stays constant across horizons, which makes coefficients comparable and the code simpler. Expose target-anchoring through a per-horizon lag argument if anyone asks.

**And decide the sample-truncation rule deliberately.** `sktime`'s own two APIs default it oppositely — `make_reduction(strategy="direct")` has `windows_identical=True` (n=113) while `DirectReductionForecaster` has `False` (n=118), giving different h=1 coefficients `1.005639629996196` versus `1.0103484992289582` for the same nominal request [V]. The `False` variant's h=1 fit is exactly the plain AR(2) recursive fit.

### 2.3 Lagged (past-only) exogenous regressors — HIGH value, LOW cost, ~30–50 lines

`arx` takes exog **contemporaneously only** [S] `src/arx.jl:183`. `sktime` supports *only* the lagged form; `forecastML` defaults to it. Caller-side lagging is the workaround, but it silently shortens the sample, which is exactly where off-by-ones live.

**And the three references make three incompatible default choices** (charter §5.10), with three silent failures at forecast time [V]:

| # | reference | failure |
|---|---|---|
| D22 | `sktime` `dirrec` | **silently discards X entirely** — with-X and without-X differ by 0.000e+00; the source says `# todo: logic for X below is broken` |
| D23 | `sktime` recursive/direct | **silently extrapolates missing future X** via `np.pad(..., "edge")`, max diff 0.9851 against correct |
| — | `forecastML` | `predict_future` returning 3 of 6 rows is **accepted**, giving `NA` for the rest, no warning |

**Our behaviour: require future values for any known-future regressor and throw naming the column and the shortfall.** `mlforecast` gets this right (a loud `ValueError`) and is the model to follow.

### 2.4 Let `forecast` accept future exog — HIGH value, LOW cost

Today `_arx_forecast_row` **throws** on any `x*` column, and the docstring calls it "a real, deliberate gap, not an oversight" [S]. The consequence is concrete: **`arx` + `fourier_terms` cannot forecast at all**, even though Fourier terms are perfectly known in the future and `fourier_terms(period, K; h=...)` already generates them.

All three references supply future regressors. **This is a small change that unlocks an existing pairing**, and it is arguably the single highest value-per-line item in Stage 12.

---

## 3. What to build only if there is appetite

### 3.1 Rolling and expanding lag features — MEDIUM value, MEDIUM cost

`mlforecast`'s largest feature-side contribution: ~20 classes, **1885 lines** [S]. `arx` has no lag-transform concept.

**Recommend a small closed set** — `RollingMean/Std/Min/Max`, `ExpandingMean`, perhaps `SeasonalRollingMean` — **every one anchored at a positive lag, with lag 0 rejected at construction.**

That rejection is not a detail. It is precisely how `mlforecast` makes leakage structurally impossible (`ValueError: lags must be positive integers`) and precisely where `sktime` fails: `WindowSummarizer(lag_feature={'mean':[[0,1]]})` produces a feature that **is literally `y_t`** — max |feature − y| = **0.000e+00**, in-sample R² = **1.0000000000** — accepted with no warning, and the leak propagates into a fitted forecaster (charter D21, re-verified here).

**Copy `mlforecast`'s design decision verbatim on this one point.**

Under recursion, rolling features are **recomputed from predictions, not frozen** — both Python references do this, and `sktime`'s docstring says so explicitly [S]. Match them, and document it, because a rolling mean at step 4 is then the mean of three predicted values.

### 3.2 Blocked CV — MEDIUM value, LOW cost, ~40 lines

`forecastML::create_windows`: hold out a contiguous block and train on **everything outside it, both sides**. Verified: 117 usable rows, 20-row block, train n = **97 = 117 − 20** for every window [V]. Cites Bergmeir–Hyndman–Koo (2018).

It is the right harness for a *direct* reduction and the only CV shape `tscv` cannot express. Add `mdape` and `rmsse` to `accuracy.jl` while there.

### 3.3 A `refit` policy for `tscv` — LOW–MEDIUM, trivial

`mlforecast`'s `refit=True/False/int`. Measured effect on the AR(2) fixture: `refit=False` differs from `refit=True` by **0.05061**, `refit=2` by **0.02185** [V]. Cheap, and a real lever once an expensive injected model is possible.

---

## 4. What NOT to build, with the measurements that settle it

### 4.1 DirRec and multi-output — provably identical to Direct under OLS

Measured, then re-proved from scratch via OLS orthogonality [V]:

| identity | max difference, h=1…6 |
|---|---|
| DirRec ≡ Direct (fixture A) | **4.441e-16** |
| DirRec ≡ Direct (fixture B) | 2.931e-14 |
| multi-output ≡ Direct | **2.220e-16** |

DirRec genuinely expands its feature count (`n_features_in_` = `[2,3,4,5,6,7]`), so this is not a no-op — it is orthogonality. With a random forest the identity **breaks** (|dirrec − direct| = 0.309531).

**With OLS as the injected default these are untestable as distinct strategies, and they diverge only for a regressor the no-dependency rule means this package will never ship. Note the identity in a docstring instead of implementing it.**

### 4.2 A new `ml_reduction` entry point

§1. Extend `arx` instead.

### 4.3 Conformal prediction intervals

The existing analytic psi-weight standard errors already beat all three references, none of which offers intervals from a reduction: `sktime` raises `NotImplementedError`, `mlforecast` has conformal-only and **silently omits the interval columns** if `prediction_intervals=` was not passed at fit time, `forecastML` has none. [V]

### 4.4 Heavy charts

`forecastML` is **41% ggplot2** (2163 lines across 8 methods); `mlforecast` ships **zero** plotting [V]. At most reuse `RecipesBase` for the two things both `forecastML` and `sktime` bother to plot: a lag/feature map and a CV-fold layout. **Feature-importance plotting is pointless** — no reference has it, and OLS has nothing to plot.

---

## 5. Fixtures

Three, saved as CSV at `%.17g`, every design rank-verified.

| id | definition | shape | md5 |
|---|---|---|---|
| A | AR(2), `y = 1.0 + 0.6y₁ − 0.3y₂ + e` | 120×3 | `c5ea75ea042344fa77950d8a7ddc8803` |
| B | period-12 seasonal + trend + AR(1) | 144×4 | `a9ce6dff087bd1decb3656b271f87377` |
| C | AR(1) + `x_known` + `x_past`, **last 6 rows carry x with y = NA** | 138×5 | `1082373dd67e14c39b2bc813555df0f0` |

### 5.1 Expectations [V]

**Fixture A, `lags=[1,2]`, `trend=:c`, design (118, 3), rank 3:**

| | value |
|---|---|
| const | **1.010348499228958** |
| y.L1 | **0.6026972101854623** |
| y.L2 | **−0.35837911855381488** |
| sigma2 (n-denominator) | 0.22541381721778347 |
| recursive h=1…6 | 1.2806577561017716, 1.2243195830708189, 1.2892814985203096, 1.3486242885217865, 1.3611090285019425, 1.3473663296846201 |

Reproduced independently in this session to 15 digits.

**Fixture C, `arx(y, 1, exog=[x_known x_past])` versus `mlforecast`'s default** [V]:

| | `arx`-style | `mlforecast` |
|---|---|---|
| const | 0.001863153934060024 | 0.0018631539340603354 |
| y.L1 | 0.38423957639220263 | 0.38423957639220185 |
| x_known | 2.0268793035159356 | 2.0268793035159347 |
| x_past | 1.5035550438207037 | 1.5035550438207026 |

Max difference **1.110e-15 — the identical model.** True DGP: 0.4 / 2.0 / 1.5.

**Direct strategy, fixture A, the anchoring comparison** — agreement only at h = 6 [V]:

| h | `forecastML` `lc={h,h+1}` | `sktime` `windows_identical=True` | diff |
|---|---|---|---|
| 1 | 1.28467221075666660 | 1.25786654152838939 | 2.68e-02 |
| 3 | 1.31496094770183225 | 1.30701018705539918 | 7.95e-03 |
| **6** | **1.38399214655808045** | **1.38399214655808067** | **2.22e-16** |

---

## 6. Verification plan

### 6.1 Tier 1

With OLS injected the whole pipeline is deterministic and matchable to ~1e-15 across four implementations in two languages.

| test | target | tolerance |
|---|---|---|
| recursive with OLS injected ≡ `arx` + `forecast(::ARXModel)` | §5.1 | **1e-14** |
| recursive against `mlforecast`, `sktime`, `forecastML` | §5.1 | **2e-15** |
| arbitrary lag subsets | `lags=[1,3]` | 1e-15 |
| known-future exog | fixture C, §5.1 | 1e-15 |
| direct at `h = fh_max` | 1.38399214655808045 | **2e-16** |
| direct below `h = fh_max` | our stated anchoring only | — |
| DirRec ≡ Direct, multi-output ≡ Direct | §4.1 | asserted as identities, **not implemented** |

### 6.2 The leakage probe — parameterised by strategy

Build the design, perturb `y[t₀]` by +1000, rebuild, and check whether any feature value *in row t₀* moves.

**And parameterise it by strategy.** A uniform "target == `y_t`" assumption produces a **false positive** on multi-output reductions, where row t legitimately has `y_lag_1 = y[t]` and targets `y[t+1…t+6]` — a survey hit this and corrected it rather than filing a bug [V]. The correct multi-output probe perturbs `y[t+1]`, the first target.

Assert: every shipped lag transform is clean; **lag 0 is rejected at construction**; and a `sktime`-style lag-0 feature would have been caught (reproduce the R² = 1.0 case as a negative test against our own validation).

### 6.3 Rank-check every fixture design

A deterministic sinusoid satisfies `x_t = 2cos(ω)x_{t−1} − x_{t−2}`, so any number of its lags spans only two dimensions — measured rank **2/6** [V]. R's `lm` silently returns `NA` coefficients where `arx`'s rank check throws. **Assert `rank(X) == size(X,2)` for every fixture design in the suite**, and keep `arx`'s throwing behaviour — it is better than both Python references (charter D33: `mlforecast`'s dummy encoding gives rank 14 of 15 with condition 1.686e+17 and sklearn returns a silent minimum-norm solution).

### 6.4 Testset inventory

| # | Testset | Assertions (min) |
|---|---|---|
| 1 | injected OLS ≡ `arx`, fixture A | 12 |
| 2 | injected OLS ≡ `arx`, lag subsets and seasonal | 14 |
| 3 | against `mlforecast`/`sktime`/`forecastML`, recursive | 16 |
| 4 | a non-OLS injected function works (a hand-written ridge) | 10 |
| 5 | the injected contract: `fit_fn`/`predict_fn` shapes, and refusals | 14 |
| 6 | direct strategy, h = fh_max exact | 10 |
| 7 | direct strategy, the anchoring is ours and documented | 10 |
| 8 | direct ≠ recursive by 5.79e-02 | 6 |
| 9 | DirRec ≡ Direct, multi-output ≡ Direct (identities, not features) | 8 |
| 10 | past-only exog, and the sample-length bookkeeping | 14 |
| 11 | known-future exog required at forecast; throws naming the column | 12 |
| 12 | **`arx` + `fourier_terms` now forecasts** | 12 |
| 13 | leakage probe, recursive and direct | 14 |
| 14 | leakage probe, multi-output (the corrected form) | 8 |
| 15 | lag 0 rejected at construction | 8 |
| 16 | rolling features clean under recursion (recomputed, not frozen) | 12 |
| 17 | rank check on all three fixtures | 10 |
| 18 | blocked CV: train n = 97 = 117 − 20 | 10 |
| 19 | `mdape`, `rmsse` | 10 |
| 20 | `refit` policy, measured effect | 8 |
| 21 | performance: allocations, `@code_warntype` | 8 |

**21 testsets, ≥ 246 assertions.**

---

## 7. Budget

| Piece | Julia |
|---|---|
| inject the regressor into `arx` | 60–100 |
| direct strategy | 80–150 |
| past-only exog | 30–50 |
| future exog in `forecast` | 40–60 |
| rolling/expanding lag features (optional) | 150–220 |
| blocked CV + two metrics | 60–80 |
| `refit` policy | 20–30 |
| **total, core four** | **210–360** |
| **total with the optional two** | **440–690** |

Compare with the references' 9578 (`mlforecast`), 5219 (`forecastML`) and 3331+992 (`sktime`) lines. **The ratio is the finding**: most of their bulk is either framework ceremony, plotting, or a lag-matrix builder we already have.

---

## 8. Build order

1. **Inject the regressor**, tested to reproduce `arx` exactly with OLS. Nothing else until this passes.
2. **Future exog in `forecast`** — small, unlocks `fourier_terms`.
3. **Past-only exog.**
4. **The direct strategy**, with the anchoring and truncation rules pinned.
5. Blocked CV and the two metrics.
6. Rolling features, if wanted, with lag 0 rejected from the first commit.
7. `refit`.

---

## 9. What not to do

1. **Do not build a new `ml_reduction` that re-derives the lag matrix.** §1.
2. **Do not build DirRec or multi-output.** §4.1 — provably identical to Direct under OLS.
3. **Do not permit a lag-0 feature.** §3.1, D21.
4. **Do not accept a fitted estimator object.** §2.1 — take functions.
5. **Do not silently extrapolate missing future exog.** D23.
6. **Do not copy `mlforecast`'s dummy encoding.** D33 — `arx`'s rank check is the better behaviour.
7. **Do not assume the two "direct" reductions agree.** §2.2 — they agree only at `h = fh_max`.
8. **Do not use a deterministic series as a fixture.** §6.3.
9. **Do not apply a uniform leakage probe across strategies.** §6.2.
10. **Do not build conformal intervals.** §4.3 — the existing psi-weight SEs are ahead of all three references.
