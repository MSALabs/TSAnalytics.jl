# Stage 12.1 — Intermittent-demand forecasting
## Croston, SBA, TSB, ADIDA, IMAPA, and the ADI/CV² classifier

**Prerequisite reading:** `stage-12-overall-handoff.md`. §5.1–5.3 and defects D24–D26 and D32 are this item's traps.

**Difficulty: Easy.** The forecasting is a thin layer over simple exponential smoothing, which the package already owns. The genuine work is the degenerate-input policy (§4) and the absent prediction intervals (§5) — neither of which is forecasting code.

**Scope widened from the roadmap.** It lists "Croston's method / SBA / TSB". `forecast` has **only** classical Croston [V]; `statsforecast` has six models. Building five plus the classifier costs little more than building three.

---

## 1. What this builds

A series that is mostly zeros with sporadic non-zero demands. Croston's insight is to smooth the **non-zero demands** and the **inter-arrival intervals** separately and divide.

### Exported surface

| Name | Kind | Notes |
|---|---|---|
| `croston` | function | classical; `method=:classic\|:optimized\|:sba` |
| `tsb` | function | Teunter–Syntetos–Babai — a different state, so a separate entry point |
| `adida`, `imapa` | functions | temporal aggregation |
| `IntermittentModel` | type | `<: TimeSeriesModel`, `<: UnivariateModel` |
| `demand_intervals` | function | → `DemandIntervals`, the decomposition and plot payload |
| `adi_cv2` | function | → `ADICV2Result`, the classification |
| `cfe`, `spis` | functions | Cumulative Forecast Error and scaled Periods In Stock — added to `src/accuracy.jl` |
| `forecast` / `predict` | methods | exact aliases |

---

## 2. The algorithms, read from source

All line counts and formulas [S] `statsforecast` v2.1.1 `models.py` and `src/ses.cpp`, with behaviour [V].

**The SES kernel** (`ses.cpp:28-50`), which is Croston's engine:

```
fitted[0] = x[0]
fitted[i] = α·x[i−1] + (1−α)·fitted[i−1]
forecast   = α·x[n−1] + (1−α)·fitted[n−1]
then fitted[0] := NaN
```

`initial="simple"` — the level starts at the first element. This is why the two references agree to 17 digits (charter §5.1), and it is **hardcoded in R's `croston2`, not user-settable**.

**The decomposition** (3 code lines total):

```
demand(x)    = x[x > 0]
intervals(x) = diff(indices_of_nonzero + 1, prepend = 0)     # first interval = 1-based index of the first non-zero
```

R's `diff(c(0, (1:length(x))[x>0]))` is the same expression.

| Method | Formula | Core size |
|---|---|---|
| **Croston classic** | `ses(demand, α) / ses(intervals, α)`, with a guard returning `ses(demand)` when the interval forecast is 0 | **21 code lines** |
| **Croston optimized** | two **independent** α, by golden-section search minimising one-step SSE | **35 code lines** + 26 lines of C++ |
| **SBA** | Croston classic **× (1 − α_interval/2)** | **5 code lines** |
| **TSB** | `ses(probability, α_p) × ses(demand, α_d)`, where `probability = (y ≠ 0)` over **all** t. **No interval state.** | **16 code lines** |
| **ADIDA** | aggregate into non-overlapping buckets of width `round(mean(intervals))`, SSE-optimised SES on the bucket sums, divide by the width | **25 code lines** |
| **IMAPA** | ADIDA at every level 1…`round(mean(intervals))`, arithmetic mean of the per-level forecasts | **22 code lines** |

**Total genuinely new algorithmic surface: ~110 lines of Python glue over ~130 lines of C++ kernel**, of which the SES kernel (~35 lines) we already have in `src/holtwinters.jl`.

### 2.1 Three source-level traps

1. **SBA's factor is a hardcoded `0.95`** [S] `models.by:5401` — `out["mean"] *= 0.95`. It equals `1 − α/2` only because classic hardcodes α = 0.1 (charter D24). **Compute `1 − α_interval/2`.** The literature form is confirmed independently from `tsintermittent/R/crost.R:152`, which also carries an **SBJ** variant (`1 − α/(2−α)`) that neither target reference implements. Worth a docstring mention, not an implementation.
2. **`CrostonOptimized`'s search is bounded to [0.1, 0.3]**, hardcoded, tolerance 1e-12 [S] `ses.cpp:52-77`. **The bound is genuinely binding** — on fixture F1 the unconstrained SSE-optimal interval α is **0.317** and golden section returns **0.29999999999967963**; on F3 the demand α clamps at the lower bound, **0.10000000000032039**. Two of four fixtures hit a bound. [V] "Optimized" is a *restricted* search; say so.
3. **TSB's docstring claims it optimises α "like in the optimized Croston's method". The code does not.** [V] Both α are user-supplied with no defaults. Do not inherit the claim.

### 2.2 A degenerate case worth knowing

For a constant series every α gives equal SSE, the golden-section `else break` fires, and the function returns the **midpoint 0.2** — verified `golden_section_ses(ones(10), 0.1, 0.3) = 0.20000000000000001` [V]. Fixture F2's demand series is all 1s, so its optimised demand α is exactly 0.2. That is a good pinned test of the tie-break.

---

## 3. Fixtures and full-precision expectations

**The CSV is the fixture; the seed is provenance only.** Note F1's provenance is **R**, not numpy: `set.seed(7); rpois(60, 0.4)` reproduces it exactly, while neither numpy generator does [V].

| id | file | definition | n | nonzero | sum | ADI | CV² |
|---|---|---|---|---|---|---|---|
| F1 | `intermittent.csv` | R `set.seed(7); rpois(60, 0.4)` | 60 | 22 | 30 | 2.5 | 0.18158730158730163 |
| F2 | `fx_sparse_p010_n120.csv` | R `set.seed(101); rpois(120, 0.1)` | 120 | 15 | 15 | 7.7333333333333334 | 0 |
| F3 | `fx_dense_p150_n120.csv` | R `set.seed(202); rpois(120, 1.5)` | 120 | 91 | 187 | 1.3186813186813187 | 0.25450541908547586 |
| ~~F4~~ | ~~`fx_partx_n51.csv`~~ | **GPL data — do not ship** (charter §4.1) | 51 | 17 | 32 | 2.8235294117647061 | 0.41918945312500006 |

**F3 is deliberately a boundary case**: ADI 1.3187 sits just below the 1.32 cutoff and CV² 0.2545 just below 0.49, so it classifies as Croston rather than SBA — the only fixture that tests the classifier's boundary.

### 3.1 Point forecasts — **flat in h**, verified bit-identical for h = 1…6 on all 24 (fixture × model) cells [V]

| | F1 | F2 | F3 |
|---|---|---|---|
| **Croston classic** | 0.65909246562518586 | 0.11168100226728453 | 1.7150503549950999 |
| **Croston optimized** | 0.67243689556737085 | 0.12378068088891507 | 1.7852955852683219 |
| **SBA** | 0.62613784234392655 | 0.1060969521539203 | 1.6292978372453448 |
| **TSB** (α_d = α_p = 0.2) | 0.2447982165028067 | 0.10398268223642748 | 1.7508977992762016 |
| **ADIDA** | 0.42114174223778827 | 0.11809029046769849 | 1.7486544426757562 |
| **IMAPA** | 0.29789731076415754 | 0.12038743043325673 | 1.7486544426757562 |

**On F3, ADIDA and IMAPA return the identical value** — not a coincidence: the mean interval 1.3187 rounds to 1, so IMAPA averages over the single level agg = 1, which *is* ADIDA at level 1. Assert it as a structural identity.

### 3.2 Component levels and optimised alphas (α = 0.1 for the levels) [V]

| | F1 | F2 | F3 |
|---|---|---|---|
| demand level | 1.2978605425231977 | 1 | 1.9398661738758045 |
| interval level | 1.9691630692396169 | 8.9540743698441609 | 1.1310840922110108 |
| α_demand (optimised) | 0.2406879687366284 | 0.20000000000000001 | **0.10000000000032039** ← lower bound |
| α_interval (optimised) | **0.29999999999967963** ← upper bound | 0.2623181302768971 | 0.13774151423786604 |

**R reports the same demand and interval levels to all 17 digits** (`f$model$demand$mean[1]`, `f$model$period$mean[1]`), and `demand / interval` reproduces the point forecast exactly. [V]

### 3.3 In-sample MASE, naive-1 denominator [V]

| | F1 | F2 | F3 |
|---|---|---|---|
| Croston classic | 1.0885456588355449 | 0.75327412615990097 | 0.76547402721288704 |
| Croston optimized | 1.0003334636707115 | 0.84657935076697566 | 0.76526682256932366 |
| SBA | 1.060205332415507 | 0.74061041985190545 | 0.76995408408447064 |
| TSB | 0.93462033008444145 | 0.89314620347791829 | 0.78723985113066774 |
| ADIDA | 0.86465563418349412 | 0.81819563743934332 | 0.77289356516392216 |
| IMAPA | 0.88322692682056458 | 0.83499151570749741 | 0.78032048148946886 |

Denominators: F1 **0.77966101694915257**, F3 **1.3277310924369747**.

**Do not transcribe `forecast::accuracy()`'s MASE** (charter D32). On a bare numeric vector it silently uses mean absolute deviation about the mean: **1.3400472874157467** instead of **1.0885456588355449**. Our `mase` at `src/accuracy.jl:124` computes the standard denominator and reproduces the correct column [V].

### 3.4 Fitted values — `rtol=1e-12`, not exact

| fixture | n compared | bit-exact | max abs diff | max rel diff |
|---|---|---|---|---|
| F1 | 59 | 10 | 3.55e-15 | 3.54e-15 |
| F2 | 119 | 19 | 4.72e-16 | 4.22e-15 |
| F3 | 119 | 10 | 5.11e-15 | 4.66e-15 |

**Diagnosed, not merely reported:** R recomputes the whole Croston from scratch on every prefix — `for (i in 1:(n-1)) fits[i+1] <- croston2(x[1:i], ...)`, i.e. **O(n²)** — while statsforecast does one SES pass and carries values forward through zero runs. Mathematically identical (during a run of zeros neither prefix changes), so the ~20-ulp residual is accumulated summation-order error. [V]

**Use the single-pass O(n) form** and test fitted values with `≈` at `rtol=1e-12`. Point forecasts test exactly.

---

## 4. The degenerate-input policy — the real design work

**Neither reference ever refuses on non-negative input.** Both always return a number, and they agree on every one of thirteen edge cases. [V]

| case | series | both references |
|---|---|---|
| all zeros, n=10 | `0×10` | **0** |
| one non-zero, middle | `0,0,0,5,0×6` | 1.25 |
| one non-zero, **last** | `0×9,5` | 0.5 |
| **one non-zero, first** | `5,0×9` | **5** ← a 10× error |
| n=1, non-zero | `3` | 3 |
| n=2 | `0,3` / `3,0` / `3,4` | 1.5 / 3 / 3.1 |
| no zeros, n=6 | `1,2,3,2,1,2` | 1.39241 |
| negative | `1,-1,0,2` | R **errors**; statsforecast returns **1** |

### Decisions

1. **Refuse a single arrival at t = 1.** `[5,0,…,0]` → both references return 5 where the empirical rate is 0.5. Croston cannot estimate an inter-arrival rate from one arrival at the first observation. Throw `ArgumentError` naming the fact. **This is the clearest honest-refusal case in four stages**, and the test asserting it should name both references' agreed answer in a comment so the deviation is deliberate and visible.
2. **Reject negatives**, as R does (charter D25 — statsforecast is internally inconsistent, counting a negative as an arrival that contributes no demand).
3. **All-zero returns zero**, matching both — but **by an explicit branch**, not by falling through to a naive forecast as `_croston_*` does (it returns `y[-1]`, which equals 0 only because the last element happens to be 0).
4. **SBA must not debias a single observation.** `[3] → 2.85` in statsforecast is debiasing a quantity with no bias to correct. Refuse or pass through unmodified; decide and document.
5. **Do not port `croston2`'s NA branch** — `else if (length(y) <= 1 || length(tt) <= 1)` is **unreachable dead code** (`tt` always has `length(y)` elements, so the preceding branch catches it). Confirmed empirically across all thirteen cases: no input produced NA. [V]

---

## 5. Prediction intervals — there is no reference, and `Forecast` requires them

**Neither reference offers analytic or simulation intervals.** [V]

- `forecast::croston` calls `ses(..., PI=FALSE)` explicitly; `$lower`/`$upper`/`$level` are NULL on all fixtures.
- `statsforecast` **raises** on `level=` without `prediction_intervals=`: *"You have to instantiate the class with prediction_intervals to calculate them"*.
- The only route is `ConformalIntervals`, a **model-agnostic empirical wrapper** over cross-validated residual quantiles — and it behaves badly here. Measured on F1 with `n_windows=4, h=6`, the 80% lower bounds are `0.17234347515770043, −0.23741674806336879, 0.041930759852654917, −0.012534817798934506, …` — **negative lower bounds on a non-negative count series**, non-monotone in h. [V]

**This collides with the package's own types.** `Forecast` (`src/forecast.jl:15`) *requires* `se`, `lower`, `upper` [S].

### Decision

**Return a dedicated `IntermittentForecast` rather than a `Forecast` with empty interval matrices**, and say in the docstring that no reference implementation of intervals exists for this model family. If intervals are wanted later, the defensible routes are a parametric Poisson or negative-binomial model on lead-time aggregated demand, or a bootstrap over the demand/interval pair — **both original work with no oracle**, and both out of scope here.

Returning a `Forecast` with zero-filled `se` would be the silent-approximation failure the charter's §2.3 rules out.

---

## 6. The ADI/CV² classifier

**Neither target reference implements it** — exhaustive grep for `1.32`, `0.49`, `lumpy`, `erratic`, `ADI`, `cv2` across all of `statsforecast`'s `.py`/`.cpp` and all 360 `forecast` namespace objects returns **zero hits** [V].

The cutoffs **are** confirmed at source, in GPL `tsintermittent/R/idclass.R:113` — `use.sba <- (p > 1.32 | v > 0.49)` — reinforced by its plot polygon at lines 188/229. So the familiar 1.32 / 0.49 are real, and now verified rather than assumed. [V]

Definitions, reproduced independently [V]:

- `p` = mean of `c(nzd[1], diff(nzd))` — the average inter-demand interval, **the same interval construction as both references**
- `v` = `(sd(z)/mean(z))²` over **non-zero demands only**, with the **sample** standard deviation (n−1 denominator)

Three further rules in the same file: `KHa` (Kostenko–Hyndman approximate) `v > 2 − 1.5p`; `KH` exact; `PK`/`PKa` adding `use.ses <- p ≤ 1`.

**One honest caveat to carry into the docs.** `idclass.R` implements a **binary SBA-versus-Croston** decision. The four-quadrant labels *smooth / intermittent / erratic / lumpy* appear nowhere in it — they come from the Syntetos–Boylan–Croston paper the cutoffs are drawn from. **If we ship the quadrant chart with those labels, that naming is ours**, and the docstring should say so.

Measured classification [V]: F1 → SBA (p = 2.5), F2 → SBA (p = 7.733), **F3 → Croston** (p = 1.3187 < 1.32, v = 0.2545 < 0.49), partx → SBA (p = 2.8235).

---

## 7. Metrics — two cheap additions that make SBA demonstrable

`src/accuracy.jl`'s `mase` is already the standard metric with the right zero-guard (`max(mae_naive, eps)`), and it reproduces the verified column in §3.3 [V/S]. **It suffices for point accuracy.**

But **MASE cannot see bias**, which is the entire thing SBA exists to correct. `statsforecast`'s companion `utilsforecast.losses` ships intermittent-specific metrics we lack [S]:

| metric | what it is | size |
|---|---|---|
| **`cfe`** | Cumulative Forecast Error — signed total; positive means under-forecast | 21 lines |
| `pis` | absolute Periods In Stock, unscaled | 22 lines |
| **`spis`** | scaled PIS — sum of absolute errors ÷ mean in-sample demand; scale-independent, so it aggregates across series | 55 lines |
| `tweedie_deviance` | compound Poisson–Gamma, the standard likelihood for intermittent counts | 79 lines |

**Add `cfe` and `spis`** — roughly five lines each in Julia. Without them we would ship a bias correction with no metric that can observe it, and the book chapter would have nothing to show. `tweedie_deviance` is a larger decision; note it and move on.

---

## 8. Verification plan

### 8.1 Tier 1 — exact

| test | target | tolerance |
|---|---|---|
| Croston classic, F1/F2/F3 | §3.1 row 1 | **exact** |
| all six methods × three fixtures | §3.1 | **exact** on point forecasts |
| flatness in h | h=1…6 bit-identical, all 18 cells | **exact** |
| demand and interval levels | §3.2 | 1e-15 |
| optimised α, including both bound hits | §3.2 | 1e-12 |
| ADIDA ≡ IMAPA on F3 | structural identity | **exact** |
| fitted values | R's vectors | **`rtol=1e-12`** (§3.4) |
| MASE, all six × three | §3.3 | 1e-12 |
| ADI/CV² | §6 | 1e-14 |

### 8.2 Reductions

1. **Croston with no zeros reduces to SES on the raw series.** Intervals are all 1, so the interval level is 1 and the forecast is `ses(y, α)`. Check against `holt_winters`' simple path — an existing, verified component. On `1,2,3,2,1,2` the reference gives 1.39241.
2. **SBA with α = 0 reduces to classic.** The factor becomes 1.
3. **TSB with a never-zero series reduces to SES**: the probability level goes to 1.
4. **ADIDA at aggregation level 1 is Croston-free SES on the raw series**, and equals IMAPA when the mean interval rounds to 1 (F3).

### 8.3 Edge cases

The full thirteen-row table of §4, each asserting **our** behaviour, with the references' agreed answer in a comment. Four of the thirteen are refusals for us.

### 8.4 The tie-break

F1's mean interval is **exactly 2.5**. statsforecast is self-inconsistent here (charter D26: banker's rounding in the mean path, half-away-from-zero in the fitted path, a 20% difference — ADIDA 0.42114174223778827 at agg=2 against 0.50681015732311729 at agg=3). **Pick a rule, document it, and test both values** so the choice is visible.

### 8.5 Testset inventory

| # | Testset | Assertions (min) |
|---|---|---|
| 1 | Croston classic, three fixtures, exact | 10 |
| 2 | Croston optimized, including both α bound hits | 14 |
| 3 | SBA, and the `1 − α/2` factor against the hardcoded 0.95 | 12 |
| 4 | TSB, three fixtures | 10 |
| 5 | ADIDA, three fixtures | 10 |
| 6 | IMAPA, and the ADIDA identity on F3 | 10 |
| 7 | flatness in h, all 18 cells | 18 |
| 8 | demand and interval levels | 12 |
| 9 | fitted values, `rtol=1e-12` | 10 |
| 10 | MASE, all six methods | 18 |
| 11 | `cfe` and `spis` | 12 |
| 12 | ADI/CV² and the classification, including F3's boundary | 14 |
| 13 | reduction: no zeros = SES | 8 |
| 14 | reductions: α=0, never-zero, agg=1 | 12 |
| 15 | edge cases: the nine we answer | 20 |
| 16 | refusals: single arrival at t=1, negatives, SBA on n=1 | 14 |
| 17 | all-zero via an explicit branch | 8 |
| 18 | the 2.5 tie-break, both values pinned | 8 |
| 19 | `IntermittentForecast` carries no fabricated intervals | 8 |
| 20 | `demand_intervals` decomposition sums correctly | 10 |
| 21 | `show` reports the method and the classification | 6 |
| 22 | performance: allocations, no recursion | 8 |

**22 testsets, ≥ 252 assertions.**

---

## 9. Budget

| Piece | Reference | Julia |
|---|---|---|
| demand/interval decomposition | 3 code lines | 20–30 |
| Croston classic + SBA | 21 + 5 | 60–80 |
| bounded golden-section α search | 26 lines C++ | 50–70 (or reuse `src/optim.jl`) |
| TSB | 16 | 40–60 |
| ADIDA + IMAPA | 25 + 22 | 80–110 |
| ADI/CV² classifier | — (neither reference) | 50–70 |
| `cfe`, `spis` | 21 + 55 | 20–30 |
| refusal policy and guards | — | 60–90 |
| `IntermittentForecast`, `show` | — | 80–110 |
| **total** | ~240 | **460–650** |

Note the ratio: the refusal policy and the result type together cost more than every forecasting method combined. That is the correct allocation for this item.

---

## 10. What not to do

1. **Do not use the hardcoded 0.95.** §2.1, D24.
2. **Do not inherit TSB's docstring claim** that it optimises α. §2.1.
3. **Do not describe `CrostonOptimized` as unrestricted.** Its [0.1, 0.3] bound binds on two of four fixtures.
4. **Do not return a `Forecast` with zero-filled intervals.** §5.
5. **Do not transcribe `forecast::accuracy()`'s MASE.** D32.
6. **Do not port the unreachable NA branch.** §4.5.
7. **Do not ship the `partx` fixture.** Charter §4.1 — GPL data.
8. **Do not claim the four-quadrant labels come from a reference.** §6.
9. **Do not recompute fitted values per-prefix.** §3.4 — R's O(n²) form is the source of the only disagreement.
