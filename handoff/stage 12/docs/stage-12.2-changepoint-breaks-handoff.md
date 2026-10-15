# Stage 12.2 — Change-point detection and structural-break testing

**Prerequisite reading:** `stage-12-overall-handoff.md`. §1.1 explains the merge, §6 the one hard island, and defects D19, D20 and D27–D29 are this item's traps.

**Difficulty: Moderate, with one hard island.** The search algorithms are small and the costs are closed forms. The time goes on penalty conventions (§3) and on inference, where the supF null distribution is blocked by licensing rather than by mathematics (§7).

**This item merges two roadmap entries.** `development-sequence.md` lists change-point detection; the book's Frontier chapter separately lists structural-break and CUSUM tests. `strucchange::breakpoints` is Bai–Perron dynamic programming, which is `changepoint`'s SegNeigh with a regression cost — the same dynamic program.

---

## 1. What this builds

| Name | Kind | Covers |
|---|---|---|
| `changepoints` | function | PELT, binary segmentation, SegNeigh/exact DP |
| `ChangePointResult` | type | breakpoints, segment parameters, the criterion path |
| `breakpoints` | function | Bai–Perron: the whole path, with selection afterwards |
| `BreakpointsResult` | type | the RSS/BIC table, segment coefficients |
| `break_confint` | function | Bai (1997) break-date confidence intervals |
| `fstats` | function | the F-statistic path over candidate break dates |
| `FStatsResult` | type | `<: HypothesisTest`, critical-values-only shape |
| `efp` | function | empirical fluctuation processes — the CUSUM family |
| `EFPResult` | type | the process path and its boundary |

---

## 2. Architecture — follow `ruptures`, which is both the cleanest and the only permissive one

Of the three references, only `ruptures` factors the problem properly, and it happens to be **BSD-2** so its structure can be followed directly (charter §4).

| reference | factorisation |
|---|---|
| **`ruptures`** | **clean three-way.** Detector takes `model=` (resolved by a cost factory) or `custom_cost=` (anything with `.fit(signal)`, `.error(start,end)`, `.min_size`); the penalty is an argument to `predict()`, not an object [S] |
| `changepoint` | **2.5-way.** Search and cost separated by a C function pointer, but **MBIC is implemented as six extra cost functions, not a penalty** — choosing `penalty="MBIC"` silently swaps the cost (the cause of D19) [S] |
| `strucchange` | **none.** `breakpoints.formula` has **no penalty argument at all** — a pure constrained DP returning the optimal segmentation for *every* m, with selection afterwards by `which.min(BIC)` [S] |

### Decisions

**Three-way separation: search × cost × penalty**, as `ruptures` does, with the cost as an injectable object carrying its own minimum segment length (`ruptures` floors the user's `min_size` up by `cost.min_size` — 5 for `ar`, 3 for `clinear` [S]).

**And adopt `strucchange`'s "compute the whole path, select later" shape for the Bai–Perron side.** It is architecturally different from the other two and it is the better shape for inference: the RSS/BIC table over m is itself a deliverable, and it is what the selection plot draws.

**MBIC is a penalty in our API, and PELT refuses it.** §3.2.

---

## 3. Penalties — the main time sink

### 3.1 The formulas, from source, each confirmed by `penalty="Manual"` [V]

With `diffparam` = extra parameters per segment (**1** for a Gaussian mean change), at n = 300:

| name | formula | value |
|---|---|---|
| `SIC0`/`BIC0` | `diffparam·log n` | 5.7037824747 |
| `SIC`/`BIC` | `(diffparam+1)·log n` | 11.4075649493 |
| `MBIC` | `(diffparam+2)·log n` **plus a modified cost** | 17.1113474240 |
| `AIC0` | `2·diffparam` | 2.0 |
| `AIC` | `2(diffparam+1)` | 4.0 |
| `Hannan-Quinn0` | `2·diffparam·log log n` | 3.4822590936 |
| `Hannan-Quinn` | `2(diffparam+1)·log log n` | 6.9645181872 |
| `Asymptotic` | closed-form extreme-value surface | 18.35698 at α = 0.05 |

`SIC1`/`BIC1`/`AIC1` are deprecated and `stop()`.

### 3.2 MBIC is not a penalty, and it breaks PELT — defect D19

The full implemented criterion is `Σ_i [RSS_i + log(n_i)] + m(diffparam+2) log n` — the `log(n_i)` term is **per-segment, inside the cost** [S] `cost_general_functions.c:58`.

That cost violates PELT's K = 0 pruning inequality. Measured: **143 of 190** random segment splits violate it under the MBIC cost against **0 of 190** under plain RSS. Concretely, a length-100 segment split 50/50 gives `log 50 + log 50 − log 100 = +3.218876 > 0`. [V]

The consequence, re-verified independently in this session [V]:

| route | τ | MBIC criterion |
|---|---|---|
| `cpt.mean(penalty="MBIC", method="PELT")` — **the package default** | **187** | **285.00735894** |
| brute force over every candidate | 139 | 282.94063889 |
| `method="BinSeg"` | 139 | 282.94063889 |
| `fpop` (independent, exact) | 139 | — |

**Suboptimality 2.06672005, silently.** The *approximate* search gets the right answer where exact PELT does not.

**Decision: `changepoints(..., search=:pelt, penalty=:mbic)` throws**, naming the pruning violation and suggesting `:segneigh`. And **assert the pruning condition per cost** in a test, so a future cost cannot silently break exactness.

Arithmetic note: the implemented criterion is **not** Zhang & Siegmund (2007) — theirs implies `(2m−1) log n` where `changepoint` carries `3m log n`, a difference of `(m+1) log n`. [U — this is arithmetic from the source, not a check against the paper. Verify before citing.]

### 3.3 The penalty scale maps one-to-one — worth having checked

`changepoint`'s `mll_mean(x,x2,x3,n) = x2 − x²/n` and `ruptures`' `CostL2.error = var·n_seg` are the same quantity [S]. Measured over **8 fixtures × 6 penalties × 2 minimum segment lengths = 96 configurations: identical 96/96** with `jump=1` [V]. Corroborated by `fpop::Fpop(λ = 2 log n)` returning the identical 28 breakpoints on the variance fixture.

So `pen` needs no rescaling. **The silent mismatch is `jump`, not the scale** (D20).

### 3.4 The Gaussian mean cost is not scale-invariant

`mll_mean` is a bare sum of squares, **not a log-likelihood** — σ² is implicitly 1 [S]. A series with σ² = 4 is effectively penalised at `β/4` in standardised units, so `BIC = 2 log n` is **not scale-invariant**.

**Decision: document it loudly and offer a `standardize` keyword that is off by default**, so the reference's numbers reproduce but a user is told what the penalty means. This is the least obvious of the three cost traps and it will surprise anyone who changes units.

`ruptures`' `normal` cost instead estimates σ² per segment; `strucchange` concentrates one global σ² out of the pooled RSS. Three references, three objectives — and `ruptures` `normal` ≡ `changepoint` `cpt.meanvar` at equal penalty despite differing by the constant `n(log 2π + 1)`, which is segmentation-independent [V].

---

## 4. Conventions

### 4.1 The breakpoint index — the trap is Julia-specific

Verified on a deterministic 10-point step [V]:

| reference | output | meaning |
|---|---|---|
| `changepoint` `cpts()` | **5** | 1-based inclusive end of the preceding segment |
| `strucchange` `$breakpoints` | **5** | 1-based inclusive end |
| `ruptures` `predict()` | **[5, 10]** | 0-based exclusive end, series end appended |
| `ecp::e.divisive` | `1, 101, 201, 301` | 1-based **start** of each segment |

The three primary references agree numerically — a 0-based exclusive end equals a 1-based inclusive end. **`ecp` genuinely differs**, proving the hazard.

**Decision: our breakpoints are the 1-based inclusive end of the preceding segment**, matching `changepoint` and `strucchange`, and we do **not** append `n`. State it in the docstring with the `ruptures` comparison spelled out, because `ruptures`' *documentation* describes 100 as the start of the new regime, which in 1-based Julia is **101**.

### 4.2 Minimum segment length — three defaults, and they decide the answer

[V] `changepoint` **1** (2 for var/meanvar, 3 for reg); `ruptures` **2**, floored up per cost; `strucchange` **`h=0.15` → 45 at n=300**.

This alone flips three of eight fixtures. On 4-sd shifts at t=3 and t=297, `changepoint` and `ruptures` return exactly `3, 297` while `strucchange` returns **nothing**, supF p = 0.449 — because h = 45 makes a 3-long edge segment inadmissible. [V]

**Decision: one `min_segment_length`, defaulting to 2** (`ruptures`' value, and the smallest that makes a variance estimable), with the three reference defaults in the docstring and a test on the edge fixture asserting which reference we match.

### 4.3 `ruptures`' `jump` — do not reproduce it

Default **5**, silently approximating, never documented as such (D20). **We have no `jump` parameter.** Every comparison against `ruptures` must pass `jump=1`, and §6 requires off-grid fixtures so the hazard cannot hide.

### 4.4 Acceptance ties

`changepoint`'s BinSeg accepts on `reduction ≥ pen` [S] `BinSeg_one_func_minseglen.c:131` while PELT's `min_which` effectively accepts on `reduction > pen`. A tie gives *more* changepoints in BinSeg and *fewer* in PELT. Pick `>` for both and note it.

---

## 5. Fixtures

Ten, all saved as CSV at `%.17g`. The starter fixture reproduces as `numpy.random.default_rng(20261015).normal(0,1,300)` then `y[100:200] += 5.0`, sum **482.7709235930279** [V].

| id | definition | n | sum |
|---|---|---|---|
| `fx0` | mean 0→5→0 at 100/200 | 300 | 482.7709235930 |
| `fxA_var` | **sd 1→3→1, no mean change** | 300 | 21.3283562235 |
| `fxB_weak` | **0.5-sd shift at 150** — near the detection limit | 300 | 55.0267780869 |
| `fxC_null` | **no change at all** | 300 | −15.0027143194 |
| `fxD_edge` | 4-sd shifts at t=3 and t=297 | 300 | 1180.8870876074 |
| `fxE_n2000` | four shifts, for timing | 2000 | 1453.7450065195 |
| `fxF_offgrid` | **shifts at 97 and 203 — off the jump=5 grid** | 300 | 509.2309163920 |
| `fxG_many` | 9 alternating ±1.2 shifts every 30 | 300 | −25.2722157596 |
| `fxH_spike` | 1-point and 2-point spikes | 200 | 13.9174452915 |
| `fxI_step` | deterministic `[0]*5 + [10]*5` — the index probe | 10 | 50 |

**`fxF_offgrid` is the most important fixture in the item.** `fx0`'s breaks at 100 and 200 are both multiples of 5, which hid D20 entirely.

### 5.1 What each fixture establishes [V]

| fixture | result | what it tests |
|---|---|---|
| `fx0` | all three → `100, 200`; `confint` → `[99,101]`, `[199,201]`; BIC(m=2) = **838.563795**; supF = **90.053285** | the happy path |
| `fxA_var` | mean cost → 14 spurious breaks; `cpt.var`/`cpt.meanvar`/`Pelt(normal)` → `100, 199`; `strucchange` correctly silent (supF 2.921401, p 0.570466); **`efp(type="Score-CUSUM")` is the only test that detects it, p = 0.00038111** against OLS-CUSUM's 0.503186 | cost selection matters more than penalty |
| `fxB_weak` | PELT+MBIC → 187; everything else → 139; `confint` → `[112, 139, 162]`; supF 33.104626, p 3.65063626e-07 | D19 |
| `fxC_null` | **no false positives at any default**, all three; supF 3.945065 p 0.382972; all 7 efp types p > 0.25 | the null |
| `fxD_edge` | `3, 297` from two; **nothing** from `strucchange` | §4.2 |
| `fxE_n2000` | **unanimous** `400,800,1200,1600` | timing |
| `fxF_offgrid` | `97, 203` with `jump=1`; **`95,100,200,205` at `ruptures`' defaults** | D20 |
| `fxG_many` | 8 cpts at MBIC, 9 at BIC; `strucchange` silent — **two stacked defaults**, h=45 > the 30-long segments *and* max breaks `ceiling(300/45)−2 = 5 < 9` | default interaction |

---

## 6. Verification plan

### 6.1 Tier 1 — exact

| test | target | tolerance |
|---|---|---|
| PELT, all 8 fixtures × 6 penalties × 2 min lengths | `changepoint`/`ruptures`' agreed breakpoints | **exact** (96/96 established) |
| the penalty formulas | §3.1, all eight | 1e-12 |
| Bai–Perron RSS/BIC table | `summary.breakpointsfull`, every m | **machine precision** (diff = 0 established) |
| `break_confint` | `[99,100,101]`, `[199,200,201]`, `[112,139,162]`, and `fxE`'s four | **exact integers** |
| F-statistic path | supF 90.053285, 33.104626, 3.945065, 2.921401 | 1e-9 |
| efp paths | OLS-CUSUM 3.926752 on fx0 | 1e-9 |
| CUSUM p-values (closed form) | OLS-CUSUM 8.09353e-14 on fx0 | 1e-12 |

**Note the F-statistic convention**: `strucchange`'s "F statistics" are `(RSS_r − RSS_u)/(RSS_u/(n−2k))`, which is **k × the classical Chow F** — identical only at k = 1 [S]. Decide which we report and say so.

### 6.2 Mandatory reductions

1. **PELT ≡ SegNeigh/exact DP** for any cost satisfying K = 0. Verified to hold on two fixtures for plain RSS [V]; it is how D19 was found. **Also assert that PELT refuses MBIC.**
2. **BinSeg is not exact, and the test should say so.** On `fxG_many` at BIC, PELT = SegNeigh = `19,60,90,121,148,180,210,240,270` while BinSeg gives `19,60,90,121,151,180,210,216,240,270` — ten breaks, two misplaced. On `fxA_var` BinSeg returns **nothing** where PELT returns 14, because greedy splitting stops when no single split clears the penalty. [V]
3. **A single break at a known location** reduces to the Chow test at that date, and `fstats` at that τ must equal `(RSS_r − RSS_u)/(RSS_u/(n−2k))` computed directly.
4. **`fxI_step`** pins the index convention exactly (§4.1).

### 6.3 Independent cross-checks

Five further GPL/LGPL packages are installed and all agree on the main fixtures [V]: `fpop`, `wbs`, `Segmentor3IsBack`, `changepoint.np`, `ecp`. **`fpop` is the best oracle** — it is exact, fast, and does not share `changepoint`'s MBIC defect. Use it where `changepoint`'s defaults cannot be trusted.

### 6.4 Testset inventory

| # | Testset | Assertions (min) |
|---|---|---|
| 1 | PELT on fx0, six penalties | 14 |
| 2 | the eight penalty formulas | 10 |
| 3 | PELT ≡ exact DP, two costs | 10 |
| 4 | **PELT refuses MBIC, naming the pruning violation** | 8 |
| 5 | the pruning condition holds for every shipped cost | 12 |
| 6 | BinSeg is approximate — the `fxG_many` and `fxA_var` divergences | 12 |
| 7 | `fxF_offgrid` — breaks at 97, 203 exactly | 10 |
| 8 | `fxA_var` — mean cost spurious, variance cost correct | 14 |
| 9 | `fxB_weak` — 139, not 187 | 8 |
| 10 | `fxC_null` — no false positives at any default | 12 |
| 11 | `fxD_edge` and the minimum-segment-length default | 12 |
| 12 | `fxH_spike` | 8 |
| 13 | the index convention on `fxI_step` | 10 |
| 14 | cost inventory: mean, var, meanvar, Poisson, exponential | 16 |
| 15 | scale non-invariance is documented and `standardize` works | 10 |
| 16 | Bai–Perron RSS/BIC table, every m | 14 |
| 17 | Bai–Perron on `fx0`, `fxF`, `fxE` | 12 |
| 18 | `break_confint` on four fixtures | 16 |
| 19 | `fstats` path and supF on four fixtures | 14 |
| 20 | the k× convention versus the classical Chow F | 8 |
| 21 | efp: all implemented types on `fx0` | 16 |
| 22 | **Score-CUSUM detects the variance change where OLS-CUSUM does not** | 8 |
| 23 | CUSUM p-values, closed form | 12 |
| 24 | refusals: `min_segment_length` too large, m too large, empty series | 12 |
| 25 | determinism across repeated threaded runs | 6 |
| 26 | performance: n=2000 near 0.002 s, allocations, `@code_warntype` | 10 |

**26 testsets, ≥ 294 assertions.**

---

## 7. The supF null distribution — scope it out, and say why

Charter §6 states the problem. Restated here as a decision:

`strucchange`'s `pvalue.Fstats` is **38 lines of code plus 2234 lines of GPL coefficient tables** implementing Hansen (1997)'s fitted chi-square approximation. The survey **reproduced the mechanism to diff = 0 on five fixtures** [V], so the algorithm is understood; the tables are the obstacle.

**Ship 12.2 without supF p-values.** Report:

- the **supF, aveF and expF statistics**, which are exact and need nothing
- critical values **by simulation** of the limiting process, with the number of replications and the seed recorded, labelled as simulated rather than tabulated
- or nothing, with the statistic alone — which is what `tsDyn`'s `vcov.setar` effectively does for the threshold and what Stage 11 accepted there

**What to build instead, because it is free:**

- **`break_confint`** — `pargmaxV` is **27 lines of closed-form CDF** (Bai 1997), two branches in Φ and exp, **no table** [S]. Measured outputs give exact regression targets.
- **The CUSUM family** — Rec-CUSUM has a closed form; OLS-CUSUM, Score-CUSUM and RE use a Kolmogorov–Smirnov series `1 − (1 + 2Σ_{i=1}^{100} (−1)^i e^{−2i²x²})^k`. Both implementable alongside the existing `_std_normal_cdf` [S].
- **MOSUM is the only family needing tables** — defer it.

And **do not reproduce D27**: `strucchange`'s MOSUM p-values saturate at exactly 0.01 and its CUSUM p-values underflow to exactly 0, both via constant extrapolation, both reported as exact. On a manufactured 50-sd break: OLS-MOSUM p = 0.01, OLS-CUSUM p = 0. [V] Return the bound with a flag, or `NaN`, but not a number that looks exact.

---

## 8. Performance

Charter §7.1 gives the measured bar: `changepoint`'s PELT in C is **0.002 s** at n = 2000, against `ruptures`' exact PELT at **6.122 s** and `strucchange` at **10.367 s**.

The design that gets there:

1. **Precompute prefix sums once** (`Σy`, `Σy²`); every segment cost is then O(1). All twelve of `changepoint`'s costs are 111 lines of C total because they are closed forms on `(Σy, Σy², n)` [S].
2. **The candidate set is a `Vector{Int}` pruned in place**, not a dictionary copied per candidate — `ruptures` copies the whole partition dict per step, which is most of its 3000× gap.
3. **Bai–Perron's triangular RSS is the one O(n²) piece.** `@views`, `mul!`, and reuse `_ols`'s QR — its Stage 11 zero-weight support is a natural fit for segment masking.
4. Zero allocation inside the DP loop, asserted; `@code_warntype` clean.
5. `Threads.@threads` over the triangular RSS rows, with the Stage 11 closure-race lesson applied (`local`, plus a determinism test).

---

## 9. Budget

| Piece | Reference | Julia |
|---|---|---|
| PELT | 222 lines C (~70 of real logic) | 120–160 |
| BinSeg | 239 lines C | 100–140 |
| exact DP / SegNeigh | 23 R + in-file | 80–120 |
| costs (mean, var, meanvar, Poisson, exponential) | 111 lines C total | 90–130 |
| penalties incl. CROPS | 115 + 154 | 110–150 |
| Bai–Perron path + BIC selection | 142 + 54 + 29 | 180–240 |
| `break_confint` + `pargmaxV` | 128 + 27 | 130–170 |
| `fstats` | 131 | 110–150 |
| `efp`, CUSUM family only | 360 (all 9 types) | 180–240 |
| result types, `show` | — | 150–200 |
| **total** | | **1250–1700** |

---

## 10. Build order

1. **Prefix sums and the cost interface.** Nothing else works without them.
2. **Exact DP (SegNeigh)** — the slow, obviously-correct reference implementation.
3. **PELT**, tested to equal exact DP. **Then** the pruning-condition assertion, which is what makes MBIC's refusal principled rather than arbitrary.
4. **BinSeg**, with its inexactness pinned as a test rather than hidden.
5. **Penalties**, including the MBIC refusal.
6. **Bai–Perron path + BIC selection**, against the diff = 0 table.
7. **`break_confint`** — closed form, no table, exact integer targets.
8. **`fstats`** and the CUSUM family.
9. MOSUM and supF p-values: **deferred** (§7).

Steps 1–4 are where the exactness lives; nothing after them can be trusted if any is loose.

---

## 11. What not to do

1. **Do not use `changepoint`'s defaults as an oracle.** D19. Use `fpop`, `method="SegNeigh"`, or `penalty="Manual"`.
2. **Do not compare against `ruptures` without `jump=1`.** D20.
3. **Do not test only on round-numbered breakpoints.** `fx0` hid D20 completely.
4. **Do not implement MBIC as a cost.** §3.2.
5. **Do not let PELT run with a cost that violates K = 0.** §3.2.
6. **Do not append `n` to the breakpoint vector.** §4.1.
7. **Do not transcribe the Hansen (1997) tables.** §7.
8. **Do not report a saturated or underflowed p-value as exact.** D27.
9. **Do not report `strucchange`'s F as the classical Chow F.** §6.1 — it is k× it.
10. **Do not assume a mean-change cost finds a variance change.** `fxA_var` gives 14 spurious breaks; only Score-CUSUM detects it as a test.
