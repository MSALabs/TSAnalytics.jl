# Stage 12 handoff bundle

**Assembled:** 2026-10-15  
**Repository HEAD at writing:** `2eccb82`  
**Contents:** 110 files — this README, 6 handoff documents, 17 fixtures, 34 ground-truth files and 52 reference scripts.

Every count and checksum here was computed by a script, not typed.

---

## Read in this order

| # | Document | Lines |
|---|---|---|
| 1 | `docs/stage-12-overall-handoff.md` | 397 |
| 2 | `docs/stage-12.1-intermittent-demand-handoff.md` | 314 |
| 3 | `docs/stage-12.2-changepoint-breaks-handoff.md` | 315 |
| 4 | `docs/stage-12.3-lag-feature-regression-handoff.md` | 294 |
| 5 | `docs/stage-12-plot-recipes-handoff.md` | 178 |
| 6 | `docs/stage-12-documentation-handoff.md` | 295 |
| | **total** | **1793** |

The charter is a prerequisite for the rest. Its **§1.2** is the finding that
reshapes the stage: `arx` plus `forecast(::ARXModel)` already *is* the recursive
ML reduction, matched to 2e-15 against three references, so Stage 12.3 is
rescoped rather than built.

**Build order is not document order.** 12.1 and 12.2 are independent; 12.3
depends on nothing new and could go first if a quick win is wanted.

---

## Fixtures

Simulated series travel as CSV, never as a seed — neither R's nor NumPy's RNG
stream is portable. Note F1's provenance is **R** (`set.seed(7); rpois(60, 0.4)`),
not numpy.

| File | Rows (incl. header) | sha256 (first 16) |
|---|---|---|
| `fixtures/changepoint_fx.csv` | 301 | `2ecba13fabf82d59` |
| `fixtures/fixture_a_ar2.csv` | 121 | `292a95f1d680634d` |
| `fixtures/fixture_b_seasonal.csv` | 145 | `937a997f22f4a43d` |
| `fixtures/fixture_c_exog.csv` | 139 | `79ee524a46208708` |
| `fixtures/fx0_regen.csv` | 301 | `2ecba13fabf82d59` |
| `fixtures/fxA_var.csv` | 301 | `7f1d10dd2f7c088e` |
| `fixtures/fxB_weak.csv` | 301 | `202b2142a9f16829` |
| `fixtures/fxC_null.csv` | 301 | `af269114a1e08674` |
| `fixtures/fxD_edge.csv` | 301 | `190a8fad6a133058` |
| `fixtures/fxE_n2000.csv` | 2001 | `659b47b969f6b8a7` |
| `fixtures/fxF_offgrid.csv` | 301 | `b1b721b9b0ddd983` |
| `fixtures/fxG_many.csv` | 301 | `945a84a4436ae6b1` |
| `fixtures/fxH_spike.csv` | 201 | `fb7a1182f85dda31` |
| `fixtures/fxI_step.csv` | 11 | `fef41c625fbd0d4a` |
| `fixtures/fx_dense_p150_n120.csv` | 121 | `f1397fa7d910d335` |
| `fixtures/fx_sparse_p010_n120.csv` | 121 | `09fa409260f461cb` |
| `fixtures/intermittent.csv` | 61 | `e4841a597368ae1e` |

**One fixture is deliberately absent.** The intermittent-demand survey built a
fourth fixture from `expsmooth::partx`, which is **GPL data**. The outputs computed
on it are ours; the inputs are not. It is excluded from this bundle and must not
ship in `test/` — see charter §4.1.

The two most important fixtures are `changepoint/fxF_offgrid.csv`, whose breaks at
97 and 203 are off the multiple-of-5 grid that hid defect D20 entirely, and
`fx_dense_p150_n120.csv`, whose ADI of 1.3187 sits just inside the 1.32
classification boundary.

---

## Ground truth

34 files. Subject to the licensing rule in charter §4 — and Stage 12 is the
first stage where the primary references are mostly **permissive**:

| Source | License | Transcribable? |
|---|---|---|
| `ruptures` 1.1.10 | BSD-2-Clause | **yes** |
| `forecastML` 0.9.0 | MIT | **yes** |
| `sktime` 1.2.0, `scikit-learn` 1.9.1 | BSD-3-Clause | **yes** |
| `statsforecast` 2.1.1, `mlforecast` 1.1.0 | Apache-2.0 | **yes** — note the NOTICE clause |
| `forecast`, `changepoint`, `strucchange`, `fpop`, and five others | GPL / LGPL | **no** — checks only |

The one blocked item is `strucchange`'s 2234 lines of Hansen (1997) coefficient
tables, which the supF p-value needs. Charter §6 scopes those p-values out and
explains what to build instead — the break-date confidence intervals and the
CUSUM family are closed-form and need no table.

---

## Scripts

| Directory | Files | Covers |
|---|---|---|
| `scripts/` | 1 | the first-pass survey |
| `scripts/intermittent/` | 11 | 12.1 |
| `scripts/changepoint/` | 13 | 12.2 |
| `scripts/mlreduction/` | 27 | 12.3 |

Every `[V]` tag in the handoffs points at one of these. Four findings were
re-verified independently before being written down: `changepoint`'s MBIC
suboptimality (2.06672005), `ruptures`' `jump=5` default, `sktime`'s lag-0 leak
(R² = 1.0, max |feature − y| = 0), and the `arx` equivalence (reproduced to 15
digits).

---

## What is new relative to the roadmap

1. **12.3 is rescoped, not built** — most of it ships as `arx`, `forecast`,
   `tscv`, `fourier_terms` and `accuracy`. Four things are genuinely missing.
2. **12.2 merges two roadmap entries** — change-point detection and the
   structural-break/CUSUM gap the book's Frontier chapter lists separately.
3. **12.1 widens** from three methods to five plus a classifier; `forecast` has
   only classical Croston.
4. **Fifteen reference defects**, D19–D33, four of which change what gets built.
5. **Intermittent demand appears in none of the six textbooks** — including
   Hyndman's own lecture notes, despite his maintaining `forecast::croston`.
