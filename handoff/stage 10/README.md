# Stage 10 handoff bundle

Generated 2026-10-13 against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

## Read in this order

1. `stage-10-overall-handoff.md` — the stage charter. Scope, governing
   principles, reference environment, the cross-cutting convention traps
   (§5, including the running list of reference defects at §5.10), the
   shared architecture decisions (§6), performance and test policy
   (§7, §8). **Read this first; the item handoffs cite it and do not
   repeat it.**
2. `stage-10.1-var-handoff.md` — VAR and lag-order selection.
3. `stage-10.2-granger-handoff.md` — Granger causality and
   instantaneous causality. Depends on 10.1 plus one new numerical
   primitive, `_f_ccdf` (its §4).
4. `stage-10-dependency-list.md` — the survey the above are built on.

## fixtures/

| CSV | Shape | Role |
|---|---|---|
| `var2_p1.csv` | 200 × 2 | primary synthetic fixture, VAR(1) |
| `var3_p2.csv` | 250 × 3 | true order 2; causality blocks |
| `var1_k1.csv` | 200 × 1 | k=1 refusal + the 10.0 reduction test |
| `var2_short.csv` | 30 × 2 | df edge cases |
| `var2_nearunit.csv` | 200 × 2 | near-boundary stability (max root 0.974) |
| `var2_explosive.csv` | 60 × 2 | `is_stable == false` (max root 1.040) |
| `uschange.csv` | 187 × 5 | fpp3 benchmark (already in `test_data/`) |
| `econ5.csv` | 161 × 5 | Shumway & Stoffer; conditioning stress |
| `climhyd.csv` | 454 × 6 | k=6 block sweep; the p-value tail cases |
| `us4.csv` | 187 × 4 | the (2 cause, 2 caused) case statsmodels cannot do |

| JSON | Contents |
|---|---|
| `ref_R.json` | 12 fitted VARs + 8 selection tables, `vars` 1.6.1, 17 sig digits |
| `ref_py.json` | statsmodels 0.15.0 counterparts |
| `ref_R_extra.json` | stability edge cases + R error messages |
| `ref_py_extra.json` | same from statsmodels |
| `ref_R_granger.json` | 15 system tests, 15 instantaneous, 6 pairwise (`vars` + `lmtest`) |
| `ref_py_granger.json` | statsmodels counterparts, incl. its own error strings and the 4 `grangercausalitytests` variants |

Load these in the test files and assert broadly; they hold considerably
more than the numbers quoted inline in the handoffs.

## scripts/

| Script | What it does |
|---|---|
| `gen_fixtures.R` | regenerates every synthetic CSV bit-for-bit (seeded) |
| `ref_R.R` | `vars`/`VARselect` reference values → `ref_R.json` |
| `extra2.R` | stability edge cases + R error messages |
| `ref_py.py` | statsmodels VAR reference values → `ref_py.json` |
| `cmp.py` | R-vs-Python comparison; reproduces every finding in charter §5 |
| `ref_R_granger.R` | `vars::causality` + `lmtest::grangertest` → `ref_R_granger.json` |
| `ref_py_granger.py` | statsmodels causality tests → `ref_py_granger.json` |

Copy `fixtures/*.csv` to `test/fixtures/var/` and `scripts/*` to
`test/verification/var/` and `test/verification/granger/`, so every
number can be regenerated rather than trusted.

## Reference environment

CRAN is unreachable from the sandbox; charter §3 has the working recipe
(Ubuntu archive for most packages, CRAN's GitHub mirror for `vars` and
`tsDyn`). Versions used: R 4.3.3, `vars` 1.6.1, `urca` 1.3.3, `tseries`
0.10.55, `tsDyn` 11.0.5.2, `lmtest` 0.9.40, `forecast` 8.21.1,
`statsmodels` 0.15.0.

**Already re-established on the implementation machine** — see the
charter's "Environment re-established" subsection in §3. `vars` 1.6.1 is
installed there from CRAN directly, `tsDyn` is unavailable and not
needed, `statsmodels` stays at 0.14.1 deliberately, and all four JSONs
below were regenerated and diffed against the bundled copies: 5,584
values, nothing differing by more than `1e-9`.
