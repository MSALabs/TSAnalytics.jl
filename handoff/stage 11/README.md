# Stage 11 handoff bundle

**Assembled:** 2026-10-14  
**Repository HEAD at writing:** `ad0010e`  
**Contents:** 164 files — this README, 7 handoff documents, 16 fixtures, 70 ground-truth files and 70 reference scripts.

Every count and checksum in this README was computed by a script, not typed.

---

## Read in this order

| # | Document | Lines |
|---|---|---|
| 1 | `docs/stage-11-overall-handoff.md` | 701 |
| 2 | `docs/stage-11.0-primitives-handoff.md` | 581 |
| 3 | `docs/stage-11.1-unobserved-components-handoff.md` | 567 |
| 4 | `docs/stage-11.2-dynamic-factor-handoff.md` | 572 |
| 5 | `docs/stage-11.3-markov-switching-handoff.md` | 619 |
| 6 | `docs/stage-11.4-tar-star-handoff.md` | 773 |
| 7 | `docs/stage-11-plot-recipes-handoff.md` | 446 |
| | **total** | **4259** |

The charter (document 1) is a prerequisite for all the others. It fixes the
tagging convention, the licensing constraints, the 16-item reference-defect
list and the tiered test policy that every sub-stage depends on.

**Build order is not document order.** The charter's §1 explains why:
11.0 splits, and the honest sequencing is
**11.0 §5+§7 → 11.1 → 11.0 §2+§3+§6 → 11.2**, with 11.3 and 11.4
independent after 11.0 §7.

---

## Fixtures

All referenced by name from the handoffs. Simulated series travel as CSV, never
as a seed — neither R's nor NumPy's RNG stream is portable.

| File | Rows (incl. header) | sha256 (first 16) |
|---|---|---|
| `fixtures/fixtureA.csv` | 201 | `5db3de25e207b644` |
| `fixtures/fixtureA_miss.csv` | 201 | `d393bde9e7c0bb05` |
| `fixtures/fixtureA_miss_idx.json` | 1 | `1dfa646122e40136` |
| `fixtures/fixtureA_truefactor.csv` | 201 | `6ea5e1d9a4518ea7` |
| `fixtures/fixtureB.csv` | 251 | `d2f978d31962523c` |
| `fixtures/fixtureB_truefactor.csv` | 251 | `230b83fba7abba2a` |
| `fixtures/fixture_lynx.csv` | 115 | `c8e279fbc88f1843` |
| `fixtures/fixture_lynx_setar.json` | 14 | `33d55f6253994775` |
| `fixtures/fixture_meta.json` | 121 | `8f0c2e92604ad50d` |
| `fixtures/fixture_setar_sim.csv` | 1001 | `96f87cee4a20b1bf` |
| `fixtures/hamilton_rgnp.csv` | 136 | `0243abfdf487f800` |
| `fixtures/hamilton_rgnp.json` | 1 | `3d7a537d75408f7c` |
| `fixtures/nile.csv` | 101 | `e297df0996cf01f4` |
| `fixtures/sunspot_hansen.csv` | 290 | `e4c88338c7b4d2fe` |
| `fixtures/uc_R_filter.csv` | 101 | `02b866a4278469e4` |
| `fixtures/ukgas.csv` | 109 | `a23a593ffc468f08` |

`fixture_meta.json` carries the true generating parameters and checksums for the
two dynamic-factor fixtures. `uc_R_filter.csv` is R's own filtered and smoothed
state for the Nile local-level model at its fitted variances — the file behind
the 1.26e-08 agreement in the charter's §5.1.

---

## Ground truth

70 files. Reference output recorded for use as test expectations, subject
to the licensing rule in the charter's §4: **statsmodels figures (BSD-3) may be
transcribed; figures from `dfms`, `MSwM`, `tsDyn` and base R (all GPL) are
independent checks only.**

| Prefix | Source | Licence status |
|---|---|---|
| `sm_*` | statsmodels 0.15.0 | **transcribable** |
| `ss_*`, `exp_*`, `dfms_*` | `dfms` 1.0.1 | GPL-3 — check only |
| `compare_*`, `equiv_*` | cross-reference comparisons | derived |

---

## Scripts

| Directory | Files | Covers |
|---|---|---|
| `scripts/` | 15 | 11.0 / 11.1 state-space and the spot-checks |
| `scripts/dfm/` | 19 | 11.2 dynamic factor |
| `scripts/ms/` | 17 | 11.3 Markov-switching |
| `scripts/tar/` | 19 | 11.4 TAR/STAR |

Every `[V]` tag in the handoffs points at one of these. Three are worth singling
out:

- **`scripts/tar/f2_handroll.R`** — an independent base-R SETAR grid search that
  reproduces `tsDyn` **bit-exactly** (threshold difference 0, coefficient
  max-abs-difference 0, SSR difference 0). The template for the Julia
  implementation and the strongest verification asset in the stage.
- **`scripts/uc_match.py`** with **`scripts/uc_fixed2.R`** — the pair that showed
  R's `StructTS` and statsmodels' `UnobservedComponents` agree to 1.26e-08 once
  the initialisation is matched, inverting the conclusion a first look produces.
- **`scripts/dfm/compare2.R`** — the diagnosis that `dfms`' apparent 0.92
  log-likelihood advantage is entirely its free initial state condition.

---

## Reference environment

| Component | Version | Obtained |
|---|---|---|
| R | 4.3.3 | pre-installed |
| statsmodels | 0.15.0 | pre-installed |
| `dfms` | 1.0.1 | `apt-get install r-cran-collapse liblapack-dev libblas-dev gfortran`, then CRAN GitHub mirror + `R CMD INSTALL` |
| `MSwM` | 1.5 | `apt-get install r-cran-nlme`, then mirror + `R CMD INSTALL` |
| `tsDyn` | 11.0.5.2 | already installed from the Stage 10 session |
| `aeon` | 1.6.0 | `pip install aeon` |

**CRAN is unreachable from the sandbox**; the Ubuntu archive and
`github.com/cran/<pkg>` are the two working routes. **Julia is not installed**,
so nothing here was checked against the package's actual behaviour — every
claim about TSAnalytics.jl is tagged `[S]` (read from source) or `[U]`.

---

## What is new relative to the roadmap

1. **11.0 is a new row** — three sub-stages need the same missing machinery.
2. **Three dependency corrections** in `development-sequence.md`: 11.1 does not
   need the exact diffuse filter, 11.2 does not depend on 10.1, and 11.3 needs
   an `_ols` fix the roadmap does not mention.
3. **The roadmap's "weak Python support" for 11.4 is wrong** — `aeon` ships a
   SETAR forecaster that agrees with `tsDyn` to 12 digits.
4. **Sixteen reference defects**, D1–D16, one of which (D7) is ours.
