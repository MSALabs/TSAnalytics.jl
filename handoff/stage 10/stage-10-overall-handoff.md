# Handoff: Stage 10 — Multivariate (stage charter)

**Status: pending.** This is the *stage-level* document: scope, shared
decisions, conventions, reference environment and policy. Per-item
handoffs sit beneath it; `stage-10.1-var-handoff.md` is the first.

For a fresh Claude Code session with no prior context. Read this before
any Stage 10 item handoff.

Built against **TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`** (*Stage 9.4: TBATS, and a
third bug in R's forecast intervals*, 2026-10-02).

---

## 0. Tagging convention

| Tag | Meaning |
|---|---|
| `[V]` | **Verified by execution** — the command ran, the number shown is what it printed |
| `[S]` | **Read from source** at the commit above, or from an installed reference's source |
| `[U]` | Carried forward, **not** re-verified. Re-verify before relying on it |

Every reference number in this document and in `stage-10.1` is `[V]`.
Every claim about TSAnalytics internals is `[S]` — **Julia is not
installed in the sandbox these documents were written in**, so nothing
was checked by running TSAnalytics itself. The first implementation
session must re-establish a Julia runtime and re-verify any
package-internal claim before building on it.

---

## 1. What Stage 10 is

Vector time series: VAR, Granger causality, impulse responses, forecast
error variance decomposition, cointegration (Engle-Granger,
Phillips-Ouliaris, Johansen), VECM, structural VAR.

It is the **only stage on a fully independent dependency branch** — it
needs Stage 1's regression primitive and nothing from the state-space
engine, the optimiser backbone (for most items), or Stages 5–9.

`[S]` `src/abstract.jl` already anticipates it: `TimeSeriesModel`'s
docstring lists `VAR` among expected subtypes, and `UnivariateModel`'s
says, verbatim, *"Marker for models with a single observed series (as
opposed to VAR/VECM, which are inherently multivariate)"*. So a
`VARModel` subtypes `TimeSeriesModel` directly — no hierarchy change.

### Revised item list

Two rows are new relative to `development-sequence.md`, and one roadmap
row splits in two. Rationale in `stage-10-dependency-list.md`.

| # | Item | Status | Difficulty |
|---|---|---|---|
| **10.0** | Multi-response OLS, `tsmatrix`, companion matrix | **new row** | Easy |
| 10.1 | VAR + lag-order selection | pending — `stage-10.1-var-handoff.md` | Easy |
| 10.2 | Granger causality | pending — `stage-10.2-granger-handoff.md` | Easy–Moderate |
| 10.3 | IRF, FEVD (+ analytic and bootstrap standard errors) | pending | Easy–Moderate |
| **10.4a** | Cointegration — Engle-Granger (tau) | **split from 10.4** | Moderate |
| **10.4b** | Cointegration — Phillips-Ouliaris (Z) | **split from 10.4** | Moderate |
| 10.5 | Cointegration — Johansen | pending | Moderate–Hard |
| 10.6 | VECM | pending | Hard |
| 10.7 | Structural VAR | pending | Hard |
| **10.8** | Multivariate residual diagnostics | **new row** | Easy–Moderate |

**Build order, on evidence rather than numbering:**

```
10.0 → 10.1 → 10.3 → 10.8 → 10.2 → {10.4a, 10.4b} → 10.5 → 10.6 → 10.7
```

10.1 and 10.3 come first because they are the two items where both
references agree to machine precision (§5), so any disagreement during
implementation is a bug rather than a convention dispute. 10.8 comes
third because a fitted VAR is not shippable without residual
diagnostics. 10.4a/b are independent of everything above and can run in
parallel.

---

## 2. Governing principles (inherited, non-negotiable)

These are the project's standing rules. They are restated here because a
fresh session will not have them.

1. **Run it, don't recall it.** Every reference number is produced by
   executing the reference. No number in a test comes from memory, from
   a paper, or from a textbook table unless the table itself is the
   artefact being transcribed.
2. **Reference, never port.** `development-sequence.md:1815`. Read a
   reference's source to understand a correct design, then rebuild it
   natively. Do not translate it line by line. This has a licensing
   dimension in Stage 10 — see §4.
3. **Refuse to guess.** When a quantity cannot be computed properly,
   throw an `ArgumentError` naming what is missing. Do not return
   `NaN`, do not clip, do not silently approximate. The package has six
   documented instances of this (PP's `ρ` p-value, KPSS's clipped
   p-values, Durbin-Watson's missing p-value, EGARCH's analytic
   multi-step forecast, X-11's three-year minimum, GARCH `dist=:t`).
   Stage 10 adds at least three more: Johansen's p-value (§5.6),
   Engle-Granger beyond its table coverage (§5.7), and `p` too large for
   the sample (§5.8).
4. **Reduction testing.** Prove a general implementation by collapsing
   it exactly onto an already-verified special case. Used in STL,
   time-varying SSM, AR-GARCH and ETS. Stage 10's reductions: a `k=1`
   VAR onto the existing AR machinery (§5.9), a VECM at rank `k` onto a
   VAR in differences, an SVAR with `A = I` onto a plain VAR.
5. **`predict` and `forecast` are exact aliases**, everywhere. `[S]`
   `src/forecast.jl:434` already states this as package policy:
   *"`forecast` and `predict` are exact aliases here as everywhere
   else"*. Every Stage 10 model that forecasts exposes both, and a test
   asserts they return identical objects.
6. **Tests outnumber features.** `[V]` The package is currently
   **17,279 source lines and 18,207 test lines** — a 1.05:1 ratio.
   Stage 10 holds that ratio or better.
7. **Document the disagreement, not just the answer.** When two
   references disagree, the test pins *both*, the docstring explains
   which this package follows and why, and the manual says what to pass
   to reproduce the other.

---

## 3. Reference environment (reproducible)

`[V]` **CRAN is unreachable from this sandbox** — `cannot open URL
'https://cloud.r-project.org/src/contrib/PACKAGES'`. This blocked
`forecast::ets` for the whole of the Stage 9A work. Two routes around it
both work, and together they make **every Stage 10 item
dual-verifiable** — a genuine change from earlier stages.

```bash
apt-get update -qq                     # REQUIRED first: without it
                                       # r-base-core has "no installation candidate"
apt-get install -y --no-install-recommends r-base-core
apt-get install -y --no-install-recommends \
  r-cran-urca r-cran-tseries r-cran-lmtest r-cran-strucchange \
  r-cran-sandwich r-cran-mass r-cran-zoo r-cran-forecast r-cran-jsonlite \
  r-cran-mnormt r-cran-mgcv r-cran-nnet r-cran-matrix \
  r-cran-foreach r-cran-generics r-cran-desolve

# vars and tsDyn are NOT packaged for Ubuntu. CRAN's GitHub mirror is reachable.
for p in vars deSolve tseriesChaos tsDyn; do
  git clone --depth 1 -q https://github.com/cran/$p /tmp/rsrc/$p
done
R CMD INSTALL /tmp/rsrc/vars
R CMD INSTALL /tmp/rsrc/tseriesChaos    # needs deSolve installed first
R CMD INSTALL /tmp/rsrc/tsDyn

pip install statsmodels --break-system-packages
```

`[V]` Resulting versions, all confirmed working: **R 4.3.3**, `vars`
**1.6.1**, `urca` **1.3.3**, `tseries` **0.10.55**, `tsDyn`
**11.0.5.2**, `lmtest` **0.9.40**, `forecast` **8.21.1**, `statsmodels`
**0.15.0**.

`[V]` 1,130 `r-cran-*` packages are available in the Ubuntu archive —
worth checking there first for any future R dependency.

### Environment re-established on the implementation machine (2026-10-13)

The recipe above is for the Linux sandbox these documents were written
in. The actual implementation machine is **Windows**, and the whole
reference bundle was re-verified there before any code was written.

**Installed, and what differs from the authoring sandbox:**

| Package | Authoring sandbox | Implementation machine |
|---|---|---|
| R | 4.3.3 | **4.6.0** |
| `vars` | 1.6.1 | **1.6.1** — exact match, installed from CRAN as a binary |
| `urca` | 1.3.3 | 1.3.4 |
| `tseries` | 0.10.55 | 0.10.63 |
| `lmtest` | 0.9.40 | 0.9.40 — exact match |
| `strucchange` | — | 1.6.0 (pulled in by `vars`) |
| `forecast` | 8.21.1 | 9.0.2 |
| `jsonlite` | — | 2.0.0 |
| Python | — | 3.8.10 |
| `statsmodels` | 0.15.0 | **0.14.1** |
| `numpy` | 2.5.3 | **1.24.3** |

`install.packages("vars")` is all that was needed; the `apt-get` and
GitHub-mirror route is not required on Windows.

**`tsDyn` is NOT installable here and does not need to be.** It has no
binary for R 4.6.0, and there is no Rtools on this machine
(`pkgbuild::has_build_tools()` is `FALSE`, no `make` on `PATH`), so the
charter's `R CMD INSTALL` route cannot run either. It matters less than
it looks: `tsDyn::VECM` is only the **third** R reference for 10.6, and
`urca::cajorls` and `vars::vec2var` are both present, so 10.6 stays
dual-verifiable against R and `statsmodels VECM`. Installing Rtools is
the fix if a third opinion is ever wanted.

**Do not upgrade `statsmodels` to 0.15.0.** `[V]` 0.14.1 is the newest
release that exists for Python 3.8, so upgrading would mean upgrading
Python — and every earlier stage's Python references were generated on
3.8.10 / statsmodels 0.14.1. The version gap was tested instead of
assumed away: all four reference JSONs were regenerated on this machine
and diffed against the bundled ones, value by value.

| Bundle file | Values compared | Max relative difference | Structural mismatches |
|---|---|---|---|
| `ref_R.json` | 2,319 | `8.4e-15` | none but the R version string |
| `ref_R_granger.json` | 186 | **`0.0` — bit-identical** | none but the R version string |
| `ref_py.json` | 2,743 | `6.2e-11` | none but the version strings |
| `ref_py_granger.json` | 336 | `7.7e-13` | **none at all** |

**5,584 reference numbers, nothing above `1e-9`**, across R 4.6.0 vs
4.3.3, statsmodels 0.14.1 vs 0.15.0 and numpy 1.24.3 vs 2.5.3. The worst
case, `6.2e-11`, is on `econ5` — the badly-conditioned `k = 5` fixture
this bundle deliberately includes as the conditioning stress case, where
even R-vs-Python agreement degrades to `~5e-13`. So the bundle is fully
reproducible here and the recorded numbers can be trusted as they stand.

`[V]` The one version-sensitive **defect** claim was checked directly
rather than inferred: statsmodels' `test_inst_causality` `IndexError`
(§5.10 item 2) reproduces identically on 0.14.1 — same
`sig_mask[causing_ind, caused_ind] = 1` line, same error as soon as the
causing set has more than one member. §5.10 stands.

### Which reference serves which item

`[V]` Every entry below was executed.

| Item | R | Python |
|---|---|---|
| 10.1 | `vars::VAR`, `vars::VARselect` | `VAR`, `.select_order` |
| 10.2 | `vars::causality`, `lmtest::grangertest` | `.test_causality(kind='f'|'wald')` |
| 10.3 | `vars::irf`, `fevd`, `Psi`, `Phi` | `.irf()`, `.fevd()` |
| 10.4a | *none built-in* — hand-roll in R | `stattools.coint(method='aeg')` |
| 10.4b | `tseries::po.test`, `urca::ca.po` | *none* |
| 10.5 | `urca::ca.jo` | `coint_johansen` |
| 10.6 | `urca::cajorls`, `vars::vec2var`, `tsDyn::VECM` | `VECM` |
| 10.7 | `vars::SVAR`, `BQ`, `SVEC` | `SVAR` |
| 10.8 | `vars::serial.test`, `normality.test`, `arch.test` | thin |
| EG tables | `urca:::.urcval` (niv = 1…12) | `adfvalues.mackinnonp/crit` |

---

## 4. Licensing — a hard constraint on how references are used

`urca`, `vars`, `tsDyn`, `tseries` are **GPL**. `statsmodels` is
**BSD-3**. TSAnalytics.jl is **MIT**.

`[S]` The precedent is already in the codebase. `src/unitroot.jl:105–117`
records that the ADF/PP tables were transcribed from `statsmodels` and
`arch` (BSD/NCSA), with the R packages used only for end-to-end checks.

**Keep that line.** For Stage 10:

- Transcribe numeric tables **only** from `statsmodels`.
- Use `urca`/`vars`/`tsDyn` to **verify**, never as a transcription
  source.
- Where `statsmodels` carries no table at all — Phillips-Ouliaris — take
  the numbers from the published paper, not from reading `ca-po.R`.
  `[S]` `urca/R/ca-po.R:66,76` hardcodes them as a `c(5,3,3)` array
  (m = 2…6 × three levels × three variants); useful to know it exists and
  to check against, not to copy.

Dataset licensing, settled earlier and unchanged: `vars::Canada` — the
canonical `vars` example — is GPL-2+ and **cannot be bundled** in the
MIT core. Use the GPL-3 datasets already redistributed in `data/`.

---

## 5. Cross-cutting conventions — the traps, with evidence

Every item below was found by executing both references on identical
data. These are stage-wide: they will bite more than one item, which is
why they live in the charter rather than in one item's handoff.

### 5.1 Deterministic-term ordering is opposite between references

`[V]` R's `Bcoef()` is `K × (Kp + ndet)` with deterministic terms
**last**, columns `[y1.l1, y2.l1, …, const]`. statsmodels' `params` is
`(ndet + Kp) × K` with deterministic terms **first**, and transposed.

Verified by reordering and matching on eight fitted models — maximum
absolute difference `2.5e-16` (k=2) to `5.5e-13` (k=5, econ5).

**Decision: TSAnalytics stores coefficients as `A::Vector{Matrix}` of
length `p`, each `K × K`, plus a separate `deterministic::Matrix`
(`K × ndet`).** Neither reference's flat layout. This makes `A[j]`
directly indexable for the companion matrix and the IRF recursion, and
removes the ordering question from every downstream item.

### 5.2 Trend vocabularies are mutually incomplete

`[V]` Verified on `var2_p1.csv`, k=2, p=1:

| R `type=` | statsmodels `trend=` | Agree? | log-likelihood |
|---|---|---|---|
| `"const"` | `"c"` | **yes** | −500.8313597101 both |
| `"none"` | `"n"` | **yes** | −522.3719059322 both |
| `"trend"` (trend, **no constant**) | *no equivalent* | — | R −506.6828693867 |
| `"both"` | `"ct"` | **yes** | −500.5285172459 both |
| *no equivalent* | `"ctt"` (const + t + t²) | — | py −497.6112301085 |

The trap: `type="trend"` and `trend="ct"` sound equivalent and are not.
R's `"trend"` has `ndet=1`; `"ct"` has `ndet=2`. `[V]` They differ by
6.15 in log-likelihood, and the mismatch propagates into order selection
— `VARselect(type="trend")` vs `select_order(trend="ct")` differ by
`4.3e-2` in AIC at lag 1.

**Decision: a `deterministic::Symbol` keyword spanning both
vocabularies** — `:none`, `:const`, `:trend` (trend only, no constant,
matching R), `:both` (const + trend, matching `"ct"`), `:quadratic`
(matching `"ctt"`). Five options, a superset of each reference, each
mapped explicitly in the docstring. Reject R's `type=`/statsmodels'
`trend=` spelling; both are ambiguous.

### 5.3 `roots` are exact reciprocals between references

`[V]` The single most dangerous trap in the stage, because it silently
inverts a stability test.

`vars::roots()` returns **moduli of the companion-matrix eigenvalues**
(stable if all `< 1`). statsmodels' `.roots` returns **roots of the
characteristic polynomial** (stable if all `|·| > 1`). Verified on five
fixtures, `max|R − 1/py|` between `1.1e-16` and `9.3e-15`:

| Fixture | R (companion moduli) | statsmodels (char. roots) |
|---|---|---|
| `var2_p1`, p=1 | 0.358411725751, 0.501276142801 | 1.99490842, 2.79008729 |
| `var2_nearunit`, p=1 | 0.871893192744, **0.974494137410** | 1.02617344, 1.14692947 |
| `var2_explosive`, p=1 | 0.941776653648, **1.040418520983** | 0.96115167, 1.06182288 |

**Decision: `roots(m::VARModel)` returns companion eigenvalue moduli
(R's convention, stable if `< 1`)**, because "the largest root is 0.97,
close to the boundary" is the reading practitioners expect, and
`is_stable(m) = maximum(roots(m)) < 1`. The docstring states the
reciprocal relationship to statsmodels explicitly, and a test asserts
`roots_julia ≈ 1 ./ abs.(roots_statsmodels)`.

### 5.4 Order-selection criteria hold the sample fixed; a fitted model does not

`[V]` This one would silently produce different numbers from both
references with no error anywhere.

Both `VARselect(lag.max=L)` and `select_order(L)` estimate **every**
candidate `p` on the **same** `T = n − L` observations, so the criteria
are comparable across `p`. A single `fit(p)` uses all `n − p`
observations available to it. On `var2_p1.csv` with `L = 8`:

```
select_order(8).aic[lag 1]          = -0.6050354269424555   (T = 192)
VAR(A).fit(1).aic                   = -0.5819716231697688   (T = 199)
VAR(A[L-1:]).fit(1).aic             = -0.6050354269424555   (T = 192)  <- matches
```

`[V]` And the four criteria, recomputed by hand and matching both
references to `1e-12` at p = 1, 2, 3, 8, with `Σ̃ = E'E / T` (the ML
divisor, **not** the df-adjusted one) and `nfree = Kp + ndet` **per
equation**:

```
aic  = log det(Σ̃) + (2 / T)              · K · nfree
bic  = log det(Σ̃) + (log T / T)          · K · nfree
hqic = log det(Σ̃) + (2 log log T / T)    · K · nfree
fpe  = ((T + nfree) / (T − nfree))^K · det(Σ̃)
```

`[V]` Also: `select_order(L)` returns **L+1 values (lags 0…L)** where
`VARselect(lag.max=L)` returns **L (lags 1…L)**. Off-by-one when
comparing.

**Decision:** the selection function holds the sample fixed at
`n − lag_max` and says so in its docstring; a fitted model's own
`aic`/`bic` use its own sample, matching both references. A test pins
*both* numbers so the difference can never be mistaken for a bug.

### 5.4b The trend column runs `p+1 : n`, and the wrong choice is silent

`[V]` Both references build the linear trend on the **calendar index of
the estimation sample** (`p+1 … n`), not a within-sample counter
(`1 … T`). They agree with each other on both deterministic coefficients
to `9.5e-16`.

The reason this needs pinning: an origin shift is absorbed entirely by
the constant, so **fit, residuals, fitted values, Σ, the log-likelihood
and every information criterion are identical under both forms.** Only
the reported intercept moves — by exactly `p ×` the trend coefficient.
Verified directly on `var2_p1.csv`, `p = 1`, `:both`:

```
trend = p+1:n   |const - reference| = 1.97e-15      <- matches
trend = 1:T     |const - reference| = 4.922e-04     ( = 1 x 4.92e-4 )
```

A test must assert the intercept, not just the fit, or this passes
undetected.

### 5.5 Σ has three conventions, and all three agree across references

`[V]` Good news, pinned anyway. On `var2_p1.csv`, p=1, const:

| Convention | Value `[1,1]` | R | Python |
|---|---|---|---|
| df-adjusted, `E'E/(T − Kp − ndet)` | 1.024427598172 | `summary()` Σ | `sigma_u` |
| ML, `E'E/T` | 1.008983966038 | — | `sigma_u_mle` |

Agreement `6.7e-16` (df) and `8.9e-16` (ML) on k=2; `9.7e-13` on econ5
(k=5, values of order 10⁴). **Both must be stored**: the df version
feeds standard errors and forecast intervals, the ML version feeds the
log-likelihood and the information criteria.

### 5.6 Johansen: statistics agree, nothing around them does

Three traps, all `[V]` on a cointegrated pair (`x` a random walk,
`z = x + 0.5ε`, n = 200):

**(a) Output order is reversed.** `urca` returns `[7.028307,
82.589857]` (r ≤ 1 first); `statsmodels` returns `[82.589857, 7.028307]`
(r = 0 first). Identical numbers, opposite order.

**(b) `ecdet="none"` maps to `det_order=0`, not `-1`.** The names invite
the opposite mapping:

| `urca` `ecdet` | trace statistics | `statsmodels` `det_order` | `lr1` |
|---|---|---|---|
| `"none"` | 7.028307, 82.589857 | `-1` | 0.143865, 75.230862 |
| `"const"` | 7.666556, 83.230455 | `0` | **7.028307, 82.589857** ← matches `"none"` |
| `"trend"` | 7.516541, 83.661089 | `1` | 7.254090, 83.398337 |

`ca.jo(ecdet="none")` still includes an **unrestricted** constant;
`det_order=-1` means no deterministic term at all.

**(c) The critical-value tables disagree and neither gives a p-value.**
`[V]` 5% trace: `urca` (8.18, 17.95) — Osterwald-Lenum (1992);
`statsmodels` (15.4943, 3.8415) — its own `coint_tables.c_sjt`, indexed
on `n − r`, whose smallest entry 3.8415 is just the χ²(1) 95% point.
Reversing one does not reconcile them. `[V]` Coverage: `c_sja`/`c_sjt`
cover n = 1…12 and `det_order ∈ {−1, 0, 1}`; **`det_order = 2` returns
`[nan, nan, nan]` silently**, no error.

**Decisions for 10.5:** ship Osterwald-Lenum (matching `urca` and most
published applied work); `pvalue()` **refuses** rather than
approximating; an out-of-range deterministic order **errors** rather
than returning `NaN`.

`[V]` **(d) Use the generalized symmetric eigenproblem.** statsmodels
emits `ComplexWarning: Casting complex values to real discards the
imaginary part` from `vecm.py:717–718` on every `coint_johansen` call —
its eigenvalues come back complex and are truncated. `eigen(A, B)` on
the symmetric generalized problem returns real eigenvalues by
construction. Already in `LinearAlgebra`; no new dependency.

### 5.7 Engle-Granger and Phillips-Ouliaris are different tests

`[V]` The roadmap lists them as one row with two references. On the same
cointegrated fixture:

```
tseries::po.test(W)        statistic = -188.7648   p = 0.01 (capped, with a warning)
statsmodels coint(aeg)     statistic =  -13.3784   p = 5.04e-24
```

Phillips-Ouliaris is a Z-type (normalized bias) statistic;
augmented Engle-Granger is a tau-type (t-ratio). Different null
distributions, different tables, different scales. Hence the 10.4a /
10.4b split.

`[V]` **The N > 1 tables that `src/unitroot.jl:123–126` deliberately
skipped are cross-checkable.** `urca`'s internal
`.urcval(arg, nobs, niv, itt, itv, nc)` exposes MacKinnon's surfaces for
`niv = 1…12`. *(The exported `punitroot`/`qunitroot` hide this — their
`N` argument is the **sample size**, not the number of series. An easy
and costly misreading.)* Constant case, nobs = 200:

| N | `urca` 5% crit | `statsmodels` 5% crit | `urca` p(−3.3670) | `sm` p(−3.3670) |
|---|---|---|---|---|
| 2 | −3.366924 | −3.36685108 | 0.049991 | 0.046150 |
| 3 | −3.783338 | −3.78374380 | 0.127301 | 0.120313 |
| 6 | −4.791978 | −4.79194880 | 0.560486 | 0.551075 |
| 8 | −5.344673 | *(crit only)* | 0.812263 | *(no p-value)* |
| 12 | −6.294929 | *(crit only)* | 0.984520 | *(no p-value)* |

Agreement: ~`4e-4` on critical values, ~`0.005–0.009` on p-values — two
vintages of MacKinnon's response surfaces, not a disagreement about the
test.

`[V]` **Coverage limits, which bound what 10.4a may honestly return:**
statsmodels' p-value surfaces (`tau_*_smallp`, `tau_*_largep`) are shape
**(6,3)** and **(6,4)** → **N ≤ 6**. Its critical-value tables
(`tau_*_2010`) are **(12,3,4)** → N ≤ 12, *except* `tau_nc_2010` which
is **(1,3,4)** → no-constant is **N = 1 only**.

### 5.8 Both references silently accept nonsense sample sizes

`[V]` `n = 8`, `k = 2`, `p = 5` → 11 parameters per equation from 3
observations:

```
R   VAR(A[1:8,], p=5, type="const")   ->  "no error"
py  VAR(A[:8]).fit(5)                 ->  "no error"
```

`[V]` And the other edge cases:

| Case | R | Python |
|---|---|---|
| `p = 0` | `length of 'dimnames' [2] not equal to array extent` | **no error** |
| `NaN` in input | `NAs in y.` | `LinAlgError: SVD did not converge in Linear Least Squares` (plus LAPACK writing `DLASCL parameter number 4 had an illegal value` to stderr) |
| invalid trend | `'arg' should be one of "const", "trend", "both", "none"` | `ValueError: trend must be one of: 'c', 'ct', 'ctt', 'n'` |

**Decision:** TSAnalytics refuses all four by name with an actionable
message. This is the honest-refusal pattern, and here it is strictly
better than both references.

### 5.9 Both references refuse `k = 1`

`[V]` Exact messages:

```
R : The matrix 'y' should contain at least two variables. For univariate
    analysis consider ar() and arima() in package stats.
py: ValueError: Only gave one variable to VAR
```

**Decision: the public `fit_var` refuses `k = 1` by name too** — no
surprise relative to either reference, and the univariate path is
already far better served by `fit_arx`/`fit_arima`. But the **internal
multi-response OLS primitive (10.0) must handle `k = 1`**, and a
reduction test runs it against `stats::ar.ols`, which is exactly what a
`k = 1` VAR(p) with a constant is. `[V]` Reference values on
`var1_k1.csv` (n = 200):

```
ar.ols(z, order.max=1, aic=FALSE, demean=FALSE, intercept=TRUE)
  ar        = 0.6484308401025832
  intercept = 0.32794476411707674
  var.pred  = 0.9538204266778918
  asy.se    = 0.05410479122534204   (on ar)
            = 0.08644379310823685   (on intercept)

order.max=2:  ar = [0.673140896685106, -0.03945717137380811]
              intercept = 0.33998905461259055
              var.pred  = 0.9560758286522902
```

### 5.10 Reference defects found so far (running list)

Keep this list current — each item's handoff adds to it. These are
places where a test must **not** be written against the reference.

1. **`vars::causality` p-values underflow to exactly zero.** `[V]`
   `vars/R/causality.R:135` computes `1 - pf(...)` and `:167` computes
   `1 - pchisq(...)`, instead of the upper tail. On `climhyd` with
   p = 2, an ordinary Granger F of `13.359738021985` with df (10, 2634)
   is reported as `p = 0.000000e+00` where the true tail is
   `3.664479e-23`; the instantaneous χ² of `184.728185617809` on 5 df is
   reported as `0` against a true `5.229214e-38`. statsmodels gets both
   right. **Consequence:** TSAnalytics needs `_f_ccdf` (10.2 §4.3) and
   must not compare p-values against `vars` in the tail.
2. **statsmodels `test_inst_causality` raises `IndexError` whenever
   both the causing and caused sets have more than one member.** `[V]`
   `var_model.py:2337` does `sig_mask[causing_ind, caused_ind] = 1`,
   two fancy index arrays that only broadcast when one has length 1.
   Verified failing at (2,4), (3,3), (4,2) on k=6, (2,3) on k=5 and
   (2,2) on k=4; working at (1, k−1) and (k−1, 1). It always errors and
   never returns a silently wrong answer — checked deliberately.
   **Consequence:** `vars` is the only reference for the general block
   case; its `df = |causing| · |caused|`.
3. **Both references silently accept nonsense inputs** — §5.8 for VAR
   sizes; for causality, `vars` accepts an omitted `cause` (warns and
   uses the first variable) and statsmodels accepts `cause` = every
   variable and overlapping cause/caused sets.

*(For context, the project found three bugs in R's `forecast` during
Stage 9 — the ETS no-trend seasonal interval variance, the ETS(M,Ad,M)
forecast contradicting its own fitted recursion, and the seasonal TBATS
intervals being 2.8× too narrow. Items 1 and 2 above are the fourth and
fifth.)*

### 5.11 Correction to the dependency list on 10.2

`stage-10-dependency-list.md` §4.2 said Granger causality "must pick a
convention" because three references gave different p-values. That
framing was wrong. Grouped properly there are **two different tests,
each with two references that agree exactly**:

- **System test** (VAR-based): `vars::causality` ≡ statsmodels
  `test_causality`. `[V]` Agreement on all 15 cases tested — F to
  `≤ 7.6e-12`, p to `≤ 7.5e-15`, identical df.
- **Pairwise test** (single-equation): `lmtest::grangertest` ≡
  statsmodels `grangercausalitytests[...]["ssr_ftest"]`. `[V]`
  Bit-identical F.

The statistics coincide only when `k = 2`; the p-values never do,
because `df2` differs by a factor of `K` (386 vs 193 on `var2_p1`,
p = 2). So 10.2 exposes **both** tests rather than choosing — see
`stage-10.2-granger-handoff.md` §3.

---

## 6. Shared architecture decisions

Settled here so no item re-litigates them.

### 6.1 Data layout: observations × variables

`Y::AbstractMatrix` with **rows = time, columns = variables** — matching
both references and every bundled multivariate CSV. Column-major Julia
means a single variable's series is contiguous, which is what the
residual and fitted loops walk.

**Do not transpose to variables × time** even though the VAR algebra
reads more naturally that way. The regressor matrix `Z` is built once
and consumed by BLAS, which does not care; the input layout should match
what users have.

### 6.2 `tsmatrix`, not overloaded `tsvalues`

`[S]` `src/interface.jl` documents `tsvalues(x) -> AbstractVector{<:Real}`
with two methods. The container-agnostic contract is vector-shaped by
construction, and `collect(Float64, A)` on a matrix already returns a
matrix — so the fallback **silently violates its own documented return
contract** today rather than erroring.

Add a companion:

```julia
tsmatrix(x) -> AbstractMatrix{<:Real}
```

with methods for `AbstractMatrix`, a `Tuple`/`Vector` of vectors, and a
`NamedTuple` of columns (which is what `dataset()` already returns for
multivariate entries, `[S]` `src/dataset_catalogue.jl`). Plus:

```julia
tsnames(x) -> Union{Nothing, Vector{Symbol}}
```

Variable names are not cosmetic in Stage 10 — `[V]` `vars::irf()` returns
`$irf$y2[, "y1"]`, named in both directions, and every `irf`, `fevd` and
`causality` result is indexed by variable. Names default to
`[:y1, …, :yK]` when the container carries none.

### 6.3 A new `VARForecast`, not a generalised `Forecast`

`[S]` `src/forecast.jl:15`:

```julia
struct Forecast
    point::Vector{Float64}       # univariate by construction
    se::Vector{Float64}
    levels::Vector{Float64}
    lower::Matrix{Float64}       # horizon × levels
    upper::Matrix{Float64}
    horizon::Int
    model_name::String
end
```

A VAR forecast is `horizon × K` points with `horizon × K × levels`
bands. Generalising `Forecast` is a **breaking change to a type nine
model families already return**. Add `VARForecast` alongside it, with
the same field *names* where they mean the same thing so the two read
alike.

`[S]` Note there is **no plot recipe for `Forecast` at all** —
`src/recipes.jl` covers `ACFResult`, `PeriodogramResult`, the three
decompositions and the two diagnostic-plot types, and nothing else. So a
VAR fan chart is the package's first forecast recipe, not an extension
of an existing one. `vars::fanchart` is the reference.

### 6.4 `HypothesisTest` needs a way to say "critical values only"

`[S]` `src/abstract.jl:64`: `pvalue(t::HypothesisTest) = t.pvalue`,
which assumes every test has one. Johansen does not (§5.6c), and
Phillips-Ouliaris' p-value is capped (§5.7).

Add a small, contained extension to `abstract.jl`:

```julia
critical_values(t::HypothesisTest) -> Union{Nothing, NamedTuple}
```

returning `(; levels, values)` or `nothing`, and have `pvalue` throw a
documented `ArgumentError` for tests that genuinely have none — naming
`critical_values` as the thing to call instead. Do **not** store `NaN`
in `t.pvalue`; that is the failure mode §5.6c criticises statsmodels
for.

### 6.5 No new package dependency

`[S]` `Project.toml` is unchanged by Stage 10. Specifically:

- `LinearAlgebra` provides `qr`, `cholesky`, `Hermitian`, `kron`,
  `eigen` and the **generalized** `eigen(A, B)` Johansen needs.
- `_chisq_ccdf` (`src/diagnostics.jl:13`), `_f_cdf` (`:126`),
  `_beta_inc_reg` (`:86`) and `_std_normal_cdf` (`:610`) already cover
  every tail the stage needs.
- `Random` is already present for bootstrap paths.

**Do not add `Distributions.jl`.** It is the obvious instinct on
reaching a multivariate stage and it would be the package's first new
dependency since 0.1.0, for tails that already exist.

### 6.6 Shared fixtures

`[V]` All bundled already, nothing to download.

| File | Shape | Role |
|---|---|---|
| `test_data/uschange.csv` | 187 × 5 | fpp3's VAR benchmark; already exposed as `TSAnalytics.US_CHANGE` |
| `data/econ5.csv` | 161 × 5 | Shumway & Stoffer's VAR/cointegration example |
| `data/climhyd.csv` | 454 × 6 | S&S cross-correlation example; 10.2 |
| `data/aus_production.csv` | 217 × 7 | mixed-frequency multivariate |

**Correction applied 2026-10-13 (first implementation session).** `econ5`
and `climhyd` were listed as `160 × 5` and `453 × 6` here, in the README,
in the dependency list §1 (tagged `[V]`) and in 10.1 §3. Both were off by
one. Re-measured three ways on the bundled files, which are byte-identical
to `data/`: `wc -l` gives 162 and 455 total lines, every file ends in a
newline and contains no blank lines, and both reference stacks agree that
the first line is a header — `R read.csv` and `pandas.read_csv` each
return **161 × 5** and **454 × 6**. `uschange` at `187 × 5` was correct.
Corrected in all five documents. The lesson is the charter's own §2.1: two
`[V]` tags were carrying recalled numbers.

Plus the seeded synthetic fixtures generated for 10.1, which the later
items reuse — see `stage-10.1-var-handoff.md` §3 and the `fixtures/`
directory in this bundle.

---

## 7. Performance policy

Julia's selling point is that a correct implementation is also a fast
one. The stage-wide rules:

1. **Factorize once.** A VAR's `K` equations share one regressor matrix.
   One `qr(Z)`, `K` back-substitutions. Never a loop of `_ols` calls.
2. **Type stability everywhere.** Every exported function gets a
   `@code_warntype` check in review. No `Any` in a hot loop, no
   abstractly-typed struct fields (`Vector{Matrix{Float64}}`, not
   `Vector{Matrix}`).
3. **Allocation-free hot loops.** The IRF recursion, the FEVD
   accumulation and the forecast recursion run `h` times over `K × K`
   matrices. Preallocate and use `mul!`, `@views`, `@inbounds`. Target
   **zero allocations** inside the recursion after setup, asserted with
   `@allocated`.
4. **Exploit structure instead of calling a general routine.** `[V]`
   Both references compute `np.linalg.pinv(duplication_matrix(k))` — an
   SVD-based pseudo-inverse. But `D'D` is *exactly diagonal* with
   entries 1 and 2, so `D⁺ = (D'D)⁻¹D'` is closed-form. Verified for
   k = 2, 3, 5, 8: `max|pinv(D) − closed form| = 2.22e-16`, `D'D`
   off-diagonal mass exactly `0.0`. Faster **and** exact.
5. **Threading over independent fits only.** Order selection fits
   `lag_max` independent models; `Threads.@threads` over them is safe
   and is the pattern `auto_ets` already uses. A test asserts the
   parallel and serial paths select the identical model.
6. **Benchmark against the reference, and record it.** Each item's
   handoff carries a timing table for `k ∈ {2, 5, 10}` on its own
   fixtures. "Faster than R/Python" is a claim, so it gets a number.
7. **Never trade accuracy for speed silently.** `_ols`'s QR default
   exists because Cholesky squares the condition number. The same choice
   applies here: QR by default, Cholesky available, documented.

---

## 8. Test policy

Per item, the battery must cover:

1. **Exact agreement with R**, every fitted quantity, on at least three
   fixtures including one real dataset.
2. **Exact agreement with Python**, same quantities, same fixtures.
3. **Reference-vs-reference disagreements pinned** — both numbers
   asserted, with a comment naming the cause.
4. **Reduction onto an already-verified case.**
5. **Parameter recovery** from a known DGP, where a DGP exists.
6. **Edge cases**: minimum viable sample, near-boundary stability,
   explosive, `k = 1`, `p = 0`, `NaN`, invalid keywords.
7. **Error messages**: every refusal asserted by type *and* that the
   message names the fix.
8. **`predict` ≡ `forecast`** returns identical objects.
9. **Type stability** (`@inferred` where practical) and **allocation**
   assertions on the hot loop.
10. **Threading determinism** where a parallel path exists.

Each item's verification scripts live in
`test/verification/<item>/` — both the `.R` and the `.py`, so a future
session can regenerate every number rather than trusting the JSON.

---

## 9. What NOT to do

- Do **not** start with 10.5 or 10.7 because they look most impressive.
  They have the most convention traps and the least reference consensus.
- Do **not** write `fit_var` before 10.0's multi-response OLS exists; it
  will end up with a loop of `_ols` calls baked in.
- Do **not** generalise `Forecast`.
- Do **not** add `Distributions.jl`.
- Do **not** copy table numbers out of `urca` or `vars` (GPL).
- Do **not** report a p-value for Johansen.
- Do **not** trust the roadmap's `Depends on` column — it is wrong for
  10.5 and the 10.4 row describes two different tests. See
  `stage-10-dependency-list.md`.
- Do **not** assume `roots` means the same thing as in either reference
  without re-reading §5.3.

---

## 10. On finishing each item

1. Mark the item `✅ built` in `development-sequence.md` with the same
   level of detail the Stage 9 rows carry — what was verified against
   what, what disagreed, what was refused and why.
2. Add `Status: implemented ✅` at the top of the item's handoff; keep
   the handoff rather than deleting it (`handoff/README.md:44–53`).
3. Record any **new** reference-vs-reference disagreement in this
   charter's §5, so the next item inherits it.
4. Add the manual chapter. `[S]` The docs are complete except
   `docs/src/api/decomposition.md` (a 10-line stub) — Stage 10 should
   not add a second stub.

---

## Appendix: outstanding items from earlier stages

Unrelated to Stage 10 but still open, listed so they are not lost:

- The **"Diwali production surge" framing is wrong** and needs
  correcting in Introduction chapters 5, 16, 20, 36 and 40. `[V]` on
  real IIP data the coefficient is **−0.0317, t −3.07, p 0.0021** —
  production *falls* ~3.2%. Factories close for the holiday; retail
  consumption rises, but IIP measures production.
- Two home-page inaccuracies: the stale `residuals`/`forecast`
  sentence, and the dataset count (the page says 90; `data/` holds 88
  files).
- `docs/src/api/decomposition.md` is still a 10-line stub.
- `development-sequence.md`'s rewrite never landed in
  **SeasonalAdjustment.jl**.
- Decide whether to add Durbin & Koopman and Ladiray & Quenneville PDFs,
  or re-attribute Part VI and chapter 38 — `[U]` those citations rest on
  training memory, not on documents in hand.
