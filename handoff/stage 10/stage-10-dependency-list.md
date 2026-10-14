# Stage 10 (Multivariate) — Fresh Dependency List

Written for a fresh Claude Code session with no prior context.
Built against **TSAnalytics.jl HEAD `4bb66f38027be3b4fb538de408449f174128c8e0`**
(`Stage 9.4: TBATS, and a third bug in R's forecast intervals`,
Fri 2026-10-02), not against the roadmap's own `Depends on` column —
which is checked and corrected below, as Stages 8.3 and 8.5 both had to
correct theirs.

**Status: dependency survey only.** No implementation, no API proposal,
no test fixtures. Those come next, per item.

## Tagging convention (used throughout)

| Tag | Meaning |
|---|---|
| `[V]` | **Verified by execution in this session** — the command ran, the number below is what it printed |
| `[S]` | **Read from source** at the commit above, or from an installed reference's own source |
| `[U]` | Carried forward from an earlier session, **not** re-verified. Re-verify before relying on it |

No `[U]` items appear in sections 1–4. Every code-side claim is `[S]`
(read at HEAD) and every reference-side claim is `[V]` (executed).
Julia is **not** installed in this sandbox, so nothing here was checked
by running TSAnalytics itself — hence `[S]`, never `[V]`, for package
internals.

---

## 1. What Stage 10 inherits (already built — do not rebuild)

All line numbers are at HEAD. All `[S]`.

### 1.1 Regression

| Helper | Location | What it gives Stage 10 |
|---|---|---|
| `_ols(X, y; weights, method)` | `src/unitroot.jl:51` | QR (default) or Cholesky, GLS via `weights`, returns `(beta, resid, se)`. The roadmap's "10.1 depends on 1.4" is **correct** |
| `_diagvec(A)` | `src/unitroot.jl:99` | diagonal extraction for the `se` path |
| `_optimize(objective, x0; ...)` → `OptimResult` | `src/optim.jl:57`, struct at `:17` | Needed **only** by 10.6 (VECM ML) and 10.7 (SVAR scoring/ML). 10.1–10.5 are closed-form or eigen problems and need no optimiser |

**One real gap in `_ols`, and it is the gap that matters most for
10.1.** The signature is `y::AbstractVector{<:Real}` — single response
only. A VAR's `k` equations share one regressor matrix `Z`, so the
right computation is **one QR factorization of `Z`, then `k`
back-substitutions**, not `k` calls to `_ols`. Calling `_ols` in a loop
re-factorizes `Z` every time: for `k` variables that is `k`× the
factorization cost for no benefit, and it is exactly the kind of
avoidable constant factor that makes a Julia package look no faster
than R. See §2.1.

### 1.2 Distributions — nothing new needed

| Helper | Location | Used by |
|---|---|---|
| `_chisq_ccdf(x, df)` | `src/diagnostics.jl:13` | 10.2 Wald form; multivariate portmanteau (§2.4) |
| `_f_cdf(x, d1, d2)` | `src/diagnostics.jl:126` | 10.2 F form |
| `_beta_inc_reg(a, b, x)` | `src/diagnostics.jl:86` | backs `_f_cdf` |
| `_std_normal_cdf(z)` | `src/diagnostics.jl:610` | IRF bootstrap percentile bands, if normal-approximation bands are also offered |

`[S]` **No `Distributions.jl` dependency is required by Stage 10.** The
chi-square and F tails the stage needs already exist, hand-rolled on
the regularized incomplete gamma and beta functions. This is worth
stating because the obvious instinct on reaching a multivariate stage
is to reach for `Distributions.jl`, and it would be the package's first
new dependency since 0.1.0.

### 1.3 Critical-value plumbing

| Helper | Location | Reusable for |
|---|---|---|
| `_horner(coefs, x)` | `src/unitroot.jl:129` | Engle-Granger p-values use the **identical** MacKinnon response-surface form as `adf_test` — same Horner evaluation, only a different coefficient row (N > 1 instead of N = 1) |
| `_interp_pvalue_left(stat, table)` | `src/unitroot.jl:304` | left-tail table interpolation → Phillips-Ouliaris |
| `_interp_pvalue_right(stat, table)` | `src/unitroot.jl:559` | right-tail → Johansen trace / max-eigenvalue |

### 1.4 Tests that can be reused — with one trap

`[S]` `adf_test` (`src/unitroot.jl:401`), `pp_test` (`:792`),
`kpss_test` (`:660`), `_bartlett_lrv(u, l)` (`:595`, Newey-West
long-run variance with a Bartlett kernel).

**The trap, stated in the source itself.** `src/unitroot.jl:123–126`:

> Only N=1 (a single time series -- the only case `adf_test`/`pp_test`
> ever need) is transcribed; statsmodels'/arch's own N=2:12
> (multi-series cointegration) tables are not, so this project doesn't
> carry untested numbers nobody calls.

So `adf_test` is reusable as the **second step** of Engle-Granger
(the ADF regression on the cointegrating residual), but its *p-value is
wrong for that use* — it reads the N=1 table. 10.4 must supply its own
N>1 tables. This is a deliberate, documented deferral now coming due,
not an oversight. See §3.1.

### 1.5 Type hierarchy

`[S]` `src/abstract.jl`: `TimeSeriesModel` (`:13`), `StateSpaceModel`
(`:32`), `UnivariateModel` (`:40`), `HypothesisTest` (`:50`), with
`statistic(t) = t.statistic` (`:57`) and `pvalue(t) = t.pvalue` (`:64`).

Two things already anticipate this stage:

- `UnivariateModel`'s docstring says, verbatim, *"Marker for models with
  a single observed series (as opposed to VAR/VECM, which are inherently
  multivariate)"* — so a `VARModel` subtypes `TimeSeriesModel` directly
  and the hierarchy needs no change.
- `TimeSeriesModel`'s own docstring already lists `VAR` among the
  concrete subtypes it expects.

**But `HypothesisTest` has a hard edge.** `pvalue(t) = t.pvalue` assumes
every test has one. Johansen does not — see §4.4. That is a contract
decision to make before 10.5, not during it.

### 1.6 Fixture data — already bundled, nothing to download

| File | Shape | Why it is the right fixture |
|---|---|---|
| `test_data/uschange.csv` | 187 × 5 (Consumption, Income, Production, Savings, Unemployment) | `[V]` fpp3's own VAR example; already exposed as `TSAnalytics.US_CHANGE` (`src/datasets.jl:46`), whose docstring already calls it a *"Multivariate/VAR … benchmark"* |
| `data/econ5.csv` | 160 × 5 | `[V]` Shumway & Stoffer's `econ5` — their VAR and cointegration chapter example. GPL-3, already redistributed |
| `data/climhyd.csv` | 453 × 6 | `[V]` S&S cross-correlation example; useful for 10.2 |
| `data/aus_production.csv` | 217 × 7 | `[V]` already bundled, mixed-frequency multivariate |

`[S]` `src/dataset_catalogue.jl` already handles multi-column datasets
(`n_cols > 2`, real column names in `notes`), so the catalogue needs no
structural change to serve a VAR example.

`vars::Canada` — the canonical `vars` example — is **GPL-2+** and
therefore cannot be bundled in the MIT core package. Use `uschange` and
`econ5`, which are already there. (If `Canada` is wanted for
line-by-line comparability with `vars`' own documentation, it belongs in
the separate GPL-3 `TSAnalyticsDatasets.jl` discussed earlier.)

### 1.7 Package dependencies

`[S]` `Project.toml` currently: `Dates`, `DelimitedFiles`, `FFTW`,
`ForwardDiff`, `LinearAlgebra`, `Optim`, `Random`, `RecipesBase`,
`Statistics`, `StatsAPI`, `StatsBase`.

**Stage 10 needs no new entry.** Specifically:

- `LinearAlgebra` already provides `qr`, `cholesky`, `Hermitian`, `eigen`
  — and `eigen(A, B)`, the **generalized** symmetric eigenproblem, which
  is exactly what Johansen's reduced-rank step is. See §4.5.
- `StatsBase: CoefTable` is already imported for coefficient printing.
- `RecipesBase` already in, so IRF/FEVD recipes add no dependency.

---

## 2. What does **not** exist and must be built

All `[S]`.

### 2.1 Multi-response least squares

As §1.1. A VAR(p) on `k` variables with a constant has
`Z ∈ ℝ^{(n-p) × (kp+1)}` shared across all `k` equations. Required:
one factorization, `k` solves, and the residual covariance
`Σ̂ = E'E/(n-p-kp-1)` from the same pass. Either a `_ols` method taking
`Y::AbstractMatrix`, or a separate `_mols`. This is a Stage 1-level
primitive arriving late, and it should be written as one — not buried
inside `fit_var`.

### 2.2 A multivariate container accessor

`[S]` `src/interface.jl` documents `tsvalues(x) -> AbstractVector{<:Real}`
with exactly two methods: `tsvalues(x::AbstractVector{<:Real}) = x` and
`tsvalues(x) = collect(Float64, x)`. The whole container-agnostic
contract — the thing that lets `acf` accept a `TSFrame` column, a
`DataFrame` column, a `TimeArray`, or a generator — is **vector-shaped
by construction**.

Stage 10 is the first stage whose input is a matrix or a set of named
columns. This needs a decided answer, not an accident:

1. a `tsmatrix(x) -> AbstractMatrix{<:Real}` companion, with methods for
   `AbstractMatrix`, a tuple/vector of vectors, and a `NamedTuple` of
   columns (which is what `dataset()` already returns for multivariate
   entries); or
2. extra `tsvalues` methods — but the docstring's stated return
   contract then becomes false, and `collect(Float64, A)` on a matrix
   already returns a matrix, so today the fallback *silently* violates
   its own documented contract rather than erroring.

Option 1, plus **variable names**, which a VAR needs in a way no
univariate model does: every `irf`, `fevd` and Granger result is
indexed by variable, and `vars`/`statsmodels` both label their output.
`[V]` `vars::irf()` returns `$irf$y2[, "y1"]` — named both ways.

### 2.3 A multivariate forecast type

`[S]` `src/forecast.jl:15`:

```julia
struct Forecast
    point::Vector{Float64}
    se::Vector{Float64}
    levels::Vector{Float64}
    lower::Matrix{Float64}      # horizon × levels
    upper::Matrix{Float64}
    horizon::Int
    model_name::String
end
```

Strictly univariate: `point` is one vector, and `lower`/`upper` are
`horizon × levels`. A VAR forecast is `horizon × k` points with
`horizon × k × levels` bands. Either a `VARForecast`, or generalise
`Forecast` — and generalising is a **breaking change to a type that nine
model families already return**, so it should be decided deliberately.

`[S]` Related: **there is no plot recipe for `Forecast` at all.**
`src/recipes.jl` has recipes for `ACFResult`, `PeriodogramResult`,
`ClassicalDecomposition`, `STLDecomposition`, `MSTLDecomposition`,
`DiagnosticPlotResult`, `GarchDiagnosticPlotResult` — and nothing for
`Forecast`. So "add a fan chart for VAR" is not an extension of an
existing recipe; it is the first forecast recipe in the package.
(`vars::fanchart` is the reference.)

### 2.4 Multivariate diagnostics — none exist

`[S]` grep across `src/` finds no Hosking, no Li-McLeod, no multivariate
Jarque-Bera, no multivariate ARCH-LM. `ljungbox_test`
(`src/diagnostics.jl:236–245`) is univariate and takes a single series.

A fitted VAR is not usable without these — "is the residual vector
white?" is the first question anyone asks of one. All three references
exist and were executed:

`[V]` `vars::serial.test(v, lags.pt=8)` → Hosking portmanteau
statistic `17.14725`, df `24`.
Also `vars::normality.test`, `vars::arch.test`.

This is genuinely new work and deserves its own numbered item rather
than being smuggled into 10.1.

### 2.5 Tables to transcribe

Three table families, none present:

1. MacKinnon cointegration tables for N = 2…12 → 10.4 (§3.1)
2. Johansen trace and max-eigenvalue critical values → 10.5 (§4.4)
3. Phillips-Ouliaris critical values → 10.4b (§3.1)

### 2.6 Companion-matrix machinery

`[S]` A `k`-variate VAR(p) stacks to a `kp × kp` companion matrix,
needed for stability (`vars::roots`), for `Psi`/`Phi` (the MA
representation behind IRF), and for the FEVD accumulation. The package
has `_ar_stationary`/`stationary_cov` for the **univariate** AR case
(`src/autoreg.jl`, used by Stage 9.4) but nothing `k`-variate.

### 2.7 Bootstrap IRF confidence bands

`[V]` `vars::irf()` defaults to `boot=TRUE` with 100 runs; I ran it with
`boot=FALSE` deliberately so the point IRF could be compared exactly
(§4.1). Bootstrap bands are the default in both references and will be
expected — they need a seeded, reproducible resampling path, and
`Random` is already a dependency.

---

## 3. Reference dependencies — all now available, which is new

This is the part that changed most. **CRAN is still unreachable from
this sandbox** `[V]` (`cannot open URL
'https://cloud.r-project.org/src/contrib/PACKAGES'`), which blocked
`forecast::ets` for the whole of the Stage 9A work. Two routes around it
both work:

- **Ubuntu archive** `[V]` — `r-cran-urca`, `r-cran-tseries`,
  `r-cran-lmtest`, `r-cran-strucchange`, `r-cran-sandwich`,
  `r-cran-mass`, `r-cran-forecast`, `r-cran-mgcv`, `r-cran-nnet`,
  `r-cran-mnormt`, `r-cran-matrix`, `r-cran-foreach`, `r-cran-generics`,
  `r-cran-desolve` (1,130 `r-cran-*` packages are available in total).
  `apt-get update` is required first — without it `r-base-core` reports
  *"no installation candidate"*.
- **CRAN's GitHub mirror** `[V]` — `vars` and `tsDyn` are **not** in the
  Ubuntu archive, but `github.com/cran/<pkg>` is clonable and
  `R CMD INSTALL` on the clone works.

### 3.1 Exact reproduction recipe for a fresh session

```bash
apt-get update -qq
apt-get install -y --no-install-recommends r-base-core
apt-get install -y --no-install-recommends \
  r-cran-urca r-cran-tseries r-cran-lmtest r-cran-strucchange \
  r-cran-sandwich r-cran-mass r-cran-zoo r-cran-forecast \
  r-cran-mnormt r-cran-mgcv r-cran-nnet r-cran-matrix \
  r-cran-foreach r-cran-generics r-cran-desolve
# vars and tsDyn are not packaged for Ubuntu -- use CRAN's GitHub mirror
for p in vars deSolve tseriesChaos tsDyn; do
  git clone --depth 1 -q https://github.com/cran/$p /tmp/rsrc/$p
done
R CMD INSTALL /tmp/rsrc/vars
R CMD INSTALL /tmp/rsrc/tseriesChaos      # needs deSolve first
R CMD INSTALL /tmp/rsrc/tsDyn
pip install statsmodels --break-system-packages
```

`[V]` Resulting versions: **R 4.3.3**, `vars` **1.6.1**, `urca`
**1.3.3**, `tseries` **0.10.55**, `tsDyn` **11.0.5.2**, `lmtest`
**0.9.40**, `forecast` **8.21.1**, `statsmodels` **0.15.0**.

### 3.2 Per-item reference coverage

Every entry below was **executed** this session, not looked up.

| Item | R reference | Python reference | Dual-verifiable? |
|---|---|---|---|
| 10.1 VAR + order selection | `vars::VAR`, `vars::VARselect` | `statsmodels VAR`, `.select_order` | **Yes — exact agreement** (§4.1) |
| 10.2 Granger causality | `vars::causality`, `lmtest::grangertest` | `.test_causality(kind='f'|'wald')` | Yes, statistic exact; **p-values differ** (§4.2) |
| 10.3 IRF / FEVD | `vars::irf`, `vars::fevd`, `vars::Psi`, `vars::Phi` | `.irf()`, `.fevd()` | **Yes — exact agreement** (§4.1) |
| 10.4a Engle-Granger | *none built-in* — must be hand-rolled in R | `statsmodels.tsa.stattools.coint(method='aeg')` | Partially — see §4.3 |
| 10.4b Phillips-Ouliaris | `tseries::po.test`, `urca::ca.po` | *none* | R only |
| 10.5 Johansen | `urca::ca.jo` | `coint_johansen` | Yes, statistics exact; **ordering reversed, CV tables disagree** (§4.4) |
| 10.6 VECM | `urca::cajorls`, `vars::vec2var`, `tsDyn::VECM` | `statsmodels VECM` | Yes — three independent R paths |
| 10.7 SVAR | `vars::SVAR`, `vars::BQ`, `vars::SVEC` | `statsmodels SVAR` | Yes (R side is the stronger one — the roadmap's "weak Python support" note holds) |
| §2.4 multivariate diagnostics | `vars::serial.test`, `normality.test`, `arch.test` | *thin* | R only |
| Cointegration tables | `urca:::.urcval` (niv = 1…12) | `statsmodels.tsa.adfvalues.mackinnonp/crit` | **Yes, and they cross-check** (§4.3) |

### 3.3 Licensing — a constraint on *how* these are used

`urca`, `vars` and `tsDyn` are **GPL**; `tseries` is GPL. `statsmodels`
is **BSD-3**. TSAnalytics.jl is MIT.

The project's standing policy — *"reference, never port"*
(`development-sequence.md:1815`) — already covers this, and §1.4's own
source comment shows the precedent: the N=1 tables were transcribed from
`statsmodels`/`arch` (both BSD/NCSA), with the R packages used only as
end-to-end checks.

**Keep that line for Stage 10.** Transcribe numeric tables from
`statsmodels` only; use `urca`/`vars`/`tsDyn` to *verify*. Where
`statsmodels` does not carry a table at all (Phillips-Ouliaris), the
table must come from the published paper, not from reading `ca-po.R`.
`[S]` `urca/R/ca-po.R:66,76` hardcodes the PO critical values as a
`c(5,3,3)` array (m = 2…6 × three levels × three variants) — useful to
know it exists, and to check against, but not to copy.

---

## 4. Verified findings that change the roadmap's own dependency claims

Everything in this section was executed. Fixtures: a bivariate
stationary VAR(1), n = 200, `set.seed(42)` in R
(`y1ₜ = 0.5y1ₜ₋₁ + 0.2y2ₜ₋₁ + e1ₜ`, `y2ₜ = 0.3y2ₜ₋₁ − 0.1y1ₜ₋₁ + e2ₜ`),
and a cointegrated pair (`x` a random walk, `z = x + 0.5ε`). The same
CSVs were handed to both R and Python so the comparisons are
apples-to-apples.

### 4.1 10.1 and 10.3 agree exactly across references — no convention risk

`[V]`

| Quantity | R (`vars` 1.6.1) | Python (`statsmodels` 0.15.0) |
|---|---|---|
| `VARselect` AIC order | 1 | 1 |
| VAR(2) log-likelihood | −535.8054 | −535.8054024141139 |
| Orthogonalized IRF, y1 ← y2, h = 0…4 | 0, 0.01801116, 0.1001568, 0.06915096, 0.02544945 | identical to all printed digits |
| FEVD of y1 at h = 4 | 0.9877639, 0.01223609 | 0.98776391, 0.01223609 |

This is unusually clean for this project. 10.1 and 10.3 can be verified
against two references that agree, so any disagreement found during
implementation is a bug in the implementation — not a convention
dispute to adjudicate. **Build these first.**

### 4.2 10.2: the statistic is universal, the p-value is not

`[V]` Same fixture, cause = y2, two lags:

| Reference | Statistic | p-value |
|---|---|---|
| `vars::causality(v, cause="y2")$Granger` | F = 0.9652132977 | — |
| `lmtest::grangertest(y1 ~ y2, order=2)` | F = 0.9652132977 | **0.3827328** |
| `statsmodels .test_causality(kind='f')` | F = 0.9652132977163053 | **0.3818193** |
| `statsmodels .test_causality(kind='wald')` | χ² = 1.9304265954326105 | 0.3809019 |

Three independent implementations agree on the F statistic to ten
significant figures, and the two p-values differ in the third decimal.
The cause is the denominator degrees of freedom, and
χ² = 2 × F = J × F exactly (J = 2 restrictions) confirms the Wald form
is the same quantity without the small-sample correction. **10.2 must
pick a convention, document it, and expose the other** — the same
pattern as `fit_ets`'s `sigma2 = sse/n` vs R's `sse/(n−np)`.

### 4.3 10.4's two listed references are two different tests

The roadmap lists 10.4 as *"Cointegration — Engle-Granger"* with
`tseries::po.test` (R) and `statsmodels…coint` (Python) side by side, as
though they were two views of one test. `[V]` On the identical
cointegrated fixture:

```
tseries::po.test(W)           statistic = -188.7648   p = 0.01 (capped, with a warning)
statsmodels coint(aeg)        statistic =  -13.3784   p = 5.04e-24
```

These are not the same test. `po.test` is **Phillips-Ouliaris**, a
Z-type (normalized bias) statistic; `coint(method='aeg')` is
**augmented Engle-Granger**, a tau-type (t-ratio) statistic. Different
null distributions, different tables, different scales.

**Recommendation: split 10.4 into 10.4a (Engle-Granger, tau) and 10.4b
(Phillips-Ouliaris, Z).** The current single row cannot be implemented
or verified as written.

**And the tables that 10.4a needs are cross-checkable.** `[V]`
`urca`'s internal `.urcval(arg, nobs, niv, itt, itv, nc)` exposes
MacKinnon's cointegration surfaces for `niv = 1…12` — the exported
`punitroot`/`qunitroot` wrappers hide this (their `N` argument is the
*sample size*, not the number of series, which is an easy and costly
misreading). Against `statsmodels`, constant case, nobs = 200:

| N | `urca` 5% crit | `statsmodels` 5% crit | `urca` p(−3.3670) | `sm` p(−3.3670) |
|---|---|---|---|---|
| 2 | −3.366924 | −3.36685108 | 0.049991 | 0.046150 |
| 3 | −3.783338 | −3.78374380 | 0.127301 | 0.120313 |
| 6 | −4.791978 | −4.79194880 | 0.560486 | 0.551075 |
| 8 | −5.344673 | *(crit only)* | 0.812263 | *(no p-value)* |
| 12 | −6.294929 | *(crit only)* | 0.984520 | *(no p-value)* |

Critical values agree to ~4 × 10⁻⁴, p-values to ~0.005–0.009 — two
different vintages of MacKinnon's response surfaces (urca: 1995;
statsmodels: 1994 for p, 2010 for crit), not a disagreement about the
test.

`[V]` **Coverage limits, which bound what 10.4a can honestly return:**
`statsmodels`' p-value surfaces (`tau_*_smallp`, `tau_*_largep`) are
shape **(6, 3)** and **(6, 4)** → **N ≤ 6**. Its critical-value tables
(`tau_*_2010`) are **(12, 3, 4)** → N ≤ 12, *except* `tau_nc_2010`
which is **(1, 3, 4)** → no-constant is **N = 1 only**. So
`fit`-time behaviour must be: p-value for N ≤ 6, critical values for
N ≤ 12 with a constant or trend, and an honest refusal beyond — which is
precisely this project's established pattern, not a new one.

### 4.4 10.5: statistics agree, everything around them does not

`[V]` Three separate traps, all on the same fixture.

**(a) The output order is reversed.**

```
urca::ca.jo(type="trace", ecdet="none", K=2)   teststat = [ 7.028307, 82.589857]   # r<=1 first
statsmodels coint_johansen(W, 0, 1)            lr1      = [82.589857,  7.028307]   # r=0 first
```

Identical numbers, opposite order. A test written against one reference
and checked against the other will pass on the wrong rows.

**(b) `ecdet="none"` maps to `det_order=0`, not `det_order=-1`.** The
names invite the opposite mapping. Verified by sweeping both:

| `urca` `ecdet` | trace statistics | | `statsmodels` `det_order` | `lr1` |
|---|---|---|---|---|
| `"none"` | 7.028307, 82.589857 | | `-1` | 0.143865, 75.230862 |
| `"const"` | 7.666556, 83.230455 | | `0` | **7.028307, 82.589857** ← matches `"none"` |
| `"trend"` | 7.516541, 83.661089 | | `1` | 7.254090, 83.398337 |

`ca.jo(ecdet="none")` still includes an **unrestricted** constant in the
VAR; `det_order=-1` means literally no deterministic term. Getting this
wrong silently shifts every statistic.

**(c) The critical-value tables genuinely disagree, and there are no
p-values anywhere.** `[V]` 5% trace: `urca` gives (8.18, 17.95);
`statsmodels` gives (15.4943, 3.8415) — and reversing one does not
reconcile them. `urca` uses Osterwald-Lenum (1992); `statsmodels` uses
its own `coint_tables.c_sjt`/`c_sja`, indexed on `n − r`, whose
smallest entry 3.8415 is just the χ²(1) 95% point. Neither reference
reports a p-value for Johansen at all.

`[V]` Table coverage: `c_sja`/`c_sjt` cover **n = 1…12** and
**det_order ∈ {−1, 0, 1}**; `det_order = 2` returns **`[nan, nan, nan]`
silently**, with no error.

**Three consequences for 10.5, each a decision to take before coding:**

1. Which table ships. Osterwald-Lenum (matching `urca`, and therefore
   matching most published applied work) is the defensible default.
2. `pvalue()` must **refuse**, not approximate — Johansen gives critical
   values only. This collides with `pvalue(t::HypothesisTest) = t.pvalue`
   (§1.5), so either the field holds `NaN` with a documented meaning, or
   `HypothesisTest` grows a way to say "critical values only". The
   second is better and is a small, contained change to `abstract.jl`.
   *(MacKinnon-Haug-Michelis (1999) response surfaces would give real
   p-values — a genuine differentiation opportunity, since neither
   reference has them — but those coefficients are in neither installed
   reference, so that is a separate, paper-sourced task, not something
   to assume.)*
3. An out-of-range `det_order` must error. Returning `NaN` the way
   `statsmodels` does is the exact failure mode this project refuses.

**(d) Use the generalized symmetric eigenproblem.** `[V]`
`statsmodels` emits `ComplexWarning: Casting complex values to real
discards the imaginary part` from `vecm.py:717–718` on every call — its
eigenvalues come back complex and are truncated. Julia's
`eigen(S10 S00⁻¹ S01, S11)` on the symmetric generalized problem
returns real eigenvalues **by construction**. Already available from
`LinearAlgebra`; no new dependency. This is a case where doing it
properly is also less code.

### 4.5 10.6: three references, three normalizations

`[V]` Same cointegrated fixture, rank 1, one lag:

```
statsmodels VECM(deterministic="n")   beta = [1, -0.99001921]   alpha = [0.08968542, 1.04069393]
urca::cajorls(jo, r=1)                beta = [1, -0.98338810]
tsDyn::VECM(estim="ML")               coefs[1:3] = 0.04229251, 1.005397, -0.056506
vars::vec2var(jo, r=1)                A1[1,1] = 1.062299
```

All three normalize `beta` so its first element is 1, and still
disagree in the second — because the deterministic specification and
the estimator (ML vs two-step OLS) differ. 10.6 is dual-verifiable, but
only once the deterministic case and estimator are pinned on **both**
sides. Budget for that; do not assume `beta` is comparable out of the
box.

> **Corrected 2026-10-13, on building 10.6.** "Three normalizations" was
> wrong: once the deterministic specification is matched
> (`ecdet="none"` ↔ `"co"` ↔ `include="const"`), `cajorls`, `statsmodels`
> and `tsDyn`'s ML agree on β to `≤ 1e-14`, and `tsDyn`'s 2OLS is a
> different estimator, matched exactly by `fit_vecm(…; method=:twostep)`.
> 10.6 is moderate, not hard. See the charter §5.12 and §5.13 item 21.

---

## 5. Revised dependency graph

Corrections to the roadmap's own `Depends on` column are in **bold**.

| # | Item | Roadmap says | Fresh assessment |
|---|---|---|---|
| **10.0** | **Multi-response OLS + `tsmatrix` + companion matrix** | *(absent)* | **New prerequisite row.** 10.1 cannot be written well without it (§2.1, §2.2, §2.6). A Stage-1-shaped primitive |
| 10.1 | VAR + lag-order selection | 1.4 | **1.4 + 10.0.** Correct that it needs nothing else — no state space, no optimiser |
| 10.2 | Granger causality | 10.1 | 10.1 + `_f_cdf`/`_chisq_ccdf` (both built). **Convention decision required** (§4.2) |
| 10.3 | IRF, FEVD | 10.1 | 10.1 + 10.0's companion matrix. **Also needs a `Forecast`-family decision if fan charts are in scope** (§2.3) |
| 10.4a | Cointegration — Engle-Granger (tau) | *(merged with 10.4b)* | **Split out.** 2.1 (`adf_test` regression machinery) + 1.4 + **new N=2…12 tables**. Does *not* need 10.1 |
| 10.4b | Cointegration — Phillips-Ouliaris (Z) | *(merged with 10.4a)* | **Split out.** 1.4 + `_bartlett_lrv` (built) + **new PO tables**. Does *not* need 10.1 |
| 10.5 | Cointegration — Johansen | 10.1 | **Not 10.1.** Johansen is a reduced-rank regression on `ΔY` against lagged differences and levels — it needs 10.0's multi-response OLS and `LinearAlgebra.eigen`, not a fitted `VARModel`. Plus new tables and the `HypothesisTest` contract decision (§4.4) |
| 10.6 | VECM | 10.5 | 10.5 + 10.0, and `_optimize` **only** if ML is offered alongside two-step OLS |
| 10.7 | Structural VAR | 10.1 | 10.1 + `_optimize` + 10.0 |
| **10.8** | **Multivariate residual diagnostics** (Hosking / Li-McLeod portmanteau, multivariate JB, multivariate ARCH-LM) | *(absent)* | **New row.** 10.1 + `_chisq_ccdf` (built). A fitted VAR is not shippable without it (§2.4) |

**What this means for sequencing.** The roadmap's claim that *"this
entire stage can start as soon as Stage 1 is done"* holds — and in fact
understates it: 10.4a and 10.4b do not even need the VAR. The real
critical path is:

```
10.0  (multi-response OLS, tsmatrix, companion matrix)
  ├── 10.1 ── 10.2
  │      ├── 10.3
  │      ├── 10.7
  │      └── 10.8
  └── 10.5 ── 10.6
10.4a, 10.4b   (independent of everything above except 1.4 / 2.1)
```

Suggested build order, on evidence rather than numbering:

1. **10.0**, as a Stage-1 primitive with its own tests
2. **10.1 + 10.3** together — the two items where both references agree
   exactly (§4.1), so the implementation is checked against consensus
3. **10.8**, because 10.1 is not usable without it
4. **10.2**, which is one convention decision away from done
5. **10.4a / 10.4b**, which are table work and can run in parallel
6. **10.5 → 10.6**, the heaviest, with the most convention traps
7. **10.7** last — the R reference is strong, the Python one thin

---

## 6. Open decisions that block a clean start

These need answers before the first 10.1 handoff, because each one
changes the API rather than the internals:

1. **`tsmatrix` or overloaded `tsvalues`?** (§2.2) — and how variable
   names travel through `irf`/`fevd`/`causality` output.
2. **New `VARForecast` or generalise `Forecast`?** (§2.3) — the latter
   is breaking for nine existing model families.
3. **Granger: F or Wald as the default**, and which p-value convention
   (§4.2).
4. **Johansen critical-value table: Osterwald-Lenum or MacKinnon-Haug-Michelis?**
   The second gives p-values neither reference has — real
   differentiation, but it is paper-sourced work (§4.4).
5. **`HypothesisTest` contract for a test with no p-value** (§1.5, §4.4).
6. **Is bootstrap IRF the default?** Both references default to it
   (§2.7).

## 7. What was *not* checked, and should be

Stated so no one assumes otherwise:

- `[S]`, never `[V]`, for everything inside TSAnalytics — **Julia is not
  installed in this sandbox**. Every package-internal claim here comes
  from reading HEAD, not from running it. The multi-response OLS timing
  argument in §2.1 is an operation count, not a benchmark.
- No SVAR identification scheme beyond the `Amat`/`Bmat` scoring path
  was exercised; `vars::BQ` (Blanchard-Quah) and `vars::SVEC` were
  confirmed present `[V]` but not run.
- Hamilton (1994) ch. 10–11 and Tsay (3rd ed.) ch. 8 are the two of the
  six books with the deepest VAR/cointegration treatment, and **their
  worked examples were not cross-checked against these references in
  this session.** Hamilton's PDF is an OCR'd scan and unusable as a data
  source; Tsay's data is CRSP and cannot be bundled. Any book-example
  reproduction in Stage 10 needs its data re-sourced, exactly as the
  textbook-dataset work concluded earlier.
