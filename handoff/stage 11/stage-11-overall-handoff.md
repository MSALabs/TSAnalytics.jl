# Stage 11 — Advanced state-space applications and nonlinear models
## Overall handoff (the charter)

**Written:** 2026-10-14
**Repository HEAD at writing:** `ad0010e` ("Manual and appendix corrections from the book work; record it all")
**Scope:** Stage 11.0 through 11.4, plus plot recipes
**Read this before any sub-stage handoff.** Every sub-stage document assumes the
conventions, licensing constraints and test policy fixed here.

---

## 0. Tagging convention

Carried forward from Stage 10 unchanged, because it is the only thing that
makes a handoff auditable six weeks later.

| Tag | Meaning |
|---|---|
| **[V]** | **Verified by execution in this sandbox.** A program was run, the numbers below are its output, and the script is named. |
| **[S]** | **Read from source.** A file and line number are given. Not executed. |
| **[U]** | **Unverified / carried forward.** Stated from an earlier stage or from reasoning, not re-checked at this HEAD. Treat as a claim to test, not a fact to build on. |

Anything in this charter without a tag is a *decision*, not a finding, and is
binding regardless of evidence.

---

## 1. What Stage 11 is

The roadmap lists four items. The survey below adds a fifth — **11.0** — because
three of the four need the same piece of machinery that does not exist yet, and
building it three times inside three sub-stages would be the worst possible
outcome.

### Revised item list

| # | Item | Status in roadmap | Change |
|---|---|---|---|
| **11.0** | **Shared primitives: multivariate-observation Kalman filter and smoother, partitioned initialisation, stationary VAR reparameterisation, zero-weight `_ols`** | **absent** | **NEW — must be built first** |
| 11.1 | Unobserved Components / structural time series | present | unchanged |
| 11.2 | Dynamic factor models | present | depends on **11.0**, not on "Stage 8, 10.1" as written |
| 11.3 | Markov-switching AR / regression | present | depends on **11.0** for the `_ols` weight fix only; otherwise standalone |
| 11.4 | Threshold AR (TAR/STAR) | present | unchanged; genuinely independent |

**The dependency column in `development-sequence.md` is wrong in three places**,
and this is the third consecutive stage where that has been true. Corrections:

1. **11.1 does not depend on "Stage 8" as a whole.** It depends on
   `TimeVaryingSSM` (Stage 8.1) and the existing `kalman_smoother` for it. It
   does **not** need `kalman_filter_diffuse` (Stage 8.2), because the reference
   it must match does not use exact diffuse initialisation — see §5.1. This makes
   11.1 **cheaper** than the roadmap implies.
2. **11.2 does not depend on 10.1.** Nothing in a dynamic factor model uses the
   VAR estimator. It uses `companion` (a 20-line helper from Stage 10.0) and a
   VAR *regression* for starting values, which `_mols` provides. The substantive
   dependency the roadmap misses entirely is the **multivariate-observation
   filter**, which does not exist anywhere in the package.
3. **11.3's listed dependency "4.1, 5.1" is right but incomplete.** It also needs
   `_ols` to accept zero weights, which it currently refuses — §5.9, defect D7.

### Difficulty, measured rather than estimated

| # | Item | Difficulty | Dual-verifiable? | Dominant cost |
|---|---|---|---|---|
| 11.0 | Shared primitives | **Moderate** | yes (both references) | the smoother, and getting missing-data row handling right |
| 11.1 | Unobserved Components | **Easy–Moderate** | **yes, to 1.26e-08** [V] | a mapping layer; ~600 reference lines, no new numerics |
| 11.2 | Dynamic factor | **Hard** | yes, after renormalisation | 11.0 plus the identification problem |
| 11.3 | Markov-switching | **Hard** | partially (spec mismatch) | two new recursions, multimodality, no reusable starting point |
| 11.4 | TAR / STAR | **Moderate–High** | **yes, three ways, bit-exact** [V] | the grid search's exact conventions, and LSTAR's start grid |

This ordering is deliberately not the roadmap's. **Build 11.1 first** — it is the
cheapest, it is the one with a clean machine-precision reference agreement, and
it exercises `TimeVaryingSSM` in anger before 11.0 generalises it. Then 11.0,
then 11.2. 11.3 and 11.4 are independent and can go in any order.

---

## 2. Governing principles (inherited, non-negotiable)

These are not new. They are restated because every one of them earned its place
by catching a real error in Stages 6–10.

1. **Run it, don't recall it.** No number enters the repository that was not
   produced by a program in a recorded script. This charter's own first draft
   claimed the two UC references "disagree"; three executions later the
   disagreement turned out to be an initialisation convention worth 1.26e-08.
   The claim would have been written down as fact without the executions.
2. **Reference, never port.** Read a reference's source to understand a correct
   design, then rebuild it natively in Julia. Never transcribe code. §4 makes
   this a licensing requirement, not just a style preference.
3. **The honest-refusal pattern.** Throw `ArgumentError` naming what is missing.
   Never return `NaN`, never clip to a boundary, never silently approximate.
   Stage 11 has two references that violate this (§5.9, defects D1 and D4) and
   the contrast is instructive: R's `StructTS` reports a variance of **exactly
   zero** on `log10(UKgas)` because it applies `pmax(coef * vx, 0)`, so a user
   cannot tell a true boundary solution from a clipped negative one. [V]
4. **Reduction testing.** Prove a general implementation by collapsing it onto an
   already-verified special case. Stage 11 has unusually good reductions
   available, and §8 makes three of them mandatory.
5. **`predict` and `forecast` are exact aliases** (`src/forecast.jl:434`). Every
   new model type follows this. [U]

---

## 3. Reference environment (reproducible)

All four sub-stages were measured in this sandbox. Recorded so the next session
does not have to rediscover it.

| Component | Version | How obtained |
|---|---|---|
| R | 4.3.3 "Angel Food Cake" | `/usr/bin/R`, pre-installed | [V] |
| statsmodels | 0.15.0 | pre-installed for python3.13 | [V] |
| `dfms` | 1.0.1 | `apt-get install -y r-cran-collapse liblapack-dev libblas-dev gfortran`, then clone `github.com/cran/dfms` + `R CMD INSTALL` | [V] |
| `MSwM` | 1.5 | `apt-get install -y r-cran-nlme` (3.1.164), then clone `github.com/cran/MSwM` + `R CMD INSTALL` | [V] |
| `tsDyn` | 11.0.5.2 | **already installed** from the Stage 10 session at `/usr/local/lib/R/site-library/tsDyn` | [V] |
| `aeon` | 1.6.0 | `pip install aeon` — PyPI **is** reachable | [V] |

**CRAN remains unreachable** from this sandbox, as it was throughout Stage 10.
The two working routes are the Ubuntu archive (`r-cran-*`, ~1130 packages) and
CRAN's GitHub mirror (`github.com/cran/<pkg>`). The first `R CMD INSTALL` of
`dfms` fails on missing `-llapack -lblas -lgfortran`; install those first. [V]

### Which reference serves which item

| Item | Primary (transcribable) | Independent check | Notes |
|---|---|---|---|
| 11.1 | statsmodels `UnobservedComponents` | R `stats::StructTS` (base R — GPL, check only) | Agreement **1.26e-08** at matched init [V] |
| 11.2 | statsmodels `DynamicFactor` | `dfms` (GPL-3, check only) | Agreement **1.4e-10** on the likelihood at fixed parameters [V] |
| 11.3 | statsmodels `MarkovAutoregression` | `MSwM` (GPL-2+, check only) | **Different models** — see §5.4; the statsmodels target is independently corroborated by E-views [V] |
| 11.4 | **`aeon.forecasting.machine_learning.SETAR`** (BSD) + a hand-rolled base-R grid search | `tsDyn` (GPL-2+, check only) | SETAR point estimates agree **to 12 printed digits across all three** [V] |

**The roadmap's "weak Python support" for 11.4 is wrong but nearly right.**
`aeon 1.6.0` ships a SETAR forecaster (201 lines) that matches `tsDyn` exactly on
a matched sample, but it fixes the delay at 1, selects no AR order by criterion,
and has no inference, no LSTAR and no `nthresh=2`. statsmodels has **zero**
threshold-AR content — a full-tree grep for `setar|lstar|smooth.transition|
threshold.autoregress` returns no hits. [V] So 11.4 is dual-referenced for the
point estimator and single-referenced (tsDyn only) for everything else.

---

## 4. Licensing — a hard constraint on how references are used

| Package | License | Permitted use |
|---|---|---|
| statsmodels 0.15.0 | **BSD-3** (`dist-info/licenses/LICENSE.txt`; per-file headers say "License: BSD-3" / "Simplified-BSD") [V] | Numbers **may** be transcribed as test expectations. Source may be read and its design followed. |
| `dfms` 1.0.1 | **GPL-3** (`DESCRIPTION`) [V] | Independent numerical check **only**. No source transcription, no table transcription. |
| `MSwM` 1.5 | **GPL (>= 2.0)** (`DESCRIPTION`) [V] | Independent check only. |
| `tsDyn` 11.0.5.2 | **GPL (>= 2)** (`DESCRIPTION`, `COPYING` present) [V] | Independent check only. |
| `aeon` 1.6.0 | BSD-3-Clause **by project policy — the LICENSE file was not read** | Treat as [U]. Apply the no-transcription discipline until someone reads the file. |
| R `stats::StructTS` | base R, **GPL-2 \| GPL-3** | Independent check only. |

**The practical rule, unchanged from Stage 10:** a number obtained by *running* a
GPL program is a measurement of the world and may be recorded as a test
expectation. The program's *source code and internal tables* may not be copied.
Where that line matters in Stage 11:

- **11.4 is the sharp case.** `tsDyn`'s design-matrix builders (`buildXth*`, 100
  lines across 8 functions) and its candidate-grid constructor (`makeTh`, 81
  lines) must be **re-derived from the documented behaviour captured in §5.8 and
  in the 11.4 handoff**, not copied. The behaviour is recorded precisely enough
  to do that.
- **11.3's expected values must come from statsmodels, not MSwM**, which is
  convenient because MSwM is non-reproducible under its own defaults anyway
  (§5.9, defect D6).

---

## 5. Cross-cutting conventions — the traps, with evidence

### 5.1 The references do not disagree. Initialisation does.

This is the single most important finding in the stage, and it inverts the
conclusion a first look produces.

On the Nile local-level model, the two references report:

| | σ²_level | σ²_irregular | reported log-likelihood |
|---|---|---|---|
| R `StructTS(Nile, type="level")` | **1469.147** | **15098.577** | **−643.201** |
| statsmodels `UnobservedComponents(level='local level')` | 1478.812 | 15078.010 | −632.538 |

[V] `s11/structts.R`, `s11/uc_nile.py`

A 0.66% parameter gap and a 10.66 likelihood gap. The obvious reading — two
conventions, pick one — is **wrong on both counts**.

**The likelihood gap is a reporting artefact, not a different objective.**
Evaluating statsmodels' likelihood *at R's parameters* gives **−632.537695**,
which is better than statsmodels' own reported optimum of −632.537761. R's
`loglik` is simply not the log-likelihood (§5.9, defect D1). [V] `s11/uc_diag.py`

**The parameter gap is likelihood flatness, not disagreement.** Four independent
optimisers in statsmodels — L-BFGS with tightened tolerances, Nelder-Mead,
Powell and BFGS — all converge to σ²_level = **1463.547**, σ²_irregular =
**15108.316**, llf = **−632.53768559**, which beats R's point by 9.7e-06 in
log-likelihood while differing from it by **0.38%** in the parameter. Starting
statsmodels *from* R's values moves it away to its own optimum. [V]
`s11/uc_conv.py`

So there are three defensible answers (R/Durbin & Koopman 1469.147, statsmodels
1463.547, statsmodels-default 1478.812) spanning 1% of the parameter and 1e-5 of
the objective.

**The recursions are identical.** With the initialisation matched — R uses
`a0 = y[1] = 1120`, `P0 = 1e6 · var(y)/100 = 2.86379e8`; statsmodels defaults to
`a0 = 0`, `P0 = 1e6` — and the same variances passed to both:

| quantity | max abs difference |
|---|---|
| filtered state, all 100 periods | **1.257e-08** |
| smoothed state, all 100 periods | **9.969e-09** |

[V] `s11/uc_fixed2.R`, `s11/uc_match.py`. The residual 1e-08 is the precision of
R's 7-significant-digit `coef` printing, which is what was fed back in — it is a
floor imposed by the measurement, not a disagreement.

**Three consequences, all binding:**

1. **statsmodels `UnobservedComponents` uses `approximate_diffuse` with variance
   1e6 by default, not exact diffuse.** [V] `s11/uc_init.py`. So **11.1 does not
   need the exact diffuse smoother**, which does not exist in the package and
   would have been the one genuinely hard algorithm in the item. `TimeVaryingSSM`
   with `P0 = 1e6·I` on the diffuse states reproduces the reference.
2. **Mixed initialisation by block is required.** With a stationary component
   present, statsmodels partitions: states 0–6 get `approximate_diffuse`, state 7
   (the AR component) gets `stationary`. [V] same script. This is 11.0's job.
3. **The test target is the recursion, not the optimum.** See §8.

### 5.2 A flat likelihood is a property of the problem, not a bug to fix

§5.1's measurements are not a Nile peculiarity. Every item in this stage has a
weakly identified direction:

| Item | Evidence [V] |
|---|---|
| 11.1 | σ²_level moves 0.38% for a 9.7e-06 change in llf |
| 11.2 | `dfms` and `DynamicFactorMQ` are **over-parameterised by r(r+1)/2** — the extra parameters are exactly flat directions |
| 11.3 | 35% of random starts land on a worse mode; the `switching_variance=True` default fit is itself a local optimum **1.36 llf** worse than the best found |
| 11.4 | LSTAR's γ **pins at the start-grid bound of 100** on two of three fixtures, because the profile in γ is nearly flat |

**The policy this forces:** a test that asserts a fitted parameter to 1e-6
against a reference is asserting the reference optimiser's stopping rule, not the
model. §8 fixes the tolerances per item from measured spreads.

### 5.3 No canonical identification exists for a dynamic factor model

Neither reference normalises the factor rotation.

- statsmodels `DynamicFactor` imposes **only** `Var(η) = Q = I`. The loadings are
  fully free, so the fitted Λ is an arbitrary orthogonal rotation of the truth.
  Measured on a 2-factor fixture: `Λ'Λ = [[1.48293016, 0.27623486], [0.27623486,
  1.35874864]]` — not the identity, not even diagonal. [V]
- `dfms` normalises **nothing** (`rQ` defaults to `"none"`, which means
  *unrestricted*, not *absent* — a genuinely misleading name). Fitted
  `Q = 1.97511745803 ≠ 1`. [V]
- `DynamicFactorMQ` estimates the factor-innovation Cholesky, so for r=1 it has
  **12 parameters where `DynamicFactor` has 11**. [V]

Rotation and scale are pinned in both GPL references only *implicitly*, by the
PCA starting values.

**Decision: TSAnalytics.jl will normalise explicitly.** The 11.2 handoff fixes
which normalisation. The point here is that copying either reference's
non-normalisation would make every parameter-level test require a Procrustes
rotation, which is exactly what the reference comparison in 11.2 had to do:
align via `L = chol(Q)`, `Λ̃ = ΛL`, then SVD of `Λ̃'Λ_ref = USV'` and `G = UV'`.
**The recovered rotation on the 2-factor fixture had `det = −1`** — an improper
rotation — so any alignment helper must allow reflections. [V]

### 5.4 The Markov-switching references implement different models

Not a convention difference. A different likelihood.

- statsmodels `MarkovAutoregression` is Hamilton's **mean-adjusted** form:
  `y_t = a_{S_t} + Σ φ_j (y_{t−j} − a_{S_{t−j}}) + ε_t`, with a state space of
  size `k^(p+1)`. [S] `markov_autoregression.py:169-230`, docstring `:80-90`
- `MSwM` builds lags as **plain levels regressors**: `y_t = a_{S_t} + Σ φ_j
  y_{t−j} + ε_t`, state space `k`. [S] `MSwM/R/2MSM.r:233-243`

**MSwM cannot express the Hamilton model at all.** A naive comparison gives a
13.2 log-likelihood gap — measured by re-implementing MSwM's filter faithfully
and evaluating it at statsmodels' parameters: **−194.43047679** against
statsmodels' **−181.26339428**. [V]

On a specification both *can* express (`MarkovRegression` with four levels lags),
they agree: coefficients to ≤2e-02 relative, σ to **1.5e-05**, smoothed regime
probabilities to **corr 0.99980722**, and the residual 0.0701 of log-likelihood
is **entirely** the initial-distribution convention — replacing MSwM's estimated
`iniProb = [0, 1]` with the steady state costs exactly 0.071892. [V]

### 5.5 Boundary solutions are where the references genuinely part company

On `log10(UKgas)` with a full basic structural model:

| | σ²_level | σ²_slope/trend | σ²_seasonal | σ²_irregular |
|---|---|---|---|---|
| R `StructTS(type="BSM")` | **0.000000e+00** | 1.733003e-05 | 7.136943e-04 | 3.677978e-04 |
| statsmodels, local linear trend + seasonal(4) | 2e-10 | 1.503e-06 | 6.245776e-04 | 3.433252e-04 |

[V] `s11/bsm.py`

Both drive the level variance to zero, but R reports **exactly** zero because it
clips (`coef <- pmax(coef * vx, 0)`, [S] deparsed `StructTS` line 118) while
statsmodels approaches from above. The trend variance then differs by a factor of
**11.5** and the seasonal by 14%, because once one variance is at a boundary the
rest are only weakly identified.

**Policy: boundary fixtures are qualitative tests only.** Assert that the level
variance is below some small bound, never that it equals a reference figure. The
Nile local-level case, where no parameter is at a boundary, is the quantitative
fixture.

### 5.6 The univariate-observation boundary is the real architectural line

Every Kalman routine in the package today takes `y::AbstractVector` and computes
a **scalar** `F_t`, with the gain as a division:

```
F_t = Z_t P_t Z_t' + H_t          # scalar
K_t = T_t P_t Z_t' / F_t          # division, not a solve
```

[S] `src/statespace/timevaryingssm.jl:139` and its docstring

This holds for `kalman_filter(::TimeVaryingSSM, …)`, `kalman_smoother(::TimeVaryingSSM, …)`
and `kalman_filter_diffuse`. **There is no multivariate-observation filter in the
package, and no diffuse smoother of any kind** — confirmed by grep. [V]

11.1 lives entirely inside the univariate boundary. **11.2 does not**, and that
is the whole reason 11.0 exists.

### 5.7 statsmodels' own standardisation is inconsistent, and it shifts the likelihood

`DynamicFactor` does **not** standardise; `DynamicFactorMQ` does
(`standardize=True` by default) — and its denominator **depends on the input
type**, because `endog.std(axis=0)` dispatches to pandas (ddof=1) or numpy
(ddof=0). Measured on the same data: numpy `[1.41108647, 1.16842665, 0.95231134,
0.9932577, 1.20123961]` versus DataFrame `[1.41462747, 1.17135871, 0.95470109,
0.99575019, 1.20425402]`, which shifts the likelihood by **2.506**. [V]
[S] `dynamic_factor_mq.py:1385`

`dfms` always standardises with ddof=1 (`fscale`, verified against `base::scale`
to 1.33e-15). [V]

`DynamicFactor` is also **not scale-invariant**: llf(raw) = −1180.13574349 versus
llf(standardised) = −1049.53340809 on the same fixture. [V]

**Decision: TSAnalytics.jl does not standardise inside the fitter.** It is the
caller's transform, it belongs in `src/transforms.jl` where the package already
keeps such things, and burying it makes the likelihood un-comparable.

### 5.8 `tsDyn` divides by the full series length, not the regression rows

For `log10(lynx)`: n = 114, regression rows = 112, and yet

```
mse    = SSR / 114
AIC    = 114·log(mse) + 2k
BIC    = 114·log(mse) + log(114)·k
resvar = SSR / (114 − p)
```

[V] Confirmed by reproducing the standard errors four ways; only `SSR/(114−6)`
gives `const.L`'s reported SE of **0.14330661379**. The three plausible
alternatives give 0.14397471351, 0.14465224531 and 0.14533943325.

**`_ols` uses `SSR/(n−k)` on the actual design and will not reproduce these.**
The 11.4 handoff specifies the SEs as a separate computation.

### 5.9 Reference defects found (running list)

Every one found by execution in this sandbox, not from a bug tracker.

| # | Reference | Defect | Evidence |
|---|---|---|---|
| **D1** | R `StructTS` | **`loglik` is not the log-likelihood**, and `loglik0` cannot be one either. `loglik0 − loglik = n·log(2π) + 0.5·n_obs·log(2π)` exactly — reproduced as **272.924744** with three NAs injected (n=100, n_obs=97). A `+n·log(2π)` term has the wrong sign for any Gaussian normalising constant, so at most one of the two reported figures can be a log-likelihood, and the documentation does not say which. The reported value is also off by **10.663** from the true likelihood at its own parameters. The cause is visible in the source: `loglik <- -length(y) * res$value - 0.5 * sum(!is.na(y)) * log(2*pi)` scales by `length(y)` where the concentrated form needs the number of observations the filter actually used. | [V] `s11/structts_na.R`, `s11/uc_diag.py`; [S] deparsed `StructTS` lines 122–124 |
| **D2** | R `StructTS` | **Initialises the state at the first observation** (`a <- xm` where `xm = x[1]`), making the first innovation **identically zero** — measured `resid[1] = 0.000e+00` — and discarding one observation's worth of information. | [V] `s11/uc_fixed2.R`; [S] deparsed line 12 |
| **D3** | R `StructTS` | **Clips variances at zero** (`coef <- pmax(coef * vx, 0)`), so a reported `0.000000e+00` is indistinguishable from a clipped negative value. A direct violation of the honest-refusal principle. | [V] §5.5; [S] deparsed line 118 |
| **D4** | statsmodels `UnobservedComponents` | **`level=True` silently gives a *deterministic* level** and adds an irregular with only a `SpecificationWarning`. The string shortcuts are mutually inconsistent: `'local level'` carries an irregular, `'rwalk'` does **not**, and `level=True, stochastic_level=True` drops the irregular. | [V] `s11/uc_trap.py` |
| **D5** | `dfms` | **Silently skips a period's likelihood contribution** when `det(F_t) ≤ 0` (`if(detS > 0) loglik += …`). No warning, no error. | [S] `src/KalmanFiltering.cpp` |
| **D6** | `MSwM` | **Non-reproducible under its own defaults.** Three runs with `set.seed(101)` and `parallelization=TRUE` gave llf = **−183.58332 / −183.66895 / −183.66916** — three different modes from one seed, because the `runif` restarts run inside `parLapply` with no RNG stream control. Only `parallelization=FALSE` is reproducible. | [V] `s11/ms/nondet.R` |
| **D7** | *TSAnalytics.jl itself* | **`_ols` rejects zero weights** at `src/unitroot.jl:58` and `:172` (`all(>(0), weights)`). Smoothed regime probabilities hit exact zero routinely — MSwM's own `iniProb` converged to `[0, 1]` on the canonical fixture, and statsmodels clamps at 1e-20. Any EM M-step that feeds smoothed probabilities straight in will throw. **This is ours to fix, and it is cross-cutting.** | [V] read at HEAD `ad0010e` |
| **D8** | `MSwM` | **`logLikel` stores the *negative* log-likelihood** (`loglik = -sum(log(margLik))`). Reading it as a log-likelihood flips the sign of every comparison. | [S] `2MSM.r:989` |
| **D9** | statsmodels `MarkovRegression` | **Runs its filter at order 1 even when `order == 0`**, because `_resid` does `np.repeat(..., k_regimes, axis=1)` and the filter then infers `order = cll.ndim − 2 = 1`. Invisible under steady-state initialisation (`Pπ = π`) but not under `initialize_known`: measured llf **−180.14328657** against a hand-computed `P²π` value of −180.14328933, where `P¹π` gives −180.11424133. Undocumented. | [V] `s11/ms/probe2.py` |
| **D10** | statsmodels `MarkovAutoregression` | **`.fit()` is broken at `order=0`** — `LinAlgError: SVD did not converge` from `_em_exog`, or `IndexError` with `em_iter=0` — even though `loglike` at that order is exactly correct (it reproduces `MarkovRegression`'s llf of −191.28811083). | [V] same |
| **D11** | statsmodels `MarkovSwitching` | **A fully non-switching specification is silently unidentified.** `switching_trend=False` with `switching_ar=False` and `switching_variance=False` returns the transition probabilities **untouched from their start values** (`p[0->0] = p[1->0] = 0.5`) with no warning. | [V] `s11/ms/probe2.py` |
| **D12** | `tsDyn` | **`isLinear.lstar` is broken in 11.0.5.2** — errors with `'.' in formula and no 'data' argument`. | [V] `s11/tar/f17_extra.R` |
| **D13** | `tsDyn` | **`vcov.setar` zero-pads the threshold** row and column, so there is no threshold standard error at all; and **`k` counts the threshold only when it was searched**, so AIC is not comparable between a searched and a supplied threshold (same SSR, AIC −358.374 versus −360.374). | [V] `s11/tar/f1_setar_lynx.R`, `f7_exactth.R` |
| **D14** | `tsDyn` | **`criterion="pooled-AIC"` is not implemented for `nthresh=2`** — it warns and silently falls back to SSR. | [V] trace output |
| **D15** | `tsDyn` | **`KapShinTest` and `BBCTest` carry the author's own disclaimer**: *"should be considered as experimental. No consistency checks with original paper could be made."* They cannot serve as references. | [V] run output |
| **D16** | `tsDyn` | **`logLik` is not defined for class `setar`** — it errors. Only `AIC`/`BIC`/`deviance`/`mse` exist, and all three use the §5.8 denominator. | [V] |

**D1, D6 and D7 are the three that change what gets built.** D1 means no R
likelihood may be compared to ours without reconstruction; D6 means MSwM cannot
supply expected values even in principle; D7 is a fix we owe ourselves before
11.3 can start.

### 5.10 Smaller conventions, recorded so nobody rediscovers them

| Area | Convention | Evidence |
|---|---|---|
| statsmodels transition matrix | **Left-stochastic**: `P[i,j] = Pr[S_t=i | S_{t−1}=j]`, **columns** sum to 1. MSwM agrees. | [V] `colsums = [1,1]` |
| Transition parameterisation | Multinomial logit with the **last** regime as implicit reference, applied column-wise; **k(k−1)** free parameters, confirmed for k = 2, 3, 4 | [V] round-trip exact |
| `initial_probabilities` | The distribution at **t = −order**, not t = 0 — the init loop applies `P` `order` times | [S] `markov_switching.py:200-205` |
| Label canonicalisation (k=2) | In statsmodels' parameterisation the swap is **`(a,b) → (1−b, 1−a)`**, *not* `(a,b) → (b,a)`. A label-swapped fit differed by **1.522** in parameters at an llf difference of **1.86e-08**. | [V] `s11/ms/canon.py` |
| `tsDyn` regime boundary | Low regime is **`z ≤ γ`** (inclusive). Exactly one lynx observation sits on the threshold; `sum(z ≤ th) = 78 = n_low`, `sum(z < th) = 77`. | [V] |
| `tsDyn` `thDelay` | `thDelay = d` means the threshold variable is **`y_{t−(d+1)}`** — zero-indexed on the lag. `thDelay = 0 → y_{t−1}`. | [V] |
| `tsDyn` candidate grid | Built from **`sort(z)` including ties**, not unique values, with `ceiling(trim·ng)`/`floor(ng(1−trim))` bounds and a `seq(length.out = up−down)` that yields **fractional indices R silently truncates**. Shown to differ from the `unique` variant in general (21 versus 31 candidates on a tied series). | [V] `s11/tar/f12_grid_exact.R` |
| LSTAR parameterisation | **Additive, not a convex combination**: `ŷ = X·φ_L + (X·φ_H)·G(z;γ,th)`. The printed "High regime" coefficients are **increments**. Using the convex form gives SSR 290.77 against the correct 4.337643232152 — wrong by 67×. | [V] `s11/tar/f9_lstar_param.R` |
| LSTAR transition | `G = 1/(1 + exp(−γ(z − th)))`, with γ multiplying the **raw** `(z − th)` and **no standardisation by sd(z)** — unlike the common Teräsvirta convention `γ/σ_z`. | [V] |
| `tsDyn` trim defaults | **Three different ones in one package**: `setar`/`selectSETAR` 0.15, `setarTest` 0.1, `lstar`'s start grid 0.1. | [S] signatures |
| `nthresh=2` | **Sequential/conditional**, not a joint 2-D grid: find γ₁, then γ₂ given γ₁, then re-search γ₁ given γ₂, with `max.iter = 2` alternations. | [V] trace |
| `DynamicFactor` factor array | `res.factors.smoothed` is **(k_factors, nobs)** with numpy input — transposed relative to the data. | [V] |
| statsmodels missing data | `Statespace.missing` is **(k_endog, nobs)**; partial observations are handled by BLAS-copying `y_t`, `Z_t`, `H_t` down to the observed rows. | [S] `_representation.pyx.in:621,653,663` |

---

## 6. Shared architecture decisions

### 6.1 `TimeVaryingSSM` is generalised, not replaced

The existing type already holds `T, Z, R, Q, H` as `Vector{Matrix{S}}` with
length-1 broadcast, which is the right representation. 11.0 adds **methods** for a
multivariate observation (`y::AbstractMatrix`), leaving every existing
`y::AbstractVector` method untouched and recompiling to exactly the same code.

This mirrors the decision already recorded in `timevaryingssm.jl`'s own
docstring, where `TimeVaryingSSM` was made a separate type from `GaussianSSM`
rather than a generalisation of it, precisely so `fit_arma`'s hot loop was
unaffected. [S] The same reasoning applies one level up, and the same proof is
required: a benchmark showing the univariate path is unchanged.

### 6.2 The multivariate filter uses the sequential (univariate) treatment

Durbin & Koopman §6.4. Each of the N series is brought in as its own scalar
update within period t, rather than forming an N×N `F_t` and factorising it.

**Four measured reasons, not a preference:**

1. **It reuses the scalar-gain algebra the package already has and has tested** —
   it is a loop around known-correct code rather than new linear algebra.
2. **Cost is O(N r²) per period instead of O(N³ + N² r + N r²).** [V] derived from
   the reference implementations' structure.
3. **Missing data is handled by skipping a series in the loop** — no submatrix
   machinery, where statsmodels needs `select_missing` with three buffers and
   `dfms` needs Armadillo submatrix extraction.
4. **It is measurably more robust.** At σ² = 1e-12 the conventional matrix path
   returns **−1.39e14** where the sequential path returns **−696.1687714342193**,
   which is the correct H→0 limit (the exact-zero case gives
   −696.1687714342186). **Neither reference raises an error in the broken case.**
   At ordinary parameters the two paths agree to **3.183e-12**. [V]
   `s11/dfm/singular.py`

So statsmodels' numbers remain valid expectations either way, and the cheaper,
safer algorithm is also the one that fits the existing code.

### 6.3 Initialisation is partitioned, and `approximate_diffuse` is the 11.1 default

11.0 provides an initialisation descriptor that assigns each state block one of
`:known`, `:stationary`, `:approximate_diffuse(κ)` or `:exact_diffuse`, because
§5.1 showed statsmodels mixes diffuse and stationary blocks in a single model and
11.1 cannot reproduce it otherwise.

`:approximate_diffuse` with **κ = 1e6** is 11.1's default, matching the reference.
The existing `kalman_filter_diffuse` (exact) stays available and unchanged; 11.1
does not use it, and the 11.1 handoff records why in the user-facing docs so the
choice is not mistaken for an oversight.

### 6.4 New result types, not generalised old ones

Stage 10 decided that a `VARForecast` was better than a generalised `Forecast`,
and the reasoning holds here. Stage 11 adds:

| Type | Item | Why not reuse |
|---|---|---|
| `UCModel`, `UCDecomposition` | 11.1 | `StateDecomposition` (from the plot-recipes work) is close, and **should** be reused for the *plot*; but the fitted model carries variance parameters and per-component smoothed standard errors that an ETS state decomposition has no slot for |
| `DynamicFactorModel` | 11.2 | multivariate, with a normalisation record that must travel with the fit |
| `MarkovSwitchingModel`, `RegimeProbabilities` | 11.3 | filtered/smoothed/predicted probability matrices plus expected durations |
| `SETARModel`, `LSTARModel`, `SETARSelection`, `ThresholdTest` | 11.4 | `ThresholdTest` is a `HypothesisTest` with bootstrap critical values — the "critical values only" shape Stage 10 §6.4 already had to accommodate |

### 6.5 No new package dependency

Unchanged from Stage 10. `LinearAlgebra`, `Statistics`, `Printf`, `Random` and
`RecipesBase` (soft) are what exists, and that is what Stage 11 uses.
`LinearAlgebra.eigen(A, B)` covers the generalised symmetric problem 11.2's PCA
start values want; `cholesky` covers the rest. **No `Distributions.jl`** — the
CDFs in `src/diagnostics.jl` plus `_f_ccdf` from 10.2 cover every p-value in the
stage.

### 6.6 Shared fixtures

Nine fixtures, all with recorded checksums, all bundled. Listed once here so the
sub-stage handoffs can refer to them by name.

| Fixture | Shape | Item | Provenance |
|---|---|---|---|
| `nile.csv` | 100×1 | 11.1 | `statsmodels.datasets.nile`; the Durbin & Koopman canonical case |
| `ukgas.csv` | 108×1 | 11.1 | `log10(datasets::UKgas)`; the boundary case |
| `fixtureA.csv` | 200×5 | 11.2 | simulated, seed 20260214, r=1; `sum = 135.8908272840041` |
| `fixtureB.csv` | 250×6 | 11.2 | simulated, seed 20260215, r=2; `sum = 76.35178397751577` |
| `fixtureA_miss.csv` | 200×5 | 11.2 | 12 blanked cells including one all-missing row |
| `hamilton_rgnp.csv` | 135×1 | 11.3 | **extracted from statsmodels' own test file** (`test_markov_autoregression.py:21`) — no network needed; `sum = 100.5207128600` |
| `fixture_lynx.csv` | 114×1 | 11.4 | `log10(datasets::lynx)`; sha256 `c8e279fb…` |
| `fixture_setar_sim.csv` | 1000×1 | 11.4 | simulated SETAR, known DGP; sha256 `96f87cee…`, round-trips bit-exactly |
| `sunspot_hansen.csv` | 289×1 | 11.4 | `(sqrt(sunspot.year+1)−1)*2`, Hansen's own transformation |

**Simulated fixtures travel as CSV, never as a seed.** R's and NumPy's RNG
streams are not portable, and the 11.4 agent hit this directly. The series is the
fixture; the generator script is documentation.

---

## 7. Performance policy

Accuracy first, and then Julia should be visibly worth using. Three places in
this stage where it can be, with the reference cost measured so the claim is
checkable.

### 7.1 The threshold grid search — the clearest win

`tsDyn`'s default `pooled-AIC` search over threshold × delay × m_L × m_H does
naive refits. Measured on **114 observations**:

| spec | OLS fits | wall clock (R) |
|---|---|---|
| m=2, thDelay 0:1 | 592 | **1.505 s** |
| m=3, thDelay 0:2 | 1,971 | **6.871 s** |
| m=5, thDelay 0:4 | 8,875 | **34.888 s** |

[V] `s11/tar/f13_cost.R`

Every one of those is a full factorisation of a matrix that differs from the last
by a handful of rows. **Sort by the threshold variable once, then update `X'X`
and `X'y` incrementally as the split point advances**: the inner loop becomes
O(k²) per candidate instead of O(n k²) with a factorisation. A 10–100× speedup is
available and should be measured and recorded, not asserted.

### 7.2 The filter's inner loop

Mandatory, as in Stage 6:

- `@views` on every slice; no allocation inside the time loop
- `mul!` accumulation rather than `*` into temporaries
- Pre-allocated work buffers **typed on the parameter eltype**, not
  `Vector{Float64}` — otherwise ForwardDiff cannot flow through, which is the
  mistake statsmodels works around by instantiating its Cython kernel four times
  (including `complex128` purely for complex-step derivatives, [S]
  `markov_switching.py:1130-1177`). Julia's real AD is cleaner, but only if the
  buffers cooperate.
- `@code_warntype` clean on every fit entry point
- `@allocated` zero inside the loop body, asserted in a test

### 7.3 Threading over independent fits

`Threads.@threads` over the restart search in 11.3 (statsmodels' `search_reps`)
and over the threshold grid in 11.4, both of which are embarrassingly parallel
and both of which are the dominant cost. `tsDyn` has an `hpc="foreach"` path;
ours should be the default, with a measured speedup recorded.

### 7.4 What not to optimise

The likelihood is evaluated thousands of times; the smoother once. Optimise the
filter, write the smoother for clarity. And do not pre-emptively avoid the
`k^(p+1)` state expansion in 11.3 — for the canonical fixture k=2, p=4 that is 32
states, which is nothing.

---

## 8. Test policy

**Accuracy first** means the tests have to be honest about what is actually
determined. §5.2 showed that asserting a fitted parameter tightly against a
reference asserts the reference's stopping rule. So the policy changes shape from
Stage 10.

### 8.1 The three-tier assertion rule

| Tier | What | Tolerance | Why |
|---|---|---|---|
| **1. Recursions at fixed parameters** | log-likelihood, innovations, filtered and smoothed states, given a parameter vector | **1e-10 or tighter** | This is what is actually determined. 11.1 achieved **1.26e-08** against R limited only by R's print precision; 11.2 achieved **1.4e-10** against `dfms` | 
| **2. Fitted optima** | the parameter vector a fitter converges to | **per-item, from measured spread** — 11.1: 1e-2 relative; 11.2: 5e-3 relative after alignment; 11.3: 1e-4 absolute after label canonicalisation; 11.4: exact for SETAR, 1e-4 for LSTAR coefficients and ~1e-5 for its SEs | Measured, not guessed. See each sub-stage handoff for the sweep that fixed the number |
| **3. Qualitative** | boundary cases, sign patterns, regime counts, ordering | assertion of the property, never a figure | §5.5 — at a boundary the references diverge by 11× and both are "right" |

**Tier 1 is where the test suite's weight goes.** A fixed-parameter likelihood
test is reproducible to machine precision, catches real errors, and does not
break when an optimiser is tuned. A fitted-optimum test is a regression test on
our own optimiser, which is worth having but is a different thing and should be
labelled as such in the testset name.

### 8.2 Mandatory reductions

Three, each proving a general implementation against something already verified:

1. **The multivariate filter at N=1 must reproduce the existing univariate filter
   bit-for-bit** on the same system. Not "to 1e-12" — the sequential treatment at
   N=1 *is* the scalar recursion, so any difference is a bug.
2. **`MarkovAutoregression` at `order=0` must reproduce `MarkovRegression`.**
   statsmodels' own likelihoods agree *exactly* here (−191.28811083), even though
   its `.fit()` is broken at that order (defect D10) — so this reduction tests us
   against a reference the reference itself cannot run.
3. **LSTAR as γ → ∞ must approach SETAR** at the same threshold. The transition
   function becomes an indicator, and the measured `G` range on the lynx fixture
   already reaches `[3.4e-09, 0.9964]`, so this is testable rather than notional.

Additionally: **SETAR with a supplied threshold must reproduce the hand-rolled
grid search bit-exactly**, which it already does in R (threshold difference 0,
coefficient max-abs-difference 0, SSR difference 0 [V]).

### 8.3 Label canonicalisation is part of the API, not the test

11.2 and 11.3 both have unidentified orderings (factor rotation, regime labels).
**The fitted object must expose a canonical form**, because otherwise every user
comparing two fits hits the same problem the tests do. 11.3's canonicalisation
rule is given in §5.10 and is *not* the obvious one.

### 8.4 Coverage targets

Carried forward from Stage 10's standard. Per sub-stage: **≥ 15 testsets, ≥ 250
assertions**, with every reference figure in the handoff appearing in at least
one. The specific inventories are in the sub-stage documents.

### 8.5 Test the defects

Each of D1–D16 that affects a user-visible choice gets a test asserting **our**
behaviour, with a comment naming the reference defect. This is what Stage 10 did
with the `1 − CDF` underflow and it is why that trap did not come back.

---

## 9. What NOT to do

1. **Do not build the exact diffuse smoother for 11.1.** The reference uses
   approximate diffuse with κ = 1e6 (§5.1). Building the exact version would be
   the hardest thing in the item, and it would make agreement *worse*, not better.
2. **Do not form an N×N `F_t` and invert it.** §6.2. `dfms` forms a full inverse
   and statsmodels offers seven inversion methods; the sequential treatment needs
   none of them.
3. **Do not copy either reference's non-identification in 11.2.** Normalise
   explicitly (§5.3), or every parameter test needs a Procrustes rotation.
4. **Do not compare R's `StructTS$loglik` to anything.** It is not a
   log-likelihood (D1).
5. **Do not use MSwM numbers as expectations.** Different model (§5.4) and
   non-reproducible by default (D6).
6. **Do not transcribe `tsDyn`'s grid or design-matrix code.** GPL (§4). The
   behaviour is documented precisely enough in §5.10 to re-derive.
7. **Do not implement `star`/`addRegime`/`testRegime`, `KapShinTest` or
   `BBCTest`.** The first is 420+ lines whose γ pins at the bound even on the
   canonical example; the latter two carry the author's own disclaimer that they
   were never checked against the papers (D15), so they cannot be verified at all.
8. **Do not standardise inside a fitter.** §5.7.
9. **Do not let a default produce an unidentified model silently.** statsmodels
   does this three times (D4, D11) and it is exactly the honest-refusal case:
   throw, naming which component would have to vary.
10. **Do not assert a bootstrap critical value or a seed-dependent p-value.**
    `setarTest`'s defaults (`nboot = 10`) and `delta.test`'s (`B = 49`) are far
    too small to be stable; the SSRs and F statistics are the portable parts.

---

## 10. On finishing each item

Unchanged, because it works:

1. Every figure in the sub-stage handoff appears in a test.
2. The testset count and assertion count are recorded in
   `development-sequence.md`, along with what was dual-verified and what was not.
3. Anything that could not be verified is written down as such, in the docstring
   the user reads — not just in the handoff.
4. The reference-defect list in §5.9 gets appended to, never rewritten.
5. A benchmark, run and recorded, for anything §7 claimed.

---

## Appendix A: outstanding items inherited from earlier stages

Not Stage 11's job, but they are open and this is where the next session will
look.

| Item | Status |
|---|---|
| `development-sequence.md:1909` records "47 `@example` blocks" across 19 chapters; the diff of `127e04c` contains **41 splits across 18 introduction chapters plus 2 manual pages** | Bookkeeping; the total is defensible, the attribution to one commit is not |
| `development-sequence.md:713` still opens Tier 2.4 with "**Three of six items built**" while line 880 of the same section ends "Tier 2.4 is complete, and so is Tier 2." | Stale header |
| The Durbin & Koopman and Ladiray & Quenneville citations in the state-space Part and the X-13 chapter rest on training memory, not editions in hand | **The one place in 52 chapters where the verification discipline is not satisfied.** Stage 11 leans on Durbin & Koopman §6.4 heavily (§6.2), which raises the stakes |
| `SeasonalAdjustment.jl`'s own `development-sequence.md` rewrite never landed | Downstream package |
| The 14,825-passing test figure is the commit's claim | Julia is **not installed** in this sandbox; it could not be re-run here, and neither could anything in this charter be checked against the actual package behaviour |

## Appendix B: scripts behind every [V] in this document

All under `/tmp/.../scratchpad/s11/`:

`structts.R`, `structts_src.R`, `structts_llik.R`, `structts_na.R`,
`uc_nile.py`, `uc_init.py`, `uc_trap.py`, `uc_diag.py`, `uc_conv.py`,
`uc_fixed2.R`, `uc_match.py`, `bsm.py`, `chk_dfm.py` — 11.0/11.1 and the
spot-checks.

`dfm/` — 19 scripts, fixtures A/B and the missing-data variant, `dfms` state-space
exports. `ms/` — 17 scripts, the Hamilton fixture, filtered/smoothed probability
arrays, MSwM comparisons. `tar/` — 18 scripts including `f2_handroll.R`, the
independent base-R SETAR that agrees bit-exactly.
