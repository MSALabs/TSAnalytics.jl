# Stage 11.2 — Dynamic factor models

**Prerequisite reading:** `stage-11-overall-handoff.md` and
`stage-11.0-primitives-handoff.md` **in full**. This item cannot start until
11.0 §2, §3 and §6 are built and passing their reductions.

**Difficulty: Hard — but the cost is concentrated in 11.0, not here.** Once the
multivariate filter and smoother exist, the model itself is a mapping layer
comparable to 11.1's. What makes the item hard is the identification problem
(§3), which no amount of 11.0 can remove.

**Roadmap correction.** `development-sequence.md` lists 11.2's dependency as
"Stage 8, 10.1 (multivariate)". **It does not depend on 10.1.** Nothing in a
dynamic factor model uses the VAR estimator. It uses `companion` and a VAR
*regression* for starting values. The dependency the roadmap misses is the
multivariate-observation filter, which is 11.0.

---

## 1. The model

`N` observed series driven by `r ≪ N` common factors:

```
y_t = Λ f_t + ε_t          ε_t ~ N(0, H),  H diagonal
f_t = A₁ f_{t-1} + … + A_p f_{t-p} + η_t        η_t ~ N(0, Q)
```

State-space form: states are the stacked factor lags (`companion` from Stage
10.0), `Z = [Λ 0 … 0]`, `H` the idiosyncratic covariance.

### Scope, fixed here

| In | Out |
|---|---|
| `r ≥ 1` factors, factor VAR order `p ≥ 1` | mixed-frequency (monthly/quarterly) construction |
| diagonal idiosyncratic covariance | AR(≥1) idiosyncratic errors |
| missing data | the news/vintage decomposition |
| direct maximum likelihood | EM |

**Mixed frequency is out of scope.** `DynamicFactorMQ`'s two constructors spend
**210 + 171 lines** purely on that bookkeeping [S], for a capability nobody has
asked for.

**EM is out of scope for the first pass**, and §5.3 gives the measured reason.

**AR idiosyncratic errors are out of scope but cheap to add later** — both
references default to white noise (`error_order=0`, `dfms`' `idio.ar1=FALSE`)
[V], so the default path is the verified one.

---

## 2. Reference sizes, measured

| | lines | core | boilerplate |
|---|---|---|---|
| statsmodels `dynamic_factor.py` | 1334 | **725 code lines (74.1%)** | 253 |
| statsmodels `dynamic_factor_mq.py` | 4441 | 1588 | 691 |
| `dfms` (R + C++) | 3926 R + 619 C++ | — | — |

[V] AST breakdown, `s11/dfm/ast_breakdown.py` and `split.py`

The anchor worth remembering: **`dfms`' entire multivariate Kalman filter plus
smoother is 304 lines of Armadillo C++** (`SKF` 106, `FIS` 65, `SKFS` 133), and
its basic EM step is **35 lines of R**. That is 11.0's budget, and it is why the
filter belongs there rather than here.

What remains for 11.2 itself, from `DynamicFactor`'s core:

| piece | reference lines |
|---|---|
| `start_params` | 129 |
| `__init__` | 106 |
| `transform_params` / `untransform_params` | 98 / 91 → **11.0 §6 covers most of this** |
| `param_names` / `state_names` | 58 / 21 |
| `update` (parameters → system matrices) | **44 code lines** |
| the `_initialize_*` helpers | ~130 across eight functions |

**`update` is 44 lines.** That is the whole mapping from a parameter vector onto
`(Z, T, R, Q, H)` once the pieces exist, and it is the clearest signal that this
item's difficulty does not live in the model.

---

## 3. Identification — the decision this item turns on

### 3.1 Neither reference normalises

Charter §5.3, measured [V]:

| reference | restriction | evidence on a 2-factor fixture |
|---|---|---|
| statsmodels `DynamicFactor` | **only `Q = I`** | `Λ'Λ = [[1.48293016, 0.27623486], [0.27623486, 1.35874864]]` — not I, not diagonal |
| `DynamicFactorMQ` | **nothing** — estimates the factor-innovation Cholesky | r=1 gives **12 parameters where `DynamicFactor` has 11**; `state_cov = [[2.14725]]` |
| `dfms` | **nothing** — `rQ` defaults to `"none"`, meaning *unrestricted* | `Q = [[1.717335061017, 0.117645886309], [0.117645886309, 1.207762844832]]` |

`dfms` and `DynamicFactorMQ` are **over-parameterised by r(r+1)/2**, and those
extra parameters are exactly flat directions. Rotation and scale are pinned only
implicitly, by the PCA starting values. `dfms`' `pos.corr=TRUE` merely flips
eigenvector signs so the principal components covary positively with
`rowMeans(X)` [S] `DFM.R:462-463` — a sign convention, not an identification.

### 3.2 The decision

**TSAnalytics.jl normalises explicitly: `Q = I` and `Λ` lower-triangular in its
leading r×r block, with positive diagonal.**

- `Q = I` matches `DynamicFactor`, so **its numbers transcribe directly** and the
  likelihood is comparable without renormalisation.
- Lower-triangular `Λ` with positive diagonal removes the remaining `r(r−1)/2`
  rotation and the `r` sign flips, making the fit **unique**. This is the
  standard factor-analysis normalisation and it costs nothing at estimation time
  (parameterise the free elements and fill the rest with zeros).

The parameter count is then `Nr − r(r−1)/2` for `Λ`, `pr²` for the factor VAR,
and `N` for `H` — against `DynamicFactor`'s `Nr + pr² + N`. **Our model has
`r(r−1)/2` fewer parameters and the same maximised likelihood**, because the
removed directions are flat. Assert that: §6.4.

`normalisation` is recorded on the fitted object, and `rotate_factors(m, G)` is
provided so a user who wants a different rotation (varimax, or a theory-driven
one) can get it without refitting.

### 3.3 The consequence for tests

Comparing **our normalised** Λ to **the reference's unnormalised** Λ requires
alignment. The procedure that worked, and which belongs in `test/` as a helper:

1. **Renormalise to `Q = I`**: `L = chol(Q)`, then `Λ̃ = Λ L`, `Ã = L⁻¹ A L`,
   `f̃ = L⁻¹ f`.
2. **Orthogonal Procrustes**: SVD of `Λ̃' Λ_ours = U S V'`, set `G = U V'`, apply
   `Λ̃G`, `G'ÃG`, `f̃G`.

**The recovered `G` on fixture B had `det = −1`** — an improper rotation [V]. Any
alignment helper must allow reflections. A helper that assumes `det = +1` will
fail on this exact fixture, which is why it is in the suite.

---

## 4. Fixtures

Both simulated, both travelling as CSV because neither R's nor NumPy's RNG stream
is portable (charter §6.6). The generator is documentation; the CSV is the
fixture.

### Fixture A — `fixtureA.csv`, 200 × 5, r = 1, p = 1

Seed 20260214, 200-period burn-in discarded, `Var(η) = I`.

| | value |
|---|---|
| true Λ | `[1.0, 0.8, −0.6, 0.5, 0.9]` |
| true A | `[0.7]` |
| true σ² | `[0.30, 0.50, 0.40, 0.60, 0.35]` |
| checksum `sum` | **135.8908272840041** |
| checksum `sumsq` | **1393.7582635434787** |

[V] re-read in this session: `sum = 135.8908272840`, `sumsq = 1393.7582635435`.

### Fixture B — `fixtureB.csv`, 250 × 6, r = 2, p = 1

Seed 20260215.

| | value |
|---|---|
| true Λ | `[[1.0, 0], [0.8, 0.4], [−0.6, 0.7], [0.5, −0.5], [0.9, 0.2], [0.2, 1.0]]` |
| true A | `[[0.7, 0.1], [−0.2, 0.5]]` |
| true σ² | `[0.30, 0.50, 0.40, 0.60, 0.35, 0.45]` |
| checksum `sum` | **76.35178397751577** |
| checksum `sumsq` | **2775.7926750184297** |

### Fixture A with missing data — `fixtureA_miss.csv`

12 blanked cells, including one **fully missing row** at t = 150 and a 3-run in
column 2. Indices in `fixtureA_miss_idx.json`.

### Standardisation

**The fixtures are stored raw.** `dfms` standardises internally with ddof = 1 and
offers no opt-out; `DynamicFactor` does not standardise at all; and
`DynamicFactorMQ`'s denominator depends on whether the input is a DataFrame
(charter §5.7, a 2.506 likelihood shift). So the test must standardise explicitly
with **ddof = 1** before feeding either reference, and our fitter does not
standardise at all (charter §5.7 decision).

`DynamicFactor` is **not scale-invariant**: llf(raw) = **−1180.13574349** against
llf(standardised) = **−1049.53340809** on fixture A [V]. Assert that the
expectation is for the standardised input, in a comment, or someone will feed raw
data and chase the difference.

---

## 5. Estimation

### 5.1 Direct maximum likelihood

`_optimize` over: the free elements of `Λ`, the factor VAR coefficients through
`_constrain_stationary` (11.0 §6), and `log` of the idiosyncratic variances.
`Q = I` is fixed, not estimated.

### 5.2 Initialisation

**Stationary**, matching `DynamicFactor`, which sets
`kwargs.setdefault("initialization", "stationary")` [S] — `a₀ = 0` and `P₀` the
stationary covariance of the companion system, **recomputed from the current
parameters at every likelihood evaluation**.

That recomputation is not optional and it is why 11.0 §5 factored
`_stationary_cov` out of `stationary_cov(::GaussianSSM)`: it must be callable on
the companion block at every objective evaluation, and the doubling iteration is
what makes that affordable (the existing docstring records 1 ms against ~2 s for
the `kron` solve at r = 54, a measured 1874× [S]).

### 5.3 Why the initial-condition convention matters more than anything else here

This is the finding that explains an apparent 0.92-log-likelihood gap between
the references, and it would otherwise look like a bug in one of them.

`dfms` **estimates the initial state condition as a free parameter**
(Shumway–Stoffer style): `Estep` returns `F_0 = Fs.row(0)` and
`P_0 = Psmooth.slice(0)`, and `EMstepDGR` does
`list2env(Estep(...), environment())`, which **overwrites the incoming
`F_0`/`P_0`** [S] `EM_algorithm.cpp`, `EMDGR.R`.

Re-evaluating `dfms`' own reported likelihood both ways [V] `s11/dfm/compare2.R`:

| fixture / method | reported llf | refit, EM-estimated init | refit, stationary init |
|---|---|---|---|
| A / DGR | −1048.7844157358 | **−1048.7844156799** | −1049.5339687249 |
| A / BM | −1048.6094260610 | **−1048.6094259563** | −1049.5365101600 |
| B / DGR | −1514.0497003041 | **−1514.0497004146** | −1514.9430284588 |
| B / BM | −1513.6983954855 | **−1513.6983953343** | −1514.9462867741 |

The reported value reproduces to ~1e-7 with the EM-estimated init and is off by
~0.92 with the stationary init. **On the common stationary-init objective,
statsmodels' optimum (−1049.5334) beats `dfms`' BM point (−1049.5365).** So both
found essentially the same maximum and the gap is entirely convention.

Corroborating detail: **the maximum factor-path difference is at t = 0 in every
single case**, and drops an order of magnitude by t ≥ 10 (A/BM: 0.1149 → 0.009427,
median 0.002878) [V].

And the degeneracy is real: fixture A's BM run drove `P₀ → 0.00243315` against
the stationary value of **3.396691164135642** — the classic pathology of treating
the initial state as free [V].

The same mechanism lives inside statsmodels:
`DynamicFactorMQ.fit_em(em_initialization=True)` is the **default**, and turning
it off closes the gap [V]:

| | llf | iterations |
|---|---|---|
| `DynamicFactor` ML, stationary init | −1052.0396799558 | — |
| `DynamicFactorMQ` EM, `em_initialization=True` | −1047.5168375259 | 2000 (hit the cap) |
| `DynamicFactorMQ` EM, `em_initialization=False` | −1049.5337142499 | 24 |

**Decision: stationary initialisation, always.** It is the defensible choice (the
factor process *is* stationary by construction, so its stationary distribution is
the correct prior), it matches the transcribable reference, and it avoids a known
degeneracy. Document it, and document that `dfms` differs — otherwise someone
will compare against `dfms` and report a bug.

### 5.4 Why not EM

Measured [V]:

- statsmodels' EM machinery is **496 code lines** (`fit_em` 315, the three
  maximisation steps 114 + 85 + 49, expectation 19, iteration 10), against
  `DynamicFactor`'s whole 725-line core for direct ML.
- On fixture A, **`dfms`' BM EM needed 2187 iterations** at tol 1e-10 where
  direct ML converged in **35** (552 function calls).
- The whole model is **11 parameters** for N=5, r=1.

EM earns its keep for large `N` or heavy missingness. Neither applies to anything
in this package's test suite or documentation. **Defer it, and record the
measurement** so the deferral is a decision rather than an omission.

### 5.5 Starting values

PCA on the standardised data → loadings by OLS → VAR on the principal components
→ idiosyncratic variances from the residuals. `dfms`' entire initialisation is
**28 lines** (`init_cond`) [S]; statsmodels' `start_params` is 129 but mostly
branches over `error_cov_type × error_order`, and the default path is ~30.

`_mols` (Stage 10.0) covers the VAR regression; `LinearAlgebra.eigen` covers the
PCA. After PCA, **rotate the starting loadings to the normalisation of §3.2**
(an LQ factorisation of the leading r×r block), so the optimiser starts inside
the parameterisation rather than on its boundary.

`start_params`' llf on fixture A is **−1699.4011064899** against the optimum's
−1049.53 [V] — a useful sanity figure: if our start is much worse than that, the
PCA step is wrong.

---

## 6. Verification plan

### 6.1 Tier 1 — the likelihood at fixed parameters

**The single most important test in the item**, and it has two independent
corroborations.

Fixture A, standardised (ddof = 1), at these parameters:

| | value |
|---|---|
| Λ | 0.70090792, 0.60429241, −0.57537284, 0.52373624, 0.66507451 |
| A | 0.6482473 |
| Q | 1 |
| diag(H) | 0.14812824, 0.36549749, 0.42432314, 0.52214416, 0.23249862 |
| init | stationary, `a₀ = 0`, `P₀ = Q/(1−A²)` |

| source | llf |
|---|---|
| statsmodels `DynamicFactor`, reported | **−1049.5334081297972** |
| **`dfms`' own Armadillo `SKF` at the same parameters and `P₀`** | **−1049.53340812981** |
| agreement between the two | **1.4e-10** |
| independent re-fit in this session | −1049.5334080883072 |

[V] `s11/dfm/compare.R`, `s11/chk_dfm.py`

Two independently written filters — one Python/Cython, one C++ — agreeing to
1e-10 is strong evidence the quantity is right and not merely reproducible. It
also confirms the 2π constant, the `log|F_t|` term and the `P₀` convention all
agree between them.

**Assert 1e-10 at the fixed parameter vector. Assert the fitted optimum to
5e-3 relative.** The 4.1e-08 spread between the two statsmodels runs above is
optimiser-level, and §8.1 of the charter is exactly this distinction.

### 6.2 Tier 2 — fitted parameters after alignment

Fixture A, r = 1 (sign is the only indeterminacy, so alignment is trivial):

| | statsmodels ML |
|---|---|
| Λ | 0.70090682, 0.60429108, −0.57537393, 0.52373497, 0.66507265 |
| A | 0.64824775 |
| diag(H) | 0.14812824, 0.36549749, 0.42432314, 0.52214416, 0.23249862 |
| llf | −1049.5334081297972 |
| aic / bic | 2121.0668162595944 / 2157.348307291623 |
| `coefficients_of_determination` | 0.91086263, 0.67706417, 0.61379053, 0.50857762, 0.82011444 |

[V] `s11/dfm/fit_sm.py`

Measured agreement after renormalisation and Procrustes, as a guide to
tolerances [V] `s11/dfm/align.py`:

| fixture / reference method | max Δ Λ (rel) | max Δ A (rel) | max Δ diag(H) (rel) | factor-path corr |
|---|---|---|---|---|
| A / `dfms` DGR | 8.80e-04 | 2.05e-03 | 2.53e-03 | 0.9999969005 |
| A / `dfms` BM | 3.47e-03 | 2.03e-03 | 5.23e-03 | 0.9999809951 |
| B / `dfms` DGR | 1.64e-03 | 3.33e-03 | 2.44e-03 | 0.9999970574 / 0.9999992791 |
| B / `dfms` BM | 3.96e-03 | 3.74e-03 | 6.22e-03 | 0.9999640063 / 0.9999973339 |

Those are *cross-reference* agreements across different estimators and
initialisations, so they bound what is achievable against `dfms`. Against
statsmodels with matched conventions, expect much better — **assert 5e-3 relative
against `dfms`, 1e-4 against statsmodels.**

### 6.3 Missing data

| | llf |
|---|---|
| complete | **−1049.5334080883072** |
| `fixtureA_miss.csv`, 12 NaNs | **−1035.826629949621** |

The likelihood **rises** (12 fewer density terms); per contributing observation it
is stable at −1.04953341 against −1.04840752 [V].

Plus the monotonicity ladder from 11.0 §2.4 — it is the same data and the same
assertion, and belongs in whichever suite runs first.

Also assert the missing-data **bookkeeping**: `nmissing` counts
`{0: 193, 1: 5, 2: 1, 5: 1}`, and the periods with any missing are
`[5, 40, 41, 42, 100, 150, 199]` [V].

### 6.4 The normalisation tests

These test §3.2's decision and have no reference analogue:

1. **Uniqueness.** Fit fixture B from 20 random starting values. After our
   normalisation, every converged fit must give the **same** Λ to 1e-5 — no
   rotation, no sign flips. This is the whole payoff of §3.2 and the test that
   would fail if we copied either reference.
2. **Likelihood equality with the unnormalised model.** Our maximised llf must
   equal statsmodels' to 1e-6 despite having `r(r−1)/2` fewer parameters,
   because the removed directions are flat. On fixture B that is one fewer
   parameter.
3. **`rotate_factors` is likelihood-preserving.** Applying any orthogonal `G`
   and recomputing must leave the llf unchanged to 1e-10.
4. **The Procrustes helper handles `det = −1`.** Fixture B's recovered rotation
   was `[[0.630687, 0.776037], [0.776037, −0.630687]]`, a reflection [V]. Assert
   the helper recovers it.

### 6.5 Reductions

1. **`r = N` is a saturated model.** With as many factors as series the model can
   fit exactly; assert the idiosyncratic variances go to their lower bound.
   (Note both references **refuse** `k_factors >= k_endog` — statsmodels throws
   `ValueError: Number of factors must be less than the number of endogenous
   variables` [V]. Decide deliberately whether we refuse too; refusing is
   defensible and matches the reference, but the reason should be in the message.)
2. **`r = 1`, `p = 1`, `N = 1` reduces to an ARMA(1,1)-like state-space model**
   that `fit_arma` can check. The factor plus idiosyncratic noise on a single
   series is exactly a local-level-style model, so `fit_uc` from 11.1 can also
   check it. Agree to 1e-6 on the llf.
3. **`A = 0` reduces to static factor analysis**, where the factors are i.i.d.
   and the MLE loadings are the eigenvectors of the correlation matrix.
   (Note `factor_order=0` in statsmodels does **not** do this — it sets
   `_factor_order = max(1, factor_order)·k_factors = 1` and `T = [[0.]]`, making
   the factor white noise, with llf = **−1094.48584214** [V]. Our API should
   either refuse `p = 0` as `dfms` does — `p >= 1` is mandatory there [V] — or
   mean it literally. **Refuse**, and say so.)

### 6.6 Convention traps as tests

Each from the charter or measured here [V]:

| trap | assertion |
|---|---|
| `error_cov_type` default is `'diagonal'`, not `'scalar'`; `k_params` for N=5,r=1 is 7 / 11 / 21 for scalar / diagonal / unstructured | our parameter count matches the diagonal case |
| `DynamicFactor` is not scale-invariant | raw vs standardised llf differ by the measured 130.6 |
| `res.factors.smoothed` is **(k_factors, nobs)** with numpy input — transposed relative to the data | our factor accessor returns observations × factors, and a test asserts the orientation explicitly |
| `dfms`' `logLik` is the **last E-step** value, at the parameters *entering* the final M-step, under the EM-estimated init — not the filter likelihood at the returned matrices | never compared directly |
| `dfms`' `m$P_0` is the **PCA-initial** stationary `P₀`; `m$ss_full$P_0` is the final EM-estimated one. Using the wrong one costs ~0.9 llf | the comparison script uses `ss_full` |
| `k_endog < 2` throws in statsmodels | we throw too, with a message naming the requirement |

### 6.7 Testset inventory

| # | Testset | Assertions (min) |
|---|---|---|
| 1 | fixture A llf at fixed parameters, 1e-10 | 8 |
| 2 | fixture B llf at fixed parameters | 8 |
| 3 | fixture A fitted optimum vs statsmodels, 1e-4 | 14 |
| 4 | fixture B fitted optimum after alignment, 5e-3 | 16 |
| 5 | normalisation uniqueness: 20 random starts → same Λ | 12 |
| 6 | likelihood equality with the unnormalised model | 6 |
| 7 | `rotate_factors` preserves the likelihood | 8 |
| 8 | Procrustes helper handles `det = −1` | 8 |
| 9 | missing data: 12-cell llf and per-obs stability | 10 |
| 10 | missing data: `nmissing` bookkeeping | 10 |
| 11 | missing data: monotonicity ladder | 10 |
| 12 | reduction: N=1, r=1 agrees with `fit_uc` | 8 |
| 13 | reduction: `A = 0` static factor analysis | 10 |
| 14 | refusals: `r ≥ N`, `N < 2`, `p < 1` | 12 |
| 15 | starting values: PCA llf ≈ −1699.40, better than that after one step | 8 |
| 16 | stationary init recomputed per evaluation (changing A changes `P₀`) | 10 |
| 17 | factor array orientation is observations × factors | 6 |
| 18 | `coefficients_of_determination` matches the reference | 10 |
| 19 | fixture B `coefficients_of_determination`, shape (6,2) | 14 |
| 20 | forecast: point and variance, and the alias identity | 12 |
| 21 | degenerate: zero-variance series throws | 8 |
| 22 | performance: allocations, N-scaling, `@code_warntype` | 10 |
| 23 | `show` reports the normalisation and convergence | 6 |

**23 testsets, ≥ 234 assertions.** The charter's 250 floor is met once 11.0's
filter tests (which this item exercises) are counted alongside.

---

## 7. Variance decomposition — the one derived quantity worth exporting

Both references compute the share of each series' variance explained by each
factor, and it is the most-used DFM output after the factor path itself.

| reference | name | fixture B, shape (6, 2) [V] |
|---|---|---|
| statsmodels | `coefficients_of_determination` | `[[0.54470929, 0.4650776], [0.72358049, 0.07323958], [0.01237265, 0.88042512], [0.01408733, 0.68099783], [0.7396314, 0.25439981], [0.33786968, 0.43269318]]` |
| `dfms` | `summary(m)$R2` | fixture B BM: `0.89, 0.74, 0.87, 0.68, 0.89, 0.88`; `R2_stats`: mean 0.8252, median 0.8758, sd 0.0918, min 0.6765, max 0.8898 |

Note these are **different quantities** — statsmodels gives per-factor R², `dfms`
gives the total across all factors. Export both: `factor_r2(m)` returning the
N×r matrix and `series_r2(m)` returning the N-vector. Say in the docstring which
reference each corresponds to.

Fixture A's per-series R² is `[0.91086263, 0.67706417, 0.61379053, 0.50857762,
0.82011444]` [V] — and note it correlates with the true loadings' magnitude,
which is a good sanity test.

### Bai–Ng information criteria

`dfms` offers `ICr` for choosing `r` (52 lines) [S]. Measured on fixture B
(IC1/IC2/IC3) [V]:

| r | IC1 | IC2 | IC3 |
|---|---|---|---|
| 1 | −0.47458253 | −0.47053491 | −0.47770195 |
| 2 | **−1.26002227** | **−1.25192703** | **−1.26626110** |
| 3 | −1.37817140 | … | … |
| 4 | −1.62434149 | … | … |

**These do not select the true r = 2 on this fixture** — they keep decreasing.
Worth implementing (it is a short, well-defined formula) and worth a test
asserting the measured values, but **not** worth presenting as a reliable
selector. Say so in the docstring: the criteria are consistent asymptotically in
both N and T, and N = 6 is not asymptotic.

---

## 8. Charts

Detail in the plot-recipes handoff. What 11.2 owes it, and what neither reference
ships:

| plot | statsmodels | `dfms` | ours |
|---|---|---|---|
| factor paths **with bands** | `factors.smoothed` + `smoothed_cov`, no plot | `plot.dfm` draws paths, no bands | **yes** — we have `smoothed_cov` and the band is the obvious use |
| series + factors overlaid | — | `plot.dfm(type="joint")` | yes |
| per-factor panels | — | `plot.dfm(type="individual")` | yes |
| R² bar chart | `plot_coefficients_of_determination` (62 lines) [S] | `summary$R2` | yes |
| loadings heatmap | **none** | **none** | **yes** — the obvious missing plot, and cheap |
| scree / IC vs r | — | `screeplot`, `plot.ICr` | yes |
| residual boxplot by series | — | `plot.dfm(type="residual")` | yes |

**Neither reference ships a factor-path-with-bands plot or a loadings heatmap.**
Both are easy with what the fit already carries, and they are the two most
useful. [V] all reference plot methods exercised, `s11/dfm/charts.R`,
`charts_sm.py`.

---

## 9. Budget

| Piece | Reference | Julia estimate |
|---|---|---|
| spec parsing, parameter layout | `__init__` 106 + `param_names` 58 + `state_names` 21 [S] | 150–200 |
| `update` (parameters → matrices) | **44 code lines** [S] | 60–80 |
| normalisation (§3.2) and `rotate_factors` | — (neither reference has it) | 100–140 |
| starting values (PCA → OLS → VAR) | `dfms` `init_cond` 28; statsmodels 129 [S] | 80–110 |
| variance decomposition + Bai–Ng IC | 29 + 52 [S] | 70–100 |
| forecast | — | 50 |
| `show` | `summary` 132 [S] | 50 |
| **total, excluding 11.0** | | **560–730** |

11.0 carries the 550–750 for the filter, smoother, initialisation and stationary
reparameterisation. **The honest total for "dynamic factor models" is therefore
~1100–1500 lines**, and that is the number to quote — but most of it is reusable
infrastructure, not DFM-specific.

---

## 10. Suggested build order

1. Multivariate filter and smoother passing their 11.0 reductions.
2. Likelihood at fixed parameters on fixture A, hitting **−1049.5334081** to
   1e-10. Nothing else until this passes.
3. ML estimation at **r = 1** — no rotation indeterminacy beyond sign, and
   fixture A gives exact targets.
4. `r = 2` with the §3.2 normalisation, plus the uniqueness test (§6.4.1).
5. Missing data.
6. Variance decomposition, IC, charts.
7. EM only if someone asks, with §5.4's measurements as the reason it waited.

---

## 11. What not to do

1. **Do not form an N×N `F_t` and invert or factorise it.** 11.0 §6.2.
2. **Do not copy either reference's non-identification.** §3.
3. **Do not estimate the initial state condition.** §5.3 — it costs ~0.9 llf of
   comparability and drove `P₀` to 0.0024 against a stationary 3.397 on a
   fixture where nothing was wrong.
4. **Do not standardise inside the fitter.** Charter §5.7.
5. **Do not compare to `dfms`' `logLik`** without reconstructing it: it is the
   last E-step value under a different initial condition (§6.6).
6. **Do not assume `det(G) = +1`** in the alignment helper. §3.3.
7. **Do not build mixed-frequency support.** §1.
8. **Do not present Bai–Ng IC as a reliable selector of `r`.** §7 — measured not
   to recover the true r = 2 on fixture B.
9. **Do not let `det(F_t) ≤ 0` pass silently.** `dfms` skips the contribution
   (charter defect D5); 11.0 §2.5 throws.
