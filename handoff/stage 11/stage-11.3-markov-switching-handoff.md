# Stage 11.3 — Markov-switching autoregression and regression

**Prerequisite reading:** `stage-11-overall-handoff.md`, and **§7 of
`stage-11.0-primitives-handoff.md`** — the `_ols` zero-weight fix, which is the
only thing this item needs from 11.0 and which currently **blocks it outright**.

**Difficulty: Hard, and for an unusual reason — nothing in the package is
reusable.** There is no hidden-Markov machinery of any kind: no forward-backward,
no Hamilton filter, no EM for mixtures. Measured budget **≈1100–1600 lines**,
which puts the item at the size of `src/diagnostics.jl` (1222) or
`src/unitroot.jl` (1041), and unlike those it has no incremental starting point.

**The compensating good news:** the canonical fixture ships inside statsmodels'
own test file (no network), and its target likelihood is **independently
corroborated by E-views to 8 digits**.

---

## 1. The model

Two of them, and the distinction matters because the two references implement
different ones.

### 1.1 Markov-switching regression (the simpler, build first)

```
y_t = x_t' β_{S_t} + ε_t,    ε_t ~ N(0, σ²_{S_t})
```

with `S_t ∈ {1 … k}` an unobserved first-order Markov chain with transition
matrix `P`. Any subset of `{β, σ²}` may switch.

### 1.2 Markov-switching autoregression — Hamilton's mean-adjusted form

```
y_t = a_{S_t} + Σ_{j=1}^p φ_j (y_{t−j} − a_{S_{t−j}}) + ε_t
```

[S] `markov_autoregression.py:169-230` (`predict_conditional`), docstring
`:80-90`

**Note the `a_{S_{t−j}}` terms.** The regime at each of the `p` lags enters, so
the conditional likelihood array has shape `(k,)^(p+1) × nobs` and the filter
runs over `k^(p+1)` joint states. For the canonical `k=2, p=4` that is **32**,
which is nothing — do not pre-emptively optimise it away (charter §7.4).

### 1.3 The references implement different models (charter §5.4)

`MSwM` builds the lags as **plain levels regressors** — `y_t = a_{S_t} + Σ φ_j
y_{t−j} + ε_t` — with a state space of size `k`, not `k^(p+1)`. [S]
`MSwM/R/2MSM.r:233-243`

**MSwM cannot express the Hamilton model at all.** Measured: re-implementing
MSwM's filter faithfully and evaluating it at statsmodels' parameters gives
**−194.43047679** against statsmodels' **−181.26339428** — a 13.2 gap that is
specification, not optimisation. [V] `s11/ms/crosscheck.py`

**Both models are worth having.** `fit_ms_regression` with lagged `y` in `exog`
is MSwM's model; `fit_ms_autoregression` is Hamilton's. Document the difference
prominently, because a user comparing against either reference needs to know
which one they have. The §6.5 reduction tests the pair against each other.

---

## 2. Reference sizes, measured

[V] AST breakdown, `s11/ms/ast_sizes.py`; `s11/ms/mswm_sizes.R`

### statsmodels

| file | total | core | boilerplate |
|---|---|---|---|
| `markov_switching.py` | 2407 | **≈1260** | ≈990 (`MarkovSwitchingResults` alone is 508, `summary` 162) |
| `markov_autoregression.py` | 493 | **327** | 32 |
| `markov_regression.py` | 456 | **304** | 32 |

**Critically: the hot loops are Cython, not Python.** `_hamilton_filter.*.so`
(278 KB) and `_kim_smoother.*.so` (258 KB) exist, and `markov_switching.py:23-33`
imports `{s,d,c,z}hamilton_filter_log` and `{s,d,c,z}kim_smoother_log` from them
[S]. The Python `cy_hamilton_filter_log` (145 lines) and `cy_kim_smoother_log`
(80) are allocation / log-space / initialisation wrappers. The actual recursions
are `_hamilton_filter.pyx.in:31-193` (**163 lines per dtype**) and
`_kim_smoother.pyx.in:31-144` (**114 per dtype**).

**So the 2407-line figure understates the algorithm by ~277 lines of tight
recursion.** An estimate built from the Python line count alone would be wrong,
which is precisely the Stage 10 IRF mistake in a new costume.

Note also *why* there are four dtype instantiations including `complex128`:
statsmodels uses `approx_fprime_cs`/`approx_hess_cs` — **complex-step
derivatives** [S] `markov_switching.py:1130-1177`. Julia's real AD is cleaner,
but only if the filter's buffers are typed on the parameter eltype (§4.4).

### MSwM 1.5

1295 lines in `R/2MSM.r`, no compiled code (`NeedsCompilation: no`).

| function | lines | role |
|---|---|---|
| `.MSM.lm.msmFilter` | **39** | Hamilton filter — **levels space**, `dnorm`, products, no logsumexp |
| `.MSM.msmSmooth` | **27** | Kim smoother, levels space, also stores pairwise `smoTransMat` |
| `.MSM.lm.maximEM` | **18** | M-step: `lm(y~.-1, weights=c(t(w)))` on a `kronecker`-replicated design |
| `.MSM.linear.iteraEM` | **21** | EM loop; `iniProb ← smoProb[1,]` |
| `.MSM.em` | **77** | `maxiterOuter=5` random restarts × `maxiterInner=10`, then full EM |
| plots | **240** | four S4 methods |
| **Gaussian-lm core** | **≈291** | |

**39 lines for the filter.** That is the honest lower bound on the recursion
itself — the rest of statsmodels' bulk is log-space safety, the `k^(p+1)`
striding, and the parameter bookkeeping, all of which we need too.

---

## 3. The fixture, and the target

### 3.1 It ships inside statsmodels

The canonical Hamilton (1989) US real GNP growth series is a **literal in
statsmodels' own test file** at `test_markov_autoregression.py:21` (variable
`rgnp`). Verified present in this session [V]. No network, no dataset download.

| | value |
|---|---|
| n | **135**, quarterly, 1951Q1–1984Q3 |
| `sum` | **100.5207128600** |
| `mean` | **0.7445978730** |
| first 5 | 2.59316421, 2.20217133, 0.45827562, 0.9687438, −0.24130757 |
| last 5 | 1.61550522, 2.70930099, 1.24461412, 0.50835466, 0.14802167 |

Extracted to `hamilton_rgnp.csv` / `.json`.

### 3.2 The target, independently corroborated

`MarkovAutoregression(rgnp, k_regimes=2, order=4, switching_ar=False)`.
nobs = 131, k_params = 9. [V] `s11/ms/fit_sm.py`

| parameter | value |
|---|---|
| `p[0->0]` | **+0.7546643177** |
| `p[1->0]` | **+0.0959150489** |
| `const[0]` | **−0.3588026189** |
| `const[1]` | **+1.1635220287** |
| `sigma2` | **+0.5913638324** |
| `ar.L1` | **+0.0134798930** |
| `ar.L2` | **−0.0575297622** |
| `ar.L3` | **−0.2469918252** |
| `ar.L4` | **−0.2129275902** |
| **llf** | **−181.2633942812** |
| aic / bic / hqic | 380.52678856 / 406.40356447 / 391.04167901 |
| `P` (left-stochastic) | `[[0.75466432, 0.09591505], [0.24533568, 0.90408495]]` |
| `expected_durations` | **[4.07604793, 10.42589261]** |
| `initial_probabilities` | [0.28106914, 0.71893086] — steady state, matching the eigenvector to **4.4e-16** |

**statsmodels' own test file records `"llf": -181.26339` as the E-views
expectation** at `test_markov_autoregression.py:502` — verified in this session
[V]. So the target is corroborated by an independent commercial implementation to
8 digits, which is better provenance than anything in Stage 10 had.

The same file records `"llf_fit_em": -183.85444` at line 504 [V] — see §5.3.

### 3.3 The probability arrays

Regime 1 is the high-growth regime. Full series in `sm_ar4_smoothed.csv` and
`sm_ar4_filtered.csv`. [V]

| | first 5 | last 5 | sum |
|---|---|---|---|
| `smoothed[:,1]` | 0.96809796, 0.99107108, 0.99855938, 0.95853315, 0.5406366 | 0.99832379, 0.99993898, 0.99685836, 0.96921009, 0.92771603 | **93.2939799393** |
| `filtered[:,1]` | 0.77672362, 0.94919266, 0.99632016, 0.99025791, 0.94010952 | 0.99537703, 0.99983291, 0.99375605, 0.95616539, 0.92771603 | **96.6874124793** |

**These are the tier-1 targets**, because they are the filter's output at a fixed
parameter vector and are determined to machine precision. statsmodels' own tests
use `atol=1e-5`; we can do better and should assert **1e-6**.

Note the last filtered and last smoothed value coincide (0.92771603), as they
must — at `t = T` there is no future information. Assert it; it is a free
structural check.

---

## 4. What to build

### 4.1 The Hamilton filter, in log space

**Log space is not optional.** MSwM works in levels (`dnorm`, products,
`/margLik`) with no logsumexp [S], which is underflow-prone for long series or
well-separated regimes. statsmodels is fully log-space with the max-trick
logsumexp. Follow statsmodels.

The recursion, for joint state `j` over `k^(p+1)` states:

```
log-predicted:  lp_t[j]  = logsumexp_i ( log P[j_0, i_0] + lf_{t-1}[i] )   # over compatible i
log-joint:      lj_t[j]  = lp_t[j] + llik_t[j]
log-marginal:   lm_t     = logsumexp_j lj_t[j]
log-filtered:   lf_t[j]  = lj_t[j] - lm_t
loglik         += lm_t
```

where the "compatible `i`" structure is the `k^(p+1)` striding: the joint state
tracks `(S_t, S_{t−1}, …, S_{t−p})`, so a transition shifts the window. statsmodels
carries four stride constants for this (`k_regimes_order_m1`, `_order`,
`_order_p1`, `_order_p2`) [S], which is a sign of how easy it is to get wrong.

**Write the striding as an explicit index map built once**, not as arithmetic
inline in the loop. It is clearer, it is testable on its own (§6.2), and it costs
nothing — the map is `k^(p+1)` integers.

### 4.2 The Kim smoother

Backward recursion over the pairwise joint probabilities:

```
Pr[S_t = i, S_{t+1} = j | Y_T] = Pr[S_{t+1} = j | Y_T] · Pr[S_t = i | Y_t] · P[j,i] / Pr[S_{t+1} = j | Y_t]
```

then marginalise. The pairwise quantities are also what the EM transition M-step
needs, so compute them once and return them.

### 4.3 The transition parameterisation

**Multinomial logit (softmax) with the last regime as the implicit reference,
applied column-wise.** [S] `markov_switching.py:1659-1666`: for each *from*-regime
`j`, `constrained = exp(u − logsumexp([0, u]))`.

Verified: `transform_params` on `[0.5, −1.2]` → `[0.62245933, 0.23147522]`,
matching `exp(.5)/(1+exp(.5))` and `exp(−1.2)/(1+exp(−1.2))`; the round trip
returns `[0.5, −1.2]` exactly. [V] `s11/ms/probe_sm.py`

**Free parameters = `k(k−1)`**, confirmed for k = 2, 3, 4 (transition parameter
counts 2, 6, 12) [V].

**The matrix is left-stochastic**: `P[i,j] = Pr[S_t = i | S_{t−1} = j]`, so
**columns** sum to 1. Verified `colsums = [1,1]`, `rowsums = [0.8506, 1.1494]`
[V]. MSwM agrees. **Document this in the docstring** — it is the opposite of the
convention many textbooks use, and a transposed `P` gives plausible-looking
nonsense.

For `k > 2` statsmodels' inverse uses a **numerical root-finder**
(`_untransform_logistic`) [S]. Ours should solve it in closed form where
possible: with the last regime as reference, `u_i = log(p_i / p_k)`, which is
exact and differentiable. **Verify the round trip at k = 3, 4** against the
reference's numerical answer — if they agree, ours is better and simpler.

### 4.4 The optimiser path, and the AD constraint

`_optimize` covers the outer ML step, **but the filter's buffers must be typed on
the parameter eltype**. A preallocated `Vector{Float64}` will block ForwardDiff
and force a numerical gradient. This is the one real performance-correctness
tension in the item and it is the reason statsmodels instantiates its Cython
kernel four times. Do it right once:

```julia
VT = promote_type(eltype(params), Float64)
lf = Vector{VT}(undef, nstates)
```

exactly as `kalman_filter_diffuse` already does [S]
`src/statespace/diffuseinit.jl`. Follow that precedent.

### 4.5 EM

Four M-steps: the transition matrix (closed form from the pairwise smoothed
probabilities), the regression coefficients (**weighted least squares**), the
variance, and the mean-adjusted AR coefficients.

The weighted-LS step is where `_ols` comes in — MSwM's M-step is literally
`lm(y~.-1, weights=c(t(w)))` on a `kronecker`-replicated design [S]
`2MSM.r:1160`. **And it is where `_ols` currently throws**: smoothed regime
probabilities hit exact zero routinely (MSwM's own `iniProb` converged to
`[0, 1]` on this very fixture [V]), and `src/unitroot.jl:58` and `:172` assert
`all(>(0), weights)`. 11.0 §7 fixes it.

**EM is a warm start, not the estimator.** §5.3.

### 4.6 Expected durations

`1/(1 − P_ii)`, giving **[4.07604793, 10.42589261]** on the canonical fit [V].
`MSwM` computes this **not at all** (`grep -i duration` → 0 hits) [V], and
statsmodels exposes it but prints no duration table. Put it in `show`.

---

## 5. Multimodality — part of the algorithm, not a caveat

### 5.1 Measured, on the canonical specification

25 random perturbed starts with `em_iter=0` [V] `s11/ms/localopt.py`,
`s11/ms/canon.py`:

| | result |
|---|---|
| default fit (`em_iter=5` + BFGS) | **−181.26339428** |
| `search_reps=5` / `20` / `50` | −181.26339429 / −181.26339427 / −181.26339426 |
| 25 random starts | 17 finite; **11/17 (65%) reached the global max**; worst **−183.66916511** |
| among the 11 global-max hits, after label canonicalisation | **llf spread 2.41e-08, max parameter deviation 3.40e-05** |

Distinct modes found: **−181.2634** (global), **−182.8848**, and a flat
no-switching ridge at **−183.6683 / −183.6691 / −183.6692**.

### 5.2 Other specifications are worse

| spec | k_params | default llf | best of 25 starts | % at global max |
|---|---|---|---|---|
| `switching_ar=False`, common σ² | 9 | −181.26339428 | −181.26339426 | 65% |
| `switching_ar=True` (all switching) | 13 | −176.25055597 | −176.25055483 | 22% (9/25 finite) |
| `switching_variance=True` | 10 | −182.03690320 | **−180.67728500** | 53% |

**The `switching_variance=True` default fit is itself a local optimum, 1.36 llf
worse than the best found.** [V] That is the reference's default behaviour on the
canonical dataset, and it is why §5.4 makes a restart search our default.

### 5.3 Pure EM fails, and the warm start is what makes BFGS work

[V] same scripts:

| | llf |
|---|---|
| `em_iter=0` | **−182.49888744** (a different, worse optimum; parameter deviation 1.13) |
| `em_iter=1` | −181.26339426 |
| `em_iter ∈ {5, 20, 100}` | −181.26339428, identically |
| `_fit_em(maxiter=500, tol=1e-10)` alone | **−183.85545**, with `p[0->0]=0.4748, p[1->0]=0.4770` — the degenerate no-persistence ridge |

statsmodels' own test file records `llf_fit_em = −183.85444` [V], so this is
documented reference behaviour, not an artefact of my run.

**Design: EM for a small fixed number of iterations (default 5) as a warm start,
then gradient-based ML.** Exactly statsmodels' `fit` structure (153 lines, of
which the EM warm start and `search_reps` are most) [S]. Expose `em_iter`;
document that `em_iter=0` lands on a worse optimum on the canonical fixture, with
the figure.

### 5.4 The restart search is the default

`Threads.@threads` over perturbed starting values (charter §7.3) — embarrassingly
parallel, and the dominant cost. Default `n_starts = 10`, with the measured
65%-hit-rate in the docstring so a user understands why it is not 1.

Report **how many distinct modes were found** on the fitted object. A user whose
data produces three modes 1.4 apart should be told, not handed the first one.

---

## 6. Verification plan

### 6.1 Tier 1 — the filter and smoother at fixed parameters

**The spine of the suite.** Feed the §3.2 parameter vector; assert:

| quantity | target | tolerance |
|---|---|---|
| llf | **−181.2633942812** | 1e-9 |
| `filtered[:,1]`, all 131 | `sm_ar4_filtered.csv` | 1e-6 |
| `smoothed[:,1]`, all 131 | `sm_ar4_smoothed.csv` | 1e-6 |
| `sum(filtered[:,1])` | **96.6874124793** | 1e-6 |
| `sum(smoothed[:,1])` | **93.2939799393** | 1e-6 |
| `expected_durations` | [4.07604793, 10.42589261] | 1e-7 |
| `initial_probabilities` | [0.28106914, 0.71893086] | 1e-8 |
| steady state = eigenvector of P | — | **4.4e-16** |
| `filtered[T] == smoothed[T]` | structural identity | 1e-12 |

### 6.2 The striding map, tested alone

The `k^(p+1)` index map (§4.1) is the likeliest bug site and is testable without
any reference:

- For `k=2, p=4`: 32 states; each state's `(S_t, …, S_{t−4})` decoding round-trips.
- A transition from state `i` to state `j` is permitted iff `j`'s lag window is
  `i`'s window shifted — assert the permitted set has exactly `k` members per `i`.
- The marginal of the joint filtered distribution over `S_t` must sum to 1 at
  every `t`, to 1e-12.

### 6.3 Tier 2 — the fitted optimum, with measured tolerances

From §5.1's spread among global-max hits: llf spread **2.41e-08**, max parameter
deviation **3.40e-05**, per-coordinate spreads all ≤ 6.40e-05.

**So: assert llf to 1e-6 absolute and parameters to 1e-4 absolute, after label
canonicalisation.** State the measured spread in the comment.

Also: **assert our llf at our optimum ≥ −181.2633942812 − 1e-6**. One-sided, and
the thing actually worth guaranteeing.

### 6.4 Label canonicalisation — and the rule is not the obvious one

A random start returned `params = [0.90408471, 0.24532915, 1.16351609,
−0.35881218, …]` — exactly the label-swapped default, with **max parameter
difference 1.522** at an **llf difference of 1.86e-08**. [V]

In statsmodels' parameterisation `P = [[a, b], [1−a, 1−b]]` with `a = p[0->0]`,
`b = p[1->0]`, the k=2 canonicalisation is

**`(a, b) → (1 − b, 1 − a)`**, plus swapping `const[0] ↔ const[1]`

— **not** `(a, b) → (b, a)`. Verified round-trip exact [V] `s11/ms/canon.py`.

**This must be a public function, not a test helper** (charter §8.3): a user
comparing two fits hits the same problem. `canonicalise(m)` orders regimes by
ascending intercept (or ascending variance when only the variance switches) and
permutes everything consistently. Assert that `canonicalise` is idempotent and
that it maps the swapped parameter vector above onto the canonical one.

### 6.5 Reductions

1. **`MarkovAutoregression` at `order=0` must reproduce `MarkovRegression`.**
   statsmodels' *likelihoods* agree exactly here: `MAR(order=0).loglike(MR_params)
   = −191.28811083 == MR.llf` [V]. **And statsmodels' own `.fit()` is broken at
   that order** — `LinAlgError: SVD did not converge` from `_em_exog`, or
   `IndexError` with `em_iter=0` (charter defect D10). So this reduction tests us
   against a reference the reference cannot run, which makes it more valuable, not
   less. Assert to 1e-10.
2. **`k = 1` reduces to ordinary OLS / AR.** One regime means no switching; the
   fit must equal `_ols` or `fit_arma` exactly. Both references make `k=1`
   awkward; we should either support it as an exact reduction or refuse it
   cleanly. **Support it** — it is a free structural test.
3. **Non-switching parameters are recovered when the regimes are identical.**
   Simulate with `P = [[0.5, 0.5], [0.5, 0.5]]` and identical regime parameters;
   the fit must recover the pooled estimates and the smoothed probabilities must
   be flat at the steady state.

### 6.6 The MSwM cross-check — how to do it correctly

MSwM is GPL and non-reproducible by default, so **no MSwM number enters the test
suite** (charter §4, §9.5). It is used as evidence that the *statsmodels* target
is right, and that comparison had to be constructed:

The correct statsmodels counterpart of MSwM's specification is
`MarkovRegression(y = rgnp[4:], exog = 4 levels lags, switching_trend=True,
switching_exog=False, switching_variance=False)`, nobs = 131. On that pair [V]
`s11/ms/equiv.py`:

| parameter | statsmodels | MSwM | rel. err |
|---|---|---|---|
| `const[0]` | −0.44739627 | −0.44846500 | 2.38e-03 |
| `const[1]` | +1.11296284 | +1.11673500 | 3.38e-03 |
| `ar.L1` | +0.11176305 | +0.10959380 | 1.98e-02 |
| `ar.L4` | −0.13563027 | −0.13800080 | 1.72e-02 |
| `sigma` | +0.78909867 | +0.78911052 | **1.50e-05** |
| `P[0,0]` | +0.66820303 | +0.67358280 | 7.99e-03 |
| `P[1,1]` | +0.91254068 | +0.91369098 | 1.26e-03 |
| llf | −180.18436018 | −180.11423858 | Δ = 0.0701 |

Smoothed probabilities: **corr 0.99980722**, max abs diff 0.07076606, mean abs
diff 0.00181952.

**The residual 0.0701 is entirely the initial-distribution convention.** At
MSwM's own parameters, replacing its estimated `iniProb = [0, 1]` with the
steady state costs exactly **0.071892** (−180.11424133 → −180.18613313) [V].
Secondary cause: MSwM is pure EM with no gradient polish.

Label alignment here is the **identity** permutation — regime 0 is the low
intercept in both — verified two ways (sorting on the intercept, and checking the
correlation under both permutations: +0.9998 identity against −0.9998 swapped)
[V].

**Our initial distribution is the steady state**, matching statsmodels. Document
that MSwM estimates it, with the 0.0719 figure.

### 6.7 Convention traps as tests

Each measured [V] `s11/ms/probe2.py`:

| trap | our behaviour to assert |
|---|---|
| `switching_variance` defaults to **False** in both reference classes | ours too, and documented |
| `switching_trend` defaults to **True** | ours too |
| **`MarkovAutoregression.switching_ar` defaults to `True`** — so the reference's defaults give `k_params=13, llf=−176.25055597`, **not** the textbook −181.26339 | our default is explicit and the docstring names which spec reproduces Hamilton |
| `switching_exog` defaults **True in `MarkovRegression`, False in `MarkovAutoregression`** — opposite | ours is consistent across both, and the inconsistency is noted |
| **A fully non-switching spec is silently unidentified**: statsmodels returns `p[0->0]=p[1->0]=0.5` untouched from the start, llf −183.66915690, no warning (charter defect D11) | **we throw**, naming which component must vary |
| `initial_probabilities` is the distribution at **t = −order**, not t = 0 | ours too, and the docstring says so |
| **`MarkovRegression` silently filters at order 1 even when `order==0`** (charter defect D9): measured llf −180.14328657 against a hand-computed `P²π` of −180.14328933, where `P¹π` gives −180.11424133 | ours applies `P` exactly `order` times; a test pins all three values so the distinction cannot regress |
| MSwM's `logLikel` stores the **negative** log-likelihood (defect D8) | the comparison script negates it; noted in a comment |
| MSwM's `smoProb` has **`nobs+1` rows** (row 1 is t=0) while `filtProb` has `nobs` | the comparison drops row 1; noted |

### 6.8 Testset inventory

| # | Testset | Assertions (min) |
|---|---|---|
| 1 | llf at the canonical fixed parameters, 1e-9 | 6 |
| 2 | filtered probabilities, all 131, 1e-6 | 14 |
| 3 | smoothed probabilities, all 131, 1e-6 | 14 |
| 4 | probability sums and the `t=T` identity | 8 |
| 5 | expected durations, initial probabilities, steady state | 10 |
| 6 | striding map: decode round-trip, permitted transitions | 16 |
| 7 | joint filtered marginal sums to 1 at every t | 6 |
| 8 | logit transform round-trip, k = 2, 3, 4 | 14 |
| 9 | closed-form inverse agrees with the reference's root-finder | 8 |
| 10 | left-stochastic orientation: columns sum to 1 | 6 |
| 11 | fitted optimum, tier-2 tolerances | 14 |
| 12 | our optimum is no worse than the reference's | 4 |
| 13 | `canonicalise` maps the swapped vector correctly, and is idempotent | 14 |
| 14 | reduction: `order=0` = `MarkovRegression`, 1e-10 | 10 |
| 15 | reduction: `k=1` = OLS / `fit_arma` | 10 |
| 16 | reduction: identical regimes recover pooled estimates | 12 |
| 17 | EM warm start: `em_iter=0` vs 1 vs 5, measured llf values | 10 |
| 18 | pure EM lands on −183.855, as the reference records | 6 |
| 19 | restart search finds the global max; mode count is reported | 10 |
| 20 | `switching_variance=True`: the better optimum −180.67728500 is reachable | 8 |
| 21 | refusal: fully non-switching spec throws | 8 |
| 22 | refusal: `k_regimes < 1`, `order < 0` | 8 |
| 23 | `_ols` zero-weight path works inside the EM M-step | 10 |
| 24 | log-space: no underflow on a well-separated 2000-point series | 8 |
| 25 | TVTP: time-varying transition matrix has the right shape and varies | 12 |
| 26 | performance: allocations, ForwardDiff flows, `@code_warntype` | 10 |
| 27 | `show` reports durations, regimes, convergence, mode count | 8 |

**27 testsets, ≥ 284 assertions.**

---

## 7. Time-varying transition probabilities

Supported by statsmodels (`exog_tvtp`), absent from MSwM. Verified working [V]
`s11/ms/probe_sm.py`:

```
MarkovAutoregression(rgnp, k_regimes=2, order=1, exog_tvtp=[1, dummy])
→ tvtp=True, k_tvtp=2, k_params=9
  names: p[0->0].tvtp0, p[1->0].tvtp0, p[0->0].tvtp1, p[1->0].tvtp1, …
  llf = −185.44341128
  regime_transition_matrix shape (2, 2, 135), genuinely time-varying:
  P[:,:,3] = [[0.45199, 0.19450], [0.54801, 0.80550]]
```

Free parameters `k(k−1)·k_tvtp`. Under TVTP `transform_params` is the **identity**
on the transition block — the logit lives inside
`_regime_transition_matrix_tvtp` instead [S] `markov_switching.py:675-705`.

**That llf of −185.44341128 is NOT a reusable expectation**: the run used a
synthetic dummy (`(t mod 17) < 3`), not the NBER recession indicator
statsmodels' own test uses. It demonstrates the mechanism only. [V, flagged
unverified as an expectation]

**Include TVTP** — it is a modest addition on top of the machinery and it is the
one capability that makes this more than a textbook exercise (regime
probabilities that respond to a covariate). Build the fixture properly: either
find statsmodels' own test expectation for its NBER-indicator case, or construct
a clearly documented synthetic one and record it as our own ground truth.

---

## 8. Charts

Detail in the plot-recipes handoff. The split between the references is stark and
convenient:

**statsmodels has no plotting methods at all** (`grep "def plot"
tsa/regime_switching/*.py` → 0 hits) [V]. It gives you the arrays:
`filtered_marginal_probabilities`, `smoothed_marginal_probabilities`,
`predicted_marginal_probabilities` (all 131×2), `expected_durations`,
`joint_likelihoods`, `llf_obs`.

**MSwM has four S4 plot methods, all of which ran successfully** [V]
`s11/ms/plots_mswm.R`:

| method | draws |
|---|---|
| `plotProb(x, which=1)` | per-regime panel with **smoothed and filtered probabilities together** |
| `plotProb(x, which≥2)` | the series against the smoothed probability of one regime, **shading the periods in that regime** |
| `plotReg(x, expl, regime)` | response and regressors against one regime's smoothed probabilities |
| `plotDiag(x, regime, which)` | residuals vs fitted / Normal Q-Q / ACF and PACF of residuals **and squared residuals**, pooled or per-regime |
| `plot(x)` | per-regime residuals with the conditional residuals |

**The two worth copying as recipes:** the regime-shaded series panel and the
smoothed-plus-filtered overlay. Expected duration is a derived scalar — put it in
`show`, not a plot.

---

## 9. Budget

| Piece | Reference | Julia estimate |
|---|---|---|
| Hamilton filter, log space, `k^(p+1)` | 163 Cython + 145 wrapper | 150–200 |
| Kim smoother, log space, pairwise | 114 Cython + 80 wrapper | 110–150 |
| Multinomial-logit transform + inverse | 112 + 51 | 100–140 |
| EM: four M-steps | 160 + 29 + 38 = 227 | 200–260 |
| Parameter-index bookkeeping | 150 (`MarkovSwitchingParams`) | 80–130 |
| Steady-state initial distribution | 79 | 40–60 |
| Start values + restart search | 90 + 46 + 36 | 100–140 |
| Mean-adjusted `predict_conditional` | 62 | 60–90 |
| Results, durations, `canonicalise`, `show`, 2 recipes | ~250 | 250–350 |
| **total** | | **≈1100–1600** |

---

## 10. Suggested build order

Deliberately filter-first, because every later piece is testable only once the
filter is right:

1. **Log-space Hamilton filter + Kim smoother at fixed parameters**, against
   `sm_ar4_filtered.csv` / `sm_ar4_smoothed.csv` at 1e-6 and the llf at 1e-9.
2. The striding map tests (§6.2) — these can be written before the filter.
3. Steady-state initialisation + the logit round trip.
4. `MarkovRegression` (order 0) ML against **llf = −191.28811083**.
5. The EM M-steps (needs 11.0 §7).
6. Mean-adjusted `MarkovAutoregression` against **llf = −181.2633942812**.
7. `canonicalise`, then the restart search.
8. TVTP.

---

## 11. What not to do

1. **Do not use MSwM numbers as expectations.** Different model (§1.3), and
   non-reproducible under its own defaults — three runs at `set.seed(101)` gave
   −183.58332, −183.66895, −183.66916 (charter defect D6) [V].
2. **Do not work in levels space.** §4.1.
3. **Do not rely on pure EM.** It converges to −183.855 (§5.3).
4. **Do not assume `(a,b) → (b,a)` is the label swap.** §6.4.
5. **Do not preallocate `Vector{Float64}` in the filter.** §4.4.
6. **Do not let a non-switching specification through.** Charter defect D11 —
   statsmodels returns the start values untouched with no warning.
7. **Do not apply `P` the wrong number of times at `order=0`.** Charter defect
   D9; §6.7 pins all three candidate values so the distinction is tested.
8. **Do not clamp zero weights to 1e-20** the way statsmodels does. 11.0 §7.2 —
   a clamped row still contributes to `X'X`; an excluded one does not.
9. **Do not report a single optimum without saying how many were found.** §5.4.
