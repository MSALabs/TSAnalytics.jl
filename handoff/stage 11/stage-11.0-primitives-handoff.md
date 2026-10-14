# Stage 11.0 — Shared primitives
## Multivariate-observation filter and smoother, partitioned initialisation, stationary VAR reparameterisation, zero-weight `_ols`

**Prerequisite reading:** `stage-11-overall-handoff.md`. §5.6, §6.2 and §6.3 of
the charter fix the decisions this document implements.

**This is a new roadmap row.** It does not appear in `development-sequence.md`
and must be added. It exists because 11.1, 11.2 and 11.3 each need a piece of
the same machinery, and building it three times inside three sub-stages is the
failure mode this handoff is written to prevent.

**Nothing in this item is user-facing.** Every function here is internal
(underscore-prefixed, or a method on an existing type). No new export. The
verification therefore cannot lean on a reference's public API, and §4 explains
how it leans on reductions instead.

---

## 1. What exists, exactly

Read at HEAD `ad0010e`. [S] throughout this section.

| Function | Location | Signature | Observation |
|---|---|---|---|
| `kalman_filter` | `src/statespace/gaussianssm.jl:239` | `(::GaussianSSM, y)` | **univariate**, ARMA-specific (`Z = e₁`, `H = 0`), concentrates σ² |
| `kalman_smoother` | `src/statespace/gaussianssm.jl:320` | `(::GaussianSSM, y)` | same restriction |
| `kalman_filter` | `src/statespace/timevaryingssm.jl:139` | `(::TimeVaryingSSM, y, a0, P0)` | **univariate**, general `Z/T/R/Q/H`, no concentration |
| `kalman_smoother` | `src/statespace/timevaryingssm.jl:291` | `(::TimeVaryingSSM, y, a0, P0)` | returns `(alpha, V, eta, eta_var, eps, eps_var, converged)` |
| `kalman_filter_diffuse` | `src/statespace/diffuseinit.jl:125` | `(::TimeVaryingSSM, y; diffuse_idx, a0, P_star0)` | **univariate**, exact diffuse, **filter only** |
| `stationary_cov` | `src/statespace/gaussianssm.jl:174` | `(::GaussianSSM; tol, maxiter)` | doubling iteration, `GaussianSSM`-typed |
| `_ols` | `src/unitroot.jl:51` | `(X, y; weights, method)` | vector response; **rejects zero weights** at `:58` |
| `_mols` | `src/var.jl` (Stage 10.0) | multi-response OLS | [U] not re-read at this HEAD |
| `companion` | `src/var.jl` (Stage 10.0) | companion matrix | [U] |
| `_optimize` | `src/optim.jl:57` | Nelder-Mead / L-BFGS, ForwardDiff-compatible | |

`TimeVaryingSSM` itself is the right representation and is **not changed**:

```julia
struct TimeVaryingSSM{S<:Real}
    T::Vector{Matrix{S}}
    Z::Vector{Matrix{S}}
    R::Vector{Matrix{S}}
    Q::Vector{Matrix{S}}
    H::Vector{Matrix{S}}
    r::Int
end
```

A length-1 vector broadcasts across every period (`_tv_at`), which is how a
constant matrix in an otherwise time-varying system is represented without
materialising it per period.

**What does not exist:** any multivariate-observation filter or smoother, any
diffuse smoother, any partitioned initialisation, any stationarity-enforcing
reparameterisation of a VAR coefficient block. Confirmed by grep. [V]

---

## 2. Item 11.0a — the multivariate-observation filter

### 2.1 The decision, restated

**Sequential (univariate) treatment of the multivariate series**, Durbin &
Koopman §6.4. Each of the `N` series in period `t` is brought in as its own
scalar update. The charter's §6.2 gives the four measured reasons; the one worth
repeating here is that **this makes the new code a loop around the scalar
recursion the package already has and has tested**, rather than new linear
algebra.

> **Citation warning.** The charter's Appendix A records that this project's
> Durbin & Koopman citations rest on training memory rather than an edition in
> hand. §6.4 is the section number for the univariate treatment in the 2012
> second edition as recalled, and the *algorithm* below is stated in full so it
> does not depend on the citation being right. **Verify the section number
> against a physical copy before it goes in a docstring.** [U]

### 2.2 The recursion, stated in full

For period `t` with observation vector `y_t ∈ ℝ^N`, state `a_t ∈ ℝ^r`, covariance
`P_t`, design `Z_t ∈ ℝ^{N×r}`, observation covariance `H_t`:

**Require `H_t` diagonal.** The sequential treatment as written below assumes the
measurement errors are uncorrelated across series within a period. This holds for
every model in Stage 11 (11.1's irregular is scalar; 11.2's idiosyncratic
covariance is diagonal by both references' default — `error_cov_type='diagonal'`
and `rR="diagonal"` [V]). **If `H_t` is not diagonal, throw**, naming the
requirement — do not silently use only its diagonal. A future item that needs a
dense `H` can either rotate by `chol(H)⁻¹` first or add the conventional path.

Set `a_t^{(1)} = a_t`, `P_t^{(1)} = P_t`. For `i = 1 … N`:

```
if y_t[i] is missing:
    a^{(i+1)} = a^{(i)};  P^{(i+1)} = P^{(i)}          # skip entirely
else:
    z    = Z_t[i, :]                                    # row i, length r
    v    = y_t[i] - dot(z, a^{(i)})                     # scalar innovation
    f    = dot(z, P^{(i)} * z) + H_t[i,i]               # scalar
    f <= 0  ->  throw                                    # §2.5
    k    = P^{(i)} * z / f                               # length r
    a^{(i+1)} = a^{(i)} + k * v
    P^{(i+1)} = P^{(i)} - f * (k * k')
    loglik += -0.5 * (log(2π) + log(f) + v^2 / f)
```

Then the single transition to the next period:

```
a_{t+1} = T_t * a_t^{(N+1)}
P_{t+1} = T_t * P_t^{(N+1)} * T_t' + R_t * Q_t * R_t'
```

**Note the asymmetry that makes this correct:** the within-period updates are
*filtering* updates (no transition), and the transition is applied **once**, after
all `N` series have been absorbed. Getting this wrong — applying `T_t` per series
— is the single likeliest bug, and §4.1 is the test that catches it.

**Store per `(t, i)`**, not per `t`: `v`, `f`, `k`, and `a^{(i)}`/`P^{(i)}`. The
smoother needs the within-period intermediates, and this is the one real cost of
the sequential treatment — `N` times the storage of the conventional form.
For 11.2's fixture B that is 250 × 6 × (1 + 1 + r) with r = 2, which is nothing;
record the growth in the docstring anyway so a future large-N caller is warned.

### 2.3 Signature

```julia
"""
    kalman_filter(ssm::TimeVaryingSSM, Y::AbstractMatrix{<:Real},
                  a0::AbstractVector{<:Real}, P0::AbstractMatrix{<:Real})
        -> (loglik, v, F, converged)
```

`Y` is **observations × variables** (`n × N`), matching the Stage 10 data-layout
decision (charter §6.1 of the Stage 10 document, carried forward). `v` and `F` are
returned as `n × N` matrices so a caller can form standardised residuals per
series; a missing observation gives `NaN` in both, exactly as the univariate
method already does.

`loglik` is the **full, inclusive** Gaussian log-likelihood — no concentration,
matching the existing `TimeVaryingSSM` univariate method and statsmodels'
default. The existing docstring already explains why concentration is
unavailable once `Q_t`/`H_t` carry absolute variances; the same argument applies
verbatim.

Dispatch is on `AbstractMatrix`, so **every existing `AbstractVector` call site
is untouched** and recompiles identically. §4.6 is the benchmark that proves it.

### 2.4 Missing data

`y_t[i] === NaN` means series `i` is unobserved in period `t`. The loop skips it:
no update, no likelihood contribution, and `v[t,i] = F[t,i] = NaN`.

An **entirely** missing period is then automatically prediction-only, with no
special case needed — which is the elegance of the sequential form, and is why
the conventional path needs `select_missing` with three buffers in statsmodels
([S] `_representation.pyx.in:621,653,663`) and Armadillo submatrix extraction in
`dfms`.

The reference behaviour to match, measured on `fixtureA_miss.csv` (12 blanked
cells, including one fully-missing row at t=150 and a 3-run in column 2):

| | llf |
|---|---|
| complete data | **−1049.5334080883072** |
| with the 12 NaNs | **−1035.826629949621** |

[V] `s11/dfm/missing.py`. The likelihood **rises**, correctly, because 12 fewer
density terms accumulate. Per contributing observation it is stable:
−1.04953341 against −1.04840752 across 1000 and 988 contributions.

The monotonicity ladder is worth having as a test on its own (random blanking,
seed 42):

| cells blanked | 0 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| llf | −1049.533408 | −1039.236191 | −1029.026754 | −1005.821324 | −945.217442 |
| per contributing obs | −1.049533 | −1.049734 | −1.055412 | −1.058759 | −1.050242 |

[V] same script.

### 2.5 The degenerate-`f` guard — and why it is not optional

**Both references fail silently here.** Measured, with all idiosyncratic variances
set to σ²:

| σ² | conventional matrix filter | sequential filter |
|---|---|---|
| 1e-2 | −13472.63748807269 | −13472.637488051558 |
| 1e-6 | −137368702.5288942 | — |
| **1e-12** | **−138551932441403.25** | **−696.1687714342193** |
| exactly 0 | −696.1687714342186 | −696.1687714342186 |

[V] `s11/dfm/singular.py`

The sequential path degrades gracefully to the `H → 0` limit where the
conventional path returns garbage fourteen orders of magnitude wrong — and
**neither reference raises**. `dfms` guards only with `if(detS > 0) loglik += …`,
silently skipping that period's contribution (charter defect D5).

**Ours throws.** `f ≤ 0` is `ArgumentError` naming the period, the series index
and the computed `f`. This is the honest-refusal principle applied to a case
where we have both references' behaviour measured and both are wrong.

For the *optimiser*'s sake there must also be a non-throwing path, because an
optimiser will explore parameter space that produces a degenerate `f` and should
get `-Inf` rather than an exception. Follow the convention `TimeVaryingSSM`
already set: `converged=false`, `loglik=-Inf`, `v`/`F` empty. The throwing form
is for a direct call with user-supplied parameters; the sentinel is for the
likelihood inside `_optimize`. **The existing univariate method already draws
this distinction** — match it exactly rather than inventing a second convention.
[S] `timevaryingssm.jl` docstring

### 2.6 Cost

Per period, with `N` series and `r` states:

| | sequential | conventional |
|---|---|---|
| per period | **O(N r²)** | O(N³ + N² r + N r²) |
| factorisation | none | Cholesky of an N×N matrix |
| missing data | skip in loop | submatrix extraction |

No `N³` term and no factorisation. For N=1 it reduces to O(r²), which is exactly
the existing scalar filter.

---

## 3. Item 11.0b — the multivariate smoother

```julia
    kalman_smoother(ssm::TimeVaryingSSM, Y::AbstractMatrix{<:Real},
                    a0::AbstractVector{<:Real}, P0::AbstractMatrix{<:Real})
        -> (alpha, V, eta, eta_var, eps, eps_var, converged)
```

Same return tuple as the univariate method, because callers should not have to
branch. `eps`/`eps_var` become `n × N` matrices.

The backward recursion runs over the **within-period** intermediates in reverse:
for period `t`, iterate `i = N … 1` using the stored `v[t,i]`, `f[t,i]`,
`k[t,i]`, accumulating `r` and `N` (the smoothing recursion's own quantities —
unfortunate notation collision with the number of series; **use `rvec` and
`Nmat` in the code** and say so in a comment).

This is the single most intricate piece of code in Stage 11.0. Two things make it
tractable:

1. **The N=1 reduction (§4.1) tests it against an already-verified
   implementation.** The existing univariate smoother has been checked against
   statsmodels; if the multivariate one agrees with it bit-for-bit at N=1, the
   backward recursion's structure is right and only the `i` loop can be wrong.
2. **Disturbance smoothing gives a second, independent check**: the smoothed
   disturbances must satisfy `alpha_{t+1} = T_t alpha_t + R_t eta_t` exactly.
   Assert it. This catches a mis-ordered `i` loop that a state-only comparison
   would miss.

**Write it for clarity, not speed.** The smoother runs once per fit; the filter
runs thousands of times (charter §7.4).

---

## 4. Verification plan

No public reference API computes "the sequential filter", so verification is by
reduction and by agreement with the conventional form.

### 4.1 Reduction to the univariate case — **bit-exact, not approximate**

Build a `TimeVaryingSSM` with `N = 1`. Feed the same system and the same
`a0`/`P0` to:

- `kalman_filter(ssm, y::Vector, a0, P0)` — the existing method
- `kalman_filter(ssm, reshape(y, :, 1), a0, P0)` — the new method

**Assert exact equality** on `loglik`, `v`, `F`. At `N = 1` the sequential
treatment *is* the scalar recursion with one pass through the inner loop, so there
is no floating-point reason for any difference. A tolerance here would hide a bug.

Same for the smoother, on all seven returned quantities.

This is the most valuable test in 11.0 because it leverages everything already
verified about the univariate path.

### 4.2 Agreement with statsmodels at fixed parameters — tier 1

The charter's §8.1 tier-1 target. On `fixtureA.csv` (200×5, r=1) with the
parameters below, the likelihood must match **−1049.5334080883072** to 1e-10.

Those parameters, from a `DynamicFactor` fit on the standardised fixture [V]
`s11/chk_dfm.py`:

| | value |
|---|---|
| Λ | 0.70090792, 0.60429241, −0.57537284, 0.52373624, 0.66507451 |
| A (factor AR(1)) | 0.6482473 |
| Q | 1 (fixed by the reference's normalisation) |
| diag(H) | 0.14812824, 0.36549749, 0.42432314, 0.52214416, 0.23249862 |
| `a0` | 0 |
| `P0` | stationary: `Q/(1−A²)` |

**Two independent corroborations of this likelihood already exist**, which is
unusually good for a tier-1 target:

1. `dfms`' own Armadillo filter `SKF`, run at these same parameters with the
   stationary `P0`, returns **−1049.53340812981** — agreement to **1.4e-10**
   with statsmodels' **−1049.5334081297972**. [V] Two independently written
   filters, one in Python/Cython and one in C++, agreeing to 1e-10 is strong
   evidence the *quantity* is right and not just reproducible.
2. An independent re-fit from the CSV in this session reproduced
   **−1049.5334080883072** — a 4.1e-08 difference from the agent's figure, which
   is optimiser-level, not filter-level. [V]

**This distinction is the charter's §8.1 in miniature and must be respected in
the test:** assert the likelihood **at the fixed parameter vector** to 1e-10;
assert the *fitted optimum* to 5e-3 relative. They are different claims.

### 4.3 Agreement with the conventional form

Implement the conventional matrix-`F_t` filter **as a test helper only**, not in
`src/`. At ordinary parameters the two must agree to **3.2e-12**, which is the
measured agreement between statsmodels' two paths [V]. Then assert the
divergence at σ² = 1e-12 (§2.5) — the sequential path stays near −696.17 while
the conventional one blows up.

Keeping the conventional form in `test/` rather than `src/` is deliberate: it
earns its keep as an independent check and nowhere else.

### 4.4 Missing data

The three figures from §2.4 — complete, 12-NaN, and the five-point monotonicity
ladder — each as an assertion. Plus: an entirely missing period must leave the
filtered state equal to the prediction `T_t a_t`, asserted directly.

### 4.5 Invariance checks that need no reference

- **Series permutation.** Reordering the `N` series (and `Z`, `H` to match) must
  leave `loglik` unchanged to machine precision. The sequential treatment makes
  this a real test rather than a tautology, since the series genuinely enter in
  order.
- **Orthogonal rotation of the state.** Replacing `(T, Z, R, Q, a0, P0)` by
  `(UTU', ZU', UR, Q, Ua0, UP0U')` for orthogonal `U` must leave `loglik`
  unchanged.
- **`H → 0` limit.** §2.5's exact-zero case gives −696.1687714342186; approaching
  it from 1e-12 must agree to ~1e-9.

### 4.6 Performance, and the no-regression proof

Mandatory, because §2.3's dispatch claim needs evidence:

1. **The univariate path is unchanged.** Benchmark `kalman_filter(ssm,
   y::Vector, …)` before and after 11.0 lands. Same allocations, same time
   within noise. This is the claim `timevaryingssm.jl`'s own docstring makes
   about `GaussianSSM` not being affected by `TimeVaryingSSM`'s existence, and
   it must be re-earned one level up.
2. **Zero allocations inside the time loop**, asserted with `@allocated`.
3. **`@code_warntype` clean** on both new methods.
4. **The O(N r²) scaling is visible.** Time the filter at N = 2, 5, 10, 20 with
   `r` fixed and record that it grows linearly in N, not cubically. This is the
   §6.2 claim; measure it rather than asserting it.

### 4.7 Testset inventory

| # | Testset | Assertions (min) |
|---|---|---|
| 1 | N=1 reduction, filter, bit-exact | 12 |
| 2 | N=1 reduction, smoother, all seven returns, bit-exact | 20 |
| 3 | fixture A likelihood at fixed parameters, 1e-10 | 6 |
| 4 | fixture B likelihood at fixed parameters | 6 |
| 5 | agreement with the conventional helper, ordinary parameters | 10 |
| 6 | divergence from the conventional helper at σ²=1e-12 | 6 |
| 7 | missing data: 12-cell pattern | 8 |
| 8 | missing data: monotonicity ladder | 10 |
| 9 | missing data: entirely missing period is prediction-only | 8 |
| 10 | series-permutation invariance | 6 |
| 11 | state-rotation invariance | 6 |
| 12 | `H → 0` limit | 6 |
| 13 | degenerate `f` throws, with the message naming t and i | 8 |
| 14 | degenerate `f` returns the sentinel in the optimiser path | 6 |
| 15 | dense `H` throws | 4 |
| 16 | disturbance-smoother identity `alpha_{t+1} = T alpha_t + R eta_t` | 10 |
| 17 | partitioned initialisation (§5) | 14 |
| 18 | stationary VAR reparameterisation round-trip (§6) | 16 |
| 19 | `_ols` zero-weight behaviour (§7) | 12 |
| 20 | performance: allocations, `@code_warntype`, N-scaling | 8 |

**20 testsets, ≥ 200 assertions.** Below the charter's 250 floor, which is
correct for an item with no public API — the assertion weight belongs in 11.1
and 11.2, which exercise this code through a user-facing fit.

---

## 5. Item 11.0c — partitioned initialisation

### 5.1 Why

Charter §5.1, finding 2: statsmodels partitions a mixed model's initialisation.
Measured on a full UC specification (level, trend, seasonal(4), damped stochastic
cycle, AR(1)), `k_states = 8`:

```
blocks: {(0,1,2,3,4,5,6): 'approximate_diffuse', (7,): 'stationary'}
approximate_diffuse_variance = 1000000.0
```

[V] `s11/uc_init.py`

The non-stationary components get a diffuse prior; the stationary AR component
gets its own stationary covariance. **11.1 cannot reproduce the reference
without this**, and the existing API offers only a single `(a0, P0)` pair or
`kalman_filter_diffuse`'s flat `diffuse_idx`.

### 5.2 Design

```julia
struct SSMInit
    blocks::Vector{Pair{UnitRange{Int},Symbol}}   # state range => scheme
    kappa::Float64                                 # approximate-diffuse variance
    a0::Vector{Float64}
    P0_known::Union{Nothing,Matrix{Float64}}
end
```

with schemes `:known`, `:stationary`, `:approximate_diffuse`, `:exact_diffuse`,
and a resolver `_resolve_init(ssm, init) -> (a0, P0, diffuse_idx)` that returns
what the existing filters already accept. **The filters do not change.** This is
a constructor for their arguments, which keeps the change surface small and means
`kalman_filter_diffuse` continues to work exactly as it does today.

κ defaults to **1e6**, matching the reference.

`:stationary` on a sub-block needs `stationary_cov` to work on a submatrix rather
than a whole `GaussianSSM`. Factor its doubling iteration into an internal
`_stationary_cov(T, RQR)` taking plain matrices; `stationary_cov(::GaussianSSM)`
becomes a one-line wrapper and **keeps its current behaviour and docstring**.
The existing performance note in that docstring — doubling at ~1 ms against
~2 s for the `kron` solve at r=54, a measured 1874× [S] — stays true and stays
relevant, because a sub-block of a seasonal model can still be large.

### 5.3 Guards

- Blocks must **partition** `1:r` exactly — no gap, no overlap. Throw naming the
  offending index otherwise.
- `:stationary` on a block whose `T` sub-block has a unit root must throw, not
  return a huge covariance. The doubling iteration's `converged=false` flag
  already exists for this; use it.
- `:known` requires `P0_known`; throw if absent.

---

## 6. Item 11.0d — stationarity-enforcing VAR reparameterisation

### 6.1 Why

11.2 estimates a factor VAR by unconstrained optimisation, which will wander into
the non-stationary region, where the stationary `P0` does not exist. statsmodels
spends **98 + 91 lines** on `transform_params`/`untransform_params` for exactly
this [S] `dynamic_factor.py`.

11.3 needs the same thing for a switching AR, and 11.4's LSTAR does not (its
regimes may each be explosive as long as the system is ergodic — **do not apply
this there**).

### 6.2 What to build

An Ansley–Kohn style mapping `_constrain_stationary(u) -> A` from unconstrained
`ℝ^{p k²}` to stationary VAR coefficient blocks, and its inverse. For `k = 1` it
must reduce to the familiar `A = u/√(1+u²)` partial-autocorrelation mapping, and
that reduction is the cheapest test.

**It must be ForwardDiff-differentiable**, because `_optimize` passes `Dual`
numbers through the objective and this transform sits inside it. That rules out
any branch on a comparison with a float, and it is the reason this cannot simply
call an iterative projection.

### 6.3 Tests

- **Round-trip**: `_unconstrain(_constrain(u)) ≈ u` to 1e-12, over 1000 random
  draws at `k = 1, 2, 3` and `p = 1, 2`.
- **Stationarity**: `maximum(abs, eigvals(companion(_constrain(u))))` < 1 for all
  1000 draws at each `(k, p)`. Not "usually" — all of them.
- **`k=1` reduction** to `u/√(1+u²)`, exact.
- **ForwardDiff**: the gradient of `sum(_constrain(u))` is finite and matches a
  central-difference approximation to 1e-7.
- **The stationary covariance exists** for every constrained draw —
  `_stationary_cov` returns `converged=true`.

---

## 7. Item 11.0e — `_ols` and zero weights

### 7.1 The defect

`src/unitroot.jl:58` and `:172` both assert:

```julia
all(>(0), weights) || throw(ArgumentError("weights must be positive"))
```

[V] read at HEAD `ad0010e`

11.3's EM M-step is weighted least squares with smoothed regime probabilities as
weights, and **those hit exact zero routinely**: MSwM's own initial distribution
converged to `[0, 1]` on the canonical fixture, and statsmodels clamps at 1e-20
rather than allowing zero [V]. So `_ols` as written throws in the middle of a
perfectly healthy EM iteration.

### 7.2 The fix, and the thing to be careful about

Relax to `all(>=(0), weights)` **and** require `sum(weights) > 0` and at least
`k` strictly positive weights — because a weight vector with fewer than `k`
non-zeros gives a singular system, and the honest-refusal principle says that
should throw with a message naming the shortfall, not fail inside the
factorisation.

A zero weight must mean *the row is excluded*, which is what `sqrt.(weights)`
scaling already achieves. The only real change is the guard.

**Do not instead clamp to 1e-20 the way statsmodels does.** A clamped row still
contributes to `X'X` and to the degrees of freedom; an excluded row does not. The
difference is invisible at 1e-20 and would be a silent approximation, which §2 of
the charter rules out.

### 7.3 Tests

- Zero weights exclude rows: `_ols(X, y; weights=w)` with `w[i] = 0` equals
  `_ols(X[setdiff,:], y[setdiff])` **exactly** (coefficients, residual vector on
  the retained rows, and standard errors).
- Fewer than `k` positive weights throws, with the count in the message.
- All-zero weights throws.
- Negative weights still throw.
- The existing `weights` call sites — the weighted paths in `unitroot.jl` — are
  unchanged. Re-run their existing tests; this is a widening, so nothing should
  move.

---

## 8. Ordering within 11.0

1. **§7, `_ols`** — three lines, unblocks 11.3, no dependencies. Do it first.
2. **§5, partitioned initialisation** — needed by 11.1, which the charter puts
   first overall. Note this means **11.1 can start after §5 and §7 only**; it
   does not need the multivariate filter at all.
3. **§2, the multivariate filter** — the substantial piece.
4. **§3, the multivariate smoother** — after §2 is passing its reductions.
5. **§6, the stationary reparameterisation** — needed only by 11.2.

So the honest sequencing is: **11.0 §5 + §7 → 11.1 → 11.0 §2 + §3 + §6 → 11.2**,
with 11.3 and 11.4 slotting in anywhere after §7. The charter's "build 11.1
first" and this document's "11.0 is a prerequisite" are both true because 11.0
splits.

---

## 9. Budget

| Piece | Reference size | Julia estimate |
|---|---|---|
| Multivariate filter (sequential) | `dfms` SKF 106 lines Armadillo; statsmodels a whole Cython subpackage | 150–200 |
| Multivariate smoother | `dfms` FIS 65 + SKFS 133 = 198 lines | 180–250 |
| Partitioned initialisation | statsmodels `initialization.py`; `initialize_default` 43 lines | 80–120 |
| Stationary reparameterisation | 98 + 91 = 189 lines | 120–160 |
| `_ols` guard | — | 5 |
| **Total** | | **~550–750 lines** |

Plus roughly the same again in tests, which is the right ratio for code with no
public API and four downstream dependants.

---

## 10. What this item explicitly does not build

- **The exact diffuse smoother.** 11.1 does not need it (charter §9.1) and
  nothing else in Stage 11 does either. `kalman_filter_diffuse` stays
  filter-only. If a future item needs it, Durbin & Koopman ch. 5's smoothing
  recursions are the reference and this is the place to add it.
- **The conventional matrix-`F_t` filter in `src/`.** It lives in `test/` as a
  check (§4.3).
- **A dense-`H` path.** §2.2 throws instead.
- **Mixed-frequency state construction.** `DynamicFactorMQ` spends 210 + 171
  lines in two constructors purely on that bookkeeping [S]; it is out of scope
  for 11.2 and therefore out of scope here.
