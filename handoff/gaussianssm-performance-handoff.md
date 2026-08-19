# Handoff: GaussianSSM engine performance — measured findings and patches

Status: **§2 (Lyapunov doubling) already landed, during the merge, not
here.** As this document's own §10/cross-reference anticipated: the
kron-based `stationary_cov` was replaced with the doubling iteration as
part of completing `stage-6-arima-handoff.md`'s Part A merge (its §3.2
flagged the same fix as a correctness-adjacent should-fix). Verified
against the real 364-case bulk corpus post-swap, not just this
document's own measurements. **§3 (companion structure), §4 (steady-state
freezing), and §5 (allocation) are still untouched** -- per §1 (Read this
first) and §7's own protocol, correctly deferred until Stage 6.6 has a
real `fit_arima` to benchmark against, not attempted opportunistically
during the Part A merge. Re-read §1 before starting the rest of this
document.

---

For a fresh Claude Code session picking this up with no prior context.
Companion document to `stage-6-arima-handoff.md`. That document covers
correctness and the ARIMA fitting layer; this one covers **speed only**.

## Read this first: when to do this work

**Not now.** Sequence it after the merge lands and `Pkg.test()` is green
inside TSAnalytics — §2.1 and §2.2 of the Stage 6 handoff (concrete
types, `seasonal_poly` guard) are correctness fixes and come first.
Optimizing a module that is still moving is how you end up debugging two
things at once.

The natural window is **after Stage 6.6** (`fit_arima` matching R
end-to-end, non-seasonal) and **before Stage 6.8** (seasonal end-to-end).
By then the engine's interface is stable, and you'll have a real fitting
loop to benchmark against rather than a synthetic one. Doing it before
6.8 matters because the seasonal path is the only place where any of
this is measurable — see §1.1.

## Scope boundary

This document proposes **no change to the mathematics**. Every
optimization below is an exact algebraic restructuring or a
tolerance-controlled early exit. Nothing here trades accuracy for speed,
and §7 shows the existing 364-case dual-verified corpus re-certifies all
of it without new ground truth being generated.

---

## 1. The measurement basis

All numbers below come from executing the algorithm as written in
`gaussianssm.jl`. Timings for the Lyapunov solve and step counts for
steady-state convergence are **measured**; per-step filter costs are
**flop counts**, not wall-clock. Julia will differ from any transcription
in constant factor — it will not differ in scaling, which is what every
recommendation here rests on. Re-measure in Julia with `BenchmarkTools`
(test-only dependency) before and after each change.

### 1.1 Where the problem is — and is not

| Model | r | per-fit cost, n=300, 200 evals |
|---|---|---|
| ARMA(2,2) | 3 | 0.04 s |
| SARIMA (1,1)×(1,1)₁₂ | 14 | 0.57 s |
| SARIMA (1,1)×(2,2)₁₂ | 26 | 27 s |
| SARIMA (1,1)×(1,1)₅₂ | 54 | 438 s |

**Non-seasonal ARMA has no performance problem and should not be
touched.** State dimension `r = max(p*, q*+1)` stays at 2–4, everything
fits in cache, and the current dense code is fine. Every optimization
below is dead weight there.

The cost is entirely driven by `r`, which for a seasonal model is
`≈ s·max(P, Q+1)`. That is precisely the regime `SeasonalAdjustment.jl`
lives in — monthly X-13/SEATS work is `s=12`, and weekly series are
`s=52`. A 7-minute likelihood-evaluation budget for a weekly SARIMA is a
real blocker for that downstream package, not a micro-optimization.

---

## 2. Optimization 1 — replace the `kron` Lyapunov solve

**Impact: 5×–9400×. Do this one first.** It is the largest single win,
the smallest diff, and the easiest to validate.

`stationary_cov` solves `vec(Q0) = (I − T⊗T)⁻¹ vec(RRᵀ)` by building a
dense `r²×r²` matrix — O(r⁶) time, O(r⁴) memory. Measured:

| Model | r | kron | doubling | speedup | agreement | kron memory |
|---|---|---|---|---|---|---|
| ARMA(2,2) | 3 | 0.2 ms | 0.2 ms | 1× | 1.6e-16 | 0.0 MB |
| SARIMA (1,1)×(1,1)₄ | 6 | 0.2 ms | 0.2 ms | 1× | 3.1e-16 | 0.0 MB |
| SARIMA (1,1)×(1,1)₁₂ | 14 | 1.2 ms | 0.3 ms | 5× | 1.7e-16 | 0.3 MB |
| SARIMA (2,2)×(1,1)₁₂ | 15 | 8.0 ms | 0.3 ms | 27× | 1.3e-16 | 0.4 MB |
| SARIMA (1,1)×(2,2)₁₂ | 26 | 125 ms | 0.6 ms | 200× | 1.5e-16 | 3.7 MB |
| SARIMA (1,1)×(1,1)₅₂ | 54 | 2.09 s | 1.1 ms | 1874× | 1.3e-16 | 68 MB |
| SARIMA (1,1)×(2,2)₅₂ | 106 | 38.8 s | 4.1 ms | 9414× | 7.6e-17 | 1010 MB |

The replacement is the doubling (Smith) iteration: `Q ← Q + A·Q·Aᵀ`,
`A ← A²`, which converges quadratically because `A` becomes `T^(2^k)`.

```julia
"""
    stationary_cov(ssm; tol=1e-15, maxiter=60) -> (Q0, converged)

Stationary state covariance solving `Q0 = T*Q0*T' + R*R'`, via the
doubling (Smith) iteration rather than an explicit `(I - T⊗T)` solve.
Quadratically convergent: `A` doubles to `T^(2^k)` each pass, so ~30
iterations reach machine precision for any spectral radius meaningfully
below 1.

`converged=false` means the iteration did not settle within `maxiter` --
in practice a non-stationary or unit-root `T`. Callers must check it;
see `kalman_filter`'s guard.
"""
function stationary_cov(ssm::GaussianSSM{S}; tol=1e-15, maxiter=60) where {S}
    Q = ssm.R * ssm.R'
    A = copy(ssm.T)
    converged = false
    for _ in 1:maxiter
        Qn = Q + A * Q * A'
        all(isfinite, Qn) || return (Qn, false)
        if maximum(abs, Qn .- Q) <= tol * max(maximum(abs, Qn), one(S))
            Q = Qn
            converged = true
            break
        end
        Q = Qn
        A = A * A
    end
    return ((Q .+ Q') ./ 2, converged)
end
```

### 2.1 The failure mode changes — verify the guard

This is the one genuine risk in the swap. The `kron` version *throws*
`SingularException` on an exactly-singular system and returns a negative
`Q0[1,1]` for a non-stationary model; the current `kalman_filter` guard
(`Q0[1,1] <= 0 || !all(isfinite, Q0)`) is built around that. Doubling
instead **diverges to `Inf`/`NaN`**, or silently returns a finite garbage
value if `maxiter` is hit without convergence.

I stress-tested this. Across **9,743 non-stationary draws** (AR(1),
AR(2), AR(3), ARMA(2,2)) the existing guard rejected every single one,
with and without the `converged` flag enforced — **0 silent acceptances
either way**. And the iteration stays accurate right up to the boundary:

| φ | converged | Q0 | exact 1/(1−φ²) | rel. error |
|---|---|---|---|---|
| 0.9 | true | 5.263158e+00 | 5.263158e+00 | 1.7e-16 |
| 0.99 | true | 5.025126e+01 | 5.025126e+01 | 2.8e-16 |
| 0.999 | true | 5.002501e+02 | 5.002501e+02 | 5.8e-15 |
| 0.9999 | true | 5.000250e+03 | 5.000250e+03 | 1.1e-13 |
| 0.99999 | true | 5.000025e+04 | 5.000025e+04 | 1.2e-13 |

**Still return and check the `converged` flag.** It cost nothing in the
sweep, but it is the difference between "provably rejected" and
"empirically rejected on the cases I happened to draw", and Stage 6's
optimizer will explore parameter space far more adversarially than a
uniform sweep does. Treat the flag as belt-and-braces, not redundancy.

Port the existing non-stationarity tests from `test_gaussianssm.jl`
unchanged, and add the `converged`-flag assertion alongside them.

### 2.2 No new dependency

`MatrixEquations.jl` provides `lyapd` (Bartels–Stewart) and would also
solve this. Don't take it — the doubling iteration is ~12 lines of
`LinearAlgebra`-backed matrix operations, consistent with both the
"LinearAlgebra over hand-rolled loops" principle (these are BLAS-level
products, not element loops) and the "no unused dependencies" policy.
A whole matrix-equations package for one function is a bad trade in a
package aiming to stay lean.

---

## 3. Optimization 2 — exploit the companion structure of `T`

**Impact: exactly `r`× on the dominant per-step cost.**

`build_statespace` constructs `T` in Harvey companion form: first column
is `phi_ext`, superdiagonal is 1, everything else zero. The filter then
hands it to generic dense `*`, paying O(r³) for a product that is
structurally O(r²):

```
(T·X)[i, :] = phi[i]·X[1, :] + X[i+1, :]      (X[r+1, :] := 0)
```

| Model | r | flops/step now | companion | reduction |
|---|---|---|---|---|
| SARIMA (1,1)×(1,1)₁₂ | 14 | 10,976 | 784 | 14× |
| SARIMA (1,1)×(2,2)₁₂ | 26 | 70,304 | 2,704 | 26× |
| SARIMA (1,1)×(1,1)₅₂ | 54 | 629,856 | 11,664 | 54× |

I verified the fast form is numerically identical (`allclose`, exact) to
the dense product for every model in the table.

```julia
"""Companion-form `T * X`, O(r^2) instead of O(r^3). `out` must not alias `X`."""
function _companion_mul!(out::AbstractMatrix, phi::AbstractVector, X::AbstractMatrix)
    r = length(phi)
    @inbounds for j in 1:r
        x1 = X[1, j]
        for i in 1:(r-1)
            out[i, j] = phi[i] * x1 + X[i+1, j]
        end
        out[r, j] = phi[r] * x1
    end
    return out
end

"""Companion-form `T * x`, O(r). `out` must not alias `x`."""
function _companion_mul!(out::AbstractVector, phi::AbstractVector, x::AbstractVector)
    r = length(phi)
    @inbounds x1 = x[1]
    @inbounds for i in 1:(r-1)
        out[i] = phi[i] * x1 + x[i+1]
    end
    @inbounds out[r] = phi[r] * x1
    return out
end
```

For the covariance update, `T·P·Tᵀ = (T·(T·P)ᵀ)ᵀ`, so two companion
applications and a transpose replace two dense O(r³) products. `P` is
symmetric, so the result is too.

**Aliasing is the trap here.** Both methods read `X[i+1, j]` while
writing `out[i, j]`; if `out === X` the loop corrupts as it goes. Assert
it, or always write into a distinct buffer.

### 3.1 Do not inline this assumption

The companion structure holds for Stage 6 and Stage 7, and **breaks at
Stage 8** when time-varying `T_t` arrives. Implement it as a specialized
method dispatching on a structured type — e.g. keep `phi` on the
`GaussianSSM` struct and dispatch — rather than hardcoding the shortcut
into `kalman_filter`'s body. Stage 8 then adds a general dense fallback
without unpicking this work.

---

## 4. Optimization 3 — freeze the steady state

**Impact: collapses 60–98% of the loop to O(r).**

The model is time-invariant, so `P_t` converges to a fixed point of the
Riccati recursion. After that, `F` and `K` are constant and every
remaining step is:

```
v = y[t] - a[1];   a = T·a + K·v
```

— no matrix operations at all. Measured steps to convergence (relative
tolerance 1e-12):

| Model | r | steps | % of n=300 | % of n=1000 |
|---|---|---|---|---|
| ARMA(1,1) | 2 | 16 | 5.3% | 1.6% |
| ARMA(2,2) | 3 | 17 | 5.7% | 1.7% |
| SARIMA (1,1)×(1,1)₄ | 6 | 34 | 11.3% | 3.4% |
| SARIMA (1,1)×(1,1)₁₂ | 14 | 99 | 33.0% | 9.9% |
| SARIMA (1,1)×(2,2)₁₂ | 26 | 135 | 45.0% | 13.5% |
| SARIMA (1,1)×(1,1)₅₂ | 54 | 419 | 139.7% | 41.9% |

The current code is doing full covariance algebra for the entire
remainder in every one of those cases.

**There is precedent, which matters for defensibility.** `statsmodels`'
Kalman filter exposes a `tolerance` parameter doing exactly this — it
stops updating the covariance once converged. This is not an exotic
shortcut invented here; it is standard state-space practice for
time-invariant models, and citable as such.

Two requirements:

- **Make the tolerance a keyword, defaulting conservatively** (1e-12
  relative, or stricter). Someone validating against R must be able to
  set `tol=0` to disable freezing entirely and get the unmodified
  recursion.
- **Note the r=54 row.** At n=300 the model never converges before the
  series ends, so freezing gains nothing there — it is not a
  regression, but don't expect the win at short series with high
  seasonal dimension.

---

## 5. Optimization 4 — allocation, with a real caveat

The current loop allocates roughly five fresh `r×r` temporaries per
step (`T*P`, `*T'`, `K*K'`, `R*R'`, the subtraction):

| Model | n | allocated per likelihood evaluation |
|---|---|---|
| SARIMA (1,1)×(1,1)₁₂ | 300 | 2.4 MB |
| SARIMA (1,1)×(1,1)₁₂ | 1000 | 7.8 MB |
| SARIMA (1,1)×(2,2)₁₂ | 1000 | 27.0 MB |
| SARIMA (1,1)×(1,1)₅₂ | 1000 | 116.6 MB |

At 200 optimizer iterations that's ~1.5 GB of churn for a monthly model
— GC pressure, not arithmetic.

Two easy, unconditionally safe pieces:

- `sum(log.(F))` allocates an n-vector for nothing. Accumulate `logF`
  scalar-wise in the loop. Same for the `sigma2` generator sum. This
  also very slightly *improves* accuracy — see §7.
- `R*R'` is loop-invariant. Hoist it out; it's recomputed n times today.

**The caveat: buffer preallocation fights ForwardDiff.** Stage 4.1's
`_optimize` is built on `AutoForwardDiff()`, so the objective will be
called with `Vector{ForwardDiff.Dual}`. Preallocated `Float64` buffers
will either throw or silently drop the derivative information. Options:

1. Type the buffers to the promoted element type inside the function
   (allocates once per call, not once per step — still ~99% of the win).
2. `PreallocationTools.jl` — solves exactly this, but is a new
   dependency.
3. Skip it.

**Recommendation: option 1, and only after §2–§4 are landed and
measured.** Optimizations 1–3 are all AD-safe and capture nearly all the
available gain; preallocation is the tail. Don't take a new dependency
for it.

---

## 6. Combined effect

Per fit, n=300, 200 optimizer evaluations. Lyapunov times measured; filter
times from the flop model.

| Model | r | now | optimized |
|---|---|---|---|
| ARMA(2,2) | 3 | 0.04 s | 0.04 s |
| SARIMA (1,1)×(1,1)₁₂ | 14 | 0.57 s | 0.07 s |
| SARIMA (1,1)×(2,2)₁₂ | 26 | 27.1 s | 0.16 s |
| SARIMA (1,1)×(1,1)₅₂ | 54 | 437.9 s | 0.57 s |

---

## 7. Validation — the existing corpus re-certifies all of it

**This is the strongest argument for doing the work.** No new ground
truth needs generating.

I built a combined fast filter (companion structure + steady-state
freezing + running log-sum) and ran it against all 364 cases in
`test_gaussianssm_bulk.jl`:

```
cases run: 364   passing at their own tolerance: 364
worst deviation from the R+Python ground truth: 2.440e-07  (tolerance 1e-4)
```

The current implementation's worst deviation is 2.454e-07. The optimized
version is *marginally better*, because the running scalar accumulation
avoids one pass of summation error over the `log(F)` vector. That is a
genuine, if tiny, accuracy improvement — not a regression to be
explained away.

### 7.1 Protocol for each change

Land the four optimizations as **separate commits**, each one:

1. Rerun the full 364-case bulk suite plus the 4 smoother cases. All
   must pass at their existing tolerances — do not loosen a tolerance to
   make an optimization pass. If a tolerance needs loosening, the
   optimization is wrong.
2. Rerun the non-stationarity rejection tests (§2.1).
3. Record before/after `BenchmarkTools` numbers in the commit message,
   for a seasonal model — the non-seasonal ones will show nothing.

### 7.2 Add an equivalence regression test

Keep the unoptimized filter available as `_kalman_filter_reference` (not
exported, not documented) and add a test asserting the fast and
reference paths agree to ~1e-12 on a handful of models spanning
`r ∈ {1, 2, 14, 26}`, with the steady-state tolerance set to both `0`
and its default. This is what catches a future regression when Stage 8
touches the filter — the bulk corpus catches wrong answers, but only an
equivalence test catches a fast path that has silently diverged from the
reference within tolerance.

---

## 8. Smoother implications

`kalman_smoother` benefits from §3 and §4 identically — its forward pass
is the same recursion. §4 additionally cuts its **storage**, since
`P_pred[t]` is constant after convergence and needn't be stored per `t`:

| Model | r | n=1000 storage now | with steady state |
|---|---|---|---|
| SARIMA (1,1)×(1,1)₁₂ | 14 | 1.8 MB | 0.3 MB |
| SARIMA (1,1)×(2,2)₁₂ | 26 | 5.8 MB | 0.9 MB |
| SARIMA (1,1)×(1,1)₅₂ | 54 | 24.2 MB | 10.2 MB |

Same for `K_all`. The backward recursion's `L = T − K·e1ᵀ` also has
companion structure and gets the same O(r²) treatment.

Worth doing at the same time rather than later — the smoother is what
SEATS and the Unobserved Components work will lean on, and it will run
on longer series than the filter typically does.

---

## 9. What not to optimize

- **Non-seasonal ARMA.** r ≤ 4. Nothing here helps; the added branching
  may marginally hurt. If a fast path costs clarity for r < 8, gate it.
- **`P = (P + P')/2` symmetrization.** Measured max relative `|P − Pᵀ|`
  over 5,000 steps: 2.2e-16. The recursion is self-stabilizing. Adding
  it is defensible as cheap insurance but must not be presented as a
  fix, and it is not a performance item either way.
- **Threading the filter loop.** It's an inherently sequential
  recursion. If parallelism is ever wanted, it belongs at the level of
  fitting many series at once (the PROC HPF pattern) — a different
  layer entirely, and not this package's concern yet.
- **`polymul` / `combined_ar_ma`.** Called once per likelihood
  evaluation on vectors of length ≤ 60. Irrelevant.

---

## 10. What to do with this

1. **Wait** until the merge is green and Stage 6.6 has one `fit_arima`
   matching R end-to-end. Do not start here.
2. Land §2 (Lyapunov doubling) first — biggest win, smallest diff. Port
   the non-stationarity tests and add the `converged`-flag assertion.
3. Land §4 (steady-state freezing) second — second-biggest win, AD-safe,
   with the tolerance exposed as a keyword.
4. Land §3 (companion structure) third, as a dispatching specialization
   so Stage 8 can add a dense fallback cleanly.
5. Land §5's two free pieces (running log-sum, hoisting `R*R'`). Defer
   buffer preallocation until measured, and don't take a dependency for
   it.
6. Apply §8 to the smoother in the same pass as §3 and §4.
7. Add the §7.2 equivalence test before considering any of it done.
8. Record in `development-sequence.md` that the engine was optimized
   post-6.6, with the before/after numbers, so the Stage 8
   generalization knows which shortcuts it must preserve or explicitly
   discard.

**Cross-reference:** `stage-6-arima-handoff.md` §3.2 flags the Lyapunov
issue as a correctness-adjacent "should-fix". If that got done during
the merge, §2 here is already complete — check before repeating it.
