# Handoff: Stage 8.1 — GaussianSSM, Part 2: Time-Varying System Matrices

**Status: done.** Implemented as `TimeVaryingSSM`/
`kalman_filter(::TimeVaryingSSM,...)`/`to_time_varying`
(`src/statespace/timevaryingssm.jl`), kept unexported like `GaussianSSM`
itself (the public shape still isn't settled). Summary (full detail in
`test/verification/timevarying/timevarying-ground-truth-transcript.txt`):

1. **§1's array-axis and 2D/3D-shape claims reconfirmed** by direct
   re-execution against real `statsmodels` 0.14.1.
2. **§3a's exact-reduction property reconfirmed** on fresh data (the
   handoff's own exact numbers weren't reproducible without its
   unstated generation script, but the *property* -- bit-identical
   2D-vs-3D agreement -- was, which is what matters).
3. **§3b's `Z_t`-varying case reproduced to full precision.**
4. **§5's explicitly flagged gap closed**: `T_t`/`R_t`/`Q_t`/`H_t`
   varying individually, none previously checked against real reference
   output, each now has a dedicated case verified against fresh real
   `statsmodels` output (`test/verification/timevarying/
   timevarying_cases.json`) -- all four match to ~1e-14/1e-15.
5. **§5's highest-value test built and run**: all 364 of Stage 6's own
   already dual-verified `GaussianSSM` cases replayed through the new
   `TimeVaryingSSM` path (`test/test_timevaryingssm_bulk.jl`,
   mechanically generated from `test_gaussianssm_bulk.jl`, not
   hand-transcribed) -- every one matches exactly. This required working
   out *why* an exact match is even possible given `GaussianSSM`'s own
   `sigma2`-concentration trick (documented in `to_time_varying`'s own
   docstring, not just asserted) -- the handoff's own §5 test sketch
   assumed the two paths would trivially agree without addressing this.
6. **§4's multiple-dispatch design confirmed not to regress `fit_arma`**
   via a timing smoke test.

For a fresh session picking this up with no prior context. **This is
the highest-stakes piece of engineering in the entire roadmap** — every
downstream item in Chapter Nine (diffuse initialization, SARIMAX,
AutoReg's full tier) and everything in Chapter Thirteen (structural
time series, dynamic factor models) sits directly on top of this. A
subtle bug here doesn't crash anything — it silently produces plausible
SARIMAX fits with wrong coefficients, the single worst failure mode
this project has been designed throughout to avoid. Treat the
verification standard here as the floor, not a suggestion to exceed if
time allows.

## Where this fits

- **Depends on:** Stage 6.1 (the time-invariant `GaussianSSM`, already
  dual-verified). **Does not depend on Stage 6.2's diffuse initialization
  need being resolved yet** — that's Stage 8.2's separate scope. This
  handoff is scoped narrowly and deliberately: time-varying system
  matrices only, with a **known, fixed initial state** in every test
  case, so genuinely new machinery (this stage) isn't conflated with
  the next genuinely new piece (diffuse initialization). Verified
  directly by mistake, in fact — an early verification attempt used a
  non-stationary transition with `initialize_stationary()` and correctly
  produced garbage output, which is itself a preview of exactly why
  Stage 8.2 needs to exist separately.

---

## 1. Verified reference: Durbin & Koopman, and `statsmodels`'s actual implementation

Durbin & Koopman (textbook, chs. 3-4) give the general time-varying
form:
```
alpha_{t+1} = T_t * alpha_t + R_t * eta_t,   eta_t ~ N(0, Q_t)
y_t         = Z_t * alpha_t + eps_t,          eps_t ~ N(0, H_t)
```
— every system matrix indexed by `t`, versus Stage 6's fixed `T`, `R`,
`Q`, `Z`, `H`.

**`statsmodels.tsa.statespace.representation.Representation`/
`KalmanFilter`** is the actual, inspectable implementation this is
built on for real systems like SARIMAX — confirmed via direct
construction, not just documentation:
```python
KalmanFilter(k_endog, k_states, nobs, design=None, obs_cov=None,
             transition=None, selection=None, state_cov=None, ...)
```
`design`=`Z`, `transition`=`T`, `selection`=`R`, `state_cov`=`Q`,
`obs_cov`=`H` — standard Durbin-Koopman naming.

**Confirmed array-axis convention, by direct construction and shape
inspection** — this is the single easiest thing to get backwards
silently, so it was verified by building a real example rather than
trusted from documentation:
```python
mod['design'] = np.zeros((k_endog, k_states, n))   # shape (1, 2, 6) for a real test case
mod['design', 0, 1, t] = x[t]   # time-varying loading, TIME IS THE LAST AXIS
```
**Time is always the *last* axis** of a 3D system matrix array —
`(k_endog, k_states, nobs)` for `design`, `(k_states, k_states, nobs)`
for `transition`, etc. Confirmed directly: `design[:,:,0]` and
`design[:,:,2]` returned the expected per-period slices for a
hand-constructed time-varying regressor-loading test.

**Confirmed `statsmodels` itself stores time-invariant matrices as
plain 2D arrays, not degenerate 3D arrays with `nobs=1`** — a real
time-invariant construction (`mod['design']` before any time-varying
assignment) has shape `(1, 2)`, not `(1, 2, 1)`. This is not just a
convenience — it means the reference implementation itself avoids
paying for time-varying generality in the common constant case, which
directly informs section 4's design.

## 2. Documented reference: R (not independently executed — same CRAN boundary as throughout this project)

Base R's `stats::arima` internals are time-invariant only (confirmed
back in Stage 6's own verification work). The general time-varying case
in R lives in separate packages — `dlm` (Petris) and `KFAS` (Helske) are
the two standard ones, neither installable here (CRAN unreachable,
confirmed by the same failed-install pattern as `forecast`/`lmtest`
throughout this project). Their general structure (time-indexed system
matrices, similar Durbin-Koopman-derived notation) is well-established
and stable in the literature, but no specific numbers from either are
claimed here as independently verified — `statsmodels`'s real, executed
output (sections 1 and 3) is the primary verification target for this
handoff, not a documented-only R citation.

---

## 3. The two things that were actually verified by execution, not assumed

### 3a. Exact reduction to the time-invariant case — the most important test in this whole handoff

Built two `KalmanFilter` instances on identical AR(1) data (phi=0.6,
n=100, seed=3): one using Stage 6's exact time-invariant 2D-matrix
convention, one using the general 3D time-varying convention with the
*same* values simply repeated across all 100 periods. **Result: exact,
bit-identical agreement.**
```
Time-invariant (2D) path: loglik = -147.7115960747244
Time-varying (3D, constant-valued) path: loglik = -147.7115960747244
Max abs diff in filtered_state: 0.0
loglik diff: 0.0
```
**This is the regression guard that protects every already-verified
GaussianSSM result** (the original AR(1)/AR(2)/MA(1) ground truth, the
364-case bulk suite, the smoother's dual-verified cases) — see section
5 for how to actually reuse all of that existing verification work here
essentially for free.

### 3b. A genuine time-varying case, real ground truth

A local-level-plus-time-varying-regression-coefficient system — exactly
the structural shape SARIMAX (Stage 8.3) will need, where `Z_t = [1,
x_t]` and `x_t` is a real, changing regressor value each period, not a
constant:
```python
x = [1.0, 2.0, -1.0, 0.5, 3.0, -0.5]     # the time-varying regressor
y = [2.1, 4.5, -0.8, 1.6, 6.9, -0.4]
transition = [[0.5, 0], [0, 0.5]]          # stationary, deliberately -- see section 0
state_cov = diag(0.01, 0.001);  obs_cov = [[0.1]]
initial state: known, mean=[0,0], cov=0.5*I  # explicit, not stationary-derived -- see section 0
```
Real output:
```
loglikelihood = -186.3544625437698
forecasts (one-step-ahead y): [0, 1.43181818, -0.78848842, 0.19589125, 0.6789244, 0.23150176]
filtered_state[:, :3] = [[0.9545, -0.0006, -0.0031], [0.9545, 1.5764, 0.7898]]
```
`forecasts[0] = 0` exactly is correct, not a bug — at `t=0` the state
hasn't been updated by any observation yet, so the prediction is exactly
the known prior mean (`[0,0]`) dotted with `Z_0`.

---

## 4. Design: multiple dispatch for a genuinely zero-cost time-invariant path

**The real efficiency risk with this stage**: if the generalized
time-varying filter becomes the *only* code path, and everything from
Stage 6.5 onward gets migrated to call it, the already-profiled,
already-fast time-invariant pipeline pays overhead for generality it
never uses — extra indexing, extra memory for a needlessly-allocated
time axis, possibly extra branching per iteration to check "did this
matrix change." Confirmed as a real concern, not hypothetical: even
`statsmodels`'s own Cython implementation avoids this (section 1) by
storing time-invariant matrices as plain 2D arrays rather than
degenerate 3D ones.

**Julia's actual advantage here, concretely**: multiple dispatch lets
`kalman_filter` have two genuinely separate compiled methods —

```julia
# Stage 6's existing method, UNCHANGED -- fast path, zero new overhead
function kalman_filter(ssm::GaussianSSM, y::Vector{Float64})
    # exactly the existing 6.2 implementation, untouched
end

# NEW: general time-varying method, separate type, separate compiled code
function kalman_filter(ssm::TimeVaryingSSM, y::Vector{Float64})
    # the T_t/Z_t/R_t/Q_t/H_t-indexed recursion
end
```

This isn't just "two functions with the same name" — Julia compiles
each specialization independently, so `fit_arma`'s existing objective
function (Stage 6.5, already performance-checked) calling
`kalman_filter(::GaussianSSM, ...)` gets **exactly the same machine
code it already had**, unaffected by this stage's existence. No
generic dispatch overhead, no runtime type-checking cost, nothing to
regress. This is a genuine, checkable structural advantage over how
`statsmodels`'s single Cython class has to internally branch on 2D-vs-3D
shape at the C level for the same distinction — worth stating as a real
claim to verify once both exist, not just an assumption.

### `TimeVaryingSSM`

```julia
struct TimeVaryingSSM
    T::Vector{Matrix{Float64}}   # length nobs, or length 1 if truly constant (see note)
    Z::Vector{Matrix{Float64}}
    R::Vector{Matrix{Float64}}
    Q::Vector{Matrix{Float64}}
    H::Vector{Matrix{Float64}}
    r::Int                        # state dimension
end
```
**A vector of matrices, not a 3D array** — matches Julia's own idioms
better than mimicking `statsmodels`'s `(k,k,nobs)` NumPy-array
convention directly; indexing `T[t]` reads more naturally than `T[:,:,t]`
and avoids any ambiguity about which axis is time (section 1's flagged
risk). A length-1 vector (broadcast across all `t`) is a reasonable,
cheap way to represent "this specific matrix happens to be constant
even within an otherwise time-varying system" (e.g., `Q_t` might be
constant while `Z_t` genuinely varies) — worth supporting explicitly
rather than forcing every matrix to be fully materialized per period
even when redundant.

---

## 5. Comprehensive test matrix — leveraging everything already verified, not starting from zero

### The highest-leverage tests: replay Stage 6's entire existing verification suite through the new code path

**Every dual-verified GaussianSSM case already has known-correct
ground truth** (the original AR(1)/AR(2)/MA(1) cases, the 364-case bulk
suite, the smoother's AR(2)/MA(1)/ARMA(1,1) cases). Construct a
`TimeVaryingSSM` from each existing `GaussianSSM`'s `T`/`R` (broadcasting
each as a length-1 constant vector, `Z` as the fixed `[1,0,...,0]`
selection vector repeated, `H` as zero), run it through the new
time-varying filter, and assert **exact** agreement with the original
time-invariant result:
```julia
using Test

@testset "TimeVaryingSSM reduces exactly to GaussianSSM -- full regression suite" begin
    for (ar, ma, y_data, expected_loglik) in EXISTING_GAUSSIANSSM_TEST_CASES  # from the bulk suite
        ssm = build_statespace(ar, ma)
        tv_ssm = to_time_varying(ssm, length(y_data))  # broadcasts constants across all periods
        loglik_tv, = kalman_filter(tv_ssm, y_data)
        loglik_inv, = kalman_filter(ssm, y_data)
        @test isapprox(loglik_tv, loglik_inv; atol=1e-10)     # not just close -- should be exact or near-machine-precision
        @test isapprox(loglik_tv, expected_loglik; atol=1e-4)  # and both still match the ORIGINAL R/Python ground truth
    end
end
```
This single test block, reusing the 364+ already-generated cases,
constitutes the bulk of this stage's "a lot of test cases" requirement
at essentially zero new verification cost — and it's the single most
important test in this handoff, since it's the one that would catch a
regression in everything already trusted.

### Genuinely new, time-varying-specific cases

| Case | Verified against |
|---|---|
| Time-varying regressor loading, `Z_t=[1,x_t]`, known initial state | `statsmodels` real output: `loglik=-186.3544625437698`, `filtered_state`/`forecasts` per section 3b |
| Same case, but `x_t` constant (degenerates to time-invariant-with-regression) | Should match a direct 2-state time-invariant computation of the same system |
| A single period's `Z_t` different from all others (e.g. a one-time regime marker) | Constructed and verified against a fresh `statsmodels` run -- not yet done in this handoff, flag as the next verification step |
| `T_t` varying (not just `Z_t`) -- e.g. a genuinely time-varying AR coefficient | Constructed and verified against a fresh `statsmodels` run -- also not yet done, flag explicitly rather than claim broader coverage than actually checked |

**Honest gap, stated plainly**: sections 3a/3b verify the reduction
property and one genuinely time-varying `Z_t` case. **Time-varying
`T_t`/`R_t`/`Q_t`/`H_t` specifically have not yet been independently
verified against real reference output** — only `Z_t` varying was
checked. Given this stage's stakes, do not treat those as validated by
extension just because `Z_t`'s case passed; construct and verify at
least one dedicated case per matrix before trusting the general
recursion fully.

---

## 6. What to do with this

1. Implement `TimeVaryingSSM` and its `kalman_filter` method per section
   4, as a genuinely separate dispatch, not a modification of Stage 6's
   existing method.
2. **Run the full regression suite in section 5 first**, before writing
   any new time-varying-specific test -- this is the cheapest, highest-
   value check and the one most likely to catch a real mistake early.
3. Verify the section 3b case exactly, then **construct and verify at
   least one dedicated case each for time-varying `T_t`, `R_t`, `Q_t`,
   `H_t`** -- the flagged gap in section 5 -- before considering this
   stage's core recursion trustworthy for the general case, not just
   the `Z_t`-varying one.
4. Confirm the multiple-dispatch design genuinely doesn't regress
   Stage 6.5-6.8's existing performance -- a quick `@benchmark` comparison
   of `fit_arma` before and after this stage's code exists in the same
   module, given the risk flagged in section 4.
5. Update `development-sequence.md`'s Stage 8.1 row: mark implemented,
   record the exact reduction-property confirmation and the explicitly
   flagged remaining gap (T/R/Q/H time-varying cases still needing
   dedicated verification) so it isn't mistaken for fully closed.
