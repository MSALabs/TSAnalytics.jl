# Handoff: Stage 8.2 — Kalman Filter with Exact Diffuse Initialization

**Status: done.** Implemented as `kalman_filter_diffuse`
(`src/statespace/diffuseinit.jl`), kept unexported like `GaussianSSM`/
`TimeVaryingSSM`. Summary (full detail in
`test/verification/diffuseinit/diffuseinit-ground-truth-transcript.txt`):

1. **The algorithm itself was read directly from `statsmodels`'s own
   real Cython source** (`_filters/_univariate_diffuse.pyx.in`, fetched
   from GitHub this session), not reconstructed from memory of the
   textbook alone — this resolved a real ambiguity the handoff's own
   description doesn't spell out: the diffuse-phase per-observation
   likelihood contribution is `-0.5*log(2π·F_infty)` only, no `v²/F`
   term at all.
2. **§3's central finding reconfirmed** (own regenerated data, since
   the handoff's own `y`/`x` weren't fully specified): the exact method
   agrees with R's own default `kappa=1e6`, used correctly, to a tiny
   margin — and the "bigger kappa is safer" intuition is genuinely
   false past `~1e8`-`1e10`.
3. **§5's design followed exactly**: no `kappa` parameter exposed;
   `nobs_diffuse` returned explicitly; `loglik` reports only the
   post-diffuse-phase sum, matching this project's own `nobs=n-d`
   convention.
4. **Verified to machine precision against real, directly-executed
   `statsmodels`** on the same canonical local-level +
   time-varying-regression shape as the handoff's own example.
5. **§4's `O(d)`-not-`O(n)` claim reconfirmed by direct execution**
   at `n=50/500/5000`.
6. **§6's core test matrix fully covered**, including the
   `diffuse_idx=Int[]` exact-reduction guard and the single-diffuse-
   state case. **§6's second "deeper validation" bulk cross-check
   (ARIMA(1,1,0) via differencing vs. diffuse-init) deliberately
   deferred, not silently skipped** — it needs a genuine undifferenced-
   ARIMA-to-state-space constructor, substantively Stage 8.3's own
   scope, not an extension of this function.
7. **Generalized beyond the handoff's own minimal signature**: optional
   `a0`/`P_star0` keywords for a genuinely mixed proper-plus-diffuse
   initial state, anticipating Stage 8.3's SARIMAX — verified with a
   dedicated case.

For a fresh session picking this up with no prior context. Continues
Stage 8.1's stakes directly — this is the piece that makes regression
coefficients (Stage 8.3's SARIMAX), unit-root/local-level components
(Chapter Eleven's full ETS), and structural time series (Chapter
Thirteen) all correctly initializable. Verified with the same standard
as 8.1: every claim below was actually run, not assumed from the
textbook description.

## Where this fits

- **Depends on:** Stage 8.1 (`TimeVaryingSSM`) — diffuse states are a
  specific *initialization* choice layered on top of the time-varying
  recursion, not a separate filter algorithm.
- **The single most important thing this handoff found**: the intuitive
  "just make R's `kappa` really big" approach does not simply converge
  to the exact answer as `kappa -> infinity`. It has a genuine sweet
  spot, and gets *worse* past it. This reframes why exact diffuse
  initialization matters — not "big-kappa is a crude approximation of
  something better," but "big-kappa requires knowing a hidden convention
  and tuning a knob that has a real failure mode on both sides."

---

## 1. Verified reference: R's approximate ("big-kappa") diffuse initialization

Already confirmed in Stage 6.5's own verification work: R's
`stats::arima()` signature includes `kappa = 1e+06` — this is R's actual
mechanism for diffuse-like initialization: set the initial state
variance to a very large fixed number, rather than solving for a proper
diffuse distribution. `kappa=1e6` is R's literal default value, not a
placeholder — confirmed directly from `args(stats::arima)` back in
Stage 6.

## 2. Verified reference: `statsmodels`'s two diffuse methods, directly compared

Confirmed via `dir(KalmanFilter)`: both `initialize_diffuse()` (the
exact method, Durbin & Koopman ch. 5 / Ansley & Kohn 1985) and
`initialize_approximate_diffuse(variance=...)` (the same big-kappa idea
as R's) exist side by side in the same package — an unusually clean
opportunity to compare them directly on identical data rather than
across two different, hard-to-align implementations.

**Test case**: a local level (unit root, `level_t = level_{t-1} +
eta_t`) plus a regression coefficient with no prior information at all
(`beta` constant, zero process noise, but genuinely diffuse initial
variance) — exactly the two canonical reasons this stage exists,
combined in one system:
```python
transition = [[1,0],[0,1]]      # level: random walk; beta: persists exactly
selection  = [[1],[0]]           # only the level has process noise
state_cov  = [[0.01]];  obs_cov = [[0.09]]
```

---

## 3. The central finding — verified numerically, corrected from a wrong first assumption

**First attempt, naive full-sum comparison — looked like divergence,
not convergence:**
```
EXACT diffuse total loglik: -13.845557
APPROX (kappa=1e4):  -23.057337   diff: -9.212
APPROX (kappa=1e6):  -27.661082   diff: -13.816
APPROX (kappa=1e8):  -32.266237   diff: -18.421
APPROX (kappa=1e10): -36.871403   diff: -23.026
APPROX (kappa=1e12): -41.477153   diff: -27.632
```
This looked like the approximate method getting *worse* as `kappa`
grows — the opposite of the usual "bigger kappa approximates diffuse
better" intuition. **This comparison was wrong, not the method** — see
below.

**Why**: the per-observation log-likelihood contribution during the
diffuse phase includes a `-0.5*log(F_t)` term, where `F_t` (the
one-step forecast error variance) is inflated by the huge initial state
variance for exactly the diffuse-affected early observations. As
`kappa -> infinity`, `log(F_t) -> infinity` for those specific terms —
confirmed directly, the first two per-observation values grow
increasingly negative with `kappa` (`-6.2, -8.5, -10.8, -13.1, -15.4` at
`kappa = 1e4...1e12`), while observations *after* the diffuse phase
(`obs[2]` onward) stay essentially unchanged across all `kappa` values.

**The correct comparison — exclude the first `d` diffuse-affected
observations from both sums (`d=2` here, matching the two genuinely
diffuse state elements)** — this is the standard convention in the
diffuse-likelihood literature, and it's what makes the two methods
actually comparable:
```
EXACT total (excl. first 2 obs):  -11.239210
APPROX kappa=1e4:  diff = 0.000026
APPROX kappa=1e6:  diff = 0.00000026   <- R's actual default, already excellent
APPROX kappa=1e8:  diff = 0.00000010   <- even closer
APPROX kappa=1e10: diff = 0.00000506   <- getting WORSE again
APPROX kappa=1e12: diff = -0.00057471  <- noticeably worse, real numerical instability
```

**Two real, precisely quantified conclusions:**
1. **R's actual default (`kappa=1e6`), used correctly (excluding the
   right number of initial observations), is already excellent** —
   agreement to `2.6e-7`. The approximate method isn't crude in
   practice; it's very good *when used correctly*.
2. **There is a genuine sweet spot, not "bigger is always safer."**
   Past roughly `1e8`-`1e10`, accuracy measurably *degrades* —
   floating-point cancellation from an excessively large initial
   variance, a real numerical instability, not a hypothetical one.

**What this means for why exact initialization is worth building
anyway**: not because the approximate method is inherently bad, but
because using it correctly requires (a) knowing the exact diffuse
dimension `d` to exclude the right number of initial observations — an
easy thing to get wrong silently, with no error message if you don't —
and (b) picking a `kappa` in the right range, neither too small nor too
large. The exact method has **neither fragility**: no arbitrary
exclusion convention to get right, no tuning parameter with a hidden
failure mode on both ends.

---

## 4. Efficiency — the diffuse phase is bounded, not growing with sample size

**Confirmed directly, not assumed**: `res.nobs_diffuse` (the number of
observations needed before the diffuse state collapses to a proper
distribution) was checked at `n=50`, `n=500`, and `n=5000` on the
identical system — **`nobs_diffuse=2` in every case**, exactly matching
the diffuse state dimension (`d=2`), not scaling with `n` at all.

**Practical implication for the Julia implementation**: the exact
diffuse recursion only needs its own special handling for the first `d`
observations (`d` is almost always small — 1 or 2 for a typical
regression/local-level case, rarely more than a handful even in
elaborate structural models) — after that, it's identical in cost to
the already-existing Stage 8.1 time-varying recursion. **This means
exact diffuse initialization should add `O(d)` fixed overhead, not
`O(n)`** — a genuinely cheap addition relative to the rest of the
filter pass, worth confirming with a real benchmark once implemented
but not a reason to expect any meaningful slowdown on long series.

---

## 5. Proposed Julia API

```julia
kalman_filter_diffuse(ssm::TimeVaryingSSM, y::Vector{Float64};
                       diffuse_idx::Vector{Int}) -> (loglik, sigma2, v, F, nobs_diffuse, converged)
```

Design notes:
- **`diffuse_idx`**: the state indices with no proper prior (e.g., `[1,
  2]` for a local-level-plus-regression-coefficient system) — required
  explicitly, not auto-detected. Auto-detecting "which states are
  diffuse" from a general `TimeVaryingSSM` is a real, harder problem
  (effectively asking whether `T`'s eigenvalues indicate a proper
  stationary distribution exists) — out of scope here; the caller
  (Stage 8.3's SARIMAX construction, Chapter Eleven's ETS) already knows
  which states it's building as diffuse by construction, so there's no
  need to infer it.
- **No `kappa`/approximate-variance parameter at all** — deliberately.
  Given section 3's finding, exposing an approximate-diffuse fallback
  invites exactly the fragility (wrong exclusion count, wrong kappa
  range) this stage exists to avoid. If an approximate method is ever
  wanted later for some other reason, it should be a clearly separate,
  explicitly-named function, not a keyword option easy to reach for by
  habit.
- **`nobs_diffuse` returned directly**, not hidden — callers (especially
  `fit_arima`-family functions computing AIC/BIC) need to know exactly
  how many initial observations were excluded from the likelihood sum,
  the same way Stage 6.6 needed `nobs = n - d` to be explicit and
  correct rather than assumed.

---

## 6. Comprehensive test matrix

### The critical regression tests

```julia
using Test

@testset "exact diffuse -- matches statsmodels on identical data" begin
    # exact system from section 2/3, n=50, seed=7
    loglik, sigma2, v, F, nobs_diffuse, converged = kalman_filter_diffuse(ssm, y; diffuse_idx=[1,2])
    @test converged
    @test nobs_diffuse == 2
    @test isapprox(loglik, -11.239210224444543; atol=1e-6)  # exact total, excl. first 2 obs -- section 3
end

@testset "approximate diffuse convergence -- sanity-check the exact method against the KNOWN-GOOD kappa range" begin
    # confirm the exact method's answer sits where the well-behaved
    # kappa=1e6/1e8 approximate results converge to -- a second,
    # independent check beyond direct statsmodels agreement
    approx_1e6_excl2 = -11.23920996   # from section 3's verified run
    @test isapprox(loglik_exact, approx_1e6_excl2; atol=1e-5)
end

@testset "diffuse phase length is O(d), not O(n)" begin
    for n in [50, 500, 5000]
        ssm_n = build_test_system(n)  # same structure, different length
        _, _, _, _, nobs_diffuse, _ = kalman_filter_diffuse(ssm_n, y_n; diffuse_idx=[1,2])
        @test nobs_diffuse == 2   # confirmed bounded, per section 4 -- must hold at every n
    end
end

@testset "reduces correctly for a single diffuse state" begin
    # a pure local-level model, ONE diffuse state (not two) -- confirms
    # diffuse_idx of length 1 works correctly, not just the length-2
    # case already verified
end

@testset "no diffuse states -> must match Stage 8.1's plain TimeVaryingSSM exactly" begin
    # diffuse_idx = Int[] should reduce EXACTLY to calling
    # kalman_filter(::TimeVaryingSSM, ...) directly with a proper
    # (non-diffuse) initial covariance -- another zero-cost-reduction
    # regression guard, same spirit as Stage 8.1's own section 3a
end
```

### Extend the bulk-verification pattern from Stage 8.1

Following 8.1's highest-leverage approach directly: construct diffuse
versions of representative cases from the existing GaussianSSM/8.1 test
suites (e.g., an ARIMA(1,1,0) case is *already* a diffuse-initialization
problem in disguise — the differencing-based approach Stage 6.6 uses
sidesteps this by differencing first, but the same series run through a
genuinely diffuse-initialized undifferenced state space should produce
a *consistent* likelihood once the diffuse-phase exclusion is handled
correctly). This gives a natural cross-check between Stage 6.6's
pre-differencing approach and this stage's diffuse-initialization
approach on the same underlying models — worth doing as a deeper
validation, not just a convenience.

---

## 7. Efficiency and accuracy pointers, summarized directly

**Accuracy**:
1. Never compare a diffuse-initialized likelihood by summing *all*
   observations naively — always exclude the first `nobs_diffuse`
   terms, or use the exact method's own properly-defined diffuse
   likelihood formula, which handles this internally without needing an
   external exclusion convention.
2. If an approximate/big-kappa fallback is ever built later anyway
   (against this handoff's recommendation in section 5), do not default
   to an arbitrarily huge `kappa` — section 3's data shows accuracy
   *degrades* past roughly `1e8`-`1e10`. There is a real ceiling, not
   just a floor.
3. `nobs_diffuse` must be surfaced and used correctly everywhere a
   likelihood or information criterion is computed downstream (Stage
   8.3's SARIMAX, Chapter Eleven's ETS) — an easy place to silently
   double-count or under-count observations if this isn't threaded
   through consistently.

**Efficiency**:
1. The diffuse phase is `O(d)`, confirmed constant across `n=50` to
   `n=5000` on identical structure — implement the special diffuse-phase
   handling as a short, separate prefix loop before falling through to
   the ordinary (already-fast, Stage 8.1-verified) recursion, rather
   than adding diffuse-related branching inside every iteration of the
   main loop. This keeps the post-diffuse-phase cost identical to
   Stage 8.1's already-checked performance, with the diffuse-specific
   cost strictly bounded and small.
2. Benchmark this specifically once built: confirm the `O(d)` claim
   holds in the actual Julia implementation too, not just in
   `statsmodels`'s Cython — the claim should transfer, but it's a
   real number to check, not to assume carries over automatically.

---

## 8. What to do with this

1. Implement `kalman_filter_diffuse` per section 5, structured as a
   short diffuse-phase prefix (per section 7's efficiency note) followed
   by a fallthrough to Stage 8.1's existing recursion.
2. Run the tests in section 6, especially the `diffuse_idx=Int[]`
   reduction test and the `O(d)`-scaling test — both are cheap,
   high-value regression guards in the same spirit as Stage 8.1's own
   highest-leverage tests.
3. Do **not** expose an approximate/`kappa`-style fallback by default,
   per section 5's reasoning — if one gets added later, name it
   distinctly and document section 3's sweet-spot finding directly in
   its docstring so the same mistake isn't repeated silently.
4. Update `development-sequence.md`'s Stage 8.2 row: mark implemented,
   record the corrected-assumption finding (naive full-sum comparison
   looks like divergence; the real comparison requires excluding the
   diffuse-affected observations) prominently, since it's exactly the
   kind of thing someone re-deriving this later could get wrong the same
   way this handoff's own first attempt did.
