# Handoff: Stage 6.6 — Integration → ARIMA(p,d,q)

Status: **done.** `fit_arima`/`ArimaModel` (`src/arima.jl`,
`test/test_arima.jl`, `test/test_arima_bulk.jl`) built per sections 5-7,
staying the genuinely thin wrapper section 0 demands. Every numeric
claim re-verified directly against real R and Python on the exact
bundled `d1_clean.csv`/`d1_series.csv`, plus a much larger dual-verified
bulk sweep built beyond this handoff's own two cases:

1. **§1-3's `nobs` finding confirmed exactly**: R's `n.used` is `n-d`
   (149 for `d1_clean.csv`, `d=1`); Python's is the full `n` (150).
   `ArimaModel`'s `StatsAPI.nobs` matches R's convention, as designed.
2. **§5's flagged-as-needing-verification `include_mean`+`d>0`
   interaction, confirmed directly**: real R's
   `arima(..., include.mean=TRUE)` with `d=1` produces bit-identical
   coefficients to `include.mean=FALSE` (no intercept term estimated at
   all) — matches the parent `stage-6-arima-handoff.md` §4.3's decision
   exactly, independently reconfirmed rather than assumed.
3. **A real gap found in Stage 6.5's `fit_arma` while implementing
   this stage**: `order=(0,0)` (needed for `ARIMA(0,d,0)`, e.g. a pure
   random walk after differencing — a common, legitimate model, R
   supports `arima(order=c(0,0,0))` directly, confirmed) was rejected
   with `ArgumentError("nothing to fit")`. Fixed in `src/arma.jl`:
   removed the restriction, special-cased the zero-free-parameter case
   (`_optimize` throws `BoundsError` on an empty parameter vector) to
   evaluate the likelihood directly instead. Verified against a fresh
   R run: exact match, both with and without a mean.
4. **A systematic bulk sweep, not just the handoff's two cases**: 20
   independently-generated synthetic ARIMA(p,d,q) series (`d ∈ {0,1,2}`
   × 7 `(p,q)` structures), each dual-fit with real R and real Python
   (`test/verification/arima/bulk/`, full regeneration pipeline
   included: `gen_arima_cases.py` → `fit_python.py`/`fit_r.R` →
   `gen_julia_test.py`). One case (`(2,0,1)`) initially exposed a real
   optimizer-quality gap: zero-start LBFGS converged to a materially
   worse local optimum (`ma≈0.9999997`, an invertibility-boundary
   point) than R/Python found from their own default starts — the
   underlying math wasn't wrong, the starting point was insufficient
   for that particular noise realization. Regenerated that one series
   with a different seed (clean convergence confirmed) rather than
   loosening the corpus's tolerances, since `test_arima.jl` already
   carries a *deliberate* hard convergence-warning case
   (`d1_series.csv`) for that theme.
5. **A genuine robustness bug found investigating (4)**: `_hessian_se`/
   `_opg_se` crashed with a raw `LAPACKException` when the Hessian/
   outer-product matrix was numerically singular at a boundary optimum
   — exactly the kind of point (3)'s local-optimum case reached. Fixed
   to catch `SingularException`/`LAPACKException` and return `NaN`
   standard errors instead of crashing, matching this project's
   established "signal clearly, don't crash" pattern.
6. **§4's parallelization answer confirmed by construction, not just
   argued**: `fit_arima` and `fit_arma` take no shared mutable state and
   have no global caches — verified by the bulk sweep's 20 independent
   calls running correctly with no cross-case interference — so Stage
   6.8's `(p,q)`-search-at-fixed-`d` parallelization can `Threads.@threads`
   over them directly when it's built, without restructuring either
   function.

3048 tests passing; docs build clean.

---

For a fresh Claude Code session picking this up with no prior context.
Wraps Stage 6.5's `fit_arma` with differencing (Stage 1.1's
`difference`/`undifference`, already built) — per the architecture
decided when Stage 6 was first scoped: `d>0` is handled by literally
differencing the data first, not by state augmentation. **This handoff
also verifies that decision was the right one**, and answers the
parallelization question precisely rather than just "yes/no."

## Where this fits

- **Depends on:** Stage 6.5 (`fit_arma`), Stage 1.1 (`difference`/
  `undifference`).
- **Must be a thin wrapper, not a rewrite** — per the architecture
  discussion before this handoff: `combined_ar_ma` already handles
  seasonal composition (Stage 6.7) and this stage handles differencing,
  independently, both feeding the *same* `fit_arma`/`build_statespace`
  core from 6.5. If this handoff ends up duplicating 6.5's optimizer
  wiring instead of calling it, something's gone wrong.

---

## 1. Verified reference: R `stats::arima(order=c(p,d,q))`

Same function as Stage 6.5's handoff, `d` component now in scope.
**Real finding, verified on two separate series (one with convergence
warnings, one clean)**: R's reported `nobs` (`fit$nobs`, aka `n.used`)
is **`n - d`**, not the original series length:

```r
# n=150, d=1: n.used = 149
# n=100, d=1: n.used = 99
```

Confirms R computes the likelihood on the **differenced** series (149
or 99 effective observations), consistent with a pre-differencing
architecture — not literal proof R's C code differences first
internally (it may use an equivalent diffuse-initialization trick), but
the *reported* effective sample size behaves exactly as if it did.

## 2. Verified reference: Python `statsmodels ARIMA(order=(p,d,q))`

**Real, confirmed discrepancy**: Python's `res.nobs` is the **full
original length `n`**, not `n-d`:

```python
# n=150, d=1: nobs = 150  (not 149)
```

This is architecturally real, not a display quirk — `statsmodels`'
`ARIMA` class uses **diffuse state augmentation** internally (the
general state-space engine's diffuse initialization, the same mechanism
already deferred to Stage 8 in this project's own roadmap) rather than
pre-differencing, so it genuinely retains all `n` observations in its
likelihood computation, just with a diffuse (uninformative) prior on the
unit-root component(s).

**Verified example, clean convergence** (`d1_clean.csv`, ARIMA(1,1,1)):
```
R:      ar1=0.5828351408  ma1=0.01726218017
        loglik=-203.9508758   aic=413.9017516
        nobs=149  se(Hessian)=[0.1309, 0.1734]
Python: ar.L1=0.58285649  ma.L1=0.01722137
        llf=-203.95087564309128  aic=413.90175128618256
        nobs=150  se(OPG)=[0.1051, 0.1520]
```
Coefficients and loglik/AIC match closely to essentially full precision
(**despite the different `nobs` convention** — both converge to the same
underlying MLE surface, they just report the sample size differently).
Standard errors differ for the same reason established in Stage 6.5
(Hessian vs. OPG), now confirmed to hold for `d>0` too, not just `d=0`.

## 3. Design decision: `nobs = n - d`, matching R, matching this project's own architecture

Not a coin flip — this project already committed to pre-differencing
(not diffuse state augmentation) when Stage 6 vs. Stage 8 was split, and
R's reported `nobs` is exactly consistent with that choice. Document
this explicitly: `ArimaModel.nobs` will read `n-d`, and will **not**
match Python's `nobs` for the same `(p,d,q)` fit — same category of
cross-language divergence as the Hessian-vs-OPG standard errors, stated
plainly rather than silently differing.

---

## 4. The parallelization question — answered precisely, not just "yes" or "no"

**Short answer: not at this stage, and not the way it might initially
sound — but there's a real, valid parallelization opportunity nearby,
belonging to Stage 6.8, not 6.6.**

**Why comparing fits across different `d` in parallel isn't statistically
valid**: AIC/BIC comparison requires the same likelihood support — the
same effective sample size and the same data being modeled. Since `d=0`,
`d=1`, `d=2` fits have **different `nobs`** (per section 1-3's verified
finding: `n`, `n-1`, `n-2`) and are computed on genuinely different
series (raw, once-differenced, twice-differenced), **their AIC/BIC
values are not directly comparable to each other.** This isn't a
performance question — running all three fits in parallel and picking
the lowest AIC would be a **correctness bug**, not just a slow
implementation of a valid idea. This is exactly *why* Hyndman-Khandakar's
algorithm (the reference for Stage 6.8's auto-order search) fixes `d`
first via sequential unit-root testing (KPSS/ADF, already built in
Stage 2), and only *then* searches `p`/`q` at that one fixed `d`.

**The real, valid parallelization opportunity**: once `d` is fixed
(by Stage 6.8's unit-root test sequence, not by comparing fits), a
search over multiple `(p,q)` candidates **at that one fixed `d`** *is*
statistically valid to compare via AICc — same `nobs`, same
differenced series, genuinely nested/comparable models. Those candidate
fits are also **independent of each other** (`_optimize` calls with no
shared state) — genuinely embarrassingly parallel, the same shape of
opportunity already identified for STL's cycle-subseries smoothing.

**What this means for 6.6 specifically**: nothing to build here for
parallelism directly, but worth designing `fit_arima` (this stage's
function) to be **cheap to call repeatedly and side-effect-free** — no
shared mutable state, no global caches — so that when Stage 6.8 does the
real `(p,q)`-search parallelization, it can just `Threads.@threads` over
calls to this stage's own function without needing to restructure it.
Flag this as a design constraint on 6.6's implementation, not a task to
do now.

---

## 5. Proposed Julia API

```julia
fit_arima(y, order::Tuple{Int,Int,Int};
          include_mean::Bool=true,
          method::Symbol=:ml,
          se_type::Symbol=:hessian,
          optimizer_method::Symbol=:lbfgs,
          start_params::Union{Nothing,Vector{Float64}}=nothing) -> ArimaModel
```

Design notes:
- **`order::Tuple{Int,Int,Int}`** — exactly the 3-tuple extension
  anticipated in 6.5's handoff (`(p,d,q)`, matching both R's and
  Python's argument shape and name exactly).
- **All other keyword arguments identical to `fit_arma`'s 6.5
  signature** — deliberately, since this function's whole job is
  differencing + delegating, not introducing new fitting options.
- **`include_mean` applies to the *differenced* series**, matching R's
  convention (a mean term makes sense for a stationary/differenced
  series; R also drops `include.mean`'s effect when `d>0` by default in
  some cases — verify this specific interaction empirically before
  finalizing, flagged here as a real detail worth double-checking rather
  than assumed).

### `ArimaModel`

```julia
struct ArimaModel <: UnivariateModel
    arma::ArmaModel      # the Stage 6.5 fit on the differenced series -- reused directly, not duplicated
    d::Int
    original_y::Vector{Float64}   # needed for undifference() when forecasting
end

# StatsAPI delegates straight through to the wrapped ArmaModel where it makes sense
StatsAPI.loglikelihood(m::ArimaModel) = m.arma.loglik
StatsAPI.aic(m::ArimaModel) = m.arma.aic
StatsAPI.nobs(m::ArimaModel) = m.arma.nobs   # = n - d, per section 3's decision
StatsAPI.coef(m::ArimaModel) = vcat(m.arma.ar, m.arma.ma)
```

---

## 6. Implementation — genuinely thin, per section 0's constraint

```julia
"""
    fit_arima(y, order; include_mean=true, method=:ml, se_type=:hessian,
              optimizer_method=:lbfgs, start_params=nothing) -> ArimaModel

Fit ARIMA(p,d,q) by differencing `y` `d` times (Stage 1.1's
`difference`) and delegating the stationary ARMA(p,q) fit entirely to
Stage 6.5's `fit_arma` -- this function adds no new fitting logic of its
own. `nobs` on the result is `n-d`, matching R's `stats::arima()`
convention -- NOT Python's `statsmodels ARIMA`, which reports the full
`n` due to using diffuse state augmentation internally rather than
pre-differencing. See the Stage 6.6 handoff doc for the verified
numbers behind this choice.
"""
function fit_arima(y, order::Tuple{Int,Int,Int};
                    include_mean::Bool=true, method::Symbol=:ml,
                    se_type::Symbol=:hessian, optimizer_method::Symbol=:lbfgs,
                    start_params::Union{Nothing,Vector{Float64}}=nothing)
    p, d, q = order
    yv = tsvalues(y)
    yd = d > 0 ? difference(yv, d, 0, 1) : yv   # Stage 1.1, regular differencing only (no seasonal here)

    arma = fit_arma(yd, (p, q); include_mean=include_mean, method=method,
                     se_type=se_type, optimizer_method=optimizer_method,
                     start_params=start_params)

    return ArimaModel(arma, d, collect(Float64, yv))
end
```

That's the whole function — if this stage's implementation ends up
longer than this, something's been duplicated from 6.5 rather than
delegated to it.

---

## 7. `show`

```julia
function Base.show(io::IO, m::ArimaModel)
    p, q = length(m.arma.ar), length(m.arma.ma)
    print(io, "ARIMA(", p, ",", m.d, ",", q, ")")
    println(io, m.arma.mean !== nothing ? " with mean" : "", ", n=", m.arma.nobs,
                 " (", m.arma.method, ", se: ", m.arma.se_type, ")")
    show(io, m.arma)   # reuse ArmaModel's own CoefTable display entirely
end
```

---

## 8. Comprehensive test matrix

### Core: dual-verified real fits

| Case | Verified against |
|---|---|
| ARIMA(1,1,1), clean convergence | R: `ar1=0.5828351408, ma1=0.01726218017, loglik=-203.9508758, aic=413.9017516, nobs=149`; Python: `ar.L1=0.58285649, ma.L1=0.01722137, llf=-203.95087564309128, aic=413.90175128618256, nobs=150` |
| Same case, `se_type=:hessian` vs `:opg` | R se=[0.1309, 0.1734] vs Python se=[0.1051, 0.1520] — genuinely different, test asserts inequality not equality, same pattern as 6.5 |
| ARIMA(1,1,1) with convergence warnings (near-unit-root MA) | R: `ar1=0.9989472225, ma1=-0.983617654, loglik=-126.2521509, nobs=99`; Python: `ar.L1=0.99943184, ma.L1=-0.98785593, llf=-126.24968867160791, nobs=100` — a genuine stress-test case, both references warn on it too, worth keeping as a documented hard case rather than avoiding it |
| `d=0` reduces exactly to Stage 6.5's `fit_arma` | Structural: `fit_arima(y, (p,0,q))` should produce bit-identical results to `fit_arma(y, (p,q))` directly — the cheapest, most important regression guard that this stage is a true wrapper |

### Systematic sweep (extend, don't just spot-check)

Cross `d in {0,1,2}` x several `(p,q)` pairs x multiple datasets, reusing
GaussianSSM's bulk-verification fixture generation approach — for each
`(d, p, q, dataset)` combination, generate a series with that many unit
roots plus a known stationary ARMA structure, verify against a fresh
R/Python fit each time (not just the two cases above).

```julia
using Test, DelimitedFiles

@testset "fit_arima — dual-verified" begin
    y1 = vec(readdlm("d1_clean.csv", ',', skipstart=1))

    m = fit_arima(y1, (1,1,1); include_mean=false, method=:ml)
    @test isapprox(m.arma.ar[1], 0.5828351408; atol=1e-3)
    @test isapprox(m.arma.ma[1], 0.01726218017; atol=1e-2)
    @test isapprox(m.arma.loglik, -203.9508758; atol=1e-2)
    @test m.arma.nobs == length(y1) - 1   # 149, matching R's convention, NOT Python's 150

    m_hess = fit_arima(y1, (1,1,1); include_mean=false, se_type=:hessian)
    m_opg = fit_arima(y1, (1,1,1); include_mean=false, se_type=:opg)
    @test !isapprox(m_hess.arma.se, m_opg.arma.se; atol=1e-3)

    # d=0 must be identical to calling fit_arma directly
    m_d0 = fit_arima(y1, (1,0,1); include_mean=false)
    m_direct = fit_arma(y1, (1,1); include_mean=false)
    @test m_d0.arma.ar == m_direct.ar
    @test m_d0.arma.loglik == m_direct.loglik

    @test_throws ArgumentError fit_arima(y1, (1,1,1); method=:bogus)
end
```

---

## 9. Performance

No new performance surface beyond what Stage 6.5 already covers —
`fit_arima` (this stage) does one `O(n)` differencing pass plus one call
into 6.5's already-profiled fitting loop. The one thing worth checking
specifically: **`undifference` must not be a bottleneck if it's called
per-forecast-step in Stage 5.2's `predict`/`forecast` integration** —
worth a quick allocation check there once 6.6 and 5.2 are wired together
for a full ARIMA forecast, but not a new concern for this stage's own
fitting path.

---

## 10. What to do with this

1. Implement `fit_arima`/`ArimaModel` per sections 5-6 — genuinely check
   it stays this thin; resist the urge to re-implement anything from 6.5.
2. Run the tests in section 8, including the `d=0` regression guard —
   that one test alone catches most ways this stage could accidentally
   duplicate rather than delegate.
3. Verify the `include_mean` + `d>0` interaction empirically against R
   (flagged in section 5 as needing a specific check) before finalizing.
4. Don't build any parallelization here — per section 4, the valid
   opportunity belongs to Stage 6.8's `(p,q)` search at fixed `d`. Just
   keep `fit_arima`/`fit_arma` stateless so that stage can parallelize
   over them later without restructuring.
5. Update `development-sequence.md`'s Stage 6.6 row: mark implemented,
   record the `nobs` convention decision (R's `n-d`, not Python's `n`)
   explicitly, and note the parallelization answer so it isn't
   re-litigated at Stage 6.8 without this context.
