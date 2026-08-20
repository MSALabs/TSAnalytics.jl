# Handoff: Stage 8.3 — ARIMAX / SARIMAX, `model=:mle` vs `model=:tvss`

**Status: done.** Implemented as `fit_arimax`/`fit_sarimax`/
`ArimaxModel`/`SarimaxModel` (`src/arimax.jl`), exported like
`fit_arima`/`fit_sarima`. Section 3's full test matrix implemented in
`test/test_arimax.jl` (75 assertions), plus additional coverage
(`include_mean` true/false and its `d>0` forced-off case, `method=:ml`/
`:css_ml`, `se_type=:hessian`/`:opg`, `n_restarts`/`parallel` for
`model=:tvss`, container-agnostic inputs, error paths). Verification
files moved to `test/verification/arimax/`.

Both branches implemented via a shared internal `_fit_arimax_core`
(`fit_arimax` fixes `seasonal_order=(0,0,0,1)`, `fit_sarimax` exposes it
fully) — confirmed to satisfy the reduction property directly, matching
Stage 6.7's own precedent over 6.6.

**A real bug in Stage 8.2's own `kalman_filter_diffuse` was found and
fixed while independently verifying `model=:tvss`** against real Python
`SARIMAX(time_varying_regression=True, mle_regression=False,
use_exact_diffuse=True)`: evaluated at Python's own exact fitted
parameters (not re-optimized, to isolate the cause), Julia's loglik was
off from Python's `llf` by the *same* small constant across two
different parameter settings — not optimizer noise, which wouldn't
repeat identically. Root cause: `kalman_filter_diffuse` excluded the
first `nobs_diffuse` observations' likelihood contributions entirely,
modeled on this project's `nobs = n - d` differencing convention (Stage
6.6/6.7) — that convention doesn't apply to exact diffuse
initialization. Real `statsmodels`'s own genuine default
(`res.llf`, `loglikelihood_burn=0`) *includes* the diffuse-phase
contribution via the well-defined `-0.5·log(2π·F_infty)` term (already
correctly identified in Stage 8.2's own docstring — only the *exclusion*
was the bug, not the formula). Confirmed directly on Stage 8.2's own
original verification data (`test/verification/diffuseinit/
diffuse_case1.json`), which had captured *both* the correct
(`loglik_full`) and incorrect (`loglik_excl_diffuse`) numbers side by
side from its very first run — only the wrong one was checked at the
time. Fixed in `src/statespace/diffuseinit.jl`; Stage 8.2's own test
suite (`test/test_diffuseinit.jl`) updated and re-passing; full record
in `test/verification/diffuseinit/diffuseinit-inclusive-sum-correction.txt`.

For a fresh session picking this up with no prior context. Supersedes
the previous Stage 8.3 handoff's naming twice now — first restructured
to `fit_arimax`/`fit_sarimax` (mirroring the established 6.5->6.6->6.7
pattern), and now with the model-choice argument renamed from `method`
to **`model`**. This second rename isn't cosmetic — it fixes a real
collision (`method` was already in use by Stage 6.5/6.7 for a different
choice) and is more accurate given section 1's own finding. **Read
section 1 before writing any tests.**

## Why `model`, not `method` — two independent reasons, not just style

**Reason 1 -- a real naming collision, caught late.** Stage 6.5's
`fit_arma`/`fit_sarima` already has a `method::Symbol` argument --
`:ml` vs `:css_ml`, choosing *how* to estimate a single model (pure MLE
vs. CSS-derived starting values feeding into MLE). Since this stage's
`:mle` path explicitly reuses that Stage 6.5/6.7 machinery internally,
`fit_arimax` needs to pass that same `method` keyword through. Naming
*this* stage's model-choice argument `method` too would have meant two
different arguments both called `method`, meaning two different things,
with the added confusion that `:ml` and `:mle` look like typos of each
other. A real bug waiting to happen, not a style nitpick.

**Reason 2 -- matches section 1's actual finding.** `:mle` and `:tvss`
are not two ways of computing the same thing -- they're genuinely
different statistical models (fixed coefficient vs. genuinely evolving
coefficient). Calling the argument `method` undersells that; `model`
says what's actually being chosen.

**Both keywords now coexist cleanly**: `model=:tvss, method=:css_ml` is
a perfectly sensible, unambiguous combination under this naming, and
would have been actively confusing under the old one.

---

## 1. The critical correction: `model=:mle` and `model=:tvss` are different models, not different computations of the same model

**This was verified directly, and it shapes the whole test design.**
Ran Python's actual `time_varying_regression=True, use_exact_diffuse=True`
path on the exact same data as `model=:mle`'s already-verified case:

```python
SARIMAX(y, exog=x, order=(1,0,0), trend='n',
        time_varying_regression=True, mle_regression=False, use_exact_diffuse=True)
# param_names: ['ar.L1', 'var.x1', 'sigma2']    <- NOT 'x1' (beta itself)!
# params: [0.4759, 0.0001094, 0.9793]
# llf: -213.7479
```

**`beta` itself is never an optimized MLE parameter under `model=:tvss`**
-- only its *process variance* (`var.x1`, i.e. `Q_beta`) is. `beta`
becomes a genuinely latent, time-varying **state**, recovered via the
filtered state path, confirmed to actually evolve:
```
filtered beta[:5]:  [2.000, 1.778, 1.458, 1.925, 2.133]
filtered beta[-5:]: [1.871, 1.875, 1.864, 1.869, 1.875]
```

**This means `model=:mle` (llf=-211.126, one fixed beta=1.8948) and
`model=:tvss` (llf=-213.748, beta evolving) are answering genuinely
different questions**, not two ways of computing the same answer.
Comparing their likelihoods directly, or expecting them to converge to
the same number, would be a real test-design error.

**The correct equivalence test, verified**: constrain `model=:tvss`'s
`Q_beta` to exactly zero (removing beta's ability to drift at all) --
```python
res = mod.fit_constrained({'var.x1': 0.0}, disp=False)
# ar.L1=0.47466  filtered beta converges to ~1.8946 by series end
# llf = -213.767   <- still NOT equal to :mle's -211.126
```
**Even here, the point estimates converge closely** (final filtered
beta `~1.895` vs. `model=:mle`'s `1.8948`; AR coefficient `0.4747` vs.
`0.4768`) **but the likelihoods still don't match exactly** -- because
`model=:tvss` still carries a diffuse-initialization phase (beta
genuinely starts with no prior information, exactly Stage 8.2's own
scenario), while `model=:mle` never has any diffuse phase at all. The
two approaches use different likelihood *normalizations*, consistent
with Stage 8.2's own `nobs_diffuse`-exclusion finding, even when
they're estimating mathematically the same fixed-coefficient model.

**Design implication for testing**: `model=:tvss(Q_beta=0)` should be
tested for **point-estimate convergence** to `model=:mle`'s result (a
`filtered beta` reaching close to the `:mle` value by the end of the
series), not **likelihood equality** -- those are different, both real,
both correct properties, and conflating them would produce a test that
fails for a reason that isn't a bug.

---

## 2. Proposed API

```julia
fit_arimax(y, order::Tuple{Int,Int,Int}, exog;
           include_mean::Bool=true,
           model::Symbol=:mle,               # :mle or :tvss -- WHICH MODEL (this stage's choice)
           method::Symbol=:ml,                # :ml or :css_ml -- Stage 6.5's estimation-procedure choice, unchanged, cleanly separated
           Q_beta::Union{Nothing,Vector{Float64}}=nothing,  # model=:tvss only; nothing = estimate freely, matches Python's default
           se_type::Symbol=:hessian,
           optimizer_method::Symbol=:lbfgs,
           n_restarts::Integer=1, parallel::Bool=true) -> ArimaxModel

fit_sarimax(y, order::Tuple{Int,Int,Int}, seasonal_order::Tuple{Int,Int,Int,Int}, exog;
            include_mean::Bool=true, model::Symbol=:mle, method::Symbol=:ml,
            Q_beta::Union{Nothing,Vector{Float64}}=nothing, kwargs...) -> SarimaxModel
```

- **`model=:mle` (default)**: `beta` as an ordinary joint-MLE
  parameter, `y - X*beta` subtracted before calling the unmodified
  Stage 6.5/6.7 likelihood. Needs only 6.5-6.7, nothing from Stage
  8.1/8.2. `method` (`:ml`/`:css_ml`) passes straight through to that
  underlying call, unchanged from Stage 6.5's own meaning.
- **`model=:tvss`**: `beta` as a latent state via Stage 8.1's
  `TimeVaryingSSM` + Stage 8.2's `kalman_filter_diffuse`. The optimizer
  searches `Q_beta` (and the ARMA parameters), not `beta` directly.
  `method` still applies to the ARMA parameters' own estimation
  procedure within this path.
- **`Q_beta`**: `model=:tvss`-only. `nothing` (default) estimates it
  freely (matching Python's actual default -- genuine drift allowed if
  the data supports it). An explicit `zeros(k)` forces the
  fixed-coefficient reduction case from section 1 -- useful specifically
  for the point-estimate-convergence test, not a typical end-user
  setting.

### `ArimaxModel` -- carries both estimation shapes

```julia
struct ArimaxModel <: UnivariateModel
    model::Symbol                                   # :mle or :tvss
    method::Symbol                                   # :ml or :css_ml
    beta::Union{Vector{Float64}, Nothing}            # model=:mle only -- fixed point estimate
    beta_filtered::Union{Matrix{Float64}, Nothing}   # model=:tvss only -- k x n, the evolving path
    Q_beta::Union{Vector{Float64}, Nothing}          # model=:tvss only -- estimated (or fixed) process variance
    arma::ArmaModel
    loglik::Float64
    nobs_diffuse::Union{Int, Nothing}   # model=:tvss only, per Stage 8.2's convention
    # ... aic/bic/se/converged, same shape as ArmaModel throughout this project
end
```

---

## 3. Comprehensive test matrix

### `model=:mle` -- already dual-verified (unchanged numbers, renamed keyword)

| Case | Verified against |
|---|---|
| AR(1) + 1 exog, n=150 | R: `ar=0.476774167, x=1.894806442, loglik=-211.1256527`; Python: `ar=0.47677706, x=1.89480149, llf=-211.1256527433344` |
| `exog=nothing` reduces exactly to `fit_sarima` | Structural regression guard |

### `model=:tvss` -- newly verified, real numbers, both configurations

| Case | Verified against |
|---|---|
| `model=:tvss`, `Q_beta` freely estimated (Python default) | `ar.L1=0.475909212, var.x1=0.000109371580, sigma2=0.979298992, llf=-213.74793442697535`; filtered beta path `[2.000,1.778,1.458,1.925,2.133,...,1.871,1.875,1.864,1.869,1.875]` |
| `model=:tvss`, `Q_beta=[0.0]` forced | `ar.L1=0.47465807, var.x1=0.0, sigma2=0.98227306, llf=-213.76718527012807`; filtered beta converges to `~1.895` by series end |
| `model=:tvss(Q_beta=0)` point estimates converge to `model=:mle`'s | Final filtered beta `~1.895` vs. `:mle`'s `1.8948` (close, not exact); AR coefficients `0.4747` vs `0.4768` (also close) |
| `model=:tvss(Q_beta=0)` likelihood does **NOT** equal `model=:mle`'s likelihood | `-213.767` vs `-211.126` -- confirmed real, expected, NOT a bug -- the test must assert **inequality** here, explicitly |

```julia
using Test, DelimitedFiles, Statistics

@testset "fit_arimax model=:mle -- dual-verified" begin
    d = readdlm("sarimax_exog_shared.csv", ',', skipstart=1)
    y, x = d[:,1], reshape(d[:,2], :, 1)

    m = fit_arimax(y, (1,0,0), x; include_mean=false, model=:mle)
    @test isapprox(m.beta[1], 1.894806442; atol=1e-3)
    @test isapprox(m.arma.ar[1], 0.476774167; atol=1e-3)
    @test isapprox(m.loglik, -211.1256527; atol=1e-2)
end

@testset "fit_arimax model=:tvss -- freely-estimated Q_beta, dual case" begin
    d = readdlm("sarimax_exog_shared.csv", ',', skipstart=1)
    y, x = d[:,1], reshape(d[:,2], :, 1)

    m = fit_arimax(y, (1,0,0), x; include_mean=false, model=:tvss)
    @test isapprox(m.arma.ar[1], 0.475909212; atol=1e-3)
    @test isapprox(m.Q_beta[1], 0.000109371580; atol=1e-5)
    @test isapprox(m.loglik, -213.74793442697535; atol=1e-2)
    @test isapprox(m.beta_filtered[1, 1:5], [2.0, 1.778, 1.458, 1.925, 2.133]; atol=1e-2)
    @test isapprox(m.beta_filtered[1, end-4:end], [1.871, 1.875, 1.864, 1.869, 1.875]; atol=1e-2)

    # beta genuinely varies -- this is the actual point of model=:tvss, assert it directly
    @test std(m.beta_filtered[1, :]) > 0.01
end

@testset "fit_arimax model=:tvss(Q_beta=0) -- point estimates converge, likelihood does NOT" begin
    d = readdlm("sarimax_exog_shared.csv", ',', skipstart=1)
    y, x = d[:,1], reshape(d[:,2], :, 1)

    m_mle = fit_arimax(y, (1,0,0), x; include_mean=false, model=:mle)
    m_tvss0 = fit_arimax(y, (1,0,0), x; include_mean=false, model=:tvss, Q_beta=[0.0])

    @test isapprox(m_tvss0.arma.ar[1], 0.47465807; atol=1e-3)
    # POINT ESTIMATE convergence -- final filtered beta close to :mle's fixed value
    @test isapprox(m_tvss0.beta_filtered[1, end], m_mle.beta[1]; atol=0.02)

    # LIKELIHOOD must NOT match -- asserting the difference explicitly,
    # per section 1's finding, so this isn't "corrected" into a bug later
    @test !isapprox(m_tvss0.loglik, m_mle.loglik; atol=0.5)
    @test isapprox(m_tvss0.loglik, -213.76718527012807; atol=1e-2)
end

@testset "fit_arimax model=:tvss -- nobs_diffuse populated, :mle does not have it" begin
    d = readdlm("sarimax_exog_shared.csv", ',', skipstart=1)
    y, x = d[:,1], reshape(d[:,2], :, 1)

    m_mle = fit_arimax(y, (1,0,0), x; include_mean=false, model=:mle)
    m_tvss = fit_arimax(y, (1,0,0), x; include_mean=false, model=:tvss)
    @test m_mle.nobs_diffuse === nothing
    @test m_tvss.nobs_diffuse !== nothing && m_tvss.nobs_diffuse >= 1
end

@testset "fit_sarimax extends fit_arimax exactly as 6.7 extends 6.6" begin
    d = readdlm("sarimax_exog_shared.csv", ',', skipstart=1)
    y, x = d[:,1], reshape(d[:,2], :, 1)

    m_sarimax = fit_sarimax(y, (1,0,0), (0,0,0,0), x; include_mean=false, model=:mle)
    m_arimax = fit_arimax(y, (1,0,0), x; include_mean=false, model=:mle)
    @test isapprox(m_sarimax.beta, m_arimax.beta; atol=1e-8)  # zero seasonal order -> identical
end

@testset "model and method are independent, both apply correctly" begin
    d = readdlm("sarimax_exog_shared.csv", ',', skipstart=1)
    y, x = d[:,1], reshape(d[:,2], :, 1)

    # model=:tvss + method=:css_ml -- a real, sensible combination that
    # would have been ambiguous under the old single-`method` naming
    m = fit_arimax(y, (1,0,0), x; include_mean=false, model=:tvss, method=:css_ml)
    @test m.model == :tvss
    @test m.method == :css_ml
end

@testset "error paths" begin
    d = readdlm("sarimax_exog_shared.csv", ',', skipstart=1)
    y, x = d[:,1], reshape(d[:,2], :, 1)
    @test_throws ArgumentError fit_arimax(y, (1,0,0), x; model=:bogus)
    @test_throws ArgumentError fit_arimax(y, (1,0,0), x; method=:bogus)
    @test_throws ArgumentError fit_arimax(y, (1,0,0), x; model=:mle, Q_beta=[0.0])  # Q_beta only valid for model=:tvss
end
```

---

## 4. Efficiency and accuracy pointers

**Accuracy**:
1. Never compare `model=:mle` and `model=:tvss` likelihoods directly --
   they're different models. Compare point estimates for the
   `Q_beta=0` reduction case; compare likelihoods only within the same
   model.
2. `model=:tvss`'s `nobs_diffuse` must be surfaced and used correctly
   in any AIC/BIC computed for a `:tvss`-fit model.
3. `model=:mle` remains the lower-risk default for the common case --
   `model=:tvss` is for when genuine time-varying-coefficient behavior
   is actually wanted, not a "more accurate" alternative to reach for
   by default.

**Efficiency**: `model=:mle` reuses Stage 6.5-6.7's already-profiled
pipeline directly -- no new performance surface. `model=:tvss` inherits
Stage 8.2's confirmed `O(d)`-bounded diffuse-phase overhead -- cheap
regardless of series length.

---

## 5. What to do with this

1. Implement `fit_arimax`/`fit_sarimax` with `model=:mle`/`:tvss` and
   `method=:ml`/`:css_ml` as two genuinely independent keyword
   arguments, per section 2.
2. Run the full test suite in section 3 -- the `Q_beta=0`
   inequality-of-likelihood test is the one most likely to be "fixed"
   incorrectly by someone who hasn't read section 1 first; leave a
   comment pointing back to this handoff at that assertion specifically.
3. Update `development-sequence.md`'s Stage 8.3 row to reflect the
   `model=:mle`/`:tvss` naming (not `method`) and the
   confirmed-different-models finding.
