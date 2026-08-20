# Handoff: Stage 8.5 — Regression with Autocorrelated Errors, Optional GARCH Variance

**Status: done.** Implemented as `fit_autoreg_garch`/`AutoregGarchModel`
(`src/autoregarch.jl`), exported. `garch_order=nothing` is a literal
delegation to `fit_arimax(y,(m,0,0),exog;model=:mle)`, verified
bit-identical. `garch_order=(p,q)`'s combined likelihood generalizes
this handoff's own hand-verified `m=1` presample-observation derivation
to arbitrary `m` via `stationary_cov`/`build_statespace` (reused, not
reinvented) -- confirmed to reduce to the `m=1` formula algebraically
and to match Stage 8.3's plain likelihood in the degenerate case to
`~7.7e-9`, same order of magnitude as this handoff's own `7.676e-09`.
Section 3's flagged gap (a genuine non-degenerate recovery check) closed
via a known-parameter synthetic series (`n=1000`) -- all five parameters
(`beta`, `phi`, `omega`, `alpha`, `garch_beta`) recovered within normal
finite-sample MLE error; full record in
`test/verification/autoregarch/ar-garch-ground-truth-transcript.txt`.
Tests in `test/test_autoregarch.jl`. Section 5's dependency correction
(drop Stage 8.2, keep Stage 7 + 8.3) applied in `development-sequence.md`.
The Durbin-Watson test (section 5 item 5) was not built -- noted as a
natural companion, not required for this stage.

For a fresh session picking this up with no prior context. **This
handoff overturns part of the roadmap's own dependency listing, based
on verified reasoning, the same way Stage 8.3's did.** The short
version: the non-GARCH case isn't new work at all — it's Stage 8.3
already. The genuinely new work is narrower and different than the
roadmap implied.

## The finding that reshapes this stage's scope

The roadmap's dependency listing was `8.2, 7` — implying this stage
needs Stage 8.2's exact diffuse initialization (for `beta` as a diffuse
state) plus Stage 7's GARCH machinery. **The diffuse-initialization
part doesn't apply here, for the same reason it didn't apply to Stage
8.3.**

**Regression with AR(m) errors — SAS `PROC AUTOREG`'s core model,
written out algebraically**:
```
Y_t = X_t*beta + nu_t
nu_t = phi_1*nu_{t-1} + ... + phi_m*nu_{t-m} + e_t
```
Since `nu_t = Y_t - X_t*beta` by definition, substituting gives:
```
Y_t - X_t*beta = phi_1*(Y_{t-1} - X_{t-1}*beta) + ... + e_t
```
**This is exactly Stage 8.3's `model=:mle` construction** —
`y - X*beta` subtracted, then an ordinary ARMA likelihood applied to
the residual, with `order=(m,0,0)` for pure AR errors (or `(m,0,q)` for
the more general ARMA-errors case, a strict generalization of PROC
AUTOREG's pure-AR case). **`fit_arimax(y, (m,0,0), exog; model=:mle)`
already computes this model exactly**, already verified in Stage 8.3
(the AR(1)+1-exog case, `ar=0.476774167, x=1.894806442,
loglik=-211.1256527`).

**What this means for scope**: the "full ML, no GARCH" tier of
`PROC AUTOREG` isn't a new stage — it's a naming/framing exercise on
top of Stage 8.3, already done. This stage's genuinely new work is the
**combined AR-errors-plus-GARCH-variance case specifically** — that's
the one piece neither Stage 8.3 nor Stage 7 covers alone.

---

## 1. Why no direct software reference exists for the combined case, and how it was verified instead

Checked Python's `arch.univariate.ARX` (a plausible-looking candidate)
directly — it combines **`Y`'s own lags** with `X` and a pluggable
`volatility=` process, matching this project's `arx()` (Stage 5.1)
framing, **not** `PROC AUTOREG`'s "regression, then AR-structured
residual" framing. These are different models (Stage 5.1's own
handoff already distinguished this from ARIMAX's framing). No direct
software match for the specific combination this stage needs was
found.

**Verified instead via the "reduces to an already-trusted case"
methodology**, used throughout this project when no direct reference
exists (Stage 8.1's exact reduction to time-invariant GaussianSSM,
Stage 8.3's `Q_beta=0` convergence check): construct the AR-errors +
GARCH-variance combined likelihood by hand, and confirm it collapses
**exactly** to Stage 8.3's already-verified plain likelihood when the
GARCH part degenerates to constant variance.

**First attempt — off by a real, non-trivial amount (1.038 log-likelihood
units), a genuine bug, not rounding noise**: the initial construction
dropped the AR error's first observation as a conditioning
shortcut (`e[1:]` only), rather than treating it properly as itself
drawn from the AR(1) process's own stationary distribution. **This is
exactly the kind of detail worth getting flagged prominently** — full
ML (what Stage 8.3 verified against) requires the proper stationary
variance for that first residual (`sigma2/(1-phi^2)`), not a dropped or
conditioned-away term.

**Corrected construction, verified to `7.7e-9`, essentially exact**:
```python
nu = y - x*beta
e[1:] = nu[1:] - phi*nu[:-1]
e[0] = nu[0]                          # the AR process's own first value
h[0] = (omega/(1-alpha-garch_beta)) / (1 - phi**2)   # proper stationary variance for e[0]
h[t] = omega + alpha*e[t-1]**2 + garch_beta*h[t-1]    # ordinary GARCH recursion, t>=1
loglik = -0.5 * sum(log(2*pi*h) + e**2/h)
```
With `alpha=garch_beta=0` (degenerate to constant variance): loglik
`-211.12565273565806` vs. Stage 8.3's independently-verified
`-211.1256527433344` — difference `7.7e-9`, machine-precision-level
agreement. **This confirms the combined construction is correct.**

---

## 2. Proposed Julia API

```julia
fit_autoreg_garch(y, m::Integer, exog;
                   garch_order::Union{Nothing,Tuple{Int,Int}}=nothing,  # nothing = plain AR errors, no GARCH (== fit_arimax)
                   include_mean::Bool=true,
                   se_type::Symbol=:hessian,
                   optimizer_method::Symbol=:lbfgs,
                   n_restarts::Integer=1, parallel::Bool=true) -> AutoregGarchModel
```

Design notes:
- **`garch_order=nothing` (default) should literally delegate to
  `fit_arimax(y, (m,0,0), exog; model=:mle)`** — per section 0's
  finding, there's no reason to duplicate that machinery. This is the
  cheapest, highest-value regression test for this stage (mirrors Stage
  8.1's "reduces exactly" tests): `fit_autoreg_garch(y, m, exog;
  garch_order=nothing)` must produce bit-identical results to calling
  `fit_arimax` directly.
- **`garch_order=(p,q)`**: the genuinely new path -- the combined
  likelihood from section 1, jointly optimizing `[beta; phi; omega;
  alpha; garch_beta]` in one `_optimize` call (Stage 4.1), not two
  separate fits.
- **No `dist` (Student-t etc.) option yet** -- Stage 7's own GARCH work
  supports `dist=:normal`/`:t`; this stage inherits whatever Stage 7.1
  ends up supporting rather than reinventing that choice here.

### `AutoregGarchModel`

```julia
struct AutoregGarchModel <: UnivariateModel
    beta::Vector{Float64}
    phi::Vector{Float64}              # AR-error coefficients
    garch::Union{Nothing,GarchModel}  # nothing when garch_order=nothing
    loglik::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    se::Vector{Float64}
    converged::Bool
end
```

---

## 3. Comprehensive test matrix

### The critical regression test

```julia
using Test, DelimitedFiles

@testset "fit_autoreg_garch, garch_order=nothing -- EXACT reduction to fit_arimax" begin
    d = readdlm("sarimax_exog_shared.csv", ',', skipstart=1)
    y, x = d[:,1], reshape(d[:,2], :, 1)

    m1 = fit_autoreg_garch(y, 1, x; garch_order=nothing)
    m2 = fit_arimax(y, (1,0,0), x; model=:mle)
    @test isapprox(m1.beta, m2.beta; atol=1e-8)
    @test isapprox(m1.loglik, m2.loglik; atol=1e-8)
end
```

### The combined AR-GARCH case, real verified numbers

| Case | Verified against |
|---|---|
| AR(1) errors + GARCH(1,1) variance, degenerate to constant variance (`alpha=garch_beta=0`) | Hand-verified combined likelihood matches Stage 8.3's plain ARIMAX loglik `-211.1256527433344` to `7.7e-9` |
| The first-observation stationary-variance treatment specifically | The most likely place for a silent bug to reappear -- a dedicated test asserting the degenerate-case reduction holds is the guard, not a one-off check |

```julia
@testset "fit_autoreg_garch -- combined likelihood, degenerate case" begin
    d = readdlm("sarimax_exog_shared.csv", ',', skipstart=1)
    y, x = d[:,1], reshape(d[:,2], :, 1)

    m = fit_autoreg_garch(y, 1, x; garch_order=(1,1))  # freely fit
    # separately, construct the same model with alpha/garch_beta FIXED at 0
    # and confirm it matches fit_arimax exactly, per section 1's verified construction
end
```

**Flagged gap**: only the degenerate-case reduction has been verified
numerically here -- a genuinely time-varying-volatility case (real
`alpha`/`garch_beta` > 0, fit against real generated data with known
true parameters) hasn't been run yet in this handoff. Given no direct
software reference exists for this exact combination (section 1), that
verification will need to follow the same generate-known-parameters-
and-check-recovery methodology used for Stage 7.1's GARCH bulk tests,
not a cross-language point-match.

---

## 4. Efficiency and parallelism

**Parallelism**: the multi-series design from Stage 7.1
(`fit_garch_multi`) applies here for the same reason -- fitting this
model to several independent series is embarrassingly parallel. No new
design needed; reuse Stage 7.1's pattern directly once this stage's
single-series fit exists.

**Efficiency**: the combined likelihood is two sequential recursions
computed together (the AR-error filter, then the GARCH variance
recursion on those filtered residuals) -- both `O(n)`, no new
performance surface beyond what Stage 8.3 and Stage 7.1 individually
already established. The joint optimization has a larger parameter
vector (`beta` + `phi` + `omega/alpha/garch_beta` all at once) than
either piece alone, so expect a real but modest slowdown versus fitting
either piece separately -- profile once implemented rather than assume.

---

## 5. What to do with this

1. **Update `development-sequence.md`'s Stage 8.5 dependency listing** --
   remove Stage 8.2 as a dependency (per section 0's finding, it was
   never actually needed here, the same way it wasn't needed for
   8.3's default case). Keep Stage 7 and Stage 8.3 as the real
   dependencies.
2. Implement `fit_autoreg_garch` per section 2, with `garch_order=nothing`
   as a literal delegation to `fit_arimax`, not a reimplementation.
3. Run the degenerate-case reduction test in section 3 -- this is the
   test most likely to silently break if someone "simplifies" the
   first-observation handling later without knowing why it's there.
4. Verify a genuine non-degenerate AR-GARCH case against known
   generating parameters (the flagged gap in section 3) before trusting
   this stage fully.
5. Note the Durbin-Watson test (flagged as missing back when AutoReg was
   first discussed, still not built anywhere in Stage 2) as a natural
   diagnostic companion to add alongside this stage, not required for
   it.
