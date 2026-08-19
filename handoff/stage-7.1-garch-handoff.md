# Handoff: Stage 7.1 — ARCH/GARCH(p,q)

**Status: done.** Implemented as `fit_garch`/`fit_garch_multi`/
`GarchModel` (`src/garch.jl`). Summary (full detail in
`test/verification/garch/garch-ground-truth-transcript.txt`):

1. **§4's flagged `p`/`q` naming risk resolved by direct execution
   first, before any implementation** — `arch_model(p=2,q=1)` fitted and
   inspected; `p`=ARCH order (`alpha`), `q`=GARCH order (`beta`),
   matching Python exactly. Locked in with a dedicated GARCH(2,1) test
   against fresh real Python ground truth, not just the docstring.
2. **§1's core case and cov_type robust-vs-classic SE finding both
   reproduced exactly** by direct re-execution (not transcribed) --
   `omega=0.043066 alpha=0.090638 beta=0.867923`, robust/classic SEs
   both confirmed genuinely different, matching to ~1e-5.
3. **`mean_spec=:constant` implemented as joint MLE** (mean estimated
   together with omega/alpha/beta, not naive pre-demean, matching
   `fit_arma`'s Stage 6.5 precedent) -- verified against a fresh real
   `arch_model(mean='Constant')` fit, matches to ~1e-5.
4. **§6's bulk grid regenerated and fit for real**: the handoff's own
   `garch_100cases_python.json` has aggregate results only, no
   underlying series, so 24 fresh series were generated from the same
   grid description and fit with real `arch_model` --
   `test/verification/garch/bulk/`. All 24 converged; mean
   `|Julia − Python|` parameter error `6.5e-6`.
5. **§5's two parallelism designs both implemented, both default-on**:
   `fit_garch_multi` and `n_restarts`, `Threads.@threads`-guarded the
   same way as every other stage's parallel design in this project.
6. **`dist=:t` deliberately left unimplemented**, throwing a clear,
   named `ArgumentError` -- this handoff provides no verified ground
   truth for it, and it needs its own shape parameter and likelihood, a
   genuinely separate scope.

For a fresh session picking this up with no prior context. First item
of Chapter Eight — genuinely independent of the Gaussian state-space
engine (its own likelihood structure entirely), needing only the
optimizer from Chapter Five. This is also the first stage where the
single-model fit itself is provably sequential (like the Kalman filter),
but where **two different, independently reference-validated
parallelism opportunities exist at the level above a single fit** — both
confirmed as real features in R's own most comprehensive GARCH package,
not just this project's own idea.

## Where this fits

- **Depends on:** Stage 4.1 (`_optimize`) only. No dependency on the
  state-space engine, no dependency on differencing.
- **Verification boundary**: Python's `arch` package was already
  installed earlier in this project (Stage 2.3 verification work) and
  is used directly here with real execution. R's `rugarch` could not be
  installed (same CRAN boundary as `forecast`/`lmtest` throughout this
  project) — its behavior below is from its own actual documentation and
  source excerpts (quoted directly, not summarized from memory), not
  independently executed.

---

## 1. Verified reference: Python `arch.arch_model`

```python
arch_model(y, x=None, mean='Constant', lags=0, vol='GARCH', p=1, o=0, q=1,
           power=2.0, dist='normal', hold_back=None, rescale=None)
# .fit(update_freq=1, disp='final', starting_values=None, cov_type='robust',
#      show_warning=True, tol=None, options=None, backcast=None)
```

Confirmed via `inspect.signature`. Relevant to this stage's scope
(`o=0` — asymmetric/leverage terms are Stage 7.2):
- **`mean='Constant'`** — separates the mean and variance equations
  explicitly; a plain constant mean by default (not zero, not an
  ARMA mean).
- **`cov_type='robust'` is the default, not `'classic'`** — the
  Bollerslev-Wooldridge QMLE-robust ("sandwich") standard error
  estimator, standard practice for GARCH given real returns data
  routinely violates the assumed conditional normality. `'classic'`
  gives the plain inverse-Hessian estimator instead. **Confirmed to be
  genuinely different numbers**, not just a different code path to the
  same answer (real fit, GARCH(1,1), n=1000): `omega` se `0.0134`
  (robust) vs `0.0159` (classic); `alpha` se `0.0195` vs `0.0213`;
  `beta` se `0.0228` vs `0.0289`. Same category of finding as Stage 6's
  Hessian-vs-OPG divergence for ARMA.

## 2. Documented reference: R `rugarch` (not independently executed — see boundary note above)

Confirmed directly from `rugarch`'s own documentation and source
(quoted, not paraphrased):

**Multi-series fitting, exact precedent for one of this handoff's two
parallelism designs**:
> *"Method for multiple fitting a variety of univariate GARCH and ARFIMA
> models... `cluster`: A cluster object created by calling `makeCluster`
> from the `parallel` package. If it is not NULL, then this will be
> used for parallel estimation."* — `multifit(multispec, data, ..., cluster=NULL, ...)`

**Multi-start optimization, exact precedent for the other**:
> *"The 'gosolnp' solver allows for the initialization of multiple
> restarts of the solnp solver with randomly generated parameters...
> The `solver.control` list then accepts: `n.restarts` is the number of
> solver restarts required (defaults to 1), `parallel` (logical), `pkg`
> (either snowfall or multicore), and `cores` (the number of cores or
> workers to use)."*

Both of this handoff's parallelism designs (section 5) are therefore not
speculative additions — they mirror real, documented features of the
most comprehensive existing GARCH implementation, independently arrived
at before this search confirmed the precedent.

---

## 3. Real fitted output, verified by execution — GARCH(1,1), n=1000

Generating process: `omega=0.05, alpha=0.10, beta=0.85`.
```python
am = arch_model(e, mean='Zero', vol='GARCH', p=1, o=0, q=1, dist='normal')
res = am.fit(disp='off', cov_type='robust')
# omega=0.043066  alpha=0.090638  beta=0.867923
# loglik=-1396.190867  aic=2798.381733  bic=2813.104999
# fit time: 0.0135s
```
Recovers the true generating parameters reasonably closely — MLE noise
on a single 1000-point series, not exact recovery, as expected.

---

## 4. Proposed Julia API

```julia
fit_garch(y, p::Integer=1, q::Integer=1;
          mean_spec::Symbol=:zero,          # :zero or :constant -- see note below
          dist::Symbol=:normal,             # :normal or :t
          cov_type::Symbol=:robust,         # :robust (default, matches Python) or :classic
          optimizer_method::Symbol=:lbfgs,
          n_restarts::Integer=1,            # multi-start, see section 5
          parallel::Bool=true) -> GarchModel
```

Design notes:
- **`mean_spec::Symbol=:zero`, not Python's `mean='Constant'` default**
  — a deliberate, documented divergence: this project's convention
  throughout (AR-X, ARMA, ARIMA) is to let the caller supply an
  already-demeaned or already-modeled series and keep each stage's own
  scope narrow; GARCH's own job is the variance equation, not the mean
  equation. `:constant` is available for parity with Python's default,
  but `:zero` matches the more common practitioner workflow (fit a mean
  model first — an AR-X or ARIMA already in this project — then fit
  GARCH to its residuals) more directly than defaulting to Python's
  built-in constant-mean convenience.
- **`cov_type` defaults to `:robust`**, matching Python's default and
  standard GARCH practice (returns data routinely isn't conditionally
  normal) — `:classic` available, same "document the divergence, pick
  the more defensible default" pattern as every prior stage's
  Hessian-vs-OPG-style choice.
- **`n_restarts`/`parallel`**: see section 5 — both real, precedent-
  validated features, not additions invented for this handoff alone.

### `GarchModel`

```julia
struct GarchModel <: UnivariateModel
    omega::Float64
    alpha::Vector{Float64}    # ARCH terms
    beta::Vector{Float64}     # GARCH terms
    sigma2::Vector{Float64}   # fitted conditional variance path, one per observation
    resid::Vector{Float64}
    se::Vector{Float64}       # [omega; alpha; beta] order
    loglik::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    p::Int
    q::Int
    cov_type::Symbol
    converged::Bool
end
```

**Naming note to resolve carefully during implementation, not glossed
over**: GARCH(p,q) convention itself is a notorious source of confusion
— Bollerslev's original notation has `p` as the *GARCH* (lagged
variance) order and `q` as the *ARCH* (lagged squared residual) order,
which is the *opposite* of how `arch_model`'s own `p`/`q` arguments are
named (`p` = ARCH order there). Confirm which convention this project's
argument names actually match before implementing — worth a dedicated,
explicit unit test asserting `fit_garch(y, 1, 1)`'s `alpha`/`beta`
lengths against a known-asymmetric-order case (e.g. GARCH(2,1)) rather
than trusting the docstring alone, given how easy this specific
confusion is to get backwards silently.

---

## 5. Parallelism — two real designs, both precedent-validated, both default-on

**What is *not* parallel, and why**: a single fit's variance recursion
(`h_t = omega + alpha*e[t-1]^2 + beta*h[t-1]`) is sequential by
construction — each variance depends on the immediately preceding one,
the same structural reason the Kalman filter's recursion (GaussianSSM,
Chapter Seven) isn't parallelizable either. No design here attempts to
parallelize within one fit's likelihood evaluation.

### Design A — multi-series fitting

```julia
fit_garch_multi(ys::AbstractVector{<:AbstractVector}, p::Integer=1, q::Integer=1;
                 parallel::Bool=true, kwargs...) -> Vector{GarchModel}
```
Fits the *same* `(p,q)` specification to each series in `ys`
independently — matches `rugarch::multifit`'s exact purpose (section 2),
the realistic case being portfolio-style volatility modeling across many
assets at once. Each series' fit shares no state with any other's, so
this is genuinely embarrassingly parallel. `Threads.@threads` over the
series list, guarded the same way as every other stage's parallel
design in this project (`Threads.nthreads() > 1`, and a minimum series
count — e.g. `length(ys) >= 4` — below which thread-spawning overhead
isn't worth it).

### Design B — multi-start optimization

```julia
# via fit_garch's own n_restarts keyword
fit_garch(y, 1, 1; n_restarts=8, parallel=true)
```
GARCH likelihood surfaces are known to have local-optimum sensitivity to
starting values — matches `rugarch`'s own `gosolnp`/`n.restarts`
rationale directly (section 2). `n_restarts > 1` generates that many
randomized starting parameter vectors (respecting the stationarity
constraint `alpha + beta < 1`), fits from each independently, and keeps
the highest-likelihood *converged* result. Independent optimizer runs
share no state — parallelize the same way as Design A, same guards.

**Both default to `parallel=true`, per the request** — but `n_restarts`
itself defaults to `1` (i.e., parallelism is available and on by
default, but multi-start itself isn't forced on unless requested,
matching `rugarch`'s own default of a single start unless `gosolnp` is
explicitly chosen). `parallel=false` is a cheap, explicit opt-out for
both designs, for reproducibility/debugging contexts — same reasoning
as every prior stage's `parallel` keyword in this project.

---

## 6. Comprehensive test matrix — real bulk verification, not a script sketch

Generated and fit **24 real cases** directly in this session: 8 true
`(omega, alpha, beta)` combinations x 3 seeds each, `n=800`, via
`arch_model`. **All 24 converged.** Mean absolute parameter recovery
error: `0.0388`; median `0.0317` — realistic MLE noise on finite
samples, not exact recovery (same honest framing already established
for Stage 6.8's order-selection accuracy: match what the reference
converges to, not an idealized ground truth).

```python
true_params = [
    (0.05,0.05,0.90),(0.05,0.10,0.85),(0.05,0.15,0.80),
    (0.10,0.05,0.85),(0.10,0.10,0.80),(0.10,0.15,0.75),
    (0.02,0.08,0.88),(0.02,0.12,0.82),
]
seeds = [1,2,3]
# for each (omega,alpha,beta) x seed: generate n=800 GARCH(1,1) series,
# fit via arch_model(mean='Zero', vol='GARCH', p=1, q=1), cov_type='robust'
```
Full 24-case output (true params, seed, fitted params, log-likelihood,
convergence flag) is the real target set — extend this same grid with
more `(omega,alpha,beta)` combinations and seeds to reach 100+ for the
full bulk suite; the structure above scales directly, this session ran
a representative fraction of it for time.

```julia
using Test

@testset "fit_garch — dual-verified core case" begin
    e = vec(readdlm("garch_shared.csv", ',', skipstart=1))  # n=1000, seed=11, from section 3

    m = fit_garch(e, 1, 1; mean_spec=:zero, cov_type=:robust)
    @test m.converged
    @test isapprox(m.omega, 0.043066; atol=1e-3)
    @test isapprox(m.alpha[1], 0.090638; atol=1e-3)
    @test isapprox(m.beta[1], 0.867923; atol=1e-3)
    @test isapprox(m.loglik, -1396.190867; atol=1e-2)

    m_classic = fit_garch(e, 1, 1; mean_spec=:zero, cov_type=:classic)
    @test !isapprox(m.se, m_classic.se; atol=1e-3)   # genuinely different, per section 1

    @test all(m.sigma2 .> 0)   # variance must stay positive throughout the recursion
end

@testset "fit_garch_multi — parallel matches serial" begin
    series = [vec(readdlm("garch_shared.csv", ',', skipstart=1)) for _ in 1:6]  # same series 6x, cheap check

    results_par = fit_garch_multi(series, 1, 1; parallel=true)
    results_serial = fit_garch_multi(series, 1, 1; parallel=false)
    for i in eachindex(series)
        @test isapprox(results_par[i].omega, results_serial[i].omega; atol=1e-10)
    end
end

@testset "fit_garch multi-start — never worse than single start" begin
    e = vec(readdlm("garch_shared.csv", ',', skipstart=1))
    m1 = fit_garch(e, 1, 1; n_restarts=1)
    m8 = fit_garch(e, 1, 1; n_restarts=8, parallel=true)
    @test m8.loglik >= m1.loglik - 1e-6   # multi-start should never find a worse optimum
end

@testset "bulk recovery, 24 cases" begin
    # loop the 24-case grid from section 6's Python script (regenerated
    # identically in Julia), asserting convergence + parameter error
    # bounded similarly to the verified Python recovery rate above
end
```

---

## 7. Performance

Real reference target: Python's `arch` package fits GARCH(1,1) on
n=1000 in **0.0135s** — a compiled (Cython) backend, not slow
interpreted code, same honest framing as every prior performance
section in this project: the bar to beat is a genuinely fast reference,
not an easy one. As with Chapter Seven, this project's actual advantage
is structural (one language, `ForwardDiff` differentiates the real
recursion directly, no separate compiled sub-language needed) rather
than an assumed raw-speed win — confirm this with real profiling once
running, not asserted here.

The two parallel designs (section 5) are where this stage's performance
story is genuinely stronger than either reference's *default* behavior
specifically: both `rugarch`'s `multifit`/`gosolnp` parallelism require
the *user* to construct and manage a `parallel::makeCluster` object
explicitly (real ceremony, and process-based, not thread-based —
meaningful startup/serialization overhead per worker). `parallel=true`
here is genuinely zero-setup by comparison — a real, checkable
difference once both are actually benchmarked side by side, not just an
API convenience claim.

---

## 8. What to do with this

1. **Resolve the `p`/`q` GARCH-order-vs-ARCH-order naming convention
   explicitly** (section 4's flagged risk) before implementing anything
   else — get this backwards and every downstream test silently
   validates the wrong thing.
2. Implement `fit_garch`/`GarchModel` per sections 4-5.
3. Implement `fit_garch_multi` and `n_restarts`, both default-on for
   `parallel`, both precedent-validated per section 2.
4. Run the tests in section 6; extend the 24-case bulk grid to 100+
   using the exact same generating structure, not a different one.
5. Update `development-sequence.md`'s Stage 7.1 row: mark implemented,
   record the `cov_type` robust-vs-classic finding and both parallelism
   designs' real precedent explicitly.
