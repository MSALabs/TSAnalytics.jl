# Handoff: Stage 6.5 — Non-Seasonal ARMA via State Space + MLE

Status: **done.** `fit_arma`/`ArmaModel` (`src/arma.jl`,
`test/test_arma.jl`) built per sections 5-8, verified end-to-end against
real R and Python (not just the handoff's own transcribed numbers) on
the bundled `arma11_fit.csv` (moved to `test/verification/arma/`, same
convention as `gaussianssm`'s move). Two real bugs found in this
handoff's own sketch before trusting it, plus one design gap closed
beyond what the handoff specified:

1. **§6's AIC/BIC formula is wrong**: `k = p + q + (include_mean ? 1 : 0)`
   omits `+1` for `sigma2`. Confirmed by computing R's actual `BIC()`
   directly on the fitted model -- it only matches with `k=3` (not `2`)
   for this ARMA(1,1) example. This is the identical "dof must count σ²"
   trap the *parent* `stage-6-arima-handoff.md`'s own checklist (§8)
   already flags for the full `ARIMAModel` -- this handoff's sketch just
   didn't apply that lesson to itself. Fixed: `k` now always includes
   `+ 1` for `sigma2`.
2. **`partrans`/`invpartrans` (Stage 4.2, already-shipped) are not
   `ForwardDiff`-safe**, and this handoff's entire objective-function
   design (§6) calls `partrans` from inside the AD-differentiated
   closure. Confirmed directly: `_optimize(obj_using_partrans, x0)`
   threw `MethodError: no method matching Float64(::ForwardDiff.Dual...)`
   before this fix, because both functions hardcoded
   `collect(Float64, ...)` internally rather than using the input's own
   element type. This had no caller inside an AD context before Stage
   6.5 (Stage 4.2 was built and tested standalone), so it was latent,
   not previously broken. Fixed both to be element-type-generic.
3. **`include_mean=true`'s design in §6 is incomplete**: the sketch
   pre-demeans `y` once with the sample mean, then fits ARMA on the
   residual -- but real R's `include.mean=TRUE` jointly optimizes the
   mean *with* the AR/MA coefficients, a genuinely different (and more
   accurate) point estimate on finite samples. Confirmed by an
   independent verification run this session (a mean-shifted version of
   the handoff's own series, since its own ground truth in §3 only
   covers `include_mean=false`): `fit_arma` estimates the mean jointly,
   verified to match a fresh R `arima(..., include.mean=TRUE)` run
   closely (`ar1`, `ma1`, `mean`, and all three standard errors within
   1e-3).

`ForwardDiff` is now a direct dependency (`Project.toml`) rather than
only transitive via `Optim`, since `_hessian_se`/`_opg_se` call
`ForwardDiff.hessian`/`ForwardDiff.jacobian` directly -- matching this
handoff's own §4 observation that exact, no-extra-pass automatic
differentiation is a genuine Julia-side structural advantage over R's
and Python's compiled-but-hand-differentiated backends, not just a
speed claim.

---

For a fresh Claude Code session picking this up with no prior context.
This is the integration stage: wiring three already-verified pieces
together — `GaussianSSM`'s `kalman_filter` (dual-verified, GaussianSSM
branch), Stage 4.1's `_optimize`, Stage 4.2's `partrans`/`invpartrans`
(verified against R's actual C source) — into an actual `fit_arma()`
that *finds* coefficients from data, not just evaluates a likelihood at
known ones. No new numerical machinery, but exactly the kind of
"correctly wire together three correct pieces" step where a sign error
or a mismatched convention silently produces a plausible-but-wrong fit.

## Where this fits

- **Depends on:** GaussianSSM (`kalman_filter`, dual-verified on its own
  branch), Stage 4.1 (`_optimize`), Stage 4.2 (`partrans`/`invpartrans`).
- **Scope, deliberately narrower than either reference**: non-seasonal
  ARMA(p,q) only — no differencing (`d`, Stage 6.6), no seasonal terms
  (Stage 6.7), no exogenous regressors (Stage 7). R's `arima()` and
  Python's `ARIMA` class both support all of that at once; this stage's
  signature should be designed to *extend* cleanly into 6.6/6.7 later,
  not pretend to cover them now.

---

## 1. Verified reference: R `stats::arima()`

```r
arima(x, order = c(0L,0L,0L), seasonal = list(order=c(0L,0L,0L), period=NA),
      xreg = NULL, include.mean = TRUE, transform.pars = TRUE, fixed = NULL,
      init = NULL, method = c("CSS-ML", "ML", "CSS"), n.cond,
      SSinit = c("Gardner1980", "Rossignol2011"), optim.method = "BFGS",
      optim.control = list(), kappa = 1e6)
```

Confirmed via `args(stats::arima)`. Relevant to this stage's scope:
- **`method` defaults to `"CSS-ML"`, not pure `"ML"`** — a two-stage
  hybrid: fit via conditional sum of squares first (fast, crude) to get
  starting values, then refine via full ML. `"ML"` alone skips the CSS
  stage and optimizes from a fixed default start. This is a genuine
  default-behavior fact worth matching the *spirit* of (better starting
  values -> faster, more robust convergence), not just the name.
- **`include.mean`**: whether to estimate a mean/intercept term — default
  `TRUE`.
- **`transform.pars`**: whether to use the Monahan reparametrization
  during optimization — default `TRUE`, exactly what Stage 4.2's
  `partrans` already implements.
- **`optim.method = "BFGS"`** — R's default optimizer, not L-BFGS.
- Standard errors: from **`var.coef`**, computed from the **Hessian** of
  the log-likelihood at the optimum (standard asymptotic MLE theory).

## 2. Verified reference: Python `statsmodels.tsa.arima.model.ARIMA`

```python
ARIMA(endog, exog=None, order=(0,0,0), seasonal_order=(0,0,0,0), trend=None,
      enforce_stationarity=True, enforce_invertibility=True,
      concentrate_scale=False, trend_offset=1, dates=None, freq=None,
      missing='none', validate_specification=True)
# .fit(start_params=None, transformed=True, includes_fixed=False, method=None,
#      method_kwargs=None, gls=None, gls_kwargs=None, cov_type=None, ...)
```

- `enforce_stationarity`/`enforce_invertibility`: matches the same
  purpose as R's `transform.pars`, split into two named flags (AR and
  MA constraints controllable separately — a real flexibility R's single
  `transform.pars` doesn't offer).
- **Standard errors default to `cov_type='opg'`** — outer-product-of-
  gradients, **not** the Hessian. Confirmed directly in the fitted
  summary output (section 3).

## 3. The fitting itself — dual-verified, real optimizer convergence, not fixed-coefficient evaluation

New dataset (this combination — genuine estimation, not evaluation at
known coefficients — didn't exist in any prior verification set):
```python
np.random.seed(7); n=200
e = np.random.randn(n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.6*y[t-1] + e[t] + 0.25*e[t-1]
```

**R, `method="ML"`** (pure MLE, matching this stage's actual scope):
```
ar1=0.5465817922  ma1=0.2717023914
loglik=-280.5177637  aic=567.0355274
se: ar1=0.0878534233  ma1=0.1074810562   (Hessian-based)
fit time: 0.0082s
```

**R, `method="CSS-ML"` (the actual default)**:
```
ar1=0.5465783235  ma1=0.2717063783   (essentially identical to pure ML)
loglik=-280.5177637
fit time: 0.0022s   <- ~4x faster than pure ML, from better starting values
```

**Python, default fit**:
```
ar.L1=0.54658944  ma.L1=0.27169267  sigma2=0.96440005
loglik=-280.51776373493647  aic=567.0355274698729
se: ar.L1=0.081  ma.L1=0.092   (OPG-based -- NOT the same numbers as R's Hessian-based se)
fit time: 0.0351s
```

**Three real findings here:**

1. **Coefficients converge to matching values across all three runs**
   (R-ML, R-CSS-ML, Python) — confirms the underlying MLE surface is the
   same regardless of path, as expected. Loglik/AIC match to full
   precision between R and Python (`-280.5177637...` both).
2. **Standard errors genuinely differ between R and Python** — R's
   Hessian-based (0.0879, 0.1075) vs. Python's OPG-based (0.081, 0.092)
   are close but **not the same numbers**, because they're different
   (both asymptotically valid) estimators of the same quantity. This is
   the same category of finding as GaussianSSM's smoother-convention
   work — pick one deliberately, document why, don't silently blend them.
3. **A real, honest performance gap**: R (2.2-8ms) is meaningfully
   *faster* than Python (35ms) here, both using compiled backends
   internally. This matters for section 4 — the performance bar to aim
   for is R's number, not just "beat Python," which is the more
   ambitious and more honest target.

---

## 4. Performance — real reference targets, honest about what's unverified

**Reference targets, from section 3, both compiled backends**: R
CSS-ML ~2.2ms, R pure ML ~8ms, Python ~35ms, for an ARMA(1,1) fit on
n=200. **These are real, measured numbers.** What follows is Julia
performance *guidance* aimed at these targets — **not yet verified
against an actual Julia run**, since no Julia runtime is available in
this sandbox. Treat this section as a set of concrete things to check
once it's actually running, not a claim already confirmed.

**The honest performance story, stated precisely**: R's and Python's
Kalman filters are *already* compiled (C and Cython respectively) — the
naive "Julia beats slow interpreted R/Python" framing doesn't actually
apply here, and overclaiming it would be dishonest. **The real Julia
advantage is structural, not raw-speed**: the same Julia code that's
readable and directly maintained (no separate Cython/Fortran layer) also
compiles to native machine code, and `ForwardDiff` differentiates
through it directly — R's C code and Python's Cython code both need
either hand-derived analytic gradients or numerical differencing bolted
on separately. That's a genuine productivity/correctness advantage
(one language, one file, automatic and exact derivatives) even if raw
wall-clock speed ends up merely competitive rather than dramatically
faster — say this honestly rather than promising a speed win that a
real benchmark might not deliver.

**Concrete things to check once this runs in Julia:**
- **Objective-function allocation**: `_optimize`'s objective gets called
  potentially hundreds of times per fit. `build_statespace`/
  `stationary_cov` currently allocate fresh `r x r` matrices every call —
  fine for small `r` (typical ARMA orders), worth profiling
  (`@allocated`, `BenchmarkTools.jl`) before assuming it's a problem.
  Don't pre-optimize this without a profile showing it matters — same
  discipline already applied to MSTL's parallelization design.
- **Type stability**: `@code_warntype` on the objective function closure
  — `Any`-typed fields anywhere in the closure environment silently kill
  performance and often aren't obvious from the code alone.
- **Hessian for standard errors**: since R's Hessian-based SE (section
  3) is the design's target (see section 5), and `_optimize` already
  uses `AutoForwardDiff()`, the Hessian is available via `ForwardDiff`
  directly (forward-over-forward, or `ForwardDiff.hessian` on the
  objective) — no separate numerical-differencing pass needed, unlike
  what R's C code or Python's Cython code has to do.

---

## 5. Proposed Julia API

```julia
fit_arma(y, order::Tuple{Int,Int};
         include_mean::Bool=true,
         method::Symbol=:ml,              # :ml (this stage's actual scope) or :css_ml (CSS-derived starting values, R's default spirit)
         se_type::Symbol=:hessian,        # :hessian (R-style, default) or :opg (Python-style)
         optimizer_method::Symbol=:lbfgs, # passed to _optimize (Stage 4.1)
         start_params::Union{Nothing,Vector{Float64}}=nothing) -> ArmaModel
```

Design notes:
- **`order::Tuple{Int,Int}`** — `(p, q)` only, matching this stage's
  scope exactly (not R's/Python's 3-tuple `(p,d,q)` — `d` doesn't exist
  yet, deliberately, per Stage 6.6). Naming this `order` rather than
  separate `p`/`q` arguments anticipates 6.6 extending it to a 3-tuple
  without renaming the argument.
- **`method::Symbol`**: `:ml` (default, matches this stage's literal
  scope — "ARMA via state space + MLE") / `:css_ml` (R's actual default
  behavior — CSS-derived starting values feeding into the same ML
  optimization). Not defaulting to `:css_ml` despite it being R's
  default and faster, specifically because this stage's stated scope is
  MLE — but `:css_ml` should exist and be easy to reach for, given the
  real ~4x speedup observed in section 3.
- **`se_type::Symbol=:hessian`**: matches R's approach by default (more
  standard textbook asymptotic MLE theory, and cheaply available via
  `ForwardDiff` per section 4) — `:opg` available for anyone specifically
  wanting Python-comparable numbers. Document the discrepancy explicitly
  in the docstring, same as the GaussianSSM smoother's convention notes.
- **`start_params`**: matches Python's exact name/purpose — explicit
  override, bypassing whatever `method` would otherwise compute.

### `ArmaModel`

```julia
struct ArmaModel <: UnivariateModel
    ar::Vector{Float64}
    ma::Vector{Float64}
    mean::Union{Nothing,Float64}
    se::Vector{Float64}          # matches [ar; ma; mean] order
    loglik::Float64
    sigma2::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    order::Tuple{Int,Int}
    method::Symbol
    se_type::Symbol
    converged::Bool
end
```

---

## 6. Implementation sketch

```julia
"""
    fit_arma(y, order; include_mean=true, method=:ml, se_type=:hessian,
             optimizer_method=:lbfgs, start_params=nothing) -> ArmaModel

Fit a non-seasonal ARMA(p,q) model by maximum likelihood, via the
GaussianSSM Kalman filter (dual-verified against R and Python
internally) wired to Stage 4.1's optimizer and Stage 4.2's Monahan
reparametrization. See the Stage 6.5 handoff doc for the verified
R/Python fitting ground truth this should reproduce, the real
performance reference targets, and why `se_type` defaults to R's
Hessian-based convention rather than Python's OPG-based one.
"""
function fit_arma(y, order::Tuple{Int,Int};
                   include_mean::Bool=true,
                   method::Symbol=:ml,
                   se_type::Symbol=:hessian,
                   optimizer_method::Symbol=:lbfgs,
                   start_params::Union{Nothing,Vector{Float64}}=nothing)
    method in (:ml, :css_ml) || throw(ArgumentError("method must be :ml or :css_ml"))
    se_type in (:hessian, :opg) || throw(ArgumentError("se_type must be :hessian or :opg"))
    p, q = order
    yv = tsvalues(y)
    mu = include_mean ? sum(yv)/length(yv) : 0.0
    yc = yv .- mu

    x0 = if start_params !== nothing
        start_params
    elseif method == :css_ml
        _css_start_values(yc, p, q)   # conditional-sum-of-squares starting values (see note below)
    else
        zeros(p + q)   # matches R's ML-only default start (unconstrained space -> phi=theta=0)
    end

    function objective(raw::Vector)
        phi = p > 0 ? partrans(raw[1:p]) : Float64[]
        theta = q > 0 ? partrans(raw[p+1:p+q]) : Float64[]
        ssm = build_statespace(phi, theta)
        loglik, sigma2, v, F, converged = kalman_filter(ssm, yc)
        return converged ? -loglik : 1e10   # smooth penalty for the optimizer, not a crash
    end

    result = _optimize(objective, x0; method=optimizer_method)
    phi_hat = p > 0 ? partrans(result.minimizer[1:p]) : Float64[]
    theta_hat = q > 0 ? partrans(result.minimizer[p+1:p+q]) : Float64[]
    ssm = build_statespace(phi_hat, theta_hat)
    loglik, sigma2, = kalman_filter(ssm, yc)

    se = se_type == :hessian ? _hessian_se(objective, result.minimizer) :
                                 _opg_se(objective, result.minimizer, length(yc))

    k = p + q + (include_mean ? 1 : 0)
    aic = -2*loglik + 2*k
    bic = -2*loglik + k*log(length(yc))

    return ArmaModel(phi_hat, theta_hat, include_mean ? mu : nothing, se,
                      loglik, sigma2, aic, bic, length(yc), order, method, se_type, result.converged)
end
```

**Deliberately not fully specified**: `_css_start_values` (the
conditional-sum-of-squares starting-value heuristic — standard textbook
technique: minimize squared one-step residuals treating pre-sample
values as zero, no Kalman filter needed, cheap) and `_hessian_se`/
`_opg_se` (straightforward given `ForwardDiff.hessian` is already
available via the `_optimize` machinery, but worth writing carefully
against a hand-checked small example rather than assumed correct from
the sketch above — same discipline as every "elided" section in this
project's other handoffs).

---

## 7. `show` — same `CoefTable` pattern as `ARXModel`

```julia
function Base.show(io::IO, m::ArmaModel)
    p, q = m.order
    names = vcat(["ar$i" for i in 1:p], ["ma$i" for i in 1:q],
                 m.mean !== nothing ? ["mean"] : String[])
    coefs = vcat(m.ar, m.ma, m.mean !== nothing ? [m.mean] : Float64[])
    z = coefs ./ m.se
    pval = 2 .* (1 .- _std_normal_cdf.(abs.(z)))
    ct = CoefTable(hcat(coefs, m.se, z, pval), ["Coef.", "Std. Error", "z", "Pr(>|z|)"], names)
    println(io, "ARMA(", p, ",", q, ")", m.mean !== nothing ? " with mean" : "",
                 ", n=", m.nobs, " (", m.method, ", se: ", m.se_type, ")")
    println(io, ct)
    print(io, "Log-likelihood: ", round(m.loglik, digits=2),
          "  AIC: ", round(m.aic, digits=2), "  BIC: ", round(m.bic, digits=2))
    m.converged || print(io, "\nWARNING: optimizer did not converge")
end
```

---

## 8. Comprehensive test matrix

Reuses `GaussianSSM`'s already-built bulk-verification infrastructure
(the 8-dataset, multi-order fixture set from the GaussianSSM branch) for
breadth — this handoff adds the fitting-specific cases on top.

| Case | Verified against |
|---|---|
| ARMA(1,1), `method=:ml`, real optimizer convergence | R `method="ML"`: `ar1=0.5465817922, ma1=0.2717023914, loglik=-280.5177637` (section 3) |
| ARMA(1,1), `method=:css_ml` | R `method="CSS-ML"`: `ar1=0.5465783235, ma1=0.2717063783` — should be close to but not necessarily bit-identical to the `:ml` case, given different starting points |
| `se_type=:hessian` | R's `sqrt(diag(var.coef))`: `ar1=0.0878534233, ma1=0.1074810562` |
| `se_type=:opg` | Python's reported std err: `ar.L1=0.081, ma.L1=0.092` — expect a real, non-matching difference from `:hessian`, per section 3's finding; a test asserting they're *different*, not equal, same pattern as GaussianSSM's cross-convention tests |
| AIC/BIC | Both R and Python: `aic=567.0355...` (Python's BIC: `576.930479569517` — verify against R's own BIC computation too before trusting) |

```julia
using Test, DelimitedFiles

@testset "fit_arma — ARMA(1,1), dual-verified real fit" begin
    y = vec(readdlm("arma11_fit.csv", ',', skipstart=1))  # from section 3's generating code

    m_ml = fit_arma(y, (1,1); include_mean=false, method=:ml)
    @test m_ml.converged
    @test isapprox(m_ml.ar[1], 0.5465817922; atol=1e-3)
    @test isapprox(m_ml.ma[1], 0.2717023914; atol=1e-3)
    @test isapprox(m_ml.loglik, -280.5177637; atol=1e-2)

    m_cssml = fit_arma(y, (1,1); include_mean=false, method=:css_ml)
    @test m_cssml.converged
    @test isapprox(m_cssml.ar[1], 0.5465783235; atol=1e-3)

    m_hess = fit_arma(y, (1,1); include_mean=false, se_type=:hessian)
    @test isapprox(m_hess.se[1], 0.0878534233; atol=1e-2)

    m_opg = fit_arma(y, (1,1); include_mean=false, se_type=:opg)
    @test isapprox(m_opg.se[1], 0.081; atol=1e-2)
    @test !isapprox(m_hess.se[1], m_opg.se[1]; atol=1e-4)  # genuinely different, not a bug

    @test_throws ArgumentError fit_arma(y, (1,1); method=:bogus)
    @test_throws ArgumentError fit_arma(y, (1,1); se_type=:bogus)

    io = IOBuffer()
    show(io, m_ml)
    @test occursin("ARMA(1,1)", String(take!(io)))
end
```

For broader coverage (multiple orders, multiple datasets), extend
`GaussianSSM`'s existing bulk-test loop to also call `fit_arma` and
compare against the same R/Python fixed-coefficient ground truth already
established there — the fitting result should converge back to
approximately those known coefficients when the data was generated from
them, giving a second, independent check beyond the single ARMA(1,1)
case above.

---

## 9. What to do with this

1. Implement `fit_arma`/`ArmaModel` per sections 5-6, including the
   flagged-incomplete `_css_start_values`/`_hessian_se`/`_opg_se` helpers.
2. Run the tests in section 8.
3. Profile before optimizing anything in section 4 — get it correct and
   working first, then check `@allocated`/`@code_warntype` against the
   real R/Python timing targets, and only optimize what a profile
   actually shows matters.
4. Update `development-sequence.md`'s Stage 6.5 row: mark implemented,
   record the real R (2-8ms) / Python (35ms) timing targets, and the
   Hessian-vs-OPG standard error finding so it isn't rediscovered later.
5. This is the natural point to close the "usable" bar for the
   GaussianSSM 2-week trial and the stronger Sept pitch demo — a real
   `fit_arma()` call, live, matching R/Python's fitted coefficients, is
   exactly the demo this whole track has been building toward.
