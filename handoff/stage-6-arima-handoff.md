# Handoff: Stage 6 — merging GaussianSSM into TSAnalytics.jl, then ARIMA/SARIMA fitting

Status: **Part A done, Part B not started.** `Pkg.test()` is green
(2664 assertions, including all 364+4 dual-verified cases) with the
engine merged into `src/statespace/gaussianssm.jl` and tests/data moved
into `test/`. Every item in §1 (mechanical steps), §2 (must-fix), and
the small items in §3.4 landed; §3.1's keyword form was added alongside
(not instead of) the positional one the generated corpus uses. Two
items were deliberately deferred rather than attempted, per their own
sections' guidance that they're out of scope for "get the merge green":
§3.3 (missing-data handling) and §3.5 (regenerating the seasonal corpus
with the `p=2` truncation bug fixed — a Python/R task, not a Julia code
change). **One correction to this document's own §2.1 code snippet**:
it shows `GaussianSSM{T}`/`build_statespace`'s signature changing, but
doesn't mention that `kalman_filter`'s internal `v`/`F`/`a` buffers were
still hardcoded `Vector{Float64}` -- assigning a `ForwardDiff.Dual` into
those throws, silently defeating the whole point of the generic-type
change. Fixed (typed from `eltype(stationary_cov(ssm))`) and verified
end-to-end through the real `_optimize` path, not just `ForwardDiff.gradient`
in isolation. **A second correction**: §2.1's claim that
`AbstractVector{<:Real}` fixes the `combined_ar_ma([0.5], [], ...)`
bare-`[]` ergonomic trap doesn't hold under direct testing (`Vector{Any}`
is not `AbstractVector{<:Real}`) -- switched to unconstrained
`AbstractVector` for the polynomial-coefficient arguments instead, which
does. Full details in `development-sequence.md`'s Stage 6.1-6.4 row.
Next: §3.5's corpus regeneration, then Part B (§4 onward).

---

For a fresh Claude Code session picking this up with no prior context.
Two distinct jobs live in this document and they should not be
interleaved:

- **Part A (§1–§3)** — merge the validated `GaussianSSM` engine into
  `TSAnalytics.jl` and patch what the standalone review found. Purely
  mechanical plus five small code changes. Finish this and get a green
  `Pkg.test()` before touching Part B.
- **Part B (§4–§9)** — build `fit_arima`/`fit_sarima` on top of it
  (Stage 6.5–6.8). This is where the genuinely hard decisions are, and
  §5 is the one that will cost days if it's got wrong.

## Where this fits

- **Depends on:** Stage 0.2 (`tsvalues`/`tsindex`), Stage 4.1
  (`_optimize`), Stage 4.2 (`partrans`/`invpartrans`, already verified
  against R's `arima.c`), Stage 5.1 (`CoefTable` display pattern,
  StatsAPI contract), Stage 5.2 (forecast object).
- **Blocks:** Stage 7 (SARIMAX with `xreg`), and the whole of
  `SeasonalAdjustment.jl` — RegARIMA is a thin layer over this.
- **Engine status:** Stage 6.1–6.4 are **done and validated**. 364
  dual-verified (R + Python) likelihood cases plus 4 smoother cases pass.
  Independent re-execution of the algorithm reproduced every one of them;
  worst deviation 2.5e-07 against a 1e-04 tolerance. Do not re-litigate
  the filter's correctness — spend the effort on §5 and §6 instead.

---

# PART A — THE MERGE

## 1. Mechanical steps

1. **Strip the module wrapper.** Delete the `module GaussianSSMs` /
   `end # module` pair and the header comment block explaining why the
   trailing `s` existed — that rationale dies with the wrapper. The
   Windows case-insensitivity note (§ the file's own header) is why the
   standalone repo couldn't have `GaussianSSM.jl` wrapping
   `gaussianssm.jl`; inside TSAnalytics there is no wrapper, so
   `src/statespace/gaussianssm.jl` is safe.

2. **Place at `src/statespace/gaussianssm.jl`**, included from
   `src/TSAnalytics.jl` **after `interface.jl`**. The `statespace/`
   subdirectory is deliberate: Stage 8's diffuse/time-varying
   generalization goes beside it, not inside it.

3. **Hoist the import.** Drop `using LinearAlgebra: I, kron` from the
   file; TSAnalytics already loads `LinearAlgebra` at package level.

4. **Export nothing yet.** Delete the `export` line. `GaussianSSM`,
   `kalman_filter`, and `kalman_smoother` will all change signature at
   Stage 8 (time-varying `Z_t`/`T_t`, diffuse init, a `Z` field in the
   struct). Exporting them now means either a breaking change in a
   stable release or living with a bad interface for a year. Keep them
   reachable as `TSAnalytics.kalman_filter` for tests and internal use,
   and export only once Stage 8 has settled the shape. This is the
   incremental-release policy applied honestly.

5. **Tests.** Move all three test files to `test/`, add the includes to
   `runtests.jl`, and relocate the CSVs to
   `test/verification/gaussianssm/`. Every `joinpath(@__DIR__, "..", "verification", ...)`
   needs updating — there are paths in all three files plus 8 dataset
   loads at the top of the bulk file.

6. **Normalize line endings.** `test_gaussianssm_bulk.jl` and every CSV
   under `verification/bulk/` are CRLF (generated on Windows). Add a
   `.gitattributes` with `* text=auto` before committing, or the diff
   noise will be permanent. `readdlm` handles CRLF, so this is hygiene,
   not a bug — but if the first Julia run ever fails with a `MethodError`
   on `kalman_filter(::GaussianSSM, ::Vector{Any})`, CRLF parsing is the
   first thing to check.

7. **Do not commit `Manifest.toml`.** It belongs to the standalone dev
   bundle, not to a registered package.

8. **No new dependencies.** The engine needs only `LinearAlgebra`. Keep
   it that way — see §3.2 for why the Lyapunov fix should be hand-rolled
   rather than pulling in `MatrixEquations.jl`.

9. **CI weight.** The bulk file is 2,582 lines / 364 cases and loads 8
   datasets. It's fast (the filter is O(n·r²)) but it dominates the test
   file count. Consider tagging it so it can be excluded from a
   fast local loop, e.g. `if get(ENV, "TSANALYTICS_FULL_TESTS", "1") == "1"`.

## 2. Must-fix before Part B

These two block Stage 6.5. Do them during the merge.

### 2.1 Concrete types will block ForwardDiff

`build_statespace(ar::Vector{Float64}, ma::Vector{Float64})` and
`struct GaussianSSM` with `Matrix{Float64}`/`Vector{Float64}` fields
mean that the moment `_optimize` passes a `Vector{ForwardDiff.Dual}`
through the objective, you get a `MethodError` — and Stage 4.1 was
built specifically around `AutoForwardDiff()`. Fix now, while there is
exactly one caller:

```julia
struct GaussianSSM{T<:Real}
    T::Matrix{T}
    R::Vector{T}
    r::Int
end

function build_statespace(ar::AbstractVector{<:Real}, ma::AbstractVector{<:Real})
    T = promote_type(eltype(ar), eltype(ma), Float64)
    ...
end
```

`kalman_filter`/`kalman_smoother`/`stationary_cov` all need the same
loosening on `y::AbstractVector{<:Real}`. Note the field named `T`
shadowing the type parameter `T` — rename the type parameter (`{S<:Real}`)
or the field; the current field name matches the state-space literature
and is worth keeping.

Side benefit: this also fixes the ergonomic trap that
`combined_ar_ma([0.5], [], [0.4], [], 12)` currently throws, because a
bare `[]` literal is `Vector{Any}`. Users *will* write that.

### 2.2 `seasonal_poly` silently corrupts when `s == 0`

`poly[i*s + 1]` with `s == 0` writes to `poly[1]`, destroying the
leading 1.0 of the polynomial. Demonstrated:

```
combined_ar_ma([0.5], [0.4], [0.3], [0.2], 0)  ->  ar = [-0.2], ma = [0.06]
```

No error, wrong answer. Every current caller passes `max(s, 1)`, which
masks it — and Stage 6.5 will be constructing that `s` from user input,
where `s = 0` for a non-seasonal model is the natural encoding. Add:

```julia
isempty(coefs) || s >= 1 ||
    throw(ArgumentError("seasonal_poly: s must be >= 1 when seasonal coefficients are given (got s=$s)"))
```

and decide explicitly at the `fit_sarima` boundary whether `s=0` or
`s=1` means "non-seasonal".

## 3. Should-fix, with the evidence

### 3.1 Argument-order footgun in `combined_ar_ma`

The signature is `(phi, Phi, theta, Theta, s)` — AR, **seasonal AR**,
MA, seasonal MA. The optimizer parameter vector is ordered
`(phi, theta, Phi, Theta)` (§4.2). These two orderings differ, silently,
by a transposition of the middle two arguments, and both are
`Vector{Float64}`, so a swap type-checks and returns plausible garbage.
Convert to keywords:

```julia
combined_ar_ma(; phi, seasonal_phi, theta, seasonal_theta, s)
```

The 364 generated bulk cases would need regenerating, or a positional
method retained for them. Worth it — this is exactly the class of bug
that survives to a release.

### 3.2 The `kron` Lyapunov solve is the scaling wall

`stationary_cov` builds a dense r²×r² system. Measured against a
doubling (Smith) iteration, `Q ← Q + A Q Aᵀ`, `A ← A²`:

| Model | r | kron | doubling | speedup | agreement |
|---|---|---|---|---|---|
| SARIMA (1,1)×(1,1)₁₂ | 14 | 1.2 ms | 0.3 ms | 5× | 1.7e-16 |
| SARIMA (2,2)×(1,1)₁₂ | 15 | 8.0 ms | 0.3 ms | 27× | 1.3e-16 |
| SARIMA (1,1)×(2,2)₁₂ | 26 | 125 ms | 0.6 ms | 200× | 1.5e-16 |
| SARIMA (1,1)×(1,1)₅₂ | 54 | 2.09 s | 1.1 ms | 1874× | 1.3e-16 |
| SARIMA (1,1)×(2,2)₅₂ | 106 | 38.8 s | 4.1 ms | 9414× | 7.6e-17 |

Log-likelihoods from the two paths agree to 1.1e-13 — far inside the
1e-04 bulk tolerance, so **the existing 364-case corpus re-validates the
replacement for free.** That's the cheap way to land this: swap the
solver, rerun the corpus, done.

Why it matters at Stage 6 specifically: this cost is paid *per
likelihood evaluation*. A 200-iteration L-BFGS fit of a weekly
SARIMA(1,1)×(1,1)₅₂ spends ~7 minutes in initialization alone. Even
monthly (1,1)×(2,2)₁₂ is 25 s of pure overhead per fit.

Twelve lines of `LinearAlgebra`-backed matrix ops, no new dependency —
consistent with the "LinearAlgebra over hand-rolled loops" principle,
since these are BLAS-level operations, not element loops. Symmetrize
the result (`(Q + Q')/2`) on return.

### 3.3 Missing-data handling is absent

A single `NaN` returns `converged=false, loglik=-Inf` with no
diagnostic. `stats::arima()` handles `NA` natively, and real seasonal
adjustment data has gaps. The standard fix is small — skip the update
when `y[t]` is missing:

```julia
if ismissing(y[t]) || isnan(y[t])
    v[t] = 0.0; F[t] = 1.0            # contribute nothing to the likelihood
    a = T * a                          # predict only, no gain applied
    P = T * P * T' + R * R'
else
    ... existing update ...
end
```

and exclude those `t` from both the `sigma2` sum and `sum(log(F))`,
reducing the effective `n` accordingly. **Not Stage 6 scope** if you
want to ship sooner, but it is the first thing real data hits, and
retrofitting it means touching the smoother's forward pass too. Flag the
decision explicitly in `development-sequence.md` either way.

### 3.4 Smaller items

- **Stale comment.** The trailing block in `test_gaussianssm.jl` says
  seasonal `combined_ar_ma` is "NOT YET COVERED". It is — 20 cases in
  the bulk file. Delete it; a future reader will otherwise duplicate
  work that's already done.
- **`-Inf` is not a smooth penalty.** `kalman_filter`'s docstring says
  the sentinel lets an optimizer "penalize smoothly". It does not:
  `-Inf` gives no gradient direction and will produce `NaN` derivatives
  under ForwardDiff. In practice `partrans` keeps you inside the
  stationary region so it shouldn't fire — but §4.4 says what to do when
  it does. Reword the docstring so nobody relies on the claim.
- **Inconsistent return contract.** Early exits return truncated
  `v[1:t]`/`F[1:t]`; the success path returns full-length vectors.
  Residual diagnostics downstream will assume `length(v) == n`. Return
  `Float64[]` on all failure paths, or document it loudly.
- **`n == 0` guard.** With an empty series, `sigma2` is computed as
  `0/0` and caught only incidentally by the `isfinite` check. Guard
  `n == 0` (and `n < p + q + 1`) explicitly at the top with a clear
  error message.
- **No symmetrization needed.** Measured max relative `|P − Pᵀ|` over
  5,000 steps: 2.2e-16 for ARMA(2,2) and SARIMA(1,1)×(1,1)₁₂. The
  recursion is self-stabilizing here. Adding `P = (P + P')/2` is cheap
  insurance but is not fixing an observed problem — don't let it be
  presented as a bug fix.

### 3.5 One real gap in the verification corpus

All 20 seasonal cases have p, q, P, Q ≤ 1. The `(2,0,1,0)` structure in
`SEASONAL_STRUCTS` produced four cases (ids 355, 356, 367, 368) that
`statsmodels` rejected and `merge_and_emit.py` correctly dropped — but
the *reason* is a generator bug, not a real limitation:
`coefset["ar"][:p]` with `p = 2` truncates a one-element list to one
element, so a three-parameter model was handed two parameters.

Fix `SEASONAL_COEF_SETS` in `gen_cases.py` to carry two regular AR and
two regular MA coefficients, regenerate, and the corpus extends toward
airline-model territory — SARIMA(2,1,1)(0,1,1)₁₂ and friends, which is
what `SeasonalAdjustment.jl` will actually be fitting. **Do this before
Part B**, so Stage 6.5 is building against seasonal ground truth that
covers the orders it will meet.

Everything else in that pipeline is sound: zero R-vs-Python mismatches
across all 364 surviving cases, and the drop was logged rather than
silently swallowed.

---

# PART B — ARIMA/SARIMA FITTING (Stage 6.5–6.8)

## 4. Design decisions to make before writing code

### 4.1 Proposed sub-stage split

Reconcile the numbering with `development-sequence.md`, but the natural
break points are:

| # | Deliverable |
|---|---|
| 6.5 | Differencing, parameter packing/unpacking, the objective function |
| 6.6 | `fit_arima` / `ARIMAModel` + StatsAPI contract + `show` |
| 6.7 | Forecasting: `psi_weights`, h-step prediction, re-integration, intervals |
| 6.8 | Seasonal path end-to-end + full R/Python fitted comparison |

Build 6.5 and 6.6 for **non-seasonal, d=0** first and get a single
`fit_arima(y, (1,0,1))` matching R's `arima()` before adding
differencing, then seasonality. Each of the three adds an independent
class of bug and debugging them superimposed is miserable.

### 4.2 Parameter vector layout — already empirically pinned

Use `[phi; theta; Phi; Theta]`. This is not a guess: `gen_cases.py`
built `list(ar) + list(ma) + list(sar) + list(sma)` and `run_r.R` built
`fixed = c(ar, ma, sar, sma)`, and all 364 cases agreed to 1e-04. That
ordering is confirmed against both references simultaneously. Note it
differs from `combined_ar_ma`'s argument order (§3.1).

Mean, if included, appends at the end — matching R, where `xreg`
coefficients (of which the intercept is the first) follow the ARMA
block.

### 4.3 Mean and drift — match `stats::arima`, and say so

- `d == 0`: include an estimated mean by default. Feed `y .- μ` to the
  filter with `μ` as an extra optimizer parameter. Because the engine
  has no `xreg` support and none is needed for a single constant, this
  requires no state-space change at all.
- `d > 0`: **no mean.** `stats::arima` ignores `include.mean` entirely
  when `d > 0`. Match that, and document that `forecast::Arima`'s
  `include.drift` (a linear trend, equivalently a constant on the
  differenced scale) is deliberately deferred to Stage 7, where `xreg`
  arrives properly.

Do not quietly invent a third convention here. The whole credibility of
this package rests on being explainable against a named reference.

### 4.4 Optimizer wiring

- Transform with Stage 4.2's `partrans` so the search stays inside the
  stationary/invertible region. Re-read that handoff for **which blocks
  get transformed** — R applies it to the regular AR, seasonal AR,
  regular MA and seasonal MA blocks separately, each with its own
  Monahan recursion, not to the concatenated vector.
- Remember the Stage 4.2 finding that R uses `tanh` and Python uses
  `u/√(1+u²)`. Both are valid; they produce *different raw optimizer
  coordinates* for the same fitted model. Comparisons must be on fitted
  coefficients and log-likelihood, never on the untransformed vector.
- Replace the `-Inf` sentinel at the objective boundary. If
  `converged == false`, return a large finite penalty that grows with
  distance from the feasible region rather than `Inf`, so a line search
  gets a usable direction. With `partrans` active this path should be
  unreachable; treat it firing as a bug signal worth logging, not a
  normal branch.
- **Starting values:** R runs conditional-sum-of-squares first and uses
  its estimates to seed the ML optimization. Zeros work for well-behaved
  series and fail on near-unit-root ones. If CSS isn't built, seed AR
  terms from Stage 1's `pacf` and MA terms at zero, and record the
  choice — different starting values are the most common reason two
  implementations report different "converged" answers.

### 4.5 Standard errors — the subtle one

R computes the Hessian **in the natural parametrization, not the
transformed one**: after optimizing with `transform.pars=TRUE`, it
re-evaluates at the optimum with the transform switched off to get the
Hessian, then `vcov = inv(hessian * n_used)`. If you take the Hessian
of the transformed objective and invert it, the standard errors will be
wrong in a way that looks plausible — small, positive, and quietly
misscaled. Get this right, and test `stderror()` against R explicitly
rather than only testing `coef()`.

## 5. Differencing and forecasting — read this before designing 6.7

**This is the most important section in the document.** The scope
boundary inherited from the project brief (stationary Lyapunov init
only, no diffuse initialization) forces an architectural choice here
that is nowhere written down, and getting it wrong means either wrong
prediction intervals or an out-of-scope rewrite.

`stats::arima` handles `d > 0` by **augmenting the state vector** with
the differencing operator — state dimension becomes `r + d + D·s`, and
forecasts emerge on the original scale automatically with correct
variances. That approach requires **diffuse initialization** for the
differencing states, because a random walk has no stationary
distribution. Diffuse init is explicitly Stage 8. So:

**Stage 6 must difference first, filter the stationary part, and
re-integrate manually.**

Concretely:

1. Apply `(1−B)^d (1−B^s)^D` to `y`. Effective sample
   `n_eff = n − d − D·s`. The two operators commute, so order doesn't
   matter.
2. Fit the stationary ARMA to the differenced series using the existing
   engine, unchanged.
3. **Point forecasts:** forecast the differenced series with the Kalman
   prediction recursion (§5.1), then cumulate back. This is exact — no
   approximation is involved in the point forecasts.
4. **Prediction interval variance:** this is where the manual approach
   costs you something. Cumulating point forecasts is easy; cumulating
   *variances* correctly is not, because the differenced-scale forecast
   errors are correlated. Compute the ψ-weights of the **undifferenced**
   ARIMA — i.e. of the model whose AR polynomial is
   `φ(B)·Φ(Bˢ)·(1−B)^d·(1−Bˢ)^D` and whose MA polynomial is
   `θ(B)·Θ(Bˢ)` — and use

   `Var[ŷ_{n+h}] = σ̂² · Σ_{j=0}^{h−1} ψ_j²`

   This is what `forecast::Arima` reports and is the thing to verify
   against. A `psi_weights(ar, ma, h)` function is therefore a required
   deliverable of Stage 6.7, not an optional extra. It's a
   straightforward recursion from the two polynomials.

Write this reasoning into `development-sequence.md`'s Stage 6 rows, so
that when Stage 8 lands diffuse initialization somebody can consciously
decide whether to switch to state augmentation rather than rediscovering
why it wasn't done.

### 5.1 The engine has no forecasting function yet

`kalman_filter` and `kalman_smoother` exist; there is no `predict`. The
h-step recursion is short — run the filter to the end, then continue
with no observations:

```
a_{t+1} = T · a_t
P_{t+1} = T · P_t · Tᵀ + R · Rᵀ
ŷ_{t+h} = a_{t+h}[1]
Var     = σ̂² · P_{t+h}[1,1]
```

Add it to `gaussianssm.jl` alongside the filter — same reasoning the
brief used for building the smoother early. Note that for `d == 0` this
gives the intervals directly and the ψ-weight machinery of §5 is only
needed once differencing is in play; cross-checking the two against each
other at `d = 0` is a free consistency test.

## 6. Model object, StatsAPI, and display

Follow Stage 5.1's `ARXModel` precedent exactly — that stage
deliberately went first so this one would have a pattern to copy.

- `ARIMAModel <: TimeSeriesModel` holding: orders `(p,d,q)`,
  `(P,D,Q,s)`, the fitted parameter vector, `sigma2`, `loglik`,
  `vcov`, the `GaussianSSM` used, residuals, `n`, `n_eff`, and the
  optimizer's convergence flag.
- StatsAPI: `coef`, `coefnames`, `vcov`, `stderror`, `residuals`,
  `fitted`, `predict`, `loglikelihood`, `nobs`, `dof`, `aic`, `bic`.
- **`nobs` returns `n_eff`, not `n`** — R reports `n.used` after
  differencing, and every information criterion depends on it. This is a
  classic off-by-`d` that silently shifts every AIC comparison.
- **`dof` must count σ²**: `p + q + P + Q + (mean ? 1 : 0) + 1`. R's
  `aic = -2·loglik + 2·(npar + 1)`, where the `+1` is exactly σ².
  Getting this wrong makes every model-selection result subtly wrong
  while every individual fit looks correct.
- `show` via `StatsBase.CoefTable`, per the print-formatting handoff —
  full coefficient table on the default `show`, GLM.jl-style, not
  behind a separate `summary()` call. `ARIMAModel` is the second
  consumer of that pattern after `ARXModel`; if the two diverge, fix it
  now while there are only two.
- Attach the Stage 5.2 forecast object rather than inventing a parallel
  return type. That interface exists precisely so this stage reuses it.

## 7. Test plan

### 7.1 Reuse the corpus, extended for d > 0

The fixed-coefficient technique generalizes directly — R's
`arima(y, order=c(p,d,q), seasonal=..., fixed=..., transform.pars=FALSE, optim.control=list(maxit=0))`
works with `d > 0` unchanged, and `SARIMAX(..., order=(p,d,q), concentrate_scale=True).loglike(params)`
matches it. Extend `gen_cases.py` with `d ∈ {0,1}`, `D ∈ {0,1}` and
regenerate. This tests the *differencing plus likelihood* path without
involving the optimizer at all, which is exactly the isolation you want
before debugging fits.

Watch for one thing: R and `statsmodels` must be reporting the
likelihood on the same effective sample. If a case shows a constant
offset rather than agreement, an `n` vs `n_eff` discrepancy is the first
suspect — and `merge_and_emit.py` will correctly refuse to emit it, so
check the mismatch log rather than assuming the corpus regenerated
cleanly.

### 7.2 Testing *fitted* models — don't compare coefficients naively

Comparing fitted coefficients across implementations is fragile:
different optimizers, different starting values, different convergence
tolerances. Two defensible assertions instead:

1. **Likelihood at their optimum.** Evaluate our `loglik` at R's fitted
   coefficients and require agreement to 1e-06. This tests the
   likelihood exactly, with the optimizer removed from the picture.
2. **Our optimum is no worse.** Require
   `our_loglik ≥ R_loglik − 1e-04`. If we find a *higher* likelihood
   than R, that is a legitimate pass, not a failure — say so in the test
   comment so nobody "fixes" it later.

Then compare coefficients with a loose tolerance (1e-03) as a sanity
check only, and compare `stderror()` against R explicitly (§4.5).

### 7.3 Cases that must be in the suite

- `d=1` and `d=2` non-seasonal, to catch the `n_eff` bookkeeping.
- `D=1, s=12` on a genuinely seasonal series.
- The airline model, SARIMA(0,1,1)(0,1,1)₁₂ — the single most-used
  seasonal specification in existence and the workhorse of everything
  `SeasonalAdjustment.jl` will do.
- A pure MA fit, where the concentrated likelihood surface is flattest.
- A near-unit-root AR, where starting values matter most.
- `d=0` with and without an estimated mean, verifying the mean is
  actually being estimated and not silently dropped.
- A model where `p + q = 0` (pure differencing) — degenerate, and easy
  to crash on.

## 8. Traps checklist

- [ ] `nobs` is `n_eff = n − d − D·s`, not `n`.
- [ ] `dof` includes σ².
- [ ] Hessian taken in the **natural**, not transformed, parametrization.
- [ ] Parameter order `[phi; theta; Phi; Theta]`, but `combined_ar_ma`
      takes `(phi, Phi, theta, Theta, s)`.
- [ ] MA sign convention is `y_t = … + e_t + θ₁e_{t−1}` (positive),
      matching R and `statsmodels`. **X-13 and much of the seasonal
      adjustment literature use the negative convention** — this will
      bite at `SeasonalAdjustment.jl`, so write the convention into the
      docstring now.
- [ ] No mean when `d > 0`.
- [ ] `s = 0` vs `s = 1` for "non-seasonal" decided once, at the API
      boundary, and enforced (§2.2).
- [ ] Prediction intervals use ψ-weights of the *undifferenced* model.
- [ ] Non-invertible MA is accepted by the filter and returns a finite
      likelihood — this is correct behaviour (it's a likelihood-equivalent
      alias), but means the optimizer can wander there and report a
      non-invertible fit. `partrans` on the MA blocks is what prevents
      it; verify it's actually applied.

## 9. What to do with this

1. Complete Part A. Land §2.1 and §2.2 as code changes, §3.1–§3.2 if
   time allows, and get `Pkg.test()` green with all 368 assertions
   inside TSAnalytics before starting Part B.
2. Regenerate the seasonal corpus with §3.5's fix, so Part B builds
   against seasonal ground truth that reaches p, q ≥ 2.
3. Build 6.5–6.6 for non-seasonal `d=0` only. Get one `fit_arima`
   matching R end-to-end. **This is the Go/No-Go bar from the original
   2-week GaussianSSM trial** — the engine half has already cleared;
   this is the second half.
4. Add differencing, then seasonality, as separate commits with separate
   test additions.
5. Build 6.7 per §5, including `psi_weights`.
6. Update `development-sequence.md`: mark Stage 6.1–6.4 validated, and
   record §5's reasoning about differencing-vs-diffuse-initialization in
   the Stage 6 and Stage 8 rows.
7. Note in `CLAUDE.md` that GaussianSSM ran as a parallel workstream
   alongside Stage 5, so the relative timing is on record.

**Next in sequence:** Stage 7 (SARIMAX with `xreg`), which is where the
diffuse initialization deferred throughout this document finally has to
be built.
