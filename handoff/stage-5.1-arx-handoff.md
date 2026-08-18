# Handoff: Stage 5.1 — AR-X (comprehensive, R+Python verified, GLM.jl-style show)

Status: **done.** `arx`/`ARXModel` (`src/arx.jl`, `test/test_arx.jl`) built
per sections 3-5, verified exactly against real `statsmodels.tsa.ar_model.AutoReg`
(re-derived independently, not just transcribed -- confirmed every one of
this handoff's own section 6 numbers by execution before trusting them).
Two real bugs in this handoff's own proposed code were found and fixed:
1. **Standard errors**: `_ols`'s own `se` uses the `n-k` (dof-adjusted)
   variance convention; `AutoReg`'s actual `bse` uses the `n`-denominator
   conditional-MLE convention -- confirmed by reproducing `AutoReg`'s
   exact standard errors from a from-scratch computation (~1% systematic
   difference, not negligible). `arx` computes `vcov`/`stderror` itself
   (`sigma2 * inv(X'X)`, `n`-denominator `sigma2`) rather than trusting
   `_ols`'s `se` field as section 4's original code did.
2. **Seasonal dummies**: section 4's proposed code built a dummy for
   every season `1:(period-1)` named `"seasonal.$s"`. Real `AutoReg`
   uses season `1` as the implicit baseline and builds dummies for
   seasons `2:period`, named `"s(2,period)"` etc. -- confirmed by
   execution to be an *algebraically different* parameterization
   (different reference category), not a cosmetic naming difference.
   `arx` now matches `AutoReg` exactly.

Also hardened a real crash beyond what section 6 anticipated: a singular/
collinear design (e.g. `trend=:c` plus two lags of a perfectly linear
series) lets `_ols`'s QR-based solve silently return *some* `beta` while
the separate `inv(X'X)` `arx` needs for `vcov` throws a raw
`SingularException` -- now caught and reported as a clear `ArgumentError`
instead. p-values reuse the already-validated `_chisq_ccdf` (`Z^2 ~
ChiSq(1)` gives the two-sided normal p-value directly, confirmed to
reproduce `AutoReg`'s exact `pvalues`) rather than adding a new
`_std_normal_cdf` primitive as section 5 suggested; 95% CI bounds use the
already-available precise `_confidence_z(0.05)` (`1.959964...`) instead
of the coarser `1.96` literal from section 5's draft, matching
`statsmodels`' `conf_int()` to 5-6 digits rather than ~4.

For a fresh Claude Code session picking this up with no prior context.
First of five Stage 5 handoffs, same one-by-one treatment as Stage 2.
**This is also the first model-fit type in the whole project** -- the
GLM.jl-style `CoefTable` printing design queued since the earlier
print-formatting discussion starts here, not at Stage 6 as originally
assumed (AR-X produces real coefficients; Stage 6's ARIMA isn't actually
the first model to need this).

## Where this fits

- **Depends on:** Stage 1.4 (`_ols`, both `:qr`/`:cholesky`), Stage 0.2
  (`tsvalues`/`tsindex` interface).
- **First consumer of:** the `StatsAPI` contract (`coef`, `vcov`,
  `residuals`, `predict`, `loglikelihood`, `aic`, `bic`, `nobs`) --
  establishing the pattern every later model (SARIMAX, ETS, GARCH, VAR)
  follows.

---

## 1. Verified reference: R -- two different functions, again

**`stats::ar(x, aic=TRUE, order.max=NULL, method=c("yule-walker","burg","ols","mle","yw"), ...)`**
-- confirmed via `args(stats::ar)`. **Defaults to `"yule-walker"`, not
OLS.** `method="ols"` routes internally to the function below.

**`stats::ar.ols(x, aic=TRUE, order.max=NULL, na.action=na.fail, demean=TRUE, intercept=demean, series=NULL, ...)`**
-- the actual conditional-least-squares reference this stage is scoped
to. Confirmed via `formals()` and a live fit. **No `exog` argument at
all** -- despite this stage being named "AR-X," R's OLS-based AR has no
native exogenous-regressor support. A user wanting that in R would
manually regress out the exogenous part first, then `ar.ols` the
residuals -- a real capability gap, not an oversight in this review.

Confirmed print format (`print(fit)` on a live `ar.ols` object):
```
Call:
ar.ols(x = y, aic = FALSE, order.max = 2, demean = TRUE, intercept = TRUE)

Coefficients:
      1        2
 0.1007  -0.0334

Intercept: -0.003896 (0.1083)

Order selected 2  sigma^2 estimated as  1.149
```
Terse -- coefficients, one intercept-with-se pair, no full table with
individual coefficient standard errors/p-values.

## 2. Verified reference: Python `statsmodels.tsa.ar_model.AutoReg`

```python
AutoReg(endog, lags, trend='c', seasonal=False, exog=None,
        hold_back=None, period=None, missing='none')
```

Confirmed via `inspect.signature`. Substantially richer than R's
`ar.ols`:
- `lags`: **int or a `Sequence[int]`** -- an arbitrary lag *subset*
  (e.g. `[1, 3]`, skipping lag 2 entirely), not just "up to order p."
  No R equivalent at all.
- `trend`: `'n'`/`'c'`/`'t'`/`'ct'` -- four options (no constant /
  constant only / linear trend only, no constant / constant+trend).
  R's `ar.ols` only has `demean`/`intercept` booleans, a strict subset.
- `seasonal`: adds deterministic seasonal dummy variables. No R
  equivalent in `ar.ols`.
- `exog`: the actual "X" in AR-X. **No R equivalent**, per section 1.
- `hold_back`: explicit control over how many initial observations to
  exclude from the fit (useful for comparing models across different lag
  orders on the same effective sample). No direct R equivalent.

Confirmed `summary()` output (real run, `lags=2, trend='c'`, n=100):
```
                            AutoReg Model Results
==============================================================================
Dep. Variable:                      y   No. Observations:                  100
Model:                     AutoReg(2)   Log Likelihood                -138.482
Method:               Conditional MLE   S.D. of innovations              0.994
...                                     AIC                            284.964
                                        BIC                            295.303
                                        HQIC                           289.146
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0346      0.100      0.344      0.731      -0.162       0.232
y.L1           0.5401      0.098      5.521      0.000       0.348       0.732
y.L2           0.2564      0.099      2.602      0.009       0.063       0.449
                                    Roots
=============================================================================
                  Real          Imaginary           Modulus         Frequency
AR.1            1.1850           +0.0000j            1.1850            0.0000
AR.2           -3.2918           +0.0000j            3.2918            0.5000
```
**This is exactly the `CoefTable` shape** already designed for this
package's model-fit `show` convention -- `coef`/`std err`/`z`/`P>|z|`/CI
bounds, z-statistics not t (matching the earlier GLM.jl-alignment
decision from the print-formatting handoff). The "Roots" section (AR
characteristic polynomial roots, for a stationarity sanity-check at a
glance) is a nice addition worth adopting too.

---

## 3. Proposed Julia API

```julia
arx(y, lags::Union{Integer,AbstractVector{<:Integer}};
    trend::Symbol=:c, seasonal::Bool=false, period::Union{Nothing,Integer}=nothing,
    exog=nothing, hold_back::Union{Nothing,Integer}=nothing,
    method::Symbol=:qr) -> ARXModel
```

Design notes:
- **Matches Python's `AutoReg` naming throughout** (`lags`, `trend`,
  `seasonal`, `exog`, `hold_back`, `period`) -- the closer, more
  feature-complete reference; R's `ar.ols` naming (`demean`/`intercept`)
  is noted in the docstring for cross-reference but not adopted, since
  `trend=:n/:c/:t/:ct` already subsumes both R booleans as special cases
  and matches this project's established `:n`/`:c`/`:ct`-family
  conventions from `adf_test`/`pp_test`.
- **`lags` accepts an `Integer` or a `Vector`**, matching Python's
  int-or-subset flexibility exactly -- a real capability gain over R.
- **`method::Symbol=:qr`**: passed straight through to `_ols` -- `:qr`
  default, `:cholesky` available. Not in either reference; a natural
  extension given `_ols` already supports it.
- **`y` accepts anything `tsvalues` does** (container-agnostic, per
  Stage 0.2) -- `exog` should too, for the same reason.

### `ARXModel` -- StatsAPI contract

```julia
struct ARXModel <: UnivariateModel
    coef::Vector{Float64}
    se::Vector{Float64}
    names::Vector{String}       # "const", "y.L1", "y.L2", "x1", ... -- matches Python's naming style
    resid::Vector{Float64}
    sigma2::Float64
    loglik::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    lags::Vector{Int}
    trend::Symbol
end

StatsAPI.coef(m::ARXModel) = m.coef
StatsAPI.stderror(m::ARXModel) = m.se
StatsAPI.residuals(m::ARXModel) = m.resid
StatsAPI.nobs(m::ARXModel) = m.nobs
StatsAPI.loglikelihood(m::ARXModel) = m.loglik
StatsAPI.aic(m::ARXModel) = m.aic
StatsAPI.bic(m::ARXModel) = m.bic
```

---

## 4. Implementation

```julia
using StatsAPI
using StatsBase: CoefTable   # display-only dependency, reintroduced per the print-formatting handoff

"""
    arx(y, lags; trend=:c, seasonal=false, period=nothing, exog=nothing,
        hold_back=nothing, method=:qr) -> ARXModel

Autoregressive model with optional exogenous regressors ("AR-X"), fit by
conditional least squares. Argument names follow Python's
`statsmodels.tsa.ar_model.AutoReg` (the fuller-featured reference -- R's
`stats::ar.ols` has no exogenous-regressor support at all; see the Stage
5.1 handoff doc for the full comparison). Note R's `stats::ar()` (a
different function) defaults to Yule-Walker estimation, not OLS -- this
function always uses conditional least squares, matching `ar.ols`'s and
`AutoReg`'s shared algorithm, not `ar()`'s default.

- `lags`: an `Integer` (fit lags `1:lags`) or an `AbstractVector`
  (an arbitrary lag subset, e.g. `[1,3]`) -- matches `AutoReg` exactly,
  a real capability R's `ar.ols` lacks.
- `trend`: `:n` (none) / `:c` (constant, default) / `:t` (linear trend,
  no constant) / `:ct` (constant + trend).
- `seasonal`: add deterministic seasonal dummies (requires `period`).
- `exog`: optional exogenous regressor matrix/vector -- the "X" in AR-X.
- `hold_back`: number of initial observations to exclude from the fit;
  defaults to `maximum(lags)` if not given.
- `method`: `:qr` (default) or `:cholesky`, passed to `_ols`.

`y` and `exog` both accept anything `tsvalues` does.
"""
function arx(y, lags::Union{Integer,AbstractVector{<:Integer}};
             trend::Symbol=:c, seasonal::Bool=false, period::Union{Nothing,Integer}=nothing,
             exog=nothing, hold_back::Union{Nothing,Integer}=nothing,
             method::Symbol=:qr)
    trend in (:n, :c, :t, :ct) || throw(ArgumentError("trend must be :n, :c, :t, or :ct"))
    seasonal && period === nothing && throw(ArgumentError("seasonal=true requires period"))

    yv = tsvalues(y)
    n = length(yv)
    lagset = lags isa Integer ? collect(1:lags) : collect(lags)
    maxlag = maximum(lagset)
    hb = hold_back === nothing ? maxlag : hold_back
    hb >= maxlag || throw(ArgumentError("hold_back must be >= maximum(lags)"))

    resp = yv[(hb+1):n]
    nobs = length(resp)

    cols = Vector{Vector{Float64}}()
    names = String[]

    if trend in (:c, :ct)
        push!(cols, ones(nobs)); push!(names, "const")
    end
    if trend in (:t, :ct)
        push!(cols, collect(Float64, (hb+1):n)); push!(names, "trend")
    end
    if seasonal
        for s in 1:(period-1)
            push!(cols, Float64[(mod1(t, period) == s) for t in (hb+1):n])
            push!(names, "seasonal.$s")
        end
    end
    for lag in lagset
        push!(cols, yv[(hb+1-lag):(n-lag)]); push!(names, "y.L$lag")
    end
    if exog !== nothing
        ex = tsvalues(exog)
        ex_mat = ex isa AbstractMatrix ? ex : reshape(ex, :, 1)
        for j in 1:size(ex_mat, 2)
            push!(cols, ex_mat[(hb+1):n, j]); push!(names, "x$j")
        end
    end

    X = reduce(hcat, cols)
    beta, resid, se = _ols(X, resp; method=method)

    k = length(beta)
    sigma2 = sum(abs2, resid) / nobs
    loglik = -0.5*nobs*(log(2π) + log(sigma2) + 1)
    aic_val = -2*loglik + 2*(k+1)
    bic_val = -2*loglik + (k+1)*log(nobs)

    return ARXModel(beta, se, names, resid, sigma2, loglik, aic_val, bic_val, nobs, lagset, trend)
end
```

---

## 5. `show` method — GLM.jl-style `CoefTable`, learning from Python's layout

```julia
function Base.show(io::IO, m::ARXModel)
    z = m.coef ./ m.se
    p = 2 .* (1 .- _std_normal_cdf.(abs.(z)))   # or reuse Distributions if/when adopted
    ci_lo = m.coef .- 1.96 .* m.se
    ci_hi = m.coef .+ 1.96 .* m.se
    ct = CoefTable(
        hcat(m.coef, m.se, z, p, ci_lo, ci_hi),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"],
        m.names
    )
    println(io, "AR(", maximum(m.lags), ")",
                 m.trend == :n ? "" : " with $(m.trend) trend", ", n=", m.nobs)
    println(io)
    println(io, ct)
    println(io)
    print(io, "Log-likelihood: ", round(m.loglik, digits=2),
          "   AIC: ", round(m.aic, digits=2),
          "   BIC: ", round(m.bic, digits=2))
end
```

`_std_normal_cdf` -- a small local helper (or `Distributions.Normal()`
`cdf` if that dependency is ever adopted elsewhere first; not worth
adding solely for this) -- can reuse the same rational approximation
style already used for `_confidence_z` in `stattools.jl` if a
dependency-free implementation is preferred.

---

## 6. Comprehensive test matrix — verified numerically against Python's `AutoReg`

Series: AR(2), phi=(0.5, 0.2), seed=0, n=100.

| Case | Call | Verified coefficients (order matches `names`) |
|---|---|---|
| A | `trend=:n` | `[0.54103038, 0.25630631]` |
| B | `trend=:ct` | `[0.07877254, -0.00085445, 0.53864826, 0.25365287]` |
| C | `exog` (single regressor, `y2 = y + 0.5*x1`) | `[0.00893923, 0.46309424, 0.32809481, 0.66284653]` |
| D | `lags=[1,3]` (subset, skipping lag 2) | `[0.02622097, 0.67724476, 0.07250313]` |
| E | `hold_back=10` | `nobs == 90` |

```julia
using Test, Random

@testset "arx comprehensive" begin
    Random.seed!(0)
    n = 100
    e = randn(n)
    y = zeros(n)
    for t in 3:n
        y[t] = 0.5*y[t-1] + 0.2*y[t-2] + e[t]
    end
    # NOTE: Julia's RNG differs from numpy's -- these exact coefficient
    # values were computed via numpy/statsmodels (see handoff transcript
    # above). Use structural tests here; regenerate y via a Python
    # bridge for exact-value matching if needed.

    # A: trend=:n
    mA = arx(y, 2; trend=:n)
    @test length(mA.coef) == 2
    @test mA.names == ["y.L1", "y.L2"]

    # B: trend=:ct
    mB = arx(y, 2; trend=:ct)
    @test length(mB.coef) == 4
    @test mB.names == ["const", "trend", "y.L1", "y.L2"]

    # D: lags=[1,3] -- subset, matches Python's AutoReg exactly in shape
    mD = arx(y, [1,3]; trend=:c)
    @test mD.names == ["const", "y.L1", "y.L3"]
    @test length(mD.coef) == 3

    # E: hold_back
    mE = arx(y, 2; trend=:c, hold_back=10)
    @test mE.nobs == n - 10   # VERIFY against Python's nobs=90 exactly before trusting

    # error paths
    @test_throws ArgumentError arx(y, 2; trend=:bogus)
    @test_throws ArgumentError arx(y, 2; seasonal=true)  # missing period
    @test_throws ArgumentError arx(y, 2; hold_back=1)     # hold_back < maxlag

    # show runs without erroring and contains the CoefTable header
    io = IOBuffer()
    show(io, mB)
    s = String(take!(io))
    @test occursin("Coef.", s)
    @test occursin("AIC", s)
end
```

**Flagged for verification**: test E's exact `nobs` arithmetic (this
handoff's `hb`/`nobs` indexing) needs to be checked carefully against
Python's `nobs=90` result once implemented -- included here as an
assertion with a note rather than a confidently-asserted derivation,
since the off-by-one risk here is real (same category as the `_ols`
lagged-difference and STL extrapolation indexing bugs caught earlier in
this project) and wasn't independently re-verified line-by-line the way
those were.

---

## 7. What to do with this

1. Implement `arx`/`ARXModel` per sections 3-4.
2. Implement the `show` method per section 5 -- this is the actual
   first real use of `StatsBase.CoefTable` in the project; re-add
   `StatsBase` to `Project.toml` for this reason specifically (per the
   print-formatting handoff's stated rationale: display convention, not
   a statistical algorithm).
3. Run the tests in section 6; **specifically re-derive and verify the
   `hold_back`/`nobs` indexing** flagged as uncertain there before
   trusting it.
4. For exact cross-language numeric matching (not just structural),
   pipe identical data through Julia and Python via a shared CSV, same
   technique as the Stage 3.1/3.2 handoffs.
5. Update `development-sequence.md`'s Stage 5.1 row: mark implemented,
   note that R's `ar()` default (Yule-Walker) was deliberately not
   matched (this function is CLS-only, matching `ar.ols`/`AutoReg`), and
   that this is the project's first `CoefTable`-based model display.

**Next in sequence:** Stage 5.2 (generic forecast object + prediction
intervals) -- the first consumer of `ARXModel`, and the interface every
later model's forecasting will reuse.
