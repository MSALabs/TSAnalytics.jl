# Handoff: Stage 1.3 — Full-Fledged ACF / PACF (override StatsBase.jl)

Status: implemented ✅ (`src/stattools.jl`, `src/diagnostics.jl` refactored
per section 6, `test/test_stattools.jl`, `StatsBase` dropped from
`Project.toml`). Independently verified before implementing: this session
had actual `numpy`/`statsmodels` available (`python` on PATH), so the
AR(1) reference series and every number in section 7's transcript were
regenerated from scratch rather than trusted from this doc's own
transcription -- all matched exactly, to the printed digit. The exact
series is saved at `test/fixtures/ar1_ref_series.csv` so `test_stattools.jl`
validates against real numbers, not just the structural properties this
doc had to fall back on without a runtime. The `_chisq_ccdf`
include-order concern in section 4 turned out to be a non-issue in
practice: Julia resolves a function body's calls at call-time, not at
`include`-time, so `stattools.jl` (before `diagnostics.jl`) calling
`_chisq_ccdf` (defined in `diagnostics.jl`) works without reordering or
relocating anything -- confirmed by the `qstat`-vs-`ljungbox_test` exact
match test. No other deltas from the proposed implementation.

For a fresh Claude Code session picking this up with no prior context.
Covers Stage 1.3: replacing the current thin `StatsBase.jl`-wrapping
`acf`/`pacf` with native implementations matching R and Python's fuller
feature sets. **Two real, verified discrepancies were found between R and
Python's defaults during this review** — see section 3 — worth reading
even if you don't implement everything else here.

## Where this fits

- **Depends on:** Stage 0.2 (`tsvalues` interface), Stage 1.4 (`_ols`, for
  the OLS-based PACF method).
- **Replaces:** the current `acf`/`pacf` in `stattools.jl`, which wrap
  `StatsBase.autocor`/`StatsBase.pacf` and add only a simple constant-width
  confidence band. `StatsBase.pacf` supports two methods (`:regression`,
  `:yulewalker`) and has no `alpha`/Bartlett/qstat/adjusted-denominator
  support — the gap this handoff closes.
- **Also affects:** `diagnostics.jl`'s `ljungbox_test`/`qs_test` currently
  call `StatsBase.autocor` directly. Once native `acf` exists, refactor
  those to call it instead (see section 6) — at that point `StatsBase.jl`
  may become removable from `Project.toml` entirely, worth checking once
  this is wired in.

---

## 1. Verified reference: R `stats::acf()` / `stats::pacf()`

```r
acf(x, lag.max = NULL, type = c("correlation","covariance","partial"),
    plot = TRUE, na.action = na.fail, demean = TRUE, ...)
pacf(x, lag.max = NULL, plot = TRUE, na.action = na.fail, ...)
```

- `lag.max` default: `floor(10 * (log10(n) - log10(m)))` where `m` = number
  of series (1 for univariate) → `floor(10*log10(n))` for a single series.
- `type`: `"correlation"` (default) / `"covariance"` / `"partial"` — `acf`
  only; `pacf` is always "partial", **fixed algorithm, no `method=`
  argument at all**. R's own docs describe it as *"fitting autoregressive
  models of successively higher orders"* — this is the Durbin-Levinson
  recursion applied to the sample autocovariances.
- `na.action`: `na.fail` by default (errors on any `NA`); `na.pass`
  computes from complete cases.
- `demean`: `TRUE` default — subtract the sample mean before computing
  covariances.
- **No `adjusted`-style denominator toggle** — R always divides
  autocovariances by `n` (not `n-k`), with no option to change this.
- **No `alpha`/confidence-interval computation inside `acf()`/`pacf()`
  themselves** — bands are a `plot.acf()`-time concern
  (`ci.type = c("white","ma")`), not something the computing functions
  return.
- **No Ljung-Box integration** — `Box.test()` is a separate function call.

## 2. Verified reference: Python `statsmodels.tsa.stattools`

```python
acf(x, adjusted=False, nlags=None, qstat=False, fft=True, alpha=None,
    bartlett_confint=True, missing='none')
pacf(x, nlags=None, method='ywadjusted', alpha=None)
```

**`acf` arguments:**

| Arg | Meaning |
|---|---|
| `adjusted` | `True` → autocovariance denominator `n-k`; `False` (default) → `n`. R has no equivalent, always behaves like `adjusted=False`. |
| `nlags` | default `min(10*log10(nobs), nobs-1)` — matches R's `lag.max` default exactly |
| `qstat` | if `True`, also returns the Ljung-Box Q-statistic and p-value **per lag** |
| `fft` | if `True` (default), compute via FFT for speed on long series |
| `alpha` | if given, returns a confidence interval per lag |
| `bartlett_confint` | **default `True`** — the CI std uses Bartlett's formula (widening with lag): `var[k] = (1/n)*(1 + 2*sum(acf[1:k-1]**2))` for `k>=2`, `var[1]=1/n`. If `False`, uses the simple constant `1/n` for every lag instead. |
| `missing` | `'none'`/`'raise'`/`'conservative'`/`'drop'` |

**`pacf` arguments:** `method` has 8 named options, verified from source:

| Method string | Meaning |
|---|---|
| `"yw"` / `"ywadjusted"` | Yule-Walker, `n-k` denominator. **Default.** |
| `"ywm"` / `"ywmle"` | Yule-Walker, `n` denominator (unadjusted). |
| `"ols"` | regression of the series on its lags + a constant. |
| `"ols-inefficient"` | same, but using one common sample size for every lag's coefficient. |
| `"ols-adjusted"` | OLS with a bias adjustment. |
| `"ld"` / `"ldadjusted"` | Levinson-Durbin recursion, bias-corrected. |
| `"ldb"` / `"ldbiased"` | Levinson-Durbin recursion, not bias-corrected. |
| `"burg"` | Burg's method. |

`pacf`'s `alpha` CI uses a **simple constant** `1/sqrt(n)` band for every
lag — no Bartlett-style widening (theoretically justified differently:
under a true-AR(p) null, PACF beyond order `p` has asymptotic variance
`1/n` regardless of lag).

## 3. Two real discrepancies found — worth knowing before picking a default

Verified empirically (not just from docs) on a simulated AR(1) series,
n=200, φ=0.6:

**(a) Bartlett bands genuinely widen; simple bands don't.** ACF band
half-widths at lags 1-5 with `bartlett_confint=True`: `[0.139, 0.188,
0.208, 0.217, 0.220]`. With `bartlett_confint=False`: constant `0.139` at
every lag. **The package's current band formula is the simple,
non-default one** — this is a real gap, not a cosmetic one.

**(b) R's `pacf()` matches statsmodels' non-default `"ywm"`, not its
default `"yw"`.** Verified numerically — lag-1 PACF: `yw`=`0.654596`,
`ywm`=`0.651323`, `ols`=`0.667193`. Small but genuinely different numbers.
R's fixed algorithm (always `n`-denominator, never `n-k`) is mathematically
`ywm`, **not** statsmodels' actual default. Anyone comparing "R vs. Python
PACF output" naively would see a discrepancy that isn't a bug in either —
it's a default-choice difference between the two ecosystems. Worth
documenting this explicitly in the Julia docstring so it isn't mistaken
for an implementation bug later.

**Design decision for Julia, given this:** default to statsmodels'
`:yw` (adjusted) as the more modern/standard choice, but support `:ywm`
explicitly and document that it's the one that matches R exactly.

---

## 4. Proposed Julia API

```julia
acf(x, lags=nothing; alpha::Real=0.05, demean::Bool=true, adjusted::Bool=false,
    bartlett::Bool=true, qstat::Bool=false) -> ACFResult

pacf(x, lags=nothing; alpha::Real=0.05, method::Symbol=:yw) -> ACFResult
```

Design notes:
- `bartlett::Bool=true` as the **default** — matches statsmodels' actual
  default behavior (widening bands), a genuine upgrade from the package's
  current constant-width bands. This is a **breaking change to prior
  default output** if the old `acf` is already in use anywhere — flag it
  clearly in a changelog, don't let it slide in silently.
- `adjusted::Bool=false` — matches statsmodels' parameter name and default
  exactly. R has no equivalent (always behaves like `false`).
- `method::Symbol` for `pacf`: `:yw` (default), `:ywm`, `:ols` implemented
  now; `:burg` explicitly **not implemented** — flagged as a documented
  gap rather than silently omitted, since it's a genuinely different
  algorithm (not just a different denominator/sample-size choice like the
  other three) and lower priority than getting the common cases right.
- `qstat::Bool=false` — when `true`, populates `ACFResult`'s `qstat`/
  `pvalues` fields using the **same** chi-squared tail helper
  (`_chisq_ccdf`) already built for `ljungbox_test`/`qs_test` in
  `diagnostics.jl`. **Dependency note:** either include `diagnostics.jl`
  before this file so `_chisq_ccdf` is in scope, or (cleaner) move
  `_chisq_ccdf` and its Lanczos/continued-fraction helpers into a small
  shared file (e.g. `special_functions.jl`) included early, since
  "diagnostics depends on stats tools" is the more natural dependency
  direction than the reverse. Do the refactor if starting fresh; if
  `diagnostics.jl` already exists in your local repo, just reorder the
  `include`s.
- **No `fft` option.** Deliberately not implemented — the project's
  established policy is no unused/heavy dependencies in core
  `TSAnalytics`, and `FFTW.jl` is scoped to `TSFeatures.jl` only (see
  `development-sequence.md`, Stage F.3). Direct O(n·maxlag) computation is
  used instead. Fine for typical series lengths; if a very-long-series use
  case genuinely needs FFT-accelerated ACF later, that's an argument for a
  package extension, not a new hard dependency — revisit if it actually
  becomes a bottleneck, don't pre-optimize for it now.
- **`missing`/NA handling**: not fully built out to statsmodels' 4-way
  policy here — that's a package-wide concern (how does `tsvalues`
  generally handle `missing`/`NaN`?), not specific to ACF/PACF, and out of
  scope for this handoff. Minimum viable behavior: throw an
  `ArgumentError` if `NaN` is present, mirroring R's `na.fail` default,
  rather than silently producing wrong numbers. Revisit properly when/if a
  dedicated missing-data stage gets scheduled.

### `ACFResult` — extended, not replaced

```julia
struct ACFResult
    lags::Vector{Int}
    values::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    n::Int
    qstat::Union{Nothing,Vector{Float64}}
    pvalues::Union{Nothing,Vector{Float64}}
end

# keep the old 5-positional-arg call sites working
ACFResult(lags, values, lower, upper, n) = ACFResult(lags, values, lower, upper, n, nothing, nothing)
```

---

## 5. Implementation

```julia
using LinearAlgebra: dot

"""
    _acovf(y, maxlag; demean=true, adjusted=false)

Sample autocovariance at lags 0:maxlag. `adjusted=false` (default)
divides by `n` at every lag, matching R's fixed behavior and
statsmodels' `adjusted=False` default. `adjusted=true` divides by `n-k`
at lag `k`, matching statsmodels' `adjusted=True`.
"""
function _acovf(y::AbstractVector{<:Real}, maxlag::Int; demean::Bool=true, adjusted::Bool=false)
    n = length(y)
    yc = demean ? y .- (sum(y)/n) : collect(Float64, y)
    out = Vector{Float64}(undef, maxlag + 1)
    for k in 0:maxlag
        s = 0.0
        for t in 1:(n-k)
            s += yc[t] * yc[t+k]
        end
        out[k+1] = s / (adjusted ? (n - k) : n)
    end
    return out
end

"""
    acf(x, lags=nothing; alpha=0.05, demean=true, adjusted=false, bartlett=true, qstat=false)

Full-fledged sample autocorrelation function, natively implemented
(no StatsBase dependency). Matches R's `stats::acf()` and Python's
`statsmodels.tsa.stattools.acf()` -- see the module handoff doc for the
verified argument-by-argument comparison and two real default-behavior
discrepancies found between them.

`bartlett=true` (the default, matching statsmodels' default, NOT R's
`plot.acf` default) uses Bartlett's formula for the confidence band,
which widens with lag: `var[k] = (1+2*sum(acf[1:k-1].^2))/n`. Set
`bartlett=false` for the simple constant-width `1/n` band instead (what
this function returned in an earlier version, unconditionally).

`adjusted=false` (default) divides autocovariances by `n`; `true` uses
`n-k`, matching statsmodels' `adjusted` argument exactly. R has no
equivalent option.

`qstat=true` additionally populates `ACFResult.qstat`/`.pvalues` with the
per-lag Ljung-Box statistic, reusing the same chi-squared tail helper as
`ljungbox_test`.

`x` accepts anything `tsvalues` does.
"""
function acf(x, lags::Union{Nothing,AbstractVector{<:Integer}}=nothing;
             alpha::Real=0.05, demean::Bool=true, adjusted::Bool=false,
             bartlett::Bool=true, qstat::Bool=false)
    y = tsvalues(x)
    any(isnan, y) && throw(ArgumentError("acf: NaN present; R's na.fail-equivalent default -- see handoff doc for missing-data policy notes"))
    n = length(y)
    ls = lags === nothing ? _default_lags(n, 0) : lags
    maxlag = maximum(ls)

    acov = _acovf(y, maxlag; demean=demean, adjusted=adjusted)
    full_acf = acov ./ acov[1]
    vals = full_acf[ls .+ 1]

    z = _confidence_z(alpha)
    if bartlett
        varacf = zeros(maxlag + 1)
        varacf[1] = 0.0
        maxlag >= 1 && (varacf[2] = 1.0 / n)
        for k in 2:maxlag
            varacf[k+1] = (1.0 + 2*sum(full_acf[2:k].^2)) / n
        end
        bound = z .* sqrt.(varacf[ls .+ 1])
        lower, upper = -bound, bound
    else
        bound = z / sqrt(n)
        lower = fill(-bound, length(ls))
        upper = fill(bound, length(ls))
    end

    qs, pv = nothing, nothing
    if qstat
        posl = filter(>(0), ls)  # lag 0 excluded, matches statsmodels convention
        qs = Float64[]; pv = Float64[]
        for k in posl
            rho_upto_k = full_acf[2:k+1]
            Q = n * (n + 2) * sum(rho_upto_k[j]^2 / (n - j) for j in 1:k)
            push!(qs, Q)
            push!(pv, _chisq_ccdf(Q, k))
        end
    end

    return ACFResult(collect(ls), vals, lower, upper, n, qs, pv)
end

"""_durbin_levinson(acov, maxlag) -- PACF via the Durbin-Levinson
recursion applied to autocovariances `acov[1:maxlag+1]` (index 1 =
lag 0). Verified against `statsmodels.tsa.stattools.pacf` output to 6
decimal places for both adjusted and unadjusted inputs -- see handoff
doc section for the verification script."""
function _durbin_levinson(acov::AbstractVector{<:Real}, maxlag::Int)
    phi = zeros(maxlag, maxlag)
    pacf_vals = zeros(maxlag)
    v = acov[1]
    phi[1,1] = acov[2] / acov[1]
    pacf_vals[1] = phi[1,1]
    v *= (1 - phi[1,1]^2)
    for k in 2:maxlag
        s = acov[k+1]
        for j in 1:k-1
            s -= phi[k-1,j] * acov[k-j+1]
        end
        phi[k,k] = s / v
        for j in 1:k-1
            phi[k,j] = phi[k-1,j] - phi[k,k]*phi[k-1,k-j]
        end
        pacf_vals[k] = phi[k,k]
        v *= (1 - phi[k,k]^2)
    end
    return pacf_vals
end

"""_pacf_ols(y, maxlag) -- PACF via successive OLS regressions on a
common sample (matches statsmodels' "ols"/"ols-inefficient" family),
reusing the package's own `_ols` helper rather than a separate
implementation."""
function _pacf_ols(y::AbstractVector{<:Real}, maxlag::Int)
    n = length(y)
    out = Vector{Float64}(undef, maxlag)
    for k in 1:maxlag
        resp = y[(k+1):n]
        nobs = length(resp)
        cols = [ones(nobs)]
        for lag in 1:k
            push!(cols, y[(k+1-lag):(n-lag)])
        end
        X = reduce(hcat, cols)
        beta, = _ols(X, resp)
        out[k] = beta[end]  # coefficient on y[t-k]
    end
    return out
end

"""
    pacf(x, lags=nothing; alpha=0.05, method=:yw)

Full-fledged sample partial autocorrelation function, natively
implemented. `method`:
- `:yw` (default): Yule-Walker / Durbin-Levinson, `n-k` denominator --
  matches statsmodels' default (`"yw"`/`"ywadjusted"`).
- `:ywm`: Yule-Walker / Durbin-Levinson, `n` denominator -- matches R's
  `pacf()` exactly (R has no adjusted-denominator option; this is what it
  computes).
- `:ols`: successive regression on a common sample -- matches
  statsmodels' `"ols"`/`"ols-inefficient"`.
- `:burg` is **not implemented** -- a genuinely different algorithm
  (Burg's method), documented as a known gap rather than silently
  omitted.

`x` accepts anything `tsvalues` does.
"""
function pacf(x, lags::Union{Nothing,AbstractVector{<:Integer}}=nothing;
              alpha::Real=0.05, method::Symbol=:yw)
    method in (:yw, :ywm, :ols) ||
        throw(ArgumentError("method must be :yw, :ywm, or :ols (:burg not yet implemented)"))
    y = tsvalues(x)
    n = length(y)
    ls = lags === nothing ? _default_lags(n, 1) : lags
    maxlag = maximum(ls)

    vals_full = if method == :ols
        _pacf_ols(y, maxlag)
    else
        acov = _acovf(y, maxlag; demean=true, adjusted=(method == :yw))
        _durbin_levinson(acov, maxlag)
    end
    vals = vals_full[ls]

    z = _confidence_z(alpha)
    bound = z / sqrt(n)   # statsmodels uses the simple constant band for PACF too
    lower = fill(-bound, length(ls))
    upper = fill(bound, length(ls))

    return ACFResult(collect(ls), vals, lower, upper, n)
end
```

(`_default_lags` and `_confidence_z` already exist in the current
`stattools.jl` -- reuse them as-is.)

---

## 6. Integration notes

- **`diagnostics.jl` refactor**: `ljungbox_test` and `qs_test` currently
  call `StatsBase.autocor` directly. Change them to call the new native
  `acf(x, lags; bartlett=false).values` internally (bands aren't needed
  there, just the raw values) instead. Once done, grep the repo for any
  remaining `using StatsBase` — if nothing else needs it, remove it from
  `Project.toml` entirely.
- **`_chisq_ccdf` location**: as noted above, consider relocating it out
  of `diagnostics.jl` if `acf`'s `qstat` feature needs it and you want the
  cleaner dependency direction.
- This is a **breaking change** to `acf`'s default output (Bartlett bands
  instead of constant bands). If any code already depends on the old
  constant-band behavior, that's a real behavior change to flag, not a
  bug to silently work around.

---

## 7. Hand-verified test values

Computed against `statsmodels` directly (see verification transcript);
treat as trustworthy. Series: AR(1), φ=0.6, seed=0, n=200 (regenerate with
the exact loop below to match).

```julia
using Test, Random

@testset "acf/pacf full-fledged" begin
    Random.seed!(0)
    n = 200
    e = randn(n)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.6*y[t-1] + e[t]
    end
    # NOTE: Julia's RNG stream differs from numpy's even with the "same"
    # seed -- these exact values were computed via numpy/statsmodels, so
    # regenerate y using a Julia<->Python bridge or accept structural
    # tests (band widening, method differences) over exact-value tests
    # unless you specifically pipe the same y through both languages.

    # Structural test (RNG-independent): Bartlett bands must widen
    r_bartlett = acf(y, 0:5; bartlett=true)
    r_simple = acf(y, 0:5; bartlett=false)
    @test r_bartlett.upper[end] > r_bartlett.upper[2]      # widens with lag
    @test all(==(r_simple.upper[2]), r_simple.upper[2:end]) # simple stays constant

    # Structural test: yw, ywm, ols should differ (not be bugs if they do)
    p_yw = pacf(y, 1:5; method=:yw)
    p_ywm = pacf(y, 1:5; method=:ywm)
    p_ols = pacf(y, 1:5; method=:ols)
    @test p_yw.values != p_ywm.values   # genuinely different, per section 3

    # adjusted=true/false must differ
    a_adj = acf(y, 0:5; adjusted=true)
    a_unadj = acf(y, 0:5; adjusted=false)
    @test a_adj.values != a_unadj.values

    # qstat sanity: matches ljungbox_test on the same lags/data
    r_q = acf(y, 1:5; qstat=true, bartlett=false)
    lb = ljungbox_test(y, collect(1:5))
    @test isapprox(r_q.qstat[end], lb.statistic; atol=1e-6)  # Q at max lag matches Ljung-Box Q

    # invalid method
    @test_throws ArgumentError pacf(y, 1:5; method=:burg)
end
```

If you want **exact** cross-language numeric matches (not just
structural ones), pipe the identical data through both: generate `y` in
Python, save to CSV, load in Julia via `DelimitedFiles`, then compare
directly against the exact values already verified above:

```
acf (unadjusted, no fft): [1.0, 0.65132332, 0.45384408, 0.30065664, 0.18339111, 0.21470952]
bartlett var (k=0..5):    [0.0, 0.005, 0.00924222, 0.01130197, 0.01220591, 0.01254223]
pacf yw  (lags 1-5): [0.654596, 0.052374, -0.023542, -0.031385, 0.18513]
pacf ywm (lags 1-5): [0.651323, 0.051447, -0.023062, -0.030508, 0.179158]
pacf ols (lags 1-5): [0.667193, 0.062016, -0.031762, -0.037513, 0.179504]
```

---

## 8. What to do with this

1. Replace `acf`/`pacf` in `stattools.jl` with the implementation in
   section 5.
2. Refactor `diagnostics.jl` per section 6.
3. Check whether `StatsBase` can be dropped from `Project.toml` entirely.
4. Run the structural tests in section 7; if you want exact cross-language
   validation, generate matching data in Python per the note there.
5. Update `development-sequence.md`'s Stage 1.3 row: mark implemented,
   note the `:burg` gap and the missing-data-policy deferral explicitly
   so they don't get silently forgotten.
