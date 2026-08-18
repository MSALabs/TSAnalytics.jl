using LinearAlgebra: qr, dot, I, cholesky, Hermitian, PosDefException

export ADFTest, KPSSTest, PPTest, adf_test, kpss_test, pp_test

# ---------------------------------------------------------------------------
# Shared OLS helper (QR-based, for performance and numerical stability)
# ---------------------------------------------------------------------------

"""
    _ols(X, y; weights=nothing, method::Symbol=:qr)

Fit y = X*beta + e by (weighted) least squares. `method=:qr` (default)
factorizes X directly via QR -- the more numerically robust choice,
following the same design as GLM.jl's `DensePredQR` (its own docs note
the Cholesky path is faster but less accurate). `method=:cholesky`
instead factorizes the normal equations X'X -- faster in practice for
well-conditioned problems, but loses accuracy proportional to `cond(X)^2`
rather than `cond(X)`, since `cond(X'X) = cond(X)^2` -- per GLM.jl's own
documented tradeoff (verified from its source, not reimplemented from it
-- see `handoff/stage-1.4-ols-cholesky-handoff.md` for the exact quote
and reasoning).

GLS is implemented identically for both methods via the standard
weighted reduction: scale rows of `X` and `y` by `sqrt(weights)`, then
solve the resulting OLS problem with the chosen method -- this reproduces
`DensePredQR`/`DensePredChol`'s row-weights behaviour without a second
code path per method.

Returns `(beta, residuals, se)` -- `residuals` are on the original
(unweighted) scale; `se` uses the usual `sigma^2*(X'X)^-1` sandwich (or its
weighted analogue), not HAC. HAC standard errors are handled separately
where needed (KPSS's long-run variance).

`:cholesky` throws an `ArgumentError` (not a raw `PosDefException`) if
X'X isn't positive definite -- e.g. collinear regressors -- and points
the caller at `:qr`, which handles that case more gracefully (via a
least-squares solve, not a clean minimum-norm one -- see the pivoting
note below).

Not yet implemented: pivoted/rank-deficient handling for either method
(GLM.jl's `dropcollinear`, via `QRPivoted`/`CholeskyPivoted`). Worth
adding once a caller actually needs it rather than speculatively --
GLM.jl's own issue tracker documents real numerical subtlety in that path
(permutation bookkeeping interacting badly with `predict()` under
ill-conditioning), so it's worth doing carefully, from GLM.jl's
`linpred.jl` as the direct reference, when there's a concrete test case
to validate against.
"""
function _ols(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
              weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
              method::Symbol=:qr)
    method in (:qr, :cholesky) || throw(ArgumentError("method must be :qr or :cholesky"))
    n, k = size(X)
    if weights !== nothing
        length(weights) == n || throw(ArgumentError("weights must have length n"))
        all(>(0), weights) || throw(ArgumentError("weights must be positive"))
        sw = sqrt.(weights)
        Xw = X .* sw
        yw = y .* sw
    else
        Xw = X
        yw = y
    end

    dof = n - k
    dof > 0 || throw(ArgumentError("not enough observations for the requested regression"))

    local beta, XtX_inv
    if method === :qr
        F = qr(Xw)
        beta = F \ yw
        R = F.R
        Rinv = R \ Matrix{Float64}(I(k))
        XtX_inv = Rinv * Rinv'
    else # :cholesky
        XtX = Xw' * Xw
        Xty = Xw' * yw
        C = try
            cholesky(Hermitian(Matrix(XtX)))
        catch e
            e isa PosDefException &&
                throw(ArgumentError("_ols: method=:cholesky failed -- X'X is not positive definite " *
                                     "(likely collinear regressors); try method=:qr instead"))
            rethrow()
        end
        beta = C \ Xty
        XtX_inv = C \ Matrix{Float64}(I(k))
    end

    resid = y - X * beta                    # residuals on the ORIGINAL scale
    wresid = yw - Xw * beta                  # weighted residuals, for sigma2
    sigma2 = dot(wresid, wresid) / dof
    se = sqrt.(sigma2 .* _diagvec(XtX_inv))
    return beta, resid, se
end

_diagvec(A) = [A[i, i] for i in 1:size(A, 1)]

# ---------------------------------------------------------------------------
# Augmented Dickey-Fuller
# ---------------------------------------------------------------------------

"""
    ADFTest <: HypothesisTest

Result of an Augmented Dickey-Fuller test. `regression` is one of `:n`
(no constant, no trend), `:c` (constant only), `:ct` (constant + linear
trend), `:ctt` (constant + linear + quadratic trend), matching Python's
`statsmodels.tsa.stattools.adfuller`'s `regression` argument exactly --
see [`adf_test`](@ref) for the full R-vs-Python comparison this naming is
based on. `pvalue` is an approximate p-value interpolated among
MacKinnon's asymptotic critical values -- see [`adf_test`](@ref) for the
accuracy caveat.
"""
struct ADFTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    lags::Int
    regression::Symbol
    n::Int
end

function Base.show(io::IO, t::ADFTest)
    println(io, "Augmented Dickey-Fuller test")
    println(io, "  regression          : ", t.regression)
    println(io, "  lags                : ", t.lags)
    println(io, "  n                   : ", t.n)
    println(io, "  test statistic      : ", round(t.statistic, digits=4))
    print(io,   "  p-value (approx.)   : ", round(t.pvalue, digits=4))
end

# MacKinnon asymptotic (T -> infinity) critical values for the tau
# statistic, at 1/5/10 percent, for each regression case. Extracted
# directly from statsmodels' own `mackinnoncrit(N=1, regression=..., nobs=Inf)`
# (its authoritative source, itself citing MacKinnon 2010) rather than
# re-derived -- see handoff/stage-2.1-adf-handoff.md.
const _ADF_CRIT = Dict(
    :n   => ((0.01, -2.56574), (0.05, -1.941),   (0.10, -1.61682)),
    :c   => ((0.01, -3.43035), (0.05, -2.86154), (0.10, -2.56677)),
    :ct  => ((0.01, -3.95877), (0.05, -3.41049), (0.10, -3.12705)),
    :ctt => ((0.01, -4.37113), (0.05, -3.83239), (0.10, -3.55326)),
)

"""Linearly interpolate an approximate p-value from a small table of
(alpha, critical value) pairs, assuming the statistic is more negative for
smaller p (standard for a left-tailed unit-root test). This is a stopgap
for the full MacKinnon response-surface p-value and is documented as such;
it is adequate for deciding significance at conventional levels but should
not be quoted to more than ~1 significant figure."""
function _interp_pvalue_left(stat::Real, table)
    sorted = sort(collect(table); by = x -> x[2])  # most negative first
    if stat <= sorted[1][2]
        return sorted[1][1] / 2  # beyond the 1% point
    elseif stat >= sorted[end][2]
        return min(1.0, sorted[end][1] * 2)
    end
    for i in 1:length(sorted)-1
        a1, c1 = sorted[i]
        a2, c2 = sorted[i+1]
        if c1 <= stat <= c2
            w = (stat - c1) / (c2 - c1)
            return a1 + w * (a2 - a1)
        end
    end
    return NaN
end

"""
    adf_test(x; regression::Symbol=:c, maxlag::Union{Nothing,Integer}=nothing,
             autolag::Union{Nothing,Symbol}=:aic) -> ADFTest

Augmented Dickey-Fuller test of the null hypothesis that `x` has a unit
root, against the alternative of stationarity (around the chosen
deterministic term). Fits, by OLS,

    Δy_t = μ (+ β t) (+ β2 t^2) + γ y_{t-1} + Σ_{i=1}^{p} φ_i Δy_{t-i} + ε_t

and returns the t-statistic on γ.

Argument names and defaults follow Python's
`statsmodels.tsa.stattools.adfuller` -- verified from its actual source,
not just its docs -- because there is no single unambiguous "the R
behavior" to follow instead: `tseries::adf.test` always fits
constant+trend with no way to request anything else, while
`fUnitRoots::adfTest` offers 3 of Python's 4 `regression` options (no
`:ctt`) under the different name `type`. See
`handoff/stage-2.1-adf-handoff.md` for the full comparison.

- `regression`: `:n` (no constant, no trend) / `:c` (constant only,
  **default**, matches Python's default) / `:ct` (constant + trend) /
  `:ctt` (constant + linear + quadratic trend).
- `maxlag`: ceiling on the number of augmenting lags. `nothing` (default)
  computes `ceil(12*(n/100)^0.25)` (Schwert 1989, via Greene), capped at
  `n÷2 - ntrend - 1` where `ntrend` is the number of deterministic
  regressors -- both exactly matching `adfuller`'s own source. An
  explicit `maxlag` is validated against that same cap rather than
  silently clamped. R's `tseries::adf.test` uses a **different** default
  formula entirely (`trunc((n-1)^(1/3))`, Said-Dickey/Banerjee et al.
  1993) -- pass that explicitly for exact R replication (see below).
- `autolag`: `:aic` (default, matches Python's default) searches lags
  `0:maxlag` and picks the AIC-minimizing one; `:bic` likewise for BIC;
  `:tstat` starts at `maxlag` and steps down, stopping at the first lag
  whose own highest-order coefficient has `|t| >= 1.6449`
  (`quantile(Normal(), 0.95)`, a **one-sided** 5% threshold -- verified
  from `adfuller`'s `_autolag` source, not guessed) -- falling back to 0
  lags if none qualify; `nothing` uses `maxlag` lags directly with no
  search (this is what `tseries::adf.test` effectively always does).
  During an `:aic`/`:bic`/`:tstat` search, every candidate lag is fit
  using the *same* `maxlag`-trimmed sample size, matching `adfuller`'s
  own approach (needed for the criteria to be comparable across
  candidates at all) -- the final chosen lag is then refit at its own
  natural, untrimmed sample size, exactly as `adfuller` does.
- **No `alternative` argument.** In `tseries::adf.test`, `alternative`
  ("stationary"/"explosive") doesn't change the regression or statistic
  at all -- verified from source -- it only affects how the conclusion is
  *worded*. Not worth the surface area for a labeling-only option.

Replicate `tseries::adf.test(x)` exactly with:
```julia
n = length(tsvalues(x))
adf_test(x; regression=:ct, autolag=nothing, maxlag=trunc(Int, (n-1)^(1/3)))
```

!!! note "p-value accuracy"
    The reported p-value is linearly interpolated among MacKinnon's
    asymptotic critical values, not the finite-sample response-surface
    p-value used by `statsmodels`/R (nor `tseries::adf.test`'s own
    Banerjee et al. table, which only ever applies to its one fixed
    constant+trend case). Adequate for a significant/not-significant call
    at 1/5/10%; exact p-values are a planned improvement.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(300));

julia> adf_test(y).pvalue > 0.10   # random walk: fails to reject the unit-root null
true

julia> adf_test(y; regression=:n, autolag=:bic).regression
:n
```
"""
function adf_test(x; regression::Symbol=:c, maxlag::Union{Nothing,Integer}=nothing,
                   autolag::Union{Nothing,Symbol}=:aic)
    regression in (:n, :c, :ct, :ctt) ||
        throw(ArgumentError("regression must be :n, :c, :ct, or :ctt"))
    autolag === nothing || autolag in (:aic, :bic, :tstat) ||
        throw(ArgumentError("autolag must be :aic, :bic, :tstat, or nothing"))

    y = tsvalues(x)
    n0 = length(y)
    ntrend = regression == :n ? 0 : length(String(regression))  # :c->1, :ct->2, :ctt->3

    if maxlag === nothing
        maxp = ceil(Int, 12 * (n0/100)^0.25)
        maxp = min(n0 ÷ 2 - ntrend - 1, maxp)
        maxp >= 0 || throw(ArgumentError("sample size is too short to use the selected regression component"))
    else
        maxlag >= 0 || throw(ArgumentError("maxlag must be >= 0"))
        maxlag <= n0 ÷ 2 - ntrend - 1 ||
            throw(ArgumentError("maxlag must be less than n÷2 - 1 - ntrend, where ntrend is the number of deterministic regressors"))
        maxp = Int(maxlag)
    end

    dy = diff(y)

    # Fit the ADF regression using own-lag order `p`, with the response and
    # y_{t-1} trimmed to accommodate `window` lags (not necessarily `p`
    # itself). During an autolag search, `window` is fixed at `maxp` for
    # every candidate so their AIC/BIC/t-stats are computed on identical
    # sample sizes -- required for those criteria to be comparable at all
    # (verified from `adfuller`'s source). The final chosen lag is re-fit
    # with `window == p`, giving it its own natural (larger, if p < maxp)
    # sample -- matching `adfuller`'s explicit re-run after search.
    function fit_at(p::Integer, window::Integer)
        resp = dy[(window+1):end]
        nobs = length(resp)
        ylag = y[(window+1):(n0-1)]

        cols = Vector{Vector{Float64}}()
        push!(cols, ylag)
        for i in 1:p
            push!(cols, dy[(window+1-i):(end-i)])
        end
        gamma_idx = 1
        if regression != :n
            pushfirst!(cols, ones(nobs))
            gamma_idx += 1
        end
        last_lag_idx = p > 0 ? gamma_idx + p : gamma_idx
        if regression in (:ct, :ctt)
            push!(cols, collect(1.0:nobs))
        end
        if regression == :ctt
            push!(cols, collect(1.0:nobs).^2)
        end
        X = reduce(hcat, cols)

        beta, resid, se = _ols(X, resp)
        k = size(X, 2)
        rss = sum(abs2, resid)
        aic = nobs*log(rss/nobs) + 2*k
        bic = nobs*log(rss/nobs) + k*log(nobs)
        tstat = beta[gamma_idx] / se[gamma_idx]
        tstat_last = beta[last_lag_idx] / se[last_lag_idx]
        return (tstat=tstat, tstat_last=tstat_last, aic=aic, bic=bic, nobs=nobs, p=p)
    end

    chosen_p = if autolag === nothing
        maxp
    elseif autolag in (:aic, :bic)
        results = [fit_at(p, maxp) for p in 0:maxp]
        crit = autolag == :aic ? getfield.(results, :aic) : getfield.(results, :bic)
        results[argmin(crit)].p
    else # :tstat
        results = [fit_at(p, maxp) for p in 0:maxp]
        stop = 1.6448536269514722  # quantile(Normal(), 0.95) -- one-sided 5%, verified from adfuller's `_autolag` source
        best = maxp
        for p in maxp:-1:0
            best = p
            abs(results[p+1].tstat_last) >= stop && break
        end
        best
    end

    final = fit_at(chosen_p, chosen_p)
    pval = _interp_pvalue_left(final.tstat, _ADF_CRIT[regression])

    return ADFTest(final.tstat, pval, chosen_p, regression, final.nobs)
end

# ---------------------------------------------------------------------------
# KPSS
# ---------------------------------------------------------------------------

"""
    KPSSTest <: HypothesisTest

Result of a KPSS test. `regression` is `:c` (level-stationarity) or `:ct`
(trend-stationarity), matching Python's `statsmodels.tsa.stattools.kpss`
argument name and value convention (also used by [`adf_test`](@ref), for
consistency across both tests). The null hypothesis is
(trend-)stationarity, the opposite null to `ADFTest` -- using both
together is standard practice since they can disagree in ambiguous
cases.
"""
struct KPSSTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    regression::Symbol
    lags::Int
    n::Int
end

function Base.show(io::IO, t::KPSSTest)
    println(io, "KPSS test")
    println(io, "  null hypothesis     : ", t.regression == :ct ? "trend-stationary" : "level-stationary")
    println(io, "  bandwidth (l)       : ", t.lags)
    println(io, "  n                   : ", t.n)
    println(io, "  test statistic      : ", round(t.statistic, digits=4))
    print(io,   "  p-value (approx.)   : ", round(t.pvalue, digits=4))
end

# Kwiatkowski, Phillips, Schmidt & Shin (1992), Table 1
const _KPSS_CRIT = Dict(
    :level => ((0.01, 0.739), (0.05, 0.463), (0.10, 0.347)),
    :trend => ((0.01, 0.216), (0.05, 0.146), (0.10, 0.119)),
)

function _interp_pvalue_right(stat::Real, table)
    sorted = sort(collect(table); by = x -> x[2])  # increasing critical value
    if stat <= sorted[1][2]
        return min(1.0, sorted[1][1] * 2)
    elseif stat >= sorted[end][2]
        return sorted[end][1] / 2
    end
    for i in 1:length(sorted)-1
        a1, c1 = sorted[i]
        a2, c2 = sorted[i+1]
        if c1 <= stat <= c2
            w = (stat - c1) / (c2 - c1)
            return a1 + w * (a2 - a1)
        end
    end
    return NaN
end

"""
    kpss_test(x; regression::Symbol=:c, nlags::Union{Symbol,Integer}=:short) -> KPSSTest

KPSS test of the null hypothesis that `x` is (trend-)stationary, against
the alternative of a unit root. Argument names follow Python's
`statsmodels.tsa.stattools.kpss` (`regression`, matching [`adf_test`](@ref)
too) -- see `handoff/stage-2.2-kpss-handoff.md` for the full comparison
against R's `tseries::kpss.test`. Both references use the same
Kwiatkowski, Phillips, Schmidt & Shin (1992) Table 1 critical values,
already implemented here -- a confirmed non-discrepancy, unlike `adf_test`.

- `regression`: `:c` (constant only, **default** -- matches Python's
  default and R's `null="Level"`) or `:ct` (constant + trend -- matches
  R's `null="Trend"`).
- `nlags`: Newey-West/Bartlett kernel bandwidth for the long-run variance
  estimate.
  - `:short` (**default here**) -- matches R's actual default
    (`lshort=TRUE`): `trunc(4*(n/100)^0.25)`. R has no Python equivalent
    for this specific formula.
  - `:legacy` -- `ceil(12*(n/100)^0.25)`, capped at `n-1`, matching
    Python's `nlags="legacy"` **exactly, verified from `kpss`'s actual
    source** (not its docs, which read as truncation -- it uses `ceil`,
    the same `ceil`-not-`floor` discrepancy already found in `adf_test`'s
    `maxlag` formula). Whether this is bit-identical to R's
    `lshort=FALSE` could not be independently verified (no R available in
    this environment) -- treat "matches Python's `legacy` exactly" as the
    verified claim, "and R's `lshort=FALSE`" as unconfirmed.
  - an explicit `Integer` -- direct override, must be `< n`.
  - `:auto` (Python's actual default -- the Hobijn et al. (1998)
    data-dependent method) is **not yet implemented** and throws
    `ArgumentError` rather than silently substituting a different formula
    under that name.

!!! note "Default mismatch with Python, by necessity, not oversight"
    This function's default (`:short`) matches R's default, not Python's
    (`:auto`/Hobijn, unimplemented). Pass `nlags=:legacy` if you
    specifically want the bandwidth choice verified to match Python's
    `"legacy"` mode.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(300));

julia> kpss_test(y).pvalue <= 0.05   # random walk: rejects level-stationarity
true

julia> kpss_test(y; regression=:ct, nlags=:legacy).regression
:ct

julia> kpss_test(y; nlags=:auto)
ERROR: ArgumentError: nlags=:auto (Hobijn et al. 1998 data-dependent method) is not yet implemented -- use :short, :legacy, or an explicit Integer. See handoff/stage-2.2-kpss-handoff.md for the reference to implement against.
```
"""
function kpss_test(x; regression::Symbol=:c, nlags::Union{Symbol,Integer}=:short)
    regression in (:c, :ct) || throw(ArgumentError("regression must be :c or :ct"))
    y = tsvalues(x)
    n = length(y)

    l = if nlags isa Integer
        nlags >= 0 || throw(ArgumentError("nlags must be >= 0"))
        nlags < n || throw(ArgumentError("nlags must be < n (got nlags=$nlags, n=$n)"))
        Int(nlags)
    elseif nlags === :short
        min(trunc(Int, 4 * (n/100)^0.25), n-1)
    elseif nlags === :legacy
        min(ceil(Int, 12 * (n/100)^0.25), n-1)
    elseif nlags === :auto
        throw(ArgumentError("nlags=:auto (Hobijn et al. 1998 data-dependent method) is not yet " *
                             "implemented -- use :short, :legacy, or an explicit Integer. " *
                             "See handoff/stage-2.2-kpss-handoff.md for the reference to implement against."))
    else
        throw(ArgumentError("nlags must be :short, :legacy, :auto, or an Integer"))
    end

    X = regression == :ct ? hcat(ones(n), collect(1.0:n)) : reshape(ones(n), n, 1)
    _, resid, _ = _ols(X, y)

    S = cumsum(resid)
    numerator = sum(abs2, S) / n^2

    gamma0 = sum(abs2, resid) / n
    lrv = gamma0
    for k in 1:l
        w = 1 - k / (l + 1)            # Bartlett kernel
        gk = dot(view(resid, 1:n-k), view(resid, 1+k:n)) / n
        lrv += 2 * w * gk
    end

    stat = numerator / lrv
    pval = _interp_pvalue_right(stat, _KPSS_CRIT[regression == :ct ? :trend : :level])

    return KPSSTest(stat, pval, regression, l, n)
end

# ---------------------------------------------------------------------------
# Phillips-Perron
# ---------------------------------------------------------------------------

"""
    PPTest <: HypothesisTest

Result of a Phillips-Perron test. `trend` is one of `:n`/`:c`/`:ct`
(matching [`adf_test`](@ref)'s `regression` values). `test_type` is
`:tau` (t-stat based) or `:rho` (coefficient based) -- see
[`pp_test`](@ref); `pvalue` is `NaN` for `test_type=:rho`, since its
asymptotic null distribution isn't the one the interpolation table here
covers (a documented gap, not a silently wrong number).
"""
struct PPTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    lags::Int
    trend::Symbol
    test_type::Symbol
    n::Int
end

function Base.show(io::IO, t::PPTest)
    println(io, "Phillips-Perron test")
    println(io, "  trend               : ", t.trend)
    println(io, "  test type           : ", t.test_type)
    println(io, "  lags                : ", t.lags)
    println(io, "  n                   : ", t.n)
    println(io, "  test statistic      : ", round(t.statistic, digits=4))
    print(io,   "  p-value (approx.)   : ", isnan(t.pvalue) ? "NaN (see test_type=:rho note)" : string(round(t.pvalue, digits=4)))
end

"""
    pp_test(x; trend::Symbol=:c, test_type::Symbol=:tau, lags::Union{Nothing,Integer}=nothing) -> PPTest

Phillips-Perron test of the null hypothesis that `x` has a unit root.
Unlike [`adf_test`](@ref), the underlying regression has no augmenting
lagged-difference terms at all -- it fits a plain
`y_t = ρ·y_{t-1} + [μ] + [δ·t] + u_t` and instead corrects the resulting
statistic for serial correlation via a Newey-West long-run-variance
adjustment (the same Bartlett-kernel formula [`kpss_test`](@ref) already
uses for its own long-run variance). Confirmed directly from `arch`'s own
docstring: *"Unlike the ADF test, the regression estimated includes only
one lag of the dependent variable, in addition to trend terms."*

Argument names follow `arch.unitroot.PhillipsPerron` (Python) -- note the
name `trend`, not `regression` (`adfuller`'s/`kpss`'s name for the same
concept) -- `arch` itself calls it `trend`, and this function follows
that specific package's naming rather than forcing cross-function
consistency at the cost of matching its own reference. R's
`tseries::pp.test` offers no comparable flexibility (always fits
constant+trend, unlike Python); see `handoff/stage-2.3-pp-handoff.md` for
the full comparison, including three genuine R-vs-Python default
disagreements (lag formula, statistic type, critical-value table).

This implementation was verified before being written in Julia at all:
`arch`'s own source formula was transcribed to Python and confirmed to
reproduce `arch`'s actual numerical output to ~1e-13 (see the handoff
doc), then independently re-verified here across all six
`trend`×`test_type` combinations on `test/fixtures/ar1_ref_series.csv`.

- `trend`: `:n` (no constant, no trend) / `:c` (constant only,
  **default**) / `:ct` (constant + trend).
- `test_type`: `:tau` (t-stat based, **default**, matches `arch`'s
  default) / `:rho` (coefficient based, matches R's `tseries::pp.test`
  default -- R and Python default to *opposite* variants).
- `lags`: Newey-West truncation lag. `nothing` (default) computes
  `ceil(12*(n/100)^0.25)`, matching `arch`'s default -- itself R's
  *non-default* `lshort=FALSE` formula (R's actual default uses
  `trunc(4*(n/100)^0.25)`; pass that explicitly for R-default
  equivalence).

!!! note "`:rho`'s p-value is `NaN`, not an approximation"
    `:rho`'s asymptotic null distribution (the Dickey-Fuller "Z"/
    coefficient distribution) is genuinely different from `:tau`'s
    (t-distribution-like) -- reusing the `:tau` critical-value table for
    it would be a wrong number presented as a real one. A proper `:rho`
    table is a known follow-up, not implemented here.

!!! note "p-value accuracy (`:tau`)"
    Same caveat as `adf_test`: approximate, via linear interpolation
    among MacKinnon's asymptotic critical values -- not the finite-sample
    response-surface p-value, and not Banerjee et al.'s table (what
    `tseries::pp.test` specifically uses).

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(300));

julia> pp_test(y).pvalue > 0.10   # random walk: fails to reject the unit-root null
true

julia> isnan(pp_test(y; test_type=:rho).pvalue)
true
```
"""
function pp_test(x; trend::Symbol=:c, test_type::Symbol=:tau, lags::Union{Nothing,Integer}=nothing)
    trend in (:n, :c, :ct) || throw(ArgumentError("trend must be :n, :c, or :ct"))
    test_type in (:tau, :rho) || throw(ArgumentError("test_type must be :tau or :rho"))
    y = tsvalues(x)
    n_full = length(y)
    l = lags === nothing ? ceil(Int, 12 * (n_full/100)^0.25) : Int(lags)
    l >= 0 || throw(ArgumentError("lags must be >= 0"))

    lhs = collect(Float64, y[2:end])
    rhs_y = collect(Float64, y[1:end-1])
    nobs = length(lhs)

    cols = Vector{Vector{Float64}}()
    push!(cols, rhs_y)
    trend in (:c, :ct) && push!(cols, ones(nobs))
    trend == :ct && push!(cols, collect(1.0:nobs))
    X = reduce(hcat, cols)
    k = size(X, 2)

    beta, u, se = _ols(X, lhs)   # plain OLS (non-HAC) se -- required by the formula below
    rho = beta[1]
    sigma = se[1]
    sigma2 = sigma^2

    s2 = dot(u, u) / (nobs - k)
    s = sqrt(s2)
    gamma0 = dot(u, u) / nobs

    cov = sum(abs2, u)
    for j in 1:l
        w = 1 - j/(l+1)             # Bartlett kernel
        gamma = dot(view(u, j+1:nobs), view(u, 1:nobs-j))
        cov += w * 2 * gamma
    end
    lam2 = cov / nobs
    lam = sqrt(lam2)

    stat_tau = sqrt(gamma0/lam2)*((rho-1)/sigma) - 0.5*((lam2-gamma0)/lam)*(nobs*sigma/s)
    stat_rho = nobs*(rho-1) - 0.5*(nobs^2 * sigma2/s2)*(lam2-gamma0)
    stat = test_type == :tau ? stat_tau : stat_rho

    # :rho's asymptotic null distribution (Dickey-Fuller "Z") is NOT the
    # tau/t-distribution family _ADF_CRIT tabulates -- NaN, not a wrong number.
    pval = test_type == :tau ? _interp_pvalue_left(stat, _ADF_CRIT[trend]) : NaN

    return PPTest(stat, pval, l, trend, test_type, nobs)
end
