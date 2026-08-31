using LinearAlgebra: qr, dot, I, cholesky, Hermitian, PosDefException

export ADFTest, KPSSTest, PPTest, adf_test, kpss_test, pp_test,
       adf_pvalue_response_surface, adf_critical_values_response_surface,
       pp_rho_pvalue, kpss_hobijn_autolag

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
# ADF / PP response-surface p-values (MacKinnon 1994/2010)
# ---------------------------------------------------------------------------
#
# Transcribed directly from two real, installed, executable reference
# implementations -- not reconstructed from the paper:
#   - statsmodels.tsa.adfvalues (`mackinnonp`/`mackinnoncrit`, the ADF
#     "tau"/ADF-t tables) -- handoff/verification/adfvalues_source.py.
#   - arch.unitroot.critical_values.dickey_fuller (`adf_z_*` tables, the
#     "ADF-z" family `arch.unitroot.PhillipsPerron` uses for the `:rho`
#     statistic). These are genuinely DIFFERENT numbers from the
#     similarly-named z_star_c/z_c_smallp/z_c_largep tables in
#     adfvalues_source.py (confirmed by direct execution -- those are for
#     a different Z-test); `arch`'s own `PhillipsPerron` computes
#     `stat_rho` via the *exact same formula* this file's `pp_test`
#     already does, and its p-value comes from these `adf_z_*` tables --
#     that end-to-end match is what confirms they're the right ones.
# Both cross-checked end-to-end in test/verification/unitroot/ against
# real `adfuller()`/`PhillipsPerron()` output on the shared AR(1)
# fixture, plus direct `mackinnonp()` calls across many (stat,
# regression) combinations -- not just "coefficients copied correctly".
#
# Only N=1 (a single time series -- the only case `adf_test`/`pp_test`
# ever need) is transcribed; statsmodels'/arch's own N=2:12
# (multi-series cointegration) tables are not, so this project doesn't
# carry untested numbers nobody calls.

"Horner evaluation of `d0 + d1*x + d2*x^2 + ...` for ascending coefficients."
_horner(coefs, x::Real) = foldr((c, acc) -> c + x * acc, coefs; init=0.0)

const _ADF_TAU_STAR_N1 = Dict(:n => -1.04, :c => -1.61, :ct => -2.89, :ctt => -3.21)
const _ADF_TAU_MIN_N1 = Dict(:n => -19.04, :c => -18.83, :ct => -16.18, :ctt => -17.17)
const _ADF_TAU_MAX_N1 = Dict(:n => Inf, :c => 2.74, :ct => 0.7, :ctt => 0.54)
const _ADF_TAU_SMALLP_N1 = Dict(
    :n => (0.6344, 1.2378, 0.032496),
    :c => (2.1659, 1.4412, 0.038269),
    :ct => (3.2512, 1.6047, 0.049588),
    :ctt => (4.0003, 1.658, 0.048288),
)
const _ADF_TAU_LARGEP_N1 = Dict(
    :n => (0.4797, 0.93557, -0.06999, 0.033066),
    :c => (1.7339, 0.93202, -0.12745, -0.010368),
    :ct => (2.5261, 0.61654, -0.37956, -0.060285),
    :ctt => (3.0778, 0.49529, -0.41477, -0.059359),
)

"""
    adf_pvalue_response_surface(teststat, regression; N::Integer=1) -> Float64

MacKinnon (1994) response-surface p-value for an ADF `tau` statistic --
the actual method `statsmodels.tsa.stattools.adfuller` and R use, more
accurate than the linear-interpolation p-value available via
[`adf_test`](@ref)'s `pvalue_method=:interpolated`. `N=1` (a single
series -- always the case for a plain ADF test) is the only supported
value; see this file's response-surface section header for why N=2:12
aren't transcribed.

# Examples
```jldoctest
julia> using TSAnalytics

julia> round(adf_pvalue_response_surface(-3.5, :c), digits=6)  # matches statsmodels' mackinnonp(-3.5, 'c', 1) exactly
0.007987
```
"""
function adf_pvalue_response_surface(teststat::Real, regression::Symbol; N::Integer=1)
    regression in (:n, :c, :ct, :ctt) || throw(ArgumentError("regression must be :n, :c, :ct, or :ctt"))
    N == 1 || throw(ArgumentError("adf_pvalue_response_surface: only N=1 is supported (transcribed from statsmodels' N=1 table only)"))
    teststat > _ADF_TAU_MAX_N1[regression] && return 1.0
    teststat < _ADF_TAU_MIN_N1[regression] && return 0.0
    coefs = teststat <= _ADF_TAU_STAR_N1[regression] ? _ADF_TAU_SMALLP_N1[regression] : _ADF_TAU_LARGEP_N1[regression]
    return _std_normal_cdf(_horner(coefs, teststat))
end

const _ADF_CRIT_2010_N1 = Dict(
    :n => ((-2.56574, -2.2358, -3.627, 0.0), (-1.941, -0.2686, -3.365, 31.223), (-1.61682, 0.2656, -2.714, 25.364)),
    :c => ((-3.43035, -6.5393, -16.786, -79.433), (-2.86154, -2.8903, -4.234, -40.040), (-2.56677, -1.5384, -2.809, 0.0)),
    :ct => ((-3.95877, -9.0531, -28.428, -134.155), (-3.41049, -4.3904, -9.036, -45.374), (-3.12705, -2.5856, -3.925, -22.380)),
    :ctt => ((-4.37113, -11.5882, -35.819, -334.047), (-3.83239, -5.9057, -12.490, -118.284), (-3.55326, -3.6596, -5.293, -63.559)),
)

"""
    adf_critical_values_response_surface(regression; nobs=nothing, N::Integer=1) -> (p1, p5, p10)

MacKinnon (2010) finite-sample response-surface critical values for the
ADF `tau` statistic, at 1%/5%/10%. `nobs=nothing` (default) returns the
asymptotic values (matching the existing `_ADF_CRIT` table used by
[`adf_test`](@ref)'s `:interpolated` p-value method); an `Integer`
`nobs` gives the finite-sample-corrected values via
`c0 + c1/nobs + c2/nobs^2 + c3/nobs^3`. `N=1` is the only supported
value (see [`adf_pvalue_response_surface`](@ref)).

# Examples
```jldoctest
julia> using TSAnalytics

julia> round.(collect(adf_critical_values_response_surface(:c; nobs=100)); digits=5)  # matches statsmodels' mackinnoncrit(1,'c',100) exactly
3-element Vector{Float64}:
 -3.4975
 -2.89091
 -2.58243
```
"""
function adf_critical_values_response_surface(regression::Symbol; nobs::Union{Nothing,Integer}=nothing, N::Integer=1)
    regression in (:n, :c, :ct, :ctt) || throw(ArgumentError("regression must be :n, :c, :ct, or :ctt"))
    N == 1 || throw(ArgumentError("adf_critical_values_response_surface: only N=1 is supported"))
    coefs = _ADF_CRIT_2010_N1[regression]
    vals = nobs === nothing ? getindex.(coefs, 1) : _horner.(coefs, 1 / nobs)
    return (p1=vals[1], p5=vals[2], p10=vals[3])
end

const _PP_Z_STAR = Dict(:n => -1.79146, :c => -5.04709, :ct => -9.22766)
const _PP_Z_SMALLP = Dict(
    :n => (0.05872, -0.69633, 0.02471, -0.04283),
    :c => (1.94205, -1.47677, 0.21163, -0.06288),
    :ct => (4.05596, -2.34128, 0.41403, -0.08312),
)
const _PP_Z_LARGEP = Dict(
    :n => (0.56681, 0.67544, 0.06881, 0.00235),
    :c => (1.70059, 0.49465, 0.02636, 0.00055),
    :ct => (2.60323, 0.39217, 0.01321, 0.00019),
)

"""
    pp_rho_pvalue(teststat, regression) -> Float64

MacKinnon "ADF-z" response-surface p-value for the Phillips-Perron
`:rho` statistic -- what [`pp_test`](@ref)`(x; test_type=:rho)` uses
(replacing a previously-`NaN` p-value). Transcribed from, and verified
end-to-end against, `arch.unitroot.PhillipsPerron(..., test_type="rho")`'s
own real executed output (Python `arch` package) -- see this file's
response-surface section header for why these tables, not the
similarly-named ones in `adfvalues_source.py`.

# Examples
```jldoctest
julia> using TSAnalytics

julia> 0.0 <= pp_rho_pvalue(-80.0, :c) <= 1.0
true
```
"""
function pp_rho_pvalue(teststat::Real, regression::Symbol)
    regression in (:n, :c, :ct) || throw(ArgumentError("regression must be :n, :c, or :ct"))
    if teststat <= _PP_Z_STAR[regression]
        return _std_normal_cdf(_horner(_PP_Z_SMALLP[regression], log(abs(teststat))))
    else
        return _std_normal_cdf(_horner(_PP_Z_LARGEP[regression], teststat))
    end
end

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
based on. `pvalue` is, by default, the MacKinnon response-surface p-value
(see [`adf_pvalue_response_surface`](@ref)) -- the same method
`statsmodels`/R use; pass `pvalue_method=:interpolated` for the older,
cruder linear-interpolation approximation instead.
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
    `pvalue_method=:response_surface` (**default**) matches
    `statsmodels`/R's own finite-sample-accurate method exactly (see
    [`adf_pvalue_response_surface`](@ref)). `pvalue_method=:interpolated`
    keeps the older, cruder linear-interpolation-among-asymptotic-critical-values
    approximation available (nor `tseries::adf.test`'s own Banerjee et
    al. table, which only ever applies to its one fixed constant+trend
    case) -- adequate for a significant/not-significant call at
    1/5/10%, not for quoting an exact p-value.

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
                   autolag::Union{Nothing,Symbol}=:aic, pvalue_method::Symbol=:response_surface)
    regression in (:n, :c, :ct, :ctt) ||
        throw(ArgumentError("regression must be :n, :c, :ct, or :ctt"))
    autolag === nothing || autolag in (:aic, :bic, :tstat) ||
        throw(ArgumentError("autolag must be :aic, :bic, :tstat, or nothing"))
    pvalue_method in (:response_surface, :interpolated) ||
        throw(ArgumentError("pvalue_method must be :response_surface or :interpolated"))

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
    pval = pvalue_method == :response_surface ?
           adf_pvalue_response_surface(final.tstat, regression) :
           _interp_pvalue_left(final.tstat, _ADF_CRIT[regression])

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

"""
    kpss_hobijn_autolag(resids, nobs) -> Int

Hobijn, Franses & Ooms (1998) data-dependent lag-selection rule for
KPSS's long-run variance bandwidth -- what [`kpss_test`](@ref)'s
`nlags=:auto` uses. Transcribed directly from a real, short, portable
implementation (`JimVaranelli/KPSS-autolag` on GitHub, submitted to
`statsmodels`) rather than the paper -- see
`handoff/verification/KPSS_hobijn_autolag.py`. Verified against real
`statsmodels.tsa.stattools.kpss(..., nlags="auto")` output on this
project's own bundled Nile and sunspot series
(`test/verification/unitroot/`).
"""
function kpss_hobijn_autolag(resids::AbstractVector{<:Real}, nobs::Integer)
    covlags = floor(Int, nobs^(2 / 9))
    s0 = sum(abs2, resids) / nobs
    s1 = 0.0
    for i in 1:covlags
        resids_prod = dot(view(resids, i+1:nobs), view(resids, 1:nobs-i)) / (nobs / 2)
        s0 += resids_prod
        s1 += i * resids_prod
    end
    s_hat = s1 / s0
    gamma_hat = 1.1447 * (s_hat^2)^(1 / 3)
    return min(nobs, floor(Int, gamma_hat * nobs^(1 / 3)))
end

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
  - `:auto` (Python's actual default) -- the Hobijn, Franses & Ooms
    (1998) data-dependent method, via [`kpss_hobijn_autolag`](@ref).

!!! note "Default mismatch with Python, by choice, not a gap"
    This function's default (`:short`) matches R's default, not Python's
    (`:auto`/Hobijn) -- `:auto` is fully implemented and available, just
    not the default here, to match this project's own established R
    default. Pass `nlags=:auto` for Python's actual default behavior, or
    `nlags=:legacy` for Python's non-default `"legacy"` mode.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(300));

julia> kpss_test(y).pvalue <= 0.05   # random walk: rejects level-stationarity
true

julia> kpss_test(y; regression=:ct, nlags=:legacy).regression
:ct

julia> kpss_test(y; nlags=:auto).lags >= 0
true
```
"""
function kpss_test(x; regression::Symbol=:c, nlags::Union{Symbol,Integer}=:short)
    regression in (:c, :ct) || throw(ArgumentError("regression must be :c or :ct"))
    y = tsvalues(x)
    n = length(y)

    X = regression == :ct ? hcat(ones(n), collect(1.0:n)) : reshape(ones(n), n, 1)
    _, resid, _ = _ols(X, y)

    l = if nlags isa Integer
        nlags >= 0 || throw(ArgumentError("nlags must be >= 0"))
        nlags < n || throw(ArgumentError("nlags must be < n (got nlags=$nlags, n=$n)"))
        Int(nlags)
    elseif nlags === :short
        min(trunc(Int, 4 * (n/100)^0.25), n-1)
    elseif nlags === :legacy
        min(ceil(Int, 12 * (n/100)^0.25), n-1)
    elseif nlags === :auto
        min(kpss_hobijn_autolag(resid, n), n-1)
    else
        throw(ArgumentError("nlags must be :short, :legacy, :auto, or an Integer"))
    end

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
[`pp_test`](@ref). `pvalue` uses the MacKinnon response-surface method
for both variants ([`adf_pvalue_response_surface`](@ref) for `:tau`,
[`pp_rho_pvalue`](@ref) for `:rho`), matching `arch.unitroot.PhillipsPerron`
exactly.
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
    print(io,   "  p-value             : ", round(t.pvalue, digits=4))
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

!!! note "`:rho`'s p-value"
    `:rho`'s asymptotic null distribution (the Dickey-Fuller "Z"/
    coefficient distribution) is genuinely different from `:tau`'s
    (t-distribution-like) -- computed via [`pp_rho_pvalue`](@ref)'s own
    separate MacKinnon "ADF-z" table, not by reusing `:tau`'s table.

!!! note "p-value method"
    Both variants use the MacKinnon response-surface method (matching
    `arch.unitroot.PhillipsPerron` exactly), not linear interpolation
    and not Banerjee et al.'s table (what `tseries::pp.test`
    specifically uses).

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(300));

julia> pp_test(y).pvalue > 0.10   # random walk: fails to reject the unit-root null
true

julia> pp_test(y; test_type=:rho).pvalue > 0.10   # same conclusion under the :rho variant
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
    # tau/t-distribution family adf_pvalue_response_surface tabulates --
    # pp_rho_pvalue's own separate "ADF-z" table, verified end-to-end
    # against arch.unitroot.PhillipsPerron (which computes this exact
    # stat_rho formula).
    pval = test_type == :tau ? adf_pvalue_response_surface(stat, trend) : pp_rho_pvalue(stat, trend)

    return PPTest(stat, pval, l, trend, test_type, nobs)
end
