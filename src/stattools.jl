export acf, pacf, ACFResult

"""
    ACFResult

Container for an (partial) autocorrelation function together with its
white-noise confidence bounds, so plotting recipes and printed summaries
don't have to recompute them. Shared by both [`acf`](@ref) and
[`pacf`](@ref) -- `kind` is what distinguishes a result that came from
one versus the other.

# Fields
- `lags::Vector{Int}`
- `values::Vector{Float64}`
- `lower::Vector{Float64}`, `upper::Vector{Float64}`: pointwise `1-alpha`
  confidence band under the white-noise null.
- `n::Int`: number of observations the (P)ACF was computed from.
- `qstat::Union{Nothing,Vector{Float64}}`, `pvalues::Union{Nothing,Vector{Float64}}`:
  per-lag Ljung-Box statistic/p-value, populated only when [`acf`](@ref) is
  called with `qstat=true` (`nothing` otherwise, including always for
  [`pacf`](@ref)).
- `kind::Symbol`: `:acf` or `:pacf`, whichever function produced this
  result -- defaults to `:acf` on the backward-compatible constructors
  below that don't take it explicitly (`acf`/`pacf` themselves always
  pass it explicitly and are unaffected by that default).
"""
struct ACFResult
    lags::Vector{Int}
    values::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
    n::Int
    qstat::Union{Nothing,Vector{Float64}}
    pvalues::Union{Nothing,Vector{Float64}}
    kind::Symbol
end
ACFResult(lags, values, lower, upper, n, qstat, pvalues) =
    ACFResult(lags, values, lower, upper, n, qstat, pvalues, :acf)
ACFResult(lags, values, lower, upper, n) =
    ACFResult(lags, values, lower, upper, n, nothing, nothing, :acf)

function Base.show(io::IO, r::ACFResult)
    # "ACFResult" is the one Julia type both acf and pacf return (see the
    # docstring) -- kept visible here rather than printed as a nonexistent
    # "PACFResult", with `kind` as the parenthetical that actually
    # distinguishes which function produced this particular result.
    label = r.kind == :pacf ? "PACF" : "ACF"
    print(io, "ACFResult (", label, "): ", length(r.lags), " lags (",
          first(r.lags), ":", last(r.lags), "), n=", r.n)
    r.qstat !== nothing && print(io, ", with Ljung-Box qstat/pvalues")
end

function _confidence_z(alpha::Real)
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0,1)"))
    # inverse standard normal CDF via a rational (Acklam-style) approximation
    # avoids a hard Distributions.jl dependency just for a z-quantile.
    p = 1 - alpha / 2
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)
    plow = 0.02425
    phigh = 1 - plow
    if p < plow
        q = sqrt(-2*log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= phigh
        q = p - 0.5
        r = q*q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2*log(1-p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

_default_lags(n::Integer, from::Integer) = from:min(n-1, floor(Int, 10*log10(n)))

"""
    _acovf(y, maxlag; demean=true, adjusted=false)

Sample autocovariance at lags `0:maxlag`. `adjusted=false` (default)
divides by `n` at every lag, matching R's fixed behavior and
statsmodels' `adjusted=False` default. `adjusted=true` divides by `n-k`
at lag `k`, matching statsmodels' `adjusted=True`.
"""
function _acovf(y::AbstractVector{<:Real}, maxlag::Integer; demean::Bool=true, adjusted::Bool=false)
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

Full-fledged sample autocorrelation function, natively implemented (no
`StatsBase` dependency). Matches R's `stats::acf()` and Python's
`statsmodels.tsa.stattools.acf()` -- with two verified, genuine
default-behavior discrepancies between those two references worth
knowing about (not implementation bugs if you see them elsewhere):

- **Confidence bands.** `bartlett=true` (the default here, matching
  statsmodels' default -- NOT what `plot.acf()` uses in R by default)
  uses Bartlett's formula, which widens with lag:
  `var[k] = (1 + 2*sum(acf[1:k-1].^2)) / n` for `k >= 2`, `var[1] = 1/n`.
  Set `bartlett=false` for the simple constant-width `z/sqrt(n)` band
  instead.
- **Denominator.** `adjusted=false` (default) divides autocovariances by
  `n` at every lag, matching R's only behavior. `adjusted=true` divides
  by `n-k` at lag `k`, matching statsmodels' `adjusted=true` (statsmodels'
  own default). R has no equivalent option -- it always behaves like
  `adjusted=false`.

`qstat=true` additionally populates `ACFResult.qstat`/`.pvalues` with the
per-lag Ljung-Box statistic and p-value (lag 0 excluded), using the same
`_chisq_ccdf` tail helper as [`ljungbox_test`](@ref).

No `fft` option: this project doesn't take `FFTW.jl` as a core dependency
(see `development-sequence.md`); direct O(n·maxlag) computation is used
instead, adequate for typical series lengths.

`x` accepts anything [`tsvalues`](@ref) does. Throws `ArgumentError` if
`x` contains `NaN` (mirrors R's `na.fail` default; no partial/dropped-NA
handling is implemented yet).

See also [`pacf`](@ref).

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(60));

julia> acf(y, 0:3).values[1]   # lag-0 autocorrelation is always exactly 1
1.0

julia> r_bartlett = acf(y, 0:5; bartlett=true);

julia> r_simple = acf(y, 0:5; bartlett=false);

julia> r_bartlett.upper[end] > r_bartlett.upper[2]   # Bartlett bands widen with lag
true

julia> r_simple.upper[end] == r_simple.upper[2]      # simple bands stay constant
true
```
"""
function acf(x, lags::Union{Nothing,AbstractVector{<:Integer}}=nothing;
             alpha::Real=0.05, demean::Bool=true, adjusted::Bool=false,
             bartlett::Bool=true, qstat::Bool=false)
    y = tsvalues(x)
    any(isnan, y) && throw(ArgumentError("acf: NaN present (no missing-data policy implemented yet; see handoff/stage-1.3-acf-pacf-handoff.md)"))
    n = length(y)
    ls = lags === nothing ? _default_lags(n, 0) : lags
    maxlag = maximum(ls)

    acov = _acovf(y, maxlag; demean=demean, adjusted=adjusted)
    full_acf = acov ./ acov[1]
    vals = full_acf[ls .+ 1]

    z = _confidence_z(alpha)
    if bartlett
        varacf = zeros(maxlag + 1)
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

    return ACFResult(collect(ls), vals, lower, upper, n, qs, pv, :acf)
end

"""
    _durbin_levinson(acov, maxlag)

Partial autocorrelation via the Durbin-Levinson recursion applied to
autocovariances `acov[1:maxlag+1]` (index 1 = lag 0) -- "fitting
autoregressive models of successively higher orders", per R's own
description of `pacf()`'s algorithm.
"""
function _durbin_levinson(acov::AbstractVector{<:Real}, maxlag::Integer)
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

"""
    _pacf_ols(y, maxlag)

Partial autocorrelation via successive OLS regressions of `y[t]` on a
constant and `y[t-1], ..., y[t-k]` (matches statsmodels' `"ols"`/
`"ols-inefficient"` family), reusing the package's own `_ols` helper
rather than a separate implementation.
"""
function _pacf_ols(y::AbstractVector{<:Real}, maxlag::Integer)
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
implemented (no `StatsBase` dependency). `method`:

- `:yw` (default): Yule-Walker / Durbin-Levinson, `n-k` denominator --
  matches statsmodels' default (`"yw"`/`"ywadjusted"`).
- `:ywm`: Yule-Walker / Durbin-Levinson, `n` denominator -- matches R's
  `pacf()` exactly. **R has no adjusted-denominator option**, so
  comparing "R vs. Python PACF" naively will show a small but genuine
  discrepancy at the default settings of each (verified numerically, not
  a bug in either) -- pass `method=:ywm` for a direct R match.
- `:ols`: successive regression on a common sample -- matches
  statsmodels' `"ols"`/`"ols-inefficient"`.
- `:burg` is **not implemented** (a genuinely different algorithm, not
  just a different denominator/sample-size choice) -- a documented gap
  rather than a silent omission.

The confidence band is always the simple constant `z/sqrt(n)` (no
Bartlett-style widening) at every lag, matching both R and statsmodels'
PACF band convention -- theoretically justified differently from `acf`'s
band: under a true AR(p) null, PACF beyond order `p` has asymptotic
variance `1/n` regardless of lag.

`x` accepts anything [`tsvalues`](@ref) does.

See also [`acf`](@ref).

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(60));

julia> p_yw = pacf(y, 1:3; method=:yw);

julia> p_ywm = pacf(y, 1:3; method=:ywm);

julia> p_yw.values == p_ywm.values   # genuinely different denominators -- not a bug
false

julia> length(pacf(y, 1:5).values)
5

julia> pacf(y, 1:5; method=:burg)
ERROR: ArgumentError: method must be :yw, :ywm, or :ols (:burg not yet implemented)
```
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
    bound = z / sqrt(n)
    lower = fill(-bound, length(ls))
    upper = fill(bound, length(ls))

    return ACFResult(collect(ls), vals, lower, upper, n, nothing, nothing, :pacf)
end
