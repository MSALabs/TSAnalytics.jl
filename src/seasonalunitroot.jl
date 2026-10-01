export seasonal_strength, ocsb_test, nsdiffs, OCSBTest

"""
    seasonal_strength(y, period) -> Float64

Strength of seasonality, in `[0, 1]`: how much of the non-trend variation
a seasonal component accounts for.

    Fs = max(0, min(1, 1 - var(remainder) / var(remainder + seasonal)))

`0` means the seasonal component explains nothing the remainder does not;
`1` means the remainder is negligible beside it. A constant series returns
`0.0` directly, since decomposing one leaves only floating-point noise in
both components and their ratio carries no information. This is the measure
fpp3 introduces as a feature and the one R's `forecast::nsdiffs` uses by
default, through its internal `seas.heuristic`.

The components come from an STL decomposition with `seasonal_window=11`,
`seasonal_degree=0` and `inner=2`, which is what `forecast::mstl` passes
through to `stl()` for a single seasonal period -- read from R's own
source, not inferred. Variances use the `n-1` denominator, matching R's
`var`.

!!! note "Agrees with R exactly once its jump shortcut is disabled"
    R's `stl()` defaults to `s.jump = ceiling(s.window/10)` and
    similarly for the trend and low-pass steps, so it evaluates each
    Loess only every few points and interpolates between. This package
    has no jump parameter and evaluates at every point.

    Against `stl(..., s.jump=1, t.jump=1, l.jump=1)` the two agree to
    **all ten printed digits** on every series in
    `test/verification/nsdiffs/`. Against R's defaults they differ by
    `1e-4` to `8e-3` -- entirely R's interpolation, and never enough to
    move the `0.64` decision in [`nsdiffs`](@ref).

```jldoctest
julia> using TSAnalytics

julia> y = [100 + 10sin(2pi*t/12) + 0.1t for t in 1:120];

julia> seasonal_strength(y, 12) > 0.9
true
```

See also [`nsdiffs`](@ref), [`stl_decompose`](@ref).
"""
function seasonal_strength(y, period::Integer)
    yv = Float64.(collect(tsvalues(y)))
    period >= 2 || throw(ArgumentError("seasonal_strength: period must be >= 2, got $period"))
    n = length(yv)
    n > 2 * period || throw(ArgumentError(
        "seasonal_strength: need more than 2*period observations, got n=$n for period=$period"))

    # A constant series has no seasonality, but STL on one produces seasonal
    # and remainder components that are both floating-point noise, whose ratio
    # is arbitrary -- measured at 0.08 on one such series, which would read as
    # a real (if weak) finding. R never hits this because its `nsdiffs` checks
    # `is.constant` before calling the heuristic at all; answer it directly.
    all(≈(yv[1]), yv) && return 0.0

    d = stl_decompose(yv, period; seasonal_window=11, seasonal_degree=0, inner=2)
    vare = _var_n1(d.resid)
    varse = _var_n1(d.resid .+ d.seasonal)
    varse > 0 || return 0.0
    return clamp(1 - vare / varse, 0.0, 1.0)
end

_var_n1(v::AbstractVector{<:Real}) = begin
    n = length(v)
    n > 1 || return 0.0
    m = sum(v) / n
    sum(abs2, v .- m) / (n - 1)
end

"""
    OCSBTest

Result of [`ocsb_test`](@ref). Fields:

- `statistic`: the OCSB test statistic -- the t-value on the `Z5`
  regressor, which is what both R and `pmdarima` report.
- `critical`: the 5 % critical value for this `period`.
- `lag_order`: the lag order actually used.
- `lag_method`: how it was chosen.
- `n`: observations in the final regression.

**There is no p-value.** The null distribution is non-standard and the
critical values come from a smoothing of simulation results, available
at 5 % only -- so `statistic > critical` is the whole test, and a
p-value would have to be invented. Reported honestly rather than
fabricated, the same way [`nyblom_test`](@ref) reports critical values.
"""
struct OCSBTest
    statistic::Float64
    critical::Float64
    lag_order::Int
    lag_method::Symbol
    n::Int
end

function Base.show(io::IO, t::OCSBTest)
    print(io, "OCSB seasonal unit root test (Osborn, Chui, Smith & Birchenhall 1988)")
    print(io, "\n  statistic      = ", round(t.statistic, digits=6))
    print(io, "\n  5% critical    = ", round(t.critical, digits=6))
    print(io, "\n  lag order      = ", t.lag_order, "  (", t.lag_method, ")")
    print(io, "\n  observations   = ", t.n)
end

"""
    _ocsb_critical(period) -> Float64

5 % OCSB critical value, smoothed over simulation results across
seasonal periods:

    -0.2937411 * exp(-0.2850853*(log m - 0.7656451)
                     - 0.05983644*(log m - 0.7656451)^2) - 1.652202

Transcribed from `forecast::calcOCSBCritVal`, which is the only published
statement of the smoothing; `pmdarima` carries the same constants.
"""
function _ocsb_critical(period::Integer)
    lm = log(period)
    c = lm - 0.7656451
    return -0.2937411 * exp(-0.2850853 * c + (-0.05983644) * c^2) - 1.652202
end

# Fit the OCSB auxiliary regression at a given lag order, returning the
# fitted model's residual sum of squares, rank, observation count and the
# t-value on Z5.
#
# The construction follows Osborn et al. (1988) as `forecast::ocsb.test`
# states it -- read from R's own source. Every regression here is fitted
# WITHOUT an intercept (`lm(y ~ 0 + .)`), and Z4/Z5 enter lagged by 1 and
# by `period` respectively. Both details matter and both differ from
# `pmdarima`, which adds a constant to the auxiliary fit and does not lag
# Z4/Z5; see `ocsb_test`'s docstring.
function _fit_ocsb(x::Vector{Float64}, period::Int, lag::Int, maxlag::Int)
    n = length(x)

    # absolute-time indexing throughout, so the alignment R gets from `cbind`
    # on ts objects is explicit rather than implied by array offsets
    dx = [x[t] - x[t-1] for t in 2:n]                      # index t -> dx[t-1]
    dsx = [x[t] - x[t-period] for t in (period+1):n]       # index t -> dsx[t-period]
    getdx(t) = dx[t-1]
    getdsx(t) = dsx[t-period]
    gety(t) = getdsx(t) - getdsx(t - 1)                    # y = diff(diff(x, period))

    t_lo = period + 2 + maxlag
    t_lo <= n || throw(ArgumentError(
        "ocsb_test: not enough observations -- need more than period + 2 + maxlag " *
        "= $(period + 2 + maxlag), got n=$n"))
    rows = t_lo:n

    # stage 1: y on its own lags, no intercept
    yv = [gety(t) for t in rows]
    beta = if lag > 0
        A = [gety(t - k) for t in rows, k in 1:lag]
        _ols_nointercept(A, yv)
    else
        Float64[]
    end

    # stage 2: Z4 and Z5, the same AR filter applied to the two differences
    resid_at(getter, t) = begin
        r = getter(t)
        for k in 1:lag
            r -= beta[k] * getter(t - k)
        end
        r
    end
    z4 = [resid_at(getdsx, t - 1) for t in rows]        # lagged by 1
    z5 = [resid_at(getdx, t - period) for t in rows]    # lagged by period

    # stage 3: y on its lags plus Z4 and Z5, no intercept; the statistic is
    # the t-value on Z5
    k_ar = lag
    X = Matrix{Float64}(undef, length(rows), k_ar + 2)
    for (i, t) in enumerate(rows)
        for k in 1:k_ar
            X[i, k] = gety(t - k)
        end
        X[i, k_ar+1] = z4[i]
        X[i, k_ar+2] = z5[i]
    end

    coefs, se, rss, dof = _ols_with_se(X, yv)
    tval = se[end] > 0 ? coefs[end] / se[end] : NaN
    return (tval=tval, rss=rss, k=size(X, 2), nobs=length(rows), dof=dof)
end

_ols_nointercept(A::AbstractMatrix{Float64}, b::Vector{Float64}) = A \ b

function _ols_with_se(X::Matrix{Float64}, y::Vector{Float64})
    n, k = size(X)
    n > k || throw(ArgumentError("ocsb_test: $n observations for $k regressors"))
    coefs = X \ y
    resid = y .- X * coefs
    rss = sum(abs2, resid)
    dof = n - k
    s2 = rss / dof
    XtXinv = inv(Symmetric(X' * X))
    se = sqrt.(max.(diag(s2 .* XtXinv), 0.0))
    return coefs, se, rss, dof
end

# Gaussian log-likelihood of an OLS fit, as R's logLik.lm computes it, so
# AIC/BIC/AICc here match R's own lag selection.
_ols_loglik(rss::Float64, n::Int) = -n / 2 * (log(2pi) + log(rss / n) + 1)

"""
    ocsb_test(y, period; lag_method=:aic, maxlag=3) -> OCSBTest

Osborn, Chui, Smith & Birchenhall (1988) test for whether a series needs
**seasonal** differencing. `statistic > critical` means difference it.

`lag_method` is `:fixed`, `:aic`, `:bic` or `:aicc`. Under `:fixed` the
lag order is exactly `maxlag`; otherwise lag orders `1:maxlag` are fitted
and the criterion chosen. Defaults match what `forecast::nsdiffs` passes
(`maxlag=3`, AIC), not `ocsb.test`'s own bare defaults.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                                 "verification", "nsdiffs", "nile.csv")));

julia> t = ocsb_test(y, 4);

julia> t.statistic < t.critical   # no seasonal differencing needed
true
```

!!! warning "This follows R, and `pmdarima` differs in three ways"
    R's `forecast::ocsb.test` is the reference here -- it is the
    canonical implementation and `pmdarima` cites its source. Verified
    against it directly. `pmdarima`'s `OCSBTest` departs from it on
    three points, each of which moves the statistic:

    1. **It adds a constant** to the stage-one auxiliary regression.
       R fits `lm(y ~ 0 + .)`, with no intercept.
    2. **It does not lag Z4 and Z5.** R enters them as `lag(Z4, -1)` and
       `lag(Z5, -period)`.
    3. **Its lag-selection index is off by one relative to R's.** R takes
       `which.min` (1-based) and then uses `id - 1`; `pmdarima` takes a
       0-based `argmin` and also subtracts one, so it can reach `-1`.

    Point 3 is arguably a bug in `pmdarima` and points 1–2 are genuine
    specification differences. This package reproduces R, including R's
    own `id - 1`, so a reported `lag_order` is one below the lag that
    minimised the criterion. That is R's behaviour, not a transcription
    slip.

    The gap is not cosmetic. Measured directly, on the four series in
    `test/verification/nsdiffs/`:

    | series | R and this package | `pmdarima` |
    |---|---|---|
    | `cardox240` | `-1.53264645` | `-1.52134327` |
    | `ets_y` | `-1.11432803` | `-1.13726582` |
    | `airp` | `1.51876238` | **`2.77013411`** |
    | `nile` | `-9.33780588` | `-8.77490357` |

    All four still reach the same `D`, but a statistic that differs by
    `1.25` cannot be used as a cross-check on the other.

See also [`nsdiffs`](@ref), [`seasonal_strength`](@ref), [`adf_test`](@ref).
"""
function ocsb_test(y, period::Integer; lag_method::Symbol=:aic, maxlag::Integer=3)
    lag_method in (:fixed, :aic, :bic, :aicc) ||
        throw(ArgumentError("lag_method must be :fixed, :aic, :bic, or :aicc, got :$lag_method"))
    period >= 2 || throw(ArgumentError(
        "ocsb_test: period must be >= 2 -- the data must be seasonal for a seasonal unit root test"))
    maxlag >= 0 || throw(ArgumentError("maxlag must be >= 0, got $maxlag"))

    x = Float64.(collect(tsvalues(y)))
    any(isnan, x) && throw(ArgumentError("ocsb_test: NaN present (no missing-data policy implemented yet)"))

    lag = Int(maxlag)
    if maxlag > 0 && lag_method !== :fixed
        ics = Float64[]
        for l in 1:maxlag
            f = try
                _fit_ocsb(x, Int(period), l, Int(maxlag))
            catch e
                e isa Union{ArgumentError,LinearAlgebra.SingularException,LinearAlgebra.LAPACKException} || rethrow()
                nothing
            end
            push!(ics, f === nothing ? NaN : _ocsb_ic(lag_method, f))
        end
        all(isnan, ics) && throw(ArgumentError(
            "ocsb_test: every lag order up to maxlag=$maxlag produced a singular regression -- " *
            "try a longer series, a smaller maxlag, or test=:seas"))
        # R: `id <- which.min(icvals); maxlag <- id - 1`. Kept as R has it,
        # deliberately -- see the docstring.
        lag = argmin(replace(ics, NaN => Inf)) - 1
    end

    f = _fit_ocsb(x, Int(period), lag, lag)
    isfinite(f.tval) || throw(ArgumentError(
        "ocsb_test: the regression did not reach a solution -- try a longer series or test=:seas"))
    return OCSBTest(f.tval, _ocsb_critical(period), lag, lag_method, f.nobs)
end

function _ocsb_ic(method::Symbol, f)
    ll = _ols_loglik(f.rss, f.nobs)
    k = f.k + 1          # +1 for sigma2, matching R's AIC.lm
    method === :aic && return -2ll + 2k
    method === :bic && return -2ll + k * log(f.nobs)
    # AICc as forecast::ocsb.test computes it
    return -2ll + 2k + (2k * (k + 1)) / (f.nobs - k - 1)
end

"""
    nsdiffs(y, period; test=:seas, max_D=1, maxlag=3, lag_method=:aic) -> Int

How many **seasonal** differences the series needs, `0` to `max_D`.

This is the seasonal counterpart of the repeated [`kpss_test`](@ref) that
[`auto_arima`](@ref) uses to pick `d`, and supplying it is what lets
`auto_arima` choose `D` itself.

| `test` | Decides by | Matches |
|---|---|---|
| `:seas` (default) | [`seasonal_strength`](@ref)` > 0.64` | **R's `nsdiffs` default** |
| `:ocsb` | [`ocsb_test`](@ref)'s statistic against its critical value | `pmdarima`'s default, and R's `test="ocsb"` |

**The default follows R, which does not use OCSB.** `forecast::nsdiffs`
defaults to `test="seas"` -- an STL seasonal-strength heuristic with a
`0.64` threshold -- and offers `"ocsb"`, `"hegy"` and `"ch"` as
alternatives, the latter two requiring a separate package.
`pmdarima.arima.nsdiffs` defaults to OCSB. Pass `test=:ocsb` for that
convention.

Neither Canova-Hansen nor HEGY is implemented here; both are documented
gaps rather than silent omissions, and `:ch`/`:hegy` are rejected by
name.

```jldoctest
julia> using TSAnalytics

julia> y = [100 + 10sin(2pi*t/12) + 0.1t for t in 1:120];

julia> nsdiffs(y, 12)
1

julia> nsdiffs(cumsum(fill(1.0, 120)), 12)   # trend, no seasonality
0
```

A constant series, or one shorter than two full periods, returns `0`
without running anything -- there is no seasonality to test.

See also [`seasonal_strength`](@ref), [`ocsb_test`](@ref), [`auto_arima`](@ref).
"""
function nsdiffs(y, period::Integer; test::Symbol=:seas, max_D::Integer=1,
                  maxlag::Integer=3, lag_method::Symbol=:aic)
    test in (:seas, :ocsb) || throw(ArgumentError(
        test in (:hegy, :ch) ?
            "nsdiffs: test=:$test is not implemented (it needs HEGY/Canova-Hansen machinery " *
            "this package does not have yet) -- use :seas or :ocsb" :
            "nsdiffs: test must be :seas or :ocsb, got :$test"))
    max_D >= 0 || throw(ArgumentError("max_D must be >= 0, got $max_D"))
    period >= 2 || throw(ArgumentError(
        "nsdiffs: period must be >= 2 -- non-seasonal data needs no seasonal differencing"))

    x = Float64.(collect(tsvalues(y)))
    max_D == 0 && return 0
    # matching R: a constant series, or too little data, is 0 rather than an error
    all(≈(x[1]), x) && return 0
    length(x) <= 2 * period && return 0

    D = 0
    while D < max_D
        needs = try
            if test === :seas
                seasonal_strength(x, period) > 0.64
            else
                t = ocsb_test(x, period; maxlag=maxlag, lag_method=lag_method)
                t.statistic > t.critical
            end
        catch e
            e isa Union{ArgumentError,LinearAlgebra.SingularException,LinearAlgebra.LAPACKException} || rethrow()
            # R warns and stops differencing rather than failing the search
            @warn "nsdiffs: the $test test errored at D=$D; returning $D seasonal differences" exception=e
            return D
        end
        needs || break
        D += 1
        x = diff(x, period)
        length(x) <= 2 * period && break
        all(≈(x[1]), x) && break
    end
    return D
end
