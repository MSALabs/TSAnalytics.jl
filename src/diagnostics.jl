export LjungBoxTest, QSTest, JarqueBeraTest, ljungbox_test, qs_test, jarque_bera_test

"""_chisq_ccdf(x, df) -- upper tail P(X > x) for X ~ chi-squared(df),
via the regularized upper incomplete gamma function, computed by a
continued-fraction expansion. Avoids taking a hard Distributions.jl
dependency purely for a chi-square tail probability."""
function _chisq_ccdf(x::Real, df::Real)
    x < 0 && return 1.0
    a = df / 2
    z = x / 2
    return _upper_incomplete_gamma_reg(a, z)
end

function _upper_incomplete_gamma_reg(a::Real, x::Real; maxiter::Int=200, tol::Float64=1e-14)
    x <= 0 && return 1.0
    if x < a + 1
        # series expansion for the lower incomplete gamma, then complement
        return 1 - _lower_incomplete_gamma_series(a, x; maxiter=maxiter, tol=tol)
    else
        # continued fraction (Lentz's algorithm) for the upper incomplete gamma
        tiny = 1e-300
        b = x + 1 - a
        c = 1 / tiny
        d = 1 / b
        h = d
        for i in 1:maxiter
            an = -i * (i - a)
            b += 2
            d = an * d + b
            abs(d) < tiny && (d = tiny)
            c = b + an / c
            abs(c) < tiny && (c = tiny)
            d = 1 / d
            del = d * c
            h *= del
            abs(del - 1) < tol && break
        end
        lng = _loggamma(a)
        return exp(-x + a*log(x) - lng) * h
    end
end

function _lower_incomplete_gamma_series(a::Real, x::Real; maxiter::Int=200, tol::Float64=1e-14)
    lng = _loggamma(a)
    ap = a
    summ = 1 / a
    del = summ
    for _ in 1:maxiter
        ap += 1
        del *= x / ap
        summ += del
        abs(del) < abs(summ) * tol && break
    end
    return summ * exp(-x + a*log(x) - lng)
end

# Lanczos approximation for log(Gamma(x))
function _loggamma(x::Real)
    g = 7
    c = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)
    if x < 0.5
        return log(pi / sin(pi*x)) - _loggamma(1 - x)
    end
    x -= 1
    a = c[1]
    t = x + g + 0.5
    for i in 1:g+1
        a += c[i+1] / (x + i)
    end
    return 0.5*log(2*pi) + (x+0.5)*log(t) - t + log(a)
end

# ---------------------------------------------------------------------------
# Ljung-Box
# ---------------------------------------------------------------------------

"""
    LjungBoxTest <: HypothesisTest

Result of a (portmanteau) Ljung-Box test for residual autocorrelation.
`lags` are the lags actually tested; `fitdf` is the number of ARMA
parameters already estimated (subtracted from the chi-squared degrees of
freedom, as is standard when testing model residuals rather than raw
data). `bp_statistic`/`bp_pvalue` are the Box-Pierce analogues, populated
only when [`ljungbox_test`](@ref) is called with `boxpierce=true`
(`nothing` otherwise).
"""
struct LjungBoxTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    lags::Vector{Int}
    df::Int
    bp_statistic::Union{Nothing,Float64}
    bp_pvalue::Union{Nothing,Float64}
end
LjungBoxTest(statistic, pvalue, lags, df) = LjungBoxTest(statistic, pvalue, lags, df, nothing, nothing)

function Base.show(io::IO, t::LjungBoxTest)
    println(io, "Ljung-Box test")
    println(io, "  lags tested   : ", t.lags)
    println(io, "  df            : ", t.df)
    println(io, "  Q statistic   : ", round(t.statistic, digits=4))
    println(io, "  p-value       : ", round(t.pvalue, digits=4))
    if t.bp_statistic !== nothing
        println(io, "  Box-Pierce Q  : ", round(t.bp_statistic, digits=4))
        print(io,   "  Box-Pierce p  : ", round(t.bp_pvalue, digits=4))
    else
        print(io, "  (pass boxpierce=true for the Box-Pierce statistic too)")
    end
end

"""
    ljungbox_test(x, lags=nothing; fitdf=0, boxpierce=false, clip_negative=false) -> LjungBoxTest

Ljung-Box (and optionally Box-Pierce) portmanteau test of the null
hypothesis that `x` is white noise (no autocorrelation). Ljung-Box is
always the primary statistic returned:
    Q = n(n+2) * sum_{k in lags} rho_k^2 / (n - k)
asymptotically chi-squared with `length(lags) - fitdf` degrees of
freedom. R's `stats::Box.test()` actually **defaults** to the older,
weaker Box-Pierce statistic instead (`type="Box-Pierce"`) -- both
Wikipedia and `statsmodels`' own docs note Ljung-Box has better
finite-sample properties, so this deliberately does not default-match R
here; see `handoff/stage-2.4-ljungbox-handoff.md` for the full
comparison, including the exact R/Python formulas.

- `lags`: `nothing` (default) computes `min(10, n÷5)`, matching Python's
  current `acorr_ljungbox` default (R's default is a single fixed
  `lag=1`, unusually small in practice). An `Integer` `h` gives the
  cumulative statistic through lag `h` (tests `1:h`) -- matches both
  references' single-lag behavior exactly. An `AbstractVector` sums
  **exactly** the given lags (e.g. seasonal lags only, what
  [`qs_test`](@ref) needs) -- this is **not** the same as passing a lag
  list to Python's `acorr_ljungbox`, which instead reports one
  *cumulative* statistic per listed lag as separate rows (verified
  directly: `acorr_ljungbox(y, lags=[5,10])` gives two increasing
  cumulative statistics, not one number over exactly `{5,10}`). See the
  handoff doc for a worked numeric example of the difference -- a Python
  user passing a lag vector expecting per-lag cumulative rows will
  silently get a different number here.
- `fitdf`: degrees of freedom to subtract (e.g. `p+q` for ARMA(p,q)
  residuals) -- matches R's argument name exactly.
- `boxpierce`: if `true`, also compute the Box-Pierce statistic
  (`Q = n*sum(rho_k^2)`, no `(n-k)` denominator, no `n+2` factor),
  populating `bp_statistic`/`bp_pvalue`. Matches Python's argument name
  and default (`false`) exactly.
- `clip_negative`: if `true`, autocorrelations are clamped to `max(0,
  rho)` before squaring at each lag -- a "one-sided" portmanteau variant
  that only counts positive autocorrelation as evidence against the
  white-noise null. This is what [`qs_test`](@ref) uses internally
  (JDemetra+'s own documented QS formula); exposed here directly since
  it's a legitimate variant in its own right, not exclusively a QS
  implementation detail.

`x` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = randn(300);

julia> length(ljungbox_test(y).lags)   # default lags: min(10, 300÷5) = 10
10

julia> r = ljungbox_test(y, [5, 10]; boxpierce=true);

julia> r.bp_statistic !== nothing
true

julia> r.statistic > r.bp_statistic   # Ljung-Box's (n+2)/(n-k) scaling inflates it relative to Box-Pierce
true

julia> ljungbox_test(y, 10; clip_negative=true).statistic <= ljungbox_test(y, 10).statistic
true
```
"""
function ljungbox_test(x; fitdf::Integer=0, boxpierce::Bool=false, clip_negative::Bool=false)
    n = length(tsvalues(x))
    ljungbox_test(x, min(10, n ÷ 5); fitdf=fitdf, boxpierce=boxpierce, clip_negative=clip_negative)
end

function ljungbox_test(x, lags::Integer; fitdf::Integer=0, boxpierce::Bool=false, clip_negative::Bool=false)
    ljungbox_test(x, collect(1:lags); fitdf=fitdf, boxpierce=boxpierce, clip_negative=clip_negative)
end

function ljungbox_test(x, lags::AbstractVector{<:Integer}; fitdf::Integer=0, boxpierce::Bool=false, clip_negative::Bool=false)
    y = tsvalues(x)
    n = length(y)
    rho_raw = acf(y, collect(lags); bartlett=false).values
    rho = clip_negative ? max.(rho_raw, 0.0) : rho_raw

    Q = n * (n + 2) * sum(rho[i]^2 / (n - lags[i]) for i in eachindex(lags))
    df = length(lags) - fitdf
    df > 0 || throw(ArgumentError("degrees of freedom must be positive; reduce fitdf or add lags"))
    pval = _chisq_ccdf(Q, df)

    bp_stat, bp_pval = if boxpierce
        Qbp = n * sum(rho[i]^2 for i in eachindex(lags))
        (Qbp, _chisq_ccdf(Qbp, df))
    else
        (nothing, nothing)
    end

    return LjungBoxTest(Q, pval, collect(lags), df, bp_stat, bp_pval)
end

# ---------------------------------------------------------------------------
# QS test (seasonal-lag portmanteau, as used by X-13ARIMA-SEATS/JDemetra+)
# ---------------------------------------------------------------------------

"""
    QSTest <: HypothesisTest

The QS statistic: a Ljung-Box-type portmanteau test restricted to the
seasonal lags `s` and `2s`, used (e.g. by X-13ARIMA-SEATS and JDemetra+) to
check for residual seasonality after seasonal adjustment or after fitting
a seasonal ARIMA model.
"""
struct QSTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    period::Int
end

function Base.show(io::IO, t::QSTest)
    println(io, "QS test (residual seasonality)")
    println(io, "  period (s)    : ", t.period)
    println(io, "  QS statistic  : ", round(t.statistic, digits=4))
    print(io,   "  p-value       : ", round(t.pvalue, digits=4))
end

"""
    qs_test(x, period::Integer) -> QSTest

QS statistic for residual seasonality at seasonal period `period`, per
JDemetra+'s definition (verified directly from its documentation, not
assumed): a one-sided Ljung-Box variant restricted to the first two
seasonal lags, `period` and `2*period`, counting only *positive*
autocorrelation as evidence of seasonality --

    QS = n(n+2) * sum_{i=1}^{2} [max(0, rho_hat_{i*period})]^2 / (n - i*period)

*"The QS test is a variant of the Ljung-Box test computed on seasonal
lags, where we only consider positive auto-correlations"* -- a negative
correlation at a seasonal lag isn't evidence of seasonality and should
contribute nothing to the statistic, not be squared into a false-positive
contribution.

Implemented as a thin wrapper over [`ljungbox_test`](@ref) with
`clip_negative=true`, matching JDemetra+'s own architecture (its
`QSTest` class calls the more general `ec.tstoolkit.stats.LjungBoxTest`,
per the JDemetra+ documentation).

!!! note "Differencing is the caller's responsibility"
    JDemetra+ and X-13ARIMA-SEATS use *different* default differencing
    conventions before computing this statistic (JDemetra+: first
    difference once; X-13/TRAMO-SEATS: `max(1, min(d+D, 2))`), and the
    literature notes the statistic is biased under under-differencing.
    Since the two references don't agree with each other, this function
    does not silently difference `x` -- pass an appropriately
    differenced series for your use case.

Used to check seasonally adjusted series or SARIMA residuals for
leftover seasonality (the same diagnostic reported in X-13ARIMA-SEATS
output). `x` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> t = 0:239; y = sin.(2π .* t ./ 24) .+ 0.3 .* randn(MersenneTwister(7), 240);

julia> r = qs_test(y, 12);

julia> r.statistic < ljungbox_test(y, [12, 24]; clip_negative=false).statistic  # clipping only ever reduces the statistic
true

julia> isapprox(r.statistic, ljungbox_test(y, [12, 24]; clip_negative=true).statistic; atol=1e-10)
true

julia> qs_test(y, 1)
ERROR: ArgumentError: period must be >= 2
```
"""
function qs_test(x, period::Integer)
    period >= 2 || throw(ArgumentError("period must be >= 2"))
    y = tsvalues(x)
    n = length(y)
    2*period < n || throw(ArgumentError("series too short relative to period for QS test"))

    lb = ljungbox_test(y, [period, 2*period]; clip_negative=true)
    return QSTest(lb.statistic, lb.pvalue, period)
end

# ---------------------------------------------------------------------------
# Jarque-Bera
# ---------------------------------------------------------------------------

"""
    JarqueBeraTest <: HypothesisTest

Result of a Jarque-Bera normality test. `skewness`/`kurtosis` are the
sample (biased, `n`-denominator) estimators used to compute `statistic`
-- exposed directly since they're useful diagnostic information on their
own, matching Python's richer return value (`statsmodels`'
`jarque_bera` returns them too; R's `tseries::jarque.bera.test` doesn't).
"""
struct JarqueBeraTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    skewness::Float64
    kurtosis::Float64
    n::Int
end

function Base.show(io::IO, t::JarqueBeraTest)
    println(io, "Jarque-Bera normality test")
    println(io, "  skewness      : ", round(t.skewness, digits=4))
    println(io, "  kurtosis      : ", round(t.kurtosis, digits=4))
    println(io, "  n             : ", t.n)
    println(io, "  JB statistic  : ", round(t.statistic, digits=4))
    print(io,   "  p-value       : ", round(t.pvalue, digits=4))
end

"""
    jarque_bera_test(x) -> JarqueBeraTest

Jarque-Bera test of the null hypothesis that `x` is drawn from a normal
distribution, based on sample skewness and kurtosis:

    JB = n * (S^2/6 + (K-3)^2/24)

where `S` and `K` are the *biased* (population, `n`-denominator) skewness
and kurtosis estimators -- confirmed to match both R's
`tseries::jarque.bera.test` (verified from source) and Python's
`statsmodels.stats.stattools.jarque_bera` (verified numerically against
real `statsmodels` output, to full floating-point precision) exactly;
unlike most of this package's other diagnostic tests, no cross-language
discrepancy was found here -- see `handoff/stage-2.6-jarque-bera-handoff.md`.
`JB` is asymptotically chi-squared with 2 degrees of freedom under the
null.

Returns `skewness` and `kurtosis` directly (matching Python's richer
return value, which R's version doesn't provide) -- useful diagnostic
info on its own, not just an intermediate computation.

`x` accepts anything [`tsvalues`](@ref) does. Commonly applied to
regression or ARMA residuals, but works on any vector.

!!! note "Reliability at small n"
    Both references note the chi-squared approximation is asymptotic;
    SciPy's docs specifically recommend n > 2000 for the test to be
    reliable. Treat results on short series with proportionate caution --
    this isn't unique to this implementation, it's inherent to the test.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y_normal = randn(2000);

julia> jarque_bera_test(y_normal).pvalue > 0.05   # normal data: should not reject normality
true

julia> y_skewed = randn(MersenneTwister(1), 2000).^2;

julia> jarque_bera_test(y_skewed).pvalue < 0.01   # chi-sq(1)-distributed: should reject normality
true
```
"""
function jarque_bera_test(x)
    y = tsvalues(x)
    n = length(y)
    n >= 2 || throw(ArgumentError("jarque_bera_test: need at least 2 observations"))

    m1 = sum(y) / n
    c = y .- m1
    m2 = sum(abs2, c) / n
    m3 = sum(c.^3) / n
    m4 = sum(c.^4) / n

    skewness = m3 / m2^1.5
    kurtosis = m4 / m2^2

    JB = (n / 6) * (skewness^2 + (kurtosis - 3)^2 / 4)
    pval = _chisq_ccdf(JB, 2)

    return JarqueBeraTest(JB, pval, skewness, kurtosis, n)
end
