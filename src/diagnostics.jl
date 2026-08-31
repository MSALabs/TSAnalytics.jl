using LinearAlgebra: eigen, I

export LjungBoxTest, QSTest, JarqueBeraTest, DurbinWatsonTest, ARCHLMTest, DKHeteroTest,
       ljungbox_test, qs_test, jarque_bera_test, durbin_watson_test,
       arch_lm_test, dk_heteroskedasticity_test, durbin_watson_pvalue_exact

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

"_beta_inc_reg(a, b, x) -- regularized incomplete beta function I_x(a,b),
via Lentz's continued-fraction algorithm (Numerical Recipes' `betacf`),
reusing this file's own `_loggamma`. Avoids a Distributions.jl
dependency purely for an F-distribution tail probability
([`dk_heteroskedasticity_test`](@ref))."
function _beta_inc_reg(a::Real, b::Real, x::Real)
    x <= 0 && return 0.0
    x >= 1 && return 1.0
    lbeta = _loggamma(a) + _loggamma(b) - _loggamma(a + b)
    bt = exp(-lbeta + a * log(x) + b * log(1 - x))
    return x < (a + 1) / (a + b + 2) ? bt * _betacf(a, b, x) / a : 1 - bt * _betacf(b, a, 1 - x) / b
end

function _betacf(a::Real, b::Real, x::Real; maxiter::Int=200, tol::Float64=1e-14)
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1 - qab * x / qap
    abs(d) < tiny && (d = tiny)
    d = 1 / d
    h = d
    for m in 1:maxiter
        m2 = 2m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        abs(d) < tiny && (d = tiny)
        c = 1 + aa / c
        abs(c) < tiny && (c = tiny)
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        abs(d) < tiny && (d = tiny)
        c = 1 + aa / c
        abs(c) < tiny && (c = tiny)
        d = 1 / d
        del = d * c
        h *= del
        abs(del - 1) < tol && break
    end
    return h
end

"_f_cdf(x, d1, d2) -- CDF of the F(d1,d2) distribution via the
regularized incomplete beta function."
function _f_cdf(x::Real, d1::Real, d2::Real)
    x <= 0 && return 0.0
    return _beta_inc_reg(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2))
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

# ---------------------------------------------------------------------------
# ARCH-LM (Engle 1982)
# ---------------------------------------------------------------------------

"""
    ARCHLMTest <: HypothesisTest

Result of Engle's (1982) Lagrange Multiplier test for ARCH
(autoregressive conditional heteroskedasticity) in a residual series.
"""
struct ARCHLMTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    lags::Int
    n::Int
end

function Base.show(io::IO, t::ARCHLMTest)
    println(io, "ARCH-LM test (Engle 1982)")
    println(io, "  lags          : ", t.lags)
    println(io, "  n             : ", t.n)
    println(io, "  LM statistic  : ", round(t.statistic, digits=4))
    print(io,   "  p-value       : ", round(t.pvalue, digits=4))
end

"""
    arch_lm_test(resid, lags::Integer=4) -> ARCHLMTest

Engle's (1982) Lagrange Multiplier test for ARCH (conditional
heteroskedasticity) in `resid`: regresses squared residuals `e_t^2` on
an intercept and their own `lags` lagged values,

    LM = (n - lags) * R^2

asymptotically chi-squared with `lags` degrees of freedom under the
null of no ARCH effects (`R^2 == 0`). Matches Tsay's textbook
presentation and R's `FinTS::ArchTest` formula -- verified by direct
execution (real OLS + chi-squared tail, `test/verification/diagnostics/`),
not just transcribed from the paper.

`resid` accepts anything [`tsvalues`](@ref) does; typically applied to
ARIMA-model residuals to check whether a GARCH-type model is warranted.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(123); e = randn(500); resid = zeros(500); resid[1] = e[1];

julia> for t in 2:500; resid[t] = sqrt(0.2 + 0.7*resid[t-1]^2) * e[t]; end

julia> arch_lm_test(resid).pvalue < 0.01   # genuine ARCH effects: rejects the no-ARCH null
true

julia> arch_lm_test(randn(MersenneTwister(1), 500)).pvalue > 0.05   # white noise: fails to reject
true
```
"""
function arch_lm_test(resid, lags::Integer=4)
    lags >= 1 || throw(ArgumentError("lags must be >= 1"))
    e2 = collect(Float64, tsvalues(resid)) .^ 2
    n = length(e2)
    n > lags + 1 || throw(ArgumentError("arch_lm_test: series too short for the requested number of lags"))

    yv = e2[(lags+1):end]
    nobs = length(yv)
    cols = [ones(nobs)]
    for i in 1:lags
        push!(cols, e2[(lags-i+1):(n-i)])
    end
    X = reduce(hcat, cols)

    _, resids_reg, _ = _ols(X, yv)
    ss_res = sum(abs2, resids_reg)
    ss_tot = sum(abs2, yv .- sum(yv) / nobs)
    r2 = 1 - ss_res / ss_tot

    stat = nobs * r2
    pval = _chisq_ccdf(stat, lags)

    return ARCHLMTest(stat, pval, lags, n)
end

# ---------------------------------------------------------------------------
# Durbin & Koopman heteroskedasticity (variance-ratio) test
# ---------------------------------------------------------------------------

"""
    DKHeteroTest <: HypothesisTest

Result of Durbin & Koopman's variance-ratio F-test for
heteroskedasticity in a residual series.
"""
struct DKHeteroTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    h::Int
    n::Int
end

function Base.show(io::IO, t::DKHeteroTest)
    println(io, "Durbin & Koopman heteroskedasticity test")
    println(io, "  h (block size): ", t.h)
    println(io, "  n             : ", t.n)
    println(io, "  F statistic   : ", round(t.statistic, digits=4))
    print(io,   "  p-value       : ", round(t.pvalue, digits=4))
end

"""
    dk_heteroskedasticity_test(resid) -> DKHeteroTest

Durbin & Koopman's variance-ratio test for heteroskedasticity in
`resid` (e.g. state-space/ARIMA model residuals): splits `resid` into
thirds of length `h = n÷3` and compares the sum of squares in the
*last* third against the *first* third (the middle third is discarded,
as in Durbin & Koopman's own presentation),

    H = sum(resid[end-h+1:end].^2) / sum(resid[1:h].^2)

which is `F(h, h)`-distributed under the null of constant variance;
two-sided p-value `2*min(F(H), 1-F(H))`. Verified by direct execution
(real F-distribution tail probability, `test/verification/diagnostics/`).

`resid` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(456); n = 300; t = 0:n-1;

julia> resid_hetero = randn(n) .* (1 .+ 3 .* t ./ n);  # variance genuinely grows over time

julia> dk_heteroskedasticity_test(resid_hetero).pvalue < 0.01   # rejects homoskedasticity
true

julia> dk_heteroskedasticity_test(randn(MersenneTwister(1), n)).pvalue > 0.05   # constant variance: fails to reject
true
```
"""
function dk_heteroskedasticity_test(resid)
    e = collect(Float64, tsvalues(resid))
    n = length(e)
    h = n ÷ 3
    h >= 1 || throw(ArgumentError("dk_heteroskedasticity_test: series too short"))
    stat = sum(abs2, e[(n-h+1):end]) / sum(abs2, e[1:h])
    Fcdf = _f_cdf(stat, h, h)
    pval = 2 * min(Fcdf, 1 - Fcdf)
    return DKHeteroTest(stat, pval, h, n)
end

# ---------------------------------------------------------------------------
# Durbin-Watson
# ---------------------------------------------------------------------------

"_std_normal_cdf(z) -- standard normal CDF via the identity
erf(x) = P(1/2, x^2) (regularized lower incomplete gamma, x >= 0),
reusing this file's own `_upper_incomplete_gamma_reg` rather than a
fresh erf polynomial approximation or a `Distributions.jl` dependency
just for a normal tail probability."
function _std_normal_cdf(z::Real)
    z == 0 && return 0.5
    x2 = (abs(z) / sqrt(2))^2
    erfx = 1 - _upper_incomplete_gamma_reg(0.5, x2)
    return z > 0 ? 0.5 * (1 + erfx) : 0.5 * (1 - erfx)
end

"""_pan_prob(x, a, c, niter) -- Farebrother's (1980, 1984) Applied
Statistics Algorithm AS 153 (AS R53 correction, "Pan's procedure") --
literal translation of `lmtest::dwtest`'s actual Fortran source
(`src/pan.f`, downloaded and read directly from the `lmtest` CRAN
source package -- not reconstructed from the paper). Computes

    P(a[1]*U[1]^2 + ... + a[M]*U[M]^2 < x*(U[1]^2+...+U[M]^2) + c)

for independent standard normal `U[1..M]`. For the Durbin-Watson use
case (`c=0` always), `x` is the observed DW statistic and `a` are the
nonzero eigenvalues of `M_X*A_dw` (see `_dw_annihilator_eigenvalues`).
`a` need not be pre-sorted -- ascending or descending both work,
matching the Fortran routine's own direction-detection. Verified
against the algorithm's own self-documented Farebrother (1984) table
(9 cases, `test/verification/durbinwatson/`) and, end-to-end with
`_dw_annihilator_eigenvalues`, against real `lmtest::dwtest(exact=TRUE)`
output."""
function _pan_prob(x::Real, a::AbstractVector{<:Real}, c::Real, niter::Integer)
    M = length(a)
    M >= 1 || throw(ArgumentError("_pan_prob: need at least one eigenvalue"))
    Av = Vector{Float64}(undef, M + 1)   # Av[i+1] == Fortran's 0-based A(i)
    Av[1] = Float64(x)
    Av[2:end] .= Float64.(a)

    ascending = !(Av[2] > Av[M+1])
    H, Kdir, I = ascending ? (1, 1, M) : (M, -1, 1)

    nu = 0
    found = false
    idx = H
    while true
        if Av[idx+1] >= x
            nu = idx
            found = true
            break
        end
        idx == I && break
        idx += Kdir
    end

    if !found
        c >= 0 && return 1.0
        throw(ArgumentError("_pan_prob: c < 0 with all a[i] < x is not implemented (unused by durbin_watson_test)"))
    end
    if nu == H && c <= 0
        return 0.0
    end

    Kdir == 1 && (nu -= 1)
    h = M - nu
    y = c == 0 ? Float64(h - nu) : c * (Av[2] - Av[M+1])

    local d, j1, j2, j3, j4
    if y >= 0
        d = 2
        h = nu
        Kdir = -Kdir
        j1, j2, j3, j4 = 0, 2, 3, 1
    else
        d = -2
        nu += 1
        j1, j2, j3, j4 = M - 2, M - 1, M + 1, M
    end

    pin = pi / (2 * niter)
    sum_ = 0.5 * (Kdir + 1)
    sgn = Kdir / Float64(niter)
    n2 = 2 * niter - 1

    hmod2 = h - 2 * (h ÷ 2)
    for _l1 in 1:(hmod2+1)
        for L2 in j2:d:nu
            sum1 = Av[j4+1]
            prod0 = Av[L2+1]
            u = 0.5 * (sum1 + prod0)
            v = 0.5 * (sum1 - prod0)
            sum1 = 0.0
            for ii in 1:2:n2
                yy = u - v * cos(ii * pin)
                num = yy - x
                prod = exp(-c / num)
                for kk in 1:j1
                    prod *= num / (yy - Av[kk+1])
                end
                for kk in j3:M
                    prod *= num / (yy - Av[kk+1])
                end
                sum1 += sqrt(abs(prod))
            end
            sgn = -sgn
            sum_ += sgn * sum1
            j1 += d
            j3 += d
            j4 += d
        end
        if d == 2
            j3 -= 1
        else
            j1 += 1
        end
        j2 = 0
        nu = 0
    end

    return sum_
end

"""_dw_annihilator_eigenvalues(X) -- the n-k eigenvalues (real parts,
near-zero ones discarded) of `M*A`, where `M = I - X*(X'X)^-1*X'` is the
OLS annihilator/projection matrix and `A` is the tridiagonal
`(1,2,2,...,2,1)`/`-1` matrix such that `DW = e'Ae / e'e` for OLS
residuals -- exactly `lmtest::dwtest`'s own real setup
(`A <- diag(c(1,rep(2,n-2),1)); A[abs(row(A)-col(A))==1] <- -1;
MA <- (I - X*Q1*X') %*% A`), read directly from its source
(`R/dwtest.R`)."""
function _dw_annihilator_eigenvalues(X::AbstractMatrix{<:Real}; tol::Float64=1e-10)
    n, k = size(X)
    A = zeros(n, n)
    A[1, 1] = 1.0
    A[n, n] = 1.0
    for i in 2:n-1
        A[i, i] = 2.0
    end
    for i in 1:n-1
        A[i, i+1] = -1.0
        A[i+1, i] = -1.0
    end
    XtX_inv = inv(X' * X)
    Mproj = Matrix{Float64}(I, n, n) - X * XtX_inv * X'
    MA = Mproj * A
    ev = real.(eigen(MA).values)
    ev = ev[abs.(ev).>tol]
    return sort(ev; rev=true)
end

"""
    durbin_watson_pvalue_exact(dw_stat, X; iterations=15, alternative=:greater) -> Float64

Exact p-value for a Durbin-Watson statistic via Farebrother's (1980,
1984) Applied Statistics Algorithm AS 153 (AS R53 correction, "Pan's
procedure") -- the actual algorithm and Fortran source R's
`lmtest::dwtest(exact=TRUE)` uses, read and translated directly from
`lmtest`'s real CRAN source (`src/pan.f`, `R/dwtest.R`), not
reconstructed from the citation alone (see `_pan_prob`).

`X` is the regression design matrix -- needed to compute the null
distribution's eigenvalues; the exact distribution genuinely depends on
`X`, not just `resid` (unlike the `:approx` normal approximation).
Verified end-to-end against real `lmtest::dwtest(exact=TRUE)` output on
two real series (`test/verification/durbinwatson/`), matching to 6+
significant figures, including the intermediate eigenvalues themselves.

# Examples
```jldoctest
julia> using TSAnalytics

julia> X = hcat(ones(20), 1.0:20);

julia> 0.0 <= durbin_watson_pvalue_exact(1.5, X) <= 1.0
true
```
"""
function durbin_watson_pvalue_exact(dw_stat::Real, X::AbstractMatrix{<:Real};
                                     iterations::Integer=15, alternative::Symbol=:greater)
    alternative in (:greater, :less, :two_sided) ||
        throw(ArgumentError("alternative must be :greater, :less, or :two_sided"))
    ev = _dw_annihilator_eigenvalues(X)
    isempty(ev) && throw(ArgumentError("durbin_watson_pvalue_exact: no nonzero eigenvalues found"))
    pdw = clamp(_pan_prob(dw_stat, ev, 0.0, iterations), 0.0, 1.0)
    return alternative == :greater ? pdw :
           alternative == :less    ? 1 - pdw :
                                      2 * min(pdw, 1 - pdw)
end

"""
    DurbinWatsonTest <: HypothesisTest

Result of a Durbin-Watson test for first-order autocorrelation in
regression residuals. `method` is `:approx` (large-sample normal
approximation) or `:exact` (Farebrother's AS 153 -- see
[`durbin_watson_pvalue_exact`](@ref)); see [`durbin_watson_test`](@ref)
for which is used by default.
"""
struct DurbinWatsonTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    alternative::Symbol
    method::Symbol
    n::Int
end

function Base.show(io::IO, t::DurbinWatsonTest)
    println(io, "Durbin-Watson test")
    println(io, "  alternative   : ", t.alternative)
    println(io, "  method        : ", t.method)
    println(io, "  n             : ", t.n)
    println(io, "  DW statistic  : ", round(t.statistic, digits=4))
    print(io,   "  p-value       : ", round(t.pvalue, digits=4))
end

"""
    durbin_watson_test(resid, X=nothing; alternative=:greater, method=nothing) -> DurbinWatsonTest

Durbin-Watson test for first-order autocorrelation in regression
residuals `resid`. The statistic itself --

    DW = sum((resid[t] - resid[t-1])^2 for t in 2:n) / sum(resid[t]^2 for t in 1:n)

-- matches Python's `statsmodels.stats.stattools.durbin_watson` exactly
(confirmed directly from its source, not just its docstring: same
formula, no additional correction). `DW` is always in `[0, 4]`; `DW ≈ 2`
indicates no first-order autocorrelation, `DW < 2` positive
autocorrelation, `DW > 2` negative autocorrelation.

**`alternative` defaults to `:greater`**, matching R's `lmtest::dwtest`
default (confirmed directly: `args(dwtest)` shows
`alternative = c("greater", "two.sided", "less")`, `"greater"` first) --
a one-sided test specifically for *positive* autocorrelation, the
classical econometric convention, not a two-sided default.

**`method`**: `nothing` (**default**) picks `:exact` when `X` is
provided and `n < 100`, `:approx` otherwise -- matching
`DescTools::DurbinWatsonTest`'s own real default-switching convention
(`exact=NULL` resolves the same way). Pass `:exact` or `:approx`
explicitly to override.
- `:exact` -- Farebrother's (1980, 1984) Applied Statistics Algorithm
  AS 153 ("Pan's procedure"), the same algorithm and Fortran source
  R's `lmtest::dwtest(exact=TRUE)` uses (read and translated directly
  from `lmtest`'s real CRAN source, not reconstructed from the
  citation -- see [`durbin_watson_pvalue_exact`](@ref)). Requires the
  regression design matrix `X` (the exact null distribution genuinely
  depends on it, not just `resid`). Verified end-to-end against real
  `lmtest::dwtest(exact=TRUE)` output, matching to 6+ significant
  figures (`test/verification/durbinwatson/`).
- `:approx` -- treats `z = (DW - 2) / sqrt(4/n)` as approximately
  standard normal, a large-sample approximation that ignores `X`
  entirely; usable without `X`. Close to but not identical to the exact
  p-value at finite `n` (e.g. `0.1579` here vs R's exact `0.1564` on a
  borderline case).

`resid` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> d = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "durbinwatson", "durbinwatson_ar1.csv"), ','; skipstart=1);

julia> x, y = d[:, 1], d[:, 2];

julia> X = hcat(ones(length(x)), x);

julia> _, resid, _ = TSAnalytics._ols(X, y);

julia> t = durbin_watson_test(resid);

julia> round(t.statistic, digits=4)  # matches real statsmodels.stats.stattools.durbin_watson on this series
0.6363

julia> t.pvalue < 0.001   # strong evidence of positive autocorrelation
true

julia> durbin_watson_test(resid, X; method=:exact).method
:exact
```
"""
function durbin_watson_test(resid, X::Union{Nothing,AbstractMatrix{<:Real}}=nothing;
                             alternative::Symbol=:greater, method::Union{Symbol,Nothing}=nothing)
    alternative in (:greater, :less, :two_sided) ||
        throw(ArgumentError("alternative must be :greater, :less, or :two_sided"))
    meth = method === nothing ? (X !== nothing && (length(tsvalues(resid)) < 100) ? :exact : :approx) : method
    meth in (:approx, :exact) || throw(ArgumentError("method must be :approx, :exact, or nothing"))
    meth == :exact && X === nothing &&
        throw(ArgumentError("durbin_watson_test: method=:exact requires the design matrix X"))

    e = tsvalues(resid)
    n = length(e)
    n >= 2 || throw(ArgumentError("durbin_watson_test: need at least 2 observations"))

    dw = sum(abs2, diff(e, 1)) / sum(abs2, e)

    pval = if meth == :exact
        durbin_watson_pvalue_exact(dw, X; alternative=alternative)
    else
        z = (dw - 2.0) / sqrt(4.0 / n)
        alternative == :greater ? _std_normal_cdf(z) :
        alternative == :less    ? 1 - _std_normal_cdf(z) :
                                   2 * min(_std_normal_cdf(z), 1 - _std_normal_cdf(z))
    end

    return DurbinWatsonTest(dw, pval, alternative, meth, n)
end
