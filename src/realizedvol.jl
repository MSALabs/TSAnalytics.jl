export realized_variance, bipower_variation, jump_test, realized_semivariance, realized_measures, JumpTest

"_MU_4_3_INV3 = mu_{4/3}^{-3}, mu_{4/3} = E|Z|^{4/3} = 2^{2/3}*Gamma(7/6)/Gamma(1/2)
for standard normal Z -- the tripower-quarticity scaling constant, confirmed
directly from `highfrequency::rTPQuar`'s own source (not transcribed from a
paper): `N*(N/(N-2))*((gamma(0.5)/(2^(2/3)*gamma(7/6)))^3)*sum(...)`. Computed
via this file's own `_loggamma` (already used for the chi-squared machinery in
diagnostics.jl) rather than a fresh `SpecialFunctions.jl` dependency for one
constant."
const _MU_4_3_INV3 = exp(3 * (_loggamma(0.5) - (2 / 3) * log(2) - _loggamma(7 / 6)))

"""
    realized_variance(r) -> Float64

Realized variance (Andersen & Bollerslev 1998): `sum(r_i^2)` over one
period's (e.g. one trading day's) intraday return vector `r`. The
foundational nonparametric volatility estimator this whole stage builds
on -- consistently estimates *total* quadratic variation, including any
jump contribution (unlike [`bipower_variation`](@ref), which is
jump-robust).

`r` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics

julia> realized_variance([0.01, -0.02, 0.005])
0.000525
```
"""
function realized_variance(r)
    x = tsvalues(r)
    return sum(abs2, x)
end

"""
    bipower_variation(r) -> Float64

Bipower variation (Barndorff-Nielsen & Shephard 2004):
`(π/2) * sum(|r[i]| * |r[i+1]| for i in 1:n-1)`. Jump-robust -- unlike
[`realized_variance`](@ref), consistently estimates only the continuous
(diffusive) component of variance, since a single large jump return
only ever enters *one* product term (paired with its ordinary
neighbor), not squared into the sum the way it dominates `RV`.

Confirmed to `π/2` (`= mu_1^{-2}`, `mu_1 = E|Z| = sqrt(2/π)` for
standard normal `Z`) with *no* finite-sample `N/(N-2)` correction
factor by reading `highfrequency::RBPVar`'s actual executed source
directly (`highfrequency` turned out to be installable in this
sandbox, unlike most R packages referenced earlier in this project --
some published presentations of this formula do include an `N/(N-2)`
correction, but the real, executable reference implementation
`rBPCov` (for a univariate series) actually calls doesn't).

`r` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics

julia> round(bipower_variation([0.01, -0.02, 0.005]), digits=6)
0.000471
```
"""
function bipower_variation(r)
    x = tsvalues(r)
    n = length(x)
    n >= 2 || throw(ArgumentError("bipower_variation: need at least 2 observations"))
    s = 0.0
    for i in 1:(n - 1)
        s += abs(x[i]) * abs(x[i + 1])
    end
    return (pi / 2) * s
end

"_tripower_quarticity(x) -- jump-robust estimator of integrated
quarticity (Barndorff-Nielsen & Shephard 2004/2006), needed only to
standardize [`jump_test`](@ref)'s asymptotic variance, not exposed as
its own public function (the handoff's own API section 3 doesn't list
it as one either). Confirmed directly from `highfrequency::rTPQuar`'s
actual source: `N*(N/(N-2))*mu_{4/3}^{-3}*sum(|x[i]|^{4/3}*|x[i+1]|^{4/3}*|x[i+2]|^{4/3})`."
function _tripower_quarticity(x::AbstractVector)
    n = length(x)
    n >= 3 || throw(ArgumentError("_tripower_quarticity: need at least 3 observations"))
    s = 0.0
    for i in 1:(n - 2)
        s += (abs(x[i]) * abs(x[i + 1]) * abs(x[i + 2]))^(4 / 3)
    end
    return n * (n / (n - 2)) * _MU_4_3_INV3 * s
end

"""
    JumpTest <: HypothesisTest

Result of [`jump_test`](@ref): the Barndorff-Nielsen & Shephard (2006)
test for jumps in one period's intraday return path. `jump_variance =
max(RV - BV, 0)` -- clipped, matching the convention used throughout
the jump-variation literature (a raw negative `RV - BV` is finite-sample
noise, not a real negative jump contribution).
"""
struct JumpTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    rv::Float64
    bv::Float64
    jump_variance::Float64
end

function Base.show(io::IO, t::JumpTest)
    println(io, "Barndorff-Nielsen & Shephard jump test")
    println(io, "  RV             : ", t.rv)
    println(io, "  BV             : ", t.bv)
    println(io, "  jump variance  : ", t.jump_variance)
    println(io, "  z statistic    : ", round(t.statistic, digits=4))
    print(io,   "  p-value        : ", round(t.pvalue, digits=4))
end

"""
    jump_test(r) -> JumpTest

Barndorff-Nielsen & Shephard (2006) test for jumps in one period's
intraday return path `r`: under the null of no jumps,
[`realized_variance`](@ref) and [`bipower_variation`](@ref) should be
statistically indistinguishable (`BV` consistently estimates only the
continuous component; `RV` estimates continuous *plus* jump variation).
The "ratio" form of the test statistic (matching
`highfrequency::BNSjumpTest(type="ratio", max=TRUE)`, confirmed
directly from its own source, not transcribed from a paper):

    Z = sqrt(n) * (1 - BV/RV) / sqrt((θ-2) * max(1, TQ/BV²))

`θ = π²/4 + π - 3` (the asymptotic variance constant specific to the
`BV` estimator, confirmed directly from `highfrequency::tt("BV")`'s
source), `TQ` the jump-robust tripower quarticity. **`max(1, TQ/BV²)`,
not `TQ/BV²` alone** -- `highfrequency`'s own *default* is actually
`max=FALSE` (`TQ/BV²` alone), but this project uses the `max=TRUE`
variant deliberately: it's the more robust, more commonly cited form in
the jump-test literature (guards against `TQ/BV²` dipping below 1 in
finite samples, which would otherwise understate the denominator and
inflate false positives) -- a documented, deliberate divergence from the
reference's own default, not an oversight.

Verified by full statistical calibration across 1000 simulated trading
days (500 no-jump, 500 with a real injected jump), not just formula
transcription -- see `test/verification/realizedvol/` for the Monte
Carlo run confirming correct test size under the null (~5% false-positive
rate at the conventional `|Z|>1.96` threshold) and high power against a
real jump (100% detection at the same threshold in the verified run).

`r` accepts anything [`tsvalues`](@ref) does; needs at least 3
observations (`TQ`'s own minimum).

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); r = randn(78) .* (0.15 / sqrt(252) / sqrt(78));

julia> jump_test(r).jump_variance >= 0
true
```
"""
function jump_test(r)
    x = tsvalues(r)
    n = length(x)
    n >= 3 || throw(ArgumentError("jump_test: need at least 3 observations"))
    rv = realized_variance(x)
    bv = bipower_variation(x)
    tq = _tripower_quarticity(x)
    theta = pi^2 / 4 + pi - 3
    product = max(1.0, tq / bv^2)
    z = sqrt(n) * (1 - bv / rv) / sqrt((theta - 2) * product)
    pval = 2 * _std_normal_cdf(-abs(z))
    jump_var = max(rv - bv, 0.0)
    return JumpTest(z, pval, rv, bv, jump_var)
end

"""
    realized_semivariance(r) -> (positive=Float64, negative=Float64)

Realized semivariance (Barndorff-Nielsen, Kinnebrock & Shephard 2010):
splits [`realized_variance`](@ref) into contributions from positive-
and negative-return intervals, `positive + negative == realized_variance(r)`
exactly. Useful for asymmetric risk measures (e.g. "downside realized
volatility") directly, not derivable from `RV`/`BV` alone. Confirmed
directly from `highfrequency::rSVar`'s own source: `sum((r[r.>0]).^2)`/
`sum((r[r.<0]).^2)`, zero contribution from exact-zero returns.

`r` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics

julia> sv = realized_semivariance([0.01, -0.02, 0.005]);

julia> isapprox(sv.positive + sv.negative, realized_variance([0.01, -0.02, 0.005]); atol=1e-18)
true
```
"""
function realized_semivariance(r)
    x = tsvalues(r)
    pos = 0.0
    neg = 0.0
    for xi in x
        xi > 0 && (pos += xi^2)
        xi < 0 && (neg += xi^2)
    end
    return (positive=pos, negative=neg)
end

"""
    realized_measures(periods::AbstractVector{<:AbstractVector}; parallel::Bool=true)
        -> Vector{<:NamedTuple}

Computes `(rv, bv, jump)` for each period in `periods` independently --
realized measures are inherently computed per-period (typically per
trading day) and applied repeatedly across however many periods a
dataset spans, a cleaner, more natural parallel structure than any
other GARCH-family stage's: years of daily data means thousands of
completely independent per-day calculations, not an add-on parallelism
capability.

`parallel=true` by default; `Threads.@threads` only actually engages
when `Threads.nthreads() > 1` and `length(periods) >= 4` (below which
thread-spawning overhead isn't worth it), the same guarded pattern as
every other parallel design in this project.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); periods = [randn(78) .* 0.01 for _ in 1:5];

julia> results = realized_measures(periods);

julia> length(results)
5

julia> results[1].rv > 0
true
```
"""
function realized_measures(periods::AbstractVector; parallel::Bool=true)
    n = length(periods)
    results = Vector{NamedTuple{(:rv, :bv, :jump),Tuple{Float64,Float64,JumpTest}}}(undef, n)
    use_threads = parallel && Threads.nthreads() > 1 && n >= 4
    if use_threads
        Threads.@threads for i in 1:n
            r = periods[i]
            results[i] = (rv=realized_variance(r), bv=bipower_variation(r), jump=jump_test(r))
        end
    else
        for i in 1:n
            r = periods[i]
            results[i] = (rv=realized_variance(r), bv=bipower_variation(r), jump=jump_test(r))
        end
    end
    return results
end
