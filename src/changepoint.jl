# ---------------------------------------------------------------------------
# Change points and structural breaks (Stage 12.2,
# handoff/stage 12/docs/stage-12.2-changepoint-breaks-handoff.md).
#
# Two traditions of one problem. Detection -- PELT (Killick, Fearnhead &
# Eckley 2012), binary segmentation, exact optimal partitioning -- searches
# for the segmentation minimising a cost plus a penalty per change. Testing
# and dating -- Bai & Perron (1998, 2003) -- computes the best segmentation
# for every number of breaks and selects afterwards, then puts confidence
# intervals on the dates (Bai 1997) and tests for any break at all (the
# F-statistic path of Andrews 1993; the CUSUM family of fluctuation tests).
#
# Search, cost and penalty are separate, as in ruptures. A segment of
# y[i..j] has its cost from prefix sums in O(1). Breakpoints are the last
# index of the segment before the change (1-based), as changepoint's cpts()
# and strucchange's breakpoints report; ruptures reports the same numbers as
# 0-based exclusive segment ends, with n appended.
# ---------------------------------------------------------------------------

export changepoints, ChangePointResult, cp_penalty
export breakpoints, BreakpointsResult, break_confint, fstats, FStatsResult, efp, EFPResult

# --- costs ------------------------------------------------------------------

# the cost kind is a type parameter, so each search loop compiles for one
# cost with no branch on the kind inside it
struct _CPCost{K}
    S1::Vector{Float64}    # prefix sums of y (or of (y - mu)^2 for :var)
    S2::Vector{Float64}    # prefix sums of y^2
    extra::Bool            # MBIC's per-segment log(length)
end

const _CP_COSTS = (:mean, :var, :meanvar, :poisson, :exponential)
# extra parameters per segment, as changepoint's `diffparam`
_cp_diffparam(kind) = kind === :meanvar ? 2 : 1
# the smallest segment each cost can price (a variance needs two points)
_cp_minlen(kind) = kind in (:var, :meanvar) ? 2 : 1

function _CPCost(y::Vector{Float64}, kind::Symbol, mu, extra)
    if kind === :var
        μ = mu === nothing ? mean(y) : Float64(mu)
        z = (y .- μ) .^ 2
        return _CPCost{:var}(vcat(0.0, cumsum(z)), Float64[], extra)
    end
    return _CPCost{kind}(vcat(0.0, cumsum(y)), vcat(0.0, cumsum(y .^ 2)), extra)
end

# the cost of the segment y[i..j], i <= j
@inline function _cp_cost(c::_CPCost{K}, i::Int, j::Int) where {K}
    n = j - i + 1
    @inbounds s = c.S1[j + 1] - c.S1[i]
    v = if K === :mean
        @inbounds (c.S2[j + 1] - c.S2[i]) - s * s / n
    elseif K === :var
        n * (log(2π) + log(max(s / n, 1e-300)) + 1)
    elseif K === :meanvar
        @inbounds σ2 = ((c.S2[j + 1] - c.S2[i]) - s * s / n) / n
        σ2 <= 0 && (σ2 = 1e-11)
        n * (log(2π) + log(σ2) + 1)
    elseif K === :poisson
        s == 0 ? 0.0 : 2 * s * (log(n) - log(s))
    else    # :exponential
        2 * n * (log(s) - log(n))
    end
    return c.extra ? v + log(n) : v
end

# --- penalties --------------------------------------------------------------

"""
    cp_penalty(name, n; cost=:mean) -> Float64

The penalty per change point for a named criterion, with `p` the number of
extra parameters a change adds (1 for a change in mean, variance, a
Poisson or an exponential rate; 2 for mean and variance together):

| name | penalty |
|---|---|
| `:bic` (= `:sic`) | `(p + 1) log n` |
| `:bic0` | `p log n` |
| `:mbic` | `(p + 2) log n`, and a modified cost (below) |
| `:aic` | `2(p + 1)` |
| `:aic0` | `2p` |
| `:hq` | `2(p + 1) log log n` |
| `:hq0` | `2p log log n` |

`changepoint`'s names and values, confirmed by its `penalty = "Manual"`.
**MBIC is not only a penalty**: it also adds `log(length)` of every
segment to the cost, which [`changepoints`](@ref) does when given
`penalty = :mbic` -- and which breaks PELT's pruning, so PELT refuses it.
"""
function cp_penalty(name::Symbol, n::Integer; cost::Symbol=:mean)
    p = _cp_diffparam(cost)
    n >= 3 || throw(ArgumentError("cp_penalty: n must be at least 3"))
    return name in (:bic, :sic) ? (p + 1) * log(n) :
           name in (:bic0, :sic0) ? p * log(n) :
           name === :mbic ? (p + 2) * log(n) :
           name === :aic ? 2.0 * (p + 1) :
           name === :aic0 ? 2.0 * p :
           name === :hq ? 2.0 * (p + 1) * log(log(n)) :
           name === :hq0 ? 2.0 * p * log(log(n)) :
           throw(ArgumentError("cp_penalty: unknown penalty :$name (use :bic, :bic0, :mbic, " *
                               ":aic, :aic0, :hq, :hq0, or a number)"))
end

# --- searches -----------------------------------------------------------------

# optimal partitioning: F(t) = min over the last change tau of
# F(tau) + C(tau+1..t) + beta, every segment at least m long. With
# `prune`, a candidate is dropped once F(tau) + C(tau+1..t) > F(t) (PELT's
# K = 0 rule), which is exact for any cost where splitting never raises it.
# Ties go to the earliest candidate.
function _cp_optimal(c::_CPCost, n::Int, β::Float64, m::Int, prune::Bool)
    F = fill(Inf, n + 1)          # F[t + 1] = best criterion for y[1..t]
    last = zeros(Int, n + 1)
    F[1] = -β
    cand = Int[0]
    vals = Float64[]
    for t in m:n
        t - m >= m && push!(cand, t - m)          # a new admissible last change
        resize!(vals, length(cand))
        best = Inf
        arg = 0
        @inbounds for (k, τ) in enumerate(cand)
            v = F[τ + 1] + _cp_cost(c, τ + 1, t) + β
            vals[k] = v
            if v < best
                best = v
                arg = τ
            end
        end
        F[t + 1] = best
        last[t + 1] = arg
        if prune
            keep = 1
            @inbounds for k in eachindex(cand)
                if vals[k] - β <= best
                    cand[keep] = cand[k]
                    keep += 1
                end
            end
            resize!(cand, keep - 1)
        end
    end
    cps = Int[]
    t = n
    while t > 0
        τ = last[t + 1]
        τ > 0 && pushfirst!(cps, τ)
        t = τ
    end
    return cps, F[n + 1]
end

# binary segmentation (Scott & Knott 1974): repeatedly make the single split
# that lowers the total cost most, and stop at the first whose gain does not
# exceed the penalty (changepoint accepts a gain equal to it; > is used here,
# as PELT's tie rule) -- greedy, so it can stop before splits PELT would make
function _cp_binseg(c::_CPCost, n::Int, β::Float64, m::Int, Q::Int)
    segs = [(1, n)]
    cps_order = Int[]
    total = [_cp_cost(c, 1, n)]
    for _ in 1:Q
        bestgain, bestseg, bestτ = -Inf, 0, 0
        for (si, (a, b)) in enumerate(segs)
            whole = _cp_cost(c, a, b)
            for τ in (a + m - 1):(b - m)
                g = whole - _cp_cost(c, a, τ) - _cp_cost(c, τ + 1, b)
                if g > bestgain
                    bestgain, bestseg, bestτ = g, si, τ
                end
            end
        end
        (bestseg == 0 || bestgain <= β) && break
        a, b = segs[bestseg]
        splice!(segs, bestseg, [(a, bestτ), (bestτ + 1, b)])
        push!(cps_order, bestτ)
        push!(total, total[end] - bestgain)
    end
    k = length(cps_order)
    return sort(cps_order), total[end] + k * β
end

"""
    ChangePointResult

Detected change points, from [`changepoints`](@ref): `breakpoints` (the
last index of each segment but the last), the `cost`, `penalty` value and
`search` used, the `criterion` reached, and each segment's mean (and
variance, for the variance costs).
"""
struct ChangePointResult
    y::Vector{Float64}
    breakpoints::Vector{Int}
    cost::Symbol
    search::Symbol
    penalty::Float64
    penalty_name::Union{Symbol,Nothing}
    criterion::Float64
    min_segment_length::Int
    segment_means::Vector{Float64}
    segment_vars::Vector{Float64}
    standardized::Bool
end

"""
    changepoints(y; cost=:mean, penalty=:bic, search=:pelt, min_segment_length=2,
                 mu=nothing, standardize=false, max_changepoints=20) -> ChangePointResult

Change points in `y`: the segmentation minimising the total segment cost
plus `penalty` per change.

**Costs** (`changepoint`'s, from prefix sums so each segment is O(1)):
`:mean` -- a change in mean, the segment's residual sum of squares;
`:var` -- a change in variance about the known mean `mu` (default the
sample mean); `:meanvar` -- both, the Gaussian `−2 log L`; `:poisson`
and `:exponential` -- a change in rate.

**The mean cost is not scale-invariant.** It is a sum of squares, with the
noise variance implicitly 1: on a series with variance 4, a BIC penalty
of `2 log n` acts like `log n / 2` in standardised units, and far more
changes are found. That is `changepoint`'s and `ruptures`' convention, kept
so their numbers reproduce; `standardize = true` divides `y` by a robust
noise scale (the median absolute difference between neighbours, over
√2 × 0.6745, which a mean shift barely moves) so the penalty means what it
says.

**Penalty**: a number, or a name from [`cp_penalty`](@ref).

**Searches**: `:pelt` (exact and fast; the default), `:exact` (optimal
partitioning without pruning -- slower, the same answer, the reference
PELT is tested against), and `:binseg` (binary segmentation, greedy and
**not exact**: it can miss a segmentation PELT finds, or find more
changes). `:pelt` refuses `penalty = :mbic`: MBIC's per-segment
`log(length)` makes splitting a segment raise the cost, which voids
PELT's pruning -- `changepoint`'s own default (`MBIC` with `PELT`) returns
a segmentation 2.07 worse on its own criterion than the optimum, without a
warning. Use `search = :exact`.

**Minimum segment length** defaults to 2 (`ruptures`' default; the
smallest that leaves a variance to estimate), raised to 2 for the variance
costs. `changepoint`'s is 1 for a mean change and `strucchange`'s 15% of
`n` (45 points at `n = 300`), and the choice decides the answer near the
ends of a series. There is no `jump`: `ruptures`' default of 5 restricts
change points to multiples of 5 and silently returns `95, 100, 200, 205`
for true changes at 97 and 203.

**Breakpoints** are the last index of the segment before each change --
`changepoint`'s `cpts()` and `strucchange`'s `breakpoints`. `ruptures`
reports the same numbers (0-based exclusive ends) with `n` appended; its
documentation reads them as the start of the next segment, which in
1-based Julia would be one more.
"""
function changepoints(y; cost::Symbol=:mean, penalty=:bic, search::Symbol=:pelt,
                      min_segment_length::Integer=2, mu=nothing, standardize::Bool=false,
                      max_changepoints::Integer=20)
    yv = collect(Float64, tsvalues(y))
    n = length(yv)
    n >= 2 || throw(ArgumentError("changepoints: need at least 2 observations"))
    any(!isfinite, yv) && throw(ArgumentError("changepoints: y contains missing/non-finite values"))
    cost in _CP_COSTS || throw(ArgumentError("changepoints: cost must be one of $(join(_CP_COSTS, ", "))"))
    search in (:pelt, :exact, :binseg) || throw(ArgumentError("changepoints: search must be :pelt, :exact or :binseg"))
    cost === :poisson && any(y -> y < 0 || !isinteger(y), yv) &&
        throw(ArgumentError("changepoints: the Poisson cost needs non-negative counts"))
    cost === :exponential && any(<=(0), yv) &&
        throw(ArgumentError("changepoints: the exponential cost needs positive values"))
    m = max(Int(min_segment_length), _cp_minlen(cost))
    m >= 1 || throw(ArgumentError("changepoints: min_segment_length must be >= 1"))
    n >= m || throw(ArgumentError("changepoints: min_segment_length $m exceeds n = $n"))
    if standardize
        cost === :mean || throw(ArgumentError("changepoints: standardize applies to the mean cost"))
        s = median(abs.(diff(yv))) / (sqrt(2) * 0.6744897501960817)
        s > 0 || throw(ArgumentError("changepoints: the series has no noise to standardise by"))
        yv = yv ./ s
    end
    mbic = penalty === :mbic
    β = penalty isa Symbol ? cp_penalty(penalty, n; cost=cost) : Float64(penalty)
    β >= 0 || throw(ArgumentError("changepoints: the penalty must be >= 0"))
    if mbic && search === :pelt
        throw(ArgumentError(
            "changepoints: PELT cannot use MBIC -- its per-segment log(length) term makes " *
            "splitting a segment raise the cost (a length-100 segment split in half: " *
            "log 50 + log 50 − log 100 = +3.22), which voids PELT's pruning; use search = :exact"))
    end
    c = _CPCost(yv, cost, mu, mbic)
    cps, crit = search === :binseg ? _cp_binseg(c, n, β, m, Int(max_changepoints)) :
                _cp_optimal(c, n, β, m, search === :pelt)
    edges = vcat(0, cps, n)
    means = [mean(view(yv, (edges[i] + 1):edges[i + 1])) for i in 1:(length(edges) - 1)]
    vars = [edges[i + 1] - edges[i] > 1 ? var(view(yv, (edges[i] + 1):edges[i + 1]); corrected=false) : 0.0
            for i in 1:(length(edges) - 1)]
    return ChangePointResult(yv, cps, cost, search, β, penalty isa Symbol ? penalty : nothing, crit,
                             m, means, vars, standardize)
end

function Base.show(io::IO, r::ChangePointResult)
    k = length(r.breakpoints)
    println(io, "Change points: ", k == 0 ? "none" : join(r.breakpoints, ", "), "  (", r.cost,
            " cost, ", r.search, ", penalty ", r.penalty_name === nothing ? "" : "$(r.penalty_name) = ",
            round(r.penalty; sigdigits=6), ", minimum segment ", r.min_segment_length, ")")
    print(io, "  segment means: ", join(round.(r.segment_means; sigdigits=5), ", "))
    r.standardized && print(io, "\n  (y standardised by a robust noise scale)")
end

# --- Bai & Perron ------------------------------------------------------------------

# RSS of the OLS fit on rows i..j, from cumulated cross-products
struct _BPCross
    XX::Vector{Matrix{Float64}}
    Xy::Vector{Vector{Float64}}
    yy::Vector{Float64}
end
function _BPCross(X::Matrix{Float64}, y::Vector{Float64})
    n, k = size(X)
    XX = Vector{Matrix{Float64}}(undef, n + 1); Xy = Vector{Vector{Float64}}(undef, n + 1)
    yy = zeros(n + 1)
    XX[1] = zeros(k, k); Xy[1] = zeros(k)
    for t in 1:n
        x = view(X, t, :)
        XX[t + 1] = XX[t] .+ x * x'
        Xy[t + 1] = Xy[t] .+ x .* y[t]
        yy[t + 1] = yy[t] + y[t]^2
    end
    return _BPCross(XX, Xy, yy)
end
function _bp_rss(c::_BPCross, i::Int, j::Int)
    A = c.XX[j + 1] .- c.XX[i]
    b = c.Xy[j + 1] .- c.Xy[i]
    return (c.yy[j + 1] - c.yy[i]) - dot(b, A \ b)
end

"""
    BreakpointsResult

The Bai–Perron breakpoint path, from [`breakpoints`](@ref): for every
number of breaks `m = 0 … max_breaks`, the optimal break dates
(`dates[m + 1]`), the residual sum of squares `rss` and `bic`; the
selected `breaks` (minimum BIC), the minimum segment length `h`, and the
regression design.
"""
struct BreakpointsResult
    y::Vector{Float64}
    X::Matrix{Float64}
    h::Int
    max_breaks::Int
    rss::Vector{Float64}
    bic::Vector{Float64}
    dates::Vector{Vector{Int}}
    breaks::Vector{Int}
end

"""
    breakpoints(y; X=nothing, h=0.15, max_breaks=nothing) -> BreakpointsResult

Bai & Perron's (2003) dynamic programme: for every number of breaks `m`
up to `max_breaks`, the break dates minimising the total residual sum of
squares of a regression of `y` on `X` (default a constant: a change in
mean) fitted separately in each segment, every segment at least `h` long
(`h < 1` is a fraction of `n`, floored; default 15%, as `strucchange`).
Nothing is decided during the search; the number of breaks is chosen
afterwards by BIC, `n log(RSS/n) + n(log 2π + 1) + log(n)(k + 1)(m + 1)`
for `k` regressors, and the whole path is returned. `max_breaks` defaults
to `⌈n/h⌉ − 2`. This is `strucchange`'s `breakpoints`, and its RSS and BIC
table is reproduced to rounding.

Note the default `h`: at `n = 300` no segment can be shorter than 45
points, so changes near either end, or closer than 45 apart, cannot be
found at all -- the reason `strucchange` sees nothing in a series that
[`changepoints`](@ref) splits at 3 and 297.
"""
function breakpoints(y; X=nothing, h::Real=0.15, max_breaks::Union{Nothing,Integer}=nothing)
    yv = collect(Float64, tsvalues(y))
    n = length(yv)
    any(!isfinite, yv) && throw(ArgumentError("breakpoints: y contains missing/non-finite values"))
    Xm = X === nothing ? ones(n, 1) : (X isa AbstractVector ? reshape(Float64.(X), :, 1) : Matrix{Float64}(X))
    size(Xm, 1) == n || throw(ArgumentError("breakpoints: X must have n rows"))
    k = size(Xm, 2)
    hh = h < 1 ? floor(Int, h * n) : Int(h)
    hh > k || throw(ArgumentError("breakpoints: the minimum segment length h = $hh must exceed the $k regressors"))
    2hh <= n || throw(ArgumentError("breakpoints: h = $hh leaves room for no break in n = $n"))
    M = max_breaks === nothing ? ceil(Int, n / hh) - 2 : Int(max_breaks)
    M = min(M, n ÷ hh - 1)
    M >= 0 || throw(ArgumentError("breakpoints: max_breaks must be >= 0"))
    cross = _BPCross(Xm, yv)
    # the triangular RSS table, segments of at least hh points, on threads
    R = fill(Inf, n, n)
    Threads.@threads for i in 1:(n - hh + 1)
        for j in (i + hh - 1):n
            R[i, j] = _bp_rss(cross, i, j)
        end
    end
    # D[m+1][j]: best RSS of y[1..j] with m breaks; P: the last break
    D = [fill(Inf, n) for _ in 0:M]
    P = [zeros(Int, n) for _ in 0:M]
    for j in hh:n
        D[1][j] = R[1, j]
    end
    for m in 1:M
        for j in ((m + 1) * hh):n
            best, arg = Inf, 0
            for τ in (m * hh):(j - hh)
                v = D[m][τ] + R[τ + 1, j]
                if v < best
                    best, arg = v, τ
                end
            end
            D[m + 1][j], P[m + 1][j] = best, arg
        end
    end
    rss = [D[m + 1][n] for m in 0:M]
    dates = Vector{Int}[]
    for m in 0:M
        bp = Int[]
        j = n
        for mm in m:-1:1
            τ = P[mm + 1][j]
            pushfirst!(bp, τ)
            j = τ
        end
        push!(dates, bp)
    end
    bic = [n * (log(2π) + log(rss[m + 1] / n) + 1) + log(n) * (k + 1) * (m + 1) for m in 0:M]
    sel = argmin(bic) - 1
    return BreakpointsResult(yv, Xm, hh, M, rss, bic, dates, dates[sel + 1])
end

function Base.show(io::IO, r::BreakpointsResult)
    println(io, "Bai-Perron breakpoints, h = ", r.h, ", up to ", r.max_breaks, " breaks; BIC selects ",
            isempty(r.breaks) ? "none" : "$(length(r.breaks)) at " * join(r.breaks, ", "))
    for m in 0:r.max_breaks
        println(io, "  m = ", m, "  RSS ", lpad(round(r.rss[m + 1]; digits=4), 12), "  BIC ",
                lpad(round(r.bic[m + 1]; digits=4), 12), m == 0 ? "" : "   " * join(r.dates[m + 1], ", "))
    end
end

# --- Bai (1997) break-date confidence intervals --------------------------------

# the distribution function of the location of the maximum of
# V(s) = W1(-s) - |s|/2 (s <= 0), sqrt(phi) W2(s) - xi |s|/2 (s > 0)
# -- Bai (1997), Appendix B; phi = xi (phi2/phi1)^2
function _bai_cdf(x::Float64, xi::Float64, phi1::Float64, phi2::Float64)
    φ = xi * (phi2 / phi1)^2
    lnΦ(z) = log(_std_normal_cdf(z))
    if x < 0
        a = abs(x)
        f = xi / φ
        return -exp(log(a) / 2 - a / 8 - log(2π) / 2) -
               (φ / xi * (φ + 2xi) / (φ + xi)) * exp(f * (1 + f) * a / 2 + lnΦ(-(0.5 + f) * sqrt(a))) +
               exp(log(a / 2 - 2 + (φ + 2xi)^2 / ((φ + xi) * xi)) + lnΦ(-sqrt(a) / 2))
    else
        a = abs(x)
        f = xi^2 / φ
        return 1 + sqrt(f) * exp(log(a) / 2 - f * a / 8 - log(2π) / 2) +
               (xi / φ * (2φ + xi) / (φ + xi)) * exp((φ + xi) * a / 2 + lnΦ(-(φ + xi / 2) / sqrt(φ) * sqrt(a))) -
               exp(log((2φ + xi)^2 / ((φ + xi) * φ) - 2 + f * a / 2) + lnΦ(-sqrt(f) * sqrt(a) / 2))
    end
end

function _bai_quantile(p, xi, phi1, phi2, side)
    # bracket by steps of 1000, then bisect to well below a unit
    lo, hi = 0.0, 0.0
    if side > 0
        while _bai_cdf(hi, xi, phi1, phi2) < p
            hi += 1000
        end
    else
        while _bai_cdf(lo, xi, phi1, phi2) > p
            lo -= 1000
        end
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        (_bai_cdf(mid, xi, phi1, phi2) < p) ? (lo = mid) : (hi = mid)
        hi - lo < 1e-10 && break
    end
    return (lo + hi) / 2
end

"""
    break_confint(r::BreakpointsResult; breaks=nothing, level=0.95) -> Matrix{Int}

Confidence intervals for the break dates (Bai 1997): one row per break,
`[lower, date, upper]`. The distribution of the estimated date is that of
the location of the maximum of a two-sided Brownian motion with drift,
whose distribution function Bai gives in closed form -- no table. Segments
on either side of a break may differ in regressor moments and in error
variance (`strucchange`'s defaults `het.reg = het.err = TRUE`), so the
interval need not be symmetric. `breaks` picks the number of breaks (default
the BIC choice). Refused when there are no breaks to date.
"""
function break_confint(r::BreakpointsResult; breaks::Union{Nothing,Integer}=nothing, level::Real=0.95)
    0 < level < 1 || throw(ArgumentError("break_confint: level must be in (0, 1)"))
    bp = breaks === nothing ? r.breaks : r.dates[Int(breaks) + 1]
    isempty(bp) && throw(ArgumentError("break_confint: no breaks to date (BIC chose none)"))
    a2 = (1 - level) / 2
    X, y = r.X, r.y
    n = length(y)
    edges = vcat(0, bp, n)
    seg(i) = (edges[i] + 1):edges[i + 1]
    fit(rows) = (b = X[rows, :] \ y[rows]; e = y[rows] .- X[rows, :] * b;
                 (b, sum(abs2, e) / length(rows), X[rows, :]' * X[rows, :] / length(rows)))
    out = zeros(Int, length(bp), 3)
    b2, s2, Q2 = fit(seg(1))
    for i in 2:length(edges) - 1
        b1, s1, Q1 = b2, s2, Q2
        b2, s2, Q2 = fit(seg(i))
        δ = b2 .- b1
        Qp1, Qp2 = dot(δ, Q1 * δ), dot(δ, Q2 * δ)
        xi = Qp2 / Qp1
        phi1, phi2 = sqrt(s1), sqrt(s2)
        up = _bai_quantile(1 - a2, xi, phi1, phi2, 1) * phi1^2 / Qp1
        lo = _bai_quantile(a2, xi, phi1, phi2, -1) * phi1^2 / Qp1
        d = bp[i - 1]
        out[i - 1, :] = [d - ceil(Int, up), d, d - floor(Int, lo)]
    end
    return out
end

# --- the F-statistic path -------------------------------------------------------------

"""
    FStatsResult

The F-statistic path of [`fstats`](@ref): `stats[j]` tests a single break
after observation `dates[j]`; `supF`, `aveF` and `expF` summarise it, and
`argmax` is the most likely break date.
"""
struct FStatsResult
    dates::Vector{Int}
    stats::Vector{Float64}
    supF::Float64
    aveF::Float64
    expF::Float64
    argmax::Int
    k::Int
    nobs::Int
end

"""
    fstats(y; X=nothing, from=0.15, to=nothing) -> FStatsResult

For every candidate break date `τ` from `from` to `to` (fractions of `n`
are floored; `to` defaults to `n − from`), the statistic for a break in
all coefficients of the regression of `y` on `X` (default a constant) at
`τ`: `(RSS_whole − RSS_split) / (RSS_split / (n − 2k))`. With `k`
coefficients this is **`k` times the classical Chow F** -- the Wald form
`strucchange` reports; the two coincide for a change in mean (`k = 1`).
The summaries are Andrews' supF, and Andrews & Ploberger's aveF and
`expF = log(mean(exp(F/2)))`.

**No p-values.** Their null distribution is non-standard, and the
response-surface tables `strucchange` uses for it (Hansen 1997) are not
available under a compatible licence. The statistics are exact.
"""
function fstats(y; X=nothing, from::Real=0.15, to::Union{Nothing,Real}=nothing)
    yv = collect(Float64, tsvalues(y))
    n = length(yv)
    Xm = X === nothing ? ones(n, 1) : (X isa AbstractVector ? reshape(Float64.(X), :, 1) : Matrix{Float64}(X))
    k = size(Xm, 2)
    f = from < 1 ? floor(Int, from * n) : Int(from)
    t = to === nothing ? n - f : (to < 1 ? floor(Int, to * n) : Int(to))
    f = max(f, k + 1)
    t = min(t, n - k - 1)
    f <= t || throw(ArgumentError("fstats: 'from' ($f) is past 'to' ($t)"))
    cross = _BPCross(Xm, yv)
    whole = _bp_rss(cross, 1, n)
    dates = collect(f:t)
    stats = similar(dates, Float64)
    for (j, τ) in enumerate(dates)
        u = _bp_rss(cross, 1, τ) + _bp_rss(cross, τ + 1, n)
        stats[j] = (whole - u) / (u / (n - 2k))
    end
    ef = log(mean(exp.(stats ./ 2)))
    return FStatsResult(dates, stats, maximum(stats), mean(stats), ef, dates[argmax(stats)], k, n)
end

function Base.show(io::IO, r::FStatsResult)
    println(io, "F statistics for a single break, dates ", first(r.dates), ":", last(r.dates), " (k = ", r.k, ")")
    print(io, "  supF = ", round(r.supF; sigdigits=8), " at ", r.argmax, ", aveF = ", round(r.aveF; sigdigits=8),
          ", expF = ", round(r.expF; sigdigits=8),
          "\n  (no p-values: the null distribution is non-standard and its tables are not available)")
end

# --- empirical fluctuation processes ----------------------------------------------

"""
    EFPResult

An empirical fluctuation process, from [`efp`](@ref): the `process`
(time × dimensions, starting at 0), its `type`, the test `statistic`,
`pvalue`, and whether that p-value is only a bound (`pvalue_bound`).
"""
struct EFPResult
    type::Symbol
    process::Matrix{Float64}
    statistic::Float64
    pvalue::Float64
    pvalue_bound::Bool
    limit::Symbol
    nobs::Int
end

# the symmetric square root of a positive definite matrix
_sqrtm_sym(A) = (E = eigen(Symmetric(A)); E.vectors * Diagonal(sqrt.(E.values)) * E.vectors')

# the recursive residuals of y on X
function _recursive_residuals(X, y)
    n, k = size(X)
    w = Float64[]
    for t in (k + 1):n
        A = X[1:(t - 1), :]
        b = A \ y[1:(t - 1)]
        x = X[t, :]
        push!(w, (y[t] - dot(x, b)) / sqrt(1 + dot(x, (A' * A) \ x)))
    end
    return w
end

# upper tails, computed as tails so that a tiny p-value is not lost to
# cancellation (strucchange's 1 - (1 - p)^k returns exactly 0 there: D27)
function _p_bridge_max(x, k)
    x < 0.1 && return (1.0, false)
    s = 0.0
    for i in 1:100
        s += (-1)^(i + 1) * exp(-2 * i^2 * x^2)
    end
    tail = 2s                               # P(sup |bridge| > x), one dimension
    return (-expm1(k * log1p(-tail)), false)
end
function _p_motion_max(x, k)
    # Brownian motion, boundary x(1 + 2t): Ploberger & Kramer's closed form
    Φ(z) = _std_normal_cdf(z)
    Φc(z) = _std_normal_cdf(-z)
    tail = 2 * (Φc(3x) + exp(-4x^2) * (Φ(x) + Φ(5x) - 1) - exp(-16x^2) * Φc(x))
    tail = clamp(tail, 0.0, 1.0)
    return (-expm1(k * log1p(-tail)), false)
end

"""
    efp(y; X=nothing, type=:ols_cusum) -> EFPResult

An empirical fluctuation process for the stability of the regression of
`y` on `X` (default a constant), and its test (Kuan & Hornik 1995; Zeileis
et al. 2002). Under stability each process behaves like a Brownian bridge
or motion; a break shows as an excursion.

- `:ols_cusum` -- cumulated OLS residuals, scaled by their standard
  deviation and `√n` (Ploberger & Krämer 1992): a Brownian bridge.
- `:rec_cusum` -- cumulated recursive residuals (Brown, Durbin & Evans
  1975): a Brownian motion, judged against boundaries that widen as
  `1 + 2t`.
- `:score_cusum` -- cumulated scores of the Gaussian likelihood, for each
  coefficient **and the variance**, decorrelated: a `k + 1`-dimensional
  Brownian bridge. The only one of these that sees a change in variance.
- `:re` -- the fluctuation of the recursive estimates about the full-sample
  estimate: a `k`-dimensional Brownian bridge.

The statistic is the largest absolute value of the process (divided by
the boundary for `:rec_cusum`); its p-value comes from the limiting
process in closed form -- the Kolmogorov series for a bridge, Ploberger &
Krämer's formula for the motion -- so no table is needed. The tails are
computed directly, so a p-value of `1e-20` is reported as such rather
than as exactly zero (`strucchange` prints 0). The MOSUM processes, whose
p-values need tables, are not offered.
"""
function efp(y; X=nothing, type::Symbol=:ols_cusum)
    yv = collect(Float64, tsvalues(y))
    n = length(yv)
    Xm = X === nothing ? ones(n, 1) : (X isa AbstractVector ? reshape(Float64.(X), :, 1) : Matrix{Float64}(X))
    k = size(Xm, 2)
    n > k + 2 || throw(ArgumentError("efp: too few observations"))
    b = Xm \ yv
    e = yv .- Xm * b
    if type === :ols_cusum
        σ = sqrt(sum(abs2, e) / (n - k))
        proc = reshape(vcat(0.0, cumsum(e)) ./ (σ * sqrt(n)), :, 1)
        stat = maximum(abs, view(proc, 2:size(proc, 1), :))
        p, bound = _p_bridge_max(stat, 1)
        lim = :bridge
    elseif type === :rec_cusum
        w = _recursive_residuals(Xm, yv)
        σ = std(w)
        proc = reshape(vcat(0.0, cumsum(w)) ./ (σ * sqrt(n - k)), :, 1)
        m = length(w)
        stat = maximum(abs(proc[j + 1]) / (1 + 2 * j / m) for j in 1:m)
        p, bound = _p_motion_max(stat, 1)
        lim = :motion
    elseif type === :score_cusum
        σ2 = sum(abs2, e) / n
        S = hcat(Xm .* e, e .^ 2 .- σ2) ./ sqrt(n)
        Q12 = _sqrtm_sym(S' * S)
        C = vcat(zeros(1, k + 1), cumsum(S; dims=1))
        proc = C * inv(Q12)
        stat = maximum(abs, view(proc, 2:size(proc, 1), :))
        p, bound = _p_bridge_max(stat, k + 1)
        lim = :bridge
    elseif type === :re
        σ = sqrt(sum(abs2, e) / (n - k))
        proc = zeros(n - k + 1, k)
        for i in k:(n - 1)
            bi = Xm[1:i, :] \ yv[1:i]
            Qi = _sqrtm_sym(Xm[1:i, :]' * Xm[1:i, :]) ./ (σ * sqrt(i))
            proc[i - k + 2, :] = Qi * (bi .- b) .* i ./ sqrt(n)
        end
        stat = maximum(abs, view(proc, 2:size(proc, 1), :))
        p, bound = _p_bridge_max(stat, k)
        lim = :bridge
    else
        throw(ArgumentError("efp: type must be :ols_cusum, :rec_cusum, :score_cusum or :re " *
                            "(the MOSUM processes need tables, and are not offered)"))
    end
    return EFPResult(type, proc, stat, p, bound, lim, n)
end

statistic(r::EFPResult) = r.statistic
pvalue(r::EFPResult) = r.pvalue

function Base.show(io::IO, r::EFPResult)
    nm = Dict(:ols_cusum => "OLS-based CUSUM", :rec_cusum => "Recursive CUSUM",
              :score_cusum => "Score-based CUSUM", :re => "Recursive estimates (RE)")[r.type]
    print(io, nm, " test: statistic = ", round(r.statistic; sigdigits=8), ", p = ",
          r.pvalue < 1e-4 ? string(round(r.pvalue; sigdigits=4)) : string(round(r.pvalue; digits=6)),
          " (limit: Brownian ", r.limit, ")")
end
