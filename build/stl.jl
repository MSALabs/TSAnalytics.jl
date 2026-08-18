export stl_decompose, STLDecomposition

"""
    STLDecomposition

Result of [`stl_decompose`](@ref). `weights` is the final robustness
weight vector (Cleveland et al. 1990's `rw`) -- all `1.0` when the
effective outer count is `0` (the `robust=false` default), otherwise one
weight per observation in `[0, 1]`, matching R's `stl()`'s `\$weights` and
Python's `STL.fit()`'s `.weights`: low values mark points the outer loop
treated as outliers and downweighted in the cycle-subseries/trend Loess
fits, `0` meaning fully excluded.
"""
struct STLDecomposition
    observed::Vector{Float64}
    trend::Vector{Float64}
    seasonal::Vector{Float64}
    resid::Vector{Float64}
    period::Int
    weights::Vector{Float64}
end

"_nextodd(x) -- smallest odd integer >= x, matching R's `stl()`'s own `nextodd` helper."
_nextodd(x::Real) = (m = ceil(Int, x); isodd(m) ? m : m + 1)

"""
_stl_est1(x, y, x0, k, degree, robweights) -- STL's OWN internal
single-point local-regression estimator, evaluated at `x0`. **Not the
same algorithm as `_lowess_fit1`**, despite both being "Loess" -- this is
Cleveland et al.'s original NETLIB Fortran `stlest`/Python `_stl.pyx`'s
`_est` closed-form routine, verified directly from `_stl.pyx` source, not
assumed to generalize from the already-validated general-purpose
`lowess()` this package's `_lowess` matches. The two agree numerically
whenever a neighborhood has comfortably more points than `degree+1` (why
`_lowess_fit1` originally seemed fine for every Stage 3.2 test, which all
used periods with 4-6 points per cycle-subseries) -- but they are
genuinely different formulas, and diverge for `mstl_decompose`'s large
periods relative to series length (a period like `168` on `n=500` gives
cycle-subseries as short as 2-3 points): `_lowess_fit1` requires strictly
more active points than `degree+1` and falls back to the single nearest
point otherwise; `_est` never falls back -- it always returns at least a
tricube-weighted mean (needing only one point with positive weight), and
separately decides whether to *also* apply a linear correction based on
whether the neighborhood's weighted position-variance is large enough to
support one (`sqrt(c) > 0.001*(n-1)`), not on a raw point-count threshold.

Same neighbor selection and width-extension as `_lowess_fit1` (`k > n`
extends the tricube width by `(k-n) \\div 2`, additive/integer-division,
per Stage 3.2's own verified fix) -- confirmed algebraically identical to
`_est`'s own `h = max(xs-nleft, nright-xs)` construction for the
sequential-integer `x` STL always uses internally (a contiguous index
range's two endpoint distances bound the same maximum as a k-nearest
search over integers). What differs is everything *after* neighbor
selection: closed-form weighted mean (`ys = sum(w*y)`) plus, only when
`degree > 0` and the neighborhood isn't x-degenerate, a linear tilt
(`w *= b*(x-xbar) + 1` where `b = (x0-xbar)/c`) -- rather than routing
through a general weighted-OLS solve.
"""
function _stl_est1(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                    x0::Real, k::Integer, degree::Integer,
                    robweights::AbstractVector{<:Real})
    n = length(x)
    keff = min(k, n)
    d = abs.(x .- x0)
    nb = partialsortperm(d, 1:keff)
    h = maximum(view(d, nb))
    k > n && (h += (k - n) ÷ 2)

    h9 = 0.999h
    h1 = 0.001h
    w = zeros(keff)
    a = 0.0
    for (j, idx) in enumerate(nb)
        r = d[idx]
        r > h9 && continue
        w[j] = r <= h1 ? 1.0 : _tricube(r / h)  # r > h1 >= 0 implies h > 0 here, so r/h is well-defined
        w[j] *= robweights[idx]
        a += w[j]
    end
    a <= 0 && return NaN

    w ./= a
    if h > 0 && degree > 0
        xn = view(x, nb)
        xbar = sum(w[j] * xn[j] for j in 1:keff)
        c = sum(w[j] * (xn[j] - xbar)^2 for j in 1:keff)
        rng = n - 1.0
        if sqrt(c) > 0.001 * rng
            b = (x0 - xbar) / c
            for j in 1:keff
                w[j] *= b * (xn[j] - xbar) + 1.0
            end
        end
    end

    return sum(w[j] * y[nb[j]] for j in 1:keff)
end

"""
_stl_smooth(x, y; k, degree, xout=x, weights=ones(length(x))) -- thin
driver over [`_stl_est1`](@ref), evaluating it at every point in `xout`.
STL's internal analogue of `_lowess` (which it deliberately does NOT
call -- see `_stl_est1`'s docstring): always a single pass (STL's own
robustness reweighting is the *outer* loop in `stl_decompose`, handled by
recomputing `weights` between calls, not by any iteration in here).
"""
function _stl_smooth(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                      k::Integer, degree::Integer,
                      xout::AbstractVector{<:Real}=x,
                      weights::AbstractVector{<:Real}=ones(length(x)))
    return [_stl_est1(x, y, x0, k, degree, weights) for x0 in xout]
end

"""
_stl_lowpass(C, period, l_window, l_degree) -- the STL low-pass step:
a length-`period` moving average applied twice, then a length-3 moving
average once (Cleveland et al. 1990's specific three-pass construction,
not a single filter), each pass implemented via [`convolution_filter`](@ref)
and trimmed of its edge `NaN`s (equivalent to -- and reusing the already
-validated primitive for -- Cleveland's own "shrinking" moving average),
then Loess-smoothed with `l_window`/`l_degree`. Reduces a length
`n + 2*period` input to exactly length `n` (`2*(period-1) + (3-1) =
2*period` positions consumed by the three shrinking passes).
"""
function _stl_lowpass(C::AbstractVector{<:Real}, period::Integer, l_window::Integer, l_degree::Integer)
    ma1 = filter(!isnan, convolution_filter(C, fill(1/period, period); sides=2))
    ma2 = filter(!isnan, convolution_filter(ma1, fill(1/period, period); sides=2))
    ma3 = filter(!isnan, convolution_filter(ma2, fill(1/3, 3); sides=2))
    xL = collect(1.0:length(ma3))
    return _stl_smooth(xL, ma3; k=l_window, degree=l_degree)
end

"""
_stl_robustness_weights(resid) -- STL's outer-loop bisquare robustness
weights (Cleveland et al. 1990's `rw`), one weight per residual:
`cmad = 6*median(|resid|)`; weight `1` for `|resid| <= 0.001*cmad`,
bisquare of `|resid|/cmad` in between, `0` for `|resid| >= 0.999*cmad`
(the `0.001`/`0.999` edge-snapping avoids near-zero/near-one bisquare
noise right at the boundary). `cmad == 0` (a perfect fit) returns all
ones. Verified directly against Python's `_stl.pyx` `_rwts` -- its
partition-based `3*(rw_part[mid0]+rw_part[mid1])` is algebraically
`6*median(|resid|)` for both even and odd `n` (the two middle-partition
indices coincide for odd `n`, doubling the same value), which is also
the *fixed* median computation `statsmodels`'s own docs say it uses in
place of the original NETLIB Fortran's documented buggy one -- so
`Statistics.median` here is deliberately the correct target, not an
approximation of it.
"""
function _stl_robustness_weights(resid::AbstractVector{<:Real})
    absresid = abs.(resid)
    cmad = 6 * median(absresid)
    cmad == 0 && return ones(length(absresid))
    c1 = 0.001 * cmad
    c9 = 0.999 * cmad
    return [r <= c1 ? 1.0 : r <= c9 ? _bisquare(r / cmad) : 0.0 for r in absresid]
end

"""
_stl_cycle_subseries!(C, detrend, robweights, period, n, s_window, s_degree, use_threads) --
the cycle-subseries smoothing step (Cleveland et al. 1990's step 2): split
`detrend` into `period` independent subseries (every `period`-th value,
offset by phase `i`), Loess-smooth each one, and write into `C`'s three
regions (`C[i]` the one-point-before extension, `C[period+idx]` the
in-sample fit at original position `idx`, `C[(m+1)*period+i]` the
one-point-after extension, where `m` is *this phase's own* subseries
length). **The `(m+1)*period+i` after-index (not `period+n+i`) matters
whenever `n` isn't an exact multiple of `period`** -- verified directly
from `_stl.pyx`'s `_ss` (`season[m*np+j]` for `m` in `0..k+1`, `k` the
phase's own subseries length -- decoding its 0-indexed layout gives
exactly `(m+1)*period+i` in this file's 1-indexed convention). Every
Stage 3.2 test used `period=12,n=48` or `period=7,n=42` -- both exact
multiples, where every phase has the *same* subseries length and
`period+n+i` and `(m+1)*period+i` coincide, silently masking this as a
real bug until `mstl_decompose`'s `period=168,n=500` (verified against
real `statsmodels` to expose it: subseries lengths of `2` and `3` in the
*same* call, `period+n+i` wrong for the shorter-subseries phases).

**Provably embarrassingly parallel**: every phase `i` reads only shared,
read-only inputs (`detrend`, `robweights`) and writes to indices of `C`
disjoint from every other phase -- every index phase `i` ever writes
(`i` itself, `period+idx`, `(m+1)*period+i`) is congruent to `i` modulo
`period`, so a different phase `i2` can never collide with it regardless
of `m`, and within one phase the after-index is always strictly past its
own largest in-sample index -- confirmed from
`handoff/stage-3.3-mstl-handoff.md` section 3, which
verified from both R's and Python's actual MSTL source that this is the
*only* genuine parallelism opportunity anywhere in STL/MSTL (the
`iterate`/period loop in `mstl_decompose` and the outer/inner loops here
are all provably sequential, each reading state the previous pass just
wrote). Threaded via `Threads.@threads` when `use_threads` -- callers
gate this on `Threads.nthreads() > 1` (spawning tasks on a
single-threaded Julia process is pure overhead) and `period >= 4`
(thread-spawning overhead plausibly exceeds the work for very small
periods; not rigorously benchmarked, a starting guess per the handoff).
"""
function _stl_cycle_subseries!(C::Vector{Float64}, detrend::AbstractVector{<:Real},
                                robweights::AbstractVector{<:Real}, period::Integer, n::Integer,
                                s_window::Integer, s_degree::Integer, use_threads::Bool)
    if use_threads
        Threads.@threads for i in 1:period
            idxs = i:period:n
            m = length(idxs)
            xs = collect(1.0:m)
            ys = detrend[idxs]
            xout = vcat(0.0, xs, Float64(m + 1))
            fit = _stl_smooth(xs, ys; k=s_window, degree=s_degree, xout=xout,
                              weights=robweights[idxs])
            C[i] = fit[1]
            for (j, idx) in enumerate(idxs)
                C[period + idx] = fit[1 + j]
            end
            C[(m + 1) * period + i] = fit[end]
        end
    else
        for i in 1:period
            idxs = i:period:n
            m = length(idxs)
            xs = collect(1.0:m)
            ys = detrend[idxs]
            xout = vcat(0.0, xs, Float64(m + 1))
            fit = _stl_smooth(xs, ys; k=s_window, degree=s_degree, xout=xout,
                              weights=robweights[idxs])
            C[i] = fit[1]
            for (j, idx) in enumerate(idxs)
                C[period + idx] = fit[1 + j]
            end
            C[(m + 1) * period + i] = fit[end]
        end
    end
    return C
end

"""
_stl_inner(y, period; ...) -- the STL inner loop (Cleveland et al. 1990,
section 3.1). Repeated `inner` times:

1. Detrend: `y - trend` (`trend` starts at `trend_init` -- all zeros on
   the very first outer-loop pass, matching Python's `_stl.pyx`, whose
   `fit()` zeros `self._trend`/`self._season` exactly once, before the
   first `_onestp` call, and never again; each subsequent outer-loop pass
   continues refining the *previous* pass's trend rather than restarting
   from scratch -- a real bug when first implemented here, since
   restarting `trend` at zero on every outer pass silently returns a
   different, wrong answer whenever `outer > 0`).
2. Split into `period` cycle-subseries, Loess-smooth each independently
   (`s_window`/`s_degree`), extending one point before and one after
   each subseries' own extent (`_stl_smooth`'s `xout` support, evaluating
   at cycle-index 0 and `m+1` for a subseries of length `m`).
3. Low-pass filter the smoothed, re-interleaved cycle-subseries
   (`_stl_lowpass`).
4. Seasonal = smoothed cycle-subseries minus the low-pass result.
5. Trend = Loess-smooth (`t_window`/`t_degree`) of `y - seasonal`.

`robweights` (default all-ones, i.e. `robust=false`'s behavior) is the
*outer*-loop robustness weight vector -- constant across all `inner`
passes within one call (it only changes between calls, in
`stl_decompose`'s outer loop), applied to the cycle-subseries and trend
Loess calls but deliberately NOT to the low-pass step's Loess call,
exactly matching Python's `_stl.pyx` (`_onestp`'s low-pass `_ess` call
hardcodes `userw=False`; only `_ss`, the cycle-subseries step, and the
final trend `_ess` call receive `self._use_rw`/`self._rw`).

`parallel` gates whether the cycle-subseries step (the one genuinely
parallel piece of this loop, see [`_stl_cycle_subseries!`](@ref)) uses
threads -- only engaged when `Threads.nthreads() > 1` (a single-threaded
Julia process gets zero benefit and pure overhead from spawning tasks)
and `period >= 4`.

Returns the final `(trend, seasonal, resid)`.
"""
function _stl_inner(y::Vector{Float64}, period::Integer;
                     s_window::Integer, s_degree::Integer,
                     t_window::Integer, t_degree::Integer,
                     l_window::Integer, l_degree::Integer,
                     inner::Integer, robweights::AbstractVector{<:Real}=ones(length(y)),
                     trend_init::AbstractVector{<:Real}=zeros(length(y)), parallel::Bool=true)
    n = length(y)
    trend = collect(Float64, trend_init)
    seasonal = zeros(n)
    use_threads = parallel && Threads.nthreads() > 1 && period >= 4

    for _ in 1:inner
        detrend = y .- trend

        C = Vector{Float64}(undef, n + 2period)
        _stl_cycle_subseries!(C, detrend, robweights, period, n, s_window, s_degree, use_threads)

        L = _stl_lowpass(C, period, l_window, l_degree)
        seasonal = C[(period+1):(period+n)] .- L

        deseasonalized = y .- seasonal
        xt = collect(1.0:n)
        trend = _stl_smooth(xt, deseasonalized; k=t_window, degree=t_degree, weights=robweights)
    end

    resid = y .- trend .- seasonal
    return trend, seasonal, resid
end

"""
    stl_decompose(x, period::Integer;
                  seasonal_window::Integer=7, seasonal_degree::Integer=1,
                  trend_window::Union{Nothing,Integer}=nothing, trend_degree::Integer=1,
                  low_pass_window::Union{Nothing,Integer}=nothing,
                  low_pass_degree::Union{Nothing,Integer}=nothing,
                  robust::Bool=false,
                  inner::Union{Nothing,Integer}=nothing,
                  outer::Union{Nothing,Integer}=nothing,
                  parallel::Bool=true) -> STLDecomposition

Seasonal-Trend decomposition using Loess (STL; Cleveland, Cleveland,
McRae & Terpenning, 1990). Matches both R's `stats::stl()` and Python's
`statsmodels.tsa.seasonal.STL` exactly once parameters are matched --
verified directly by running both on identical data
(`handoff/stage-3.2-stl-handoff.md`/`stage-3.2-transcript.txt`): every
default-value difference between the two references (`seasonal_degree`,
the `low_pass_degree`/`trend_degree` coupling, jump-parameter shortcuts,
`inner`'s non-robust default) is a genuine discrepancy in *defaults*, not
in the algorithm itself.

- `seasonal_window`/`trend_window`/`low_pass_window`: neighborhood span
  for each Loess smoothing step, as an explicit point count (STL's own
  convention, not a fraction). Every explicitly-given window must be an
  **odd integer >= 3** (`trend_window`/`low_pass_window` must also be
  `> period`) -- this is where R and Python genuinely diverge: R's own
  NETLIB Fortran (`stl.f`'s top-level `stl` subroutine) silently
  *auto-corrects* any window to `max(3,w)` then `+1` if even, while
  Python's `_stl.pyx` raises on the same bad input. This package
  deliberately follows Python's stricter, explicit-error behavior here,
  consistent with the same choice already made for `classical_decompose`.
  `trend_window`/`low_pass_window` default (`nothing`) to the standard
  auto-formulas: `trend_window = nextodd(ceil(1.5*period/(1-1.5/seasonal_window)))`
  (identical in both references, verified from R's actual source) and
  `low_pass_window = nextodd(period)` (R's stated default -- Python's own
  default, `nextodd(period+1)`, differs for odd `period`; R's formula is
  used here, so the auto-computed default is deliberately *not* held to
  the same `> period` floor as an explicit value).
- `seasonal_degree`/`trend_degree`/`low_pass_degree`: local polynomial
  degree for each smoothing step (`0` or `1`). `low_pass_degree`
  defaults (`nothing`) to *following* `trend_degree` -- R's dynamic
  coupling, judged the more defensible default (no obvious reason
  low-pass smoothness should be decoupled from trend smoothness), rather
  than Python's fixed default of `1` regardless of `trend_degree`.
- `inner`: number of inner-loop passes. `nothing` defaults to `5`
  (non-robust) or `2` (robust) -- Python's defaults, chosen over R's
  leaner `2`/`1` as the more defensible precision-first choice for a
  package prioritizing correctness over the 1990s-Fortran-era
  performance concession R's default reads as.
- `robust`/`outer`: **`robust` itself has no direct computational
  effect** in either reference -- verified directly from R's `stl.R`
  (its `.Fortran()` call passes only numeric `ni`/`no` iteration counts,
  never a robust flag) and Python's `_stl.pyx` (`self.robust` is read
  only to pick `inner`/`outer` *defaults*, `2`/`15`, identical in both
  references). The outer/robustness loop is gated entirely by the
  *effective* `outer` count being `> 0`, confirmed empirically:
  `robust=true` with an explicit `outer=0` and `robust=false` with the
  same `outer=0` (and matched `inner`) produce bit-identical output on
  real `statsmodels` data. When the effective outer count is `> 0`,
  `outer` outer-loop passes run: each computes bisquare robustness
  weights (`cmad = 6*median(|resid|)`, weight `1`/bisquare/`0` below
  `0.001*cmad`/between/above `0.999*cmad`) from the *previous* pass's
  residuals, then re-runs the `inner`-pass loop with those weights
  applied to the cycle-subseries and trend Loess steps only -- **not**
  the low-pass step, exactly matching `_stl.pyx`'s `_onestp` (its
  low-pass `_ess` call hardcodes `userw=False`). This package
  deliberately targets Python's *fixed* bisquare/median computation, not
  R's original NETLIB Fortran, which `statsmodels`'s own docs confirm has
  a documented median-computation bug under `robust=TRUE` -- verified
  directly (not just from the docs) to cause a real, consistent ~1e-3
  numeric divergence from Python on data with genuine outliers (two
  points, `+30`/`-25`, injected into `monthly.csv`); Julia's own
  `Statistics.median` is algebraically identical to `_stl.pyx`'s
  partition-based fixed computation for both even and odd `n`, so no
  hand-rolled median substitute was needed here.
- `parallel`: threads the cycle-subseries smoothing step (the `period`
  independent per-phase Loess fits -- see
  [`_stl_cycle_subseries!`](@ref)) via `Threads.@threads`, the *only*
  genuinely parallel piece of STL -- everything else (the `inner`/outer
  loops, the low-pass and trend Loess steps) reads state the previous
  step just wrote, confirmed sequential from both R's and Python's actual
  MSTL source (`handoff/stage-3.3-mstl-handoff.md` section 3). Not in
  either reference (both are single-threaded); default `true` but a
  no-op unless Julia was launched with more than one thread
  (`Threads.nthreads() > 1`) and `period >= 4` -- neither condition holds,
  it silently falls back to a plain loop, so this is always safe to leave
  on. Numerically identical to `parallel=false` (each phase writes to
  disjoint output positions, no shared mutable state), so it only affects
  wall-clock time, never the result -- most worth enabling for
  [`mstl_decompose`](@ref)'s large weekly/daily periods (e.g. `period=168`
  for hourly data with weekly seasonality), where a single-threaded
  cycle-subseries loop is genuinely a bottleneck.

!!! note "`seasonal_window=:periodic` is not yet implemented"
    R's `s.window="periodic"` mode (seasonal component = plain mean per
    within-cycle position, no Loess) has no numeric verification target
    in this package yet and is a documented gap, not silently
    unsupported.

`x` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); t = 0:47; y = (100 .+ 0.5 .* t) .+ 10 .* sin.(2π .* t ./ 12) .+ randn(48);

julia> r = stl_decompose(y, 12; seasonal_window=7, seasonal_degree=0, trend_window=19,
                          trend_degree=1, low_pass_window=13, low_pass_degree=1, inner=2);

julia> length(r.trend) == length(y)
true

julia> r2 = stl_decompose(y, 12; robust=true, outer=5);

julia> length(r2.trend) == length(y)
true

julia> r3 = stl_decompose(y, 12; outer=-1);
ERROR: ArgumentError: outer must be >= 0
```
"""
function stl_decompose(x, period::Integer;
                        seasonal_window::Integer=7, seasonal_degree::Integer=1,
                        trend_window::Union{Nothing,Integer}=nothing, trend_degree::Integer=1,
                        low_pass_window::Union{Nothing,Integer}=nothing,
                        low_pass_degree::Union{Nothing,Integer}=nothing,
                        robust::Bool=false,
                        inner::Union{Nothing,Integer}=nothing,
                        outer::Union{Nothing,Integer}=nothing,
                        parallel::Bool=true)
    period >= 2 || throw(ArgumentError("period must be >= 2"))
    (isodd(seasonal_window) && seasonal_window >= 3) ||
        throw(ArgumentError("seasonal_window must be an odd integer >= 3, got $seasonal_window"))
    if trend_window !== nothing
        (isodd(trend_window) && trend_window >= 3) ||
            throw(ArgumentError("trend_window must be an odd integer >= 3, got $trend_window"))
        trend_window > period ||
            throw(ArgumentError("trend_window must be > period ($period), got $trend_window"))
    end
    if low_pass_window !== nothing
        (isodd(low_pass_window) && low_pass_window >= 3) ||
            throw(ArgumentError("low_pass_window must be an odd integer >= 3, got $low_pass_window"))
        low_pass_window > period ||
            throw(ArgumentError("low_pass_window must be > period ($period), got $low_pass_window"))
    end
    seasonal_degree in (0, 1) || throw(ArgumentError("seasonal_degree must be 0 or 1"))
    trend_degree in (0, 1) || throw(ArgumentError("trend_degree must be 0 or 1"))

    y = tsvalues(x)
    n = length(y)
    any(!isfinite, y) && throw(ArgumentError("stl_decompose: missing/non-finite values not supported"))
    n >= 2*period || throw(ArgumentError("need at least 2 full periods ($(2*period) observations), got $n"))

    t_window = trend_window === nothing ?
        _nextodd(ceil(Int, 1.5*period / (1 - 1.5/seasonal_window))) : trend_window
    l_window = low_pass_window === nothing ? _nextodd(period) : low_pass_window
    l_degree = low_pass_degree === nothing ? trend_degree : low_pass_degree
    l_degree in (0, 1) || throw(ArgumentError("low_pass_degree must be 0 or 1"))

    inner_n = inner === nothing ? (robust ? 2 : 5) : inner
    inner_n >= 1 || throw(ArgumentError("inner must be >= 1"))
    outer_n = outer === nothing ? (robust ? 15 : 0) : outer
    outer_n >= 0 || throw(ArgumentError("outer must be >= 0"))

    # `robust` itself has no direct computational effect in either R or Python --
    # verified from R's stl.R (the Fortran call receives only numeric `ni`/`no`
    # counts, never a robust flag) and Python's _stl.pyx (`robust` only selects
    # inner/outer *defaults*). The outer/robustness loop is really gated by the
    # *effective* outer count being > 0, which is what's checked here -- not the
    # raw `robust` argument -- so an explicit `outer` also triggers it even
    # with the default `robust=false`, exactly as it would in R/statsmodels.
    yf = collect(Float64, y)
    robweights = ones(n)
    trend = zeros(n)
    local seasonal, resid
    for pass in 0:outer_n
        trend, seasonal, resid = _stl_inner(yf, period;
                                             s_window=seasonal_window, s_degree=seasonal_degree,
                                             t_window=t_window, t_degree=trend_degree,
                                             l_window=l_window, l_degree=l_degree,
                                             inner=inner_n, robweights=robweights, trend_init=trend,
                                             parallel=parallel)
        pass == outer_n && break
        robweights = _stl_robustness_weights(resid)
    end

    return STLDecomposition(yf, trend, seasonal, resid, period, robweights)
end
