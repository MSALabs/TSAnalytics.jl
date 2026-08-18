# Loess/lowess primitive underlying STL (Stage 3.2). Deliberately unexported,
# analogous to `_ols` (Stage 1.4): an internal numerical engine other public
# functions (`stl_decompose`) build on, not a public regression API in its
# own right. See handoff/stage-3.2-stl-handoff.md, which flagged this as the
# genuinely-new prerequisite STL needs and recommended building it as its own
# independently-verified unit first.
#
# Algorithm and every numeric convention below (frac -> neighbor count via
# truncation, not rounding; tricube weights; bisquare robustness reweighting
# on |resid|/(6*median(|resid|))) verified directly from
# `statsmodels.nonparametric.smoothers_lowess_old` -- the readable
# pre-Cython predecessor of `statsmodels`' actual `lowess()` (the current
# version is a compiled .pyd with no introspectable source). Confirms that
# weighted local fits use `sqrt(weight)` row-scaling into an ordinary
# least-squares solve (exactly `_ols`'s existing convention, reused directly
# here) -- notably NOT what a quick read of `SeasonalTrendLoess.jl`'s
# `loess.jl` does (it scales rows by the raw weight, not its square root,
# before an unweighted solve -- mathematically a different, non-standard
# weighting scheme; not something to port, per the "reference, never port"
# policy, and not what the validated Python/statsmodels reference does).
#
# `degree` (0/1/2) and off-grid `xout` evaluation were added once
# `stl_decompose`'s inner loop actually needed them (R's `s.degree=0`
# default, and extending each cycle-subseries one period beyond its own
# extent) -- both validated against real `statsmodels` too (its `lowess`
# accepts an `xvals` argument for exactly this off-grid case).
#
# `k > n` (requesting more neighbors than exist -- routine for STL's
# cycle-subseries smoothing, where `s_window` is typically far larger
# than the handful of points in one subseries) uses every available
# point, with the tricube neighborhood width *extended by
# `(k-n) ÷ 2` (integer division)* rather than silently capping `k` at
# `n`. This is an ADDITIVE adjustment, not the multiplicative `*(k/n)`
# stretch `SeasonalTrendLoess.jl`'s `loess.jl` uses -- that formula was
# tried first here too, and it was close but measurably wrong once
# checked against real STL output. The additive version, and the integer
# (truncating) division specifically, come directly from the actual
# NETLIB STL Fortran source (`stlest`, read via
# `github.com/wch/r-source`'s mirror of `src/library/stats/src/stl.f` --
# the exact routine R's own `stl()` calls into): `h = max(xs-nleft,
# nright-xs); if (len .gt. n) h = h + (len-n)/2` with `len`/`n` declared
# `integer` -- Fortran integer division truncates before the result is
# used, which is NOT the same number as real division for odd `k-n`
# (verified to matter: this is the one-line difference between output
# that's close-but-wrong and output that matches real `statsmodels` to
# 8+ significant digits on `monthly.csv`).

_tricube(u::Real) = u < 1 ? (1 - u^3)^3 : 0.0
_bisquare(u::Real) = u < 1 ? (1 - u^2)^2 : 0.0

"""
_lowess_fit1(x, y, x0, k, degree, robweights) -- locally-weighted
degree-`degree` polynomial fit evaluated at `x0`, using the `min(k,
length(x))` nearest neighbors (by `|x - x0|`) with tricube weights
(neighborhood width extended by `(k - length(x)) ÷ 2`, integer division,
when `k` exceeds the number of available points -- see the file-level
comment for why this is additive, not multiplicative, and why the
integer division matters), further scaled by `robweights` (all-ones on a
non-robust pass). `x0` may lie outside the range of `x` (extrapolation)
-- the neighbor search and weighting are unaffected either way, since
both are purely distance-based. Returns the fitted value at `x0` (the
fitted polynomial's intercept, since the local regressors are centered
at `x0`).
"""
function _lowess_fit1(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                       x0::Real, k::Integer, degree::Integer,
                       robweights::AbstractVector{<:Real})
    n = length(x)
    keff = min(k, n)
    d = abs.(x .- x0)
    nb = partialsortperm(d, 1:keff)
    width = maximum(view(d, nb))
    k > n && (width += (k - n) ÷ 2)

    w = if width == 0
        ones(keff)  # all keff neighbors coincide with x0
    else
        [_tricube(d[i] / width) for i in nb]
    end
    w = w .* view(robweights, nb)

    active = nb[findall(>(0), w)]
    # _ols needs strictly more observations than regression coefficients
    # (degree+1: intercept plus `degree` polynomial terms) for a
    # nondegenerate (dof > 0) weighted solve.
    length(active) <= degree + 1 && return y[nb[argmin(view(d, nb))]]  # can't fit; nearest-point fallback

    wa = w[w .> 0]
    cols = [ones(length(active))]
    for p in 1:degree
        push!(cols, (x[active] .- x0) .^ p)
    end
    X = reduce(hcat, cols)
    beta, = _ols(X, y[active]; weights=wa)
    return beta[1]
end

"""
    _lowess(x, y; frac=2/3, k=nothing, degree=1, it=3, xout=x, weights=nothing)

Locally-weighted (tricube kernel) polynomial regression smoother -- Loess/
lowess (Cleveland, 1979), the algorithm underlying STL (Cleveland et al.,
1990).

- `k`: explicit neighborhood span, as a point count -- STL's own
  `s.window`/`t.window`/`l.window` convention (a literal window size, not
  a fraction). Takes precedence over `frac` when given.
- `frac`: neighborhood span as a fraction of `n`, matching R's `loess()`/
  Python's `lowess()` convention -- used to compute `k = trunc(Int,
  frac*n)` (truncation, not rounding, matching `statsmodels` exactly)
  when `k` is not given explicitly.
- `degree`: local polynomial degree, `0`, `1`, or `2`.
- `it`: robustness iterations, reweighting each point by the bisquare
  function of its residual magnitude relative to `6*median(|residuals|)`
  (residuals larger than that get weight 0) -- matching both R's and
  Python's convention. Residuals (and therefore robustness weights) are
  always computed at the original `x`/`y` points, regardless of `xout`.
- `xout`: points to evaluate the final fit at (default: `x` itself).
  These may lie outside the range of `x` -- e.g. extending a
  cycle-subseries one period beyond its own extent, as `stl_decompose`'s
  inner loop needs.
- `weights`: optional externally-supplied per-point weights (same length
  as `x`/`y`), seeding `it`'s pass-0 robustness weights instead of the
  default all-ones -- how `stl_decompose`'s own outer/robustness loop
  (a separate mechanism from this `it`, always called with `it=0` from
  `stl_decompose`) injects its own reweighting into the cycle-subseries
  and trend Loess calls.
"""
function _lowess(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                  frac::Real=2/3, k::Union{Nothing,Integer}=nothing,
                  degree::Integer=1, it::Integer=3,
                  xout::AbstractVector{<:Real}=x,
                  weights::Union{Nothing,AbstractVector{<:Real}}=nothing)
    n = length(x)
    length(y) == n || throw(ArgumentError("x and y must have the same length"))
    degree in (0, 1, 2) || throw(ArgumentError("degree must be 0, 1, or 2"))
    it >= 0 || throw(ArgumentError("it must be >= 0"))
    weights === nothing || length(weights) == n ||
        throw(ArgumentError("weights must have the same length as x and y"))
    kk = if k === nothing
        0 < frac <= 1 || throw(ArgumentError("frac must be in (0, 1]"))
        trunc(Int, frac * n)
    else
        k >= 1 || throw(ArgumentError("k must be >= 1"))
        Int(k)
    end
    kk = max(kk, degree + 1)  # not clamped to n -- k > n is legitimate (stretched width, see above)

    robweights = weights === nothing ? ones(n) : collect(Float64, weights)
    fitted = similar(y, Float64)
    for pass in 0:it
        if pass > 0
            resid = abs.(y .- fitted)
            s = median(resid)
            robweights = if s == 0
                ones(n)
            else
                u = resid ./ (6s)
                [ui >= 1 ? 0.0 : _bisquare(ui) for ui in u]
            end
        end
        fitted = [_lowess_fit1(x, y, x[i], kk, degree, robweights) for i in 1:n]
    end

    return [_lowess_fit1(x, y, x0, kk, degree, robweights) for x0 in xout]
end
