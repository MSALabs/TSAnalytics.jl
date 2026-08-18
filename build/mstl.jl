export mstl_decompose, MSTLDecomposition

"""
    MSTLDecomposition

Result of [`mstl_decompose`](@ref). `seasonal` is `n x length(periods)` --
one column per period, in the same (ascending) order as `periods` --
rather than Python's `squeeze`-to-vector-when-one-period convenience,
kept a fixed-shape `Matrix` here for type stability. `weights` is the
final robustness weight vector from the *last* period's *last* STL fit
(matching both R's and Python's MSTL, which also only expose the last
fit's weights, not one per period).
"""
struct MSTLDecomposition
    observed::Vector{Float64}
    trend::Vector{Float64}
    seasonal::Matrix{Float64}
    resid::Vector{Float64}
    periods::Vector{Int}
    weights::Vector{Float64}
end

"""
    mstl_decompose(x, periods::Union{Integer,AbstractVector{<:Integer}};
                  windows::Union{Nothing,Integer,AbstractVector{<:Integer}}=nothing,
                  lambda::Union{Nothing,Real,Symbol}=nothing,
                  iterate::Integer=2,
                  parallel::Bool=true,
                  stl_kwargs::NamedTuple=NamedTuple()) -> MSTLDecomposition

Multiple Seasonal-Trend decomposition using Loess (MSTL; Bandura, Hyndman
& Bergmeir, 2021), for series with more than one seasonal period (e.g. an
hourly series with both daily and weekly seasonality). A thin sequential
wrapper over [`stl_decompose`](@ref): one [`STL`](@ref stl_decompose) fit
per period, `iterate` times, each period's fit re-added to the residual
before its own re-fit and removed again after -- matching both R's
`forecast::mstl` and Python's `statsmodels.tsa.seasonal.MSTL` exactly,
verified directly from Python's actual (pure-Python, not compiled)
source (`statsmodels/tsa/stl/mstl.py`) rather than its docs.

- `periods`: one period (`Integer`) or several (any integer vector) --
  sorted ascending internally, matching both references. Every period
  must be `>= 2`; any period `>= length(x)/2` is **dropped with a
  warning**, not an error, matching Python's confirmed source behavior
  (R's `mstl.R` does the same `msts[msts < n/2]` filtering).
- `windows`: the `seasonal_window` to use for each period's STL fit, in
  the same order as (post-sort) `periods`. `nothing` (default) computes
  `7 + 4*i` for `i` in `1:length(periods)` -- confirmed identical in both
  references (Python's `_default_seasonal_windows`, R's `7 + 4*seq_len`).
  A scalar broadcasts the same window to every period -- a deliberate
  convenience, **not** matching Python's actual source: passing a scalar
  `windows` there with more than one period raises `ValueError` (its
  `_process_windows` wraps a scalar as a length-1 tuple without
  broadcasting it, then a length check against `periods` fails). Broadcast
  is unambiguous and clearly useful here, so it's kept rather than
  reproducing what reads as an oversight rather than a considered choice.
- `lambda`: `nothing` (default, no transform) applies none; a `Real`
  applies the classic Box-Cox transform (`(x^lambda-1)/lambda`, or
  `log(x)` at `lambda=0`) to `x` *before* decomposing -- **matching
  Python's confirmed source behavior, the transform is never inverted**:
  `observed`/`trend`/`seasonal`/`resid` are all returned in the
  transformed scale when `lambda` is given, not back-transformed to the
  original units. Requires strictly positive `x` (the transform's own
  domain restriction). **`lambda=0` here genuinely applies the log
  transform, unlike Python**: its source uses `elif self.lmbda:` to
  decide whether to transform at all, and `0.0` is falsy in Python, so
  `lmbda=0.0` there silently applies *no* transform (confirmed by
  execution, not just reading the source) -- an accidental quirk of using
  truthiness for a numeric check, not a considered choice worth
  reproducing, so this package checks `lambda !== nothing` instead.
  `lambda=:auto` (MLE-estimated, like Python's
  `scipy.stats.boxcox(x, lmbda=None)` or R's Guerrero-method default) is
  **not yet implemented** -- it needs a 1-D likelihood optimizer this
  package doesn't have yet (Stage 4.1), so it throws `ArgumentError`
  rather than silently falling back to `lambda=nothing`.
- `iterate`: number of passes over all periods, refining each one using
  the others' current estimates. Forced to `1` internally when there's
  only one period (matching both references -- with one period there's
  nothing to refine against).
- `parallel`: forwarded to every [`stl_decompose`](@ref) call -- see its
  own docstring. This is where `parallel=true` actually pays off: MSTL's
  whole use case is large periods (an hourly series' weekly period is
  `168`), exactly where the cycle-subseries threading helps most.
- `stl_kwargs`: extra keywords forwarded to every `stl_decompose` call
  (e.g. `(seasonal_degree=0, robust=true)`) -- matches Python's
  `stl_kwargs` dict pass-through design (chosen over R's `...` splat for
  better discoverability). Passing `seasonal_window`, `period`, or
  `parallel` here conflicts with this function's own arguments and
  raises Julia's normal duplicate-keyword error rather than Python's
  silent override-and-drop.

!!! note "Missing data is not imputed"
    Python's own source docstring states plainly: *"Missing data must be
    handled outside of this class."* -- confirmed directly from source,
    not inferred. This package follows that: non-finite input throws
    `ArgumentError`, matching every other function in this package. R's
    `forecast::mstl` documentation describes imputing via `na.interp`
    first, per the algorithm's original paper -- but that claim is from
    the paper, not independently confirmed against R's actual shipped
    source (no R available in this environment to check); Python's
    behavior is source-verified, so it's the one adopted here.

`x` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); t = 1:200; y = (100 .+ 0.01 .* t) .+ 5 .* sin.(2π .* t ./ 24) .+ 3 .* sin.(2π .* t ./ 48) .+ randn(200);

julia> r = mstl_decompose(y, [24, 48]);

julia> size(r.seasonal) == (200, 2)
true

julia> r.periods
2-element Vector{Int64}:
 24
 48
```
"""
function mstl_decompose(x, periods::Union{Integer,AbstractVector{<:Integer}};
                         windows::Union{Nothing,Integer,AbstractVector{<:Integer}}=nothing,
                         lambda::Union{Nothing,Real,Symbol}=nothing,
                         iterate::Integer=2,
                         parallel::Bool=true,
                         stl_kwargs::NamedTuple=NamedTuple())
    periods_v = periods isa Integer ? [Int(periods)] : collect(Int, periods)
    isempty(periods_v) && throw(ArgumentError("periods must be non-empty"))
    all(>=(2), periods_v) || throw(ArgumentError("every period must be >= 2"))

    windows_v = windows === nothing ? nothing :
        windows isa Integer ? fill(Int(windows), length(periods_v)) : collect(Int, windows)
    windows_v !== nothing && length(windows_v) != length(periods_v) &&
        throw(ArgumentError("periods and windows must have the same length"))

    y = tsvalues(x)
    n = length(y)
    any(!isfinite, y) && throw(ArgumentError("mstl_decompose: missing/non-finite values not supported"))

    # sort periods ascending, carrying any explicit windows along as pairs;
    # default windows (7 + 4*i) are assigned positionally after sorting,
    # matching Python's confirmed source (they have no per-period identity)
    if windows_v === nothing
        periods_sorted = sort(periods_v)
        windows_sorted = [7 + 4i for i in 1:length(periods_sorted)]
    else
        order = sortperm(periods_v)
        periods_sorted = periods_v[order]
        windows_sorted = windows_v[order]
    end

    # drop periods >= n/2 (warn, don't error) -- matches Python's confirmed
    # source (statsmodels/tsa/stl/mstl.py's `_process_periods_and_windows`)
    keep = periods_sorted .< n / 2
    if !all(keep)
        @warn "mstl_decompose: dropping period(s) >= half the series length ($(n/2))" dropped=periods_sorted[.!keep]
        periods_sorted = periods_sorted[keep]
        windows_sorted = windows_sorted[keep]
    end
    isempty(periods_sorted) &&
        throw(ArgumentError("no periods remain after dropping those >= half the series length ($(n/2))"))

    num_seasons = length(periods_sorted)
    iterate_n = num_seasons == 1 ? 1 : iterate
    iterate_n >= 1 || throw(ArgumentError("iterate must be >= 1"))

    yt = if lambda === nothing
        collect(Float64, y)
    elseif lambda === :auto
        throw(ArgumentError("mstl_decompose: lambda=:auto (MLE Box-Cox estimation) is not yet " *
                             "implemented -- pass an explicit Real lambda, or nothing"))
    elseif lambda isa Real
        all(>(0), y) ||
            throw(ArgumentError("mstl_decompose: lambda requires strictly positive data for the Box-Cox transform"))
        lambda == 0 ? log.(Float64.(y)) : (Float64.(y) .^ lambda .- 1) ./ lambda
    else
        throw(ArgumentError("lambda must be `nothing`, `:auto`, or a Real"))
    end

    seasonal = zeros(n, num_seasons)
    deseas = copy(yt)
    local res
    for _ in 1:iterate_n
        for i in 1:num_seasons
            deseas = deseas .+ seasonal[:, i]
            res = stl_decompose(deseas, periods_sorted[i]; seasonal_window=windows_sorted[i],
                                 parallel=parallel, stl_kwargs...)
            seasonal[:, i] = res.seasonal
            deseas = deseas .- seasonal[:, i]
        end
    end

    trend = res.trend
    resid = deseas .- trend
    return MSTLDecomposition(yt, trend, seasonal, resid, periods_sorted, res.weights)
end
