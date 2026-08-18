export expanding_window_split, sliding_window_split, tscv

"""
    expanding_window_split(n; initial_window, step_length=1, fh=1) -> Vector{Tuple{Vector{Int},Vector{Int}}}

Generate `(train_idx, test_idx)` pairs for expanding-window
cross-validation: the training window grows by `step_length` each fold,
starting from `initial_window` observations. Matches `sktime`'s
`ExpandingWindowSplitter` exactly (verified against real index output --
see `handoff/stage-5.4-cv-handoff.md` section 1) and corresponds to R's
`forecast::tsCV(window=NULL)` (the default).

`fh`: an `Integer` (single horizon) or `AbstractVector` (multiple
horizons evaluated per fold, e.g. `fh=[1,2,3]`). Indices are 1-based,
Julia convention.

# Examples
```jldoctest
julia> folds = expanding_window_split(20; initial_window=10);

julia> length(folds)
10

julia> folds[1]
([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11])
```
"""
function expanding_window_split(n::Integer; initial_window::Integer,
                                 step_length::Integer=1, fh::Union{Integer,AbstractVector{<:Integer}}=1)
    initial_window >= 1 || throw(ArgumentError("initial_window must be >= 1"))
    step_length >= 1 || throw(ArgumentError("step_length must be >= 1"))
    horizons = fh isa Integer ? [fh] : collect(fh)
    all(>=(1), horizons) || throw(ArgumentError("fh entries must be >= 1"))
    max_h = maximum(horizons)
    folds = Tuple{Vector{Int},Vector{Int}}[]
    origin = initial_window
    while origin + max_h <= n
        train_idx = collect(1:origin)
        test_idx = [origin + h for h in horizons]
        push!(folds, (train_idx, test_idx))
        origin += step_length
    end
    return folds
end

"""
    sliding_window_split(n; window_length, step_length=1, fh=1) -> Vector{Tuple{Vector{Int},Vector{Int}}}

Generate `(train_idx, test_idx)` pairs for sliding (fixed-size) window
cross-validation. Matches `sktime`'s `SlidingWindowSplitter` exactly
(verified) and corresponds to R's `forecast::tsCV(window=<integer>)`.

# Examples
```jldoctest
julia> folds = sliding_window_split(20; window_length=10);

julia> folds[1]
([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11])

julia> folds[2]
([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12])
```
"""
function sliding_window_split(n::Integer; window_length::Integer,
                               step_length::Integer=1, fh::Union{Integer,AbstractVector{<:Integer}}=1)
    window_length >= 1 || throw(ArgumentError("window_length must be >= 1"))
    step_length >= 1 || throw(ArgumentError("step_length must be >= 1"))
    horizons = fh isa Integer ? [fh] : collect(fh)
    all(>=(1), horizons) || throw(ArgumentError("fh entries must be >= 1"))
    max_h = maximum(horizons)
    folds = Tuple{Vector{Int},Vector{Int}}[]
    start = 1
    while start + window_length - 1 + max_h <= n
        train_idx = collect(start:(start + window_length - 1))
        test_idx = [start + window_length - 1 + h for h in horizons]
        push!(folds, (train_idx, test_idx))
        start += step_length
    end
    return folds
end

"""
    tscv(y, fit_forecast_fn; h=1, window=nothing, initial=0, step_length=1) -> Matrix{Float64}

Rolling-origin cross-validation, R-`forecast::tsCV`-style: for each
fold, `fit_forecast_fn(train_data, hmax)` is called (`hmax =
maximum(h)`) and its point forecast(s) compared against the actual
held-out value(s), returning a matrix of errors (`actual - forecast`),
one row per fold, one column per requested horizon in `h` -- matches
R's error convention exactly (verified numerically against real
`forecast::tsCV()` output; R additionally pads its result to a
full-series-length `ts` with leading/trailing `NA`s for time-index
alignment, which `tscv` omits since a fold that can never be computed --
e.g. the trailing horizon(s) beyond the series end -- carries no
information plain `NaN` padding would add). `window=nothing` (default)
is expanding-window (matches R's `window=NULL` default); an `Integer`
gives a fixed-size sliding window. `initial` skips that many
observations before the first possible origin (matches R's argument
name and role exactly).

Built on [`expanding_window_split`](@ref)/[`sliding_window_split`](@ref)
internally, so the two never silently disagree about fold boundaries.
`fit_forecast_fn` must accept `(train_data, hmax)` and return a vector
of at least `hmax` point forecasts (only indices in `h` are read).

# Examples
```jldoctest
julia> y = collect(1.0:20.0);

julia> naive_forecast(train, h) = fill(train[end], h);

julia> errs = tscv(y, naive_forecast; h=1, initial=10);

julia> size(errs)
(10, 1)

julia> all(isapprox.(errs, 1.0; atol=1e-10))
true
```
"""
function tscv(y, fit_forecast_fn; h::Union{Integer,AbstractVector{<:Integer}}=1,
              window::Union{Nothing,Integer}=nothing, initial::Integer=0, step_length::Integer=1)
    yv = tsvalues(y)
    n = length(yv)
    horizons = h isa Integer ? [h] : collect(h)

    folds = if window === nothing
        expanding_window_split(n; initial_window=max(1, initial), step_length=step_length, fh=horizons)
    else
        sliding_window_split(n; window_length=window, step_length=step_length, fh=horizons)
    end

    errors = fill(NaN, length(folds), length(horizons))
    max_h = maximum(horizons)
    for (i, (train_idx, test_idx)) in enumerate(folds)
        train_data = yv[train_idx]
        fc = fit_forecast_fn(train_data, max_h)
        for (j, hh) in enumerate(horizons)
            errors[i, j] = yv[test_idx[j]] - fc[hh]
        end
    end
    return errors
end
