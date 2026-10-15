export mae, rmse, mape, smape, mase, accuracy, mdape, rmsse, cfe, spis

"""
    mae(actual, predicted) -> Float64

Mean absolute error: `mean(|actual - predicted|)`. Accepts anything
`tsvalues` does; `actual` and `predicted` must have the same length.

# Examples
```jldoctest
julia> mae([3.0, -0.5, 2, 7, 2], [2.5, 0.0, 2, 8, 1.25])
0.55
```
"""
function mae(actual, predicted)
    a = tsvalues(actual)
    p = tsvalues(predicted)
    length(a) == length(p) || throw(ArgumentError("actual and predicted must have the same length"))
    return sum(abs.(a .- p)) / length(a)
end

"""
    rmse(actual, predicted) -> Float64

Root mean squared error: `sqrt(mean((actual - predicted)^2))`. Accepts
anything `tsvalues` does; `actual` and `predicted` must have the same
length.

# Examples
```jldoctest
julia> rmse([3.0, -0.5, 2, 7, 2], [2.5, 0.0, 2, 8, 1.25])
0.6422616289332564
```
"""
function rmse(actual, predicted)
    a = tsvalues(actual)
    p = tsvalues(predicted)
    length(a) == length(p) || throw(ArgumentError("actual and predicted must have the same length"))
    return sqrt(sum(abs2, a .- p) / length(a))
end

"""
    mape(actual, predicted; as_percentage=true) -> Float64

Mean absolute percentage error:
`mean(|actual - predicted| / max(|actual|, eps))`. Matches `sktime`'s
`mean_absolute_percentage_error(symmetric=false)` exactly (verified from
source). `as_percentage=true` (default) scales to R's `forecast::accuracy()`
convention (e.g. `33.69` meaning 33.69%, also verified by direct execution);
`as_percentage=false` gives `sktime`'s raw-fraction convention (`0.3369`)
directly.

# Examples
```jldoctest
julia> mape([3.0, -0.5, 2, 7, 2], [2.5, 0.0, 2, 8, 1.25])
33.69047619047619
```
"""
function mape(actual, predicted; as_percentage::Bool=true)
    a = tsvalues(actual)
    p = tsvalues(predicted)
    length(a) == length(p) || throw(ArgumentError("actual and predicted must have the same length"))
    epsval = eps(Float64)
    v = sum(abs.(a .- p) ./ max.(abs.(a), epsval)) / length(a)
    return as_percentage ? 100 * v : v
end

"""
    smape(actual, predicted; as_percentage=true) -> Float64

Symmetric MAPE:
`mean(|actual - predicted| / max((|actual| + |predicted|)/2, eps))` --
matches `sktime`'s formula exactly (verified from source; one of at
least three genuinely different published "sMAPE" definitions -- see
`handoff/stage-5.3-accuracy-handoff.md` section 3 for why this one).
`as_percentage=true` (default) matches [`mape`](@ref)'s scaling
convention.

!!! warning "Use with caution"
    Hyndman (author of the MASE metric and R's `forecast` package) has
    publicly recommended against using sMAPE at all: *"There seems
    little point using the sMAPE... [prefer] the original MAPE... or
    MASE instead."* It is also not truly symmetric despite its name, and
    at least three incompatible published formulas exist under this same
    name. Provided here because it is still widely requested and used in
    forecasting competitions, not as an endorsement -- prefer [`mase`](@ref)
    or [`mape`](@ref) where a choice is available.

# Examples
```jldoctest
julia> smape([3.0, -0.5, 2, 7, 2], [2.5, 0.0, 2, 8, 1.25])
55.53379953379953
```
"""
function smape(actual, predicted; as_percentage::Bool=true)
    a = tsvalues(actual)
    p = tsvalues(predicted)
    length(a) == length(p) || throw(ArgumentError("actual and predicted must have the same length"))
    epsval = eps(Float64)
    v = sum(abs.(a .- p) ./ max.((abs.(a) .+ abs.(p)) ./ 2, epsval)) / length(a)
    return as_percentage ? 100 * v : v
end

"""
    mdape(actual, predicted; as_percentage=true) -> Float64

Median absolute percentage error: `median(|actual − predicted| / max(|actual|, eps))`,
the median counterpart of [`mape`](@ref), on the same scale (a percentage
by default, `sktime`'s fraction with `as_percentage=false`). One large
relative error -- a near-zero actual -- moves the mean and not the median.

```jldoctest
julia> mdape([100.0, 200.0, 300.0], [110.0, 190.0, 0.0])
10.0
```
"""
function mdape(actual, predicted; as_percentage::Bool=true)
    a = tsvalues(actual)
    p = tsvalues(predicted)
    length(a) == length(p) || throw(ArgumentError("actual and predicted must have the same length"))
    isempty(a) && throw(ArgumentError("mdape: no observations"))
    v = median(abs.(a .- p) ./ max.(abs.(a), eps(Float64)))
    return as_percentage ? 100 * v : v
end

"""
    rmsse(actual, predicted, train; sp::Integer=1) -> Float64

Root mean squared scaled error: the out-of-sample root mean squared error
divided by the in-sample root mean squared error of the (seasonal-)naive
forecast on `train` -- the squared-error counterpart of [`mase`](@ref), and
the M5 competition's metric. `sktime`'s
`mean_squared_scaled_error(square_root=True)`.

```jldoctest
julia> rmsse([3.0, 5.0], [2.0, 5.0], [1.0, 2.0, 4.0])
0.4472135954999579
```
"""
function rmsse(actual, predicted, train; sp::Integer=1)
    a = tsvalues(actual)
    p = tsvalues(predicted)
    tr = tsvalues(train)
    length(a) == length(p) || throw(ArgumentError("actual and predicted must have the same length"))
    sp >= 1 || throw(ArgumentError("sp must be >= 1"))
    length(tr) > sp || throw(ArgumentError("train must have more than sp observations"))
    msd = sum(abs2, tr[(sp + 1):end] .- tr[1:(end - sp)]) / (length(tr) - sp)
    return sqrt((sum(abs2, a .- p) / length(a)) / max(msd, eps(Float64)))
end

"""
    cfe(actual, predicted) -> Float64

Cumulative forecast error, `sum(actual − predicted)`: the signed total, so
**positive means the forecasts were too low** in total, negative too high.
The bias measure for intermittent demand, where a forecast that is
systematically too high or too low costs stock-outs or holding, and the
one that shows what SBA corrects -- MASE cannot see bias.
`utilsforecast`'s `cfe` computes the opposite sign, `sum(predicted − actual)`,
while documenting this one.

```jldoctest
julia> cfe([2.0, 0.0, 3.0], [1.0, 1.0, 1.0])
2.0
```
"""
function cfe(actual, predicted)
    a = tsvalues(actual)
    p = tsvalues(predicted)
    length(a) == length(p) || throw(ArgumentError("actual and predicted must have the same length"))
    return sum(a .- p)
end

"""
    spis(actual, predicted, train) -> Float64

Scaled periods in stock: the sum of absolute forecast errors divided by
the mean demand over `train`, so that it is free of the series' scale and
can be compared and averaged across items -- `utilsforecast`'s `spis`.

```jldoctest
julia> spis([2.0, 0.0, 3.0], [1.0, 1.0, 1.0], [0.0, 4.0, 2.0])
2.0
```
"""
function spis(actual, predicted, train)
    a = tsvalues(actual)
    p = tsvalues(predicted)
    tr = tsvalues(train)
    length(a) == length(p) || throw(ArgumentError("actual and predicted must have the same length"))
    isempty(tr) && throw(ArgumentError("spis: train is empty"))
    m = mean(tr)
    m > 0 || throw(ArgumentError("spis: the training demand has mean zero, so nothing scales it"))
    return sum(abs.(a .- p)) / m
end

"""
    mase(actual, predicted, train; sp::Integer=1) -> Float64

Mean absolute scaled error: the out-of-sample MAE of `actual`/`predicted`
divided by the in-sample MAE of the (seasonal-)naive benchmark computed
on `train`. Matches `sktime`'s `mean_absolute_scaled_error` exactly
(verified from source). Requires `train` -- the training series -- as a
genuine mathematical necessity, not an incidental API choice: MASE cannot
be computed from `actual`/`predicted` alone.

`sp=1` (default): naive benchmark (`train[t]` predicts `train[t+1]`).
`sp=period`: seasonal-naive benchmark, for a series with that seasonal
period.

# Examples
```jldoctest
julia> mase([3.0, -0.5, 2, 7, 2], [2.5, 0.0, 2, 8, 1.25], [5.0, 0.5, 4, 6, 3, 5, 2])
0.18333333333333335
```
"""
function mase(actual, predicted, train; sp::Integer=1)
    a = tsvalues(actual)
    p = tsvalues(predicted)
    tr = tsvalues(train)
    length(a) == length(p) || throw(ArgumentError("actual and predicted must have the same length"))
    sp >= 1 || throw(ArgumentError("sp must be >= 1"))
    length(tr) > sp || throw(ArgumentError("train must have more than sp observations"))
    mae_naive = sum(abs.(tr[(sp + 1):end] .- tr[1:(end - sp)])) / (length(tr) - sp)
    mae_pred = mae(a, p)
    return mae_pred / max(mae_naive, eps(Float64))
end

"""
    accuracy(actual, predicted, train=nothing; sp::Integer=1) -> NamedTuple

Convenience wrapper: computes MAE, RMSE, and MAPE (percentage) always,
plus MASE if `train` is given. Mirrors R's `forecast::accuracy()` table
conceptually; returns a `NamedTuple` rather than a table object for
simplicity.

# Examples
```jldoctest
julia> acc = accuracy([3.0, -0.5, 2, 7, 2], [2.5, 0.0, 2, 8, 1.25], [5.0, 0.5, 4, 6, 3, 5, 2]);

julia> acc.mae
0.55

julia> haskey(acc, :mase)
true
```
"""
function accuracy(actual, predicted, train=nothing; sp::Integer=1)
    m = (mae=mae(actual, predicted), rmse=rmse(actual, predicted), mape=mape(actual, predicted))
    train === nothing && return m
    return merge(m, (mase=mase(actual, predicted, train; sp=sp),))
end
