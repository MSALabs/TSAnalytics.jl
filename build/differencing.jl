export diffinv, tsdiff, tsundiff

"""
    diff(x, lag::Integer; differences::Integer=1)

Lag-`lag`, order-`differences` differencing, generalizing `Base.diff`
(which only covers `lag=1, differences=1`) to also cover seasonal
differencing (`lag = period`) and repeated differencing -- matching R's
`base::diff(x, lag, differences)`. `x` accepts anything [`tsvalues`](@ref)
does, so a `TSFrame`/`TimeArray`/`DataFrame` column works directly.

`differences` is applied iteratively: one lag-`lag` difference, then
another lag-`lag` difference of *that* result, `differences` times total
-- not a single closed-form pass. This iterative composition is also how
regular (`lag=1`) and seasonal (`lag=period`) differencing combine for
ARIMA/SARIMA's `d`/`D` orders: call this function twice, once per lag,
e.g. `diff(diff(y, 12), 1)` for one seasonal plus one regular difference.

`lag` is a required positional argument here, unlike the rest of this
package's keyword-heavy style -- deliberately, so this method's arity
(2 required positional arguments) never overlaps with `Base.diff`'s own
single-argument method. `diff(x)` (one argument) always calls
`Base.diff` unchanged; `diff(x, lag; ...)` (two arguments) calls this
method. See [`tsdiff`](@ref) for an all-keyword alias that reads more
naturally when `lag` is left at its default of `1`.

See also [`diffinv`](@ref), its exact inverse.

# Examples
```jldoctest
julia> using TSAnalytics

julia> diff([1.0, 3.0, 6.0, 10.0], 1)   # same as Base.diff([1.0,3.0,6.0,10.0])
3-element Vector{Float64}:
 2.0
 3.0
 4.0

julia> diff([1.0, 3.0, 6.0, 10.0], 1; differences=2)
2-element Vector{Float64}:
 1.0
 1.0

julia> diff(1.0:12.0, 4)   # seasonal difference, period 4
8-element Vector{Float64}:
 4.0
 4.0
 4.0
 4.0
 4.0
 4.0
 4.0
 4.0
```
"""
function Base.diff(x, lag::Integer; differences::Integer=1)
    lag >= 1 || throw(ArgumentError("lag must be >= 1, got $lag"))
    differences >= 0 || throw(ArgumentError("differences must be >= 0, got $differences"))
    y = tsvalues(x)
    n = length(y)
    lag * differences < n ||
        throw(ArgumentError("series too short: need more than lag*differences=$(lag*differences) observations, got $n"))
    # collect eagerly: slicing+broadcasting a range whose differences happen
    # to be constant (e.g. `1.0:12.0`) collapses back into a StepRangeLen
    # rather than a plain Vector, unlike Base.diff, which always returns a
    # Vector regardless of input type -- matching that here for consistency.
    r = collect(y)
    for _ in 1:differences
        r = r[(lag+1):end] .- r[1:(end-lag)]
    end
    return r
end

"""
    diffinv(x; lag::Integer=1, differences::Integer=1, xi=nothing)

Exact inverse of [`diff`](@ref): reconstructs a series from its
lag-`lag`, order-`differences` difference `x`, given the `lag*differences`
seed values `xi` that immediately precede `x` in the original series'
timeline. Matches R's `stats::diffinv`, including its convention of
returning `xi` prepended to the reconstruction -- output length is
`length(x) + lag*differences`.

`xi` defaults to zeros (matching `diffinv`'s R default), which turns a
series of increments into a cumulative level series starting from zero:
for `lag=1, differences=1`, `diffinv(x)` is `vcat(0.0, cumsum(x))`.

`xi` is not necessarily the *historical head* of the series -- the same
function also covers turning forecasted differenced values back into
level forecasts, once model fitting lands (Stage 6 of
`development-sequence.md`): there, `xi` is the *tail* of already-observed
history, and `x` is the model's forecasted differenced values extending
beyond it.

See also [`tsundiff`](@ref) for an alias.

# Examples
```jldoctest
julia> using TSAnalytics

julia> diffinv([2.0, 3.0, 4.0]; xi=[1.0])
4-element Vector{Float64}:
  1.0
  3.0
  6.0
 10.0

julia> diffinv([2.0, 3.0, 4.0])   # xi defaults to zeros -- cumulative sum from 0
4-element Vector{Float64}:
 0.0
 2.0
 5.0
 9.0

julia> diff(diffinv([2.0, 3.0, 4.0]; xi=[1.0]), 1) == [2.0, 3.0, 4.0]
true
```
"""
function diffinv(x; lag::Integer=1, differences::Integer=1, xi=nothing)
    lag >= 1 || throw(ArgumentError("lag must be >= 1, got $lag"))
    differences >= 0 || throw(ArgumentError("differences must be >= 0, got $differences"))
    z = tsvalues(x)
    m = lag * differences
    seed = xi === nothing ? zeros(Float64, m) : tsvalues(xi)
    length(seed) == m ||
        throw(ArgumentError("xi must have length lag*differences=$m, got $(length(seed))"))

    differences == 0 && return collect(Float64, z)

    # seeds[k] = first `lag` values of the (k-1)-th order lag-`lag`
    # difference of the ORIGINAL series -- derivable directly from `seed`,
    # since that quantity depends only on the original series' first
    # lag*k values, all of which are contained in `seed` (length lag*differences).
    seeds = Vector{Vector{Float64}}(undef, differences)
    level = collect(Float64, seed)
    for k in 1:differences
        seeds[k] = level[1:lag]
        if k < differences
            level = level[(lag+1):end] .- level[1:(end-lag)]
        end
    end

    # Integrate outward: `result` starts as the differences-th order
    # series `z`; each step uses seeds[k] to recover one order less.
    result = collect(Float64, z)
    for k in differences:-1:1
        s = seeds[k]
        m2 = length(result)
        rebuilt = Vector{Float64}(undef, m2 + lag)
        rebuilt[1:lag] = s
        for i in 1:m2
            rebuilt[lag+i] = rebuilt[i] + result[i]
        end
        result = rebuilt
    end
    return result
end

"""
    tsdiff(x; lag::Integer=1, differences::Integer=1)

Alias for [`diff`](@ref)`(x, lag; differences=differences)`, spelled with
`lag` as a keyword rather than a required positional argument -- reads
more naturally than `diff(x, 1)` when `lag` is left at its default,
since bare `diff(x)` (one argument, no `lag`) calls `Base.diff` instead.

# Examples
```jldoctest
julia> using TSAnalytics

julia> tsdiff([1.0, 3.0, 6.0, 10.0])
3-element Vector{Float64}:
 2.0
 3.0
 4.0

julia> tsdiff(1.0:12.0; lag=4)
8-element Vector{Float64}:
 4.0
 4.0
 4.0
 4.0
 4.0
 4.0
 4.0
 4.0
```
"""
tsdiff(x; lag::Integer=1, differences::Integer=1) = diff(tsvalues(x), lag; differences=differences)

"""
    tsundiff(x; lag::Integer=1, differences::Integer=1, xi=nothing)

Alias for [`diffinv`](@ref)`(x; lag=lag, differences=differences, xi=xi)`.

# Examples
```jldoctest
julia> using TSAnalytics

julia> tsundiff([2.0, 3.0, 4.0]; xi=[1.0])
4-element Vector{Float64}:
  1.0
  3.0
  6.0
 10.0
```
"""
tsundiff(x; lag::Integer=1, differences::Integer=1, xi=nothing) = diffinv(x; lag=lag, differences=differences, xi=xi)
