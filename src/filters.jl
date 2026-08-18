export convolution_filter, recursive_filter, moving_average

"""
    convolution_filter(x, filt; sides::Integer=2, circular::Bool=false)

Linear filtering by convolution -- a (possibly weighted, possibly
one-sided) moving average:

    y[i] = filt[1]*x[i+o] + filt[2]*x[i+o-1] + ... + filt[p]*x[i+o-(p-1)]

where `o` is an offset determined by `sides`. Matches R's
`stats::filter(x, filt, method="convolution", sides=sides,
circular=circular)` and Python's
`statsmodels.tsa.filters.filtertools.convolution_filter(x, filt,
nsides=sides)` exactly, including the reverse-time-order convention for
`filt` -- kept deliberately (rather than flipped to a perhaps more
intuitive forward order) so results validate directly against those
reference implementations without a translation step.

- `sides = 2` (default): centered moving average, `o = length(filt) ÷ 2`.
  If `length(filt)` is even, more of the filter sits forward in time than
  backward, matching R's documented convention.
- `sides = 1`: one-sided/causal -- coefficients apply to the current value
  and the past only (`o = 0`).
- `circular = true`: wrap the filter around the ends of the series
  (an R-only feature; statsmodels has no equivalent, hence `false` as the
  default to match Python's only available behaviour). When `false`,
  positions without enough data return `NaN`, matching both references.

`x` accepts anything [`tsvalues`](@ref) does.

See also [`recursive_filter`](@ref) and [`moving_average`](@ref).

# Examples
```jldoctest
julia> using TSAnalytics

julia> isapprox(convolution_filter(1.0:5.0, [1/3, 1/3, 1/3]), [NaN, 2.0, 3.0, 4.0, NaN]; nans=true)
true

julia> isapprox(convolution_filter(1.0:5.0, [1/3, 1/3, 1/3]; sides=1), [NaN, NaN, 2.0, 3.0, 4.0]; nans=true)
true
```
"""
function convolution_filter(x, filt::AbstractVector{<:Real}; sides::Integer=2, circular::Bool=false)
    sides in (1, 2) || throw(ArgumentError("sides must be 1 or 2, got $sides"))
    xv = tsvalues(x)
    n = length(xv)
    p = length(filt)
    out = fill(NaN, n)
    o = sides == 2 ? p ÷ 2 : 0

    for i in 1:n
        acc = 0.0
        ok = true
        for j in 1:p
            idx = i + o - (j - 1)
            if circular
                idx = mod1(idx, n)
            elseif idx < 1 || idx > n
                ok = false
                break
            end
            acc += filt[j] * xv[idx]
        end
        out[i] = ok ? acc : NaN
    end
    return out
end

"""
    recursive_filter(x, ar_coeff; init=nothing)

Autoregressive (recursive/IIR) filtering, with an *implied* unit
coefficient at lag 0:

    y[i] = x[i] + ar_coeff[1]*y[i-1] + ar_coeff[2]*y[i-2] + ... + ar_coeff[p]*y[i-p]

Matches R's `stats::filter(x, ar_coeff, method="recursive", init=init)`
and Python's `statsmodels.tsa.filters.filtertools.recursive_filter(x,
ar_coeff, init=init)` exactly, including the reverse-time-order
convention for both `ar_coeff` and `init`.

`init` supplies the values of `y` just prior to the start of the series,
in reverse time order (`init[1]` = y at time 0, i.e. immediately before
the first observation; `init[2]` = y at time -1; ...). Defaults to zeros.
No stability check is performed -- as in R, the output may diverge if
`ar_coeff` doesn't correspond to a stable/invertible filter.

`x` and `init` both accept anything [`tsvalues`](@ref) does.

See also [`convolution_filter`](@ref).

# Examples
```jldoctest
julia> using TSAnalytics

julia> recursive_filter(ones(5), [0.5])
5-element Vector{Float64}:
 1.0
 1.5
 1.75
 1.875
 1.9375
```
"""
function recursive_filter(x, ar_coeff::AbstractVector{<:Real}; init=nothing)
    xv = tsvalues(x)
    n = length(xv)
    p = length(ar_coeff)
    initv = init === nothing ? zeros(Float64, p) : tsvalues(init)
    length(initv) == p || throw(ArgumentError("init must have the same length as ar_coeff, got $(length(initv)) and $p"))

    out = Vector{Float64}(undef, n)
    yprev = collect(Float64, initv)   # yprev[k] holds y[i-k] at the start of step i

    for i in 1:n
        acc = xv[i]
        for k in 1:p
            acc += ar_coeff[k] * yprev[k]
        end
        out[i] = acc
        for k in p:-1:2
            yprev[k] = yprev[k-1]
        end
        p >= 1 && (yprev[1] = acc)
    end
    return out
end

"""
    moving_average(x, order::Integer; centre::Bool=true)

Simple moving average smoother -- the convenience entry point most users
actually reach for, built directly on [`convolution_filter`](@ref) rather
than duplicating its logic. Matches R's `forecast::ma(x, order,
centre=TRUE)` exactly (name and formula, including the R spelling of
`centre`):

    T̂ₜ = (1/m) * sum(y[t+j] for j in -k:k),   k = (m-1)/2

- **Odd `order`**: a plain centered `order`-length uniform-weight filter.
- **Even `order`, `centre=false`**: a plain `order`-length uniform-weight
  filter (asymmetric -- includes one more future observation than past,
  matching R's documented behaviour for this case).
- **Even `order`, `centre=true`** (the default, and the case that matters
  for seasonal decomposition with an even period like 12): the classic
  "2×`order`" double moving average -- a single `(order+1)`-length filter
  with half-weighted endpoints `[1/(2m), 1/m, ..., 1/m, 1/(2m)]`. This is
  the exact filter X-11-style seasonal decomposition uses for even
  periods, which is why it's the default here despite plain rolling
  means elsewhere (e.g. `pandas.Series.rolling`) not doing this.

`x` accepts anything [`tsvalues`](@ref) does. Positions without enough
surrounding data return `NaN`, same as [`convolution_filter`](@ref).

# Examples
```jldoctest
julia> using TSAnalytics

julia> isapprox(moving_average(1.0:12.0, 3),   # odd order
                 [NaN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, NaN]; nans=true)
true

julia> isapprox(moving_average(1.0:12.0, 4),   # even order, centred 2x4 MA (default)
                 [NaN, NaN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, NaN, NaN]; nans=true)
true
```
"""
function moving_average(x, order::Integer; centre::Bool=true)
    order >= 1 || throw(ArgumentError("order must be >= 1, got $order"))
    if isodd(order) || !centre
        filt = fill(1 / order, order)
    else
        filt = vcat(1 / (2*order), fill(1 / order, order - 1), 1 / (2*order))
    end
    return convolution_filter(x, filt; sides=2)
end
