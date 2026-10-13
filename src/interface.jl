export tsvalues, tsindex, tsmatrix, tsnames

"""
    tsvalues(x) -> AbstractVector{<:Real}

Extract a plain, dense numeric vector from `x`. TSAnalytics does not depend
on, or write integration code for, any particular time series container.
Instead, every public function accepts anything `tsvalues` can be called on:

- Any `AbstractVector{<:Real}` -- returned as-is, zero copy.
- Any other iterable -- a range, a generator, a `TSFrames.TSFrame` column,
  `TimeSeries.values(ta)`, a `DataFrames.DataFrame` column, or a container
  from a package that doesn't exist yet -- materialized via `collect` into
  a `Vector{Float64}`.

If you're working with a structured container, extract the column with
that container's own native accessor and hand the result to TSAnalytics.
There is deliberately no TSAnalytics-specific integration code for any of
them -- each of the following already works today, unmodified:

```julia
using TSAnalytics

# Plain vector
acf(randn(200))

# TSFrames.jl -- column indexing already returns a plain Vector
using TSFrames
tsf = TSFrame(...)
acf(tsf[:, :Close])

# TimeSeries.jl -- values(ta) is already a Vector/Matrix
using TimeSeries
ta = TimeArray(...)
acf(values(ta))

# DataFrames.jl
using DataFrames
df = DataFrame(...)
acf(df.Close)

# A lazy iterator, or a container from a package written after this one
acf(Iterators.map(x -> x^2, 1:100))
```

For genuinely custom behaviour (e.g. avoiding a copy for your own storage
format), define a method on your own type:

    TSAnalytics.tsvalues(x::YourType) = ...   # return an AbstractVector{<:Real}

This is ordinary Julia multiple dispatch -- no change to TSAnalytics, no
extension mechanism, no new dependency in either direction. Container
authors can add this method in *their* package if they want first-class
support, but nothing requires it.
"""
tsvalues(x::AbstractVector{<:Real}) = x
tsvalues(x) = collect(Float64, x)

"""
    tsmatrix(x) -> AbstractMatrix{<:Real}

Extract a plain numeric matrix from `x`, **rows = time, columns =
variables**. The multivariate companion to [`tsvalues`](@ref), and the
entry point every Stage 10 model uses.

Accepts:

- Any `AbstractMatrix{<:Real}` — returned as-is, zero copy.
- A `NamedTuple` of equal-length numeric columns — which is exactly what
  [`dataset`](@ref) returns for its multivariate entries, so
  `fit_var(dataset("econ5"), 2)` works with no unpacking. Column order is
  the `NamedTuple`'s own, and [`tsnames`](@ref) recovers the names.
- A `Tuple` or `AbstractVector` of equal-length numeric vectors, treated
  as columns.
- Anything else `collect` turns into a matrix.

## Why this is not more `tsvalues` methods

`tsvalues`' documented contract is `AbstractVector{<:Real}`, and the
container-agnostic promise is vector-shaped by construction. Adding
matrix methods to it would make that docstring false.

There is a sharper reason. `tsvalues`' fallback is
`collect(Float64, x)`, and on a matrix `collect` returns a **matrix** —
so `tsvalues(rand(2, 2))` already returns a `Matrix{Float64}` today,
silently violating its own stated return type rather than erroring. A
separate name fixes that by not pretending the two cases are one.

```jldoctest
julia> using TSAnalytics

julia> tsmatrix([1.0 4.0; 2.0 5.0; 3.0 6.0])
3×2 Matrix{Float64}:
 1.0  4.0
 2.0  5.0
 3.0  6.0

julia> tsmatrix((a = [1.0, 2.0], b = [3.0, 4.0]))
2×2 Matrix{Float64}:
 1.0  3.0
 2.0  4.0

julia> tsnames((a = [1.0, 2.0], b = [3.0, 4.0]))
2-element Vector{Symbol}:
 :a
 :b
```

See also [`tsvalues`](@ref), [`tsnames`](@ref).
"""
tsmatrix(x::AbstractMatrix{<:Real}) = x

function tsmatrix(x::NamedTuple)
    isempty(x) && throw(ArgumentError("tsmatrix: the NamedTuple has no columns"))
    cols = [tsvalues(c) for c in values(x)]
    return _tsmatrix_from_columns(cols, "NamedTuple")
end

function tsmatrix(x::Union{Tuple,AbstractVector})
    isempty(x) && throw(ArgumentError("tsmatrix: no columns given"))
    all(c -> c isa Union{AbstractVector,Tuple}, x) ||
        throw(ArgumentError("tsmatrix: expected a collection of column vectors, " *
                            "got elements of type $(typeof(first(x)))"))
    cols = [tsvalues(c) for c in x]
    return _tsmatrix_from_columns(cols, "collection of vectors")
end

function tsmatrix(x)
    m = collect(x)
    m isa AbstractMatrix{<:Real} && return Float64.(m)
    throw(ArgumentError("tsmatrix: cannot read $(typeof(x)) as an " *
                        "observations × variables matrix. Pass a matrix, a " *
                        "NamedTuple of columns, or a vector of column vectors."))
end

function _tsmatrix_from_columns(cols::AbstractVector, what::AbstractString)
    n = length(first(cols))
    all(c -> length(c) == n, cols) || throw(DimensionMismatch(
        "tsmatrix: the $what has columns of unequal length " *
        "$(Tuple(length(c) for c in cols))"))
    n > 0 || throw(ArgumentError("tsmatrix: the columns are empty"))
    M = Matrix{Float64}(undef, n, length(cols))
    @inbounds for (j, c) in enumerate(cols)
        for i in 1:n
            M[i, j] = c[i]
        end
    end
    return M
end

"""
    tsnames(x) -> Union{Nothing, Vector{Symbol}}

The variable names `x` carries, or `nothing` when it carries none.

Names are not cosmetic in the multivariate models: an impulse response,
a forecast error variance decomposition and a Granger causality result
are all indexed *by variable*, and both references label their output in
both directions — `vars::irf()` returns `$(")")irf$(")")y2[, "y1"]`. A
model that loses the names makes its own output unreadable.

Returns the keys of a `NamedTuple`, and `nothing` for a bare matrix or a
vector of vectors. Callers default to `[:y1, :y2, …]` when this is
`nothing`.

See also [`tsmatrix`](@ref).
"""
tsnames(x::NamedTuple) = collect(Symbol, keys(x))
tsnames(::Any) = nothing

"""
    tsindex(x) -> Union{Nothing, AbstractVector}

Extract a paired timestamp/index vector from `x`, if `x` carries one
natively. Defaults to `nothing`, meaning "use `1:n`". Most containers don't
make this discoverable from the numeric column alone -- once you've sliced
a `TSFrame` or `DataFrame` down to a single column, the dates live on the
parent object, not the column -- so functions that accept a paired index
(seasonal-period inference, forecast-horizon labeling) take it via an
explicit `index=` keyword rather than relying on this method, e.g.:

```julia
acf(tsf[:, :Close]; index = TSFrames.index(tsf))
acf(values(ta);      index = timestamp(ta))
acf(df.Close;        index = df.Date)
```

Define a `tsindex` method on your own combined value+timestamp type if you
want it picked up automatically instead.
"""
tsindex(x) = nothing
