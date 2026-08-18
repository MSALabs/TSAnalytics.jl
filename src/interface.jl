export tsvalues, tsindex

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
