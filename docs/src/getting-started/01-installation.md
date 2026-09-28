# Installation

```julia
] add TSAnalytics
```

Julia 1.9 or later.

## Checking it works

```jldoctest installation
julia> using TSAnalytics

julia> y = dataset("cardox").value;   # monthly CO₂ at Mauna Loa

julia> length(y)
781

julia> round(acf(y, 1:1).values[1], digits=3)
0.995
```

If that runs, everything in this documentation will run. The
`dataset` call is reading a real series bundled with the package —
84 of them ship, so most examples here need no download and no setup.

## What you do not need

**No time series container.** This is the first thing that
distinguishes the package, and it is worth stating before you write any
code: TSAnalytics does not define a time series type, does not require
one, and does not integrate with any particular one.

Every public function accepts anything [`tsvalues`](@ref) can be called
on, which by default means anything iterable:

```jldoctest installation
julia> using TSAnalytics

julia> acf([1.0, 3.0, 2.0, 5.0, 4.0, 7.0], 1:2).values |> length
2

julia> acf(1:10, 1:2).values |> length
2
```

A plain `Vector` works. So does a `TSFrames.TSFrame` column, a
`TimeSeries.TimeArray`'s `values(ta)`, or a `DataFrames.jl` column —
not because the package has integration code for any of them, but
because each of those already hands you a plain `Vector` through its
*own* accessor before TSAnalytics sees anything. There is no adapter
layer to learn, and nothing to convert.

The practical consequence: you can adopt this package without adopting
an ecosystem, and you can drop it without unpicking one.

## Optional companions

| Package | Why | Needed? |
|---|---|---|
| `Plots.jl` | Every result type has a plot recipe — `plot(acf(y, 1:20))`, `plot(stl_decompose(y, 12))`, `plot(diagnostic_plot(m))` | Only if you want pictures |
| `DataFrames.jl` | `dataset("x", DataFrame)` materialises into one | Only if you already use it |

Nothing else is required. The package's own dependencies are
deliberately minimal — see the "no unused dependencies" policy in
`CLAUDE.md` — and it pulls in no plotting stack of its own: the
recipes are declared through `RecipesBase`, which means they cost you
nothing until you load `Plots` yourself.

## Where to go next

[Your First Model](02-first-model.md) fits something in about fifteen
lines. If you would rather see the design reasoning first,
[Where Next](05-where-next.md) is a map of the rest of the
documentation.
