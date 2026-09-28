# X-13 from Julia

[SeasonalAdjustment.jl](https://github.com/MSALabs/SeasonalAdjustment.jl)
is the sibling package. It wraps the actual Census Bureau
X-13ARIMA-SEATS binary — the same executable national statistical
offices run in production, via the freely redistributable
[`x13prebuilt`](https://github.com/x13org/x13prebuilt) build — and
gives it a Julia interface.

This chapter is a pointer, not a tutorial. It shows enough to make the
shape of the thing concrete, then hands off to that package's own
documentation.

!!! note "The examples here are not executed"
    Unlike every other code block in this book, the blocks below are
    not run at build time — SeasonalAdjustment.jl is not a dependency
    of these docs, and pulling the Census binary into this build would
    be the wrong trade. They are written against the real API and are
    correct at time of writing; the authoritative, executed versions
    live in that package's own documentation.

## Installation

Neither package is in Julia's General registry yet, so both go in
together — two separate `Pkg.add` calls fail in either order.

```julia
using Pkg
Pkg.add([
    PackageSpec(url="https://github.com/MSALabs/TSAnalytics.jl"),
    PackageSpec(url="https://github.com/MSALabs/SeasonalAdjustment.jl"),
])
```

The X-13 binary itself arrives as a Julia artifact. There is nothing to
install by hand and no system dependency to satisfy;
`x13_binary_available()` will tell you whether it resolved.

## The simplest possible run

```julia
using SeasonalAdjustment

res = x13(dataset("airline"))
```

`x13` builds a specification, runs the binary, parses its output, and
returns a typed `X13Result`. A bare call like this turns nothing on —
no automatic model selection, no outlier detection, no calendar
testing. It is the "what does the default do" call, not the one you
will use.

## The call you actually want

```julia
res = x13(dataset("airline");
          automdl  = true,          # let X-13 search for the ARIMA order
          outlier  = true,          # detect and classify extreme values
          aictest  = [:td, :easter],# test whether trading-day and Easter effects earn their place
          transform = :auto)        # decide log vs level from the data

arima_model(res)        # "(0 1 1)(0 1 1)"
transformfunction(res)  # :log
mstats(res).q           # 0.2
length(outliers(res))   # 1
```

Four things happened there that have no equivalent anywhere in
TSAnalytics.

**`automdl`** ran X-13's own model search and landed on
`(0 1 1)(0 1 1)` — the airline model, which is the right answer for
this series and the one [`auto_arima`](@ref) also finds. The agreement
is reassuring rather than surprising; both are implementing the same
idea.

**`aictest`** did something `auto_arima` cannot: it *tested whether
calendar regressors belong in the model*, by AIC, and included them
only if they did. Trading-day and Easter effects are pre-adjusted out
before the seasonal filters run.

**`outlier`** identified one extreme value, classified it, and reported
it. Not downweighted — named, with a type and a date, in output you can
publish.

**`mstats(res).q`** is the Q statistic, X-11's summary quality measure,
built from eleven `M` statistics each targeting a specific failure
mode. Below `1.0` conventionally means the adjustment is acceptable;
`0.2` is good. This is what "published diagnostics" means in practice —
a single number a downstream user can check without re-running
anything.

## What comes back

`X13Result` carries the standard tables under names that match the
X-13 documentation, so agency guidance transfers directly:

| Accessor | Table | Is |
|---|---|---|
| `components(res)` | D10–D13 | Seasonal, seasonally adjusted, irregular, trend |
| `udg(res)` | `.udg` | The full diagnostics dictionary |
| `qs(res)` | — | QS test for residual seasonality |
| `outliers(res)` | — | Detected outliers, typed and dated |
| `arima_model(res)` | — | The order that was selected |
| `forecast(res)` / `backcast(res)` | — | The ARIMA extension the filters used |

`series(res)` gives the input back, and `open_output(res)` opens the
binary's own text output when you need to read what it actually did.

## SEATS instead of X-11

```julia
res = x13(dataset("airline"); seats = true, automdl = true, transform = :auto)
```

SEATS derives its filters from the fitted ARIMA model rather than
applying X-11's fixed moving-average cascade. Two philosophies, one
program, the same RegARIMA pre-adjustment feeding both. Which to prefer
is a live question in the field; agencies differ, and the package
supports both because the binary does.

## The escape hatch

`x13` is the curated entry point. Underneath it,
`X13Spec`/`run_x13`/`parse_output` are exposed separately, so anything
the binary can do is reachable — a spec file written by hand, a partial
table selection, a batch run across many series. Nothing is locked
behind the convenience function.

That layering is deliberate, and it is the same shape as the rest of
this project: a good default that covers most use, with the general
machinery available when it does not.

## Where to go next

SeasonalAdjustment.jl's own documentation carries the real treatment —
specification options, every diagnostic, the plotting recipes, the
bundled datasets, and the guidance on reading a Q statistic properly.

The next chapter works the one case that motivates all of this from
inside this book's own narrative: an Indian series with a moving
festival in it, which is where the calendar machinery stops being an
abstraction.

## See also

- [Chapter 38](38-official-seasonal-adjustment.md) — why this program exists
- [Chapter 40](40-adjusting-an-indian-series.md) — the case that needs it
- [Chapter 34](34-regression-with-arima-errors.md) — RegARIMA, which is the pre-adjustment step
