# Getting Started

## Installation

```julia
] add TSAnalytics
```

!!! warning "About the examples below"
    These examples were written without a Julia runtime available (see the
    project's development history) and are **not yet verified to produce
    exactly these outputs** — statistical test results on a specific random
    realization aren't fully predictable without running them, and RNG
    streams can shift across Julia versions. Run these locally first, using
    `] test` and manual REPL checks, and convert them to `jldoctest` blocks
    (with `checkdocs`/`doctest=true` already wired up in `make.jl`) once
    confirmed — at that point Documenter.jl will enforce them staying
    accurate on every future change, which is the actual point of doctests.
    Until then, treat this page as illustrative, not verified.

## A first example

TSAnalytics accepts any container that satisfies [`tsvalues`](@ref) — a
plain `Vector`, a `TSFrames.TSFrame` column, `TimeSeries.values(ta)`, or a
`DataFrames.jl` column all work identically. This example just uses plain
vectors.

```julia
using TSAnalytics, Random

Random.seed!(1)
y = cumsum(randn(500))          # a random walk

adf_test(y).pvalue > 0.10        # expected: fails to reject the unit-root null
kpss_test(y).pvalue <= 0.05      # expected: rejects level-stationarity
```

## Checking a series for autocorrelation

```julia
dy = diff(y)
r = acf(dy, 0:5)
length(r.values)                 # 6

ljungbox_test(dy, 10).pvalue > 0.05   # expected: differenced series looks like white noise
```

## Design notes worth knowing before you dig further

- **No container lock-in.** See [`tsvalues`](@ref) — every function above
  also accepts a `TSFrame`/`TimeArray`/`DataFrame` column directly.
- **Approximate p-values, by design, for now.** `adf_test` and `kpss_test`
  report p-values interpolated among asymptotic critical values, not the
  finite-sample MacKinnon response-surface p-values R/statsmodels use —
  adequate for a significant/not-significant call at 1/5/10%, tracked as a
  known limitation. See [`ADFTest`](@ref) and [`KPSSTest`](@ref).
- **Reference, never port.** Every algorithm here is implemented natively
  from its primary paper or textbook, validated against R/Python output
  numbers on standard series (Nile, AirPassengers, sunspots) — never by
  translating another package's source. See the project's
  `development-sequence.md` for the full policy and roadmap.
