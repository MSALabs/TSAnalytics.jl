# Was It Any Good?

TSAnalytics accepts any container that satisfies [`tsvalues`](@ref) — a
plain `Vector`, a `TSFrames.TSFrame` column, `TimeSeries.values(ta)`, or a
`DataFrames.jl` column all work identically. This chapter's examples just
use plain vectors.

## Checking a series before you model it

```jldoctest getting-started
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(500));  # a random walk

julia> adf_test(y).pvalue > 0.10    # fails to reject the unit-root null
true

julia> kpss_test(y).pvalue <= 0.05  # rejects level-stationarity
true
```

## Checking a series for autocorrelation

```jldoctest getting-started
julia> dy = diff(y);

julia> length(acf(dy, 0:5).values)
6

julia> ljungbox_test(dy, 10).pvalue > 0.05   # differenced series looks like white noise
true
```

## Checking a fitted model's residuals

Once you've fit a model (see [Your First Model](02-first-model.md)), the
same question applies to its residuals: [`diagnostic_plot`](@ref)
produces the standard 4-panel residual display — standardized residuals
over time, the ACF of the residuals, a normal Q-Q plot, and Ljung-Box
p-values across a range of lags:

```jldoctest getting-started
julia> resid = randn(MersenneTwister(2), 200);

julia> r = diagnostic_plot(resid);

julia> length(r.ljungbox_pvalues) == r.nlag - r.ppq
true
```

See [Plotting](../manual/10-plotting.md) for the visual, rendered version
of this and every other result type in the package.
