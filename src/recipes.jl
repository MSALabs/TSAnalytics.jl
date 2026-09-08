# ---------------------------------------------------------------------------
# RecipesBase.jl plotting recipes -- deliberately a soft dependency, not
# Plots.jl/Makie.jl itself (matches how Distributions.jl and R/Python's own
# generic plot()/autoplot() dispatch work): `plot(result)` works for any
# caller with a plotting backend loaded, without this package taking on a
# real plotting dependency. `periodogram`/`spectral_density` (Part A)
# originally returned plain NamedTuples with no recipe (a bare
# `NamedTuple` recipe would apply far too broadly -- it would hijack
# plotting for *any* unrelated NamedTuple in a session with TSAnalytics
# loaded). `PeriodogramResult` (`src/spectral.jl`) replaced that
# NamedTuple specifically so a real recipe could be added below --
# see `handoff/periodogram-recipe-handoff.md`.
# ---------------------------------------------------------------------------

export seasonal_subseries_plot

"""
    seasonal_subseries_plot(x, period) -> (values, position, cycle, means)

Data behind fpp3's own signature "subseries plot" (`feasts::gg_subseries()`):
for each of the `period` positions within a seasonal cycle (e.g. each
calendar month, for `period=12`), the subseries of `x` at that position
across cycles, together with that position's own mean -- the classic
"how has each month's value moved across years" view.

`position[i]` is `x[i]`'s 1-based position within its cycle (`1:period`,
wrapping); `cycle[i]` is which cycle (year) `x[i]` belongs to;
`means[p]` is the mean of all `values[i]` with `position[i] == p`.
Plot by grouping `values` by `position` into `period` panels/lines, each
with a horizontal reference line at its own `means[p]` -- deliberately a
plain NamedTuple, not a dedicated result type/recipe: unlike
[`periodogram`](@ref) (a single fixed display, `freq` vs `spec`), this
one has no one natural rendering (panels? overlaid lines? which axis is
which?), so a recipe would just be encoding one opinionated choice among
several equally valid ones.

`x` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics

julia> x = 100 .+ 10 .* sin.(2π .* (1:96) ./ 12) .+ (1:96) ./ 10;

julia> r = seasonal_subseries_plot(x, 12);

julia> length(r.means) == 12
true

julia> r.position[1:14]
14-element Vector{Int64}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
  1
  2
```
"""
function seasonal_subseries_plot(x, period::Integer)
    period >= 2 || throw(ArgumentError("seasonal_subseries_plot: period must be >= 2"))
    xv = Float64.(collect(tsvalues(x)))
    n = length(xv)
    n >= period || throw(ArgumentError("seasonal_subseries_plot: series shorter than period"))
    position = [((i - 1) % period) + 1 for i in 1:n]
    cycle = [((i - 1) ÷ period) + 1 for i in 1:n]
    means = [mean(xv[position.==p]) for p in 1:period]
    return (values=xv, position=position, cycle=cycle, means=means)
end

"""
    @recipe f(r::ACFResult)

Bar/stem plot of an [`ACFResult`](@ref) (from [`acf`](@ref) or
[`pacf`](@ref)) with its pointwise white-noise confidence band shaded
and a zero reference line -- the standard ACF/PACF display.
"""
@recipe function f(r::ACFResult)
    legend --> false
    xlabel --> "Lag"
    ylabel --> (r.kind == :pacf ? "Partial Autocorrelation" : "Autocorrelation")

    @series begin
        seriestype := :line
        fillrange := r.upper
        fillalpha := 0.15
        linealpha := 0
        color := :steelblue
        label := ""
        r.lags, r.lower
    end
    @series begin
        seriestype := :hline
        linestyle := :dash
        color := :gray
        label := ""
        [0.0]
    end
    seriestype --> :sticks
    marker --> :circle
    r.lags, r.values
end

"_decimate_local_max(x, y, maxpoints) -- for plotting only: if
`length(x) > maxpoints`, partitions into `maxpoints` bins and keeps the
`(x,y)` pair with the *largest* `y` in each bin, never a naive
fixed-stride subsample. A narrow spectral peak can fall between
strided sample indices and be silently hidden by naive downsampling;
local-max retention always keeps whichever point is the single largest
in its bin, so the global peak (the largest point overall, hence the
largest in whichever bin it falls into) is always preserved exactly."
function _decimate_local_max(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, maxpoints::Integer)
    n = length(x)
    n <= maxpoints && return collect(x), collect(y)
    binsize = n / maxpoints
    xout = Vector{eltype(x)}(undef, maxpoints)
    yout = Vector{eltype(y)}(undef, maxpoints)
    for b in 1:maxpoints
        lo = floor(Int, (b - 1) * binsize) + 1
        hi = max(lo, min(n, floor(Int, b * binsize)))
        idx = lo - 1 + argmax(view(y, lo:hi))
        xout[b] = x[idx]
        yout[b] = y[idx]
    end
    return xout, yout
end

"""
    @recipe f(r::PeriodogramResult)

The log-scale periodogram plot -- `spec.pgram(plot=TRUE)`'s default
display in base R, and `astsa::mvspec`'s own standard output (Shumway
& Stoffer, *Time Series Analysis and Its Applications*, Chapter 4).
Shows spectral power against frequency on a log10 y-axis, the
conventional scale for spectral estimates given their typically large
dynamic range. For `r.kind == :spectral_density`, the title also shows
`df`/`bandwidth` -- a user comparing against R's own `spec.pgram`
output needs these visible, not just reachable on the struct.

Series with more than 10,000 plotted points are decimated via
[`_decimate_local_max`](@ref) before plotting (display speed only --
[`periodogram`](@ref)/[`spectral_density`](@ref) themselves are
unaffected); this always preserves the true spectral peak exactly,
unlike a naive strided downsample.

# Example
```julia
julia> x = 5 .* sin.(2π .* (1:96) ./ 12) .+ randn(96) .* 0.5;

julia> r = periodogram(x; taper=0.1);

julia> plot(r)  # peak visible at frequency 1/12, matching the injected periodicity
```

See also [`periodogram`](@ref), [`spectral_density`](@ref) for the
underlying computation, verified directly against base R's `spec.pgram`
(see `handoff/stage-1-fillgap-handoff.md`).
"""
@recipe function f(r::PeriodogramResult)
    freq, spec = length(r.freq) > 10_000 ? _decimate_local_max(r.freq, r.spec, 10_000) : (r.freq, r.spec)
    seriestype --> :path
    yscale --> :log10
    xlabel --> "frequency"
    ylabel --> "spectrum"
    legend --> false
    title --> (r.kind == :spectral_density ?
               "Smoothed periodogram (df=$(round(r.df, digits=2)), bandwidth=$(round(r.bandwidth, digits=5)))" :
               "Periodogram")
    freq, spec
end

"""
    @recipe f(d::ClassicalDecomposition)
    @recipe f(d::STLDecomposition)
    @recipe f(d::MSTLDecomposition)

Stacked observed/trend/seasonal/remainder panels sharing a common x-axis
-- the standard `plot(decompose(...))` display R's own `plot.decomposed.ts`
and Python's `DecomposeResult.plot()` both produce. [`MSTLDecomposition`](@ref)
gets one seasonal panel per period in `periods`, since it has more than
one seasonal component.
"""
@recipe function f(d::ClassicalDecomposition)
    layout --> (4, 1)
    legend --> false
    link --> :x
    t = 1:length(d.observed)
    @series begin
        subplot := 1
        ylabel := "observed"
        t, d.observed
    end
    @series begin
        subplot := 2
        ylabel := "trend"
        t, d.trend
    end
    @series begin
        subplot := 3
        ylabel := "seasonal"
        t, d.seasonal
    end
    @series begin
        subplot := 4
        ylabel := "remainder"
        seriestype := :scatter
        markersize := 2
        t, d.resid
    end
end

@recipe function f(d::STLDecomposition)
    layout --> (4, 1)
    legend --> false
    link --> :x
    t = 1:length(d.observed)
    @series begin
        subplot := 1
        ylabel := "observed"
        t, d.observed
    end
    @series begin
        subplot := 2
        ylabel := "trend"
        t, d.trend
    end
    @series begin
        subplot := 3
        ylabel := "seasonal"
        t, d.seasonal
    end
    @series begin
        subplot := 4
        ylabel := "remainder"
        seriestype := :scatter
        markersize := 2
        t, d.resid
    end
end

@recipe function f(d::MSTLDecomposition)
    nperiods = length(d.periods)
    layout --> (2 + nperiods + 1, 1)
    legend --> false
    link --> :x
    t = 1:length(d.observed)
    @series begin
        subplot := 1
        ylabel := "observed"
        t, d.observed
    end
    @series begin
        subplot := 2
        ylabel := "trend"
        t, d.trend
    end
    for (i, p) in enumerate(d.periods)
        @series begin
            subplot := 2 + i
            ylabel := "seasonal($p)"
            t, view(d.seasonal, :, i)
        end
    end
    @series begin
        subplot := 2 + nperiods + 1
        ylabel := "remainder"
        seriestype := :scatter
        markersize := 2
        t, d.resid
    end
end

"""
    @recipe f(r::DiagnosticPlotResult)

The standard 4-panel residual diagnostic display (from
[`diagnostic_plot`](@ref)): standardized residuals over time, ACF of
the residuals, a normal Q-Q plot with the `y=x` reference line, and
Ljung-Box p-values across lags with a `0.05` reference line -- matching
`astsa::sarima()`'s/`SARIMAXResults.plot_diagnostics()`'s confirmed
real structure.
"""
@recipe function f(r::DiagnosticPlotResult)
    layout --> (2, 2)
    legend --> false

    @series begin
        subplot := 1
        seriestype := :line
        title := "Standardized Residuals"
        1:length(r.std_resid), r.std_resid
    end

    @series begin
        subplot := 2
        seriestype := :sticks
        marker := :circle
        title := "ACF of Residuals"
        xlabel := "Lag"
        r.acf_lags, r.acf
    end

    @series begin
        subplot := 3
        seriestype := :scatter
        markersize := 3
        title := "Normal Q-Q Plot"
        xlabel := "Theoretical Quantiles"
        ylabel := "Sample Quantiles"
        r.qq_theoretical, r.qq_sample
    end
    @series begin
        subplot := 3
        seriestype := :line
        linestyle := :dash
        color := :gray
        lo, hi = extrema(r.qq_theoretical)
        [lo, hi], [lo, hi]
    end

    @series begin
        subplot := 4
        seriestype := :scatter
        markersize := 3
        title := "p-values for Ljung-Box statistic"
        xlabel := "Lag"
        ylims := (0.0, 1.0)
        r.lb_lags, r.ljungbox_pvalues
    end
    @series begin
        subplot := 4
        seriestype := :hline
        linestyle := :dash
        color := :blue
        [0.05]
    end
end
