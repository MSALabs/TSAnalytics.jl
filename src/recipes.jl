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
    ylabel --> (r.kind == :pacf ? "Partial Autocorrelation" :
                 r.kind == :ccf  ? "Cross-Correlation" : "Autocorrelation")

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
               r.kind == :spec_ar ? "AR spectrum" : "Periodogram")
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

"""
    @recipe f(r::GarchDiagnosticPlotResult)

Six-panel variance-model diagnostic display: conditional standard
deviation against `|resid|`, standardized residuals, their ACF, the ACF
of their squares, a normal Q-Q plot, and the news-impact curve. The
squared-residual ACF is the panel with no counterpart in the ARIMA-side
display, and the one that says whether the variance model actually
worked.

The news-impact panel is omitted for `model=:egarch`, whose curve lives
on a log-variance scale (see [`GarchDiagnosticPlotResult`](@ref)); the
layout drops to five panels in that case rather than drawing an empty
frame.
"""
@recipe function f(r::GarchDiagnosticPlotResult)
    has_nic = r.news_impact_e !== nothing
    layout --> (2, 3)
    legend --> false
    # six panels need more room than Plots' 600x400 default, or every title clips
    size --> (1000, 650)
    titlefontsize --> 9

    @series begin
        subplot := 1
        seriestype := :line
        title := "Conditional SD vs |resid|"
        color := :gray
        alpha := 0.45
        1:length(r.abs_resid), r.abs_resid
    end
    @series begin
        subplot := 1
        seriestype := :line
        linewidth := 1.5
        title := "Conditional SD vs |resid|"
        1:length(r.sigma), r.sigma
    end

    @series begin
        subplot := 2
        seriestype := :line
        title := "Standardized Residuals"
        1:length(r.std_resid), r.std_resid
    end

    @series begin
        subplot := 3
        seriestype := :sticks
        marker := :circle
        title := "ACF of Std. Residuals"
        xlabel := "Lag"
        r.acf_lags, r.acf
    end

    @series begin
        subplot := 4
        seriestype := :sticks
        marker := :circle
        title := "ACF of Squared Std. Residuals"
        xlabel := "Lag"
        r.acf_lags, r.acf_sq
    end

    @series begin
        subplot := 5
        seriestype := :scatter
        markersize := 3
        title := "Normal Q-Q Plot"
        xlabel := "Theoretical Quantiles"
        ylabel := "Sample Quantiles"
        r.qq_theoretical, r.qq_sample
    end
    @series begin
        subplot := 5
        seriestype := :line
        linestyle := :dash
        color := :gray
        title := "Normal Q-Q Plot"
        lo, hi = extrema(r.qq_theoretical)
        [lo, hi], [lo, hi]
    end

    if has_nic
        @series begin
            subplot := 6
            seriestype := :line
            linewidth := 1.5
            title := "News Impact Curve"
            xlabel := "shock"
            ylabel := "next conditional variance"
            r.news_impact_e, r.news_impact_sigma2
        end
    end
end

"_decimate_minmax(x, y, maxpoints) -- for plotting a time path only: if
`length(x) > maxpoints`, splits it into `maxpoints ÷ 2` bins and keeps both
the smallest and the largest point of each bin, in time order. A line
chart needs its troughs as much as its peaks, which is why this is not
`_decimate_local_max` (which keeps maxima only, right for a spectrum and
wrong for a series): either extreme of any bin is always drawn."
function _decimate_minmax(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, maxpoints::Integer)
    n = length(x)
    n <= maxpoints && return collect(x), collect(y)
    nb = max(1, maxpoints ÷ 2)
    binsize = n / nb
    xo = Vector{eltype(x)}(); yo = Vector{eltype(y)}()
    sizehint!(xo, 2nb); sizehint!(yo, 2nb)
    for b in 1:nb
        lo = floor(Int, (b - 1) * binsize) + 1
        hi = max(lo, min(n, floor(Int, b * binsize)))
        v = view(y, lo:hi)
        i1 = lo - 1 + argmin(v); i2 = lo - 1 + argmax(v)
        for i in (i1 <= i2 ? (i1, i2) : (i2, i1))
            push!(xo, x[i]); push!(yo, y[i])
        end
    end
    return xo, yo
end

const _PLOT_MAXPOINTS = 5_000

# One forecast fan on the current subplot: one ribbon per level, widest
# first so narrower ones draw on top, then (optionally) the history, then
# the point path. Shared by the Forecast and VARForecast recipes.
function _fan_series!(series, t, point, lower, upper, levels, history; sp=1, labels=true)
    order = sortperm(levels; rev=true)
    for (k, j) in enumerate(order)
        push!(series, (; subplot=sp, seriestype=:line, fillrange=upper[:, j],
                       fillalpha=0.12 + 0.10 * (k - 1), linealpha=0.0, color=:steelblue,
                       label=labels ? string(Int(round(levels[j])), "%") : "",
                       x=t, y=lower[:, j]))
    end
    if history !== nothing
        n0 = length(history)
        hx, hy = _decimate_minmax(collect(1:n0), history, _PLOT_MAXPOINTS)
        push!(series, (; subplot=sp, seriestype=:line, fillrange=nothing, fillalpha=nothing,
                       linealpha=1.0, color=:black, label=labels ? "observed" : "",
                       x=hx, y=hy))
    end
    push!(series, (; subplot=sp, seriestype=:line, fillrange=nothing, fillalpha=nothing,
                   linealpha=1.0, color=:steelblue, label=labels ? "forecast" : "",
                   x=t, y=point))
    return series
end

_history_values(h) = h === nothing ? nothing : Float64.(collect(tsvalues(h)))

"""
    @recipe f(fc::Forecast)

Fan chart of a [`Forecast`](@ref): the point forecast with one shaded
ribbon per prediction-interval level, widest outermost. Pass the series
it came from as `history` to draw it to the left of the forecast origin,
which is what makes a fan chart readable:

    plot(fc; history = y)

Without `history` the x-axis is the horizon `1 … h`; with it, the series
sits at `1 … n` and the forecast at `n+1 … n+h`. Long histories are
thinned for display by min-max decimation (both extremes of every bin
are kept). `[S]` Matches R's `forecast:::plot.forecast`; Python has no
model-level equivalent (`plot_predict` needs a pandas index and draws one
band). One recipe serves every forecasting model in the package.
"""
@recipe function f(fc::Forecast)
    history = _history_values(pop!(plotattributes, :history, nothing))
    legend --> :topleft
    title --> fc.model_name
    n0 = history === nothing ? 0 : length(history)
    xlabel --> (history === nothing ? "horizon" : "")
    t = collect((n0 + 1):(n0 + fc.horizon))
    for s in _fan_series!(NamedTuple[], t, fc.point, fc.lower, fc.upper, fc.levels, history)
        @series begin
            seriestype := s.seriestype
            s.fillrange === nothing || (fillrange := s.fillrange)
            s.fillalpha === nothing || (fillalpha := s.fillalpha)
            linealpha := s.linealpha
            color --> s.color
            label := s.label
            s.x, s.y
        end
    end
end

"""
    @recipe f(fc::VARForecast)

One fan chart per variable, stacked with a shared x-axis — the
[`Forecast`](@ref) recipe's fan, from the same helper. `history` takes
the data matrix (or anything [`tsmatrix`](@ref) accepts). `[S]` R's
`plot.varprd`/`vars::fanchart`; `statsmodels`' `plot_forecast`.
"""
@recipe function f(fc::VARForecast)
    hist = pop!(plotattributes, :history, nothing)
    H = hist === nothing ? nothing : Float64.(tsmatrix(hist))
    K = length(fc.names)
    H === nothing || size(H, 2) == K || throw(ArgumentError(
        "plot(::VARForecast): history has $(size(H, 2)) columns for $K variables"))
    layout --> (K, 1)
    link --> :x
    legend --> false
    n0 = H === nothing ? 0 : size(H, 1)
    t = collect((n0 + 1):(n0 + fc.horizon))
    for i in 1:K
        h = H === nothing ? nothing : H[:, i]
        for s in _fan_series!(NamedTuple[], t, fc.point[:, i], fc.lower[:, i, :],
                              fc.upper[:, i, :], fc.levels, h; sp=i, labels=false)
            @series begin
                subplot := i
                title := string(fc.names[i])
                seriestype := s.seriestype
                s.fillrange === nothing || (fillrange := s.fillrange)
                s.fillalpha === nothing || (fillalpha := s.fillalpha)
                linealpha := s.linealpha
                color --> s.color
                s.x, s.y
            end
        end
    end
end

"""
    @recipe f(vf::VolatilityForecast)

The forecast conditional **standard deviation** `σ` — the square root of
the stored variance, labelled as such — across the horizon. When the
forecast was simulated (`variance_paths` present), a shaded 5–95% band of
the simulated `σ` paths is drawn behind it; the analytic forecast has no
interval, and none is invented. Pass `history` (e.g. the fitted
conditional standard deviations) to draw it before the forecast origin.
"""
@recipe function f(vf::VolatilityForecast)
    history = _history_values(pop!(plotattributes, :history, nothing))
    legend --> :topleft
    title --> "volatility forecast ($(vf.method))"
    ylabel --> "σ (conditional standard deviation)"
    n0 = history === nothing ? 0 : length(history)
    t = collect((n0 + 1):(n0 + vf.horizon))
    if vf.variance_paths !== nothing
        S = sqrt.(max.(vf.variance_paths, 0.0))
        lo = [quantile(view(S, :, h), 0.05) for h in 1:vf.horizon]
        hi = [quantile(view(S, :, h), 0.95) for h in 1:vf.horizon]
        @series begin
            seriestype := :line
            fillrange := hi
            fillalpha := 0.2
            linealpha := 0.0
            color := :steelblue
            label := "5–95% of simulated paths"
            t, lo
        end
    end
    if history !== nothing
        @series begin
            seriestype := :line
            color := :black
            label := "observed"
            _decimate_minmax(collect(1:n0), history, _PLOT_MAXPOINTS)
        end
    end
    @series begin
        seriestype := :line
        color := :steelblue
        linewidth --> 2
        label := "forecast σ"
        t, sqrt.(max.(vf.variance, 0.0))
    end
end

"""
    @recipe f(d::StateDecomposition)

Stacked panels sharing the x-axis: observed, level, slope (if any), and
one season panel per seasonal period — `[S]` the panels of R's
`plot.ets`/`plot.tbats`, titled "Decomposition by …" as R does.
"""
@recipe function f(d::StateDecomposition)
    npan = 2 + (d.slope !== nothing) + length(d.seasonal)
    layout --> (npan, 1)
    legend --> false
    link --> :x
    plot_title --> "Decomposition by " * d.model_name
    plot_titlefontsize --> 11
    panels = Tuple{String,Vector{Float64}}[("observed", d.observed), ("level", d.level)]
    d.slope === nothing || push!(panels, ("slope", d.slope))
    for (i, s) in enumerate(d.seasonal)
        lab = length(d.seasonal) == 1 ? "season" :
              "season $(isinteger(d.periods[i]) ? Int(d.periods[i]) : d.periods[i])"
        push!(panels, (lab, s))
    end
    for (k, (lab, v)) in enumerate(panels)
        @series begin
            subplot := k
            seriestype := :line
            ylabel := lab
            _decimate_minmax(d.t, v, _PLOT_MAXPOINTS)
        end
    end
end

"""
    @recipe f(r::ARMARootsResult)

The inverse AR and MA roots against the unit circle, equal axis scales —
`[S]` R's `plot.Arima`/`autoplot.Arima`. A polynomial with no roots emits
no series at all, rather than an empty one.
"""
@recipe function f(r::ARMARootsResult)
    aspect_ratio --> :equal
    legend --> :topright
    title --> "Inverse roots: $(r.model_name)"
    xlabel --> "real"
    ylabel --> "imaginary"
    @series begin
        seriestype := :path          # :line would sort by x and zigzag across the circle
        color := :gray
        linestyle := :dash
        label := ""
        θ = range(0, 2pi; length=361)
        cos.(θ), sin.(θ)
    end
    if !isempty(r.ar_inverse)
        @series begin
            seriestype := :scatter
            markershape := :circle
            label := "AR"
            real.(r.ar_inverse), imag.(r.ar_inverse)
        end
    end
    if !isempty(r.ma_inverse)
        @series begin
            seriestype := :scatter
            markershape := :utriangle
            label := "MA"
            real.(r.ma_inverse), imag.(r.ma_inverse)
        end
    end
end

"""
    @recipe f(d::TSDisplayResult)

The series across the top, its ACF and PACF side by side beneath —
`[S]` R's `forecast::ggtsdisplay`. Each correlogram is drawn exactly as
the [`ACFResult`](@ref) recipe draws it (band, zero line, sticks).
"""
@recipe function f(d::TSDisplayResult)
    layout --> @layout [a{0.5h}; b c]
    legend --> false
    @series begin
        subplot := 1
        seriestype := :line
        _decimate_minmax(d.t, d.x, _PLOT_MAXPOINTS)
    end
    for (sp, r) in ((2, d.acf), (3, d.pacf))
        @series begin
            subplot := sp
            seriestype := :line
            fillrange := r.upper
            fillalpha := 0.15
            linealpha := 0
            color := :steelblue
            r.lags, r.lower
        end
        @series begin
            subplot := sp
            seriestype := :hline
            linestyle := :dash
            color := :gray
            [0.0]
        end
        @series begin
            subplot := sp
            seriestype := :sticks
            marker := :circle
            ylabel := (r.kind == :pacf ? "PACF" : "ACF")
            r.lags, r.values
        end
    end
end

"""
    @recipe f(r::IRFResult)

A `responses × impulses` grid (row = response, column = impulse) of
impulse responses against horizon, titled `impulse → response`, with a
zero line. The orthogonalized (or structural) responses are drawn. When
the result carries uncertainty its band is shaded and **labelled by
method** — a bootstrap percentile band, or `±z·se` from the analytic
standard errors — since the two differ and a bare "95%" would hide which
one it is. `[S]` The one-figure grid is `statsmodels`' `IRAnalysis.plot`
layout; R's `plot.varirf` pages one impulse at a time.
"""
@recipe function f(r::IRFResult)
    nr, ni = length(r.responses), length(r.impulses)
    layout --> (nr, ni)
    legend --> false
    titlefontsize --> 8
    hz = collect(r.horizons)
    band = r.method === :bootstrap ? (r.lower, r.upper, "$(r.level)% bootstrap") :
           (r.method === :analytic && r.stderr_orth !== nothing) ?
           ([P .- 1.96 .* S for (P, S) in zip(r.orth, r.stderr_orth)],
            [P .+ 1.96 .* S for (P, S) in zip(r.orth, r.stderr_orth)], "±1.96 se (analytic)") :
           nothing
    for a in 1:nr, b in 1:ni
        sp = (a - 1) * ni + b
        ttl = "$(r.impulses[b]) → $(r.responses[a])"
        if band !== nothing
            @series begin
                subplot := sp
                title := ttl
                seriestype := :line
                fillrange := [M[a, b] for M in band[2]]
                fillalpha := 0.2
                linealpha := 0.0
                label := band[3]
                hz, [M[a, b] for M in band[1]]
            end
        end
        @series begin
            subplot := sp
            seriestype := :line
            title := ttl
            hz, [M[a, b] for M in r.orth]
        end
        @series begin
            subplot := sp
            title := ttl
            seriestype := :hline
            linestyle := :dash
            color := :gray
            [0.0]
        end
    end
end

"""
    @recipe f(f::FEVDResult)

One panel per variable, its forecast error variance split by shock as
**stacked bars** summing to one, horizons `1 … h` — `[V]` the display of
both R's `plot.varfevd` and `statsmodels`' `FEVD.plot`. Core Plots has no
working `bar_position = :stack` (`[V]` GR ignores it with a warning), so
the stack is built explicitly: each shock's bar sits on the cumulative
share of the shocks before it.
"""
@recipe function f(fv::FEVDResult)
    K = size(fv.decomp, 1)
    h = collect(fv.horizons)
    layout --> (K, 1)
    ylims --> (0.0, 1.0)
    legend --> :outerright
    for i in 1:K
        base = zeros(length(h))
        for j in 1:K
            top = base .+ fv.decomp[i, :, j]
            @series begin
                subplot := i
                seriestype := :bar
                fillrange := copy(base)
                bar_width --> 0.8
                title := "FEVD of $(fv.names[i])"
                label := i == 1 ? string(fv.names[j]) : ""
                h, top
            end
            base = top
        end
    end
end

"""
    @recipe f(r::RootsPlotResult)

A VAR's companion eigenvalues in the complex plane against the unit
circle, equal axis scales.
"""
@recipe function f(r::RootsPlotResult)
    aspect_ratio --> :equal
    legend --> false
    title --> string(r.model_name, r.stable ? ": stable" : ": not stable")
    xlabel --> "real"
    ylabel --> "imaginary"
    @series begin
        seriestype := :path          # :line would sort by x and zigzag across the circle
        color := :gray
        linestyle := :dash
        θ = range(0, 2pi; length=361)
        cos.(θ), sin.(θ)
    end
    @series begin
        seriestype := :scatter
        markersize --> 5
        real.(r.eigenvalues), imag.(r.eigenvalues)
    end
end

"""
    @recipe f(r::VARDiagnosticPlotResult)

A `K × 4` grid, one row per equation: standardized residuals, their ACF,
a normal Q-Q plot, and Ljung-Box p-values — the
[`DiagnosticPlotResult`](@ref) panels, drawn the same way, per equation.
"""
@recipe function f(r::VARDiagnosticPlotResult)
    K = length(r.names)
    layout --> (K, 4)
    legend --> false
    titlefontsize --> 8
    size --> (1100, 260 * K)
    for (i, d) in enumerate(r.equations)
        sp = 4 * (i - 1)
        nm = r.names[i]
        @series begin
            subplot := sp + 1
            seriestype := :line
            title := "$nm: standardized residuals"
            1:length(d.std_resid), d.std_resid
        end
        @series begin
            subplot := sp + 2
            seriestype := :sticks
            marker := :circle
            title := "$nm: residual ACF"
            d.acf_lags, d.acf
        end
        @series begin
            subplot := sp + 3
            seriestype := :scatter
            markersize := 2
            title := "$nm: normal Q-Q"
            d.qq_theoretical, d.qq_sample
        end
        @series begin
            subplot := sp + 3
            seriestype := :line
            linestyle := :dash
            color := :gray
            title := "$nm: normal Q-Q"
            lo, hi = extrema(d.qq_theoretical)
            [lo, hi], [lo, hi]
        end
        @series begin
            subplot := sp + 4
            seriestype := :scatter
            markersize := 2
            ylims := (0.0, 1.0)
            title := "$nm: Ljung-Box p-values"
            d.lb_lags, d.ljungbox_pvalues
        end
        @series begin
            subplot := sp + 4
            seriestype := :hline
            linestyle := :dash
            color := :blue
            title := "$nm: Ljung-Box p-values"
            [0.05]
        end
    end
end

"""
    @recipe f(r::CointegrationPlotResult)

One row per cointegrating relation: the relation over time with its mean,
and its ACF beside it — a real equilibrium wanders around a fixed level
and its autocorrelations die out.
"""
@recipe function f(r::CointegrationPlotResult)
    nr = size(r.relations, 2)
    layout --> (nr, 2)
    legend --> false
    titlefontsize --> 9
    for j in 1:nr
        v = r.relations[:, j]
        @series begin
            subplot := 2j - 1
            seriestype := :line
            title := "cointegrating relation $j"
            _decimate_minmax(r.t, v, _PLOT_MAXPOINTS)
        end
        @series begin
            subplot := 2j - 1
            seriestype := :hline
            linestyle := :dash
            color := :gray
            title := "cointegrating relation $j"
            [sum(v) / length(v)]
        end
        a = r.acfs[j]
        @series begin
            subplot := 2j
            title := "relation $j: ACF"
            seriestype := :line
            fillrange := a.upper
            fillalpha := 0.15
            linealpha := 0
            color := :steelblue
            a.lags, a.lower
        end
        @series begin
            subplot := 2j
            title := "relation $j: ACF"
            seriestype := :hline
            linestyle := :dash
            color := :gray
            [0.0]
        end
        @series begin
            subplot := 2j
            seriestype := :sticks
            marker := :circle
            title := "relation $j: ACF"
            a.lags, a.values
        end
    end
end
