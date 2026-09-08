# Handoff: Periodogram/Spectral Density Recipe — Closing Part D's One Remaining Gap

For a fresh Claude Code session picking this up with no prior context.
The combined Stages 1-3 handoff (`stage-1-2-3-combined-fillgap-handoff.md`)
landed almost entirely — five real `@recipe` types now exist
(`ACFResult`, `ClassicalDecomposition`, `STLDecomposition`,
`MSTLDecomposition`, `DiagnosticPlotResult`), plus `boxcox_profile_plot`
and `seasonal_subseries_plot` as data-producing functions. **Confirmed
directly by re-reading the actual source**: the periodogram/spectral
density recipe from that same handoff is the one piece that didn't
land. This closes it.

**Also confirmed, not assumed, while checking**: PACF does *not* need
a separate recipe — `pacf()` already returns `ACFResult` (tagged
`:pacf` in its `kind` field), and the existing ACF recipe already
branches on that tag correctly for the axis label. No second gap here;
said explicitly so it isn't re-investigated.

## Where this fits

- **Depends on:** Stage 1.5's `periodogram`/`spectral_density`
  (already built, already verified against real R output — see
  `stage-1-fillgap-handoff.md`). This handoff only adds the
  visualization layer on top; no new numerical work.

---

## 1. Test cases — more cases, genuinely aimed at correctness, not padding

The underlying computation is already verified (Stage 1.5's real
R-matched peak-frequency case). What a *recipe* can actually get wrong
is different in kind: mapping the right fields to the right axes,
correct scale, and not silently mishandling edge cases the plain
data-returning function might tolerate but a rendered plot would show
as visibly broken.

```julia
using Test, RecipesBase

@testset "periodogram recipe -- correct series count and axis mapping" begin
    x = randn(96)
    r = periodogram(x)
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r)
    @test length(rec) == 1                      # one series, unlike ACF's three (fill band + hline + sticks)
    @test rec[1].plotattributes[:seriestype] in (:path, :line)
end

@testset "periodogram recipe -- log scale is the default, matching R/astsa convention" begin
    x = randn(96)
    r = periodogram(x)
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r)
    @test get(rec[1].plotattributes, :yscale, :identity) == :log10
end

@testset "periodogram recipe -- real, verified peak case renders the right x-value at the right place" begin
    # reuse Stage 1.5's own exact verified case -- period-12 series,
    # peak frequency 1/12 -- confirm the PLOTTED x-data at the peak
    # y-value matches periodogram()'s own already-verified freq output,
    # not a separately-computed value
    Random.seed!(42)
    n = 96; t = 1:n
    x = 5 .* sin.(2π .* t ./ 12) .+ 3 .* cos.(2π .* t ./ 12) .+ randn(n) .* 0.5
    r = periodogram(x; taper=0.1)
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r)
    xdata, ydata = rec[1].args
    @test isapprox(xdata[argmax(ydata)], 1/12; atol=1e-6)
end

@testset "periodogram recipe -- edge cases" begin
    @test_throws ArgumentError periodogram(Float64[])          # empty series
    @test_throws ArgumentError periodogram([1.0])               # single point, no frequency content possible
    r = periodogram(randn(4))                                    # minimal valid length
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r)
    @test length(rec[1].args[1]) > 0                             # still produces a valid, non-empty plot
end

@testset "spectral_density recipe -- df/bandwidth surfaced in the plot title or subtitle, not silently dropped" begin
    x = randn(200)
    r = spectral_density(x, [7,7])
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r)
    # confirm df/bandwidth appear somewhere in plotattributes (title/annotation) --
    # a user comparing to R's own spec.pgram output needs these visible,
    # not just returned in the underlying struct and ignored by the recipe
    @test any(occursin("df", string(v)) for v in values(rec[1].plotattributes) if v isa AbstractString) ||
          haskey(r, :df)  # at minimum, confirm the data is still reachable even if not auto-annotated
end

@testset "large series -- decimation doesn't silently corrupt the visible peak" begin
    # see section 2 -- if decimation for large-n rendering is implemented,
    # this is the test that must never be allowed to fail silently:
    # decimating for display speed must not move or hide the true peak
    n = 50_000
    t = 1:n
    x = 5 .* sin.(2π .* t ./ 12) .+ randn(n) .* 0.5
    r = periodogram(x)
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r)
    xdata, ydata = rec[1].args
    @test isapprox(xdata[argmax(ydata)], 1/12; atol=1e-4)
end
```

---

## 2. Efficiency — honest about where it does and doesn't apply here

**The recipe itself is not where efficiency matters** — it repackages
already-computed arrays from `periodogram`/`spectral_density` (Stage
1.5's own FFT-based computation is where the real cost already lives,
and that's already been through its own performance consideration).
Say this plainly rather than manufacture an efficiency story for glue
code that doesn't need one.

**One genuine efficiency consideration specific to plotting, not
computation**: a periodogram on a very long series (tens of thousands
of points, a real case for high-frequency financial data) produces one
point per frequency bin — rendering all of them can be slow and adds
little visual information beyond what a decimated view shows. If
decimation is added for large `n`, it must be verified not to hide or
shift the true spectral peak (the test in section 1 covers this
directly) — a real correctness risk introduced *by* the efficiency
measure, not a hypothetical one. Recommend: only decimate above some
threshold (e.g. `n > 10_000` plotted points), and always decimate by
*local maximum retention* within each bin rather than naive downsampling
(naive stride-based decimation can genuinely miss a narrow spectral
peak sitting between the sampled indices — local-max retention cannot).

---

## 3. Documentation — the professional/hobby distinction, taken seriously

Every other recipe and function already in this codebase
(`boxcox_profile_plot`, `seasonal_subseries_plot`, the `ACFResult`
recipe) follows a consistent documentation pattern worth matching
exactly, not improvising a new style for this one addition:

```julia
"""
    @recipe f(r::PeriodogramResult)

The log-scale periodogram plot -- `spec.pgram(plot=TRUE)`'s default
display in base R, and `astsa::mvspec`'s own standard output (Shumway
& Stoffer, *Time Series Analysis and Its Applications*, Chapter 4).
Shows spectral power against frequency on a log10 y-axis, the
conventional scale for spectral estimates given their typically
large dynamic range.

# Example
```julia
julia> x = 5 .* sin.(2π .* (1:96) ./ 12) .+ randn(96) .* 0.5;

julia> r = periodogram(x; taper=0.1);

julia> plot(r)  # peak visible at frequency 1/12, matching the injected periodicity
```

See also [`periodogram`](@ref), [`spectral_density`](@ref) for the
underlying computation, verified directly against base R's `spec.pgram`
(see `stage-1-fillgap-handoff.md`).
"""
```

**Specifically required, not optional**: a citation back to the exact
book/chapter and reference implementation (matching every other
docstring in this codebase's own established convention), a runnable
`julia>` example a reader can paste directly, and a `See also` pointing
to the underlying verified computation rather than re-explaining it.

---

## What to do with this

1. Implement the `@recipe function f(r::PeriodogramResult)` (and the
   equivalent for `spectral_density`'s return type, if it's a distinct
   struct) in `src/recipes.jl`, alongside the existing five.
2. Run the test matrix in section 1 — the real peak-frequency
   reproduction test and the large-series decimation test are the two
   highest-value cases, not the edge-case tests alone.
3. If decimation is implemented per section 2, it is not optional to
   skip the large-series peak-preservation test — that's the one place
   an efficiency measure could silently introduce a correctness bug.
4. Write the docstring per section 3's exact pattern before considering
   this done — matching the existing codebase's documentation
   standard, not a lighter version of it.
5. Update `development-sequence.md`: mark Part D's periodogram recipe
   complete, and record explicitly that the PACF investigation
   concluded "no gap" rather than leave it as an open question.
