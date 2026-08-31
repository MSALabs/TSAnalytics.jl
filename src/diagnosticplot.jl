export DiagnosticPlotResult, diagnostic_plot

"""
    _diagnostic_nlag(; period, ppq, fitdf=0) -> Int

Lag count for [`diagnostic_plot`](@ref)'s Ljung-Box panel, transcribed
directly from `astsa::sarima()`'s real R source (not reconstructed from
a description):
```r
nlag = ifelse(S<7, 20, 3*S); nlag = min(nlag, 52)
ppq = p+q+P+Q - fixed + abs(fitdf)
if (nlag < ppq+8) nlag = ppq+8
```
`period` is the seasonal period `S`; `ppq` is `p+q+P+Q-fixed` (already
summed by the caller -- `abs(fitdf)` is added on top of it here,
matching the R formula's own separate treatment of the two terms).
"""
function _diagnostic_nlag(; period::Integer, ppq::Integer, fitdf::Integer=0)
    nlag = period < 7 ? 20 : 3 * period
    nlag = min(nlag, 52)
    ppq_total = ppq + abs(fitdf)
    return nlag < ppq_total + 8 ? ppq_total + 8 : nlag
end

"""
    DiagnosticPlotResult

Result of [`diagnostic_plot`](@ref) -- the data behind `astsa::sarima()`'s/
`SARIMAXResults.plot_diagnostics()`'s standard 4-panel residual display
(standardized residuals over time, ACF of residuals, a normal Q-Q plot,
and Ljung-Box p-values across a range of lags). `ppq` is the *effective*
lag offset actually used (`ppq + abs(fitdf)` from
`_diagnostic_nlag`'s own formula) -- `length(ljungbox_pvalues) ==
nlag - ppq` always holds.
"""
struct DiagnosticPlotResult
    std_resid::Vector{Float64}
    acf_lags::Vector{Int}
    acf::Vector{Float64}
    qq_theoretical::Vector{Float64}
    qq_sample::Vector{Float64}
    lb_lags::Vector{Int}
    ljungbox_pvalues::Vector{Float64}
    nlag::Int
    ppq::Int
end

"""
    diagnostic_plot(resid; fitdf=0, ppq=0, period=1, lags=nothing) -> DiagnosticPlotResult

Residual diagnostics matching the standard 4-panel display found,
confirmed by reading the real source of all three, in R base's
`tsdiag()`, `astsa::sarima()`, and Python's
`SARIMAXResults.plot_diagnostics()`: standardized residuals over time,
the ACF of the residuals, a normal Q-Q plot, and Ljung-Box p-values
computed across a *range* of lags (not a single number) with a `0.05`
reference line.

- `fitdf`: number of estimated ARMA parameters to subtract from each
  Ljung-Box test's degrees of freedom (matching [`ljungbox_test`](@ref)'s
  own argument).
- `ppq`: `p+q+P+Q-fixed` -- used only to compute the default `lags`
  count via `_diagnostic_nlag` (`astsa::sarima`'s own
  convention); irrelevant if `lags` is given explicitly.
- `period`: the seasonal period `S`, also only used by the default
  `lags` formula.
- `lags`: total number of lags to show; `nothing` (default) uses
  `_diagnostic_nlag(period=period, ppq=ppq, fitdf=fitdf)`.
  Ljung-Box p-values are computed for lags `(ppq+abs(fitdf)+1):lags`
  (a lag count at or below the number of already-fitted parameters
  gives an invalid, non-positive degrees of freedom).

Use `plot(diagnostic_plot(resid))` with a `Plots.jl`/similar backend
loaded (via the `RecipesBase.jl` recipe on [`DiagnosticPlotResult`](@ref))
for the actual 4-panel figure; this function itself has no plotting
dependency.

**Model-aware overloads** `diagnostic_plot(resid, m)` for
`m::Union{ArmaModel,ArimaModel,SarimaModel}` derive `fitdf`/`ppq`/`period`
automatically from `m`'s `order`/`seasonal_order` fields -- `resid`
still has to be supplied explicitly, since none of these model types
currently store or expose fitted residuals (`ArmaModel`/`SarimaModel`
keep only the fitted coefficients, not even the original series;
`ArimaModel` keeps `original_y` but no direct residual accessor either)
-- a real, pre-existing gap in the model API noted in
`development-sequence.md`, not something this stage silently
sidesteps.

`resid` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); resid = randn(200);

julia> r = diagnostic_plot(resid);

julia> length(r.ljungbox_pvalues) == r.nlag - r.ppq
true

julia> r.nlag  # period=1 default -> _diagnostic_nlag(period=1, ppq=0) = 20
20
```
"""
function diagnostic_plot(resid; fitdf::Integer=0, ppq::Integer=0, period::Integer=1,
                          lags::Union{Nothing,Integer}=nothing)
    e = collect(Float64, tsvalues(resid))
    n = length(e)
    n >= 3 || throw(ArgumentError("diagnostic_plot: need at least 3 residuals"))

    ppq_eff = ppq + abs(fitdf)
    nlag = lags === nothing ? _diagnostic_nlag(period=period, ppq=ppq, fitdf=fitdf) : Int(lags)
    nlag > ppq_eff || throw(ArgumentError("diagnostic_plot: lags ($nlag) must exceed ppq+abs(fitdf) ($ppq_eff)"))
    nlag < n || throw(ArgumentError("diagnostic_plot: lags ($nlag) must be < length(resid) ($n)"))

    mu = sum(e) / n
    sigma = sqrt(sum(abs2, e .- mu) / (n - 1))
    sigma > 0 || throw(ArgumentError("diagnostic_plot: residuals have zero variance"))
    std_resid = (e .- mu) ./ sigma

    acf_lags = collect(1:nlag)
    acf_vals = acf(e, acf_lags).values

    lb_lags = collect((ppq_eff+1):nlag)
    lb_pvalues = [ljungbox_test(e, collect(1:L); fitdf=fitdf).pvalue for L in lb_lags]

    qq_sample = sort(std_resid)
    qq_theoretical = [_std_normal_quantile((i - 0.5) / n) for i in 1:n]

    return DiagnosticPlotResult(std_resid, acf_lags, acf_vals, qq_theoretical, qq_sample,
                                 lb_lags, lb_pvalues, nlag, ppq_eff)
end

"""
    diagnostic_plot(resid, m::ArmaModel; fitdf=nothing, lags=nothing) -> DiagnosticPlotResult
    diagnostic_plot(resid, m::ArimaModel; kwargs...) -> DiagnosticPlotResult
    diagnostic_plot(resid, m::SarimaModel; kwargs...) -> DiagnosticPlotResult

Model-aware overload deriving `ppq`/`period` (and, unless overridden,
`fitdf`) automatically from `m`'s `order`/`seasonal_order` --
`p+q(+P+Q)`, and `period=1` for `ArmaModel`/`ArimaModel`, `period=s`
for `SarimaModel`. `fitdf` defaults to `0` (no fixed/constrained
parameters supported by these model types), matching
[`diagnostic_plot`](@ref)'s own default; pass it explicitly to override.
`resid` must still be supplied -- see [`diagnostic_plot`](@ref)'s own
docstring for why it can't be pulled from `m` directly.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = randn(150);

julia> m = fit_arma(y, (1, 1));

julia> r = diagnostic_plot(randn(MersenneTwister(2), 150), m);

julia> r.ppq  # p+q = 1+1 = 2
2
```
"""
function diagnostic_plot(resid, m::ArmaModel; fitdf::Union{Nothing,Integer}=nothing,
                          lags::Union{Nothing,Integer}=nothing)
    p, q = m.order
    fd = fitdf === nothing ? 0 : fitdf
    return diagnostic_plot(resid; fitdf=fd, ppq=p + q, period=1, lags=lags)
end

diagnostic_plot(resid, m::ArimaModel; kwargs...) = diagnostic_plot(resid, m.arma; kwargs...)

function diagnostic_plot(resid, m::SarimaModel; fitdf::Union{Nothing,Integer}=nothing,
                          lags::Union{Nothing,Integer}=nothing)
    p, _, q = m.order
    P, _, Q, s = m.seasonal_order
    fd = fitdf === nothing ? 0 : fitdf
    return diagnostic_plot(resid; fitdf=fd, ppq=p + q + P + Q, period=s, lags=lags)
end
