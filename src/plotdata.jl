export state_decomposition, StateDecomposition
export arma_roots, ARMARootsResult
export ts_display, TSDisplayResult
export seasonal_plot, lag_plot
export roots_plot, RootsPlotResult
export VARDiagnosticPlotResult
export cointegration_plot, CointegrationPlotResult

# ---------------------------------------------------------------------------
# Plot data (handoff/stage 10/plot-recipes-handoff.md).
#
# The package's four plotting patterns, and which each function here uses:
#   A  a recipe directly on an existing result type   (Forecast, VARForecast,
#      VolatilityForecast, IRFResult, FEVDResult -- in recipes.jl)
#   B  a multi-panel recipe on fields a result already has
#   C  a computing function -> a panel struct -> a recipe  (everything below
#      with a struct: the numbers are tested without a plotting backend)
#   D  a data-returning function and deliberately NO recipe, when there is
#      no single natural rendering  (seasonal_plot, lag_plot)
# No plotting dependency: RecipesBase only.
# ---------------------------------------------------------------------------

"""
    StateDecomposition

The fitted state paths of a model, built by [`state_decomposition`](@ref):
`observed`, `level` (`nothing` for a model without one), `slope`
(`nothing` without a trend), and `seasonal` — one series per seasonal
period — all on the time positions `t`. `plot(d)` draws them stacked, as
R's `plot.ets`/`plot.tbats` do.

For an unobserved-components model the decomposition also carries each
component's smoothed standard error (`level_se`, `slope_se`,
`seasonal_se`; `nothing` where there is none), the further components in
`extra` as `(name, path, se)` -- cycle, autoregressive, regression,
irregular -- and the one-step `fitted` values, which the recipe overlays
on the observed series. Exponential-smoothing models have no state
standard errors, and their fields stay `nothing`/empty.
"""
struct StateDecomposition
    t::Vector{Int}
    observed::Vector{Float64}
    level::Union{Nothing,Vector{Float64}}
    slope::Union{Nothing,Vector{Float64}}
    seasonal::Vector{Vector{Float64}}
    periods::Vector{Float64}
    model_name::String
    level_se::Union{Nothing,Vector{Float64}}
    slope_se::Union{Nothing,Vector{Float64}}
    seasonal_se::Vector{Union{Nothing,Vector{Float64}}}
    extra::Vector{Tuple{String,Vector{Float64},Union{Nothing,Vector{Float64}}}}
    fitted::Union{Nothing,Vector{Float64}}
end

# the exponential-smoothing form: no standard errors, no further components
StateDecomposition(t, observed, level, slope, seasonal, periods, model_name) =
    StateDecomposition(t, observed, level, slope, seasonal, periods, model_name, nothing, nothing,
                       Union{Nothing,Vector{Float64}}[nothing for _ in seasonal],
                       Tuple{String,Vector{Float64},Union{Nothing,Vector{Float64}}}[], nothing)

"""
    state_decomposition(m) -> StateDecomposition

The decomposition of a fitted exponential-smoothing model into its state
paths — observed, level, slope and seasonal — matching R's display panel
for panel:

- **[`fit_ets`](@ref)** (`ETSModel`): `observed`, `level`, `slope` if the
  trend is not `:none`, and `season` if seasonal — at most four panels.
  `[S]` That is `forecast:::plot.ets` exactly: its seasonal panel is the
  *current* seasonal state `s1`, one series, not all `m` seasonal states.
- **[`fit_tbats`](@ref)** (`TBATSModel`): one `season` panel per seasonal
  period, each the period's harmonics summed through the measurement
  vector from the state one step back — `[S]` `forecast::tbats.components`,
  which `plot.tbats` draws. (`plot.bats` takes a single state per period
  instead; that is BATS, not TBATS.) With a Box-Cox `lambda` every panel,
  `observed` included, is on the **transformed** scale, as in R.
- **[`holt_winters`](@ref)** (`ExponentialSmoothingModel`): the same
  panels, from the first observation after the season its heuristic
  initialization consumes.

The observed series is rebuilt from the model — `fitted + resid` for
additive errors, `fitted .* (1 .+ resid)` for multiplicative ETS errors,
`boxcox(fitted) + resid` for TBATS — so the data need not be passed back.
`ThetaModel` keeps no state path, so it has no decomposition.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = Float64.(readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)[:, end]);

julia> d = state_decomposition(fit_ets(y, 12; error=:mul, trend=:add, seasonal=:mul));

julia> d.observed ≈ y, d.slope === nothing, length(d.seasonal)
(true, false, 1)
```
"""
function state_decomposition(m::ETSModel)
    obs = m.error === :mul ? m.fitted .* (1 .+ m.resid) : m.fitted .+ m.resid
    slope = m.trend === :none ? nothing : copy(m.trend_component)
    seas = m.seasonal === :none ? Vector{Float64}[] : [_ets_s1_path(m)]
    per = m.seasonal === :none ? Float64[] : [Float64(m.period)]
    name = string("ETS(", _ets_letter(m.error), ",", _ets_letter(m.trend), ",",
                  _ets_letter(m.seasonal), ")")
    return StateDecomposition(collect(1:m.nobs), obs, copy(m.level), slope, seas, per, name)
end

# R's s1: the seasonal state just updated at t, s_t. `seasonal_component[t]`
# holds the term that ENTERED observation t, s_{t-m}; so s_t is
# `seasonal_component[t+m]` for t <= n-m, and the final buffer for the rest.
function _ets_s1_path(m::ETSModel)
    n, p = m.nobs, length(m.s0)
    sN = _ets_final_seasonal(m)
    return [t <= n - p ? m.seasonal_component[t + p] : sN[t - (n - p)] for t in 1:n]
end

_ets_letter(s::Symbol) = s === :add ? "A" : s === :mul ? "M" : s === :damped ? "Ad" : "N"

function state_decomposition(m::TBATSModel)
    P = _tbats_state_path(m)
    n = m.nobs
    obs = m.lambda === nothing ? m.fitted .+ m.resid : boxcox(m.fitted, m.lambda) .+ m.resid
    has_t = m.beta !== nothing
    i0 = 1 + has_t
    seas = Vector{Float64}[]
    c = 0
    for k in m.k
        # the s_j states of this period's harmonics (w = 1 on them, 0 on s*_j),
        # one step back -- the seasonal term in the measurement equation at t
        cols = (i0 + c + 1):(i0 + c + k)
        push!(seas, vec(sum(P[1:n, cols]; dims=2)))
        c += k
    end
    name = string("TBATS{", join(string.(m.k), ","), "}",
                  m.lambda === nothing ? "" : " (Box-Cox scale)")
    return StateDecomposition(collect(1:n), obs, P[2:end, 1], has_t ? P[2:end, 2] : nothing,
                              seas, Float64.(m.periods), name)
end

function state_decomposition(m::ExponentialSmoothingModel)
    L = length(m.fitted)
    obs = m.fitted .+ m.resid
    off = m.period === nothing ? 0 : m.period
    seas = m.seasonal_component === nothing ? Vector{Float64}[] : [copy(m.seasonal_component)]
    per = m.seasonal_component === nothing ? Float64[] : [Float64(m.period)]
    return StateDecomposition(collect((off + 1):(off + L)), obs, copy(m.level),
                              m.trend_component === nothing ? nothing : copy(m.trend_component),
                              seas, per, "Holt-Winters")
end

"""
    ARMARootsResult

The inverse roots of a fitted ARIMA-family model's AR and MA polynomials,
built by [`arma_roots`](@ref). `plot(r)` draws them against the unit
circle.
"""
struct ARMARootsResult
    ar_inverse::Vector{ComplexF64}
    ma_inverse::Vector{ComplexF64}
    model_name::String
end

"""
    arma_roots(m) -> ARMARootsResult

The **inverse** roots of the AR polynomial `φ(B)Φ(Bˢ)` and the MA
polynomial `θ(B)Θ(Bˢ)` — seasonal factors multiplied out — for
[`fit_arma`](@ref), [`fit_arima`](@ref), [`fit_sarima`](@ref),
[`fit_arimax`](@ref) and [`fit_sarimax`](@ref) models. Inside the unit
circle means stationary (AR) and invertible (MA); a root near the circle
is the thing to look at. `[S]` That is R's `plot.Arima`/`autoplot.Arima`
display; Python has none.

They are the eigenvalues of the polynomials' companion matrices: for
`1 − φ₁B − … − φ_pBᵖ`, the roots of `wᵖ − φ₁wᵖ⁻¹ − … − φ_p`, and for
`1 + θ₁B + … + θ_qB^q`, of `w^q + θ₁w^{q−1} + … + θ_q`.

```jldoctest
julia> using TSAnalytics, Random

julia> m = fit_arma(cumsum(randn(MersenneTwister(1), 300)) .* 0.05 .+ randn(MersenneTwister(2), 300), (1, 1));

julia> r = arma_roots(m);

julia> length(r.ar_inverse), length(r.ma_inverse)
(1, 1)
```
"""
arma_roots(m::ArmaModel) = _arma_roots(m.ar, m.ma, "ARMA$(m.order)")
arma_roots(m::ArimaModel) = _arma_roots(m.arma.ar, m.arma.ma,
                                        "ARIMA($(m.arma.order[1]),$(m.d),$(m.arma.order[2]))")
function arma_roots(m::SarimaModel)
    ar, ma = combined_ar_ma(; phi=m.phi, theta=m.theta, seasonal_phi=m.Phi,
                            seasonal_theta=m.Theta, s=m.seasonal_order[4])
    return _arma_roots(ar, ma, "SARIMA$(m.order)$(m.seasonal_order)")
end
arma_roots(m::ArimaxModel) = _arma_roots(m.arma.ar, m.arma.ma, "ARIMAX")
function arma_roots(m::SarimaxModel)
    r = arma_roots(m.arma)
    return ARMARootsResult(r.ar_inverse, r.ma_inverse, "SARIMAX")
end

function _companion_eigs(firstrow::AbstractVector{<:Real})
    p = length(firstrow)
    p == 0 && return ComplexF64[]
    C = zeros(p, p)
    C[1, :] .= firstrow
    for i in 2:p
        C[i, i - 1] = 1.0
    end
    return ComplexF64.(eigvals(C))
end

_arma_roots(ar, ma, name) = ARMARootsResult(_companion_eigs(ar), _companion_eigs(-ma), name)

"""
    TSDisplayResult

A series with its ACF and PACF, built by [`ts_display`](@ref).
"""
struct TSDisplayResult
    t::Vector{Int}
    x::Vector{Float64}
    acf::ACFResult
    pacf::ACFResult
end

"""
    ts_display(x; lags=nothing) -> TSDisplayResult

The standard first look at a series: the series itself on top, its
[`acf`](@ref) and [`pacf`](@ref) side by side beneath — R's
`forecast::ggtsdisplay`. `lags` defaults to `acf`'s own default.

```jldoctest
julia> using TSAnalytics

julia> d = ts_display(sin.(1:120) .+ 0.1 .* (1:120));

julia> d.acf.kind, d.pacf.kind
(:acf, :pacf)
```
"""
function ts_display(x; lags::Union{Nothing,Integer}=nothing)
    xv = Float64.(collect(tsvalues(x)))
    n = length(xv)
    n >= 3 || throw(ArgumentError("ts_display: need at least 3 observations, got $n"))
    L = lags === nothing ? nothing : collect(1:Int(lags))
    return TSDisplayResult(collect(1:n), xv, acf(xv, L), pacf(xv, L))
end

"""
    seasonal_plot(x, period) -> (cycles, positions, cycle_ids)

Data for a seasonal plot: the series cut into its cycles — each cycle's
values in `cycles`, their within-cycle positions (`1:period`, possibly
partial for the first and last cycle) in `positions`, and the cycle's
number in `cycle_ids` — ready to overlay one line per cycle on a
`1:period` axis.

Deliberately data only, not a dedicated result type/recipe, for the same
reason as [`seasonal_subseries_plot`](@ref): there is no one natural
rendering. R's `ggseasonplot` overlays cycles as lines, its `polar = TRUE`
variant draws the same data on a circle, and `statsmodels`' `seasonal_plot`
does something different again, so a recipe would just be encoding one
opinionated choice among several equally valid ones.

```jldoctest
julia> using TSAnalytics

julia> c = seasonal_plot(1.0:30.0, 12);

julia> length(c.cycles), c.positions[end]
(3, [1, 2, 3, 4, 5, 6])
```
"""
function seasonal_plot(x, period::Integer)
    period >= 2 || throw(ArgumentError("seasonal_plot: period must be >= 2"))
    xv = Float64.(collect(tsvalues(x)))
    n = length(xv)
    n >= period || throw(ArgumentError("seasonal_plot: series shorter than period"))
    ncyc = cld(n, period)
    cycles = [xv[((c - 1) * period + 1):min(c * period, n)] for c in 1:ncyc]
    positions = [collect(1:length(v)) for v in cycles]
    return (cycles=cycles, positions=positions, cycle_ids=collect(1:ncyc))
end

"""
    lag_plot(x, lags=1:4) -> (lags, pairs)

Data for a lag plot: for each `k` in `lags`, the pair `(x_{t-k}, x_t)` as
two equal-length vectors in `pairs[i]` — ready to scatter one panel per
lag.

Deliberately data only, not a dedicated result type/recipe, for the same
reason as [`seasonal_subseries_plot`](@ref): there is no one natural
rendering. R's `gglagplot` draws the points as a panel grid, its
`gglagchull` draws convex hulls of the same data, and either can be
coloured by season or not, so a recipe would just be encoding one
opinionated choice among several equally valid ones.

```jldoctest
julia> using TSAnalytics

julia> l = lag_plot(1.0:10.0, [1, 3]);

julia> l.pairs[2]
([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
```
"""
function lag_plot(x, lags::AbstractVector{<:Integer}=1:4)
    xv = Float64.(collect(tsvalues(x)))
    n = length(xv)
    isempty(lags) && throw(ArgumentError("lag_plot: lags must be non-empty"))
    all(k -> 1 <= k < n, lags) || throw(ArgumentError("lag_plot: every lag must be in 1:$(n - 1)"))
    return (lags=collect(Int, lags), pairs=[(xv[1:(n - k)], xv[(k + 1):n]) for k in lags])
end

"""
    RootsPlotResult

The complex eigenvalues of a VAR's companion matrix, built by
[`roots_plot`](@ref).
"""
struct RootsPlotResult
    eigenvalues::Vector{ComplexF64}
    stable::Bool
    model_name::String
end

"""
    roots_plot(m) -> RootsPlotResult

The eigenvalues of a VAR's companion matrix in the complex plane, against
the unit circle — the multivariate counterpart of [`arma_roots`](@ref):
every point inside the circle means a stable VAR. [`roots`](@ref) returns
their moduli; this keeps the complex values, so oscillating dynamics
(conjugate pairs off the real axis) show too. For a [`VECMModel`](@ref)
the levels VAR's companion ([`var_rep`](@ref)) is used, whose `K − r` unit
roots sit on the circle by construction.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> r = roots_plot(fit_var(Y, 2));

julia> length(r.eigenvalues), r.stable
(4, true)
```
"""
roots_plot(m::VARModel) = RootsPlotResult(eigvals(companion(m)), is_stable(m), "VAR($(m.p))")
roots_plot(m::VECMModel) = RootsPlotResult(eigvals(_companion(var_rep(m))), false,
                                           "VECM(lags=$(m.lags), rank=$(m.rank))")

"""
    VARDiagnosticPlotResult

A residual panel for a fitted VAR, built by
`diagnostic_plot(m::VARModel)`: one [`DiagnosticPlotResult`](@ref) per
equation, in the model's variable order.
"""
struct VARDiagnosticPlotResult
    names::Vector{Symbol}
    equations::Vector{DiagnosticPlotResult}
end

"""
    diagnostic_plot(m::VARModel; lags=nothing) -> VARDiagnosticPlotResult

The four-panel residual display of [`diagnostic_plot`](@ref) for every
equation of a fitted VAR — standardized residuals, their ACF, a normal
Q-Q plot and Ljung-Box p-values — as a `K × 4` grid, one row per
equation. `[V]` R spreads this over `plot.varest` and `plot.varcheck`;
`statsmodels` has only its `K × K` `plot_acorr` grid.

The per-equation Ljung-Box p-values are a screening view, not the joint
test: for that use [`portmanteau_test`](@ref) or [`var_diagnostics`](@ref),
which account for the cross-equation structure. Structural-stability
(CUSUM) panels, which `vars::plot.varstabil` draws, are not implemented.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> d = diagnostic_plot(fit_var(Y, 2));

julia> length(d.equations), d.names
(2, [:y1, :y2])
```
"""
function diagnostic_plot(m::VARModel; lags::Union{Nothing,Integer}=nothing)
    eqs = [diagnostic_plot(m.resid[:, i]; lags=lags) for i in 1:length(m.names)]
    return VARDiagnosticPlotResult(copy(m.names), eqs)
end

"""
    CointegrationPlotResult

The estimated cointegrating relations over time, with each relation's
ACF, built by [`cointegration_plot`](@ref).
"""
struct CointegrationPlotResult
    t::Vector{Int}
    relations::Matrix{Float64}
    acfs::Vector{ACFResult}
    beta::Matrix{Float64}
    names::Vector{Symbol}
end

"""
    cointegration_plot(m::VECMModel) -> CointegrationPlotResult
    cointegration_plot(Y, t::JohansenTest; rank=coint_rank(t)) -> CointegrationPlotResult

Each estimated equilibrium relation `β'Ỹ_{t-1}` over time, beside its
autocorrelation function — the visual check that the relation really is
stationary, which is the whole claim a cointegration test makes. `[V]`
`urca`'s `plot` for `ca.jo` draws the same; Python has nothing like it.

`β` is identity-normalized, and `Ỹ` is augmented by a restricted
deterministic term when the model has one — exactly the error-correction
term a VECM regresses on. A [`JohansenTest`](@ref) does not keep its data,
so `Y` is passed alongside it.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "coint3_r1.csv"), ','; skipstart=1);

julia> c = cointegration_plot(fit_vecm(Y; lags=1, rank=1));

julia> size(c.relations), length(c.acfs)
((248, 1), 1)
```
"""
function cointegration_plot(m::VECMModel)
    _, _, X1, T, _ = _vecm_design(m.Y, m.lags, m.deterministic)
    return _coint_plot_result(collect((m.lags + 2):(m.lags + 1 + T)), X1 * m.beta,
                              m.beta, m.names)
end

function cointegration_plot(Y, t::JohansenTest; rank::Integer=coint_rank(t))
    1 <= rank <= t.K - 1 || throw(ArgumentError(
        "cointegration_plot: rank must be in 1:$(t.K - 1), got $rank" *
        (rank == 0 ? " -- the test found no cointegrating relation to plot" : "")))
    Ym = Float64.(tsmatrix(Y))
    size(Ym, 2) == t.K || throw(ArgumentError(
        "cointegration_plot: the data have $(size(Ym, 2)) columns but the test has K = $(t.K)"))
    _, _, X1, T, _ = _vecm_design(Ym, t.lags, t.deterministic)
    T == t.nobs || throw(ArgumentError(
        "cointegration_plot: the data have $(size(Ym, 1)) rows, not the ones the test was run on"))
    B = normalize_beta(t, rank; scheme=:identity)
    return _coint_plot_result(collect((t.lags + 2):(t.lags + 1 + T)), X1 * B, B, t.names)
end

_coint_plot_result(t, R, B, names) =
    CointegrationPlotResult(t, R, [acf(R[:, j]) for j in 1:size(R, 2)], Matrix(B), copy(names))

# ===========================================================================
# Stage 11 plot data (handoff/stage 11/docs/stage-11-plot-recipes-handoff.md).
# Every plottable output gets a named struct: a recipe on a bare NamedTuple
# would capture every NamedTuple in the session.
# ===========================================================================

export factor_paths, FactorPathResult, loadings_plot, LoadingsResult
export factor_r2_plot, FactorR2Result, factor_selection_plot, FactorSelectionResult
export factor_residual_plot, FactorResidualResult
export regime_plot, RegimeProbabilityResult
export threshold_regime_plot, ThresholdRegimeResult, threshold_phase_plot, ThresholdPhaseResult
export threshold_profile, ThresholdProfileResult, transition_plot, TransitionFunctionResult

const _Extra = Tuple{String,Vector{Float64},Union{Nothing,Vector{Float64}}}

# --- 11.1 unobserved components ---------------------------------------------

"""
    state_decomposition(m::UCModel) -> StateDecomposition

The smoothed components of an unobserved-components model with their
standard errors: `level`, `slope` (the stochastic trend's slope), the
seasonal, then in `extra` the cycle, autoregressive, regression and
irregular components, and the one-step `fitted` values. The recipe draws
each component with a ribbon at ±1.96 standard errors -- **a 95% interval
for the smoothed component**, not a forecast interval -- except the
irregular, which is a residual and gets none. The standard errors are
`NaN` over the periods the diffuse initialisation consumes, where the
smoother's variance is not yet defined; the band starts after them.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> nile = readdlm(TSAnalytics.NILE, ','; skipstart=1)[:, 2];

julia> d = state_decomposition(uc_local_level(nile));

julia> d.slope === nothing, first.(d.extra)
(true, ["irregular"])

julia> length(d.level) == length(d.level_se) == 100
true
```
"""
function state_decomposition(m::UCModel)
    d = uc_decomposition(m)
    n = length(m.y)
    seas = d.seasonal === nothing ? Vector{Float64}[] : [d.seasonal]
    sse = Union{Nothing,Vector{Float64}}[d.seasonal_se for _ in seas]
    per = d.seasonal === nothing ? Float64[] : [Float64(d.period)]
    extra = _Extra[]
    d.cycle === nothing || push!(extra, ("cycle", d.cycle, d.cycle_se))
    d.ar === nothing || push!(extra, ("autoregressive", d.ar, d.ar_se))
    d.regression === nothing || push!(extra, ("regression", d.regression, nothing))
    push!(extra, ("irregular", d.irregular, nothing))
    # the one-step prediction is not defined over the diffuse periods
    fit = copy(m.fitted)
    fit[1:min(m.burn, n)] .= NaN
    return StateDecomposition(collect(1:n), copy(m.y), d.level, d.trend, seas, per,
                              "unobserved components", d.level_se, d.trend_se, sse, extra, fit)
end

"""
    diagnostic_plot(m::UCModel; kwargs...) -> DiagnosticPlotResult

The residual diagnostics of [`diagnostic_plot`](@ref) on the model's
one-step prediction errors, leaving out the periods the diffuse
initialisation consumes and any missing observations.
"""
function diagnostic_plot(m::UCModel; kwargs...)
    e = m.resid[(m.burn + 1):end]
    return diagnostic_plot(e[.!isnan.(e)]; kwargs...)
end

# --- 11.2 dynamic factor models ---------------------------------------------

"""
    FactorPathResult

The smoothed factors of a dynamic factor model and their standard errors
(both **observations × factors** -- statsmodels' `factors.smoothed` is the
transpose), with the data, built by [`factor_paths`](@ref). `plot(r)` draws
one panel per factor with a ±1.96 se band; `plot(r; overlay_series=true)`
draws every series in grey with the factors over them, in one panel (R
`dfms`' `type = "joint"`).
"""
struct FactorPathResult
    t::Vector{Int}
    factors::Matrix{Float64}
    se::Matrix{Float64}
    series::Matrix{Float64}
    series_names::Vector{String}
end

"""
    factor_paths(m::DFMModel) -> FactorPathResult

The smoothed factor paths with their standard errors, observations ×
factors. Neither reference draws the band: statsmodels has no factor plot
and `dfms` draws the paths alone.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(TSAnalytics.US_CHANGE, ','; skipstart=1);

julia> r = factor_paths(fit_dfm(standardize(Y), 1));

julia> size(r.factors), size(r.series)
((187, 1), (187, 5))
```
"""
function factor_paths(m::DFMModel)
    n = size(m.Y, 1)
    return FactorPathResult(collect(1:n), copy(m.factors), copy(m.factors_se), copy(m.Y),
                            string.(m.names))
end

"""
    LoadingsResult

The loadings matrix `Λ` (series × factors) with the series names, built by
[`loadings_plot`](@ref). `plot(r)` draws it as a heatmap on a diverging
colour scale centred at zero.
"""
struct LoadingsResult
    loadings::Matrix{Float64}
    series_names::Vector{String}
end

"""
    loadings_plot(m::DFMModel) -> LoadingsResult

Which series load on which factor. Under the package's identification
(lower-triangular leading block, positive diagonal) the pattern is
interpretable rather than an artefact of an arbitrary rotation; after
[`rotate_factors`](@ref) it shows the rotated loadings.
"""
loadings_plot(m::DFMModel) = LoadingsResult(copy(m.Lambda), string.(m.names))

"""
    FactorR2Result

Per-factor R² (series × factors, [`factor_r2`](@ref)) and the share
explained by all factors together ([`series_r2`](@ref)), built by
[`factor_r2_plot`](@ref). `plot(r)` draws grouped bars, one group per
series, with the joint R² as a marker over each group.
"""
struct FactorR2Result
    per_factor::Matrix{Float64}
    total::Vector{Float64}
    series_names::Vector{String}
end

"""
    factor_r2_plot(m::DFMModel) -> FactorR2Result

Both variance decompositions at once. They are different quantities -- the
per-factor R² does not add up to the joint one unless the factors are
uncorrelated -- which is why both are drawn.
"""
factor_r2_plot(m::DFMModel) = FactorR2Result(factor_r2(m), series_r2(m), string.(m.names))

"""
    FactorSelectionResult

The eigenvalues of the series' correlation matrix and Bai & Ng's criteria
for `r = 1 … rmax`, built by [`factor_selection_plot`](@ref). `plot(r)`
draws the scree plot and the three criteria side by side.
"""
struct FactorSelectionResult
    eigenvalues::Vector{Float64}
    r::Vector{Int}
    IC1::Vector{Float64}
    IC2::Vector{Float64}
    IC3::Vector{Float64}
    N::Int
    T::Int
end

"""
    factor_selection_plot(Y, rmax) -> FactorSelectionResult

A scree plot and [`bai_ng_ic`](@ref) together. **The criteria are
consistent only as both `N` and `T` grow**: with a handful of series they
often keep falling past the true number of factors (on a six-series fixture
built with two they keep falling through four), so the minimum is not an
answer. The recipe says so on the plot when `N < 20`.
"""
function factor_selection_plot(Y, rmax::Integer)
    M = Matrix{Float64}(tsmatrix(Y))
    ic = bai_ng_ic(M, rmax)
    ev = sort(eigvals(Symmetric(cor(M))); rev=true)
    return FactorSelectionResult(ev, ic.r, ic.IC1, ic.IC2, ic.IC3, size(M, 2), size(M, 1))
end

"""
    FactorResidualResult

Each series' idiosyncratic residual `y − Λ f` and its five-number summary
(quartiles by linear interpolation, whiskers at the most extreme points
within 1.5 IQR, the points beyond them as outliers), built by
[`factor_residual_plot`](@ref). `plot(r)` draws one box per series.
"""
struct FactorResidualResult
    residuals::Matrix{Float64}
    series_names::Vector{String}
    q1::Vector{Float64}
    median::Vector{Float64}
    q3::Vector{Float64}
    whisker_low::Vector{Float64}
    whisker_high::Vector{Float64}
    outliers::Vector{Vector{Float64}}
end

"""
    factor_residual_plot(m::DFMModel) -> FactorResidualResult

The residuals by series as box plots (R `dfms`' `type = "residual"`): a
series the factors do not explain stands out as a wide box.
"""
function factor_residual_plot(m::DFMModel)
    E = m.Y .- m.factors * m.Lambda'
    N = size(E, 2)
    q1, md, q3, lo, hi = (zeros(N) for _ in 1:5)
    out = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        e = filter(!isnan, E[:, i])
        q1[i], md[i], q3[i] = quantile(e, (0.25, 0.5, 0.75))
        iqr = q3[i] - q1[i]
        inside = filter(x -> q1[i] - 1.5iqr <= x <= q3[i] + 1.5iqr, e)
        lo[i], hi[i] = extrema(inside)
        out[i] = filter(x -> x < lo[i] || x > hi[i], e)
    end
    return FactorResidualResult(E, string.(m.names), q1, md, q3, lo, hi, out)
end

# --- 11.3 Markov switching ----------------------------------------------------

"""
    RegimeProbabilityResult

Filtered and smoothed regime probabilities (each `nobs × k`), the series on
the same periods, and the expected durations (`nothing` with time-varying
transition probabilities), built by [`regime_plot`](@ref). R `MSwM`'s
`smoProb` has `nobs + 1` rows (its first is the smoothed distribution at
`t = 0`); these have `nobs`.

`plot(r)` draws one panel per regime with the smoothed and filtered
probabilities overlaid -- the gap between them is what the smoother adds.
`plot(r; mode=:shaded, regime=1)` draws the series with the periods where
`Pr(S_t = regime | all data) > 0.5` shaded, the figure of the published
Markov-switching literature.
"""
struct RegimeProbabilityResult
    t::Vector{Int}
    y::Vector{Float64}
    filtered::Matrix{Float64}
    smoothed::Matrix{Float64}
    durations::Union{Nothing,Vector{Float64}}
end

"""
    regime_plot(m::MSModel) -> RegimeProbabilityResult

The data for the regime-probability plots. The last filtered and the last
smoothed probability coincide (the smoother starts from the filter's end).
The pairwise transition probabilities `Pr(S_t, S_{t−1} | all data)` get no
plot: a heatmap per period, `k²` small multiples or a transition-count
summary are all defensible and none is obviously right.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> gnp = readdlm(joinpath(pkgdir(TSAnalytics), "handoff", "stage 11", "fixtures",
                              "hamilton_rgnp.csv"), ','; skipstart=1)[:, 2];

julia> r = regime_plot(canonicalise(fit_ms_autoregression(gnp, 4)));

julia> size(r.smoothed), r.filtered[end, :] ≈ r.smoothed[end, :]
((131, 2), true)
```
"""
function regime_plot(m::MSModel)
    p = m.spec.p
    n = length(m.y)
    dur = m.P === nothing ? nothing : expected_durations(m)
    return RegimeProbabilityResult(collect((p + 1):n), m.y[(p + 1):n], copy(m.filtered),
                                   copy(m.smoothed), dur)
end

"""
    diagnostic_plot(m::MSModel; regime=nothing, kwargs...) -> DiagnosticPlotResult

Residual diagnostics. With `regime = nothing`, on the residual averaged
over the smoothed regime distribution (`residuals(m)`); with `regime = r`,
on that regime's conditional residual weighted by `Pr(S_t = r | all data)`,
so the periods the model assigns elsewhere fade out (R `MSwM`'s per-regime
`plotDiag`).
"""
function diagnostic_plot(m::MSModel; regime::Union{Nothing,Integer}=nothing, kwargs...)
    regime === nothing && return diagnostic_plot(StatsAPI.residuals(m); kwargs...)
    1 <= regime <= m.spec.k || throw(ArgumentError(
        "diagnostic_plot: regime must be in 1:$(m.spec.k), got $regime"))
    E, _ = _ms_conditional_resid(m)
    return diagnostic_plot(E[:, regime] .* m.smoothed[:, regime]; kwargs...)
end

ts_display(m::MSModel; kwargs...) = ts_display(m.y; kwargs...)

# --- 11.4 threshold models ----------------------------------------------------

"""
    ThresholdRegimeResult

The series on the rows a SETAR model used, the threshold variable `z`, the
regime of each row (the low regime is `z ≤ th`, inclusive), the
threshold(s) and the fitted values, built by
[`threshold_regime_plot`](@ref). `plot(r)` draws the series coloured by
regime with the thresholds as horizontal lines (R `tsDyn`'s `plot.setar`).
A point's regime is set by the threshold variable `z_t = y_{t−thDelay−1}`,
not by `y_t` itself, so a point above the line can belong to the low regime.
"""
struct ThresholdRegimeResult
    t::Vector{Int}
    y::Vector{Float64}
    z::Vector{Float64}
    regime::Vector{Int}
    thresholds::Vector{Float64}
    fitted::Vector{Float64}
end

"""
    threshold_regime_plot(m::SETARModel) -> ThresholdRegimeResult

Which regime each observation falls in. On log10(lynx) exactly one
observation sits on the threshold and belongs to the low regime.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = log10.(readdlm(TSAnalytics.LYNX, ','; skipstart=1)[:, 2]);

julia> r = threshold_regime_plot(fit_setar(y, 2; thDelay=1, warn=false));

julia> count(==(1), r.regime), r.regime[63], r.z[63] == r.thresholds[1]
(78, 1, true)
```
"""
function threshold_regime_plot(m::SETARModel)
    _, _, z, rows = _tar_embed(m.y, m.m, m.thDelay)
    return ThresholdRegimeResult(collect(rows), m.y[rows], z, copy(m.regime), copy(m.thresholds),
                                 copy(m.fitted))
end

"""
    ThresholdPhaseResult

`y_t` against `y_{t−lag}` with each point's regime and fitted value, built
by [`threshold_phase_plot`](@ref). `plot(r)` draws the points coloured by
regime and the fitted values as open markers.
"""
struct ThresholdPhaseResult
    lagged::Vector{Float64}
    y::Vector{Float64}
    fitted::Vector{Float64}
    regime::Vector{Int}
    lag::Int
    nregimes::Int
end

"""
    threshold_phase_plot(m::SETARModel; lag=1) -> ThresholdPhaseResult

The phase plot of R `tsDyn`'s `plot.setar`, for any number of regimes --
`tsDyn` does not draw it with two thresholds. With more than one lag in the
model the fitted value is not a function of `y_{t−lag}` alone, so it is
drawn as points, not a curve.
"""
function threshold_phase_plot(m::SETARModel; lag::Integer=1)
    1 <= lag <= m.m || throw(ArgumentError("threshold_phase_plot: lag must be in 1:$(m.m)"))
    _, lagf, _, _ = _tar_embed(m.y, m.m, m.thDelay)
    return ThresholdPhaseResult(lagf(lag), m.y[(m.m + 1):end], copy(m.fitted), copy(m.regime),
                                Int(lag), length(m.thresholds) + 1)
end

"""
    ThresholdProfileResult

The search criterion at every candidate threshold, built by
[`threshold_profile`](@ref). `plot(r)` draws the profile with the chosen
threshold marked.
"""
struct ThresholdProfileResult
    candidates::Vector{Float64}
    values::Vector{Float64}
    chosen::Float64
    criterion::Symbol
    conditional_on::Union{Nothing,Float64}
end

"""
    threshold_profile(m::SETARModel; criterion=:ssr) -> ThresholdProfileResult

The profile the threshold search minimised: the SSR at every candidate, or
(`criterion = :aic`) `tsDyn`'s AIC, `n log(SSR/n) + 2k` on the full series
length. `tsDyn`'s `plot.selectSETAR` refuses anything but the SSR; the
profile is as meaningful for the AIC. With two thresholds it is the profile
of the lower threshold with the upper one fixed (`conditional_on`), the
last step of the sequential search. A threshold that was supplied, not
searched, has no profile and is refused.

There is no confidence set for the threshold (Hansen 2000's likelihood-ratio
inversion): it is not built, and a shaded region with no stated coverage
would be worse than none.
"""
function threshold_profile(m::SETARModel; criterion::Symbol=:ssr)
    m.searched || throw(ArgumentError("threshold_profile: the threshold was supplied, not searched"))
    criterion in (:ssr, :aic) || throw(ArgumentError("threshold_profile: criterion must be :ssr or :aic"))
    c = first.(m.profile)
    v = last.(m.profile)
    n = length(m.y)
    criterion === :aic && (v = n .* log.(v ./ n) .+ 2 * StatsAPI.dof(m))
    nt = length(m.thresholds)
    # the stored profile is over the threshold that was searched last
    chosen = c[argmin(v)]
    other = nt == 1 ? nothing : m.thresholds[findfirst(!=(chosen), m.thresholds)]
    return ThresholdProfileResult(c, v, chosen, criterion, other)
end

"""
    TransitionFunctionResult

The fitted LSTAR transition `G(z_t)` at every row, with `z`, the slope
`gamma`, the location `th` and whether `gamma` ran to the start grid's
bound, built by [`transition_plot`](@ref). `plot(r)` draws `G` against `z`
(with the logistic curve) and against time.
"""
struct TransitionFunctionResult
    t::Vector{Int}
    z::Vector{Float64}
    G::Vector{Float64}
    gamma::Float64
    th::Float64
    gamma_at_bound::Bool
end

"""
    transition_plot(m::LSTARModel) -> TransitionFunctionResult

How abrupt the transition is -- which neither reference plots (`tsDyn`
only colours points by `G`). A slope at the start grid's bound is noted on
the plot: the curve is then a step because the parameter is at a boundary,
not because a step was estimated.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = log10.(readdlm(TSAnalytics.LYNX, ','; skipstart=1)[:, 2]);

julia> r = transition_plot(fit_lstar(y, 2; thDelay=1));

julia> minimum(r.G) < 1e-7, maximum(r.G) > 0.99, r.gamma_at_bound
(true, true, false)
```
"""
function transition_plot(m::LSTARModel)
    _, _, z, rows = _tar_embed(m.y, m.m, m.thDelay)
    return TransitionFunctionResult(collect(rows), z, copy(m.G), m.gamma, m.th, m.gamma_at_bound)
end

diagnostic_plot(m::SETARModel; kwargs...) = diagnostic_plot(StatsAPI.residuals(m); kwargs...)
diagnostic_plot(m::LSTARModel; kwargs...) = diagnostic_plot(StatsAPI.residuals(m); kwargs...)
ts_display(m::SETARModel; kwargs...) = ts_display(m.y; kwargs...)
ts_display(m::LSTARModel; kwargs...) = ts_display(m.y; kwargs...)
