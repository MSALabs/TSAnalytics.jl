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

The fitted state paths of an exponential-smoothing model, built by
[`state_decomposition`](@ref): `observed`, `level`, `slope` (`nothing`
without a trend), and `seasonal` — one series per seasonal period —
all on the time positions `t`. `plot(d)` draws them stacked, as R's
`plot.ets`/`plot.tbats` do.
"""
struct StateDecomposition
    t::Vector{Int}
    observed::Vector{Float64}
    level::Vector{Float64}
    slope::Union{Nothing,Vector{Float64}}
    seasonal::Vector{Vector{Float64}}
    periods::Vector{Float64}
    model_name::String
end

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
