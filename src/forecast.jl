export forecast, Forecast, mean_forecast, naive, seasonal_naive, drift, psi_weights

"""
    Forecast

Result of `StatsAPI.predict`/[`forecast`](@ref).

- `point`/`se`: point forecast and standard error, one entry per step
  `1:horizon`.
- `levels`: the confidence levels used, as percentages (e.g. `95.0`).
- `lower`/`upper`: `horizon x length(levels)` matrices, one column per
  level, in the same order as `levels`.
- `model_name`: a short label for `show` (e.g. `"AR(2)"`).
"""
struct Forecast
    point::Vector{Float64}
    se::Vector{Float64}
    levels::Vector{Float64}
    lower::Matrix{Float64}
    upper::Matrix{Float64}
    horizon::Int
    model_name::String
end

"_arx_forecast_row(names, t, yext) -- one row of the design matrix at
absolute time `t`, in exactly the column order `arx()` itself built
`names` in (see its docstring), reading lag terms from `yext` (the
original series extended in-place with each step's own forecast so far).
Throws a clear error for `exog` columns (`\"x1\"`, ...) rather than
silently zeroing them or guessing -- forecasting an AR-X model with
exogenous regressors needs *future* exog values this function has no way
to know, a real, deliberate gap, not an oversight."
function _arx_forecast_row(names::Vector{String}, t::Integer, yext::Vector{Float64})
    row = Vector{Float64}(undef, length(names))
    for (i, nm) in enumerate(names)
        row[i] = if nm == "const"
            1.0
        elseif nm == "trend"
            Float64(t)
        elseif startswith(nm, "s(")
            m = match(r"^s\((\d+),(\d+)\)$", nm)
            season, period = parse(Int, m.captures[1]), parse(Int, m.captures[2])
            mod1(t, period) == season ? 1.0 : 0.0
        elseif startswith(nm, "y.L")
            lag = parse(Int, nm[4:end])
            yext[t - lag]
        elseif startswith(nm, "x")
            throw(ArgumentError("predict: cannot forecast a model fit with `exog` -- " *
                                 "future exog values are needed and aren't available " *
                                 "(not supported)"))
        else
            error("predict: unrecognized ARXModel coefficient name \"$nm\"")
        end
    end
    return row
end

"""
    StatsAPI.predict(model::ARXModel, horizon; level=[80.0, 95.0]) -> Forecast

Extends `StatsAPI.predict`; see [`forecast`](@ref) for the full
documentation (matching this project's existing convention of
documenting a `StatsAPI` extension's *exported* wrapper, not the bare
extension method itself -- `StatsAPI.coef`/`vcov` on `ARXModel`, Stage
5.1, likewise carry no separate docstring of their own).
"""
function StatsAPI.predict(model::ARXModel, horizon::Integer; level::Vector{<:Real}=[80.0, 95.0])
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    isempty(level) && throw(ArgumentError("level must be non-empty"))
    all(0 .< level .< 100) || throw(ArgumentError("level entries must be in (0, 100)"))
    any(nm -> startswith(nm, "x"), model.names) &&
        throw(ArgumentError("predict: cannot forecast a model fit with `exog` -- " *
                             "future exog values are needed and aren't available " *
                             "(not supported)"))

    n = length(model.y)
    yext = copy(model.y)
    point = Vector{Float64}(undef, horizon)
    for h in 1:horizon
        t = n + h
        row = _arx_forecast_row(model.names, t, yext)
        yhat = dot(row, model.coef)
        push!(yext, yhat)
        point[h] = yhat
    end

    # Box-Jenkins psi-weights (impulse response): psi[1] == psi_0 == 1,
    # generalizes correctly to an arbitrary lag subset (verified formula,
    # see the handoff doc) -- phi[k] is aligned with model.lags[k] since
    # arx() pushes "y.L$lag" names in exactly lagset's own order.
    lag_idx = findall(nm -> startswith(nm, "y.L"), model.names)
    phi = model.coef[lag_idx]
    psi = zeros(horizon)
    psi[1] = 1.0
    for j in 2:horizon
        s = 0.0
        for (k, lag) in enumerate(model.lags)
            lag < j && (s += phi[k] * psi[j-lag])
        end
        psi[j] = s
    end
    se = [sqrt(model.sigma2 * sum(abs2, view(psi, 1:h))) for h in 1:horizon]

    z = [_confidence_z(1 - l/100) for l in level]  # level is a percentage, e.g. 95.0 -> alpha=0.05
    lower = reduce(hcat, [point .- zi .* se for zi in z])
    upper = reduce(hcat, [point .+ zi .* se for zi in z])

    model_name = "AR(" * string(maximum(model.lags)) * ")"
    return Forecast(point, se, Float64.(level), lower, upper, horizon, model_name)
end

"""
    forecast(model::ARXModel, horizon; level=[80.0, 95.0]) -> Forecast

Forecast `horizon` steps ahead from a fitted AR-X model, with prediction
intervals at each level in `level` (percentages, e.g. `95.0` for a 95%
interval -- matches R's `forecast()` convention, not Python's single
`alpha`; this function's own name is a direct alias for R users looking
for `forecast()` specifically -- `StatsAPI.predict(model, horizon;
level=level)` is exactly equivalent). Standard errors use the
known-parameters forecast-error propagation formula (Box-Jenkins
psi-weights), verified to exactly reproduce **both**
`statsmodels`' `AutoReg.get_prediction()` **and** R's
`predict.ar()`/`forecast.ar()` output (both executed directly in this
session, not assumed) -- see `handoff/stage-5.2-forecast-handoff.md`.
Does not adjust for parameter estimation uncertainty, matching both
references' actual (not idealized) behavior.

Point forecasts are computed by the same recursion `arx()` itself uses to
build its design matrix, walking `model.names` (see [`ARXModel`](@ref))
column by column at each future time step, feeding each step's own
forecast back in as pseudo-history once the horizon exceeds the original
sample -- so it generalizes correctly to a lag *subset* (e.g. `[1,3]`),
`trend`/`seasonal` continuation, and any combination thereof. Throws a
clear `ArgumentError` for models fit with `exog` (future exog values
would be needed and aren't available) rather than silently ignoring the
regressor.

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1));

julia> m = arx(y, 2);

julia> f = forecast(m, 3);

julia> f.horizon
3

julia> size(f.lower) == (3, 2)  # default level=[80.0, 95.0], 2 columns
true
```
"""
forecast(model, horizon; level::Vector{<:Real}=[80.0, 95.0]) = StatsAPI.predict(model, horizon; level=level)

function Base.show(io::IO, f::Forecast)
    println(io, f.model_name, " forecast, ", f.horizon, " steps ahead")
    println(io)
    header = ["h", "Point Forecast"]
    for l in f.levels
        li = isinteger(l) ? string(Int(l)) : string(l)
        push!(header, "Lo $li"); push!(header, "Hi $li")
    end
    println(io, join(header, "  "))
    for h in 1:f.horizon
        row = [string(h), string(round(f.point[h], digits=4))]
        for j in eachindex(f.levels)
            push!(row, string(round(f.lower[h, j], digits=4)))
            push!(row, string(round(f.upper[h, j], digits=4)))
        end
        println(io, join(row, "  "))
    end
end

"_benchmark_forecast(point, se, level, model_name) -- shared tail end for
all four benchmark methods below: validates `level` the same way
`StatsAPI.predict(::ARXModel, ...)` already does, builds `lower`/`upper`
via the same `_confidence_z` (precise normal quantile, not the coarse
1.28/1.96 literals) rather than the rounded literals fpp3 itself quotes,
and returns the `Forecast`."
function _benchmark_forecast(point::Vector{Float64}, se::Vector{Float64},
                              level::Vector{<:Real}, model_name::String)
    isempty(level) && throw(ArgumentError("level must be non-empty"))
    all(0 .< level .< 100) || throw(ArgumentError("level entries must be in (0, 100)"))
    z = [_confidence_z(1 - l / 100) for l in level]
    lower = reduce(hcat, [point .- zi .* se for zi in z])
    upper = reduce(hcat, [point .+ zi .* se for zi in z])
    return Forecast(point, se, Float64.(level), lower, upper, length(point), model_name)
end

"""
    mean_forecast(y, h::Integer; level=[80.0, 95.0]) -> Forecast

The mean benchmark method (Hyndman & Athanasopoulos, fpp3 §5.2): every
step forecasts the sample mean of `y`, the right benchmark for a series
with neither trend nor seasonality. Standard error grows only through
uncertainty in the mean itself, so `se` is constant across the horizon
(fpp3 Table 5.2: `σ̂ * sqrt(1 + 1/T)`), unlike [`naive`](@ref)/[`drift`](@ref).

Named `mean_forecast` rather than `mean` to avoid shadowing
`Statistics.mean`.

# Examples
```jldoctest
julia> mean_forecast([10.0, 12, 14, 16, 18], 2).point
2-element Vector{Float64}:
 14.0
 14.0
```
"""
function mean_forecast(y, h::Integer; level::Vector{<:Real}=[80.0, 95.0])
    h >= 1 || throw(ArgumentError("h must be >= 1"))
    yv = tsvalues(y)
    n = length(yv)
    n >= 2 || throw(ArgumentError("mean_forecast needs at least 2 observations"))
    m = Statistics.mean(yv)
    resid = yv .- m
    sigma = sqrt(sum(abs2, resid) / (n - 1))
    point = fill(m, h)
    se = fill(sigma * sqrt(1 + 1 / n), h)
    return _benchmark_forecast(point, se, level, "Mean")
end

"""
    naive(y, h::Integer; level=[80.0, 95.0]) -> Forecast

The naive benchmark method (fpp3 §5.2): every step repeats the last
observed value, `y[end]`. The right benchmark for a random walk, and the
one [`mase`](@ref) computes internally for its own denominator (`sp=1`) --
calling `naive` directly makes that same benchmark available as a
forecast you can plot or feed to [`tscv`](@ref), rather than writing it
by hand.

# Examples
```jldoctest
julia> naive([10.0, 12, 14, 16, 18], 3).point
3-element Vector{Float64}:
 18.0
 18.0
 18.0
```
"""
function naive(y, h::Integer; level::Vector{<:Real}=[80.0, 95.0])
    h >= 1 || throw(ArgumentError("h must be >= 1"))
    yv = tsvalues(y)
    n = length(yv)
    n >= 2 || throw(ArgumentError("naive needs at least 2 observations"))
    resid = diff(yv)
    sigma = sqrt(sum(abs2, resid) / (n - 1))
    point = fill(yv[end], h)
    se = [sigma * sqrt(hh) for hh in 1:h]
    return _benchmark_forecast(point, se, level, "Naive")
end

"""
    seasonal_naive(y, h::Integer, m::Integer; level=[80.0, 95.0]) -> Forecast

The seasonal-naive benchmark method (fpp3 §5.2): step `h` repeats the
value from the same point in the most recent complete season,
`y[end + h - m*(k+1)]` with `k = (h-1) ÷ m`. The seasonal counterpart of
[`mase`](@ref)'s own benchmark (`sp=m`). Standard error is a genuine
**step function** of `h` (fpp3 Table 5.2: `σ̂ * sqrt(k+1)`) -- constant
within each block of `m` horizons, then jumping at the next one.

# Examples
```jldoctest
julia> seasonal_naive([1.0, 2, 3, 4, 5, 6, 7, 8], 6, 4).point
6-element Vector{Float64}:
 5.0
 6.0
 7.0
 8.0
 5.0
 6.0
```
"""
function seasonal_naive(y, h::Integer, m::Integer; level::Vector{<:Real}=[80.0, 95.0])
    h >= 1 || throw(ArgumentError("h must be >= 1"))
    m >= 1 || throw(ArgumentError("m must be >= 1"))
    yv = tsvalues(y)
    n = length(yv)
    n > m || throw(ArgumentError("seasonal_naive needs more than m observations"))
    resid = yv[(m + 1):end] .- yv[1:(end - m)]
    sigma = sqrt(sum(abs2, resid) / (n - m))
    point = Vector{Float64}(undef, h)
    se = Vector{Float64}(undef, h)
    for hh in 1:h
        k = (hh - 1) ÷ m
        point[hh] = yv[n + hh - m * (k + 1)]
        se[hh] = sigma * sqrt(k + 1)
    end
    return _benchmark_forecast(point, se, level, "Seasonal naive")
end

"""
    drift(y, h::Integer; level=[80.0, 95.0]) -> Forecast

The drift benchmark method (fpp3 §5.2): extrapolates the straight line
through the first and last observations, `y[end] + h * (y[end] - y[1]) /
(T - 1)`. Equivalent to naive with a trend term added. Standard error
(fpp3 Table 5.2: `σ̂ * sqrt(h * (1 + h/(T-1)))`) is computed from the
residuals of that same line, `diff(y) .- slope`, which is why `drift`
needs one more observation than [`naive`](@ref) to have a residual left
to estimate `σ̂` from.

# Examples
```jldoctest
julia> drift([10.0, 12, 14, 16, 18], 3).point
3-element Vector{Float64}:
 20.0
 22.0
 24.0
```
"""
function drift(y, h::Integer; level::Vector{<:Real}=[80.0, 95.0])
    h >= 1 || throw(ArgumentError("h must be >= 1"))
    yv = tsvalues(y)
    n = length(yv)
    n >= 3 || throw(ArgumentError("drift needs at least 3 observations"))
    slope = (yv[end] - yv[1]) / (n - 1)
    resid = diff(yv) .- slope
    sigma = sqrt(sum(abs2, resid) / (n - 2))
    point = [yv[end] + hh * slope for hh in 1:h]
    se = [sigma * sqrt(hh * (1 + hh / (n - 1))) for hh in 1:h]
    return _benchmark_forecast(point, se, level, "Drift")
end

"""
    psi_weights(ar, ma, h::Integer) -> Vector{Float64}

MA(∞) representation (Box-Jenkins impulse response) of an ARMA process
given in *state-space* coefficient convention -- `ar[i]` multiplies
`y_{t-i}`, `ma[j]` multiplies `e_{t-j}` (the same convention
`build_statespace`/`combined_ar_ma` use, i.e. what you
already have for an `ArmaModel`'s `.ar`/`.ma`, or what `combined_ar_ma`
returns for a seasonal model). Returns `h` weights, `psi[1] == ψ₀ == 1`
through `psi[h] == ψ_{h-1}`, via the standard recursion
`ψⱼ = Σᵢ ar[i]·ψⱼ₋ᵢ + (j<=length(ma) ? ma[j] : 0)`.

Verified directly against R's `ARMAtoMA()` on several AR/MA/ARMA cases,
including a case with a unit root in `ar` (i.e. `ar` including a
differencing factor, as [`forecast`](@ref)`(::ArimaModel, ...)` builds
internally) -- unlike a stationary ARMA's weights, these do not decay,
which is the exact, verified mechanism behind ARIMA's fanning
prediction intervals (ψⱼ² accumulates without bound as `j` grows).

# Examples
```jldoctest
julia> psi_weights([0.5], Float64[], 4)  # AR(1): ψⱼ = 0.5^j
4-element Vector{Float64}:
 1.0
 0.5
 0.25
 0.125
```
"""
function psi_weights(ar::AbstractVector{<:Real}, ma::AbstractVector{<:Real}, h::Integer)
    h >= 1 || throw(ArgumentError("psi_weights: h must be >= 1"))
    psi = zeros(Float64, h)
    psi[1] = 1.0
    p, q = length(ar), length(ma)
    for j in 1:(h-1)
        s = 0.0
        for i in 1:min(j, p)
            s += ar[i] * psi[j-i+1]
        end
        j <= q && (s += ma[j])
        psi[j+1] = s
    end
    return psi
end

"_undifferenced_ar(ar, d) -- the state-space AR coefficients of
`ar` combined with `d` factors of `(1-B)`, in the same state-space
convention (`ar[i]` multiplies `y_{t-i}`) -- the AR polynomial that
[`psi_weights`](@ref) needs to see the unit root(s) differencing
introduces, so that the resulting weights do not decay (see
`handoff/stage-6-arima-handoff.md` §5, point 4: this is what
`forecast::Arima` itself uses `Var[ŷ] = σ̂²·Σψⱼ²` on -- the psi-weights
of the *undifferenced* model, not the stationary one actually fit)."
function _undifferenced_ar(ar::AbstractVector{<:Real}, d::Integer)
    ar_natural = vcat([1.0], -collect(Float64, ar))
    diff_poly = [1.0]
    for _ in 1:d
        diff_poly = polymul(diff_poly, [1.0, -1.0])
    end
    combined = polymul(ar_natural, diff_poly)
    return -combined[2:end]
end

"_forecast_arma_diffed(w, v, ar, ma, mu, horizon) -- forward AR/MA
recursion on an already-differenced, demeaned series `w` with its own
fitted innovations `v` (from `kalman_filter`), extending both `horizon`
steps into the future (future innovations forecast as `0`, their
expectation) and returning the `horizon` point forecasts *with the mean
added back*, still on the differenced scale. Shared by the `ArimaModel`
and `SarimaModel` methods below -- the only difference between the two
is how `w`/`ar`/`ma` were built (plain vs. `combined_ar_ma`) and how the
result gets re-integrated afterwards."
function _forecast_arma_diffed(w::Vector{Float64}, v::Vector{Float64},
                                ar::AbstractVector{<:Real}, ma::AbstractVector{<:Real},
                                mu::Real, horizon::Integer)
    n = length(w)
    p, q = length(ar), length(ma)
    wext = vcat(w, zeros(horizon))
    eext = vcat(v, zeros(horizon))  # future innovations forecast as their expectation, 0
    for h in 1:horizon
        t = n + h
        s = 0.0
        for i in 1:p
            s += ar[i] * wext[t-i]
        end
        for j in 1:q
            s += ma[j] * eext[t-j]
        end
        wext[t] = s
    end
    return wext[(n+1):end] .+ mu
end

"""
    StatsAPI.predict(model::ArimaModel, horizon; level=[80.0, 95.0]) -> Forecast

Extends `StatsAPI.predict`; see [`forecast`](@ref) for the full
documentation.
"""
function StatsAPI.predict(model::ArimaModel, horizon::Integer; level::Vector{<:Real}=[80.0, 95.0])
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    isempty(level) && throw(ArgumentError("level must be non-empty"))
    all(0 .< level .< 100) || throw(ArgumentError("level entries must be in (0, 100)"))

    arma = model.arma
    d = model.d
    y = model.original_y
    ar, ma = arma.ar, arma.ma
    mu = arma.mean === nothing ? 0.0 : arma.mean

    yd = d > 0 ? diff(y, 1; differences=d) : copy(y)
    w = yd .- mu
    ssm = build_statespace(ar, ma)
    _, sigma2, v, _, converged = kalman_filter(ssm, w)
    converged || throw(ArgumentError("predict: the fitted model's Kalman filter did not converge on its own data"))

    point_diff = _forecast_arma_diffed(w, v, ar, ma, mu, horizon)
    point = if d > 0
        seed = y[(end-d+1):end]
        tsundiff(point_diff; differences=d, xi=seed)[(d+1):end]
    else
        point_diff
    end

    ar_full = _undifferenced_ar(ar, d)
    psi = psi_weights(ar_full, ma, horizon)
    se = [sqrt(sigma2 * sum(abs2, view(psi, 1:h))) for h in 1:horizon]

    z = [_confidence_z(1 - l/100) for l in level]
    lower = reduce(hcat, [point .- zi .* se for zi in z])
    upper = reduce(hcat, [point .+ zi .* se for zi in z])
    p, q = length(ar), length(ma)
    model_name = "ARIMA($p,$d,$q)"
    return Forecast(point, se, Float64.(level), lower, upper, horizon, model_name)
end

"""
    forecast(model::ArimaModel, horizon; level=[80.0, 95.0]) -> Forecast

Forecast `horizon` steps ahead from a fitted [`ArimaModel`](@ref), with
prediction intervals at each level in `level` (percentages -- see
[`forecast`](@ref)`(::ARXModel, ...)` for the convention). Point
forecasts are exact (Box-Jenkins §5, no approximation): the underlying
stationary ARMA is forecast forward on the differenced scale, using the
model's own fitted innovations up to the end of the sample and their
expectation (`0`) beyond it, then re-integrated back to the original
scale via [`tsundiff`](@ref), seeded with `model.original_y`'s own last
`d` values -- exactly reproducing the original series where the two
overlap (verified: `tsundiff∘diff` round-trips to machine precision).

**Prediction interval variance is the one place this genuinely differs
from a stationary model's forecast**: computed from the psi-weights of
the *undifferenced* ARIMA polynomial -- `ar` combined with `d` factors
of `(1-B)` via [`psi_weights`](@ref) -- not the stationary ARMA that was
actually fit, matching `forecast::Arima`'s own documented approach
(`handoff/stage-6-arima-handoff.md` §5). A stationary model's
psi-weights decay to zero and the forecast variance converges to a
constant; here they do not decay (the differencing operator contributes
a literal unit root to the polynomial `psi_weights` sees), so the
variance grows without bound -- the honest consequence of a unit root
already met in Chapter 19, not a modelling artefact.

Verified against real R `predict(arima(...))`/`forecast::forecast.Arima`
and Python `statsmodels`' `get_forecast()` on the same fitted ARIMA(1,1,0)
and ARIMA(0,1,1) models: point forecasts and prediction-interval widths
match both references to several significant figures.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(100));

julia> m = fit_arima(y, (1, 1, 0));

julia> f = forecast(m, 5);

julia> f.horizon
5

julia> f.se[5] > f.se[1]  # variance grows with horizon under a unit root
true
```
"""
forecast(model::ArimaModel, horizon::Integer; level::Vector{<:Real}=[80.0, 95.0]) =
    StatsAPI.predict(model, horizon; level=level)

"""
    StatsAPI.predict(model::SarimaModel, y, horizon; level=[80.0, 95.0]) -> Forecast

Extends `StatsAPI.predict`; see [`forecast`](@ref) for the full
documentation. `y` is required -- unlike [`ArimaModel`](@ref),
[`SarimaModel`](@ref) does not store the series it was fit on (see its
own docstring), so it must be supplied again here, exactly as
[`diagnostic_plot`](@ref)`(resid, ::SarimaModel)` already requires for
the same reason.
"""
function StatsAPI.predict(model::SarimaModel, y, horizon::Integer; level::Vector{<:Real}=[80.0, 95.0])
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    isempty(level) && throw(ArgumentError("level must be non-empty"))
    all(0 .< level .< 100) || throw(ArgumentError("level entries must be in (0, 100)"))

    p, d, q = model.order
    P, D, Q, s = model.seasonal_order
    yv = Float64.(collect(tsvalues(y)))
    yD = D > 0 ? diff(yv, s; differences=D) : yv
    yd = d > 0 ? diff(yD, 1; differences=d) : yD
    mu = model.mean === nothing ? 0.0 : model.mean
    w = yd .- mu

    ar, ma = combined_ar_ma(; phi=model.phi, theta=model.theta,
                             seasonal_phi=model.Phi, seasonal_theta=model.Theta, s=s)
    ssm = build_statespace(ar, ma)
    _, sigma2, v, _, converged = kalman_filter(ssm, w)
    converged || throw(ArgumentError("predict: the fitted model's Kalman filter did not converge on its own data"))

    point_diff = _forecast_arma_diffed(w, v, ar, ma, mu, horizon)
    point_afterD = if d > 0
        seed = yD[(end-d+1):end]
        tsundiff(point_diff; differences=d, xi=seed)[(d+1):end]
    else
        point_diff
    end
    point = if D > 0
        seed = yv[(end-s*D+1):end]
        tsundiff(point_afterD; lag=s, differences=D, xi=seed)[(s*D+1):end]
    else
        point_afterD
    end

    ar_reg_natural = vcat([1.0], -model.phi)
    ar_seas_natural = seasonal_poly(model.Phi, s; sign=-1.0)
    diff_reg = [1.0]
    for _ in 1:d
        diff_reg = polymul(diff_reg, [1.0, -1.0])
    end
    diff_seas = [1.0]
    for _ in 1:D
        diff_seas = polymul(diff_seas, seasonal_poly([1.0], s; sign=-1.0))
    end
    ar_full_natural = polymul(polymul(ar_reg_natural, ar_seas_natural), polymul(diff_reg, diff_seas))
    ar_full = -ar_full_natural[2:end]
    psi = psi_weights(ar_full, ma, horizon)
    se = [sqrt(sigma2 * sum(abs2, view(psi, 1:h))) for h in 1:horizon]

    z = [_confidence_z(1 - l/100) for l in level]
    lower = reduce(hcat, [point .- zi .* se for zi in z])
    upper = reduce(hcat, [point .+ zi .* se for zi in z])
    P_, Q_ = length(model.Phi), length(model.Theta)
    model_name = "ARIMA($p,$d,$q)($P_,$D,$Q_)[$s]"
    return Forecast(point, se, Float64.(level), lower, upper, horizon, model_name)
end

"""
    forecast(model::SarimaModel, y, horizon; level=[80.0, 95.0]) -> Forecast

Forecast `horizon` steps ahead from a fitted [`SarimaModel`](@ref); see
[`forecast`](@ref)`(::ArimaModel, ...)` for the underlying method, which
this generalizes exactly the way `combined_ar_ma` generalizes
`build_statespace` to the seasonal case -- both differencing
orders (`d` at lag 1, `D` at lag `model.seasonal_order[4]`) are
re-integrated, in the reverse of the order [`fit_sarima`](@ref) applied
them (`D` first, then `d`, when differencing; `d` undone first, then
`D`, when re-integrating), and the prediction-interval psi-weights see
the full seasonal-and-differenced AR polynomial
(`φ(B)Φ(Bˢ)(1-B)^d(1-Bˢ)^D`), not just the stationary part that was fit.

`y` must be supplied -- see `StatsAPI.predict`'s own docstring for why.

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = Float64.(readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)[:, 2]);

julia> m = fit_sarima(log.(y), (0, 1, 1), (0, 1, 1, 12));

julia> f = forecast(m, log.(y), 12);

julia> f.horizon
12
```
"""
forecast(model::SarimaModel, y, horizon::Integer; level::Vector{<:Real}=[80.0, 95.0]) =
    StatsAPI.predict(model, y, horizon; level=level)
