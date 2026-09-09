export forecast, Forecast, mean_forecast, naive, seasonal_naive, drift

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
