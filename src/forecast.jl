export forecast, Forecast

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
