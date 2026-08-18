export holt_winters, ExponentialSmoothingModel

"""
    ExponentialSmoothingModel <: UnivariateModel

Result of [`holt_winters`](@ref). Simple exponential smoothing (no
`trend`/`seasonal`), Holt's linear method (`trend` only), or full
Holt-Winters (`trend` and/or `seasonal`) are all represented by this one
struct -- unused components are `nothing`.

- `alpha`/`beta`/`gamma`: fitted (or user-fixed) smoothing parameters.
  `beta`/`gamma` are `nothing` when there is no trend/seasonal component.
- `level`/`trend_component`/`seasonal_component`: the state path, one
  entry per fitted point (`fitted`/`resid` are the same length).
- `sse`: sum of squared one-step-ahead errors, the quantity `alpha`/
  `beta`/`gamma` were fit to minimize.
- `seasonal_type`/`period`: `nothing` unless a seasonal component was fit.
- `initialization_method`: `:heuristic` or `:estimated` -- see
  [`holt_winters`](@ref).
"""
struct ExponentialSmoothingModel <: UnivariateModel
    alpha::Float64
    beta::Union{Nothing,Float64}
    gamma::Union{Nothing,Float64}
    level::Vector{Float64}
    trend_component::Union{Nothing,Vector{Float64}}
    seasonal_component::Union{Nothing,Vector{Float64}}
    fitted::Vector{Float64}
    resid::Vector{Float64}
    sse::Float64
    seasonal_type::Union{Nothing,Symbol}
    period::Union{Nothing,Int}
    initialization_method::Symbol
end

"_hw_recursion(yv, period, has_trend, seasonal_type, l0, b0, s0, alpha, beta, gamma) -- the
Holt-Winters state recursion, generic in the element type of the state/
parameters (plain `Float64` for a direct fixed-parameter fit, or
`ForwardDiff.Dual` when called from inside `_optimize`'s objective).
Matches R's `stats::HoltWinters()` C-level recursion **exactly**
(verified this session by reproducing its fitted/level/trend/season
output bit-for-bit -- to 1e-13 -- on `data/airpassengers.csv` across
additive/multiplicative/with-trend/without-trend cases; see
`handoff/stage-5.5-holtwinters-handoff.md`). The one non-obvious point,
found only by comparing R's actual second-seasonal-cycle output (the
first cycle can't distinguish this, since every season slot is still at
its untouched initial value): the seasonal update `s_t` uses the
**already-updated** `l_t`, not `l_{t-1}+b_{t-1}` -- i.e.
`s_t = gamma*(y_t - l_t) + (1-gamma)*s_{t-m}` (additive) /
`gamma*(y_t/l_t) + (1-gamma)*s_{t-m}` (multiplicative), not the
`l_{t-1}+b_{t-1}` form some textbook presentations use. `start` is the
number of leading observations consumed by initialization before the
first fitted point (`period` if seasonal, `2` for Holt, `1` for simple
ES -- matches R's `start.time` exactly)."
function _hw_recursion(yv::Vector{Float64}, period::Integer, has_trend::Bool,
                        seasonal_type::Union{Nothing,Symbol},
                        l0, b0, s0::AbstractVector, alpha, beta, gamma)
    T = promote_type(typeof(float(l0)), typeof(float(b0)), eltype(float.(s0)),
                      typeof(float(alpha)), typeof(float(beta)), typeof(float(gamma)))
    n = length(yv)
    m = seasonal_type === nothing ? 0 : period
    start = seasonal_type !== nothing ? m : (has_trend ? 2 : 1)
    l = T(l0)
    b = T(b0)
    s = T.(s0)
    npts = n - start
    fitted = Vector{T}(undef, npts)
    levels = Vector{T}(undef, npts)
    trends = has_trend ? Vector{T}(undef, npts) : T[]
    seasons = seasonal_type !== nothing ? Vector{T}(undef, npts) : T[]
    sse = zero(T)
    for (i, t) in enumerate((start + 1):n)
        si = seasonal_type !== nothing ? mod1(t, m) : 1
        lb = has_trend ? l + b : l
        xhat = seasonal_type === :multiplicative ? lb * s[si] :
               seasonal_type === :additive ? lb + s[si] : lb
        fitted[i] = xhat
        sse += (yv[t] - xhat)^2
        l_new = seasonal_type === :multiplicative ? alpha * (yv[t] / s[si]) + (1 - alpha) * lb :
                seasonal_type === :additive ? alpha * (yv[t] - s[si]) + (1 - alpha) * lb :
                alpha * yv[t] + (1 - alpha) * lb
        b_new = has_trend ? beta * (l_new - l) + (1 - beta) * b : b
        if seasonal_type !== nothing
            s_new = seasonal_type === :multiplicative ? gamma * (yv[t] / l_new) + (1 - gamma) * s[si] :
                    gamma * (yv[t] - l_new) + (1 - gamma) * s[si]
            s[si] = s_new
            seasons[i] = s_new
        end
        levels[i] = l_new
        has_trend && (trends[i] = b_new)
        l, b = l_new, b_new
    end
    return (sse=sse, fitted=fitted, level=levels,
            trend=(has_trend ? trends : nothing),
            seasonal=(seasonal_type !== nothing ? seasons : nothing), start=start)
end

"_hw_heuristic_init(yv, m, has_trend, seasonal_type) -- deterministic
initial level/trend/seasonal states, matching R's `stats::HoltWinters()`
initialization exactly: for a seasonal model, `classical_decompose`
(Stage 3.1, itself already verified against R's `decompose()`) on the
first `2*period` points, OLS of the extracted trend against its own
compact position index for `l0`/`b0`, and the decomposition's `figure`
directly for `s0` -- literally R's own algorithm, since R's C source
calls `decompose()` too (see the Stage 5.5 handoff doc section 1).
Non-seasonal: `l0=y[2],b0=y[2]-y[1]` (Holt) or `l0=y[1]` (simple ES),
matching R's `expsmooth` branch exactly."
function _hw_heuristic_init(yv::Vector{Float64}, m::Integer, has_trend::Bool,
                             seasonal_type::Union{Nothing,Symbol})
    if seasonal_type !== nothing
        wind = 2 * m
        decomp = classical_decompose(yv[1:wind], m; model=seasonal_type)
        valid = findall(!isnan, decomp.trend)
        X = hcat(ones(length(valid)), Float64.(1:length(valid)))
        coefs, = _ols(X, decomp.trend[valid])
        return coefs[1], coefs[2], copy(decomp.figure)
    elseif has_trend
        return yv[2], yv[2] - yv[1], Float64[]
    else
        return yv[1], 0.0, Float64[]
    end
end

"""
    holt_winters(y, period=nothing; trend=nothing, seasonal=nothing,
                 initialization_method=:heuristic, alpha=nothing, beta=nothing, gamma=nothing)
                 -> ExponentialSmoothingModel

Exponential smoothing (simple / Holt's linear / Holt-Winters), fit by
minimizing the sum of squared one-step-ahead errors (SSE).

- `trend`: `nothing` (no trend) or `:additive`.
- `seasonal`: `nothing`, `:additive`, or `:multiplicative`; requires
  `period` when given.
- `alpha`/`beta`/`gamma`: `nothing` (default) estimates the parameter;
  a value in `[0, 1]` pins it. `beta` only applies with `trend`,
  `gamma` only with `seasonal`.
- `initialization_method`:
  - `:heuristic` (default) -- fixed initial level/trend/seasonal states
    computed once via `classical_decompose` (Stage 3.1) plus OLS,
    **matching R's `stats::HoltWinters()` exactly** (verified this
    session by reproducing its fitted values to 1e-13 on
    `TSAnalytics.AIR_PASSENGERS` -- see `handoff/stage-5.5-holtwinters-handoff.md`).
    Only `alpha`/`beta`/`gamma` (whichever aren't pinned) are optimized;
    the initial states stay fixed throughout.
  - `:estimated` -- initial level/trend/seasonal states are optimized
    *jointly* with `alpha`/`beta`/`gamma`, matching Python's
    `statsmodels.tsa.holtwinters.ExponentialSmoothing` default
    philosophy.

**`:heuristic` and `:estimated` are not just two starting guesses for
the same fit -- they solve genuinely different optimization problems**
and can converge to substantially different `alpha`/`beta`/`gamma` and
SSE on identical data (confirmed by direct execution of both R's and
Python's actual defaults on the same series -- see the Stage 5.5 handoff
doc section 3). `:heuristic` is the default here because it reuses
already-verified code (`classical_decompose`), is deterministic, and
only ever searches 1-3 dimensions.

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> ap = readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1);

julia> y = Float64.(ap[:, 2]);

julia> m = holt_winters(y, 12; trend=:additive, seasonal=:additive, alpha=0.3, beta=0.05, gamma=0.2);

julia> round(m.sse, digits=4)
92402.9671
```
"""
function holt_winters(y, period::Union{Nothing,Integer}=nothing;
                       trend::Union{Nothing,Symbol}=nothing,
                       seasonal::Union{Nothing,Symbol}=nothing,
                       initialization_method::Symbol=:heuristic,
                       alpha::Union{Nothing,Real}=nothing,
                       beta::Union{Nothing,Real}=nothing,
                       gamma::Union{Nothing,Real}=nothing)
    initialization_method in (:heuristic, :estimated) ||
        throw(ArgumentError("initialization_method must be :heuristic or :estimated"))
    trend === nothing || trend == :additive ||
        throw(ArgumentError("trend must be nothing or :additive"))
    seasonal === nothing || seasonal in (:additive, :multiplicative) ||
        throw(ArgumentError("seasonal must be nothing, :additive, or :multiplicative"))
    seasonal !== nothing && period === nothing &&
        throw(ArgumentError("seasonal smoothing requires period"))
    seasonal !== nothing && period !== nothing && period < 2 &&
        throw(ArgumentError("period must be >= 2"))
    beta !== nothing && trend === nothing &&
        throw(ArgumentError("beta given but trend is nothing (no trend component to smooth)"))
    gamma !== nothing && seasonal === nothing &&
        throw(ArgumentError("gamma given but seasonal is nothing (no seasonal component to smooth)"))
    for (nm, v) in (("alpha", alpha), ("beta", beta), ("gamma", gamma))
        v === nothing || (0 <= v <= 1) ||
            throw(ArgumentError("$nm must be in [0, 1]"))
    end
    alpha == 0 && throw(ArgumentError("alpha must not be 0 (cannot fit a model without a level)"))

    yv = Float64.(collect(tsvalues(y)))
    n = length(yv)
    has_trend = trend !== nothing
    m = seasonal === nothing ? 0 : Int(period)
    seasonal !== nothing && n < 2 * m &&
        throw(ArgumentError("need at least 2 full periods ($(2*m) observations), got $n"))
    !has_trend && seasonal === nothing && n < 2 &&
        throw(ArgumentError("need at least 2 observations"))
    has_trend && seasonal === nothing && n < 3 &&
        throw(ArgumentError("need at least 3 observations for Holt's linear method"))

    l0h, b0h, s0h = _hw_heuristic_init(yv, m, has_trend, seasonal)

    free_syms = Symbol[]
    alpha === nothing && push!(free_syms, :alpha)
    has_trend && beta === nothing && push!(free_syms, :beta)
    seasonal !== nothing && gamma === nothing && push!(free_syms, :gamma)

    _extract(theta, j0) = begin
        p = Dict{Symbol,eltype(theta)}()
        j = j0
        for s in free_syms
            p[s] = 1 / (1 + exp(-theta[j]))
            j += 1
        end
        p
    end

    # method=:nelder_mead, not the package default :lbfgs: verified by
    # direct testing (not assumed) that :lbfgs takes a single raw,
    # unregularized step on this sigmoid-bounded SSE objective and
    # falsely reports convergence once alpha/beta/gamma saturate near
    # 0/1 (where the sigmoid's gradient vanishes) -- e.g. converging to
    # SSE=93495 in 1 iteration on `AIR_PASSENGERS` where the true optimum
    # is SSE=21860 (confirmed against real R). :nelder_mead, gradient-free
    # and step-size-adaptive, reliably finds R's exact optimum on the same
    # problem (alpha/beta/gamma and SSE match to 5+ digits) and the
    # optimization here is low-dimensional for :heuristic (at most 3
    # smoothing parameters) but can reach period+3 dimensions for
    # :estimated (initial states are free too) -- verified directly that
    # NelderMead's default 1000-iteration budget is not always enough at
    # that size (12 monthly seasonal states + 3 smoothing + level + trend
    # = 17 dims on `AIR_PASSENGERS` needed ~12900 iterations to actually
    # converge, not 1000), hence the generous `iterations` below rather
    # than trusting the default blindly.
    if initialization_method == :heuristic
        l0, b0, s0 = l0h, b0h, s0h
        if isempty(free_syms)
            fitalpha, fitbeta, fitgamma = alpha, (has_trend ? beta : 0.0), (seasonal !== nothing ? gamma : 0.0)
        else
            x0 = zeros(length(free_syms))
            objective_h(theta) = begin
                p = _extract(theta, 1)
                a = alpha === nothing ? p[:alpha] : alpha
                bb = has_trend ? (beta === nothing ? p[:beta] : beta) : 0.0
                g = seasonal !== nothing ? (gamma === nothing ? p[:gamma] : gamma) : 0.0
                _hw_recursion(yv, m, has_trend, seasonal, l0, b0, s0, a, bb, g).sse
            end
            res = _optimize(objective_h, x0; method=:nelder_mead, iterations=20_000)
            p = _extract(res.minimizer, 1)
            fitalpha = alpha === nothing ? p[:alpha] : alpha
            fitbeta = has_trend ? (beta === nothing ? p[:beta] : beta) : 0.0
            fitgamma = seasonal !== nothing ? (gamma === nothing ? p[:gamma] : gamma) : 0.0
        end
    else # :estimated
        n_free_p = length(free_syms)
        x0 = vcat(zeros(n_free_p), [l0h], has_trend ? [b0h] : Float64[], seasonal !== nothing ? s0h : Float64[])
        objective_e(theta) = begin
            p = _extract(theta, 1)
            a = alpha === nothing ? p[:alpha] : alpha
            bb = has_trend ? (beta === nothing ? p[:beta] : beta) : 0.0
            g = seasonal !== nothing ? (gamma === nothing ? p[:gamma] : gamma) : 0.0
            j = n_free_p + 1
            l0v = theta[j]; j += 1
            b0v = has_trend ? theta[j] : 0.0
            has_trend && (j += 1)
            s0v = seasonal !== nothing ? theta[j:(j + m - 1)] : eltype(theta)[]
            _hw_recursion(yv, m, has_trend, seasonal, l0v, b0v, s0v, a, bb, g).sse
        end
        res = _optimize(objective_e, x0; method=:nelder_mead, iterations=20_000)
        theta = res.minimizer
        p = _extract(theta, 1)
        fitalpha = alpha === nothing ? p[:alpha] : alpha
        fitbeta = has_trend ? (beta === nothing ? p[:beta] : beta) : 0.0
        fitgamma = seasonal !== nothing ? (gamma === nothing ? p[:gamma] : gamma) : 0.0
        j = n_free_p + 1
        l0 = theta[j]; j += 1
        b0 = has_trend ? theta[j] : 0.0
        has_trend && (j += 1)
        s0 = seasonal !== nothing ? theta[j:(j + m - 1)] : Float64[]
    end

    result = _hw_recursion(yv, m, has_trend, seasonal, l0, b0, s0, fitalpha, fitbeta, fitgamma)
    start = result.start
    resid = yv[(start + 1):n] .- result.fitted

    return ExponentialSmoothingModel(fitalpha, has_trend ? fitbeta : nothing,
                                      seasonal !== nothing ? fitgamma : nothing,
                                      result.level, result.trend, result.seasonal,
                                      result.fitted, resid, result.sse,
                                      seasonal, seasonal !== nothing ? m : nothing,
                                      initialization_method)
end

function Base.show(io::IO, mdl::ExponentialSmoothingModel)
    kind = mdl.seasonal_type !== nothing ? "Holt-Winters ($(mdl.seasonal_type))" :
           mdl.trend_component !== nothing ? "Holt's linear" : "Simple exponential smoothing"
    println(io, kind, " (initialization: ", mdl.initialization_method, ")")
    println(io, "  alpha : ", round(mdl.alpha, digits=4))
    mdl.beta !== nothing && println(io, "  beta  : ", round(mdl.beta, digits=4))
    mdl.gamma !== nothing && println(io, "  gamma : ", round(mdl.gamma, digits=4))
    print(io, "  SSE   : ", round(mdl.sse, digits=4))
end
