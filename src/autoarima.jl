export auto_arima

_nobs_model(m::ArimaModel) = m.arma.nobs
_nobs_model(m::SarimaModel) = m.nobs
_nparam_model(m::ArimaModel) = length(m.arma.ar) + length(m.arma.ma) + (m.arma.mean !== nothing ? 1 : 0)
_nparam_model(m::SarimaModel) = length(m.phi) + length(m.theta) + length(m.Phi) + length(m.Theta) +
                                 (m.mean !== nothing ? 1 : 0)

"_ic_value(m, ic) -- AIC/BIC read directly off the fitted model; AICc
computed here (not stored on either model type) as
`aic + 2k(k+1)/(n-k-1)`, `k = nparam+1` matching the same +1-for-sigma2
convention `fit_arma`/`fit_sarima` already use for their own `aic`/`bic`.
`Inf` (never negative/NaN) when `n-k-1 <= 0`, so an overparametrized
candidate a stepwise/exhaustive search stumbles into is disqualified by
comparison rather than crashing the search."
function _ic_value(m, ic::Symbol)
    ic === :aic && return StatsAPI.aic(m)
    ic === :bic && return StatsAPI.bic(m)
    n = _nobs_model(m)
    k = _nparam_model(m) + 1
    denom = n - k - 1
    denom > 0 || return Inf
    return StatsAPI.aic(m) + (2 * k * (k + 1)) / denom
end

"_try_fit -- fits one (p,q,P,Q) candidate at fixed d/D/s, returning
`nothing` (never rethrowing) when the candidate is structurally
infeasible (`fit_arima`/`fit_sarima`'s own `ArgumentError`s: too few
observations after differencing, etc.) -- matches R/`pmdarima`'s own
behavior of silently skipping candidates that can't be fit rather than
aborting the whole search."
function _try_fit(yv, p, d, q, P, D, Q, s, seasonal, include_mean, method, se_type, optimizer_method)
    try
        return seasonal ?
               fit_sarima(yv, (p, d, q), (P, D, Q, s); include_mean=include_mean, method=method,
                          se_type=se_type, optimizer_method=optimizer_method) :
               fit_arima(yv, (p, d, q); include_mean=include_mean, method=method,
                         se_type=se_type, optimizer_method=optimizer_method)
    catch e
        e isa ArgumentError || rethrow()
        return nothing
    end
end

"_select_d -- repeated KPSS test (Stage 2.2), matching both references'
`test='kpss'`/`test=\"kpss\"` default: difference while the null of
stationarity is rejected (`pvalue <= alpha`), up to `max_d`. Stops early
(without erroring) if the series becomes too short to test or
`kpss_test` itself errors on a degenerate (e.g. constant) series --
that's treated as \"nothing more to detect\", not a failure."
function _select_d(y0::Vector{Float64}, max_d::Integer, alpha::Real)
    y = y0
    d = 0
    while d < max_d && length(y) > 10
        pv = try
            kpss_test(y).pvalue
        catch
            break
        end
        pv > alpha && break
        y = diff(y, 1; differences=1)
        d += 1
    end
    return d
end

"_stepwise_search -- Hyndman-Khandakar's actual algorithm (Hyndman &
Khandakar 2008 §3.2): fit four base models
(`(2,d,2)(1,D,1)`, `(0,d,0)(0,D,0)`, `(1,d,0)(1,D,0)`, `(0,d,1)(0,D,1)`,
each capped at `max_p`/`max_q`/`max_P`/`max_Q`), take the best by
`ic`, then repeatedly consider the ±1 neighbors of the current best on
each of p/q/P/Q, refit them, and move to the single best-improving
neighbor -- stopping when no neighbor improves `ic`. A plain greedy
hill-climb over a finite, `max_order`-bounded grid, so it always
terminates; a small `cache` avoids refitting a candidate visited more
than once (neighbor sets from adjacent steps can overlap).

`fitfn(p, q, P, Q)` is the only piece that differs between `auto_arima`
(fits `fit_arima`/`fit_sarima`) and Stage 8.4's `auto_arimax` (fits
`fit_arimax`/`fit_sarimax`, `d`/`D`/`s`/`exog` already closed over by
the caller) -- the search algorithm itself (candidate generation, IC
comparison, caching, trace printing) is identical either way, so it's
injected rather than duplicated."
function _stepwise_search(fitfn, seasonal, max_p, max_q, max_P, max_Q, max_order, ic, trace)
    cache = Dict{NTuple{4,Int},Any}()

    function fitcand(p, q, P, Q)
        key = (p, q, P, Q)
        haskey(cache, key) && return cache[key]
        if p + q + P + Q > max_order
            cache[key] = nothing
            return nothing
        end
        m = fitfn(p, q, P, Q)
        cache[key] = m
        if trace
            label = seasonal ? "(p=$p,q=$q)(P=$P,Q=$Q)" : "(p=$p,q=$q)"
            println("  ", label, " : ", m === nothing ? "failed" : string(ic, "=", round(_ic_value(m, ic), digits=4)))
        end
        return m
    end

    p0, q0 = min(2, max_p), min(2, max_q)
    p1, q1 = min(1, max_p), min(1, max_q)
    P0 = seasonal ? min(1, max_P) : 0
    Q0 = seasonal ? min(1, max_Q) : 0

    starts = unique([(p0, q0, P0, Q0), (0, 0, 0, 0), (p1, 0, P0, 0), (0, q1, 0, Q0)])

    best, best_ic, best_key = nothing, Inf, nothing
    for key in starts
        m = fitcand(key...)
        m === nothing && continue
        v = _ic_value(m, ic)
        isfinite(v) || continue
        if v < best_ic
            best, best_ic, best_key = m, v, key
        end
    end

    best === nothing && throw(ArgumentError(
        "auto_arima: no candidate model in the initial search could be fit -- " *
        "try widening max_p/max_q/max_order or check the series has enough observations"))

    improved = true
    while improved
        improved = false
        p, q, P, Q = best_key
        neighbors = NTuple{4,Int}[]
        p < max_p && push!(neighbors, (p + 1, q, P, Q))
        p > 0 && push!(neighbors, (p - 1, q, P, Q))
        q < max_q && push!(neighbors, (p, q + 1, P, Q))
        q > 0 && push!(neighbors, (p, q - 1, P, Q))
        if seasonal
            P < max_P && push!(neighbors, (p, q, P + 1, Q))
            P > 0 && push!(neighbors, (p, q, P - 1, Q))
            Q < max_Q && push!(neighbors, (p, q, P, Q + 1))
            Q > 0 && push!(neighbors, (p, q, P, Q - 1))
        end

        local_best, local_best_ic, local_best_key = nothing, Inf, nothing
        for key in neighbors
            m = fitcand(key...)
            m === nothing && continue
            v = _ic_value(m, ic)
            isfinite(v) || continue
            if v < local_best_ic
                local_best, local_best_ic, local_best_key = m, v, key
            end
        end

        if local_best !== nothing && local_best_ic < best_ic - 1e-8
            best, best_ic, best_key = local_best, local_best_ic, local_best_key
            improved = true
        end
    end

    return best
end

"_exhaustive_search -- `stepwise=false`: the genuinely parallel mode
(handoff §4, confirmed by both references' own docs: R's
`parallel=`/`num.cores=` and Python's `n_jobs=` apply *only* here).
Fits every `(p,q,P,Q)` with `p+q+P+Q <= max_order` over
`Threads.@threads` when `parallel=true` and more than one thread is
actually available (and the grid is non-trivial) -- same guarded
pattern as MSTL's own `parallel` keyword. Threading changes only which
order the candidates are fit in, never which one wins: every thread
writes to its own `results[i]` slot, and the arg-min pass is strictly
sequential afterwards.

`fitfn(p, q, P, Q)` -- see `_stepwise_search`'s own docstring for why
this is injected rather than duplicated between `auto_arima` and
Stage 8.4's `auto_arimax`."
function _exhaustive_search(fitfn, seasonal, max_p, max_q, max_P, max_Q, max_order, ic, parallel, trace)
    P_range = seasonal ? (0:max_P) : (0:0)
    Q_range = seasonal ? (0:max_Q) : (0:0)
    candidates = vec([(p, q, P, Q) for p in 0:max_p, q in 0:max_q, P in P_range, Q in Q_range
                       if p + q + P + Q <= max_order])
    isempty(candidates) && throw(ArgumentError("auto_arima: no candidate orders satisfy max_order=$max_order"))

    results = Vector{Any}(undef, length(candidates))
    use_threads = parallel && Threads.nthreads() > 1 && length(candidates) >= 4
    if use_threads
        Threads.@threads for i in eachindex(candidates)
            p, q, P, Q = candidates[i]
            results[i] = fitfn(p, q, P, Q)
        end
    else
        for i in eachindex(candidates)
            p, q, P, Q = candidates[i]
            results[i] = fitfn(p, q, P, Q)
        end
    end

    best, best_ic = nothing, Inf
    for (i, m) in enumerate(results)
        m === nothing && continue
        v = _ic_value(m, ic)
        isfinite(v) || continue
        if trace
            p, q, P, Q = candidates[i]
            label = seasonal ? "(p=$p,q=$q)(P=$P,Q=$Q)" : "(p=$p,q=$q)"
            println("  ", label, " : ", ic, "=", round(v, digits=4))
        end
        if v < best_ic
            best, best_ic = m, v
        end
    end
    best === nothing && throw(ArgumentError("auto_arima: no candidate model could be fit in the exhaustive search"))
    return best
end

"""
    auto_arima(y; d=nothing, D=nothing, max_p=5, max_q=5, max_P=2, max_Q=2,
               max_order=5, max_d=2, max_D=1, seasonal=false, m=1,
               information_criterion=:aicc, alpha=0.05, stepwise=true,
               parallel=true, include_mean=true, method=:ml, se_type=:hessian,
               optimizer_method=:lbfgs, trace=false) -> Union{ArimaModel,SarimaModel}

Automatic order selection (Hyndman & Khandakar 2008), built entirely on
top of [`fit_arima`](@ref)/[`fit_sarima`](@ref) (Stage 6.6/6.7) -- this
function contributes no new fitting math of its own, only the search
over `(p,q,P,Q)` and (optionally) `d`.

**`information_criterion` defaults to `:aicc`**, matching R
`forecast::auto.arima`'s default (`ic="aicc"`) rather than Python
`pmdarima`'s default (`'aic'`) -- AICc is the Hyndman-Khandakar paper's
own recommendation for finite samples; `:aic` and `:bic` are both also
accepted.

**`d`/`D` -- `nothing` means "detect automatically"**, matching both
references' `NA`/`None` convention (Julia's `nothing` in their place).
`d` is detected by repeated [`kpss_test`](@ref) (matching both
references' `test="kpss"`/`test='kpss'` default): difference while the
null of stationarity is rejected, up to `max_d`.

**`D` cannot be auto-detected in this version -- `seasonal=true`
requires `D` passed explicitly.** Neither of the two seasonal
unit-root tests either reference uses for this (R's default
Canova-Hansen, Python's default OCSB) exists anywhere in this project
yet; rather than silently default `D` to something unverified, this is
a documented, temporary limitation (see
`handoff/stage-6.8-autoarima-handoff.md` §3) -- pass `D` explicitly
(commonly `0` or `1`) for seasonal data.

**`include_mean` is passed straight through, not searched** -- unlike
both references, which also search a constant on/off as part of the
stepwise procedure. A deliberate, documented simplification for this
first version (same "narrower, correct thing first" scoping as
`D`-detection above), not an oversight; `fit_arima`/`fit_sarima` still
silently force it off whenever `d>0` or `D>0`, exactly as in Stage
6.6/6.7.

**`stepwise` (default `true`) is inherently sequential** -- the actual
Hyndman-Khandakar greedy hill-climb: four base models, then repeated
±1 neighbor moves on `p`/`q`/`P`/`Q`, taking the single best-improving
move each round until none improves. **`stepwise=false` is a full grid
search over every `(p,q,P,Q)` with `p+q+P+Q <= max_order`, and is the
only mode `parallel` affects** (`Threads.@threads`, guarded the same
way as MSTL's own `parallel` keyword) -- confirmed as the literal
documented scope of both R's `parallel=`/`num.cores=` and Python's
`n_jobs=`; `parallel` is a silent no-op when `stepwise=true`, since a
sequential hill-climb has nothing to parallelize.

Candidates that can't be fit (too few observations after differencing,
etc.) are silently skipped, matching both references -- not an error,
unless *no* candidate anywhere in the search could be fit.

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "arima", "d1_clean.csv"), ','; skipstart=1));

julia> m = auto_arima(y; max_p=3, max_q=3, max_d=2);

julia> m isa ArimaModel
true

julia> m.d  # differencing order auto-selected via repeated kpss_test
1
```
"""
function auto_arima(y; d::Union{Nothing,Integer}=nothing, D::Union{Nothing,Integer}=nothing,
                     max_p::Integer=5, max_q::Integer=5, max_P::Integer=2, max_Q::Integer=2,
                     max_order::Integer=5, max_d::Integer=2, max_D::Integer=1,
                     seasonal::Bool=false, m::Integer=1,
                     information_criterion::Symbol=:aicc, alpha::Real=0.05,
                     stepwise::Bool=true, parallel::Bool=true,
                     include_mean::Bool=true, method::Symbol=:ml, se_type::Symbol=:hessian,
                     optimizer_method::Symbol=:lbfgs, trace::Bool=false)
    information_criterion in (:aic, :aicc, :bic) ||
        throw(ArgumentError("information_criterion must be :aic, :aicc, or :bic"))
    max_p >= 0 && max_q >= 0 && max_P >= 0 && max_Q >= 0 ||
        throw(ArgumentError("max_p/max_q/max_P/max_Q must be non-negative"))
    max_order >= 0 || throw(ArgumentError("max_order must be non-negative"))
    max_d >= 0 && max_D >= 0 || throw(ArgumentError("max_d/max_D must be non-negative"))
    0 < alpha < 1 || throw(ArgumentError("alpha must be in (0, 1)"))

    Dsel = 0
    s = 1
    if seasonal
        m >= 2 || throw(ArgumentError("seasonal=true requires m >= 2, got m=$m"))
        D === nothing && throw(ArgumentError(
            "auto_arima: automatic seasonal-differencing-order (D) detection is not yet " *
            "implemented -- no Canova-Hansen/OCSB seasonal unit-root test exists in this " *
            "project yet (handoff/stage-6.8-autoarima-handoff.md §3). Pass D explicitly " *
            "for seasonal=true."))
        D >= 0 || throw(ArgumentError("D must be non-negative"))
        D <= max_D || throw(ArgumentError("D=$D exceeds max_D=$max_D"))
        Dsel = D
        s = m
    end

    yv = Float64.(collect(tsvalues(y)))
    yseas = Dsel > 0 ? diff(yv, s; differences=Dsel) : yv

    dsel = d === nothing ? _select_d(yseas, max_d, alpha) : Int(d)
    dsel >= 0 || throw(ArgumentError("d must be non-negative"))
    dsel <= max_d || throw(ArgumentError("d=$dsel exceeds max_d=$max_d"))

    fitfn(p, q, P, Q) = _try_fit(yv, p, dsel, q, P, Dsel, Q, s, seasonal, include_mean, method, se_type, optimizer_method)

    return stepwise ?
           _stepwise_search(fitfn, seasonal, max_p, max_q, max_P, max_Q, max_order, information_criterion, trace) :
           _exhaustive_search(fitfn, seasonal, max_p, max_q, max_P, max_Q, max_order, information_criterion,
                               parallel, trace)
end
