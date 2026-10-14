# ---------------------------------------------------------------------------
# Stage 11.1 -- Unobserved components (structural time series).
# handoff/stage 11/docs/stage-11.1-unobserved-components-handoff.md
#
# No new numerics: a component specification is mapped onto (Z, T, R, Q, H)
# and handed to the existing TimeVaryingSSM filter and smoother, with the
# partitioned prior of Stage 11.0c. Primary reference statsmodels
# `UnobservedComponents` (Harvey 1989; Durbin & Koopman 2012, ch. 3);
# R's `StructTS` an independent check.
# ---------------------------------------------------------------------------

export fit_uc, UCModel, uc_decomposition, UCDecomposition,
       uc_local_level, uc_local_linear_trend, uc_random_walk, uc_smooth_trend, uc_bsm

struct _UCSpec
    level::Symbol            # :none | :deterministic | :stochastic
    trend::Symbol
    period::Int              # 0 = no seasonal
    seasonal_kind::Symbol    # :dummy | :trig
    harmonics::Int
    seasonal_stochastic::Bool
    cycle::Symbol
    damped_cycle::Bool
    ar::Int
    irregular::Bool
    nexog::Int
end

# State layout: index ranges of each component, and the parameter names in
# the reference's order (sigma2.irregular, sigma2.level, sigma2.trend,
# sigma2.seasonal | sigma2.freq_seasonal_s(h), sigma2.cycle,
# frequency.cycle, damping.cycle, sigma2.ar, ar.L1..Lp, beta.x1..xk).
struct _UCLayout
    r::Int
    level::UnitRange{Int}
    trend::UnitRange{Int}
    seasonal::UnitRange{Int}
    cycle::UnitRange{Int}
    ar::UnitRange{Int}
    zload::Vector{Float64}        # observation loadings (Z)
    seasonal_load::Vector{Int}    # seasonal states that load on y
    names::Vector{String}
    nshock::Int                   # columns of R
end

function _uc_layout(s::_UCSpec)
    i = 0
    take(k) = (rng = (i + 1):(i + k); i += k; rng)
    lv = s.level === :none ? (1:0) : take(1)
    tr = s.trend === :none ? (1:0) : take(1)
    se = s.period == 0 ? (1:0) :
         s.seasonal_kind === :dummy ? take(s.period - 1) : take(2 * s.harmonics)
    cy = s.cycle === :none ? (1:0) : take(2)
    ar = take(s.ar)
    r = i
    z = zeros(r)
    isempty(lv) || (z[first(lv)] = 1.0)
    sload = Int[]
    if !isempty(se)
        if s.seasonal_kind === :dummy
            push!(sload, first(se))
        else
            append!(sload, first(se):2:last(se))      # the cosine state of each harmonic
        end
        z[sload] .= 1.0
    end
    isempty(cy) || (z[first(cy)] = 1.0)
    isempty(ar) || (z[first(ar)] = 1.0)

    names = String[]
    s.irregular && push!(names, "sigma2.irregular")
    s.level === :stochastic && push!(names, "sigma2.level")
    s.trend === :stochastic && push!(names, "sigma2.trend")
    if s.period > 0 && s.seasonal_stochastic
        push!(names, s.seasonal_kind === :dummy ? "sigma2.seasonal" :
                     "sigma2.freq_seasonal_$(s.period)($(s.harmonics))")
    end
    s.cycle === :stochastic && push!(names, "sigma2.cycle")
    s.cycle === :none || push!(names, "frequency.cycle")
    s.cycle !== :none && s.damped_cycle && push!(names, "damping.cycle")
    if s.ar > 0
        push!(names, "sigma2.ar")
        append!(names, ["ar.L$j" for j in 1:s.ar])
    end
    append!(names, ["beta.x$j" for j in 1:s.nexog])

    nshock = (s.level === :stochastic) + (s.trend === :stochastic) +
             (s.period > 0 && s.seasonal_stochastic ?
              (s.seasonal_kind === :dummy ? 1 : 2 * s.harmonics) : 0) +
             (s.cycle === :stochastic ? 2 : 0) + (s.ar > 0)
    return _UCLayout(r, lv, tr, se, cy, ar, z, sload, names, nshock)
end

# Unpack the natural parameter vector into its pieces.
# explicit offsets, not a closure over a mutated index (which Julia boxes)
function _uc_unpack(s::_UCSpec, p::AbstractVector{S}) where {S}
    i = 0
    s2irr = s.irregular ? p[i += 1] : zero(S)
    s2lv = s.level === :stochastic ? p[i += 1] : zero(S)
    s2tr = s.trend === :stochastic ? p[i += 1] : zero(S)
    s2se = s.period > 0 && s.seasonal_stochastic ? p[i += 1] : zero(S)
    s2cy = s.cycle === :stochastic ? p[i += 1] : zero(S)
    lam = s.cycle === :none ? zero(S) : p[i += 1]
    rho = s.cycle !== :none && s.damped_cycle ? p[i += 1] : one(S)
    s2ar = s.ar > 0 ? p[i += 1] : zero(S)
    phi = p[(i + 1):(i + s.ar)]
    beta = p[(i + s.ar + 1):(i + s.ar + s.nexog)]
    return (; s2irr, s2lv, s2tr, s2se, s2cy, lam, rho, s2ar, phi, beta)
end

# (T, Z, R, Q, H) for natural parameters p; generic in eltype (ForwardDiff).
function _uc_system(s::_UCSpec, L::_UCLayout, p::AbstractVector)
    u = _uc_unpack(s, p)
    S = eltype(p) === Any ? Float64 : promote_type(eltype(p), Float64)
    r = L.r
    T = zeros(S, r, r)
    if !isempty(L.level)
        T[first(L.level), first(L.level)] = 1
        isempty(L.trend) || (T[first(L.level), first(L.trend)] = 1)
    end
    isempty(L.trend) || (T[first(L.trend), first(L.trend)] = 1)
    if !isempty(L.seasonal)
        o = first(L.seasonal) - 1
        if s.seasonal_kind === :dummy
            m = s.period - 1
            for j in 1:m
                T[o + 1, o + j] = -1            # sum-to-zero
            end
            for j in 2:m
                T[o + j, o + j - 1] = 1         # lags
            end
        else
            for j in 1:s.harmonics
                lj = 2pi * j / s.period
                c, sn = cos(lj), sin(lj)
                a, b = o + 2j - 1, o + 2j
                T[a, a] = c;  T[a, b] = sn
                T[b, a] = -sn; T[b, b] = c
            end
        end
    end
    if !isempty(L.cycle)
        a, b = first(L.cycle), last(L.cycle)
        c, sn = cos(u.lam), sin(u.lam)
        T[a, a] = u.rho * c;  T[a, b] = u.rho * sn
        T[b, a] = -u.rho * sn; T[b, b] = u.rho * c
    end
    if s.ar > 0
        o = first(L.ar) - 1
        for j in 1:s.ar
            T[o + 1, o + j] = u.phi[j]
        end
        for j in 2:s.ar
            T[o + j, o + j - 1] = 1
        end
    end

    k = max(L.nshock, 1)
    R = zeros(S, r, k)
    qd = zeros(S, k)
    c = 0
    push_shock!(state, var) = (c += 1; R[state, c] = 1; qd[c] = var)
    s.level === :stochastic && push_shock!(first(L.level), u.s2lv)
    s.trend === :stochastic && push_shock!(first(L.trend), u.s2tr)
    if s.period > 0 && s.seasonal_stochastic
        if s.seasonal_kind === :dummy
            push_shock!(first(L.seasonal), u.s2se)
        else
            for st in L.seasonal
                push_shock!(st, u.s2se)
            end
        end
    end
    if s.cycle === :stochastic
        push_shock!(first(L.cycle), u.s2cy); push_shock!(last(L.cycle), u.s2cy)
    end
    s.ar > 0 && push_shock!(first(L.ar), u.s2ar)
    Q = Matrix{S}(Diagonal(qd))
    Z = reshape(S.(L.zload), 1, r)
    H = fill(S(u.s2irr), 1, 1)
    return T, Z, R, Q, H, u.beta
end

_uc_init(L::_UCLayout, kappa::Real) =
    SSMInit(isempty(L.ar) ? [1:L.r => :approximate_diffuse] :
            L.r == length(L.ar) ? [L.ar => :stationary] :
            [1:(first(L.ar) - 1) => :approximate_diffuse, L.ar => :stationary]; kappa=kappa)

# Log-likelihood at natural parameters. The first `burn` periods -- one per
# approximate-diffuse state -- are left out, as the reference does
# (statsmodels' loglikelihood_burn = k_states minus the AR order, measured).
function _uc_loglik(s::_UCSpec, L::_UCLayout, p::AbstractVector, y::AbstractVector,
                    X::Union{Nothing,AbstractMatrix}, kappa::Real, burn::Int,
                    known::Union{Nothing,Tuple})
    T, Z, R, Q, H, beta = _uc_system(s, L, p)
    S = eltype(T)
    ya = X === nothing ? S.(y) : S.(y) .- X * beta
    a0, P0 = if known === nothing
        # an AR block at (numerically) a unit root has no stationary prior;
        # during a search that is an infeasible point, not an error
        res = try
            _resolve_init(T, R * Q * R', _uc_init(L, kappa))
        catch e
            e isa ArgumentError || rethrow()
            return S(-Inf)
        end
        res[1], res[2]
    else
        S.(known[1]), S.(known[2])
    end
    ssm = TimeVaryingSSM{S}([T], [Z], [R], [Q], [H], L.r)
    ll, v, F, ok = kalman_filter(ssm, ya, a0, P0)
    ok || return S(-Inf)
    acc = zero(S); nobs = 0
    for t in (burn + 1):length(ya)
        isnan(v[t]) && continue
        acc += log(F[t]) + v[t]^2 / F[t]
        nobs += 1
    end
    return -0.5 * (nobs * log(2pi) + acc)
end

"""
    UCModel <: StateSpaceModel

A fitted unobserved-components model, returned by [`fit_uc`](@ref).
`coef`, `coefnames`, `vcov`, `stderror`, `loglikelihood`, `aic`, `bic`,
`nobs`, `residuals` (one-step prediction errors), `fitted` (one-step
predictions), and `forecast`/`predict`. Components and their standard errors
come from [`uc_decomposition`](@ref).
"""
struct UCModel <: StateSpaceModel
    y::Vector{Float64}
    exog::Union{Nothing,Matrix{Float64}}
    spec::_UCSpec
    layout::_UCLayout
    params::Vector{Float64}
    loglik::Float64
    nobs::Int
    burn::Int
    kappa::Float64
    known_init::Union{Nothing,Tuple{Vector{Float64},Matrix{Float64}}}
    T::Matrix{Float64}
    Z::Matrix{Float64}
    R::Matrix{Float64}
    Q::Matrix{Float64}
    H::Matrix{Float64}
    fitted::Vector{Float64}
    resid::Vector{Float64}
    alpha::Matrix{Float64}
    V::Vector{Matrix{Float64}}
    converged::Bool
    iterations::Int
    estimated::Bool
end

function _uc_validate(level, trend, seasonal, seasonal_kind, harmonics, seasonal_stochastic,
                      cycle, damped_cycle, ar, irregular)
    for (nm, v) in (("level", level), ("trend", trend), ("cycle", cycle))
        v in (:none, :deterministic, :stochastic) || throw(ArgumentError(
            "fit_uc: $nm must be :none, :deterministic or :stochastic, got $(repr(v))"))
    end
    trend !== :none && level === :none && throw(ArgumentError(
        "fit_uc: a trend needs a level -- the trend is the level's slope"))
    ar >= 0 || throw(ArgumentError("fit_uc: ar must be >= 0, got $ar"))
    period = 0; h = 0
    if seasonal !== nothing
        period = Int(seasonal)
        period >= 2 || throw(ArgumentError("fit_uc: the seasonal period must be >= 2, got $period"))
        seasonal_kind in (:dummy, :trig) || throw(ArgumentError(
            "fit_uc: seasonal_kind must be :dummy or :trig, got $(repr(seasonal_kind))"))
        if seasonal_kind === :trig
            h = harmonics === nothing ? fld(period, 2) : Int(harmonics)
            1 <= h <= fld(period, 2) || throw(ArgumentError(
                "fit_uc: harmonics must be between 1 and floor(period/2) = $(fld(period, 2)), got $h"))
        else
            harmonics === nothing || throw(ArgumentError(
                "fit_uc: harmonics applies to seasonal_kind=:trig only"))
        end
    end
    damped_cycle && cycle === :none && throw(ArgumentError(
        "fit_uc: damped_cycle=true needs a cycle"))
    stochastic = level === :stochastic || trend === :stochastic ||
                 (period > 0 && seasonal_stochastic) || cycle === :stochastic || ar > 0
    (stochastic || irregular) || throw(ArgumentError(
        "fit_uc: this specification has no disturbance at all -- every component is " *
        "deterministic and irregular=false. At least one component must be :stochastic, " *
        "or the irregular must be present. (statsmodels warns and silently adds an " *
        "irregular here; that is a different model from the one asked for.)"))
    return period, h
end

"""
    fit_uc(y; level=:none, trend=:none, seasonal=nothing, seasonal_kind=:dummy,
           harmonics=nothing, seasonal_stochastic=true, cycle=:none,
           damped_cycle=false, ar=0, irregular=true, exog=nothing,
           init_kappa=1e6, params=nothing, fixed=nothing,
           initial_state=nothing, initial_cov=nothing) -> UCModel

Fit an **unobserved-components** (structural time series) model by maximum
likelihood,
```
y_t = μ_t + γ_t + c_t + ψ_t + x_t'β + ε_t
      level  seasonal  cycle  AR  regression  irregular
```
each component with its own variance, then recover every component, with a
standard error, by the smoother ([`uc_decomposition`](@ref)). Unlike
[`stl_decompose`](@ref), which splits a series by a filter you choose, this
splits it by a fitted model, so each component comes with an interval.

**The specification is explicit.** `level`, `trend` and `cycle` each take
`:none`, `:deterministic` or `:stochastic`; there are no Boolean or string
shortcuts, because the reference's are traps (`level=True` gives a
*deterministic* level, and its string names disagree about whether an
irregular is present). The convenience constructors [`uc_local_level`](@ref),
[`uc_local_linear_trend`](@ref), [`uc_random_walk`](@ref),
[`uc_smooth_trend`](@ref) and [`uc_bsm`](@ref) each state their expansion.

- `seasonal`: the period, with `seasonal_kind = :dummy` (`period − 1`
  states, one sum-to-zero disturbance) or `:trig` (`harmonics` rotating
  pairs, default `floor(period/2)`, one common variance);
  `seasonal_stochastic = false` makes it fixed.
- `cycle`: a stochastic or deterministic cycle with estimated frequency in
  `(0, π)`; `damped_cycle = true` adds a damping factor in `(0, 1)`.
- `ar`: an AR(`ar`) component, kept stationary.
- `exog`: regressors with **fixed** coefficients estimated by maximum
  likelihood. For coefficients that drift over time use
  [`fit_arimax`](@ref)`(…; model = :tvss)`, which already does it.

**Initialisation is approximate diffuse, deliberately.** The non-stationary
states get a prior variance `init_kappa = 1e6`; the AR block gets its own
stationary covariance. The package does have an exact diffuse filter
([`kalman_filter_diffuse`](@ref)), and its absence here is not an
oversight: the reference uses approximate diffuse, matching it is what makes
the numbers comparable, and the exact filter has no smoother. `init_kappa`
is a prior, not a technicality -- the likelihood depends on it. As in the
reference, the first period for each diffuse state is left out of the
likelihood -- one period per non-stationary state, with any prior.
`initial_state`/`initial_cov` replace the prior with a known one (R's
`StructTS` uses `y[1]` and `1e4 var(y)`).

**Variances are fitted on the log scale**, so a boundary solution is a small
positive number you can see -- never clipped to zero. (R's `StructTS` clips
with `pmax(·, 0)`, so a true boundary and a clipped negative estimate look
the same.)

`params` evaluates the model at given natural parameters, in
`coefnames` order, without optimising (the reference's `smooth`);
`fixed` pins some parameters by name (`Dict("sigma2.level" => 0.0, …)`) and
estimates the rest.

A specification with no disturbance at all -- every component deterministic
and no irregular -- is refused rather than patched.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> nile = readdlm(TSAnalytics.NILE, ','; skipstart=1)[:, 2];

julia> m = fit_uc(nile; level=:stochastic);

julia> round(m.loglik, digits=4)
-632.5377

julia> round.(m.params, digits=1)
2-element Vector{Float64}:
 15108.3
  1463.5
```
"""
function fit_uc(y; level=:none, trend=:none, seasonal=nothing,
                seasonal_kind::Symbol=:dummy, harmonics=nothing, seasonal_stochastic::Bool=true,
                cycle=:none, damped_cycle::Bool=false, ar::Integer=0,
                irregular::Bool=true, exog=nothing, init_kappa::Real=1e6, params=nothing,
                fixed=nothing, initial_state=nothing, initial_cov=nothing)
    yv = Vector{Float64}(tsvalues(y))
    n = length(yv)
    period, h = _uc_validate(level, trend, seasonal, seasonal_kind, harmonics,
                             seasonal_stochastic, cycle, damped_cycle, ar, irregular)
    X = exog === nothing ? nothing : (exog isa AbstractVector ? reshape(Float64.(exog), :, 1) :
                                      Matrix{Float64}(exog))
    X === nothing || size(X, 1) == n || throw(ArgumentError(
        "fit_uc: exog has $(size(X, 1)) rows, y has $n"))
    X === nothing || !any(isnan, X) || throw(ArgumentError("fit_uc: exog contains NaN"))
    init_kappa > 0 || throw(ArgumentError("fit_uc: init_kappa must be positive"))
    s = _UCSpec(level, trend, period, seasonal_kind, h, seasonal_stochastic, cycle,
                damped_cycle, Int(ar), irregular, X === nothing ? 0 : size(X, 2))
    L = _uc_layout(s)
    known = nothing
    if initial_state !== nothing || initial_cov !== nothing
        (initial_state !== nothing && initial_cov !== nothing) || throw(ArgumentError(
            "fit_uc: give both initial_state and initial_cov, or neither"))
        length(initial_state) == L.r && size(initial_cov) == (L.r, L.r) || throw(ArgumentError(
            "fit_uc: initial_state/initial_cov must match the $(L.r) states"))
        known = (Vector{Float64}(initial_state), Matrix{Float64}(initial_cov))
    end
    # one period per non-stationary state, whatever the prior: the reference
    # fixes its burn with the model, and keeps it under initialize_known
    burn = L.r - s.ar
    nobs = count(!isnan, @view yv[(burn + 1):end])
    np = length(L.names)
    nobs > np || throw(ArgumentError(
        "fit_uc: $nobs usable observations (after $burn left out for the diffuse " *
        "initialisation) for $np parameters"))

    obs = filter(!isnan, yv)
    vy = length(obs) > 1 ? var(obs) : 1.0
    vy > 0 || (vy = 1.0)

    fixed_idx, fixed_val = Int[], Float64[]
    if fixed !== nothing
        for (k, v) in pairs(fixed)
            nm = string(k)
            j = findfirst(==(nm), L.names)
            j === nothing && (j = findfirst(==(nm), replace.(L.names, "." => "_")))
            j === nothing && throw(ArgumentError(
                "fit_uc: fixed names $(repr(nm)), not a parameter of this model; " *
                "parameters are $(join(L.names, ", "))"))
            push!(fixed_idx, j); push!(fixed_val, Float64(v))
        end
    end

    lltarget(p) = _uc_loglik(s, L, p, yv, X, init_kappa, burn, known)
    if params !== nothing
        length(params) == np || throw(ArgumentError(
            "fit_uc: params has $(length(params)) entries, the model has $np ($(join(L.names, ", ")))"))
        pnat = Vector{Float64}(params)
        converged, iters, estimated = true, 0, false
    else
        pnat, converged, iters = _uc_optimise(s, L, yv, X, vy, lltarget, fixed_idx, fixed_val)
        estimated = true
    end
    ll = lltarget(pnat)
    isfinite(ll) || throw(ArgumentError(
        "fit_uc: the filter broke down at these parameters (a non-positive prediction variance)"))

    T, Z, R, Q, H, beta = _uc_system(s, L, pnat)
    ya = X === nothing ? yv : yv .- X * beta
    a0, P0 = known === nothing ? _resolve_init(T, R * Q * R', _uc_init(L, init_kappa))[1:2] :
             known
    ssm = TimeVaryingSSM{Float64}([T], [Z], [R], [Q], [H], L.r)
    _, v, F, _ = kalman_filter(ssm, ya, a0, P0)
    alpha, V, _, _, _, _, ok = kalman_smoother(ssm, ya, a0, P0)
    ok || throw(ArgumentError("fit_uc: the smoother broke down at the fitted parameters"))
    fitted = [isnan(yv[t]) ? NaN : yv[t] - v[t] for t in 1:n]
    return UCModel(yv, X, s, L, pnat, ll, nobs, burn, Float64(init_kappa), known,
                   T, Z, R, Q, H, fitted, v, alpha, V, converged, iters, estimated)
end

_logistic(x) = 1 / (1 + exp(-x))
_logit(p) = log(p / (1 - p))

# Unconstrained u <-> natural p. Variances: p = vy * exp(u) -- the reference's
# scaling by var(y), imitated for conditioning, without its clipping.
# frequency: (0, π) logistic; damping: (0, 1) logistic; AR: partrans.
function _uc_transform(s::_UCSpec, L::_UCLayout, u::AbstractVector, vy::Real)
    S = eltype(u)
    p = Vector{S}(undef, length(u))
    for (j, nm) in enumerate(L.names)
        if startswith(nm, "sigma2.")
            p[j] = vy * exp(u[j])
        elseif nm == "frequency.cycle"
            p[j] = pi * _logistic(u[j])
        elseif nm == "damping.cycle"
            p[j] = _logistic(u[j])
        elseif startswith(nm, "beta.")
            p[j] = u[j]
        end
    end
    if s.ar > 0
        j0 = findfirst(==("ar.L1"), L.names)
        p[j0:(j0 + s.ar - 1)] = partrans(u[j0:(j0 + s.ar - 1)])
    end
    return p
end

function _uc_untransform(s::_UCSpec, L::_UCLayout, p::AbstractVector, vy::Real)
    u = similar(p, Float64)
    for (j, nm) in enumerate(L.names)
        if startswith(nm, "sigma2.")
            u[j] = log(max(p[j], 1e-300) / vy)
        elseif nm == "frequency.cycle"
            u[j] = _logit(p[j] / pi)
        elseif nm == "damping.cycle"
            u[j] = _logit(p[j])
        elseif startswith(nm, "beta.")
            u[j] = p[j]
        end
    end
    if s.ar > 0
        j0 = findfirst(==("ar.L1"), L.names)
        u[j0:(j0 + s.ar - 1)] = invpartrans(p[j0:(j0 + s.ar - 1)])
    end
    return u
end

function _uc_start(s::_UCSpec, L::_UCLayout, y, X, vy; lam=nothing)
    p = zeros(length(L.names))
    nstoch = count(startswith("sigma2."), L.names) - s.irregular
    for (j, nm) in enumerate(L.names)
        if nm == "sigma2.irregular"
            p[j] = nstoch == 0 ? vy : vy / 2
        elseif startswith(nm, "sigma2.")
            p[j] = s.irregular ? vy / (2 * nstoch) : vy / nstoch
        elseif nm == "frequency.cycle"
            p[j] = lam === nothing ? 2pi / 8 : lam
        elseif nm == "damping.cycle"
            p[j] = 0.7
        end
    end
    if s.nexog > 0
        ok = .!isnan.(y)
        b = X[ok, :] \ y[ok]
        p[(end - s.nexog + 1):end] = b
    end
    return p
end

function _uc_optimise(s, L, y, X, vy, lltarget, fixed_idx, fixed_val)
    np = length(L.names)
    free = setdiff(1:np, fixed_idx)
    # several starting frequencies for a cycle (its likelihood is multimodal
    # in the frequency); one start otherwise
    lams = s.cycle === :none ? [nothing] : [2pi / 4, 2pi / 8, 2pi / 16, 2pi / 32]
    best = nothing
    for lam in lams
        p0 = _uc_start(s, L, y, X, vy; lam=lam)
        p0[fixed_idx] = fixed_val
        u0 = _uc_untransform(s, L, p0, vy)
        function obj(uf)
            u = Vector{eltype(uf)}(u0)
            u[free] = uf
            ll = lltarget(_uc_transform(s, L, u, vy))
            isfinite(ll) ? -ll : eltype(uf)(1e15)
        end
        isempty(free) && return (p0, true, 0)
        res = try
            _optimize(obj, u0[free]; method=:lbfgs, autodiff=true, g_tol=1e-8, iterations=2000)
        catch e
            e isa ArgumentError || e isa DomainError || rethrow()
            nothing
        end
        res === nothing && continue
        if best === nothing || res.minimum < best[2].minimum
            u = copy(u0); u[free] = res.minimizer
            best = (u, res)
        end
    end
    best === nothing && throw(ArgumentError("fit_uc: the optimiser failed from every start"))
    u, res = best
    p = _uc_transform(s, L, u, vy)
    p[fixed_idx] = fixed_val
    return p, res.converged, res.iterations
end

# --- convenience constructors ------------------------------------------------

"""
    uc_local_level(y; kw...) -> UCModel

`fit_uc(y; level = :stochastic, irregular = true)` -- the local level model:
a random-walk level observed with noise.
"""
uc_local_level(y; kw...) = fit_uc(y; level=:stochastic, irregular=true, kw...)

"""
    uc_local_linear_trend(y; kw...) -> UCModel

`fit_uc(y; level = :stochastic, trend = :stochastic, irregular = true)`.
"""
uc_local_linear_trend(y; kw...) = fit_uc(y; level=:stochastic, trend=:stochastic,
                                         irregular=true, kw...)

"""
    uc_random_walk(y; kw...) -> UCModel

`fit_uc(y; level = :stochastic, irregular = false)` -- a random walk with
**no** irregular; the series is its own level.
"""
uc_random_walk(y; kw...) = fit_uc(y; level=:stochastic, irregular=false, kw...)

"""
    uc_smooth_trend(y; kw...) -> UCModel

`fit_uc(y; level = :deterministic, trend = :stochastic, irregular = true)` --
the level moves only through a slowly varying slope, giving a smooth trend.
"""
uc_smooth_trend(y; kw...) = fit_uc(y; level=:deterministic, trend=:stochastic,
                                   irregular=true, kw...)

"""
    uc_bsm(y, period; kw...) -> UCModel

The basic structural model: `fit_uc(y; level = :stochastic, trend =
:stochastic, seasonal = period, seasonal_kind = :dummy, irregular = true)` --
R's `StructTS(type = "BSM")`.
"""
uc_bsm(y, period::Integer; kw...) = fit_uc(y; level=:stochastic, trend=:stochastic,
                                            seasonal=period, seasonal_kind=:dummy,
                                            irregular=true, kw...)

# --- StatsAPI ------------------------------------------------------------------

StatsAPI.coef(m::UCModel) = copy(m.params)
StatsAPI.coefnames(m::UCModel) = copy(m.layout.names)
StatsAPI.loglikelihood(m::UCModel) = m.loglik
StatsAPI.nobs(m::UCModel) = m.nobs
StatsAPI.dof(m::UCModel) = length(m.params)
StatsAPI.aic(m::UCModel) = -2m.loglik + 2length(m.params)
StatsAPI.bic(m::UCModel) = -2m.loglik + log(m.nobs) * length(m.params)
StatsAPI.residuals(m::UCModel) = copy(m.resid)
StatsAPI.fitted(m::UCModel) = copy(m.fitted)

"""
    vcov(m::UCModel)

Inverse of the observed information (the Hessian of the negative
log-likelihood in the natural parameters). Throws when the Hessian is not
positive definite -- typically a variance sitting on its zero boundary,
where the usual asymptotics do not apply -- rather than returning `NaN`s.
"""
function StatsAPI.vcov(m::UCModel)
    f(p) = -_uc_loglik(m.spec, m.layout, p, m.y, m.exog, m.kappa, m.burn, m.known_init)
    Hs = ForwardDiff.hessian(f, m.params)
    C = cholesky(Symmetric(Hs); check=false)
    issuccess(C) || throw(ArgumentError(
        "vcov(::UCModel): the information matrix is not positive definite at the " *
        "estimate -- usually a variance on its zero boundary, where the standard " *
        "errors do not exist. coef and loglikelihood are still valid."))
    return inv(C)
end
StatsAPI.stderror(m::UCModel) = sqrt.(diag(StatsAPI.vcov(m)))

function _uc_describe(s::_UCSpec)
    parts = String[]
    s.level === :none || push!(parts, "$(s.level) level")
    s.trend === :none || push!(parts, "$(s.trend) trend")
    if s.period > 0
        push!(parts, (s.seasonal_stochastic ? "stochastic " : "deterministic ") *
                     (s.seasonal_kind === :dummy ? "seasonal($(s.period))" :
                      "trig seasonal($(s.period), $(s.harmonics) harmonics)"))
    end
    s.cycle === :none || push!(parts, (s.damped_cycle ? "damped " : "") * "$(s.cycle) cycle")
    s.ar > 0 && push!(parts, "AR($(s.ar))")
    s.nexog > 0 && push!(parts, "$(s.nexog) regressor$(s.nexog > 1 ? "s" : "")")
    s.irregular && push!(parts, "irregular")
    return join(parts, " + ")
end

function Base.show(io::IO, m::UCModel)
    println(io, "Unobserved components: ", _uc_describe(m.spec))
    println(io, "  n = $(length(m.y)), $(m.burn) left out for the diffuse initialisation, ",
            "nobs = $(m.nobs)")
    for (nm, v) in zip(m.layout.names, m.params)
        println(io, "  ", rpad(nm, 26), lpad(string(round(v; sigdigits=8)), 16))
    end
    println(io, "  log-likelihood = ", round(m.loglik; digits=4),
            "   AIC = ", round(StatsAPI.aic(m); digits=4), "   BIC = ", round(StatsAPI.bic(m); digits=4))
    print(io, m.estimated ? (m.converged ? "  optimiser converged" :
                             "  optimiser did NOT converge -- the likelihood is flat here; " *
                             "treat the estimates with care") :
                           "  evaluated at given parameters (not estimated)")
end

# --- decomposition -------------------------------------------------------------

"""
    UCDecomposition

The smoothed components of a fitted [`UCModel`](@ref), each with its
standard error (`nothing` for a component the model does not have):
`level`, `trend`, `seasonal`, `cycle`, `ar` (with `*_se`), `regression`
(fixed coefficients, so no standard error), `irregular` (`y` minus every
other component), `names` (the components present, in order) and `period`.
"""
struct UCDecomposition
    level::Union{Nothing,Vector{Float64}}
    level_se::Union{Nothing,Vector{Float64}}
    trend::Union{Nothing,Vector{Float64}}
    trend_se::Union{Nothing,Vector{Float64}}
    seasonal::Union{Nothing,Vector{Float64}}
    seasonal_se::Union{Nothing,Vector{Float64}}
    cycle::Union{Nothing,Vector{Float64}}
    cycle_se::Union{Nothing,Vector{Float64}}
    ar::Union{Nothing,Vector{Float64}}
    ar_se::Union{Nothing,Vector{Float64}}
    regression::Union{Nothing,Vector{Float64}}
    irregular::Vector{Float64}
    names::Vector{Symbol}
    period::Union{Nothing,Int}
end

"""
    uc_decomposition(m::UCModel) -> UCDecomposition

Each component's smoothed path and its standard error, read from the
smoother's state mean and covariance: what a model-based decomposition gives
that [`stl_decompose`](@ref) cannot. The components sum, with the
irregular, to the series exactly.

The standard errors are `NaN` over the first periods that the likelihood
leaves out (one per diffuse state): there the approximate-diffuse prior's
`1e6` variance makes the smoothed variance a difference of huge numbers,
and its value is noise -- negative, in the reference's own output.
"""
function uc_decomposition(m::UCModel)
    L, s = m.layout, m.spec
    n = length(m.y)
    comp(idx) = isempty(idx) ? (nothing, nothing) : _uc_component(m, idx)
    lv, lvse = comp(L.level)
    tr, trse = comp(L.trend)
    se, sese = comp(L.seasonal_load)
    cy, cyse = comp(isempty(L.cycle) ? Int[] : [first(L.cycle)])
    ar, arse = comp(isempty(L.ar) ? Int[] : [first(L.ar)])
    reg = s.nexog == 0 ? nothing : m.exog * m.params[(end - s.nexog + 1):end]
    # the trend is the level's slope, not an additive component: it does not
    # enter the sum
    total = zeros(n)
    for c in (lv, se, cy, ar, reg)
        c === nothing || (total .+= c)
    end
    irr = m.y .- total
    names = Symbol[]
    lv === nothing || push!(names, :level)
    tr === nothing || push!(names, :trend)
    se === nothing || push!(names, :seasonal)
    cy === nothing || push!(names, :cycle)
    ar === nothing || push!(names, :ar)
    reg === nothing || push!(names, :regression)
    s.irregular && push!(names, :irregular)
    return UCDecomposition(lv, lvse, tr, trse, se, sese, cy, cyse, ar, arse, reg, irr, names,
                           s.period == 0 ? nothing : s.period)
end

function _uc_component(m::UCModel, idx)
    w = zeros(m.layout.r)
    w[collect(idx)] .= 1.0
    n = size(m.alpha, 2)
    path = [dot(w, view(m.alpha, :, t)) for t in 1:n]
    # Over the first `burn` periods the smoothed variance is P - P N P with
    # P ~ kappa = 1e6: catastrophic cancellation, and the reference's own
    # values there are negative (as low as -0.97 on UKgas). Reported as NaN
    # -- the same periods the likelihood leaves out -- rather than clipped to
    # a zero that would claim certainty where the prior dominates.
    se = [t <= m.burn ? NaN : sqrt(max(dot(w, m.V[t] * w), 0.0)) for t in 1:n]
    return path, se
end

# --- forecasting ---------------------------------------------------------------

"""
    forecast(m::UCModel, h; level=[80.0, 95.0]) -> Forecast
    forecast(m::UCModel, newexog, h; level=[80.0, 95.0]) -> Forecast

Propagate the last smoothed state through the transition, accumulating
`R Q R'` and adding the irregular variance. A diffuse component's forecast
variance grows without bound, which is correct and not suppressed; a
deterministic component adds none. A model with regressors needs their
future values, one row per step. `predict` is the same function.
"""
function forecast(m::UCModel, horizon::Integer; level=[80.0, 95.0])
    m.spec.nexog == 0 || throw(ArgumentError(
        "forecast(::UCModel): this model has $(m.spec.nexog) regressor(s); pass their future " *
        "values as forecast(m, newexog, horizon)"))
    return _uc_forecast(m, nothing, horizon, level)
end

function forecast(m::UCModel, newexog, horizon::Integer; level=[80.0, 95.0])
    m.spec.nexog > 0 || throw(ArgumentError(
        "forecast(::UCModel): the model has no regressors; use forecast(m, horizon)"))
    Xf = newexog isa AbstractVector ? reshape(Float64.(newexog), :, 1) : Matrix{Float64}(newexog)
    size(Xf) == (horizon, m.spec.nexog) || throw(ArgumentError(
        "forecast(::UCModel): newexog must be $horizon × $(m.spec.nexog), got $(size(Xf))"))
    return _uc_forecast(m, Xf, horizon, level)
end

StatsAPI.predict(m::UCModel, horizon::Integer; level=[80.0, 95.0]) =
    forecast(m, horizon; level=level)
StatsAPI.predict(m::UCModel, newexog, horizon::Integer; level=[80.0, 95.0]) =
    forecast(m, newexog, horizon; level=level)

function _uc_forecast(m::UCModel, Xf, horizon::Integer, level)
    horizon >= 1 || throw(ArgumentError("forecast(::UCModel): horizon must be >= 1"))
    T, R, Q = m.T, m.R, m.Q
    z = vec(m.Z)
    RQR = R * Q * R'
    n = size(m.alpha, 2)
    # a_{n+1|n} = T a_{n|n}; the smoothed state at n IS the filtered one
    a = T * m.alpha[:, n]
    P = T * m.V[n] * T' + RQR
    beta = m.spec.nexog == 0 ? Float64[] : m.params[(end - m.spec.nexog + 1):end]
    point = zeros(horizon); se = zeros(horizon)
    for h in 1:horizon
        point[h] = dot(z, a) + (Xf === nothing ? 0.0 : dot(view(Xf, h, :), beta))
        se[h] = sqrt(max(dot(z, P * z) + m.H[1, 1], 0.0))
        a = T * a
        P = T * P * T' + RQR
    end
    zq = [_confidence_z(1 - l / 100) for l in level]
    lower = reduce(hcat, [point .- zi .* se for zi in zq])
    upper = reduce(hcat, [point .+ zi .* se for zi in zq])
    return Forecast(point, se, Float64.(level), lower, upper, horizon,
                    "UC(" * _uc_describe(m.spec) * ")")
end
