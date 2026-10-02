export fit_ets, auto_ets, ETSModel, notation

"""
    ETSModel

Result of [`fit_ets`](@ref) / [`auto_ets`](@ref). Fields:

- `trend`: `:none`, `:add` or `:damped`.
- `seasonal`: `:none` or `:add`.
- `alpha`, `beta`, `gamma`, `phi`: smoothing parameters. `beta`/`gamma`
  are `nothing` when the component is absent, `phi` `nothing` unless the
  trend is damped.
- `l0`, `b0`, `s0`: fitted initial level, trend and seasonal states.
  `s0` is **oldest-first**, so `s0[end]` is the season immediately
  before the first observation.
- `level`, `trend_component`, `seasonal_component`: the fitted state
  paths.
- `fitted`, `resid`, `sse`: one-step-ahead fits and errors.
- `sigma2`: `sse/nobs`, the concentrated innovation variance.
- `loglik`, `aic`, `aicc`, `bic`: the **full Gaussian** log-likelihood
  and its criteria — see [`fit_ets`](@ref) on why this differs from R's
  `ets` by a constant.
- `nparams`: free parameters, counting the initial states and `sigma2`.
- `nobs`, `period`, `converged`.

`notation` gives the conventional label, e.g. `"ETS(A,Ad,A)"`.
"""
struct ETSModel
    trend::Symbol
    seasonal::Symbol
    alpha::Float64
    beta::Union{Nothing,Float64}
    gamma::Union{Nothing,Float64}
    phi::Union{Nothing,Float64}
    l0::Float64
    b0::Union{Nothing,Float64}
    s0::Vector{Float64}
    level::Vector{Float64}
    trend_component::Vector{Float64}
    seasonal_component::Vector{Float64}
    fitted::Vector{Float64}
    resid::Vector{Float64}
    sse::Float64
    sigma2::Float64
    loglik::Float64
    aic::Float64
    aicc::Float64
    bic::Float64
    nparams::Int
    nobs::Int
    period::Int
    converged::Bool
end

"""
    notation(m::ETSModel) -> String

The conventional ETS label, `"ETS(E,T,S)"` — always `A` for the error,
since only the additive-error family is implemented.
"""
notation(m::ETSModel) = string("ETS(A,",
    m.trend === :none ? "N" : m.trend === :add ? "A" : "Ad", ",",
    m.seasonal === :none ? "N" : "A", ")")

function Base.show(io::IO, m::ETSModel)
    println(io, notation(m), "  n=", m.nobs,
            m.seasonal === :none ? "" : "  period=$(m.period)",
            m.converged ? "" : "  (NOT CONVERGED)")
    print(io, "  alpha = ", round(m.alpha, digits=6))
    m.beta  !== nothing && print(io, "   beta = ", round(m.beta, digits=6))
    m.gamma !== nothing && print(io, "   gamma = ", round(m.gamma, digits=6))
    m.phi   !== nothing && print(io, "   phi = ", round(m.phi, digits=6))
    println(io)
    print(io, "  SSE = ", round(m.sse, digits=4),
          "   sigma2 = ", round(m.sigma2, digits=6))
    println(io)
    print(io, "  loglik = ", round(m.loglik, digits=4),
          "   AIC = ", round(m.aic, digits=3),
          "   AICc = ", round(m.aicc, digits=3),
          "   BIC = ", round(m.bic, digits=3))
    m.converged || print(io, "\nWARNING: the optimizer did not converge")
end

# ---------------------------------------------------------------------------
# the recursion
# ---------------------------------------------------------------------------

"""
    _ets_recursion!(fitted, resid, lvl, trd, ssn, y, alpha, beta, gamma, phi,
                    l0, b0, s0, trend, seasonal) -> sse

The additive-error ETS recursion (Hyndman's innovations state space
form), writing its outputs into the caller's buffers and returning the
sum of squared errors.

    yhat_t = l_{t-1} + phi*b_{t-1} + s_{t-m}
    e_t    = y_t - yhat_t
    l_t    = l_{t-1} + phi*b_{t-1} + alpha*e_t
    b_t    = phi*b_{t-1} + beta*e_t
    s_t    = s_{t-m} + gamma*e_t

with `phi = 1` unless the trend is damped, and each component dropped
when absent. `beta` is in the `beta = alpha*beta_star` parameterisation
R's `ets` and `statsmodels` both report in.

`s0` is **oldest-first**: `s0[1]` is the season `m` steps before the
first observation and `s0[end]` the one immediately before it. That is
the opposite of R's `ets`, which lists `s0` as the *most recent* — a
convention difference confirmed by testing both orders against R's SSE,
where only this one reproduces it.

Allocation-free by construction: the seasonal state is a plain vector
with a rotating index rather than a `circshift`, which would allocate on
every one of `n` steps.
"""
function _ets_recursion!(fitted::AbstractVector, resid::AbstractVector,
                          lvl::AbstractVector, trd::AbstractVector, ssn::AbstractVector,
                          y::AbstractVector, alpha, beta, gamma, phi,
                          l0, b0, s0::AbstractVector, trend::Symbol, seasonal::Symbol)
    n = length(y)
    m = length(s0)
    has_t = trend !== :none
    has_s = seasonal !== :none
    ph = trend === :damped ? phi : one(alpha)

    l = l0
    b = has_t ? b0 : zero(alpha)
    # rotating seasonal buffer; si points at the season to be used next
    sbuf = has_s ? collect(s0) : similar(s0, 0)
    si = 1
    sse = zero(alpha) * zero(eltype(y))

    @inbounds for t in 1:n
        tr = has_t ? ph * b : zero(l)
        se = has_s ? sbuf[si] : zero(l)
        yh = l + tr + se
        e = y[t] - yh
        fitted[t] = yh
        resid[t] = e
        sse += e^2

        lnew = l + tr + alpha * e
        if has_t
            b = ph * b + beta * e
        end
        if has_s
            sbuf[si] = se + gamma * e
            si = si == m ? 1 : si + 1
        end
        l = lnew
        lvl[t] = l
        has_t && (trd[t] = b)
        has_s && (ssn[t] = se)
    end
    return sse
end

# ---------------------------------------------------------------------------
# parameter packing
# ---------------------------------------------------------------------------

"_ets_nfree(trend, seasonal, m) -> (n_smooth, n_state) -- free smoothing
parameters and free initial states. The seasonal states contribute `m-1`,
not `m`: they are constrained to sum to zero, so the last is determined,
which is why R's `ets` reports `s0..s_{m-2}` and this package's
`nparams` matches R's `length(par)` exactly."
function _ets_nfree(trend::Symbol, seasonal::Symbol, m::Int)
    ns = 1                                   # alpha
    trend !== :none && (ns += 1)             # beta
    trend === :damped && (ns += 1)           # phi
    seasonal !== :none && (ns += 1)          # gamma
    nst = 1                                  # l0
    trend !== :none && (nst += 1)            # b0
    seasonal !== :none && (nst += m - 1)     # m-1 free seasonals
    return ns, nst
end

"_ets_unpack(p, trend, seasonal, m) -> (alpha, beta, gamma, phi, l0, b0, s0)
-- read a parameter vector laid out as
`[alpha, beta?, gamma?, phi?, l0, b0?, s_free...]`. The final seasonal
state is `-sum(s_free)`, enforcing the sum-to-zero normalisation rather
than estimating a redundant parameter."
function _ets_unpack(p::AbstractVector, trend::Symbol, seasonal::Symbol, m::Int)
    T = eltype(p)
    i = 1
    alpha = p[i]; i += 1
    beta = trend !== :none ? (v = p[i]; i += 1; v) : zero(T)
    gamma = seasonal !== :none ? (v = p[i]; i += 1; v) : zero(T)
    phi = trend === :damped ? (v = p[i]; i += 1; v) : one(T)
    l0 = p[i]; i += 1
    b0 = trend !== :none ? (v = p[i]; i += 1; v) : zero(T)
    s0 = if seasonal !== :none
        free = p[i:(i + m - 2)]
        vcat(free, -sum(free))
    else
        T[]
    end
    return alpha, beta, gamma, phi, l0, b0, s0
end

"""
_ets_in_region(alpha, beta, gamma, phi, trend, seasonal, constraint) -- whether
the smoothing parameters lie in the admissible parameter region.

`:traditional` is the textbook box, and the one R's `ets` defaults to:
`0 < alpha < 1`, `0 < beta < alpha`, `0 < gamma < 1 - alpha`,
`0.8 < phi < 0.98`. The `beta < alpha` and `gamma < 1 - alpha` couplings
are what make it a region rather than a box in the individual
parameters.

`:admissible` is the strictly wider set on which the underlying linear
system is stable (Hyndman et al. 2008, section 10.1) -- it allows
parameters the traditional region forbids while still giving a
forecastable model. Implemented here as the traditional bounds relaxed
to `0 < alpha < 2`, `0 < beta < 4 - 2*alpha` and `gamma` positive with
`alpha + gamma < 2`, plus the same `phi` range.
"""
function _ets_in_region(alpha, beta, gamma, phi, trend::Symbol, seasonal::Symbol,
                         constraint::Symbol)
    has_t = trend !== :none
    has_s = seasonal !== :none
    if constraint === :traditional
        0 < alpha < 1 || return false
        has_t && !(0 < beta < alpha) && return false
        has_s && !(0 < gamma < 1 - alpha) && return false
    else
        0 < alpha < 2 || return false
        has_t && !(0 < beta < 4 - 2 * alpha) && return false
        has_s && !(gamma > 0 && alpha + gamma < 2) && return false
    end
    trend === :damped && !(0.8 <= phi <= 0.98) && return false
    return true
end

# ---------------------------------------------------------------------------
# initial values
# ---------------------------------------------------------------------------

"""
_ets_heuristic_init(y, m, trend, seasonal) -> (l0, b0, s_full)

Initial states, delegating to [`holt_winters`](@ref)'s own
`_hw_heuristic_init` rather than reimplementing it.

That delegation is the point, not a convenience. The reduction test that
`fit_ets` at pinned smoothing parameters reproduces `holt_winters`
exactly is only meaningful if the two share an initialisation -- and
since `holt_winters` is independently verified against R's `HoltWinters`
to 1e-13, the reduction then inherits that verification instead of
comparing two of this package's own guesses. An earlier version computed
its own decomposition here and the reduction failed by 14%.

Returns the **full** `m`-vector of seasonal figures, in calendar order
and *not* renormalised to sum to zero -- `classical_decompose`'s
`figure` as `holt_winters` consumes it. The sum-to-zero
reparameterisation applies only when the initial states are being
optimised, where it removes a redundant parameter.
"""
function _ets_heuristic_init(y::Vector{Float64}, m::Int, trend::Symbol, seasonal::Symbol)
    st = seasonal === :none ? nothing : :additive
    l0, b0, fig = _hw_heuristic_init(y, m, trend !== :none, st)
    return l0, b0, fig
end

# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

"""
    fit_ets(y, period=1; trend=:none, seasonal=:none, constraint=:traditional,
            fixed=nothing, initial=nothing, optimizer_method=:nelder_mead) -> ETSModel

Fit an **additive-error** ETS model by maximum likelihood.

| `trend` | `seasonal` | Model | Also known as |
|---|---|---|---|
| `:none` | `:none` | ETS(A,N,N) | Simple exponential smoothing |
| `:add` | `:none` | ETS(A,A,N) | Holt's linear |
| `:damped` | `:none` | ETS(A,Ad,N) | Damped Holt |
| `:none` | `:add` | ETS(A,N,A) | Seasonal, no trend |
| `:add` | `:add` | ETS(A,A,A) | Additive Holt-Winters |
| `:damped` | `:add` | ETS(A,Ad,A) | Damped additive Holt-Winters |

Those six are the whole **linear** ETS family — the subset expressible
as a linear Gaussian state space model. Multiplicative error and
multiplicative seasonality are the non-linear half, need
simulation-based intervals, and are rejected by name rather than
silently approximated.

## What this adds over `holt_winters`

[`holt_winters`](@ref) implements the classical recursions and reports
`sse`. This adds a **likelihood**, and so `aic`/`aicc`/`bic` and
automatic model selection ([`auto_ets`](@ref)), plus the damped trend
parameter `phi`. The initial states are optimised jointly with the
smoothing parameters, as R's `ets` and `statsmodels` both do.

The two **do** reduce onto each other, but not by passing the same
numbers to both — and this is worth stating because it is easy to get
wrong:

1. **The parameterisations differ.** `holt_winters` uses the classical
   form, `fit_ets` the innovations form. The map is
   `beta = alpha*beta_star` and `gamma = gamma_star*(1 - alpha)`,
   derivable by substituting `e_t = y_t - (l+b+s)` into the classical
   updates.
2. **The windows differ.** For a seasonal model `holt_winters` starts
   its recursion at `t = m+1`, scoring `n - m` observations; `fit_ets`
   scores all `n`.

Matching both, the two agree to **1e-13** in SSE and `4e-14` on every
fitted value — and since `holt_winters` is independently verified
against R's `HoltWinters` to 1e-13, that makes the reduction a real
check rather than two of this package's own guesses agreeing.

## The likelihood convention, which differs from R by a constant

`loglik` is the **full Gaussian** log-likelihood,

    -n/2 * (log(2*pi) + log(sse/n) + 1)

matching `statsmodels` and, more importantly, matching what every other
model in this package reports — so `aic` here is comparable with
[`fit_arima`](@ref)'s.

**R's `ets` reports `-(n/2)*log(sse)` instead.** This one exceeds R's
by exactly

    n/2 * (log(n) - log(2*pi) - 1)

which on a 120-point series is `116.976881` (and on an 84-point one,
`66.903469`). Verified: on the two undamped non-seasonal models, where
both optimisers reach the same optimum, `statsmodels`' log-likelihood
minus R's equals that constant to four decimal places. So R's `loglik`, `aic` and `aicc` are **not**
comparable with this package's without adding it; the *rankings* are
unaffected, because it is a constant.

!!! note "`statsmodels`' damped fits are at its `phi` bound"
    Verified on the bundled fixture: `statsmodels` returns `phi = 0.98`,
    its upper bound, for both damped models, with SSE `5044.65` and
    `420.31`. R finds interior optima (`phi = 0.918` and `0.949`) with
    **better** SSE, `5002.51` and `412.36`.

    So for the damped models **R is the better reference**, and this
    package targets R there. The handoff for this stage flagged the
    `0.98` as "worth investigating rather than an error"; this is the
    resolution.

    **`0.98` is not a `statsmodels` quirk, though — it is R's bound too**,
    and this package adopts the same `0.8 <= phi <= 0.98` range
    deliberately. A `phi` of `0.98` is therefore not by itself a sign of
    anything wrong: on `log(dataset("jj").value)` R's own damped fit
    returns `phi = 0.97995483`, and this one returns `0.98` with a better
    SSE. What distinguished the fixture case was that R found an
    *interior* optimum there and `statsmodels` did not.

`fixed` pins smoothing parameters, as a `NamedTuple` — e.g.
`fixed=(alpha=0.4, beta=0.1, gamma=0.3)`. Pinned parameters are removed
from the optimisation rather than merely started from.

`constraint` selects the admissible parameter region: `:traditional`
(default, R's own) or `:admissible`, which is strictly wider and so can
only fit at least as well.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                                 "verification", "ets", "ets_y.csv")));

julia> m = fit_ets(y, 4; trend=:add, seasonal=:add);

julia> notation(m)
"ETS(A,A,A)"

julia> m.nparams
9
```

See also [`auto_ets`](@ref), [`holt_winters`](@ref), [`forecast`](@ref).
"""
function fit_ets(y, period::Integer=1; trend::Symbol=:none, seasonal::Symbol=:none,
                  constraint::Symbol=:traditional,
                  fixed::Union{Nothing,NamedTuple}=nothing,
                  initial::Union{Nothing,Symbol}=nothing,
                  optimizer_method::Symbol=:nelder_mead)
    trend in (:none, :add, :damped) || throw(ArgumentError(
        trend in (:mul, :multiplicative, :mult) ?
            "fit_ets: trend=:$trend is not implemented -- multiplicative trend is part of " *
            "the non-linear ETS family, which needs simulation-based intervals. " *
            "Use :none, :add or :damped." :
            "fit_ets: trend must be :none, :add or :damped, got :$trend"))
    seasonal in (:none, :add) || throw(ArgumentError(
        seasonal in (:mul, :multiplicative, :mult) ?
            "fit_ets: seasonal=:$seasonal is not implemented -- multiplicative seasonality " *
            "is part of the non-linear ETS family. Use :none or :add." :
            "fit_ets: seasonal must be :none or :add, got :$seasonal"))
    constraint in (:traditional, :admissible) ||
        throw(ArgumentError("fit_ets: constraint must be :traditional or :admissible"))
    initial === nothing || initial in (:heuristic, :estimated) ||
        throw(ArgumentError("fit_ets: initial must be :heuristic or :estimated"))

    yv = Float64.(collect(tsvalues(y)))
    n = length(yv)
    n >= 3 || throw(ArgumentError("fit_ets: need at least 3 observations, got $n"))
    any(isnan, yv) && throw(ArgumentError("fit_ets: NaN present (no missing-data policy yet)"))

    m = Int(period)
    if seasonal !== :none
        m >= 2 || throw(ArgumentError(
            "fit_ets: seasonal=:add needs period >= 2, got $m"))
        n >= 2 * m || throw(ArgumentError(
            "fit_ets: seasonal=:add needs at least 2 full periods ($(2m) observations), got $n"))
    else
        m = max(m, 1)
    end
    mseas = seasonal === :none ? 0 : m

    fx = fixed === nothing ? NamedTuple() : fixed
    for key in keys(fx)
        key in (:alpha, :beta, :gamma, :phi) ||
            throw(ArgumentError("fit_ets: `fixed` may only name alpha, beta, gamma or phi, got $key"))
    end
    haskey(fx, :beta) && trend === :none &&
        throw(ArgumentError("fit_ets: `fixed` names beta but trend=:none"))
    haskey(fx, :gamma) && seasonal === :none &&
        throw(ArgumentError("fit_ets: `fixed` names gamma but seasonal=:none"))
    haskey(fx, :phi) && trend !== :damped &&
        throw(ArgumentError("fit_ets: `fixed` names phi but trend is not :damped"))

    l0h, b0h, sfh = _ets_heuristic_init(yv, max(m, 2), trend, seasonal)

    # which smoothing parameters are free
    smooth_names = Symbol[:alpha]
    trend !== :none && push!(smooth_names, :beta)
    seasonal !== :none && push!(smooth_names, :gamma)
    trend === :damped && push!(smooth_names, :phi)
    free_names = filter(s -> !haskey(fx, s), smooth_names)

    # Deterministic multi-start over the smoothing parameters. A single start
    # is not enough: on the damped seasonal model Nelder-Mead walks `phi` to
    # its 0.98 bound and stops there -- which is exactly where `statsmodels`
    # lands too, with a worse SSE than the interior optimum R finds. Restarting
    # from a small `gamma` and a lower `phi` recovers it. Deterministic rather
    # than randomized so a fit is reproducible without a seed.
    START_GRID = [Dict(:alpha => 0.3,  :beta => 0.1,  :gamma => 0.1,    :phi => 0.95),
                  Dict(:alpha => 0.3,  :beta => 0.2,  :gamma => 0.0005, :phi => 0.92),
                  Dict(:alpha => 0.1,  :beta => 0.05, :gamma => 0.05,   :phi => 0.98),
                  Dict(:alpha => 0.6,  :beta => 0.1,  :gamma => 0.3,    :phi => 0.88)]
    sfree0 = if seasonal === :none
        Float64[]
    else
        centred = sfh .- sum(sfh) / length(sfh)   # sum-to-zero for the free form
        centred[1:(end-1)]
    end
    state0 = vcat(l0h, trend === :none ? Float64[] : [b0h], sfree0)
    heuristic_only = initial === :heuristic
    starts = [heuristic_only ? Float64[defaults[s] for s in free_names] :
               vcat(Float64[defaults[s] for s in free_names], state0)
               for defaults in START_GRID]

    function assemble(x::AbstractVector)
        T = eltype(x)
        vals = Dict{Symbol,T}()
        i = 1
        for s in free_names
            vals[s] = x[i]; i += 1
        end
        for s in smooth_names
            haskey(vals, s) || (vals[s] = T(fx[s]))
        end
        alpha = vals[:alpha]
        beta = trend !== :none ? vals[:beta] : zero(T)
        gamma = seasonal !== :none ? vals[:gamma] : zero(T)
        phi = trend === :damped ? vals[:phi] : one(T)
        if heuristic_only
            # verbatim, so the reduction onto holt_winters is exact
            l0 = T(l0h); b0 = T(b0h)
            s0 = seasonal === :none ? T[] : T.(sfh)
        else
            l0 = x[i]; i += 1
            b0 = trend !== :none ? (v = x[i]; i += 1; v) : zero(T)
            s0 = if seasonal !== :none
                free = x[i:(i + mseas - 2)]
                vcat(free, -sum(free))
            else
                T[]
            end
        end
        return alpha, beta, gamma, phi, l0, b0, s0
    end

    fitted = Vector{Float64}(undef, n); resid = similar(fitted)
    lvl = similar(fitted); trd = zeros(n); ssn = zeros(n)

    function objective(x::AbstractVector)
        alpha, beta, gamma, phi, l0, b0, s0 = assemble(x)
        _ets_in_region(alpha, beta, gamma, phi, trend, seasonal, constraint) || return 1e10
        f = Vector{Float64}(undef, n); r = Vector{Float64}(undef, n)
        lv = Vector{Float64}(undef, n); td = zeros(n); sn = zeros(n)
        sse = _ets_recursion!(f, r, lv, td, sn, yv, alpha, beta, gamma, phi,
                               l0, b0, seasonal === :none ? Float64[] : collect(s0),
                               trend, seasonal)
        isfinite(sse) ? sse : 1e10
    end

    res = if isempty(starts[1])
        (minimizer=Float64[], converged=true)
    else
        attempts = [try
                        _optimize(objective, x; method=optimizer_method, autodiff=false)
                    catch e
                        e isa ArgumentError ? nothing : rethrow()
                    end for x in starts]
        valid = [r for r in attempts if r !== nothing]
        isempty(valid) && throw(ArgumentError(
            "fit_ets: the optimizer failed from every starting point"))
        # best objective wins outright: a non-converged point with a lower SSE
        # is still the better fit, and preferring a converged worse one would
        # discard it
        best = valid[argmin([objective(r.minimizer) for r in valid])]
        # Polish: restart from the best point, repeatedly. Nelder-Mead stops
        # short of its tolerance on a long flat valley -- which ETS likelihood
        # surfaces have -- so a single pass from a distant start both reports
        # `converged = false` and leaves real improvement on the table.
        # Rebuilding the simplex around the current best recovers it. Bounded
        # at a handful of rounds and stopped as soon as a round buys nothing,
        # so this is cheap in the common case where the first fit was already
        # at the optimum.
        for _ in 1:8
            cur = objective(best.minimizer)
            nxt = try
                _optimize(objective, best.minimizer; method=optimizer_method, autodiff=false)
            catch e
                e isa ArgumentError ? nothing : rethrow()
            end
            nxt === nothing && break
            improvement = cur - objective(nxt.minimizer)
            if objective(nxt.minimizer) <= cur
                best = nxt
            end
            improvement <= 1e-10 * max(abs(cur), 1.0) && break
        end
        best
    end
    alpha, beta, gamma, phi, l0, b0, s0 = assemble(res.minimizer)
    sse = _ets_recursion!(fitted, resid, lvl, trd, ssn, yv, alpha, beta, gamma, phi,
                           l0, b0, seasonal === :none ? Float64[] : collect(s0),
                           trend, seasonal)

    ns, nst = _ets_nfree(trend, seasonal, mseas == 0 ? 1 : mseas)
    npinned = length(keys(fx))
    nparams = (ns - npinned) + (heuristic_only ? 0 : nst) + 1   # +1 for sigma2
    sigma2 = sse / n
    loglik = -n / 2 * (log(2pi) + log(sigma2) + 1)
    aic = -2 * loglik + 2 * nparams
    aicc = n - nparams - 1 > 0 ? aic + 2 * nparams * (nparams + 1) / (n - nparams - 1) : Inf
    bic = -2 * loglik + nparams * log(n)

    return ETSModel(trend, seasonal, alpha,
                     trend !== :none ? beta : nothing,
                     seasonal !== :none ? gamma : nothing,
                     trend === :damped ? phi : nothing,
                     l0, trend !== :none ? b0 : nothing, collect(s0),
                     lvl, trd, ssn, fitted, resid, sse, sigma2,
                     loglik, aic, aicc, bic, nparams, n,
                     seasonal === :none ? Int(period) : m, res.converged)
end

"""
    auto_ets(y, period=1; seasonal=true, constraint=:traditional, ...) -> ETSModel

Fit every applicable **linear** ETS model and return the one with the
lowest **AICc**, which is Hyndman's own recommendation for finite
samples and what R's `ets` selects by.

Six candidates with `seasonal=true` and a `period >= 2`; three when
seasonality is excluded or the series is too short for two full periods.

The candidates are independent fits sharing no state, so they are fitted
in parallel when Julia is started with more than one thread — guarded the
same way as every other threaded path here. **The parallel and serial
paths select the identical model**, not merely an equally good one, which
is asserted in the test suite rather than assumed.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                                 "verification", "ets", "ets_y.csv")));

julia> m = auto_ets(y, 4);

julia> notation(m)
"ETS(A,A,A)"
```

That series was simulated from an ETS(A,A,A) process, so recovering it is
an end-to-end check on the whole selection machinery rather than on any
one component.

See also [`fit_ets`](@ref), [`auto_arima`](@ref).
"""
function auto_ets(y, period::Integer=1; seasonal::Bool=true,
                   constraint::Symbol=:traditional,
                   optimizer_method::Symbol=:nelder_mead, parallel::Bool=true)
    yv = Float64.(collect(tsvalues(y)))
    n = length(yv)
    m = Int(period)
    use_seasonal = seasonal && m >= 2 && n >= 2 * m

    specs = [(:none, :none), (:add, :none), (:damped, :none)]
    use_seasonal && append!(specs, [(:none, :add), (:add, :add), (:damped, :add)])

    fits = Vector{Union{Nothing,ETSModel}}(undef, length(specs))
    tryfit(i) = try
        fit_ets(yv, m; trend=specs[i][1], seasonal=specs[i][2],
                 constraint=constraint, optimizer_method=optimizer_method)
    catch e
        e isa ArgumentError ? nothing : rethrow()
    end
    if parallel && Threads.nthreads() > 1 && length(specs) >= 4
        Threads.@threads for i in eachindex(specs)
            fits[i] = tryfit(i)
        end
    else
        for i in eachindex(specs)
            fits[i] = tryfit(i)
        end
    end

    ok = [f for f in fits if f !== nothing]
    isempty(ok) && throw(ArgumentError(
        "auto_ets: no candidate model could be fitted -- the series may be too short"))
    return ok[argmin([f.aicc for f in ok])]
end

# ---------------------------------------------------------------------------
# forecasting
# ---------------------------------------------------------------------------

"""
_ets_propagate(l, b, s, alpha, beta, gamma, phi, trend, seasonal, horizon)

Run the ETS state forward `horizon` steps with **zero** innovations,
returning the per-step `yhat`.

Used twice, for two different things, because the recursion is linear in
the state:

- fed the fitted final state, it gives the **point forecasts**;
- fed a unit innovation's state perturbation `(alpha, beta, gamma)`, it
  gives the **psi weights** — the effect of a shock at time `n` on
  `yhat_{n+h}` — from which the forecast variance follows as
  `sigma2 * (1 + sum(psi[1:h-1].^2))`.

Deriving the `F`/`w`/`g` matrices by hand for all six models and then
powering `F` would be the textbook route and six chances to make an
indexing error. Propagating the recursion that is already verified
against R to 1e-6 is the same computation with none of them.
"""
function _ets_propagate(l0, b0, s0::AbstractVector, alpha, beta, gamma, phi,
                         trend::Symbol, seasonal::Symbol, horizon::Integer)
    has_t = trend !== :none
    has_s = seasonal !== :none
    ph = trend === :damped ? phi : 1.0
    m = length(s0)
    l = l0
    b = has_t ? b0 : 0.0
    sbuf = has_s ? collect(s0) : Float64[]
    si = 1
    out = Vector{Float64}(undef, horizon)
    for h in 1:horizon
        tr = has_t ? ph * b : 0.0
        se = has_s ? sbuf[si] : 0.0
        out[h] = l + tr + se
        # zero innovation: the states just evolve
        l = l + tr
        has_t && (b = ph * b)
        if has_s
            si = si == m ? 1 : si + 1
        end
    end
    return out
end

"""
    forecast(m::ETSModel, horizon; level=[80.0, 95.0]) -> Forecast
    predict(m::ETSModel, horizon; level=[80.0, 95.0]) -> Forecast

Forecast `horizon` steps ahead from a fitted [`ETSModel`](@ref), with
prediction intervals at each `level` (percentages, matching
[`forecast`](@ref)'s convention throughout this package).

The point forecast is the state propagated forward with zero
innovations, which for the linear family is exact:

    yhat_{n+h} = l_n + (phi + phi^2 + ... + phi^h) * b_n + s_{n+h-m*ceil(h/m)}

so an undamped trend extrapolates linearly in `h` while a damped one
converges to `l_n + phi/(1-phi) * b_n`. The seasonal term repeats with
the period.

Intervals come from the exact linear-model variance
`sigma2 * (1 + sum(psi[1:h-1].^2))`, where `psi_j` is the effect of a
time-`n` innovation on `yhat_{n+j+1}`. **Like every other interval in
this package these treat the fitted parameters as known**, so they are
slightly too narrow; R and `statsmodels` do the same.

!!! note "R's ETS intervals are about 3.5% wider, and only for one reason"
    `sigma2` here is `sse/n`, the maximum-likelihood estimate — the same
    quantity the log-likelihood is built from, matching `statsmodels`
    and [`fit_arima`](@ref). **R's `ets` uses `sse/(n - np)`**, the
    degrees-of-freedom-adjusted estimate, where `np` counts the fitted
    parameters and initial states.

    On the bundled fixture that is `3.48894` against R's `3.73864`, a
    ratio of `sqrt(120/112) = 1.0351`. Substituting R's `sigma2` into
    this package's own interval arithmetic reproduces R's 95% bound to
    **all eight printed decimals** (`174.24518033`), so the point
    forecast, the psi weights and the interval construction are all
    identical — the entire gap is the denominator.

    Neither is wrong. The ML estimate keeps `sigma2`, `loglik` and `aic`
    mutually consistent, which is why it is the one stored; R's is the
    better small-sample choice for an interval specifically. The same
    `n` versus `n-k` choice is already documented for [`arx`](@ref).

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                                 "verification", "ets", "ets_y.csv")));

julia> f = forecast(fit_ets(y, 4; trend=:add, seasonal=:add), 8);

julia> f.horizon
8

julia> f.model_name
"ETS(A,A,A)"

julia> issorted(f.se)
true
```

See also [`fit_ets`](@ref), [`auto_ets`](@ref), [`accuracy`](@ref).
"""
function StatsAPI.predict(model::ETSModel, horizon::Integer;
                           level::Vector{<:Real}=[80.0, 95.0])
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    isempty(level) && throw(ArgumentError("level must be non-empty"))
    all(0 .< level .< 100) || throw(ArgumentError("level entries must be in (0, 100)"))

    beta = model.beta === nothing ? 0.0 : model.beta
    gamma = model.gamma === nothing ? 0.0 : model.gamma
    phi = model.phi === nothing ? 1.0 : model.phi
    b0 = model.b0 === nothing ? 0.0 : model.b0

    # final state: the recursion's own last values, with the seasonal buffer
    # left where the fitted pass ended
    n = model.nobs
    mseas = length(model.s0)
    lN = model.level[end]
    bN = model.trend === :none ? 0.0 : model.trend_component[end]
    sN = if model.seasonal === :none
        Float64[]
    else
        # rebuild the buffer as it stood after observation n: re-run to collect it
        sb = collect(model.s0)
        si = 1
        for t in 1:n
            sb[si] = sb[si] + gamma * model.resid[t]
            si = si == mseas ? 1 : si + 1
        end
        circshift(sb, -(si - 1))        # rotate so index 1 is next to be used
    end

    point = _ets_propagate(lN, bN, sN, model.alpha, beta, gamma, phi,
                            model.trend, model.seasonal, horizon)

    # psi weights: the same propagation from a unit shock's state perturbation
    dps = model.seasonal === :none ? Float64[] : vcat(zeros(mseas - 1), gamma)
    psi = _ets_propagate(model.alpha, beta, dps, model.alpha, beta, gamma, phi,
                          model.trend, model.seasonal, horizon)
    se = [sqrt(model.sigma2 * (1 + sum(abs2, view(psi, 1:(h-1))))) for h in 1:horizon]

    z = [_confidence_z(1 - l / 100) for l in level]
    lower = reduce(hcat, [point .- zi .* se for zi in z])
    upper = reduce(hcat, [point .+ zi .* se for zi in z])
    return Forecast(point, se, Float64.(level), lower, upper, horizon, notation(model))
end

forecast(model::ETSModel, horizon::Integer; level::Vector{<:Real}=[80.0, 95.0]) =
    StatsAPI.predict(model, horizon; level=level)

StatsAPI.coef(m::ETSModel) = vcat(m.alpha,
    m.beta === nothing ? Float64[] : [m.beta],
    m.gamma === nothing ? Float64[] : [m.gamma],
    m.phi === nothing ? Float64[] : [m.phi])
StatsAPI.residuals(m::ETSModel) = m.resid
StatsAPI.fitted(m::ETSModel) = m.fitted
StatsAPI.nobs(m::ETSModel) = m.nobs
StatsAPI.loglikelihood(m::ETSModel) = m.loglik
StatsAPI.aic(m::ETSModel) = m.aic
StatsAPI.bic(m::ETSModel) = m.bic
