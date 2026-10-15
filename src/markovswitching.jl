# ---------------------------------------------------------------------------
# Stage 11.3 -- Markov-switching regression and autoregression.
# handoff/stage 11/docs/stage-11.3-markov-switching-handoff.md
#
# Hamilton (1989): an unobserved first-order Markov chain S_t ∈ {1..k}
# switches the parameters of a regression, or of an autoregression in its
# mean-adjusted form
#     y_t − μ_{S_t,t} = Σ_j φ_j (y_{t−j} − μ_{S_{t−j},t−j}) + ε_t.
# Because the regime at every lag enters, the filter runs over the k^(p+1)
# joint states (S_t, S_{t−1}, …, S_{t−p}). Log-space Hamilton filter, Kim
# smoother, EM as a warm start (Hamilton 1990), then gradient ML from
# several starts. Primary reference statsmodels MarkovRegression /
# MarkovAutoregression; the canonical fixture's likelihood is corroborated by
# E-views. R's MSwM fits a different model (lags as plain regressors).
# ---------------------------------------------------------------------------

export fit_ms_regression, fit_ms_autoregression, MSModel, canonicalise,
       expected_durations, regime_probabilities

struct _MSSpec
    k::Int
    p::Int                    # 0 = regression; >= 1 = mean-adjusted AR
    trend::Bool
    nx::Int                   # exog columns
    sw_trend::Bool
    sw_exog::Bool
    sw_ar::Bool
    sw_var::Bool
    q::Int                    # transition covariates (1 = constant, no TVTP)
end

# --- the joint-state index map, built once ----------------------------------
# State j <-> digits (s_0, …, s_p) of S_t, S_{t-1}, …, S_{t-p} (0-based
# regimes), j = 1 + Σ s_m k^(p-m). A transition i -> j is permitted iff j's
# window is i's shifted by one: (j_1, …, j_p) == (i_0, …, i_{p-1}).
struct _MSStates
    digits::Matrix{Int}            # J × (p+1)
    pred::Vector{Vector{Int}}      # predecessors of j
    succ::Vector{Vector{Int}}      # successors of i
end

function _ms_states(k::Int, p::Int)
    J = k^(p + 1)
    D = zeros(Int, J, p + 1)
    for j in 1:J
        x = j - 1
        for m in (p + 1):-1:1
            D[j, m] = x % k
            x ÷= k
        end
    end
    enc(d) = 1 + sum(d[m] * k^(p + 1 - m) for m in 1:(p + 1))
    pred = Vector{Vector{Int}}(undef, J)
    succ = Vector{Vector{Int}}(undef, J)
    for j in 1:J
        # predecessors: (j_1, …, j_p, x)
        pred[j] = [enc(vcat(D[j, 2:end], x)) for x in 0:(k - 1)]
        # successors: (x, j_0, …, j_{p-1})
        succ[j] = [enc(vcat(x, D[j, 1:p])) for x in 0:(k - 1)]
    end
    return _MSStates(D, pred, succ)
end

_logsumexp(v) = (m = maximum(v); isfinite(m) ? m + log(sum(exp.(v .- m))) : m)

# --- parameters ----------------------------------------------------------------
# natural order: transition (k(k-1) q, grouped by to-regime then from-regime,
# the reference's order), const, exog, sigma2, ar.

function _ms_names(s::_MSSpec)
    nm = String[]
    for c in 1:s.q, i in 0:(s.k - 2), j in 0:(s.k - 1)
        push!(nm, s.q == 1 ? "p[$j->$i]" : "p[$j->$i].tvtp$(c - 1)")
    end
    reg(base, sw) = sw ? ["$base[$r]" for r in 0:(s.k - 1)] : [base]
    s.trend && append!(nm, reg("const", s.sw_trend))
    for c in 1:s.nx
        append!(nm, reg("x$c", s.sw_exog))
    end
    append!(nm, reg("sigma2", s.sw_var))
    for l in 1:s.p
        append!(nm, reg("ar.L$l", s.sw_ar))
    end
    return nm
end

# explicit offsets: a closure mutating a captured index would be boxed by
# Julia and make every field type-unstable (it cost a factor of ~10)
function _ms_unpack(s::_MSSpec, θ::AbstractVector{T}) where {T}
    k = s.k
    ntr = (k - 1) * k * s.q
    tr = θ[1:ntr]
    o = ntr
    nc = s.trend ? (s.sw_trend ? k : 1) : 0
    c = θ[(o + 1):(o + nc)]; o += nc
    nb = s.sw_exog ? k : 1
    B = Vector{T}[θ[(o + (m - 1) * nb + 1):(o + m * nb)] for m in 1:s.nx]; o += s.nx * nb
    nv = s.sw_var ? k : 1
    σ2 = θ[(o + 1):(o + nv)]; o += nv
    na = s.sw_ar ? k : 1
    ϕ = Vector{T}[θ[(o + (l - 1) * na + 1):(o + l * na)] for l in 1:s.p]
    return (; tr, c, B, σ2, ϕ)
end

# left-stochastic P (P[i,j] = Pr(S_t = i | S_{t-1} = j), columns sum to 1)
# from the probability parameters p[j->i], i = 0..k-2
function _ms_P_from_probs(tr::AbstractVector, k::Int)
    T = eltype(tr)
    P = Matrix{T}(undef, k, k)
    for j in 1:k
        acc = zero(T)
        for i in 1:(k - 1)
            P[i, j] = tr[(i - 1) * k + j]
            acc += P[i, j]
        end
        P[k, j] = one(T) - acc
    end
    return P
end

# TVTP: P_t from the logits, column-wise softmax with the last regime as the
# reference (logit 0)
function _ms_P_tvtp(tr::AbstractVector, k::Int, q::Int, z::AbstractVector)
    T = promote_type(eltype(tr), eltype(z))
    P = Matrix{T}(undef, k, k)
    for j in 1:k
        l = zeros(T, k)
        for i in 1:(k - 1)
            for c in 1:q
                l[i] += tr[(c - 1) * (k - 1) * k + (i - 1) * k + j] * z[c]
            end
        end
        m = maximum(l)
        e = exp.(l .- m)
        P[:, j] = e ./ sum(e)
    end
    return P
end

# steady-state distribution of a left-stochastic P (eigenvector for 1),
# by solving (I - P) π = 0 with Σ π = 1
function _ms_steady(P::AbstractMatrix)
    k = size(P, 1)
    A = vcat(Matrix{eltype(P)}(I, k, k) - P, ones(eltype(P), 1, k))
    b = vcat(zeros(eltype(P), k), one(eltype(P)))
    return A \ b
end

# --- the filter ----------------------------------------------------------------

# μ_{r,t} for regime r (1-based) at time t
function _ms_mu(s::_MSSpec, u, X, r::Int, t::Int)
    μ = s.trend ? u.c[s.sw_trend ? r : 1] : zero(eltype(u.σ2))
    for c in 1:s.nx
        μ += X[t, c] * u.B[c][s.sw_exog ? r : 1]
    end
    return μ
end

# log-density of y_t for every joint state
function _ms_condll!(ll, s::_MSSpec, u, st::_MSStates, y, X, t::Int)
    for j in axes(st.digits, 1)
        r0 = st.digits[j, 1] + 1
        e = y[t] - _ms_mu(s, u, X, r0, t)
        for l in 1:s.p
            rl = st.digits[j, l + 1] + 1
            e -= u.ϕ[l][s.sw_ar ? r0 : 1] * (y[t - l] - _ms_mu(s, u, X, rl, t - l))
        end
        σ2 = u.σ2[s.sw_var ? r0 : 1]
        ll[j] = -0.5 * (log(2pi) + log(σ2) + e^2 / σ2)
    end
    return ll
end

# P for the transition into time t
_ms_Pt(s::_MSSpec, u, Z, t) = s.q == 1 ? _ms_P_from_probs(u.tr, s.k) :
                               _ms_P_tvtp(u.tr, s.k, s.q, view(Z, t, :))

"""
internal: the log-space Hamilton filter. Returns (loglik, lf, lp) with lf/lp
the filtered / predicted joint log-probabilities, J × nobs.
"""
function _ms_filter(s::_MSSpec, θ::AbstractVector, st::_MSStates, y, X, Z;
                    init=nothing)
    u = _ms_unpack(s, θ)
    T = promote_type(eltype(θ), Float64)
    k, p = s.k, s.p
    J = size(st.digits, 1)
    n = length(y)
    nobs = n - p
    lf = Matrix{T}(undef, J, nobs)
    lp = Matrix{T}(undef, J, nobs)
    ll = Vector{T}(undef, J)
    # the first window (S_{p+1}, …, S_1): π(S_1) and P applied p times
    P1 = _ms_Pt(s, u, Z, 1)
    π0 = init === nothing ? _ms_steady(P1) : T.(init)
    any(<(0), π0) && return (T(-Inf), lf, lp)
    # log P for the transitions into t = 2..p+1, computed once
    lPinit = [log.(max.(_ms_Pt(s, u, Z, t), floatmin(Float64))) for t in 2:(p + 1)]
    for j in 1:J
        v = log(max(π0[st.digits[j, p + 1] + 1], floatmin(Float64)))
        for m in p:-1:1
            v += lPinit[p + 1 - m][st.digits[j, m] + 1, st.digits[j, m + 1] + 1]
        end
        lp[j, 1] = v
    end
    total = zero(T)
    for τ in 1:nobs
        t = τ + p
        if τ > 1
            Pt = _ms_Pt(s, u, Z, t)
            lPt = log.(max.(Pt, floatmin(Float64)))
            for j in 1:J
                a0 = st.digits[j, 1] + 1
                mx = T(-Inf)
                for i in st.pred[j]
                    mx = max(mx, lPt[a0, st.digits[i, 1] + 1] + lf[i, τ - 1])
                end
                acc = zero(T)
                for i in st.pred[j]
                    acc += exp(lPt[a0, st.digits[i, 1] + 1] + lf[i, τ - 1] - mx)
                end
                lp[j, τ] = mx + log(acc)
            end
        end
        _ms_condll!(ll, s, u, st, y, X, t)
        mx = T(-Inf)
        for j in 1:J
            ll[j] += lp[j, τ]
            mx = max(mx, ll[j])
        end
        isfinite(mx) || return (T(-Inf), lf, lp)
        acc = zero(T)
        for j in 1:J
            acc += exp(ll[j] - mx)
        end
        lm = mx + log(acc)
        for j in 1:J
            lf[j, τ] = ll[j] - lm
        end
        total += lm
    end
    return (total, lf, lp)
end

# Kim smoother on the joint states; returns the smoothed joint
# log-probabilities and the summed pairwise regime probabilities
# Σ_t Pr(S_t = a, S_{t-1} = b | Y) for the EM transition step.
function _ms_smooth(s::_MSSpec, θ, st::_MSStates, y, X, Z, lf, lp)
    u = _ms_unpack(s, θ)
    J, nobs = size(lf)
    k, p = s.k, s.p
    ls = similar(lf)
    ls[:, nobs] = lf[:, nobs]
    pair = zeros(k, k)
    for τ in (nobs - 1):-1:1
        t1 = τ + 1 + p
        lPt = log.(max.(_ms_Pt(s, u, Z, t1), floatmin(Float64)))
        for i in 1:J
            acc = [lPt[st.digits[j, 1] + 1, st.digits[i, 1] + 1] + ls[j, τ + 1] - lp[j, τ + 1]
                   for j in st.succ[i]]
            ls[i, τ] = lf[i, τ] + _logsumexp(acc)
        end
        for i in 1:J, j in st.succ[i]
            a, b = st.digits[j, 1] + 1, st.digits[i, 1] + 1
            pair[a, b] += exp(lf[i, τ] + lPt[a, b] + ls[j, τ + 1] - lp[j, τ + 1])
        end
    end
    return ls, pair
end

_ms_marginal(st::_MSStates, L::AbstractMatrix, k::Int) =
    [sum(exp(L[j, τ]) for j in axes(st.digits, 1) if st.digits[j, 1] == r - 1)
     for τ in axes(L, 2), r in 1:k]

"""
    MSModel

A fitted Markov-switching model, from [`fit_ms_regression`](@ref) or
[`fit_ms_autoregression`](@ref). `P` is the **left-stochastic** transition
matrix -- `P[i, j] = Pr(S_t = i | S_{t−1} = j)`, **columns** sum to one,
the opposite of many textbooks -- or `nothing` when the transition
probabilities vary with covariates. Regime probabilities are in
[`regime_probabilities`](@ref); `n_modes` counts the distinct optima the
restart search found, and `mode_logliks` lists them.
"""
struct MSModel <: TimeSeriesModel
    y::Vector{Float64}
    exog::Union{Nothing,Matrix{Float64}}
    exog_tvtp::Union{Nothing,Matrix{Float64}}
    spec::_MSSpec
    params::Vector{Float64}
    names::Vector{String}
    P::Union{Nothing,Matrix{Float64}}
    initial::Vector{Float64}
    loglik::Float64
    nobs::Int
    filtered::Matrix{Float64}
    smoothed::Matrix{Float64}
    converged::Bool
    estimated::Bool
    n_modes::Int
    mode_logliks::Vector{Float64}
end

# --- unconstrained <-> natural --------------------------------------------------
# transition: logits log(p[j->i] / p[j->k-1]) (closed form; the reference
# root-finds this for k > 2); TVTP logits are already unconstrained.
# variances: log. AR: partrans per regime (stationary). means/exog: raw.

function _ms_to_natural(s::_MSSpec, w::AbstractVector)
    T = eltype(w)
    k = s.k
    θ = Vector{T}(undef, length(w))
    ntr = (k - 1) * k * s.q
    if s.q == 1
        for j in 1:k
            l = vcat([w[(i - 1) * k + j] for i in 1:(k - 1)], zero(T))
            m = maximum(l)
            e = exp.(l .- m)
            pr = e ./ sum(e)
            for i in 1:(k - 1)
                θ[(i - 1) * k + j] = pr[i]
            end
        end
    else
        θ[1:ntr] = w[1:ntr]
    end
    names = _ms_names(s)
    for idx in (ntr + 1):length(w)
        θ[idx] = startswith(names[idx], "sigma2") ? exp(w[idx]) : w[idx]
    end
    _ms_ar_map!(θ, s, names, partrans)
    return θ
end

function _ms_to_unconstrained(s::_MSSpec, θ::AbstractVector)
    k = s.k
    w = Vector{Float64}(undef, length(θ))
    ntr = (k - 1) * k * s.q
    if s.q == 1
        P = _ms_P_from_probs(θ[1:ntr], k)
        for j in 1:k, i in 1:(k - 1)
            w[(i - 1) * k + j] = log(max(P[i, j], 1e-12) / max(P[k, j], 1e-12))
        end
    else
        w[1:ntr] = θ[1:ntr]
    end
    names = _ms_names(s)
    for idx in (ntr + 1):length(θ)
        w[idx] = startswith(names[idx], "sigma2") ? log(θ[idx]) : θ[idx]
    end
    _ms_ar_map!(w, s, names, invpartrans)
    return w
end

# apply f to each regime's AR coefficient vector in place
function _ms_ar_map!(v, s::_MSSpec, names, f)
    s.p == 0 && return v
    for r in 1:(s.sw_ar ? s.k : 1)
        idx = [findfirst(==(s.sw_ar ? "ar.L$l[$(r - 1)]" : "ar.L$l"), names) for l in 1:s.p]
        v[idx] = f(v[idx])
    end
    return v
end

# --- EM (a warm start) ------------------------------------------------------------

function _ms_em_step(s::_MSSpec, θ, st, y, X, Z)
    ll, lf, lp = _ms_filter(s, θ, st, y, X, Z)
    isfinite(ll) || return θ
    ls, pair = _ms_smooth(s, θ, st, y, X, Z, lf, lp)
    k, p = s.k, s.p
    names = _ms_names(s)
    θn = copy(θ)
    # transition: closed form from the pairwise probabilities
    if s.q == 1
        for j in 1:k
            tot = sum(pair[:, j])
            tot > 0 || continue
            for i in 1:(k - 1)
                θn[(i - 1) * k + j] = clamp(pair[i, j] / tot, 1e-8, 1 - 1e-8)
            end
        end
        P = _ms_P_from_probs(θn[1:((k - 1) * k)], k)
        any(<(1e-8), P) && return θ
    end
    # joint-state weights
    J, nobs = size(ls)
    W = exp.(ls)
    u = _ms_unpack(s, θn)
    # regression / AR: alternate (ECM) -- φ given μ, then μ (and β) given φ
    if p >= 1
        # φ given μ: per joint state, the mean-adjusted regression
        nar = s.sw_ar ? k : 1
        for r in 1:nar
            rows = Vector{Float64}[]; ys = Float64[]; ws = Float64[]
            for τ in 1:nobs, j in 1:J
                r0 = st.digits[j, 1] + 1
                s.sw_ar && r0 != r && continue
                t = τ + p
                σ2 = u.σ2[s.sw_var ? r0 : 1]
                push!(ys, y[t] - _ms_mu(s, u, X, r0, t))
                push!(rows, [y[t - l] - _ms_mu(s, u, X, st.digits[j, l + 1] + 1, t - l) for l in 1:p])
                push!(ws, W[j, τ] / σ2)
            end
            count(>(0), ws) > p || continue
            b, = _ols(reduce(vcat, permutedims.(rows)), ys; weights=ws)
            for l in 1:p
                θn[findfirst(==(s.sw_ar ? "ar.L$l[$(r - 1)]" : "ar.L$l"), names)] = b[l]
            end
        end
        u = _ms_unpack(s, θn)
    end
    # μ (const, β) given φ: e = ỹ_t − (μ_{s0,t} − Σ φ_l μ_{s_l,t−l}), linear in
    # the mean parameters
    mnames = [nm for nm in names if startswith(nm, "const") || startswith(nm, "x")]
    if !isempty(mnames)
        mi = Dict(nm => c for (c, nm) in enumerate(mnames))
        rows = Vector{Float64}[]; ys = Float64[]; ws = Float64[]
        for τ in 1:nobs, j in 1:J
            t = τ + p
            r0 = st.digits[j, 1] + 1
            ϕ = [u.ϕ[l][s.sw_ar ? r0 : 1] for l in 1:p]
            row = zeros(length(mnames))
            for (lag, coef) in zip(0:p, vcat(1.0, -ϕ))
                rl = st.digits[j, lag + 1] + 1
                tl = t - lag
                if s.trend
                    row[mi[s.sw_trend ? "const[$(rl - 1)]" : "const"]] += coef
                end
                for c in 1:s.nx
                    row[mi[s.sw_exog ? "x$c[$(rl - 1)]" : "x$c"]] += coef * X[tl, c]
                end
            end
            push!(rows, row)
            push!(ys, y[t] - sum((ϕ[l] * y[t - l] for l in 1:p); init=0.0))
            push!(ws, W[j, τ] / u.σ2[s.sw_var ? r0 : 1])
        end
        if count(>(0), ws) > length(mnames)
            b, = _ols(reduce(vcat, permutedims.(rows)), ys; weights=ws)
            for (c, nm) in enumerate(mnames)
                θn[findfirst(==(nm), names)] = b[c]
            end
        end
        u = _ms_unpack(s, θn)
    end
    # variances from the weighted squared residuals
    num = zeros(k); den = zeros(k)
    for τ in 1:nobs, j in 1:J
        t = τ + p
        r0 = st.digits[j, 1] + 1
        e = y[t] - _ms_mu(s, u, X, r0, t)
        for l in 1:p
            e -= u.ϕ[l][s.sw_ar ? r0 : 1] * (y[t - l] - _ms_mu(s, u, X, st.digits[j, l + 1] + 1, t - l))
        end
        num[r0] += W[j, τ] * e^2
        den[r0] += W[j, τ]
    end
    if s.sw_var
        for r in 1:k
            den[r] > 0 && (θn[findfirst(==("sigma2[$(r - 1)]"), names)] = max(num[r] / den[r], 1e-8))
        end
    else
        θn[findfirst(==("sigma2"), names)] = max(sum(num) / sum(den), 1e-8)
    end
    return θn
end

# --- starting values ---------------------------------------------------------------

function _ms_start(s::_MSSpec, y, X)
    n = length(y); p = s.p; k = s.k
    # a one-regime fit by least squares: y_t on [1, x_t, y_{t-1..p}]
    cols = Vector{Float64}[]
    s.trend && push!(cols, ones(n - p))
    for c in 1:s.nx
        push!(cols, X[(p + 1):n, c])
    end
    for l in 1:p
        push!(cols, y[(p + 1 - l):(n - l)])
    end
    D = reduce(hcat, cols)
    b = D \ y[(p + 1):n]
    e = y[(p + 1):n] .- D * b
    sd = std(e)
    ϕ = p > 0 ? b[(end - p + 1):end] : Float64[]
    names = _ms_names(s)
    θ = zeros(length(names))
    for (idx, nm) in enumerate(names)
        if startswith(nm, "p[")
            # p[j->i]: stay with probability 0.8, split the rest
            m = match(r"p\[(\d+)->(\d+)\]", nm)
            j, i = parse(Int, m[1]), parse(Int, m[2])
            θ[idx] = s.q == 1 ? (i == j ? 0.8 : 0.2 / (k - 1)) : (i == j ? 1.0 : 0.0)
        elseif startswith(nm, "const")
            μ = s.trend ? b[1] / (1 - sum(ϕ)) : 0.0
            r = occursin('[', nm) ? parse(Int, match(r"\[(\d+)\]", nm)[1]) : nothing
            θ[idx] = r === nothing ? μ : μ + sd * (r - (k - 1) / 2)
        elseif startswith(nm, "x")
            c = parse(Int, match(r"^x(\d+)", nm)[1])
            θ[idx] = b[(s.trend ? 1 : 0) + c]
        elseif startswith(nm, "sigma2")
            θ[idx] = sd^2
        elseif startswith(nm, "ar.L")
            l = parse(Int, match(r"ar\.L(\d+)", nm)[1])
            θ[idx] = clamp(ϕ[l], -0.9, 0.9) * 0.5
        end
    end
    return θ
end

# --- fitting -----------------------------------------------------------------------

function _ms_check_spec(s::_MSSpec)
    s.k >= 1 || throw(ArgumentError("Markov switching: k (the number of regimes) must be >= 1"))
    s.p >= 0 || throw(ArgumentError("Markov switching: the AR order must be >= 0"))
    if s.k >= 2
        sw = (s.trend && s.sw_trend) || (s.nx > 0 && s.sw_exog) || (s.p > 0 && s.sw_ar) || s.sw_var
        sw || throw(ArgumentError(
            "Markov switching: nothing switches -- with k = $(s.k) regimes, at least one of the " *
            "intercept (switching_trend, needs trend = true), the regressors (switching_exog), " *
            "the AR coefficients (switching_ar) or the variance (switching_variance) must vary " *
            "by regime. Otherwise the transition probabilities are not identified (statsmodels " *
            "returns its start values untouched here, with no warning)."))
    end
    return s
end

function _ms_fit(s::_MSSpec, y, X, Z; params, init, em_iter, n_starts, rng, iterations)
    st = _ms_states(s.k, s.p)
    names = _ms_names(s)
    if params !== nothing
        length(params) == length(names) || throw(ArgumentError(
            "Markov switching: params has $(length(params)) entries; the model has " *
            "$(length(names)) ($(join(names, ", ")))"))
        θ = Vector{Float64}(params)
        ll = _ms_filter(s, θ, st, y, X, Z; init=init)[1]
        isfinite(ll) || throw(ArgumentError("Markov switching: the likelihood is not finite at params"))
        return θ, ll, true, false, 1, [ll], st
    end
    θs = _ms_start(s, y, X)
    w0 = _ms_to_unconstrained(s, θs)
    starts = [w0]
    for _ in 2:n_starts
        push!(starts, w0 .+ 0.5 .* randn(rng, length(w0)))
    end
    results = Vector{Any}(undef, length(starts))
    run_one(w) = begin
        θ = _ms_to_natural(s, w)
        for _ in 1:em_iter
            θ = _ms_em_step(s, θ, st, y, X, Z)
        end
        wθ = try _ms_to_unconstrained(s, θ) catch; w end
        obj(v) = (l = _ms_filter(s, _ms_to_natural(s, v), st, y, X, Z; init=init)[1];
                  isfinite(l) ? -l : eltype(v)(1e15))
        res = try
            _optimize(obj, wθ; method=:bfgs, autodiff=true, g_tol=1e-8, iterations=iterations)
        catch e
            e isa ArgumentError || e isa DomainError || e isa SingularException ||
                e isa PosDefException || rethrow()
            nothing
        end
        res === nothing ? nothing : (_ms_to_natural(s, res.minimizer), -res.minimum, res.converged)
    end
    if Threads.nthreads() > 1 && length(starts) >= 4
        Threads.@threads for idx in eachindex(starts)
            results[idx] = run_one(starts[idx])
        end
    else
        for idx in eachindex(starts)
            results[idx] = run_one(starts[idx])
        end
    end
    ok = [r for r in results if r !== nothing && isfinite(r[2])]
    isempty(ok) && throw(ArgumentError("Markov switching: the optimiser failed from every start"))
    best = ok[argmax([r[2] for r in ok])]
    # distinct optima, after the label ambiguity: log-likelihoods 1e-4 apart
    lls = sort([r[2] for r in ok]; rev=true)
    modes = Float64[]
    for l in lls
        any(m -> abs(m - l) < 1e-4, modes) || push!(modes, l)
    end
    return best[1], best[2], best[3], true, length(modes), modes, st
end

function _ms_build(s, y, X, Z, θ, ll, conv, est, nmodes, modes, st, init)
    lf, lp = _ms_filter(s, θ, st, y, X, Z; init=init)[2:3]
    ls, _ = _ms_smooth(s, θ, st, y, X, Z, lf, lp)
    u = _ms_unpack(s, θ)
    P = s.q == 1 ? _ms_P_from_probs(u.tr, s.k) : nothing
    π0 = init === nothing ? _ms_steady(_ms_Pt(s, u, Z, 1)) : Vector{Float64}(init)
    return MSModel(y, X, Z, s, θ, _ms_names(s), P, π0, ll, length(y) - s.p,
                   _ms_marginal(st, lf, s.k), _ms_marginal(st, ls, s.k), conv, est, nmodes, modes)
end

const _MS_DOC_COMMON = """
- `k`: the number of regimes (`k = 1` is allowed and is an exact reduction
  to least squares).
- `switching_trend`, `switching_exog`, `switching_variance` (and, for the
  autoregression, `switching_ar`): which parameters vary by regime. A
  specification in which **nothing** switches is refused -- its transition
  probabilities are not identified, and statsmodels silently returns its
  starting values for them.
- `exog_tvtp`: covariates for **time-varying transition probabilities**
  (include a column of ones for a constant); the transition parameters are
  then logits.
- `initial`: the regime distribution at `t = −order` (before the first
  usable observation); default the steady state of `P`, as statsmodels.
  R's `MSwM` estimates it instead, which moves the likelihood (by 0.072 on
  the canonical fixture).
- `em_iter` EM steps (default 5) warm-start each of `n_starts` (default 10)
  gradient fits, run on threads; the best is returned and `n_modes` reports
  how many distinct optima were found. **The likelihood is multimodal**: on
  Hamilton's GNP series only about two thirds of random starts reach the
  global maximum, pure EM stops on a no-persistence ridge, and with a
  switching variance the reference's own default fit is a local optimum 1.36
  below the best.
- `params` evaluates the model at given parameters (in `coefnames` order)
  without fitting.
"""

"""
    fit_ms_regression(y; k=2, exog=nothing, trend=true, switching_trend=true,
                      switching_exog=false, switching_variance=false,
                      exog_tvtp=nothing, initial=nothing, em_iter=5,
                      n_starts=10, rng=Random.default_rng(), params=nothing) -> MSModel

Markov-switching regression (Hamilton 1989):
```
y_t = c_{S_t} + x_t' β_{S_t} + ε_t,   ε_t ~ N(0, σ²_{S_t})
```
with `S_t` a first-order Markov chain with left-stochastic transition matrix
`P` (`P[i,j] = Pr(S_t = i | S_{t−1} = j)`). Putting lagged `y` in `exog`
gives R's `MSwM` model, which is **not** Hamilton's switching autoregression
-- that is [`fit_ms_autoregression`](@ref).

$(_MS_DOC_COMMON)
`switching_exog` defaults to `false` here and in the autoregression; the
reference defaults it to `true` in one class and `false` in the other.
"""
function fit_ms_regression(y; k::Integer=2, exog=nothing, trend::Bool=true,
                           switching_trend::Bool=true, switching_exog::Bool=false,
                           switching_variance::Bool=false, exog_tvtp=nothing, initial=nothing,
                           em_iter::Integer=5, n_starts::Integer=10,
                           rng=Random.default_rng(), params=nothing, iterations::Integer=1000)
    return _ms_entry(y, 0; k, exog, trend, switching_trend, switching_exog,
                     switching_ar=false, switching_variance, exog_tvtp, initial, em_iter,
                     n_starts, rng, params, iterations)
end

"""
    fit_ms_autoregression(y, order; k=2, exog=nothing, trend=true,
                          switching_trend=true, switching_exog=false,
                          switching_ar=false, switching_variance=false,
                          exog_tvtp=nothing, initial=nothing, em_iter=5,
                          n_starts=10, rng=Random.default_rng(), params=nothing) -> MSModel

Hamilton's (1989) Markov-switching autoregression, in its **mean-adjusted**
form:
```
y_t − μ_{S_t} = Σ_{j=1}^{order} φ_j (y_{t−j} − μ_{S_{t−j}}) + ε_t
```
The regime at every lag enters, so the filter runs over the `k^(order+1)`
joint regime states. `fit_ms_autoregression(y, 4)` with the defaults --
switching mean, common AR coefficients and variance -- is Hamilton's model
for US GNP growth, and reproduces statsmodels'
`MarkovAutoregression(…, switching_ar=False)` (whose own default,
`switching_ar=True`, is a different, 13-parameter model).

$(_MS_DOC_COMMON)
R's `MSwM` cannot express this model: it puts the lags in as plain levels.
On the canonical fixture its specification evaluated at statsmodels'
parameters is 13.2 log-likelihood units away -- a different model, not a
different optimum.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> gnp = readdlm(joinpath(pkgdir(TSAnalytics), "handoff", "stage 11", "fixtures",
                              "hamilton_rgnp.csv"), ','; skipstart=1)[:, 2];

julia> m = canonicalise(fit_ms_autoregression(gnp, 4));

julia> round(m.loglik, digits=4)
-181.2634

julia> round.(expected_durations(m), digits=2)
2-element Vector{Float64}:
  4.08
 10.43
```
"""
function fit_ms_autoregression(y, order::Integer; k::Integer=2, exog=nothing, trend::Bool=true,
                               switching_trend::Bool=true, switching_exog::Bool=false,
                               switching_ar::Bool=false, switching_variance::Bool=false,
                               exog_tvtp=nothing, initial=nothing, em_iter::Integer=5,
                               n_starts::Integer=10, rng=Random.default_rng(), params=nothing,
                               iterations::Integer=1000)
    order >= 0 || throw(ArgumentError("fit_ms_autoregression: order must be >= 0, got $order"))
    return _ms_entry(y, Int(order); k, exog, trend, switching_trend, switching_exog,
                     switching_ar, switching_variance, exog_tvtp, initial, em_iter,
                     n_starts, rng, params, iterations)
end

function _ms_entry(y, p; k, exog, trend, switching_trend, switching_exog, switching_ar,
                   switching_variance, exog_tvtp, initial, em_iter, n_starts, rng, params,
                   iterations)
    yv = Vector{Float64}(tsvalues(y))
    any(isnan, yv) && throw(ArgumentError("Markov switching: y contains NaN"))
    n = length(yv)
    X = exog === nothing ? zeros(n, 0) :
        (exog isa AbstractVector ? reshape(Float64.(exog), :, 1) : Matrix{Float64}(exog))
    size(X, 1) == n || throw(ArgumentError("Markov switching: exog has $(size(X, 1)) rows, y has $n"))
    Z = exog_tvtp === nothing ? ones(n, 1) :
        (exog_tvtp isa AbstractVector ? reshape(Float64.(exog_tvtp), :, 1) : Matrix{Float64}(exog_tvtp))
    size(Z, 1) == n || throw(ArgumentError("Markov switching: exog_tvtp has $(size(Z, 1)) rows, y has $n"))
    q = exog_tvtp === nothing ? 1 : size(Z, 2)
    kk = Int(k)
    s = _ms_check_spec(_MSSpec(kk, p, trend, size(X, 2), switching_trend && kk > 1,
                               switching_exog && kk > 1, switching_ar && kk > 1,
                               switching_variance && kk > 1, kk > 1 ? q : 1))
    n - p > length(_ms_names(s)) || throw(ArgumentError(
        "Markov switching: $(n - p) usable observations for $(length(_ms_names(s))) parameters"))
    if initial !== nothing
        length(initial) == kk && all(>=(0), initial) && sum(initial) ≈ 1 || throw(ArgumentError(
            "Markov switching: initial must be a probability vector of length k = $kk"))
    end
    em_iter >= 0 || throw(ArgumentError("Markov switching: em_iter must be >= 0"))
    n_starts >= 1 || throw(ArgumentError("Markov switching: n_starts must be >= 1"))
    θ, ll, conv, est, nmodes, modes, st = _ms_fit(s, yv, X, Z; params=params, init=initial,
                                                    em_iter=Int(em_iter), n_starts=Int(n_starts),
                                                    rng=rng, iterations=Int(iterations))
    return _ms_build(s, yv, exog === nothing ? nothing : X, exog_tvtp === nothing ? nothing : Z,
                     θ, ll, conv, est, nmodes, modes, st, initial)
end

# k = 1 needs no transition: the chain is a single state; the filter above
# handles it with J = 1 (P = [1]).

# --- results ------------------------------------------------------------------------

"""
    regime_probabilities(m::MSModel; kind=:smoothed) -> Matrix   # nobs × k

`Pr(S_t = r | …)` for each usable observation (the first `order` are lags):
`kind = :smoothed` conditions on the whole sample, `:filtered` on data to `t`.
"""
function regime_probabilities(m::MSModel; kind::Symbol=:smoothed)
    kind === :smoothed && return copy(m.smoothed)
    kind === :filtered && return copy(m.filtered)
    throw(ArgumentError("regime_probabilities: kind must be :smoothed or :filtered"))
end

# Residuals given the regime path, averaged over the smoothed joint
# distribution: E = nobs × k, E[τ, r] = E[e_t | S_t = r, all data] (the
# lagged regimes integrated out), and the pooled Σ_r Pr(S_t = r) E[τ, r].
function _ms_conditional_resid(m::MSModel)
    s = m.spec
    st = _ms_states(s.k, s.p)
    n = length(m.y)
    X = m.exog === nothing ? zeros(n, 0) : m.exog
    Z = m.exog_tvtp === nothing ? ones(n, 1) : m.exog_tvtp
    _, lf, lp = _ms_filter(s, m.params, st, m.y, X, Z; init=m.initial)
    ls, _ = _ms_smooth(s, m.params, st, m.y, X, Z, lf, lp)
    u = _ms_unpack(s, m.params)
    J, nobs = size(ls)
    E = zeros(nobs, s.k)
    W = zeros(nobs, s.k)
    for τ in 1:nobs
        t = τ + s.p
        for j in 1:J
            r0 = st.digits[j, 1] + 1
            e = m.y[t] - _ms_mu(s, u, X, r0, t)
            for l in 1:s.p
                rl = st.digits[j, l + 1] + 1
                e -= u.ϕ[l][s.sw_ar ? r0 : 1] * (m.y[t - l] - _ms_mu(s, u, X, rl, t - l))
            end
            w = exp(ls[j, τ])
            E[τ, r0] += w * e
            W[τ, r0] += w
        end
    end
    E ./= max.(W, floatmin(Float64))
    return E, vec(sum(W .* E; dims=2))
end

"""
    residuals(m::MSModel) -> Vector
    fitted(m::MSModel) -> Vector

The residual averaged over the smoothed regime distribution,
`Σ_r Pr(S_t = r | all data) E[e_t | S_t = r]`, for each usable observation
(the first `order` are lags), and `y_t` less it. A smoothed quantity --
it uses the whole sample -- so it is a diagnostic, not a forecast error.
"""
StatsAPI.residuals(m::MSModel) = _ms_conditional_resid(m)[2]
StatsAPI.fitted(m::MSModel) = m.y[(m.spec.p + 1):end] .- StatsAPI.residuals(m)

"""
    expected_durations(m::MSModel) -> Vector

`1 / (1 − P[r, r])`: the expected number of periods a regime lasts once
entered. (Not defined for time-varying transition probabilities.)
"""
function expected_durations(m::MSModel)
    m.P === nothing && throw(ArgumentError(
        "expected_durations: the transition probabilities vary over time"))
    return [1 / (1 - m.P[r, r]) for r in 1:m.spec.k]
end

"""
    canonicalise(m::MSModel) -> MSModel

The same fit with its regimes relabelled in a fixed order -- ascending
intercept, or ascending variance when only the variance switches -- so two
fits can be compared. The label ambiguity is real: a random start can return
exactly the relabelled optimum (parameters 1.5 apart, likelihood equal). For
`k = 2` relabelling is **not** `(p[0->0], p[1->0]) → (p[1->0], p[0->0])` but
`→ (1 − p[1->0], 1 − p[0->0])`, with the regime-specific parameters swapped.
"""
function canonicalise(m::MSModel)
    s, k = m.spec, m.spec.k
    k == 1 && return m
    u = _ms_unpack(s, m.params)
    key = s.trend && s.sw_trend ? u.c :
          s.sw_var ? u.σ2 :
          s.nx > 0 && s.sw_exog ? u.B[1] :
          s.p > 0 && s.sw_ar ? u.ϕ[1] : collect(1.0:k)
    perm = sortperm(collect(key))          # new regime r is old regime perm[r]
    perm == collect(1:k) && return m
    names = m.names
    θ = copy(m.params)
    if s.q == 1
        P = _ms_P_from_probs(u.tr, k)
        Pn = P[perm, perm]
        for j in 1:k, i in 1:(k - 1)
            θ[(i - 1) * k + j] = Pn[i, j]
        end
    else
        # TVTP logits are relative to the last regime; relabel by recomputing
        throw(ArgumentError("canonicalise: not available with time-varying transition probabilities"))
    end
    for (idx, nm) in enumerate(names)
        mm = match(r"^(.*)\[(\d+)\]$", nm)
        (mm === nothing || startswith(nm, "p[")) && continue
        r = parse(Int, mm[2]) + 1
        src = "$(mm[1])[$(perm[r] - 1)]"
        θ[idx] = m.params[findfirst(==(src), names)]
    end
    st = _ms_states(k, s.p)
    init = m.initial[perm]
    Z = m.exog_tvtp === nothing ? ones(length(m.y), 1) : m.exog_tvtp
    X = m.exog === nothing ? zeros(length(m.y), 0) : m.exog
    ll = _ms_filter(s, θ, st, m.y, X, Z; init=init)[1]
    return _ms_build(s, m.y, m.exog, m.exog_tvtp, θ, ll, m.converged, m.estimated, m.n_modes,
                     m.mode_logliks, st, init)
end

StatsAPI.coef(m::MSModel) = copy(m.params)
StatsAPI.coefnames(m::MSModel) = copy(m.names)
StatsAPI.loglikelihood(m::MSModel) = m.loglik
StatsAPI.nobs(m::MSModel) = m.nobs
StatsAPI.dof(m::MSModel) = length(m.params)
StatsAPI.aic(m::MSModel) = -2m.loglik + 2length(m.params)
StatsAPI.bic(m::MSModel) = -2m.loglik + log(m.nobs) * length(m.params)

function Base.show(io::IO, m::MSModel)
    s = m.spec
    kind = s.p == 0 ? "regression" : "autoregression, order $(s.p) (mean-adjusted)"
    println(io, "Markov-switching $kind, $(s.k) regime$(s.k > 1 ? "s" : ""), nobs = $(m.nobs)")
    for (nm, v) in zip(m.names, m.params)
        println(io, "  ", rpad(nm, 18), lpad(string(round(v; sigdigits=8)), 16))
    end
    if m.P !== nothing && s.k > 1
        println(io, "  expected durations: ", round.(expected_durations(m); digits=4))
    end
    println(io, "  log-likelihood = ", round(m.loglik; digits=4), "   AIC = ",
            round(StatsAPI.aic(m); digits=4), "   BIC = ", round(StatsAPI.bic(m); digits=4))
    if m.estimated
        print(io, "  ", m.converged ? "optimiser converged" : "optimiser did NOT converge",
              "; ", m.n_modes, " distinct optim", m.n_modes == 1 ? "um" : "a",
              " found across the starts")
        m.n_modes > 1 && print(io, " (", join(round.(m.mode_logliks; digits=4), ", "), ")")
    else
        print(io, "  evaluated at given parameters (not estimated)")
    end
end
