export fit_arimax, fit_sarimax, ArimaxModel, SarimaxModel

"""
    ArimaxModel <: UnivariateModel

Result of [`fit_arimax`](@ref): ARIMA(p,d,q) with exogenous regressors,
either `model=:mle` (`beta` a fixed jointly-estimated coefficient) or
`model=:tvss` (`beta_filtered` a genuinely time-varying latent state) --
see `fit_arimax`'s own docstring for why these are different models, not
two ways of computing the same one.

- `model`: `:mle` or `:tvss`.
- `method`: `:ml` or `:css_ml` -- Stage 6.5's own estimation-procedure
  choice for the ARMA part, independent of `model`.
- `beta`/`beta_filtered`/`Q_beta`: exactly one of `beta` (`:mle`) or
  `beta_filtered`/`Q_beta` (`:tvss`) is populated, the other `nothing`.
- `arma`: the underlying `ArmaModel` -- for `:mle`, a genuine `fit_arma`-
  shaped result on `y - X*beta` (already differenced); for `:tvss`, its
  `ar`/`ma`/`sigma2`/`se` fields hold the jointly-estimated ARMA
  coefficients from the combined filter, but its own `loglik`/`aic`/
  `bic`/`nobs`/`converged` are **not** authoritative for the overall
  model -- use this struct's own top-level fields for those, which are
  always correct for whichever `model` was actually fit.
- `d`/`original_y`: as `ArimaModel` (Stage 6.6).
- `exog`: the regressor matrix as given, plus a constant column if
  `include_mean=true` (`n x k`).
- `nobs_diffuse`: `:tvss` only (`nothing` for `:mle`) -- diagnostic only:
  the number of initial diffuse-phase observations (how long `beta`'s
  diffuse prior took to resolve). Does **not** reduce `nobs`/`loglik`'s
  own observation count -- `kalman_filter_diffuse` sums the full,
  inclusive likelihood over all `nd` observations (see its own
  docstring).
- `se`/`loglik`/`aic`/`bic`/`nobs`/`converged`: authoritative for the
  overall model. `se` order: `[beta; ar; ma]` for `:mle`,
  `[ar; ma; sigma2; Q_beta]` for `:tvss` (matching `coef`'s own order).
"""
struct ArimaxModel <: UnivariateModel
    model::Symbol
    method::Symbol
    beta::Union{Vector{Float64},Nothing}
    beta_filtered::Union{Matrix{Float64},Nothing}
    Q_beta::Union{Vector{Float64},Nothing}
    arma::ArmaModel
    d::Int
    original_y::Vector{Float64}
    exog::Matrix{Float64}
    se::Vector{Float64}
    loglik::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    nobs_diffuse::Union{Int,Nothing}
    converged::Bool
end

"""
    SarimaxModel <: UnivariateModel

Result of [`fit_sarimax`](@ref) -- exactly [`ArimaxModel`](@ref)'s shape,
extended with `seasonal_order` and an underlying `SarimaModel` (Stage
6.7) rather than `ArmaModel`, the same relationship [`SarimaModel`](@ref)
has to [`ArmaModel`](@ref).
"""
struct SarimaxModel <: UnivariateModel
    model::Symbol
    method::Symbol
    beta::Union{Vector{Float64},Nothing}
    beta_filtered::Union{Matrix{Float64},Nothing}
    Q_beta::Union{Vector{Float64},Nothing}
    arma::SarimaModel
    seasonal_order::Tuple{Int,Int,Int,Int}
    original_y::Vector{Float64}
    exog::Matrix{Float64}
    se::Vector{Float64}
    loglik::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    nobs_diffuse::Union{Int,Nothing}
    converged::Bool
end

"_arma_blocks(a) -> (phi, theta, Phi, Theta) -- `ArimaxModel`/`SarimaxModel`
wrap either an `ArmaModel` (no seasonal fields at all, `P=Q=0` always)
or a `SarimaModel` (`phi`/`theta`/`Phi`/`Theta` stored separately, not
combined) -- this normalizes the two shapes for `coef`/`show`, which
need seasonal blocks generically without caring which struct they're
looking at."
_arma_blocks(a::ArmaModel) = (a.ar, a.ma, Float64[], Float64[])
_arma_blocks(a::SarimaModel) = (a.phi, a.theta, a.Phi, a.Theta)

for M in (:ArimaxModel, :SarimaxModel)
    @eval begin
        StatsAPI.loglikelihood(m::$M) = m.loglik
        StatsAPI.aic(m::$M) = m.aic
        StatsAPI.bic(m::$M) = m.bic
        StatsAPI.nobs(m::$M) = m.nobs
        function StatsAPI.coef(m::$M)
            phi, theta, Phi, Theta = _arma_blocks(m.arma)
            m.model == :mle ? vcat(m.beta, phi, theta, Phi, Theta) :
                               vcat(phi, theta, Phi, Theta, m.arma.sigma2, m.Q_beta)
        end
    end
end

"_arimax_unpack(raw, p, q, P, Q, s, k) -> (beta, ar, ma, yc) -- `:mle`
path's natural-parametrization unpack, seasonal-general via
`combined_ar_ma` (reduces exactly to the plain non-seasonal case when
`P=Q=0`, already proven by Stage 6.7's own regression suite): `beta`
first (`k` entries, unconstrained -- an ordinary linear-regression
coefficient, no transform needed), then the four raw AR/MA blocks
(already-transformed by the caller when used inside the optimizer
objective, left natural for the Hessian/OPG objective -- matching
`_sarima_unpack`'s own established split)."
function _arimax_unpack(raw::AbstractVector, yd::Vector{Float64}, Xd::Matrix{Float64},
                         p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer, k::Integer)
    T = eltype(raw)
    beta = raw[1:k]
    i = k
    phi = p > 0 ? raw[(i + 1):(i + p)] : T[]
    i += p
    theta = q > 0 ? raw[(i + 1):(i + q)] : T[]
    i += q
    Phi = P > 0 ? raw[(i + 1):(i + P)] : T[]
    i += P
    Theta = Q > 0 ? raw[(i + 1):(i + Q)] : T[]
    ar, ma = combined_ar_ma(phi, Phi, theta, Theta, s)
    yc = yd .- Xd * beta
    return beta, ar, ma, yc
end

"_arimax_natural_objective(raw, yd, Xd, p, q, P, Q, s, k) -- negative
log-likelihood as a function of the *natural* `[beta; phi; theta; Phi;
Theta]`, for `_hessian_se`/`_opg_se` (Stage 6.5's own generic helpers,
reused directly)."
function _arimax_natural_objective(raw::AbstractVector, yd::Vector{Float64}, Xd::Matrix{Float64},
                                    p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer, k::Integer)
    T = eltype(raw)
    beta, ar, ma, yc = _arimax_unpack(raw, yd, Xd, p, q, P, Q, s, k)
    ssm = build_statespace(ar, ma)
    loglik, sigma2, v, F, converged = kalman_filter(ssm, yc)
    return converged ? -loglik : T(1e10)
end

"_arimax_loglik_contributions(raw, yd, Xd, p, q, P, Q, s, k) -- per-
observation contributions, for the OPG standard-error estimator."
function _arimax_loglik_contributions(raw::AbstractVector, yd::Vector{Float64}, Xd::Matrix{Float64},
                                       p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer, k::Integer)
    T = eltype(raw)
    beta, ar, ma, yc = _arimax_unpack(raw, yd, Xd, p, q, P, Q, s, k)
    ssm = build_statespace(ar, ma)
    loglik, sigma2, v, F, converged = kalman_filter(ssm, yc)
    n = length(yd)
    contribs = Vector{T}(undef, n)
    if !converged
        fill!(contribs, T(-1e10) / n)
        return contribs
    end
    for t in 1:n
        contribs[t] = -0.5 * (log(2π) + log(sigma2) + log(F[t]) + v[t]^2 / (sigma2 * F[t]))
    end
    return contribs
end

"_arimax_tvss_unpack(raw, p, q, P, Q, s, k, Q_beta_fixed, reparam) ->
(ar, ma, sigma2, Qbeta) -- `model=:tvss`'s parameter layout `[phi;
theta; Phi; Theta; log(sigma2); log(Q_beta)?]` (or the natural, untransformed
equivalent when `reparam=false`, for the Hessian/OPG evaluation point --
`_optimize` itself always calls this with `reparam=true`, matching every
other fit_* function's own natural-vs-transformed split). `ar`/`ma`
already combined via `combined_ar_ma` (seasonal-general, reduces exactly
to plain `(phi,theta)` when `P=Q=0`)."
function _arimax_tvss_unpack(raw::AbstractVector, p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer,
                              k::Integer, Q_beta_fixed::Union{Nothing,Vector{Float64}}, reparam::Bool)
    T = eltype(raw)
    i = 0
    phi = p > 0 ? raw[(i + 1):(i + p)] : T[]
    i += p
    theta = q > 0 ? raw[(i + 1):(i + q)] : T[]
    i += q
    Phi = P > 0 ? raw[(i + 1):(i + P)] : T[]
    i += P
    Theta = Q > 0 ? raw[(i + 1):(i + Q)] : T[]
    i += Q
    if reparam
        phi = p > 0 ? partrans(phi) : T[]
        theta = q > 0 ? partrans(theta) : T[]
        Phi = P > 0 ? partrans(Phi) : T[]
        Theta = Q > 0 ? partrans(Theta) : T[]
    end
    sigma2 = reparam ? exp(raw[i + 1]) : raw[i + 1]
    i += 1
    Qbeta = if Q_beta_fixed !== nothing
        T.(Q_beta_fixed)
    else
        qraw = raw[(i + 1):(i + k)]
        reparam ? exp.(qraw) : qraw
    end
    ar, ma = combined_ar_ma(phi, Phi, theta, Theta, s)
    return ar, ma, sigma2, Qbeta
end

"_arimax_tvss_system(ar, ma, sigma2, Qbeta, Xd, k) -> (tvssm, a0,
P_star0, r_arma, ok) -- builds the combined [ARMA companion block; k
beta states] `TimeVaryingSSM` described in `fit_arimax`'s own docstring
(`T` block-diagonal, `Z_t=[1,0,...,0,x_t']`, `R`/`Q` stacking the ARMA
disturbance loading against each beta's own process-noise loading, the
ARMA block's initial covariance scaled by `sigma2`, the beta block left
for the caller to mark diffuse). One shared builder rather than
duplicating this construction in the optimizer objective, the SE
evaluation, and the post-fit filtered-beta extraction, all three of
which need the exact same system."
function _arimax_tvss_system(ar::AbstractVector, ma::AbstractVector, sigma2, Qbeta::AbstractVector,
                              Xd::Matrix{Float64}, k::Integer)
    T = promote_type(typeof(sigma2), eltype(Qbeta), Float64)
    ssm_arma = build_statespace(ar, ma)
    r_arma = ssm_arma.r
    Q0_arma, sc_converged = stationary_cov(ssm_arma)
    if !sc_converged || !all(isfinite, Q0_arma) || Q0_arma[1, 1] <= 0
        return nothing, nothing, nothing, r_arma, false
    end

    r_full = r_arma + k
    n = size(Xd, 1)
    Tfull = zeros(T, r_full, r_full)
    Tfull[1:r_arma, 1:r_arma] .= ssm_arma.T
    for j in 1:k
        Tfull[r_arma + j, r_arma + j] = one(T)
    end

    Rfull = zeros(T, r_full, 1 + k)
    Rfull[1:r_arma, 1] .= ssm_arma.R
    for j in 1:k
        Rfull[r_arma + j, 1 + j] = one(T)
    end

    Qfull = zeros(T, 1 + k, 1 + k)
    Qfull[1, 1] = sigma2
    for j in 1:k
        Qfull[1 + j, 1 + j] = Qbeta[j]
    end

    Zseq = [reshape(vcat(one(T), zeros(T, r_arma - 1), T.(Xd[t, :])), 1, r_full) for t in 1:n]
    tvssm = TimeVaryingSSM{T}([Tfull], Zseq, [Rfull], [Qfull], [zeros(T, 1, 1)], r_full)

    a0 = zeros(T, r_full)
    P_star0 = zeros(T, r_full, r_full)
    P_star0[1:r_arma, 1:r_arma] .= sigma2 .* Q0_arma
    return tvssm, a0, P_star0, r_arma, true
end

"_arimax_tvss_loglik(raw, yd, Xd, p, q, P, Q, s, k, Q_beta_fixed,
reparam) -> (loglik, v, F, nobs_diffuse, converged) -- unpacks `raw`,
builds the combined system, and runs `kalman_filter_diffuse` with the
`k` beta states marked diffuse (`diffuse_idx = r_arma+1:r_arma+k`) --
the single call every `model=:tvss` code path (optimizer objective,
Hessian/OPG SE, post-fit extraction) goes through, so they can never
drift out of sync with each other."
function _arimax_tvss_loglik(raw::AbstractVector, yd::Vector{Float64}, Xd::Matrix{Float64},
                              p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer, k::Integer,
                              Q_beta_fixed::Union{Nothing,Vector{Float64}}, reparam::Bool)
    ar, ma, sigma2, Qbeta = _arimax_tvss_unpack(raw, p, q, P, Q, s, k, Q_beta_fixed, reparam)
    tvssm, a0, P_star0, r_arma, ok = _arimax_tvss_system(ar, ma, sigma2, Qbeta, Xd, k)
    ok || return (eltype(raw)(-Inf), eltype(raw)[], eltype(raw)[], 0, false)
    return kalman_filter_diffuse(tvssm, yd; diffuse_idx=collect((r_arma + 1):(r_arma + k)), a0=a0, P_star0=P_star0)
end

"_arimax_tvss_natural_objective(raw, yd, Xd, p, q, P, Q, s, k,
Q_beta_fixed) -- negative log-likelihood as a function of the *natural*
`[phi;theta;Phi;Theta;sigma2;Q_beta?]`, for `_hessian_se`. Uses
`-loglik` from `_arimax_tvss_loglik` directly -- **not** a separately
recomputed sum of per-observation terms -- so it can never numerically
drift from what `kalman_filter_diffuse` itself actually returns (that
mismatch is exactly what made the diffuse-phase exclusion easy to get
wrong: those initial observations' likelihood contribution has no
`v_t^2/F_t` term at all, see `kalman_filter_diffuse`'s own docstring)."
function _arimax_tvss_natural_objective(raw::AbstractVector, yd::Vector{Float64}, Xd::Matrix{Float64},
                                         p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer, k::Integer,
                                         Q_beta_fixed::Union{Nothing,Vector{Float64}})
    T = eltype(raw)
    loglik, = _arimax_tvss_loglik(raw, yd, Xd, p, q, P, Q, s, k, Q_beta_fixed, false)
    return isfinite(loglik) ? -loglik : T(1e10)
end

"_arimax_tvss_loglik_contributions(raw, yd, Xd, p, q, P, Q, s, k,
Q_beta_fixed, nobs_diffuse_fixed) -- per-observation log-likelihood
terms for the OPG standard-error estimator, over the `nd -
nobs_diffuse_fixed` **post**-diffuse-phase observations only (the
diffuse-phase ones have no comparable per-observation contribution --
same reasoning as `_arimax_tvss_natural_objective`). `nobs_diffuse_fixed`
is `nobs_diffuse` at the already-fitted optimum, held fixed rather than
recomputed at each perturbed point `ForwardDiff.jacobian` evaluates --
it's a discrete, data-driven quantity (confirmed `O(d)` fixed by Stage
8.2) that doesn't change under the infinitesimal perturbations
`ForwardDiff` uses, and a fixed-length output is required for the
Jacobian regardless."
function _arimax_tvss_loglik_contributions(raw::AbstractVector, yd::Vector{Float64}, Xd::Matrix{Float64},
                                            p::Integer, q::Integer, P::Integer, Q::Integer, s::Integer, k::Integer,
                                            Q_beta_fixed::Union{Nothing,Vector{Float64}}, nobs_diffuse_fixed::Integer)
    T = eltype(raw)
    n = length(yd)
    n_eff = n - nobs_diffuse_fixed
    loglik, v, F, nobs_diffuse, converged = _arimax_tvss_loglik(raw, yd, Xd, p, q, P, Q, s, k, Q_beta_fixed, false)
    if !converged
        return fill(T(-1e10) / n_eff, n_eff)
    end
    contribs = Vector{T}(undef, n_eff)
    for (idx, t) in enumerate((nobs_diffuse_fixed + 1):n)
        contribs[idx] = -0.5 * (log(2π) + log(F[t]) + v[t]^2 / F[t])
    end
    return contribs
end

"""
    fit_arimax(y, order::Tuple{Int,Int,Int}, exog;
               include_mean::Bool=true,
               model::Symbol=:mle, method::Symbol=:ml,
               Q_beta::Union{Nothing,Vector{Float64}}=nothing,
               se_type::Symbol=:hessian, optimizer_method::Symbol=:lbfgs,
               n_restarts::Integer=1, parallel::Bool=true,
               start_params::Union{Nothing,Vector{Float64}}=nothing) -> ArimaxModel

Fit ARIMA(p,d,q) with exogenous regressors `exog` (`n x k`, or a vector
for `k=1`).

**`model=:mle` and `model=:tvss` are different models, not different
computations of the same model -- verified directly, not assumed.**
Confirmed by running Python's actual `SARIMAX(...,
time_varying_regression=True, mle_regression=False,
use_exact_diffuse=True)` on the same data as the already-verified
`model=:mle` case: `beta` itself is **never** an optimized MLE
parameter under `model=:tvss` -- only its *process variance* (`Q_beta`)
is; `beta` becomes a genuinely latent, time-varying state, recovered
via the filtered state path (confirmed to actually evolve: `filtered
beta[:5] = [2.0, 1.778, 1.458, 1.925, 2.133]` on the reference series,
not a constant). Constraining `Q_beta` to exactly zero (removing
`beta`'s ability to drift) makes the two models' *point estimates*
converge closely (final filtered `beta ≈ 1.895` vs. `model=:mle`'s
`1.8948`), but their *likelihoods still don't match* (`-213.767` vs.
`-211.126`) -- both `loglik`s sum over the same `nd` observations (Stage
8.2's `kalman_filter_diffuse` sums the *full*, inclusive likelihood,
`nobs_diffuse` observations included, not excluded -- see its own
docstring), so the gap isn't an observation-count mismatch. It's that
`model=:tvss`'s likelihood genuinely is a different mathematical object
for the early observations: `beta` starts with **no prior information
at all** (Stage 8.2's own diffuse-initialization scenario), so its
likelihood contribution there is the diffuse marginal term
`-0.5*log(2π*F_infty,t)` (no `v_t^2/F_t` at all) -- fundamentally not
the same quantity as `model=:mle`'s ordinary fixed-parameter Gaussian
density for that same observation, even though both eventually sum over
identical `nd` totals. **Never compare the two models' `loglik`/`aic`/
`bic` directly** -- compare point estimates (e.g. a `:tvss(Q_beta=0)`
fit's final `beta_filtered` against `:mle`'s `beta`) for the reduction
property instead.

**`model=:mle` (default)**: `beta` an ordinary joint-MLE parameter,
`y .- X*beta` (both already differenced `d` times, matching Stage
6.6's own convention -- differencing distributes over the linear
`X*beta` term exactly, since `beta` is constant) fed to the unmodified
Stage 6.5 likelihood. Needs only Stage 6.5-6.7, nothing from Stage
8.1/8.2. `method` (`:ml`/`:css_ml`) passes straight through to the
underlying ARMA estimation, unchanged from Stage 6.5's own meaning --
deliberately a *different* keyword from `model`, not reused, since
Stage 6.5/6.7 already established `method` for a different choice
(`:ml` vs `:css_ml`) and `:ml`/`:mle` read as typos of each other.

**`model=:tvss`**: `beta` a latent state via Stage 8.1's `TimeVaryingSSM`
(the ARMA companion-form block and the `k` beta states stacked into one
combined state vector, `Z_t = [1,0,...,0, x_t']`) and Stage 8.2's
`kalman_filter_diffuse` (the beta block starts genuinely diffuse -- no
prior information at all -- while the ARMA block starts from its own
proper stationary distribution, scaled by the jointly-estimated
`sigma2`; the mixed proper-plus-diffuse `a0`/`P_star0` generalization
Stage 8.2 added anticipating exactly this use). The optimizer searches
`[ar; ma; sigma2; Q_beta]` (`sigma2` is a genuine free parameter here,
**not concentrated** the way Stage 6.5's own `fit_arma` does it --
confirmed directly: real `SARIMAX(time_varying_regression=True)`
reports `sigma2` as an explicit fitted parameter, `param_names =
['ar.L1', 'var.x', 'sigma2']`, three parameters, not two). `Q_beta`:
`nothing` (default) estimates it freely (matching Python's own default
-- genuine drift allowed if the data supports it, via `exp(raw)`,
ensuring non-negativity); an explicit `Q_beta` (e.g. `zeros(k)`) fixes
it instead of searching over it -- useful specifically for the
point-estimate-convergence property above, not a typical end-user
setting.

`include_mean=true` adds a constant-`1` column to `exog` internally,
the same joint-estimation treatment as any other regressor -- **but
silently forced to `false` whenever `d>0` or `D>0`**, matching Stage
6.6/6.7's own `include_mean` convention: a differenced constant column
is identically zero (`diff` of a constant is `0`), making its
coefficient completely unidentifiable, the same underlying reason a
plain ARMA mean is dropped after differencing.

Verified end-to-end against real R `stats::arima(xreg=)` and real
Python `statsmodels SARIMAX` (`model=:mle`) and real Python `SARIMAX(
time_varying_regression=True, use_exact_diffuse=True)` (`model=:tvss`,
both `Q_beta` freely estimated and fixed to zero) on the same shared
series -- see `handoff/stage-8.3-arimax-sarimax-handoff-v3.md` §1, §3.

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> d = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "arimax", "sarimax_exog_shared.csv"), ','; skipstart=1);

julia> y, x = d[:, 1], reshape(d[:, 2], :, 1);

julia> m = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle);

julia> m.converged
true

julia> round(m.beta[1], digits=4)
1.8948

julia> round(m.arma.ar[1], digits=4)
0.4768
```
"""
function fit_arimax(y, order::Tuple{Int,Int,Int}, exog;
                     include_mean::Bool=true,
                     model::Symbol=:mle,
                     method::Symbol=:ml,
                     Q_beta::Union{Nothing,Vector{Float64}}=nothing,
                     se_type::Symbol=:hessian,
                     optimizer_method::Symbol=:lbfgs,
                     n_restarts::Integer=1,
                     parallel::Bool=true,
                     start_params::Union{Nothing,Vector{Float64}}=nothing)
    _fit_arimax_core(y, order, (0, 0, 0, 1), exog, ArimaxModel;
                      include_mean=include_mean, model=model, method=method, Q_beta=Q_beta,
                      se_type=se_type, optimizer_method=optimizer_method,
                      n_restarts=n_restarts, parallel=parallel, start_params=start_params)
end

"""
    fit_sarimax(y, order::Tuple{Int,Int,Int}, seasonal_order::Tuple{Int,Int,Int,Int}, exog;
                kwargs...) -> SarimaxModel

Seasonal extension of [`fit_arimax`](@ref) -- every keyword argument,
both `model` choices, and every finding in `fit_arimax`'s own docstring
apply identically here; the ARMA part is combined via
`combined_ar_ma` (Stage 6.7) instead of plain `(phi,theta)`, the
same relationship [`fit_sarima`](@ref) has to `fit_arma`. `seasonal_order
= (0,0,0,anything)` reduces `fit_sarimax` exactly to `fit_arimax`
(verified directly, not assumed -- `combined_ar_ma` itself already has
this reduction property dual-verified from Stage 6.7).

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> d = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "arimax", "sarimax_exog_shared.csv"), ','; skipstart=1);

julia> y, x = d[:, 1], reshape(d[:, 2], :, 1);

julia> m = fit_sarimax(y, (1, 0, 0), (0, 0, 0, 4), x; include_mean=false, model=:mle);

julia> m.converged
true

julia> round(m.beta[1], digits=4)   # seasonal_order=(0,0,0,·) reduces exactly to fit_arimax
1.8948
```
"""
function fit_sarimax(y, order::Tuple{Int,Int,Int}, seasonal_order::Tuple{Int,Int,Int,Int}, exog;
                      include_mean::Bool=true,
                      model::Symbol=:mle,
                      method::Symbol=:ml,
                      Q_beta::Union{Nothing,Vector{Float64}}=nothing,
                      se_type::Symbol=:hessian,
                      optimizer_method::Symbol=:lbfgs,
                      n_restarts::Integer=1,
                      parallel::Bool=true,
                      start_params::Union{Nothing,Vector{Float64}}=nothing)
    _fit_arimax_core(y, order, seasonal_order, exog, SarimaxModel;
                      include_mean=include_mean, model=model, method=method, Q_beta=Q_beta,
                      se_type=se_type, optimizer_method=optimizer_method,
                      n_restarts=n_restarts, parallel=parallel, start_params=start_params)
end

"_fit_arimax_core -- shared implementation behind `fit_arimax`/
`fit_sarimax`; `constructor` is `ArimaxModel` or `SarimaxModel`, the two
structs differing only in whether they carry an `ArmaModel` or
`SarimaModel` and a `seasonal_order` field, per their own docstrings."
function _fit_arimax_core(y, order::Tuple{Int,Int,Int}, seasonal_order::Tuple{Int,Int,Int,Int}, exog,
                           constructor;
                           include_mean::Bool, model::Symbol, method::Symbol,
                           Q_beta::Union{Nothing,Vector{Float64}}, se_type::Symbol,
                           optimizer_method::Symbol, n_restarts::Integer, parallel::Bool,
                           start_params::Union{Nothing,Vector{Float64}})
    model in (:mle, :tvss) || throw(ArgumentError("model must be :mle or :tvss"))
    method in (:ml, :css_ml) || throw(ArgumentError("method must be :ml or :css_ml"))
    se_type in (:hessian, :opg) || throw(ArgumentError("se_type must be :hessian or :opg"))
    model == :mle && Q_beta !== nothing &&
        throw(ArgumentError("Q_beta is only valid for model=:tvss"))
    n_restarts >= 1 || throw(ArgumentError("n_restarts must be >= 1"))

    p, d, q = order
    P, D, Q, s = seasonal_order
    p >= 0 && d >= 0 && q >= 0 || throw(ArgumentError("order must be non-negative: got $order"))
    P >= 0 && D >= 0 && Q >= 0 || throw(ArgumentError("seasonal_order must be non-negative: got $seasonal_order"))
    s >= 1 || throw(ArgumentError("seasonal period s must be >= 1, got $s"))
    (P > 0 || D > 0 || Q > 0) && s < 2 &&
        throw(ArgumentError("seasonal_order needs period s >= 2 when P, D, or Q > 0 (got s=$s)"))

    yv = Float64.(collect(tsvalues(y)))
    n0 = length(yv)
    Xmat0 = exog isa AbstractMatrix ? Float64.(exog) : reshape(Float64.(collect(exog)), :, 1)
    size(Xmat0, 1) == n0 ||
        throw(ArgumentError("exog must have the same number of rows as y (got $(size(Xmat0, 1)) vs $n0)"))

    include_mean_effective = (d > 0 || D > 0) ? false : include_mean
    Xmat = include_mean_effective ? hcat(Xmat0, ones(n0)) : Xmat0
    k = size(Xmat, 2)
    k >= 1 || throw(ArgumentError("fit_arimax: exog must have at least one column (or include_mean=true with d=D=0)"))

    if Q_beta !== nothing
        length(Q_beta) == k ||
            throw(ArgumentError("Q_beta must have length k=$k (matching exog columns" *
                                 (include_mean_effective ? "+1 for include_mean" : "") * ")"))
        all(>=(0), Q_beta) || throw(ArgumentError("Q_beta entries must be non-negative"))
    end

    function difference(v::AbstractVector)
        vd = v
        D > 0 && (vd = diff(vd, s; differences=D))
        d > 0 && (vd = diff(vd, 1; differences=d))
        return vd
    end
    yd = difference(yv)
    Xd = reduce(hcat, [difference(Xmat[:, j]) for j in 1:k])
    nd = length(yd)

    if model == :mle
        nparam = k + p + q + P + Q
        nd > nparam || throw(ArgumentError("fit_arimax: not enough observations ($nd after differencing) for order $order, " *
                                            "seasonal_order $seasonal_order, $k exog columns"))

        objective_mle(raw::AbstractVector) = begin
            T = eltype(raw)
            beta = raw[1:k]
            i = k
            phi = p > 0 ? partrans(raw[(i + 1):(i + p)]) : T[]
            i += p
            theta = q > 0 ? partrans(raw[(i + 1):(i + q)]) : T[]
            i += q
            Phi = P > 0 ? partrans(raw[(i + 1):(i + P)]) : T[]
            i += P
            Theta = Q > 0 ? partrans(raw[(i + 1):(i + Q)]) : T[]
            ar, ma = combined_ar_ma(phi, Phi, theta, Theta, s)
            ssm = build_statespace(ar, ma)
            loglik, sigma2, v, F, converged = kalman_filter(ssm, yd .- Xd * beta)
            converged ? -loglik : T(1e10)
        end

        beta0 = Xd \ yd
        x0 = if start_params !== nothing
            length(start_params) == nparam || throw(ArgumentError("start_params must have length $nparam"))
            start_params
        elseif method == :css_ml
            resid0 = yd .- Xd * beta0
            css = _sarima_css_start_values(resid0, p, q, P, Q, s)
            vcat(beta0, css)
        else
            vcat(beta0, zeros(p + q + P + Q))
        end

        result = _optimize(objective_mle, x0; method=optimizer_method)
        n0v = result.minimizer
        beta_hat = n0v[1:k]
        i = k
        phi_hat = p > 0 ? partrans(n0v[(i + 1):(i + p)]) : Float64[]
        i += p
        theta_hat = q > 0 ? partrans(n0v[(i + 1):(i + q)]) : Float64[]
        i += q
        Phi_hat = P > 0 ? partrans(n0v[(i + 1):(i + P)]) : Float64[]
        i += P
        Theta_hat = Q > 0 ? partrans(n0v[(i + 1):(i + Q)]) : Float64[]
        ar_hat, ma_hat = combined_ar_ma(phi_hat, Phi_hat, theta_hat, Theta_hat, s)
        resid_hat = yd .- Xd * beta_hat
        ssm = build_statespace(ar_hat, ma_hat)
        loglik, sigma2, = kalman_filter(ssm, resid_hat)

        params_hat = vcat(beta_hat, phi_hat, theta_hat, Phi_hat, Theta_hat)
        se = se_type == :hessian ?
             _hessian_se(raw -> _arimax_natural_objective(raw, yd, Xd, p, q, P, Q, s, k), params_hat) :
             _opg_se(raw -> _arimax_loglik_contributions(raw, yd, Xd, p, q, P, Q, s, k), params_hat)

        kfull = nparam + 1  # +1 for sigma2
        aic = -2 * loglik + 2 * kfull
        bic = -2 * loglik + kfull * log(nd)

        arma_fields = if seasonal_order == (0, 0, 0, 1) || constructor === ArimaxModel
            ArmaModel(phi_hat, theta_hat, nothing, se[(k + 1):end], loglik, sigma2, aic, bic, nd,
                      (p, q), method, se_type, result.converged)
        else
            nothing
        end
        arma_seasonal = constructor === SarimaxModel ?
            SarimaModel(phi_hat, theta_hat, Phi_hat, Theta_hat, nothing, se[(k + 1):end], loglik, sigma2,
                        aic, bic, nd, order, seasonal_order, method, se_type, result.converged) : nothing

        arma_result = constructor === ArimaxModel ? arma_fields : arma_seasonal
        common = (model=:mle, method=method, beta=beta_hat, beta_filtered=nothing, Q_beta=nothing,
                  se=se, loglik=loglik, aic=aic, bic=bic, nobs=nd, nobs_diffuse=nothing,
                  converged=result.converged)
        return constructor === ArimaxModel ?
               ArimaxModel(common.model, common.method, common.beta, common.beta_filtered, common.Q_beta,
                            arma_result, d, yv, Xmat, common.se, common.loglik, common.aic, common.bic,
                            common.nobs, common.nobs_diffuse, common.converged) :
               SarimaxModel(common.model, common.method, common.beta, common.beta_filtered, common.Q_beta,
                             arma_result, seasonal_order, yv, Xmat, common.se, common.loglik, common.aic,
                             common.bic, common.nobs, common.nobs_diffuse, common.converged)
    else
        # model = :tvss
        nq = Q_beta === nothing ? k : 0
        nparam = p + q + P + Q + 1 + nq
        nd > nparam || throw(ArgumentError("fit_arimax: not enough observations ($nd after differencing) for order $order, " *
                                            "seasonal_order $seasonal_order, $k exog columns, model=:tvss"))

        function objective_tvss(raw::AbstractVector)
            T = eltype(raw)
            loglik, v, F, nobs_diffuse, converged = _arimax_tvss_loglik(raw, yd, Xd, p, q, P, Q, s, k, Q_beta, true)
            converged && isfinite(loglik) ? -loglik : T(1e10)
        end

        x0 = if start_params !== nothing
            length(start_params) == nparam || throw(ArgumentError("start_params must have length $nparam"))
            start_params
        else
            vcat(zeros(p + q + P + Q), log(max(var(yd), 1e-4)), fill(log(1e-4), nq))
        end

        starts = Vector{Vector{Float64}}(undef, n_restarts)
        starts[1] = x0
        for i in 2:n_restarts
            rng = Random.MersenneTwister(3000 + i)
            starts[i] = x0 .+ 0.1 .* randn(rng, length(x0))
        end

        results = Vector{Any}(undef, n_restarts)
        use_threads = parallel && Threads.nthreads() > 1 && n_restarts >= 4
        if use_threads
            Threads.@threads for i in 1:n_restarts
                results[i] = try
                    _optimize(objective_tvss, starts[i]; method=optimizer_method)
                catch
                    nothing
                end
            end
        else
            for i in 1:n_restarts
                results[i] = try
                    _optimize(objective_tvss, starts[i]; method=optimizer_method)
                catch
                    nothing
                end
            end
        end
        valid = [(r, objective_tvss(r.minimizer)) for r in results if r !== nothing]
        isempty(valid) && throw(ArgumentError("fit_arimax: optimizer failed from every starting point (model=:tvss)"))
        converged_valid = filter(rv -> rv[1].converged, valid)
        best = isempty(converged_valid) ? valid[argmin(last.(valid))] : converged_valid[argmin(last.(converged_valid))]
        result, best_obj = best
        raw_hat = result.minimizer

        i = 0
        phi_hat = p > 0 ? partrans(raw_hat[(i + 1):(i + p)]) : Float64[]
        i += p
        theta_hat = q > 0 ? partrans(raw_hat[(i + 1):(i + q)]) : Float64[]
        i += q
        Phi_hat = P > 0 ? partrans(raw_hat[(i + 1):(i + P)]) : Float64[]
        i += P
        Theta_hat = Q > 0 ? partrans(raw_hat[(i + 1):(i + Q)]) : Float64[]
        i += Q
        sigma2_hat = exp(raw_hat[i + 1])
        i += 1
        Qbeta_hat = Q_beta === nothing ? exp.(raw_hat[(i + 1):(i + k)]) : Q_beta

        loglik, v_hat, F_hat, nobs_diffuse, conv_hat = _arimax_tvss_loglik(
            raw_hat, yd, Xd, p, q, P, Q, s, k, Q_beta, true)

        # rebuild the fitted system (via the shared builder, so it's guaranteed
        # identical to what `loglik`/`nobs_diffuse` above were computed from)
        # to additionally extract the filtered beta path.
        ar_hat, ma_hat = combined_ar_ma(phi_hat, Phi_hat, theta_hat, Theta_hat, s)
        tvssm, a0, P_star0, r_arma, ok = _arimax_tvss_system(ar_hat, ma_hat, sigma2_hat, Qbeta_hat, Xd, k)
        ok || throw(ArgumentError("fit_arimax: fitted model=:tvss system has a non-stationary ARMA block"))
        filtered_states = _kalman_filter_diffuse_filtered_states(
            tvssm, yd; diffuse_idx=collect((r_arma + 1):(r_arma + k)), a0=a0, P_star0=P_star0)
        beta_filtered = filtered_states[(r_arma + 1):(r_arma + k), :]

        natural_hat = vcat(phi_hat, theta_hat, Phi_hat, Theta_hat, sigma2_hat,
                            Q_beta === nothing ? Qbeta_hat : Float64[])
        se_full = se_type == :hessian ?
                  _hessian_se(nat -> _arimax_tvss_natural_objective(nat, yd, Xd, p, q, P, Q, s, k, Q_beta),
                              natural_hat) :
                  _opg_se(nat -> _arimax_tvss_loglik_contributions(nat, yd, Xd, p, q, P, Q, s, k, Q_beta, nobs_diffuse),
                          natural_hat)

        kfull = nparam
        aic = -2 * loglik + 2 * kfull
        bic = -2 * loglik + kfull * log(nd)

        arma_placeholder_se = se_full[1:(p + q + P + Q)]
        arma_result = constructor === ArimaxModel ?
            ArmaModel(phi_hat, theta_hat, nothing, arma_placeholder_se, loglik, sigma2_hat, aic, bic,
                      nd, (p, q), method, se_type, result.converged) :
            SarimaModel(phi_hat, theta_hat, Phi_hat, Theta_hat, nothing, arma_placeholder_se, loglik,
                        sigma2_hat, aic, bic, nd, order, seasonal_order, method, se_type,
                        result.converged)

        return constructor === ArimaxModel ?
               ArimaxModel(:tvss, method, nothing, beta_filtered, Qbeta_hat, arma_result, d, yv, Xmat,
                            se_full, loglik, aic, bic, nd, nobs_diffuse, result.converged) :
               SarimaxModel(:tvss, method, nothing, beta_filtered, Qbeta_hat, arma_result, seasonal_order,
                             yv, Xmat, se_full, loglik, aic, bic, nd, nobs_diffuse, result.converged)
    end
end

"_kalman_filter_diffuse_filtered_states(ssm, y; diffuse_idx, a0, P_star0)
-> states::Matrix{Float64} (`r x n`) -- like `kalman_filter_diffuse`, but
returns the full matrix of *filtered* states `a_t|t` (not just
`v`/`F`/`loglik`), needed for `model=:tvss`'s `beta_filtered` output. A
separate function rather than extending `kalman_filter_diffuse`'s own
return signature, since no other caller needs the full state path and
Stage 8.1/8.2's own established convention is to keep the filter's
return shape minimal (matching `GaussianSSM`'s own `kalman_filter`,
which doesn't return the state path either). Runs the identical
recursion `kalman_filter_diffuse` does (same branch conditions, same
`L0`/`L1`/`K0`/`K1` updates) -- only called on an already-fitted,
already-`converged`-checked system, so it doesn't need its own
divergence guard."
function _kalman_filter_diffuse_filtered_states(ssm::TimeVaryingSSM, y::AbstractVector{<:Real};
                                                 diffuse_idx::AbstractVector{<:Integer},
                                                 a0::AbstractVector{<:Real}, P_star0::AbstractMatrix{<:Real})
    n = length(y)
    r = ssm.r
    a = Vector{Float64}(a0)
    P = Matrix{Float64}(P_star0)
    Pinf = zeros(r, r)
    for i in diffuse_idx
        Pinf[i, i] = 1.0
    end
    diffuse_active = !isempty(diffuse_idx)
    states = Matrix{Float64}(undef, r, n)
    Ir = Matrix{Float64}(I, r, r)

    for t in 1:n
        Tt = _tv_at(ssm.T, t)
        z = vec(_tv_at(ssm.Z, t))
        Rt = _tv_at(ssm.R, t)
        Qt = _tv_at(ssm.Q, t)
        Ht = _tv_at(ssm.H, t)[1, 1]

        vt = y[t] - dot(z, a)
        Fstar = max(dot(z, P * z) + Ht, 0.0)
        Finf = diffuse_active ? max(dot(z, Pinf * z), 0.0) : 0.0

        local a_filt, P_filt, Pinf_filt
        if diffuse_active && Finf > _TOLERANCE_DIFFUSE
            F1 = 1 / Finf
            M = P * z
            Minf = Pinf * z
            K0 = Minf .* F1
            K1 = M .* F1 .- Minf .* (Fstar * F1^2)
            L0 = Ir - K0 * z'
            L1 = -K1 * z'
            a_filt = a + K0 .* vt
            P_filt = P * L0' + Pinf * L1'
            Pinf_filt = Pinf * L0'
        elseif Fstar > _TOLERANCE_DIFFUSE
            K0 = P * z ./ Fstar
            L0 = Ir - K0 * z'
            a_filt = a + K0 .* vt
            P_filt = P * L0'
            Pinf_filt = Pinf
        else
            a_filt = a
            P_filt = P
            Pinf_filt = Pinf
        end

        states[:, t] = a_filt
        a = Tt * a_filt
        P = Tt * P_filt * Tt' + Rt * Qt * Rt'
        if diffuse_active
            Pinf = Tt * Pinf_filt * Tt'
            diffuse_active = sum(abs2, Pinf) > _TOLERANCE_DIFFUSE
        end
    end
    return states
end

function Base.show(io::IO, m::Union{ArimaxModel,SarimaxModel})
    phi, theta, Phi, Theta = _arma_blocks(m.arma)
    p, q = length(phi), length(theta)
    label = m isa SarimaxModel ? "SARIMAX" : "ARIMAX"
    k = m.model == :mle ? length(m.beta) : length(m.Q_beta)
    println(io, label, "(", p, ",", q, "), ", k, " exog, model=", m.model,
            ", n=", m.nobs, m.nobs_diffuse === nothing ? "" : " (nobs_diffuse=$(m.nobs_diffuse))")
    if m.model == :mle
        println(io, "  beta: ", round.(m.beta, digits=4))
    else
        println(io, "  Q_beta: ", round.(m.Q_beta, digits=6))
        println(io, "  final filtered beta: ", round.(m.beta_filtered[:, end], digits=4))
    end
    print(io, "  ar: ", round.(phi, digits=4), "  ma: ", round.(theta, digits=4))
    isempty(Phi) && isempty(Theta) || print(io, "  sar: ", round.(Phi, digits=4), "  sma: ", round.(Theta, digits=4))
    println(io)
    print(io, "Log-likelihood: ", round(m.loglik, digits=2),
          "   AIC: ", round(m.aic, digits=2), "   BIC: ", round(m.bic, digits=2))
    m.converged || print(io, "\nWARNING: optimizer did not converge")
end
