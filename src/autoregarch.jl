export fit_autoreg_garch, AutoregGarchModel

"""
    AutoregGarchModel <: UnivariateModel

Result of [`fit_autoreg_garch`](@ref): SAS `PROC AUTOREG`'s regression-
with-AR(m)-errors model, optionally with a jointly-fitted GARCH variance.

- `beta`: regression coefficients (`k` entries, `k` = `exog`'s columns
  plus an intercept if `include_mean=true` -- same convention as
  [`fit_arimax`](@ref)'s own `beta`).
- `phi`: AR(m) error coefficients (`nu_t = y_t - X_t·beta`, `nu_t =
  phi_1·nu_{t-1} + ... + phi_m·nu_{t-m} + e_t`).
- `garch`: `nothing` when `garch_order=nothing` (plain, constant-variance
  AR errors -- this is exactly [`fit_arimax`](@ref)'s `model=:mle`, see
  [`fit_autoreg_garch`](@ref)'s own docstring); otherwise the fitted
  variance-equation coefficients, packaged as a [`GarchModel`](@ref) for
  reuse of its own display/`coef` conventions -- **its own `loglik`/
  `aic`/`bic`/`nobs`/`se`/`cov_type` are not independently meaningful**
  (they mirror this struct's own top-level fields/`se_type` choice,
  since `beta`/`phi`/the variance equation are all fit *jointly*, not in
  two separate stages) -- use `AutoregGarchModel`'s own top-level fields.
- `loglik`/`aic`/`bic`/`nobs`/`se`/`converged`: authoritative for the
  overall model. `se` order: `[beta; phi]` (`garch_order=nothing`) or
  `[beta; phi; omega; alpha; garch_beta]` (matching `coef`'s own order).
"""
struct AutoregGarchModel <: UnivariateModel
    beta::Vector{Float64}
    phi::Vector{Float64}
    garch::Union{Nothing,GarchModel}
    loglik::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    se::Vector{Float64}
    converged::Bool
end

StatsAPI.loglikelihood(m::AutoregGarchModel) = m.loglik
StatsAPI.aic(m::AutoregGarchModel) = m.aic
StatsAPI.bic(m::AutoregGarchModel) = m.bic
StatsAPI.nobs(m::AutoregGarchModel) = m.nobs
StatsAPI.coef(m::AutoregGarchModel) = m.garch === nothing ? vcat(m.beta, m.phi) :
                                       vcat(m.beta, m.phi, m.garch.omega, m.garch.alpha, m.garch.beta)

"""
_ar_garch_loglik(beta, phi, omega, alpha, ggbeta, yv, Xmat, m) ->
(loglik, contribs, h, ok) -- the combined AR(m)-errors + GARCH(p,q)
likelihood (handoff section 1), generalized from the handoff's own
verified `m=1` derivation to general `m` via this project's *existing*
Kalman-filter stationary-covariance machinery, not new approximate math:

The first `m` residuals `nu_{1:m}` have no available lags to form an
ordinary AR residual from -- treating them as literally drawn from the
AR(m) process's own exact stationary joint distribution (mean 0,
covariance `h_bar .* Sigma_m`, `h_bar = omega/(1-sum(alpha)-sum(ggbeta))`
GARCH's own unconditional variance, `Sigma_m = stationary_cov` of the
*same* `build_statespace(phi, [])` companion form
[`fit_arma`](@ref)/[`fit_arimax`](@ref) already use for any stationary
AR/ARMA polynomial) gives their *exact* joint log-density as one block
contribution -- this is the literal generalization of "full ML needs the
proper stationary variance for the first residual, not a dropped or
conditioned-away term" (handoff section 1) from a single value to an
`m`-dimensional one. **Confirmed to reduce exactly to the handoff's own
verified `m=1` formula**: for `m=1`, `Sigma_1 = [1/(1-phi^2)]` (the
plain AR(1) stationary variance under unit innovation variance, this
project's own `stationary_cov` convention -- see `to_time_varying`'s
docstring), so `h_bar .* Sigma_1 = h_bar/(1-phi^2)`, and the resulting
univariate presample term is algebraically identical to the handoff's
own `e[1]=nu[1]`, `h[1]=(omega/(1-alpha-garch_beta))/(1-phi^2)`
construction, verified to `7.7e-9` against Stage 8.3's own already-
verified plain-ARIMAX likelihood in the degenerate (`alpha=garch_beta=0`)
case.

For `t=m+1,...,n`: the ordinary AR(m) residual `e_t = nu_t -
phi_1·nu_{t-1} - ... - phi_m·nu_{t-m}` and GARCH(p,q) variance
recursion (reusing `_garch_sigma2_path`'s own formula, inlined here
since the presample block `h[1:m]` needs to seed it with `h_bar`
directly rather than an empirical `_garch_backcast`, `_garch_backcast`
being specifically an approximation for when no better information
exists -- here the exact `h_bar` already is that better information).
"""
function _ar_garch_loglik(beta::AbstractVector, phi::AbstractVector, omega, alpha::AbstractVector,
                           ggbeta::AbstractVector, yv::Vector{Float64}, Xmat::Matrix{Float64}, m::Integer)
    T = promote_type(eltype(beta), eltype(phi), typeof(omega), eltype(alpha), eltype(ggbeta), Float64)
    n = length(yv)
    nu = yv .- Xmat * beta

    persistence = sum(alpha) + sum(ggbeta)
    if !(0 < omega) || !(0 <= persistence < 1)
        return T(-1e10), T[], T[], false
    end
    h_bar = omega / (1 - persistence)

    ssm = build_statespace(phi, T[])
    Sigma_m, sc_ok = stationary_cov(ssm)
    if !sc_ok || !all(isfinite, Sigma_m)
        return T(-1e10), T[], T[], false
    end

    e = Vector{T}(undef, n)
    h = Vector{T}(undef, n)
    for t in 1:m
        e[t] = nu[t]
        h[t] = h_bar
    end
    p, q = length(alpha), length(ggbeta)
    for t in (m + 1):n
        e[t] = nu[t] - dot(phi, view(nu, (t - 1):-1:(t - m)))
        s2 = omega
        for j in 1:p
            s2 += alpha[j] * (t - j >= 1 ? e[t - j]^2 : h_bar)
        end
        for kk in 1:q
            s2 += ggbeta[kk] * (t - kk >= 1 ? h[t - kk] : h_bar)
        end
        h[t] = s2
    end

    nu_pre = view(nu, 1:m)
    Hm = h_bar .* Sigma_m
    Hm_chol = try
        cholesky(Symmetric(Hm))
    catch
        return T(-1e10), T[], T[], false
    end
    presample_ll = -0.5 * (m * log(2 * pi) + logdet(Hm_chol) + dot(nu_pre, Hm_chol \ nu_pre))
    main_contribs = @. -0.5 * (log(2 * pi) + log($view(h, (m + 1):n)) + $view(e, (m + 1):n)^2 / $view(h, (m + 1):n))

    loglik = presample_ll + sum(main_contribs)
    isfinite(loglik) || return T(-1e10), T[], T[], false
    return loglik, vcat(presample_ll, main_contribs), h, true
end

"""
    fit_autoreg_garch(y, m::Integer, exog;
                       garch_order::Union{Nothing,Tuple{Int,Int}}=nothing,
                       include_mean::Bool=true,
                       se_type::Symbol=:hessian, optimizer_method::Symbol=:lbfgs,
                       n_restarts::Integer=1, parallel::Bool=true,
                       start_params::Union{Nothing,Vector{Float64}}=nothing) -> AutoregGarchModel

SAS `PROC AUTOREG`'s regression-with-autocorrelated-errors model,
optionally with a jointly-fitted GARCH(p,q) variance:
```
Y_t = X_t·beta + nu_t
nu_t = phi_1·nu_{t-1} + ... + phi_m·nu_{t-m} + e_t
```

**`garch_order=nothing` (default, plain AR(m) errors, constant
variance) is exactly [`fit_arimax`](@ref)'s `model=:mle` construction --
literally delegates to it, not a reimplementation**: substituting
`nu_t = Y_t - X_t·beta` into the AR(m) recursion above gives `Y_t -
X_t·beta = phi_1·(Y_{t-1} - X_{t-1}·beta) + ... + e_t`, i.e.
`fit_arimax(y, (m,0,0), exog; model=:mle)` exactly (verified directly:
bit-identical `beta`/`loglik` to calling `fit_arimax` yourself). This is
the "full ML, no GARCH" tier of `PROC AUTOREG` -- already built as of
Stage 8.3, not new work here.

**`garch_order=(p,q)` is the genuinely new case**: `beta`, `phi`, and
the GARCH(p,q) variance-equation coefficients (`omega`/`alpha`/
`garch_beta`, [`fit_garch`](@ref)'s own `p`=ARCH/`q`=GARCH naming, plain
`:garch` only -- GJR/EGARCH variants combined with AR errors are outside
this stage's verified scope) are optimized **jointly**, in one
`_optimize` call, not two sequential fits. No direct software reference
exists for this exact combination (checked directly: Python's
`arch.univariate.ARX` combines `Y`'s own lags with `X` and a
`volatility=` process instead -- [`arx`](@ref)'s own framing, not
`PROC AUTOREG`'s "regression, then AR-structured residual" framing;
these are different models). Verified instead via the "reduces to an
already-trusted case" methodology used throughout this project when no
direct reference exists (Stage 8.1's exact reduction to
`GaussianSSM`-time-invariant, Stage 8.3's `Q_beta=0` convergence check):
constraining `alpha=garch_beta=0` (degenerate to constant variance)
reduces this construction to Stage 8.3's already-verified plain
likelihood, confirmed to `7.7e-9` -- see `_ar_garch_loglik`'s own
docstring for exactly how the general-`m` presample-block likelihood is
built and why it reduces to the handoff's own hand-verified `m=1`
formula exactly.

`include_mean=true` (default) adds a constant-`1` column to `exog`,
identically to [`fit_arimax`](@ref)'s own convention (`beta` gains one
more entry) -- no `d`/differencing concept exists in this model at all
(unlike `fit_arimax`), so unlike there, `include_mean` is never silently
forced off.

`se_type=:hessian`/`:opg` -- same meaning as every other `fit_*`
function in this project (Stage 6.5's `_hessian_se`/`_opg_se`, reused
directly). For `:opg`, the presample block (all `m` observations
together) contributes **one** row to the per-contribution Jacobian, not
`m` separate ones -- it genuinely is one joint block of information
about the score, not `m` independently-scored observations.

`nobs` is the full series length `n` -- the presample block's own exact
joint likelihood already accounts for all `m` of its observations (no
`nobs = n - m` convention needed, unlike differencing-based `d`/`D` --
the same "sum every observation's genuine likelihood contribution,
nothing burned" principle Stage 8.2's `kalman_filter_diffuse` was
corrected to follow during Stage 8.3, here true from the start since
this construction never had a burn-based first attempt).

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> d = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "arimax", "sarimax_exog_shared.csv"), ','; skipstart=1);

julia> y, x = d[:, 1], reshape(d[:, 2], :, 1);

julia> m1 = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=nothing);

julia> m2 = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle);

julia> isapprox(m1.beta, m2.beta; atol=1e-8)
true

julia> isapprox(m1.loglik, m2.loglik; atol=1e-8)
true
```
"""
function fit_autoreg_garch(y, m::Integer, exog;
                            garch_order::Union{Nothing,Tuple{Int,Int}}=nothing,
                            include_mean::Bool=true,
                            se_type::Symbol=:hessian,
                            optimizer_method::Symbol=:lbfgs,
                            n_restarts::Integer=1,
                            parallel::Bool=true,
                            start_params::Union{Nothing,Vector{Float64}}=nothing)
    m >= 1 || throw(ArgumentError("m (AR-error order) must be >= 1"))
    se_type in (:hessian, :opg) || throw(ArgumentError("se_type must be :hessian or :opg"))
    n_restarts >= 1 || throw(ArgumentError("n_restarts must be >= 1"))

    if garch_order === nothing
        am = fit_arimax(y, (m, 0, 0), exog; include_mean=include_mean, model=:mle,
                         se_type=se_type, optimizer_method=optimizer_method,
                         n_restarts=n_restarts, parallel=parallel, start_params=start_params)
        return AutoregGarchModel(am.beta, am.arma.ar, nothing, am.loglik, am.aic, am.bic, am.nobs,
                                  am.se, am.converged)
    end

    p, q = garch_order
    p >= 1 || throw(ArgumentError("garch_order[1] (ARCH order) must be >= 1"))
    q >= 0 || throw(ArgumentError("garch_order[2] (GARCH order) must be >= 0"))

    yv = Float64.(collect(tsvalues(y)))
    n0 = length(yv)
    Xmat0 = exog isa AbstractMatrix ? Float64.(exog) : reshape(Float64.(collect(exog)), :, 1)
    size(Xmat0, 1) == n0 ||
        throw(ArgumentError("exog must have the same number of rows as y (got $(size(Xmat0, 1)) vs $n0)"))
    Xmat = include_mean ? hcat(Xmat0, ones(n0)) : Xmat0
    k = size(Xmat, 2)
    k >= 1 || throw(ArgumentError("fit_autoreg_garch: exog must have at least one column (or include_mean=true)"))

    nparam = k + m + 1 + p + q
    n0 > nparam || throw(ArgumentError("fit_autoreg_garch: not enough observations ($n0) for m=$m, " *
                                        "garch_order=$garch_order, $k exog columns"))

    function unpack(raw::AbstractVector, reparam::Bool)
        T = eltype(raw)
        beta = raw[1:k]
        phi = reparam ? partrans(raw[(k + 1):(k + m)]) : raw[(k + 1):(k + m)]
        i = k + m
        omega = reparam ? exp(raw[i + 1]) : raw[i + 1]
        i += 1
        if reparam
            prop = _stationary_softmax(raw[(i + 1):(i + p + q)])
            alpha = prop[1:p]
            ggbeta = q > 0 ? prop[(p + 1):(p + q)] : T[]
        else
            alpha = raw[(i + 1):(i + p)]
            ggbeta = q > 0 ? raw[(i + p + 1):(i + p + q)] : T[]
        end
        return beta, phi, omega, alpha, ggbeta
    end

    function objective(raw::AbstractVector)
        T = eltype(raw)
        beta, phi, omega, alpha, ggbeta = unpack(raw, true)
        loglik, = _ar_garch_loglik(beta, phi, omega, alpha, ggbeta, yv, Xmat, m)
        isfinite(loglik) ? -loglik : T(1e10)
    end

    beta0 = Xmat \ yv
    nu0 = yv .- Xmat * beta0
    x0 = if start_params !== nothing
        length(start_params) == nparam || throw(ArgumentError("start_params must have length $nparam"))
        start_params
    else
        garch_raw = _garch_start_raw(p, q, false, 0.0, var(nu0), q > 0 ? 0.05 : 0.3, q > 0 ? 0.90 : 0.0)
        vcat(beta0, zeros(m), garch_raw)
    end

    starts = Vector{Vector{Float64}}(undef, n_restarts)
    starts[1] = x0
    for i in 2:n_restarts
        rng = Random.MersenneTwister(4000 + i)
        starts[i] = x0 .+ 0.1 .* randn(rng, length(x0))
    end

    results = Vector{Any}(undef, n_restarts)
    use_threads = parallel && Threads.nthreads() > 1 && n_restarts >= 4
    if use_threads
        Threads.@threads for i in 1:n_restarts
            results[i] = try
                _optimize(objective, starts[i]; method=optimizer_method)
            catch
                nothing
            end
        end
    else
        for i in 1:n_restarts
            results[i] = try
                _optimize(objective, starts[i]; method=optimizer_method)
            catch
                nothing
            end
        end
    end
    valid = [(r, objective(r.minimizer)) for r in results if r !== nothing]
    isempty(valid) && throw(ArgumentError("fit_autoreg_garch: optimizer failed from every starting point"))
    converged_valid = filter(rv -> rv[1].converged, valid)
    best = isempty(converged_valid) ? valid[argmin(last.(valid))] : converged_valid[argmin(last.(converged_valid))]
    result, best_obj = best
    raw_hat = result.minimizer

    beta_hat, phi_hat, omega_hat, alpha_hat, ggbeta_hat = unpack(raw_hat, true)
    loglik, _, h_hat, ok = _ar_garch_loglik(beta_hat, phi_hat, omega_hat, alpha_hat, ggbeta_hat, yv, Xmat, m)
    ok || throw(ArgumentError("fit_autoreg_garch: fitted parameters do not give a valid model"))

    nu_hat = yv .- Xmat * beta_hat
    e_hat = Vector{Float64}(undef, n0)
    e_hat[1:m] .= nu_hat[1:m]
    for t in (m + 1):n0
        e_hat[t] = nu_hat[t] - dot(phi_hat, view(nu_hat, (t - 1):-1:(t - m)))
    end

    params_hat = vcat(beta_hat, phi_hat, omega_hat, alpha_hat, ggbeta_hat)
    se = se_type == :hessian ?
         _hessian_se(nat -> begin
                         T = eltype(nat)
                         beta, phi, omega, alpha, ggbeta = unpack(nat, false)
                         loglik_n, = _ar_garch_loglik(beta, phi, omega, alpha, ggbeta, yv, Xmat, m)
                         isfinite(loglik_n) ? -loglik_n : T(1e10)
                     end, params_hat) :
         _opg_se(nat -> begin
                     T = eltype(nat)
                     beta, phi, omega, alpha, ggbeta = unpack(nat, false)
                     _, contribs, = _ar_garch_loglik(beta, phi, omega, alpha, ggbeta, yv, Xmat, m)
                     isempty(contribs) ? fill(T(-1e10) / (n0 - m + 1), n0 - m + 1) : contribs
                 end, params_hat)

    aic = -2 * loglik + 2 * nparam
    bic = -2 * loglik + nparam * log(n0)

    garch_se = se[(k + m + 1):end]
    garch = GarchModel(:garch, omega_hat, alpha_hat, nothing, ggbeta_hat, :zero, nothing, h_hat, e_hat,
                        garch_se, loglik, aic, bic, n0, p, q, :classic, result.converged)

    return AutoregGarchModel(beta_hat, phi_hat, garch, loglik, aic, bic, n0, se, result.converged)
end

function Base.show(io::IO, m::AutoregGarchModel)
    println(io, "AUTOREG(", length(m.phi), " AR-errors", m.garch === nothing ? "" :
                ", GARCH($(m.garch.p),$(m.garch.q)) variance", "), ", length(m.beta), " exog, n=", m.nobs)
    println(io, "  beta: ", round.(m.beta, digits=4))
    println(io, "  phi: ", round.(m.phi, digits=4))
    if m.garch !== nothing
        println(io, "  omega: ", round(m.garch.omega, digits=4), "  alpha: ", round.(m.garch.alpha, digits=4),
                "  garch_beta: ", round.(m.garch.beta, digits=4))
    end
    print(io, "Log-likelihood: ", round(m.loglik, digits=2),
          "   AIC: ", round(m.aic, digits=2), "   BIC: ", round(m.bic, digits=2))
    m.converged || print(io, "\nWARNING: optimizer did not converge")
end
