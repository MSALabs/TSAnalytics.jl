export fit_vecm, VECMModel, var_rep

# ---------------------------------------------------------------------------
# Stage 10.6 -- the vector error-correction model.
#
# Primary sources: Johansen (1988, 1991) for the maximum-likelihood
# estimator (reduced-rank regression, shared with johansen_test), Engle &
# Granger (1987) for the two-step estimator, and Lütkepohl (2005), §7.2-7.3,
# for the standard errors of the cointegrating vectors and the levels-VAR
# representation. urca::cajorls + vars::vec2var, statsmodels' VECM and
# tsDyn::VECM are the references; once the deterministic specification is
# matched, all three agree to machine precision.
# ---------------------------------------------------------------------------

"""
    VECMModel <: TimeSeriesModel

A fitted vector error-correction model, returned by [`fit_vecm`](@ref):

    ΔY_t = α β' Ỹ_{t-1} + Γ_1 ΔY_{t-1} + … + Γ_k ΔY_{t-k} + (unrestricted deterministic) + u_t

with `Ỹ_{t-1}` the lagged levels, augmented by a deterministic term when
one is restricted to the cointegrating space.

- `alpha` (`K × r`), `beta` (`K₁ × r`, `K₁ = K + 1` when a term is
  restricted; its coefficient is the last row). `beta` is stored with its
  leading `r × r` block the identity — `statsmodels`' and `cajorls`'
  normalization — and `alpha` matches it, so `alpha * beta'` is the
  impact matrix. [`normalize_beta`](@ref) gives other forms.
- `gamma`: `lags` matrices, each `K × K`; `det_coef`: `K × n` for the
  unrestricted deterministic terms (constant, then trend).
- `sigma_df` (divisor `T − q`, `q` regressors per equation) and
  `sigma_mle` (divisor `T`).
- `stderr_alpha`, `stderr_gamma` (`K × K·lags`), `stderr_det`: from the
  regression given `β`, with the ML `Σ`, as `statsmodels` computes them.
  `stderr_beta`: Lütkepohl's (2005, Prop. 7.6) asymptotic standard
  errors of the free elements of `β`, zero on the normalized block;
  `nothing` for `method = :twostep`, whose static regression gives none
  that are valid.
- `resid` (`T × K`), `loglik`, `nobs` (`T`), `lags` (**differenced**
  lags), `rank`, `deterministic`, `method`.
"""
struct VECMModel <: TimeSeriesModel
    Y::Matrix{Float64}
    names::Vector{Symbol}
    lags::Int
    rank::Int
    deterministic::Symbol
    method::Symbol
    alpha::Matrix{Float64}
    beta::Matrix{Float64}
    gamma::Vector{Matrix{Float64}}
    det_coef::Matrix{Float64}
    sigma_df::Matrix{Float64}
    sigma_mle::Matrix{Float64}
    stderr_alpha::Matrix{Float64}
    stderr_beta::Union{Nothing,Matrix{Float64}}
    stderr_gamma::Matrix{Float64}
    stderr_det::Matrix{Float64}
    resid::Matrix{Float64}
    loglik::Float64
    nobs::Int
end

"""
    _vecm_design(Y, lags, deterministic) -> (W, X0, X1, T)

The pieces of the error-correction regression, aligned on `t = lags+2 … n`:
`W` the short-run regressors (`lags` lagged differences, then the
unrestricted deterministic terms), `X0 = ΔY_t`, and `X1` the lagged levels
augmented by a restricted term. The same layout `_johansen_rrr` builds,
so the two cannot drift apart.
"""
function _vecm_design(Y::Matrix{Float64}, lags::Int, deterministic::Symbol)
    n, K = size(Y)
    dY = diff(Y; dims=1)
    T = n - lags - 1
    restricted = deterministic in (:const_restricted, :trend_restricted)
    nun = deterministic in (:const, :trend_restricted) ? 1 : deterministic === :trend ? 2 : 0
    W = Matrix{Float64}(undef, T, K * lags + nun)
    X0 = Matrix{Float64}(undef, T, K)
    X1 = Matrix{Float64}(undef, T, K + restricted)
    @inbounds for i in 1:T
        t = lags + 1 + i
        for j in 1:lags, c in 1:K
            W[i, (j - 1) * K + c] = dY[t - 1 - j, c]
        end
        nun >= 1 && (W[i, K * lags + 1] = 1.0)
        nun == 2 && (W[i, K * lags + 2] = t)
        for c in 1:K
            X0[i, c] = dY[t - 1, c]
            X1[i, c] = Y[t - 1, c]
        end
        restricted && (X1[i, K + 1] = deterministic === :const_restricted ? 1.0 : Float64(t))
    end
    return W, X0, X1, T, nun
end

"""
    fit_vecm(Y; lags, rank, deterministic=:const, method=:ml) -> VECMModel

Fit a vector error-correction model of cointegration rank `rank` with
`lags` **differenced** lags (`statsmodels`' `k_ar_diff`; `urca`'s `K` is
`lags + 1`).

`deterministic` is [`johansen_test`](@ref)'s five-value vocabulary, with
the same meanings, so a rank chosen there and a model fitted here cannot
silently refer to different specifications:

| `deterministic` | `statsmodels` | `urca` |
|---|---|---|
| `:none` | `"n"` | — |
| `:const` | `"co"` (constant **outside** the relation) | `ecdet = "none"` + `cajorls` |
| `:const_restricted` | `"ci"` (constant **inside**) | `ecdet = "const"` |
| `:trend` | `"colo"` | — |
| `:trend_restricted` | `"coli"` | `ecdet = "trend"` |

`[V]` `"co"` and `"ci"` are close — on `coint2_yes`, `β₂` is
`−0.6673249` against `−0.6673217` and the log-likelihoods differ by
`0.022` — and they are different models.

## Two estimators

- `method = :ml` (default): Johansen's reduced-rank maximum likelihood —
  `β` from the same eigenproblem as [`johansen_test`](@ref), then `α`,
  `Γ` and the deterministic terms by least squares given `β`. This is
  what all three references compute by default and what the Johansen
  rank test assumes. `[V]` `urca::cajorls`, `statsmodels`' `VECM` and
  `tsDyn::VECM(estim = "ML")` agree on `β` to `≤ 1e-14`.
- `method = :twostep`: Engle & Granger's two-step estimator — `β` from a
  static least-squares regression of the first series on the others (with
  any *restricted* deterministic term), then the error-correction
  regression. Consistent but not efficient, and a genuinely different
  estimate (`[V]` `β = (1, −0.4931, −0.9655)` against ML's
  `(1, −0.4965, −0.9777)` on `coint3_r1`); it matches
  `tsDyn::VECM(estim = "2OLS")`. **Rank 1 only** — `[V]` `tsDyn` warns and
  silently switches to ML when asked for more; this refuses instead.

## Refusals

`rank` must be in `1 … K−1`: rank `0` is a VAR in differences and rank
`K` a stationary VAR in levels, and neither is a VECM. `[V]` `statsmodels`
accepts both (returning an empty `β` for 0 and the identity for `K`), and
also accepts `k_ar_diff = 0`, which is refused here too.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "coint3_r1.csv"), ','; skipstart=1);

julia> m = fit_vecm(Y; lags=1, rank=1);

julia> round.(vec(m.beta); digits=6)
3-element Vector{Float64}:
  1.0
 -0.496505
 -0.97773

julia> round.(vec(m.alpha); digits=6)
3-element Vector{Float64}:
  0.036929
 -0.234278
  0.960762
```

See also [`johansen_test`](@ref), [`var_rep`](@ref), [`forecast`](@ref).
"""
function fit_vecm(Y; lags::Integer, rank::Integer, deterministic::Symbol=:const,
                  method::Symbol=:ml)
    deterministic in _JOHANSEN_DET || throw(ArgumentError(
        "fit_vecm: deterministic must be one of $(join(string.(':', _JOHANSEN_DET), ", ")), " *
        "got :$deterministic"))
    method in (:ml, :twostep) || throw(ArgumentError(
        "fit_vecm: method must be :ml or :twostep, got :$method"))
    Y isa AbstractVector{<:Real} && throw(ArgumentError(
        "fit_vecm: a VECM needs at least two series, got a single vector."))
    Ym = Float64.(tsmatrix(Y))
    nms = tsnames(Y)
    n, K = size(Ym)
    K >= 2 || throw(ArgumentError("fit_vecm: a VECM needs at least two series, got $K."))
    lags >= 1 || throw(ArgumentError(
        "fit_vecm: lags counts differenced lags and must be at least 1, got $lags. " *
        "(statsmodels silently accepts 0.)"))
    1 <= rank <= K - 1 || throw(ArgumentError(
        "fit_vecm: rank must be in 1:$(K - 1) for $K series, got $rank. Rank 0 is a " *
        "VAR in differences and rank $K a stationary VAR in levels -- neither is a " *
        "VECM; use fit_var on diff(Y) or on Y. (statsmodels accepts both, returning " *
        "an empty beta for 0.)"))
    method === :twostep && rank > 1 && throw(ArgumentError(
        "fit_vecm: method = :twostep estimates a single cointegrating relation by a " *
        "static regression; rank $rank needs method = :ml. (tsDyn warns and silently " *
        "switches to ML here.)"))
    any(!isfinite, Ym) && throw(ArgumentError("fit_vecm: the input contains NaN or Inf."))
    T0 = n - lags - 1
    T0 > rank + K * lags + 3 || throw(ArgumentError(
        "fit_vecm: not enough observations -- lags = $lags on $n rows leaves $T0."))

    W, X0, X1, T, nun = _vecm_design(Ym, Int(lags), deterministic)
    K1 = size(X1, 2)
    r = Int(rank)

    beta = if method === :ml
        _, V, _, _ = _johansen_rrr(Ym, Int(lags), deterministic)
        B = V[:, 1:r]
        B / B[1:r, :]                                 # identity-normalized
    else
        # Engle-Granger step one, on all n observations of the levels: the
        # first series on the others and any RESTRICTED term. An unrestricted
        # constant stays outside the relation, in the error-correction
        # regression -- which is what tsDyn's include = "const" does, and why
        # its 2OLS beta is not the static regression with an intercept.
        Xs = Ym[:, 2:K]
        if deterministic === :const_restricted
            Xs = hcat(Xs, ones(n))
        elseif deterministic === :trend_restricted
            Xs = hcat(Xs, collect(1.0:n))
        end
        b = Xs \ Ym[:, 1]
        reshape(vcat(1.0, -b), K1, 1)
    end

    ect = X1 * beta
    Z = hcat(ect, W)
    q = size(Z, 2)
    _, E, _, ZtZ_inv = _mols(Z, X0)
    C = transpose(Z \ X0)                             # K x q
    alpha = Matrix(C[:, 1:r])
    gamma = [Matrix(C[:, (r + (j - 1) * K + 1):(r + j * K)]) for j in 1:lags]
    det_coef = Matrix(C[:, (r + K * lags + 1):q])

    sigma_mle = (E' * E) ./ T
    sigma_df = (E' * E) ./ (T - q)
    se = sqrt.(max.(_diagvec(ZtZ_inv) * transpose(_diagvec(sigma_mle)), 0.0))   # q x K
    se_alpha = Matrix(transpose(se[1:r, :]))
    se_gamma = Matrix(transpose(se[(r + 1):(r + K * lags), :]))
    se_det = Matrix(transpose(se[(r + K * lags + 1):q, :]))

    se_beta = nothing
    if method === :ml
        # Lütkepohl (2005), Prop. 7.6: with the levels partialled on W,
        # cov(vec(β_(K-r)')) = (Y₂ M Y₂')⁻¹ ⊗ (α' Σ⁻¹ α)⁻¹
        X1p = X1 - W * (W \ X1)
        Y2 = X1p[:, (r + 1):K1]
        cov_b = kron(inv(Symmetric(Y2' * Y2)), inv(Symmetric(alpha' * (sigma_mle \ alpha))))
        sb = zeros(Float64, K1, r)
        d = sqrt.(max.(_diagvec(cov_b), 0.0))         # vec of the r x (K1-r) matrix β_(K-r)'
        for j in 1:(K1 - r), i in 1:r
            sb[r + j, i] = d[(j - 1) * r + i]
        end
        se_beta = sb
    end

    loglik = -T / 2 * (K * log(2pi) + logdet(cholesky(Hermitian(sigma_mle))) + K)
    names = nms === nothing ? [Symbol("y", i) for i in 1:K] : collect(Symbol, nms)
    return VECMModel(Ym, names, Int(lags), r, deterministic, method, alpha, beta, gamma,
                     det_coef, sigma_df, sigma_mle, se_alpha, se_beta, se_gamma, se_det,
                     E, loglik, T)
end

"""
    normalize_beta(m::VECMModel, r=m.rank; scheme=:first_unit) -> Matrix{Float64}

The fitted cointegrating vectors in a conventional normalization; see
the [`JohansenTest`](@ref) method, which shares the implementation, so a
Johansen test and a VECM with the same `deterministic` give the same
normalized `β`.

A VECM stores `β` with an identity block (`statsmodels`' and `cajorls`'
form). At rank 1 both schemes coincide. At rank 2 or more the
`:first_unit` scheme is a property of the **eigenvector** basis that
[`johansen_test`](@ref) returns, not of the cointegrating space, and the
identity-normalized columns can lead with zero; it is refused here rather
than divided by zero — use `:identity`, which agrees with the Johansen
test at every rank.
"""
function normalize_beta(m::VECMModel, r::Integer=m.rank; scheme::Symbol=:first_unit)
    1 <= r <= m.rank || throw(ArgumentError(
        "normalize_beta: r must be in 1:$(m.rank), the model's rank, got $r"))
    if scheme === :first_unit && r > 1
        throw(ArgumentError(
            "normalize_beta: scheme = :first_unit at rank $r is a property of Johansen's " *
            "eigenvector basis, which a VECM does not keep -- use scheme = :identity " *
            "(which matches johansen_test's), or normalize_beta on the JohansenTest."))
    end
    return _normalize_beta(m.beta, r, scheme, m.rank)
end

"""
    var_rep(m::VECMModel) -> Vector{Matrix{Float64}}

The VECM written as a VAR in levels, `lags + 1` coefficient matrices:

    A₁ = I + α β' + Γ₁,    A_j = Γ_j − Γ_{j−1}  (j = 2 … lags),    A_{lags+1} = −Γ_lags

(`β'` restricted to its first `K` rows; a restricted deterministic term
moves into the VAR's intercept or trend). This is what [`forecast`](@ref)
and [`irf`](@ref) use — the VAR recursions of Stages 10.1 and 10.3, not a
second implementation — and the test suite checks that the levels VAR
reproduces the VECM's own residuals exactly.

`[V]` Matches `vars::vec2var` and `statsmodels`' `var_rep`.

The levels VAR has `K − r` unit roots **by construction** — that is what
cointegration means — so it is not stable, and its long-run effects do
not exist: [`lr_effects`](@ref) refuses a `VECMModel`.
"""
function var_rep(m::VECMModel)
    K = length(m.names)
    Pi = m.alpha * transpose(m.beta[1:K, :])
    A = Vector{Matrix{Float64}}(undef, m.lags + 1)
    A[1] = Matrix{Float64}(I, K, K) + Pi + m.gamma[1]
    for j in 2:m.lags
        A[j] = m.gamma[j] - m.gamma[j - 1]
    end
    A[m.lags + 1] = -m.gamma[m.lags]
    return A
end

# The levels VAR's deterministic part, in fit_var's vocabulary and layout.
function _vecm_var_det(m::VECMModel)
    K = length(m.names)
    det = m.deterministic
    if det === :none
        return :none, zeros(K, 0)
    elseif det === :const
        return :const, m.det_coef[:, 1:1]
    elseif det === :const_restricted
        return :const, m.alpha * transpose(m.beta[K + 1:K + 1, :])
    elseif det === :trend
        return :both, m.det_coef[:, 1:2]
    else # :trend_restricted
        return :both, hcat(m.det_coef[:, 1], m.alpha * transpose(m.beta[K + 1, :]))
    end
end

# The VECM as a VARModel, for delegation only. Its coefficient covariance is
# not the VAR's (the levels VAR's coefficients are a reduced-rank function of
# the VECM's), so the parameter-uncertainty fields are left empty and every
# path that would read them refuses before it gets there.
function _vecm_as_var(m::VECMModel)
    A = var_rep(m)
    det, dc = _vecm_var_det(m)
    p = m.lags + 1
    Yl = m.Y[(p + 1):end, :]
    return VARModel(copy(m.Y), copy(m.names), p, det, A, dc, m.sigma_df, m.sigma_mle,
                    zeros(length(m.names), 0), zeros(0, 0), m.resid, Yl - m.resid,
                    m.loglik, m.nobs, _companion(A), true)
end

"""
    forecast(m::VECMModel, h; levels=[80.0, 95.0]) -> VARForecast
    predict(m::VECMModel, h; levels=[80.0, 95.0]) -> VARForecast

Forecasts of the **levels**, through the VAR representation
([`var_rep`](@ref)) and the VAR forecast recursion — not a second
implementation. Intervals use the degrees-of-freedom-adjusted `Σ` and,
like the VAR's, ignore parameter uncertainty; because the levels are
I(1), they widen without bound. `[V]` Point forecasts match
`vars::predict(vec2var(…))` and `statsmodels`' `VECMResults.predict`.
"""
function StatsAPI.predict(m::VECMModel, h::Integer; levels::Vector{<:Real}=[80.0, 95.0])
    f = StatsAPI.predict(_vecm_as_var(m), h; levels=levels)
    return VARForecast(f.point, f.se, f.levels, f.lower, f.upper, f.names, f.horizon,
                       "VECM(lags=$(m.lags), rank=$(m.rank))")
end

forecast(m::VECMModel, h::Integer; levels::Vector{<:Real}=[80.0, 95.0]) =
    StatsAPI.predict(m, h; levels=levels)

"""
    irf(m::VECMModel, h; cumulative=false, impulse=nothing, response=nothing) -> IRFResult
    fevd(m::VECMModel, h) -> FEVDResult

Impulse responses and variance decomposition of a VECM, computed from its
levels VAR ([`var_rep`](@ref)) by the Stage 10.3 recursion. Point
estimates only: the analytic standard errors of 10.3 are for an
unrestricted VAR's coefficients, and the VECM's reduced-rank levels
coefficients have a different (singular) covariance, so `stderr` and
`boot` are refused rather than answered with the wrong formula.

Because the levels VAR has unit roots, the responses of a VECM do **not**
die out: they converge to the permanent effect of the shock.
"""
function irf(m::VECMModel, h::Integer; cumulative::Bool=false, impulse=nothing,
             response=nothing, stderr::Bool=false, boot::Bool=false, kwargs...)
    (stderr || boot) && throw(ArgumentError(
        "irf(::VECMModel): point estimates only -- the analytic and bootstrap " *
        "uncertainty paths of irf are for an unrestricted VAR, and the VECM's " *
        "reduced-rank levels coefficients have a different covariance."))
    r = irf(_vecm_as_var(m), h; cumulative=cumulative, impulse=impulse, response=response)
    return r
end

fevd(m::VECMModel, h::Integer) = fevd(_vecm_as_var(m), h)

roots(m::VECMModel) = roots(_vecm_as_var(m))
is_stable(::VECMModel) = false

function lr_effects(m::VECMModel; kwargs...)
    throw(ArgumentError(
        "lr_effects: a VECM's levels VAR has $(length(m.names) - m.rank) unit root(s) " *
        "by construction, so the cumulative response does not converge and the " *
        "long-run effect does not exist. The responses converge to the permanent " *
        "effect of a shock instead -- see irf(m, h; cumulative=false) at a long horizon."))
end
lr_effect_stderr(m::VECMModel; kwargs...) = lr_effects(m)

# --------------------------------------------------------------------------
# StatsAPI contract
# --------------------------------------------------------------------------

"""
    coef(m::VECMModel) -> Matrix{Float64}

`K × q`: `α`, then `Γ_1 … Γ_lags`, then the unrestricted deterministic
terms — the order of the error-correction regression. [`coefnames`](@ref)
labels the columns.
"""
StatsAPI.coef(m::VECMModel) = hcat(m.alpha, reduce(hcat, m.gamma), m.det_coef)

function StatsAPI.coefnames(m::VECMModel)
    out = [Symbol("ect", i) for i in 1:m.rank]
    for j in 1:m.lags, v in m.names
        push!(out, Symbol("d", v, "_l", j))
    end
    m.deterministic in (:const, :trend, :trend_restricted) && push!(out, :const)
    m.deterministic === :trend && push!(out, :trend)
    return out
end

StatsAPI.residuals(m::VECMModel) = m.resid
StatsAPI.nobs(m::VECMModel) = m.nobs
StatsAPI.loglikelihood(m::VECMModel) = m.loglik

"""
    dof(m::VECMModel)

The number of free mean parameters: `K·r` in `α`, `r(K₁ − r)` in `β` after
normalization, `K²·lags` in `Γ` and `K` per unrestricted deterministic
term. `aic` and `bic` are `−2 logL + 2·dof` and `−2 logL + log(T)·dof`.
"""
function StatsAPI.dof(m::VECMModel)
    K = length(m.names)
    K1 = size(m.beta, 1)
    return K * m.rank + m.rank * (K1 - m.rank) + K^2 * m.lags + K * size(m.det_coef, 2)
end
StatsAPI.aic(m::VECMModel) = -2 * m.loglik + 2 * StatsAPI.dof(m)
StatsAPI.bic(m::VECMModel) = -2 * m.loglik + log(m.nobs) * StatsAPI.dof(m)

function Base.show(io::IO, m::VECMModel)
    K = length(m.names)
    println(io, "VECM  K=", K, "  rank=", m.rank, "  differenced lags=", m.lags,
            "  T=", m.nobs, "  method=:", m.method)
    println(io, "  variables: ", join(string.(m.names), ", "))
    println(io, "  deterministic: ", _JO_DET_NAME[m.deterministic])
    B = normalize_beta(m, m.rank; scheme=:identity)
    for j in 1:m.rank
        println(io, "  beta[", j, "] = (", join(string.(round.(B[:, j]; digits=6)), ", "), ")")
    end
    print(io, "  logLik = ", round(m.loglik, digits=4),
          "   AIC = ", round(StatsAPI.aic(m), digits=4),
          "   BIC = ", round(StatsAPI.bic(m), digits=4))
end
