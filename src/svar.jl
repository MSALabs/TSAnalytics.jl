export fit_svar, blanchard_quah, SVARModel

# ---------------------------------------------------------------------------
# Stage 10.7 -- structural VARs: the A-, B- and AB-models and Blanchard-Quah.
#
# Primary sources: Amisano & Giannini (1997), "Topics in Structural VAR
# Econometrics", and Lütkepohl (2005), §9.1-9.3 (the AB-model, its
# identification and its concentrated likelihood); Blanchard & Quah (1989),
# AER 79(4), for the long-run restriction. vars::SVAR/BQ and statsmodels'
# SVAR are the numerical references.
#
# Structural impulse responses reuse Stage 10.3's recursion with P = A⁻¹B in
# place of the Cholesky factor; nothing here recomputes the MA representation.
# ---------------------------------------------------------------------------

"""
    SVARModel <: TimeSeriesModel

A structural VAR, returned by [`fit_svar`](@ref) or
[`blanchard_quah`](@ref): the reduced form `var`, and the structural
matrices in `A u_t = B ε_t`, `ε_t ~ (0, I)`.

- `svar_type`: `:A`, `:B`, `:AB` or `:bq`.
- `A`, `B`: the estimates; `A_mask`, `B_mask` mark the free elements.
- `stderr_A`, `stderr_B`: asymptotic standard errors of the free elements
  (zero on restricted ones), from the inverse information matrix;
  `nothing` for Blanchard-Quah, which is a closed-form transformation.
- `P = A⁻¹B`: the impact matrix — what the structural impulse responses
  multiply the moving-average matrices by.
- `lrim`: the long-run impact matrix `(I − Σ A_j)⁻¹ P` (Blanchard-Quah
  only; lower triangular by construction).
- `loglik`: the concentrated structural log-likelihood (see
  [`fit_svar`](@ref) for the `Σ` it uses).
- `lr_test`: `(; statistic, df, pvalue)` for an over-identified model,
  `nothing` for a just-identified one.
- `converged`, `iterations`, `method`.
"""
struct SVARModel <: TimeSeriesModel
    var::VARModel
    svar_type::Symbol
    A::Matrix{Float64}
    B::Matrix{Float64}
    A_mask::BitMatrix
    B_mask::BitMatrix
    stderr_A::Union{Nothing,Matrix{Float64}}
    stderr_B::Union{Nothing,Matrix{Float64}}
    P::Matrix{Float64}
    lrim::Union{Nothing,Matrix{Float64}}
    loglik::Float64
    lr_test::Union{Nothing,NamedTuple{(:statistic, :df, :pvalue),Tuple{Float64,Int,Float64}}}
    converged::Bool
    iterations::Int
    method::Symbol
end

# Split a user restriction matrix into (values, free mask). `missing` marks
# a free element; NaN is refused by name, because statsmodels' np.nan trap
# (its free marker is the string "E", and a float NaN silently yields an
# empty mask) shows what an ambiguous marker costs.
function _svar_pattern(M, K::Int, which::String)
    M === nothing && return Matrix{Float64}(I, K, K), falses(K, K)
    size(M) == (K, K) || throw(ArgumentError(
        "fit_svar: $which must be $K × $K for a K = $K VAR, got $(size(M, 1)) × $(size(M, 2))"))
    vals = Matrix{Float64}(undef, K, K)
    mask = falses(K, K)
    for j in 1:K, i in 1:K
        x = M[i, j]
        if ismissing(x)
            mask[i, j] = true
            vals[i, j] = 0.0
        elseif x isa Real && isnan(x)
            throw(ArgumentError(
                "fit_svar: $which[$i, $j] is NaN. Mark free parameters with `missing`, not " *
                "NaN -- NaN reads as a computed value, and statsmodels' equivalent " *
                "(np.nan where it expects the string \"E\") silently estimates nothing."))
        elseif x isa Real
            vals[i, j] = Float64(x)
        else
            throw(ArgumentError("fit_svar: $which[$i, $j] must be a number or `missing`, got $(typeof(x))"))
        end
    end
    return vals, mask
end

# θ -> (A, B), filling the free positions in column-major order, A first.
function _svar_AB(θ::AbstractVector{T}, A0, Am, B0, Bm) where {T}
    A = Matrix{T}(A0)
    B = Matrix{T}(B0)
    k = 0
    for idx in eachindex(Am)
        Am[idx] && (k += 1; A[idx] = θ[k])
    end
    for idx in eachindex(Bm)
        Bm[idx] && (k += 1; B[idx] = θ[k])
    end
    return A, B
end

# The concentrated log-likelihood of the AB-model given the reduced-form Σ
# (Lütkepohl 2005, (9.3.2)):
#   ℓ = −KT/2 ln 2π + T ln|det A| − T ln|det B| − T/2 tr(A'B'⁻¹B⁻¹A Σ)
#
# Its score and the information matrix below are in closed form rather than
# by automatic differentiation. With ForwardDiff, a dual whose value is zero
# compares equal to zero, so the generic `\`, `det` and `lu` -- which test a
# matrix's structure first -- treated a free element at its zero start value
# as a structural zero and silently dropped its derivative. Explicit `lu`
# avoided it on Julia 1.9 and not on 1.12; the analytic forms avoid it on
# every version.
function _svar_loglik(θ, A0, Am, B0, Bm, S, T::Int)
    A, B = _svar_AB(θ, A0, Am, B0, Bm)
    K = size(S, 1)
    FA = lu(A; check=false)
    FB = lu(B; check=false)
    (issuccess(FA) && issuccess(FB)) || return oftype(float(one(eltype(A))), -Inf)
    M = FB \ A                                 # B⁻¹A
    return -K * T / 2 * log(2pi) + T * logabsdet(FA)[1] - T * logabsdet(FB)[1] -
           T / 2 * tr(transpose(M) * M * S)
end

_svar_sigma(θ, A0, Am, B0, Bm) = (AB = _svar_AB(θ, A0, Am, B0, Bm);
                                  P = lu(AB[1]) \ AB[2]; P * transpose(P))

# The score, in closed form (Lütkepohl 2005, §9.3): with M = B⁻ᵀB⁻¹,
#   ∂ℓ/∂A = T A⁻ᵀ − T M A S,     ∂ℓ/∂B = −T B⁻ᵀ + T M A S Aᵀ B⁻ᵀ,
# read off at the free positions (A's first, column-major, then B's).
function _svar_score(θ, A0, Am, B0, Bm, S, T::Int)
    A, B = _svar_AB(θ, A0, Am, B0, Bm)
    Ai = inv(lu(A)); Bi = inv(lu(B))
    M = transpose(Bi) * Bi
    gA = T .* (transpose(Ai) .- M * A * S)
    gB = T .* (-transpose(Bi) .+ M * A * S * transpose(A) * transpose(Bi))
    return vcat(gA[Am], gB[Bm])
end

# J = ∂vec Σ(θ)/∂θ for Σ(θ) = A⁻¹BBᵀA⁻ᵀ, in closed form:
#   ∂Σ/∂A_ij = −(A⁻¹E_ij Σ + Σ E_ji A⁻ᵀ),   ∂Σ/∂B_ij = A⁻¹(E_ij Bᵀ + B E_ji)A⁻ᵀ
function _svar_jacobian(θ, A0, Am, B0, Bm)
    A, B = _svar_AB(θ, A0, Am, B0, Bm)
    K = size(A, 1)
    Ai = inv(lu(A))
    Sg = Ai * B * transpose(B) * transpose(Ai)
    J = zeros(K * K, count(Am) + count(Bm))
    c = 0
    for idx in CartesianIndices(Am)
        Am[idx] || continue
        c += 1
        i, j = Tuple(idx)
        D = -(Ai[:, i] * transpose(Sg[j, :]))           # A⁻¹ E_ij Σ
        J[:, c] = vec(D + transpose(D))
    end
    for idx in CartesianIndices(Bm)
        Bm[idx] || continue
        c += 1
        i, j = Tuple(idx)
        D = Ai[:, i] * transpose(Ai * B[:, j])          # A⁻¹ E_ij Bᵀ A⁻ᵀ
        J[:, c] = vec(D + transpose(D))
    end
    return J
end

# Expected information for the free parameters: the covariance-structure
# form T/2 · J'(Σ⁻¹ ⊗ Σ⁻¹)J, J = ∂vec Σ(θ)/∂θ, Σ(θ) = A⁻¹BB'A'⁻¹ -- the
# same information Amisano & Giannini's scoring algorithm iterates with.
function _svar_information(θ, A0, Am, B0, Bm, T::Int)
    J = _svar_jacobian(θ, A0, Am, B0, Bm)
    Si = inv(Symmetric(_svar_sigma(θ, A0, Am, B0, Bm)))
    return (T / 2) .* (transpose(J) * kron(Si, Si) * J)
end

"""
    fit_svar(m::VARModel; A=nothing, B=nothing, method=:scoring,
             maxiter=200, tol=1e-10) -> SVARModel

Estimate a structural VAR on the fitted reduced form `m`:

    A u_t = B ε_t,    ε_t ~ (0, I_K),    so  Σ_u = A⁻¹ B B' A'⁻¹

Give `A` (an **A-model**, `B = I`), `B` (a **B-model**, `A = I`) or both
(an **AB-model**), as `K × K` matrices with **`missing`** marking the
free elements and numbers fixing the rest:

```julia
fit_svar(m; A = [1.0 0.0; missing 1.0], B = [missing 0.0; 0.0 missing])
```

`missing`, not `NaN`: `NaN` is refused by name. `[V]` `statsmodels`' marker
is the string `"E"`, and passing `np.nan` instead silently gives it an
empty set of parameters and an opaque error.

## Identification is checked first

`Σ_u` has `K(K+1)/2` distinct elements, so at most that many parameters
can be identified (Lütkepohl 2005, §9.1.2):

- more free parameters than `K(K+1)/2`, or free parameters at the same
  position in `A` and `B`: **refused** before any optimization —
  `[V]` `vars` does the same; `statsmodels` fails deep inside with a
  singular-matrix error;
- exactly `K(K+1)/2`: **just-identified** — `A⁻¹B(A⁻¹B)' = Σ_u` exactly,
  and there is nothing to test;
- fewer: **over-identified** — the restrictions are testable, and
  `lr_test` carries Lütkepohl's likelihood-ratio statistic
  `T(ln|Σ̃_r| − ln|Σ̃_u|)` on `K(K+1)/2 − nfree` degrees of freedom.

The rank condition is checked too: a specification whose information
matrix is singular is refused rather than handed to the optimizer.

!!! warning "A unit-diagonal `A` with `B = I` is over-identified"
    `A = [1 0; missing 1]` alone has **one** free parameter for a
    `2 × 2` `Σ_u`: it also forces both structural variances to one. It
    is the A-model `vars` and `statsmodels` are usually shown with, and
    `[V]` on `var2_p1` `vars` itself rejects it (`LR = 121.6` on 2 df).
    The just-identified recursive model is `A = [1 0; missing 1]` with a
    free diagonal `B`, or a lower-triangular free `B` — both reproduce
    the Cholesky factor exactly. (`statsmodels` reports the *reduced-form*
    log-likelihood as its SVAR `llf` whatever the restrictions, which
    hides this.)

## Estimation

`method = :scoring` (default) is Amisano & Giannini's scoring algorithm
— Newton steps with the expected information — which is what
`vars::SVAR(estmethod = "scoring")` uses; `:mle` maximizes the same
likelihood with `_optimize`'s quasi-Newton method. Both start from a
feasible point built from `Σ_u`. `[V]` `vars`' `estmethod = "direct"`
stops short of the optimum (`A₂₁ = −0.293750…` against `−0.293821`) and R
warns about it; it is not reproduced.

The likelihood is concentrated with the reduced form's
**degrees-of-freedom-adjusted** `Σ_u` (`m.sigma_df`), which is what both
references decompose — `[V]` their `P P'` is `Σ_df`. So `loglik` at a
just-identified optimum is the Gaussian log-likelihood at `Σ_df`, which
differs from `loglikelihood(m)` (computed at the ML `Σ`) by
`T/2 (ln|Σ_df| − ln|Σ_ml|)`; the LR test compares like with like.

Standard errors are the square roots of the inverse information matrix
for the free parameters. Structural impulse responses and variance
decompositions are [`irf`](@ref)`(s, h)` and [`fevd`](@ref)`(s, h)`.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> m = fit_var(Y, 2);

julia> s = fit_svar(m; A = [1.0 0.0; missing 1.0], B = [missing 0.0; 0.0 missing]);

julia> s.lr_test === nothing          # just-identified
true

julia> s.P * s.P' ≈ m.sigma_df
true

julia> round(s.A[2, 1]; digits=6)
-0.293821
```

See also [`blanchard_quah`](@ref), [`irf`](@ref).
"""
function fit_svar(m::VARModel; A=nothing, B=nothing, method::Symbol=:scoring,
                  maxiter::Integer=200, tol::Real=1e-10)
    method in (:scoring, :mle) || throw(ArgumentError(
        "fit_svar: method must be :scoring or :mle, got :$method"))
    maxiter >= 1 || throw(ArgumentError("fit_svar: maxiter must be at least 1, got $maxiter"))
    tol > 0 || throw(ArgumentError("fit_svar: tol must be positive, got $tol"))
    (A === nothing && B === nothing) && throw(ArgumentError(
        "fit_svar: give A (an A-model), B (a B-model) or both, with `missing` marking " *
        "the free elements. For Blanchard-Quah long-run restrictions use blanchard_quah."))
    K = length(m.names)
    A0, Am = _svar_pattern(A, K, "A")
    B0, Bm = _svar_pattern(B, K, "B")
    svar_type = A === nothing ? :B : B === nothing ? :A : :AB
    nfree = count(Am) + count(Bm)
    nmax = K * (K + 1) ÷ 2
    nfree == 0 && throw(ArgumentError(
        "fit_svar: no free parameters -- mark the elements to estimate with `missing`."))
    any(Am .& Bm) && throw(ArgumentError(
        "fit_svar: free parameters at the same position in A and B -- the model is not " *
        "identified (vars refuses this too)."))
    nfree > nmax && throw(ArgumentError(
        "fit_svar: $nfree free parameters, but Σ_u has only $nmax distinct elements for " *
        "K = $K: the model is not identified (the order condition). Fix more elements " *
        "of A or B."))

    S = m.sigma_df
    T = m.T
    # a feasible start built from Σ_u: free diagonal entries of B at the
    # residual standard deviations, of A at their reciprocals; the rest 0
    θ = Float64[]
    for idx in CartesianIndices(Am)
        Am[idx] && push!(θ, idx[1] == idx[2] ? 1 / sqrt(S[idx[1], idx[1]]) : 0.0)
    end
    for idx in CartesianIndices(Bm)
        Bm[idx] && push!(θ, idx[1] == idx[2] ? sqrt(S[idx[1], idx[1]]) : 0.0)
    end
    f(t) = _svar_loglik(t, A0, Am, B0, Bm, S, T)

    # the rank condition, at a generic point near the start
    θg = θ .+ 0.1 .* sin.(1:nfree)
    Ig = try
        _svar_information(θg, A0, Am, B0, Bm, T)
    catch err
        err isa Union{SingularException,LinearAlgebra.LAPACKException,PosDefException} ||
            rethrow()
        throw(ArgumentError(
            "fit_svar: A or B is singular for every value of the free parameters (a " *
            "row or column with no nonzero element), so Σ_u = A⁻¹BB'A'⁻¹ cannot be formed."))
    end
    (all(isfinite, Ig) && rank(Ig; rtol=1e-8) == nfree) || throw(ArgumentError(
        "fit_svar: the restrictions satisfy the order condition but not the rank " *
        "condition -- the information matrix is singular, so the free parameters are " *
        "not identified. Change which elements of A and B are free."))

    converged = false
    iters = 0
    if method === :scoring
        ℓ = f(θ)
        for it in 1:maxiter
            iters = it
            g = _svar_score(θ, A0, Am, B0, Bm, S, T)
            Iθ = _svar_information(θ, A0, Am, B0, Bm, T)
            step = Symmetric(Iθ) \ g
            λ = 1.0
            θn = θ .+ step
            ℓn = f(θn)
            while !(ℓn >= ℓ - 1e-12 * abs(ℓ)) && λ > 1e-8
                λ /= 2
                θn = θ .+ λ .* step
                ℓn = f(θn)
            end
            Δ = maximum(abs.(θn .- θ))
            θ, ℓ = θn, ℓn
            if Δ < tol
                converged = true
                break
            end
        end
    else
        res = _optimize(t -> -f(t), θ; method=:bfgs, autodiff=false, g_tol=tol,
                        iterations=Int(maxiter))
        θ = res.minimizer
        converged = res.converged
        iters = res.iterations
    end
    isfinite(f(θ)) || throw(ArgumentError(
        "fit_svar: the optimizer reached a singular A or B; the specification is not usable."))

    Ae, Be = _svar_AB(θ, A0, Am, B0, Bm)
    Ae, Be, θ = _svar_sign_normalize(Ae, Be, Am, Bm, A0, B0)
    Iθ = _svar_information(θ, A0, Am, B0, Bm, T)
    V = inv(Symmetric(Iθ))
    se = sqrt.(max.(_diagvec(V), 0.0))
    seA = zeros(K, K)
    seB = zeros(K, K)
    k = 0
    for idx in eachindex(Am)
        Am[idx] && (k += 1; seA[idx] = se[k])
    end
    for idx in eachindex(Bm)
        Bm[idx] && (k += 1; seB[idx] = se[k])
    end

    P = Ae \ Be
    lr = nothing
    if nfree < nmax
        Sr = P * transpose(P)
        stat = T * (logdet(cholesky(Hermitian(Sr))) - logdet(cholesky(Hermitian(S))))
        df = nmax - nfree
        lr = (statistic=stat, df=df, pvalue=_chisq_ccdf(stat, df))
    end
    return SVARModel(m, svar_type, Ae, Be, Am, Bm, seA, seB, P, nothing, f(θ), lr,
                     converged, iters, method)
end

# Shock signs are not identified: ε_j → −ε_j flips column j of P. Make the
# diagonal of P's structural matrices positive where a free row of A, or a
# free column of B, allows it without touching a fixed nonzero element.
function _svar_sign_normalize(A, B, Am, Bm, A0, B0)
    K = size(A, 1)
    A = copy(A); B = copy(B)
    for j in 1:K
        if B[j, j] < 0 && all(i -> Bm[i, j] || B0[i, j] == 0, 1:K)
            B[:, j] .*= -1
        end
        if A[j, j] < 0 && all(i -> Am[j, i] || A0[j, i] == 0, 1:K) &&
           all(i -> Bm[j, i] || B0[j, i] == 0, 1:K)
            A[j, :] .*= -1
            B[j, :] .*= -1
        end
    end
    θ = vcat(A[Am], B[Bm])
    return A, B, θ
end

"""
    blanchard_quah(m::VARModel) -> SVARModel

Blanchard & Quah's (1989) long-run identification: the structural shocks
are orthogonal and the **long-run** impact matrix
`Ξ = (I − Σ_j A_j)⁻¹ B` is lower triangular — the last shock has no
permanent effect on the first variable, and so on. It is just-identified
and has a closed form:

    Ξ = chol(Ψ_∞ Σ_u Ψ_∞')ᴸ,    B = (I − Σ_j A_j) Ξ,    A = I

with `Ψ_∞ = (I − Σ_j A_j)⁻¹` and `Σ_u` the degrees-of-freedom-adjusted
residual covariance. `B B' = Σ_u` exactly, and `lrim` is lower triangular
exactly — both asserted in the test suite with no reference involved.
`[V]` `B` and `lrim` match `vars::BQ`; `statsmodels` has no equivalent.

**An unstable VAR is refused**: the long-run matrix `Ψ_∞` does not exist
when a companion root is on or outside the unit circle, which is the same
refusal as [`lr_effects`](@ref). `vars::BQ` inverts `I − Σ A_j`
regardless.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> bq = blanchard_quah(fit_var(Y, 2));

julia> round.(bq.lrim; digits=4)
2×2 Matrix{Float64}:
 1.7871  0.0
 1.0737  1.1792
```
"""
function blanchard_quah(m::VARModel)
    _check_stable_for_lr(m, "blanchard_quah")
    K = length(m.names)
    Psi = _lr_matrix(m)
    Xi = Matrix(cholesky(Hermitian(Psi * m.sigma_df * transpose(Psi))).L)
    Bq = (Matrix{Float64}(I, K, K) - sum(m.A)) * Xi
    Aq = Matrix{Float64}(I, K, K)
    ll = -K * m.T / 2 * log(2pi) - m.T / 2 * logdet(cholesky(Hermitian(m.sigma_df))) -
         K * m.T / 2
    return SVARModel(m, :bq, Aq, Bq, falses(K, K), falses(K, K), nothing, nothing,
                     Bq, Xi, ll, nothing, true, 0, :closed_form)
end

"""
    irf(s::SVARModel, h; cumulative=false, impulse=nothing, response=nothing) -> IRFResult
    fevd(s::SVARModel, h) -> FEVDResult

Structural impulse responses `Φ_i P` with `P = A⁻¹B`, and the variance
decomposition by structural shock — Stage 10.3's recursion with the
structural impact matrix in place of the Cholesky factor, not a second
implementation. The result's `orth` field holds the structural responses
and `factorization` records `:structural` or `:blanchard_quah`; `nonorth`
is the reduced form's, unchanged. Point estimates only: the analytic
standard errors of [`irf_stderr`](@ref) are for the Cholesky factor.

For a recursive just-identified model the structural responses equal the
orthogonalized ones of [`irf`](@ref)`(s.var, h)` up to the sign of each
shock.
"""
function irf(s::SVARModel, h::Integer; cumulative::Bool=false, impulse=nothing,
             response=nothing, stderr::Bool=false, boot::Bool=false, kwargs...)
    (stderr || boot) && throw(ArgumentError(
        "irf(::SVARModel): point estimates only -- analytic and bootstrap bands for " *
        "structural responses are not implemented."))
    _check_horizon(h, "irf")
    m = s.var
    imp = impulse === nothing ? collect(1:length(m.names)) : _var_indices(m, impulse; who="irf")
    res = response === nothing ? collect(1:length(m.names)) : _var_indices(m, response; who="irf")
    Phi = _ma_rep(m.A, h)
    Psi = _impact_ma_rep(Phi, s.P)
    if cumulative
        Phi = _cumsum_mats(Phi)
        Psi = _cumsum_mats(Psi)
    end
    return IRFResult(_sub_mats(Psi, res, imp), _sub_mats(Phi, res, imp), nothing, nothing,
                     nothing, nothing, nothing, nothing, 0:Int(h), copy(m.names),
                     m.names[res], m.names[imp], cumulative, nothing, :none, nothing,
                     s.svar_type === :bq ? :blanchard_quah : :structural)
end

function fevd(s::SVARModel, h::Integer)
    _check_horizon(h, "fevd")
    return _fevd_from_psi(_impact_ma_rep(_ma_rep(s.var.A, h - 1), s.P), Int(h), copy(s.var.names))
end

StatsAPI.loglikelihood(s::SVARModel) = s.loglik
StatsAPI.nobs(s::SVARModel) = s.var.T
StatsAPI.residuals(s::SVARModel) = s.var.resid

"""
    structural_shocks(s::SVARModel) -> Matrix{Float64}

The estimated structural shocks `ε_t = P⁻¹ u_t`, `T × K`. For a
just-identified model their sample covariance (with the reduced form's
degrees-of-freedom divisor) is the identity.
"""
structural_shocks(s::SVARModel) = Matrix(transpose(s.P \ transpose(s.var.resid)))

const _SVAR_NAME = Dict(:A => "A-model", :B => "B-model", :AB => "AB-model",
                        :bq => "Blanchard-Quah (long-run restrictions)")

function Base.show(io::IO, s::SVARModel)
    K = length(s.var.names)
    println(io, "Structural VAR, ", _SVAR_NAME[s.svar_type], "  K=", K, "  T=", s.var.T)
    println(io, "  variables: ", join(string.(s.var.names), ", "))
    if s.svar_type !== :bq
        nfree = count(s.A_mask) + count(s.B_mask)
        println(io, "  free parameters: ", nfree, " of ", K * (K + 1) ÷ 2,
                s.lr_test === nothing ? "  (just-identified)" : "  (over-identified)")
    end
    for (nm, M) in (("A", s.A), ("B", s.B))
        println(io, "  ", nm, " = ")
        for i in 1:K
            println(io, "    ", join(lpad.(string.(round.(M[i, :]; digits=6)), 11), " "))
        end
    end
    if s.lr_test !== nothing
        println(io, "  LR test of the over-identifying restrictions: ",
                round(s.lr_test.statistic; digits=4), " on ", s.lr_test.df, " df, p = ",
                _fmt_p(s.lr_test.pvalue))
    end
    print(io, s.svar_type === :bq ? "  closed form" :
              (s.converged ? "  converged" : "  NOT CONVERGED") * " in $(s.iterations) iterations (:$(s.method))")
end
