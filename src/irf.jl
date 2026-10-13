export ma_rep, orth_ma_rep, coef_alpha
export irf, IRFResult, irf_stderr, cum_effect_stderr, lr_effects, lr_effect_stderr
export fevd, FEVDResult

# ---------------------------------------------------------------------------
# Stage 10.3 -- impulse responses and forecast error variance decomposition.
#
# Primary source: Lütkepohl (2005), "New Introduction to Multiple Time Series
# Analysis", §2.3.2-2.3.3 (MA representation, impulse responses, FEVD) and
# §3.7 (their asymptotic distributions). vars::irf/fevd and statsmodels'
# IRAnalysis/FEVD are the numerical references; both agree with each other
# to machine precision on every point estimate.
#
# The recursion takes the impact factor `P` as an argument everywhere, so
# Stage 10.7's structural VAR reuses it with `P = A⁻¹B` instead of the
# Cholesky factor rather than writing a second one.
# ---------------------------------------------------------------------------

"""
    ma_rep(m::VARModel, h) -> Vector{Matrix{Float64}}

The first `h+1` matrices of the moving-average representation of a fitted
VAR, `Φ₀ = I` and `Φ_i = Σ_{j=1}^{min(i,p)} Φ_{i-j} A_j`. These are the
**non-orthogonalized** impulse responses: `Φ_i[response, impulse]` is the
response of `response` `i` periods after a unit shock to the reduced-form
error of `impulse`.

Returned **horizon first**, as a vector of `K × K` matrices, so `Φ[i+1]` is
`Φ_i`. The two references disagree on every layout while agreeing on every
number — `vars` puts the horizon *last* in a `K × K × (h+1)` array,
`statsmodels` first — and when `K == h+1` a transposed implementation
agrees with a correct one in shape, so the test suite checks this layout
on a fixture where that happens.

The order `Φ_{i-j} A_j` (not `A_j Φ_{i-j}`) is the one that matches both
references, with `A_j[i, m]` the coefficient of equation `i` on variable
`m` at lag `j`. The recursion accumulates into each matrix in place, so it
allocates the `h+1` result matrices and nothing else.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> m = fit_var(Y, 2);

julia> Phi = ma_rep(m, 8);

julia> length(Phi), Phi[1] == [1.0 0.0; 0.0 1.0], Phi[2] == m.A[1]
(9, true, true)

julia> round(Phi[3][2, 1]; digits=10)    # response of y2 to y1, two periods on
0.1999228499
```

See also [`orth_ma_rep`](@ref), [`irf`](@ref).
"""
function ma_rep(m::VARModel, h::Integer)
    _check_horizon(h, "ma_rep"; allow_zero=true)
    return _ma_rep(m.A, h)
end

"""
    _cholesky_factor(m) -> Matrix{Float64}

The lower-triangular `P` with `PP' = Σ`, from the **degrees-of-freedom
adjusted** residual covariance. `[V]` That is the `Σ` both references
orthogonalize with: R's `t(chol(Sigma_df))` equals `statsmodels`'
`cholesky(sigma_u)` to `≤ 6.7e-14` on all six fixtures, and both equal the
orthogonalized response at horizon 0 exactly.
"""
_cholesky_factor(m::VARModel) = Matrix(cholesky(Hermitian(m.sigma_df)).L)

"""
    _impact_ma_rep(Phi, P) -> Vector{Matrix{Float64}}

`Ψ_i = Φ_i P` for every horizon. This is the only place the impact factor
enters the point estimates: the Cholesky factor gives the orthogonalized
responses, and Stage 10.7 passes `A⁻¹B` for structural ones.
"""
_impact_ma_rep(Phi::Vector{Matrix{Float64}}, P::AbstractMatrix) = [Φ * P for Φ in Phi]

"""
    orth_ma_rep(m::VARModel, h) -> Vector{Matrix{Float64}}

The orthogonalized moving-average matrices `Ψ_i = Φ_i P`, `i = 0 … h`,
where `P` is the lower-triangular Cholesky factor of the residual
covariance (`PP' = Σ`). `Ψ_i[response, impulse]` is the response to a
one-standard-deviation orthogonalized shock.

!!! warning "The answer depends on the order of the variables"
    The Cholesky factor makes the first variable contemporaneously
    affected only by its own shock, the second by the first two, and so
    on. Reordering the columns of the data is therefore **not** cosmetic:
    it changes every orthogonalized response, while the
    non-orthogonalized [`ma_rep`](@ref) is merely relabelled. Choosing an
    order is a claim about contemporaneous causation. The fitted model's
    `names` record the order that was used.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> Psi = orth_ma_rep(fit_var(Y, 2), 4);

julia> round.(Psi[1]; digits=10)     # the Cholesky factor itself
2×2 Matrix{Float64}:
 1.00586   0.0
 0.295542  0.731323
```

See also [`ma_rep`](@ref), [`irf`](@ref).
"""
function orth_ma_rep(m::VARModel, h::Integer)
    _check_horizon(h, "orth_ma_rep"; allow_zero=true)
    return _impact_ma_rep(_ma_rep(m.A, h), _cholesky_factor(m))
end

function _check_horizon(h::Integer, who::AbstractString; allow_zero::Bool=false)
    lo = allow_zero ? 0 : 1
    h >= lo || throw(ArgumentError(
        "$who: the horizon must be at least $lo, got $h. (vars errors with " *
        "\"subscript out of bounds\" here, and statsmodels returns an empty " *
        "result without complaint.)"))
    return nothing
end

"""
    coef_alpha(m::VARModel) -> Vector{Float64}

The VAR's slope coefficients as one vector, `α = vec([A_1 … A_p])` —
the `K × Kp` matrix of lag coefficients flattened **column-major**, which
is Lütkepohl's ordering and the one the analytic impulse-response
covariance is written against.

`[V]` The ordering is not a convention to guess. A central-difference
Jacobian of the MA recursion reproduces `statsmodels`' analytic
covariance to `2.1e-12` with this flattening, and misses by `3.9e-2` —
about 40% of the standard error, and plausible-looking — with the
row-major alternative. The test suite asserts the layout directly.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> m = fit_var(Y, 2);

julia> a = coef_alpha(m);

julia> length(a), a[1:2] == m.A[1][:, 1], a[5:6] == m.A[2][:, 1]
(8, true, true)
```
"""
coef_alpha(m::VARModel) = vec(reduce(hcat, m.A))

"""
    _cov_alpha(m) -> Matrix{Float64}

`Σ_α`, the covariance of [`coef_alpha`](@ref): the lag block of
`vcov(m) = kron(Σ_df, (Z'Z)⁻¹)`, permuted from `vec(B)`'s
equation-by-equation order into `α`'s. Element `A_j[i, c]` sits at
`α`-position `((j-1)K + c - 1)K + i` and at `vec(B)`-position
`(i-1)·nz + (j-1)K + c`.
"""
function _cov_alpha(m::VARModel)
    K = length(m.names)
    p = m.p
    nz = K * p + _var_ndet(m.deterministic)
    n = K * K * p
    pos = Vector{Int}(undef, n)
    @inbounds for col in 1:(K * p), i in 1:K
        pos[(col - 1) * K + i] = (i - 1) * nz + col
    end
    return m.vcov_full[pos, pos]
end

"""
    _cov_sigma(m) -> Matrix{Float64}

The covariance of `vech(Σ̂)`, `2 D⁺ (Σ ⊗ Σ) D⁺' / T` (Lütkepohl 2005,
Prop. 3.4), with `D⁺` the closed-form duplication pseudo-inverse and `T`
the effective sample. `[V]` `T = n - p` is the scaling that reproduces the
references' orthogonalized standard error at horizon 0, where only this
term contributes.
"""
function _cov_sigma(m::VARModel)
    K = length(m.names)
    Dp = _dupl_pinv(K)
    S = m.sigma_df
    return (2 / m.T) .* (Dp * kron(S, S) * transpose(Dp))
end

"""
    _chol_jacobian(P) -> Matrix{Float64}

`H = ∂vec(P)/∂vech(Σ)'` for the lower-triangular Cholesky factor `P` of
`Σ`, in closed form (Lütkepohl 2005, Prop. 3.6 and §A.13):

    H = L'{ L (I_{K²} + K_{KK}) (P ⊗ I_K) L' }⁻¹

with `L` the elimination matrix (`vech = L vec`) and `K_{KK}` the
commutation matrix. All three are `O(K⁴)` dense and small. Verified
against a central-difference derivative of `cholesky` itself in the test
suite.
"""
function _chol_jacobian(P::AbstractMatrix)
    K = size(P, 1)
    vidx = _vech_indices(K)
    nv = length(vidx)
    L = zeros(Float64, nv, K * K)
    for (r, (i, j)) in enumerate(vidx)
        L[r, (j - 1) * K + i] = 1.0
    end
    Kcomm = zeros(Float64, K * K, K * K)
    for i in 1:K, j in 1:K
        Kcomm[(i - 1) * K + j, (j - 1) * K + i] = 1.0
    end
    inner = L * (I + Kcomm) * kron(P, Matrix{Float64}(I, K, K)) * transpose(L)
    return transpose(L) / inner
end

"""
    _irf_G(m, Phi, h) -> Vector{Matrix{Float64}}

`G_i = ∂vec(Φ_i)/∂α'`, `i = 0 … h`, each `K² × K²p`, from the recursion

    G_0 = 0,    G_i = G_{i-1} (A*' ⊗ I_K) + (J ⊗ Φ_{i-1})

with `A*` the companion matrix and `J = [I_K 0 … 0]`. That is Lütkepohl's
(3.7.4), `G_i = Σ_{m<i} J(A*')^{i-1-m} ⊗ Φ_m`, rearranged so each horizon
costs one multiplication instead of a fresh sum of Kronecker products:
`(X A*') ⊗ Φ = (X ⊗ Φ)(A*' ⊗ I)`. `J ⊗ Φ` is `Φ` repeated down the
diagonal of the first `K²` columns, so it is added in place rather than
built.
"""
function _irf_G(m::VARModel, Phi::Vector{Matrix{Float64}}, h::Integer)
    K = length(m.names)
    p = m.p
    nr, nc = K * K, K * K * p
    Akron = kron(transpose(m.C), Matrix{Float64}(I, K, K))
    G = Vector{Matrix{Float64}}(undef, h + 1)
    G[1] = zeros(Float64, nr, nc)
    for i in 1:h
        Gi = G[i] * Akron
        Φ = Phi[i]                        # Φ_{i-1}
        @inbounds for b in 1:K, c in 1:K, r in 1:K
            Gi[(b - 1) * K + r, (b - 1) * K + c] += Φ[r, c]
        end
        G[i + 1] = Gi
    end
    return G
end

# sqrt(diag(M S M')) without forming M S M'
function _quad_diag_sqrt(M::AbstractMatrix, S::AbstractMatrix)
    MS = M * S
    out = Vector{Float64}(undef, size(M, 1))
    @inbounds for r in 1:size(M, 1)
        acc = 0.0
        for c in 1:size(M, 2)
            acc += MS[r, c] * M[r, c]
        end
        out[r] = acc
    end
    return out
end

_as_KK(v::Vector{Float64}, K::Integer) = reshape(sqrt.(max.(v, 0.0)), K, K)

"""
    irf_stderr(m::VARModel, h; orth=false, cumulative=false) -> Vector{Matrix{Float64}}

Analytic asymptotic standard errors of the impulse responses at horizons
`0 … h`, horizon first, each `K × K` and laid out `[response, impulse]`
like [`ma_rep`](@ref).

Non-orthogonalized (Lütkepohl 2005, Prop. 3.6):
`cov(vec Φ_i) = G_i Σ_α G_i'`, with `Σ_α` the covariance of
[`coef_alpha`](@ref). Orthogonalized:
`cov(vec Ψ_i) = C_i Σ_α C_i' + C̄_i Σ_σ C̄_i'` with `C_i = (P' ⊗ I) G_i`,
`C̄_i = (I ⊗ Φ_i) H`, `Σ_σ` the covariance of `vech Σ̂` and `H` the
derivative of the Cholesky factor. `cumulative=true` gives the standard
errors of the running sums `Σ_{j≤i} Φ_j` (or `Ψ_j`), whose derivatives
are the running sums of the same matrices.

## Two things worth knowing

**The orthogonalized formula is verified by simulation, not by the
book.** `statsmodels`' own source marks the textbook statement as wrong at
this equation. The version implemented here was checked two independent
ways in the Stage 10 handoff: a central-difference Jacobian of the
recursion (agreement `≤ 2.1e-12`), and a 4000-replication parametric
bootstrap of the sampling distribution (agreement within `1.4–4.8%`,
against a Monte Carlo error of about `1.1%`). `vars` has no analytic
standard errors at all, so `statsmodels` is the only software reference.

**At horizon 0 the two forms differ structurally.** `Φ_0 = I` is not
estimated, so its standard errors are exactly zero. `Ψ_0 = P` depends on
the estimated `Σ`, so its standard errors are not — except above the
diagonal, where `P` is zero by construction.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> m = fit_var(Y, 2);

julia> irf_stderr(m, 4)[1]                 # Φ₀ = I is not estimated
2×2 Matrix{Float64}:
 0.0  0.0
 0.0  0.0

julia> round.(irf_stderr(m, 4; orth=true)[1]; digits=6)
2×2 Matrix{Float64}:
 0.050546  0.0
 0.054053  0.03675
```

See also [`cum_effect_stderr`](@ref), [`lr_effect_stderr`](@ref), [`irf`](@ref).
"""
function irf_stderr(m::VARModel, h::Integer; orth::Bool=false, cumulative::Bool=false)
    _check_horizon(h, "irf_stderr"; allow_zero=true)
    K = length(m.names)
    Phi = _ma_rep(m.A, h)
    G = _irf_G(m, Phi, h)
    if cumulative
        for i in 2:(h + 1)
            G[i] = G[i] + G[i - 1]
            Phi[i] = Phi[i] + Phi[i - 1]
        end
    end
    Sa = _cov_alpha(m)
    out = Vector{Matrix{Float64}}(undef, h + 1)
    if !orth
        for i in 1:(h + 1)
            out[i] = _as_KK(_quad_diag_sqrt(G[i], Sa), K)
        end
        return out
    end
    P = _cholesky_factor(m)
    H = _chol_jacobian(P)
    Ss = _cov_sigma(m)
    PtI = kron(transpose(P), Matrix{Float64}(I, K, K))
    IK = Matrix{Float64}(I, K, K)
    for i in 1:(h + 1)
        C = PtI * G[i]
        Cbar = kron(IK, Phi[i]) * H
        out[i] = _as_KK(_quad_diag_sqrt(C, Sa) .+ _quad_diag_sqrt(Cbar, Ss), K)
    end
    return out
end

"""
    cum_effect_stderr(m::VARModel, h; orth=false) -> Vector{Matrix{Float64}}

Standard errors of the cumulative impulse responses `Σ_{j≤i} Φ_j`,
`i = 0 … h` — shorthand for `irf_stderr(m, h; orth, cumulative=true)`.
At horizon 1 they equal the plain standard errors, since `Φ_0` contributes
nothing.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> round.(cum_effect_stderr(fit_var(Y, 2), 2)[3]; digits=6)
2×2 Matrix{Float64}:
 0.134482  0.169476
 0.108328  0.136645
```
"""
cum_effect_stderr(m::VARModel, h::Integer; orth::Bool=false) =
    irf_stderr(m, h; orth=orth, cumulative=true)

function _check_stable_for_lr(m::VARModel, who::AbstractString)
    is_stable(m) || throw(ArgumentError(
        "$who: the VAR is not stable (largest companion root " *
        "$(round(maximum(roots(m)); digits=6)) >= 1), so the cumulative " *
        "response does not converge and the long-run effect does not exist. " *
        "Finite-horizon impulse responses are still well defined -- use irf. " *
        "(Both references compute (I - sum A_j)^-1 regardless and return a " *
        "number with no meaning.)"))
    return nothing
end

"""
    lr_effects(m::VARModel; orth=false) -> Matrix{Float64}

The long-run (total) effects `Ψ_∞ = (I - Σ_j A_j)⁻¹`, the limit of the
cumulative responses; with `orth=true`, `Ψ_∞ P`.

**Refuses an unstable VAR.** The cumulative response only converges when
every companion root is inside the unit circle. `[V]` Neither reference
checks: on `var2_explosive` (largest root `1.0404`) `statsmodels` inverts
`I - Σ A_j` anyway and returns a number that is not the limit of
anything. Here that is an `ArgumentError` naming the problem, while the
finite-horizon [`irf`](@ref) keeps working.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> round.(lr_effects(fit_var(Y, 2)); digits=4)
2×2 Matrix{Float64}:
 1.8092  -0.1176
 0.6702   1.5399
```
"""
function lr_effects(m::VARModel; orth::Bool=false)
    _check_stable_for_lr(m, "lr_effects")
    Psi = _lr_matrix(m)
    return orth ? Psi * _cholesky_factor(m) : Psi
end

function _lr_matrix(m::VARModel)
    K = length(m.names)
    S = Matrix{Float64}(I, K, K)
    for Aj in m.A
        S .-= Aj
    end
    return inv(S)
end

"""
    lr_effect_stderr(m::VARModel; orth=false) -> Matrix{Float64}

Standard errors of [`lr_effects`](@ref). With `Ψ_∞ = (I - A(1))⁻¹`,
`∂vec(Ψ_∞)/∂α' = (Ψ_∞' ⊗ Ψ_∞) [I_{K²} … I_{K²}]` (`p` blocks, since
`A(1) = Σ_j A_j`). The orthogonalized form adds the `Σ` term exactly as
[`irf_stderr`](@ref) does. Refuses an unstable VAR, for the same reason.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> round.(lr_effect_stderr(fit_var(Y, 2)); digits=6)
2×2 Matrix{Float64}:
 0.239755  0.302223
 0.21395   0.269695
```
"""
function lr_effect_stderr(m::VARModel; orth::Bool=false)
    _check_stable_for_lr(m, "lr_effect_stderr")
    K = length(m.names)
    Psi = _lr_matrix(m)
    F = repeat(kron(transpose(Psi), Psi), 1, m.p)
    Sa = _cov_alpha(m)
    if !orth
        return _as_KK(_quad_diag_sqrt(F, Sa), K)
    end
    P = _cholesky_factor(m)
    IK = Matrix{Float64}(I, K, K)
    B = kron(transpose(P), IK) * F
    Bbar = kron(IK, Psi) * _chol_jacobian(P)
    return _as_KK(_quad_diag_sqrt(B, Sa) .+ _quad_diag_sqrt(Bbar, _cov_sigma(m)), K)
end

# ---------------------------------------------------------------------------
# irf: the result object and the bootstrap
# ---------------------------------------------------------------------------

"""
    IRFResult

Impulse responses returned by [`irf`](@ref). Every matrix is
`[response, impulse]` and every vector of them is **horizon first**, with
`horizons == 0:h` — so `r.orth[1]` is the impact response.

- `orth`, `nonorth`: orthogonalized (Cholesky) and non-orthogonalized
  responses, plain or cumulative as `cumulative` says.
- `lower`, `upper`: bootstrap percentile bands for `orth`;
  `lower_nonorth`, `upper_nonorth` the same for `nonorth`. `nothing`
  unless `boot=true`.
- `stderr_orth`, `stderr_nonorth`: analytic standard errors, `nothing`
  unless `stderr=true`.
- `names`: the model's variable order, which **is** the Cholesky ordering;
  `responses`, `impulses`: the variables actually returned.
- `level`, `runs`: the bootstrap settings; `method`: `:none`,
  `:analytic` or `:bootstrap`, so a plot can say what its bands are.
- `factorization`: `:cholesky`, recording how the orthogonalized shocks
  were identified.
"""
struct IRFResult
    orth::Vector{Matrix{Float64}}
    nonorth::Vector{Matrix{Float64}}
    lower::Union{Nothing,Vector{Matrix{Float64}}}
    upper::Union{Nothing,Vector{Matrix{Float64}}}
    lower_nonorth::Union{Nothing,Vector{Matrix{Float64}}}
    upper_nonorth::Union{Nothing,Vector{Matrix{Float64}}}
    stderr_orth::Union{Nothing,Vector{Matrix{Float64}}}
    stderr_nonorth::Union{Nothing,Vector{Matrix{Float64}}}
    horizons::UnitRange{Int}
    names::Vector{Symbol}
    responses::Vector{Symbol}
    impulses::Vector{Symbol}
    cumulative::Bool
    level::Union{Nothing,Float64}
    method::Symbol
    runs::Union{Nothing,Int}
    factorization::Symbol
end

_cumsum_mats(v::Vector{Matrix{Float64}}) = accumulate(+, v)
_sub_mats(v::Vector{Matrix{Float64}}, r, c) = [M[r, c] for M in v]
_sub_mats(::Nothing, r, c) = nothing

"""
    irf(m::VARModel, h; cumulative=false, impulse=nothing, response=nothing,
        stderr=false, boot=false, runs=100, level=95.0,
        rng=Random.default_rng(), parallel=true) -> IRFResult

Impulse responses of a fitted VAR at horizons `0 … h`, both
orthogonalized (Cholesky) and non-orthogonalized, as an
[`IRFResult`](@ref).

- `cumulative=true` returns running sums `Σ_{j≤i}` instead.
- `impulse`, `response` select variables by name or index (a single one
  or a vector); `nothing` keeps all of them.
- `stderr=true` attaches the analytic standard errors of
  [`irf_stderr`](@ref). Point estimates alone are the common case, so
  they do not pay for a covariance nobody asked for.
- `boot=true` attaches residual-bootstrap percentile bands at `level`
  percent from `runs` replications.

!!! warning "Orthogonalized responses depend on the variable order"
    See [`orth_ma_rep`](@ref). Reordering the data's columns changes every
    orthogonalized response; it is a statement about which shocks move
    which variables within the period, not a formatting choice. The
    result's `names` record the order used.

## The bootstrap

Residuals are centred and resampled **by row**, keeping their
contemporaneous correlation; each artificial series is rebuilt through the
fitted recursion from the real first `p` observations, refitted with the
same lag order and deterministic terms, and its responses recomputed. That
is `vars`' scheme. The band is the elementwise `(1-level/100)/2` and
`(1+level/100)/2` quantiles across replications.

Each replication draws from its **own** generator, seeded from `rng`, so a
seeded run is reproducible and the threaded and serial paths return
bit-identical bands. `runs` defaults to `vars`' `100`, which is thin for a
95% percentile interval — `1000` or more is advisable when the band
matters.

Bands cannot be checked numerically against `vars`: a different
resampling stream gives different numbers by design. The test suite
checks what a band claims instead — that it covers the true response of a
known VAR at close to its nominal rate.

```jldoctest
julia> using TSAnalytics, DelimitedFiles, Random

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> r = irf(fit_var(Y, 2), 8);

julia> r.horizons, round(r.orth[2][1, 1]; digits=10)
(0:8, 0.4716609801)

julia> b = irf(fit_var(Y, 2), 4; boot=true, runs=50, rng=MersenneTwister(1));

julia> b.method, b.runs, size(b.lower[1])
(:bootstrap, 50, (2, 2))
```

See also [`fevd`](@ref), [`ma_rep`](@ref), [`irf_stderr`](@ref).
"""
function irf(m::VARModel, h::Integer;
             cumulative::Bool=false, impulse=nothing, response=nothing,
             stderr::Bool=false, boot::Bool=false, runs::Integer=100,
             level::Real=95.0, rng::AbstractRNG=Random.default_rng(),
             parallel::Bool=true)
    _check_horizon(h, "irf")
    imp = impulse === nothing ? collect(1:length(m.names)) :
          _var_indices(m, impulse; who="irf")
    res = response === nothing ? collect(1:length(m.names)) :
          _var_indices(m, response; who="irf")
    if boot
        runs >= 1 || throw(ArgumentError("irf: runs must be at least 1, got $runs"))
        0 < level < 100 || throw(ArgumentError(
            "irf: level is a percentage in (0, 100), got $level"))
    end

    Phi = _ma_rep(m.A, h)
    Psi = _impact_ma_rep(Phi, _cholesky_factor(m))
    if cumulative
        Phi = _cumsum_mats(Phi)
        Psi = _cumsum_mats(Psi)
    end

    se_o = se_n = nothing
    if stderr
        se_o = irf_stderr(m, h; orth=true, cumulative=cumulative)
        se_n = irf_stderr(m, h; orth=false, cumulative=cumulative)
    end

    lo = up = lon = upn = nothing
    if boot
        lo, up, lon, upn = _irf_bootstrap(m, h, Int(runs), Float64(level), rng,
                                          parallel, cumulative)
    end
    method = boot ? :bootstrap : (stderr ? :analytic : :none)

    return IRFResult(_sub_mats(Psi, res, imp), _sub_mats(Phi, res, imp),
                     _sub_mats(lo, res, imp), _sub_mats(up, res, imp),
                     _sub_mats(lon, res, imp), _sub_mats(upn, res, imp),
                     _sub_mats(se_o, res, imp), _sub_mats(se_n, res, imp),
                     0:Int(h), copy(m.names), m.names[res], m.names[imp],
                     cumulative, boot ? Float64(level) : nothing, method,
                     boot ? Int(runs) : nothing, :cholesky)
end

"""
    _irf_bootstrap(m, h, runs, level, rng, parallel, cumulative)

Residual-bootstrap percentile bands for both response forms. One
generator per replication, seeded deterministically from `rng`, so the
threaded and serial paths agree bit for bit. Each replication allocates
its own series and regressor buffers; nothing is shared across threads
except the read-only fit.
"""
function _irf_bootstrap(m::VARModel, h::Int, runs::Int, level::Float64,
                        rng::AbstractRNG, parallel::Bool, cumulative::Bool)
    K = length(m.names)
    p = m.p
    n = size(m.Y, 1)
    T = m.T
    E = m.resid .- mean(m.resid; dims=1)
    Z, _, _ = build_var_Z(m.Y, p, m.deterministic)
    ndet = _var_ndet(m.deterministic)
    detpart = ndet == 0 ? zeros(Float64, T, K) :
              Z[:, (K * p + 1):end] * transpose(m.det_coef)
    seeds = rand(rng, UInt64, runs)

    draws_o = Array{Float64}(undef, runs, h + 1, K, K)
    draws_n = Array{Float64}(undef, runs, h + 1, K, K)

    function one!(r)
        rr = Xoshiro(seeds[r])
        Ys = Matrix{Float64}(undef, n, K)
        Ys[1:p, :] .= view(m.Y, 1:p, :)
        idx = rand(rr, 1:T, T)
        @inbounds for t in 1:T
            row = p + t
            for i in 1:K
                acc = detpart[t, i] + E[idx[t], i]
                for j in 1:p, c in 1:K
                    acc += m.A[j][i, c] * Ys[row - j, c]
                end
                Ys[row, i] = acc
            end
        end
        mb = fit_var(Ys, p; deterministic=m.deterministic)
        Phi = _ma_rep(mb.A, h)
        Psi = _impact_ma_rep(Phi, _cholesky_factor(mb))
        if cumulative
            Phi = _cumsum_mats(Phi)
            Psi = _cumsum_mats(Psi)
        end
        @inbounds for s in 1:(h + 1), c in 1:K, i in 1:K
            draws_o[r, s, i, c] = Psi[s][i, c]
            draws_n[r, s, i, c] = Phi[s][i, c]
        end
        return nothing
    end

    if parallel && Threads.nthreads() > 1
        Threads.@threads for r in 1:runs
            one!(r)
        end
    else
        for r in 1:runs
            one!(r)
        end
    end

    a = (1 - level / 100) / 2
    lo, up = _percentile_bands(draws_o, a)
    lon, upn = _percentile_bands(draws_n, a)
    return lo, up, lon, upn
end

# Top-level on purpose: as a closure inside `_irf_bootstrap` its `lo`/`up`
# would be captured from the enclosing scope, and the second call would
# silently overwrite the first call's bands -- a bug the coverage test caught.
function _percentile_bands(D::Array{Float64,4}, a::Float64)
    _, H, K, _ = size(D)
    lo = [Matrix{Float64}(undef, K, K) for _ in 1:H]
    up = [Matrix{Float64}(undef, K, K) for _ in 1:H]
    for s in 1:H, c in 1:K, i in 1:K
        v = D[:, s, i, c]
        lo[s][i, c] = quantile(v, a)
        up[s][i, c] = quantile(v, 1 - a)
    end
    return lo, up
end

function Base.show(io::IO, r::IRFResult)
    kind = r.cumulative ? "cumulative impulse responses" : "impulse responses"
    println(io, "VAR ", kind, ", horizons ", first(r.horizons), ":", last(r.horizons))
    println(io, "  ordering (Cholesky): ", join(string.(r.names), ", "))
    println(io, "  impulses:  ", join(string.(r.impulses), ", "))
    println(io, "  responses: ", join(string.(r.responses), ", "))
    if r.method === :bootstrap
        print(io, "  bands: residual bootstrap, $(r.level)% percentile, $(r.runs) runs")
    elseif r.method === :analytic
        print(io, "  uncertainty: analytic asymptotic standard errors")
    else
        print(io, "  point estimates only")
    end
end

# ---------------------------------------------------------------------------
# FEVD
# ---------------------------------------------------------------------------

"""
    FEVDResult

A forecast error variance decomposition, returned by [`fevd`](@ref).

- `decomp[variable, horizon, shock]`: the share of `variable`'s
  `horizon`-step forecast error variance due to orthogonalized `shock`.
  Each `decomp[i, h, :]` sums to one.
- `horizons == 1:h` — **not** `0:h`. A one-step forecast already has
  error, so the decomposition starts at 1, while the impulse response
  starts at the impact period 0. The two objects deliberately do not
  share a horizon axis, and nothing silently aligns them.
- `names`: the variable order, which is the Cholesky ordering.
"""
struct FEVDResult
    decomp::Array{Float64,3}
    horizons::UnitRange{Int}
    names::Vector{Symbol}
end

"""
    fevd(m::VARModel, h) -> FEVDResult

Forecast error variance decomposition at horizons `1 … h`. The
contribution of orthogonalized shock `j` to the `H`-step forecast error
variance of variable `i` is `Σ_{s=0}^{H-1} Ψ_s[i, j]²`, normalized by its
sum over `j` — which is the `H`-step mean squared error itself, since
`Σ_j Ψ_s[i, j]² = (Φ_s Σ Φ_s')_{ii}`. Like the orthogonalized
[`irf`](@ref), it depends on the variable order.

Both references agree with each other to `≤ 9.1e-15` on six fixtures, and
both mishandle `h = 0` — `vars` errors with "subscript out of bounds",
`statsmodels` returns an empty decomposition. That is refused here.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> f = fevd(fit_var(Y, 2), 8);

julia> f.horizons, size(f.decomp)
(1:8, (2, 8, 2))

julia> round.(f.decomp[2, 1, :]; digits=6)   # y2 at one step: its own shock and y1's
2-element Vector{Float64}:
 0.140386
 0.859614
```

See also [`irf`](@ref), [`FEVDResult`](@ref).
"""
function fevd(m::VARModel, h::Integer)
    _check_horizon(h, "fevd")
    return _fevd_from_psi(orth_ma_rep(m, h - 1), Int(h), copy(m.names))
end

function _fevd_from_psi(Psi::Vector{Matrix{Float64}}, h::Int, names::Vector{Symbol})
    K = size(Psi[1], 1)
    D = Array{Float64,3}(undef, K, h, K)
    acc = zeros(Float64, K, K)
    @inbounds for H in 1:h
        Ψ = Psi[H]
        for j in 1:K, i in 1:K
            acc[i, j] += Ψ[i, j]^2
        end
        for i in 1:K
            tot = 0.0
            for j in 1:K
                tot += acc[i, j]
            end
            for j in 1:K
                D[i, H, j] = acc[i, j] / tot
            end
        end
    end
    return FEVDResult(D, 1:h, names)
end

function Base.show(io::IO, f::FEVDResult)
    K = size(f.decomp, 1)
    h = last(f.horizons)
    println(io, "Forecast error variance decomposition, horizons 1:", h)
    println(io, "  ordering (Cholesky): ", join(string.(f.names), ", "))
    print(io, "  share of each variable's ", h, "-step error variance by shock:")
    for i in 1:K
        print(io, "\n    ", rpad(string(f.names[i]), 12))
        for j in 1:K
            print(io, "  ", lpad(string(round(f.decomp[i, h, j]; digits=4)), 7))
        end
    end
end
