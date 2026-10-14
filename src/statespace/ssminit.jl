# ---------------------------------------------------------------------------
# Partitioned initialisation (Stage 11.0c, handoff/stage 11/
# stage-11.0-primitives-handoff.md §5).
#
# A mixed model's states do not share one prior. statsmodels partitions an
# unobserved-components model with level, trend, seasonal, cycle and an AR
# block as {non-AR states: approximate diffuse, kappa = 1e6; AR block:
# stationary} -- measured, `s11/uc_init.py`. The existing filters take one
# (a0, P0) pair, or kalman_filter_diffuse's flat `diffuse_idx`; this is a
# constructor for those arguments, so the filters themselves do not change.
#
# Internal: no export.
# ---------------------------------------------------------------------------

"""
    SSMInit(blocks; kappa=1e6, a0=nothing, P0_known=nothing)

A prior for the initial state, given block by block. `blocks` is a vector of
`range => scheme` pairs that must partition `1:r` exactly, with `scheme` one
of `:approximate_diffuse` (`P0 = kappa * I` on the block), `:stationary`
(the block's own stationary covariance -- the block must not be driven by
states outside it), `:known` (the block of `P0_known`) and `:exact_diffuse`
(passed through as `diffuse_idx` for [`kalman_filter_diffuse`](@ref)).
"""
struct SSMInit
    blocks::Vector{Pair{UnitRange{Int},Symbol}}
    kappa::Float64
    a0::Union{Nothing,Vector{Float64}}
    P0_known::Union{Nothing,Matrix{Float64}}
end

SSMInit(blocks::AbstractVector; kappa::Real=1e6, a0=nothing, P0_known=nothing) =
    SSMInit(Pair{UnitRange{Int},Symbol}[UnitRange{Int}(first(b)) => Symbol(last(b)) for b in blocks],
            Float64(kappa), a0 === nothing ? nothing : Vector{Float64}(a0),
            P0_known === nothing ? nothing : Matrix{Float64}(P0_known))

const _INIT_SCHEMES = (:approximate_diffuse, :stationary, :known, :exact_diffuse)

"""
    _stationary_cov(T, RQR; tol=1e-15, maxiter=60) -> (P, converged)

Solve `P = T P T' + RQR` by the doubling iteration -- the core of
[`stationary_cov`](@ref), on plain matrices so it applies to a sub-block.
`converged = false` on a unit root or divergence; it never throws.
"""
function _stationary_cov(T::AbstractMatrix, RQR::AbstractMatrix; tol::Real=1e-15,
                         maxiter::Integer=60)
    S = promote_type(eltype(T), eltype(RQR), Float64)
    P = Matrix{S}(RQR)
    A = Matrix{S}(T)
    converged = false
    for _ in 1:maxiter
        Pn = P + A * P * A'
        all(isfinite, Pn) || return (Pn, false)
        if maximum(abs, Pn .- P) <= tol * max(maximum(abs, Pn), one(S))
            P = Pn
            converged = true
            break
        end
        P = Pn
        A = A * A
    end
    return ((P .+ P') ./ 2, converged)
end

"""
    _resolve_init(T, RQR, init::SSMInit) -> (a0, P0, diffuse_idx)

Turn a partitioned prior into what the filters accept. `T` and `RQR` are the
(time-invariant) transition matrix and state-noise covariance `R Q R'`.
"""
function _resolve_init(T::AbstractMatrix, RQR::AbstractMatrix, init::SSMInit)
    r = size(T, 1)
    S = promote_type(eltype(T), eltype(RQR), Float64)
    covered = zeros(Int, r)
    for (rng, scheme) in init.blocks
        scheme in _INIT_SCHEMES || throw(ArgumentError(
            "SSMInit: unknown scheme :$scheme; use one of $(join(string.(':', _INIT_SCHEMES), ", "))"))
        (first(rng) >= 1 && last(rng) <= r) || throw(ArgumentError(
            "SSMInit: block $rng lies outside the $r states"))
        covered[rng] .+= 1
    end
    gap = findfirst(==(0), covered)
    gap === nothing || throw(ArgumentError(
        "SSMInit: state $gap is in no block -- the blocks must partition 1:$r"))
    over = findfirst(>(1), covered)
    over === nothing || throw(ArgumentError(
        "SSMInit: state $over is in more than one block -- the blocks must partition 1:$r"))

    a0 = init.a0 === nothing ? zeros(S, r) : Vector{S}(init.a0)
    length(a0) == r || throw(ArgumentError("SSMInit: a0 has length $(length(a0)), need $r"))
    P0 = zeros(S, r, r)
    diffuse_idx = Int[]
    for (rng, scheme) in init.blocks
        if scheme === :approximate_diffuse
            for i in rng
                P0[i, i] = init.kappa
            end
        elseif scheme === :exact_diffuse
            append!(diffuse_idx, rng)
        elseif scheme === :known
            init.P0_known === nothing && throw(ArgumentError(
                "SSMInit: block $rng is :known but no P0_known was given"))
            P0[rng, rng] = init.P0_known[rng, rng]
        else # :stationary
            outside = setdiff(1:r, rng)
            all(iszero, T[rng, outside]) || throw(ArgumentError(
                "SSMInit: the :stationary block $rng is driven by states outside it " *
                "(T[$rng, other] is non-zero), so it has no stationary distribution of its own"))
            Pb, ok = _stationary_cov(T[rng, rng], RQR[rng, rng])
            ok || throw(ArgumentError(
                "SSMInit: the :stationary block $rng has a unit root or is explosive -- " *
                "its stationary covariance does not exist"))
            P0[rng, rng] = Pb
        end
    end
    return a0, P0, diffuse_idx
end
