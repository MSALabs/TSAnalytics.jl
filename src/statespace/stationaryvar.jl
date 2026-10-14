# ---------------------------------------------------------------------------
# Stationarity-enforcing reparameterisation of VAR coefficients (Stage 11.0d,
# handoff/stage 11/docs/stage-11.0-primitives-handoff.md §6).
#
# An unconstrained optimiser over a VAR block wanders into the explosive
# region, where a stationary prior does not exist. This maps unconstrained
# R^(p k^2) onto the stationary region and back, in the manner of Ansley &
# Kohn (1986): through normalised partial autocorrelation matrices, which
# parameterise stationary processes freely (Morf, Vieira & Kailath 1978).
#
#   1. U_s (k x k, unconstrained) -> P_s = L_s^{-1} U_s, L_s = chol(I + U_s U_s'),
#      so P_s P_s' = I - (I + U_s U_s')^{-1}: every singular value below one.
#   2. With the innovation variance fixed at I, a backward pass gives the
#      factors S_s of the forward prediction-error variances
#      (S_p = I, S_s = S_{s+1} C_{s+1}^{-1}, C = chol(I - P P')), and the
#      backward ones start from the same factor, T_0 = S_0.
#   3. The Whittle (multivariate Durbin-Levinson) recursion from there:
#        Phi_{s+1,s+1}  = S_s P_{s+1} T_s^{-1}
#        Phi*_{s+1,s+1} = T_s P_{s+1}' S_s^{-1}
#        Phi_{s+1,j}    = Phi_{s,j}  - Phi_{s+1,s+1}  Phi*_{s,s+1-j}
#        Phi*_{s+1,j}   = Phi*_{s,j} - Phi*_{s+1,s+1} Phi_{s,s+1-j}
#        S_{s+1} = S_s chol(I - P P'),  T_{s+1} = T_s chol(I - P'P)
#      and the VAR coefficients are Phi_{p,1..p}.
#
# Fixing the INNOVATION variance (not Gamma_0) is what makes the map onto:
# with Gamma_0 fixed instead, a VAR(1) could only reach coefficient matrices
# of norm below one, not every matrix with spectral radius below one.
#
# Cholesky factors throughout (no eigendecomposition, no comparison against
# a float), so ForwardDiff's Dual numbers pass through. The inverse has no
# closed form under this factor convention -- the starting factor S_0 is
# fixed by the whole recursion -- so it is a Newton solve.
#
# Internal: no export.
# ---------------------------------------------------------------------------

function _constrain_stationary(u::AbstractVector, k::Integer, p::Integer)
    length(u) == p * k^2 || throw(ArgumentError(
        "_constrain_stationary: need p*k^2 = $(p * k^2) values, got $(length(u))"))
    S = eltype(u)
    Id = Matrix{S}(I, k, k)
    Ps = Vector{Matrix{S}}(undef, p)
    for s in 1:p
        U = reshape(u[((s - 1) * k^2 + 1):(s * k^2)], k, k)
        L = cholesky(Symmetric(Id + U * U')).L
        Ps[s] = L \ U
    end
    # backward pass: S_p = I, S_s = S_{s+1} C_{s+1}^{-1}
    Sf = Vector{Matrix{S}}(undef, p + 1)
    Sf[p + 1] = Id
    for s in p:-1:1
        C = cholesky(Symmetric(Id - Ps[s] * Ps[s]')).L
        Sf[s] = Sf[s + 1] / C
    end
    Tb = Sf[1]
    fwd = Matrix{S}[]          # Phi_{s,1..s}
    bwd = Matrix{S}[]          # Phi*_{s,1..s}
    for s in 0:(p - 1)
        Ss = Sf[s + 1]
        P = Ps[s + 1]
        new_f = Ss * P / Tb
        new_b = Tb * P' / Ss
        f2 = Matrix{S}[fwd[j] - new_f * bwd[s + 1 - j] for j in 1:s]
        b2 = Matrix{S}[bwd[j] - new_b * fwd[s + 1 - j] for j in 1:s]
        fwd = push!(f2, new_f)
        bwd = push!(b2, new_b)
        Tb = Tb * cholesky(Symmetric(Id - P' * P)).L
    end
    return fwd
end

# Inverse: a Newton solve of _constrain_stationary(u) = A, by continuation.
# Scaling A_j -> t^j A_j multiplies every companion eigenvalue by exactly t,
# so the path from 0 (whose preimage is u = 0) to A stays stationary, and
# Newton only ever takes a short step from the previous solution. A naive
# start at u = vec(A) fails on a third of random VAR(2)s near the unit circle.
function _unconstrain_stationary(A::AbstractVector{<:AbstractMatrix}; tol::Real=1e-13,
                                 maxiter::Integer=50)
    p = length(A)
    k = size(A[1], 1)
    maximum(abs, eigvals(_companion(Matrix{Float64}.(A)))) < 1 || throw(ArgumentError(
        "_unconstrain_stationary: the VAR is not stationary -- it has no unconstrained image"))
    target(t) = reduce(vcat, [vec(t^j .* A[j]) for j in 1:p])
    function newton(u, tgt)
        resid(x) = reduce(vcat, vec.(_constrain_stationary(x, k, p))) .- tgt
        r = resid(u)
        for _ in 1:maxiter
            nr = norm(r)
            nr <= tol * max(1.0, norm(tgt)) && return (u, true)
            # the Jacobian is taken with exact zeros nudged to 1e-300: on Julia
            # 1.12 generic matmul skips entries for which iszero holds, and
            # ForwardDiff 0.10's iszero ignores the partials, so a Dual with
            # value 0 and a nonzero derivative (u = 0 at the start of the
            # continuation) silently loses its derivative and J is singular
            du = ForwardDiff.jacobian(resid, map(x -> iszero(x) ? 1e-300 : x, u)) \ r
            step = 1.0
            while true
                un = u .- step .* du
                rn = all(isfinite, un) ? resid(un) : fill(Inf, length(r))
                if norm(rn) < nr
                    u, r = un, rn
                    break
                end
                step /= 2
                step < 1e-12 && return (u, false)
            end
        end
        return (u, norm(r) <= 1e3 * tol * max(1.0, norm(tgt)))
    end
    u = zeros(p * k^2)
    t, dt = 0.0, 0.25
    while t < 1
        tn = min(1.0, t + dt)
        un, ok = try
            newton(u, target(tn))
        catch e
            e isa PosDefException || e isa ArgumentError || rethrow()
            (u, false)
        end
        if ok
            u, t = un, tn
            dt = min(2dt, 0.25)
        else
            dt /= 4
            dt < 1e-8 && throw(ArgumentError(
                "_unconstrain_stationary: continuation stalled at t = $t -- the VAR is too close " *
                "to the unit circle to invert reliably"))
        end
    end
    return u
end
