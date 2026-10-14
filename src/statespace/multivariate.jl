# ---------------------------------------------------------------------------
# Multivariate-observation filter and smoother (Stage 11.0a/b,
# handoff/stage 11/docs/stage-11.0-primitives-handoff.md §2-§3).
#
# The SEQUENTIAL (univariate) treatment: each of the N series observed in a
# period enters as its own scalar update, and the transition is applied once,
# after all N have been absorbed. A loop around the scalar recursion the
# package already has -- no N x N factorisation, missing values skipped in the
# loop, O(N r^2) per period. It requires a diagonal H_t (uncorrelated
# measurement errors within a period), which every Stage 11 model has; a dense
# H is refused rather than silently diagonalised.
#
# The algorithm is stated in full in the docstring, so it does not rest on a
# section-number citation (the charter records that this project's Durbin &
# Koopman section numbers are from memory, not an edition in hand).
# ---------------------------------------------------------------------------

function _mv_check_H(H::AbstractMatrix, t::Integer)
    N = size(H, 1)
    @inbounds for j in 1:N, i in 1:N
        i == j && continue
        iszero(H[i, j]) || throw(ArgumentError(
            "kalman_filter: H at period $t is not diagonal (H[$i,$j] = $(H[i, j])). The " *
            "sequential multivariate treatment needs uncorrelated measurement errors within " *
            "a period; it does not silently use the diagonal of a dense H."))
    end
    return nothing
end

"""
    kalman_filter(ssm::TimeVaryingSSM, Y::AbstractMatrix{<:Real},
                  a0::AbstractVector{<:Real}, P0::AbstractMatrix{<:Real})
        -> (loglik, v, F, converged)

Kalman filter for **several series observed together**: `Y` is
observations × series (`n × N`), `Z_t` is `N × r` and `H_t` must be
diagonal. `v` and `F` are `n × N` (`NaN` where a value is missing), and
`loglik` is the full Gaussian log-likelihood, with no concentration --
exactly as the vector method.

The series are absorbed one at a time within each period (the sequential,
or univariate, treatment of a multivariate model), and the transition is
applied once per period, after all of them:
```
for each period t, starting from a = a_t, P = P_t:
    for i = 1..N, skipping a missing Y[t,i]:
        z = Z_t[i,:];  v = Y[t,i] - z'a;  f = z'P z + H_t[i,i]
        k = P z / f;   a = a + k v;       P = P - f k k'
        loglik += -½(log 2π + log f + v²/f)
    a_{t+1} = T_t a;   P_{t+1} = T_t P T_t' + R_t Q_t R_t'
```
No `N × N` matrix is factorised, a missing value is simply skipped, and a
wholly missing period is prediction-only with no special case. Cost
`O(N r²)` per period.

**A non-positive `f` throws**, naming the period, the series and the value.
Both references fail silently there instead -- one returns a likelihood
fourteen orders of magnitude wrong as the measurement variances approach
zero, the other skips the period. A dense `H_t` also throws.

`Y` a matrix dispatches here; a vector keeps the univariate method, which is
unchanged.
"""
function kalman_filter(ssm::TimeVaryingSSM, Y::AbstractMatrix{<:Real},
                       a0::AbstractVector{<:Real}, P0::AbstractMatrix{<:Real})
    return _kf_mv(ssm, Y, a0, P0, true)
end

# strict = true throws on a degenerate f (a direct call); strict = false
# returns the sentinel (-Inf, empty, empty, false) for an optimiser's
# objective, matching the univariate method's convention.
function _kf_mv(ssm::TimeVaryingSSM, Y::AbstractMatrix{<:Real}, a0::AbstractVector{<:Real},
                P0::AbstractMatrix{<:Real}, strict::Bool)
    n, N = size(Y)
    n == 0 && throw(ArgumentError("kalman_filter: Y must have at least one row"))
    r = ssm.r
    length(a0) == r || throw(ArgumentError("kalman_filter: a0 must have length r=$r"))
    size(P0) == (r, r) || throw(ArgumentError("kalman_filter: P0 must be r x r = $r x $r"))
    size(_tv_at(ssm.Z, 1), 1) == N || throw(ArgumentError(
        "kalman_filter: Z has $(size(_tv_at(ssm.Z, 1), 1)) rows but Y has $N columns"))
    S = promote_type(eltype(P0), eltype(a0), eltype(ssm.T[1]), eltype(Y), Float64)
    a = Vector{S}(a0)
    P = Matrix{S}(P0)
    v = fill(S(NaN), n, N)
    F = fill(S(NaN), n, N)
    Pz = zeros(S, r)
    ta = zeros(S, r)
    TP = zeros(S, r, r)
    kq = size(_tv_at(ssm.R, 1), 2)
    RQ = zeros(S, r, kq)
    acc = zero(S)
    nobs = 0
    @inbounds for t in 1:n
        Tt = _tv_at(ssm.T, t); Zt = _tv_at(ssm.Z, t)
        Rt = _tv_at(ssm.R, t); Qt = _tv_at(ssm.Q, t); Ht = _tv_at(ssm.H, t)
        _mv_check_H(Ht, t)
        for i in 1:N
            yi = Y[t, i]
            isnan(yi) && continue
            za = zero(S)
            for b in 1:r
                za += Zt[i, b] * a[b]
            end
            f = S(Ht[i, i])
            for c in 1:r
                s = zero(S)
                for b in 1:r
                    s += P[c, b] * Zt[i, b]
                end
                Pz[c] = s
                f += Zt[i, c] * s
            end
            vi = yi - za
            if !(f > 0) || !isfinite(f)
                strict && throw(ArgumentError(
                    "kalman_filter: prediction variance f = $f <= 0 at period $t, series $i. " *
                    "The measurement variances or the state covariance have collapsed; the " *
                    "likelihood is not defined here (it is not silently skipped)."))
                return (S(-Inf), Matrix{S}(undef, 0, 0), Matrix{S}(undef, 0, 0), false)
            end
            for c in 1:r
                a[c] += Pz[c] * vi / f
            end
            for b in 1:r, c in 1:r
                P[c, b] -= Pz[c] * Pz[b] / f
            end
            v[t, i] = vi
            F[t, i] = f
            acc += log(f) + vi^2 / f
            nobs += 1
        end
        mul!(ta, Tt, a); copyto!(a, ta)
        mul!(TP, Tt, P); mul!(P, TP, transpose(Tt))
        mul!(RQ, Rt, Qt); mul!(P, RQ, transpose(Rt), true, true)
    end
    return (-0.5 * (nobs * log(2pi) + acc), v, F, true)
end

"""
    kalman_smoother(ssm::TimeVaryingSSM, Y::AbstractMatrix{<:Real},
                    a0::AbstractVector{<:Real}, P0::AbstractMatrix{<:Real})
        -> (alpha, V, eta, eta_var, eps, eps_var, converged)

Fixed-interval smoother for several series observed together, the backward
pass of the sequential treatment [`kalman_filter`](@ref)'s matrix method
uses. Within each period the backward recursion runs over the series in
reverse, `i = N…1`, with the filtering gain `k = P z / f` of that series'
update:
```
r ← z v/f + L' r,   N ← z z'/f + L' N L,   L = I − k z'
```
and between periods `r ← T' r`, `N ← T' N T`. Then
`alpha_t = a_t + P_t r`, `V_t = P_t − P_t N P_t` from the period's
predicted state, and the disturbances
`eta_t = Q R' r_t`, `Var = Q − Q R' N_t R Q` (with `r_t`, `N_t` the
values carried out of period `t + 1`), and per series
`eps[t,i] = H_ii (v/f − k' r)`, `Var = H_ii − H_ii² (1/f + k' N k)`.

Returns `alpha` (`r × n`), `V` (`n` matrices), `eta` (`k × n`, one column of
state disturbances per period), `eta_var` (`n` matrices), `eps` and
`eps_var` (`n × N`, `NaN` where a value is missing). `eta` and `eta_var` are a
matrix and matrices, not the vector method's scalars, because a multivariate
model's state disturbance is a vector; the smoothed quantities satisfy
`alpha_{t+1} = T_t alpha_t + R_t eta_t` exactly.
"""
function kalman_smoother(ssm::TimeVaryingSSM, Y::AbstractMatrix{<:Real},
                         a0::AbstractVector{<:Real}, P0::AbstractMatrix{<:Real})
    n, N = size(Y)
    n == 0 && throw(ArgumentError("kalman_smoother: Y must have at least one row"))
    r = ssm.r
    length(a0) == r || throw(ArgumentError("kalman_smoother: a0 must have length r=$r"))
    size(P0) == (r, r) || throw(ArgumentError("kalman_smoother: P0 must be r x r = $r x $r"))
    S = promote_type(eltype(P0), eltype(a0), eltype(ssm.T[1]), eltype(Y), Float64)

    # forward pass, storing the period's predicted state and, per series,
    # the innovation, its variance and the filtering gain
    a = Vector{S}(a0); P = Matrix{S}(P0)
    a_pred = Vector{Vector{S}}(undef, n); P_pred = Vector{Matrix{S}}(undef, n)
    v = fill(S(NaN), n, N); F = fill(S(NaN), n, N)
    K = zeros(S, r, N, n)
    for t in 1:n
        Tt = _tv_at(ssm.T, t); Zt = _tv_at(ssm.Z, t)
        Rt = _tv_at(ssm.R, t); Qt = _tv_at(ssm.Q, t); Ht = _tv_at(ssm.H, t)
        _mv_check_H(Ht, t)
        a_pred[t] = copy(a); P_pred[t] = copy(P)
        for i in 1:N
            isnan(Y[t, i]) && continue
            z = Zt[i, :]
            Pz = P * z
            f = dot(z, Pz) + Ht[i, i]
            (f > 0 && isfinite(f)) || throw(ArgumentError(
                "kalman_smoother: prediction variance f = $f <= 0 at period $t, series $i"))
            vi = Y[t, i] - dot(z, a)
            k = Pz ./ f
            a = a .+ k .* vi
            P = P .- f .* (k * k')
            v[t, i] = vi; F[t, i] = f; K[:, i, t] = k
        end
        a = Tt * a
        P = Tt * P * Tt' + Rt * Qt * Rt'
    end

    # backward pass. `rvec`/`Nmat`: the smoothing recursion's own r and N,
    # named apart from the state dimension r and the series count N.
    rvec = zeros(S, r); Nmat = zeros(S, r, r)
    kq = size(_tv_at(ssm.R, 1), 2)
    alpha = Matrix{S}(undef, r, n); V = Vector{Matrix{S}}(undef, n)
    eta = Matrix{S}(undef, kq, n); eta_var = Vector{Matrix{S}}(undef, n)
    eps = fill(S(NaN), n, N); eps_var = fill(S(NaN), n, N)
    for t in n:-1:1
        Tt = _tv_at(ssm.T, t); Zt = _tv_at(ssm.Z, t)
        Rt = _tv_at(ssm.R, t); Qt = _tv_at(ssm.Q, t); Ht = _tv_at(ssm.H, t)
        # rvec/Nmat here are r_t, N_t: what period t+1 passes back
        eta[:, t] = Qt * (Rt' * rvec)
        eta_var[t] = Qt - Qt * (Rt' * Nmat * Rt) * Qt
        rvec = Tt' * rvec
        Nmat = Tt' * Nmat * Tt
        for i in N:-1:1
            isnan(Y[t, i]) && continue
            z = Zt[i, :]
            k = K[:, i, t]
            f = F[t, i]
            h = Ht[i, i]
            eps[t, i] = h * (v[t, i] / f - dot(k, rvec))
            eps_var[t, i] = h - h^2 * (1 / f + dot(k, Nmat * k))
            L = Matrix{S}(I, r, r) .- k * z'
            rvec = z .* (v[t, i] / f) .+ L' * rvec
            Nmat = (z * z') ./ f .+ L' * Nmat * L
        end
        alpha[:, t] = a_pred[t] .+ P_pred[t] * rvec
        V[t] = P_pred[t] .- P_pred[t] * Nmat * P_pred[t]
    end
    return (alpha, V, eta, eta_var, eps, eps_var, true)
end
