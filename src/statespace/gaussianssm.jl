

using LinearAlgebra: I, kron

export GaussianSSM, build_statespace, stationary_cov, kalman_filter, kalman_smoother, combined_ar_ma

# ---------------------------------------------------------------------------
# Polynomial algebra for the multiplicative seasonal model
# (needed to build the combined AR/MA coefficients SARIMA requires;
# pure ARMA callers can skip straight to build_statespace with plain
# ar/ma vectors)
# ---------------------------------------------------------------------------

"""Convolution of two polynomials given as coefficient vectors, constant
term first (index 1 = coefficient of B^0)."""
function polymul(a::Vector{Float64}, b::Vector{Float64})
    na, nb = length(a), length(b)
    c = zeros(Float64, na + nb - 1)
    for i in 1:na, j in 1:nb
        c[i+j-1] += a[i] * b[j]
    end
    return c
end

"""Place seasonal AR/MA coefficients at lags s, 2s, 3s, ... within a
(1 + ...) polynomial. `sign=-1.0` for AR-style polynomials (so the
returned state-space coefficients come out with the conventional
positive sign after negation in `combined_ar_ma`), `sign=1.0` for MA."""
function seasonal_poly(coefs::Vector{Float64}, s::Int; sign::Float64=1.0)
    P = length(coefs)
    poly = zeros(Float64, P*s + 1)
    poly[1] = 1.0
    for i in 1:P
        poly[i*s + 1] = sign * coefs[i]
    end
    return poly
end

"""
    combined_ar_ma(phi, Phi, theta, Theta, s) -> (ar, ma)

Multiplies the regular and seasonal AR/MA polynomials to obtain the
combined ARMA coefficients used in the state-space form, following the
standard multiplicative SARIMA convention:
    phi(B) Phi(B^s) w_t = theta(B) Theta(B^s) e_t

Re-verified with its own dedicated fixed-coefficient R/Python comparison
across many seasonal structures and two datasets -- see
verification/bulk/. 364 dual-verified (R + Python) cases now cover this
function, including its previously-untested seasonal path.

For a *non-seasonal* ARMA(p,q) -- i.e. Phi = Theta = Float64[], s
irrelevant -- this reduces to just `ar = phi`, `ma = theta` (covered by
test/test_gaussianssm.jl's degenerate-case check).
"""
function combined_ar_ma(phi::Vector{Float64}, Phi::Vector{Float64},
                         theta::Vector{Float64}, Theta::Vector{Float64}, s::Int)
    ar_reg  = vcat([1.0], -phi)
    ar_seas = seasonal_poly(Phi, s; sign=-1.0)
    ma_reg  = vcat([1.0], theta)
    ma_seas = seasonal_poly(Theta, s; sign=1.0)

    ar_full = polymul(ar_reg, ar_seas)
    ma_full = polymul(ma_reg, ma_seas)

    ar = -ar_full[2:end]   # state-space AR coefficients (multiply lagged y)
    ma = ma_full[2:end]    # state-space MA coefficients (multiply lagged innovations)
    return ar, ma
end

# ---------------------------------------------------------------------------
# GaussianSSM: Harvey (1993) state-space form, time-invariant case only
# (per the project brief's scope boundary -- no time-varying Z/T/R here,
# that's Stage 8's generalization, deliberately out of scope for this track)
# ---------------------------------------------------------------------------

"""
    GaussianSSM

Time-invariant linear Gaussian state-space representation for a
stationary ARMA(p*, q*) process, Harvey (1993) companion form. `r =
max(p*, q*+1)`. Build via [`build_statespace`](@ref) rather than the
constructor directly.
"""
struct GaussianSSM
    T::Matrix{Float64}   # r x r transition
    R::Vector{Float64}   # r-vector, innovation loading
    r::Int
end

"""
    build_statespace(ar, ma) -> GaussianSSM

Harvey (1993) state-space form for a stationary/invertible ARMA(p*,q*)
process with combined coefficients `ar`, `ma` (already-combined, i.e.
the output of [`combined_ar_ma`](@ref) for a seasonal model, or the raw
AR/MA coefficients directly for plain ARMA).
"""
function build_statespace(ar::Vector{Float64}, ma::Vector{Float64})
    p = length(ar)
    q = length(ma)
    r = max(p, q + 1)

    phi_ext = zeros(Float64, r); phi_ext[1:p] .= ar
    theta_ext = zeros(Float64, r); theta_ext[1] = 1.0; theta_ext[2:q+1] .= ma

    T = zeros(Float64, r, r)
    for i in 1:r
        T[i, 1] = phi_ext[i]
    end
    for i in 1:(r-1)
        T[i, i+1] = 1.0
    end
    R = theta_ext
    return GaussianSSM(T, R, r)
end

"""
    stationary_cov(ssm::GaussianSSM) -> Matrix{Float64}

Solves the discrete Lyapunov equation `Q0 = T*Q0*T' + R*R'` for the
stationary state covariance (with sigma^2 = 1, concentrated out
separately), via vectorization: `vec(Q0) = (I - T⊗T)^{-1} vec(RR')`.

Returns a matrix that may not be positive definite if `ssm`'s AR
polynomial isn't actually stationary (e.g. during an optimizer's search
before Stage 4.2's `partrans` constrains it) -- callers should check
before using it, see `kalman_filter`'s own guard for the expected pattern.
"""
function stationary_cov(ssm::GaussianSSM)
    r = ssm.r
    A = kron(ssm.T, ssm.T)
    rhs = vec(ssm.R * ssm.R')
    vecQ0 = (I(r*r) - A) \ rhs
    return reshape(vecQ0, r, r)
end

"""
    kalman_filter(ssm::GaussianSSM, y::Vector{Float64}) -> (loglik, sigma2, v, F, converged)

Exact Gaussian log-likelihood via the Kalman filter with stationary
initialization (Lyapunov equation) and sigma^2 concentrated out
analytically. Dual-verified (R's `arima()` and Python's
`statsmodels SARIMAX`) across 364+ cases -- see verification/bulk/ and
test/test_gaussianssm_bulk.jl -- covering pure AR(1-3), pure MA(1-3),
ARMA up to (2,2), and seasonal SARIMA, across synthetic and real/textbook
datasets (Nile, sunspots, macro GDP growth, El Nino SST, Mauna Loa CO2).

Returns `v`/`F` (the one-step prediction errors and their variances at
every time point) in addition to the scalar `loglik`/`sigma2`/`converged`
-- useful for residual diagnostics and consumed internally by
`kalman_smoother`'s independent forward pass.

`converged=false` (with `loglik=-Inf`) signals a non-stationary or
numerically degenerate `ssm` -- e.g. fed AR coefficients outside the
stationary region -- rather than throwing, so an optimizer's objective
function can penalize it smoothly instead of crashing mid-search.
"""
function kalman_filter(ssm::GaussianSSM, y::Vector{Float64})
    n = length(y)
    r = ssm.r
    T = ssm.T
    R = ssm.R

    Q0 = try
        stationary_cov(ssm)
    catch
        return (-Inf, NaN, Float64[], Float64[], false)
    end
    if Q0[1,1] <= 0 || !all(isfinite, Q0)
        return (-Inf, NaN, Float64[], Float64[], false)
    end

    a = zeros(Float64, r)
    P = Q0
    v = Vector{Float64}(undef, n)
    F = Vector{Float64}(undef, n)

    for t in 1:n
        v[t] = y[t] - a[1]
        F[t] = P[1, 1]
        if F[t] <= 0 || !isfinite(F[t])
            return (-Inf, NaN, v[1:t], F[1:t], false)
        end
        PZ = P[:, 1]
        K = (T * PZ) / F[t]
        a = T * a + K * v[t]
        P = T * P * T' - F[t] * (K * K') + R * R'
    end

    sigma2 = sum(v[t]^2 / F[t] for t in 1:n) / n
    if sigma2 <= 0 || !isfinite(sigma2)
        return (-Inf, NaN, v, F, false)
    end
    loglik = -0.5 * (n*log(2π) + n*log(sigma2) + sum(log.(F)) + n)
    return (loglik, sigma2, v, F, true)
end

"""
    kalman_smoother(ssm::GaussianSSM, y::Vector{Float64}) -> (alpha::Matrix{Float64}, V::Vector{Matrix{Float64}})

Fixed-interval Kalman smoother (Durbin & Koopman 2001, Sec. 4.4 classic
backward recursion), for the same time-invariant stationary state-space
model `kalman_filter` evaluates. Runs its own forward pass (stationary
Lyapunov initialization, identical to `kalman_filter`) rather than reusing
`kalman_filter`'s return values, since the backward recursion needs the
*predicted* state/covariance/gain at every t (`a_t|t-1`, `P_t|t-1`, `K_t`),
which `kalman_filter` doesn't expose.

Backward recursion, with `Z = e1` (observation picks the first state
element) and `L_t = T - K_t*Z'`:
```
r_n = 0, N_n = 0
r_{t-1} = v_t/F_t * e1 + L_t' r_t
N_{t-1} = 1/F_t * e1 e1' + L_t' N_t L_t
alpha_t = a_t|t-1 + P_t|t-1 * r_{t-1}
V_t     = P_t|t-1 - P_t|t-1 * N_{t-1} * P_t|t-1
```

Returns:
  - `alpha`: `r x n` matrix, smoothed state `E[alpha_t | y_1:n]` per column.
    `alpha[1, :]` reproduces `y` exactly -- there is no observation noise in
    this state-space form (`Z` picks a state element directly), so smoothing
    cannot change the observed component, only the unobserved ones.
  - `V`: length-`n` vector of `r x r` matrices, `Var[alpha_t | y_1:n]`,
    scaled by the concentrated `sigma^2` (unlike the filter's internal `P`,
    which is computed in `sigma^2 = 1` units and only scaled at the very
    end for `loglik`).

Verified against `statsmodels` `SARIMAX(...).smooth(params)`'s own
`smoothed_state`/`smoothed_state_cov` (which uses an identical
transition/selection/design convention) -- see
`verification/bulk/smoother_check/`.

Throws `ArgumentError` for a non-stationary/degenerate `ssm`. Unlike
`kalman_filter`'s `converged=false` sentinel (meant to let an optimizer's
objective function penalize smoothly mid-search), the smoother is meant to
run once on an already-valid model for diagnostics, so throwing here is
fine -- there's no search loop to keep alive.
"""
function kalman_smoother(ssm::GaussianSSM, y::Vector{Float64})
    n = length(y)
    r = ssm.r
    T = ssm.T
    R = ssm.R

    Q0 = try
        stationary_cov(ssm)
    catch
        throw(ArgumentError("kalman_smoother: ssm is not stationary (Lyapunov equation has no solution)"))
    end
    if Q0[1,1] <= 0 || !all(isfinite, Q0)
        throw(ArgumentError("kalman_smoother: ssm is not stationary (degenerate stationary covariance)"))
    end

    a_pred = Vector{Vector{Float64}}(undef, n)
    P_pred = Vector{Matrix{Float64}}(undef, n)
    K_all  = Vector{Vector{Float64}}(undef, n)
    v = Vector{Float64}(undef, n)
    F = Vector{Float64}(undef, n)

    a = zeros(Float64, r)
    P = Q0
    for t in 1:n
        a_pred[t] = a
        P_pred[t] = P
        v[t] = y[t] - a[1]
        F[t] = P[1, 1]
        if F[t] <= 0 || !isfinite(F[t])
            throw(ArgumentError("kalman_smoother: degenerate F[$t] <= 0 during forward pass"))
        end
        K = (T * P[:, 1]) / F[t]
        K_all[t] = K
        a = T * a + K * v[t]
        P = T * P * T' - F[t] * (K * K') + R * R'
    end

    sigma2 = sum(v[t]^2 / F[t] for t in 1:n) / n
    if sigma2 <= 0 || !isfinite(sigma2)
        throw(ArgumentError("kalman_smoother: degenerate concentrated sigma2"))
    end

    e1 = zeros(Float64, r); e1[1] = 1.0

    r_vec = zeros(Float64, r)
    N = zeros(Float64, r, r)

    alpha = Matrix{Float64}(undef, r, n)
    V = Vector{Matrix{Float64}}(undef, n)

    for t in n:-1:1
        L = T - K_all[t] * e1'
        r_prev = (v[t] / F[t]) .* e1 .+ L' * r_vec
        N_prev = (1.0 / F[t]) .* (e1 * e1') .+ L' * N * L

        alpha[:, t] = a_pred[t] + P_pred[t] * r_prev
        V[t] = sigma2 .* (P_pred[t] - P_pred[t] * N_prev * P_pred[t])

        r_vec = r_prev
        N = N_prev
    end

    return alpha, V
end

end # module GaussianSSMs
