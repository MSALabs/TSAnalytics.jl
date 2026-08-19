# ---------------------------------------------------------------------------
# Polynomial algebra for the multiplicative seasonal model
# (needed to build the combined AR/MA coefficients SARIMA requires;
# pure ARMA callers can skip straight to build_statespace with plain
# ar/ma vectors)
# ---------------------------------------------------------------------------

"_gssm_elt(v) -- `eltype(v)`, unless `v` is empty and its declared eltype
is `Any` (a bare `[]` literal), in which case `Float64` -- an empty
`Vector{Any}` never actually contributes a value to a `promote_type`
computation, but `zeros(Any, ...)` itself throws (`zero(::Type{Any})`
doesn't exist), so the `Any` has to be filtered out before it ever
reaches `zeros`/`promote_type`."
_gssm_elt(v::AbstractVector) = (isempty(v) && eltype(v) === Any) ? Float64 : eltype(v)

"""Convolution of two polynomials given as coefficient vectors, constant
term first (index 1 = coefficient of B^0). `AbstractVector`, not
`AbstractVector{<:Real}`, deliberately -- so a bare `[]` literal
(`Vector{Any}`) works for an absent polynomial, not just `Float64[]`."""
function polymul(a::AbstractVector, b::AbstractVector)
    na, nb = length(a), length(b)
    c = zeros(promote_type(_gssm_elt(a), _gssm_elt(b), Float64), na + nb - 1)
    for i in 1:na, j in 1:nb
        c[i+j-1] += a[i] * b[j]
    end
    return c
end

"""Place seasonal AR/MA coefficients at lags s, 2s, 3s, ... within a
(1 + ...) polynomial. `sign=-1.0` for AR-style polynomials (so the
returned state-space coefficients come out with the conventional
positive sign after negation in `combined_ar_ma`), `sign=1.0` for MA."""
function seasonal_poly(coefs::AbstractVector, s::Integer; sign::Real=1.0)
    isempty(coefs) || s >= 1 ||
        throw(ArgumentError("seasonal_poly: s must be >= 1 when seasonal coefficients are given (got s=$s)"))
    P = length(coefs)
    poly = zeros(promote_type(_gssm_elt(coefs), typeof(sign), Float64), P*s + 1)
    poly[1] = 1.0
    for i in 1:P
        poly[i*s + 1] = sign * coefs[i]
    end
    return poly
end

"""
    combined_ar_ma(phi, Phi, theta, Theta, s) -> (ar, ma)
    combined_ar_ma(; phi, seasonal_phi, theta, seasonal_theta, s) -> (ar, ma)

Multiplies the regular and seasonal AR/MA polynomials to obtain the
combined ARMA coefficients used in the state-space form, following the
standard multiplicative SARIMA convention:
    phi(B) Phi(B^s) w_t = theta(B) Theta(B^s) e_t

Re-verified with its own dedicated fixed-coefficient R/Python comparison
across many seasonal structures and two datasets -- see
`test/verification/gaussianssm/bulk/`. 364 dual-verified (R + Python)
cases now cover this function, including its previously-untested
seasonal path.

For a *non-seasonal* ARMA(p,q) -- i.e. Phi = Theta = Float64[], s
irrelevant -- this reduces to just `ar = phi`, `ma = theta` (covered by
`test/test_gaussianssm.jl`'s degenerate-case check).

The keyword form exists because the positional order here
(`phi, Phi, theta, Theta`) is a transposition away from the ARIMA
optimizer's own parameter-vector layout (`phi, theta, Phi, Theta` --
see `handoff/stage-6-arima-handoff.md` §4.2) -- both are
`AbstractVector`, so a swap type-checks silently and returns plausible
garbage. Prefer the keyword form at any new call site; the positional
form stays available (and is what the generated 364-case corpus in
`test_gaussianssm_bulk.jl` uses) rather than being removed.
"""
function combined_ar_ma(phi::AbstractVector, Phi::AbstractVector,
                         theta::AbstractVector, Theta::AbstractVector, s::Integer)
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

combined_ar_ma(; phi, theta, seasonal_phi=Float64[], seasonal_theta=Float64[], s::Integer=1) =
    combined_ar_ma(phi, seasonal_phi, theta, seasonal_theta, s)

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

Parametrized on the element type (`GaussianSSM{S}`) rather than fixed to
`Float64`, so it works unchanged when `_optimize` (Stage 4.1) passes a
`Vector{ForwardDiff.Dual}` through the objective during optimization.
The type parameter is named `S`, not `T`, to avoid colliding with the
`T` field name (state-space literature convention for the transition
matrix, kept as-is).
"""
struct GaussianSSM{S<:Real}
    T::Matrix{S}   # r x r transition
    R::Vector{S}   # r-vector, innovation loading
    r::Int
end

"""
    build_statespace(ar, ma) -> GaussianSSM

Harvey (1993) state-space form for a stationary/invertible ARMA(p*,q*)
process with combined coefficients `ar`, `ma` (already-combined, i.e.
the output of [`combined_ar_ma`](@ref) for a seasonal model, or the raw
AR/MA coefficients directly for plain ARMA).
"""
function build_statespace(ar::AbstractVector{<:Real}, ma::AbstractVector{<:Real})
    S = promote_type(eltype(ar), eltype(ma), Float64)
    p = length(ar)
    q = length(ma)
    r = max(p, q + 1)

    phi_ext = zeros(S, r); phi_ext[1:p] .= ar
    theta_ext = zeros(S, r); theta_ext[1] = one(S); theta_ext[2:q+1] .= ma

    T = zeros(S, r, r)
    for i in 1:r
        T[i, 1] = phi_ext[i]
    end
    for i in 1:(r-1)
        T[i, i+1] = one(S)
    end
    R = theta_ext
    return GaussianSSM{S}(T, R, r)
end

"""
    stationary_cov(ssm::GaussianSSM; tol=1e-15, maxiter=60) -> (Q0, converged)

Stationary state covariance solving `Q0 = T*Q0*T' + R*R'`, via the
doubling (Smith) iteration `Q <- Q + A*Q*A'`, `A <- A*A` rather than the
naive `vec(Q0) = (I - T kron T)^{-1} vec(RR')` solve (which is O(r^6)
time / O(r^4) memory -- a seasonal `SARIMA(1,1)x(1,1)_52` reaches
`r=54`, where that solve alone costs ~2 seconds per likelihood
evaluation; doubling costs ~1 millisecond, a 1874x measured speedup,
with agreement to 1.3e-16 -- see
`handoff/gaussianssm-performance-handoff.md` §2 and §3.2 of the Stage 6
handoff). Quadratically convergent: `A` becomes `T^(2^k)` each pass, so
~30 iterations reach machine precision for any spectral radius
meaningfully below 1.

`converged=false` means the iteration did not settle within `maxiter`
(in practice a non-stationary or unit-root `T`) or diverged to a
non-finite value -- callers must check it explicitly, see
`kalman_filter`'s guard, since (unlike the `kron` solve, which throws
`SingularException` on an exactly-singular system) this never throws.
Stress-tested across 9,743 non-stationary draws (AR(1-3), ARMA(2,2)):
the existing `Q0[1,1] <= 0 || !all(isfinite, Q0)` guard style rejects
every one, with or without this flag -- but the flag is
belt-and-braces, not redundant, once an optimizer explores parameter
space more adversarially than a uniform sweep.
"""
function stationary_cov(ssm::GaussianSSM{S}; tol::Real=1e-15, maxiter::Integer=60) where {S<:Real}
    Q = ssm.R * ssm.R'
    A = copy(ssm.T)
    converged = false
    for _ in 1:maxiter
        Qn = Q + A * Q * A'
        all(isfinite, Qn) || return (Qn, false)
        if maximum(abs, Qn .- Q) <= tol * max(maximum(abs, Qn), one(S))
            Q = Qn
            converged = true
            break
        end
        Q = Qn
        A = A * A
    end
    return ((Q .+ Q') ./ 2, converged)
end

"""
    kalman_filter(ssm::GaussianSSM, y::AbstractVector{<:Real}) -> (loglik, sigma2, v, F, converged)

Exact Gaussian log-likelihood via the Kalman filter with stationary
initialization (Lyapunov equation) and sigma^2 concentrated out
analytically. Dual-verified (R's `arima()` and Python's
`statsmodels SARIMAX`) across 364+ cases -- see
`test/verification/gaussianssm/bulk/` and `test_gaussianssm_bulk.jl` --
covering pure AR(1-3), pure MA(1-3), ARMA up to (2,2), and seasonal
SARIMA, across synthetic and real/textbook datasets (Nile, sunspots,
macro GDP growth, El Nino SST, Mauna Loa CO2).

Returns `v`/`F` (the one-step prediction errors and their variances at
every time point) in addition to the scalar `loglik`/`sigma2`/`converged`
-- useful for residual diagnostics and consumed internally by
`kalman_smoother`'s independent forward pass.

`converged=false` signals a non-stationary or numerically degenerate
`ssm` (e.g. fed AR coefficients outside the stationary region), or an
empty series, rather than throwing, so an optimizer's objective function
can catch it and act rather than crashing mid-search. In that case
`loglik=-Inf` and `v`/`F` are `Float64[]` (not a truncated prefix, on
every failure path, including a mid-series numerical breakdown) --
**`-Inf` is a hard rejection, not a smooth penalty**: it carries no
usable gradient direction and produces `NaN` derivatives under
ForwardDiff. `partrans` (Stage 4.2) is what actually keeps an optimizer
inside the stationary region during a search; a caller building an
objective function around this must supply its own smooth penalty for
the `converged=false` case rather than relying on `-Inf` itself to guide
the search.
"""
function kalman_filter(ssm::GaussianSSM, y::AbstractVector{<:Real})
    n = length(y)
    n == 0 && throw(ArgumentError("kalman_filter: y must be non-empty"))
    r = ssm.r
    T = ssm.T
    R = ssm.R

    Q0, sc_converged = stationary_cov(ssm)
    if !sc_converged || Q0[1,1] <= 0 || !all(isfinite, Q0)
        return (-Inf, NaN, Float64[], Float64[], false)
    end

    VT = eltype(Q0)   # Float64 normally; ForwardDiff.Dual during _optimize's AD pass
    a = zeros(VT, r)
    P = Q0
    v = Vector{VT}(undef, n)
    F = Vector{VT}(undef, n)

    for t in 1:n
        v[t] = y[t] - a[1]
        F[t] = P[1, 1]
        if F[t] <= 0 || !isfinite(F[t])
            return (-Inf, NaN, Float64[], Float64[], false)
        end
        PZ = P[:, 1]
        K = (T * PZ) / F[t]
        a = T * a + K * v[t]
        P = T * P * T' - F[t] * (K * K') + R * R'
    end

    sigma2 = sum(v[t]^2 / F[t] for t in 1:n) / n
    if sigma2 <= 0 || !isfinite(sigma2)
        return (-Inf, NaN, Float64[], Float64[], false)
    end
    loglik = -0.5 * (n*log(2π) + n*log(sigma2) + sum(log.(F)) + n)
    return (loglik, sigma2, v, F, true)
end

"""
    kalman_smoother(ssm::GaussianSSM, y::AbstractVector{<:Real}) -> (alpha::Matrix{Float64}, V::Vector{Matrix{Float64}})

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
`test/verification/gaussianssm/bulk/smoother_check/`.

Throws `ArgumentError` for a non-stationary/degenerate/empty `ssm`/`y`.
Unlike `kalman_filter`'s `converged=false` sentinel (meant to let an
optimizer's objective function catch it and act, mid-search), the
smoother is meant to run once on an already-valid model for diagnostics,
so throwing here is fine -- there's no search loop to keep alive.
"""
function kalman_smoother(ssm::GaussianSSM, y::AbstractVector{<:Real})
    n = length(y)
    n == 0 && throw(ArgumentError("kalman_smoother: y must be non-empty"))
    r = ssm.r
    T = ssm.T
    R = ssm.R

    Q0, sc_converged = stationary_cov(ssm)
    if !sc_converged
        throw(ArgumentError("kalman_smoother: ssm is not stationary (Lyapunov doubling iteration did not converge)"))
    end
    if Q0[1,1] <= 0 || !all(isfinite, Q0)
        throw(ArgumentError("kalman_smoother: ssm is not stationary (degenerate stationary covariance)"))
    end

    a_pred = Vector{Vector{Float64}}(undef, n)
    P_pred = Vector{Matrix{Float64}}(undef, n)
    K_all  = Vector{Vector{Float64}}(undef, n)
    v = Vector{Float64}(undef, n)
    F = Vector{Float64}(undef, n)

    a = zeros(eltype(Q0), r)
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
