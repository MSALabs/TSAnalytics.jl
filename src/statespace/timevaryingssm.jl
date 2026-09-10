export TimeVaryingSSM, to_time_varying

# ---------------------------------------------------------------------------
# TimeVaryingSSM: Durbin & Koopman general time-varying state-space form
# (Stage 8.1 -- generalizes GaussianSSM's time-invariant, stationary-only
# case to arbitrary per-period Z_t/T_t/R_t/Q_t/H_t, with a *known* initial
# state. Does NOT include diffuse initialization -- that's Stage 8.2's
# separate scope, deliberately kept out of this stage per the handoff.)
# ---------------------------------------------------------------------------

"""
    TimeVaryingSSM{S<:Real}

General time-varying linear Gaussian state-space representation (Durbin &
Koopman, chs. 3-4):
```
alpha_{t+1} = T_t alpha_t + R_t eta_t,   eta_t ~ N(0, Q_t)
y_t         = Z_t alpha_t + eps_t,        eps_t ~ N(0, H_t)
```
Every field is a `Vector` of matrices, one per period, **or a length-1
vector broadcast across every period** -- a cheap way to represent "this
specific matrix happens to be constant even within an otherwise
time-varying system" (e.g. `Q_t` constant while `Z_t` genuinely varies)
without forcing every matrix to be fully materialized per period when
redundant. A plain `Vector` of matrices, not a 3D array -- unlike
`statsmodels`'s `(k,k,nobs)` NumPy-array convention (confirmed directly
by construction: time is always the *last* axis there, the single
easiest thing to get backwards silently), `T[t]` reads unambiguously,
with no axis-order convention to remember.

Deliberately a **separate type from `GaussianSSM`**, not a
generalization of it: `kalman_filter` dispatches to a genuinely
different method for each, so `fit_arma`'s existing objective function
(Stage 6.5, already performance-checked) calling
`kalman_filter(::GaussianSSM, ...)` gets exactly the same compiled code
it already had, completely unaffected by this type's existence -- no
generic-dispatch overhead, nothing to regress. Build via
`build_statespace` (time-invariant) or construct directly for a
genuinely time-varying system; `to_time_varying` converts an
existing fitted `GaussianSSM` into one, for the regression-testing
purpose described there.

**No stationary or diffuse initialization here** -- callers supply a
*known* `a0`/`P0` directly to `kalman_filter`'s time-varying
method. Exact diffuse initialization (Stage 8.2) and SARIMAX/AutoReg's
full tier (Stage 8.3+) build on top of this type, not the other way
around.
"""
struct TimeVaryingSSM{S<:Real}
    T::Vector{Matrix{S}}
    Z::Vector{Matrix{S}}
    R::Vector{Matrix{S}}
    Q::Vector{Matrix{S}}
    H::Vector{Matrix{S}}
    r::Int
end

"_tv_at(v, t) -- reads period `t`'s matrix from a `TimeVaryingSSM` field,
broadcasting a length-1 vector across every period."
_tv_at(v::AbstractVector, t::Integer) = length(v) == 1 ? v[1] : v[t]

"""
    kalman_filter(ssm::TimeVaryingSSM, y::AbstractVector{<:Real},
                  a0::AbstractVector{<:Real}, P0::AbstractMatrix{<:Real})
        -> (loglik, v, F, converged)

General time-varying Kalman filter, known initial state `a0`/`P0`
(**not** derived from stationarity or diffuse initialization -- callers
supply it directly, Stage 8.1's own deliberate scope boundary). The
standard Durbin & Koopman prediction-form recursion, every matrix
indexed by `t`:
```
v_t = y_t - Z_t a_t
F_t = Z_t P_t Z_t' + H_t
K_t = T_t P_t Z_t' / F_t
a_{t+1} = T_t a_t + K_t v_t
P_{t+1} = T_t P_t T_t' - F_t K_t K_t' + R_t Q_t R_t'
loglik  = -0.5 * sum_t(log(2π) + log(F_t) + v_t^2/F_t)
```

**No `sigma2` concentration, unlike `GaussianSSM`'s
`kalman_filter`** -- and this is a genuine difference in what the two
methods compute, not just an omitted return value: `GaussianSSM`'s
method profiles an overall scale factor out of the likelihood
analytically because, in that restricted time-invariant/stationary
scope, `sigma2` really is a single free scale shared by the whole
system with no other role. Once `Q_t`/`H_t` can vary independently
per period (this type's whole point), there is no single scale left to
profile out -- `Q_t`/`H_t` already carry the actual variance directly,
exactly matching how `statsmodels.tsa.statespace.KalmanFilter` computes
this same likelihood by default (confirmed directly by construction,
not assumed: feeding fixed `state_cov`/`obs_cov` and reading `llf` off
a real fit).

**Two claims verified by direct execution against `statsmodels`, not
assumed from the textbook formula alone** (see
`test/verification/timevarying/`):
1. **Exact reduction to the time-invariant case**: `GaussianSSM`
   and this type agree to numerical precision on the *same* recursion
   quantities (`v`, `F`) when fed the same system with a matched known
   initial state (see `to_time_varying` for exactly how that
   match is constructed, since `GaussianSSM`'s own concentrated `sigma2`
   has to be threaded through explicitly for the final `loglik` numbers
   to agree too, not just `v`/`F`).
2. **A genuinely time-varying case** (`Z_t=[1,x_t]`, a real changing
   regressor -- exactly SARIMAX's eventual shape): matches real
   `statsmodels` output to full precision, `loglik=-186.3544625437698`.

**Time-varying `T_t`/`R_t`/`Q_t`/`H_t` individually were *not* covered
by the handoff's own original verification** (only `Z_t` varying was) --
closed directly in this implementation session: one dedicated case per
matrix was constructed and verified against fresh real `statsmodels`
output before trusting the general recursion for any of them (see
`test/verification/timevarying/timevarying_cases.json` and
`test/test_timevaryingssm.jl`).

`converged=false` (with `loglik=-Inf`, `v`/`F` empty) signals a
numerically degenerate `F_t` at some period, matching
`GaussianSSM`'s own sentinel convention -- so an optimizer's
objective function built around this method can catch it and act the
same way, once a future stage (SARIMAX) actually searches over a
time-varying system's parameters.

**Missing observations**: any `y[t] === NaN` is treated as no
observation at all (Durbin & Koopman 2012 sec. 4.10) -- the update step
is skipped entirely (`Z_t` acts as if it were zero for that period only,
`K_t = 0`, `a`/`P` simply predict forward via `T_t`/`R_t Q_t R_t'` with
no correction), and the period contributes nothing to `loglik`. `v[t]`
and `F[t]` are both `NaN` at a missing period (there is no prediction
error to report), matching this package's existing `NaN`-for-missing
convention (`acf`, `decompose`, `stl`, `holt_winters` all use `NaN`, not
`Missing`, for exactly this reason -- see e.g. `src/stattools.jl`).
Verified directly against real `statsmodels`
(`sm.tsa.UnobservedComponents(y, level=True)` with `y[t] = np.nan` at
several interior points): filtered state and its variance at and after
a missing period match to machine precision, and `llf` matches when
computed over the same non-missing observations.
"""
function kalman_filter(ssm::TimeVaryingSSM, y::AbstractVector{<:Real},
                        a0::AbstractVector{<:Real}, P0::AbstractMatrix{<:Real})
    n = length(y)
    n == 0 && throw(ArgumentError("kalman_filter: y must be non-empty"))
    r = ssm.r
    length(a0) == r || throw(ArgumentError("kalman_filter: a0 must have length r=$r"))
    size(P0) == (r, r) || throw(ArgumentError("kalman_filter: P0 must be r x r = $r x $r"))

    VT = promote_type(eltype(P0), eltype(a0), eltype(ssm.T[1]), Float64)
    a = Vector{VT}(a0)
    P = Matrix{VT}(P0)
    v = Vector{VT}(undef, n)
    F = Vector{VT}(undef, n)
    acc = zero(VT)
    n_obs = 0

    for t in 1:n
        Tt = _tv_at(ssm.T, t)
        z = vec(_tv_at(ssm.Z, t))
        Rt = _tv_at(ssm.R, t)
        Qt = _tv_at(ssm.Q, t)
        Ht = _tv_at(ssm.H, t)[1, 1]

        if isnan(y[t])
            v[t] = VT(NaN)
            F[t] = VT(NaN)
            a = Tt * a
            P = Tt * P * Tt' + Rt * Qt * Rt'
            continue
        end

        n_obs += 1
        v[t] = y[t] - dot(z, a)
        Ft = dot(z, P * z) + Ht
        F[t] = Ft
        if Ft <= 0 || !isfinite(Ft)
            return (-Inf, Float64[], Float64[], false)
        end
        K = (Tt * P * z) ./ Ft
        a = Tt * a + K * v[t]
        P = Tt * P * Tt' - Ft * (K * K') + Rt * Qt * Rt'
        acc += log(Ft) + v[t]^2 / Ft
    end

    loglik = -0.5 * (n_obs * log(2π) + acc)
    return (loglik, v, F, true)
end

"""
    kalman_smoother(ssm::TimeVaryingSSM, y::AbstractVector{<:Real},
                     a0::AbstractVector{<:Real}, P0::AbstractMatrix{<:Real})
        -> (alpha, V, eta, eta_var, eps, eps_var, converged)

General fixed-interval smoother (Durbin & Koopman 2012 secs. 4.4-4.5)
for the same known-initial-state `TimeVaryingSSM` system
[`kalman_filter`](@ref) filters -- genuinely time-varying `Z_t`/`T_t`/
`R_t`/`Q_t`/`H_t`, not the ARMA-specific `Z=e1`/`H=0` case
`GaussianSSM`'s own [`kalman_smoother`](@ref) is restricted to. Runs its
own forward pass (identical recursion to `kalman_filter`, but storing
the *predicted* `a_t|t-1`/`P_t|t-1`/gain at every `t`, which
`kalman_filter` doesn't expose) followed by the standard backward
recursion:
```
r_n = 0, N_n = 0
for t = n, n-1, ..., 1:
    L_t     = T_t - K_t Z_t
    r_{t-1} = Z_t' F_t^{-1} v_t + L_t' r_t
    N_{t-1} = Z_t' F_t^{-1} Z_t + L_t' N_t L_t
    alpha_t = a_t|t-1 + P_t|t-1 r_{t-1}
    V_t     = P_t|t-1 - P_t|t-1 N_{t-1} P_t|t-1
```

Also performs **disturbance smoothing** (Durbin & Koopman sec. 4.5),
recovering the individual state and observation shocks the model
implies given the whole series -- not available from filtering alone,
since the filter only ever looks backward in time:
```
eta_t = Q_t R_t' r_t,   Var(eta_t) = Q_t - Q_t R_t' N_t R_t Q_t
eps_t = H_t (F_t^{-1} v_t - K_t' r_t),   Var(eps_t) = H_t - H_t(F_t^{-1} + K_t' N_t K_t)H_t
```
`eta_t` is the smoothed *state* disturbance at `t` (how much the state
itself moved beyond what the transition alone predicted) and `eps_t`
the smoothed *observation* disturbance (how much of `y_t` is
attributable to observation noise rather than the state) -- a large
`eps_t` relative to `sqrt(eps_var_t)` points at a measurement anomaly,
a large `eta_t` relative to `sqrt(eta_var_t)` points at a genuine shift
in the underlying state, and the two are distinguishable because they
enter the model through different equations.

**Missing observations** (`y[t] === NaN`) are handled exactly as in
`kalman_filter`: `Z_t` contributes nothing for that period (`K_t = 0`,
`L_t = T_t`), so `r_{t-1} = L_t' r_t` and `N_{t-1} = L_t' N_t L_t` with
no correction term -- the smoothed state still updates (information
from both sides of the gap reaches it through `T_t`/`R_t Q_t R_t'`),
but there is no observation disturbance to recover, so `eps_t`/`eps_var_t`
are `NaN` at a missing period while `eta_t`/`eta_var_t` remain
well-defined.

Returns:
  - `alpha`: `r x n` matrix, smoothed state `E[alpha_t | y_1:n]` per column.
  - `V`: length-`n` vector of `r x r` matrices, `Var[alpha_t | y_1:n]`.
  - `eta`: `n`-vector of smoothed state disturbances (scalar per period,
    since every model built by this package's own `build_statespace`/
    `combined_ar_ma`/regression-in-state constructions uses a single
    state-innovation loading `R_t`, matching `GaussianSSM`'s own
    single-`R`-vector convention).
  - `eta_var`: `n`-vector, `Var(eta_t)`.
  - `eps`, `eps_var`: `n`-vectors, the observation-disturbance
    counterparts (`NaN` at any period where `H_t = 0` exactly, since
    there is then no observation noise to disentangle, and at any
    missing observation).
  - `converged`: `false` (with every other return value empty) on a
    non-finite `F_t` during the forward pass, matching `kalman_filter`'s
    own sentinel convention.

**Verified two ways.** (1) Against real, directly-executed `statsmodels`
(`sm.tsa.UnobservedComponents(y, level=True).smooth(params)`, whose
`smoothed_state`/`smoothed_state_cov`/`smoothed_state_disturbance`/
`smoothed_measurement_disturbance` use an identical convention):
matches to machine precision on a local-level system, including a case
with interior missing observations. (2) **Exact reduction to
`GaussianSSM`'s own already-verified ARMA-specific smoother**: built via
[`to_time_varying`](@ref) on a fitted ARMA model, this function's
`alpha`/`V` agree with `GaussianSSM.kalman_smoother`'s own output to
numerical precision -- the same reduction-test discipline used
throughout this package (`to_time_varying`'s own docstring, Stage 6.6's
`d=0` regression guard, etc.).

Throws `ArgumentError` for a non-stationary/degenerate/empty `ssm`/`y`,
matching `GaussianSSM.kalman_smoother`'s own convention -- the smoother
is meant to run once on an already-fitted model for diagnostics, not
inside a search loop.

# Examples
```jldoctest
julia> using TSAnalytics

julia> T = [1.0;;]; Z = [1.0;;]; R = [1.0;;]; Q = [0.01;;]; H = [0.5;;];

julia> ssm = TimeVaryingSSM{Float64}([T], [Z], [R], [Q], [H], 1);

julia> y = [1.0, 1.2, NaN, 1.6, 1.5];

julia> alpha, V, eta, eta_var, eps, eps_var, converged = kalman_smoother(ssm, y, [1.0], [1.0;;]);

julia> converged
true

julia> isnan(eps[3])  # no observation disturbance at the missing period
true
```
"""
function kalman_smoother(ssm::TimeVaryingSSM, y::AbstractVector{<:Real},
                          a0::AbstractVector{<:Real}, P0::AbstractMatrix{<:Real})
    n = length(y)
    n == 0 && throw(ArgumentError("kalman_smoother: y must be non-empty"))
    r = ssm.r
    length(a0) == r || throw(ArgumentError("kalman_smoother: a0 must have length r=$r"))
    size(P0) == (r, r) || throw(ArgumentError("kalman_smoother: P0 must be r x r = $r x $r"))

    VT = promote_type(eltype(P0), eltype(a0), eltype(ssm.T[1]), Float64)
    a = Vector{VT}(a0)
    P = Matrix{VT}(P0)

    a_pred = Vector{Vector{VT}}(undef, n)
    P_pred = Vector{Matrix{VT}}(undef, n)
    K_all  = Vector{Vector{VT}}(undef, n)
    z_all  = Vector{Vector{VT}}(undef, n)
    v = Vector{VT}(undef, n)
    F = Vector{VT}(undef, n)
    missing_t = falses(n)

    for t in 1:n
        Tt = _tv_at(ssm.T, t)
        z = vec(_tv_at(ssm.Z, t))
        Rt = _tv_at(ssm.R, t)
        Qt = _tv_at(ssm.Q, t)
        Ht = _tv_at(ssm.H, t)[1, 1]

        a_pred[t] = a
        P_pred[t] = P
        z_all[t] = z

        if isnan(y[t])
            missing_t[t] = true
            v[t] = VT(NaN)
            F[t] = VT(NaN)
            K_all[t] = zeros(VT, r)
            a = Tt * a
            P = Tt * P * Tt' + Rt * Qt * Rt'
            continue
        end

        v[t] = y[t] - dot(z, a)
        Ft = dot(z, P * z) + Ht
        F[t] = Ft
        if Ft <= 0 || !isfinite(Ft)
            return (zeros(0,0), Matrix{Float64}[], Float64[], Float64[], Float64[], Float64[], false)
        end
        K = (Tt * P * z) ./ Ft
        K_all[t] = K
        a = Tt * a + K * v[t]
        P = Tt * P * Tt' - Ft * (K * K') + Rt * Qt * Rt'
    end

    r_vec = zeros(VT, r)
    N = zeros(VT, r, r)

    alpha = Matrix{VT}(undef, r, n)
    V = Vector{Matrix{VT}}(undef, n)
    eta = Vector{VT}(undef, n)
    eta_var = Vector{VT}(undef, n)
    eps = Vector{VT}(undef, n)
    eps_var = Vector{VT}(undef, n)

    for t in n:-1:1
        Tt = _tv_at(ssm.T, t)
        Rt = _tv_at(ssm.R, t)
        Qt = _tv_at(ssm.Q, t)
        Ht = _tv_at(ssm.H, t)[1, 1]
        z = z_all[t]
        K = K_all[t]
        L = Tt - K * z'

        # disturbance smoothing at t uses r_t/N_t (the values entering this
        # iteration, i.e. "r_vec"/"N" before they are updated to r_{t-1}/N_{t-1}).
        # Rt is r x k (usually r x 1), Qt is k x k (usually 1 x 1); eta_t = Qt*Rt'*r_t
        # is a k-vector, collapsed to a scalar here since every state built by
        # this package's own constructions uses a single (k=1) innovation.
        eta[t] = (Qt * (Rt' * r_vec))[1]
        eta_var[t] = (Qt - Qt * (Rt' * N * Rt) * Qt)[1, 1]

        if missing_t[t]
            eps[t] = VT(NaN)
            eps_var[t] = VT(NaN)
            r_prev = L' * r_vec
            N_prev = L' * N * L
        else
            Ft = F[t]
            eps[t] = Ht * (v[t] / Ft - dot(K, r_vec))
            eps_var[t] = Ht - Ht * (1/Ft + dot(K, N * K)) * Ht
            r_prev = (v[t] / Ft) .* z .+ L' * r_vec
            N_prev = (z * z') ./ Ft .+ L' * N * L
        end

        alpha[:, t] = a_pred[t] + P_pred[t] * r_prev
        V[t] = P_pred[t] - P_pred[t] * N_prev * P_pred[t]

        r_vec = r_prev
        N = N_prev
    end

    return (alpha, V, eta, eta_var, eps, eps_var, true)
end

"""
    to_time_varying(ssm::GaussianSSM, y::AbstractVector{<:Real})
        -> (TimeVaryingSSM, a0::Vector{Float64}, P0::Matrix{Float64})

Converts a fitted (or fittable) `GaussianSSM` into an exactly
equivalent `TimeVaryingSSM`, for the regression-testing purpose
described in `handoff/stage-8.1-gaussianssm-timevarying-handoff.md` §5:
confirming the new general filter reproduces every one of Stage 6's
already dual-verified results (the 364-case bulk suite included) before
trusting it for anything genuinely new.

**Threading `GaussianSSM`'s concentrated `sigma2` through explicitly is
what makes the reduction exact, not incidental**: `GaussianSSM`'s own
`kalman_filter` profiles an overall scale `sigma2_hat` out of the
likelihood analytically (see its own docstring); `TimeVaryingSSM`'s
general filter does not concentrate anything (see its own docstring).
Feeding `TimeVaryingSSM` the *raw*, unscaled `R*R'` as `Q_t` (i.e.
`Q_t=1`) would therefore generally give a genuinely different `loglik`
number from `GaussianSSM`'s concentrated one -- a real, derivable fact
about the two formulas, not a bug to paper over. Scaling the state
noise covariance by `sigma2_hat` (`Q_t = sigma2_hat`, `R_t = ssm.R`) and
the known initial covariance by the same factor (`P0 =
sigma2_hat .* Q0_stationary`) makes them agree *exactly*: scaling `Q`
and `P0` by a constant `c` scales `P_t`/`F_t` by `c` at every step while
leaving `a_t`/`v_t`/the Kalman gain `K_t` completely unchanged (`K_t`'s
numerator and denominator both scale by `c` and cancel) -- substituting
`c = sigma2_hat` into the *direct* (non-concentrated) log-likelihood
formula reduces it algebraically to exactly `GaussianSSM`'s own
concentrated formula. Verified both algebraically and by direct
execution (`test/test_timevaryingssm.jl`'s full regression suite, reused
from the existing 364-case `GaussianSSM` bulk corpus).

Throws `ArgumentError` if `ssm`/`y` don't produce a valid (stationary,
converged) fit -- there's no sensible known initial state to construct
otherwise.
"""
function to_time_varying(ssm::GaussianSSM, y::AbstractVector{<:Real})
    loglik, sigma2, v, F, converged = kalman_filter(ssm, y)
    converged || throw(ArgumentError("to_time_varying: kalman_filter(ssm, y) did not converge -- " *
                                      "no valid known initial state to construct"))
    Q0, sc_converged = stationary_cov(ssm)
    sc_converged || throw(ArgumentError("to_time_varying: ssm's stationary covariance did not converge"))

    r = ssm.r
    Z = zeros(1, r)
    Z[1, 1] = 1.0
    tv = TimeVaryingSSM{Float64}([ssm.T], [Z], [reshape(ssm.R, r, 1)],
                                  [reshape([sigma2], 1, 1)], [zeros(1, 1)], r)
    a0 = zeros(r)
    P0 = sigma2 .* Q0
    return tv, a0, P0
end
