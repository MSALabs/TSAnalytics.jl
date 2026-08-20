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

    for t in 1:n
        Tt = _tv_at(ssm.T, t)
        z = vec(_tv_at(ssm.Z, t))
        Rt = _tv_at(ssm.R, t)
        Qt = _tv_at(ssm.Q, t)
        Ht = _tv_at(ssm.H, t)[1, 1]

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

    loglik = -0.5 * (n * log(2π) + acc)
    return (loglik, v, F, true)
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
