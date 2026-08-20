# ---------------------------------------------------------------------------
# kalman_filter_diffuse: exact diffuse initialization (Stage 8.2), layered
# directly on top of Stage 8.1's TimeVaryingSSM recursion -- Durbin & Koopman
# (2012) ch. 5, univariate treatment (their ch. 6.4/Koopman & Durbin 2000),
# reimplemented natively from the textbook algorithm; the exact branching
# logic and array bookkeeping were cross-checked directly against
# `statsmodels`'s own real Cython source
# (statsmodels/tsa/statespace/_filters/_univariate_diffuse.pyx.in, fetched
# and read directly this session, not assumed from the textbook alone) --
# `statsmodels` is used only to confirm the algorithm and validate output
# numbers, never as code copied from.
# ---------------------------------------------------------------------------

"_TOLERANCE_DIFFUSE = 1e-10 -- confirmed directly from statsmodels's own
Cython source (`self.tolerance_diffuse = 1e-10`, `_kalman_filter.pyx.in`),
not a round number picked independently. Used both for the per-observation
`F_infty_t > tol` / `F_star_t > tol` branch decisions and for detecting
when `P_infty` has collapsed to (numerically) zero globally (`sum(Pinfty.^2)
> tol`, matching statsmodels's own Frobenius-norm-squared collapse check)."
const _TOLERANCE_DIFFUSE = 1e-10

"""
    kalman_filter_diffuse(ssm::TimeVaryingSSM, y::AbstractVector{<:Real};
                           diffuse_idx::AbstractVector{<:Integer},
                           a0::AbstractVector{<:Real}=zeros(ssm.r),
                           P_star0::AbstractMatrix{<:Real}=zeros(ssm.r, ssm.r))
        -> (loglik, v, F, nobs_diffuse, converged)

Exact diffuse Kalman filter (Durbin & Koopman 2012 ch. 5): the state
indices in `diffuse_idx` start with **no proper prior at all** (`P_infty`
initialized as the identity on those coordinates, zero elsewhere -- the
standard `P0 = P_star + kappa*P_infty` limit as `kappa -> infinity`,
without ever actually forming a large finite `kappa`); the remaining
coordinates start from the ordinary known `a0`/`P_star0` (both default to
zero, matching `TimeVaryingSSM`'s own "known initial state" philosophy --
Stage 8.3's SARIMAX will typically want a genuinely mixed system, proper
ARMA-stationary states alongside diffuse regression coefficients, hence
these being real keyword arguments here rather than hardcoded zero,
beyond what this handoff's own minimal proposed signature showed).

**Deliberately no `kappa`/approximate-variance parameter at all.**
Verified directly (`test/verification/diffuseinit/`) that the intuitive
"just make `kappa` really big" approach has a genuine sweet spot, not a
limit that keeps improving: comparing R's actual default (`kappa=1e6`)
against this exact method, agreement is excellent (`2.6e-7` in the
reference case) -- but accuracy measurably *degrades* past roughly
`1e8`-`1e10` (floating-point cancellation from an excessively large
initial variance, a real numerical instability, not hypothetical).
The exact method has no such fragility, which is the actual reason it's
worth building, not because the approximate method is inherently crude
-- it isn't, when used correctly.

**`loglik` sums *every* observation's likelihood contribution,
`nobs_diffuse` included** -- this is a correction from an earlier
version of this function (and its docstring) that excluded the first
`nobs_diffuse` observations, modeled on this project's `nobs = n - d`
differencing convention (Stage 6.6/6.7). That convention doesn't apply
here: confirmed directly against real, directly-executed `statsmodels`
(`res.llf` on `kf.initialize_diffuse()`, **not** a manually-summed
subset of `res.llf_obs`) that the diffuse-phase contribution
(`-0.5*log(2π*F_infty,t)`, confirmed directly from `statsmodels`'s own
source -- no `v_t^2/F_t` term at all, a genuinely different quantity
from an ordinary observation's) is a well-defined, finite quantity that
*is* included in the reported total, not burned -- exactly the point of
"exact" diffuse initialization (Durbin & Koopman): unlike the
approximate `kappa` method, which genuinely needs the initial
observations discarded because their likelihood is unreliable for large
finite `kappa`, the exact method's diffuse-phase term is fully valid on
its own and SARIMAX's own default `loglikelihood_burn` for it is `0`.
On `test/verification/diffuseinit/diffuse_case1.csv`'s own system,
`res.llf = -14.53870388541507` (the genuine default) vs.
`np.sum(res.llf_obs[res.nobs_diffuse:]) = -11.239210224444564` (a
manually-sliced, non-default figure this function's own first attempt
was mistakenly compared against) -- both numbers were captured side by
side in `diffuse_case1.json` (`loglik_full` vs. `loglik_excl_diffuse`)
from the very first verification run, but only the non-default one was
checked at the time. `nobs_diffuse` is still returned -- genuinely
useful diagnostic information (how many leading observations were
diffuse-influenced) -- it just no longer changes what `loglik` sums.
`v`/`F` are returned for **every** observation (length `n`, `F` always
the ordinary `F_star,t`) for full residual-diagnostic usefulness,
matching `GaussianSSM`/`TimeVaryingSSM`'s own existing convention.

**The diffuse phase is `O(d)` fixed overhead, not `O(n)`** -- confirmed
directly by execution (`n=50/500/5000` on an identical structure all
gave `nobs_diffuse=2`, not scaling with `n`). Implemented as the
handoff itself recommends: a single loop tracks `P_infty` only until it
collapses to (numerically) zero -- detected via
`sum(P_infty.^2) <= tolerance_diffuse`, `statsmodels`'s own Frobenius-
norm-squared convention -- at which point this function stops updating
`P_infty` at all and the remaining recursion is *exactly*
`kalman_filter(::TimeVaryingSSM,...)`'s own already-verified per-step
math (confirmed algebraically: once `P_infty=0`, this function's
filtered-covariance update `P_t*(I-K_0 Z_t)'` and `TimeVaryingSSM`'s own
prediction-form update are the identical recursion, just derived via a
different intermediate factorization -- the same relationship already
proven in `to_time_varying`'s own docstring for the Stage 8.1 case).

Verified against real, directly-executed `statsmodels`
(`kf.initialize_diffuse()`, confirmed matching this function's own
`diffuse_idx` covering every state): `nobs_diffuse` and `loglik` (the
genuine, default, fully-inclusive `res.llf`, re-verified against Stage
8.3's own SARIMAX `time_varying_regression=True` scenario too -- see
`test/verification/arimax/arimax-ground-truth-transcript.txt`) both
match to machine precision on a local-level-plus-time-varying-
regression-coefficient system, the same structural shape as this
handoff's own worked example (see
`test/verification/diffuseinit/diffuseinit-ground-truth-transcript.txt`
for the original numeric comparison and
`test/verification/diffuseinit/diffuseinit-inclusive-sum-correction.txt`
for this correction's own record).

`converged=false` (`loglik=-Inf`, `v`/`F` empty) on a degenerate `F_t`
at any step, matching `TimeVaryingSSM`'s own sentinel convention.
"""
function kalman_filter_diffuse(ssm::TimeVaryingSSM, y::AbstractVector{<:Real};
                                diffuse_idx::AbstractVector{<:Integer},
                                a0::AbstractVector{<:Real}=zeros(ssm.r),
                                P_star0::AbstractMatrix{<:Real}=zeros(ssm.r, ssm.r))
    n = length(y)
    n == 0 && throw(ArgumentError("kalman_filter_diffuse: y must be non-empty"))
    r = ssm.r
    length(a0) == r || throw(ArgumentError("kalman_filter_diffuse: a0 must have length r=$r"))
    size(P_star0) == (r, r) || throw(ArgumentError("kalman_filter_diffuse: P_star0 must be r x r = $r x $r"))
    all(1 .<= diffuse_idx .<= r) || throw(ArgumentError("kalman_filter_diffuse: diffuse_idx entries must be in 1:r=$r"))
    length(unique(diffuse_idx)) == length(diffuse_idx) ||
        throw(ArgumentError("kalman_filter_diffuse: diffuse_idx must not contain duplicates"))

    VT = promote_type(eltype(P_star0), eltype(a0), eltype(ssm.T[1]), Float64)
    a = Vector{VT}(a0)
    P = Matrix{VT}(P_star0)
    Pinf = zeros(VT, r, r)
    for i in diffuse_idx
        Pinf[i, i] = one(VT)
    end
    diffuse_active = !isempty(diffuse_idx)

    v = Vector{VT}(undef, n)
    F = Vector{VT}(undef, n)
    acc = zero(VT)
    nobs_diffuse = 0
    Ir = Matrix{VT}(I, r, r)

    for t in 1:n
        Tt = _tv_at(ssm.T, t)
        z = vec(_tv_at(ssm.Z, t))
        Rt = _tv_at(ssm.R, t)
        Qt = _tv_at(ssm.Q, t)
        Ht = _tv_at(ssm.H, t)[1, 1]

        vt = y[t] - dot(z, a)
        v[t] = vt
        Fstar = max(dot(z, P * z) + Ht, zero(VT))
        F[t] = Fstar
        Finf = diffuse_active ? max(dot(z, Pinf * z), zero(VT)) : zero(VT)

        local a_filt, P_filt, Pinf_filt
        if diffuse_active && Finf > _TOLERANCE_DIFFUSE
            nobs_diffuse += 1
            F1 = 1 / Finf
            M = P * z
            Minf = Pinf * z
            K0 = Minf .* F1
            K1 = M .* F1 .- Minf .* (Fstar * F1^2)
            L0 = Ir - K0 * z'
            L1 = -K1 * z'
            a_filt = a + K0 .* vt
            P_filt = P * L0' + Pinf * L1'
            Pinf_filt = Pinf * L0'
            acc += log(Finf)   # diffuse-phase contribution -- see docstring: no v^2/F term
        elseif Fstar > _TOLERANCE_DIFFUSE
            K0 = P * z ./ Fstar
            L0 = Ir - K0 * z'
            a_filt = a + K0 .* vt
            P_filt = P * L0'
            Pinf_filt = Pinf
            acc += log(Fstar) + vt^2 / Fstar
        else
            if !isfinite(Fstar) || (!diffuse_active && Fstar <= 0)
                return (-Inf, Float64[], Float64[], nobs_diffuse, false)
            end
            # both F_infty and F_star numerically zero at this step: genuinely
            # uninformative observation, matching statsmodels's own no-op
            # (neither branch fires in its source either)
            a_filt = a
            P_filt = P
            Pinf_filt = Pinf
        end

        a = Tt * a_filt
        P = Tt * P_filt * Tt' + Rt * Qt * Rt'
        if diffuse_active
            Pinf = Tt * Pinf_filt * Tt'
            diffuse_active = sum(abs2, Pinf) > _TOLERANCE_DIFFUSE
        end
    end

    loglik = -0.5 * (n * log(2π) + acc)
    return (loglik, v, F, nobs_diffuse, true)
end
