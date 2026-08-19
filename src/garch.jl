export fit_garch, fit_garch_multi, GarchModel

"""
    GarchModel <: UnivariateModel

Result of [`fit_garch`](@ref): GARCH(p,q) (Bollerslev 1986), GJR-GARCH
(Glosten, Jagannathan & Runkle 1993), or EGARCH (Nelson 1991), selected
via `fit_garch`'s `model` keyword. `p` = ARCH (lagged squared/absolute
residual) order, `q` = GARCH (lagged variance) order --
**matching Python `arch.arch_model(p=,q=)`'s own naming, not Bollerslev's
original notation** (which has `p`/`q` swapped) -- see [`fit_garch`](@ref)
for why.

- `model`: `:garch`, `:gjr`, or `:egarch`.
- `omega`/`alpha`/`gamma`/`beta`: fitted variance-equation coefficients
  (`alpha` length `p`, `beta` length `q`). `gamma` is `nothing` for
  `:garch`, a length-1 vector for `:gjr`/`:egarch` (this implementation
  fixes the asymmetry order at 1, a deliberate scope narrowing -- see
  [`fit_garch`](@ref)).
- `mean_spec`/`mu`: `:zero` (`mu === nothing`, `resid = y`) or `:constant`
  (`mu` jointly estimated with the variance parameters, not a naive
  pre-demean -- see [`fit_garch`](@ref)).
- `sigma2`: fitted conditional variance path, one per observation
  (always the *level* variance, even for `:egarch`, whose recursion is
  on `log(sigma2)` internally).
- `resid`: `y .- mu` (`== y` when `mean_spec == :zero`).
- `se`: standard errors, in `[mu?; omega; alpha; gamma?; beta]` order
  (matching `coef`'s own ordering -- `gamma` sits *between* `alpha` and
  `beta`, matching `arch_model`'s own parameter display order).
- `cov_type`: `:robust` (Bollerslev-Wooldridge sandwich estimator, the
  default) or `:classic` (plain inverse-Hessian) -- see [`fit_garch`](@ref).
- `converged`: whether the optimizer reported convergence.
"""
struct GarchModel <: UnivariateModel
    model::Symbol
    omega::Float64
    alpha::Vector{Float64}
    gamma::Union{Nothing,Vector{Float64}}
    beta::Vector{Float64}
    mean_spec::Symbol
    mu::Union{Nothing,Float64}
    sigma2::Vector{Float64}
    resid::Vector{Float64}
    se::Vector{Float64}
    loglik::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    p::Int
    q::Int
    cov_type::Symbol
    converged::Bool
end

StatsAPI.loglikelihood(m::GarchModel) = m.loglik
StatsAPI.aic(m::GarchModel) = m.aic
StatsAPI.bic(m::GarchModel) = m.bic
StatsAPI.nobs(m::GarchModel) = m.nobs
StatsAPI.coef(m::GarchModel) = vcat(m.mu === nothing ? Float64[] : [m.mu], m.omega, m.alpha,
                                     m.gamma === nothing ? Float64[] : m.gamma, m.beta)
StatsAPI.residuals(m::GarchModel) = m.resid

"_garch_backcast(e, tau) -- exponentially-weighted (0.94^i, normalized)
average of the first `tau = min(75, n)` squared residuals, used to seed
the variance recursion's unavailable pre-sample lags. Not this project's
own invention: matches the real `arch` package's own `backcast` method
exactly (confirmed directly from its source, `arch.univariate.volatility.
GARCH.backcast`) -- resolving what would otherwise be a genuine
implementation ambiguity (any fixed positive seed value converges to the
same MLE asymptotically, but the exact number affects finite-sample
numbers, and this project validates finite-sample numbers directly
against real `arch` output)."
function _garch_backcast(e::AbstractVector{<:Real}, tau::Integer)
    w = 0.94 .^ (0:(tau - 1))
    w ./= sum(w)
    return dot(w, view(e, 1:tau) .^ 2)
end

"_stationary_softmax(raw) -- maps a length-`k` unconstrained vector to `k`
non-negative proportions summing to strictly less than 1, via a softmax
over `k` categories plus one *implicit* zero-logit category. Shared by
[`_garch_unpack`](@ref) (over the combined `alpha;beta` block) and
`fit_garch`'s `:egarch` unpack (over `beta` alone) -- the same
from-scratch \"non-negative, sum < 1\" reparametrization trick applied to
whichever block of coefficients needs the GARCH stationarity constraint
in a given model variant."
function _stationary_softmax(raw::AbstractVector)
    wgt = exp.(raw)
    denom = 1 + sum(wgt)
    return wgt ./ denom
end

"_garch_sigma2_path(e, omega, alpha, beta, backcast) -- the variance
recursion `sigma2[t] = omega + sum_j alpha[j]*e[t-j]^2 + sum_k
beta[k]*sigma2[t-k]`, substituting `backcast` for any lag reaching before
`t=1` (matching `arch`'s own `garch_recursion`, confirmed directly from
its Python reference implementation). Sequential by construction -- each
`sigma2[t]` depends on `sigma2[t-1]`, the same structural reason the
Kalman filter recursion isn't parallelizable either (Stage 6's own
finding); no attempt is made to parallelize within one fit's likelihood
evaluation, only across independent fits (see [`fit_garch_multi`](@ref)
and `fit_garch`'s own `n_restarts`)."
function _garch_sigma2_path(e::AbstractVector, omega, alpha::AbstractVector, beta::AbstractVector,
                             backcast::Real)
    T = promote_type(eltype(e), typeof(omega))
    n = length(e)
    p, q = length(alpha), length(beta)
    sigma2 = Vector{T}(undef, n)
    for t in 1:n
        s2 = omega
        for j in 1:p
            s2 += alpha[j] * (t - j >= 1 ? e[t - j]^2 : backcast)
        end
        for k in 1:q
            s2 += beta[k] * (t - k >= 1 ? sigma2[t - k] : backcast)
        end
        sigma2[t] = s2
    end
    return sigma2
end

"_gjr_sigma2_path(e, omega, alpha, gamma1, beta, backcast) -- GJR-GARCH
(Glosten, Jagannathan & Runkle 1993): the plain GARCH recursion plus one
asymmetric term `gamma1*e[t-1]^2*I(e[t-1]<0)` -- a leverage effect,
negative shocks raise variance more than positive ones of the same size.
Asymmetry order fixed at 1 regardless of `p` (this implementation's
deliberate scope narrowing -- see [`fit_garch`](@ref)), matching
`arch_model(o=1)`'s pre-sample convention exactly (confirmed directly
from its `garch_recursion` source): the missing pre-sample asymmetric
term is `gamma1*0.5*backcast`, the `0.5` reflecting `E[I(e<0)]` for a
symmetric innovation, not the full `backcast` plain GARCH's own lags use."
function _gjr_sigma2_path(e::AbstractVector, omega, alpha::AbstractVector, gamma1, beta::AbstractVector,
                           backcast::Real)
    T = promote_type(eltype(e), typeof(omega), typeof(gamma1))
    n = length(e)
    p, q = length(alpha), length(beta)
    sigma2 = Vector{T}(undef, n)
    for t in 1:n
        s2 = omega
        for j in 1:p
            s2 += alpha[j] * (t - j >= 1 ? e[t - j]^2 : backcast)
        end
        s2 += if t - 1 >= 1
            gamma1 * e[t - 1]^2 * (e[t - 1] < 0 ? 1.0 : 0.0)
        else
            gamma1 * 0.5 * backcast
        end
        for k in 1:q
            s2 += beta[k] * (t - k >= 1 ? sigma2[t - k] : backcast)
        end
        sigma2[t] = s2
    end
    return sigma2
end

"_egarch_sigma2_path(e, omega, alpha, gamma1, beta, backcast_ln) --
EGARCH (Nelson 1991): `log(sigma2[t]) = omega + sum_i alpha[i]*(|z[t-i]|
- sqrt(2/pi)) + gamma1*z[t-1] + sum_k beta[k]*log(sigma2[t-k])`, `z[t] =
e[t]/sigma[t]` the standardized residual -- confirmed directly from
`arch.univariate.volatility.EGARCH`'s own docstring/source, not a
third-party formula (the handoff's own flagged risk: EGARCH's
`alpha`/`gamma` naming and scaling is *not* universally standardized
across independent implementations, unlike GJR-GARCH's). **Pre-sample
convention genuinely differs from plain/GJR-GARCH's**, confirmed
directly from `egarch_recursion`'s source: missing pre-sample `alpha`/
`gamma` terms are simply *omitted* (not substituted with `backcast`),
only the `beta` (log-variance) lags fall back to `backcast_ln`. `z[t]`
is only computable once `sigma2[t]` itself is known, so `std_resid`/
`abs_std_resid` are built up one step at a time inside the same loop,
each used only by *later* time steps -- still sequential by
construction, same reason as every other variance recursion in this
project. `log(sigma2[t])` is clamped to `[-50,50]` before exponentiating
as a numerical safety net during optimizer excursions (not part of
`arch`'s own formula, which uses its own `var_bounds`-based clamp
instead -- this project's reparametrization makes `omega`/`alpha`/
`gamma` genuinely unconstrained the same way `arch`'s own bounds do, so
some such safety net is needed either way)."
function _egarch_sigma2_path(e::AbstractVector, omega, alpha::AbstractVector, gamma1, beta::AbstractVector,
                              backcast_ln::Real)
    T = promote_type(eltype(e), typeof(omega), typeof(gamma1))
    n = length(e)
    p, q = length(alpha), length(beta)
    sqrt2opi = sqrt(2 / pi)
    lnsigma2 = Vector{T}(undef, n)
    z = Vector{T}(undef, n)
    absz = Vector{T}(undef, n)
    for t in 1:n
        ls2 = omega
        for i in 1:p
            t - i >= 1 && (ls2 += alpha[i] * (absz[t - i] - sqrt2opi))
        end
        t - 1 >= 1 && (ls2 += gamma1 * z[t - 1])
        for k in 1:q
            ls2 += beta[k] * (t - k >= 1 ? lnsigma2[t - k] : backcast_ln)
        end
        ls2 = clamp(ls2, -50.0, 50.0)
        lnsigma2[t] = ls2
        s = exp(ls2 / 2)
        z[t] = e[t] / s
        absz[t] = abs(z[t])
    end
    return exp.(lnsigma2)
end

"_neg_ll_from_sigma2(e, sigma2) -- per-observation negative Gaussian
log-likelihood contributions, `0.5*(log(2π) + log(sigma2[t]) +
e[t]^2/sigma2[t])` (matching `arch`'s own `Normal.loglikelihood`,
confirmed directly from source, negated to match this project's
`_optimize`-minimizes convention). Shared across all three model
variants -- once `sigma2` (the *level* variance, even for `:egarch`) is
known, the Gaussian likelihood formula is identical regardless of which
recursion produced it."
function _neg_ll_from_sigma2(e::AbstractVector, sigma2::AbstractVector)
    return @. 0.5 * (log(2 * pi) + log(sigma2) + e^2 / sigma2)
end

"_garch_unpack(raw, p, q, has_mean) -- maps the unconstrained optimizer
vector to `(mu, omega, alpha, beta)`. `omega = exp(raw[i])` enforces
`omega > 0`; `alpha`/`beta` come from [`_stationary_softmax`](@ref) over
the combined `p+q` block, so `alpha .>= 0`, `beta .>= 0`, and
`sum(alpha) + sum(beta) < 1` (the GARCH stationarity/positivity
constraint) hold by construction for any finite `raw` -- a from-scratch
reparametrization (not `partrans`, which is Monahan's AR/MA-specific
transform), analogous in spirit but not formula to how `partrans` frees
Stage 6's optimizer from the AR/MA stationarity constraint."
function _garch_unpack(raw::AbstractVector, p::Integer, q::Integer, has_mean::Bool)
    T = eltype(raw)
    i = 1
    mu = has_mean ? raw[i] : zero(T)
    has_mean && (i += 1)
    omega = exp(raw[i])
    i += 1
    prop = _stationary_softmax(raw[i:(i + p + q - 1)])
    alpha = prop[1:p]
    beta = q > 0 ? prop[(p + 1):(p + q)] : T[]
    return mu, omega, alpha, beta
end

"_gjr_unpack(raw, p, q, has_mean) -- as [`_garch_unpack`](@ref) for
`omega`/`alpha`/`beta`, plus one extra raw entry (at the end) mapped to
`gamma1` via a sigmoid onto the *asymmetric* interval
`(-alpha[1], 2*(1-persistence_ab))`. The two GJR-GARCH constraints
(confirmed directly from `arch`'s own `GARCH.constraints()` source) are
each one-sided, not a symmetric cap: `alpha[1] + gamma1 >= 0` only
bounds `gamma1` *below* by `-alpha[1]` (a large *positive* `gamma1` is
never a problem for this constraint -- an earlier symmetric
`g_max*tanh(...)` version of this function wrongly capped `gamma1`'s
magnitude on both sides, silently preventing the optimizer from ever
reaching a true `gamma1 > alpha[1]`, caught by comparing against real
`arch` output, not assumed correct from the formula alone); `sum(alpha)
+ 0.5*gamma1 + sum(beta) < 1` only bounds `gamma1` *above* by
`2*(1-persistence_ab)` (`0.5` reflecting `E[I(e<0)]` for symmetric
innovations). Both hold by construction for any finite raw vector."
function _gjr_unpack(raw::AbstractVector, p::Integer, q::Integer, has_mean::Bool)
    T = eltype(raw)
    i = 1
    mu = has_mean ? raw[i] : zero(T)
    has_mean && (i += 1)
    omega = exp(raw[i])
    i += 1
    prop = _stationary_softmax(raw[i:(i + p + q - 1)])
    i += p + q
    alpha = prop[1:p]
    beta = q > 0 ? prop[(p + 1):(p + q)] : T[]
    persistence_ab = sum(prop)
    lower = -alpha[1]
    upper = 2 * (1 - persistence_ab)
    gamma1 = lower + (upper - lower) / (1 + exp(-raw[i]))
    return mu, omega, alpha, [gamma1], beta
end

"_egarch_unpack(raw, p, q, has_mean) -- `omega`/`alpha`/`gamma1` are
identity-mapped (genuinely unconstrained on the log-variance scale,
matching `arch.univariate.volatility.EGARCH`'s own `(-inf,inf)` bounds
for these -- confirmed directly from its `bounds()` source); `beta`
alone goes through [`_stationary_softmax`](@ref) (`beta .>= 0`,
`sum(beta) < 1`, the log-variance-AR stationarity condition, confirmed
from `EGARCH.constraints()`'s own source: `sum(beta) < 1`)."
function _egarch_unpack(raw::AbstractVector, p::Integer, q::Integer, has_mean::Bool)
    T = eltype(raw)
    i = 1
    mu = has_mean ? raw[i] : zero(T)
    has_mean && (i += 1)
    omega = raw[i]
    i += 1
    alpha = raw[i:(i + p - 1)]
    i += p
    gamma1 = raw[i]
    i += 1
    beta = q > 0 ? _stationary_softmax(raw[i:(i + q - 1)]) : T[]
    return mu, omega, alpha, [gamma1], beta
end

"_garch_start_raw(p, q, has_mean, mu0, e0_var, alpha0_total, beta0_total)
-- inverts [`_stationary_softmax`](@ref) to produce raw parameters
giving the requested natural starting point (`alpha0_total`/
`beta0_total` split evenly across `p`/`q` terms respectively), used both
for the default single start and to seed each randomized multi-start
restart. For `model=:gjr`, an extra `0.0` is appended (`tanh(0)=0`, a
neutral zero-asymmetry start)."
function _garch_start_raw(p::Integer, q::Integer, has_mean::Bool, mu0::Real, e0_var::Real,
                           alpha0_total::Real, beta0_total::Real; gjr::Bool=false)
    persistence = clamp(alpha0_total + beta0_total, 0.01, 0.98)
    s = persistence / (1 - persistence)
    wgt = Vector{Float64}(undef, p + q)
    for j in 1:p
        wgt[j] = (alpha0_total / p) * (1 + s)
    end
    for k in 1:q
        wgt[p + k] = (beta0_total / q) * (1 + s)
    end
    raw_ab = log.(max.(wgt, 1e-8))
    omega0 = max(e0_var * (1 - persistence), 1e-8)
    raw = has_mean ? vcat(mu0, log(omega0), raw_ab) : vcat(log(omega0), raw_ab)
    return gjr ? vcat(raw, 0.0) : raw
end

"_egarch_start_raw(p, q, has_mean, mu0, e0_var, beta0_total, alpha0, gamma0)
-- `omega`/`alpha`/`gamma` starting values are natural values directly
(identity-mapped, see [`_egarch_unpack`](@ref)); `beta0_total` inverted
through [`_stationary_softmax`](@ref) as in [`_garch_start_raw`](@ref)."
function _egarch_start_raw(p::Integer, q::Integer, has_mean::Bool, mu0::Real, e0_var::Real,
                            beta0_total::Real, alpha0::Real, gamma0::Real)
    beta0_total = clamp(beta0_total, 0.01, 0.98)
    s = beta0_total / (1 - beta0_total)
    raw_beta = q > 0 ? log.(fill(max((beta0_total / q) * (1 + s), 1e-8), q)) : Float64[]
    omega0 = log(max(e0_var, 1e-8)) * (1 - beta0_total)
    raw_alpha = fill(alpha0, p)
    return vcat(has_mean ? [mu0] : Float64[], omega0, raw_alpha, gamma0, raw_beta)
end

"""
    fit_garch(y, p::Integer=1, q::Integer=1;
              model::Symbol=:garch,
              mean_spec::Symbol=:zero, dist::Symbol=:normal,
              cov_type::Symbol=:robust, optimizer_method::Symbol=:lbfgs,
              n_restarts::Integer=1, parallel::Bool=true,
              start_params::Union{Nothing,Vector{Float64}}=nothing) -> GarchModel

Fit GARCH(p,q) (`model=:garch`), GJR-GARCH (`model=:gjr`, Glosten,
Jagannathan & Runkle 1993), or EGARCH (`model=:egarch`, Nelson 1991) by
maximum likelihood, Gaussian innovations.

**`p`/`q` naming, resolved explicitly (a well-known source of confusion
-- Bollerslev's original notation has `p` as the *GARCH*/variance order
and `q` as the *ARCH*/residual order, the opposite of what's used here)**:
this project's `p` is the ARCH order (`alpha`, length `p`) and `q` is the
GARCH order (`beta`, length `q`), **matching both Python `arch_model(p=,q=)`
and R `rugarch`'s `garchOrder=c(p,q)`** (confirmed directly: fitting
`arch_model(p=2,q=1)` and inspecting which fitted names came back; and
`rugarch`'s own docs stating *"the first element denotes the ARCH order
and the second the GARCH order"*) -- both software ecosystems agree,
even though some textbook citations of Bollerslev's original notation
swap `p`/`q`.

**`model=:gjr`/`:egarch` add one asymmetry ("leverage") term, `gamma`,
fixed at order 1 regardless of `p`** -- a deliberate scope narrowing
(this implementation doesn't expose Python's more general independent
`o`-order), matching `arch_model(o=1)`/`rugarch`'s `garchOrder`
convention for the overwhelmingly standard GJR(1,1)/EGARCH(1,1) usage.
**GJR-GARCH's asymmetric term is `gamma*e[t-1]^2*I(e[t-1]<0)`** added to
the same level-variance recursion as plain GARCH -- confirmed matching
`arch_model(vol='GARCH', o=1)` exactly (GJR-GARCH is *not* a separate
`vol=` spec in Python, unlike R's `rugarch`, which names it
`"gjrGARCH"` separately). **EGARCH is a genuinely separate recursion, on
`log(sigma2)`**: `log(sigma2[t]) = omega + sum_i alpha[i]*(|z[t-i]| -
sqrt(2/pi)) + gamma*z[t-1] + sum_k beta[k]*log(sigma2[t-k])`, `z[t] =
e[t]/sigma[t]` -- confirmed directly from `arch.univariate.volatility.
EGARCH`'s own source, **not assumed from a third-party formula**: an
independent academic source directly comparing its own EGARCH
implementation against `rugarch` found EGARCH's `alpha`/`gamma` naming
and scaling genuinely differs across independent implementations
(unlike GJR-GARCH's, which matched cleanly between R and Python). This
implementation's EGARCH formula is verified against Python's actual
executed output as the primary target for exactly this reason, not
against any third-party convention. `omega`/`alpha`/`gamma` all have no
positivity constraint on the log-variance scale (`m.omega < 0` is
expected and correct, not a bug -- confirmed directly: Python's own
verified example has `omega=-0.007737`).

**`mean_spec` defaults to `:zero`, not Python's `mean='Constant'`
default** -- a deliberate divergence matching this project's convention
throughout (AR-X, ARMA, ARIMA/SARIMA): GARCH's own scope is the variance
equation, not the mean equation; the common practitioner workflow this
project is built around is fitting a mean model first (`arx`/`fit_arma`/
`fit_arima`), then `fit_garch` on its residuals. `:constant` is available
for parity with Python's default -- when used, `mu` is estimated
*jointly* with the variance parameters by the same MLE (matching real
`arch_model(mean='Constant')` behavior), not by pre-demeaning with the
sample mean first, the same "joint, not naive pre-demean" choice already
made for `fit_arma`'s `include_mean` (Stage 6.5).

**`dist=:normal` is the only distribution currently implemented.**
`dist=:t` is accepted in the signature (documented future extension) but
throws a clear `ArgumentError` rather than silently falling back --
Python's `arch_model(dist='t')` needs an additional estimated
degrees-of-freedom shape parameter and its own log-likelihood, genuinely
new scope no handoff so far provides verified ground truth for.

**`cov_type` defaults to `:robust`**, matching Python's default and
standard GARCH practice (financial returns routinely violate conditional
normality) -- the Bollerslev-Wooldridge QMLE-robust ("sandwich")
estimator `inv(H)*cov(scores)*inv(H)/n`, vs. `:classic`'s plain
`inv(H)/n`, `H` the average Hessian of the *negative* log-likelihood at
the optimum. Confirmed to be genuinely different numbers, not just a
different code path to the same answer, both by reading `arch`'s own
`compute_param_cov` source directly and by fitting real data (Stage
7.1's own `garch_shared.csv`: `omega` se `0.0134` robust vs `0.0159`
classic).

**Both parallelism designs (Stage 7.1) reused unchanged, not
redesigned** -- confirmed directly from `rugarch`'s own documentation
that `multifit`/`gosolnp`'s parallelism options are generic across
variance-model type (`sGARCH`/`gjrGARCH`/`eGARCH` alike), not
model-specific overloads. See [`fit_garch_multi`](@ref) (matches
`rugarch::multifit`) and `n_restarts` (matches `rugarch`'s `gosolnp`);
`n_restarts > 1` fits from that many randomized starting points
(respecting each model's own stationarity constraint by construction --
see `_garch_unpack`/`_gjr_unpack`/`_egarch_unpack`) and keeps the
highest-likelihood *converged* result. `parallel=true` by default for
both designs; `Threads.@threads` only actually engages when
`Threads.nthreads() > 1` and there's enough work to be worth the
thread-spawning overhead (`n_restarts >= 4`), same guarded pattern as
every other parallel design in this project.

A single fit's variance recursion is sequential by construction (each
`sigma2[t]` depends on `sigma2[t-1]`, and EGARCH's standardized
residuals depend on the *just-computed* `sigma2[t]` before being used by
later steps) -- no attempt is made to parallelize within one fit's
likelihood evaluation, the same structural reason the Kalman filter
recursion isn't parallelizable (Stage 6).

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> e = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "garch", "garch_shared.csv"), ','; skipstart=1));

julia> m = fit_garch(e, 1, 1);

julia> m.converged
true

julia> round(m.omega, digits=4)
0.0431

julia> round(m.alpha[1], digits=4)
0.0906

julia> round(m.beta[1], digits=4)
0.8679

julia> egjr = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "garch", "gjr_shared.csv"), ','; skipstart=1));

julia> mg = fit_garch(egjr, 1, 1; model=:gjr);

julia> mg.gamma !== nothing
true

julia> me = fit_garch(egjr, 1, 1; model=:egarch);

julia> me.omega < 0   # NOT a bug -- log-variance omega has no positivity constraint
true
```
"""
function fit_garch(y, p::Integer=1, q::Integer=1;
                    model::Symbol=:garch,
                    mean_spec::Symbol=:zero,
                    dist::Symbol=:normal,
                    cov_type::Symbol=:robust,
                    optimizer_method::Symbol=:lbfgs,
                    n_restarts::Integer=1,
                    parallel::Bool=true,
                    start_params::Union{Nothing,Vector{Float64}}=nothing)
    model in (:garch, :gjr, :egarch) || throw(ArgumentError("model must be :garch, :gjr, or :egarch"))
    mean_spec in (:zero, :constant) || throw(ArgumentError("mean_spec must be :zero or :constant"))
    dist == :normal || throw(ArgumentError(
        "dist=:t is not yet implemented (needs an estimated degrees-of-freedom shape " *
        "parameter and its own log-likelihood) -- use :normal"))
    cov_type in (:robust, :classic) || throw(ArgumentError("cov_type must be :robust or :classic"))
    p >= 1 || throw(ArgumentError("p (ARCH order) must be >= 1"))
    q >= 0 || throw(ArgumentError("q (GARCH order) must be >= 0"))
    n_restarts >= 1 || throw(ArgumentError("n_restarts must be >= 1"))

    yv = Float64.(collect(tsvalues(y)))
    n = length(yv)
    has_mean = mean_spec == :constant
    has_gamma = model != :garch
    nparam = (has_mean ? 1 : 0) + 1 + p + q + (has_gamma ? 1 : 0)
    n > nparam || throw(ArgumentError("fit_garch: not enough observations ($n) for p=$p, q=$q" *
                                       (has_mean ? " with a mean" : "")))

    mu0 = has_mean ? sum(yv) / n : 0.0
    e0 = yv .- mu0
    tau = min(75, n)
    backcast = _garch_backcast(e0, tau)
    backcast_ln = log(backcast)

    function unpack(raw::AbstractVector)
        if model == :garch
            mu, omega, alpha, beta = _garch_unpack(raw, p, q, has_mean)
            return mu, omega, alpha, nothing, beta
        elseif model == :gjr
            return _gjr_unpack(raw, p, q, has_mean)
        else
            return _egarch_unpack(raw, p, q, has_mean)
        end
    end

    sigma2_of(e::AbstractVector, omega, alpha, gamma, beta) =
        model == :garch ? _garch_sigma2_path(e, omega, alpha, beta, backcast) :
        model == :gjr ? _gjr_sigma2_path(e, omega, alpha, gamma[1], beta, backcast) :
        _egarch_sigma2_path(e, omega, alpha, gamma[1], beta, backcast_ln)

    objective(raw::AbstractVector) = begin
        mu, omega, alpha, gamma, beta = unpack(raw)
        sigma2 = sigma2_of(yv .- mu, omega, alpha, gamma, beta)
        s = sum(_neg_ll_from_sigma2(yv .- mu, sigma2))
        isfinite(s) ? s : oftype(s, 1e10)
    end

    x0 = if start_params !== nothing
        length(start_params) == nparam ||
            throw(ArgumentError("start_params must have length p+q+1$(has_mean ? "+1" : "")$(has_gamma ? "+1" : "") = $nparam"))
        start_params
    elseif model == :egarch
        _egarch_start_raw(p, q, has_mean, mu0, var(e0), 0.90, 0.1, -0.05)
    else
        _garch_start_raw(p, q, has_mean, mu0, var(e0), q > 0 ? 0.05 : 0.3, q > 0 ? 0.90 : 0.0;
                          gjr=(model == :gjr))
    end

    starts = Vector{Vector{Float64}}(undef, n_restarts)
    starts[1] = x0
    for i in 2:n_restarts
        rng = Random.MersenneTwister(1000 + i)
        persistence = 0.5 + 0.48 * rand(rng)
        if model == :egarch
            alpha0 = 0.05 + 0.25 * rand(rng)
            gamma0 = -0.15 * rand(rng)
            starts[i] = _egarch_start_raw(p, q, has_mean, mu0, var(e0), persistence, alpha0, gamma0)
        else
            alpha0 = persistence * (0.05 + 0.5 * rand(rng))
            beta0 = q > 0 ? persistence - alpha0 : 0.0
            alpha0 = q > 0 ? alpha0 : persistence
            starts[i] = _garch_start_raw(p, q, has_mean, mu0, var(e0), alpha0, beta0; gjr=(model == :gjr))
        end
    end

    results = Vector{Any}(undef, n_restarts)
    use_threads = parallel && Threads.nthreads() > 1 && n_restarts >= 4
    if use_threads
        Threads.@threads for i in 1:n_restarts
            results[i] = try
                _optimize(objective, starts[i]; method=optimizer_method)
            catch
                nothing
            end
        end
    else
        for i in 1:n_restarts
            results[i] = try
                _optimize(objective, starts[i]; method=optimizer_method)
            catch
                nothing
            end
        end
    end

    valid = [(r, objective(r.minimizer)) for r in results if r !== nothing]
    isempty(valid) && throw(ArgumentError("fit_garch: optimizer failed from every starting point"))
    converged_valid = filter(rv -> rv[1].converged, valid)
    best = isempty(converged_valid) ? valid[argmin(last.(valid))] : converged_valid[argmin(last.(converged_valid))]
    result, best_obj = best

    n0 = result.minimizer
    mu_hat, omega_hat, alpha_hat, gamma_hat, beta_hat = unpack(n0)
    e_hat = yv .- mu_hat
    sigma2_hat = sigma2_of(e_hat, omega_hat, alpha_hat, gamma_hat, beta_hat)
    loglik = -best_obj

    params_natural = vcat(has_mean ? [mu_hat] : Float64[], omega_hat, alpha_hat,
                           gamma_hat === nothing ? Float64[] : gamma_hat, beta_hat)
    natural_contributions(np::AbstractVector) = begin
        i = 1
        mu = has_mean ? np[i] : zero(eltype(np))
        has_mean && (i += 1)
        omega = np[i]
        i += 1
        alpha = np[i:(i + p - 1)]
        i += p
        gamma = has_gamma ? [np[i]] : nothing
        has_gamma && (i += 1)
        beta = q > 0 ? np[i:(i + q - 1)] : eltype(np)[]
        sigma2 = sigma2_of(yv .- mu, omega, alpha, gamma, beta)
        _neg_ll_from_sigma2(yv .- mu, sigma2)
    end

    H = ForwardDiff.hessian(np -> sum(natural_contributions(np)), params_natural) ./ n
    invH = try
        inv(H)
    catch e
        e isa Union{LinearAlgebra.SingularException,LinearAlgebra.LAPACKException} || rethrow()
        fill(NaN, nparam, nparam)
    end
    covmat = if any(isnan, invH)
        invH
    elseif cov_type == :classic
        invH ./ n
    else
        scores = ForwardDiff.jacobian(natural_contributions, params_natural)
        score_cov = Statistics.cov(scores)
        invH * score_cov * invH ./ n
    end
    se = sqrt.(max.(diag(covmat), 0.0))

    aic = -2 * loglik + 2 * nparam
    bic = -2 * loglik + nparam * log(n)

    return GarchModel(model, omega_hat, alpha_hat, gamma_hat, beta_hat, mean_spec,
                       has_mean ? mu_hat : nothing, sigma2_hat, e_hat, se, loglik, aic, bic, n, p, q,
                       cov_type, result.converged)
end

"""
    fit_garch_multi(ys::AbstractVector{<:AbstractVector}, p::Integer=1, q::Integer=1;
                     parallel::Bool=true, kwargs...) -> Vector{GarchModel}

Fits the *same* `(p,q,model)` specification to each series in `ys`
independently -- matches `rugarch::multifit`'s exact purpose (confirmed
directly from its own documentation: *"Method for multiple fitting a
variety of univariate GARCH and ARFIMA models... `cluster`: ... used for
parallel estimation."*), the realistic case being portfolio-style
volatility modeling across many assets at once. Each series' fit shares
no state with any other's -- genuinely embarrassingly parallel.
`kwargs...` are passed through to [`fit_garch`](@ref) unchanged (so
`model`, `mean_spec`, `cov_type`, `n_restarts`, etc. all apply
identically to every series -- confirmed directly from `rugarch`'s own
docs that this parallelism is generic across variance-model type, not
`sGARCH`-specific).

`parallel=true` by default; `Threads.@threads` only actually engages
when `Threads.nthreads() > 1` and `length(ys) >= 4` (below which
thread-spawning overhead isn't worth it), the same guarded pattern as
every other parallel design in this project.

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> e = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "garch", "garch_shared.csv"), ','; skipstart=1));

julia> results = fit_garch_multi([e, e], 1, 1);

julia> length(results)
2

julia> results[1].omega == results[2].omega
true
```
"""
function fit_garch_multi(ys::AbstractVector, p::Integer=1, q::Integer=1;
                          parallel::Bool=true, kwargs...)
    m = length(ys)
    results = Vector{GarchModel}(undef, m)
    use_threads = parallel && Threads.nthreads() > 1 && m >= 4
    if use_threads
        Threads.@threads for i in 1:m
            results[i] = fit_garch(ys[i], p, q; kwargs...)
        end
    else
        for i in 1:m
            results[i] = fit_garch(ys[i], p, q; kwargs...)
        end
    end
    return results
end

function Base.show(io::IO, m::GarchModel)
    label = m.model == :garch ? "GARCH" : m.model == :gjr ? "GJR-GARCH" : "EGARCH"
    names = vcat(m.mu !== nothing ? ["mu"] : String[], ["omega"],
                 ["alpha$i" for i in 1:m.p], m.gamma !== nothing ? ["gamma1"] : String[],
                 ["beta$i" for i in 1:m.q])
    coefs = vcat(m.mu !== nothing ? [m.mu] : Float64[], m.omega, m.alpha,
                  m.gamma !== nothing ? m.gamma : Float64[], m.beta)
    z = coefs ./ m.se
    pval = _chisq_ccdf.(z .^ 2, 1)
    ct = StatsBase.CoefTable(hcat(coefs, m.se, z, pval), ["Coef.", "Std. Error", "z", "Pr(>|z|)"], names)
    println(io, label, "(", m.p, ",", m.q, ")", m.mu !== nothing ? " with mean" : "",
            ", n=", m.nobs, " (cov: ", m.cov_type, ")")
    println(io)
    println(io, ct)
    print(io, "Log-likelihood: ", round(m.loglik, digits=2),
          "   AIC: ", round(m.aic, digits=2), "   BIC: ", round(m.bic, digits=2))
    m.converged || print(io, "\nWARNING: optimizer did not converge")
end
