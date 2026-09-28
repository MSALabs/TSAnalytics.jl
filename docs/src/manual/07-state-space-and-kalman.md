# State Space and the Kalman Filter

Every model in this package that involves a likelihood — ARMA, ARIMA,
SARIMA, ARIMAX, unobserved components — is fitted by writing it in
state-space form and running one Kalman filter. There is **one engine,
not one per model**, and it is exported rather than hidden, so you can
build on it directly.

This page is about using that engine yourself. If you only want to fit
an ARIMA, you never need it.

```@example ssm
using TSAnalytics, Statistics

y = dataset("gtemp_both").value   # global temperature anomaly, 174 years
nothing # hide
```

## Two forms, one filter

| Type | Shape | Used by |
|---|---|---|
| [`GaussianSSM`](@ref) | Time-invariant ARMA companion form. `Z` is implicitly `e₁`, `H` implicitly `0` | `fit_arma`, `fit_arima`, `fit_sarima` |
| [`TimeVaryingSSM`](@ref) | Arbitrary per-period `Z_t`, `T_t`, `R_t`, `Q_t`, `H_t` | Drifting-coefficient regression, diffuse initialisation |

[`kalman_filter`](@ref) and [`kalman_smoother`](@ref) are each defined
for both. The narrow form exists because it is faster — with `Z` and
`H` known at compile time the inner loop drops several matrix
operations per step — not because it does anything the general form
cannot.

## Build a model

```@example ssm
ssm = build_statespace([0.7], [0.3])   # ARMA(1,1)
println("state dimension r = ", ssm.r)
println("T = ", ssm.T)
println("R = ", ssm.R)
```

[`build_statespace`](@ref) takes the *combined* AR and MA coefficients
and returns Harvey's companion form, with `r = max(p, q+1)`. Construct
through it rather than calling the struct directly.

```@example ssm
P0, ok = stationary_cov(ssm)
println("P0[1,1] = ", round(P0[1,1], digits=5), "   converged = ", ok)
println("theory  = ", round((1 + 2*0.7*0.3 + 0.3^2) / (1 - 0.7^2), digits=5))
```

[`stationary_cov`](@ref) solves for the stationary state covariance by
doubling — about thirty iterations reach machine precision for any
spectral radius meaningfully below 1. It returns a **tuple**, value and
a convergence flag, and the flag must be checked: it never throws.

`P0[1,1]` is the process variance, and it reproduces the textbook
ARMA(1,1) formula exactly. That is the cheapest available check that
you built the model you meant to.

### Seasonal models

```@example ssm
ar, ma = combined_ar_ma([0.5], [0.4], [0.3], [0.2], 4)
println("AR : ", round.(ar, digits=4))
println("MA : ", round.(ma, digits=4))
```

[`combined_ar_ma`](@ref) multiplies the seasonal and non-seasonal
polynomials out. A SARIMA is not a separate state-space form — it is an
ordinary ARMA whose coefficients happen to be mostly zero, which is why
`fit_sarima` needs no filter of its own.

## Run the filter

```@example ssm
m = fit_arima(y, (0, 1, 1))
ssm_fit = build_statespace(Float64[], [m.arma.ma[1]])
dy = diff(y)

loglik, sigma2, v, F, converged = kalman_filter(ssm_fit, dy)
println("loglik    : ", round(loglik, digits=4))
println("sigma2    : ", round(sigma2, digits=6))
println("converged : ", converged)
println("matches fit_arima: ", isapprox(loglik, m.arma.loglik; atol=1e-8))
```

Five returns, positionally: log-likelihood, the concentrated `sigma2`,
the one-step prediction errors `v`, their variances `F` (in
`sigma2 = 1` units), and a flag.

That last line is the point of this section. `fit_arima` did nothing
you cannot do by hand — it searched for the coefficient, and this is
the likelihood it was searching on.

**`v ./ sqrt.(F)` is the standardised residual vector**, which is what
`residuals` returns and what the diagnostics expect:

```@example ssm
z = v ./ sqrt.(F)
println("sd = ", round(std(z), digits=5), "   sqrt(sigma2) = ", round(sqrt(sigma2), digits=5))
println("Ljung-Box(10): p = ", round(ljungbox_test(z, 10; fitdf=1).pvalue, digits=4))
```

`sigma2` is concentrated out of the likelihood rather than optimised,
so `F` comes back in units where it equals 1 — scale by `sqrt(sigma2)`
only if you want variances on the original scale.

## Failure is a flag, not an exception

```@example ssm
bad = build_statespace([1.5], Float64[])   # AR coefficient outside the unit circle
lb, sb, vb, Fb, cb = kalman_filter(bad, dy)
println("converged = ", cb, "   loglik = ", lb, "   length(v) = ", length(vb))
```

A non-stationary or numerically degenerate model returns
`converged=false` with `loglik = -Inf` and **empty** `v`/`F` — never a
truncated prefix, on any failure path. It does not throw, because an
optimiser's objective function needs to catch this mid-search and
recover.

!!! warning "`-Inf` is a rejection, not a penalty"
    It carries no gradient direction and produces `NaN` derivatives
    under ForwardDiff. What actually keeps the optimiser inside the
    stationary region here is `partrans`, Monahan's reparameterisation,
    which makes non-stationary points unreachable by construction.

    If you build your own objective on this filter, supply a smooth
    penalty for the `converged=false` case rather than relying on
    `-Inf` to steer the search.

[`kalman_smoother`](@ref) takes the opposite view: it throws on a
degenerate model. It is meant to run once on an already-valid fit for
diagnostics, so there is no search loop to keep alive.

```@example ssm
alpha, V = kalman_smoother(ssm_fit, dy)
println("smoothed states : ", size(alpha), "   covariances: ", length(V))
```

`alpha` is `r × n` — the state estimate at each time using the **whole**
series, not just the past. `V[t]` is its covariance, scaled by
`sigma2`, unlike the filter's internal `P`.

The filter tells you what you knew at time `t`. The smoother tells you
what you know now about time `t`. For anything retrospective —
extracting a trend, imputing a gap, reading off a level shift — the
smoother is the one you want.

## Widening to the general form

```@example ssm
tv, a0, P0f = to_time_varying(ssm_fit, dy)
ll2, v2, F2, c2 = kalman_filter(tv, dy, a0, P0f)
println("TimeVaryingSSM loglik : ", round(ll2, digits=8))
println("GaussianSSM loglik    : ", round(loglik, digits=8))
println("identical: ", isapprox(ll2, loglik; atol=1e-8))
```

[`to_time_varying`](@ref) rewrites a fitted `GaussianSSM` in the
general form and hands back the initial state it implies. The two
filters agree to machine precision, which is the standing check that
the fast path has not drifted from the general one.

Note the different signature: the `TimeVaryingSSM` method takes `a0`
and `P0` explicitly and returns **four** values, not five — there is no
concentrated `sigma2`, because `Q_t` is given rather than estimated.

`to_time_varying` throws rather than returning a flag if the model does
not filter — there is no sensible initial state to construct from a
model that does not work.

## Non-stationary states

The stationary covariance does not exist for a random walk, so a model
with a non-stationary state component cannot be initialised the usual
way. [`kalman_filter_diffuse`](@ref) handles it: the states named in
`diffuse_idx` start with infinite variance, and the filter tracks that
separately until the data has pinned them down.

```julia
kalman_filter_diffuse(tv, y; diffuse_idx=[1], a0=zeros(tv.r), P_star0=zeros(tv.r, tv.r))
```

The diffuse phase costs `O(d)` — a fixed number of steps, `d` being
roughly the number of diffuse states — not `O(n)`. Once `P_infty`
collapses to numerically zero the recursion becomes exactly the
ordinary one. `nobs_diffuse` in the result reports how many
observations it took.

This is the mechanism behind `statsmodels`' full-`n` `nobs` convention:
it augments the state instead of differencing, so it genuinely keeps
every observation. See [Coming from R or
Python](11-coming-from-r-python.md).

## Missing observations

`NaN` entries in `y` are skipped by both `TimeVaryingSSM` methods:
predict-only, with the covariances still propagating through `T_t` and
`R_t Q_t R_t'`, `v[t]` and `F[t]` returned as `NaN`, and no
contribution to the likelihood. A gap costs precision, not validity —
which is the main practical reason to reach for the state-space form
over a direct likelihood.

## See also

- [Part VI](../introduction/29-the-state-space-form.md) — the derivation, and why every model here reduces to this one
- [Chapter 32](../introduction/32-time-varying-systems.md) — drifting coefficients worked end to end
- [API: State Space](../api/state-space.md)
