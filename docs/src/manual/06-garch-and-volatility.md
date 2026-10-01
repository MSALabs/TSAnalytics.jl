# GARCH and Volatility

Everything so far has modelled the *level* of a series. GARCH models
its **variance** — the observation that on financial data, large moves
cluster together and small moves cluster together, even though the
direction of the next move stays unpredictable.

```@example garch
using TSAnalytics, Statistics

r = dataset("sp500.gr").value   # 2,728 daily S&P 500 returns, 2001-2011
println("n = ", length(r), "   sd = ", round(std(r), digits=5))
nothing # hide
```

## Establish that you need one

```@example garch
println("Ljung-Box(10) : p = ", round(ljungbox_test(r, 10).pvalue, digits=6))
println("ARCH-LM(12)   : p = ", round(arch_lm_test(r, 12).pvalue, digits=6))
```

[`arch_lm_test`](@ref) is the entry ticket. It regresses squared
returns on their own lags: if the squares are predictable, the variance
is not constant and a GARCH term has something to explain. Fit one
without checking and you will sometimes fit three parameters to
describe noise.

See [Diagnostics](02-diagnostics.md) for the case where the two tests
disagree — which is the interesting one.

## Fit it

```@example garch
m = fit_garch(r, 1, 1)
```

```@example garch
println("omega     : ", round(m.omega, sigdigits=4))
println("alpha     : ", round(m.alpha[1], digits=5))
println("beta      : ", round(m.beta[1], digits=5))
println("persistence (alpha+beta): ", round(m.alpha[1] + m.beta[1], digits=6))
println("converged : ", m.converged)
```

`alpha` is how much yesterday's *shock* moves today's variance; `beta`
is how much yesterday's *variance* does. Their sum is persistence, and
at `0.992` this series forgets a volatility shock very slowly — which
is what a decade containing 2008 looks like.

Persistence must stay below `1` for the unconditional variance to
exist. At `0.992` it does, barely; the implied long-run daily
volatility is

```@example garch
println("unconditional sigma: ", round(sqrt(m.omega / (1 - m.alpha[1] - m.beta[1])), digits=5))
```

!!! warning "`p` and `q` are the ARCH and GARCH orders, in that order"
    `fit_garch(y, p, q)` takes **`p` as the ARCH order** (the `alpha`
    terms, on squared residuals) and **`q` as the GARCH order** (the
    `beta` terms, on lagged variance).

    This matches Python's `arch_model(p=, q=)` and R's
    `rugarch::garchOrder=c(p,q)` — both ecosystems agree. It is the
    **opposite** of Bollerslev's original notation, which some
    textbooks still quote. For GARCH(1,1) it does not matter; for
    GARCH(2,1) it very much does.

## The mean equation is not fitted by default

`mean_spec` defaults to `:zero`, not Python's `mean='Constant'`. That
is deliberate: GARCH's job here is the variance equation, and the
workflow this package is built around is to fit a mean model first and
hand `fit_garch` its residuals.

```julia
ma = fit_arma(y, (1, 1))
mg = fit_garch(residuals(ma), 1, 1)
```

Pass `mean_spec=:constant` for parity with Python's default. When you
do, `mu` is estimated **jointly** with the variance parameters rather
than by subtracting the sample mean first — the same choice
[`fit_arma`](@ref) makes about `include_mean`.

## Asymmetry: the leverage effect

A plain GARCH treats a −3 % day and a +3 % day as identical news.
Markets do not. Two models add an asymmetry term:

```@example garch
mg = fit_garch(r, 1, 1; model=:gjr)
me = fit_garch(r, 1, 1; model=:egarch)

for (lab, mm) in (("garch", m), ("gjr", mg), ("egarch", me))
    println(rpad(lab, 7), " loglik = ", round(mm.loglik, digits=2),
            "   AIC = ", round(mm.aic, digits=1))
end
println("GJR alpha : ", round(mg.alpha[1], digits=5))
println("GJR gamma : ", round(mg.gamma[1], digits=5))
```

Read those two GJR numbers together, because they are more emphatic
than a leverage effect usually is. `alpha` — the symmetric response,
applying to shocks of either sign — is driven to **zero**. All of the
news impact has moved into `gamma`, which applies only when the shock
was negative.

On this series, over this decade, the model's answer is that a
*positive* return carries essentially no information about tomorrow's
variance and a negative one carries all of it. The AIC improvement of
113 units over plain GARCH is not a rounding error, and neither is the
finding: asymmetry is the single most reliable stylised fact in equity
returns.

| `model` | Recursion | Asymmetry enters as |
|---|---|---|
| `:garch` | On `sigma2` | — |
| `:gjr` | On `sigma2` | `gamma * e[t-1]^2 * I(e[t-1] < 0)` |
| `:egarch` | On `log(sigma2)` | `gamma * z[t-1]` |

!!! note "`omega < 0` is correct for EGARCH"
    EGARCH models the log-variance, so its `omega`, `alpha` and `gamma`
    carry no positivity constraint. `me.omega` comes back negative here
    and that is the expected result, not a failed fit.

`gamma` is fixed at order 1 for both, regardless of `p` — the
overwhelmingly standard GJR(1,1)/EGARCH(1,1) usage, matching
`arch_model(o=1)`.

## Standard errors

```@example garch
mc = fit_garch(r, 1, 1; cov_type=:classic)
println("robust  : ", round.(m.se, sigdigits=4))
println("classic : ", round.(mc.se, sigdigits=4))
```

`cov_type` defaults to **`:robust`** — the Bollerslev-Wooldridge QMLE
sandwich — matching Python's `arch` and standard GARCH practice,
because financial returns routinely violate the conditional normality
the likelihood assumes. The classic errors here run a fifth to two
fifths smaller — exactly the direction you would expect them to be
wrong in, and enough to change a marginal significance verdict.

Note the spelling: GARCH uses `cov_type=:robust|:classic` while the
ARMA family uses `se_type=:hessian|:opg|:robust`. The two are kept
distinct on purpose, because each matches the reference implementation
its users are coming from.

## Diagnose it

Fitting a GARCH does not excuse you from checking it. Standardise the
residuals and run the same tests again:

```@example garch
z = m.resid ./ sqrt.(m.sigma2)
println("Ljung-Box(10) on z  : p = ", round(ljungbox_test(z, 10).pvalue, digits=4))
println("ARCH-LM(12) on z    : p = ", round(arch_lm_test(z, 12).pvalue, digits=5))
```

The portmanteau test is now clean. **ARCH-LM still rejects at
`p = 0.003`** — GARCH(1,1) has absorbed most of the clustering but not
all of it. That is a genuine finding about this series, not a defect,
and it is why the asymmetric models improved the fit as much as they
did.

```@example garch
sb = sign_bias_test(z, m.resid)
println("sign bias       : p = ", round(sb.sign_bias_pvalue, digits=4))
println("positive bias   : p = ", round(sb.positive_sign_bias_pvalue, digits=4))
println("joint effect    : p = ", round(sb.joint_effect_pvalue, sigdigits=3))
```

[`sign_bias_test`](@ref) asks specifically whether the sign of the
previous shock still matters after the model has had its say. The joint
test rejects decisively, confirming what the GJR fit found — and it
still rejects on the GJR residuals, so even the asymmetric model has
not captured all of it.

Honest summary for this series: GJR-GARCH is clearly better than GARCH,
and neither is finished.

### The six-panel view

```@example garch
d = diagnostic_plot(m)
println("panels backed by: ", propertynames(d))
```

Handed a [`GarchModel`](@ref), [`diagnostic_plot`](@ref) returns the
variance-model display — fitted conditional volatility, absolute
residuals, standardised residuals, their ACF, the ACF of their squares,
a Q-Q plot, and the **news impact curve**, which draws variance
response against shock size and makes the asymmetry visible directly.

```@example garch
println("news impact curve for :egarch -> ",
        diagnostic_plot(me).news_impact_e === nothing ? "not available" : "available")
```

It is `nothing` for `:egarch`, whose log-variance recursion has no
directly comparable curve on the variance scale.

## Forecast the variance

```@example garch
fc = forecast_volatility(m, 20)
println("method     : ", fc.method)
println("sigma h=1  : ", round(sqrt(fc.variance[1]), digits=5))
println("sigma h=20 : ", round(sqrt(fc.variance[end]), digits=5))
```

The forecast decays from the current conditional volatility toward the
unconditional level — slowly, because persistence is `0.992`. That
decay *is* the forecast; a GARCH model has nothing to say about the
direction of returns, only about their scale.

`method=:auto` (the default) picks `:analytic` where a closed form
exists (`:garch`, `:gjr`) and `:simulation` where it does not:

```@example garch
using Random
Random.seed!(1)
fe = forecast_volatility(me, 5)
println("EGARCH: method = ", fe.method, ", simulations = ", fe.simulations,
        ", paths = ", size(fe.variance_paths))
```

Asking for `:analytic` explicitly on an EGARCH beyond one step throws
rather than silently substituting simulation under the requested name —
which is what `arch` does too.

`variance_paths` is the full simulated ensemble, so you can take
quantiles from it rather than only the mean:

```@example garch
q = [0.05, 0.5, 0.95]
println("5/50/95% of sigma at h=5: ",
        round.(sqrt.(quantile(fe.variance_paths[:, end], q)), digits=5))
```

## Several series at once

[`fit_garch_multi`](@ref) fits the same specification across a vector
of series, matching `rugarch::multifit`. `n_restarts > 1` refits from
that many randomised starting points and keeps the best *converged*
result, matching `rugarch`'s `gosolnp` — worth reaching for when a fit
comes back `converged=false`.

Both take `parallel=true`, which engages only when Julia is started
with more than one thread. A single fit is inherently sequential: each
`sigma2[t]` depends on `sigma2[t-1]`, so there is nothing within one
likelihood evaluation to parallelise.

## Known limits

| | |
|---|---|
| `dist=:t` | Accepted in the signature, throws a clear error. Normal innovations only. |
| `gamma` order | Fixed at 1 for `:gjr`/`:egarch`. |
| `rugarch` comparison | Do not compare likelihoods directly — see below. |

!!! warning "`rugarch` seeds the variance recursion differently"
    This package backcasts the first variance with an
    exponentially-weighted average of the first 75 squared residuals
    (decay `0.94`), matching `arch` exactly. `rugarch` assigns the
    whole-sample mean of squared residuals instead.

    That single number was worth `1.05` log-likelihood units on a
    1,260-point series. Holding parameters fixed and changing only the
    seed reproduces `rugarch`'s likelihood to `1.58e-09` — so the
    formulas are otherwise identical and the entire gap is the seed.
    `rugarch` figures cannot be used as targets for anything
    likelihood-based without aligning it first. See
    [Coming from R or Python](11-coming-from-r-python.md).

## See also

- [Chapters 24–28](../introduction/24-why-variance-changes.md) — why variance changes, and what each model assumes about it
- [Diagnostics](02-diagnostics.md) — the tests used above
- [API: GARCH](../api/garch.md)
