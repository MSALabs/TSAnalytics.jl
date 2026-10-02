# Fitting ARMA Models

Three functions, nested: [`fit_arma`](@ref) for a stationary series,
[`fit_arima`](@ref) to add differencing, [`fit_sarima`](@ref) to add a
seasonal component. Each is a genuine implementation rather than a
wrapper, but the reduction properties hold exactly — `fit_arima` with
`d=0` matches `fit_arma`, and `fit_sarima` with a null seasonal order
matches `fit_arima`.

```@example fitting
using TSAnalytics, Random

Random.seed!(5)
z = zeros(300)
for t in 2:300
    z[t] = 0.7z[t-1] + randn()
end
y = dataset("cardox").value[1:240]
nothing # hide
```

## Fit a known order

```@example fitting
m = fit_arma(z, (1, 0))
```

Printing a fitted model gives the coefficient table: estimate, standard
error, z-statistic, p-value and a 95% interval per parameter.

```@example fitting
println("ar          : ", round(m.ar[1], digits=4))
println("se          : ", round(m.se[1], digits=4))
println("loglik      : ", round(m.loglik, digits=3))
println("converged   : ", m.converged)
```

With differencing and a seasonal part:

```@example fitting
ms = fit_sarima(y, (1,1,1), (0,1,1,12))
println("AIC: ", round(ms.aic, digits=2), "   BIC: ", round(ms.bic, digits=2))
```

## The `nobs` convention

```@example fitting
println("length(y)             : ", length(y))
println("fit_arima(y,(1,1,1))  : ", fit_arima(y, (1,1,1)).arma.nobs)
println("fit_sarima ... (0,1,1,12): ", ms.nobs)
```

`n - d - D*s`. Differencing consumes observations, and this package
reports the number it actually computed a likelihood on.

!!! warning "This is the most common cross-language surprise here"
    R's `arima` agrees (`n - d`). **`statsmodels` reports the full
    `n`**, because it uses diffuse state augmentation and genuinely
    keeps every observation.

    Since AIC and BIC are both built from `nobs`, **information
    criteria are not comparable between this package and `statsmodels`
    without accounting for it.** Neither convention is wrong; they
    answer different questions about what an observation is once you
    have differenced. See
    [Coming from R or Python](12-coming-from-r-python.md).

## Including a mean

```@example fitting
println("stationary series, include_mean=true : ", round(fit_arma(z, (1,0); include_mean=true).mean, digits=4))
println("differenced series, include_mean=true: ", fit_arima(y, (1,1,1); include_mean=true).arma.mean)
```

**`include_mean` is silently forced off whenever `d > 0` or `D > 0`.**
That is not a bug and it matches R: differencing a constant gives
zero, so the mean's coefficient is not identifiable — there is nothing
in the differenced series for it to estimate. The field comes back
`nothing` rather than `0.0`, so you can tell "not estimated" from
"estimated as zero".

When the mean *is* estimated, it is estimated **jointly** with the
ARMA coefficients, not by subtracting the sample mean first. R does
the same; the two approaches give different answers.

## Standard errors

```@example fitting
for st in (:hessian, :opg, :robust)
    mm = fit_arma(z, (1,0); se_type=st)
    println(rpad(string(st), 9), " se = ", round(mm.se[1], digits=5))
end
```

Three conventions, same fit:

| `se_type` | What it is | Matches |
|---|---|---|
| `:hessian` (default) | Inverse observed-information Hessian | R's `arima` |
| `:opg` | Outer product of gradients | `statsmodels`' default |
| `:robust` | Huber-White sandwich `H⁻¹(J'J)H⁻¹` | Nothing in R — `sandwich::vcovHC` cannot consume an `arima` object |

The coefficients are identical across all three; only the standard
errors move — sometimes enough to change a significance verdict.
[Chapter 18](../introduction/18-fitting-arma.md) compares seven
estimates of one standard error across three languages.

The Hessian is evaluated on the **natural** parameters, not on the
transformed space the optimiser searched. Taking it on the transformed
objective gives plausible-looking numbers that are quietly wrong.

## Residuals

```@example fitting
using StatsAPI: residuals
r = residuals(ms)
println("residuals: ", length(r), "   model nobs: ", ms.nobs)
```

Every fitted type retains the series it was fitted to, so `residuals(m)`
needs nothing else. The two-argument form `residuals(m, y)` computes the
same residuals against a series you supply and throws if the fitted
parameters cannot filter it — useful for checking you have the pairing
right, not for scoring a different series.

The length is `nobs`, not `length(y)` — R pads its own back to the full
input length. Feed these straight to
[the diagnostic tests](02-diagnostics.md).

## When `converged` is false

```@example fitting
println("converged: ", ms.converged)
```

Always check it. `converged=false` means the optimiser stopped without
meeting its tolerance. The fit is still returned — it is not an error — but treat
the coefficients and especially the standard errors with suspicion.

Common causes and what to do:

| Cause | Fix |
|---|---|
| Order too high for the data | Reduce `p`/`q`; check the ACF/PACF first |
| Near a unit root | Difference once more, or accept it |
| Poor starting values | Try `method=:css_ml`, which warm-starts from a conditional-sum-of-squares fit |
| Genuinely flat likelihood | The data may not identify the model |

`NaN` entries in `se` with `converged=true` mean the information matrix
was not positive-definite at the optimum — usually an MA coefficient
sitting on the invertibility boundary, where that parameter's standard
error is genuinely undefined rather than small.

```@example fitting
mb = fit_arma(y, (1, 1))              # undifferenced, so the MA term hits 1.0
println("ma        : ", round(mb.ma[1], digits=6))
println("se        : ", mb.se)
println("converged : ", mb.converged)
```

That is reported honestly rather than papered over with a pseudo-inverse
— and rather than clamped to `0.0`, which is what this used to do and
which printed a `z` of `Inf` against a coefficient the model knew
nothing about.

## Standard errors as a matrix

```@example fitting
using StatsAPI: vcov, stderror, coef
V = vcov(ms)
println("size     : ", size(V), "   coefficients: ", length(coef(ms)))
println("stderror : ", round.(stderror(ms), digits=5))
println("off-diagonal corr(phi, theta): ",
        round(V[1,2] / sqrt(V[1,1]*V[2,2]), digits=4))
```

[`vcov`](https://juliastats.org/StatsAPI.jl/) returns the full covariance
matrix, not just the diagonal that `se` reports. The off-diagonal terms
are the part you cannot recover afterwards, and they are what a joint
test or a linear combination of coefficients needs — a forecast's
variance, for instance, depends on them.

`coef`, `stderror` and `vcov` agree on ordering and length, **including
the estimated mean** when there is one.

## Estimation method

```@example fitting
for mth in (:ml, :css_ml)
    mm = fit_arma(z, (1,0); method=mth)
    println(rpad(string(mth), 7), " ar = ", round(mm.ar[1], digits=6),
            "   loglik = ", round(mm.loglik, digits=4))
end
```

`:ml` (default) optimises the exact likelihood from zero starting
values. `:css_ml` first runs a cheap conditional-sum-of-squares fit and
starts the exact likelihood from there — the same two-stage strategy
R's `arima` uses, and both agree with R's `arima(z, order=c(1,0,0))`
to six decimal places.

On well-behaved data the two land on the same optimum, as above. The
reason to reach for `:css_ml` is a long or heavily parameterised model
where starting from zeros leaves the optimiser a long way from the
answer — the warm start costs little and can be the difference between
`converged=true` and `false`.

!!! note "A warm start that lands on the boundary is discarded"
    The conditional-sum-of-squares objective flattens completely as a
    coefficient approaches the stationarity boundary, to the point
    where its gradient underflows to zero. A gradient-based search that
    oversteps into that region will report convergence there, and
    passing that unit root on as a starting value contaminates the
    exact-likelihood stage.

    This package detects it and retries derivative-free, falling back
    to a cold start if that fails too, so `:css_ml` cannot return a
    worse answer than `:ml` for this reason. R's `arima` guards the
    same failure more bluntly, by refusing outright with
    `non-stationary AR part from CSS`.

## See also

- [Automatic Order Selection](05-automatic-order-selection.md) — when you do not know the order
- [Forecasting](09-forecasting-and-accuracy.md) — what to do with the fit
- [API: ARMA Models](../api/arma-models.md)
