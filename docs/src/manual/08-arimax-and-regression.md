# ARIMAX and Regression with ARIMA Errors

Sometimes the thing you want to explain is not in the series' own past.
A holiday, a policy change, a price, a lockdown — an external variable
that the ARIMA structure has no way to represent.

Three functions cover it, at increasing cost:

| | Estimation | Errors | Use when |
|---|---|---|---|
| [`arx`](@ref) | Conditional least squares | AR lags in the mean | Fast, and the lag structure is simple |
| [`fit_arimax`](@ref) | Joint maximum likelihood | Full ARIMA | You want the ARIMA machinery |
| [`fit_sarimax`](@ref) | Joint maximum likelihood | Full SARIMA | Seasonal data |

```@example arimax
using TSAnalytics, Dates, Statistics
using LinearAlgebra: diag

iip = dataset("iip_india")
y, dt = iip.value, iip.date
println("n = ", length(y), "   ", first(dt), " to ", last(dt))
nothing # hide
```

## A regressor the calendar cannot supply

India's index of industrial production is monthly, so a SARIMA can
represent any effect that recurs at a fixed twelve-month lag. Diwali is
not one: it moves between October and November on the lunar calendar.

```@example arimax
diwali_month = Dict(
    2011=>10, 2012=>11, 2013=>11, 2014=>10, 2015=>11, 2016=>10,
    2017=>10, 2018=>11, 2019=>10, 2020=>11, 2021=>11, 2022=>10,
    2023=>11, 2024=>11, 2025=>10, 2026=>11,
)
Xdiwali = Float64.([month(d) == diwali_month[year(d)] ? 1.0 : 0.0 for d in dt])
Xcovid  = Float64.([Date(2020,3,1) <= d <= Date(2020,8,1) ? 1.0 : 0.0 for d in dt])
println("Diwali-month indicator fires ", Int(sum(Xdiwali)), " times")
println("lockdown indicator fires     ", Int(sum(Xcovid)), " times")
```

This is the canonical case for an exogenous regressor: something with a
real, repeated effect and **no fixed lag**. No amount of extra data
teaches a fixed-period model to see a moving feature.

## Does it earn its place?

```@example arimax
m0 = fit_sarima(y, (1,0,0), (0,1,1,12); include_mean=false)
m1 = fit_sarimax(y, (1,0,0), (0,1,1,12), hcat(Xdiwali, Xcovid); include_mean=false)

println("without regressors : loglik = ", round(m0.loglik, digits=3), "   AIC = ", round(m0.aic, digits=2))
println("with regressors    : loglik = ", round(m1.loglik, digits=3), "   AIC = ", round(m1.aic, digits=2))
```

An AIC improvement of 28 for two parameters. The coefficients say what
happened:

```@example arimax
println("beta          : ", round.(m1.beta, digits=4))
println("standard errs : ", round.(m1.se[1:2], digits=4))
```

Industrial production **falls** by `4.11` index points in the Diwali
month — `t = -3.5`, decisively non-zero. That is the right sign:
Diwali is a holiday, plants shut, and the consumer-spending surge that
people associate with the festival lives in retail, not in factory
output. The lockdown dummy picks up `-20.2`, which needs no
explanation.

`exog` is `n × k`, or a plain vector when `k = 1`, and must have the
same number of rows as `y`. `beta` comes back in column order.

!!! note "`se` covers everything, `beta` first"
    `m1.se` has one entry per estimated parameter — the exogenous
    coefficients first, then the ARMA terms. `m1.beta` has only the
    exogenous ones, so index `se` to match.

## AR-X: the cheap version

[`arx`](@ref) fits the same idea by conditional least squares instead
of maximum likelihood — no Kalman filter, no optimiser, just a QR
decomposition:

```@example arimax
a = arx(y, 2; exog=Xdiwali)
println(a.names)
println("coefficients : ", round.(a.coef, digits=4))
```

It is essentially instant, and it is the right tool when the error
structure is plain AR lags rather than a full ARIMA. Argument names
follow Python's `statsmodels.tsa.ar_model.AutoReg`, which is the
fuller-featured reference — R's `ar.ols` has no exogenous support at
all.

Two capabilities worth knowing about:

```@example arimax
sub = arx(y, [1, 3]; exog=Xdiwali)     # lag subset -- lag 2 skipped entirely
seas = arx(y, 2; seasonal=true, period=12, exog=Xdiwali)
println("lag subset : ", sub.names)
println("seasonal   : ", length(seas.names), " coefficients")
```

`lags` takes a vector for an arbitrary subset, and `seasonal=true` adds
`period-1` deterministic dummies. **Season 1 is the reference
category**, not season `period` — the dummies are named `s(2,12)`
through `s(12,12)`, matching `AutoReg` exactly. A different reference
category is an equally valid parameterisation of the same model, but it
gives genuinely different coefficient values, so this matters when
comparing output.

### Robust standard errors

```@example arimax
for st in (:hessian, :robust)
    aa = arx(y, 2; exog=Xdiwali, se_type=st)
    println(rpad(string(st), 9), " ", round.(sqrt.(diag(aa.vcov)), digits=4))
end
```

The difference is not cosmetic. The AR coefficients' standard errors
nearly **double** under the sandwich estimator, while the Diwali
dummy's *falls* from `2.24` to `0.90`. For a linear model the sandwich
is exactly White's HC0, and this matches R's
`sandwich::vcovHC(fit, type="HC0")` to eight digits.

Heteroskedasticity is the norm on a series that spans a lockdown.
`:hessian` assumes it away; `:robust` does not.

!!! note "Standard errors use the conditional-MLE convention"
    `arx` reports `sigma2 = SSR/n`, matching `AutoReg`'s own `bse`,
    not the degrees-of-freedom-adjusted `SSR/(n-k)` an ordinary OLS
    routine would use. The two differ by `sqrt(n/(n-k))` — about 1 %
    at typical sample sizes. Small, but not nothing.

## Coefficients that drift

`fit_arimax` takes `model=:tvss`, which makes `beta` a **latent
time-varying state** rather than a fixed parameter. What gets estimated
is `Q_beta`, the variance of its random walk:

```@example arimax
mt = fit_arimax(y, (1,0,0), Xdiwali; include_mean=false, model=:tvss)
mm = fit_arimax(y, (1,0,0), Xdiwali; include_mean=false)   # model=:mle, the default

println("Q_beta                : ", round.(mt.Q_beta, sigdigits=4))
println("final filtered beta   : ", round(mt.beta_filtered[end], digits=4))
println("fixed-coefficient beta: ", round.(mm.beta, digits=4))
```

`Q_beta` comes back at `1.8e-18` — numerically zero. The data is saying
the Diwali effect does **not** drift over these fifteen years, and the
filtered coefficient converges to exactly the fixed-coefficient
estimate. That agreement is the reduction property working: with the
drift shut off, the two models describe the same thing.

!!! warning "Never compare `:mle` and `:tvss` likelihoods"
    They are different mathematical objects, not two computations of
    one model. Under `:tvss`, `beta` starts with no prior information
    at all, so its early likelihood contributions are diffuse marginal
    terms rather than ordinary Gaussian densities. Both sums run over
    the same number of observations and the gap is still real.

    Compare **point estimates** instead, exactly as above. `loglik`,
    `aic` and `bic` are not comparable across the two.

## Automatic selection with regressors

[`auto_arimax`](@ref) is `auto_arima` with an `exog` argument — the
same search, with the regressors held in every candidate. The
regressors are not searched over; deciding which external variables
belong in the model is a modelling question, not a fitting one.

## Known limit

**Neither `ArimaxModel` nor `SarimaxModel` has a `forecast` method
yet.** They fit, they report coefficients and standard errors, and they
pass residuals to the diagnostics — but producing a forecast requires
future values of the regressors, and the machinery for that is not
built. `fit_arima`/`fit_sarima` forecasts are unaffected.

## See also

- [Chapter 36](../introduction/36-calendar-effects.md) — this example worked in full, including why a moving festival defeats every decomposition method
- [Forecasting](09-forecasting-and-accuracy.md) — for the models that do forecast
- [API: ARIMAX](../api/arimax.md)
