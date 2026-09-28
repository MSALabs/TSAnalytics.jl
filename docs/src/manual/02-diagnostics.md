# Diagnostics

Twelve hypothesis tests, covering three separate questions: is this
series stationary, did my model capture the structure, and is the
variance behaving.

All of them return a result type with `.statistic` and `.pvalue`, and
none of them prints a verdict — deciding what a p-value means is the
caller's job, not the library's.

```@example diagnostics
using TSAnalytics, Random

Random.seed!(3)
wn = randn(400)                 # white noise
Random.seed!(3)
rw = cumsum(randn(400))         # a random walk
nothing # hide
```

## Is it stationary?

Three tests, and you should routinely run at least two:

```@example diagnostics
println("random walk : ADF p=", round(adf_test(rw).pvalue, digits=4),
        "   KPSS p=", round(kpss_test(rw).pvalue, digits=4),
        "   PP p=", round(pp_test(rw).pvalue, digits=4))
println("white noise : ADF p=", round(adf_test(wn).pvalue, digits=4),
        "   KPSS p=", round(kpss_test(wn).pvalue, digits=4),
        "   PP p=", round(pp_test(wn).pvalue, digits=4))
```

**The reason to run two is that their nulls are opposites.**
[`adf_test`](@ref) and [`pp_test`](@ref) have "there is a unit root" as
the null; [`kpss_test`](@ref) has "the series is stationary" as the
null. One test alone cannot separate "the data says stationary" from
"the data says nothing".

| ADF | KPSS | Conclusion |
|---|---|---|
| rejects | does not reject | Stationary |
| does not reject | rejects | Unit root — difference it |
| does not reject | does not reject | Underpowered; the data cannot resolve it |
| rejects | rejects | Neither fits — suspect a trend or a break |

Note the random walk's ADF p-value above is `0.22`, not something
dramatic. **ADF has low power against a persistent alternative**, which
is exactly why the KPSS result (`0.005`, decisive) earns its place.

!!! note "KPSS p-values are clamped"
    `kpss_test` returns p-values from a printed table with bounds at
    `0.01` and `0.10`, so `0.005` and `0.2` mean "beyond the table" in
    each direction rather than precise values. Both R and `statsmodels`
    do the same and warn the same way.

`kpss_test` takes `nlags=:short` by default, matching R. Pass
`nlags=:auto` for the Hobijn data-dependent rule, which is
`statsmodels`' default.

## Did the model capture the structure?

```@example diagnostics
lb = ljungbox_test(wn, 10)
println("Ljung-Box(10): Q = ", round(lb.statistic, digits=4), "   p = ", round(lb.pvalue, digits=4))
```

[`ljungbox_test`](@ref) pools autocorrelation across lags into one
statistic. On model residuals, pass `fitdf` — the number of ARMA
parameters estimated — so the degrees of freedom are charged correctly.
Omitting it makes the test too forgiving.

```@example diagnostics
bp = ljungbox_test(wn, 10; boxpierce=true)
println("Box-Pierce: Q = ", round(bp.bp_statistic, digits=4), "   p = ", round(bp.bp_pvalue, digits=4))
```

!!! warning "Two divergences live here"
    **The default statistic differs from R.** `Box.test` defaults to
    Box-Pierce; this defaults to Ljung-Box, which has better
    finite-sample properties. Pass `boxpierce=true` for R's default.

    **A vector of lags means something different from Python.**

    ```@example diagnostics
    println("exactly lags {5,10}: Q = ", round(ljungbox_test(wn, [5,10]).statistic, digits=4))
    ```

    This sums **exactly those two lags** into one statistic. Python's
    `acorr_ljungbox(y, lags=[5,10])` instead returns **two cumulative
    statistics**, through lag 5 and through lag 10. Pass an `Integer`
    for the cumulative behaviour both packages share.

### Seasonality specifically

```@example diagnostics
monthly = [100 + 10sin(2π*t/12) + randn() for t in 1:240]
println("QS test p = ", round(qs_test(monthly, 12).pvalue, digits=6))
```

[`qs_test`](@ref) targets the seasonal lags rather than spreading
attention across all of them. On monthly data it is the test most
likely to catch a real failure, because the seasonal lag is where a
twelve-month model most often falls short — and a portmanteau test
diluted across 24 lags can miss it.

### Regression residuals

```@example diagnostics
println("Durbin-Watson: ", round(durbin_watson_test(wn).statistic, digits=4))
```

[`durbin_watson_test`](@ref) tests first-order autocorrelation
specifically. `2` means none; below `2` positive, above `2` negative.
Pass the design matrix `X` for the exact p-value (the Pan/Farebrother
algorithm) rather than the normal approximation.

## Is the variance behaving?

This is a genuinely separate question, and the most common place a
"finished" model turns out not to be:

```@example diagnostics
Random.seed!(9)
e = zeros(600); s2 = ones(600)
for t in 2:600
    s2[t] = 0.05 + 0.12*e[t-1]^2 + 0.85*s2[t-1]
    e[t] = sqrt(s2[t])*randn()
end
println("Ljung-Box(10) : p = ", round(ljungbox_test(e, 10).pvalue, digits=4))
println("ARCH-LM(12)   : p = ", round(arch_lm_test(e, 12).pvalue, digits=6))
```

**Read those two together.** The portmanteau test says there is no
linear structure left — `p = 0.94`, about as clean as it gets. The
ARCH-LM test says the *squares* are strongly predictable —
`p = 0.005`. Both are correct. The sign of the next value is
unpredictable; its **size** is not.

A model can pass every autocorrelation test and still be wrong about
what it claims to know, which is what [GARCH](06-garch-and-volatility.md)
exists to fix.

[`dk_heteroskedasticity_test`](@ref) is the other variance test — a
Durbin-Koopman F-test comparing the variance of the first and last
thirds of a residual series. It is better than ARCH-LM at
distinguishing a one-off level shift in variance from genuine
clustering, because a shift and clustering both produce squared-residual
autocorrelation.

## Normality

```@example diagnostics
println("Jarque-Bera p = ", round(jarque_bera_test(wn).pvalue, digits=4))
```

[`jarque_bera_test`](@ref) combines skewness and excess kurtosis.
Failing it rarely invalidates point forecasts, but it does undermine
*interval* forecasts, which assume the distribution it is testing.

## Variance-model diagnostics

Two tests apply specifically to a fitted conditional-variance model.
Both were implemented to match `rugarch` rather than their originating
papers — the two disagree on construction, and the reference
implementation is what people compare against.

| Test | Asks |
|---|---|
| [`sign_bias_test`](@ref) | Is there asymmetry a symmetric GARCH missed — do negative shocks affect variance differently? |
| [`nyblom_test`](@ref) | Did the fitted parameters stay constant across the sample? |

```@example diagnostics
m = fit_garch(e, 1, 1)
sb = sign_bias_test(m.resid ./ sqrt.(m.sigma2), m.resid)
println("sign bias, joint effect: p = ", round(sb.joint_effect_pvalue, digits=4))
```

`nyblom_test` takes a score matrix rather than a model — the
per-observation gradient contributions — so it applies to anything that
can produce them. See [Chapter 32](../introduction/32-time-varying-systems.md)
for a worked example, and note it reports **critical values rather than
a p-value**, because its null distribution is non-standard and
tabulated only at three levels.

## All at once

```@example diagnostics
d = diagnostic_plot(wn; period=1)
println("lags examined: ", d.nlag, "   minimum Ljung-Box p: ", round(minimum(d.ljungbox_pvalues), digits=4))
```

[`diagnostic_plot`](@ref) assembles the standard four-panel residual
display and returns the values behind it. **Pass `period=12` for
monthly data** — the default lag count is `20`, which never reaches the
seasonal lag; supplying the period raises it to `36`.

Handed a [`GarchModel`](@ref) instead of residuals, it returns six
variance-model panels — see [GARCH](06-garch-and-volatility.md).

## See also

- [Was It Any Good?](../getting-started/03-was-it-any-good.md) — the same tests as a workflow
- [Coming from R or Python](11-coming-from-r-python.md) — every default that differs
- [API: Diagnostics](../api/diagnostics.md) — full argument lists
