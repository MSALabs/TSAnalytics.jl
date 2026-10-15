# Regime Switching

A Markov-switching model lets a series' parameters — its mean, its
variance, its autoregressive coefficients — jump between a small number of
**regimes** that are never observed directly. The regime follows a Markov
chain: which regime the series is in this period depends only on which it
was in last period, through a matrix of transition probabilities that is
estimated with everything else. What comes back is, for every period, the
probability that the series was in each regime.

!!! warning "The transition matrix is left-stochastic"
    `P[i, j] = Pr(S_t = i | S_{t−1} = j)`: **columns** sum to one. This is
    Hamilton's convention (1994, printed p. 678, PDF 693) and `statsmodels`'.
    Shumway & Stoffer (2017, Property 6.7) write the transpose, with rows
    summing to one. A transposed `P` gives plausible-looking nonsense, so
    check which one a formula assumes before using it.

## Hamilton's model of the business cycle

Hamilton (1989) fitted an AR(4) to US real GNP growth whose **mean**
switches between two regimes, with the autoregression taken around the
regime's mean:

```math
y_t - \mu_{S_t} = \sum_{j=1}^{4} \phi_j \,(y_{t-j} - \mu_{S_{t-j}}) + \varepsilon_t,
\qquad \varepsilon_t \sim N(0, \sigma^2)
```

That is [`fit_ms_autoregression`](@ref). Because the lags are taken around
*their own* regimes' means, the likelihood depends on the last five
regimes, and the filter runs over `2⁵ = 32` combinations — the expansion
Hamilton describes (1994, printed p. 698, PDF 713).

```@example ms
using TSAnalytics, Plots, DelimitedFiles
using StatsAPI: coef, coefnames
gnp = Float64.(readdlm(TSAnalytics.HAMILTON_GNP, ',', skipstart = 1)[:, 2])
m = canonicalise(fit_ms_autoregression(gnp, 4))
```

[`canonicalise`](@ref) orders the regimes by their mean, so regime 1 —
labelled `0` in the parameter names, as `statsmodels` does — is the low
one. The estimates against Hamilton's own Table 22.1 (1994, printed p. 698,
PDF 713), which prints them to two decimals:

```@example ms
hamilton = Dict("const[0]" => -0.36, "const[1]" => 1.16, "sigma2" => 0.59,
                "ar.L1" => 0.01, "ar.L2" => -0.06, "ar.L3" => -0.25, "ar.L4" => -0.21)
for (nm, b) in zip(coefnames(m), coef(m))
    haskey(hamilton, nm) && println(rpad(nm, 9), lpad(round(b, digits = 4), 8), "   Hamilton: ", hamilton[nm])
end
println("P[2,2] (stay in expansion) = ", round(m.P[2, 2], digits = 4), "   Hamilton p₁₁ = 0.90")
println("P[1,1] (stay in recession) = ", round(m.P[1, 1], digits = 4), "   Hamilton p₂₂ = 0.75")
```

Every estimate agrees to the two decimals printed. Hamilton numbers the
regimes the other way round — his state 1 is expansion — so his `p₁₁` is
our `P[2,2]`. He prints no log-likelihood; `statsmodels`' is `−181.26339`,
and this package matches it to `8.5e-14`.

The last line of the model's display matters: the restarts found **two**
optima. The likelihood of a switching model is multimodal, and the second
one, at `−183.67`, is a degenerate solution with almost no persistence.
[`fit_ms_autoregression`](@ref) runs several starts on threads and reports
how many distinct optima they reached, so a fit that landed on the wrong
one does not pass silently.

## Regime probabilities

The filter gives `Pr(S_t | data up to t)`; the smoother revises it with the
whole sample. [`regime_probabilities`](@ref) returns either, and
[`regime_plot`](@ref) draws them:

```@example ms
plot(regime_plot(m); size = (700, 440))
```

Where the dashed filtered line and the solid smoothed one disagree is
where hindsight helps: a quarter of weak growth looks like the start of a
recession in real time and is revised once the following quarters show
recovery. The standard figure shades the periods the smoother assigns to
the low regime:

```@example ms
plot(regime_plot(m); mode = :shaded, regime = 1, size = (700, 300))
```

How long a regime lasts once entered is `1 / (1 − P[i,i])`:

```@example ms
println("expected durations (quarters): ", round.(expected_durations(m), digits = 2))
```

about four quarters of recession and ten of expansion. Those two numbers
go in a table, not a chart.

## Regressions, switching variances, covariates

[`fit_ms_regression`](@ref) is the order-0 case and allows regressors:
`y_t = c_{S_t} + x_t′β + ε_t`, with `switching_exog`, `switching_variance`
and (for the autoregression) `switching_ar` choosing what switches. A
specification in which nothing switches is refused — its transition
probabilities are not identified, and `statsmodels` silently returns its
starting values for them. `exog_tvtp` makes the transition probabilities
depend on covariates (Diebold, Lee & Weinbach 1994).

Residual diagnostics can be read per regime: `diagnostic_plot(m; regime = r)`
weights each period's residual by the probability that it was in regime
`r`, so the periods the model puts elsewhere fade out.

```@example ms
plot(diagnostic_plot(m; regime = 2); size = (700, 480))
```

## India: a variance regime

Year-on-year growth of the Index of Industrial Production, April 2012 to
March 2026, with a mean and a variance that both switch:

```@example ms
using Dates
iip = dataset("iip_india")
yi = 100 .* (log.(iip.value[13:end]) .- log.(iip.value[1:(end - 12)]))
di = iip.date[13:end]
mi = canonicalise(fit_ms_regression(yi; switching_variance = true))
```

```@example ms
turbulent = di[regime_probabilities(mi)[:, 1] .> 0.5]
println(length(turbulent), " months in the turbulent regime: ", Dates.format.(turbulent, "yyyy-mm"))
```

```@example ms
plot(regime_plot(mi); mode = :shaded, regime = 1, size = (700, 300),
     xlabel = "months from April 2012")
```

The model finds what a reader would: a calm regime of steady growth and a
turbulent one with a variance about 150 times larger, occupied by the
pandemic months of 2020 — and again twelve months later, in 2021, when
year-on-year growth compares the recovery with the collapse. A switching
model fitted to year-on-year rates sees a shock twice; that is a property
of the transformation, and worth knowing before reading a second "crisis"
into the second episode. The handful of isolated months outside those
spells — March 2013, autumn 2019, mid-2022 — are single large surprises
that the turbulent regime's wide variance absorbs more cheaply than the
calm one's.

## See also

- [API: Regime Switching](../api/regime-switching.md)
- [Threshold and Smooth-Transition Models](20-threshold-models.md) — regimes set by an observed variable instead of a hidden chain
- [Coming from R or Python](16-coming-from-r-python.md) — `MSwM` and `statsmodels`, and where they go wrong
