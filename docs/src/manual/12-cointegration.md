# Cointegration

Two random walks wander. Usually they wander independently, and a
regression of one on the other finds a "relationship" that is nothing but
two trends — Granger and Newbold's spurious regression. Sometimes,
though, some linear combination of them does **not** wander: consumption
and income, a spot and a futures price, two yields on the same curve.
Those series are **cointegrated**, and the stationary combination is an
equilibrium the system keeps returning to.

This chapter covers the three ways this package tests for that, and what
each one is for. The examples use simulated series whose answer is known:

```@example coint
using TSAnalytics, DelimitedFiles
fx(name) = readdlm(joinpath(pkgdir(TSAnalytics), "test", "fixtures", "var", name), ','; skipstart=1)

yes = fx("coint2_yes.csv")   # x a random walk, z = 2 + 1.5x + stationary noise
no  = fx("coint2_no.csv")    # two independent random walks
r1  = fx("coint3_r1.csv")    # three series, one cointegrating relation
r2  = fx("coint3_r2.csv")    # three series sharing one trend: two relations
nothing # hide
```

## Engle-Granger: regress, then test the residual

The oldest test is two steps: regress one series on the others, then ask
whether the residual has a unit root. [`coint_test`](@ref) does exactly
that, running [`adf_test`](@ref) on the residual, and reports MacKinnon's
response-surface p-value for the cointegration case:

```@example coint
coint_test(yes[:, [2, 1]]; lags = 4, autolag = nothing)
```

```@example coint
round(pvalue(coint_test(no; lags = 4, autolag = nothing)); digits = 3)
```

Two things to know before relying on it.

**The first column is special.** It is the dependent variable of the
first-step regression, and the test is not symmetric: put the series
first that you would normalize the relation on.

**Pass the lag if you want to reproduce a number.** At a fixed lag the
statistic agrees exactly with both references. With automatic selection,
`statsmodels` and an R implementation built from `urca::ur.df` search
differently and land on different lags; this package uses
[`adf_test`](@ref)'s search, which is `statsmodels`'.

MacKinnon's p-value surfaces cover at most six series. For seven to
twelve, `coint_test` returns critical values and `pvalue` refuses by
name; beyond twelve nothing is tabulated and the test is refused.

## Phillips-Ouliaris: the residual-based tests that are invariant

Phillips & Ouliaris (1990) proposed two alternatives built on long-run
covariances. [`phillips_ouliaris_test`](@ref) with `kind = :Pz` — the
default — is the useful one: a trace statistic that does not single out
any series, so the column order no longer matters.

```@example coint
t = phillips_ouliaris_test(r1)
```

```@example coint
reject(t; level = 1.0), reject(phillips_ouliaris_test(no); level = 10.0)
```

There is no p-value, only the paper's critical values; [`reject`](@ref)
compares in the right direction, which matters because a **large**
Phillips-Ouliaris statistic rejects while a **small** Engle-Granger one
does.

The `deterministic` choice changes conclusions, not decimals. On a pair
whose equilibrium itself contains a trend, the variance-ratio statistic
`Pu` goes from no rejection to rejection at 1% as the trend is allowed
for:

```@example coint
tr = fx("coint2_trend.csv")
[round(phillips_ouliaris_test(tr; kind = :Pu, deterministic = d).statistic; digits = 2)
 for d in (:none, :const, :trend)]
```

!!! note "This package follows the paper where `urca` does not"
    With a constant or trend, `Pz` is built from demeaned (or detrended)
    data, as Phillips & Ouliaris specify and as their tables assume.
    `urca::ca.po` uses the raw levels; its statistic changes when a
    constant is added to the data, and under the null it almost never
    rejects. `Pu`, and `Pz` without deterministic terms, agree with
    `ca.po` to `1e-10`.

## Johansen: how many relations?

The residual-based tests answer "is there cointegration?". With more
than two series the better question is "how much?" — three series can
share one, two or no relations. Johansen's procedure answers it with a
sequence of likelihood-ratio tests, one per candidate rank:

```@example coint
j = johansen_test(r2; lags = 1)
```

[`coint_rank`](@ref) runs the sequence — test `rank ≤ 0`, then `≤ 1`, …,
and stop at the first hypothesis not rejected:

```@example coint
coint_rank(johansen_test(r1; lags = 1)), coint_rank(j), coint_rank(johansen_test(no; lags = 1))
```

One, two and none: the three designs, recovered.

[`cointegration_plot`](@ref) shows what was found: each estimated
relation `β'y` over time. A real equilibrium wanders around a fixed level
and keeps coming back.

```@example coint
using Plots
plot(cointegration_plot(r2, j); size = (700, 450))
```

### `lags` counts differenced lags

`lags` is the number of lagged **differences** in the error-correction
form — `statsmodels`' `k_ar_diff`. `urca` counts the levels VAR's order
instead, so its `K = 2` is `lags = 1` here. It is required: there is no
safe default.

### Deterministic terms: read the table, not the names

| `deterministic` | meaning | `urca` | `statsmodels` |
|---|---|---|---|
| `:none` | nothing | — | `det_order = -1` |
| `:const` | unrestricted constant (the data drift) | `ecdet = "none"` | `det_order = 0` |
| `:const_restricted` | constant only inside the relation | `ecdet = "const"` | — |
| `:trend` | unrestricted constant and trend | — | — |
| `:trend_restricted` | trend only inside the relation | `ecdet = "trend"` | — |

`urca`'s `ecdet = "none"` is the single most common source of confusion:
it still fits a constant, just not inside the cointegrating relation. It
is `:const`.

### Critical values, and the one place this disagrees with `urca`

There is no p-value — the tables are quantiles (Osterwald-Lenum 1992),
and [`critical_values`](@ref) returns them. Each deterministic case uses
the table the paper assigns it. For `:const` that is Table 1, which
`statsmodels`' own simulations agree with; `urca` uses Table 1.1\*, a
side case where the data have no drift, and reports larger critical
values. The test statistics themselves agree with both references to
`1e-10`.

### The cointegrating vectors

The eigenvectors are only defined up to scale, so
[`normalize_beta`](@ref) puts them in a conventional form — first element
one (`urca`'s `V`), or an identity block (`cajorls`):

```@example coint
round.(normalize_beta(j, 2; scheme = :identity); digits = 3)
```

`r2`'s design has relations `d₁ + d₃` and `d₂ + 2d₃`, and the third row
recovers `(1, 2)`. Beyond the true rank the extra vectors are still
returned, but they mean no more than the rank you asserted.

## The VECM: modelling the system you found

Once the rank is known, [`fit_vecm`](@ref) estimates the error-correction
model it implies,

```math
\Delta y_t = lpha eta' y_{t-1} + \Gamma_1 \Delta y_{t-1} + \dots + \Gamma_k \Delta y_{t-k} + \mu + u_t
```

where `β' y_{t-1}` is how far the system is from its equilibria and `α`
how fast each series moves back:

```@example coint
v = fit_vecm(r1; lags = 1, rank = 1)
```

```@example coint
round.(vec(v.alpha); digits = 3)
```

The third series does almost all of the adjusting; the first two barely
respond to the disequilibrium. Use the same `deterministic` as the
Johansen test that chose the rank — the vocabulary is shared so that the
two cannot silently refer to different models, and they give the same
`β`.

`method = :ml` (the default) is Johansen's maximum likelihood, what all
three references compute. `method = :twostep` is Engle and Granger's
static-regression estimator: consistent but less efficient, rank 1 only,
and a different number — not a normalization of the same one.

Forecasts and impulse responses come from the model written as a VAR in
levels, [`var_rep`](@ref), through the same recursions as
[`fit_var`](@ref)'s. That VAR has unit roots **by construction** — that
is what cointegration means — so its forecasts' intervals keep widening,
its impulse responses settle at a permanent effect instead of dying out,
and [`lr_effects`](@ref) refuses it.

## Which test to use

- **Two series, a natural dependent variable:** [`coint_test`](@ref) —
  simple, and it has a p-value.
- **Two to six series, no natural normalization:**
  [`phillips_ouliaris_test`](@ref) with `kind = :Pz`.
- **More than two series, or you need the number of relations and the
  vectors themselves:** [`johansen_test`](@ref), then [`fit_vecm`](@ref)
  at the rank it chooses.

All three test the null of **no** cointegration. Failing to reject is
not evidence for it; tests of the reverse null exist but behave badly,
for reasons Phillips & Ouliaris (1990, §6) set out.

## See also

- [Vector Autoregression](11-vector-autoregression.md) — the stationary case
- [Coming from R or Python](14-coming-from-r-python.md) — the reference divergences in one place
- [API: Multivariate](../api/multivariate.md)
