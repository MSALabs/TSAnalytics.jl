# Vector Autoregression

Everything up to here has modelled one series at a time. A vector
autoregression models several together, letting each one depend on the
recent past of all of them:

```math
y_t = \nu + A_1 y_{t-1} + \dots + A_p y_{t-p} + u_t
```

where `y_t` is a `K`-vector and each `A_j` is a `K × K` matrix. That is
the whole model. Its usefulness is that it gives you somewhere to ask
the questions a univariate model cannot even phrase: does one series
help predict another, and if one is shocked, what happens to the rest.

## Fitting one

`Y` has **rows = time and columns = variables** — the orientation both
reference implementations use, and what the bundled multivariate
datasets carry. [`fit_var`](@ref) accepts anything
[`tsmatrix`](@ref) understands, including the `NamedTuple` that
[`dataset`](@ref) returns, so no unpacking is needed:

```@example var
using TSAnalytics
using StatsAPI: coefnames, aic      # not re-exported; StatsAPI owns the generics

m = fit_var(dataset("econ5"), 2)
```

Note the last line of that summary. On this data the fitted VAR(2) is
**not stable** — its largest companion root is just over 1 — which is
worth knowing before forecasting from it, and is the sort of thing that
is easy to miss if the model does not say so.

The coefficients come back as `A[j]`, a `K × K` matrix per lag, with a
separate `det_coef` for the deterministic terms:

```@example var
println("A[1] is ", size(m.A[1]), ", A[2] is ", size(m.A[2]))
println("deterministic block: ", size(m.det_coef))
println("coefnames: ", coefnames(m)[1:3], " … ", coefnames(m)[end])
```

Neither reference stores it that way — `vars` returns one flat
`K × (Kp + ndet)` matrix with the deterministic terms last,
`statsmodels` a transposed `(ndet + Kp) × K` with them first — and they
were verified to agree with each other once reordered. Separating the
two removes the ordering question from everything built on top. If you
want the flat form, `coef(m)` gives it, in R's order.

### Deterministic terms

```@example var
println(fit_var(dataset("econ5"), 1; deterministic=:none).loglik)
println(fit_var(dataset("econ5"), 1; deterministic=:const).loglik)
println(fit_var(dataset("econ5"), 1; deterministic=:both).loglik)
```

Five values are available, and they deliberately span **both**
references' vocabularies, which are mutually incomplete:

| `deterministic` | terms | R `type=` | `statsmodels` `trend=` |
|---|---|---|---|
| `:none` | — | `"none"` | `"n"` |
| `:const` | 1 | `"const"` | `"c"` |
| `:trend` | t | `"trend"` | *no equivalent* |
| `:both` | 1, t | `"both"` | `"ct"` |
| `:quadratic` | 1, t, t² | *no equivalent* | `"ctt"` |

!!! warning "`:trend` is not `\"ct\"`"
    R's `type="trend"` means a trend with **no constant**, while
    `statsmodels`' `trend="ct"` means constant **and** trend. They sound
    equivalent and are not: on the bundled fixture they differ by `6.15`
    in log-likelihood, and the mismatch propagates into order selection.
    That is why the keyword here is spelled like neither of them.

## Choosing the order

```@example var
s = var_select(dataset("econ5"); lag_max=6)
```

Four criteria, and they disagree — which is normal, not a problem. AIC
and FPE pick the longest lag they are offered here, BIC the shortest.

### The sample is held fixed, and that is the whole subtlety

Every candidate order is estimated on the **same** `T = n - lag_max`
observations, so the criteria are comparable with one another. A single
fitted model uses all `n - p` observations available to it, which is
more. Both numbers are correct and both references report both:

```@example var
println("var_select, lag 2 : ", round(s.aic[2], digits=10), "   (T = ", s.T, ")")
m2 = fit_var(dataset("econ5"), 2)
println("aic(fit_var(…, 2)): ", round(aic(m2), digits=10), "   (T = ", m2.T, ")")
```

If you want the selection figure for one order, slice the data yourself
— `fit_var(Y[lag_max - p + 1:end, :], p)` reproduces it exactly.

## Stability

```@example var
m1 = fit_var(dataset("econ5"), 1)
println("roots      : ", round.(roots(m1)[1:3], digits=6))
println("is_stable  : ", is_stable(m1))
```

!!! warning "The two references mean different things by `roots`"
    This is the one convention in this chapter most likely to cause real
    harm, because getting it wrong silently **inverts** a stability test.

    `vars::roots()` returns the moduli of the companion-matrix
    eigenvalues: stable if all `< 1`. `statsmodels`' `.roots` returns the
    roots of the characteristic polynomial: stable if all `|·| > 1`.
    They are exact reciprocals, verified on five fixtures to between
    `1.1e-16` and `9.3e-15`.

    [`roots`](@ref) follows R, because "the largest root is 0.97, close
    to the boundary" is the reading practitioners expect. For
    `statsmodels`' convention take `1 ./ roots(m)`.

## Forecasting

```@example var
f = forecast(m2, 4)
println("point is ", size(f.point), ", bands are ", size(f.lower))
println("unemp: ", round.(f.point[:, 1], digits=4))
```

A VAR forecast is `horizon × K` points with `horizon × K × levels`
bands, so it returns a [`VARForecast`](@ref) rather than the univariate
[`Forecast`](@ref) nine other model families use. Generalising that type
would have been a breaking change to all of them.

The intervals come from the df-adjusted `Σ` accumulated through the
moving-average representation, and — as in both references — they
**treat the estimated coefficients as known**, so they are slightly too
narrow. The same honesty applies as to [`fit_ets`](@ref)'s `sigma2`.

## Granger causality

Does one variable help predict another, beyond what the other's own past
already says? That is a question about *predictive* content, not about
mechanism, and the name has misled people for fifty years.

```@example var
t = granger_test(m2; cause = :gnp)
```

`caused` defaults to every variable not named in `cause`, and both can
be blocks.

### There are two different tests, and you should pick deliberately

```@example var
Y = tsmatrix(dataset("econ5"))
sys  = granger_test(fit_var(Y, 2); cause = 2)          # the system test
pair = granger_test(Y[:, 1], Y[:, 2]; lags = 2)        # the pairwise test
println("system   F = ", round(statistic(sys), digits=6), "  df2 = ", sys.df2)
println("pairwise F = ", round(statistic(pair), digits=6), "  df2 = ", pair.df2)
```

- **`granger_test(m::VARModel; cause, caused)`** is the VAR-based Wald/F
  test — `vars::causality` and `statsmodels`' `test_causality`. Its
  `df2` is the *system* residual degrees of freedom, `K·T − K·(Kp+ndet)`.
- **`granger_test(y, x; lags)`** is the single-equation F test —
  `lmtest::grangertest` and `statsmodels`' `grangercausalitytests`.
  Its `df2` is one equation's, `T − 2·lags − 1`.

The F **statistics** coincide only when the caused set has one member,
which for the default `caused` means `K = 2`. **The p-values never
coincide**, because `df2` differs by a factor of `K`. On the bivariate
fixture at `p = 2` the same `F` of `0.730864` gives `0.4821580341` from
the system test and `0.4828206979` from the pairwise one.

### Instantaneous causality

Granger causality is about the lags. Contemporaneous correlation in the
residuals is a separate question:

```@example var
instantaneous_causality(m2; cause = :gnp)
```

This is **not** a directional statement — contemporaneous correlation is
symmetric, so swapping the two sets returns the identical statistic.

## Two places where the references are wrong

Both are worth knowing because a test written against the reference
would fail where this package is right.

### `vars` reports p = 0 for any moderately strong rejection

`vars::causality` computes its p-values as `1 - pf(...)` and
`1 - pchisq(...)` rather than taking the upper tail, so they underflow to
**exactly zero** in ordinary use. On `climhyd` an `F` of `13.36` on
`(10, 2634)` degrees of freedom:

```@example var
C = fit_var(dataset("climhyd"), 2)
g = granger_test(C; cause = :Temp)
println("F = ", round(statistic(g), digits=6), "   df = (", g.df1, ", ", g.df2, ")")
println("this package : p = ", round(pvalue(g), sigdigits=6))
println("1 - CDF form : p = ", 1 - TSAnalytics._f_cdf(statistic(g), g.df1, g.df2))
```

The second line is what `vars` prints. `statsmodels` calls a survival
function and gets it right, so this is a defect rather than a
convention — and it is why this package's p-values are not comparable
with `vars` in the tail.

### `statsmodels` cannot test instantaneous causality between two blocks

Its `test_inst_causality` raises `IndexError` whenever **both** the
causing and the caused set have more than one member, because it indexes
a mask with two fancy index arrays that only broadcast when one has
length 1. Verified failing at (2,4), (3,3) and (4,2) on `K = 6`, and
working at (1, K−1) and (K−1, 1). It always errors rather than returning
a wrong answer.

So for a general block `vars` is the only reference, and this package
agrees with it there:

```@example var
instantaneous_causality(C; cause = [:Temp, :DewPt, :CldCvr])
```

## What is not here yet

- **Impulse responses and FEVD** (10.3), **residual diagnostics** (10.8),
  **cointegration** (10.4, 10.5), **VECM** (10.6) and **structural VARs**
  (10.7). The companion matrix and the moving-average representation this
  chapter's forecast already uses are what those are built on, and
  [`companion`](@ref) is exported for that reason.
- **Exogenous regressors**, restricted VARs and Bayesian priors. The
  regressor construction would carry exogenous columns untouched, so it
  is a clean later addition rather than a rewrite.
- `K = 1`. Refused by name, as both references refuse it; a single series
  is better served by [`arx`](@ref) or [`fit_arima`](@ref).

## See also

- [Forecasting and Accuracy](09-forecasting-and-accuracy.md) — evaluating what comes out of `forecast`
- [Coming from R or Python](13-coming-from-r-python.md) — every documented divergence in one place
- [API: Multivariate](../api/multivariate.md)
