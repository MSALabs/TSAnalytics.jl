# Primitives

The operations everything else is built from: differencing, filtering,
autocorrelation, the frequency domain, and variance-stabilising
transformations.

Every function here accepts anything [`tsvalues`](@ref) can be called
on — a `Vector`, a `TSFrame` column, a `DataFrame` column. Examples use
plain vectors.

```@example primitives
using TSAnalytics, Plots

y = dataset("cardox").value[1:240]   # monthly CO₂, 20 years
nothing # hide
```

## Difference a series, and undo it

```@example primitives
dy = diff(y)
println("original: ", length(y), "   differenced: ", length(dy))
```

[`diff`](@ref) takes `lag` and `differences` keywords for seasonal and
repeated differencing:

```@example primitives
d_seasonal = diff(y, 12)              # seasonal difference, lag 12
d_both = diff(diff(y, 12), 1)         # one seasonal, then one ordinary
println("seasonal: ", length(d_seasonal), "   both: ", length(d_both))
```

[`diffinv`](@ref) inverts it, given the starting values the difference
threw away:

```@example primitives
back = diffinv(dy; xi=[y[1]])
println("recovered the original exactly: ", isapprox(back, y; atol=1e-9))
```

`xi` is the initial condition. Without it `diffinv` starts from zero
and you get the right *shape* at the wrong level — a common and silent
mistake when reversing a differenced forecast.

## Apply a moving average

```@example primitives
ma = moving_average(y, 12)
println("length: ", length(ma), "   first finite value at index: ", findfirst(!isnan, ma))
plot(y; label="observed", legend=:topleft)
plot!(ma; label="12-month centred MA", linewidth=2)
```

The result is the same length as the input, with `NaN` at both ends
where the window does not fit. A centred even-order average is
automatically double-averaged so it stays aligned — the usual
`2×12-MA` convention for monthly data.

For asymmetric or recursive filtering, use the two lower-level
functions directly:

| Function | Computes | Use for |
|---|---|---|
| [`convolution_filter`](@ref) | `sum(coef[j] * x[t-j])` | Moving averages, smoothing, any FIR filter |
| [`recursive_filter`](@ref) | `x[t] + sum(coef[j] * y[t-j])` | Exponential smoothing, AR-style recursions |

Both keep the argument order and the `sides` spelling R's `stats::filter`
uses, so a filter specification transfers across without rearrangement.

## Autocorrelation

```@example primitives
a = acf(y, 1:24)
plot(a; title="ACF of CO₂")
```

```@example primitives
println("first five: ", round.(acf(y, 1:5).values, digits=4))
```

The plot draws the confidence band automatically. By default it is the
constant `±1.96/√n` band; pass `bartlett=true` for Bartlett's widening
band, which is the more honest one when you are reading a *decay*
rather than testing a single lag.

## Partial autocorrelation, and its three methods

```@example primitives
println("yw  : ", round.(pacf(y, 1:5).values, digits=4))
println("ols : ", round.(pacf(y, 1:5; method=:ols).values, digits=4))
```

Three methods are available and they do **not** agree on a strongly
trending series like this one:

| `method` | What it is | Matches |
|---|---|---|
| `:yw` (default) | Yule-Walker, `n-k` denominator | `statsmodels`' `pacf_yw` |
| `:ywm` | Yule-Walker, `n` denominator | **R's `pacf()`** |
| `:ols` | Successive regression | `statsmodels`' `pacf_ols` |
| `:burg` | Burg's maximum-entropy recursion | `statsmodels`' `pacf_burg` |

The spread above is not a bug. On a near-unit-root series the
Yule-Walker estimator is heavily biased toward zero, and `:ols` is
generally the better choice when you intend to read an order off the
result.

`:burg` is the odd one out, and worth a note on why.

!!! note "R's `pacf()` ignores its own `method` argument"
    Verified by direct execution: R's `"yule-walker"`, `"burg"`,
    `"ols"` and `"mle"` all return **bit-identical** output, equal to
    this package's `:ywm`. Burg in R lives in `ar.burg()`, which
    estimates AR coefficients rather than partial autocorrelations.

    So R is not a reference for this option, and `:burg` is validated
    against `statsmodels` alone — the one place in the package where a
    `pacf` method has a single reference rather than two.

Burg is a genuinely different estimator, not another denominator on the
same one: it minimises the forward **and** backward prediction error
jointly, which makes it better behaved near a unit root. On a strongly
trending series it will sit between `:yw` and `:ols`.

## Two series at a time

```@example primitives
rec = dataset("rec").value
soi = dataset("soi").value
c = ccf(soi, rec, 16)
println("peak at lag ", c.lags[argmax(abs.(c.values))],
        "   value ", round(c.values[argmax(abs.(c.values))], digits=4))
plot(c; title="SOI against Recruitment")
```

[`ccf`](@ref) is the cross-correlation function: everything above asks
what a series says about its own past, and this asks what one series
says about another's.

**Lag `k` estimates the correlation between `x[t+k]` and `y[t]`**, which
is R's convention and the one all six reference books use. A peak at a
*negative* lag says the first series **leads** the second.

Here the strongest relationship is at lag `-6` and it is `-0.60`: the
Southern Oscillation Index leads fish recruitment by about six months,
and the sign is negative, so warmer water now means fewer fish half a
year later. Note that finding it needs `argmax(abs.(...))` rather than
`argmax` — a lead relationship can perfectly well be an inverse one,
and this is the kind of structure an ACF cannot see at all.

!!! warning "Python returns only non-negative lags"
    `statsmodels`' `ccf(x, y)` returns lags `0:n-1` and nothing below
    zero, so on a pair where `x` leads `y` **it does not contain the
    peak at all** — you have to call `ccf(y, x)` to find it. The values
    agree exactly where they overlap: R's lag `-k` is `statsmodels`'
    `ccf(y, x)` at `+k`.

    Reaching for `argmax` after porting from Python will give a
    different answer here. That is the intended difference.

`ccf(x, x, k)` reproduces `acf(x, 0:k)` exactly on its non-negative
half — same `1/n` denominator — which is the cheapest check that you
have the normalisation you expect.

## Find a period you did not already know

```@example primitives
pg = periodogram(y)
i = argmax(pg.spec)
println("peak frequency: ", round(pg.freq[i], digits=5), "  →  period ", round(1/pg.freq[i], digits=2))
plot(pg; title="periodogram")
```

The peak lands at frequency `0.08333`, which is `1/12` — the annual
cycle, recovered without anyone having told the function the data was
monthly. This is what the frequency domain is *for*: the ACF can only
confirm a period you already suspected.

A raw periodogram is a noisy estimator no matter how long the series
is. [`spectral_density`](@ref) smooths it with a Daniell kernel:

```@example primitives
sd = spectral_density(y, [5])
println("bandwidth: ", round(sd.bandwidth, digits=5), "   equivalent df: ", round(sd.df, digits=2))
```

The span vector controls the smoothing; more spans means a smoother
estimate with more degrees of freedom and less resolution. Base R's
`spec.pgram` tapers by default (`taper=0.1`) and this does not — see
[Chapter 6](../introduction/06-the-frequency-domain.md) for why that
matters when comparing output.

## Stabilise a variance

```@example primitives
lambda = guerrero_lambda(y, 12)
println("Guerrero lambda: ", round(lambda, digits=4))
```

[`guerrero_lambda`](@ref) picks the Box-Cox parameter that best
stabilises variance across seasonal blocks. Near `0` means take logs;
near `1` means the series is fine as it is.

[`boxcox_lambda`](@ref) is the other one, and they answer different
questions:

```@example primitives
println("guerrero (variance stabilisation): ", round(guerrero_lambda(y, 12), digits=4))
println("MLE (normality)                  : ", round(boxcox_lambda(y), digits=4))
```

| | Minimises | Needs a period | Matches |
|---|---|---|---|
| [`guerrero_lambda`](@ref) | Coefficient of variation across seasonal blocks | Yes | R's `BoxCox.lambda` default |
| [`boxcox_lambda`](@ref) | Negative profile log-likelihood | No | `scipy.stats.boxcox`, once unbounded |

They disagree sharply here, and the reason is worth knowing: this
series sits at `315`–`340`, a range of under 8 % of its own level. Over
such a narrow window every power transform is nearly a linear rescaling
of every other, the profile likelihood is almost flat, and the MLE
drifts to whichever bound it was given — the `-1.0` above is the bound,
not a finding. Guerrero's `-0.003` is the one to act on, because
stabilising variance across seasonal blocks is a question this data can
actually answer.

**Prefer `guerrero_lambda` on seasonal data**, and read a `boxcox_lambda`
result that lands exactly on a bound as "the data does not identify
this" rather than as an estimate.

Both default to searching `[-1, 2]`, matching R. **`scipy.stats.boxcox`
is unbounded**, so where the unconstrained optimum falls outside that
interval the two disagree and `boxcox_lambda` returns the boundary —
widen `bounds` to reproduce scipy. A lambda of `-1.8` is a
reciprocal-squared transform few people intend, which is why both R and
this package bound it.

Apply it with [`boxcox`](@ref), and reverse with [`boxcox_inv`](@ref):

```@example primitives
yt = boxcox(y, lambda)
println("round-trips: ", isapprox(boxcox_inv(yt, lambda), y; atol=1e-8))
```

Reversing a *forecast* needs more care than reversing the data — the
naive back-transform gives a median rather than a mean. See
[Chapter 7](../introduction/07-transformations.md) for the bias
correction.

## See also

- [Diagnostics](02-diagnostics.md) — testing what you find here
- [Plotting](10-plotting.md) — every result type's recipe
- [API: Primitives](../api/primitives.md) — full argument lists
