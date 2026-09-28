# Beyond the Defaults

Four settings account for most of what anyone changes first. This page
is about those, and about the one limitation you need to know before
you hit it.

## When the automatic order is wrong

[`auto_arima`](@ref) searches; it does not divine. When you already
know the order — from theory, from a previous fit, or from a
correlogram — say so directly:

```jldoctest beyond
julia> using TSAnalytics

julia> y = dataset("cardox").value[1:240];

julia> m = fit_sarima(y, (1,1,1), (0,1,1,12));

julia> round(m.aic, digits=2)
127.03
```

Reasons to override the search are concrete rather than stylistic:

- **You are reproducing a published result** and need that exact order.
- **The search is slow** on a long series and you are refitting in a
  loop — a known order skips the search entirely.
- **Theory constrains the model.** A series you know to be a random
  walk plus noise has a specification; discovering it by AICc is
  slower and less reliable than asserting it.
- **The search picked something implausible.** On short samples it
  often does — Chapter 21 of the Introduction measures how often, and
  the answer is sobering.

## Seasonal data, and the one gap

```jldoctest beyond
julia> m_auto = auto_arima(y; seasonal=true, m=12, D=1);

julia> m_auto.order, m_auto.seasonal_order
((1, 1, 1), (0, 1, 1, 12))
```

`m=12` gives the period. `D=1` takes one seasonal difference — and
**you must supply `D` yourself.**

!!! warning "There is no seasonal unit-root test yet"
    `auto_arima` chooses the ordinary differencing order `d` for you,
    by running a KPSS test repeatedly. It cannot choose `D` the same
    way, because the seasonal equivalent is not implemented. R's
    `auto.arima` uses the Canova-Hansen test for this and Python's
    `pmdarima` uses OCSB; this package has neither yet.

    Left alone, `D` defaults to `0` — so a seasonal series fitted
    without an explicit `D` will quietly get no seasonal differencing
    at all, and the result will look worse than it should for a reason
    nothing announces. **On monthly or quarterly data with a visible
    annual cycle, pass `D=1`.**

    This is a real gap, stated plainly rather than left to be
    discovered. It is on the roadmap.

If you are unsure whether `D=1` is warranted, the cheap check is the
one from the previous page: fit both, and run `qs_test` on each set of
residuals. Leftover seasonality shows up there.

## Transformations

A series whose swings grow with its level wants a transformation before
modelling, not a bigger model:

```jldoctest beyond
julia> lambda = guerrero_lambda(y, 12);

julia> round(lambda, digits=3)
-0.003
```

[`guerrero_lambda`](@ref) estimates the Box-Cox parameter that best
stabilises the variance. A `lambda` near `0` means **take logs**; near
`1` means leave it alone. At `-0.003` this series is telling you a log
transform is as close to optimal as makes no difference.

Whether to act on that is a judgement call. Logging changes what the
model is about — additive effects on a logged series are multiplicative
on the original — and it complicates forecasting, because the naive
back-transform of a forecast mean is not the mean of the
back-transformed forecast. Chapter 7 of the Introduction works through
the bias correction.

## Which information criterion

```jldoctest beyond
julia> m_bic = auto_arima(y; seasonal=true, m=12, D=1, information_criterion=:bic);

julia> m_bic.order, m_bic.seasonal_order
((0, 1, 1), (0, 1, 1, 12))
```

**A different criterion selected a different model on identical data.**
The default `:aicc` kept an AR term that `:bic` dropped, and what `:bic`
landed on — ARIMA(0,1,1)(0,1,1)[12] — is the airline model exactly.

That AR term is worth about `0.6` of AIC. AICc considers that a fair
trade for one parameter; BIC, whose penalty grows with sample size,
does not. Neither is wrong.

| Criterion | Penalty | Use when |
|---|---|---|
| `:aicc` (default) | `2k` plus a small-sample correction | Default. Forecasting, and any sample where `n/k` is not large |
| `:aic` | `2k` | Comparing against Python — `pmdarima` defaults here |
| `:bic` | `k·log(n)` | You want the smallest defensible model, or believe a true finite-order model exists |

`:aicc` matches R's `auto.arima` default. `:aic` matches `pmdarima`'s.
**This is a live source of cross-language confusion**: the same series
through both packages can return different orders purely because of
this default, with no disagreement about any underlying computation.

The practical advice is duller than the theory: on a forecasting task,
the criterion rarely matters as much as the debate suggests, and
[Chapter 23](../introduction/23-evaluating-honestly.md) argues you
should be comparing out-of-sample anyway. Where it *does* matter is
when you intend to interpret the order as a fact about the process —
which, at typical sample sizes, is a claim the data usually cannot
support.

## Standard errors

One more, since it changes numbers rather than models:

```jldoctest beyond
julia> m_rob = fit_sarima(y, (1,1,1), (0,1,1,12); se_type=:robust);

julia> length(m_rob.se) == 3
true
```

`se_type` accepts `:hessian` (default, matching R), `:opg` (matching
`statsmodels`' default), and `:robust` (a Huber-White sandwich). The
coefficients are identical across all three; only their standard errors
differ, sometimes enough to change a significance verdict.
[Chapter 18](../introduction/18-fitting-arma.md) compares seven
estimates of the same standard error across three languages.

## Next

[Where Next](05-where-next.md) maps the rest of the documentation —
which section answers which kind of question.
