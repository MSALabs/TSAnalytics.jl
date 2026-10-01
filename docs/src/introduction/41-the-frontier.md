# The Frontier

Forty chapters in, it is worth being precise about where this package
stops. Not because the gaps are embarrassing — every library has them —
but because knowing which wall you are standing at tells you which
direction to walk.

This chapter is a map of what is missing, why, and what you would use
instead today.

## The shape of what exists

Everything in this book reduces to a small number of things:

- a **Kalman filter** on a Gaussian state-space model, which ARMA,
  ARIMA, SARIMA, ARIMAX and drifting-coefficient regression all
  compile down to;
- a **conditional-variance recursion**, which GARCH, GJR and EGARCH
  share;
- a set of **decomposition** procedures — classical, STL, MSTL — that
  do not involve a likelihood at all;
- and **diagnostics**, which are independent of all of the above and
  will happily test residuals you produced some other way.

That shape is the reason the gaps are where they are. Anything that
fits the state-space form is comparatively cheap to add; anything that
does not needs new machinery.

## One series at a time

The largest single gap is that **everything here is univariate**. There
is no VAR, no cointegration testing, no vector error-correction model,
no impulse response function.

This is not an oversight of emphasis. A VAR is a different object: the
state is a vector, the coefficients are matrices, and lag-order
selection has to trade off `k²p` parameters rather than `p`. Granger
causality, impulse responses and forecast error variance decomposition
all follow from it, and cointegration — Engle-Granger, then Johansen —
is where the interesting economics lives, because it is the machinery
for saying two non-stationary series move together.

Until it exists, R's `vars` and `urca` remain the reference, with
`statsmodels`' `VAR` and `VECM` on the Python side. The
[`adf_test`](@ref)/[`kpss_test`](@ref) pair here will get you as far as
establishing that each series individually has a unit root, which is
the precondition for asking the cointegration question at all — and
then you will need to leave.

## Exponential smoothing, properly

[`holt_winters`](@ref) covers the classical recursions. What it does
not do is the **full ETS taxonomy** — all thirty error/trend/season
combinations, each with its own state-space representation, selected
automatically by AICc. That is what R's `ets()` gives you, and it
matters because the automatic selection is genuinely good: ETS
routinely wins forecasting competitions against models chosen by hand.

The infrastructure is largely in place. ETS in state-space form runs on
the same Kalman filter Part VI describes, and the local-level and
local-trend components are non-stationary by construction, which is
precisely what [diffuse initialisation](33-diffuse-initialisation.md)
exists to handle. It is a matter of writing the thirty forms and the
selection loop, not of new theory.

Damped-trend variants, the Theta method and TBATS sit behind it in the
same queue.

## Components you can name

**Unobserved components models** — a local level, a local trend, a
stochastic seasonal, a cycle, each with its own variance, fitted
jointly — are the other obvious state-space application that is not yet
built. `fit_arimax(...; model=:tvss)` in
[Chapter 35](35-coefficients-that-drift.md) is a special case of one,
and the diffuse filter it uses is the hard part.

The appeal of a UC model over an ARIMA is interpretability: you get a
trend you can plot and a seasonal you can plot, with standard errors,
rather than a set of coefficients that jointly imply them. For anyone
who has to explain a model to somebody who does not want to hear about
polynomials in the lag operator, that is the whole argument.

## Regimes

Nothing here allows the model itself to change. **Markov-switching**
models let the parameters jump between a small number of unobserved
states, with the transition probabilities estimated; **threshold**
models (TAR, SETAR, STAR) let them change as a function of an observed
variable crossing a boundary.

[`nyblom_test`](@ref) will tell you the parameters did not stay
constant. It will not tell you what to do about it, and the honest
answer today is: split the sample, or go elsewhere. R's `MSwM` and
`tsDyn` are the references; Python's coverage is thinner, which makes
this one of the places a Julia implementation would not merely be
catching up.

## Distributions other than the normal

`fit_garch` accepts `dist=:t` in its signature and throws. Financial
returns are famously fat-tailed, and a Student-*t* GARCH is the
standard response — it needs an extra estimated degrees-of-freedom
parameter and its own likelihood, which is real new scope rather than a
keyword.

The `cov_type=:robust` default is the partial answer already in place:
the Bollerslev-Wooldridge sandwich gives you standard errors that
survive the normality assumption being wrong, even though the
likelihood itself still assumes it. That is QMLE, and it is a
legitimate way to work — but it does not fix the *intervals*, which
still come from a normal.

The same applies more broadly: every interval in this package is
Gaussian, and [`jarque_bera_test`](@ref) is in the checklist precisely
because you should know when that assumption is doing work it cannot
support.

## Smaller, specific gaps

| Missing | Consequence | Reference |
|---|---|---|
| HEGY and Canova-Hansen seasonal unit-root tests | [`nsdiffs`](@ref) offers the seasonal-strength heuristic and OCSB, which is what R and `pmdarima` actually default to; `uroot`-style HEGY/CH are not built | R's `uroot` package |
| `forecast` for `model=:tvss` | `model=:mle` forecasts with future regressors; the drifting-coefficient variant needs a projected `beta` path | R has no direct analogue |
| Intermittent-demand methods (Croston, SBA, TSB) | Series that are mostly zeros are not served | `forecast::croston`, `statsforecast` |
| Change-point detection | A break has to be found by eye | `changepoint` (R), `ruptures` (Python) |
| Missing-data policy beyond the state-space path | `NaN` handling is per-function rather than uniform | — |

These are genuine gaps but narrow ones: R reaches HEGY and
Canova-Hansen only through the separate `uroot` package, and neither is
its default.

## The neighbouring package

Official seasonal adjustment — X-11's lineage, X-13ARIMA-SEATS, the
diagnostics statistical agencies publish — is deliberately **not** here.
It is a different package,
[SeasonalAdjustment.jl](38-official-seasonal-adjustment.md), for the
reasons Part VIII sets out: it is a different kind of activity, with
institutional requirements that have no analogue in exploratory
modelling.

The boundary is clean. [`stl_decompose`](@ref) is how *you* decompose a
series. X-13 is how a statistical office publishes one.

## What will not change

Some things are not gaps, and will not be filled:

**No container integration.** No dependency on TSFrames, TimeSeries or
DataFrames, ever. Every function here accepts anything the
`tsvalues`/`tsindex` interface can be called on, which by default means
any iterable. Adding a container dependency would buy convenience for
one community at the cost of everyone else's.

**No ported code.** Every algorithm here is implemented from its
primary source. R and Python are used to resolve ambiguity when a paper
underspecifies something, and to validate output numbers — never to
translate from. That is slower, and it is the reason the divergences
documented throughout this book are *decisions* rather than accidents.

**No silent fallbacks.** `dist=:t` errors rather than quietly using a
normal. `pacf(method=:mle)` errors rather than quietly using
Yule-Walker — which is what R does. `forecast_volatility(egarch_model, 5; method=:analytic)` errors
rather than quietly simulating. A wrong number you trust is worse than
an error you can read.

## Where to start

If you want to contribute, the ordering that makes sense is roughly the
order of this chapter: full ETS is large but needs no new theory, and
VAR is the biggest single addition, opening the whole multivariate
track behind it.

`development-sequence.md` in the repository carries the full staged
roadmap with dependencies — what is built, what is next, and what each
piece needs before it can start.

## See also

- [Appendix B](B-verification.md) — the standard every number here was held to
- [Appendix C](C-further-reading.md) — where to read further on everything above
- [Coming from R or Python](../manual/11-coming-from-r-python.md) — what to reach for in the meantime
