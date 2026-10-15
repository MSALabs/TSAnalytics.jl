# The Frontier

Sixty chapters in, it is worth being precise about where this
package stops. Not because the gaps are embarrassing — every library has
them — but because knowing which wall you are standing at tells you
which direction to walk.

This chapter is a map of what is missing, why, and what you would use
instead today.

## The shape of what exists

Everything in this book reduces to a small number of things:

- a **Kalman filter** on a Gaussian state-space model, which ARMA,
  ARIMA, SARIMA, ARIMAX and drifting-coefficient regression all
  compile down to;
- an **innovations recursion** for exponential smoothing, in which the
  initial states are estimated as parameters, so that ETS, Theta and
  TBATS need no diffuse initialisation at all;
- a **conditional-variance recursion**, which GARCH, GJR and EGARCH
  share;
- **multivariate least squares and reduced-rank regression**, on which
  the VAR, the Johansen test and the VECM all stand;
- a **sequential multivariate Kalman filter**, which unobserved
  components and dynamic factor models compile down to — the same
  state-space engine, now taking a matrix of series;
- a **filter over a discrete hidden state** — Hamilton's — for
  Markov-switching models, and a **least-squares search over a
  threshold** for the threshold autoregressions;
- a **dynamic programme over segmentations**, which places change points
  (PELT) and dates structural breaks (Bai–Perron);
- **regression on a table of lagged features**, the reduction that
  autoregression, the direct strategy and any plugged-in learner share;
- a set of **decomposition** procedures — classical, STL, MSTL — that
  do not involve a likelihood at all;
- and **diagnostics**, which are independent of all of the above and
  will happily test residuals you produced some other way.

That shape is the reason the gaps are where they are. Anything that
fits one of those forms is comparatively cheap to add; anything that
does not needs new machinery.

## Series that move together: what is still missing

This section used to say that everything here was univariate. It no
longer is. Part X builds the multivariate track from the VAR
([Chapter 45](45-two-series-at-once.md)) through Granger causality,
impulse responses, the three cointegration tests, the VECM and
structural identification ([Chapter 51](51-shocks-with-names.md)), each
number checked against R's `vars` and `urca` and against `statsmodels`.

What that track does not yet have:

| Missing | Why it matters | Reference |
|---|---|---|
| Error bands for structural impulse responses | `irf` on an [`SVARModel`](@ref) gives point estimates only; the analytic bands are for the Cholesky factor | `vars::irf(…, boot=TRUE)` on an `svarest` |
| Sims-Zha error bands | The bootstrap percentile band is the only interval that is not asymptotic | `statsmodels`' `err_band_sz1`–`sz3`; `vars` has none |
| Generalised (Pesaran-Shin) impulse responses | Order-invariant responses, the usual answer to the ordering problem of [Chapter 48](48-tracing-a-shock.md) | neither `vars` nor `statsmodels` has them |
| Likelihood-ratio tests on restricted `β` and `α` | Testing a theory-given cointegrating vector, or weak exogeneity | `urca::blrtest`, `urca::alrtest` |
| Gregory-Hansen | Cointegration with a structural break in the relation | Gregory & Hansen (1996); in neither `urca` nor `tsDyn` |
| Threshold VECM | An equilibrium that is restored only outside a band | `tsDyn::TVECM` |
| Sign-restricted SVAR | Identification by the signs of the responses rather than zeros | neither `vars` nor `statsmodels` ships one |
| Toda-Yamamoto and frequency-domain causality | Causality tests that do not require pre-testing for unit roots | — |
| Mixed-frequency and EM-estimated factor models | [`fit_dfm`](@ref) maximises the likelihood directly, on series of one frequency | `statsmodels DynamicFactorMQ`, `dfms` |
| Factor-augmented VARs | A VAR in a few observed series plus the factors of many | — |

None of these is a small addition, and each of them is a place where
R and Python disagree with each other often enough that a careful
implementation would earn its keep.

## Exponential smoothing: the other fifteen

Part V teaches the fifteen ETS forms that exist — R's whole default
search space — together with Theta and TBATS. The fifteen that remain
are the multiplicative-**trend** forms, which R also excludes by default
because they are unstable, and the ETS(A,·,M) trio, which R refuses
outright because the level update divides an additive error by a
seasonal factor. A reader who wants a multiplicative trend should be
told plainly that no mainstream package offers one by default either.

TBATS is built without the automatic search R's `tbats()` wraps around
it and without ARMA errors; both are additions on top of what exists,
not changes to it.

## Components you can name: what is still missing

This section used to say that **unobserved components** models were the
obvious state-space application not yet built. They are built now —
[Chapter 52](52-the-components-you-cannot-see.md) and
[Chapter 53](53-when-a-component-stops-moving.md): a level, a slope, a
seasonal, a cycle and an autoregressive part, each with its own variance,
fitted jointly, every component returned with a standard error — and so
are **dynamic factor models**
([Chapter 54](54-many-series-few-forces.md)), the same idea across many
series at once.

What they do not yet have:

- **Drifting regression coefficients inside a UC model.** Regressors
  enter [`fit_uc`](@ref) with fixed coefficients; a coefficient that is
  itself a random walk is still `fit_arimax(…; model=:tvss)`, in
  [Chapter 39](39-coefficients-that-drift.md).
- **Exact diffuse initialisation for UC.** The non-stationary components
  start from a large-variance prior, as `statsmodels` does, and the
  periods it consumes are left out of the likelihood and reported
  without a standard error rather than with a meaningless one.
- **Forecast intervals that account for estimated variances** — the
  same limitation every reference shares.

## Regimes: what is still missing

The model can change now. **Markov-switching** regressions and
autoregressions ([Chapter 55](55-the-economy-has-two-moods.md)) let the
parameters jump between
unobserved states, with fixed or covariate-driven transition
probabilities; **threshold** autoregressions
([Chapter 56](56-a-different-model-on-each-side-of-a-line.md)) let them
change when an observed lag crosses a
boundary, abruptly or smoothly, and [`setar_test`](@ref) asks whether a
threshold is there at all. Each is checked against R's `MSwM` or `tsDyn`
and `statsmodels`, and in several places goes further than both — a
three-regime phase plot, a threshold profile for any criterion, a count
of how many optima the restarts found.

What remains:

| Missing | Why it matters | Reference |
|---|---|---|
| A confidence set for the threshold | The threshold's distribution is non-standard; Hansen's likelihood-ratio inversion is the usual answer, and no standard error is reported in its place | Hansen (2000); neither `tsDyn` nor `statsmodels` |
| Automatic order search for LSTAR, and more than two smooth regimes | [`fit_lstar`](@ref) fits the order it is given | `tsDyn::selectLSTAR`, `tsDyn::star` |
| Momentum TAR (a differenced threshold variable) | Asymmetric adjustment in the *direction* of the change | `tsDyn::setar(…, model="MTAR")` |
| Markov-switching VARs | Several series switching together | — |

## Change points and breaks: what is still missing

Changes that do not come back are built now:
[Chapter 57](57-a-change-that-does-not-come-back.md) detects and dates
them (PELT, binary segmentation, Bai–Perron with confidence intervals for
the dates, the F and fluctuation tests), and
[Chapter 58](58-watching-a-model-drift.md) watches for them as they
happen. Each number is checked against R's `changepoint` and
`strucchange` and Python's `ruptures`. What remains is mostly a matter of
tables and licences, not mathematics:

| Missing | Why it matters | Reference |
|---|---|---|
| p-values for supF, aveF and expF | The statistics are computed; their null distributions are non-standard. Hansen's (1997) approximation is GPL-licensed, and Andrews' (1993) tables are in his paper. None of the six textbooks prints them | `strucchange::sctest` |
| MOSUM fluctuation tests | Their p-values also need tables; the CUSUM family has closed-form limits and is built | `strucchange::efp(type = "OLS-MOSUM")` |
| The full penalty path (CROPS) as a recipe | [`penalty_path`](@ref) returns every segmentation; only the elbow is drawn, since no single rendering of the rest is natural | `changepoint::cpt.range` |
| Multivariate and non-parametric change points | Changes in a covariance matrix, or with no distributional cost | `changepoint.np`, `ecp` |
| Monitoring charts as functions | The CUSUM and EWMA of Chapter 58 are a few lines of code there, not a function with an average-run-length calculator | `qcc`, `spc` (R) |

## Intermittent demand and lag regression: what is still missing

Mostly-zero series ([Chapter 59](59-mostly-nothing-then-something.md))
have Croston, SBA, TSB, ADIDA and IMAPA, each forecast agreeing with
`statsforecast` to the last digit. The reduction of forecasting to
regression ([Chapter 60](60-any-regressor-you-like.md)) accepts any
regressor and adds the direct strategy. What was deliberately not built:

| Not built | Why | Reference |
|---|---|---|
| Prediction intervals for intermittent methods | No reference implementation computes any, and the methods have no consistent underlying model to derive them from | — |
| DirRec and multi-output strategies | Identical to the direct strategy under least squares, so they add nothing until a non-linear regressor is plugged in | `sktime` |
| Conformal prediction intervals | A distribution-free interval for any plugged-in regressor; it needs a calibration design of its own | `mlforecast`'s `PredictionIntervals` |
| Feature importances and hyperparameter search | Least squares has no importances, and search belongs to the learner's own library | — |

## Distributions other than the normal

GARCH is no longer Gaussian-only: `fit_garch(...; dist=:t)` estimates a
Student-*t* shape alongside the variance recursion, and it agrees with
Python's `arch` and R's `rugarch`. What remains Gaussian is everything
else. Every **forecast interval** in this package — ARIMA, ETS, VAR —
is a normal interval around the point forecast, and
[`jarque_bera_test`](@ref) and [`normality_test`](@ref) are in the
checklist precisely because you should know when that assumption is
doing work it cannot support.

## Smaller, specific gaps

| Missing | Consequence | Reference |
|---|---|---|
| HEGY and Canova-Hansen seasonal unit-root tests | [`nsdiffs`](@ref) offers the seasonal-strength heuristic and OCSB, which is what R and `pmdarima` actually default to; `uroot`-style HEGY/CH are not built | R's `uroot` package |
| `forecast` for `model=:tvss` | `model=:mle` forecasts with future regressors; the drifting-coefficient variant needs a projected `beta` path | R has no direct analogue |
| Missing-data policy beyond the state-space path | `NaN` handling is per-function rather than uniform | — |

These are genuine gaps but narrow ones: R reaches HEGY and
Canova-Hansen only through the separate `uroot` package, and neither is
its default.

## The neighbouring package

Official seasonal adjustment — X-11's lineage, X-13ARIMA-SEATS, the
diagnostics statistical agencies publish — is deliberately **not** here.
It is a different package,
[SeasonalAdjustment.jl](42-official-seasonal-adjustment.md), for the
reasons Part IX sets out: it is a different kind of activity, with
institutional requirements that have no analogue in exploratory
modelling.

The boundary is clean. [`stl_decompose`](@ref) is how *you* decompose a
series. X-13 is how a statistical office publishes one.

## What will not change

Some things are not gaps, and will not be filled:

**No container integration.** No dependency on TSFrames, TimeSeries or
DataFrames, ever. Every function here accepts anything the
`tsvalues`/`tsindex` interface can be called on, which by default means
any iterable — and, for the multivariate functions, anything
[`tsmatrix`](@ref) can turn into a matrix, including a `NamedTuple` of
columns. Adding a container dependency would buy convenience for one
community at the cost of everyone else's.

**No ported code.** Every algorithm here is implemented from its
primary source. R and Python are used to resolve ambiguity when a paper
underspecifies something, and to validate output numbers — never to
translate from. That is slower, and it is the reason the divergences
documented throughout this book are *decisions* rather than accidents.

**No silent fallbacks.** `pacf(method=:mle)` errors rather than quietly
using Yule-Walker — which is what R does.
`forecast_volatility(egarch_model, 5; method=:analytic)` errors rather
than quietly simulating. `pvalue` on a Phillips-Ouliaris test errors,
because the paper tabulates critical values and no p-values, rather
than inventing one by interpolation. A wrong number you trust is worse
than an error you can read.

## Where to start

If you want to contribute, the ordering that makes sense is roughly the
order of this chapter. Bands for structural impulse responses and the
restricted-`β` tests complete the multivariate track that Part X
opened; a confidence set for the threshold completes the regime models,
and p-values for the supremum break tests complete Part XI's break
dating, once a licence-compatible source for their distribution exists.

`development-sequence.md` in the repository carries the full staged
roadmap with dependencies — what is built, what is next, and what each
piece needs before it can start.

## See also

- [Appendix B](B-verification.md) — the standard every number here was held to
- [Appendix C](C-further-reading.md) — where to read further on everything above
- [Coming from R or Python](../manual/16-coming-from-r-python.md) — what to reach for in the meantime
