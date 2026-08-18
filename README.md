# TSAnalytics.jl

A native Julia time series analysis package: ARIMA/SARIMAX, seasonal
decomposition, exponential smoothing, and the diagnostic tests that go
with them -- built incrementally, in the spirit of GLM.jl (separation of
model/predictor concerns, `StatsAPI` conventions, `LinearAlgebra`-backed
numerics).

**Status: pre-release, active development.** This repository currently
contains the foundational descriptive-statistics, diagnostic-testing, and
decomposition layers, plus the optimizer/reparametrization infrastructure
every MLE-fit model will share. Model fitting itself (ARIMA/SARIMAX via a
shared state-space engine) is next. See [Roadmap](#roadmap) below.

## What's implemented so far

- `diff`, `diffinv` -- regular and seasonal differencing and its exact
  inverse, generalizing `Base.diff`'s single `lag=1, differences=1` case
  to arbitrary lag/order combinations (matching R's `diff`/`diffinv`);
  `tsdiff`/`tsundiff` are all-keyword aliases
- `convolution_filter`, `recursive_filter` -- linear (FIR/IIR) filtering
  primitives matching R's `stats::filter()` and
  `statsmodels.tsa.filters.filtertools`; `moving_average` is a
  convenience wrapper matching `forecast::ma()`, including the "2×m"
  centered double moving average X-11-style decomposition needs for even
  periods
- `acf`, `pacf` -- autocorrelation / partial autocorrelation, natively
  implemented (no `StatsBase` dependency), matching R's `stats::acf()`/
  `pacf()` and Python's `statsmodels.tsa.stattools.acf`/`pacf` feature
  sets: Bartlett-widening or simple confidence bands, adjusted/unadjusted
  denominators, per-lag Ljung-Box `qstat`, and three `pacf` methods
  (`:yw`/`:ywm`/`:ols`)
- `adf_test` -- Augmented Dickey-Fuller unit-root test, argument names
  matching `statsmodels.tsa.stattools.adfuller` exactly (`regression=
  :n`/`:c`/`:ct`/`:ctt`; `:aic`/`:bic`/`:tstat` automatic lag-order search
  or a fixed `maxlag`) -- validated against real `statsmodels` output
  across all 12 `regression`×`autolag` combinations, matching statistic,
  chosen lag, and sample size simultaneously
- `kpss_test` -- KPSS stationarity test, argument names matching
  `statsmodels.tsa.stattools.kpss` (`regression=:c`/`:ct`, consistent
  with `adf_test`'s naming too), with a Bartlett-kernel long-run variance
  estimate; `nlags=:short` (R's default) / `:legacy` (matches Python's
  `"legacy"` exactly, verified from source) / an explicit `Integer` --
  `:auto` (Python's actual default, the Hobijn et al. 1998 method) is a
  documented, explicit gap, not silently approximated
- `pp_test` -- Phillips-Perron unit-root test, argument names matching
  `arch.unitroot.PhillipsPerron` (`trend=:n`/`:c`/`:ct`, `test_type=
  :tau`/`:rho`); algorithmically distinct from `adf_test` (no augmenting
  lags in the regression at all -- serial correlation corrected via the
  same Newey-West/Bartlett adjustment `kpss_test` uses) -- verified to
  reproduce real `arch` output across all 6 `trend`×`test_type`
  combinations before being trusted
- `ljungbox_test` -- Ljung-Box portmanteau test (`boxpierce=true` for the
  Box-Pierce statistic too -- R's `Box.test()` actually defaults to the
  older, weaker Box-Pierce statistic; this deliberately doesn't
  default-match R there), general lag sets (so it covers the seasonal-lag
  variant too -- with a genuinely different meaning from a Python lag
  list, documented explicitly: exact-set sum here, per-lag cumulative
  rows in `statsmodels`), `fitdf` support for testing residuals of a
  fitted model
- `qs_test` -- the QS seasonal-residual test used by X-13ARIMA-SEATS /
  JDemetra+; a thin wrapper over `ljungbox_test`'s new `clip_negative`
  option, matching JDemetra+'s own `QSTest`-calls-`LjungBoxTest`
  architecture (verified against its published documentation) -- fixed a
  real correctness bug along the way: negative autocorrelations at the
  seasonal lags must be clipped to zero before squaring, per JDemetra+'s
  documented formula, which the earlier build omitted
- `jarque_bera_test` -- Jarque-Bera normality test; the one diagnostic
  test in this package where R and Python's formulas agree exactly (no
  default-behavior discrepancy to document) -- verified against real
  `statsmodels` output to full floating-point precision; returns
  `skewness`/`kurtosis` directly alongside the statistic/p-value
- `classical_decompose` -- classical (moving-average) seasonal
  decomposition, matching both R's `stats::decompose()` and Python's
  `statsmodels.tsa.seasonal.seasonal_decompose()` exactly at their shared
  defaults (both references' exact source read directly and
  cross-checked against each other and against this implementation, not
  from documentation alone); adopts two genuine Python-only capabilities
  R has no equivalent for at all (`two_sided=false` causal trend,
  `extrapolate_trend` edge extrapolation); deliberately stricter
  validation than R's silently-broken-on-bad-input behavior, matching
  Python's instead
- `stl_decompose` -- Seasonal-Trend decomposition using Loess (STL;
  Cleveland et al. 1990), matching both R's `stats::stl()` and Python's
  `statsmodels.tsa.seasonal.STL` exactly once parameters are matched --
  including the outer/robustness loop (`robust`/`outer`), verified
  against real `statsmodels` on data with genuine outliers to 13+
  significant digits; window parameters (`seasonal_window`/
  `trend_window`/`low_pass_window`) are validated as odd and `>= 3`
  (`>period` for the latter two), matching Python's strict behavior
  rather than R's silent auto-correction; `outer` runs whenever the
  *effective* outer count is `> 0` (via `robust=true`'s default, or an
  explicit `outer` regardless of `robust` -- confirmed from both
  references that `robust` itself has no direct computational effect),
  deliberately targeting Python's fixed bisquare/median computation
  rather than R's original NETLIB Fortran, which has a documented
  median-computation bug under `robust=TRUE`; `STLDecomposition` exposes
  the final robustness `weights` (all `1.0` when the outer loop never
  ran); a `parallel::Bool=true` keyword threads the cycle-subseries step
  (`Threads.@threads`, the one genuinely parallel piece of STL, confirmed
  from both R's and Python's actual MSTL source) -- numerically identical
  to `parallel=false` either way, since every phase writes to disjoint
  output positions
- `mstl_decompose` -- Multiple Seasonal-Trend decomposition (MSTL;
  Bandura, Hyndman & Bergmeir, 2021) for series with more than one
  seasonal period (e.g. hourly data with both daily and weekly
  seasonality); a thin sequential wrapper over `stl_decompose` (the
  `iterate`×`periods` loop is provably sequential in both references, not
  parallelizable), matching real `statsmodels.tsa.seasonal.MSTL`'s actual
  source to 10+ significant digits. `lambda` applies the classic Box-Cox
  transform before decomposing (never inverted back, matching Python) --
  `lambda=0` deliberately applies the correct `log(x)`, unlike Python's
  own source, where `lmbda=0.0` is silently a no-op due to a Python
  truthiness quirk (confirmed by execution); `lambda=:auto` (MLE
  estimation) is a documented, explicit gap, not yet built. Periods `>=`
  half the series length are dropped with a warning rather than erroring,
  matching both references
- `_optimize`/`OptimResult` -- thin, backend-decoupled wrapper around
  `Optim.jl` (L-BFGS/BFGS/gradient-free Nelder-Mead), the shared
  optimization engine every future MLE-fit model (ARIMA, SARIMAX, ETS,
  GARCH) will call into; private/unexported like `_ols`, not a public
  regression API in its own right. Uses `autodiff=:forward`, not the
  newer `ADTypes.jl`-based interface some current `Optim.jl` docs show --
  verified by execution that the `v1.11.x` line this package's own
  `julia = "1.9"` compat floor resolves to doesn't support that interface
  at all
- `partrans`/`invpartrans` -- the Monahan (1984) stationarity/
  invertibility-preserving reparametrization every future MLE-fit AR/MA
  model needs, matching R's `stats::arima` C source (`arima.c`) exactly.
  Deliberately does **not** match Python's `statsmodels` convention
  (`u/sqrt(1+u^2)` plus a sign negation) -- both are individually valid
  parameterizations of the same constrained space (verified numerically
  distinct, not assumed); only final fitted AR/MA coefficients, not raw
  optimizer-space parameters, are expected to be comparable across an
  R-based and Python-based implementation

All of the above are implemented natively (no calls out to R or Python,
`Optim.jl` for general-purpose numerical optimization aside -- the same
category of dependency as `LinearAlgebra`/BLAS, not a statistical
algorithm this package should own),
using `LinearAlgebra`'s QR (default) or Cholesky factorization for the
internal regression steps and a from-scratch Lanczos/continued-fraction
chi-squared tail, to avoid a hard `Distributions.jl` dependency this
early. `Distributions.jl` may be adopted later once heavier distributional
machinery (state-space likelihoods, forecast intervals) needs it anyway.

## Known limitations (tracked, not hidden)

- `adf_test` and `kpss_test` report **approximate** p-values (linear
  interpolation among asymptotic critical values), not the finite-sample
  MacKinnon response-surface p-values that R/`statsmodels` use. Adequate
  for a significant/not-significant call at 1/5/10%; exact p-values are a
  planned refinement.
- `pacf`'s `:burg` method (Burg's method) is not implemented -- a
  genuinely different algorithm from the three that are (`:yw`/`:ywm`/
  `:ols`), not just a denominator/sample-size variant.
- `kpss_test`'s `nlags=:auto` (Python's actual default -- the Hobijn et
  al. 1998 data-dependent bandwidth method) is not implemented; the
  default here (`:short`) matches R's default instead, documented
  explicitly rather than silently approximated under the `:auto` name.
- `acf`/`pacf` reject `NaN` input outright (mirroring R's `na.fail`
  default); there's no partial/dropped-observation missing-data policy
  yet across the package.

## Documentation policy

Documentation isn't a post-hoc pass — it's enforced structurally:

- **Every exported function gets a full docstring before it's considered
  done**, not after. `docs/make.jl` sets `checkdocs=:exports`, which fails
  the docs build if any exported name is missing one — this is a CI gate,
  not a guideline.
- **Docstrings include a runnable example** where practical, written as a
  `jldoctest` block once verified locally (see the caveat on
  `docs/src/getting_started.md` about examples written without a Julia
  runtime available — verify, then convert). `make.jl` sets
  `doctest=true`, so a docstring's example output silently going stale is
  a CI failure, not a documentation debt that quietly accumulates.
- **Narrative docs** (`docs/src/getting_started.md` and future tutorial
  pages) reuse the same benchmark series already used in the test suite
  (Nile, AirPassengers, sunspots) rather than inventing separate examples
  — one less thing to keep in sync.
- **API reference** (`docs/src/api.md`) is organized by category, not
  alphabetically or by file — it should read as a guide to the package's
  shape, not a dump of its namespace.
- Docs build and deploy via `.github/workflows/Docs.yml` on every push to
  `main`, so the published site never drifts far from what's actually in
  the package.

## Design principles

0. **Container-agnostic by construction, not by integration.** TSAnalytics
   does not depend on, or write extension code for, TSFrames.jl,
   TimeSeries.jl, DataFrames.jl, or any other time series container. Every
   public function accepts anything the two-method `tsvalues`/`tsindex`
   interface (see `src/interface.jl`) can be called on -- which, by
   default, is *any* iterable via `collect`. A plain `Vector`, a
   `TSFrame` column, `TimeSeries.values(ta)`, a `DataFrame` column, or a
   container from a package that doesn't exist yet all work identically,
   with zero TSAnalytics-side knowledge of which one you're using. This
   deliberately avoids coupling TSAnalytics's release cadence to any other
   package's — see `test/test_interface.jl` for a from-scratch mock
   container proving this end-to-end.
0.5. **Reference, never port.** Every algorithm in this package is
   implemented natively in Julia from its primary source -- the original
   paper or textbook (Durbin & Koopman for state space, Hyndman &
   Khandakar (2008) for auto-ARIMA, Monahan (1984) for the stationarity
   reparametrization, Kwiatkowski et al. (1992) for KPSS, and so on) --
   never by translating another package's source code, in any language.
   R (`stats::arima`, `forecast`), Python (`statsmodels`), and other Julia
   packages (`StateSpaceModels.jl`, `MarSwitching.jl`) are used for two
   things only: (a) understanding what a well-tested implementation of the
   algorithm actually does when the paper is ambiguous, and (b) validating
   *output numbers* against, on standard benchmark series. This matters
   concretely for the state-space engine: statsmodels' own Kalman filter
   recursion is Cython, not Python -- it exists specifically because a
   pure-Python loop is too slow for that hot path. Julia doesn't have that
   problem; a native Julia loop backed by `LinearAlgebra`/BLAS reaches the
   same speed without a compiled sub-layer, which is the entire point of
   writing this in Julia rather than wrapping something else.
1. **StatsAPI-first.** Once model fitting lands, every model implements
   `fit`, `coef`, `vcov`, `residuals`, `predict`, `loglikelihood`, `aic`,
   `bic` -- the same contract GLM.jl uses, for free interoperability with
   the JuliaStats ecosystem.
2. **One state-space engine, many models.** ARIMA/SARIMAX,
   UnobservedComponents, and linear ETS will all be expressed as a
   `GaussianSSM` and share a single, well-tested Kalman filter/smoother,
   rather than each model reimplementing filtering.
3. **Validated against reference implementations.** Every model is
   checked against R (`stats::arima`, `forecast`) or `statsmodels` on
   standard benchmark series (AirPassengers, Nile, co2, sunspots), not
   just internal consistency.
4. **`LinearAlgebra` over hand-rolled loops** wherever it's the faster
   and more numerically stable choice -- QR/Cholesky factorizations for
   regression, BLAS-backed matrix ops in the Kalman filter.

## Roadmap

Small, stable releases every 3 months rather than one large one.

| Release | Target | Scope |
|---|---|---|
| **v0.1.0** | Month 3 | This diagnostics layer + generalized Kalman filter/smoother (diffuse init) + AR-X, ARIMA, SARIMAX + auto-order selection + residual diagnostics wired together |
| v0.2.0 | Month 6 | STL / classical / MSTL decomposition, ETS (additive + damped trend), exact ADF/KPSS p-values |
| v0.3.0 | Month 9 | VAR/VECM, Granger causality, impulse responses, cointegration tests |
| v0.4.0 | Month 12 | UnobservedComponents (structural time series), Markov-switching, HP/BK/CF filters |

## Installation (once registered)

```julia
] add TSAnalytics
```

Until then:

```julia
] add https://github.com/<your-org>/TSAnalytics.jl
```

## Quick example

```julia
using TSAnalytics, Random

Random.seed!(1)
y = cumsum(randn(500))          # a random walk

adf_test(y)                      # fails to reject unit root, as expected
kpss_test(y)                     # rejects level-stationarity, as expected

r = acf(diff(y), 0:10)           # differenced series should look like white noise
ljungbox_test(diff(y), 10)       # confirms it
```
