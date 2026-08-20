# TSAnalytics.jl

[![docs latest](https://img.shields.io/badge/docs-dev-blue.svg)](https://msalabs.github.io/TSAnalytics.jl/dev/)
[![CI](https://github.com/MSALabs/TSAnalytics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/MSALabs/TSAnalytics.jl/actions/workflows/CI.yml)

A native Julia time series analysis package: ARIMA/SARIMAX, seasonal
decomposition, exponential smoothing, and the diagnostic tests that go
with them -- built incrementally, in the spirit of GLM.jl (separation of
model/predictor concerns, `StatsAPI` conventions, `LinearAlgebra`-backed
numerics).

**Status: pre-release, active development.** This repository currently
contains the foundational descriptive-statistics, diagnostic-testing, and
decomposition layers, the optimizer/reparametrization infrastructure
every MLE-fit model will share, its first fitted model (`arx`) with
forecasting (`forecast`), accuracy metrics (`mae`/`rmse`/`mape`/
`smape`/`mase`/`accuracy`), rolling-origin cross-validation (`tscv` and
friends), and classical exponential smoothing (`holt_winters`) --
completing Stage 5 -- plus the shared Gaussian state-space engine and
its first consumer, non-seasonal ARMA maximum-likelihood fitting
(`fit_arma`), its differencing-aware wrapper (`fit_arima`,
ARIMA(p,d,q)), full seasonal ARIMA (`fit_sarima`,
ARIMA(p,d,q)(P,D,Q)_s), automatic order selection (`auto_arima`,
Hyndman-Khandakar), and the full GARCH-family volatility toolkit --
GARCH/GJR-GARCH/EGARCH fitting (`fit_garch`), multi-step forecasting
(`forecast_volatility`), and nonparametric realized volatility measures
(`realized_variance`, `jump_test`, ...), completing Stage 7. Stage 8's
engine generalization (time-varying `Z_t`/`T_t`/`R_t`/`Q_t`/`H_t`
state-space matrices, exact diffuse initialization) now has its first
user-facing consumer: full ARIMAX/SARIMAX with exogenous regressors
(`fit_arimax`, `fit_sarimax`) -- two genuinely different ways to treat
the regression coefficient, `model=:mle` (fixed, jointly-estimated) or
`model=:tvss` (a real latent, time-varying state), verified directly
against real R and Python. See [Roadmap](#roadmap) below.

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
- `durbin_watson_test` -- Durbin-Watson test for first-order
  autocorrelation in regression residuals; statistic matches
  `statsmodels.stats.stattools.durbin_watson` exactly (verified against
  its source), `alternative=:greater` matches R `lmtest::dwtest`'s
  default. `method=:approx` (large-sample normal approximation) is the
  only method implemented, honestly documented as cruder than R's exact
  method (which needs the eigenvalues of an `X`-derived matrix, not
  computable from residuals alone -- the same reason Python's own
  version skips a p-value entirely); `method=:exact` throws a clear,
  named error rather than silently approximating under that name
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
- `arx`/`ARXModel` -- **the project's first fitted model**: autoregressive
  with optional exogenous regressors ("AR-X"), fit by conditional least
  squares. Matches Python's `statsmodels.tsa.ar_model.AutoReg` throughout
  (`lags` as an integer *or* an arbitrary subset like `[1,3]`,
  `trend=:n/:c/:t/:ct`, `seasonal`, `exog`, `hold_back`) rather than R's
  much narrower `stats::ar.ols` (booleans only, no `exog` support at
  all); deliberately conditional-least-squares only, not matching R's
  `stats::ar()` (a different function, defaulting to Yule-Walker). First
  real consumer of the `StatsAPI` contract (`coef`/`vcov`/`stderror`/
  `residuals`/`nobs`/`loglikelihood`/`aic`/`bic`) and of
  `StatsBase.CoefTable`-based `show` output. Verified exactly against
  real `AutoReg` — coefficients, standard errors, `sigma2`,
  log-likelihood, AIC, BIC, and p-values — across every `trend` value, an
  `exog` regressor, a lag *subset*, `seasonal` dummies, and explicit
  `hold_back`; standard errors are computed independently rather than
  reused from `_ols` (whose own `se` uses a different, `n-k`-denominator
  convention than `AutoReg`'s actual `n`-denominator one — confirmed
  numerically, not assumed, a ~1% systematic difference), and a singular/
  collinear design raises a clear `ArgumentError` rather than a raw
  `SingularException`
- `forecast`/`Forecast` -- multi-step-ahead forecasting with prediction
  intervals from a fitted `ARXModel`, R's `forecast(level=...)`
  convention (multiple nested percentage levels in one call, default
  `[80.0, 95.0]`) rather than Python's single-`alpha` `get_prediction`.
  Standard errors use the known-parameters Box-Jenkins psi-weight
  propagation formula, verified to exactly reproduce **both** real
  `statsmodels`' `AutoReg.get_prediction()` **and** real R's
  `predict.ar()`/`forecast.ar()`/`print.forecast()` output (R genuinely
  is installed on the dev machine this was built on, just not on `PATH`
  -- an earlier assumption to the contrary was wrong). Point forecasts
  walk the fitted model's own column labels generically (not hand-coded
  per `trend`/`seasonal` case), so an arbitrary lag *subset*, `trend`/
  `seasonal` continuation, and combinations of both are all verified
  exact against fresh reference numbers, not just the default case.
  Forecasting a model fit with `exog` raises a clear error (future exog
  values would be needed and aren't available) rather than silently
  ignoring the regressor
- `mae`, `rmse`, `mape`, `smape`, `mase`, `accuracy` -- forecast accuracy
  metrics, verified exactly against real `sktime`'s
  `mean_absolute_error`/`mean_squared_error`/
  `mean_absolute_percentage_error`/`mean_absolute_scaled_error` (source
  read directly, not just docs) and against real R's
  `forecast::accuracy()`, confirming R reports `MAPE` as a percentage
  (e.g. `33.69`) while `sktime` uses a raw fraction (`0.3369`) --
  `as_percentage=true` is the default, matching R's more commonly
  expected convention. `mase` takes the training series as a required
  argument (a mathematical necessity: it can't be computed from
  actual/predicted alone) and supports a seasonal-naive benchmark via
  `sp`. `smape`'s docstring carries a prominent warning, quoting Hyndman
  (its own metric family's author) recommending against using sMAPE at
  all -- prefer plain MAPE or MASE instead
- `expanding_window_split`, `sliding_window_split`, `tscv` --
  rolling-origin cross-validation, both shapes R and Python actually use:
  the low-level splitters return `(train_idx, test_idx)` pairs matching
  `sktime`'s `ExpandingWindowSplitter`/`SlidingWindowSplitter` exactly
  (verified against real index output); `tscv` is the R-`forecast::tsCV`-
  style "batteries included" wrapper (user-supplied fit-forecast
  callable, returns an error matrix), built *on top of* the splitters so
  the two conventions can't silently disagree. Cross-checked against real
  `forecast::tsCV()` output too: error values agree exactly at every
  fold R can compute; `tscv` just omits R's always-`NA` padding rows
  (folds that can never be computed, e.g. beyond the series end)
- `holt_winters`/`ExponentialSmoothingModel` -- simple / Holt's linear /
  Holt-Winters exponential smoothing, fit by SSE minimization.
  `initialization_method=:heuristic` (default) uses fixed initial states
  from `classical_decompose`+OLS, **verified to reproduce R's
  `stats::HoltWinters()` fitted/level/trend/season output bit-for-bit**
  (~1e-13) across additive, multiplicative, with/without trend, and
  simple-ES cases -- catching a genuine formula subtlety along the way
  (the seasonal update uses the *already-updated* level, not
  `l_{t-1}+b_{t-1}`, invisible until comparing R's second seasonal
  cycle). `:estimated` instead jointly optimizes the initial states with
  the smoothing parameters, matching Python's
  `statsmodels.tsa.holtwinters.ExponentialSmoothing` default philosophy
  -- confirmed by direct execution that these are genuinely different
  optimization problems (different `alpha`/`beta`/`gamma`/SSE on
  identical data), not just different starting guesses. Also found and
  fixed a real optimizer-selection bug along the way: `_optimize`'s
  default `:lbfgs` silently converges to a badly wrong optimum on this
  sigmoid-bounded objective (verified by direct testing), so
  `holt_winters` uses `:nelder_mead` with a generous iteration budget
  instead
- `fit_arma`/`ArmaModel` -- non-seasonal ARMA(p,q) maximum-likelihood
  fitting via the shared `GaussianSSM` state-space engine (Durbin &
  Koopman companion form; time-invariant, stationary Lyapunov
  initialization), wired to the Monahan reparametrization (`partrans`)
  and the Stage 4.1 optimizer. Verified end-to-end against real R
  `stats::arima()` and real Python `statsmodels.tsa.arima.model.ARIMA`.
  `include_mean=true` (default) estimates the mean *jointly* with the
  AR/MA coefficients, matching R's actual behavior -- not a one-time
  sample-mean subtraction, which was checked directly against R and
  found to give a different, wrong point estimate. `se_type=:hessian`
  (default, R's convention) vs. `:opg` (Python's) are confirmed
  genuinely different numbers, both valid, via `ForwardDiff.hessian`/
  `ForwardDiff.jacobian` -- exact automatic differentiation, no
  numerical-differencing pass, a real structural advantage over R's and
  Python's compiled-but-hand-differentiated backends. Also fixed two
  latent bugs surfaced by building this: `partrans`/`invpartrans`
  (Stage 4.2) weren't `ForwardDiff`-safe (hardcoded `Float64` internally,
  harmless until something called them from inside an autodiff'd
  optimizer objective, which this stage is the first to do), and an
  AIC/BIC formula omitting `sigma2` from the parameter count (confirmed
  wrong against R's actual `BIC()` output)
- `fit_arima`/`ArimaModel` -- ARIMA(p,d,q): differences `y` (Stage 1.1's
  `diff`) and delegates the stationary fit entirely to `fit_arma`, a
  genuinely thin wrapper verified by a `d=0` regression guard
  (bit-identical to calling `fit_arma` directly, across 5 order
  combinations). `nobs = n - d`, matching R's `stats::arima()` exactly
  -- deliberately not Python's `statsmodels ARIMA`, whose `nobs` is the
  full `n` because it uses diffuse state augmentation internally rather
  than pre-differencing (verified directly, on two series). `include_mean`
  is silently forced to `false` whenever `d > 0`, matching R's actual
  behavior exactly (confirmed: `include.mean=TRUE`/`FALSE` give
  bit-identical fits when `d>0` in real R). Verified against a 20-case
  dual R+Python bulk sweep (`d ∈ {0,1,2}` × 7 `(p,q)` structures,
  `test/verification/arima/bulk/`, full regeneration pipeline included)
  beyond the two cases in its own handoff -- which surfaced a real
  optimizer-quality gap (a zero-start LBFGS landing on a materially
  worse invertibility-boundary local optimum for one ARMA(2,1) case;
  fixed by regenerating that series, not by loosening the corpus's
  tolerances) and a real robustness bug in `fit_arma`'s standard-error
  helpers (`_hessian_se`/`_opg_se` crashed with a raw `LAPACKException`
  on a singular Hessian at that same kind of boundary optimum -- now
  returns `NaN` standard errors instead of crashing). Also closed a real
  gap in `fit_arma` itself found while building this: `order=(0,0)`
  (needed for `ARIMA(0,d,0)`, e.g. a pure random walk after
  differencing -- common, and R supports it directly) was previously
  rejected as "nothing to fit"; now matches R's
  `arima(order=c(0,0,0))` exactly, with or without a mean
- `fit_sarima`/`SarimaModel` -- full seasonal ARIMA(p,d,q)(P,D,Q)_s.
  Unlike `fit_arima`, not a thin wrapper -- `combined_ar_ma` (already
  dual-verified for 20 seasonal cases) has to run *inside* the optimizer
  loop, since the combined polynomial changes every iteration across
  four independently-searched blocks, so this stage recomposes the
  actual numerical primitives (`partrans`, `build_statespace`,
  `kalman_filter`, `_optimize`, the CSS recursion) rather than
  delegating to `fit_arma` directly. `nobs = n - d - D*s`, extending
  `fit_arima`'s convention (verified directly against real R);
  `include_mean` silently forced off whenever `d>0` **or** `D>0`
  (confirmed directly that `D>0` alone also drops the mean, not just
  assumed to extend from the non-seasonal case). Verified against a
  dedicated synthetic case with all four polynomial blocks nonzero
  (`p,q,P,Q > 0`) -- explicitly flagged as untested in its own handoff,
  since the primary reference case only exercised `P`/`Phi` -- plus a
  12-case dual R+Python bulk sweep (`test/verification/sarima/bulk/`,
  full regeneration pipeline included). Building the bulk sweep also
  surfaced a real Windows-filesystem gotcha: case-insensitive filename
  collisions between generated series (`P_only.csv`/`p_only.csv`) that
  silently overwrote data, the same class of issue this package's own
  `gaussianssm.jl` merge had already flagged. Refactored Stage 6.5's
  `_hessian_se`/`_opg_se`/CSS-recursion helpers to be generic (a
  closure-based natural-objective/contributions/unpack argument instead
  of hardcoded ARMA-specific parameters) so this stage reuses them
  directly rather than duplicating the Hessian/OPG/CSS logic -- verified
  no regression against the full Stage 6.5/6.6 suite before building on
  top of the refactor
- `auto_arima` -- automatic order selection (Hyndman & Khandakar 2008),
  built entirely on `fit_arima`/`fit_sarima`; contributes no new fitting
  math, only the `(p,q,P,Q)` search. `stepwise` (default) is the actual
  Hyndman-Khandakar greedy hill-climb (four base models, then repeated
  ±1 neighbor moves, taking the best-improving move each round);
  `stepwise=false` is a genuinely parallel exhaustive grid search
  (`Threads.@threads`, same guarded pattern as MSTL's `parallel`
  keyword) -- confirmed as the literal documented scope of both R's
  `parallel=`/`num.cores=` and Python's `n_jobs=`, which apply *only* to
  this non-default mode. `information_criterion` defaults to `:aicc`,
  matching R's default (not Python's `:aic`). `d` is auto-detected via
  repeated `kpss_test`; `D` is **not** auto-detected -- neither seasonal
  unit-root test either reference uses (R's Canova-Hansen, Python's
  OCSB) exists in this project yet, so `seasonal=true` requires `D`
  passed explicitly, an honest documented limitation rather than a
  silent default. Verified against 20 synthetic series (of a documented
  36-case sweep) dual-checked against real `pmdarima.auto_arima` --
  Julia's selected order matches pmdarima's own on 17/20 (85%), well
  above the 33.3% true-order-recovery rate even pmdarima itself achieves
  at this sample size, confirming the project's own accuracy framing:
  matching the reference's search is the meaningful, checkable bar, not
  recovering the unknowable true order. Building this surfaced and fixed
  a real pre-existing bug in `fit_sarima` (Stage 6.7): its optimizer call
  was missing the zero-free-parameters guard `fit_arma` already had,
  crashing on the pure-white-noise `(0,·,0)(0,·,0)` candidate that
  `auto_arima`'s own base-model search is the first thing to actually
  try
- `fit_arimax`/`fit_sarimax`/`ArimaxModel`/`SarimaxModel` -- (S)ARIMA
  with exogenous regressors, two genuinely different models behind one
  function, confirmed directly against real Python `SARIMAX(
  time_varying_regression=True, mle_regression=False,
  use_exact_diffuse=True)`, not assumed: `model=:mle` (default) treats
  `beta` as an ordinary joint-MLE coefficient (`y - X·beta`, both
  differenced exactly as `fit_arima`/`fit_sarima`'s own convention, fed
  to the unmodified Stage 6 likelihood -- no Stage 8.1/8.2 machinery
  needed); `model=:tvss` makes `beta` a genuinely latent, time-varying
  **state**, recovered via Stage 8.1's `TimeVaryingSSM` + Stage 8.2's
  `kalman_filter_diffuse` -- only its process variance `Q_beta` is
  optimized, `beta` itself never is (matching real `SARIMAX`'s own
  reported parameter list exactly). Named `model`, deliberately not
  `method` -- `method` was already Stage 6.5/6.7's `:ml`/`:css_ml`
  estimation-procedure choice, and both are needed simultaneously
  (`model=:tvss, method=:css_ml` is a real, sensible combination).
  Constraining `Q_beta=0` makes `model=:tvss`'s point estimates converge
  closely to `model=:mle`'s, but their likelihoods still genuinely
  differ -- `model=:tvss`'s diffuse-phase observations carry a different
  likelihood formula entirely (a diffuse marginal term, not an ordinary
  fixed-parameter density), asserted as an explicit inequality in the
  test suite rather than "fixed" into equality. Building this caught and
  fixed a real bug in Stage 8.2's own `kalman_filter_diffuse`: its
  `loglik` was excluding diffuse-phase observations entirely, when real
  `statsmodels`'s own genuine default includes them (`loglikelihood_burn
  = 0` under exact diffuse init) -- caught because `model=:tvss`,
  evaluated at Python's own exact fitted parameters, showed the *same*
  small constant loglik offset across two different parameter settings,
  ruling out optimizer noise. `fit_sarimax` is not a thin wrapper over
  `fit_arimax` (same reasoning as `fit_sarima` over `fit_arima`) -- both
  share an internal `_fit_arimax_core`, verified to satisfy the
  reduction property (`seasonal_order=(0,0,0,·)` exactly matches
  `fit_arimax`) directly
- `fit_garch`/`fit_garch_multi`/`GarchModel` -- GARCH(p,q) volatility
  modeling (Bollerslev 1986), own likelihood entirely independent of the
  Gaussian state-space engine. **`p`=ARCH order, `q`=GARCH order,
  matching Python `arch_model(p=,q=)`'s own naming** -- the opposite of
  Bollerslev's original notation, resolved by direct execution (fitting
  `arch_model(p=2,q=1)` and checking which fitted names came back)
  before writing anything else, per the handoff's own explicitly flagged
  risk. A from-scratch reparametrization keeps the optimizer unconstrained
  while guaranteeing `omega>0`, `alpha,beta>=0`, `sum(alpha)+sum(beta)<1`
  by construction (`omega=exp(raw)`; `alpha`/`beta` a softmax over `p+q`
  categories plus one implicit zero-logit category). `cov_type=:robust`
  (default) is a genuine Bollerslev-Wooldridge sandwich estimator
  (`inv(H)*cov(scores)*inv(H)/n` via `ForwardDiff`, not finite
  differences) -- confirmed genuinely different numbers from `:classic`
  both from reading `arch`'s own covariance source and from a real fit.
  Verified to ~1e-5 against real Python `arch_model` on the primary case,
  exactly on a dedicated GARCH(2,1) case (locking in the naming
  resolution), and exactly on a `mean_spec=:constant` case (joint
  mean+GARCH MLE, not a naive pre-demean -- same rigor as `fit_arma`'s
  `include_mean`); a 24-case bulk sweep matched real `arch_model` to a
  mean absolute parameter error of `6.5e-6`. Two parallelism designs,
  both precedent-validated against real `rugarch` documentation (not
  independently executed -- R's `rugarch` itself couldn't be installed):
  `fit_garch_multi` (matches `rugarch::multifit`) and `n_restarts`
  multi-start (matches `rugarch`'s `gosolnp`), both `Threads.@threads`-
  parallel by default. `dist=:t` deliberately not yet implemented --
  throws a clear, named error rather than silently falling back.
  **`model=:gjr`/`:egarch` extend the same function with asymmetric
  ("leverage") volatility** -- GJR-GARCH adds `gamma*e[t-1]^2*I(e[t-1]<0)`
  to the same level-variance recursion (confirmed matching
  `arch_model(vol='GARCH', o=1)` exactly); EGARCH is a genuinely
  separate recursion on `log(sigma2)`, implemented directly from `arch`'s
  own EGARCH source as the primary target (not a third-party formula --
  an independent academic source found EGARCH's `alpha`/`gamma`
  convention genuinely isn't standardized across implementations, unlike
  GJR-GARCH's). `m.omega < 0` for EGARCH is expected, not a bug --
  log-variance has no positivity constraint. **Two real bugs caught
  before shipping, both by comparing against real Python output
  immediately**: a first-draft GJR `gamma` reparametrization used a
  symmetric cap that silently prevented the optimizer from ever
  exceeding `gamma <= alpha` (the true constraint only bounds `gamma`
  from below), caught because the fitted `gamma` came out suspiciously
  exactly equal to `alpha`; and a tuple-ordering bug in the `model=
  :garch` dispatch path that briefly broke plain GARCH entirely during
  this extension's own refactor, caught by re-running Stage 7.1's
  already-passing test suite immediately after. Verified exactly against
  real Python on both a dual-verified GJR-GARCH(1,1) and EGARCH(1,1)
  case, plus a 48-case bulk sweep matching to a mean absolute parameter
  error of `6.3e-6`
- `forecast_volatility`/`VolatilityForecast` -- multi-step conditional
  variance forecasting. A genuinely different parallelism story than
  `fit_garch` itself: not multi-series/multi-restart, but thousands of
  independent Monte Carlo simulation paths, needed because EGARCH has no
  closed-form multi-step forecast at all (confirmed by direct execution:
  real `arch`'s `forecast(method='analytic')` throws for EGARCH beyond
  one step). `method=:auto` resolves to `:analytic` for GARCH/GJR-GARCH
  and `:simulation` for EGARCH automatically; requesting `:analytic`
  explicitly on EGARCH beyond one step still throws, matching Python's
  own behavior. The analytic multi-step recursion is implemented
  directly from `arch`'s own source, generalized for real multi-lag
  `p`/`q`. **Caught a real transcription typo in the source handoff
  itself** while verifying (a digit-transposition in one transcribed
  reference value) -- confirmed twice independently via direct
  re-execution before trusting either number. The simulation engine's
  own correctness is checked against its own analytic answer (50,000
  paths, `0.000725` max absolute difference, comparable to real `arch`'s
  own `0.00098`). Parallelism here is the strongest case in the whole
  GARCH module -- `rugarch`'s own bootstrap-forecast docs actively
  *recommend* parallelizing this specific workload, unprompted, stronger
  precedent than `fit_garch`'s own two parallel designs. A 216-check bulk
  verification reuses all 72 already-fitted models from the `fit_garch`
  bulk sweeps directly, no new data generation needed
- `realized_variance`/`bipower_variation`/`jump_test`/
  `realized_semivariance`/`realized_measures` -- nonparametric
  intraday-return volatility measures (Andersen & Bollerslev 1998;
  Barndorff-Nielsen & Shephard 2004/2006/2010), no dependency on
  `fit_garch` at all -- a fundamentally different, quadratic-variation-
  based approach. **The reference situation here is the reverse of every
  other Stage 7 sub-stage**: the originally-planned Python reference
  (`arch.realized`) doesn't exist -- confirmed directly, `arch` has no
  `realized` submodule at all, and no mature Python equivalent exists
  anywhere in that ecosystem. R's `highfrequency` turned out to be
  installable in this environment (unlike most R packages referenced
  earlier in this project), upgrading verification to full direct
  execution: every formula read from real, executed source, matched to
  full displayed precision on a numeric cross-check, not transcribed
  from documentation. **Found and fixed a real formula discrepancy**
  along the way -- the originally-cited bipower-variation formula
  included a finite-sample correction factor the actual reference
  implementation doesn't apply. `jump_test` deliberately uses the more
  robust `max(1, TQ/BV²)` variant of the BNS jump-test statistic, not
  `highfrequency`'s own less conservative default, documented
  explicitly. Verified by full statistical calibration across 1000
  simulated trading days (500 no-jump, 500 with a real injected jump)
  rather than point-matching a single reference output -- correct test
  size and high power confirmed directly, not assumed. `realized_measures`
  is the most naturally parallel design in the whole GARCH-family chapter:
  parallelizing over independent trading days is the actual default shape
  of how this gets used, not an add-on capability

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
] add https://github.com/MSALabs/TSAnalytics.jl
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
