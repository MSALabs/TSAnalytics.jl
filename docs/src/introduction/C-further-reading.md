# Further Reading

The six reference books this project's own "reference, never port" policy
is checked against (see `development-sequence.md` and the project's
`CLAUDE.md`) — every algorithm in TSAnalytics.jl is implemented natively
from one of these (or a cited paper), never translated from another
package's source.

- **Hamilton, J. D. (1994).** *Time Series Analysis.* Princeton University
  Press. The classical econometric treatment — ARMA/ARIMA theory, unit
  roots and the Dickey-Fuller family, state-space models and the Kalman
  filter, and (G)ARCH, all derived from first principles with full proofs.
- **Hyndman, R. J. & Athanasopoulos, G.** *Forecasting: Principles and
  Practice* (3rd ed., "fpp3"). OTexts. The applied, R-`tidyverts`-flavored
  companion — decomposition (STL), Box-Cox and Guerrero's method,
  automatic ARIMA order selection, exponential smoothing (ETS), and
  forecast accuracy evaluation, all with a strong "what would you actually
  do" perspective.
- **Durbin, J. & Koopman, S. J. (2012).** *Time Series Analysis by State
  Space Methods* (2nd ed.). Oxford University Press. The authoritative
  state-space reference — the Kalman filter and smoother, diffuse
  initialization for non-stationary states, and the structural
  (unobserved-components) time series model.
- **Tsay, R. S. (2010).** *Analysis of Financial Time Series* (3rd ed.).
  Wiley. The financial-econometrics reference — ARCH/GARCH and their
  variants (GJR, EGARCH), the ARCH-LM test, and realized-volatility
  measures.
- **Ladiray, D. & Quenneville, B. (2001).** *Seasonal Adjustment with the
  X-11 Method.* Springer Lecture Notes in Statistics. The X-11/X-13-style
  seasonal adjustment reference — out of TSAnalytics.jl's own scope (see
  the sibling package `SeasonalAdjustment.jl`), but part of the same
  6-book comparison exercise that shaped this package's own diagnostic
  and decomposition coverage.
- **Shumway, R. H. & Stoffer, D. S.** *Time Series Analysis and Its
  Applications: With R Examples* (latest ed., companion to the `astsa` R
  package). Springer. The spectral-analysis reference (periodogram,
  tapering, kernel-smoothed spectral density) and the source of this
  package's `diagnostic_plot` layout, transcribed directly from
  `astsa::sarima()`'s own real source.

## Primary sources for exponential smoothing

The books above set out the methods; these are the papers the code was
built from.

- **Holt, C. C. (2004).** Forecasting seasonals and trends by
  exponentially weighted moving averages. *International Journal of
  Forecasting* 20(1), 5–10. The 1957 Office of Naval Research memorandum
  that introduced the trend recursion, reprinted.
- **Winters, P. R. (1960).** Forecasting sales by exponentially weighted
  moving averages. *Management Science* 6(3), 324–342. The seasonal
  extension — [`holt_winters`](@ref).
- **Gardner, E. S. & McKenzie, E. (1985).** Forecasting trends in time
  series. *Management Science* 31(10), 1237–1246. The damped trend.
- **Hyndman, R. J., Koehler, A. B., Snyder, R. D. & Grose, S. (2002).**
  A state space framework for automatic forecasting using exponential
  smoothing methods. *International Journal of Forecasting* 18(3),
  439–454. The innovations state-space form and the automatic selection
  that [`auto_ets`](@ref) implements.
- **Hyndman, R. J., Koehler, A. B., Ord, J. K. & Snyder, R. D. (2008).**
  *Forecasting with Exponential Smoothing: The State Space Approach.*
  Springer. The ETS taxonomy, the parameter regions, and the class 1, 2
  and 3 variance results [`fit_ets`](@ref) uses for its intervals.
- **Assimakopoulos, V. & Nikolopoulos, K. (2000).** The theta model: a
  decomposition approach to forecasting. *International Journal of
  Forecasting* 16(4), 521–530. The Theta method, winner of the M3
  competition.
- **Hyndman, R. J. & Billah, B. (2003).** Unmasking the Theta method.
  *International Journal of Forecasting* 19(2), 287–290. The proof that
  Theta is simple exponential smoothing plus half the regression slope
  as drift — the form [`fit_theta`](@ref) actually computes.
- **De Livera, A. M., Hyndman, R. J. & Snyder, R. D. (2011).**
  Forecasting time series with complex seasonal patterns using
  exponential smoothing. *Journal of the American Statistical
  Association* 106(496), 1513–1527. TBATS — [`fit_tbats`](@ref).

## Primary sources for multivariate time series

- **Lütkepohl, H. (2005).** *New Introduction to Multiple Time Series
  Analysis.* Springer. The reference for the whole multivariate track:
  VAR estimation and order selection, Granger and instantaneous
  causality, impulse responses with their asymptotic standard errors,
  variance decomposition, the VECM, structural VARs, and the residual
  diagnostics of [`var_diagnostics`](@ref).
- **Sims, C. A. (1980).** Macroeconomics and reality. *Econometrica*
  48(1), 1–48. The case for the VAR, and for impulse responses as the
  way to read one.
- **Granger, C. W. J. (1969).** Investigating causal relations by
  econometric models and cross-spectral methods. *Econometrica* 37(3),
  424–438. — [`granger_test`](@ref).
- **Engle, R. F. & Granger, C. W. J. (1987).** Co-integration and error
  correction: representation, estimation, and testing. *Econometrica*
  55(2), 251–276. — [`coint_test`](@ref).
- **MacKinnon, J. G. (1994).** Approximate asymptotic distribution
  functions for unit-root and cointegration tests. *Journal of Business
  & Economic Statistics* 12(2), 167–176; and **MacKinnon, J. G.
  (2010).** Critical values for cointegration tests. Queen's Economics
  Department Working Paper 1227. The response surfaces behind the
  Engle-Granger p-values and critical values.
- **Phillips, P. C. B. & Ouliaris, S. (1990).** Asymptotic properties of
  residual based tests for cointegration. *Econometrica* 58(1),
  165–193. — [`phillips_ouliaris_test`](@ref).
- **Johansen, S. (1988).** Statistical analysis of cointegration
  vectors. *Journal of Economic Dynamics and Control* 12(2–3), 231–254;
  **Johansen, S. & Juselius, K. (1990).** Maximum likelihood estimation
  and inference on cointegration — with applications to the demand for
  money. *Oxford Bulletin of Economics and Statistics* 52(2), 169–210;
  and **Johansen, S. (1995).** *Likelihood-Based Inference in
  Cointegrated Vector Autoregressive Models.* Oxford University Press.
  — [`johansen_test`](@ref) and [`fit_vecm`](@ref).
- **Osterwald-Lenum, M. (1992).** A note with quantiles of the
  asymptotic distribution of the maximum likelihood cointegration rank
  test statistics. *Oxford Bulletin of Economics and Statistics* 54(3),
  461–472. The critical values [`johansen_test`](@ref) reports.
- **MacKinnon, J. G., Haug, A. A. & Michelis, L. (1999).** Numerical
  distribution functions of likelihood ratio tests for cointegration.
  *Journal of Applied Econometrics* 14(5), 563–577. Response surfaces
  that would give the Johansen test real p-values; not yet implemented,
  which is why [`johansen_test`](@ref) reports critical values only.
- **Blanchard, O. J. & Quah, D. (1989).** The dynamic effects of
  aggregate demand and supply disturbances. *American Economic Review*
  79(4), 655–673. — [`blanchard_quah`](@ref).
- **Amisano, G. & Giannini, C. (1997).** *Topics in Structural VAR
  Econometrics* (2nd ed.). Springer. The A-, B- and AB-models of
  [`fit_svar`](@ref).
- **Hosking, J. R. M. (1980).** The multivariate portmanteau statistic.
  *Journal of the American Statistical Association* 75(371), 602–608.
  — [`portmanteau_test`](@ref).
- **Edgerton, D. & Shukur, G. (1999).** Testing autocorrelation in a
  system perspective. *Econometric Reviews* 18(4), 343–386.
  — [`es_test`](@ref).
