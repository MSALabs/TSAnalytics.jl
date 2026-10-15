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

## Primary sources for components, factors and regimes

Part XI also cites two textbooks beyond the six above, each for what it
genuinely contains:

- **Cowpertwait, P. S. P. & Metcalfe, A. V. (2009).** *Introductory Time
  Series with R.* Springer. Chapter 12's dynamic linear models: the
  clearest pictures of filtering against smoothing across a level shift,
  and of how the variance ratio drives the filter. (It does not use R's
  `StructTS`, and estimates no variances by maximum likelihood.)
- **Montgomery, D. C., Jennings, C. L. & Kulahci, M. (2008).**
  *Introduction to Time Series Analysis and Forecasting.* Wiley.
  Exponential smoothing as the optimal forecast of an IMA(1,1) (§4.8), and
  monitoring a forecasting model with tracking signals (§2.6.3), which no
  other book here covers.

The papers:

- **Durbin, J. & Koopman, S. J. (2001).** *Time Series Analysis by State
  Space Methods* (1st ed.). Oxford University Press. The univariate
  treatment of multivariate observations used by the filter beneath
  [`fit_dfm`](@ref); see Appendix B on this citation's provenance.

- **Harvey, A. C. (1989).** *Forecasting, Structural Time Series Models
  and the Kalman Filter.* Cambridge University Press. The structural
  (unobserved-components) model — level, slope, seasonal and cycle, each
  with its own disturbance — that [`fit_uc`](@ref) fits; Durbin & Koopman
  above give the filtering and smoothing.
- **Koopman, S. J. & Durbin, J. (2000).** Fast filtering and smoothing
  for multivariate state space models. *Journal of Time Series Analysis*
  21(3), 281–296. Processing a vector of observations one series at a
  time — the sequential filter beneath [`fit_dfm`](@ref).
- **Ansley, C. F. & Kohn, R. (1986).** A note on reparameterizing a
  vector autoregressive moving average model to enforce stationarity.
  *Journal of Statistical Computation and Simulation* 24(2), 99–106; and
  **Morf, M., Vieira, A. & Kailath, T. (1978).** Covariance
  characterization by partial autocorrelation matrices. *Annals of
  Statistics* 6(3), 643–648. The map that keeps a factor VAR stationary
  while the optimiser searches freely.
- **Stock, J. H. & Watson, M. W. (1991).** A probability model of the
  coincident economic indicators. In K. Lahiri & G. H. Moore (eds.),
  *Leading Economic Indicators*, Cambridge University Press, 63–89. The
  dynamic factor model as an unobserved index of the business cycle.
- **Stock, J. H. & Watson, M. W. (2002).** Forecasting using principal
  components from a large number of predictors. *Journal of the
  American Statistical Association* 97(460), 1167–1179. The case for a
  few factors behind many series, and the principal-components start.
- **Bai, J. & Ng, S. (2002).** Determining the number of factors in
  approximate factor models. *Econometrica* 70(1), 191–221. — [`bai_ng_ic`](@ref).
- **Shephard, N. G. & Harvey, A. C. (1990).** On the probability of
  estimating a deterministic component in the local level model. *Journal
  of Time Series Analysis* 11(4), 339–347. The pile-up of variance
  estimates at zero.
- **Self, S. G. & Liang, K.-Y. (1987).** Asymptotic properties of maximum
  likelihood estimators and likelihood ratio tests under nonstandard
  conditions. *Journal of the American Statistical Association* 82(398),
  605–610. The half-`χ²` test of a variance on its boundary.
- **Hamilton, J. D. (1989).** A new approach to the economic analysis of
  nonstationary time series and the business cycle. *Econometrica*
  57(2), 357–384. The Markov-switching autoregression, its filter, and
  the GNP series it was first fitted to — [`fit_ms_autoregression`](@ref).
- **Kim, C.-J. (1994).** Dynamic linear models with Markov-switching.
  *Journal of Econometrics* 60(1–2), 1–22; and **Kim, C.-J. & Nelson,
  C. R. (1999).** *State-Space Models with Regime Switching.* MIT Press.
  The smoother behind [`regime_probabilities`](@ref).
- **Diebold, F. X., Lee, J.-H. & Weinbach, G. C. (1994).** Regime
  switching with time-varying transition probabilities. In C.
  Hargreaves (ed.), *Nonstationary Time Series Analysis and
  Cointegration*, Oxford University Press, 283–302. The `exog_tvtp`
  option.
- **Tong, H. & Lim, K. S. (1980).** Threshold autoregression, limit
  cycles and cyclical data. *Journal of the Royal Statistical Society B*
  42(3), 245–292; and **Tong, H. (1990).** *Non-linear Time Series: A
  Dynamical System Approach.* Oxford University Press. The threshold
  autoregression, and the lynx series it is still demonstrated on —
  [`fit_setar`](@ref).
- **Davies, R. B. (1987).** Hypothesis testing when a nuisance parameter
  is present only under the alternative. *Biometrika* 74(1), 33–43. Why a
  threshold test needs the bootstrap.
- **Hansen, B. E. (1999).** Testing for linearity. *Journal of Economic
  Surveys* 13(5), 551–576. — [`setar_test`](@ref). And **Hansen, B. E.
  (2000).** Sample splitting and threshold estimation. *Econometrica*
  68(3), 575–603: why the threshold has no ordinary standard error, and
  the confidence set that is not yet built.
- **Teräsvirta, T. (1994).** Specification, estimation, and evaluation
  of smooth transition autoregressive models. *Journal of the American
  Statistical Association* 89(425), 208–218. — [`fit_lstar`](@ref).

## Primary sources for breaks, change points and monitoring

Hamilton (§14.3, §15.5) and Montgomery (§2.6.3, §6.7) are the textbook
treatments; the tests and algorithms come from these:

- **Chow, G. C. (1960).** Tests of equality between sets of coefficients
  in two linear regressions. *Econometrica* 28(3), 591–605. The test with
  a known break date.
- **Andrews, D. W. K. (1993).** Tests for parameter instability and
  structural change with unknown change point. *Econometrica* 61(4),
  821–856. The supremum statistics and their distribution — [`fstats`](@ref);
  see Appendix B on why no p-value is given. And **Andrews, D. W. K. &
  Ploberger, W. (1994).** Optimal tests when a nuisance parameter is
  present only under the alternative. *Econometrica* 62(6), 1383–1414:
  the average and exponential forms.
- **Hansen, B. E. (1997).** Approximate asymptotic p values for
  structural-change tests. *Journal of Business & Economic Statistics*
  15(1), 60–67. The approximation `strucchange` uses.
- **Bai, J. (1997).** Estimation of a change point in multiple regression
  models. *Review of Economics and Statistics* 79(4), 551–563. The
  break-date distribution behind [`break_confint`](@ref).
- **Bai, J. & Perron, P. (2003).** Computation and analysis of multiple
  structural change models. *Journal of Applied Econometrics* 18(1),
  1–22. The dynamic programme — [`breakpoints`](@ref).
- **Brown, R. L., Durbin, J. & Evans, J. M. (1975).** Techniques for
  testing the constancy of regression relationships over time. *Journal
  of the Royal Statistical Society B* 37(2), 149–192. The recursive CUSUM;
  and **Ploberger, W. & Krämer, W. (1992).** The CUSUM test with OLS
  residuals. *Econometrica* 60(2), 271–285 — [`efp`](@ref).
- **Zeileis, A., Leisch, F., Hornik, K. & Kleiber, C. (2002).**
  strucchange: an R package for testing for structural change in linear
  regression models. *Journal of Statistical Software* 7(2), 1–38. The
  fluctuation-process framework, and the reference implementation.
- **Killick, R., Fearnhead, P. & Eckley, I. A. (2012).** Optimal detection
  of changepoints with a linear computational cost. *Journal of the
  American Statistical Association* 107(500), 1590–1598. PELT —
  [`changepoints`](@ref).
- **Box, G. E. P. & Tiao, G. C. (1975).** Intervention analysis with
  applications to economic and environmental problems. *Journal of the
  American Statistical Association* 70(349), 70–79.
- **Lam, P.-S. (1990).** The Hamilton model with a general autoregressive
  component. *Journal of Monetary Economics* 26(3), 409–432. The shifting
  GNP trend of Hamilton's Figure 15.4.
- **Page, E. S. (1954).** Continuous inspection schemes. *Biometrika*
  41(1/2), 100–115. The CUSUM as a monitor.

## Primary sources for intermittent demand and the ML reduction

None of the six textbooks covers intermittent demand; these are the
sources:

- **Croston, J. D. (1972).** Forecasting and stock control for
  intermittent demands. *Operational Research Quarterly* 23(3), 289–303.
  — [`croston`](@ref).
- **Syntetos, A. A. & Boylan, J. E. (2005).** The accuracy of intermittent
  demand estimates. *International Journal of Forecasting* 21(2),
  303–314. The bias correction (SBA).
- **Syntetos, A. A., Boylan, J. E. & Croston, J. D. (2005).** On the
  categorization of demand patterns. *Journal of the Operational Research
  Society* 56(5), 495–503. — [`adi_cv2`](@ref).
- **Teunter, R. H., Syntetos, A. A. & Babai, M. Z. (2011).** Intermittent
  demand: linking forecasting to inventory obsolescence. *European Journal
  of Operational Research* 214(3), 606–615. — [`tsb`](@ref).
- **Nikolopoulos, K., Syntetos, A. A., Boylan, J. E., Petropoulos, F. &
  Assimakopoulos, V. (2011).** An aggregate–disaggregate intermittent
  demand approach (ADIDA) to forecasting. *Journal of the Operational
  Research Society* 62(3), 544–554. — [`adida`](@ref).
- **Petropoulos, F. & Kourentzes, N. (2015).** Forecast combinations for
  intermittent demand. *Journal of the Operational Research Society*
  66(6), 914–924. — [`imapa`](@ref).
- **Shenstone, L. & Hyndman, R. J. (2005).** Stochastic models underlying
  Croston's method for intermittent demand forecasting. *Journal of
  Forecasting* 24(6), 389–402. Why there are no intervals.
- **Hyndman, R. J. & Athanasopoulos, G. (2021).** *Forecasting:
  Principles and Practice* (3rd ed.). OTexts. The published book, which
  covers Croston's method; the `fpp` lecture notes in the reference set do
  not.
- **Taieb, S. B., Bontempi, G., Atiya, A. F. & Sorjamaa, A. (2012).** A
  review and comparison of strategies for multi-step ahead time series
  forecasting based on the NN5 forecasting competition. *Expert Systems
  with Applications* 39(8), 7067–7083. The recursive, direct, DirRec and
  multi-output strategies — [`lag_regression`](@ref).
