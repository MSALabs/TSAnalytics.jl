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
