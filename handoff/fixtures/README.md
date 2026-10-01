# Reference fixtures — generated 2026, R 4.3.3 / statsmodels 0.15.0

All values in the two handoffs were produced by the scripts here.
Regenerate with `Rscript <name>.R` or the inline Python in the handoffs.

| File | Purpose |
|---|---|
| `ets_y.csv` | 120 obs, quarterly, simulated from a true ETS(A,A,A): alpha .40 beta .10 gamma .30 sigma 2.0, seed 20260101. Built so fitted parameters land in the INTERIOR — a first attempt with a deterministic trend pinned every parameter to its bound |
| `ets_ref.json` | statsmodels output for all six linear models: params, loglik, aic, aicc, bic, sse, fitted[1:5], forecast[1:4] |
| `hw.R` | base R HoltWinters at FIXED alpha/beta/gamma — the equivalence anchor. No CRAN needed |
| `ccf_x.csv`, `ccf_y.csv` | 145 obs, y lags x by 3. R peak at lag -3 = +0.9636929430 |
| `ccf.R` | R two-sided ccf. statsmodels returns NON-NEGATIVE lags only, so ccf(x,y) there does not contain the peak |
| `ax_y.csv`, `ax_x.csv`, `ax_xf.csv` | ARIMAX: 140 fit + 10 future regressor values. AR(1) phi=0.6, true beta=2.0, seed 31 |
| `ax.R` | R arima(xreg=) fit and predict(newxreg=). Agrees with statsmodels to 5-6 dp |
| `burg.R` | pacf method comparison. FINDING: R's pacf() IGNORES its method argument — all four identical. Burg lives in ar.burg(). statsmodels is the reference for Burg PACF |

CRAN was unreachable, so `forecast::ets()` and `forecast::nsdiffs()`
could not be run. ETS likelihood quantities are single-verified against
Python; the HoltWinters anchor uses base R and needs no CRAN.
