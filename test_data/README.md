# Datasets

These CSVs are the standard benchmark time series used throughout the
time series literature (and by R's `datasets`/`forecast` packages,
`statsmodels`, and other Julia packages) -- used here purely as
validation targets: implement a function, compute the reference
statistic in R or Python on one of these series, hardcode that number
into the Julia test suite, iterate until it matches.

Sourced from `StateSpaceModels.jl` (LAMPSPUC, MIT License), which
bundles the same series for the same purpose -- retrieved from
`https://github.com/LAMPSPUC/StateSpaceModels.jl` (`datasets/` directory).
See that repository's own dataset docstrings for original provenance;
summarized here for convenience.

| File | Series | Period | Original source |
|---|---|---|---|
| `nile.csv` | Annual flow of the Nile at Aswan (10^8 m^3) | 1871-1970 | Durbin & Koopman (2012), ch. 2 -- the canonical local-level-model example |
| `airpassengers.csv` | Monthly US airline passenger totals | 1949-1960 | Box & Jenkins (1976); the canonical multiplicative-seasonality example |
| `sunspot_year.csv` | Yearly sunspot numbers | 1700-1988 | Federal Reserve Bank of St Louis / H. Tong (1996) -- classic nonlinear/AR benchmark |
| `uschange.csv` | Quarterly % changes: consumption, income, production, savings, unemployment | 1960-2016 | Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* -- multivariate/VAR and regression-with-ARIMA-errors benchmark |
| `vehicle_fatalities.csv` | Annual road traffic fatalities, Norway and Finland | 1970-2003 | Commandeur & Koopman (2007), ch. 3 -- structural time series / local linear trend example |
| `internet.csv` | Users logged onto an internet server per minute | 100 observations | Durbin & Koopman (2012), ch. 9 -- ARMA/short-series example |
| `monthly.csv` | Synthetic, n=48, period-12 (even), linear trend + sinusoidal seasonality, additive-appropriate | n/a (synthetic) | Generated for Stage 3.1 dual-verification against real R/Python `decompose`/`seasonal_decompose`; see `handoff/stage-3.1-classical-decompose-handoff.md` |
| `period7.csv` | Synthetic, n=42, period-7 (odd), linear trend + sinusoidal seasonality, additive-appropriate | n/a (synthetic) | Same generation session as `monthly.csv` (continued RNG stream) -- Stage 3.1 |
| `mult_monthly.csv` | Synthetic, n=48, period-12, strictly positive, multiplicative trend/seasonality | n/a (synthetic) | Same generation session as `monthly.csv`/`period7.csv` -- Stage 3.1 |
| `monthly_outlier.csv` | `monthly.csv` with two outliers injected (`y[10]+=30`, `y[30]-=25`, 0-indexed) | n/a (synthetic) | Stage 3.2, to actually engage `stl_decompose`'s `robust=true`/`outer` reweighting |
| `hourly_mstl.csv` | Synthetic, n=500 hourly series, daily (24) + weekly (168) seasonality + quadratic trend | n/a (synthetic) | Stage 3.3, scaled-down version of `statsmodels.tsa.seasonal.MSTL`'s own docstring example |
| `ar2_arx.csv` | Synthetic, n=100, AR(2) (`phi=(0.5,0.2)`) | n/a (synthetic) | Stage 5.1, `arx` verification against real `statsmodels.tsa.ar_model.AutoReg` |
| `arx_exog_x1.csv` | Synthetic, n=100, exogenous regressor paired with `ar2_arx.csv` | n/a (synthetic) | Stage 5.1, `arx(...; exog=...)` verification |
| `arx_exog_y2.csv` | `ar2_arx.csv .+ 0.5 .* arx_exog_x1.csv` | n/a (synthetic) | Stage 5.1, dependent series for the `exog` case |

## Which dataset for which stage

- **Stage 6 (ARIMA/SARIMA):** `airpassengers.csv` (seasonal), `internet.csv` (short, non-seasonal ARMA)
- **Stage 2/6 (unit root tests, local level):** `nile.csv` -- the textbook example for exactly this
- **Stage 9/Stage 10 (multivariate, VAR):** `uschange.csv`
- **Stage 11 (Unobserved Components):** `vehicle_fatalities.csv`
- **Stage 12/general AR benchmarking:** `sunspot_year.csv`
- **Stage 3 (decomposition):** `monthly.csv`/`period7.csv` (additive, even/odd period), `mult_monthly.csv` (multiplicative), `monthly_outlier.csv` (STL `robust=true`), `hourly_mstl.csv` (MSTL, multi-seasonal) -- synthetic, dual-verified against real R and Python
- **Stage 5 (AR-X and later univariate models):** `ar2_arx.csv`, `arx_exog_x1.csv`, `arx_exog_y2.csv` -- synthetic, verified against real `statsmodels`

## Adding a new dataset

Keep the same convention: a plain CSV with a header row, added here, with
a corresponding path constant added to `src/datasets.jl` and a row added
to the table above documenting source and intended use.
