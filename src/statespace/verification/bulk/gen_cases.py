"""
Stage 1 of the bulk verification pipeline.

Builds:
  - verification/bulk/data/*.csv       one column "y", real + synthetic datasets
  - verification/bulk/cases.json        every (dataset, order, params) case to test,
                                         pre-filtered to stationary/invertible params,
                                         with the Python-side loglik already attached
  - verification/bulk/run_r.R           R script that re-evaluates every case in cases.json

Run this first, then run_r.R (via Rscript), then merge_and_emit.py.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.datasets as smd
from statsmodels.tsa.statespace.sarimax import SARIMAX

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def save_series(name, y):
    y = np.asarray(y, dtype=float)
    pd.DataFrame({"y": y}).to_csv(os.path.join(DATA_DIR, f"{name}.csv"), index=False)
    return name, len(y)


# ---------------------------------------------------------------------------
# Datasets: the 3 existing synthetic series (already used by the original
# 3-case gate) + 5 real textbook/canonical series pulled from statsmodels'
# bundled datasets.
# ---------------------------------------------------------------------------
datasets = {}

ar1 = pd.read_csv(os.path.join(HERE, "..", "ar1_shared.csv"))
datasets["ar1_shared"] = ar1.iloc[:, 0].to_numpy(dtype=float)

ar2 = pd.read_csv(os.path.join(HERE, "..", "ar2_shared.csv"))
datasets["ar2_shared"] = ar2.iloc[:, 0].to_numpy(dtype=float)

ma1 = pd.read_csv(os.path.join(HERE, "..", "ma1_shared.csv"))
datasets["ma1_shared"] = ma1.iloc[:, 0].to_numpy(dtype=float)

nile = smd.nile.load_pandas().data
datasets["nile"] = nile["volume"].to_numpy(dtype=float)  # n=100, Durbin & Koopman's canonical example

sun = smd.sunspots.load_pandas().data
datasets["sunspots"] = sun["SUNACTIVITY"].to_numpy(dtype=float)  # n=309, classic Box-Jenkins-style series

macro = smd.macrodata.load_pandas().data
gdp = macro["realgdp"].to_numpy(dtype=float)
gdp_growth = 100.0 * np.diff(np.log(gdp))  # n=202, quarterly -> good for seasonal s=4
datasets["macro_gdp_growth"] = gdp_growth

elnino = smd.elnino.load_pandas().data
months = [c for c in elnino.columns if c != "YEAR"]
monthly = elnino[months].to_numpy(dtype=float).reshape(-1)  # flatten years x 12 months
monthly = monthly[-300:]  # last 300 obs (25 years) -> seasonal s=12, keep runtime sane
datasets["elnino_monthly"] = monthly

co2 = smd.co2.load_pandas().data
co2y = co2["co2"].dropna().to_numpy(dtype=float)
co2y = np.diff(co2y[:400])  # first ~400 weekly obs, differenced once for sane scale
datasets["co2_diff"] = co2y

for name, y in datasets.items():
    save_series(name, y)

print("Datasets:", {k: len(v) for k, v in datasets.items()})


# ---------------------------------------------------------------------------
# Parameter grid, stationarity/invertibility filtered via root check.
# Convention (matches build_statespace / R arima / statsmodels SARIMAX with
# trend='n'): y_t = phi_1 y_{t-1} + ... + e_t + theta_1 e_{t-1} + ...
# Stationary/invertible iff roots of 1 - phi_1 z - ... - phi_p z^p = 0 (resp.
# 1 + theta_1 z + ...) lie strictly outside the unit circle.
# ---------------------------------------------------------------------------
def roots_outside_unit_circle(coefs, ar_side):
    if len(coefs) == 0:
        return True
    if ar_side:
        poly = [1.0] + [-c for c in coefs]  # 1 - phi1 z - phi2 z^2 - ...
    else:
        poly = [1.0] + list(coefs)  # 1 + theta1 z + theta2 z^2 + ...
    # numpy.roots wants highest power first; poly above is lowest power first
    r = np.roots(poly[::-1])
    return bool(np.all(np.abs(r) > 1.0 + 1e-9))


AR_SETS = {
    1: [[0.3], [0.5], [0.7], [-0.3], [-0.5]],
    2: [[0.5, 0.1], [0.3, 0.3], [0.6, -0.2], [-0.4, 0.3], [0.2, 0.5]],
    3: [[0.5, -0.2, 0.1], [0.3, 0.2, -0.3], [0.6, -0.1, -0.2]],
}
MA_SETS = {
    1: [[0.2], [0.4], [0.6], [-0.3], [-0.5]],
    2: [[0.4, 0.2], [0.3, -0.3], [-0.4, 0.2], [0.5, -0.1]],
    3: [[0.4, 0.2, -0.1], [0.3, -0.2, 0.2], [-0.3, 0.2, 0.1]],
}
# (p, q) -> which AR_SETS[p]/MA_SETS[q] index lists to use (paired up to the
# shorter list's length so e.g. ARMA(2,1) gets 3 combos not 5x3=15)
ARMA_ORDERS = [(1, 1), (2, 1), (1, 2), (2, 2)]

SEASONAL_STRUCTS = [
    # (p, q, P, Q)
    (1, 0, 1, 0),
    (0, 1, 0, 1),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (1, 1, 1, 1),
    (2, 0, 1, 0),
]
SEASONAL_COEF_SETS = [
    {"ar": [0.4], "ma": [0.3], "sar": [0.5], "sma": [0.4]},
    {"ar": [-0.3], "ma": [-0.2], "sar": [-0.4], "sma": [0.3]},
]
SEASONAL_DATASETS = [("macro_gdp_growth", 4), ("elnino_monthly", 12)]

cases = []
cid = 0


def add_case(dataset, p, q, ar, ma, P=0, Q=0, sar=None, sma=None, s=0):
    global cid
    sar = sar or []
    sma = sma or []
    if not roots_outside_unit_circle(ar, ar_side=True):
        return
    if not roots_outside_unit_circle(ma, ar_side=False):
        return
    if sar and not roots_outside_unit_circle(sar, ar_side=True):
        return
    if sma and not roots_outside_unit_circle(sma, ar_side=False):
        return
    cid += 1
    cases.append(dict(
        id=cid, dataset=dataset, p=p, q=q, P=P, Q=Q, s=s,
        ar=ar, ma=ma, sar=sar, sma=sma,
    ))


for dname in datasets:
    for order, sets in AR_SETS.items():
        for coefs in sets:
            add_case(dname, order, 0, coefs, [])
    for order, sets in MA_SETS.items():
        for coefs in sets:
            add_case(dname, 0, order, [], coefs)
    for p, q in ARMA_ORDERS:
        ar_sets = AR_SETS[p]
        ma_sets = MA_SETS[q]
        n = min(len(ar_sets), len(ma_sets))
        for i in range(n):
            add_case(dname, p, q, ar_sets[i], ma_sets[i])

for dname, s in SEASONAL_DATASETS:
    for (p, q, P, Q) in SEASONAL_STRUCTS:
        for coefset in SEASONAL_COEF_SETS:
            ar = coefset["ar"][:p] if p else []
            ma = coefset["ma"][:q] if q else []
            sar = coefset["sar"][:P] if P else []
            sma = coefset["sma"][:Q] if Q else []
            add_case(dname, p, q, ar, ma, P=P, Q=Q, sar=sar, sma=sma, s=s)

print(f"Total candidate cases (post stationarity/invertibility filter): {len(cases)}")


# ---------------------------------------------------------------------------
# Python-side loglik for every case (statsmodels SARIMAX, concentrate_scale)
# ---------------------------------------------------------------------------
def py_loglik(y, p, q, ar, ma, P, Q, sar, sma, s):
    order = (p, 0, q)
    seasonal_order = (P, 0, Q, s) if s else (0, 0, 0, 0)
    mod = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                   trend='n', concentrate_scale=True)
    params = list(ar) + list(ma) + list(sar) + list(sma)
    return float(mod.loglike(params))


bad = []
for c in cases:
    y = datasets[c["dataset"]]
    try:
        ll = py_loglik(y, c["p"], c["q"], c["ar"], c["ma"], c["P"], c["Q"], c["sar"], c["sma"], c["s"])
    except Exception as e:
        ll = None
        bad.append((c["id"], str(e)))
    c["py_loglik"] = ll

if bad:
    print(f"{len(bad)} cases raised in statsmodels (dropped):")
    for i, msg in bad[:10]:
        print(" ", i, msg)
cases = [c for c in cases if c["py_loglik"] is not None and np.isfinite(c["py_loglik"])]
print(f"Cases with a finite Python loglik: {len(cases)}")

with open(os.path.join(HERE, "cases.json"), "w") as f:
    json.dump(cases, f, indent=1)

print("Wrote cases.json and datasets. Next: run run_r.R via Rscript.")


# ---------------------------------------------------------------------------
# Emit the R script that re-evaluates every case (single R session, avoids
# per-case subprocess startup cost).
# ---------------------------------------------------------------------------
r_script = r'''
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
here <- dirname(normalizePath(file_arg))
cases <- fromJSON(file.path(here, "cases.json"), simplifyDataFrame = FALSE)

data_cache <- new.env()
get_data <- function(name) {
  if (!exists(name, envir = data_cache, inherits = FALSE)) {
    y <- read.csv(file.path(here, "data", paste0(name, ".csv")))$y
    assign(name, y, envir = data_cache)
  }
  get(name, envir = data_cache, inherits = FALSE)
}

results <- data.frame(id = integer(0), r_loglik = double(0), r_error = character(0))

for (c in cases) {
  y <- get_data(c$dataset)
  ar <- unlist(c$ar); ma <- unlist(c$ma)
  sar <- unlist(c$sar); sma <- unlist(c$sma)
  fixed <- c(ar, ma, sar, sma)

  seasonal <- if (c$P > 0 || c$Q > 0) {
    list(order = c(c$P, 0, c$Q), period = c$s)
  } else {
    list(order = c(0, 0, 0), period = 0)
  }

  ll <- tryCatch({
    fit <- arima(y, order = c(c$p, 0, c$q), seasonal = seasonal,
                 include.mean = FALSE, fixed = fixed,
                 transform.pars = FALSE, optim.control = list(maxit = 0))
    as.numeric(fit$loglik)
  }, error = function(e) NA_real_)

  results <- rbind(results, data.frame(id = c$id, r_loglik = ll, r_error = ifelse(is.na(ll), "error", "")))
}

write.csv(results, file.path(here, "r_results.csv"), row.names = FALSE)
cat("Wrote r_results.csv with", nrow(results), "rows\n")
'''

with open(os.path.join(HERE, "run_r.R"), "w") as f:
    f.write(r_script)
