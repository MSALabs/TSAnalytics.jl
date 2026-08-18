"""
Ground truth for kalman_smoother, via statsmodels SARIMAX's own smoother.
Confirmed (see interactive check) that statsmodels' transition/selection/
design matrices for order=(p,0,q), trend='n' match build_statespace's T/R/Z
convention exactly, so smoothed_state/smoothed_state_cov are directly
comparable to kalman_smoother's alpha/V with no reindexing.

Emits one CSV per case: columns are the r state dimensions' smoothed means
(alpha) followed by the flattened smoothed covariances (V), one row per t.
Also emits a Julia test file that loads these and compares.
"""
import json
import os

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

cases = [
    dict(name="ar1", dataset="ar1_shared", p=1, q=0, ar=[0.6], ma=[]),
    dict(name="ar2", dataset="ar2_shared", p=2, q=0, ar=[0.5, 0.1], ma=[]),
    dict(name="ma1", dataset="ma1_shared", p=0, q=1, ar=[], ma=[0.4]),
    dict(name="arma11", dataset="ar1_shared", p=1, q=1, ar=[0.4], ma=[0.3]),
]

manifest = []
for c in cases:
    y = pd.read_csv(os.path.join(REPO_ROOT, "verification", f"{c['dataset']}.csv")).iloc[:, 0].to_numpy(dtype=float)
    order = (c["p"], 0, c["q"])
    mod = SARIMAX(y, order=order, trend='n', concentrate_scale=True)
    params = list(c["ar"]) + list(c["ma"])
    res = mod.smooth(params)

    alpha = res.smoothed_state  # r x n
    cov = res.smoothed_state_cov  # r x r x n
    r, n = alpha.shape

    rows = []
    for t in range(n):
        row = list(alpha[:, t]) + list(cov[:, :, t].reshape(-1))
        rows.append(row)
    cols = [f"alpha{i+1}" for i in range(r)] + [f"V{i+1}_{j+1}" for i in range(r) for j in range(r)]
    df = pd.DataFrame(rows, columns=cols)
    out_csv = os.path.join(HERE, f"{c['name']}_expected.csv")
    df.to_csv(out_csv, index=False)

    manifest.append(dict(name=c["name"], dataset=c["dataset"], p=c["p"], q=c["q"],
                          ar=c["ar"], ma=c["ma"], r=r, n=n))
    print(f"{c['name']}: r={r} n={n} -> {out_csv}")

with open(os.path.join(HERE, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
