"""
Generates the Durbin-Watson (Stage 2.7) ground truth this project's own
tests actually load. The roadmap-restructure handoff that proposed this
stage referenced "the exact seed=5 script in this handoff's verification
transcript" -- no such transcript file or script actually exists in this
repo (checked directly), so this is a fresh, self-contained regeneration
rather than a transcription, verified against the real
statsmodels.stats.stattools.durbin_watson source (confirmed by reading it
directly: dw = sum(diff(resid)**2) / sum(resid**2), exactly as claimed).

Saves x,y (not just residuals) so the Julia test computes its own OLS
residuals via this project's own already-verified `_ols` (Stage 1.4)
rather than trusting residuals computed only in Python -- an extra,
independent check that the two languages' OLS fits agree, not just that
the DW formula does.
"""
import json
import os

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(5)
n = 100
x = np.random.randn(n)
innovations = np.random.randn(n)

# Case A: AR(1) errors, phi=0.6 -- strong positive autocorrelation expected
e_ar = np.zeros(n)
e_ar[0] = innovations[0]
for t in range(1, n):
    e_ar[t] = 0.6 * e_ar[t - 1] + innovations[t]
y_ar = 2 + 1.5 * x + e_ar

# Case B: white-noise errors -- no autocorrelation expected
np.random.seed(5)
x_wn = np.random.randn(n)  # same x-generating draw as case A, for comparability
e_wn = np.random.randn(n)  # fresh draw, no AR recursion
y_wn = 2 + 1.5 * x_wn + e_wn

results = {}
for label, xv, yv in [("ar1", x, y_ar), ("wn", x_wn, y_wn)]:
    X = sm.add_constant(xv)
    model = sm.OLS(yv, X).fit()
    resid = model.resid.to_numpy() if hasattr(model.resid, "to_numpy") else np.asarray(model.resid)
    dw = float(durbin_watson(resid))
    results[label] = {"dw": dw, "n": n}
    np.savetxt(os.path.join(DIR, f"durbinwatson_{label}.csv"),
               np.column_stack([xv, yv]), delimiter=",", header="x,y", comments="")
    print(label, "DW =", dw, "resid[:3] =", resid[:3])

with open(os.path.join(DIR, "python_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nWrote durbinwatson_ar1.csv, durbinwatson_wn.csv, python_results.json")
