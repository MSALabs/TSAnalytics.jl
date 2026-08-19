"""
Regenerates the first 20 (true_order, seed) cases from the
Stage 6.8 handoff's own generating code (handoff/stage-6.8-autoarima-handoff.md
section 6), saves each series as a CSV, and fits each with real
pmdarima.auto_arima -- recording ground truth (selected_order, aic) for the
Julia bulk test to compare against.

Capped at 20 cases per explicit instruction (auto_arima fitting is slow --
~2-12s/case per the handoff's own timing note); extend N_CASES later.
"""
import json
import os

import numpy as np
from pmdarima import auto_arima

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

N_CASES = 20

orders = [(p, d, q) for p in range(3) for d in range(3) for q in range(3) if not (p == 0 and q == 0)]
seeds = [1, 2, 3]

cases = [(p, d, q, seed) for (p, d, q) in orders[:12] for seed in seeds][:N_CASES]

results = []
for i, (p, d, q, seed) in enumerate(cases):
    np.random.seed(seed)
    n = 150
    e = np.random.randn(n + d)
    w = np.zeros(n + d)
    for t in range(max(p, q) + 1, n + d):
        ar_part = sum(0.3 * w[t - i2] for i2 in range(1, p + 1)) if p else 0
        ma_part = sum(0.2 * e[t - i2] for i2 in range(1, q + 1)) if q else 0
        w[t] = ar_part + ma_part + e[t]
    y = w.copy()
    for _ in range(d):
        y = np.cumsum(y)
    y = y[-n:]

    case_name = f"case{i:02d}_p{p}d{d}q{q}_seed{seed}"
    np.savetxt(os.path.join(DATA_DIR, case_name + ".csv"), y, delimiter=",",
               header="y", comments="")

    model = auto_arima(y, seasonal=False, stepwise=True, suppress_warnings=True,
                        max_p=3, max_q=3, max_d=2)
    result = {
        "case": case_name,
        "true_order": [p, d, q],
        "seed": seed,
        "selected_order": list(model.order),
        "aic": float(model.aic()),
    }
    results.append(result)
    print(result)

with open(os.path.join(DIR, "python_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nWrote {len(results)} cases to python_results.json")
