"""
Regenerates the Stage 7.1 handoff's own 24-case bulk grid (section 6:
8 true (omega,alpha,beta) combinations x 3 seeds, n=800) -- the handoff's
own garch_100cases_python.json has aggregate results only (no underlying
series), so this generates fresh series, saves them as CSVs this
project's test suite actually loads, and fits each with real
arch.arch_model for fresh, directly-reproducible ground truth.
"""
import json
import os

import numpy as np
from arch import arch_model

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

true_params = [
    (0.05, 0.05, 0.90), (0.05, 0.10, 0.85), (0.05, 0.15, 0.80),
    (0.10, 0.05, 0.85), (0.10, 0.10, 0.80), (0.10, 0.15, 0.75),
    (0.02, 0.08, 0.88), (0.02, 0.12, 0.82),
]
seeds = [1, 2, 3]
n = 800

results = []
case_i = 0
for (omega, alpha, beta) in true_params:
    for seed in seeds:
        rng = np.random.RandomState(seed)
        h = np.zeros(n)
        e = np.zeros(n)
        h[0] = omega / (1 - alpha - beta)
        innov = rng.randn(n)
        e[0] = np.sqrt(h[0]) * innov[0]
        for t in range(1, n):
            h[t] = omega + alpha * e[t - 1] ** 2 + beta * h[t - 1]
            e[t] = np.sqrt(h[t]) * innov[t]

        case_name = f"case{case_i:02d}_o{omega}_a{alpha}_b{beta}_seed{seed}"
        np.savetxt(os.path.join(DATA_DIR, case_name + ".csv"), e, delimiter=",",
                   header="e", comments="")

        am = arch_model(e, mean="Zero", vol="GARCH", p=1, o=0, q=1, dist="normal")
        res = am.fit(disp="off", cov_type="robust")
        result = {
            "case": case_name,
            "true": [omega, alpha, beta],
            "seed": seed,
            "fitted": [float(res.params["omega"]), float(res.params["alpha[1]"]),
                       float(res.params["beta[1]"])],
            "loglik": float(res.loglikelihood),
            "converged": bool(res.convergence_flag == 0),
        }
        results.append(result)
        print(result)
        case_i += 1

with open(os.path.join(DIR, "python_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nWrote {len(results)} cases to python_results.json")
