"""
Regenerates the Stage 7.2 handoff's own 48-case bulk grid (section 5:
8 true parameter combinations x 3 seeds x 2 model types [:gjr, :egarch],
n=1500) -- fresh series generated from the grid description, saved as
CSVs this project's test suite actually loads, fit with real
arch.arch_model for ground truth (extends Stage 7.1's exact methodology,
same as gen_and_fit_python.py there).
"""
import json
import os

import numpy as np
from arch import arch_model

DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

true_params_gjr = [
    (0.05, 0.05, 0.10, 0.85), (0.05, 0.10, 0.05, 0.80), (0.05, 0.05, 0.15, 0.75),
    (0.10, 0.05, 0.10, 0.80), (0.10, 0.10, 0.05, 0.75), (0.10, 0.05, 0.15, 0.70),
    (0.02, 0.05, 0.08, 0.85), (0.02, 0.08, 0.10, 0.80),
]
true_params_egarch = [
    (-0.05, 0.15, -0.05, 0.90), (-0.05, 0.20, -0.10, 0.85), (-0.05, 0.15, -0.08, 0.92),
    (-0.10, 0.15, -0.05, 0.88), (-0.10, 0.20, -0.10, 0.83), (-0.10, 0.18, -0.07, 0.90),
    (-0.03, 0.12, -0.04, 0.93), (-0.03, 0.18, -0.06, 0.87),
]
seeds = [1, 2, 3]
n = 1500

results = []
case_i = 0


def gen_gjr(omega, alpha, gamma, beta, seed):
    rng = np.random.RandomState(seed)
    h = np.zeros(n)
    e = np.zeros(n)
    h[0] = omega / (1 - alpha - 0.5 * gamma - beta)
    innov = rng.randn(n)
    e[0] = np.sqrt(h[0]) * innov[0]
    for t in range(1, n):
        asym = gamma * e[t - 1] ** 2 if e[t - 1] < 0 else 0.0
        h[t] = omega + alpha * e[t - 1] ** 2 + asym + beta * h[t - 1]
        e[t] = np.sqrt(h[t]) * innov[t]
    return e


def gen_egarch(omega, alpha, gamma, beta, seed):
    rng = np.random.RandomState(seed)
    sqrt2opi = np.sqrt(2 / np.pi)
    lnh = np.zeros(n)
    e = np.zeros(n)
    z = np.zeros(n)
    lnh[0] = omega / (1 - beta)
    innov = rng.randn(n)
    e[0] = np.exp(lnh[0] / 2) * innov[0]
    z[0] = innov[0]
    for t in range(1, n):
        lnh[t] = omega + alpha * (abs(z[t - 1]) - sqrt2opi) + gamma * z[t - 1] + beta * lnh[t - 1]
        e[t] = np.exp(lnh[t] / 2) * innov[t]
        z[t] = e[t] / np.exp(lnh[t] / 2)
    return e


for (omega, alpha, gamma, beta) in true_params_gjr:
    for seed in seeds:
        e = gen_gjr(omega, alpha, gamma, beta, seed)
        case_name = f"case{case_i:02d}_gjr_o{omega}_a{alpha}_g{gamma}_b{beta}_seed{seed}"
        np.savetxt(os.path.join(DATA_DIR, case_name + ".csv"), e, delimiter=",", header="e", comments="")
        am = arch_model(e, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal")
        res = am.fit(disp="off", cov_type="robust")
        results.append({
            "case": case_name, "model": "gjr", "true": [omega, alpha, gamma, beta], "seed": seed,
            "fitted": [float(res.params["omega"]), float(res.params["alpha[1]"]),
                       float(res.params["gamma[1]"]), float(res.params["beta[1]"])],
            "loglik": float(res.loglikelihood), "converged": bool(res.convergence_flag == 0),
        })
        print(results[-1])
        case_i += 1

for (omega, alpha, gamma, beta) in true_params_egarch:
    for seed in seeds:
        e = gen_egarch(omega, alpha, gamma, beta, seed)
        case_name = f"case{case_i:02d}_egarch_o{omega}_a{alpha}_g{gamma}_b{beta}_seed{seed}"
        np.savetxt(os.path.join(DATA_DIR, case_name + ".csv"), e, delimiter=",", header="e", comments="")
        am = arch_model(e, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal")
        res = am.fit(disp="off", cov_type="robust")
        results.append({
            "case": case_name, "model": "egarch", "true": [omega, alpha, gamma, beta], "seed": seed,
            "fitted": [float(res.params["omega"]), float(res.params["alpha[1]"]),
                       float(res.params["gamma[1]"]), float(res.params["beta[1]"])],
            "loglik": float(res.loglikelihood), "converged": bool(res.convergence_flag == 0),
        })
        print(results[-1])
        case_i += 1

with open(os.path.join(DIR, "python_results.json"), "w") as f:
    json.dump(results, f, indent=2)

n_conv = sum(1 for r in results if r["converged"])
print(f"\nWrote {len(results)} cases to python_results.json ({n_conv} converged)")
