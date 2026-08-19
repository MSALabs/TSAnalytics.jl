"""
Generate synthetic ARIMA(p,d,q) series across a sweep of orders, save as
CSVs, for dual R/Python fitting -- Stage 6.6 bulk verification.
"""
import numpy as np
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__)) + "/data"
os.makedirs(OUTDIR, exist_ok=True)

# (name, p, d, q, true_ar, true_ma, n, seed)
CASES = [
    ("pq10_d0", 1, 0, 0, [0.5], [], 120, 101),
    ("pq10_d0_b", 1, 0, 0, [-0.4], [], 150, 102),
    ("pq01_d0", 0, 0, 1, [], [0.6], 120, 103),
    ("pq01_d0_b", 0, 0, 1, [], [-0.3], 150, 104),
    ("pq11_d0", 1, 0, 1, [0.5], [0.3], 150, 105),
    ("pq11_d0_b", 1, 0, 1, [0.6], [-0.4], 180, 106),
    ("pq20_d0", 2, 0, 0, [0.5, -0.2], [], 150, 107),
    ("pq02_d0", 0, 0, 2, [], [0.4, 0.2], 150, 108),
    ("pq21_d0", 2, 0, 1, [0.4, -0.1], [0.3], 180, 555),
    ("pq10_d1", 1, 1, 0, [0.5], [], 150, 201),
    ("pq01_d1", 0, 1, 1, [], [0.5], 150, 202),
    ("pq11_d1", 1, 1, 1, [0.4], [0.3], 150, 203),
    ("pq11_d1_b", 1, 1, 1, [-0.5], [0.2], 180, 204),
    ("pq20_d1", 2, 1, 0, [0.4, -0.2], [], 150, 205),
    ("pq02_d1", 0, 1, 2, [], [0.3, 0.2], 150, 206),
    ("pq00_d1", 0, 1, 0, [], [], 150, 207),
    ("pq10_d2", 1, 2, 0, [0.4], [], 150, 301),
    ("pq11_d2", 1, 2, 1, [0.4], [0.3], 180, 302),
    ("pq01_d2", 0, 2, 1, [], [0.4], 150, 303),
    ("pq00_d2", 0, 2, 0, [], [], 150, 304),
]

manifest = []
for name, p, d, q, ar_true, ma_true, n, seed in CASES:
    rng = np.random.default_rng(seed)
    # Generate the stationary ARMA(p,q) part directly via the recursion
    # (avoids needing statsmodels' ArmaProcess sign convention here --
    # we just need SOME reasonable series, R/Python will do the real fit)
    e = rng.standard_normal(n + 50)
    w = np.zeros(n + 50)
    for t in range(len(w)):
        val = e[t]
        for i, phi in enumerate(ar_true, start=1):
            if t - i >= 0:
                val += phi * w[t - i]
        for j, theta in enumerate(ma_true, start=1):
            if t - j >= 0:
                val += theta * e[t - j]
        w[t] = val
    w = w[50:]  # burn-in

    # integrate d times to build the ARIMA(p,d,q) series from the
    # stationary ARMA(p,q) innovation series
    y = w.copy()
    for _ in range(d):
        y = np.cumsum(y) + rng.standard_normal() * 0  # plain random-walk-style integration
    y = y + (0.0 if d == 0 else 0.0)

    path = f"{OUTDIR}/{name}.csv"
    with open(path, "w") as f:
        f.write("x\n")
        for v in y:
            f.write(f"{v:.15e}\n")
    manifest.append(dict(name=name, p=p, d=d, q=q, n=n))

import json
BULKDIR = os.path.dirname(OUTDIR)
with open(f"{BULKDIR}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Generated {len(manifest)} cases in {OUTDIR}")
