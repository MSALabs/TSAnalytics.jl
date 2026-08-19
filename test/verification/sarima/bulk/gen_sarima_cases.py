"""
Generate synthetic SARIMA(p,d,q)(P,D,Q)_s series across a sweep of
orders, save as CSVs, for dual R/Python fitting -- Stage 6.7 bulk
verification.
"""
import numpy as np
import os

OUTDIR = os.path.dirname(os.path.abspath(__file__)) + "/data"
os.makedirs(OUTDIR, exist_ok=True)

# (name, p, d, q, P, D, Q, s, phi, theta, Phi, Theta, n, seed)
CASES = [
    ("p_only",        1, 0, 0, 0, 0, 0, 4, [0.5],  [],     [],    [],     150, 1001),
    ("q_only",        0, 0, 1, 0, 0, 0, 4, [],     [0.4],  [],    [],     150, 1002),
    ("seasar_only",   0, 0, 0, 1, 0, 0, 4, [],     [],     [0.5], [],     150, 1003),
    ("seasma_only",   0, 0, 0, 0, 0, 1, 4, [],     [],     [],    [-0.4], 150, 1004),
    ("ar_seasar",     1, 0, 0, 1, 0, 0, 4, [0.4],  [],     [0.4], [],     150, 1005),
    ("ma_seasma",     0, 0, 1, 0, 0, 1, 4, [],     [0.35], [],    [0.3],  150, 1006),
    ("full_block",    1, 0, 1, 1, 0, 1, 4, [0.4],  [0.3],  [0.5], [-0.3], 200, 1007),
    ("ar_seasar_d1D1", 1, 1, 0, 1, 1, 0, 4, [0.3], [],     [0.4], [],     150, 1008),
    ("ma_seasma_d1",  0, 1, 1, 0, 0, 1, 4, [],     [0.4],  [],    [0.35], 150, 1009),
    ("s12_ar_seasar", 1, 0, 0, 1, 0, 0, 12, [0.4], [],     [0.4], [],     120, 1010),
    ("s12_full",      1, 0, 1, 1, 1, 0, 12, [0.35],[0.25], [0.3], [],     150, 1011),
    ("p2_seasar1",    2, 0, 0, 1, 0, 0, 4,  [0.3, -0.15], [], [0.4], [], 150, 1012),
]

manifest = []
for name, p, d, q, P, D, Q, s, phi, theta, Phi, Theta, n, seed in CASES:
    rng = np.random.default_rng(seed)
    burn = 60
    e = rng.standard_normal(n + burn)
    w = np.zeros(n + burn)
    for t in range(len(w)):
        val = e[t]
        for i, c in enumerate(phi, start=1):
            if t - i >= 0:
                val += c * w[t - i]
        for j, c in enumerate(theta, start=1):
            if t - j >= 0:
                val += c * e[t - j]
        for i, c in enumerate(Phi, start=1):
            if t - i * s >= 0:
                val += c * w[t - i * s]
        for j, c in enumerate(Theta, start=1):
            if t - j * s >= 0:
                val += c * e[t - j * s]
        w[t] = val
    w = w[burn:]

    # Inverse of seasonal-then-regular differencing: seasonal cumulative
    # sum (lag s) D times, then plain cumulative sum d times -- exact
    # inverse of diff(diff(y, s; differences=D), 1; differences=d).
    y = w.copy()
    for _ in range(D):
        y2 = np.zeros(len(y))
        for i in range(len(y)):
            y2[i] = y[i] + (y2[i - s] if i - s >= 0 else 0.0)
        y = y2
    for _ in range(d):
        y = np.cumsum(y)

    path = f"{OUTDIR}/{name}.csv"
    with open(path, "w") as f:
        f.write("x\n")
        for v in y:
            f.write(f"{v:.15e}\n")
    manifest.append(dict(name=name, p=p, d=d, q=q, P=P, D=D, Q=Q, s=s, n=len(y)))

import json
with open(os.path.join(os.path.dirname(OUTDIR), "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Generated {len(manifest)} cases in {OUTDIR}")
