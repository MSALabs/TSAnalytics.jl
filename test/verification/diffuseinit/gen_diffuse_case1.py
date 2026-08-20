import numpy as np
import json
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter

OUT = "c:/Users/user/Documents/MSA-Consulting/XKDR/TSA/TSAnalytics.jl/test/verification/diffuseinit"

np.random.seed(7)
n = 50
true_beta = 1.5
x = np.random.randn(n) * 2
level = np.zeros(n)
level_noise = np.random.randn(n) * np.sqrt(0.01)
level[0] = level_noise[0]
for t in range(1, n):
    level[t] = level[t-1] + level_noise[t]
obs_noise = np.random.randn(n) * np.sqrt(0.09)
y = level + true_beta * x + obs_noise
print("y.shape:", y.shape, "x.shape:", x.shape)

kf = KalmanFilter(k_endog=1, k_states=2, k_posdef=1, nobs=n)
design = np.zeros((1, 2, n))
for t in range(n):
    design[0, 0, t] = 1.0
    design[0, 1, t] = x[t]
kf['design'] = design
kf['transition'] = np.array([[1.0, 0.0], [0.0, 1.0]])
kf['selection'] = np.array([[1.0], [0.0]])
kf['state_cov'] = np.array([[0.01]])
kf['obs_cov'] = np.array([[0.09]])
kf.initialize_diffuse()
kf.bind(y)
print("y.shape after bind:", y.shape)
res = kf.filter()

print("loglik (full):", res.llf)
print("nobs_diffuse:", res.nobs_diffuse)
llf_obs = res.llf_obs
print("sum excl first nobs_diffuse:", np.sum(llf_obs[res.nobs_diffuse:]))
print("forecasts shape:", res.forecasts.shape)
forecasts_flat = res.forecasts.flatten()
print("forecasts_flat[:5]:", forecasts_flat[:5])
fec_flat = res.forecasts_error_cov.flatten()
print("fec_flat[:5]:", fec_flat[:5])

y_list = [float(v) for v in y]
x_list = [float(v) for v in x]
print("y_list[:3]:", y_list[:3])

results = {
    "x": x_list, "y": y_list, "true_beta": true_beta,
    "loglik_full": float(res.llf),
    "nobs_diffuse": int(res.nobs_diffuse),
    "loglik_excl_diffuse": float(np.sum(llf_obs[res.nobs_diffuse:])),
    "forecasts": [float(v) for v in forecasts_flat],
    "forecast_error_cov": [float(v) for v in fec_flat],
}
with open(OUT + "/diffuse_case1.json", "w") as f:
    json.dump(results, f, indent=2)

with open(OUT + "/diffuse_case1.csv", "w") as f:
    f.write("x,y\n")
    for xi, yi in zip(x_list, y_list):
        f.write(f"{xi},{yi}\n")
print("saved")
