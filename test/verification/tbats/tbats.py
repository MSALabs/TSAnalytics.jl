import numpy as np
from tbats import TBATS
y = np.loadtxt("test/verification/ets/ets_y.csv")
est = TBATS(seasonal_periods=[4], use_box_cox=False, use_trend=True,
            use_damped_trend=False, use_arma_errors=False, n_jobs=1)
m = est.fit(y)
p = m.params
print("alpha=%.8f beta=%.8f" % (p.alpha, p.beta))
print("gamma =", p.gamma_params)
print("sse=%.6f  variance=%.8f  aic=%.6f" % (np.sum(m.resid**2), np.var(m.resid, ddof=0), m.aic))
print("x0 =", np.round(p.x0, 8))
f, ci = m.forecast(steps=8, confidence_level=0.95)
print("mean =", " ".join("%.8f" % v for v in f))
print("lo95 =", " ".join("%.8f" % v for v in ci["lower_bound"]))
se = (f - np.asarray(ci["lower_bound"])) / 1.959963984540054
print("implied se =", " ".join("%.6f" % v for v in se))
print("se/se[0]   =", " ".join("%.6f" % v for v in se / se[0]))
