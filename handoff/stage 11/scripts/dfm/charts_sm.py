import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
y = np.loadtxt("fixtureB.csv", delimiter=",", skiprows=1)
z = (y-y.mean(0))/y.std(0,ddof=1)
df = pd.DataFrame(y, columns=[f"y{i+1}" for i in range(6)],
                  index=pd.period_range("2000-01", periods=250, freq="M"))
r = DynamicFactor(z, k_factors=2, factor_order=1).fit(disp=False, maxiter=3000)
print("=== DynamicFactorResults chart/diagnostic surface ===")
for nm in ["plot_coefficients_of_determination","plot_diagnostics","coefficients_of_determination","factors"]:
    print(f"  {nm:<36} present={hasattr(r,nm)}")
try:
    f=r.plot_coefficients_of_determination(); f.savefig("sm_cod.png"); print("  plot_coefficients_of_determination() -> OK (sm_cod.png)")
except Exception as e: print(f"  plot_coefficients_of_determination FAILED {e}")
try:
    f=r.plot_diagnostics(variable=0); f.savefig("sm_diag.png"); print("  plot_diagnostics(variable=0)          -> OK (sm_diag.png)")
except Exception as e: print(f"  plot_diagnostics FAILED {e}")
cod = np.asarray(r.coefficients_of_determination)
print(f"  coefficients_of_determination shape={cod.shape}\n{cod}")
print(f"  factors.smoothed shape={np.asarray(r.factors.smoothed).shape}; has .cov? {hasattr(r.factors,'cov')}")
print(f"  factors fields = {list(r.factors.keys()) if hasattr(r.factors,'keys') else dir(r.factors)}")

print("\n=== DynamicFactorMQResults extra surface ===")
mq = DynamicFactorMQ(df, factors=2, factor_orders=1, idiosyncratic_ar1=False)
rq = mq.fit(disp=False)
for nm in ["plot_coefficients_of_determination","get_coefficients_of_determination",
           "get_smoothed_decomposition","news","append","extend","apply","simulate","impulse_responses","summary"]:
    print(f"  {nm:<36} present={hasattr(rq,nm)}")
try:
    f=rq.plot_coefficients_of_determination(); f.savefig("mq_cod.png"); print("  MQ plot_coefficients_of_determination -> OK")
except Exception as e: print(f"  MQ plot cod FAILED {e}")
try:
    d = rq.get_smoothed_decomposition(decomposition_of="smoothed_state")
    print(f"  get_smoothed_decomposition -> OK, returns {len(d)} objects, shapes={[np.asarray(x).shape for x in d]}")
except Exception as e: print(f"  get_smoothed_decomposition FAILED {type(e).__name__}: {e}")
try:
    cd = rq.get_coefficients_of_determination(method="individual")
    print(f"  get_coefficients_of_determination(method='individual') ->\n{cd}")
except Exception as e: print(f"  FAILED {e}")
