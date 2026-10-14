import numpy as np, json, warnings
np.set_printoptions(precision=6, suppress=True, linewidth=200)
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
y = np.loadtxt("fixtureA.csv", delimiter=",", skiprows=1)
z = (y - y.mean(0))/y.std(0, ddof=1)

print("#"*95); print("Q6.1  DOES DynamicFactor STANDARDISE THE DATA?")
m1 = DynamicFactor(y, k_factors=1, factor_order=1)          # raw
m2 = DynamicFactor(z, k_factors=1, factor_order=1)          # standardised
print(f"  mod.endog[:2] raw-input  = {m1.endog[:2].ravel()}")
print(f"  input      y[:2]         = {y[:2].ravel()}")
print(f"  identical to input? {np.allclose(m1.endog, y)}  -> NO internal standardisation" if np.allclose(m1.endog,y) else "  CHANGED")
r1=m1.fit(disp=False,maxiter=2000); r2=m2.fit(disp=False,maxiter=2000)
print(f"  llf(raw)={r1.llf:.8f}   llf(std)={r2.llf:.8f}   -> differ => NOT scale-invariant")
print(f"  loadings raw = {r1.params[m1._params_loadings]}")
print(f"  loadings std = {r2.params[m2._params_loadings]}")

print("#"*95); print("Q6.2  DynamicFactorMQ: does IT standardise?")
mq = DynamicFactorMQ(y, factors=1, factor_orders=1, idiosyncratic_ar1=False)
print(f"  mq.standardize = {getattr(mq,'standardize',None)}")
print(f"  mq._endog_mean = {np.asarray(mq._endog_mean).ravel()}")
print(f"  mq._endog_std  = {np.asarray(mq._endog_std).ravel()}")
print(f"  y.mean(0)      = {y.mean(0)}")
print(f"  y.std(0,ddof=1)= {y.std(0,ddof=1)}   y.std(0,ddof=0)={y.std(0,ddof=0)}")

print("#"*95); print("Q6.3  error_order default / error_cov_type default")
m = DynamicFactor(z, k_factors=1, factor_order=1)
print(f"  error_order    default = {m.error_order}   (white noise)")
print(f"  error_cov_type default = {m.error_cov_type!r}")
print(f"  error_var      default = {m.error_var}")
print(f"  enforce_stationarity   = {m.enforce_stationarity}")
for ect in ["scalar","diagonal","unstructured"]:
    mm = DynamicFactor(z, k_factors=1, factor_order=1, error_cov_type=ect)
    print(f"    error_cov_type={ect:<13} k_params={mm.k_params:<3} parameters={dict(mm.parameters)}")
mm = DynamicFactor(z, k_factors=1, factor_order=1, error_order=1)
print(f"  error_order=1  k_params={mm.k_params} k_states={mm.k_states} params={dict(mm.parameters)}")
rr = mm.fit(disp=False, maxiter=3000)
print(f"  error_order=1  llf={rr.llf:.8f}  (vs error_order=0 llf={r2.llf:.8f})")
print(f"  error_order=1  AR coefs of idio = {rr.params[mm._params_error_transition]}")

print("#"*95); print("Q6.4  factor_order = 0")
try:
    m0 = DynamicFactor(z, k_factors=1, factor_order=0)
    print(f"  CONSTRUCTED. k_states={m0.k_states} k_params={m0.k_params} _factor_order={m0._factor_order}")
    print(f"  parameters={dict(m0.parameters)}  _unused_state={m0._unused_state}")
    print(f"  state_names={m0.state_names}  param_names={m0.param_names}")
    r0=m0.fit(disp=False,maxiter=3000); print(f"  llf={r0.llf:.8f}  (static factor, no AR)")
    print(f"  T matrix =\n{m0['transition']}")
except Exception as e:
    print(f"  RAISED {type(e).__name__}: {e}")
print("  -- k_factors=0:")
try:
    mk = DynamicFactor(z, k_factors=0, factor_order=1)
    print(f"  CONSTRUCTED k_states={mk.k_states} k_params={mk.k_params} _unused_state={mk._unused_state}")
except Exception as e:
    print(f"  RAISED {type(e).__name__}: {e}")
print("  -- k_factors >= k_endog:")
try:
    DynamicFactor(z, k_factors=5, factor_order=1)
except Exception as e: print(f"  RAISED {type(e).__name__}: {e}")
print("  -- univariate endog:")
try:
    DynamicFactor(z[:,:1], k_factors=1, factor_order=1)
except Exception as e: print(f"  RAISED {type(e).__name__}: {e}")
