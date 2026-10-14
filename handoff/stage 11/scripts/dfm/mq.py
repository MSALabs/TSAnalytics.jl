import numpy as np
np.set_printoptions(precision=6, suppress=True, linewidth=200)
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
import pandas as pd
y = np.loadtxt("fixtureA.csv", delimiter=",", skiprows=1)
df = pd.DataFrame(y, columns=[f"y{i+1}" for i in range(5)],
                  index=pd.period_range("2000-01", periods=200, freq="M"))
mq = DynamicFactorMQ(df, factors=1, factor_orders=1, idiosyncratic_ar1=False)
print(f"standardize={mq.standardize}  k_params={mq.k_params}  k_states={mq.k_states}")
print(f"param_names={list(mq.param_names)}")
res = mq.fit(disp=False)
print("\n--- fit() retvals: does it report an EM iteration count? ---")
for k,v in res.mle_retvals.items():
    print(f"  {k} = {v if not isinstance(v,(list,np.ndarray)) else f'<{type(v).__name__} len={len(v)}>'}")
print(f"\n  llf = {res.llf!r}")
print(f"  loadings =\n{np.asarray(res.params.iloc[:5])}")
print(f"  full params:\n{res.params}")
print("\n--- IDENTIFICATION: loadings / factor cov ---")
L = np.asarray(res.params.iloc[:5]).reshape(5,1)
print(f"  L'L = {L.T@L}")
fs = np.asarray(res.factors.smoothed)
print(f"  factors.smoothed shape={fs.shape}  var={np.var(fs, ddof=1, axis=0)}")
print(f"  Q (state_cov) =\n{mq['state_cov']}")
print("\n--- compare MQ EM llf vs DynamicFactor ML llf (MQ standardises with ddof=0) ---")
zq = (y - y.mean(0))/y.std(0, ddof=0)
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
rdf = DynamicFactor(zq, k_factors=1, factor_order=1).fit(disp=False, maxiter=3000)
print(f"  DynamicFactor(ML, ddof=0 std data) llf = {rdf.llf!r}")
print(f"  DynamicFactorMQ(EM)                llf = {res.llf!r}")
print(f"  diff = {res.llf - rdf.llf:.6e}")
print(f"  DF loadings = {rdf.params[:5]}")
print(f"  MQ loadings = {np.asarray(res.params.iloc[:5])}")
print("\n--- MQ results methods (chart/diagnostic surface) ---")
meths = [m for m in dir(res) if not m.startswith('_')]
print("  ", ", ".join(meths))
