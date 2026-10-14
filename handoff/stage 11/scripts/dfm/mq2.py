import numpy as np, pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
y = np.loadtxt("fixtureA.csv", delimiter=",", skiprows=1)
df = pd.DataFrame(y, columns=[f"y{i+1}" for i in range(5)],
                  index=pd.period_range("2000-01", periods=200, freq="M"))
zq = (y - y.mean(0))/y.std(0, ddof=0)
ml = DynamicFactor(zq, k_factors=1, factor_order=1).fit(disp=False, maxiter=3000)
print(f"DynamicFactor ML (stationary init)          llf = {ml.llf:.10f}")
for emi in [True, False]:
    mq = DynamicFactorMQ(df, factors=1, factor_orders=1, idiosyncratic_ar1=False)
    r = mq.fit(disp=False, em_initialization=emi, maxiter=2000, tolerance=1e-10)
    Lam = np.asarray(r.params.iloc[:5]); chol = float(r.params.iloc[6])
    print(f"DynamicFactorMQ EM em_initialization={str(emi):<5} llf = {r.llf:.10f}  iters={r.mle_retvals['iter']}")
    print(f"   loadings*chol (renorm to Q=I) = {Lam*chol}")
    print(f"   ML loadings                   = {ml.params[:5]}")
    print(f"   max|diff| after sign-align    = {np.abs(np.abs(Lam*chol)-np.abs(ml.params[:5])).max():.3e}")
print("\n--- MQ with idiosyncratic_ar1=True (default?) ---")
import inspect
sig = inspect.signature(DynamicFactorMQ.__init__)
for p in ["factors","factor_orders","factor_multiplicities","idiosyncratic_ar1","standardize"]:
    print(f"   default {p} = {sig.parameters[p].default!r}")
