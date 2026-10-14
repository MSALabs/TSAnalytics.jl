import numpy as np, pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
y = np.loadtxt("fixtureA.csv", delimiter=",", skiprows=1)
df = pd.DataFrame(y, columns=[f"y{i+1}" for i in range(5)],
                  index=pd.period_range("2000-01", periods=200, freq="M"))
for nm, inp in [("numpy ndarray", y), ("pandas DataFrame", df)]:
    m = DynamicFactorMQ(inp, factors=1, factor_orders=1, idiosyncratic_ar1=False)
    s = np.asarray(m._endog_std).ravel()
    print(f"{nm:<18} _endog_std = {s}")
print(f"{'y.std(ddof=0)':<18} = {y.std(0,ddof=0)}")
print(f"{'y.std(ddof=1)':<18} = {y.std(0,ddof=1)}")
print("\n=> DynamicFactorMQ standardisation denominator DEPENDS ON INPUT TYPE")
print("   ddof=1 (pandas) vs ddof=0 (numpy); llf differs by T*sum(log(sd1/sd0)):")
T,N = y.shape
print(f"   predicted llf shift = {T*np.sum(np.log(y.std(0,ddof=1)/y.std(0,ddof=0))):.6f}")
