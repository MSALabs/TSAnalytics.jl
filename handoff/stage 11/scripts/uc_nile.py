import numpy as np, warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.structural import UnobservedComponents
import statsmodels.api as sm

nile = sm.datasets.nile.load_pandas().data['volume'].values.astype(float)
print("n =", len(nile), "mean =", round(nile.mean(),4))

m = UnobservedComponents(nile, level='local level')
print("k_states =", m.k_states, "param_names =", m.param_names)
print("initialization =", m.ssm.initialization)
r = m.fit(disp=False)
print("llf = %.10f" % r.llf)
print("params =", " ".join("%s=%.10f"%(n,v) for n,v in zip(m.param_names, r.params)))
print("smoothed level[0,1,-1] =", [round(float(x),6) for x in r.level.smoothed[[0,1,-1]]])
