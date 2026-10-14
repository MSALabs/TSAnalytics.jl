import numpy as np, warnings; warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.structural import UnobservedComponents
import statsmodels.api as sm, pandas as pd
y = sm.datasets.nile.load_pandas().data['volume'].values.astype(float)
V, h = 1469.146619, 15098.577154
P0 = 1e6*np.var(y, ddof=1)/100
m = UnobservedComponents(y, level='local level')
m.ssm.initialize_known(np.array([y[0]]), np.array([[P0]]))
res = m.smooth(np.array([h, V]))
R = pd.read_csv("uc_R_filter.csv")
pf = res.filtered_state[0]; ps = res.smoothed_state[0]
print("a0 = %.1f  P0 = %.6g" % (y[0], P0))
print("filtered state: max abs diff vs R = %.3e   (R[1:3] %s)" %
      (np.max(np.abs(pf - R['filt'].values)), np.round(R['filt'].values[:3],6)))
print("                                              (sm[1:3] %s)" % np.round(pf[:3],6))
print("smoothed state: max abs diff vs R = %.3e" % np.max(np.abs(ps - R['smooth'].values)))
print("statsmodels llf at these params/init = %.10f" % res.llf)
print("R's first standardised residual = %.3e (identically zero: a0 = y[1])" % R['resid'].values[0])
