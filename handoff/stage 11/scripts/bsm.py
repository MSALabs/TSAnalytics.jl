import numpy as np, warnings; warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.structural import UnobservedComponents
y = np.loadtxt("ukgas.csv", skiprows=1)
print("n =", len(y))
m = UnobservedComponents(y, level='local linear trend', seasonal=4, stochastic_seasonal=True)
r = m.fit(disp=False, maxiter=2000)
print("params:", dict(zip(m.param_names, np.round(r.params, 10))))
print("llf = %.8f" % r.llf)
print("\nplot_components panels available:", [k for k in ['level','trend','seasonal','freq_seasonal','cycle','autoregressive'] if getattr(r, k, None) is not None])
