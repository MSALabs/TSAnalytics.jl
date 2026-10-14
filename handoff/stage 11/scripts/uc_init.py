import numpy as np, warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.structural import UnobservedComponents
import statsmodels.api as sm
nile = sm.datasets.nile.load_pandas().data['volume'].values.astype(float)
for spec in [dict(level='local level'),
             dict(level='local linear trend'),
             dict(level=True, trend=True, seasonal=4, cycle=True, stochastic_cycle=True, damped_cycle=True, autoregressive=1, irregular=True)]:
    m = UnobservedComponents(nile, **spec)
    init = m.ssm.initialization
    print("spec:", spec)
    print("   k_states=%d  init.initialization_type=%s" % (m.k_states, init.initialization_type))
    print("   blocks:", {k: v.initialization_type for k,v in init.blocks.items()} if init.blocks else "none")
    print("   approx_diffuse_variance =", init.approximate_diffuse_variance)
    print("   params:", m.param_names)
    print("   states:", m.state_names)
