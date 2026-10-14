import numpy as np, warnings; warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True, linewidth=150)
from statsmodels.tsa.statespace.structural import UnobservedComponents
y = np.loadtxt("ukgas.csv", skiprows=1)
specs = [("local level",          dict(level='local level')),
         ("local linear trend",   dict(level='local linear trend')),
         ("lltrend + seasonal 4", dict(level='local linear trend', seasonal=4)),
         ("llevel + freq_seas",   dict(level='local level', freq_seasonal=[{'period':4,'harmonics':2}])),
         ("llevel + damped cycle",dict(level='local level', cycle=True, stochastic_cycle=True, damped_cycle=True)),
         ("llevel + ar2",         dict(level='local level', autoregressive=2))]
for name, kw in specs:
    m = UnobservedComponents(y, **kw)
    m.update(m.start_params)
    print("="*78); print(name, " k_states =", m.k_states, " k_posdef =", m.ssm.k_posdef)
    print(" states:", m.state_names)
    print(" params:", m.param_names)
    print(" Z =", np.asarray(m.ssm['design'])[0,:,0] if np.asarray(m.ssm['design']).ndim==3 else np.asarray(m.ssm['design'])[0])
    T = np.asarray(m.ssm['transition']); T = T[:,:,0] if T.ndim==3 else T
    print(" T =\n", T)
    Rm = np.asarray(m.ssm['selection']); Rm = Rm[:,:,0] if Rm.ndim==3 else Rm
    print(" R' =", Rm.T)
