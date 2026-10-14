import numpy as np, warnings; warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.structural import UnobservedComponents
import statsmodels.api as sm
y = sm.datasets.nile.load_pandas().data['volume'].values.astype(float)
R_level, R_eps = 1469.147, 15098.577

def llf_at(params, P0=None, a0=None):
    m = UnobservedComponents(y, level='local level')
    if P0 is not None:
        m.ssm.initialize_known(np.array([a0]), np.array([[P0]]))
    return m.loglike(np.array(params))

print("statsmodels objective evaluated at R's optimum, default init (approx diffuse 1e6):")
print("   llf(R params)      = %.6f" % llf_at([R_eps, R_level]))
print("   llf(own optimum)   = %.6f" % llf_at([15078.0099864289, 1478.8120190697]))
vx = np.var(y, ddof=1)/100
print("\nR's own init: a0 = y[1] = %.1f, P0 = 1e6*var(y)/100 = %.6g" % (y[0], 1e6*vx))
print("   llf(R params, R init)    = %.6f" % llf_at([R_eps, R_level], P0=1e6*vx, a0=y[0]))
print("   R's reported loglik      = -643.201000")
print("\nR's loglik0 = -367.5194 ; R's loglik = -643.201")
n = len(y)
print("   -643.201 - (-632.5378) = %.6f" % (-643.201 - llf_at([R_eps, R_level])))
print("   0.5*n*log(2pi)        = %.6f" % (0.5*n*np.log(2*np.pi)))
print("   0.5*n*log(2pi)/... n/2 = %.6f" % (n/2*np.log(2*np.pi)))
