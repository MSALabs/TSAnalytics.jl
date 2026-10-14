import numpy as np, warnings; warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.structural import UnobservedComponents
import statsmodels.api as sm
y = sm.datasets.nile.load_pandas().data['volume'].values.astype(float)
R = np.array([15098.577, 1469.147])
m = UnobservedComponents(y, level='local level')
print("R (=Durbin&Koopman published): eps=%.4f level=%.4f  llf=%.8f" % (R[0], R[1], m.loglike(R)))
for meth, kw in [("lbfgs(default)", {}), ("lbfgs tight", dict(method='lbfgs', maxiter=5000, pgtol=1e-12, factr=1.0)),
                 ("nm", dict(method='nm', maxiter=20000, xtol=1e-12, ftol=1e-12)),
                 ("powell", dict(method='powell', maxiter=20000, xtol=1e-12, ftol=1e-12)),
                 ("bfgs", dict(method='bfgs', maxiter=5000, gtol=1e-12))]:
    try:
        r = m.fit(disp=False, **kw)
        print("%-15s eps=%12.4f level=%11.4f  llf=%.8f   |rel err vs R| = %.3e, %.3e" %
              (meth, r.params[0], r.params[1], r.llf,
               abs(r.params[0]-R[0])/R[0], abs(r.params[1]-R[1])/R[1]))
    except Exception as e:
        print(meth, "FAILED", e)
r2 = m.fit(start_params=R, disp=False, method='nm', maxiter=20000, xtol=1e-12, ftol=1e-12)
print("nm from R start : eps=%.4f level=%.4f llf=%.8f" % (r2.params[0], r2.params[1], r2.llf))
