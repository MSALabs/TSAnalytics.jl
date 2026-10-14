import numpy as np
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
y = np.loadtxt("fixtureA.csv", delimiter=",", skiprows=1)
z = (y-y.mean(0))/y.std(0,ddof=1)
m = DynamicFactor(z, k_factors=1, factor_order=1)
p = m.start_params.copy()
print(f"default inversion_method flags: invert_univariate={m.ssm.invert_univariate} solve_cholesky={m.ssm.solve_cholesky}")
print(f"                               solve_lu={m.ssm.solve_lu} invert_lu={m.ssm.invert_lu} invert_cholesky={m.ssm.invert_cholesky}")
print(f"filter_method: conventional={m.ssm.filter_conventional} univariate={m.ssm.filter_univariate} collapsed={m.ssm.filter_collapsed}")
# drive idiosyncratic variances to ~0 -> F_t = Z P Z' + H becomes rank-1 (singular for N=5)
for eps in [1e-2, 1e-6, 1e-12, 0.0]:
    pp = p.copy(); pp[m._params_error_cov] = eps
    try:
        ll = m.loglike(pp); print(f"  sigma2 = {eps:<8}  llf = {ll!r}")
    except Exception as e:
        print(f"  sigma2 = {eps:<8}  RAISED {type(e).__name__}: {e}")
print("\n--- with FILTER_UNIVARIATE (sequential scalar updates, avoids matrix F_t) ---")
m2 = DynamicFactor(z, k_factors=1, factor_order=1); m2.ssm.filter_univariate = True
for eps in [1e-2, 1e-12, 0.0]:
    pp = p.copy(); pp[m2._params_error_cov] = eps
    try: print(f"  sigma2 = {eps:<8}  llf = {m2.loglike(pp)!r}")
    except Exception as e: print(f"  sigma2 = {eps:<8}  RAISED {type(e).__name__}: {e}")
print("\n--- llf agreement conventional vs univariate filter at normal params ---")
print(f"  conventional = {m.loglike(p)!r}")
print(f"  univariate   = {m2.loglike(p)!r}")
print(f"  diff         = {abs(m.loglike(p)-m2.loglike(p)):.3e}")
