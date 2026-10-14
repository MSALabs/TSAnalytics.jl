import numpy as np, json
np.set_printoptions(precision=6, suppress=True, linewidth=200)
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
y  = np.loadtxt("fixtureA.csv", delimiter=",", skiprows=1)
ym = np.loadtxt("fixtureA_miss.csv", delimiter=",", skiprows=1)
mi = json.load(open("fixtureA_miss_idx.json"))["miss_idx"]
st = lambda a: (a - np.nanmean(a,0))/np.nanstd(a,0,ddof=1)
z, zm = st(y), st(ym)
print(f"missing cells = {len(mi)}; pattern: {mi}")
print(f"NaN count per column = {np.isnan(zm).sum(0)}; rows with >=1 NaN = {int((np.isnan(zm).any(1)).sum())}; fully-missing rows = {int((np.isnan(zm).all(1)).sum())}")

mA = DynamicFactor(z,  k_factors=1, factor_order=1); rA = mA.fit(disp=False, maxiter=3000)
mB = DynamicFactor(zm, k_factors=1, factor_order=1); rB = mB.fit(disp=False, maxiter=3000)
nm = np.array(mB.ssm._statespace.nmissing) if hasattr(mB.ssm,'_statespace') else None
print("\n--- nmissing vector mechanism (Statespace.nmissing) ---")
mB.ssm.loglike()  # force bind
ss = mB.ssm._statespace
nm = np.array(ss.nmissing)
print(f"  ss.nmissing length={len(nm)}  unique counts={dict(zip(*np.unique(nm, return_counts=True)))}")
print(f"  t where nmissing>0: {np.where(nm>0)[0].tolist()}")
print(f"  ss.missing shape={np.array(ss.missing).shape} (k_endog x nobs, int 0/1)")
print(f"  missing col t=150 -> {np.array(ss.missing)[:,150]}  (fully missing row)")
print(f"  missing col t=5   -> {np.array(ss.missing)[:,5]}")

print("\n--- llf comparison ---")
print(f"  llf complete  = {rA.llf!r}  (nobs={mA.nobs}, k_endog={mA.k_endog})")
print(f"  llf with NaNs = {rB.llf!r}")
print(f"  llf increased (fewer obs contribute)? {rB.llf > rA.llf}")
nobs_ke_A = mA.nobs*mA.k_endog
nobs_ke_B = mB.nobs*mB.k_endog - nm.sum()
print(f"  obs-dim contributions: complete={nobs_ke_A}  missing={nobs_ke_B}  (dropped {nm.sum()} = {len(mi)} cells)")
print(f"  llf per contributing obs: complete={rA.llf/nobs_ke_A:.8f}  missing={rB.llf/nobs_ke_B:.8f}")
print(f"  loadings complete={rA.params[mA._params_loadings]}")
print(f"  loadings missing ={rB.params[mB._params_loadings]}")
print(f"  sigma2  complete={rA.params[mA._params_error_cov]}")
print(f"  sigma2  missing ={rB.params[mB._params_error_cov]}")
print(f"  A complete={rA.params[mA._params_factor_transition]}  missing={rB.params[mB._params_factor_transition]}")

print("\n--- monotonicity check: llf must rise as more cells are blanked ---")
rng = np.random.default_rng(42)
prev=None
for k in [0, 10, 25, 50, 100]:
    zz = z.copy()
    if k:
        idx = rng.choice(z.size, size=k, replace=False)
        zz.ravel()[idx] = np.nan
    mm = DynamicFactor(zz, k_factors=1, factor_order=1)
    rr = mm.fit(disp=False, maxiter=3000)
    mm.ssm.loglike(); nmk = np.array(mm.ssm._statespace.nmissing).sum()
    print(f"  blanked={k:>4} (nmissing_total={nmk:>4})  llf={rr.llf:.6f}  per-obs={rr.llf/(mm.nobs*mm.k_endog-nmk):.6f}")
