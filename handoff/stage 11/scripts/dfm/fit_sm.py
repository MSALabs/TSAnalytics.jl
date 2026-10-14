import numpy as np, json, sys
np.set_printoptions(precision=8, suppress=False, linewidth=200)
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

def std(y):  # match collapse::fscale -> mean 0, sd 1 (ddof=1)
    return (y - y.mean(0)) / y.std(0, ddof=1)

out = {}
for tag, path, r in [("A","fixtureA.csv",1), ("B","fixtureB.csv",2)]:
    y = np.loadtxt(path, delimiter=",", skiprows=1)
    z = std(y)
    print("#"*90); print(f"### FIXTURE {tag}: T={y.shape[0]} N={y.shape[1]} r={r} (standardised input)")
    mod = DynamicFactor(z, k_factors=r, factor_order=1, error_order=0)
    print(f"  k_params={mod.k_params}  k_states={mod.k_states}  k_posdef={mod.ssm.k_posdef}")
    print(f"  param_names={mod.param_names}")
    print(f"  error_cov_type={mod.error_cov_type}  error_order={mod.error_order}  error_var={mod.error_var}")
    print(f"  initialization={mod.ssm.initialization}")
    sp = mod.start_params
    print(f"  start_params  = {sp}")
    print(f"  llf(start)    = {mod.loglike(sp):.10f}")
    res = mod.fit(disp=False, maxiter=2000)
    print(f"  CONVERGED={res.mle_retvals['converged']}  iters={res.mle_retvals.get('iterations')}  fcalls={res.mle_retvals.get('fcalls')}")
    print(f"  llf = {res.llf!r}")
    print(f"  aic = {res.aic!r}   bic = {res.bic!r}")
    Lam = res.params[mod._params_loadings].reshape(y.shape[1], r)
    sig = res.params[mod._params_error_cov]
    Amat= res.params[mod._params_factor_transition].reshape(r, r)
    print(f"  loadings Lambda ({y.shape[1]}x{r}) =\n{Lam}")
    print(f"  idio var sigma2 = {sig}")
    print(f"  factor AR A ({r}x{r}) =\n{Amat}")
    print(f"  Lambda'Lambda =\n{Lam.T@Lam}")
    fs = np.asarray(res.factors.smoothed)
    if fs.ndim==1: fs = fs[:,None]
    if fs.shape[0]==r and fs.shape[1]!=r: fs = fs.T
    print(f"  smoothed factors shape={fs.shape}  cov(F_smooth)=\n{np.cov(fs.T, ddof=1)}")
    print(f"  F_smooth first 3 =\n{fs[:3]}")
    print(f"  cod (coefficients_of_determination) = {np.asarray(res.coefficients_of_determination).ravel()}")
    np.savetxt(f"sm_{tag}_Fsmooth.csv", fs, delimiter=",")
    out[tag] = dict(llf=float(res.llf), aic=float(res.aic), bic=float(res.bic),
                    Lam=Lam.tolist(), sig2=sig.tolist(), A=Amat.tolist(),
                    k_params=int(mod.k_params), k_states=int(mod.k_states),
                    converged=bool(res.mle_retvals['converged']),
                    param_names=list(mod.param_names),
                    start_params=sp.tolist(), llf_start=float(mod.loglike(sp)),
                    cod=np.asarray(res.coefficients_of_determination).ravel().tolist())
json.dump(out, open("sm_results.json","w"), indent=1)
