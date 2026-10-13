"""Independent verification of statsmodels' ANALYTIC IRF standard errors.
vars has no analytic SEs (it bootstraps), so statsmodels is the only reference.
Two independent checks: (1) a numerical Jacobian of the IRF wrt vec(A),
(2) a parametric bootstrap of the sampling distribution."""
import numpy as np, csv, json
from statsmodels.tsa.api import VAR
rng = np.random.default_rng(20261013)
rd = lambda p: np.array([[float(c) for c in r] for r in list(csv.reader(open(p)))[1:]])

Y = rd("fixtures/var2_p1.csv"); n, K = Y.shape; p = 2; H = 4
r = VAR(Y).fit(p, trend="c")
A = r.coefs.copy()            # (p,K,K)
nu = r.intercept.copy()
Sig = r.sigma_u.copy()
ir = r.irf(H)
se_an      = ir.stderr()            # (H+1,K,K) non-orth
se_an_orth = ir.stderr(orth=True)

def ma_rep(A, H):
    """Phi_0 = I; Phi_i = sum_{j=1..min(i,p)} Phi_{i-j} A_j"""
    pp, K, _ = A.shape
    Phi = np.zeros((H+1, K, K)); Phi[0] = np.eye(K)
    for i in range(1, H+1):
        acc = np.zeros((K, K))
        for j in range(1, min(i, pp)+1):
            acc += Phi[i-j] @ A[j-1]
        Phi[i] = acc
    return Phi

# sanity: my recursion must reproduce statsmodels' ma_rep exactly
assert np.max(np.abs(ma_rep(A, H) - r.ma_rep(H))) < 1e-13, "ma_rep mismatch"
print("ma_rep recursion matches statsmodels to", np.max(np.abs(ma_rep(A,H)-r.ma_rep(H))))

# ---------- check 1: numerical Jacobian ----------
cov_a = np.asarray(ir.cov_a)                # (K^2 p, K^2 p)
m = K*K*p
eps = 1e-6
def pack(A, order):
    """two candidate flattenings of the coefficient array"""
    if order == "vec_stacked":      # vec([A1 ... Ap]) -- column-major of the K x Kp matrix
        return np.concatenate([A[j].T.ravel() for j in range(p)])
    return A.ravel()                # row-major of (p,K,K)
def unpack(v, order):
    if order == "vec_stacked":
        return np.stack([v[j*K*K:(j+1)*K*K].reshape(K, K).T for j in range(p)])
    return v.reshape(p, K, K)

results = {}
for order in ("vec_stacked", "rowmajor"):
    base = pack(A, order)
    J = np.zeros((H+1, K*K, m))
    for idx in range(m):
        vu = base.copy(); vu[idx] += eps
        vd = base.copy(); vd[idx] -= eps
        up = ma_rep(unpack(vu, order), H); dn = ma_rep(unpack(vd, order), H)
        for hh in range(H+1):
            # statsmodels stores IRF slices as [response, impulse]; vec column-major
            J[hh, :, idx] = ((up[hh]-dn[hh])/(2*eps)).T.ravel()
    se_num = np.zeros_like(se_an)
    for hh in range(H+1):
        C = J[hh] @ cov_a @ J[hh].T
        se_num[hh] = np.sqrt(np.maximum(np.diag(C), 0)).reshape(K, K).T
    worst = max(np.max(np.abs(se_num[hh]-se_an[hh])) for hh in range(H+1))
    results[order] = (worst, se_num)
    print(f"  flattening={order:12s} worst |numerical - analytic| over h=0..{H}: {worst:.3e}")
order = min(results, key=lambda k: results[k][0])
se_num = results[order][1]
print(f"\ncheck 1 -- numerical Jacobian vs statsmodels analytic (non-orth), flattening={order}:")
for hh in range(H+1):
    d = np.max(np.abs(se_num[hh]-se_an[hh]))
    rel = d/max(np.max(se_an[hh]), 1e-12)
    print(f"  h={hh}: max|diff| = {d:.3e}   rel = {rel:.2e}")

# ---------- check 2: parametric bootstrap ----------
Pch = np.linalg.cholesky(Sig)
NREP = 4000
acc_n = np.zeros((NREP, H+1, K, K)); acc_o = np.zeros((NREP, H+1, K, K))
for b in range(NREP):
    e = rng.standard_normal((n, K)) @ Pch.T
    Ysim = np.zeros((n, K)); Ysim[:p] = Y[:p]
    for t in range(p, n):
        acc = nu.copy()
        for j in range(p): acc = acc + A[j] @ Ysim[t-1-j]
        Ysim[t] = acc + e[t]
    # fast OLS fit
    T = n-p
    Z = np.column_stack([np.ones(T)] + [Ysim[p-1-j:n-1-j] for j in range(p)])
    X = Ysim[p:]
    B, *_ = np.linalg.lstsq(Z, X, rcond=None)
    Ah = np.stack([B[1+j*K:1+(j+1)*K, :].T for j in range(p)])
    E = X - Z@B
    Sh = E.T@E/(T - Z.shape[1])
    Ph = ma_rep(Ah, H)
    acc_n[b] = Ph
    L = np.linalg.cholesky(Sh)
    acc_o[b] = np.einsum('hij,jk->hik', Ph, L)
se_bs_n = acc_n.std(axis=0, ddof=1)
se_bs_o = acc_o.std(axis=0, ddof=1)
print(f"\ncheck 2 -- parametric bootstrap ({NREP} reps) vs statsmodels analytic:")
print("  NON-ORTHOGONALISED")
for hh in range(1, H+1):
    rel = np.max(np.abs(se_bs_n[hh]-se_an[hh])/np.maximum(se_an[hh],1e-12))
    print(f"    h={hh}: max relative deviation = {rel*100:6.2f}%   bs[0,0]={se_bs_n[hh,0,0]:.6f} an[0,0]={se_an[hh,0,0]:.6f}")
print("  ORTHOGONALISED (the one with the Lutkepohl typo statsmodels flags)")
for hh in range(0, H+1):
    rel = np.max(np.abs(se_bs_o[hh]-se_an_orth[hh])/np.maximum(se_an_orth[hh],1e-12))
    print(f"    h={hh}: max relative deviation = {rel*100:6.2f}%   bs[0,0]={se_bs_o[hh,0,0]:.6f} an[0,0]={se_an_orth[hh,0,0]:.6f}")
json.dump({"se_num":se_num.tolist(),"se_bs_nonorth":se_bs_n.tolist(),
           "se_bs_orth":se_bs_o.tolist(),"nrep":NREP,"seed":20261013},
          open("fixtures/verify_irf_se.json","w"), indent=1)
