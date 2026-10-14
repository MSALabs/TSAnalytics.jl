import numpy as np, json
np.set_printoptions(precision=6, suppress=True, linewidth=200)
sm = json.load(open("sm_results.json"))
def rd(f): return np.loadtxt(f, delimiter=",", ndmin=2)

for tag, N, r in [("A",5,1), ("B",6,2)]:
    Lsm = np.array(sm[tag]["Lam"]); Asm = np.array(sm[tag]["A"])
    Rsm = np.array(sm[tag]["sig2"]); Fsm = np.loadtxt(f"sm_{tag}_Fsmooth.csv", delimiter=",", ndmin=2)
    for meth in ["DGR","BM"]:
        p=f"ss_{tag}_{meth}_"
        A=rd(p+"A.csv"); C=rd(p+"C.csv"); Q=rd(p+"Q.csv"); R=rd(p+"R.csv")
        Fd=np.loadtxt(p+"Fqml.csv", delimiter=",", skiprows=1, ndmin=2)
        # --- renormalise dfms to Q=I convention: L = chol(Q); f~ = L^-1 f ---
        L = np.linalg.cholesky(Q[:r,:r])
        Cn = C[:, :r] @ L                   # Lambda~ = Lambda L
        An = np.linalg.solve(L, A[:r,:r] @ L)  # A~ = L^-1 A L
        Fn = np.linalg.solve(L, Fd.T).T      # f~ = L^-1 f
        # --- orthogonal Procrustes: find G (r x r, orthogonal) s.t. Cn G ~ Lsm ---
        U,s,Vt = np.linalg.svd(Cn.T @ Lsm); G = U @ Vt
        Ca = Cn @ G; Aa = G.T @ An @ G; Fa = Fn @ G
        Rd = np.diag(R)[:N]
        print("="*100)
        print(f"FIXTURE {tag}  (N={N}, r={r})   dfms em.method={meth}  vs statsmodels DynamicFactor")
        print(f"  rotation G (orthogonal, det={np.linalg.det(G):+.6f}) =\n{G}")
        print(f"  loadings: statsmodels | dfms(renorm+aligned) | abs diff")
        for i in range(N):
            print(f"    y{i+1}  " + "  ".join(f"{Lsm[i,j]:+.6f}/{Ca[i,j]:+.6f} d={abs(Lsm[i,j]-Ca[i,j]):.2e}" for j in range(r)))
        den = np.abs(Lsm).max()
        print(f"  max abs diff loadings = {np.abs(Lsm-Ca).max():.6e}   rel(/max|L|={den:.4f}) = {np.abs(Lsm-Ca).max()/den:.4e}")
        print(f"  factor AR: statsmodels=\n{Asm}\n             dfms aligned=\n{Aa}")
        print(f"  max abs diff A = {np.abs(Asm-Aa).max():.6e}  rel = {np.abs(Asm-Aa).max()/np.abs(Asm).max():.4e}")
        print(f"  idio var: sm ={Rsm}")
        print(f"            dfms={Rd}")
        print(f"  max abs diff R = {np.abs(Rsm-Rd).max():.6e}  max rel = {np.abs((Rsm-Rd)/Rsm).max():.4e}")
        print(f"  smoothed factor path:")
        for j in range(r):
            c = np.corrcoef(Fsm[:,j], Fa[:,j])[0,1]
            print(f"    f{j+1}: corr={c:.10f}  max|diff|={np.abs(Fsm[:,j]-Fa[:,j]).max():.6e}  sd_sm={Fsm[:,j].std(ddof=1):.6f} sd_dfms={Fa[:,j].std(ddof=1):.6f}")
        print(f"  cov(F_sm_aligned_dfms)=\n{np.cov(Fa.T, ddof=1)}")

# ---- WHERE does the max factor-path diff occur? (tests initial-condition attribution) ----
print("\n" + "#"*95); print("WHERE IS THE FACTOR-PATH DISAGREEMENT CONCENTRATED?")
for tag, N, r in [("A",5,1), ("B",6,2)]:
    Lsm = np.array(sm[tag]["Lam"]); Fsm = np.loadtxt(f"sm_{tag}_Fsmooth.csv", delimiter=",", ndmin=2)
    for meth in ["DGR","BM"]:
        p=f"ss_{tag}_{meth}_"
        Q=rd(p+"Q.csv"); C=rd(p+"C.csv")
        Fd=np.loadtxt(p+"Fqml.csv", delimiter=",", skiprows=1, ndmin=2)
        L=np.linalg.cholesky(Q[:r,:r]); Cn=C[:,:r]@L; Fn=np.linalg.solve(L,Fd.T).T
        U,s,Vt=np.linalg.svd(Cn.T@Lsm); G=U@Vt; Fa=Fn@G
        d=np.abs(Fsm-Fa)
        print(f"  {tag}/{meth}: argmax_t per factor = {d.argmax(0).tolist()}  (T={d.shape[0]})")
        print(f"      max|diff| = {d.max(0)}")
        print(f"      max|diff| over t>=10 = {d[10:].max(0)}   median|diff| = {np.median(d,0)}")
