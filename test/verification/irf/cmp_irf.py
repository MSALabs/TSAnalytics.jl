import json, numpy as np
R=json.load(open("fixtures/ref_R_irf.json")); P=json.load(open("fixtures/ref_py_irf.json"))
arr=lambda x: np.array(x,dtype=float)
tags=["A_p1_h6","A_p2_h8","B_p2_h6","US_p2_h6","E5_p2_h4","SH_p2_h4"]
print("=== IRF: R irf[[impulse]][horizon, response]  vs  py irfs[horizon, response, impulse] ===")
for t in tags:
    r,p=R[t],P[t]; K=p["K"]; h=p["h"]
    nm = r["names"] if isinstance(r.get("names"),list) else list(r["irf_ortho"].keys())
    for key_r, key_p in [("irf_ortho","orth_irfs"),("irf_nonortho","irfs"),
                         ("irf_ortho_cum","orth_cum_effects"),("irf_nonortho_cum","cum_effects")]:
        Rm=np.empty((h+1,K,K)); 
        for j,imp in enumerate(nm):
            Rm[:,:,j]=arr(r[key_r][imp])        # rows=horizon, cols=response
        Pm=arr(p[key_p])                         # [horizon, response, impulse]
        print(f"  {t:10s} {key_r:17s} vs {key_p:17s} max|diff| = {np.max(np.abs(Rm-Pm)):.3e}")
print("\n=== MA representation: R Psi/Phi (K,K,h+1) vs py orth_ma_rep/ma_rep (h+1,K,K) ===")
for t in tags:
    r,p=R[t],P[t]
    for key_r,key_p in [("Psi","orth_ma_rep"),("Phi","ma_rep")]:
        Rm=arr(r[key_r]); Pm=arr(p[key_p])
        Rm2 = np.moveaxis(Rm,-1,0)   # R stores (K,K,h+1); always move the horizon axis to front
        print(f"  {t:10s} {key_r:4s} R shape {Rm.shape} -> {Rm2.shape} vs py {Pm.shape}  max|diff| = {np.max(np.abs(Rm2-Pm)):.3e}")
print("\n=== FEVD: R fevd[[variable]][horizon 1..h, shock] vs py decomp[variable, horizon, shock] ===")
for t in tags:
    r,p=R[t],P[t]; K=p["K"]; h=p["h"]
    nm = r["names"] if isinstance(r.get("names"),list) else list(r["fevd"].keys())
    Rm=np.empty((K,h,K))
    for i,var in enumerate(nm): Rm[i]=arr(r["fevd"][var])
    Pm=arr(p["fevd_decomp"])
    print(f"  {t:10s} R {Rm.shape} vs py {Pm.shape}  max|diff| = {np.max(np.abs(Rm-Pm)):.3e}"
          f"   rows sum to 1: {np.max(np.abs(Pm.sum(axis=2)-1)):.2e}")
print("\n=== Cholesky factor: which Sigma does the orthogonalisation use? ===")
for t in tags:
    r,p=R[t],P[t]
    print(f"  {t:10s} |R t(chol(Sigma_df)) - py cholesky(sigma_u)| = {np.max(np.abs(arr(r['chol_lower'])-arr(p['chol_lower']))):.3e}")
