import json, numpy as np
R = json.load(open("fixtures/ref_R.json")); P = json.load(open("fixtures/ref_py.json"))
def arr(x): return np.array(x, dtype=float)
def mx(a,b): return float(np.max(np.abs(arr(a)-arr(b))))

print("=== coefficient layout: R Bcoef (K x (Kp+ndet), det LAST) vs py params ((ndet+Kp) x K, det FIRST) ===")
for tag_r, tag_p in [("A_p1_const","A_p1_const"),("A_p2_const","A_p2_const"),
                     ("B_p2_const","B_p2_const"),("B_p3_const","B_p3_const"),
                     ("US_p2_const","US_p2_const"),("E5_p2_const","E5_p2_const"),
                     ("SH_p2_const","SH_p2_const"),("A_p1_none","A_p1_none")]:
    r = R[tag_r]; p = P[tag_p]; K = p["K"]; pp = p["p"]
    B = arr(r["Bcoef"])                       # K x (Kp + ndet)
    par = arr(p["params"])                    # (ndet + Kp) x K
    ndet = par.shape[0] - K*pp
    lag_r = B[:, :K*pp]                       # K x Kp
    lag_p = par[ndet:, :].T                   # K x Kp
    det_r = B[:, K*pp:] if ndet else np.zeros((K,0))
    det_p = par[:ndet, :].T if ndet else np.zeros((K,0))
    print(f"  {tag_r:14s} K={K} p={pp} ndet={ndet}  max|lag diff|={mx(lag_r,lag_p):.3e}"
          f"  max|det diff|={mx(det_r,det_p) if ndet else 0:.3e}  cols={r['Bcoef_colnames']}")

print("\n=== Sigma conventions ===")
for tag in ["A_p1_const","B_p2_const","US_p2_const","E5_p2_const","SH_p2_const"]:
    r = R[tag]; p = P[tag]
    print(f"  {tag:14s} |R Sigma_df - py sigma_u| = {mx(r['Sigma_df'], p['sigma_u']):.3e}"
          f"   |R Sigma_mle - py sigma_u_mle| = {mx(r['Sigma_mle'], p['sigma_u_mle']):.3e}"
          f"   |R Sigma_ols - py sigma_u| = {mx(r['Sigma_ols'], p['sigma_u']):.3e}")

print("\n=== log-likelihood, se, resid, fitted, roots, forecast ===")
for tag in ["A_p1_const","A_p2_const","B_p2_const","US_p2_const","E5_p2_const","SH_p2_const","A_p1_none"]:
    r = R[tag]; p = P[tag]
    se_r = np.concatenate([arr(v) for v in r["coef_se"].values()]) if isinstance(r["coef_se"],dict) else arr(r["coef_se"])
    par = arr(p["params"]); K=p["K"]; pp=p["p"]; ndet=par.shape[0]-K*pp
    se_p_raw = arr(p["stderr"])                       # (ndet+Kp) x K
    # R lists per-equation: [lag terms..., const]; py has det first -> reorder
    se_p = np.concatenate([np.concatenate([se_p_raw[ndet:, j], se_p_raw[:ndet, j]]) for j in range(K)])
    print(f"  {tag:14s} llf: R {r['logLik']:.10f}  py {p['llf']:.10f}  diff {abs(r['logLik']-p['llf']):.2e}"
          f" | max|se| {mx(se_r, se_p):.2e}"
          f" | resid_head {mx(r['resid_head'], p['resid_head']):.2e}"
          f" | fitted_head {mx(r['fitted_head'], p['fittedvalues_head']):.2e}"
          f" | roots {mx(sorted(arr(r['roots'])), sorted(arr(p['roots']))):.2e}"
          f" | fcst {mx(r['fcst_point'], p['forecast']):.2e}")

print("\n=== trend-type mapping (deliberate mismatch check) ===")
for rt, pt in [("A_p1_trend","A_p1_trend"),("A_p1_both","A_p1_trend"),("A_p1_both","A_p1_ctt")]:
    r=R[rt]; p=P[pt]
    K=p["K"]; pp=p["p"]; par=arr(p["params"]); ndet=par.shape[0]-K*pp
    print(f"  R type={r['type']:6s} vs py trend={p['trend']:4s}: R ncol={arr(r['Bcoef']).shape[1]} "
          f"py nrow={par.shape[0]} (ndet R={arr(r['Bcoef']).shape[1]-K*pp}, py={ndet})  "
          f"llf R {r['logLik']:.6f} py {p['llf']:.6f}  match={abs(r['logLik']-p['llf'])<1e-8}")

print("\n=== forecast intervals ===")
for tag in ["A_p1_const","US_p2_const"]:
    r=R[tag]; p=P[tag]
    lo_r=arr(r["fcst_lower"]); up_r=arr(r["fcst_upper"])
    mid,lo_p,up_p = [arr(x) for x in p["forecast_interval"]]
    print(f"  {tag:14s} max|lower| {float(np.max(np.abs(lo_r-lo_p))):.3e}  max|upper| {float(np.max(np.abs(up_r-up_p))):.3e}")

print("\n=== information criteria: VARselect vs select_order ===")
for rt, pt in [("sel_A_const","sel_A_const"),("sel_B_const","sel_B_const"),
               ("sel_US_const","sel_US_const"),("sel_E5_const","sel_E5_const"),("sel_A_none","sel_A_none")]:
    r=R[rt]; p=P[pt]
    crit=arr(r["criteria"]); names=r["rownames"]
    rmap={"AIC(n)":"aic","HQ(n)":"hqic","SC(n)":"bic","FPE(n)":"fpe"}
    print(f"  {rt}: R selection {r['selection']}  py selected {p['selected']}")
    for i,nm in enumerate(names):
        key=rmap[nm]; rv=crit[i]; pv=arr(p["ics"][key])
        print(f"      {nm:7s}->{key:5s} max|diff| {float(np.max(np.abs(rv-pv))):.4e}   R[1:3]={np.round(rv[:3],6)}  py[1:3]={np.round(pv[:3],6)}")
print("\nk=1 refusals:\n  R :", R["k1_refusal_R"], "\n  py:", P["k1_refusal_py"])
