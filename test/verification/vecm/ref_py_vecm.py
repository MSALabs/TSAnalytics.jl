import json, numpy as np, csv, statsmodels, warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.vector_ar.vecm import VECM
rd=lambda p: np.array([[float(c) for c in r] for r in list(csv.reader(open(p)))[1:]])
out={}
def gr(fx,k,r,det,tag):
    Y=rd("fixtures/"+fx)
    m=VECM(Y, k_ar_diff=k, coint_rank=r, deterministic=det).fit()
    d={"fx":fx,"k_ar_diff":k,"coint_rank":r,"deterministic":det,"K":int(Y.shape[1]),
       "alpha":m.alpha.tolist(),"beta":m.beta.tolist(),
       "gamma":m.gamma.tolist(),"sigma_u":m.sigma_u.tolist(),
       "nobs":int(m.nobs),"llf":float(m.llf),
       "stderr_alpha":m.stderr_alpha.tolist(),"stderr_beta":m.stderr_beta.tolist(),
       "stderr_gamma":m.stderr_gamma.tolist(),
       "resid_head":m.resid[:2].tolist(),
       "var_rep":m.var_rep.tolist(),
       "predict":m.predict(steps=4).tolist()}
    try: d["det_coef"]=m.det_coef.tolist()
    except Exception as e: d["det_coef"]=f"{type(e).__name__}"
    try: d["det_coef_coint"]=m.det_coef_coint.tolist()
    except Exception as e: d["det_coef_coint"]=f"{type(e).__name__}"
    out[tag]=d
for a in [("coint2_yes.csv",1,1,"n","yes_k1_r1_n"),("coint2_yes.csv",1,1,"co","yes_k1_r1_co"),
          ("coint2_yes.csv",1,1,"ci","yes_k1_r1_ci"),
          ("coint3_r1.csv",1,1,"n","r1_k1_r1_n"),("coint3_r1.csv",1,2,"n","r1_k1_r2_n"),
          ("coint3_r2.csv",1,2,"n","r2_k1_r2_n"),("coint2_yes.csv",2,1,"n","yes_k2_r1_n")]:
    gr(*a)
Y=rd("fixtures/coint2_yes.csv")
for nm,fn in [("err_r0",lambda: VECM(Y,k_ar_diff=1,coint_rank=0).fit()),
              ("err_rK",lambda: VECM(Y,k_ar_diff=1,coint_rank=2).fit()),
              ("err_rK1",lambda: VECM(Y,k_ar_diff=1,coint_rank=3).fit()),
              ("err_k0",lambda: VECM(Y,k_ar_diff=0,coint_rank=1).fit()),
              ("err_baddet",lambda: VECM(Y,k_ar_diff=1,coint_rank=1,deterministic="bogus").fit())]:
    try:
        v=fn(); out[nm]=f"no error; beta={np.asarray(v.beta).ravel().tolist()[:4]}"
    except Exception as e: out[nm]=f"{type(e).__name__}: {e}"
out["meta"]={"statsmodels":statsmodels.__version__}
json.dump(out,open("fixtures/ref_py_vecm.json","w"),indent=1)
print("vecm py refs written")
