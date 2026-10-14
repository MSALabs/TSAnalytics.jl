import json, numpy as np, csv, statsmodels, warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.vector_ar.svar_model import SVAR
rd=lambda p: np.array([[float(c) for c in r] for r in list(csv.reader(open(p)))[1:]])
out={}
Y=rd("fixtures/var2_p1.csv"); Y3=rd("fixtures/var3_p2.csv")
def fit(Y, svar_type, A=None, B=None, tag=None, p=2):
    try:
        m=SVAR(Y, svar_type=svar_type, A=A, B=B)
        r=m.fit(maxlags=p, trend="c", solver="bfgs", maxiter=1000)
        d={"svar_type":svar_type,"A":np.asarray(r.A).tolist(),"B":np.asarray(r.B).tolist(),
           "sigma_u":r.sigma_u.tolist(),"llf":float(r.llf),"nobs":int(r.nobs)}
        try: d["A_mask"]=np.asarray(m.A_mask).tolist()
        except Exception: pass
        try: d["irf_orth"]=r.irf(4).svar_irfs.tolist()
        except Exception as e: d["irf_orth"]=f"{type(e).__name__}: {e}"
        out[tag]=d
    except Exception as e:
        out[tag]=f"{type(e).__name__}: {e}"
# statsmodels marks FREE parameters with the STRING "E", not np.nan --
# A.astype("U") then A == "E".  np.nan yields an all-False mask and an
# empty-array error.  Source: svar_model.py SVAR.__init__.
A2=np.array([["1","0"],["E","1"]], dtype=object)
B2=np.array([["1","0"],["E","1"]], dtype=object)
fit(Y,"A",A=A2,tag="A_k2")
fit(Y,"B",B=B2,tag="B_k2")
fit(Y,"AB",A=A2,B=np.array([["1","0"],["0","1"]],dtype=object),tag="AB_k2")
A3=np.array([["1","0","0"],["E","1","0"],["E","E","1"]], dtype=object)
fit(Y3,"A",A=A3,tag="A_k3")
for nm,fn in [("err_noA",lambda: SVAR(Y,svar_type="A").fit(maxlags=2)),
              ("err_allfree",lambda: SVAR(Y,svar_type="A",A=np.full((2,2),"E",dtype=object)).fit(maxlags=2)),
              ("err_nan_marker",lambda: SVAR(Y,svar_type="A",A=np.array([[1.0,0.0],[np.nan,1.0]])).fit(maxlags=2)),
              ("err_badtype",lambda: SVAR(Y,svar_type="bogus",A=A2).fit(maxlags=2))]:
    try:
        v=fn(); out[nm]=f"no error; A={np.asarray(v.A).tolist()}"
    except Exception as e: out[nm]=f"{type(e).__name__}: {e}"
out["meta"]={"statsmodels":statsmodels.__version__}
json.dump(out,open("fixtures/ref_py_svar.json","w"),indent=1)
print("svar py refs written")
