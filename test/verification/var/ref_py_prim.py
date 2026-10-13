import json, numpy as np, csv, statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar import util
rd=lambda p: np.array([[float(c) for c in r] for r in list(csv.reader(open(p)))[1:]])
out={}

def mols(fx,p,tag):
    Y=rd("fixtures/"+fx); n,K=Y.shape
    Z=util.get_var_endog(Y, p, trend="c")        # statsmodels' own regressor builder
    X=Y[p:]
    B,*_=np.linalg.lstsq(Z,X,rcond=None)
    E=X-Z@B; T=X.shape[0]
    r=VAR(Y).fit(p,trend="c")
    out[tag]={"K":K,"p":p,"T":int(T),"Z_dim":list(Z.shape),
              "Z_head":Z[:3].tolist(),"Z_tail":Z[-2:].tolist(),
              "B":B.tolist(),            # (ndet+Kp) x K, det FIRST
              "sigma_df":(E.T@E/(T-Z.shape[1])).tolist(),
              "sigma_mle":(E.T@E/T).tolist(),
              "resid_head":E[:2].tolist(),
              "params_from_fit":r.params.tolist(),
              "stderr_from_fit":r.stderr.tolist(),
              "B_matches_fit":bool(np.max(np.abs(B-r.params))<1e-9)}
for a in [("var2_p1.csv",1,"mols_A_p1"),("var2_p1.csv",2,"mols_A_p2"),
          ("var3_p2.csv",2,"mols_B_p2"),("uschange.csv",2,"mols_US_p2")]:
    mols(*a)

def comp(fx,p,tag):
    Y=rd("fixtures/"+fx); r=VAR(Y).fit(p,trend="c")
    C=util.comp_matrix(r.coefs)
    ev=np.linalg.eigvals(C)
    out[tag]={"companion":C.tolist(),"eig_mod":sorted(np.abs(ev).tolist(),reverse=True),
              "sm_roots_abs":sorted(np.abs(r.roots).tolist()),
              "is_stable":bool(r.is_stable())}
for a in [("var2_p1.csv",1,"comp_A_p1"),("var2_p1.csv",2,"comp_A_p2"),
          ("var3_p2.csv",2,"comp_B_p2"),("var2_p1.csv",4,"comp_A_p4"),
          ("uschange.csv",2,"comp_US_p2")]:
    comp(*a)

# trend variants of the regressor matrix: which column order?
Y=rd("fixtures/var2_p1.csv")
for tr in ("n","c","ct","ctt"):
    Z=util.get_var_endog(Y,1,trend=tr)
    out["Zlayout_"+tr]={"shape":list(Z.shape),"head":Z[:3].tolist()}

# collinear input
Ac=np.column_stack([Y, Y[:,0]])
try:
    r=VAR(Ac).fit(1,trend="c")
    out["err_collinear"]=f"no error; any NaN in params: {bool(np.isnan(r.params).any())}; cond(Z'Z) huge"
except Exception as e:
    out["err_collinear"]=f"{type(e).__name__}: {e}"
out["meta"]={"statsmodels":statsmodels.__version__}
json.dump(out,open("fixtures/ref_py_prim.json","w"),indent=1)
print("primitive py refs written")
