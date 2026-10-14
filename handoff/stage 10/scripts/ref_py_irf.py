import json, numpy as np, csv, statsmodels
from statsmodels.tsa.api import VAR
rd=lambda p: np.array([[float(c) for c in r] for r in list(csv.reader(open(p)))[1:]])
out={}

def grab(fx, p, h, tag, names=None):
    Y=rd("fixtures/"+fx); K=Y.shape[1]
    r=VAR(Y).fit(p, trend="c")
    ir=r.irf(h)
    d={"K":K,"p":p,"h":h,"nobs":int(r.nobs),
       "ma_rep":r.ma_rep(h).tolist(),            # (h+1, K, K) non-orth
       "orth_ma_rep":r.orth_ma_rep(h).tolist(),  # (h+1, K, K) orth
       "irfs":ir.irfs.tolist(),                  # [h, response, impulse]
       "orth_irfs":ir.orth_irfs.tolist(),
       "cum_effects":ir.cum_effects.tolist(),
       "orth_cum_effects":ir.orth_cum_effects.tolist(),
       "lr_effects":ir.lr_effects.tolist(),
       "orth_lr_effects":ir.orth_lr_effects.tolist(),
       "fevd_decomp":r.fevd(h).decomp.tolist(),  # [variable, horizon, shock]
       "sigma_u":r.sigma_u.tolist(),
       "chol_lower":np.linalg.cholesky(r.sigma_u).tolist(),
       # ANALYTIC standard errors -- statsmodels is the only reference for these
       "stderr":ir.stderr().tolist(),            # (h+1, K, K) non-orth
       "stderr_orth":ir.stderr(orth=True).tolist(),
       "cum_effect_stderr":ir.cum_effect_stderr().tolist(),
       "lr_effect_stderr":ir.lr_effect_stderr().tolist(),
       }
    if names: d["names"]=names
    out[tag]=d

grab("var2_p1.csv",1,6,"A_p1_h6",["y1","y2"])
grab("var2_p1.csv",2,8,"A_p2_h8",["y1","y2"])
grab("var3_p2.csv",2,6,"B_p2_h6",["x1","x2","x3"])
grab("uschange.csv",2,6,"US_p2_h6")
grab("econ5.csv",2,4,"E5_p2_h4")
grab("var2_short.csv",2,4,"SH_p2_h4",["y1","y2"])

# Cholesky-ordering dependence
Yp=rd("fixtures/var2_p1.csv")[:, [1,0]]
rp=VAR(Yp).fit(2, trend="c"); irp=rp.irf(4)
out["A_p2_permuted"]={"names":["y2","y1"],"orth_irfs":irp.orth_irfs.tolist(),
                      "irfs":irp.irfs.tolist(),"fevd_decomp":rp.fevd(4).decomp.tolist()}

A=rd("fixtures/var2_p1.csv"); r=VAR(A).fit(2,trend="c")
for nm,fn in [("err_neg_ahead",lambda: r.irf(-1)),("err_zero_ahead",lambda: r.irf(0)),
              ("err_fevd_zero",lambda: r.fevd(0))]:
    try:
        v=fn(); out[nm]=f"no error, shape {getattr(getattr(v,'irfs',None),'shape',None) or getattr(v,'decomp',np.array([])).shape}"
    except Exception as e: out[nm]=f"{type(e).__name__}: {e}"
out["meta"]={"statsmodels":statsmodels.__version__}
json.dump(out, open("fixtures/ref_py_irf.json","w"), indent=1)
print("irf py refs written")
