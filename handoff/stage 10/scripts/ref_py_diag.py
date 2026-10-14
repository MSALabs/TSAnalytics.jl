import json, numpy as np, csv, statsmodels
from statsmodels.tsa.api import VAR
rd=lambda p: np.array([[float(c) for c in r] for r in list(csv.reader(open(p)))[1:]])
out={}
def grab(fx,p,tag,lags=8):
    Y=rd("fixtures/"+fx); r=VAR(Y).fit(p,trend="c")
    d={"K":int(Y.shape[1]),"p":p,"nobs":int(r.nobs),"lags":lags}
    for adj in (False,True):
        try:
            w=r.test_whiteness(nlags=lags, adjusted=adj)
            d[f"whiteness_adj{int(adj)}"]={"stat":float(w.test_statistic),"p":float(w.pvalue),
                                          "df":int(w.df),"crit":float(w.crit_value)}
        except Exception as e:
            d[f"whiteness_adj{int(adj)}"]={"error":f"{type(e).__name__}: {e}"}
    try:
        nt=r.test_normality()
        d["normality"]={"stat":float(nt.test_statistic),"p":float(nt.pvalue),"df":int(nt.df)}
    except Exception as e:
        d["normality"]={"error":f"{type(e).__name__}: {e}"}
    out[tag]=d
for a in [("var2_p1.csv",2,"A_p2",8),("var2_p1.csv",1,"A_p1",8),("var3_p2.csv",2,"B_p2",8),
          ("uschange.csv",2,"US_p2",10),("econ5.csv",2,"E5_p2",6),
          ("var2_short.csv",2,"SH_p2",4),("climhyd.csv",2,"C_p2",8)]:
    grab(*a)
Y=rd("fixtures/var2_p1.csv"); r=VAR(Y).fit(2,trend="c")
for nm,fn in [("err_lags_le_p",lambda: r.test_whiteness(nlags=2)),
              ("err_lags_zero",lambda: r.test_whiteness(nlags=0)),
              ("err_lags_huge",lambda: r.test_whiteness(nlags=300))]:
    try:
        w=fn(); out[nm]=f"no error, df={w.df}, stat={w.test_statistic}"
    except Exception as e: out[nm]=f"{type(e).__name__}: {e}"
out["meta"]={"statsmodels":statsmodels.__version__}
json.dump(out,open("fixtures/ref_py_diag.json","w"),indent=1)
print("diag py refs written")
