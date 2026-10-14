import json, numpy as np, csv, statsmodels
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit
rd=lambda p: np.array([[float(c) for c in r] for r in list(csv.reader(open(p)))[1:]])
out={}

def eg(fx, trend, tag, maxlag=None, autolag="aic"):
    Y=rd("fixtures/"+fx)
    t,p,cv=coint(Y[:,0], Y[:,1:], trend=trend, method="aeg", maxlag=maxlag, autolag=autolag)
    out[tag]={"fx":fx,"trend":trend,"N":int(Y.shape[1]),"autolag":autolag,"maxlag":maxlag,
              "stat":float(t),"p":float(p),"crit":[float(c) for c in cv]}
for a in [("coint2_yes.csv","c","eg_yes_c"),("coint2_no.csv","c","eg_no_c"),
          ("coint3_r1.csv","c","eg_r1_c"),("coint3_r2.csv","c","eg_r2_c"),
          ("coint2_trend.csv","ct","eg_trend_ct"),("coint2_yes.csv","n","eg_yes_n"),
          ("coint2_yes.csv","ct","eg_yes_ct")]:
    eg(*a)
# fixed lag, no autolag -- reproducible without the selection path
eg("coint2_yes.csv","c","eg_yes_c_lag4", maxlag=4, autolag=None)
eg("coint2_no.csv","c","eg_no_c_lag4", maxlag=4, autolag=None)

# MacKinnon tables, the BSD-licensed transcription source
tabs={}
for reg in ("n","c","ct","ctt"):
    for N in range(1,13):
        key=f"{reg}_N{N}"
        d={}
        try: d["crit_n250"]=[float(x) for x in mackinnoncrit(N=N, regression=reg, nobs=250)]
        except Exception as e: d["crit_n250"]=f"{type(e).__name__}: {e}"
        try: d["crit_asym"]=[float(x) for x in mackinnoncrit(N=N, regression=reg, nobs=np.inf)]
        except Exception as e: d["crit_asym"]=f"{type(e).__name__}: {e}"
        pv=[]
        for s in (-5.0,-4.0,-3.5,-3.3670,-3.0,-2.0,-1.0,0.0):
            try: pv.append(float(mackinnonp(s, regression=reg, N=N)))
            except Exception as e: pv.append(f"{type(e).__name__}: {e}")
        d["pvals"]=pv
        tabs[key]=d
out["mackinnon_tables"]=tabs
out["mackinnon_stats_grid"]=[-5.0,-4.0,-3.5,-3.3670,-3.0,-2.0,-1.0,0.0]

Y=rd("fixtures/coint2_yes.csv")
for nm,fn in [("err_one_series",lambda: coint(Y[:,0], Y[:,0:0], trend="c")),
              ("err_bad_trend",lambda: coint(Y[:,0], Y[:,1], trend="bogus")),
              ("err_N13",lambda: mackinnonp(-3.0, regression="c", N=13)),
              ("err_N7_pval",lambda: mackinnonp(-3.0, regression="c", N=7)),
              ("err_n_crit_N2",lambda: mackinnoncrit(N=2, regression="n", nobs=250))]:
    try:
        v=fn(); out[nm]=f"no error -> {v}"
    except Exception as e: out[nm]=f"{type(e).__name__}: {e}"
out["meta"]={"statsmodels":statsmodels.__version__}
json.dump(out,open("fixtures/ref_py_coint.json","w"),indent=1)
print("coint py refs written")
