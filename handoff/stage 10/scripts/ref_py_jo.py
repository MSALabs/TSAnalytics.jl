import json, numpy as np, csv, statsmodels, warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank
from statsmodels.tsa.coint_tables import c_sja, c_sjt
rd=lambda p: np.array([[float(c) for c in r] for r in list(csv.reader(open(p)))[1:]])
out={}
def jo(fx, k_ar_diff, det_order, tag):
    Y=rd("fixtures/"+fx)
    j=coint_johansen(Y, det_order, k_ar_diff)
    out[tag]={"fx":fx,"k_ar_diff":k_ar_diff,"det_order":det_order,"K":int(Y.shape[1]),
              "lr1_trace":[float(x) for x in j.lr1],"lr2_maxeig":[float(x) for x in j.lr2],
              "cvt":[[float(y) for y in r] for r in j.cvt],
              "cvm":[[float(y) for y in r] for r in j.cvm],
              "eig":[float(x) for x in j.eig],
              "evec_shape":list(np.asarray(j.evec).shape),
              "evec":[[float(y) for y in r] for r in np.asarray(j.evec)]}
for a in [("coint2_yes.csv",1,-1,"yes_d-1"),("coint2_yes.csv",1,0,"yes_d0"),
          ("coint2_yes.csv",1,1,"yes_d1"),("coint2_yes.csv",2,0,"yes_d0_k2"),
          ("coint2_no.csv",1,0,"no_d0"),("coint3_r1.csv",1,0,"r1_d0"),
          ("coint3_r1.csv",1,-1,"r1_d-1"),("coint3_r2.csv",1,0,"r2_d0"),
          ("uschange.csv",1,0,"US_d0"),("econ5.csv",1,0,"E5_d0")]:
    jo(*a)
# the critical-value tables themselves
tabs={}
for n in range(1,14):
    for p in (-1,0,1,2):
        try:
            a=c_sja(n,p); t=c_sjt(n,p)
            tabs[f"n{n}_p{p}"]={"maxeig":[float(x) for x in a],"trace":[float(x) for x in t],
                                "allnan":bool(np.all(np.isnan(a)))}
        except Exception as e:
            tabs[f"n{n}_p{p}"]=f"{type(e).__name__}: {e}"
out["sj_tables"]=tabs
# rank selection
Y=rd("fixtures/coint3_r1.csv")
for meth in ("trace","maxeig"):
    r=select_coint_rank(Y, det_order=0, k_ar_diff=1, method=meth, signif=0.05)
    out[f"rank_{meth}"]={"rank":int(r.rank),"test_stats":[float(x) for x in r.test_stats],
                         "crit_vals":[float(x) for x in r.crit_vals]}
for nm,fn in [("err_onecol",lambda: coint_johansen(Y[:,0:1],0,1)),
              ("err_k0",lambda: coint_johansen(Y,0,0)),
              ("err_d2",lambda: coint_johansen(Y,2,1)),
              ("err_13col",lambda: coint_johansen(np.cumsum(np.random.default_rng(1).standard_normal((250,13)),axis=0),0,1))]:
    try:
        v=fn()
        extra=""
        if hasattr(v,"cvt"): extra=f" cvt[0]={np.asarray(v.cvt)[0]}"
        out[nm]=f"no error{extra}"
    except Exception as e: out[nm]=f"{type(e).__name__}: {e}"
out["meta"]={"statsmodels":statsmodels.__version__}
json.dump(out,open("fixtures/ref_py_johansen.json","w"),indent=1)
print("johansen py refs written")
