import json, numpy as np, csv, statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
rd=lambda p: np.array([[float(c) for c in r] for r in list(csv.reader(open(p)))[1:]])
NAMES={"var2_p1.csv":["y1","y2"],"var3_p2.csv":["x1","x2","x3"],"var2_short.csv":["y1","y2"],
       "uschange.csv":["Consumption","Income","Production","Savings","Unemployment"],
       "econ5.csv":["unemp","gnp","consum","govinv","prinv"]}
out={}

def gr(fx, p, cause, tag):
    Y=rd("fixtures/"+fx); nm=NAMES[fx]
    r=VAR(Y).fit(p, trend="c")
    causing=[nm.index(c) for c in cause]
    caused=[i for i in range(len(nm)) if i not in causing]
    f=r.test_causality(caused, causing, kind="f")
    w=r.test_causality(caused, causing, kind="wald")
    try:
        ic=r.test_inst_causality(causing)
        inst={"inst_stat":float(ic.test_statistic),"inst_p":float(ic.pvalue),"inst_df":int(ic.df)}
    except Exception as e:
        inst={"inst_error":f"{type(e).__name__}: {e}"}
    out[tag]={"K":len(nm),"p":p,"cause":cause,"nobs":int(r.nobs),
              "df_model":int(r.df_model),"df_resid":int(r.df_resid),
              "f_stat":float(f.test_statistic),"f_p":float(f.pvalue),
              "f_df":[int(x) for x in f.df],"f_crit":float(f.crit_value),
              "wald_stat":float(w.test_statistic),"wald_p":float(w.pvalue),
              "wald_df":int(w.df),"wald_crit":float(w.crit_value),
              **inst}

for args in [("var2_p1.csv",1,["y2"],"A_p1_cause_y2"),("var2_p1.csv",1,["y1"],"A_p1_cause_y1"),
             ("var2_p1.csv",2,["y2"],"A_p2_cause_y2"),("var2_p1.csv",2,["y1"],"A_p2_cause_y1"),
             ("var2_p1.csv",4,["y2"],"A_p4_cause_y2"),
             ("var3_p2.csv",2,["x1"],"B_p2_cause_x1"),("var3_p2.csv",2,["x2"],"B_p2_cause_x2"),
             ("var3_p2.csv",2,["x3"],"B_p2_cause_x3"),
             ("var3_p2.csv",2,["x1","x2"],"B_p2_cause_x1x2"),
             ("var3_p2.csv",2,["x2","x3"],"B_p2_cause_x2x3"),
             ("var2_short.csv",2,["y2"],"SH_p2_cause_y2"),
             ("uschange.csv",2,["Income"],"US_p2_cause_Income"),
             ("uschange.csv",2,["Income","Production"],"US_p2_cause_IncProd"),
             ("uschange.csv",2,["Savings"],"US_p2_cause_Savings"),
             ("econ5.csv",2,["gnp"],"E5_p2_cause_gnp")]:
    gr(*args)

# grangercausalitytests: pairwise, column order [caused, causing]
def gct(fx, i_caused, i_causing, lags, tag):
    Y=rd("fixtures/"+fx)
    res=grangercausalitytests(np.column_stack([Y[:,i_caused],Y[:,i_causing]]), maxlag=lags)
    d={}
    for lag,v in res.items():
        d[str(lag)]={k:[float(x) for x in t] for k,t in v[0].items()}
    out[tag]=d
gct("var2_p1.csv",0,1,[1,2,4],"gct_A_y2_to_y1")
gct("var2_p1.csv",1,0,[2],"gct_A_y1_to_y2")
gct("uschange.csv",0,1,[2],"gct_US_Inc_to_Cons")
gct("var3_p2.csv",2,0,[2],"gct_B_x1_to_x3")

A=rd("fixtures/var2_p1.csv"); r=VAR(A).fit(2,trend="c")
for nm,fn in [("err_cause_all",lambda: r.test_causality([0,1],[0,1],kind="f")),
              ("err_cause_bad",lambda: r.test_causality(0,[5],kind="f")),
              ("err_cause_overlap",lambda: r.test_causality([0],[0],kind="f")),
              ("err_bad_kind",lambda: r.test_causality(0,[1],kind="bogus"))]:
    try: fn(); out[nm]="no error"
    except Exception as e: out[nm]=f"{type(e).__name__}: {e}"
out["meta"]={"statsmodels":statsmodels.__version__}
json.dump(out, open("fixtures/ref_py_granger.json","w"), indent=1)
print("granger py refs written")
