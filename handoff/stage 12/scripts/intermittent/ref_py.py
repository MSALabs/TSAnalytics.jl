import numpy as np, warnings, csv, os
np.set_printoptions(precision=17, floatmode='maxprec')
from statsforecast.models import (CrostonClassic, CrostonOptimized, CrostonSBA,
                                  TSB, ADIDA, IMAPA)
from statsforecast._lib import ses as S

BASE=os.path.dirname(os.path.abspath(__file__))
def rd(p):
    return np.array([float(r[0]) for r in list(csv.reader(open(os.path.join(BASE,p))))[1:]])

FX = [("F1_poisson040_n60", rd("../intermittent.csv")),
      ("F2_sparse010_n120", rd("fx_sparse_p010_n120.csv")),
      ("F3_dense150_n120",  rd("fx_dense_p150_n120.csv")),
      ("F4_partx_n51",      rd("fx_partx_n51.csv"))]

def models():
    return [("CrostonClassic", CrostonClassic()), ("CrostonOptimized", CrostonOptimized()),
            ("CrostonSBA", CrostonSBA()), ("TSB(0.2,0.2)", TSB(alpha_d=0.2, alpha_p=0.2)),
            ("ADIDA", ADIDA()), ("IMAPA", IMAPA())]

def g(v): return f"{v:.17g}"

def mase(y, fit, sp=1):
    m = ~np.isnan(fit)
    denom = np.mean(np.abs(y[sp:] - y[:-sp]))
    return np.mean(np.abs(y[m]-fit[m]))/denom

for nm, y in FX:
    nz=int((y>0).sum())
    print("\n"+"#"*78)
    print(f"### {nm}  n={len(y)} nonzero={nz} sum={int(y.sum())} dtype_in={y.dtype}")
    for mn, mo in models():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mo.forecast(y=y, h=6, fitted=True)
        mean, fit = r["mean"], r["fitted"]
        uniq = len(set(np.round(mean,15)))
        print(f"  {mn:<18} h1={g(mean[0])}  flat={uniq==1}  mean.dtype={mean.dtype}")
        if uniq!=1: print(f"      h1..h6: {[g(v) for v in mean]}")
        print(f"      MASE(sp=1)={g(mase(y,fit))}  nNaN_fitted={int(np.isnan(fit).sum())}"
              f"  fit[1:5]={[g(v) for v in fit[1:5]]}")
    # exposed levels
    yd = y[y>0]; yi = np.diff(np.where(y!=0)[0]+1, prepend=0).astype(float)
    dp, df_ = S.ses_forecast(yd, 0.1); ip, if_ = S.ses_forecast(yi, 0.1)
    print(f"  [levels@a=0.1] demand_fcst={g(dp)}  interval_fcst={g(ip)}  ratio={g(dp/ip)}")
    ad = S.golden_section_ses(yd,0.1,0.3); ai = S.golden_section_ses(yi,0.1,0.3)
    dpo,_ = S.ses_forecast(yd, ad); ipo,_ = S.ses_forecast(yi, ai)
    print(f"  [optimized]    alpha_d={g(ad)} alpha_i={g(ai)} demand={g(dpo)} interval={g(ipo)} ratio={g(dpo/ipo)}")
    mi = yi.mean(); print(f"  [agg]  mean_interval={g(mi)}  round()={round(mi)}  np.round={np.round(mi)}")
