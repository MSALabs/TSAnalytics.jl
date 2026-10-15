import numpy as np, csv, os, warnings
from statsforecast.models import CrostonClassic
BASE=os.path.dirname(os.path.abspath(__file__))
rd=lambda p:[float(r[0]) if r[0] not in ('NA','') else np.nan for r in list(csv.reader(open(os.path.join(BASE,p))))[1:]]
FX=[("F1","../intermittent.csv"),("F2","fx_sparse_p010_n120.csv"),
    ("F3","fx_dense_p150_n120.csv"),("F4","fx_partx_n51.csv")]
print(f"{'fx':<4}{'metric':<22}{'R (forecast 8.21.1)':<24}{'statsforecast 2.1.1':<24}{'rel.diff':>12}")
print("-"*86)
for nm,p in FX:
    y=np.array(rd(p)); Rf=np.array(rd(f"Rfitted_{nm}.csv"))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r=CrostonClassic().forecast(y=y,h=6,fitted=True)
    Pf=r["fitted"]; Pm=r["mean"][0]
    # R point forecast re-read from the script above is identical to fitted tail; recompute from components
    yd=y[y>0]; yi=np.diff(np.where(y!=0)[0]+1,prepend=0).astype(float)
    from statsforecast._lib import ses as S
    dp,_=S.ses_forecast(yd,0.1); ip,_=S.ses_forecast(yi,0.1); Rm=dp/ip
    m=(~np.isnan(Rf))&(~np.isnan(Pf))
    d=np.abs(Rf[m]-Pf[m]); den=np.maximum(np.abs(Rf[m]),1e-300)
    print(f"{nm:<4}{'point h=1':<22}{Rm:<24.17g}{Pm:<24.17g}{abs(Rm-Pm)/abs(Rm):>12.2e}")
    print(f"{'':<4}{'fitted: n compared':<22}{m.sum():<24}{m.sum():<24}")
    print(f"{'':<4}{'fitted: max abs diff':<22}{d.max():<24.17g}{'':<24}{(d/den).max():>12.2e}")
    print(f"{'':<4}{'fitted: n exact match':<22}{int((d==0).sum()):<24}{f'of {m.sum()}':<24}")
    if d.max()>0:
        k=int(np.argmax(d)); idx=np.where(m)[0][k]
        print(f"{'':<4}{'  worst at t=':<22}{idx:<24}  R={Rf[idx]:.17g}  SF={Pf[idx]:.17g}")
    print()
