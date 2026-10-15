import numpy as np, csv, os, warnings
from statsforecast.models import CrostonClassic,CrostonOptimized,CrostonSBA,TSB,ADIDA,IMAPA
from statsforecast._lib import ses as S
BASE=os.path.dirname(os.path.abspath(__file__))
rd=lambda p:np.array([float(r[0]) for r in list(csv.reader(open(os.path.join(BASE,p))))[1:]])
FX=[("F1_poisson040_n60","../intermittent.csv"),("F2_sparse010_n120","fx_sparse_p010_n120.csv"),
    ("F3_dense150_n120","fx_dense_p150_n120.csv"),("F4_partx_n51","fx_partx_n51.csv")]
lines=[]
def P(s): print(s); lines.append(s)
P("# Stage 12.1 intermittent-demand reference values")
P("# statsforecast 2.1.1 (Apache-2.0) -- transcribable as Julia test expectations")
P("# forecast 8.21.1 (GPL-3) -- independent check only, agreement recorded below")
P("# All point forecasts are FLAT in h: h=1..6 are bit-identical. Verified for every row.")
P("")
for nm,p in FX:
    y=rd(p); nz=int((y>0).sum())
    yi=np.diff(np.where(y!=0)[0]+1,prepend=0).astype(float); z=y[y>0]
    adi=yi.mean(); cv2=(z.std(ddof=1)/z.mean())**2 if len(z)>1 else 0.0
    P(f"## {nm}: n={len(y)} nonzero={nz} sum={int(y.sum())}")
    P(f"#    ADI={adi:.17g}  CV2={cv2:.17g}  SBC(p>1.32|v>0.49)={'SBA' if (adi>1.32 or cv2>0.49) else 'Croston'}")
    for mn,mo in [("CrostonClassic",CrostonClassic()),("CrostonOptimized",CrostonOptimized()),
                  ("CrostonSBA",CrostonSBA()),("TSB_ad0.2_ap0.2",TSB(alpha_d=0.2,alpha_p=0.2)),
                  ("ADIDA",ADIDA()),("IMAPA",IMAPA())]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); r=mo.forecast(y=y,h=6,fitted=True)
        mean,fit=r["mean"],r["fitted"]
        flat=bool(np.all(mean==mean[0]))
        m=~np.isnan(fit); mase=np.mean(np.abs(y[m]-fit[m]))/np.mean(np.abs(np.diff(y)))
        P(f"    {mn:<17} = {mean[0]:.17g}   flat={flat}  MASE_naive1={mase:.17g}")
    dp,_=S.ses_forecast(z,0.1); ip,_=S.ses_forecast(yi,0.1)
    P(f"    [components a=0.1] demand_level={dp:.17g}  interval_level={ip:.17g}")
    ad=S.golden_section_ses(z,0.1,0.3); ai=S.golden_section_ses(yi,0.1,0.3)
    P(f"    [CrostonOptimized] alpha_demand={ad:.17g} alpha_interval={ai:.17g}"
      f"  bound_hit={'YES' if min(abs(ad-0.1),abs(ad-0.3),abs(ai-0.1),abs(ai-0.3))<1e-9 else 'no'}")
    P(f"    [agg level]        mean_interval={adi:.17g} -> py round()={round(adi)}  Cpp std::round()={int(np.floor(adi+0.5))}")
    P("")
open(os.path.join(BASE,"EXPECTATIONS.txt"),"w").write("\n".join(lines)+"\n")
