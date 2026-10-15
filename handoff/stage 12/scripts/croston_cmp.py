import numpy as np, warnings; warnings.filterwarnings("ignore")
from statsforecast.models import CrostonClassic, CrostonOptimized, CrostonSBA, TSB, ADIDA, IMAPA
y = np.loadtxt("intermittent.csv", skiprows=1).astype(float)
print("n=%d nonzero=%d sum=%.1f" % (len(y), (y>0).sum(), y.sum()))
for M, kw in [(CrostonClassic,{}), (CrostonOptimized,{}), (CrostonSBA,{}),
              (TSB,{"alpha_d":0.2,"alpha_p":0.2}), (ADIDA,{}), (IMAPA,{})]:
    try:
        m = M(**kw).fit(y)
        f = m.predict(h=3)["mean"]
        print("  %-18s %.12f" % (M.__name__, f[0]))
    except Exception as e:
        print("  %-18s FAILED %s" % (M.__name__, str(e)[:50]))
