import numpy as np, warnings, csv, os, traceback
from statsforecast.models import CrostonClassic, CrostonSBA, TSB, ADIDA, IMAPA, CrostonOptimized
from statsforecast.utils import ConformalIntervals
from statsforecast._lib import ses as S
BASE=os.path.dirname(os.path.abspath(__file__))
y=np.array([float(r[0]) for r in list(csv.reader(open(os.path.join(BASE,'../intermittent.csv'))))[1:]])
g=lambda v: f"{v:.17g}"

print("=== A. plain forecast with level= (no prediction_intervals set) ===")
for nm,M in [("CrostonClassic",CrostonClassic),("CrostonSBA",CrostonSBA),("ADIDA",ADIDA),("IMAPA",IMAPA)]:
    try:
        r=M().forecast(y=y,h=6,level=[80,95])
        print(f" {nm:<16} keys={sorted(r)}")
    except Exception as e:
        print(f" {nm:<16} RAISED {type(e).__name__}: {e}")

try:
    r=TSB(alpha_d=0.2,alpha_p=0.2).forecast(y=y,h=6,level=[80,95]); print(" TSB              keys=",sorted(r))
except Exception as e:
    print(" TSB              RAISED",type(e).__name__,e)

print("\n=== B. with ConformalIntervals ===")
ci=ConformalIntervals(h=6,n_windows=4)
import inspect
print(" ConformalIntervals signature:", inspect.signature(ConformalIntervals.__init__))
print(" ConformalIntervals source lines:", len(inspect.getsource(ConformalIntervals).splitlines()))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m=CrostonClassic(prediction_intervals=ci)
    r=m.forecast(y=y,h=6,level=[80,95])
print(" keys:",sorted(r))
for k in sorted(r): print(f"   {k:<10} {[g(v) for v in np.atleast_1d(r[k])]}")

print("\n=== C. ConformalIntervals mechanism (is it simulation or empirical quantile?) ===")
src=inspect.getsource(ConformalIntervals); print(src)
