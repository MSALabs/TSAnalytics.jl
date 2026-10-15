import numpy as np, warnings, csv, os
from statsforecast.models import CrostonClassic, ADIDA, IMAPA, _adida, _imapa
from statsforecast._lib import ses as S
BASE=os.path.dirname(os.path.abspath(__file__))
y=np.array([float(r[0]) for r in list(csv.reader(open(os.path.join(BASE,'../intermittent.csv'))))[1:]])
g=lambda v: f"{v:.17g}"
yi=np.diff(np.where(y!=0)[0]+1, prepend=0).astype(float)
mi=yi.mean()
print("=== 1. ADIDA/IMAPA aggregation-level rounding on F1 (mean_interval is exactly 2.5) ===")
print(f" mean_interval = {g(mi)}")
print(f" python round(2.5)   = {round(mi)}       <- used by _adida/_imapa 'mean' path (banker's)")
print(f" C std::round(2.5)   = 3 (half away from zero) <- used by imapa_fitted_vals")
print(f" ADIDA h1 with agg=2 : {g(S.chunk_forecast(y,2)/2)}")
print(f" ADIDA h1 with agg=3 : {g(S.chunk_forecast(y,3)/3)}")
print(f" ADIDA.forecast h1   : {g(ADIDA().forecast(y=y,h=1)['mean'][0])}  -> agg=2 confirmed")
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    fa=_adida(y=y,h=1,fitted=True)['fitted']; fi=_imapa(y=y,h=1,fitted=True)['fitted']
print(f" last fitted ADIDA={g(fa[-1])}  IMAPA={g(fi[-1])}  (fitted path uses std::round)")

print("\n=== 2. IMAPA mean-path vs fitted-path shortcut divergence ===")
print(" _imapa mean uses _chunk_sums+_optimized_ses_forecast; fitted uses C++ chunk_forecast")
print(" C++ chunk_forecast has shortcuts: n_cut<agg -> y[n-1]; n_chunks<=1 -> agg_sums[0]")
print(" python _imapa mean path has NO such shortcut -> _optimized_ses_forecast on a 1-elem array")
for n in [3,5]:
    ys=y[:n]
    try:
        a=_imapa(y=ys,h=1,fitted=False)['mean'][0]
    except Exception as e: a=f"RAISED {type(e).__name__}"
    print(f"  n={n}: _imapa mean = {a}   C++ imapa_fitted_vals[-1] = {g(S.imapa_fitted_vals(y[:n+1])[-1])}")

print("\n=== 3. negative values: does statsforecast reject? (R does) ===")
yn=np.array([1.,-1.,0.,2.])
for nm,m in [("CrostonClassic",CrostonClassic()),("ADIDA",ADIDA()),("IMAPA",IMAPA())]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            print(f"  {nm:<16} h1={g(m.forecast(y=yn,h=1)['mean'][0])}  (NO negativity check)")
    except Exception as e: print(f"  {nm:<16} RAISED {type(e).__name__}: {e}")
print("  NOTE _demand uses y>0 but _intervals/_probability use y!=0 -> diverge only on negatives")

print("\n=== 4. dtype: int input -> float32 cast in _ensure_float ===")
yint=y.astype(np.int64)
print(f"  int64 input  CrostonClassic h1 = {g(CrostonClassic().forecast(y=yint,h=1)['mean'][0])}")
print(f"  float64 input CrostonClassic h1 = {g(CrostonClassic().forecast(y=y,h=1)['mean'][0])}")
print(f"  float32 input CrostonClassic h1 = {g(CrostonClassic().forecast(y=y.astype(np.float32),h=1)['mean'][0])}")
print("  (ses kernel is Eigen VectorXd=float64, so float32 cast is lossless for small ints)")

print("\n=== 5. SBA debiasing factor: hardcoded or 1-alpha/2? ===")
from statsforecast.models import _croston_sba, _croston_classic
c=_croston_classic(y=y,h=1,fitted=False)['mean'][0]; s=_croston_sba(y=y,h=1,fitted=False)['mean'][0]
print(f"  classic={g(c)} sba={g(s)} ratio={g(s/c)}   hardcoded 0.95 (source: out['mean'] *= 0.95)")
print(f"  1-alpha/2 with alpha=0.1 -> {g(1-0.1/2)}  COINCIDES but is NOT computed from alpha")
