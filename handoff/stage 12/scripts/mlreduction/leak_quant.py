"""Quantify the sktime lag-0 leak: in-sample fit quality that is impossible
without seeing the target period."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sktime.transformations.series.summarize import WindowSummarizer
FIX="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d=pd.read_csv(f"{FIX}/fixture_a_ar2.csv"); y=d.y.to_numpy(float)
idx=pd.PeriodIndex(pd.to_datetime(d.date),freq="M")
print(f"{'lag_feature spec':34s} {'n':>5s} {'R2(in-sample)':>14s} {'resid SD':>12s}")
for lab,lf in [("mean [[1,3]]  (lag 1)",{"mean":[[1,3]]}),
               ("mean [[0,3]]  (lag 0, LEAKY)",{"mean":[[0,3]]}),
               ("mean [[0,1]]  (lag 0, w=1 -> y_t)",{"mean":[[0,1]]}),
               ("lag [1]",{"lag":[1]})]:
    F=WindowSummarizer(lag_feature=lf).fit_transform(pd.DataFrame({"y":y},index=idx))
    m=F.notna().all(axis=1).to_numpy()
    X=np.column_stack([np.ones(m.sum()), F[m].to_numpy()]); t=y[m]
    b,*_=np.linalg.lstsq(X,t,rcond=None); r=t-X@b
    r2=1-np.sum(r**2)/np.sum((t-t.mean())**2)
    print(f"{lab:34s} {m.sum():5d} {r2:14.10f} {np.std(r):12.3e}")
print()
print("A lag-0 window of size 1 is literally y_t: R2 = 1, resid SD = 0 -- a perfect")
print("in-sample fit from a 'feature'. sktime accepts this spec with NO warning.")
print("mlforecast CANNOT express it (positive-lag keys enforced); forecastML CANNOT")
print("express it via `lookback` (positive integers enforced) but CAN via")
print("`dynamic_features`, which are taken contemporaneously by design.")
