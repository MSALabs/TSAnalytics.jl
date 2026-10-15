"""CV harnesses: mlforecast.cross_validation and sktime splitters, vs this
project's expanding_window_split / sliding_window_split / tscv."""
import numpy as np, pandas as pd, warnings, inspect; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from mlforecast import MLForecast
FIX="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d=pd.read_csv(f"{FIX}/fixture_a_ar2.csv")
df=pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(d.date),"y":d.y.to_numpy(float)})
print("="*78); print("mlforecast.MLForecast.cross_validation"); print("="*78)
print("signature:"); 
for k,v in list(inspect.signature(MLForecast.cross_validation).parameters.items())[1:]:
    print(f"   {k} = {v.default!r}")
mf=MLForecast(models=LinearRegression(), freq="MS", lags=[1,2])
cv=mf.cross_validation(df, n_windows=4, h=3, step_size=3, refit=True)
print(f"\nresult shape {cv.shape} cols {list(cv.columns)}")
print(cv.to_string(index=False, float_format=lambda v: f"{v:.10g}"))
print("\n-- refit=False (fit ONCE on the first train split, reuse):")
cv2=mf.cross_validation(df, n_windows=4, h=3, step_size=3, refit=False)
print(f"   max|diff| vs refit=True = {np.max(np.abs(cv2.LinearRegression.to_numpy()-cv.LinearRegression.to_numpy())):.4g}")
print("\n-- refit=2 (refit every 2nd window):")
cv3=mf.cross_validation(df, n_windows=4, h=3, step_size=3, refit=2)
print(f"   max|diff| vs refit=True = {np.max(np.abs(cv3.LinearRegression.to_numpy()-cv.LinearRegression.to_numpy())):.4g}")
print("\n-- is the window EXPANDING or SLIDING? inspect cutoffs + train sizes:")
import utilsforecast.processing as ufp
for cutoffs,train,valid in ufp.backtest_splits(df, n_windows=4, h=3, id_col="unique_id", time_col="ds", freq="MS", step_size=3):
    print(f"   cutoff {cutoffs.iloc[0,-1]}  train n={len(train)} first={train['ds'].iloc[0].date()} last={train['ds'].iloc[-1].date()}  valid n={len(valid)}")
print("   -> train always starts at the series start => EXPANDING window. No sliding option,")
print("      no gap/purge/embargo argument in the signature.")

print()
print("="*78); print("sktime splitters"); print("="*78)
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter, SingleWindowSplitter, CutoffSplitter
try:
    from sktime.split import ExpandingGreedySplitter
    extra=[ExpandingGreedySplitter]
except Exception: extra=[]
try:
    from sktime.split import SameLocSplitter, TestPlusTrainSplitter
    extra += [SameLocSplitter, TestPlusTrainSplitter]
except Exception: pass
import sktime.split as sksp
print("sktime.split exports:", sorted(n for n in dir(sksp) if n[0].isupper()))
y=pd.Series(d.y.to_numpy(float), index=pd.PeriodIndex(pd.to_datetime(d.date), freq="M"))
ew=ExpandingWindowSplitter(initial_window=10, step_length=1, fh=[1])
sw=SlidingWindowSplitter(window_length=10, step_length=1, fh=[1])
for nm,sp in [("ExpandingWindowSplitter(iw=10,sl=1,fh=[1])",ew),("SlidingWindowSplitter(wl=10,sl=1,fh=[1])",sw)]:
    sp_list=list(sp.split(y)); 
    print(f"\n{nm}: {len(sp_list)} folds")
    for i,(tr,te) in enumerate(sp_list[:3]): print(f"   fold {i+1}: train {tr[0]}..{tr[-1]} (n={len(tr)})  test {te}")
print("\n-- does sktime offer a GAP / purge?")
for cls in (ExpandingWindowSplitter, SlidingWindowSplitter):
    ps=list(inspect.signature(cls.__init__).parameters)[1:]
    print(f"   {cls.__name__}: {ps}")
