"""LEAKAGE TEST (the most consequential check).
Method: build the feature matrix, then perturb y at a single row t0 by +1000 and
rebuild. If ANY feature value in ROW t0 changes, that feature saw the target
period -> leakage. A correctly-lagged feature in row t0 must be invariant to y[t0].
This is a direct functional-dependence probe, not an inference from R^2.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from mlforecast import MLForecast
import mlforecast.lag_transforms as lt

FIX="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d=pd.read_csv(f"{FIX}/fixture_a_ar2.csv")
BASE=pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(d.date),"y":d.y.to_numpy(float)})
T0=60   # the row we perturb (0-based position in the ORIGINAL series)

def build(cfg, yvals):
    df=BASE.copy(); df["y"]=yvals
    f=MLForecast(models=[], freq="MS", **cfg)
    return f.preprocess(df, dropna=False)

def leak_test(label, cfg):
    y0=BASE.y.to_numpy(float).copy()
    y1=y0.copy(); y1[T0]+=1000.0
    p0=build(cfg,y0); p1=build(cfg,y1)
    ds0=BASE.ds.iloc[T0]
    r0=p0[p0.ds==ds0]; r1=p1[p1.ds==ds0]
    if len(r0)==0:
        print(f"  {label:52s} row t0 dropped from design"); return
    feats=[c for c in p0.columns if c not in ("unique_id","ds","y")]
    leaky=[]
    for c in feats:
        a=float(r0[c].iloc[0]); b=float(r1[c].iloc[0])
        if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a,b,equal_nan=True):
            leaky.append((c,a,b))
    verdict = "LEAK" if leaky else "clean"
    print(f"  {label:52s} [{verdict}]  feats={feats}")
    for c,a,b in leaky:
        print(f"      !! {c}: {a:.10g} -> {b:.10g}   (moved by {b-a:.6g} when y[t0] moved by +1000)")

print("="*80); print("mlforecast: does any feature in row t see y[t]?"); print("="*80)
leak_test("lags=[1,2]", dict(lags=[1,2]))
leak_test("lag_transforms={1:[RollingMean(3)]}", dict(lag_transforms={1:[lt.RollingMean(window_size=3)]}))
leak_test("lag_transforms={1:[RollingMean(3),RollingStd(3),RollingMin(3),RollingMax(3)]}",
          dict(lag_transforms={1:[lt.RollingMean(window_size=3),lt.RollingStd(window_size=3),
                                  lt.RollingMin(window_size=3),lt.RollingMax(window_size=3)]}))
leak_test("lag_transforms={1:[ExpandingMean()]}", dict(lag_transforms={1:[lt.ExpandingMean()]}))
leak_test("lag_transforms={1:[ExponentiallyWeightedMean(0.9)]}",
          dict(lag_transforms={1:[lt.ExponentiallyWeightedMean(alpha=0.9)]}))
leak_test("lag_transforms={1:[SeasonalRollingMean(12,2)]}",
          dict(lag_transforms={1:[lt.SeasonalRollingMean(season_length=12,window_size=2)]}))
leak_test("lag_transforms={12:[RollingMean(3)]}", dict(lag_transforms={12:[lt.RollingMean(window_size=3)]}))
print()
print("### the ADVERSARIAL case: can the user ask for lag 0?")
for k in (0,):
    try:
        leak_test(f"lag_transforms={{{k}:[RollingMean(3)]}}", dict(lag_transforms={k:[lt.RollingMean(window_size=3)]}))
    except Exception as e: print(f"  lag_transforms={{{k}:...}} REJECTED {type(e).__name__}: {e}")
for k in (0,):
    try:
        leak_test(f"lags=[{k}]", dict(lags=[k]))
    except Exception as e: print(f"  lags=[{k}] REJECTED {type(e).__name__}: {e}")
print()
print("### date_features (deterministic in ds, so can never leak y) ###")
leak_test("date_features=['month','quarter','year']", dict(lags=[1],date_features=["month","quarter","year"]))

print()
print("="*80); print("sktime WindowSummarizer: same probe"); print("="*80)
from sktime.transformations.series.summarize import WindowSummarizer
def sk_leak(label, kwargs):
    y0=BASE.y.to_numpy(float).copy(); y1=y0.copy(); y1[T0]+=1000.0
    idx=pd.PeriodIndex(BASE.ds,freq="M")
    out=[]
    for yy in (y0,y1):
        ws=WindowSummarizer(**kwargs)
        out.append(ws.fit_transform(pd.DataFrame({"y":yy},index=idx)))
    a,b=out; feats=list(a.columns); leaky=[]
    for c in feats:
        va=a[c].iloc[T0]; vb=b[c].iloc[T0]
        if not np.isclose(va,vb,equal_nan=True): leaky.append((c,va,vb))
    print(f"  {label:52s} [{'LEAK' if leaky else 'clean'}]  cols={feats}")
    for c,va,vb in leaky: print(f"      !! {c}: {va:.10g} -> {vb:.10g}")
sk_leak("lag_feature={'lag':[1,2]}", dict(lag_feature={"lag":[1,2]}))
sk_leak("lag_feature={'mean':[[1,3]]}  (lag 1, window 3)", dict(lag_feature={"mean":[[1,3]]}))
sk_leak("lag_feature={'mean':[[0,3]]}  (lag 0 -- ADVERSARIAL)", dict(lag_feature={"mean":[[0,3]]}))
sk_leak("lag_feature={'std':[[1,4]],'mean':[[1,4]]}", dict(lag_feature={"std":[[1,4]],"mean":[[1,4]]}))
