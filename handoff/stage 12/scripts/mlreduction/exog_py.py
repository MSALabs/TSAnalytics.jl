"""Known-future vs past-only exogenous regressors in mlforecast and sktime."""
import numpy as np, pandas as pd, warnings, traceback; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from mlforecast import MLForecast
from mlforecast.feature_engineering import transform_exog
from sktime.forecasting.compose import make_reduction
FIX="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
C=pd.read_csv(f"{FIX}/fixture_c_exog.csv")
NY=C.y.notna().sum(); H=6
def f17(a): return "  ".join(f"{v:.17g}" for v in np.atleast_1d(np.asarray(a).ravel()))
print(f"fixture C: {NY} observed y rows, {len(C)-NY} future-only rows carrying x")
print("NOTE: mlforecast requires static_features=[] -- it DEFAULTS every extra column to STATIC\n      and raises if the values vary over time (a loud error, not a silent one).")

hist=C.iloc[:NY].copy(); fut=C.iloc[NY:].copy()
df=pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(hist.date),"y":hist.y.to_numpy(float),
                 "x_known":hist.x_known.to_numpy(float),"x_past":hist.x_past.to_numpy(float)})
Xf=pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(fut.date),
                 "x_known":fut.x_known.to_numpy(float),"x_past":fut.x_past.to_numpy(float)})

print("\n"+"="*78); print("MLFORECAST"); print("="*78)
print("--- (a) exog columns present in df: treated as KNOWN-FUTURE, used CONTEMPORANEOUSLY")
mf=MLForecast(models=LinearRegression(), freq="MS", lags=[1])
prep=mf.preprocess(df, static_features=[]); print("preprocess cols:", list(prep.columns), " shape", prep.shape)
mf.fit(df, static_features=[])
m=mf.models_["LinearRegression"]; feat=[c for c in prep.columns if c not in ("unique_id","ds","y")]
print(f"  intercept {m.intercept_:.17g}")
for nm,c in zip(feat,m.coef_): print(f"  {nm:10s} {c:.17g}")
p=mf.predict(H, X_df=Xf)
print("  predict(X_df supplied) ->", f17(p['LinearRegression'].to_numpy()))
print("\n--- (b) SAME model, X_df NOT supplied: what happens?")
try:
    p2=mf.predict(H); print("  ->", f17(p2['LinearRegression'].to_numpy()))
except Exception as e:
    print(f"  RAISES {type(e).__name__}: {e}")
print("\n--- (c) X_df supplied but MISSING a column")
try:
    p3=mf.predict(H, X_df=Xf[["unique_id","ds","x_known"]]); print("  ->", f17(p3['LinearRegression'].to_numpy()))
except Exception as e: print(f"  RAISES {type(e).__name__}: {str(e)[:220]}")
print("\n--- (d) X_df supplied with only 3 of 6 future rows")
try:
    p4=mf.predict(H, X_df=Xf.iloc[:3]); print("  ->", f17(p4['LinearRegression'].to_numpy()))
except Exception as e: print(f"  RAISES {type(e).__name__}: {str(e)[:220]}")
print("\n--- (e) PAST-ONLY handling: mlforecast.feature_engineering.transform_exog")
te=transform_exog(pd.concat([df[["unique_id","ds","x_past"]], Xf[["unique_id","ds","x_past"]]]), lags=[1,2])
print("  transform_exog(lags=[1,2]) cols:", list(te.columns), " shape", te.shape)
print("  -> this LAGS the regressor so the future rows' lags come from OBSERVED history;")
print("     the contemporaneous x_past column must then be DROPPED by the user by hand.")
mf2=MLForecast(models=LinearRegression(), freq="MS", lags=[1])
d2=df.drop(columns=["x_past"]).merge(te[te.ds.isin(df.ds)][["unique_id","ds","x_past_lag1"]], on=["unique_id","ds"])
mf2.fit(d2, static_features=[])
X2=Xf.drop(columns=["x_past"]).merge(te[te.ds.isin(Xf.ds)][["unique_id","ds","x_past_lag1"]], on=["unique_id","ds"])
print("  fitted with x_past_lag1 instead; cols:", [c for c in mf2.preprocess(d2, static_features=[]).columns if c not in ("unique_id","ds","y")])
print("  predict ->", f17(mf2.predict(H, X_df=X2)["LinearRegression"].to_numpy()))

print("\n"+"="*78); print("SKTIME make_reduction"); print("="*78)
pi=pd.PeriodIndex(pd.to_datetime(C.date),freq="M")
ys=pd.Series(C.y.to_numpy(float)[:NY], index=pi[:NY], name="y")
Xall=pd.DataFrame({"x_known":C.x_known.to_numpy(float),"x_past":C.x_past.to_numpy(float)}, index=pi)
for strat in ["recursive","direct","dirrec","multioutput"]:
    print(f"\n--- strategy={strat}")
    try:
        f=make_reduction(LinearRegression(), window_length=2, strategy=strat)
        f.fit(ys, X=Xall.iloc[:NY], fh=list(range(1,H+1)))
        nf = getattr(getattr(f,'estimator_',None),'n_features_in_',None)
        if nf is None and hasattr(f,'estimators_'): nf=[e.n_features_in_ for e in f.estimators_]
        print(f"   n_features_in_ = {nf}   (window_length=2, 1 y + 2 X = 3 vars x 2 lags = 6 expected)")
        p=f.predict(fh=list(range(1,H+1)), X=Xall)
        print("   predict(X with future rows) ->", f17(p.to_numpy()))
    except Exception as e:
        print(f"   FAIL {type(e).__name__}: {str(e)[:200]}")
print("\n--- sktime: X given in fit but NOT in predict")
f=make_reduction(LinearRegression(), window_length=2, strategy="recursive")
f.fit(ys, X=Xall.iloc[:NY], fh=list(range(1,H+1)))
try: print("   ->", f17(f.predict(fh=list(range(1,H+1))).to_numpy()))
except Exception as e: print(f"   RAISES {type(e).__name__}: {str(e)[:200]}")
print("\n--- sktime: X given in predict covering only the HISTORY (no future rows)")
try: print("   ->", f17(f.predict(fh=list(range(1,H+1)), X=Xall.iloc[:NY]).to_numpy()))
except Exception as e: print(f"   RAISES {type(e).__name__}: {str(e)[:200]}")
