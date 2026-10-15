"""Pin down the two silent exog failures."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("error", category=UserWarning)
from sklearn.linear_model import LinearRegression
from mlforecast import MLForecast
from sktime.forecasting.compose import make_reduction
FIX="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
C=pd.read_csv(f"{FIX}/fixture_c_exog.csv"); NY=C.y.notna().sum(); H=6
def f17(a): return "  ".join(f"{v:.17g}" for v in np.atleast_1d(np.asarray(a).ravel()))
hist=C.iloc[:NY]; fut=C.iloc[NY:]
df=pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(hist.date),"y":hist.y.to_numpy(float),
                 "x_known":hist.x_known.to_numpy(float)})
Xf=pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(fut.date),"x_known":fut.x_known.to_numpy(float)})
mf=MLForecast(models=LinearRegression(), freq="MS", lags=[1]); mf.fit(df, static_features=[])
print("DEFECT 1: mlforecast predict() with a SHORT X_df -- silent or loud?")
full=mf.predict(H, X_df=Xf)["LinearRegression"].to_numpy()
print("  full X_df (6 rows) :", f17(full))
for k in (1,2,3,5):
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            got=mf.predict(H, X_df=Xf.iloc[:k])["LinearRegression"].to_numpy()
            print(f"  X_df with {k} of 6 rows: {f17(got)}")
            print(f"     warnings raised: {[str(x.message)[:90] for x in w] or 'NONE'}")
            print(f"     max|diff| vs full = {np.max(np.abs(got-full)):.4g}   first divergent h = "
                  f"{int(np.argmax(~np.isclose(got,full)))+1 if not np.allclose(got,full) else '-'}")
        except Exception as e: print(f"  X_df with {k} rows RAISES {type(e).__name__}: {str(e)[:150]}")
print("  future x_known actual values:", f17(fut.x_known.to_numpy()))
print()
print("DEFECT 2: sktime dirrec silently DROPS X (source comment: 'logic for X below is broken')")
pi=pd.PeriodIndex(pd.to_datetime(C.date),freq="M")
ys=pd.Series(C.y.to_numpy(float)[:NY], index=pi[:NY], name="y")
Xall=pd.DataFrame({"x_known":C.x_known.to_numpy(float)}, index=pi)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    f=make_reduction(LinearRegression(), window_length=2, strategy="dirrec")
    f.fit(ys, X=Xall.iloc[:NY], fh=list(range(1,H+1)))
    pX=f.predict(fh=list(range(1,H+1)), X=Xall).to_numpy()
    print(f"  dirrec WITH X : n_features={[e.n_features_in_ for e in f.estimators_]}  preds {f17(pX)}")
    print(f"  warnings: {[str(x.message)[:110] for x in w] or 'NONE'}")
f2=make_reduction(LinearRegression(), window_length=2, strategy="dirrec")
f2.fit(ys, fh=list(range(1,H+1)))
p0=f2.predict(fh=list(range(1,H+1))).to_numpy()
print(f"  dirrec NO X   : n_features={[e.n_features_in_ for e in f2.estimators_]}  preds {f17(p0)}")
print(f"  max|diff| WITH-X vs NO-X = {np.max(np.abs(pX-p0)):.3e}  -> X had ZERO effect"
      if np.allclose(pX,p0) else f"  differ by {np.max(np.abs(pX-p0)):.3e}")
print()
print("DEFECT 3: sktime recursive with history-only X -> silent garbage")
f3=make_reduction(LinearRegression(), window_length=2, strategy="recursive")
f3.fit(ys, X=Xall.iloc[:NY], fh=list(range(1,H+1)))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    good=f3.predict(fh=list(range(1,H+1)), X=Xall).to_numpy()
    bad =f3.predict(fh=list(range(1,H+1)), X=Xall.iloc[:NY]).to_numpy()
    print("  X incl. 6 future rows:", f17(good))
    print("  X history only       :", f17(bad))
    print(f"  max|diff| = {np.max(np.abs(good-bad)):.4g}   warnings: {[str(x.message)[:90] for x in w] or 'NONE'}")
