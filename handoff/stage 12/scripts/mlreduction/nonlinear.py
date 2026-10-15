"""Characterise the interface with a NON-LINEAR regressor.
These numbers are NOT reproducible across versions and must NEVER become a
test expectation -- they exist only to show the interface is regressor-agnostic
and to show what changes when the injected model stops being linear."""
import numpy as np, pandas as pd, warnings, sklearn; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from mlforecast import MLForecast
from sktime.forecasting.compose import make_reduction
FIX="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d=pd.read_csv(f"{FIX}/fixture_a_ar2.csv")
def f17(a): return "  ".join(f"{v:.17g}" for v in np.atleast_1d(np.asarray(a).ravel()))
print(f"NOT A TEST EXPECTATION. sklearn {sklearn.__version__}, numpy {np.__version__}")
print("A RandomForest's splits depend on the RNG, the tie-breaking and the")
print("implementation's float accumulation order; these digits will drift.\n")
df=pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(d.date),"y":d.y.to_numpy(float)})
print("### mlforecast, lags=[1,2], RandomForest(n=50), two different random_state")
for rs in (0,1):
    mf=MLForecast(models=RandomForestRegressor(n_estimators=50,random_state=rs), freq="MS", lags=[1,2])
    mf.fit(df); p=mf.predict(6)["RandomForestRegressor"].to_numpy()
    print(f"  random_state={rs}: {f17(p)}")
mf=MLForecast(models=RandomForestRegressor(n_estimators=50,random_state=0), freq="MS", lags=[1,2])
mf.fit(df); a=mf.predict(6)["RandomForestRegressor"].to_numpy()
mf2=MLForecast(models=RandomForestRegressor(n_estimators=50,random_state=0), freq="MS", lags=[1,2])
mf2.fit(df); b=mf2.predict(6)["RandomForestRegressor"].to_numpy()
print(f"  same random_state twice, max|diff| = {np.max(np.abs(a-b)):.3e}  (reproducible WITHIN a version)")
print(f"  feature_importances_ = {np.array2string(mf.models_['RandomForestRegressor'].feature_importances_, precision=12)}")
print("  -> a non-linear model exposes feature_importances_, which OLS does not;")
print("     anything the package plots from it must be OPTIONAL, not assumed.")
print()
print("### sktime, same RF, all four strategies -- does DirRec still equal Direct?")
ys=pd.Series(d.y.to_numpy(float), index=pd.PeriodIndex(pd.to_datetime(d.date),freq="M"))
res={}
for s in ["recursive","direct","dirrec","multioutput"]:
    f=make_reduction(RandomForestRegressor(n_estimators=50,random_state=0), window_length=2, strategy=s)
    f.fit(ys, fh=list(range(1,7))); res[s]=f.predict(fh=list(range(1,7))).to_numpy()
    print(f"  {s:12s} {f17(res[s])}")
print(f"\n  |dirrec - direct| max = {np.max(np.abs(res['dirrec']-res['direct'])):.6g}")
print("  -> with OLS this was 4e-16 (an exact algebraic identity). With a NON-LINEAR")
print("     model the identity BREAKS, so DirRec is a genuinely distinct strategy")
print("     only for non-linear regressors.")
# prove the OLS identity is algebraic, not accidental: show it on fixture B too
print("\n### the OLS DirRec==Direct identity, restated and re-proved from scratch")
y=d.y.to_numpy(float); n=len(y); W=2
# direct h=2: regress y_{t+2} on [y_{t-1}, y_t]; dirrec h=2: on [y_{t-1}, y_t, y_{t+1}]
# then plug yhat_{t+1} (the direct h=1 fit) in for y_{t+1}.
rows=range(W-1, n-2)
Xw=np.column_stack([np.ones(len(list(rows)))]+[y[np.array(list(rows))-k+1] for k in (2,1)])
y1=y[np.array(list(rows))+1]; y2=y[np.array(list(rows))+2]
b1,*_=np.linalg.lstsq(Xw,y1,rcond=None)             # h=1
bd,*_=np.linalg.lstsq(Xw,y2,rcond=None)             # direct h=2
Xdr=np.column_stack([Xw,y1]); bdr,*_=np.linalg.lstsq(Xdr,y2,rcond=None)  # dirrec h=2
xq=np.array([1.0,y[-2],y[-1]])
pred_direct=xq@bd
pred_dirrec=np.concatenate([xq,[xq@b1]])@bdr
print(f"  direct h=2 prediction : {pred_direct:.17g}")
print(f"  dirrec h=2 prediction : {pred_dirrec:.17g}")
print(f"  |diff| = {abs(pred_direct-pred_dirrec):.3e}")
print("  Reason: OLS orthogonality X'(y1 - X b1) = 0 makes the composite collapse.")
print(f"  check ||X'(y1 - X b1)||_inf = {np.max(np.abs(Xw.T@(y1-Xw@b1))):.3e}")
