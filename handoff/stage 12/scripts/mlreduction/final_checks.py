import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from mlforecast import MLForecast
from sktime.forecasting.compose import make_reduction
FIX="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d=pd.read_csv(f"{FIX}/fixture_a_ar2.csv"); y=d.y.to_numpy(float); n=len(y)
def f17(a): return "  ".join(f"{v:.17g}" for v in np.atleast_1d(np.asarray(a).ravel()))
print("### (1) LAG SUBSET lags=[1,3] (skipping lag 2): arx vs mlforecast vs sktime")
L=[1,3]; P=3; hb=P; resp=y[hb:]; m=len(resp)
X=np.column_stack([np.ones(m)]+[y[hb-k:n-k] for k in L])
b,*_=np.linalg.lstsq(X,resp,rcond=None)
print(f"  arx       design {X.shape}  coefs {f17(b)}")
df=pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(d.date),"y":y})
mf=MLForecast(models=LinearRegression(), freq="MS", lags=L); mf.fit(df)
lm=mf.models_["LinearRegression"]; mb=np.concatenate([[lm.intercept_],lm.coef_])
pr=mf.preprocess(df)
print(f"  mlforecast design {(pr.shape[0], len([c for c in pr.columns if c.startswith('lag')]))}  cols {[c for c in pr.columns if c.startswith('lag')]}  coefs {f17(mb)}")
print(f"  max|diff| = {np.max(np.abs(b-mb)):.3e}  => mlforecast SUPPORTS lag subsets, matches arx")
# arx recursive forecast on the subset
hist=list(y); fc=[]
for h in range(6):
    v=float(np.array([1.0]+[hist[-k] for k in L])@b); fc.append(v); hist.append(v)
print(f"  arx        recursive h=1..6: {f17(np.array(fc))}")
print(f"  mlforecast recursive h=1..6: {f17(mf.predict(6)['LinearRegression'].to_numpy())}")
print(f"  max|diff| = {np.max(np.abs(np.array(fc)-mf.predict(6)['LinearRegression'].to_numpy())):.3e}")
print("\n  sktime make_reduction: window_length is a CONTIGUOUS window -- can it skip a lag?")
import inspect
print(f"    make_reduction params: {list(inspect.signature(make_reduction).parameters)}")
print("    -> no lag-subset argument; window_length=w always means lags 1..w.")
print("       A subset is only reachable via `transformers=WindowSummarizer(lag_feature={'lag':[1,3]})`,")
print("       which requires pooling='global'. Testing:")
from sktime.transformations.series.summarize import WindowSummarizer
idx=pd.MultiIndex.from_product([["s1"],pd.PeriodIndex(pd.to_datetime(d.date),freq="M")],names=["id","time"])
yp=pd.DataFrame({"y":y},index=idx)
try:
    f=make_reduction(LinearRegression(), strategy="recursive", window_length=None,
                     transformers=[WindowSummarizer(lag_feature={"lag":[1,3]})], pooling="global")
    f.fit(yp, fh=[1]); e=f.estimator_
    sb=np.concatenate([[e.intercept_],np.atleast_1d(e.coef_)])
    print(f"       coefs {f17(sb)}   n_features {e.n_features_in_}")
    print(f"       max|diff| vs arx (after reorder) = {np.max(np.abs(np.sort(sb)-np.sort(b))):.3e}")
except Exception as ex: print(f"       FAIL {type(ex).__name__}: {str(ex)[:150]}")

print("\n### (2) PREDICTION INTERVALS from a reduction -- does any reference give them?")
mf2=MLForecast(models=LinearRegression(), freq="MS", lags=[1,2]); mf2.fit(df, prediction_intervals=None)
try:
    p=mf2.predict(6, level=[80,95]); print("  mlforecast predict(level=...) WITHOUT prediction_intervals:", list(p.columns))
except Exception as e: print(f"  mlforecast predict(level=...) w/o fit(prediction_intervals=): {type(e).__name__}: {str(e)[:130]}")
from mlforecast.utils import PredictionIntervals
mf3=MLForecast(models=LinearRegression(), freq="MS", lags=[1,2])
mf3.fit(df, prediction_intervals=PredictionIntervals(n_windows=4, h=6))
p=mf3.predict(6, level=[80,95])
print("  mlforecast WITH PredictionIntervals(n_windows=4,h=6) ->", list(p.columns))
print("  -> CONFORMAL (empirical quantiles of CV residuals), not an analytic psi-weight SE.")
ys=pd.Series(y, index=pd.PeriodIndex(pd.to_datetime(d.date),freq="M"))
f=make_reduction(LinearRegression(), window_length=2, strategy="recursive"); f.fit(ys, fh=[1,2,3])
try:
    q=f.predict_interval(fh=[1,2,3], coverage=0.95); print("  sktime predict_interval ->\n", q.to_string())
except Exception as e: print(f"  sktime predict_interval: {type(e).__name__}: {str(e)[:160]}")
print("\n  forecastML: return_error gives point-error metrics only; no interval method in its API")
print("  (man pages: combine_forecasts, create_lagged_df, create_skeleton, create_windows,")
print("   fill_gaps, return_error, return_hyper, train_model, predict.forecast_model + 8 plots).")
