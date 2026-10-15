import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction, DirectReductionForecaster, RecursiveReductionForecaster, YfromX
from sktime.forecasting.compose._reduce import _sliding_window_transform
from sktime.forecasting.base import ForecastingHorizon
FIX="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
def f17(a): return "  ".join(f"{v:.17g}" for v in np.atleast_1d(np.asarray(a).ravel()))

for fx,label in [("fixture_a_ar2.csv","A AR(2)"), ("fixture_b_seasonal.csv","B seasonal")]:
    d=pd.read_csv(f"{FIX}/{fx}"); y=d.y.to_numpy(float)
    ys=pd.Series(y,index=pd.PeriodIndex(pd.to_datetime(d.date),freq="M"),name="y")
    print("="*74); print(f"FIXTURE {label}  n={len(y)}"); print("="*74)
    for wi in (True, False):
        f=make_reduction(LinearRegression(),window_length=2,strategy="direct",windows_identical=wi)
        f.fit(ys,fh=list(range(1,7))); p=f.predict(fh=list(range(1,7))).to_numpy()

        # recover training sample size per horizon
        print(f"direct windows_identical={wi!s:5s} fh preds: {f17(p)}")
        print(f"   per-horizon h=1 coefs: int={f.estimators_[0].intercept_:.17g} coef={f17(f.estimators_[0].coef_)}")
    f=make_reduction(LinearRegression(),window_length=2,strategy="recursive"); f.fit(ys,fh=[1])
    print(f"recursive (one model)        int={f.estimator_.intercept_:.17g} coef={f17(f.estimator_.coef_)}")
    # DirRec == Direct identity
    dr=make_reduction(LinearRegression(),window_length=2,strategy="dirrec"); dr.fit(ys,fh=list(range(1,7)))
    di=make_reduction(LinearRegression(),window_length=2,strategy="direct"); di.fit(ys,fh=list(range(1,7)))
    pdr=dr.predict(fh=list(range(1,7))).to_numpy(); pdi=di.predict(fh=list(range(1,7))).to_numpy()
    print(f"   DirRec vs Direct max|diff| = {np.max(np.abs(pdr-pdi)):.3e}   (n_features per dirrec model: {[e.n_features_in_ for e in dr.estimators_]})")
    print()

# explicit design-matrix shapes from the raw transform
print("="*74); print("RAW _sliding_window_transform SHAPES (fixture A, window_length=2)"); print("="*74)
d=pd.read_csv(f"{FIX}/fixture_a_ar2.csv"); ys=pd.Series(d.y.to_numpy(float),index=pd.PeriodIndex(pd.to_datetime(d.date),freq="M"))
for wi in (True,False):
    for fhv in ([1],[1,2,3,4,5,6]):
        yt,Xt=_sliding_window_transform(ys,window_length=2,fh=ForecastingHorizon(fhv,is_relative=True),windows_identical=wi)
        print(f"  windows_identical={wi!s:5s} fh={str(fhv):18s} yt{yt.shape} Xt{Xt.shape}")
print("  (recursive always uses fh=[1] internally -> the full 118-row sample)")
print()
print("="*74); print("sktime's NEWER reduction classes: defaults differ"); head=None; print("="*74)
import inspect
for cls in (DirectReductionForecaster, RecursiveReductionForecaster, YfromX):
    sig=inspect.signature(cls.__init__)
    print(f"  {cls.__name__}({', '.join(f'{k}={v.default!r}' for k,v in list(sig.parameters.items())[1:])})")
print()
print("make_reduction signature:")
print("  " + ", ".join(f"{k}={v.default!r}" for k,v in list(inspect.signature(make_reduction).parameters.items())))
