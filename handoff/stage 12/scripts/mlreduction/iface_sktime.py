"""What estimator interface does sktime's reduction ACTUALLY demand?"""
import numpy as np, pandas as pd, traceback, warnings
warnings.filterwarnings("ignore")
from sktime.forecasting.compose import make_reduction
from sklearn.linear_model import LinearRegression

FIX = "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d = pd.read_csv(f"{FIX}/fixture_a_ar2.csv")
y = pd.Series(d.y.values, index=pd.PeriodIndex(pd.to_datetime(d.date), freq="M"), name="y")

def probe(label, est, strategy="recursive"):
    print("=" * 72); print(f"PROBE [{strategy}]:", label)
    try:
        f = make_reduction(est, window_length=2, strategy=strategy)
        f.fit(y)
        p = f.predict(fh=[1, 2, 3])
        print("  OK ->", p.to_numpy())
    except Exception as ex:
        print(f"  FAIL {type(ex).__name__}: {ex}")
        for ln in traceback.format_exc().strip().splitlines()[-4:]:
            print("    |", ln)

probe("sklearn LinearRegression()", LinearRegression())

class DuckOLS:
    def fit(self, X, y):
        Xa=np.asarray(X,float); ya=np.asarray(y,float)
        Xd=np.column_stack([np.ones(len(Xa)),Xa]); self.beta,*_=np.linalg.lstsq(Xd,ya,rcond=None); return self
    def predict(self, X):
        Xa=np.asarray(X,float); return np.column_stack([np.ones(len(Xa)),Xa])@self.beta
probe("DuckOLS -- fit/predict only", DuckOLS())

class DuckOLSParams(DuckOLS):
    def get_params(self, deep=True): return {}
    def set_params(self, **kw): return self
probe("DuckOLSParams -- fit/predict + get_params", DuckOLSParams())

probe("bare function", lambda X: np.zeros(len(X)))

# what does make_reduction's scitype inference require?
from sktime.forecasting.compose._reduce import _infer_scitype
for lbl, obj in [("LinearRegression", LinearRegression()), ("DuckOLSParams", DuckOLSParams()),
                 ("DuckOLS", DuckOLS()), ("function", (lambda X: X))]:
    try:
        print(f"_infer_scitype({lbl}) = {_infer_scitype(obj)!r}")
    except Exception as ex:
        print(f"_infer_scitype({lbl}) FAIL {type(ex).__name__}: {ex}")
