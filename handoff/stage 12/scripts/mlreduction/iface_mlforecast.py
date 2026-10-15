"""What estimator interface does mlforecast ACTUALLY demand?
Probe 4 candidates: sklearn estimator, duck-typed fit/predict object,
a bare callable (function), and a fit/predict object that is NOT a BaseEstimator
but IS cloneable-looking."""
import numpy as np, pandas as pd, traceback
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression

FIX = "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d = pd.read_csv(f"{FIX}/fixture_a_ar2.csv")
df = pd.DataFrame({"unique_id": "s1", "ds": pd.to_datetime(d.date), "y": d.y})

def probe(label, model):
    print("=" * 72)
    print("PROBE:", label)
    try:
        f = MLForecast(models=model, freq="MS", lags=[1, 2])
        f.fit(df)
        p = f.predict(3)
        print("  OK -> predict ok, cols:", list(p.columns))
        print("  values:", p.iloc[:, -1].to_numpy())
    except Exception as ex:
        print(f"  FAIL {type(ex).__name__}: {ex}")
        tb = traceback.format_exc().strip().splitlines()
        for ln in tb[-6:]:
            print("    |", ln)

# 1. canonical sklearn estimator
probe("sklearn.linear_model.LinearRegression()", LinearRegression())

# 2. duck-typed object: fit/predict only, NOT a BaseEstimator, no get_params
class DuckOLS:
    def fit(self, X, y):
        Xa = np.asarray(X, dtype=float); ya = np.asarray(y, dtype=float)
        Xd = np.column_stack([np.ones(len(Xa)), Xa])
        self.beta, *_ = np.linalg.lstsq(Xd, ya, rcond=None)
        return self
    def predict(self, X):
        Xa = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(Xa)), Xa]) @ self.beta
probe("DuckOLS() -- fit/predict only, no BaseEstimator, no get_params", DuckOLS())

# 3. a BARE CALLABLE (plain python function)
def ols_callable(X, y=None):
    raise RuntimeError("should never be called like this")
probe("a bare function (callable, no .fit/.predict)", ols_callable)

# 4. a lambda
probe("a lambda", lambda X: np.zeros(len(X)))

# 5. object with fit/predict AND get_params/set_params but not BaseEstimator subclass
class DuckOLSParams(DuckOLS):
    def get_params(self, deep=True): return {}
    def set_params(self, **kw): return self
probe("DuckOLSParams() -- fit/predict + get_params/set_params, still not BaseEstimator", DuckOLSParams())
