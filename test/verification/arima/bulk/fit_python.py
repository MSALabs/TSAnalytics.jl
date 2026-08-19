import json, os, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

DIR = os.path.dirname(os.path.abspath(__file__))
manifest = json.load(open(f"{DIR}/manifest.json"))

results = {}
for case in manifest:
    name, p, d, q = case["name"], case["p"], case["d"], case["q"]
    y = pd.read_csv(f"{DIR}/data/{name}.csv")["x"].values
    try:
        m = ARIMA(y, order=(p, d, q), trend="n").fit()
        ar = list(m.arparams) if p > 0 else []
        ma = list(m.maparams) if q > 0 else []
        results[name] = dict(
            ar=ar, ma=ma, llf=float(m.llf), aic=float(m.aic), bic=float(m.bic),
            nobs=int(m.nobs), converged=bool(m.mle_retvals.get("converged", True)) if hasattr(m, "mle_retvals") else True,
            se=list(m.bse),
        )
    except Exception as ex:
        results[name] = dict(error=str(ex))
    print(name, results[name])

with open(f"{DIR}/python_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("done")
