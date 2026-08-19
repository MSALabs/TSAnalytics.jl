import json, os, warnings
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DIR = os.path.dirname(os.path.abspath(__file__))
manifest = json.load(open(f"{DIR}/manifest.json"))

results = {}
for case in manifest:
    name = case["name"]
    order = (case["p"], case["d"], case["q"])
    seasonal_order = (case["P"], case["D"], case["Q"], case["s"])
    y = pd.read_csv(f"{DIR}/data/{name}.csv")["x"].values
    try:
        m = SARIMAX(y, order=order, seasonal_order=seasonal_order, trend="n").fit(disp=False)
        p, q, P, Q = case["p"], case["q"], case["P"], case["Q"]
        params = list(m.params)
        j = 0
        ar = params[j:j + p]; j += p
        ma = params[j:j + q]; j += q
        sar = params[j:j + P]; j += P
        sma = params[j:j + Q]; j += Q
        results[name] = dict(
            ar=ar, ma=ma, sar=sar, sma=sma,
            llf=float(m.llf), aic=float(m.aic), bic=float(m.bic), nobs=int(m.nobs),
            se=list(m.bse[:p + q + P + Q]),
        )
    except Exception as ex:
        results[name] = dict(error=str(ex))
    print(name, results[name])

with open(f"{DIR}/python_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("done")
