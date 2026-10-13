import json, numpy as np, csv, statsmodels
from statsmodels.tsa.api import VAR

def rd(p):
    rows = list(csv.reader(open(p)))
    return np.array([[float(c) for c in r] for r in rows[1:]])

TREND = {"const": "c", "none": "n", "trend": "ct", "both": "ctt"}  # provisional -- checked below

def grab(Y, p, trend, h=5):
    m = VAR(Y)
    r = m.fit(p, trend=trend)
    out = {
        "K": int(Y.shape[1]), "p": p, "trend": trend,
        "nobs": int(r.nobs),
        "params": r.params.tolist(),          # (ndet + K*p) x K, deterministic FIRST
        "params_index": [str(x) for x in (r.model.exog_names if hasattr(r.model,'exog_names') else [])],
        "coefs": r.coefs.tolist(),            # (p, K, K)
        "intercept": np.atleast_1d(r.intercept).tolist(),
        "stderr": r.stderr.tolist(),
        "tvalues": r.tvalues.tolist(),
        "sigma_u": r.sigma_u.tolist(),
        "sigma_u_mle": r.sigma_u_mle.tolist(),
        "llf": float(r.llf),
        "aic": float(r.aic), "bic": float(r.bic), "hqic": float(r.hqic), "fpe": float(r.fpe),
        "resid_head": r.resid[:3].tolist(), "resid_tail": r.resid[-3:].tolist(),
        "fittedvalues_head": r.fittedvalues[:3].tolist(),
        "roots": np.abs(r.roots).tolist(), "is_stable": bool(r.is_stable()),
        "forecast": r.forecast(Y[-p:], h).tolist(),
        "forecast_interval": [a.tolist() for a in r.forecast_interval(Y[-p:], h, alpha=0.05)],
        "forecast_cov": r.forecast_cov(h).tolist(),
    }
    return out

def sel(Y, lag_max, trend):
    m = VAR(Y)
    s = m.select_order(lag_max, trend=trend)
    return {"lag_max": lag_max, "trend": trend,
            "selected": {k: (int(v) if v is not None else None) for k, v in s.selected_orders.items()},
            "ics": {k: [float(x) for x in v] for k, v in s.ics.items()}}

A  = rd("fixtures/var2_p1.csv");  B3 = rd("fixtures/var3_p2.csv")
K1 = rd("fixtures/var1_k1.csv");  SH = rd("fixtures/var2_short.csv")
US = rd("fixtures/uschange.csv"); E5 = rd("fixtures/econ5.csv")

out = {}
out["A_p1_const"] = grab(A, 1, "c")
out["A_p2_const"] = grab(A, 2, "c")
out["A_p1_none"]  = grab(A, 1, "n")
out["A_p1_trend"] = grab(A, 1, "ct")
out["A_p1_ctt"]   = grab(A, 1, "ctt")
out["B_p2_const"] = grab(B3, 2, "c")
out["B_p3_const"] = grab(B3, 3, "c")
out["SH_p2_const"]= grab(SH, 2, "c")
out["US_p2_const"]= grab(US, 2, "c", h=8)
out["E5_p2_const"]= grab(E5, 2, "c", h=4)

out["sel_A_const"]  = sel(A, 8, "c")
out["sel_A_none"]   = sel(A, 8, "n")
out["sel_A_trend"]  = sel(A, 8, "ct")
out["sel_B_const"]  = sel(B3, 8, "c")
out["sel_US_const"] = sel(US, 8, "c")
out["sel_E5_const"] = sel(E5, 6, "c")

try:
    VAR(K1).fit(1); out["k1_refusal_py"] = "no error"
except Exception as e:
    out["k1_refusal_py"] = f"{type(e).__name__}: {e}"

out["meta"] = {"statsmodels": statsmodels.__version__, "numpy": np.__version__}
json.dump(out, open("fixtures/ref_py.json","w"), indent=1)
print("python refs written; k=1:", out["k1_refusal_py"])
