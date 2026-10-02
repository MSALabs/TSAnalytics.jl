# Stage 9.3 second reference: statsmodels ThetaModel.
import numpy as np, pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel

def run(lbl, y, m, h):
    idx = (pd.period_range("1960Q1", periods=len(y), freq="Q") if m == 4 else
           pd.period_range("1949-01", periods=len(y), freq="M") if m == 12 else
           pd.RangeIndex(len(y)))
    s = pd.Series(np.asarray(y, float), index=idx)
    kw = dict(deseasonalize=(m > 1), use_test=(m > 1))
    if m > 1:
        kw["period"] = m
    r = ThetaModel(s, **kw).fit()
    # statsmodels stores the FULL OLS slope as b0; R's thetaf stores b0/2
    print("%s  alpha=%.10f  b0=%.10f  b0_half=%.10f  sigma2=%.10f  deseasonalize=%s"
          % (lbl, r.params["alpha"], r.params["b0"], r.params["b0"] / 2,
             r.sigma2, r.model.deseasonalize))
    print("%s  forecast=%s" % (lbl, " ".join("%.10f" % v for v in r.forecast(h).values)))
    pi = r.prediction_intervals(h, alpha=0.05)
    print("%s  lo95=%s" % (lbl, " ".join("%.10f" % v for v in pi["lower"].values)))
    print("%s  hi95=%s" % (lbl, " ".join("%.10f" % v for v in pi["upper"].values)))
    c = r.forecast_components(h)
    # NOTE: "trend" here is the FULL b0*(h-1+(1-(1-alpha)^n)/alpha); the
    # forecast applies HALF of it. ses is the constant SES level.
    print("%s  comp_ses=%.10f  comp_trend=%s" % (lbl, c["ses"].values[0],
          " ".join("%.10f" % v for v in c["trend"].values[:4])))
    print("%s  comp_seasonal=%s" % (lbl, " ".join("%.10f" % v for v in c["seasonal"].values[:min(h, 12)])))
    # and the same model with sigma2 estimated by MLE on the DEseasonalized
    # series, which is the consistent choice its default path does not make
    rm = ThetaModel(s, **kw).fit(use_mle=True)
    print("%s  mle: alpha=%.10f b0=%.10f sigma2=%.10f" % (lbl,
          rm.params["alpha"], rm.params["b0"], rm.sigma2))
    print()

y = np.loadtxt("test/verification/theta/theta_y.csv")
run("SIM4", y, 4, 8)
run("SIM1", y, 1, 8)
ap = pd.read_csv("test_data/airpassengers.csv")["passengers"].values.astype(float)
run("AIRP", ap, 12, 12)
run("LAIRP", np.log(ap), 12, 12)
nl = pd.read_csv("test_data/nile.csv")["flow"].values.astype(float)
run("NILE", nl, 1, 10)
