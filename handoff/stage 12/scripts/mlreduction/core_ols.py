"""THE key experiment: with OLS injected, is the ML reduction just arx?
Compares, on fixture A (AR(2)):
  (1) a from-scratch arx-style OLS (exactly arx.jl's construction: hold_back=maxlag,
      const + y.L1..y.Lp) plus a hand-rolled recursive forecast
  (2) mlforecast recursive + LinearRegression
  (3) sktime RecursiveTabularRegressionForecaster + LinearRegression
  (4) sktime DirectTabularRegressionForecaster + LinearRegression
  (5) sktime Multioutput + LinearRegression
  (6) sktime DirRec + LinearRegression
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
np.set_printoptions(precision=17, suppress=False, linewidth=200)
from sklearn.linear_model import LinearRegression
from mlforecast import MLForecast
from sktime.forecasting.compose import make_reduction

FIX = "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d = pd.read_csv(f"{FIX}/fixture_a_ar2.csv")
y = d.y.to_numpy(float); n = len(y)
P = 2; H = 6
def f17(a): return "  ".join(f"{v:.17g}" for v in np.atleast_1d(a))

print("#"*74); print("# (1) arx-style OLS, built exactly as src/arx.jl does"); print("#"*74)
hb = P
resp = y[hb:]                                  # y[(hb+1):n] 1-based
cols = [np.ones(len(resp))]                    # trend=:c
names = ["const"]
for lag in range(1, P+1):
    cols.append(y[hb-lag : n-lag]); names.append(f"y.L{lag}")
X = np.column_stack(cols)
print(f"design matrix shape: {X.shape}   (nobs={len(resp)}, k={X.shape[1]})")
beta_arx, *_ = np.linalg.lstsq(X, resp, rcond=None)
for nm, b in zip(names, beta_arx): print(f"  {nm:8s} {b:.17g}")
resid = resp - X @ beta_arx
print(f"  sigma2(n-denom) {np.sum(resid**2)/len(resp):.17g}")
# recursive forecast from the arx fit
hist = list(y)
fc_arx = []
for h in range(H):
    xrow = np.array([1.0] + [hist[-l] for l in range(1, P+1)])
    nxt = float(xrow @ beta_arx); fc_arx.append(nxt); hist.append(nxt)
fc_arx = np.array(fc_arx)
print("  recursive h=1..6:"); print("   ", f17(fc_arx))

print(); print("#"*74); print("# (2) mlforecast, lags=[1,2], LinearRegression, recursive (its only strategy)"); print("#"*74)
df = pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(d.date),"y":d.y})
mf = MLForecast(models=LinearRegression(), freq="MS", lags=[1,2])
prep = mf.preprocess(df)
print("preprocess() output shape:", prep.shape, " cols:", list(prep.columns))
print("preprocess() head(3):"); print(prep.head(3).to_string())
feat = [c for c in prep.columns if c not in ("unique_id","ds","y")]
print("feature cols:", feat, " -> design (incl. implicit intercept) = ", (prep.shape[0], len(feat)))
mf.fit(df)
lm_mf = mf.models_["LinearRegression"]
print(f"  intercept_ {lm_mf.intercept_:.17g}")
for nm, c in zip(feat, lm_mf.coef_): print(f"  {nm:8s} {c:.17g}")
fc_mf = mf.predict(H)["LinearRegression"].to_numpy()
print("  forecasts h=1..6:"); print("   ", f17(fc_mf))

print(); print("#"*74); print("# (3)-(6) sktime make_reduction, window_length=2, LinearRegression"); print("#"*74)
ys = pd.Series(y, index=pd.PeriodIndex(pd.to_datetime(d.date), freq="M"), name="y")
sk = {}
for strat in ["recursive","direct","multioutput","dirrec"]:
    f = make_reduction(LinearRegression(), window_length=P, strategy=strat)
    f.fit(ys, fh=list(range(1,H+1)))
    p = f.predict(fh=list(range(1,H+1))).to_numpy()
    sk[strat] = p
    print(f"-- strategy={strat}")
    est = f.estimator_ if hasattr(f,"estimator_") else None
    if isinstance(est, list):
        for i,e in enumerate(est,1):
            print(f"   model[{i}] intercept {e.intercept_ if np.ndim(e.intercept_)==0 else e.intercept_}  coef {np.array2string(np.atleast_1d(e.coef_).ravel(), precision=17)}")
    elif est is not None and hasattr(est,"coef_"):
        print(f"   intercept {est.intercept_}")
        print(f"   coef      {np.array2string(np.atleast_1d(est.coef_), precision=17)}")
    print("   forecasts:", f17(p))

print(); print("#"*74); print("# AGREEMENT TABLE (max abs diff of h=1..6 point forecasts)"); print("#"*74)
cands = {"arx-style OLS recursive": fc_arx, "mlforecast recursive": fc_mf,
         "sktime recursive": sk["recursive"], "sktime direct": sk["direct"],
         "sktime multioutput": sk["multioutput"], "sktime dirrec": sk["dirrec"]}
ks = list(cands)
print(f"{'':28s}" + "".join(f"{k[:13]:>16s}" for k in ks))
for a in ks:
    print(f"{a:28s}" + "".join(f"{np.max(np.abs(cands[a]-cands[b])):16.3e}" for b in ks))
print()
print("arx-style OLS coefs vs mlforecast LinearRegression coefs:")
mfb = np.concatenate([[lm_mf.intercept_], lm_mf.coef_])
print("  arx       ", f17(beta_arx))
print("  mlforecast", f17(mfb))
print("  max abs diff:", f"{np.max(np.abs(beta_arx-mfb)):.3e}")
sk_rec_est = make_reduction(LinearRegression(), window_length=P, strategy="recursive")
sk_rec_est.fit(ys, fh=[1]); e = sk_rec_est.estimator_
skb = np.concatenate([[e.intercept_], np.atleast_1d(e.coef_)])
print("  sktime rec", f17(skb))
print("  max abs diff vs arx:", f"{np.max(np.abs(beta_arx-skb)):.3e}")
print()
print("!! sktime window order check: its coef vector order vs arx's [y.L1, y.L2]")
print("   arx    coefs (L1,L2):", f17(beta_arx[1:]))
print("   sktime coefs        :", f17(np.atleast_1d(e.coef_)))
