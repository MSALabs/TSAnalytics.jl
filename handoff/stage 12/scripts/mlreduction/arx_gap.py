"""What does arx ALREADY cover, and what exactly is left over?
Replicate arx.jl's construction in numpy and compare against mlforecast with
(a) month dummies (= arx seasonal=true) and (b) a contemporaneous exog (= arx exog).
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from mlforecast import MLForecast
FIX="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
def f17(a): return "  ".join(f"{v:.17g}" for v in np.atleast_1d(np.asarray(a).ravel()))

def arx_ols(y, lags, trend="c", seasonal=False, period=None, exog=None):
    """Exactly src/arx.jl: hold_back=max(lags); cols = const, trend, s(2..period), y.L*, x*"""
    n=len(y); P=max(lags); hb=P; resp=y[hb:]; m=len(resp); cols=[]; names=[]
    if trend in ("c","ct"): cols.append(np.ones(m)); names.append("const")
    if trend in ("t","ct"): cols.append(np.arange(hb+1,n+1,dtype=float)); names.append("trend")
    if seasonal:
        for s in range(2,period+1):
            cols.append(np.array([1.0 if ((t-1)%period)+1==s else 0.0 for t in range(hb+1,n+1)]))
            names.append(f"s({s},{period})")
    for L in lags: cols.append(y[hb-L:n-L]); names.append(f"y.L{L}")
    if exog is not None:
        E=np.atleast_2d(exog); E=E.T if E.shape[0]==1 else E
        for j in range(E.shape[1]): cols.append(E[hb:n,j]); names.append(f"x{j+1}")
    X=np.column_stack(cols)
    b,*_=np.linalg.lstsq(X,resp,rcond=None)
    return names,b,X,resp

print("="*80); print("(A) SEASONAL: arx(seasonal=true, period=12) vs mlforecast date_features=['month']")
print("="*80)
d=pd.read_csv(f"{FIX}/fixture_b_seasonal.csv"); y=d.y.to_numpy(float)
nm,b,X,r=arx_ols(y,[1,2],seasonal=True,period=12)
print(f"arx design {X.shape}")
for a,c in zip(nm,b): print(f"  {a:12s} {c:.17g}")
df=pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(d.date),"y":y})
mf=MLForecast(models=LinearRegression(), freq="MS", lags=[1,2],
              date_features=["month"], date_features_as_dummies=True)
prep=mf.preprocess(df); mf.fit(df)
feat=[c for c in prep.columns if c not in ("unique_id","ds","y")]
lm=mf.models_["LinearRegression"]
print(f"\nmlforecast design {(prep.shape[0], len(feat))}  features: {feat}")
print(f"  intercept   {lm.intercept_:.17g}")
for a,c in zip(feat,lm.coef_): print(f"  {a:12s} {c:.17g}")
print("\n!! NOT comparable term-by-term: mlforecast's month dummies drop NO baseline")
print("   (12 dummies + intercept => rank deficient; sklearn's lstsq picks a min-norm")
print("   solution silently). arx drops season 1 as baseline (11 dummies), full rank.")
Xmf=np.column_stack([np.ones(prep.shape[0])]+[prep[c].to_numpy(float) for c in feat])
print(f"   rank(mlforecast design incl. intercept) = {np.linalg.matrix_rank(Xmf)} of {Xmf.shape[1]}  cond {np.linalg.cond(Xmf):.4g}")
print(f"   rank(arx design)                        = {np.linalg.matrix_rank(X)} of {X.shape[1]}  cond {np.linalg.cond(X):.4g}")
print("   -> arx would THROW on the mlforecast parameterisation (its rank(X) check);")
print("      sklearn returns a silent min-norm fit. FITTED VALUES still agree:")
fv_arx=X@b
fv_mf=lm.intercept_+prep[feat].to_numpy(float)@lm.coef_
print(f"   max|fitted_arx - fitted_mlforecast| = {np.max(np.abs(fv_arx-fv_mf)):.3e}  (same column space)")
print(f"   h=1..6 mlforecast recursive: {f17(mf.predict(6)['LinearRegression'].to_numpy())}")
# arx-style recursive forecast WITH seasonal dummies
n=len(y); hist=list(y); fc=[]
for h in range(6):
    t=n+h+1; row=[1.0]+[1.0 if ((t-1)%12)+1==s else 0.0 for s in range(2,13)]+[hist[-1],hist[-2]]
    v=float(np.array(row)@b); fc.append(v); hist.append(v)
print(f"   h=1..6 arx recursive       : {f17(np.array(fc))}")
print(f"   max|diff| = {np.max(np.abs(np.array(fc)-mf.predict(6)['LinearRegression'].to_numpy())):.3e}")

print()
print("="*80); print("(B) EXOG: arx(exog=x) is CONTEMPORANEOUS -- same as mlforecast's default")
print("="*80)
C=pd.read_csv(f"{FIX}/fixture_c_exog.csv"); NY=C.y.notna().sum()
yc=C.y.to_numpy(float)[:NY]; xk=C.x_known.to_numpy(float)[:NY]; xp=C.x_past.to_numpy(float)[:NY]
nm,b,X,r=arx_ols(yc,[1],exog=np.column_stack([xk,xp]))
print(f"arx design {X.shape}")
for a,c in zip(nm,b): print(f"  {a:12s} {c:.17g}")
dfc=pd.DataFrame({"unique_id":"s1","ds":pd.to_datetime(C.date[:NY]),"y":yc,"x_known":xk,"x_past":xp})
mfc=MLForecast(models=LinearRegression(), freq="MS", lags=[1]); mfc.fit(dfc, static_features=[])
lmc=mfc.models_["LinearRegression"]; fc2=[c for c in mfc.preprocess(dfc,static_features=[]).columns if c not in ("unique_id","ds","y")]
mb=np.concatenate([[lmc.intercept_],lmc.coef_])
# reorder mlforecast [x_known,x_past,lag1] -> arx [const, y.L1, x1, x2]
mb_re=np.array([mb[0], mb[3], mb[1], mb[2]])
print(f"\nmlforecast (reordered to arx's column order): {f17(mb_re)}")
print(f"arx                                        : {f17(b)}")
print(f"max abs diff = {np.max(np.abs(b-mb_re)):.3e}  => IDENTICAL MODEL")
print("\n!! arx has NO way to LAG an exog regressor: src/arx.jl line 183 takes")
print("   ex_mat[(hb+1):n, j] -- the contemporaneous row only. A past-only")
print("   regressor must be lagged by the CALLER before being passed in.")
