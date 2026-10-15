"""Build 3 fixtures for the ML-reduction scoping study.
Generated with a SEEDED numpy Generator, then WRITTEN TO CSV at full precision:
the CSV is the portable artifact, so no cross-language RNG agreement is needed.
Genuine pseudo-random innovations are used deliberately -- a purely deterministic
(sinusoidal) driver makes any 6-lag block of it rank-2, which silently produces
NA coefficients in R's lm and would mask real findings.
"""
import numpy as np, pandas as pd, os
OUT="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261015)

# ---- A: AR(2), y_t = 1.0 + 0.6 y_{t-1} - 0.3 y_{t-2} + e_t ----
n=120; e=rng.normal(0,0.5,n); y=np.zeros(n); y[0],y[1]=1.0,1.2
for t in range(2,n): y[t]=1.0+0.6*y[t-1]-0.3*y[t-2]+e[t]
pd.DataFrame({"t":np.arange(1,n+1),
    "date":pd.date_range("2010-01-01",periods=n,freq="MS").strftime("%Y-%m-%d"),
    "y":y}).to_csv(f"{OUT}/fixture_a_ar2.csv",index=False,float_format="%.17g")

# ---- B: seasonal (period 12) + trend + AR(1) ----
n=144; s=3.0*np.sin(2*np.pi*np.arange(1,n+1)/12)+1.5*np.cos(2*np.pi*np.arange(1,n+1)/12)
e=rng.normal(0,0.4,n); yb=np.zeros(n); yb[0]=10.0
for t in range(1,n): yb[t]=0.5*yb[t-1]+0.05*(t+1)+s[t]+e[t]
pd.DataFrame({"t":np.arange(1,n+1),
    "date":pd.date_range("2008-01-01",periods=n,freq="MS").strftime("%Y-%m-%d"),
    "y":yb,"month":[((t-1)%12)+1 for t in range(1,n+1)]}).to_csv(
    f"{OUT}/fixture_b_seasonal.csv",index=False,float_format="%.17g")

# ---- C: AR(1) + a KNOWN-FUTURE regressor + a PAST-ONLY regressor ----
# 6 extra rows carry x beyond the y sample, for forecast time.
n=132; H=6
x_known=rng.normal(0.5,1.0,n+H)     # e.g. a planned promo intensity: known ahead
x_past =rng.normal(0.0,0.8,n+H)     # e.g. a measured driver: observed only to the origin
e=rng.normal(0,0.3,n); yc=np.zeros(n); yc[0]=2.0
for t in range(1,n): yc[t]=0.4*yc[t-1]+2.0*x_known[t]+1.5*x_past[t]+e[t]
pd.DataFrame({"t":np.arange(1,n+H+1),
    "date":pd.date_range("2012-01-01",periods=n+H,freq="MS").strftime("%Y-%m-%d"),
    "y":list(yc)+[np.nan]*H,"x_known":x_known,"x_past":x_past}).to_csv(
    f"{OUT}/fixture_c_exog.csv",index=False,float_format="%.17g")

# ---- rank report: every design we will actually fit must be full rank ----
print("RANK CHECK (a rank-deficient design makes R's lm emit NA coefs SILENTLY)")
A=pd.read_csv(f"{OUT}/fixture_a_ar2.csv").y.to_numpy(); B=pd.read_csv(f"{OUT}/fixture_b_seasonal.csv").y.to_numpy()
Cd=pd.read_csv(f"{OUT}/fixture_c_exog.csv")
def rk(lbl,X): print(f"  {lbl:54s} {str(X.shape):12s} rank {np.linalg.matrix_rank(X):3d} cond {np.linalg.cond(X):.4g}")
for nm,v in [("A",A),("B",B)]:
    for P in (2,6):
        rk(f"{nm}: const + {P} y-lags", np.column_stack([np.ones(len(v)-P)]+[v[P-k:len(v)-k] for k in range(1,P+1)]))
yc=Cd.y.to_numpy()[:132]; xk=Cd.x_known.to_numpy(); xp=Cd.x_past.to_numpy(); nC=132
for P in (1,2,6):
    cols=[np.ones(nC-P)]+[yc[P-k:nC-k] for k in range(1,P+1)]+[xk[P-k:nC-k] for k in range(1,P+1)]+[xp[P-k:nC-k] for k in range(1,P+1)]
    rk(f"C: const + {P} lags of each of y,x_known,x_past", np.column_stack(cols))
cols=[np.ones(nC-6)]+[yc[6-k:nC-k] for k in range(1,7)]+[xk[6:nC]]+[xp[6-k:nC-k] for k in range(1,7)]
rk("C: const + 6 y-lags + CONTEMPORANEOUS x_known + 6 x_past-lags", np.column_stack(cols))
print()
for f in sorted(os.listdir(OUT)):
    d=pd.read_csv(f"{OUT}/{f}"); print(f"  {f:26s} shape={d.shape} cols={list(d.columns)}")
