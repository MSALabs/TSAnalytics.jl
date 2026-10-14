import json, numpy as np, warnings, inspect
warnings.simplefilter("ignore")
np.set_printoptions(precision=8, linewidth=200)
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
rgnp=np.array(json.load(open(W+"hamilton_rgnp.json")))

print("Q3b fixed: n transition params = count of names starting 'p['")
for k in [2,3,4]:
    mm=MarkovAutoregression(rgnp,k_regimes=k,order=4,switching_ar=False)
    nms=mm.param_names; ntr=sum(1 for n in nms if n.startswith("p["))
    print("  k=%d k_params=%2d  ntrans=%2d (k(k-1)=%2d)  rest=%s"%(k,mm.k_params,ntr,k*(k-1),nms[ntr:]))

print("\nQ6a: switching_variance default False -> single sigma2; True -> per regime")
for sv in [False,True]:
    mm=MarkovAutoregression(rgnp,k_regimes=2,order=4,switching_ar=False,switching_variance=sv)
    r=mm.fit(); print("  switching_variance=%-5s k_params=%d llf=%.8f names=%s"%(sv,mm.k_params,r.llf,mm.param_names))
    print("    params=",r.params)

print("\nQ6b: switching_ar default True (MarkovAutoregression)")
mm=MarkovAutoregression(rgnp,k_regimes=2,order=4)  # defaults
print("  defaults k_params=%d names=%s"%(mm.k_params,mm.param_names))
r=mm.fit(); print("  llf=%.8f"%r.llf); print("  params=",r.params)

print("\nQ6c: switching_trend default True; set False")
mm=MarkovAutoregression(rgnp,k_regimes=2,order=4,switching_ar=False,switching_trend=False)
print("  k_params=%d names=%s"%(mm.k_params,mm.param_names))
r=mm.fit(); print("  llf=%.8f params=%s"%(r.llf,r.params))

print("\nQ6d: k_regimes=2, order=0")
try:
    mm=MarkovAutoregression(rgnp,k_regimes=2,order=0)
    print("  k_params=%d names=%s nobs=%d"%(mm.k_params,mm.param_names,mm.nobs))
    r=mm.fit(); print("  llf=%.8f params=%s"%(r.llf,r.params))
except Exception as e:
    print("  RAISES %s: %s"%(type(e).__name__,e))
mm=MarkovRegression(rgnp,k_regimes=2)
r2=mm.fit(); print("  MarkovRegression k=2 (mixture-of-normals w/ switching mean): llf=%.8f nobs=%d"%(r2.llf,mm.nobs))

print("\nQ6f: MarkovRegression switching_exog default True vs MAR switching_exog default False")
X=np.c_[np.arange(len(rgnp))*0.01]
a=MarkovRegression(rgnp,k_regimes=2,exog=X); b=MarkovAutoregression(rgnp,k_regimes=2,order=1,exog=X)
print("  MarkovRegression(exog) names=",a.param_names)
print("  MarkovAutoregression(exog) names=",b.param_names)

print("\nQ5: local optima — fit signature & search_reps")
print(inspect.signature(MarkovAutoregression.fit))
