import json, numpy as np, warnings, traceback
warnings.simplefilter("ignore")
np.set_printoptions(precision=8, linewidth=200)
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
rgnp=np.array(json.load(open(W+"hamilton_rgnp.json")))

print("### Q6d diagnosis: MarkovAutoregression order=0")
m=MarkovAutoregression(rgnp,k_regimes=2,order=0)
mr=MarkovRegression(rgnp,k_regimes=2)
rr=mr.fit()
print("  MarkovRegression llf=%.8f params=%s"%(rr.llf,rr.params))
print("  MAR(order=0).loglike at those params = %.8f"%m.loglike(rr.params))
print("  -> same likelihood surface:", abs(m.loglike(rr.params)-rr.llf)<1e-10)
try:
    m.fit()
except Exception as e:
    tb=traceback.format_exc().splitlines()
    print("  MAR(order=0).fit() -> %s: %s"%(type(e).__name__,e))
    print("  last frames:"); [print("    ",l.strip()) for l in tb if "regime_switching" in l or "numpy" in l][-6:]
try:
    r=m.fit(em_iter=0); print("  fit(em_iter=0) OK llf=%.8f"%r.llf)
except Exception as e: print("  fit(em_iter=0) -> %s: %s"%(type(e).__name__,e))
try:
    r=m.fit(start_params=rr.params, em_iter=0); print("  fit(start=MR params, em_iter=0) llf=%.8f"%r.llf)
except Exception as e: print("  fit(start,em_iter=0) -> %s: %s"%(type(e).__name__,e))

print("\n"+"="*72); print("### Q5 local optima: MS-AR(4), switching_ar=False  (reference llf=-181.26339428)")
m=MarkovAutoregression(rgnp,k_regimes=2,order=4,switching_ar=False)
base=m.fit()
print("  default fit (em_iter=5 then bfgs): llf=%.8f"%base.llf)
for sr in [0,5,20,50]:
    r=m.fit(search_reps=sr, rng=12345)
    print("  search_reps=%-3d llf=%.8f  maxabs dev from base params=%.3e"%(sr,r.llf,np.abs(r.params-base.params).max()))
for ei in [0,1,5,20,100]:
    try:
        r=m.fit(em_iter=ei); print("  em_iter=%-4d llf=%.8f  dP=%.3e"%(ei,r.llf,np.abs(r.params-base.params).max()))
    except Exception as e: print("  em_iter=%-4d FAIL %s"%(ei,e))
print("  pure EM (_fit_em, 500 it):")
try:
    em=m._fit_em(maxiter=500,tolerance=1e-10)
    print("    llf=%.8f  params=%s"%(m.loglike(em.params),em.params))
except Exception as e: print("    FAIL",e)

print("\n  random perturbed starts (uniform jitter on untransformed start_params):")
rng=np.random.default_rng(7)
sp=m.start_params.copy()
rows=[]
for i in range(25):
    s=sp.copy()
    s[0:2]=rng.uniform(0.2,0.95,2)          # probs
    s[2:4]=sp[2:4]+rng.normal(0,1.5,2)      # consts
    s[4]=abs(sp[4]*np.exp(rng.normal(0,0.6)))  # sigma2
    s[5:]=sp[5:]+rng.normal(0,0.4,4)
    try:
        r=m.fit(start_params=s, em_iter=0, maxiter=500)
        rows.append((r.llf, r.params.copy()))
    except Exception as e:
        rows.append((np.nan,None))
lls=np.array([r[0] for r in rows])
ok=lls[~np.isnan(lls)]
print("    n=%d  n_converged=%d"%(len(lls),len(ok)))
print("    llf: max=%.8f min=%.8f median=%.8f"%(ok.max(),ok.min(),np.median(ok)))
best=max((r for r in rows if r[1] is not None), key=lambda r:r[0])
print("    best params=",best[1])
print("    best - default llf = %.3e ; maxabs param diff = %.3e"%(best[0]-base.llf, np.abs(best[1]-base.params).max()))
# how many within 1e-4 of the best
print("    #starts within 1e-4 of best llf: %d/%d"%(int((ok>best[0]-1e-4).sum()),len(ok)))
print("    distinct llf modes (rounded 3dp):", sorted(set(np.round(ok,3)), reverse=True)[:8])

print("\n  same for switching_ar=True (k_params=13):")
m2=MarkovAutoregression(rgnp,k_regimes=2,order=4)
b2=m2.fit(); print("    default llf=%.8f"%b2.llf)
sp2=m2.start_params.copy(); lls2=[]
for i in range(25):
    s=sp2.copy()
    s[0:2]=rng.uniform(0.2,0.95,2); s[2:4]=sp2[2:4]+rng.normal(0,1.5,2)
    s[4]=abs(sp2[4]*np.exp(rng.normal(0,0.6))); s[5:]=sp2[5:]+rng.normal(0,0.4,len(sp2)-5)
    try: lls2.append(m2.fit(start_params=s,em_iter=0,maxiter=500).llf)
    except Exception: pass
lls2=np.array(lls2)
print("    n_conv=%d max=%.8f min=%.8f median=%.8f"%(len(lls2),lls2.max(),lls2.min(),np.median(lls2)))
print("    distinct modes (3dp):",sorted(set(np.round(lls2,3)),reverse=True)[:8])
