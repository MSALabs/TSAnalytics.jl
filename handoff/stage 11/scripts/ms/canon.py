import json, numpy as np, warnings
warnings.simplefilter("ignore")
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
y=np.array(json.load(open(W+"hamilton_rgnp.json")))
m=MarkovAutoregression(y,k_regimes=2,order=4,switching_ar=False)
base=m.fit()
# CORRECT label-swap map for k=2 statsmodels params:
#   P = [[a, b],[1-a, 1-b]], a=p[0->0]=P[0,0], b=p[1->0]=P[0,1]
#   swapping regime labels -> (a,b) -> (1-b, 1-a);  const[0]<->const[1]
def canon(p):
    p=p.copy()
    if p[2]>p[3]:
        a,b=p[0],p[1]
        p=np.r_[1-b,1-a,p[3],p[2],p[4:]]
    return p
print("default params      :",np.round(base.params,8))
print("canon(swap of them) :",np.round(canon(np.r_[1-base.params[1],1-base.params[0],base.params[3],base.params[2],base.params[4:]]),8))
print("round-trip exact    :",np.allclose(canon(np.r_[1-base.params[1],1-base.params[0],base.params[3],base.params[2],base.params[4:]]),base.params))
rng=np.random.default_rng(20261014); sp=m.start_params.copy(); hits=[]; lls=[]
for i in range(25):
    s=sp.copy(); s[0:2]=rng.uniform(0.05,0.95,2); s[2:4]=sp[2:4]+rng.normal(0,1.5,2)
    s[4]=abs(sp[4])*np.exp(rng.normal(0,0.7)); s[5:]=sp[5:]+rng.normal(0,0.4,len(sp)-5)
    try:
        r=m.fit(start_params=s,em_iter=0,maxiter=1000)
        if np.isfinite(r.llf): lls.append(r.llf); hits.append(r.params.copy())
    except Exception: pass
lls=np.array(lls); mx=lls.max()
H=np.array([canon(p) for l,p in zip(lls,hits) if l>mx-1e-6])
print("\n%d finite / 25 starts; %d within 1e-6 of global max (llf=%.8f)"%(len(lls),len(H),mx))
print("after CORRECT canonical label ordering, spread (max-min per coord):")
for nm,v in zip(m.param_names,H.max(0)-H.min(0)): print("   %-10s %.3e"%(nm,v))
print("max abs deviation of any hit from the default fit (canonical): %.3e"%np.abs(H-canon(base.params)).max())
print("llf spread among hits: %.3e"%(lls[lls>mx-1e-6].max()-lls[lls>mx-1e-6].min()))
print("best llf found - default llf = %.3e"%(mx-base.llf))
