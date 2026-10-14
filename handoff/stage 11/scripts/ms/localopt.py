import json, numpy as np, warnings
warnings.simplefilter("ignore")
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
y=np.array(json.load(open(W+"hamilton_rgnp.json")))
rng=np.random.default_rng(20261014)
def sweep(m,label,nstart=25):
    base=m.fit(); sp=m.start_params.copy(); res=[]
    for i in range(nstart):
        s=sp.copy()
        s[0:2]=rng.uniform(0.05,0.95,2)
        s[2:4]=sp[2:4]+rng.normal(0,1.5,2)
        s[4]=abs(sp[4])*np.exp(rng.normal(0,0.7))
        s[5:]=sp[5:]+rng.normal(0,0.4,len(sp)-5)
        try:
            r=m.fit(start_params=s,em_iter=0,maxiter=1000)
            if np.isfinite(r.llf): res.append((r.llf,r.params.copy()))
        except Exception: pass
    ll=np.array([a for a,_ in res])
    print("\n%s  (k_params=%d)"%(label,m.k_params))
    print("  default fit llf = %.8f"%base.llf)
    print("  %d/%d random starts returned finite llf"%(len(ll),nstart))
    print("  llf  max=%.8f  min=%.8f  median=%.8f"%(ll.max(),ll.min(),np.median(ll)))
    print("  frac within 1e-6 of global max: %.2f"%( (ll>ll.max()-1e-6).mean() ))
    print("  distinct modes (llf rounded 4dp, top 8): %s"%sorted(set(np.round(ll,4)),reverse=True)[:8])
    # spread of params among the starts that hit the global max, after label canonicalisation
    hits=[p for l,p in res if l>ll.max()-1e-6]
    def canon(p):
        p=p.copy()
        if p[2]>p[3]:  # force const[0] < const[1]
            p=np.r_[p[1],p[0],p[3],p[2],p[4:]]
        return p
    H=np.array([canon(p) for p in hits])
    print("  among %d global-max hits, after canonical label ordering:"%len(H))
    print("    max param spread (max-min per coord) = %s"%np.round(H.max(0)-H.min(0),10))
    print("    llf spread = %.3e"%(ll[ll>ll.max()-1e-6].max()-ll[ll>ll.max()-1e-6].min()))
    return base,ll,H

m=MarkovAutoregression(y,k_regimes=2,order=4,switching_ar=False)
b1,l1,H1=sweep(m,"Hamilton MS-AR(4), switching_ar=False, switching_variance=False")
m2=MarkovAutoregression(y,k_regimes=2,order=4)
b2,l2,H2=sweep(m2,"MS-AR(4), ALL switching (switching_ar=True)")
m3=MarkovAutoregression(y,k_regimes=2,order=4,switching_ar=False,switching_variance=True)
b3,l3,H3=sweep(m3,"MS-AR(4), switching_variance=True")

print("\n### canonical test expectations (statsmodels 0.15.0, BSD-3)")
print("Hamilton MS-AR(4): llf=%.10f"%b1.llf)
for nm,v in zip(m.param_names,b1.params): print("  %-10s %+.10f"%(nm,v))
sp=b1.smoothed_marginal_probabilities; fp=b1.filtered_marginal_probabilities
print("smoothed[:,1] first5 %s"%np.round(sp[:5,1],8))
print("smoothed[:,1] last5  %s"%np.round(sp[-5:,1],8))
print("filtered[:,1] first5 %s"%np.round(fp[:5,1],8))
print("filtered[:,1] last5  %s"%np.round(fp[-5:,1],8))
print("sum(smoothed[:,1]) = %.10f   sum(filtered[:,1]) = %.10f"%(sp[:,1].sum(),fp[:,1].sum()))
print("expected_durations = %s"%np.round(b1.expected_durations,10))
print("aic=%.8f bic=%.8f hqic=%.8f"%(b1.aic,b1.bic,b1.hqic))
