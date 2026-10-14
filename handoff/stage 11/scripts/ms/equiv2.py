import json, numpy as np, warnings
warnings.simplefilter("ignore")
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
y=np.array(json.load(open(W+"hamilton_rgnp.json"))); p=4; n=len(y)
idx=np.arange(p,n); Y=y[idx]; X=np.column_stack([y[idx-j] for j in range(1,p+1)])
mwc=np.array([-0.448465,1.116735]); mwar=np.array([0.1095938,0.06516106,-0.1275574,-0.1380008])
mwsd=0.78911052; P=np.array([[0.6735828,0.08630902],[0.3264171,0.91369098]])
par=np.r_[P[0,0],P[0,1],mwc,mwar,mwsd**2]
mu=np.column_stack([mwc[0]+X@mwar, mwc[1]+X@mwar]).T
lik=np.exp(-0.5*((Y[None,:]-mu)/mwsd)**2-0.5*np.log(2*np.pi*mwsd**2))
def filt_sm(Pm,ini):
    f=ini.copy(); tot=0.0
    for t in range(lik.shape[1]):
        pred=Pm@f if t>0 else ini
        w=pred*lik[:,t]; m=w.sum(); tot+=np.log(m); f=w/m
    return tot
mr=MarkovRegression(Y,k_regimes=2,exog=X,switching_exog=False,switching_variance=False)
print("### CONFIRM: MarkovRegression's effective prior for obs 0 is P^2 @ pi")
for v,lab in [(np.array([1e-20,1.0]),"[0,1]"),(np.array([0.5,0.5]),"[.5,.5]"),(np.array([0.9,0.1]),"[.9,.1]")]:
    mr.initialize_known(v)
    sm=mr.loglike(par)
    print("  known=%-9s statsmodels=%.8f   manual(P^0 pi)=%.8f  (P pi)=%.8f  (P^2 pi)=%.8f"%(
        lab,sm,filt_sm(P,v),filt_sm(P,P@v),filt_sm(P,P@P@v)))
mr.initialize_steady_state()
ss=mr.initial_probabilities(par)
print("  steady    statsmodels=%.8f  manual=%.8f  (P pi==pi: %s)"%(mr.loglike(par),filt_sm(P,ss),np.allclose(P@ss,ss)))

print("\n### MSwM equivalent (MSwM applies P once to iniProb, so prior = P@iniProb)")
print("  MSwM reported llf                        = -180.11423858")
print("  manual filt_sm with ini = P@[0,1]        = %.8f"%filt_sm(P,P@np.array([1e-300,1.0])))

print("\n### statsmodels MLE of the MSwM-equivalent spec, started AT MSwM's params")
mr2=MarkovRegression(Y,k_regimes=2,exog=X,switching_exog=False,switching_variance=False)
r=mr2.fit(start_params=par, em_iter=0, maxiter=2000)
print("  llf=%.8f (steady-state init)"%r.llf)
for nm,v in zip(mr2.param_names,r.params): print("    %-12s %+.10f"%(nm,v))
Pb=mr2.regime_transition_matrix(r.params)[:,:,0]
print("  P=\n",np.round(Pb,8),"  sigma=%.10f"%np.sqrt(r.params[-1]))
print("\n  == aligned comparison (regime0 = low mean in BOTH; identity alignment) ==")
rows=[("const[0]",r.params[2],mwc[0]),("const[1]",r.params[3],mwc[1]),
      ("ar.L1",r.params[4],mwar[0]),("ar.L2",r.params[5],mwar[1]),
      ("ar.L3",r.params[6],mwar[2]),("ar.L4",r.params[7],mwar[3]),
      ("sigma",np.sqrt(r.params[8]),mwsd),("P[0,0]",Pb[0,0],P[0,0]),("P[1,1]",Pb[1,1],P[1,1])]
print("  %-10s %14s %14s %12s"%("param","statsmodels","MSwM","rel.err"))
for nm,a,b in rows: print("  %-10s %+14.8f %+14.8f %12.3e"%(nm,a,b,abs(a-b)/max(abs(b),1e-12)))
sp=r.smoothed_marginal_probabilities
mwp=np.loadtxt(W+"mswm_A_smoProb.csv",delimiter=",",skiprows=1)[1:]
print("  smoothed probs: corr=%.8f  maxabsdiff=%.8f  meanabsdiff=%.8f"%(
  np.corrcoef(sp[:,0],mwp[:,0])[0,1],np.abs(sp-mwp).max(),np.abs(sp-mwp).mean()))
np.savetxt(W+"compare_smoothed_equivspec.csv",np.c_[sp,mwp],delimiter=",",header="sm_r0,sm_r1,mswm_r0,mswm_r1",comments="")
json.dump({"spec":"MarkovRegression(Y=rgnp[4:], exog=4 levels lags, switching_trend=True, switching_exog=False, switching_variance=False)",
  "sm_llf":r.llf,"sm_params":list(r.params),"sm_param_names":list(mr2.param_names),
  "mswm_llf":-180.11423858,"mswm_const":list(mwc),"mswm_ar":list(mwar),"mswm_sigma":mwsd,"mswm_P":P.tolist()},
  open(W+"equiv_compare.json","w"),indent=1)
