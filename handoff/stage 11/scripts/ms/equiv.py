import json, numpy as np, warnings
warnings.simplefilter("ignore")
np.set_printoptions(precision=8, linewidth=200)
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
y=np.array(json.load(open(W+"hamilton_rgnp.json"))); p=4; n=len(y)
idx=np.arange(p,n); Y=y[idx]; X=np.column_stack([y[idx-j] for j in range(1,p+1)])
print("equivalent-spec sample: nobs=%d (y[%d:]), exog = 4 lags of y in LEVELS"%(len(Y),p))

# MSwM A params (R, seed 101)
mw=dict(c=np.array([-0.448465,1.116735]), ar=np.array([0.1095938,0.06516106,-0.1275574,-0.1380008]),
        sd=0.78911052, P=np.array([[0.6735828,0.08630902],[0.3264171,0.91369098]]),
        ini=np.array([0.0,1.0]), ll=-180.11423858)

print("\n### A. statsmodels MarkovRegression with lagged-y exog == MSwM's spec?")
mr=MarkovRegression(Y,k_regimes=2,exog=X,switching_exog=False,switching_variance=False)
print("  param_names:",mr.param_names)
par=np.r_[mw["P"][0,0],mw["P"][0,1],mw["c"],mw["ar"],mw["sd"]**2]
print("  evaluating at MSwM params, order of names above -> par =",np.round(par,6))
# steady-state init
ss=mr.initial_probabilities(par)
print("  steady-state init:",ss)
print("  loglike(steady init)          = %.8f"%mr.loglike(par))
# known init = P @ mswm_ini  (because MSwM applies P to iniProb before t=1)
mr2=MarkovRegression(Y,k_regimes=2,exog=X,switching_exog=False,switching_variance=False)
init=mw["P"]@np.maximum(mw["ini"],1e-20)
mr2.initialize_known(init)
print("  known init = P@iniProb =",init)
print("  loglike(known init)           = %.8f"%mr2.loglike(par))
print("  MSwM reported                 = %.8f"%mw["ll"])
print("  ABS DIFF                      = %.3e"%abs(mr2.loglike(par)-mw["ll"]))

print("\n### B. refit statsmodels MarkovRegression (same spec, steady-state init) from scratch")
r=mr.fit(search_reps=40,rng=99)
print("  llf=%.8f"%r.llf); 
for nm,v in zip(mr.param_names,r.params): print("    %-12s %+.10f"%(nm,v))
Pb=mr.regime_transition_matrix(r.params)[:,:,0]; print("  P=\n",Pb)
print("  sigma = %.10f"%np.sqrt(r.params[-1]))
print("  MSwM llf (free init) = %.8f ; statsmodels (steady init) = %.8f"%(mw["ll"],r.llf))
print("  loglike of statsmodels MLE under MSwM's free-init filter:")
mr3=MarkovRegression(Y,k_regimes=2,exog=X,switching_exog=False,switching_variance=False)
Pm=Pb; 
best=-1e18
for q in np.linspace(0,1,201):
    mr3.initialize_known(np.maximum(Pm@np.array([q,1-q]),1e-20)); v=mr3.loglike(r.params)
    if v>best: best,bq=v,q
print("    max over free initial dist: %.8f at iniProb=[%.3f,%.3f]  (steady gave %.8f)"%(best,bq,1-bq,r.llf))

print("\n### C. aligned parameter comparison (statsmodels MLE vs MSwM MLE, same spec)")
sm_c=r.params[2:4]; sm_ar=r.params[4:8]; sm_sd=np.sqrt(r.params[8])
print("  statsmodels: const=[%+.6f,%+.6f] ar=%s sigma=%.6f P11=%.6f P22=%.6f"%(
   sm_c[0],sm_c[1],np.round(sm_ar,6),sm_sd,Pb[0,0],Pb[1,1]))
print("  MSwM       : const=[%+.6f,%+.6f] ar=%s sigma=%.6f P11=%.6f P22=%.6f"%(
   mw["c"][0],mw["c"][1],np.round(mw["ar"],6),mw["sd"],mw["P"][0,0],mw["P"][1,1]))
# label alignment: both have regime0 = low mean -> identity
rel=lambda a,b: np.abs(np.asarray(a)-np.asarray(b))/np.maximum(np.abs(np.asarray(b)),1e-12)
print("  rel err const: %s"%np.round(rel(sm_c,mw["c"]),6))
print("  rel err ar   : %s"%np.round(rel(sm_ar,mw["ar"]),6))
print("  rel err sigma: %.6f"%rel(sm_sd,mw["sd"]))
print("  rel err P11=%.6f P22=%.6f"%(rel(Pb[0,0],mw["P"][0,0]),rel(Pb[1,1],mw["P"][1,1])))

print("\n### D. smoothed probability comparison, equivalent spec")
smp=r.smoothed_marginal_probabilities
mwp=np.loadtxt(W+"mswm_A_smoProb.csv",delimiter=",",skiprows=1)[1:]
print("  shapes",smp.shape,mwp.shape)
print("  corr(regime0)=%.6f  maxabsdiff=%.6f  meanabsdiff=%.6f"%(
   np.corrcoef(smp[:,0],mwp[:,0])[0,1], np.abs(smp-mwp).max(), np.abs(smp-mwp).mean()))
print("  corr swapped =%.6f"%np.corrcoef(smp[:,0],mwp[:,1])[0,1])
np.savetxt(W+"compare_smoothed_equivspec.csv",np.c_[smp,mwp],delimiter=",",
   header="sm_r0,sm_r1,mswm_r0,mswm_r1",comments="")

print("\n### E. for the record: Hamilton MS-AR (mean-adjusted) in MSwM? -> not expressible")
print("  MSwM msmFit builds lagged y as plain regressors (src/MSwM/R/2MSM.r:233-243),")
print("  so it cannot represent y_t - mu_{S_t} = sum phi_j (y_{t-j} - mu_{S_{t-j}}).")
