import json, numpy as np, warnings
warnings.simplefilter("ignore")
np.set_printoptions(precision=8, linewidth=200)
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
y=np.array(json.load(open(W+"hamilton_rgnp.json")))
p=4

# ---- MSwM-style filter, exactly as in MSwM::.MSM.lm.msmFilter (levels space) ----
def mswm_loglik(c0,c1,ar,sd0,sd1,P,iniProb):
    # design: rows t=p..n-1, regressors 1, y_{t-1}..y_{t-p}
    n=len(y); idx=np.arange(p,n)
    Y=y[idx]; X=np.column_stack([y[idx-j] for j in range(1,p+1)])
    mu=np.column_stack([c0+X@ar, c1+X@ar])
    sd=np.array([sd0,sd1])
    lik=np.exp(-0.5*((Y[:,None]-mu)/sd)**2)/(sd*np.sqrt(2*np.pi))
    nr=len(Y); f=np.zeros((nr,2)); ml=np.zeros(nr)
    f0 = (P@np.asarray(iniProb))*lik[0]; ml[0]=f0.sum(); f[0]=f0/ml[0]
    for t in range(1,nr):
        ft=(P@f[t-1])*lik[t]; ml[t]=ft.sum(); f[t]=ft/ml[t]
    return np.log(ml).sum(), f

# ---- statsmodels MLE ----
mod=MarkovAutoregression(y,k_regimes=2,order=4,switching_ar=False)
res=mod.fit()
sm=res.params
P_sm=mod.regime_transition_matrix(sm)[:,:,0]
print("statsmodels MLE llf (steady-state init)      = %.8f"%res.llf)
print("  params:", dict(zip(mod.param_names, np.round(sm,8))))
ss=mod.initial_probabilities(sm)
ll_mswmfilt_at_sm,_ = mswm_loglik(sm[2],sm[3],sm[5:9],np.sqrt(sm[4]),np.sqrt(sm[4]),P_sm,ss)
print("MSwM-style filter at sm params, steady init  = %.8f  (delta %.2e)"%(ll_mswmfilt_at_sm, ll_mswmfilt_at_sm-res.llf))

# ---- MSwM spec A results (from R, seed 101) ----
mw_c=np.array([-0.448465, 1.116735]); mw_ar=np.array([0.1095938,0.06516106,-0.1275574,-0.1380008])
mw_sd=0.78911052
mw_P=np.array([[0.6735828,0.08630902],[0.3264171,0.91369098]])
mw_ini=np.array([0.0,1.0])
ll_mswm_own,_=mswm_loglik(mw_c[0],mw_c[1],mw_ar,mw_sd,mw_sd,mw_P,np.maximum(mw_ini,1e-300))
print("\nMSwM reported loglik                         = -180.11423858")
print("MSwM-style filter re-evaluated at MSwM params = %.8f"%ll_mswm_own)

# steady state of MSwM's P
A=np.c_[(np.eye(2)-mw_P).T,np.ones(2)].T; ss_mw=np.linalg.pinv(A)[:,-1]
ll_mswm_ss,_=mswm_loglik(mw_c[0],mw_c[1],mw_ar,mw_sd,mw_sd,mw_P,ss_mw)
print("  same params but STEADY-STATE init           = %.8f  (lost %.6f)"%(ll_mswm_ss, ll_mswm_own-ll_mswm_ss))
print("  MSwM P steady state =",ss_mw)

# statsmodels loglik at MSwM params, steady-state init (apples-to-apples on sm's surface)
par=np.r_[mw_P[0,0],mw_P[0,1],mw_c[0],mw_c[1],mw_sd**2,mw_ar]
print("\nstatsmodels loglike at MSwM params (steady)  = %.8f"%mod.loglike(par))
# statsmodels with KNOWN init = MSwM iniProb, at MSwM params
mod2=MarkovAutoregression(y,k_regimes=2,order=4,switching_ar=False)
mod2.initialize_known(np.array([1e-20,1.0]))
print("statsmodels loglike at MSwM params (known=[0,1]) = %.8f"%mod2.loglike(par))
print("statsmodels loglike at sm   params (known=[0,1]) = %.8f"%mod2.loglike(sm))
res2=mod2.fit()
print("statsmodels MLE with known=[0,1] init: llf=%.8f"%res2.llf)
print("  params:",np.round(res2.params,8))

# refit sm with FREE-ish init matched: use known = MSwM's estimated iniProb
print("\n--- aligned comparison: statsmodels MLE vs MSwM MLE, both on steady-state surface ---")
print("  sm   llf=%.8f"%res.llf)
print("  MSwM llf(on sm surface)=%.8f -> sm better by %.6f"%(mod.loglike(par), res.llf-mod.loglike(par)))

# ---- label alignment + smoothed prob comparison ----
sm_smooth=res.smoothed_marginal_probabilities      # (131,2); col1 = high regime
mw_smooth=np.loadtxt(W+"mswm_A_smoProb.csv",delimiter=",",skiprows=1)  # (132,2) incl t=0
mw_s=mw_smooth[1:]                                  # drop the t=0 row -> (131,2)
print("\nshapes sm",sm_smooth.shape," mswm(after dropping row0)",mw_s.shape)
# sm regime0 = low mean (-0.3588); MSwM regime0(col1) = low mean (-0.4485) -> SAME order
for perm,lab in [((0,1),"identity"),((1,0),"swapped")]:
    a=sm_smooth[:,list(perm)]
    c=np.corrcoef(a[:,0],mw_s[:,0])[0,1]
    print("  perm %-8s corr(low)=%.6f  maxabsdiff=%.6f  meanabsdiff=%.6f"%(
        lab,c,np.abs(a-mw_s).max(),np.abs(a-mw_s).mean()))
np.savetxt(W+"compare_smoothed.csv",np.c_[sm_smooth,mw_s],delimiter=",",
           header="sm_r0,sm_r1,mswm_r0,mswm_r1",comments="")
