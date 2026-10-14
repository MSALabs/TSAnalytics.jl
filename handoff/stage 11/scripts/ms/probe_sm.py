import json, numpy as np, warnings
warnings.simplefilter("ignore")
np.set_printoptions(precision=8, linewidth=200)
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
rgnp=np.array(json.load(open(W+"hamilton_rgnp.json")))

print("="*70); print("Q3a: logistic / multinomial-logit verification")
m=MarkovAutoregression(rgnp,k_regimes=2,order=4,switching_ar=False)
u=np.zeros(m.k_params); u[0]=0.5; u[1]=-1.2
c=m.transform_params(u)
print("unconstrained[0:2]=",u[:2],"-> constrained[0:2]=",c[:2])
print("manual softmax: exp(0.5)/(1+exp(0.5))=%.10f  exp(-1.2)/(1+exp(-1.2))=%.10f"%(
  np.exp(.5)/(1+np.exp(.5)), np.exp(-1.2)/(1+np.exp(-1.2))))
print("round-trip untransform(transform(u))[0:2]=", m.untransform_params(c)[:2])

print("\nQ3b: k_params vs k_regimes (order=4, switching_ar=False, switching_variance=False)")
for k in [2,3,4]:
    mm=MarkovAutoregression(rgnp,k_regimes=k,order=4,switching_ar=False)
    ntr=len(np.atleast_1d(mm.parameters["regime_transition"]))
    print("  k=%d: k_params=%2d  n_transition_params=%2d  (k*(k-1)=%d)  names=%s"%(
        k,mm.k_params,ntr,k*(k-1),mm.param_names[:ntr]))

print("\nQ3c: transition matrix orientation (left-stochastic)")
res=m.fit()
P=m.regime_transition_matrix(res.params)[:,:,0]
print("P=\n",P,"\ncolsums",P.sum(0),"rowsums",P.sum(1))
print("P[i,j] = Pr[S_t=i | S_{t-1}=j]")

print("\nQ6e: initialization convention")
print("  m._initialization =",repr(m._initialization))
pi=m.initial_probabilities(res.params)
# steady state check
w,v=np.linalg.eig(P); i=np.argmin(abs(w-1)); ss=np.real(v[:,i]); ss=ss/ss.sum()
print("  initial_probabilities =",pi," eig steady-state =",ss," maxabsdiff=%.3e"%abs(pi-ss).max())

print("\n"+"="*70); print("Q3d: TVTP run")
rec=np.zeros(len(rgnp)); rec[(np.arange(len(rgnp))%17)<3]=1  # dummy
# replicate statsmodels' own tvtp fixture shape: const + a dummy
from statsmodels.tools import add_constant
exog_tvtp=np.c_[np.ones(len(rgnp)), rec]
mt=MarkovAutoregression(rgnp,k_regimes=2,order=1,exog_tvtp=exog_tvtp)
print("  tvtp:",mt.tvtp," k_tvtp:",mt.k_tvtp," k_params:",mt.k_params)
print("  param_names:",mt.param_names)
rt=mt.fit()
print("  llf=%.8f"%rt.llf)
Pt=mt.regime_transition_matrix(rt.params)
print("  P shape",Pt.shape,"  P[:,:,0]=\n",Pt[:,:,0],"\n  P[:,:,3]=\n",Pt[:,:,3])
print("  transform_params leaves tvtp params untouched:", np.allclose(
   mt.transform_params(np.arange(mt.k_params)*0.1)[:4], np.arange(4)*0.1))

print("\n"+"="*70); print("Q6: convention traps")
m0=MarkovRegression(rgnp,k_regimes=2)
print("  MarkovRegression(k=2) default: k_params=%d names=%s"%(m0.k_params,m0.param_names))
r0=m0.fit(); print("   llf=%.8f params=%s"%(r0.llf,r0.params))
mar0=MarkovAutoregression(rgnp,k_regimes=2,order=0) if True else None
