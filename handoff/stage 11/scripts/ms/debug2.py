import json, numpy as np, warnings
warnings.simplefilter("ignore")
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
y=np.array(json.load(open(W+"hamilton_rgnp.json"))); p=4; n=len(y)
idx=np.arange(p,n); Y=y[idx]; X=np.column_stack([y[idx-j] for j in range(1,p+1)])
mw=dict(c=np.array([-0.448465,1.116735]), ar=np.array([0.1095938,0.06516106,-0.1275574,-0.1380008]),
        sd=0.78911052, P=np.array([[0.6735828,0.08630902],[0.3264171,0.91369098]]), ini=np.array([0.,1.]))
par=np.r_[mw["P"][0,0],mw["P"][0,1],mw["c"],mw["ar"],mw["sd"]**2]
mr=MarkovRegression(Y,k_regimes=2,exog=X,switching_exog=False,switching_variance=False)
print("nobs",mr.nobs,"k_params",mr.k_params)
print("pmap exog:",mr.parameters['exog'],"| r0:",mr.parameters[0,'exog'],"| r1:",mr.parameters[1,'exog'],"| var:",mr.parameters['variance'])
pc=np.squeeze(mr.predict_conditional(par)); print("pc shape",pc.shape)
mu=np.column_stack([mw["c"][0]+X@mw["ar"], mw["c"][1]+X@mw["ar"]]).T
print("max|pc-mu|",np.abs(pc-mu).max())
cll=np.squeeze(mr._conditional_loglikelihoods(par)); print("cll shape",cll.shape)
sd=mw["sd"]; mylik=-0.5*((Y[None,:]-mu)/sd)**2-0.5*np.log(2*np.pi*sd**2)
print("max|cll-mylik|",np.abs(cll-mylik).max())
# my filter in levels
def filt(P,ini,lik):
    nr=lik.shape[1]; f=np.zeros((nr,2)); ml=np.zeros(nr)
    f0=(P@ini)*lik[:,0]; ml[0]=f0.sum(); f[0]=f0/ml[0]
    for t in range(1,nr):
        ft=(P@f[t-1])*lik[:,t]; ml[t]=ft.sum(); f[t]=ft/ml[t]
    return np.log(ml).sum()
lik=np.exp(mylik)
print("my MSwM filter, ini=[0,1]:      %.8f"%filt(mw["P"],np.array([1e-300,1.0]),lik))
print("my filter with ini=P@[0,1]:     %.8f  <- statsmodels-style (no extra P)"%0.0)
# statsmodels-style: predicted prob at t=1 is ini directly
def filt_sm(P,ini,lik):
    nr=lik.shape[1]; f=ini.copy(); tot=0.0
    for t in range(nr):
        pred=P@f if t>0 else ini
        w=pred*lik[:,t]; m=w.sum(); tot+=np.log(m); f=w/m
    return tot
print("statsmodels-style filter ini=steady: %.8f"%filt_sm(mw["P"],np.array([0.2091193,0.7908807]),lik))
print("statsmodels-style filter ini=P@[0,1]: %.8f"%filt_sm(mw["P"],mw["P"]@np.array([1e-300,1.0]),lik))
mr.initialize_known(mw["P"]@np.array([1e-20,1.0]))
print("statsmodels loglike(known=P@[0,1]) : %.8f"%mr.loglike(par))
mr.initialize_steady_state()
print("statsmodels loglike(steady)        : %.8f"%mr.loglike(par))
