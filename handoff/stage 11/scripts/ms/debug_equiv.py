import json, numpy as np, warnings
warnings.simplefilter("ignore")
np.set_printoptions(precision=10, linewidth=200)
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
y=np.array(json.load(open(W+"hamilton_rgnp.json"))); p=4; n=len(y)
idx=np.arange(p,n); Y=y[idx]; X=np.column_stack([y[idx-j] for j in range(1,p+1)])
mw=dict(c=np.array([-0.448465,1.116735]), ar=np.array([0.1095938,0.06516106,-0.1275574,-0.1380008]),
        sd=0.78911052, P=np.array([[0.6735828,0.08630902],[0.3264171,0.91369098]]), ini=np.array([0.,1.]))
par=np.r_[mw["P"][0,0],mw["P"][0,1],mw["c"],mw["ar"],mw["sd"]**2]
mr=MarkovRegression(Y,k_regimes=2,exog=X,switching_exog=False,switching_variance=False)
print("parameters dict:"); 
for kk in ["regime_transition","exog","variance"]:
    print("  ",kk, mr.parameters[kk])
for i in range(2): print("   regime",i,"exog:",mr.parameters[i,"exog"])
pc=mr.predict_conditional(par)   # shape (k, nobs)
mu=np.column_stack([mw["c"][0]+X@mw["ar"], mw["c"][1]+X@mw["ar"]]).T
print("predict_conditional shape",pc.shape)
print("max|pc - manual mu| =",np.abs(pc-mu).max())
print("pc[:, :3]=\n",pc[:,:3]); print("mu[:, :3]=\n",mu[:,:3])
