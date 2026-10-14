import json, numpy as np, pandas as pd
np.set_printoptions(precision=8, suppress=False, linewidth=150)
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

W="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms/"
rgnp=np.array(json.load(open(W+"hamilton_rgnp.json")))

print("### Q6: defaults via introspection")
import inspect
print("MarkovAutoregression.__init__:", inspect.signature(MarkovAutoregression.__init__))
print("MarkovRegression.__init__:", inspect.signature(MarkovRegression.__init__))

print("\n### Hamilton AR(4), k=2, switching_ar=False (canonical)")
mod = MarkovAutoregression(rgnp, k_regimes=2, order=4, switching_ar=False)
print("k_params:", mod.k_params, " nobs:", mod.nobs, " param_names:", mod.param_names)
print("start_params:", mod.start_params)
res = mod.fit()
print("llf = %.8f" % res.llf)
print("params:"); 
for n,v in zip(res.model.param_names, res.params): print("   %-22s %+.10f" % (n,v))
print("bse:", res.bse)
print("aic %.6f bic %.6f  nobs %d" % (res.aic,res.bic,res.nobs))
print("regime_transition_matrix:\n", mod.regime_transition_matrix(res.params)[:,:,0])
print("expected_durations:", res.expected_durations)
print("initial_probabilities:", mod.initial_probabilities(res.params))
sm_smoothed = res.smoothed_marginal_probabilities
sm_filtered = res.filtered_marginal_probabilities
print("smoothed_marginal_probabilities shape", sm_smoothed.shape)
print("smoothed[:,0] first10", sm_smoothed[:10,0])
print("smoothed[:,1] first10", sm_smoothed[:10,1])
np.savetxt(W+"sm_ar4_smoothed.csv", sm_smoothed, delimiter=",", header="regime0,regime1", comments="")
np.savetxt(W+"sm_ar4_filtered.csv", sm_filtered, delimiter=",", header="regime0,regime1", comments="")
json.dump({"llf":res.llf,"params":list(res.params),"param_names":list(res.model.param_names),
           "aic":res.aic,"bic":res.bic,"nobs":int(res.nobs),
           "expected_durations":list(np.asarray(res.expected_durations).ravel()),
           "P":mod.regime_transition_matrix(res.params)[:,:,0].tolist()},
          open(W+"sm_ar4_fit.json","w"), indent=1)
