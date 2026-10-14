import warnings; warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.structural import UnobservedComponents
import statsmodels.api as sm
nile = sm.datasets.nile.load_pandas().data['volume'].values.astype(float)
cases = [
 ("level=True",                      dict(level=True)),
 ("level='local level'",             dict(level='local level')),
 ("level=True,stochastic_level=True",dict(level=True, stochastic_level=True)),
 ("level='dtrend'",                  dict(level='dtrend')),
 ("level='lltrend'",                 dict(level='lltrend')),
 ("level='rwalk'",                   dict(level='rwalk')),
 ("level='rwdrift'",                 dict(level='rwdrift')),
 ("level='smooth trend'",            dict(level='smooth trend')),
 ("level='random trend'",            dict(level='random trend')),
]
for label, kw in cases:
    m = UnobservedComponents(nile, **kw)
    print(f"{label:34s} k={m.k_states}  stoch(level,trend)=({m.stochastic_level},{m.stochastic_trend})  irregular={m.irregular}  params={m.param_names}")
