"""Does the lag-0 WindowSummarizer leak reach a make_reduction FORECASTER?"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction
from sktime.transformations.series.summarize import WindowSummarizer
FIX="/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d=pd.read_csv(f"{FIX}/fixture_a_ar2.csv")
# build a PANEL (transformers require pooling='global')
idx=pd.MultiIndex.from_product([["s1"],pd.PeriodIndex(pd.to_datetime(d.date),freq="M")],names=["id","time"])
y=pd.DataFrame({"y":d.y.to_numpy(float)},index=idx)
for lab,lf in [("lag1 window3 (clean)",{"mean":[[1,3]]}),("lag0 window3 (LEAKY)",{"mean":[[0,3]]})]:
    try:
        f=make_reduction(LinearRegression(), strategy="recursive", window_length=None,
                         transformers=[WindowSummarizer(lag_feature=lf, truncate="bfill")], pooling="global")
        f.fit(y, fh=[1,2,3])
        p=f.predict(fh=[1,2,3])
        est=f.estimator_
        print(f"{lab}: FIT OK  n_features_in_={getattr(est,'n_features_in_','?')}  coef={np.array2string(np.atleast_1d(est.coef_),precision=12)}")
        print(f"    in-sample R^2-ish |coef| large => leak signature. preds={p.to_numpy().ravel()}")
    except Exception as e:
        print(f"{lab}: FAIL {type(e).__name__}: {e}")
