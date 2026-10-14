import ast, json, csv, pathlib
p="/usr/local/lib/python3.13/dist-packages/statsmodels/tsa/regime_switching/tests/test_markov_autoregression.py"
tree=ast.parse(open(p).read())
rgnp=None
for n in tree.body:
    if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=="rgnp":
        rgnp=ast.literal_eval(n.value)
assert rgnp is not None
out=pathlib.Path("/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s11/ms")
print("n =",len(rgnp))
print("first5",rgnp[:5]); print("last5",rgnp[-5:])
print("sum %.10f mean %.10f" % (sum(rgnp), sum(rgnp)/len(rgnp)))
# quarterly 1951Q1 .. ?
import pandas as pd
idx=pd.period_range(start="1951Q1",periods=len(rgnp),freq="Q")
print("range:",idx[0],"->",idx[-1])
with open(out/"hamilton_rgnp.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["quarter","rgnp"])
    for q,v in zip(idx,rgnp): w.writerow([str(q),repr(v)])
json.dump(rgnp, open(out/"hamilton_rgnp.json","w"))
print("written")
