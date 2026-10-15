"""Per-file and per-function line counts via ast, split into roles."""
import ast, os, sys, importlib.util
def mod_dir(n):
    s=importlib.util.find_spec(n); return os.path.dirname(s.origin)
MF=mod_dir("mlforecast"); SK=mod_dir("sktime")
def defs(path):
    src=open(path).read(); tree=ast.parse(src); out=[]
    def walk(node, pre=""):
        for ch in node.body:
            if isinstance(ch,(ast.FunctionDef,ast.AsyncFunctionDef)):
                out.append((pre+ch.name, ch.lineno, ch.end_lineno, ch.end_lineno-ch.lineno+1))
            elif isinstance(ch,ast.ClassDef):
                out.append((pre+ch.name+" [class]", ch.lineno, ch.end_lineno, ch.end_lineno-ch.lineno+1))
                walk(ch, pre+ch.name+".")
    walk(tree); return out, len(src.splitlines())
print("="*92); print("MLFORECAST 1.1.0 -- per-file totals and the functions that matter"); print("="*92)
files=["core.py","forecast.py","lag_transforms.py","target_transforms.py","feature_engineering.py",
       "lgb_cv.py","data_validation.py","grouped_array.py","auto.py","conformal_prediction.py","utils.py","optimization.py"]
tot=0
for f in files:
    p=os.path.join(MF,f); ds,n=defs(p); tot+=n
    print(f"\n-- {f}  ({n} lines, {len(ds)} defs)")
    for nm,a,b,L in sorted(ds,key=lambda x:-x[3])[:9]:
        print(f"     {L:5d}  L{a}-{b}  {nm}")
print(f"\nmlforecast total across those files: {tot}")
print()
print("="*92); print("SKTIME 1.2.0 -- the reduction module only"); print("="*92)
for f in ["forecasting/compose/_reduce.py","transformations/series/summarize.py","forecasting/compose/_reduce_proba.py"]:
    p=os.path.join(SK,f)
    if not os.path.exists(p): print(f"  {f}: MISSING"); continue
    ds,n=defs(p)
    print(f"\n-- {f}  ({n} lines, {len(ds)} defs)")
    for nm,a,b,L in sorted(ds,key=lambda x:-x[3])[:16]:
        print(f"     {L:5d}  L{a}-{b}  {nm}")
