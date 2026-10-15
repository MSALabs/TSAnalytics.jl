import ast, sys
p='/usr/local/lib/python3.13/dist-packages/statsforecast/models.py'
src=open(p).read(); tree=ast.parse(src)
targets={'_croston_classic','_croston_optimized','_croston_sba','_tsb','_adida','_imapa',
         '_demand','_intervals','_probability','_ses_forecast','_optimized_ses_forecast',
         '_chunk_sums','_chunk_forecast','_expand_fitted_demand','_expand_fitted_intervals'}
classes={'ADIDA','CrostonClassic','CrostonOptimized','CrostonSBA','IMAPA','TSB'}
def body_lines(node):
    # code lines excluding docstring
    b=node.body
    if b and isinstance(b[0],ast.Expr) and isinstance(b[0].value,ast.Constant) and isinstance(b[0].value.value,str):
        b=b[1:]
    if not b: return 0,0,0
    lo=min(x.lineno for x in b); hi=max(getattr(x,'end_lineno',x.lineno) for x in b)
    return lo,hi,hi-lo+1
print("--- module-level helper funcs (core algorithms) ---")
print(f"{'name':<28}{'lines':>6} {'code(no doc)':>12}  span")
for n in ast.walk(tree):
    if isinstance(n,ast.FunctionDef) and n.name in targets:
        tot=n.end_lineno-n.lineno+1
        lo,hi,cl=body_lines(n)
        print(f"{n.name:<28}{tot:>6} {cl:>12}  L{n.lineno}-{n.end_lineno}")
print()
print("--- public classes (wrapper, incl. docstrings) ---")
for n in ast.walk(tree):
    if isinstance(n,ast.ClassDef) and n.name in classes:
        tot=n.end_lineno-n.lineno+1
        meths=[(m.name,m.end_lineno-m.lineno+1,body_lines(m)[2]) for m in n.body if isinstance(m,ast.FunctionDef)]
        print(f"{n.name:<20} total={tot:>5} L{n.lineno}-{n.end_lineno}  methods={len(meths)}")
        for mn,mt,mc in meths: print(f"    {mn:<16} {mt:>4} lines, {mc:>3} code")
