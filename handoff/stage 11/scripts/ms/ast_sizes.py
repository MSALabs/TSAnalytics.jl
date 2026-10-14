import ast, sys, os
base="/usr/local/lib/python3.13/dist-packages/statsmodels/tsa/regime_switching/"
for fn in ["markov_switching.py","markov_autoregression.py","markov_regression.py"]:
    p=os.path.join(base,fn)
    src=open(p).read()
    tree=ast.parse(src)
    total=len(src.splitlines())
    print("="*78)
    print(f"FILE {fn}  total_lines={total}")
    # module level funcs
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            print(f"  [module-func] {node.name:45s} {node.lineno:5d}-{node.end_lineno:5d}  {node.end_lineno-node.lineno+1:5d}")
    for node in tree.body:
        if isinstance(node,ast.ClassDef):
            print(f"  CLASS {node.name:42s} {node.lineno:5d}-{node.end_lineno:5d}  {node.end_lineno-node.lineno+1:5d}")
            for m in node.body:
                if isinstance(m,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    deco="".join("@"+(d.id if isinstance(d,ast.Name) else getattr(d,'attr',ast.dump(d)[:12]))+" " for d in m.decorator_list)
                    print(f"      {deco}{m.name:40s} {m.lineno:5d}-{m.end_lineno:5d}  {m.end_lineno-m.lineno+1:5d}")
