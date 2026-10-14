import ast, sys, os
for path in sys.argv[1:]:
    src = open(path).read()
    tree = ast.parse(src)
    nlines = len(src.splitlines())
    print("="*100)
    print(f"FILE {path}  total_lines={nlines}")
    print("="*100)
    toplevel_func_total = 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            span = node.end_lineno - node.lineno + 1
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            print(f"\nCLASS {node.name}({bases})  lines {node.lineno}-{node.end_lineno}  span={span}")
            meth_total = 0
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    s = sub.end_lineno - sub.lineno + 1
                    meth_total += s
                    deco = ",".join(ast.unparse(d) for d in sub.decorator_list)
                    # docstring lines
                    ds = ast.get_docstring(sub)
                    dslines = len(ds.splitlines()) if ds else 0
                    print(f"    {sub.name:<40} {sub.lineno:>5}-{sub.end_lineno:<5} span={s:<5} code~{s-dslines:<5} deco={deco}")
            print(f"    -- methods total span = {meth_total}  (class span {span}, non-method {span-meth_total})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s = node.end_lineno - node.lineno + 1
            toplevel_func_total += s
            print(f"\nFUNC  {node.name:<40} {node.lineno:>5}-{node.end_lineno:<5} span={s}")
    print(f"\n[toplevel function total span = {toplevel_func_total}]")
