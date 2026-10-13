#!/usr/bin/env python3
"""Emit statsmodels' MacKinnon (1994/2010) response-surface tables as Julia literals.

statsmodels is BSD-3, so its tables may be transcribed (charter section 4);
urca's (GPL) are used only as an independent check, in test/test_coint.jl.
Run with statsmodels 0.15.0 (the Stage 10 reference version):

    python test/verification/coint/gen_mackinnon_tables.py > tables.jl

and paste the output over the block in src/unitroot.jl. The values printed
are the coefficients AFTER statsmodels' module-level scaling -- what
mackinnonp/mackinnoncrit actually evaluate.
"""
import numpy as np
import statsmodels
import statsmodels.tsa.adfvalues as a


def f(x):
    return "Inf" if np.isinf(x) else repr(float(x))


REGS = [("n", "nc"), ("c", "c"), ("ct", "ct"), ("ctt", "ctt")]
print("# statsmodels %s, statsmodels/tsa/adfvalues.py" % statsmodels.__version__)
for name, key in [("STAR", "star"), ("MIN", "min"), ("MAX", "max")]:
    print("const _ADF_TAU_%s = Dict(" % name)
    for jl, sm in REGS:
        v = np.asarray(getattr(a, "tau_%s_%s" % (key, sm)))
        print("    :%s => (%s,)," % (jl, ", ".join(f(x) for x in v)))
    print(")")
for name, key in [("SMALLP", "smallp"), ("LARGEP", "largep")]:
    print("const _ADF_TAU_%s = Dict(" % name)
    for jl, sm in REGS:
        v = np.asarray(getattr(a, "tau_%s_%s" % (sm, key)))
        print("    :%s => (%s)," % (jl, ", ".join("(" + ", ".join(f(x) for x in r) + ")" for r in v)))
    print(")")
print("const _ADF_CRIT_2010 = Dict(")
for jl, sm in REGS:
    v = np.asarray(getattr(a, "tau_%s_2010" % sm))
    rows = ", ".join("(" + ", ".join("(" + ", ".join(f(x) for x in lev) + ")" for lev in r) + ")" for r in v)
    print("    :%s => (%s,)," % (jl, rows))   # trailing comma: a 1-row table must stay a tuple
print(")")
