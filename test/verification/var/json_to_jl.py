#!/usr/bin/env python3
"""Convert the Stage 10 reference JSONs into a Julia source file.

TSAnalytics has no JSON dependency and the Stage 10 charter (6.5) says
Project.toml is unchanged by the stage, so the test suite cannot parse the
reference JSONs directly. Rather than transcribe a handful of numbers inline
and lose the rest, this emits every value as Julia literals, so the tests can
assert against the whole bundle.

Run from the repository root:

    python test/verification/var/json_to_jl.py

Regenerate the JSONs first with the R and Python scripts under
test/verification/ if anything upstream changes.
"""
import json
import os

FIX = os.path.join("test", "fixtures", "var")
OUT = os.path.join(FIX, "refs.jl")

SOURCES = [
    ("REF_R", "ref_R.json"),
    ("REF_PY", "ref_py.json"),
    ("REF_R_EXTRA", "ref_R_extra.json"),
    ("REF_PY_EXTRA", "ref_py_extra.json"),
    ("REF_R_GRANGER", "ref_R_granger.json"),
    ("REF_PY_GRANGER", "ref_py_granger.json"),
    ("REF_R_PRIM", "ref_R_prim.json"),
    ("REF_PY_PRIM", "ref_py_prim.json"),
    ("REF_R_IRF", "ref_R_irf.json"),
    ("REF_PY_IRF", "ref_py_irf.json"),
    ("VERIFY_IRF_SE", "verify_irf_se.json"),
    ("REF_R_COINT", "ref_R_coint.json"),
    ("REF_PY_COINT", "ref_py_coint.json"),
    ("REF_R_JOHANSEN", "ref_R_johansen.json"),
    ("REF_PY_JOHANSEN", "ref_py_johansen.json"),
    ("REF_R_VECM", "ref_R_vecm.json"),
    ("REF_PY_VECM", "ref_py_vecm.json"),
    ("REF_R_SVAR", "ref_R_svar.json"),
    ("REF_PY_SVAR", "ref_py_svar.json"),
    ("REF_R_DIAG", "ref_R_diag.json"),
    ("REF_PY_DIAG", "ref_py_diag.json"),
]


def jl(v):
    """Render a JSON value as a Julia literal."""
    if v is None:
        return "nothing"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "NaN"
        if v == float("inf"):
            return "Inf"
        if v == float("-inf"):
            return "-Inf"
        # repr gives the shortest round-tripping form, which is what we want:
        # these are reference values and must survive the trip exactly.
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, list):
        if not v:
            return "Any[]"
        return "Any[" + ", ".join(jl(x) for x in v) + "]"
    if isinstance(v, dict):
        if not v:
            return "Dict{String,Any}()"
        items = ", ".join('%s => %s' % (jl(str(k)), jl(x)) for k, x in v.items())
        return "Dict{String,Any}(" + items + ")"
    raise TypeError("cannot render %r" % type(v))


def main():
    parts = [
        "# GENERATED FILE -- do not edit by hand.",
        "#",
        "# Produced by test/verification/var/json_to_jl.py from the reference JSONs",
        "# in this directory. TSAnalytics has no JSON dependency and the Stage 10",
        "# charter keeps Project.toml unchanged, so the references are carried as",
        "# Julia literals instead. Regenerate with:",
        "#",
        "#     python test/verification/var/json_to_jl.py",
        "",
    ]
    for name, fname in SOURCES:
        path = os.path.join(FIX, fname)
        with open(path) as fh:
            data = json.load(fh)
        parts.append("const %s = %s" % (name, jl(data)))
        parts.append("")
    with open(OUT, "w") as fh:
        fh.write("\n".join(parts))
    print("wrote %s (%d bytes)" % (OUT, os.path.getsize(OUT)))


if __name__ == "__main__":
    main()
