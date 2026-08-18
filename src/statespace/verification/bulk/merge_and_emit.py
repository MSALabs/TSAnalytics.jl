"""
Stage 2: merge R and Python loglik results, keep only cases where both agree
(dual-language verified, same bar as the original 3 ground-truth cases), and
emit test/test_gaussianssm_bulk.jl -- a generated Julia test file, one @test
per verified case, plus a manifest CSV for human inspection.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TEST_DIR = os.path.join(REPO_ROOT, "test")

with open(os.path.join(HERE, "cases.json")) as f:
    cases = json.load(f)
cases_by_id = {c["id"]: c for c in cases}

r_results = {}
with open(os.path.join(HERE, "r_results.csv")) as f:
    for row in csv.DictReader(f):
        cid = int(row["id"])
        r_results[cid] = row

TOL = 1e-4  # absolute loglik tolerance for R vs Python agreement

matched, mismatched, r_errors = [], [], []
for cid, c in cases_by_id.items():
    r = r_results.get(cid)
    if r is None:
        continue
    if r["r_error"] == "error" or r["r_loglik"] in ("NA", ""):
        r_errors.append(cid)
        continue
    r_ll = float(r["r_loglik"])
    py_ll = c["py_loglik"]
    if abs(r_ll - py_ll) <= TOL:
        c["r_loglik"] = r_ll
        matched.append(c)
    else:
        mismatched.append((cid, c["dataset"], r_ll, py_ll, abs(r_ll - py_ll)))

print(f"Matched (dual-verified): {len(matched)}")
print(f"R errored out: {len(r_errors)}")
print(f"Mismatched beyond tol={TOL}: {len(mismatched)}")
for m in mismatched[:20]:
    print("  MISMATCH", m)

# Manifest CSV for human inspection
manifest_path = os.path.join(HERE, "verified_cases.csv")
with open(manifest_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "dataset", "p", "q", "P", "Q", "s", "ar", "ma", "sar", "sma", "r_loglik", "py_loglik"])
    for c in matched:
        w.writerow([c["id"], c["dataset"], c["p"], c["q"], c["P"], c["Q"], c["s"],
                    c["ar"], c["ma"], c["sar"], c["sma"], c["r_loglik"], c["py_loglik"]])
print(f"Wrote {manifest_path}")


# ---------------------------------------------------------------------------
# Emit the generated Julia test file
# ---------------------------------------------------------------------------
def jlvec(xs):
    if not xs:
        return "Float64[]"
    return "[" + ", ".join(repr(float(x)) for x in xs) + "]"


datasets_used = sorted({c["dataset"] for c in matched})

lines = []
lines.append("# test/test_gaussianssm_bulk.jl")
lines.append("#")
lines.append("# GENERATED FILE -- do not hand-edit. Produced by")
lines.append("# verification/bulk/gen_cases.py + run_r.R + merge_and_emit.py.")
lines.append("#")
lines.append(f"# {len(matched)} cases, each dual-verified: R's arima(..., fixed=..., ")
lines.append("# transform.pars=FALSE, optim.control=list(maxit=0)) and Python's")
lines.append("# SARIMAX(..., concentrate_scale=True).loglike(...) agree to within")
lines.append(f"# {TOL} on every case below, matching the same dual-language standard")
lines.append("# as the original 3-case gate in test_gaussianssm.jl. Covers pure AR(1-3),")
lines.append("# pure MA(1-3), ARMA(1,1)/(2,1)/(1,2)/(2,2), and seasonal SARIMA structures")
lines.append("# (the combined_ar_ma path flagged as untested in the project brief),")
lines.append("# across 3 synthetic series and 5 real/textbook datasets (Nile, sunspots,")
lines.append("# macro GDP growth, El Nino SST, Mauna Loa CO2).")
lines.append("#")
lines.append("# Regenerate with:")
lines.append("#   python verification/bulk/gen_cases.py")
lines.append('#   Rscript verification/bulk/run_r.R   # or the full path to Rscript.exe')
lines.append("#   python verification/bulk/merge_and_emit.py")
lines.append("")
lines.append("using Test, DelimitedFiles, LinearAlgebra")
lines.append("")
lines.append('@testset "GaussianSSM bulk -- dual-verified R+Python, real+synthetic datasets" begin')
lines.append("")

for dname in datasets_used:
    varname = f"y_{dname}"
    relpath = os.path.join("..", "verification", "bulk", "data", f"{dname}.csv").replace("\\", "/")
    lines.append(f'    {varname} = vec(readdlm(joinpath(@__DIR__, "{relpath}"), \',\', skipstart=1))')
lines.append("")

for c in matched:
    varname = f"y_{c['dataset']}"
    ar, ma, sar, sma, s = c["ar"], c["ma"], c["sar"], c["sma"], c["s"]
    ll = c["py_loglik"]
    lines.append(
        f"    let ar = {jlvec(ar)}, ma = {jlvec(ma)}, sar = {jlvec(sar)}, sma = {jlvec(sma)}, s = {s}"
    )
    lines.append("        ar_c, ma_c = combined_ar_ma(ar, sar, ma, sma, max(s, 1))")
    lines.append("        ssm = build_statespace(ar_c, ma_c)")
    lines.append(f"        loglik, sigma2, v, F, converged = kalman_filter(ssm, {varname})")
    lines.append(f"        @test converged  # case {c['id']}: {c['dataset']}, order=({c['p']},0,{c['q']}), seasonal=({c['P']},0,{c['Q']},{s})")
    lines.append(f"        @test isapprox(loglik, {ll!r}; atol=1e-4)")
    lines.append("    end")

lines.append("end")

out_path = os.path.join(TEST_DIR, "test_gaussianssm_bulk.jl")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {out_path} with {len(matched)} generated @test cases")
