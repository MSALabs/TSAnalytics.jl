import json, os

DIR = os.path.dirname(os.path.abspath(__file__))
manifest = json.load(open(f"{DIR}/manifest.json"))
r_results = json.load(open(f"{DIR}/r_results.json"))
py_results = json.load(open(f"{DIR}/python_results.json"))

def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

lines = []
lines.append("# test/test_arima_bulk.jl")
lines.append("#")
lines.append("# GENERATED FILE -- do not hand-edit. Produced by")
lines.append("# test/verification/arima/bulk/gen_arima_cases.py + fit_r.R + fit_python.py")
lines.append("# (see test/verification/arima/bulk/gen_julia_test.py for the merge/emit step).")
lines.append("#")
lines.append(f"# {len(manifest)} synthetic ARIMA(p,d,q) series, each dual-verified: R's")
lines.append('# arima(order=c(p,d,q), include.mean=FALSE, method="ML") and Python\'s')
lines.append('# ARIMA(order=(p,d,q), trend="n").fit() both fit the exact same data.')
lines.append("# Sweeps d in {0,1,2} across (p,q) in {(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(0,0)},")
lines.append("# each on its own independently-generated synthetic series (known true")
lines.append("# AR/MA structure, integrated d times for d>0), per")
lines.append("# handoff/stage-6.6-arima-handoff.md section 8's request for a systematic")
lines.append("# sweep beyond the handoff's own two spot-check cases.")
lines.append("#")
lines.append("# Tolerances are loose (1e-2) relative to Stage 6.5's single-case tests --")
lines.append("# these are 20 independent ML fits via two different optimizers (Julia's")
lines.append("# LBFGS vs R's BFGS / Python's L-BFGS-B), not fixed-coefficient likelihood")
lines.append("# evaluation, so small convergence-path differences are expected and normal,")
lines.append("# not a bug -- same reasoning as the single ARMA(1,1) case in")
lines.append("# test_arma.jl. loglik/aic are checked tighter (1e-1) since all three")
lines.append("# implementations are climbing the same likelihood surface regardless of")
lines.append("# path, so should land close to the same peak.")
lines.append("#")
lines.append("# Regenerate with (from test/verification/arima/bulk/):")
lines.append("#   python gen_arima_cases.py")
lines.append("#   python fit_python.py")
lines.append('#   Rscript fit_r.R   # or the full path to Rscript.exe')
lines.append("#   python gen_julia_test.py")
lines.append("")
lines.append("using DelimitedFiles")
lines.append("")
lines.append('if get(ENV, "TSANALYTICS_FULL_TESTS", "1") == "1"')
lines.append('@testset "fit_arima bulk -- dual-verified R+Python synthetic sweep" begin')

for i, case in enumerate(manifest, start=1):
    name, p, d, q, n = case["name"], case["p"], case["d"], case["q"], case["n"]
    r = r_results[name]
    py = py_results[name]
    if "error" in r or "error" in py:
        continue

    r_ar = as_list(r.get("ar"))
    r_ma = as_list(r.get("ma"))
    py_ar = as_list(py.get("ar"))
    py_ma = as_list(py.get("ma"))
    r_se = as_list(r.get("se"))

    lines.append(f"    let name = \"{name}\"  # case {i}: order=({p},{d},{q}), n={n}")
    lines.append(f'        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "{name}.csv"), \',\', skipstart=1))')
    lines.append(f"        m = fit_arima(y, ({p}, {d}, {q}); include_mean=false)")
    lines.append(f"        @test m.arma.converged")
    lines.append(f"        @test m.d == {d}")
    if p > 0:
        lines.append(f"        @test isapprox(m.arma.ar, {r_ar}; atol=1e-2)  # R")
        lines.append(f"        @test isapprox(m.arma.ar, {py_ar}; atol=1e-2)  # Python")
    else:
        lines.append(f"        @test isempty(m.arma.ar)")
    if q > 0:
        lines.append(f"        @test isapprox(m.arma.ma, {r_ma}; atol=1e-2)  # R")
        lines.append(f"        @test isapprox(m.arma.ma, {py_ma}; atol=1e-2)  # Python")
    else:
        lines.append(f"        @test isempty(m.arma.ma)")
    lines.append(f"        @test isapprox(m.arma.loglik, {r['loglik']}; atol=1e-1)  # R")
    lines.append(f"        @test isapprox(m.arma.loglik, {py['llf']}; atol=1e-1)  # Python")
    lines.append(f"        @test isapprox(m.arma.aic, {r['aic']}; atol=1e-1)  # R")
    lines.append(f"        @test isapprox(m.arma.aic, {py['aic']}; atol=1e-1)  # Python")
    # nobs: R convention (n-d), which is what fit_arima matches -- Python's differs (full n)
    lines.append(f"        @test StatsAPI.nobs(m) == {r['nobs']}  # R's n-d convention (Python reports {py['nobs']} = full n instead)")
    if r_se:
        se_str = "[" + ", ".join(f"{v}" for v in r_se) + "]"
        lines.append(f"        @test isapprox(m.arma.se, {se_str}; atol=5e-2)  # R hessian se")
    lines.append("    end")
    lines.append("")

lines.append("end")
lines.append("end # if TSANALYTICS_FULL_TESTS")
lines.append("")

OUT = os.path.join(DIR, "..", "..", "..", "test_arima_bulk.jl")  # test/verification/arima/bulk/ -> test/
with open(OUT, "w") as f:
    f.write("\n".join(lines))

print(f"wrote {len(manifest)} cases")
