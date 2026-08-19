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
lines.append("# test/test_sarima_bulk.jl")
lines.append("#")
lines.append("# GENERATED FILE -- do not hand-edit. Produced by")
lines.append("# test/verification/sarima/bulk/gen_sarima_cases.py + fit_r.R + fit_python.py")
lines.append("# (see test/verification/sarima/bulk/gen_julia_test.py for the merge/emit step).")
lines.append("#")
lines.append(f"# {len(manifest)} synthetic SARIMA(p,d,q)(P,D,Q)_s series, each dual-verified: R's")
lines.append('# arima(order=c(p,d,q), seasonal=list(order=c(P,D,Q),period=s),')
lines.append('# include.mean=FALSE, method="ML") and Python\'s')
lines.append('# SARIMAX(order=(p,d,q), seasonal_order=(P,D,Q,s), trend="n").fit() both fit')
lines.append("# the exact same data. Sweeps every polynomial block in isolation and in")
lines.append("# combination (including the handoff's own explicitly-flagged highest-risk")
lines.append("# gap: a case with p,q,P,Q all > 0, exercising the theta/Theta")
lines.append("# parameter-unpacking indices no single-seasonal-term case can reach),")
lines.append("# d/D=0 and d/D>0, and two seasonal periods (s=4, s=12).")
lines.append("#")
lines.append("# Tolerances loose (1e-2) for coefficients / (1e-1) for loglik/aic, same")
lines.append("# reasoning as test_arima_bulk.jl: independent ML fits via different")
lines.append("# optimizers, not fixed-coefficient evaluation.")
lines.append("#")
lines.append("# Regenerate with (from test/verification/sarima/bulk/):")
lines.append("#   python gen_sarima_cases.py")
lines.append("#   python fit_python.py")
lines.append('#   Rscript fit_r.R   # or the full path to Rscript.exe')
lines.append("#   python gen_julia_test.py")
lines.append("")
lines.append("using DelimitedFiles, StatsAPI")
lines.append("")
lines.append('if get(ENV, "TSANALYTICS_FULL_TESTS", "1") == "1"')
lines.append('@testset "fit_sarima bulk -- dual-verified R+Python synthetic sweep" begin')

for i, case in enumerate(manifest, start=1):
    name = case["name"]
    p, d, q, P, D, Q, s, n = case["p"], case["d"], case["q"], case["P"], case["D"], case["Q"], case["s"], case["n"]
    r = r_results[name]
    py = py_results[name]
    if "error" in r or "error" in py:
        continue

    r_ar, r_ma, r_sar, r_sma = (as_list(r.get(k)) for k in ("ar", "ma", "sar", "sma"))
    py_ar, py_ma, py_sar, py_sma = (as_list(py.get(k)) for k in ("ar", "ma", "sar", "sma"))
    r_se = as_list(r.get("se"))

    lines.append(f"    let name = \"{name}\"  # case {i}: order=({p},{d},{q}), seasonal_order=({P},{D},{Q},{s}), n={n}")
    lines.append(f'        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "{name}.csv"), \',\', skipstart=1))')
    lines.append(f"        m = fit_sarima(y, ({p}, {d}, {q}), ({P}, {D}, {Q}, {s}); include_mean=false)")
    lines.append(f"        @test m.converged")
    if p > 0:
        lines.append(f"        @test isapprox(m.phi, {r_ar}; atol=1e-2)  # R")
        lines.append(f"        @test isapprox(m.phi, {py_ar}; atol=1e-2)  # Python")
    else:
        lines.append(f"        @test isempty(m.phi)")
    if q > 0:
        lines.append(f"        @test isapprox(m.theta, {r_ma}; atol=1e-2)  # R")
        lines.append(f"        @test isapprox(m.theta, {py_ma}; atol=1e-2)  # Python")
    else:
        lines.append(f"        @test isempty(m.theta)")
    if P > 0:
        lines.append(f"        @test isapprox(m.Phi, {r_sar}; atol=1e-2)  # R")
        lines.append(f"        @test isapprox(m.Phi, {py_sar}; atol=1e-2)  # Python")
    else:
        lines.append(f"        @test isempty(m.Phi)")
    if Q > 0:
        lines.append(f"        @test isapprox(m.Theta, {r_sma}; atol=1e-2)  # R")
        lines.append(f"        @test isapprox(m.Theta, {py_sma}; atol=1e-2)  # Python")
    else:
        lines.append(f"        @test isempty(m.Theta)")
    lines.append(f"        @test isapprox(m.loglik, {r['loglik']}; atol=1e-1)  # R")
    lines.append(f"        @test isapprox(m.loglik, {py['llf']}; atol=1e-1)  # Python")
    lines.append(f"        @test isapprox(m.aic, {r['aic']}; atol=1e-1)  # R")
    lines.append(f"        @test isapprox(m.aic, {py['aic']}; atol=1e-1)  # Python")
    lines.append(f"        @test StatsAPI.nobs(m) == {r['nobs']}  # R's n-d-D*s convention (Python reports {py['nobs']})")
    if r_se:
        se_str = "[" + ", ".join(f"{v}" for v in r_se) + "]"
        lines.append(f"        @test isapprox(m.se, {se_str}; atol=5e-2)  # R hessian se")
    lines.append("    end")
    lines.append("")

lines.append("end")
lines.append("end # if TSANALYTICS_FULL_TESTS")
lines.append("")

OUT = os.path.join(DIR, "..", "..", "..", "test_sarima_bulk.jl")  # test/verification/sarima/bulk/ -> test/
with open(OUT, "w") as f:
    f.write("\n".join(lines))

print(f"wrote {len(manifest)} cases")
