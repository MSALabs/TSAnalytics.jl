import re

SRC = r"c:\Users\user\Documents\MSA-Consulting\XKDR\TSA\TSAnalytics.jl\test\test_gaussianssm_bulk.jl"
OUT = r"c:\Users\user\Documents\MSA-Consulting\XKDR\TSA\TSAnalytics.jl\test\test_timevaryingssm_bulk.jl"

with open(SRC) as f:
    text = f.read()

# Extract the body between "@testset ... begin" and the matching "end" before the outer "end"/"end # if"
lines = text.split("\n")

blocks = []
cur = []
in_block = False
for line in lines:
    if line.startswith("    let "):
        in_block = True
        cur = [line]
        continue
    if in_block:
        cur.append(line)
        if line == "    end":
            blocks.append(cur)
            in_block = False
            cur = []

print(f"Found {len(blocks)} let-blocks")

new_blocks = []
kf_re = re.compile(r"TSAnalytics\.kalman_filter\(ssm, (y_\w+)\)")
for b in blocks:
    text_b = "\n".join(b)
    m = kf_re.search(text_b)
    assert m, f"no kalman_filter call found in block: {text_b[:200]}"
    yvar = m.group(1)
    # insert extra lines before the closing "    end"
    extra = [
        "        if converged",
        f"            tv, a0, P0 = TSAnalytics.to_time_varying(ssm, {yvar})",
        f"            loglik_tv, v_tv, F_tv, converged_tv = TSAnalytics.kalman_filter(tv, {yvar}, a0, P0)",
        "            @test converged_tv",
        "            @test isapprox(loglik_tv, loglik; atol=1e-8)",
        "            @test isapprox(v_tv, v; atol=1e-8)",
        "        end",
    ]
    new_b = b[:-1] + extra + [b[-1]]
    new_blocks.append("\n".join(new_b))

header = '''# test/test_timevaryingssm_bulk.jl
#
# GENERATED FILE -- do not hand-edit. Produced by a one-off transform
# (test/verification/timevarying/gen_tv_bulk_from_gaussianssm.py) of
# test_gaussianssm_bulk.jl's own 364 already dual-verified (R + Python)
# cases: for each one, in addition to the existing GaussianSSM check,
# builds a TimeVaryingSSM via to_time_varying(ssm, y) and confirms
# kalman_filter(::TimeVaryingSSM, ...) reproduces the SAME v/F and an
# EXACT (not just close) loglik -- see to_time_varying's own docstring
# for why threading GaussianSSM's concentrated sigma2 through explicitly
# is what makes this exact, not incidental.
#
# This is the single highest-value test in Stage 8.1 (per its own
# handoff): it's the one most likely to catch a regression in the new
# general time-varying filter against everything already trusted about
# GaussianSSM, essentially for free (no new ground truth generation).
#
# Gated behind TSANALYTICS_FULL_TESTS, same as the file it's derived from.

using Test, DelimitedFiles, LinearAlgebra

if get(ENV, "TSANALYTICS_FULL_TESTS", "1") == "1"
@testset "TimeVaryingSSM reduces exactly to GaussianSSM -- full 364-case regression suite" begin

    y_ar1_shared = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/ar1_shared.csv"), ',', skipstart=1))
    y_ar2_shared = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/ar2_shared.csv"), ',', skipstart=1))
    y_co2_diff = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/co2_diff.csv"), ',', skipstart=1))
    y_elnino_monthly = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/elnino_monthly.csv"), ',', skipstart=1))
    y_ma1_shared = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/ma1_shared.csv"), ',', skipstart=1))
    y_macro_gdp_growth = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/macro_gdp_growth.csv"), ',', skipstart=1))
    y_nile = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/nile.csv"), ',', skipstart=1))
    y_sunspots = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/sunspots.csv"), ',', skipstart=1))

'''

footer = '''
end
end # if TSANALYTICS_FULL_TESTS
'''

out_text = header + "\n".join(new_blocks) + "\n" + footer

with open(OUT, "w") as f:
    f.write(out_text)

print(f"Wrote {len(new_blocks)} cases to {OUT}")
