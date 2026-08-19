# Bulk volatility-forecast verification, per Stage 7.3 handoff section 5:
# reuses all 24 GARCH (Stage 7.1) + 48 GJR-GARCH/EGARCH (Stage 7.2) already-
# fitted series -- 72 models x 3 horizons (1, 5, 10) = 216 forecast checks,
# built entirely on data already generated and dual-verified in 7.1/7.2
# rather than a new grid. No new Python ground truth needed here: the checks
# are internal consistency (positivity, stability) and analytic-vs-simulation
# agreement, exactly per the handoff's own framing -- not against a fresh
# external reference.

if get(ENV, "TSANALYTICS_FULL_TESTS", "1") == "1"
@testset "forecast_volatility bulk (72 models x 3 horizons = 216 checks)" begin
    garch_dir = joinpath(@__DIR__, "verification", "garch", "bulk", "data")
    gjr_egarch_dir = joinpath(@__DIR__, "verification", "garch", "bulk_gjr_egarch", "data")

    cases = vcat(
        [(joinpath(garch_dir, f), :garch) for f in readdir(garch_dir) if endswith(f, ".csv")],
        [(joinpath(gjr_egarch_dir, f), :gjr) for f in readdir(gjr_egarch_dir) if occursin("_gjr_", f)],
        [(joinpath(gjr_egarch_dir, f), :egarch) for f in readdir(gjr_egarch_dir) if occursin("_egarch_", f)],
    )
    @test length(cases) == 24 + 24 + 24  # 24 plain GARCH + 24 GJR + 24 EGARCH = 72

    horizons = (1, 5, 10)
    n_checks = 0
    max_analytic_sim_diff = 0.0
    for (path, model) in cases
        e = vec(readdlm(path, ','; skipstart=1))
        m = fit_garch(e, 1, 1; model=model)
        @test m.converged

        for h in horizons
            fc_auto = forecast_volatility(m, h; method=:auto)
            @test all(fc_auto.variance .> 0)
            @test length(fc_auto.variance) == h
            n_checks += 1

            if model != :egarch
                fc_sim = forecast_volatility(m, h; method=:simulation, simulations=20000)
                d = maximum(abs.(fc_sim.variance .- fc_auto.variance))
                max_analytic_sim_diff = max(max_analytic_sim_diff, d)
                @test d < 0.5   # loose Monte Carlo tolerance across the whole grid, not a single tuned case
            end
        end
    end
    @info "forecast_volatility bulk: checks run, max |simulation - analytic| across GARCH/GJR grid" n_checks max_analytic_sim_diff
    @test n_checks == 216
end
end
