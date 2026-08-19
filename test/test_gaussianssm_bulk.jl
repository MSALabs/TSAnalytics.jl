# test/test_gaussianssm_bulk.jl
#
# GENERATED FILE -- do not hand-edit (except the mechanical
# TSAnalytics.-prefixing and path updates applied when this moved into
# TSAnalytics.jl proper -- see handoff/stage-6-arima-handoff.md §1).
# Produced by verification/bulk/gen_cases.py + run_r.R + merge_and_emit.py.
#
# 364 cases, each dual-verified: R's arima(..., fixed=..., 
# transform.pars=FALSE, optim.control=list(maxit=0)) and Python's
# SARIMAX(..., concentrate_scale=True).loglike(...) agree to within
# 0.0001 on every case below, matching the same dual-language standard
# as the original 3-case gate in test_gaussianssm.jl. Covers pure AR(1-3),
# pure MA(1-3), ARMA(1,1)/(2,1)/(1,2)/(2,2), and seasonal SARIMA structures
# (the combined_ar_ma path flagged as untested in the project brief),
# across 3 synthetic series and 5 real/textbook datasets (Nile, sunspots,
# macro GDP growth, El Nino SST, Mauna Loa CO2).
#
# Regenerate with (from test/verification/gaussianssm/):
#   python bulk/gen_cases.py
#   Rscript bulk/run_r.R   # or the full path to Rscript.exe
#   python bulk/merge_and_emit.py
#
# 364 cases dominates this project's test file count relative to its
# actual runtime cost (the filter is O(n*r^2), so it's fast) -- gated
# behind TSANALYTICS_FULL_TESTS so a fast local loop can skip it
# (handoff/stage-6-arima-handoff.md §1.9).

using Test, DelimitedFiles, LinearAlgebra

if get(ENV, "TSANALYTICS_FULL_TESTS", "1") == "1"
@testset "GaussianSSM bulk -- dual-verified R+Python, real+synthetic datasets" begin

    y_ar1_shared = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/ar1_shared.csv"), ',', skipstart=1))
    y_ar2_shared = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/ar2_shared.csv"), ',', skipstart=1))
    y_co2_diff = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/co2_diff.csv"), ',', skipstart=1))
    y_elnino_monthly = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/elnino_monthly.csv"), ',', skipstart=1))
    y_ma1_shared = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/ma1_shared.csv"), ',', skipstart=1))
    y_macro_gdp_growth = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/macro_gdp_growth.csv"), ',', skipstart=1))
    y_nile = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/nile.csv"), ',', skipstart=1))
    y_sunspots = vec(readdlm(joinpath(@__DIR__, "verification/gaussianssm/bulk/data/sunspots.csv"), ',', skipstart=1))

    let ar = [0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 1: ar1_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -152.69761145364052; atol=1e-4)
    end
    let ar = [0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 2: ar1_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -143.81745153156382; atol=1e-4)
    end
    let ar = [0.7], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 3: ar1_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -141.03103076138504; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 4: ar1_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -191.87114522295067; atol=1e-4)
    end
    let ar = [-0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 5: ar1_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -204.5853421759724; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 6: ar1_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -142.2119053367943; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 7: ar1_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -146.85927522689795; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 8: ar1_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -147.47029084200474; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 9: ar1_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -186.75539114971482; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 10: ar1_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -152.1744530243335; atol=1e-4)
    end
    let ar = [0.5, -0.2, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 11: ar1_shared, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -149.4292435011469; atol=1e-4)
    end
    let ar = [0.3, 0.2, -0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 12: ar1_shared, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -154.71710689443907; atol=1e-4)
    end
    let ar = [0.6, -0.1, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 13: ar1_shared, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -149.45947085190906; atol=1e-4)
    end
    let ar = Float64[], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 14: ar1_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -160.2399415487617; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 15: ar1_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -153.2468383119139; atol=1e-4)
    end
    let ar = Float64[], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 16: ar1_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -152.48076973421448; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 17: ar1_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -197.06438750764943; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 18: ar1_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -221.5109251863897; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 19: ar1_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -147.8146267967155; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 20: ar1_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -170.89832524344175; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 21: ar1_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -194.09431907303352; atol=1e-4)
    end
    let ar = Float64[], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 22: ar1_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -156.31481707687647; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 23: ar1_shared, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -149.7659160952457; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.2, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 24: ar1_shared, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -164.36068700727483; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3, 0.2, 0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 25: ar1_shared, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -184.09698171113723; atol=1e-4)
    end
    let ar = [0.3], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 26: ar1_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -146.70280989358133; atol=1e-4)
    end
    let ar = [0.5], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 27: ar1_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -147.4376829090952; atol=1e-4)
    end
    let ar = [0.7], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 28: ar1_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -167.96338188295977; atol=1e-4)
    end
    let ar = [-0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 29: ar1_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -219.89996800221832; atol=1e-4)
    end
    let ar = [-0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 30: ar1_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -259.43270532713177; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 31: ar1_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -141.2262451302951; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 32: ar1_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -141.0057568010333; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 33: ar1_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -170.98660961070794; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 34: ar1_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -212.3485089901882; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 35: ar1_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -176.21487402153613; atol=1e-4)
    end
    let ar = [0.3], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 36: ar1_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -142.40885258526257; atol=1e-4)
    end
    let ar = [0.5], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 37: ar1_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -156.95114992538336; atol=1e-4)
    end
    let ar = [0.7], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 38: ar1_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -147.9032873024266; atol=1e-4)
    end
    let ar = [-0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 39: ar1_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -166.43361464253798; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 40: ar1_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -143.70606251958156; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 41: ar1_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -145.71487138811077; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 42: ar1_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -155.84614580799075; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar1_shared)
        @test converged  # case 43: ar1_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -160.02373156678334; atol=1e-4)
    end
    let ar = [0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 44: ar2_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -151.67676721285432; atol=1e-4)
    end
    let ar = [0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 45: ar2_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -145.55735697112763; atol=1e-4)
    end
    let ar = [0.7], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 46: ar2_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -145.04120629073037; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 47: ar2_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -185.76941728507927; atol=1e-4)
    end
    let ar = [-0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 48: ar2_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -197.83391795034163; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 49: ar2_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -143.2564441233744; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 50: ar2_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -144.44666673769729; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 51: ar2_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -151.28909602937387; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 52: ar2_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -180.73321263356888; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 53: ar2_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -147.7396100546822; atol=1e-4)
    end
    let ar = [0.5, -0.2, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 54: ar2_shared, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -151.43376697809387; atol=1e-4)
    end
    let ar = [0.3, 0.2, -0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 55: ar2_shared, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -155.0321632087179; atol=1e-4)
    end
    let ar = [0.6, -0.1, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 56: ar2_shared, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -153.12895189689985; atol=1e-4)
    end
    let ar = Float64[], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 57: ar2_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -157.9277427859475; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 58: ar2_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -153.56635568748132; atol=1e-4)
    end
    let ar = Float64[], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 59: ar2_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -155.3775940737853; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 60: ar2_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -190.9537177472616; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 61: ar2_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -214.90046790516402; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 62: ar2_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -148.47069724448596; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 63: ar2_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -168.27781235932554; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 64: ar2_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -187.14045228110112; atol=1e-4)
    end
    let ar = Float64[], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 65: ar2_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -157.13773804388222; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 66: ar2_shared, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -150.77115228858924; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.2, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 67: ar2_shared, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -160.44006667717525; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3, 0.2, 0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 68: ar2_shared, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -176.41354510854364; atol=1e-4)
    end
    let ar = [0.3], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 69: ar2_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -148.719341886635; atol=1e-4)
    end
    let ar = [0.5], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 70: ar2_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -155.6547238867168; atol=1e-4)
    end
    let ar = [0.7], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 71: ar2_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -178.58943941848386; atol=1e-4)
    end
    let ar = [-0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 72: ar2_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -212.85078867764292; atol=1e-4)
    end
    let ar = [-0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 73: ar2_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -252.09375434428534; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 74: ar2_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -145.60603160690323; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 75: ar2_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -144.85780949763316; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 76: ar2_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -181.2665960433041; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 77: ar2_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -205.23858533384796; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 78: ar2_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -169.0351565612149; atol=1e-4)
    end
    let ar = [0.3], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 79: ar2_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -147.64449908906127; atol=1e-4)
    end
    let ar = [0.5], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 80: ar2_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -160.94208578629392; atol=1e-4)
    end
    let ar = [0.7], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 81: ar2_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -144.490860126337; atol=1e-4)
    end
    let ar = [-0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 82: ar2_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -163.38906464843163; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 83: ar2_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -150.66226625250647; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 84: ar2_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -147.1562231857704; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 85: ar2_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -153.62484069846505; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ar2_shared)
        @test converged  # case 86: ar2_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -157.2289828556735; atol=1e-4)
    end
    let ar = [0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 87: ma1_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -130.69882374979608; atol=1e-4)
    end
    let ar = [0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 88: ma1_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -129.38415945099408; atol=1e-4)
    end
    let ar = [0.7], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 89: ma1_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -132.98312789542518; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 90: ma1_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -156.0908439352466; atol=1e-4)
    end
    let ar = [-0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 91: ma1_shared, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -167.2935396072795; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 92: ma1_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -132.96590308154734; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 93: ma1_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -140.7816766287781; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 94: ma1_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -125.63023777080676; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 95: ma1_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -161.76745460012273; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 96: ma1_shared, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -150.63852791062553; atol=1e-4)
    end
    let ar = [0.5, -0.2, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 97: ma1_shared, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -127.81125706261638; atol=1e-4)
    end
    let ar = [0.3, 0.2, -0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 98: ma1_shared, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -131.22838085966671; atol=1e-4)
    end
    let ar = [0.6, -0.1, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 99: ma1_shared, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -125.74188978032336; atol=1e-4)
    end
    let ar = Float64[], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 100: ma1_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -132.85317079230458; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 101: ma1_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -128.11402004283462; atol=1e-4)
    end
    let ar = Float64[], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 102: ma1_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -128.64328181993793; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 103: ma1_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -156.5500474261598; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 104: ma1_shared, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -169.98541603825566; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 105: ma1_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -128.32998460913302; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 106: ma1_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -133.90572271788884; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 107: ma1_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -163.7233503480382; atol=1e-4)
    end
    let ar = Float64[], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 108: ma1_shared, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -129.38924543331106; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 109: ma1_shared, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -127.03245422108338; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.2, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 110: ma1_shared, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -137.4381721284191; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3, 0.2, 0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 111: ma1_shared, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -162.21095772309968; atol=1e-4)
    end
    let ar = [0.3], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 112: ma1_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -127.29297984079942; atol=1e-4)
    end
    let ar = [0.5], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 113: ma1_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -132.56128784310772; atol=1e-4)
    end
    let ar = [0.7], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 114: ma1_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -152.26440479004103; atol=1e-4)
    end
    let ar = [-0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 115: ma1_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -175.1540476389148; atol=1e-4)
    end
    let ar = [-0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 116: ma1_shared, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -203.1008063323968; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 117: ma1_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -131.652541672642; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 118: ma1_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -133.60243983008175; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 119: ma1_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -150.90252885525354; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 120: ma1_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -178.07568156814546; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 121: ma1_shared, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -165.13544384064653; atol=1e-4)
    end
    let ar = [0.3], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 122: ma1_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -128.12932048524453; atol=1e-4)
    end
    let ar = [0.5], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 123: ma1_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -135.1837645249413; atol=1e-4)
    end
    let ar = [0.7], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 124: ma1_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -144.22182982385036; atol=1e-4)
    end
    let ar = [-0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 125: ma1_shared, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -134.34435134585976; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 126: ma1_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -135.56816917264817; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 127: ma1_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -131.6030374126396; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 128: ma1_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -134.11586264454024; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_ma1_shared)
        @test converged  # case 129: ma1_shared, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -137.69942842768194; atol=1e-4)
    end
    let ar = [0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 130: nile, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -791.6836317074869; atol=1e-4)
    end
    let ar = [0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 131: nile, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -760.7762814948977; atol=1e-4)
    end
    let ar = [0.7], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 132: nile, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -718.7720024585614; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 133: nile, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -851.6714485944212; atol=1e-4)
    end
    let ar = [-0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 134: nile, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -865.8993294898934; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 135: nile, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -741.4304899483194; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 136: nile, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -741.8863336891503; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 137: nile, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -777.4054750645284; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 138: nile, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -836.0316750662139; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 139: nile, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -720.1383293752218; atol=1e-4)
    end
    let ar = [0.5, -0.2, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 140: nile, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -777.5306386088787; atol=1e-4)
    end
    let ar = [0.3, 0.2, -0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 141: nile, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -804.2475948278866; atol=1e-4)
    end
    let ar = [0.6, -0.1, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 142: nile, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -791.5660887201075; atol=1e-4)
    end
    let ar = Float64[], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 143: nile, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -808.246177361179; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 144: nile, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -793.5851366698411; atol=1e-4)
    end
    let ar = Float64[], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 145: nile, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -781.4841443129908; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 146: nile, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -860.7835975896244; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 147: nile, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -893.6998554683814; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 148: nile, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -781.0858409258603; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 149: nile, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -825.8737051060355; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 150: nile, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -848.1650163014788; atol=1e-4)
    end
    let ar = Float64[], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 151: nile, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -793.8562965800174; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 152: nile, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -787.2351689708681; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.2, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 153: nile, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -801.3034389463228; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3, 0.2, 0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 154: nile, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -826.6535045011135; atol=1e-4)
    end
    let ar = [0.3], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 155: nile, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -774.7362904034135; atol=1e-4)
    end
    let ar = [0.5], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 156: nile, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -733.9604876758817; atol=1e-4)
    end
    let ar = [0.7], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 157: nile, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -704.9806918745569; atol=1e-4)
    end
    let ar = [-0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 158: nile, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -886.7945801873649; atol=1e-4)
    end
    let ar = [-0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 159: nile, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -934.1851964991639; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 160: nile, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -726.8893134350918; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 161: nile, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -715.1525810150741; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 162: nile, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -743.9772812950258; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 163: nile, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -870.9911365335822; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 164: nile, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -780.7719036206389; atol=1e-4)
    end
    let ar = [0.3], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 165: nile, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -749.4632298596487; atol=1e-4)
    end
    let ar = [0.5], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 166: nile, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -762.764976516679; atol=1e-4)
    end
    let ar = [0.7], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 167: nile, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -736.3896894671647; atol=1e-4)
    end
    let ar = [-0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 168: nile, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -818.6834418979693; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 169: nile, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -707.817130375496; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 170: nile, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -742.2200513074097; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 171: nile, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -798.206507663931; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_nile)
        @test converged  # case 172: nile, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -802.8623033782429; atol=1e-4)
    end
    let ar = [0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 173: sunspots, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1626.4391743811611; atol=1e-4)
    end
    let ar = [0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 174: sunspots, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1547.9288944137465; atol=1e-4)
    end
    let ar = [0.7], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 175: sunspots, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1465.7903181115535; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 176: sunspots, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1801.1800683412653; atol=1e-4)
    end
    let ar = [-0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 177: sunspots, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1844.5030586316163; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 178: sunspots, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1519.421579348279; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 179: sunspots, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1547.28640723055; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 180: sunspots, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1572.3535177184763; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 181: sunspots, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1764.7629210496339; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 182: sunspots, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1547.7862159478482; atol=1e-4)
    end
    let ar = [0.5, -0.2, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 183: sunspots, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1588.0275577549135; atol=1e-4)
    end
    let ar = [0.3, 0.2, -0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 184: sunspots, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1639.9464912135745; atol=1e-4)
    end
    let ar = [0.6, -0.1, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 185: sunspots, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1596.2263264084522; atol=1e-4)
    end
    let ar = Float64[], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 186: sunspots, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1671.1586289741253; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 187: sunspots, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1626.3489768815284; atol=1e-4)
    end
    let ar = Float64[], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 188: sunspots, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1588.5045530926593; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 189: sunspots, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1824.5032246692574; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 190: sunspots, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1916.2299131491707; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 191: sunspots, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1592.8404708365704; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 192: sunspots, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1713.1192928417677; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 193: sunspots, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1799.4561883273168; atol=1e-4)
    end
    let ar = Float64[], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 194: sunspots, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1624.176693224053; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 195: sunspots, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1605.5278649275806; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.2, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 196: sunspots, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1658.234643400762; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3, 0.2, 0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 197: sunspots, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1747.5372038516002; atol=1e-4)
    end
    let ar = [0.3], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 198: sunspots, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1575.7453778602724; atol=1e-4)
    end
    let ar = [0.5], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 199: sunspots, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1463.4992212567495; atol=1e-4)
    end
    let ar = [0.7], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 200: sunspots, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1386.592313149877; atol=1e-4)
    end
    let ar = [-0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 201: sunspots, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1902.7773360536453; atol=1e-4)
    end
    let ar = [-0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 202: sunspots, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -2039.2237726813432; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 203: sunspots, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1474.4281286272915; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 204: sunspots, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1462.2813189760022; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 205: sunspots, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1464.4142534144034; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 206: sunspots, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1863.7850560256634; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 207: sunspots, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1689.0856393418073; atol=1e-4)
    end
    let ar = [0.3], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 208: sunspots, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1503.0366271717776; atol=1e-4)
    end
    let ar = [0.5], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 209: sunspots, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1536.2551135593808; atol=1e-4)
    end
    let ar = [0.7], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 210: sunspots, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1541.5005901640636; atol=1e-4)
    end
    let ar = [-0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 211: sunspots, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1699.206214831037; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 212: sunspots, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1416.8098291803944; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 213: sunspots, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1522.3864083068997; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 214: sunspots, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1644.6942792539771; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_sunspots)
        @test converged  # case 215: sunspots, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1662.8435774422112; atol=1e-4)
    end
    let ar = [0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 216: macro_gdp_growth, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -286.4228989564128; atol=1e-4)
    end
    let ar = [0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 217: macro_gdp_growth, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -274.05985407912453; atol=1e-4)
    end
    let ar = [0.7], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 218: macro_gdp_growth, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -273.25585060016925; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 219: macro_gdp_growth, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -356.0479054746303; atol=1e-4)
    end
    let ar = [-0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 220: macro_gdp_growth, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -380.5884835657259; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 221: macro_gdp_growth, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -266.9781537278571; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 222: macro_gdp_growth, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -264.06096776097206; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 223: macro_gdp_growth, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -289.9442503897021; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 224: macro_gdp_growth, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -340.484891514844; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 225: macro_gdp_growth, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -266.28749357609536; atol=1e-4)
    end
    let ar = [0.5, -0.2, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 226: macro_gdp_growth, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -287.34472021597213; atol=1e-4)
    end
    let ar = [0.3, 0.2, -0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 227: macro_gdp_growth, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -298.4647863434467; atol=1e-4)
    end
    let ar = [0.6, -0.1, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 228: macro_gdp_growth, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -296.7653318244837; atol=1e-4)
    end
    let ar = Float64[], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 229: macro_gdp_growth, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -299.8610417696618; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 230: macro_gdp_growth, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -293.2713442165258; atol=1e-4)
    end
    let ar = Float64[], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 231: macro_gdp_growth, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -304.4750285073336; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 232: macro_gdp_growth, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -368.98712561116656; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 233: macro_gdp_growth, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -423.4293845799376; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 234: macro_gdp_growth, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -280.6852125715426; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 235: macro_gdp_growth, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -332.66922709749036; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 236: macro_gdp_growth, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -354.86134799263004; atol=1e-4)
    end
    let ar = Float64[], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 237: macro_gdp_growth, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -308.78540992975235; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 238: macro_gdp_growth, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -285.64682624336575; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.2, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 239: macro_gdp_growth, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -319.76346495108794; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3, 0.2, 0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 240: macro_gdp_growth, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -328.2068292061321; atol=1e-4)
    end
    let ar = [0.3], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 241: macro_gdp_growth, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -282.3531024014617; atol=1e-4)
    end
    let ar = [0.5], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 242: macro_gdp_growth, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -301.34729724208415; atol=1e-4)
    end
    let ar = [0.7], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 243: macro_gdp_growth, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -359.665453050383; atol=1e-4)
    end
    let ar = [-0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 244: macro_gdp_growth, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -414.68257155391444; atol=1e-4)
    end
    let ar = [-0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 245: macro_gdp_growth, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -500.66073100862263; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 246: macro_gdp_growth, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -275.1194314762703; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 247: macro_gdp_growth, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -272.47613368322413; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 248: macro_gdp_growth, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -367.59291763348506; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 249: macro_gdp_growth, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -393.36112560421543; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 250: macro_gdp_growth, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -299.4413136248031; atol=1e-4)
    end
    let ar = [0.3], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 251: macro_gdp_growth, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -280.4696823240187; atol=1e-4)
    end
    let ar = [0.5], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 252: macro_gdp_growth, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -328.9572248864312; atol=1e-4)
    end
    let ar = [0.7], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 253: macro_gdp_growth, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -264.03374269401854; atol=1e-4)
    end
    let ar = [-0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 254: macro_gdp_growth, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -315.2131073460648; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 255: macro_gdp_growth, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -286.4435772101262; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 256: macro_gdp_growth, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -286.0571050420898; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 257: macro_gdp_growth, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -290.0390690358656; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 258: macro_gdp_growth, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -294.1525123864346; atol=1e-4)
    end
    let ar = [0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 259: elnino_monthly, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1265.8766777865412; atol=1e-4)
    end
    let ar = [0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 260: elnino_monthly, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1166.0782497067244; atol=1e-4)
    end
    let ar = [0.7], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 261: elnino_monthly, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1016.3157830027651; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 262: elnino_monthly, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1450.6618775635252; atol=1e-4)
    end
    let ar = [-0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 263: elnino_monthly, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1493.5721059775499; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 264: elnino_monthly, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1100.949537062434; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 265: elnino_monthly, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1102.4020711122193; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 266: elnino_monthly, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1219.4860677721977; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 267: elnino_monthly, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1401.4811290903745; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 268: elnino_monthly, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1022.1257513537008; atol=1e-4)
    end
    let ar = [0.5, -0.2, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 269: elnino_monthly, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1220.1402747076554; atol=1e-4)
    end
    let ar = [0.3, 0.2, -0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 270: elnino_monthly, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1304.8405382735943; atol=1e-4)
    end
    let ar = [0.6, -0.1, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 271: elnino_monthly, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1264.748358746698; atol=1e-4)
    end
    let ar = Float64[], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 272: elnino_monthly, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1317.736745335095; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 273: elnino_monthly, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1271.7086096283988; atol=1e-4)
    end
    let ar = Float64[], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 274: elnino_monthly, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1231.903113301888; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 275: elnino_monthly, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1478.6509105231767; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 276: elnino_monthly, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1578.894472271188; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 277: elnino_monthly, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1231.9980943639762; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 278: elnino_monthly, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1371.8732879243291; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 279: elnino_monthly, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1439.3784988730822; atol=1e-4)
    end
    let ar = Float64[], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 280: elnino_monthly, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1271.693667819759; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 281: elnino_monthly, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1251.0970258357759; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.2, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 282: elnino_monthly, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1294.4209852524466; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3, 0.2, 0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 283: elnino_monthly, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1373.166956254061; atol=1e-4)
    end
    let ar = [0.3], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 284: elnino_monthly, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1211.4980592588213; atol=1e-4)
    end
    let ar = [0.5], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 285: elnino_monthly, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1065.9291488150222; atol=1e-4)
    end
    let ar = [0.7], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 286: elnino_monthly, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -877.5562980117435; atol=1e-4)
    end
    let ar = [-0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 287: elnino_monthly, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1557.2162922433743; atol=1e-4)
    end
    let ar = [-0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 288: elnino_monthly, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1700.528901917027; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 289: elnino_monthly, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1046.7265840083278; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 290: elnino_monthly, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1002.298866433582; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 291: elnino_monthly, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1079.7814163625694; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 292: elnino_monthly, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1507.9883330389684; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 293: elnino_monthly, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1225.9575391420135; atol=1e-4)
    end
    let ar = [0.3], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 294: elnino_monthly, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1125.8960891796332; atol=1e-4)
    end
    let ar = [0.5], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 295: elnino_monthly, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1165.7265943204998; atol=1e-4)
    end
    let ar = [0.7], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 296: elnino_monthly, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1083.541110892641; atol=1e-4)
    end
    let ar = [-0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 297: elnino_monthly, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1349.97276909314; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 298: elnino_monthly, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -961.580336141802; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 299: elnino_monthly, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1101.3971781102582; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 300: elnino_monthly, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1286.3991037399142; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 301: elnino_monthly, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -1300.5297478244408; atol=1e-4)
    end
    let ar = [0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 302: co2_diff, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -311.08718448650757; atol=1e-4)
    end
    let ar = [0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 303: co2_diff, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -333.6147132944657; atol=1e-4)
    end
    let ar = [0.7], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 304: co2_diff, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -365.68477439879246; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 305: co2_diff, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -328.6936059050421; atol=1e-4)
    end
    let ar = [-0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 306: co2_diff, order=(1,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -359.3029362077102; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 307: co2_diff, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -331.0087261507979; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 308: co2_diff, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -311.71910067020065; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 309: co2_diff, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -361.8974047346777; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 310: co2_diff, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -338.0474257937599; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 311: co2_diff, order=(2,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -322.7320183551899; atol=1e-4)
    end
    let ar = [0.5, -0.2, 0.1], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 312: co2_diff, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -345.29543527145336; atol=1e-4)
    end
    let ar = [0.3, 0.2, -0.3], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 313: co2_diff, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -338.9940765074633; atol=1e-4)
    end
    let ar = [0.6, -0.1, -0.2], ma = Float64[], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 314: co2_diff, order=(3,0,0), seasonal=(0,0,0,0)
        @test isapprox(loglik, -366.91946100427964; atol=1e-4)
    end
    let ar = Float64[], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 315: co2_diff, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -306.8628446716625; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 316: co2_diff, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -330.5226129621651; atol=1e-4)
    end
    let ar = Float64[], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 317: co2_diff, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -379.78814976031583; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 318: co2_diff, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -339.4187504430279; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 319: co2_diff, order=(0,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -404.77682950362237; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 320: co2_diff, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -324.64371420638514; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 321: co2_diff, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -359.3658643803949; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 322: co2_diff, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -332.1939320799471; atol=1e-4)
    end
    let ar = Float64[], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 323: co2_diff, order=(0,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -367.3010879981341; atol=1e-4)
    end
    let ar = Float64[], ma = [0.4, 0.2, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 324: co2_diff, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -334.79757552601484; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3, -0.2, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 325: co2_diff, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -362.25322750403296; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.3, 0.2, 0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 326: co2_diff, order=(0,0,3), seasonal=(0,0,0,0)
        @test isapprox(loglik, -312.2050176573385; atol=1e-4)
    end
    let ar = [0.3], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 327: co2_diff, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -338.70448684384627; atol=1e-4)
    end
    let ar = [0.5], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 328: co2_diff, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -424.2804071600648; atol=1e-4)
    end
    let ar = [0.7], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 329: co2_diff, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -542.9813865928968; atol=1e-4)
    end
    let ar = [-0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 330: co2_diff, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -398.1421427811244; atol=1e-4)
    end
    let ar = [-0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 331: co2_diff, order=(1,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -532.1636280984253; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 332: co2_diff, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -366.9386387430858; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.4], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 333: co2_diff, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -365.3770716662392; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [0.6], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 334: co2_diff, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -550.0651427125235; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [-0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 335: co2_diff, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -389.056500571897; atol=1e-4)
    end
    let ar = [0.2, 0.5], ma = [-0.5], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 336: co2_diff, order=(2,0,1), seasonal=(0,0,0,0)
        @test isapprox(loglik, -311.3969332867507; atol=1e-4)
    end
    let ar = [0.3], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 337: co2_diff, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -373.6001527818687; atol=1e-4)
    end
    let ar = [0.5], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 338: co2_diff, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -446.17495803648126; atol=1e-4)
    end
    let ar = [0.7], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 339: co2_diff, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -312.3076969539361; atol=1e-4)
    end
    let ar = [-0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 340: co2_diff, order=(1,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -320.2865970123989; atol=1e-4)
    end
    let ar = [0.5, 0.1], ma = [0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 341: co2_diff, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -410.55424469768104; atol=1e-4)
    end
    let ar = [0.3, 0.3], ma = [0.3, -0.3], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 342: co2_diff, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -361.5323130345972; atol=1e-4)
    end
    let ar = [0.6, -0.2], ma = [-0.4, 0.2], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 343: co2_diff, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -302.9907057509932; atol=1e-4)
    end
    let ar = [-0.4, 0.3], ma = [0.5, -0.1], sar = Float64[], sma = Float64[], s = 0
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_co2_diff)
        @test converged  # case 344: co2_diff, order=(2,0,2), seasonal=(0,0,0,0)
        @test isapprox(loglik, -302.34478889815136; atol=1e-4)
    end
    let ar = [0.4], ma = Float64[], sar = [0.5], sma = Float64[], s = 4
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 345: macro_gdp_growth, order=(1,0,0), seasonal=(1,0,0,4)
        @test isapprox(loglik, -271.69431929856864; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = [-0.4], sma = Float64[], s = 4
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 346: macro_gdp_growth, order=(1,0,0), seasonal=(1,0,0,4)
        @test isapprox(loglik, -403.37452964753106; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3], sar = Float64[], sma = [0.4], s = 4
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 347: macro_gdp_growth, order=(0,0,1), seasonal=(0,0,1,4)
        @test isapprox(loglik, -283.09219952353504; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.2], sar = Float64[], sma = [0.3], s = 4
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 348: macro_gdp_growth, order=(0,0,1), seasonal=(0,0,1,4)
        @test isapprox(loglik, -327.3203765134928; atol=1e-4)
    end
    let ar = [0.4], ma = Float64[], sar = Float64[], sma = [0.4], s = 4
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 349: macro_gdp_growth, order=(1,0,0), seasonal=(0,0,1,4)
        @test isapprox(loglik, -274.18630556046753; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = Float64[], sma = [0.3], s = 4
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 350: macro_gdp_growth, order=(1,0,0), seasonal=(0,0,1,4)
        @test isapprox(loglik, -334.7015437910272; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3], sar = [0.5], sma = Float64[], s = 4
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 351: macro_gdp_growth, order=(0,0,1), seasonal=(1,0,0,4)
        @test isapprox(loglik, -276.74858243290277; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.2], sar = [-0.4], sma = Float64[], s = 4
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 352: macro_gdp_growth, order=(0,0,1), seasonal=(1,0,0,4)
        @test isapprox(loglik, -396.2001549854213; atol=1e-4)
    end
    let ar = [0.4], ma = [0.3], sar = [0.5], sma = [0.4], s = 4
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 353: macro_gdp_growth, order=(1,0,1), seasonal=(1,0,1,4)
        @test isapprox(loglik, -316.9893409518713; atol=1e-4)
    end
    let ar = [-0.3], ma = [-0.2], sar = [-0.4], sma = [0.3], s = 4
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_macro_gdp_growth)
        @test converged  # case 354: macro_gdp_growth, order=(1,0,1), seasonal=(1,0,1,4)
        @test isapprox(loglik, -402.0522136226242; atol=1e-4)
    end
    let ar = [0.4], ma = Float64[], sar = [0.5], sma = Float64[], s = 12
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 357: elnino_monthly, order=(1,0,0), seasonal=(1,0,0,12)
        @test isapprox(loglik, -1026.2994082488967; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = [-0.4], sma = Float64[], s = 12
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 358: elnino_monthly, order=(1,0,0), seasonal=(1,0,0,12)
        @test isapprox(loglik, -1549.1590160775177; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3], sar = Float64[], sma = [0.4], s = 12
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 359: elnino_monthly, order=(0,0,1), seasonal=(0,0,1,12)
        @test isapprox(loglik, -1197.4373084641268; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.2], sar = Float64[], sma = [0.3], s = 12
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 360: elnino_monthly, order=(0,0,1), seasonal=(0,0,1,12)
        @test isapprox(loglik, -1363.4855705962714; atol=1e-4)
    end
    let ar = [0.4], ma = Float64[], sar = Float64[], sma = [0.4], s = 12
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 361: elnino_monthly, order=(1,0,0), seasonal=(0,0,1,12)
        @test isapprox(loglik, -1123.7172121483577; atol=1e-4)
    end
    let ar = [-0.3], ma = Float64[], sar = Float64[], sma = [0.3], s = 12
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 362: elnino_monthly, order=(1,0,0), seasonal=(0,0,1,12)
        @test isapprox(loglik, -1375.3327265133103; atol=1e-4)
    end
    let ar = Float64[], ma = [0.3], sar = [0.5], sma = Float64[], s = 12
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 363: elnino_monthly, order=(0,0,1), seasonal=(1,0,0,12)
        @test isapprox(loglik, -1099.9641853894695; atol=1e-4)
    end
    let ar = Float64[], ma = [-0.2], sar = [-0.4], sma = Float64[], s = 12
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 364: elnino_monthly, order=(0,0,1), seasonal=(1,0,0,12)
        @test isapprox(loglik, -1537.3128803547654; atol=1e-4)
    end
    let ar = [0.4], ma = [0.3], sar = [0.5], sma = [0.4], s = 12
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 365: elnino_monthly, order=(1,0,1), seasonal=(1,0,1,12)
        @test isapprox(loglik, -856.3102029218692; atol=1e-4)
    end
    let ar = [-0.3], ma = [-0.2], sar = [-0.4], sma = [0.3], s = 12
        ar_c, ma_c = TSAnalytics.combined_ar_ma(ar, sar, ma, sma, max(s, 1))
        ssm = TSAnalytics.build_statespace(ar_c, ma_c)
        loglik, sigma2, v, F, converged = TSAnalytics.kalman_filter(ssm, y_elnino_monthly)
        @test converged  # case 366: elnino_monthly, order=(1,0,1), seasonal=(1,0,1,12)
        @test isapprox(loglik, -1539.0003024461748; atol=1e-4)
    end
end
end # if TSANALYTICS_FULL_TESTS
