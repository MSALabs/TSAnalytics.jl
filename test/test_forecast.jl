using DelimitedFiles, StatsAPI

@testset "predict/forecast matched cases (exact R+Python validation)" begin
    # handoff/stage-5.2-forecast-handoff.md, section 7. Verified against
    # BOTH real statsmodels.tsa.ar_model.AutoReg.get_prediction() AND
    # real R (predict.ar() and forecast.ar(), via the `forecast` package)
    # in this session -- R was in fact installed on this machine (not on
    # PATH, but reachable via its full Rscript.exe path), contrary to an
    # earlier assumption; both references agree exactly on point
    # forecasts, standard errors, and confidence bounds.
    y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1, header=false))
    m = arx(y, 2; trend=:c)

    # Case A: default level=[80,95]
    fA = StatsAPI.predict(m, 5)
    @test isapprox(fA.point,
                    [1.4723224165841236, 1.2990086833593646, 1.1136152325414617,
                     0.9690548655286699, 0.8434514630857797]; atol=1e-8)
    @test isapprox(fA.se,
                    [0.9941579732015288, 1.129890847944828, 1.2544018545913682,
                     1.326680844514891, 1.3781074490217473]; atol=1e-8)
    @test fA.levels == [80.0, 95.0]
    @test size(fA.lower) == (5, 2)
    @test size(fA.upper) == (5, 2)
    # exact 95% CI bounds, cross-checked against both AutoReg.conf_int(0.05)
    # and R's forecast()'s "Lo 95"/"Hi 95" columns
    @test isapprox(fA.lower[:, 2], [-0.47619141, -0.91553669, -1.34496722, -1.63119181, -1.8575895]; atol=1e-6)
    @test isapprox(fA.upper[:, 2], [3.42083624, 3.51355405, 3.57219769, 3.56930154, 3.54449243]; atol=1e-6)
    # exact 80% CI bounds, cross-checked against R's forecast()'s "Lo 80"/"Hi 80"
    @test isapprox(fA.lower[:, 1], [0.1982577, -0.1490047, -0.4939654, -0.7311550, -0.9226643]; atol=1e-6)
    @test isapprox(fA.upper[:, 1], [2.746387, 2.747022, 2.721196, 2.669265, 2.609567]; atol=1e-6)

    # Case B: single level
    fB = StatsAPI.predict(m, 5; level=[95.0])
    @test size(fB.lower) == (5, 1)
    @test size(fB.upper) == (5, 1)
    @test isapprox(vec(fB.lower), fA.lower[:, 2]; atol=1e-10)

    # Case C: horizon=1 -- se must equal sqrt(sigma2) exactly (psi=[1.0] only)
    fC = StatsAPI.predict(m, 1)
    @test isapprox(fC.se[1], sqrt(m.sigma2); atol=1e-10)

    # Case D: 3 levels, must nest at every horizon
    fD = StatsAPI.predict(m, 5; level=[50.0, 80.0, 95.0])
    @test size(fD.lower) == (5, 3)
    for h in 1:5
        @test fD.lower[h, 1] > fD.lower[h, 2] > fD.lower[h, 3]
        @test fD.upper[h, 1] < fD.upper[h, 2] < fD.upper[h, 3]
    end

    # se compounds (non-decreasing) with horizon
    @test issorted(fA.se)

    # forecast() alias matches predict() exactly
    @test forecast(m, 5).point == StatsAPI.predict(m, 5).point
    @test forecast(m, 5; level=[50.0]).lower == StatsAPI.predict(m, 5; level=[50.0]).lower

    # show: matches R's confirmed print.forecast() column shape
    io = IOBuffer()
    show(io, fA)
    s = String(take!(io))
    @test occursin("AR(2)", s)
    @test occursin("Point Forecast", s)
    @test occursin("Lo 80", s) && occursin("Hi 80", s)
    @test occursin("Lo 95", s) && occursin("Hi 95", s)
end

@testset "predict/forecast trend/seasonal/lag-subset (exact statsmodels validation)" begin
    # Independently generated (not transcribed from the handoff, which only
    # covered the default trend=:c case) -- exercises the general recursion
    # against trend continuation, an arbitrary lag subset, and seasonal
    # dummy continuation, each a genuinely different code path in
    # _arx_forecast_row.
    y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1, header=false))

    m_ct = arx(y, 2; trend=:ct)
    f_ct = forecast(m_ct, 5)
    @test isapprox(f_ct.point,
                    [1.4228227794713457, 1.2222975845836224, 1.0100559098102462,
                     0.8440140618155086, 0.6998857515040073]; atol=1e-8)
    @test isapprox(f_ct.se,
                    [0.9938800412938361, 1.12889283807402, 1.2516003425039628,
                     1.3224068397092725, 1.3724008437168345]; atol=1e-8)

    m_sub = arx(y, [1, 3]; trend=:c)
    f_sub = forecast(m_sub, 5)
    @test isapprox(f_sub.point,
                    [1.4660541455774698, 1.146131331493967, 0.9351397165302536,
                     0.7658329526245548, 0.6279754293707813]; atol=1e-8)
    @test isapprox(f_sub.se,
                    [1.025523024900917, 1.2385756067965228, 1.3248828361947766,
                     1.3819155328393589, 1.4176851121975311]; atol=1e-8)

    m_seas = arx(y, 1; trend=:c, seasonal=true, period=4)
    f_seas = forecast(m_seas, 5)
    @test isapprox(f_seas.point,
                    [1.2954536285625842, 0.9621329545537095, 0.6417368146425771,
                     0.6528146411134899, 0.44406817524569514]; atol=1e-8)
    @test isapprox(f_seas.se,
                    [1.0183460630642374, 1.2566356678904838, 1.3647406888823432,
                     1.4179758421150248, 1.4450240427308862]; atol=1e-8)
end

@testset "predict/forecast parameter coverage and error paths" begin
    y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1, header=false))
    x1 = vec(readdlm(TSAnalytics.ARX_EXOG_X1, ','; skipstart=1, header=false))
    y2 = vec(readdlm(TSAnalytics.ARX_EXOG_Y2, ','; skipstart=1, header=false))

    # every trend value
    for trend in (:n, :c, :t, :ct)
        m = arx(y, 2; trend=trend)
        f = forecast(m, 5)
        @test all(isfinite, f.point)
        @test all(isfinite, f.se)
        @test issorted(f.se)
    end

    # every horizon
    for h in (1, 2, 5, 10, 20)
        m = arx(y, 2; trend=:c)
        f = forecast(m, h)
        @test f.horizon == h
        @test length(f.point) == h
        @test size(f.lower) == (h, 2)
    end

    # every level-vector size
    for levels in ([50.0], [80.0, 95.0], [10.0, 50.0, 90.0, 99.0])
        m = arx(y, 2; trend=:c)
        f = forecast(m, 5; level=levels)
        @test size(f.lower, 2) == length(levels)
        @test f.levels == levels
    end

    # lags as Integer vs equivalent explicit Vector give identical forecasts
    m_int = arx(y, 3; trend=:c)
    m_vec = arx(y, [1, 2, 3]; trend=:c)
    @test forecast(m_int, 5).point == forecast(m_vec, 5).point

    # seasonal, every period value
    for period in (3, 4, 6)
        m = arx(y, 1; trend=:c, seasonal=true, period=period)
        f = forecast(m, 8)  # long enough horizon to wrap around the period at least once
        @test all(isfinite, f.point)
    end

    # container-agnostic model fit (Float32 y) still forecasts finitely
    m_f32 = arx(Float32.(y), 2; trend=:c)
    f_f32 = forecast(m_f32, 5)
    @test all(isfinite, f_f32.point)

    # error paths
    m = arx(y, 2; trend=:c)
    @test_throws ArgumentError forecast(m, 0)
    @test_throws ArgumentError forecast(m, -1)
    @test_throws ArgumentError forecast(m, 5; level=Float64[])
    @test_throws ArgumentError forecast(m, 5; level=[0.0])
    @test_throws ArgumentError forecast(m, 5; level=[100.0])
    @test_throws ArgumentError forecast(m, 5; level=[-5.0])
    @test_throws ArgumentError forecast(m, 5; level=[50.0, 150.0])  # one bad level among good ones

    # exog models: forecasting is explicitly not supported, must throw clearly
    m_exog = arx(y2, 2; trend=:c, exog=x1)
    @test_throws ArgumentError forecast(m_exog, 5)
    m_exog_seas = arx(y2, 1; trend=:c, seasonal=true, period=4, exog=x1)
    @test_throws ArgumentError forecast(m_exog_seas, 3)
end

@testset "Forecast struct" begin
    y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1, header=false))
    m = arx(y, 2; trend=:c)
    f = forecast(m, 5)
    @test f isa TSAnalytics.Forecast
    @test f.point isa Vector{Float64}
    @test f.se isa Vector{Float64}
    @test f.levels isa Vector{Float64}
    @test f.lower isa Matrix{Float64}
    @test f.upper isa Matrix{Float64}
    @test f.horizon == 5
    @test f.model_name == "AR(2)"
end

@testset "benchmark point forecasts (fpp3 5.2, exact identities)" begin
    y = [10.0, 12, 14, 16, 18]

    @test naive(y, 3).point == [18.0, 18.0, 18.0]
    @test mean_forecast(y, 2).point == [14.0, 14.0]

    # drift: slope = (18-10)/4 = 2 per step
    @test isapprox(drift(y, 3).point, [20.0, 22.0, 24.0]; atol=1e-10)

    ys = [1.0, 2, 3, 4, 5, 6, 7, 8]  # m = 4
    @test seasonal_naive(ys, 4, 4).point == [5.0, 6.0, 7.0, 8.0]
    @test seasonal_naive(ys, 6, 4).point == [5.0, 6.0, 7.0, 8.0, 5.0, 6.0]

    @test naive(y, 3).model_name == "Naive"
    @test mean_forecast(y, 3).model_name == "Mean"
    @test drift(y, 3).model_name == "Drift"
    @test seasonal_naive(ys, 3, 4).model_name == "Seasonal naive"
end

@testset "benchmark interval shapes (fpp3 Table 5.2)" begin
    # naive: se grows as sqrt(h)
    f = naive(randn(100), 9)
    @test isapprox(f.se[4] / f.se[1], 2.0; atol=1e-10)  # sqrt(4)/sqrt(1)
    @test isapprox(f.se[9] / f.se[1], 3.0; atol=1e-10)  # sqrt(9)/sqrt(1)

    # seasonal naive: se is a step function in h, not smooth
    fs = seasonal_naive(randn(100), 8, 4)
    @test fs.se[1] == fs.se[2] == fs.se[3] == fs.se[4]  # k = 0 throughout
    @test fs.se[5] == fs.se[8]                          # k = 1 throughout
    @test fs.se[5] > fs.se[4]                           # jumps between blocks
    @test isapprox(fs.se[5] / fs.se[1], sqrt(2); atol=1e-10)

    # mean: se is constant across the horizon
    fm = mean_forecast(randn(50), 6)
    @test all(==(fm.se[1]), fm.se)

    # drift: se grows faster than naive's sqrt(h) (the h/(T-1) term inside)
    fd = drift(randn(50) .+ (1:50), 6)
    @test issorted(fd.se)
end

@testset "residual sd divisors follow fpp3 eq. 5.1 (K, M per method)" begin
    # naive divides by T-1, drift by T-2 on identical data -- confirm the
    # two genuinely differ, not just that both run
    y = cumsum(randn(30)) .+ 50
    @test naive(y, 1).se[1] != drift(y, 1).se[1]
end

@testset "MASE of the naive/seasonal-naive forecast is exactly 1 (ties to mase)" begin
    y = cumsum(randn(60)) .+ 100
    train, test = y[1:50], y[51:60]

    f = naive(train, 10)
    @test mase(test, f.point, train; sp=1) != 1.0  # a real forecast, not trivially 1

    # in-sample one-step-ahead naive against the series itself IS the
    # benchmark mase's own denominator is built from -- must tie exactly
    onestep = train[1:(end - 1)]
    @test isapprox(mase(train[2:end], onestep, train; sp=1), 1.0; atol=1e-12)

    m = 4
    onestep_seasonal = train[1:(end - m)]
    @test isapprox(mase(train[(m + 1):end], onestep_seasonal, train; sp=m), 1.0; atol=1e-12)
end

@testset "benchmark methods slot into tscv directly (Forecast-returning callback)" begin
    y = cumsum(randn(80)) .+ 50
    for f in (naive, drift, mean_forecast)
        errs = tscv(y, f; h=1, initial=20)
        @test length(errs) > 0
        @test all(isfinite, errs)
    end
    errs_seasonal = tscv(y, (train, h) -> seasonal_naive(train, h, 4); h=1, initial=20)
    @test length(errs_seasonal) > 0
end

@testset "benchmark methods: error paths and level coverage" begin
    y = [10.0, 12, 14, 16, 18]

    @test_throws ArgumentError naive(y, 0)
    @test_throws ArgumentError mean_forecast(y, -1)
    @test_throws ArgumentError drift(y, 0)
    @test_throws ArgumentError seasonal_naive(y, 0, 2)
    @test_throws ArgumentError seasonal_naive(y, 3, 0)

    @test_throws ArgumentError naive([1.0], 3)               # needs >= 2 obs
    @test_throws ArgumentError drift([1.0, 2.0], 3)           # needs >= 3 obs
    @test_throws ArgumentError seasonal_naive([1.0, 2.0], 3, 4)  # needs > m obs

    @test_throws ArgumentError naive(y, 3; level=Float64[])
    @test_throws ArgumentError naive(y, 3; level=[0.0])
    @test_throws ArgumentError naive(y, 3; level=[100.0])

    for levels in ([50.0], [80.0, 95.0], [10.0, 50.0, 90.0, 99.0])
        f = naive(y, 5; level=levels)
        @test size(f.lower, 2) == length(levels)
        @test f.levels == levels
        for h in 1:5, j in eachindex(levels)
            @test f.lower[h, j] <= f.point[h] <= f.upper[h, j]
        end
    end
end
