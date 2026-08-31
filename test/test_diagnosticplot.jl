@testset "_diagnostic_nlag (exact astsa::sarima lag-count formula)" begin
    # from handoff/stage-1-2-3-combined-fillgap-handoff.md's own test matrix
    @test TSAnalytics._diagnostic_nlag(period=4, ppq=2, fitdf=0) == 20
    @test TSAnalytics._diagnostic_nlag(period=12, ppq=2, fitdf=0) == 36
    @test TSAnalytics._diagnostic_nlag(period=12, ppq=30, fitdf=0) >= 38
    @test TSAnalytics._diagnostic_nlag(period=12, ppq=30, fitdf=0) == 38

    # S<7 -> 20, capped/padded; S>=7 -> 3*S, capped at 52
    @test TSAnalytics._diagnostic_nlag(period=1, ppq=0) == 20
    @test TSAnalytics._diagnostic_nlag(period=6, ppq=0) == 20
    @test TSAnalytics._diagnostic_nlag(period=7, ppq=0) == 21
    @test TSAnalytics._diagnostic_nlag(period=52, ppq=0) == 52     # cap engages: 3*52=156 -> 52
    @test TSAnalytics._diagnostic_nlag(period=100, ppq=0) == 52    # cap engages

    # fitdf's abs() is added on top of ppq
    @test TSAnalytics._diagnostic_nlag(period=1, ppq=0, fitdf=-5) == 20  # 20 >= 5+8=13, unaffected
    @test TSAnalytics._diagnostic_nlag(period=1, ppq=0, fitdf=20) == 28  # 20 < 20+8=28 -> 28
end

@testset "diagnostic_plot (low-level)" begin
    Random.seed!(1)
    resid = randn(200)
    r = diagnostic_plot(resid)

    @test length(r.ljungbox_pvalues) == r.nlag - r.ppq
    @test r.nlag == TSAnalytics._diagnostic_nlag(period=1, ppq=0, fitdf=0)  # == 20
    @test length(r.std_resid) == 200
    @test length(r.acf_lags) == r.nlag
    @test length(r.acf) == r.nlag
    @test length(r.qq_theoretical) == length(r.qq_sample) == 200
    @test issorted(r.qq_theoretical)
    @test issorted(r.qq_sample)
    @test all(0 .<= r.ljungbox_pvalues .<= 1)

    # standardization: mean ~0, sd ~1
    @test isapprox(sum(r.std_resid) / length(r.std_resid), 0.0; atol=1e-10)
    @test isapprox(sum(abs2, r.std_resid) / (length(r.std_resid) - 1), 1.0; atol=1e-10)

    # explicit lags overrides the default formula
    r_explicit = diagnostic_plot(resid; lags=15)
    @test r_explicit.nlag == 15

    # fitdf/ppq feed into both the default nlag and the Ljung-Box start lag
    r_fitdf = diagnostic_plot(resid; fitdf=2, ppq=3)
    @test r_fitdf.ppq == 5  # ppq + abs(fitdf)
    @test length(r_fitdf.ljungbox_pvalues) == r_fitdf.nlag - 5

    # container-agnostic
    @test diagnostic_plot(resid).std_resid == diagnostic_plot(collect(resid)).std_resid

    # error paths
    @test_throws ArgumentError diagnostic_plot(randn(2))
    @test_throws ArgumentError diagnostic_plot(resid; lags=1, ppq=5)   # lags must exceed ppq+abs(fitdf)
    @test_throws ArgumentError diagnostic_plot(resid; lags=300)         # lags must be < n
    @test_throws ArgumentError diagnostic_plot(ones(50))                # zero residual variance
end

@testset "diagnostic_plot model-aware overloads" begin
    Random.seed!(2)
    y = randn(150)
    resid_probe = randn(MersenneTwister(3), 150)

    m_arma = fit_arma(y, (1, 1))
    r_arma = diagnostic_plot(resid_probe, m_arma)
    @test r_arma.ppq == 2  # p+q = 1+1
    @test r_arma.nlag == TSAnalytics._diagnostic_nlag(period=1, ppq=2, fitdf=0)

    yd = cumsum(randn(160))
    m_arima = fit_arima(yd, (1, 1, 1))
    r_arima = diagnostic_plot(randn(MersenneTwister(4), 158), m_arima)
    @test r_arima.ppq == 2  # matches the underlying ArmaModel's p+q

    ys = 50 .+ 5 .* sin.(2π .* (1:120) ./ 12) .+ randn(MersenneTwister(5), 120)
    m_sarima = fit_sarima(ys, (1, 0, 0), (1, 0, 0, 12))
    r_sarima = diagnostic_plot(randn(MersenneTwister(6), 120), m_sarima)
    @test r_sarima.ppq == 2   # p+q+P+Q = 1+0+1+0
    @test r_sarima.nlag == TSAnalytics._diagnostic_nlag(period=12, ppq=2, fitdf=0)  # period=s=12

    # explicit fitdf override still respected
    r_override = diagnostic_plot(resid_probe, m_arma; fitdf=3)
    @test r_override.ppq == 2 + 3
end
