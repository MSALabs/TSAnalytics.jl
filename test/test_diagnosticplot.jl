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

@testset "diagnostic_plot(::GarchModel) — variance-model panels" begin
    Random.seed!(1)
    n = 600
    e = zeros(n); s2 = ones(n)
    for t in 2:n
        s2[t] = 0.05 + 0.08*e[t-1]^2 + 0.86*s2[t-1]
        e[t] = sqrt(s2[t])*randn()
    end

    m = fit_garch(e, 1, 1)
    d = diagnostic_plot(m)

    @test d isa GarchDiagnosticPlotResult
    @test length(d.std_resid) == n
    @test length(d.sigma) == n
    @test length(d.abs_resid) == n
    @test d.abs_resid == abs.(m.resid)
    @test isapprox(d.std_resid, m.resid ./ sqrt.(m.sigma2); atol=1e-12)
    @test length(d.acf) == length(d.acf_lags) == length(d.acf_sq)
    @test length(d.qq_theoretical) == length(d.qq_sample) == n
    @test issorted(d.qq_sample)          # Q-Q sample quantiles are the sorted residuals
    @test d.model == :garch

    @testset "lags keyword and its bound" begin
        d20 = diagnostic_plot(m; lags=15)
        @test length(d20.acf_lags) == 15
        @test_throws ArgumentError diagnostic_plot(m; lags=n)
    end

    @testset "news impact curve: symmetric for :garch, asymmetric for :gjr" begin
        @test d.news_impact_e !== nothing
        # GARCH is symmetric in the shock: NIC(-x) == NIC(+x)
        i_lo = argmin(abs.(d.news_impact_e .+ 2.0))
        i_hi = argmin(abs.(d.news_impact_e .- 2.0))
        @test isapprox(d.news_impact_sigma2[i_lo], d.news_impact_sigma2[i_hi]; rtol=0.05)

        # GJR must put MORE variance on a negative shock -- the leverage effect,
        # and the same asymmetry sign_bias_test is built to detect
        Random.seed!(4)
        eg = zeros(n); sg = ones(n)
        for t in 2:n
            sg[t] = 0.05 + 0.05*eg[t-1]^2 + 0.85*sg[t-1] + 0.15*(eg[t-1] < 0)*eg[t-1]^2
            eg[t] = sqrt(sg[t])*randn()
        end
        mg = fit_garch(eg, 1, 1; model=:gjr)
        dg = diagnostic_plot(mg)
        j_lo = argmin(abs.(dg.news_impact_e .+ 2.0))
        j_hi = argmin(abs.(dg.news_impact_e .- 2.0))
        @test dg.news_impact_sigma2[j_lo] > dg.news_impact_sigma2[j_hi]
    end

    @testset ":egarch omits the news impact curve rather than drawing a wrong one" begin
        me = fit_garch(e, 1, 1; model=:egarch)
        de = diagnostic_plot(me)
        @test de.model == :egarch
        @test de.news_impact_e === nothing
        @test de.news_impact_sigma2 === nothing
        @test length(de.std_resid) == n     # every other panel still populated
    end
end
