using TSAnalytics, DelimitedFiles, StatsAPI

@testset "auto_arima" begin
    y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "d1_clean.csv"), ','; skipstart=1))
    y2 = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "sarima_shared.csv"), ','; skipstart=1))

    @testset "non-seasonal stepwise (regression, hand-verified via direct julia run)" begin
        m = auto_arima(y; max_p=3, max_q=3, max_d=2)
        @test m isa ArimaModel
        @test m.d == 1
        @test length(m.arma.ar) == 1 && length(m.arma.ma) == 0
        @test isapprox(m.arma.aic, 411.9117264518479; atol=1e-6)
        @test m.arma.converged
    end

    @testset "exhaustive matches stepwise on this series (handoff §8 item 4)" begin
        m_step = auto_arima(y; max_p=3, max_q=3, max_d=2, stepwise=true)
        m_exh = auto_arima(y; max_p=3, max_q=3, max_d=2, stepwise=false)
        @test (length(m_step.arma.ar), m_step.d, length(m_step.arma.ma)) ==
              (length(m_exh.arma.ar), m_exh.d, length(m_exh.arma.ma))
        @test m_step.arma.aic == m_exh.arma.aic
    end

    @testset "exhaustive: parallel=true and parallel=false select identical model" begin
        m_par = auto_arima(y; stepwise=false, parallel=true, max_p=3, max_q=3, max_d=2)
        m_ser = auto_arima(y; stepwise=false, parallel=false, max_p=3, max_q=3, max_d=2)
        @test (length(m_par.arma.ar), m_par.d, length(m_par.arma.ma)) ==
              (length(m_ser.arma.ar), m_ser.d, length(m_ser.arma.ma))
        @test m_par.arma.aic == m_ser.arma.aic
    end

    @testset "seasonal requires D explicit (handoff §3 scope limitation)" begin
        @test_throws ArgumentError auto_arima(y2; seasonal=true, m=12)
        m = auto_arima(y2; seasonal=true, m=12, D=1, max_p=3, max_q=3, max_P=2, max_Q=2)
        @test m isa SarimaModel
        @test m.seasonal_order == (0, 1, 1, 12)
        @test isapprox(m.aic, 247.479141588352; atol=1e-6)
        @test m.converged
    end

    @testset "d/D bounds and grid-size parameters" begin
        # max_p=max_q=0 with d auto-detected -> pure differenced white noise
        m = auto_arima(y; max_p=0, max_q=0, max_d=2)
        @test (length(m.arma.ar), m.d, length(m.arma.ma)) == (0, 1, 0)

        # max_order caps total p+q even when max_p/max_q individually allow more
        m = auto_arima(y; max_p=5, max_q=5, max_order=1, max_d=2)
        @test length(m.arma.ar) + length(m.arma.ma) <= 1

        # explicit d bypasses kpss_test detection entirely
        m = auto_arima(y; d=0, max_p=3, max_q=3)
        @test m.d == 0

        @test_throws ArgumentError auto_arima(y2; seasonal=true, m=12, D=2, max_D=1)
        @test_throws ArgumentError auto_arima(y; d=5, max_d=2)
    end

    @testset "information_criterion: :aic, :aicc, :bic all produce a finite-AIC model" begin
        for ic in (:aic, :aicc, :bic)
            m = auto_arima(y; information_criterion=ic, max_p=3, max_q=3, max_d=2)
            @test m isa ArimaModel
            @test isfinite(m.arma.aic)
        end
        @test_throws ArgumentError auto_arima(y; information_criterion=:bogus)
    end

    @testset "include_mean passed through (not searched), still forced off when d>0" begin
        m_mean = auto_arima(y; d=0, max_p=2, max_q=2, include_mean=true)
        @test m_mean.arma.mean isa Float64
        m_nomean = auto_arima(y; d=0, max_p=2, max_q=2, include_mean=false)
        @test m_nomean.arma.mean === nothing
        # d>0 forces mean off regardless of include_mean, matching fit_arima's own convention
        m_d1 = auto_arima(y; d=1, max_p=2, max_q=2, include_mean=true)
        @test m_d1.arma.mean === nothing
    end

    @testset "method and se_type pass-through" begin
        m_css = auto_arima(y; method=:css_ml, max_p=3, max_q=3, max_d=2)
        @test m_css isa ArimaModel
        @test m_css.arma.method == :css_ml

        m_opg = auto_arima(y; se_type=:opg, max_p=2, max_q=2, max_d=2)
        @test m_opg.arma.se_type == :opg
        @test all(isfinite, m_opg.arma.se)

        @test_throws ArgumentError auto_arima(y; method=:bogus)
        @test_throws ArgumentError auto_arima(y; se_type=:bogus)
    end

    @testset "error paths: negative/invalid arguments" begin
        @test_throws ArgumentError auto_arima(y; max_p=-1)
        @test_throws ArgumentError auto_arima(y; max_q=-1)
        @test_throws ArgumentError auto_arima(y; max_P=-1)
        @test_throws ArgumentError auto_arima(y; max_Q=-1)
        @test_throws ArgumentError auto_arima(y; max_order=-1)
        @test_throws ArgumentError auto_arima(y; max_d=-1)
        @test_throws ArgumentError auto_arima(y2; seasonal=true, m=12, D=-1)
        @test_throws ArgumentError auto_arima(y; alpha=0.0)
        @test_throws ArgumentError auto_arima(y; alpha=1.0)
        @test_throws ArgumentError auto_arima(y2; seasonal=true, m=1, D=0)
        @test_throws ArgumentError auto_arima(y2; seasonal=true, m=0, D=0)
    end

    @testset "trace=true runs without erroring, for both search modes" begin
        mktemp() do _, io
            redirect_stdout(io) do
                auto_arima(y; max_p=2, max_q=2, max_d=2, trace=true)
                auto_arima(y; max_p=2, max_q=2, max_d=2, stepwise=false, trace=true)
            end
            flush(io)
            seekstart(io)
            @test !isempty(read(io, String))
        end
    end
end
