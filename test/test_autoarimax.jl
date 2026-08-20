using TSAnalytics, DelimitedFiles, StatsAPI

@testset "auto_arimax" begin
    d = readdlm(joinpath(@__DIR__, "verification", "autoarimax", "auto_arimax_shared.csv"), ','; skipstart=1)
    y, x = Float64.(d[:, 1]), reshape(Float64.(d[:, 2]), :, 1)

    @testset "cointegration-style case -- dual-verified against real pmdarima" begin
        m = auto_arimax(y, x; seasonal=false, stepwise=true)
        @test m isa ArimaxModel
        @test m.model == :mle
        @test m.d == 1   # matches real pmdarima's actual selection, NOT the naively-expected d=0
        @test length(m.arma.ar) == 1 && length(m.arma.ma) == 1
        @test isapprox(m.aic, 574.819; atol=0.5)
        @test isapprox(m.beta[1], 1.2254; atol=0.05)
        @test m.converged
    end

    @testset "exhaustive matches stepwise on this series" begin
        m_step = auto_arimax(y, x; seasonal=false, max_p=2, max_q=2, stepwise=true)
        m_exh = auto_arimax(y, x; seasonal=false, max_p=2, max_q=2, stepwise=false)
        @test (length(m_step.arma.ar), m_step.d, length(m_step.arma.ma)) ==
              (length(m_exh.arma.ar), m_exh.d, length(m_exh.arma.ma))
        @test isapprox(m_step.aic, m_exh.aic; atol=1e-6)
    end

    @testset "AICc parameter count includes k_exog" begin
        m = auto_arimax(y, x; seasonal=false, d=1, max_p=1, max_q=1, stepwise=false)
        nparam = TSAnalytics._nparam_model(m)
        @test nparam == length(m.beta) + length(m.arma.ar) + length(m.arma.ma)
        n = TSAnalytics._nobs_model(m)
        k = nparam + 1
        expected_aicc = m.aic + 2 * k * (k + 1) / (n - k - 1)
        @test isapprox(TSAnalytics._ic_value(m, :aicc), expected_aicc; atol=1e-8)
    end

    @testset "model is always :mle -- no model keyword exposed" begin
        m = auto_arimax(y, x; seasonal=false, d=1, max_p=1, max_q=0, stepwise=false)
        @test m.model == :mle
        @test m.beta !== nothing
        @test m.beta_filtered === nothing
        @test m.nobs_diffuse === nothing
    end

    @testset "d bypasses residual-based detection when passed explicitly" begin
        m = auto_arimax(y, x; seasonal=false, d=0, max_p=1, max_q=1, stepwise=false)
        @test m.d == 0
    end

    @testset "seasonal requires D explicit" begin
        @test_throws ArgumentError auto_arimax(y, x; seasonal=true, m=4)
        m = auto_arimax(y, x; seasonal=true, m=4, D=0, max_p=1, max_q=0, max_P=1, max_Q=0, stepwise=false)
        @test m isa SarimaxModel
        @test m.seasonal_order[4] == 4
        @test m.model == :mle
    end

    @testset "information_criterion: :aic, :aicc, :bic all produce a finite result" begin
        for ic in (:aic, :aicc, :bic)
            m = auto_arimax(y, x; information_criterion=ic, d=1, max_p=1, max_q=1, stepwise=false)
            @test isfinite(m.aic)
        end
        @test_throws ArgumentError auto_arimax(y, x; information_criterion=:bogus)
    end

    @testset "method and se_type pass-through" begin
        m_css = auto_arimax(y, x; method=:css_ml, d=1, max_p=1, max_q=0, stepwise=false)
        @test m_css.method == :css_ml
        m_opg = auto_arimax(y, x; se_type=:opg, d=1, max_p=1, max_q=0, stepwise=false)
        @test all(isfinite, m_opg.se)
        @test_throws ArgumentError auto_arimax(y, x; method=:bogus)
        @test_throws ArgumentError auto_arimax(y, x; se_type=:bogus)
    end

    @testset "container-agnostic: exog as a plain Vector (k=1)" begin
        m_vec = auto_arimax(y, vec(x); d=1, max_p=1, max_q=0, stepwise=false)
        m_mat = auto_arimax(y, x; d=1, max_p=1, max_q=0, stepwise=false)
        @test isapprox(m_vec.beta, m_mat.beta; atol=1e-8)
    end

    @testset "error paths: negative/invalid arguments" begin
        @test_throws ArgumentError auto_arimax(y, x; max_p=-1)
        @test_throws ArgumentError auto_arimax(y, x; max_q=-1)
        @test_throws ArgumentError auto_arimax(y, x; max_P=-1)
        @test_throws ArgumentError auto_arimax(y, x; max_Q=-1)
        @test_throws ArgumentError auto_arimax(y, x; max_order=-1)
        @test_throws ArgumentError auto_arimax(y, x; max_d=-1)
        @test_throws ArgumentError auto_arimax(y, x; alpha=0.0)
        @test_throws ArgumentError auto_arimax(y, x; alpha=1.0)
        @test_throws ArgumentError auto_arimax(y, x; d=10, max_d=2)
        @test_throws ArgumentError auto_arimax(y, x[1:100, :])  # row mismatch
        @test_throws ArgumentError auto_arimax(y, x; seasonal=true, m=1, D=0)
        @test_throws ArgumentError auto_arimax(y, x; seasonal=true, m=4, D=-1)
        @test_throws ArgumentError auto_arimax(y, x; seasonal=true, m=4, D=2, max_D=1)
    end

    @testset "trace=true runs without erroring, for both search modes" begin
        mktemp() do _, io
            redirect_stdout(io) do
                auto_arimax(y, x; d=1, max_p=1, max_q=1, stepwise=true, trace=true)
                auto_arimax(y, x; d=1, max_p=1, max_q=1, stepwise=false, trace=true)
            end
            flush(io)
            seekstart(io)
            @test !isempty(read(io, String))
        end
    end
end
