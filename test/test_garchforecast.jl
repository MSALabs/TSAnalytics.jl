using DelimitedFiles, Statistics

@testset "forecast_volatility" begin
    e = vec(readdlm(joinpath(@__DIR__, "verification", "garch", "garch_shared.csv"), ','; skipstart=1))
    eg = vec(readdlm(joinpath(@__DIR__, "verification", "garch", "gjr_shared.csv"), ','; skipstart=1))
    m = fit_garch(e, 1, 1)

    @testset "GARCH analytic (exact real arch.arch_model validation)" begin
        fc = forecast_volatility(m, 10; method=:analytic)
        @test fc isa VolatilityForecast
        @test fc.method == :analytic
        @test fc.variance_paths === nothing
        @test fc.simulations === nothing
        @test fc.horizon == 10
        @test length(fc.variance) == 10
        @test isapprox(fc.variance, [0.7852, 0.7957, 0.8058, 0.8155, 0.8248,
                                      0.8337, 0.8422, 0.8503, 0.8582, 0.8657]; atol=1e-3)
    end

    @testset "GJR-GARCH analytic (exact real arch.arch_model validation)" begin
        m_gjr = fit_garch(eg, 1, 1; model=:gjr)
        fc = forecast_volatility(m_gjr, 5; method=:analytic)
        # handoff's own transcribed last value (1.7777) is a digit-transposition typo,
        # confirmed by direct re-execution of real arch_model: the true value is 1.67775
        @test isapprox(fc.variance, [1.88458, 1.82791, 1.77469, 1.72470, 1.67775]; atol=1e-3)
    end

    @testset "EGARCH: horizon=1 analytic works, horizon>1 throws (matches real arch's ValueError)" begin
        m_ego = fit_garch(eg, 1, 1; model=:egarch)
        fc1 = forecast_volatility(m_ego, 1; method=:analytic)
        @test fc1.method == :analytic
        @test length(fc1.variance) == 1
        @test fc1.variance[1] > 0

        @test_throws ArgumentError forecast_volatility(m_ego, 5; method=:analytic)
        @test_throws ArgumentError forecast_volatility(m_ego, 2; method=:analytic)
    end

    @testset "EGARCH simulation (structural + rough numeric agreement)" begin
        m_ego = fit_garch(eg, 1, 1; model=:egarch)
        fc = forecast_volatility(m_ego, 5; method=:simulation, simulations=20000)
        @test fc.method == :simulation
        @test fc.variance_paths !== nothing
        @test size(fc.variance_paths) == (20000, 5)
        @test all(fc.variance .> 0)
        # rough agreement with real arch's own simulation-based forecast (Monte Carlo
        # noise on both sides -- not expected to be bit-identical, different RNG streams)
        @test isapprox(fc.variance, [1.6389, 1.5938, 1.5484, 1.5047, 1.4693]; atol=0.05)
    end

    @testset "method=:auto picks correctly" begin
        m_garch = fit_garch(e, 1, 1; model=:garch)
        m_gjr = fit_garch(eg, 1, 1; model=:gjr)
        m_ego = fit_garch(eg, 1, 1; model=:egarch)

        @test forecast_volatility(m_garch, 5; method=:auto).method == :analytic
        @test forecast_volatility(m_gjr, 5; method=:auto).method == :analytic
        @test forecast_volatility(m_ego, 5; method=:auto).method == :simulation
        @test forecast_volatility(m_ego, 1; method=:auto).method == :simulation   # even h=1, :egarch always simulates under :auto
    end

    @testset "simulation vs analytic convergence (the correctness check on the simulation engine itself)" begin
        fc_s = forecast_volatility(m, 5; method=:simulation, simulations=50000)
        fc_a = forecast_volatility(m, 5; method=:analytic)
        @test maximum(abs.(fc_s.variance .- fc_a.variance)) < 0.005   # Monte Carlo tolerance
    end

    @testset "parallel simulation matches serial in distribution" begin
        m_ego = fit_garch(eg, 1, 1; model=:egarch)
        fc_par = forecast_volatility(m_ego, 5; simulations=5000, parallel=true)
        fc_serial = forecast_volatility(m_ego, 5; simulations=5000, parallel=false)
        @test isapprox(mean(fc_par.variance), mean(fc_serial.variance); atol=0.1)
        @test all(fc_par.variance .> 0) && all(fc_serial.variance .> 0)
    end

    @testset "general p,q analytic forecast (not just (1,1))" begin
        for (p, q) in [(2, 1), (1, 2), (2, 2)]
            mpq = fit_garch(e, p, q)
            fc = forecast_volatility(mpq, 8; method=:analytic)
            @test length(fc.variance) == 8
            @test all(fc.variance .> 0)
            # simulation should still converge to the analytic forecast for general p,q
            fc_s = forecast_volatility(mpq, 3; method=:simulation, simulations=30000)
            @test maximum(abs.(fc_s.variance .- fc.variance[1:3])) < 0.05
        end
    end

    @testset "error paths" begin
        @test_throws ArgumentError forecast_volatility(m, 5; method=:bogus)
        @test_throws ArgumentError forecast_volatility(m, 0)
        @test_throws ArgumentError forecast_volatility(m, -1)
        @test_throws ArgumentError forecast_volatility(m, 5; simulations=0)
    end
end
