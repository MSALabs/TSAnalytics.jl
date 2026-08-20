using DelimitedFiles, Random, Statistics, StatsAPI

@testset "realized volatility" begin
    r_check = vec(readdlm(joinpath(@__DIR__, "verification", "realizedvol", "rv_check.csv"), ','; skipstart=1))

    @testset "exact real highfrequency validation (n=78)" begin
        @test isapprox(realized_variance(r_check), 7.713353e-05; atol=1e-10)
        @test isapprox(bipower_variation(r_check), 7.724151e-05; atol=1e-10)

        sv = realized_semivariance(r_check)
        @test isapprox(sv.negative, 3.613574e-05; atol=1e-10)
        @test isapprox(sv.positive, 4.099779e-05; atol=1e-10)
        @test isapprox(sv.positive + sv.negative, realized_variance(r_check); atol=1e-15)

        jt = jump_test(r_check)
        @test jt isa JumpTest
        @test isapprox(jt.statistic, -0.01584209; atol=1e-6)
        @test isapprox(jt.pvalue, 0.9873604; atol=1e-6)
        @test jt.rv == realized_variance(r_check)
        @test jt.bv == bipower_variation(r_check)
        @test jt.jump_variance == max(jt.rv - jt.bv, 0.0)
    end

    @testset "small hand-checkable example" begin
        r = [0.01, -0.02, 0.005]
        @test realized_variance(r) == 0.01^2 + 0.02^2 + 0.005^2
        @test isapprox(bipower_variation(r), (pi / 2) * (0.01 * 0.02 + 0.02 * 0.005); atol=1e-15)
        sv = realized_semivariance(r)
        @test sv.positive == 0.01^2 + 0.005^2
        @test sv.negative == 0.02^2
    end

    @testset "realized volatility -- statistical calibration (per handoff section 5, 1000 cases)" begin
        Random.seed!(42)
        n_intraday = 78
        sigma_daily = 0.15 / sqrt(252)
        n_days = 500

        ratios_nojump = Float64[]
        rejections_nojump = 0
        for _ in 1:n_days
            r = randn(n_intraday) .* (sigma_daily / sqrt(n_intraday))
            rv = realized_variance(r)
            bv = bipower_variation(r)
            push!(ratios_nojump, rv / bv)
            jt = jump_test(r)
            abs(jt.statistic) > 1.96 && (rejections_nojump += 1)
        end
        @test isapprox(mean(ratios_nojump), 1.0; atol=0.05)
        @test isapprox(rejections_nojump / n_days, 0.05; atol=0.03)

        ratios_jump = Float64[]
        rejections_jump = 0
        for _ in 1:n_days
            r = randn(n_intraday) .* (sigma_daily / sqrt(n_intraday))
            r[rand(1:n_intraday)] += rand([-1, 1]) * 0.02
            rv = realized_variance(r)
            bv = bipower_variation(r)
            push!(ratios_jump, rv / bv)
            jt = jump_test(r)
            abs(jt.statistic) > 1.96 && (rejections_jump += 1)
        end
        @test mean(ratios_jump) > 2.0
        @test rejections_jump / n_days > 0.9

        for _ in 1:20
            r = randn(n_intraday) .* (sigma_daily / sqrt(n_intraday))
            @test jump_test(r).jump_variance >= 0
        end
    end

    @testset "realized_measures: parallel matches serial" begin
        Random.seed!(7)
        periods = [randn(78) .* 0.01 for _ in 1:20]
        par = realized_measures(periods; parallel=true)
        serial = realized_measures(periods; parallel=false)
        @test length(par) == length(serial) == 20
        for i in eachindex(periods)
            @test isapprox(par[i].rv, serial[i].rv; atol=1e-12)
            @test isapprox(par[i].bv, serial[i].bv; atol=1e-12)
            @test isapprox(par[i].jump.statistic, serial[i].jump.statistic; atol=1e-12)
        end
    end

    @testset "StatsAPI-style HypothesisTest accessors" begin
        jt = jump_test(r_check)
        @test statistic(jt) == jt.statistic
        @test pvalue(jt) == jt.pvalue
    end

    @testset "container-agnostic" begin
        @test realized_variance(1:5) == realized_variance(collect(1.0:5))
        @test bipower_variation(1:5) == bipower_variation(collect(1.0:5))
    end

    @testset "error paths" begin
        @test_throws ArgumentError bipower_variation([1.0])
        @test_throws ArgumentError jump_test([1.0, 2.0])
        @test_throws ArgumentError TSAnalytics._tripower_quarticity([1.0, 2.0])
    end

    @testset "show method runs without erroring" begin
        io = IOBuffer()
        show(io, jump_test(r_check))
        s = String(take!(io))
        @test occursin("jump test", s)
        @test occursin("z statistic", s)
    end
end
