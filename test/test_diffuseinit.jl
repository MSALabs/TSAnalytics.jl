using DelimitedFiles, Random, LinearAlgebra

@testset "kalman_filter_diffuse" begin
    T_ = TSAnalytics

    @testset "exact diffuse matches real statsmodels (local level + time-varying regression)" begin
        d = readdlm(joinpath(@__DIR__, "verification", "diffuseinit", "diffuse_case1.csv"), ','; skipstart=1)
        x, y = Float64.(d[:, 1]), Float64.(d[:, 2])
        Zseq = [reshape([1.0, xt], 1, 2) for xt in x]
        ssm = T_.TimeVaryingSSM{Float64}([reshape([1.0 0.0; 0.0 1.0], 2, 2)], Zseq, [reshape([1.0, 0.0], 2, 1)],
                                          [reshape([0.01], 1, 1)], [reshape([0.09], 1, 1)], 2)
        loglik, v, F, nobs_diffuse, converged = T_.kalman_filter_diffuse(ssm, y; diffuse_idx=[1, 2])
        @test converged
        @test nobs_diffuse == 2
        # loglik is the full, inclusive sum (statsmodels' own genuine `res.llf`,
        # loglikelihood_burn=0) -- NOT np.sum(llf_obs[nobs_diffuse:]), a
        # manually-sliced, non-default figure this test asserted before Stage
        # 8.3's own verification caught the discrepancy (see
        # test/verification/diffuseinit/diffuseinit-inclusive-sum-correction.txt).
        # Both numbers are in diffuse_case1.json: loglik_full vs. loglik_excl_diffuse.
        @test isapprox(loglik, -14.53870388541507; atol=1e-8)
        forecasts = y .- v
        @test isapprox(forecasts[1:5], [0.0, -0.86562006, 0.33425629, 1.43040012, -2.04550969]; atol=1e-6)
        @test isapprox(F[1:5], [0.09, 0.10269377, 0.15853232, 0.13515254, 0.18103155]; atol=1e-6)
        @test length(v) == length(F) == length(y)   # full-length, not truncated
    end

    @testset "diffuse_idx=Int[] reduces exactly to plain kalman_filter(::TimeVaryingSSM,...)" begin
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arma", "arma11_fit.csv"), ','; skipstart=1))
        ssm0 = T_.build_statespace([0.5466], [0.2717])
        tv, a0, P0 = T_.to_time_varying(ssm0, y)

        loglik_plain, v_plain, F_plain, conv_plain = T_.kalman_filter(tv, y, a0, P0)
        loglik_d, v_d, F_d, nobs_d, conv_d = T_.kalman_filter_diffuse(tv, y; diffuse_idx=Int[], a0=a0, P_star0=P0)

        @test conv_d
        @test nobs_d == 0
        @test loglik_d == loglik_plain   # exact, not just approximate
        @test isapprox(v_d, v_plain; atol=1e-10)
        @test isapprox(F_d, F_plain; atol=1e-10)
    end

    @testset "single diffuse state (pure local level, d=1)" begin
        n = 30
        Random.seed!(11)
        level = cumsum(randn(n) .* sqrt(0.02))
        y = level .+ randn(n) .* sqrt(0.1)
        ssm = T_.TimeVaryingSSM{Float64}([reshape([1.0], 1, 1)], [reshape([1.0], 1, 1)], [reshape([1.0], 1, 1)],
                                          [reshape([0.02], 1, 1)], [reshape([0.1], 1, 1)], 1)
        loglik, v, F, nobs_diffuse, converged = T_.kalman_filter_diffuse(ssm, y; diffuse_idx=[1])
        @test converged
        @test nobs_diffuse == 1   # a single diffuse state resolves in exactly one observation here
        @test isfinite(loglik)
        @test length(v) == n
    end

    @testset "diffuse phase length is O(d), not O(n)" begin
        for n in (50, 500, 5000)
            Random.seed!(7)
            true_beta = 1.5
            x = randn(n) .* 2
            level = cumsum(randn(n) .* sqrt(0.01))
            y = level .+ true_beta .* x .+ randn(n) .* sqrt(0.09)
            Zseq = [reshape([1.0, xt], 1, 2) for xt in x]
            ssm = T_.TimeVaryingSSM{Float64}([reshape([1.0 0.0; 0.0 1.0], 2, 2)], Zseq, [reshape([1.0, 0.0], 2, 1)],
                                              [reshape([0.01], 1, 1)], [reshape([0.09], 1, 1)], 2)
            loglik, v, F, nobs_diffuse, converged = T_.kalman_filter_diffuse(ssm, y; diffuse_idx=[1, 2])
            @test converged
            @test nobs_diffuse == 2   # bounded regardless of n, per section 4's finding
        end
    end

    @testset "mixed proper + diffuse state (a0/P_star0 keyword generalization beyond the handoff's own signature)" begin
        # state 1: diffuse local level; state 2: a KNOWN, non-diffuse constant with proper (tiny) prior variance
        n = 20
        Random.seed!(3)
        level = cumsum(randn(n) .* sqrt(0.02))
        known_const = 5.0
        y = level .+ known_const .+ randn(n) .* sqrt(0.05)
        ssm = T_.TimeVaryingSSM{Float64}([reshape([1.0 0.0; 0.0 1.0], 2, 2)], [reshape([1.0, 1.0], 1, 2)],
                                          [reshape([1.0, 0.0], 2, 1)], [reshape([0.02], 1, 1)],
                                          [reshape([0.05], 1, 1)], 2)
        a0 = [0.0, known_const]
        P_star0 = zeros(2, 2)  # state 2 known exactly -- zero proper variance
        loglik, v, F, nobs_diffuse, converged = T_.kalman_filter_diffuse(
            ssm, y; diffuse_idx=[1], a0=a0, P_star0=P_star0)
        @test converged
        @test nobs_diffuse == 1
        @test isfinite(loglik)
    end

    @testset "converged=false on a degenerate system" begin
        y = [1.0, 2.0, 3.0]
        ssm = T_.TimeVaryingSSM{Float64}([reshape([0.5], 1, 1)], [reshape([1.0], 1, 1)], [reshape([1.0], 1, 1)],
                                          [reshape([0.0], 1, 1)], [reshape([-10.0], 1, 1)], 1)
        loglik, v, F, nobs_diffuse, converged = T_.kalman_filter_diffuse(ssm, y; diffuse_idx=Int[])
        @test !converged
        @test loglik == -Inf
        @test isempty(v)
        @test isempty(F)
    end

    @testset "error paths" begin
        ssm = T_.TimeVaryingSSM{Float64}([reshape([1.0], 1, 1)], [reshape([1.0], 1, 1)], [reshape([1.0], 1, 1)],
                                          [reshape([0.02], 1, 1)], [reshape([0.1], 1, 1)], 1)
        y = [1.0, 2.0, 3.0]
        @test_throws ArgumentError T_.kalman_filter_diffuse(ssm, Float64[]; diffuse_idx=[1])
        @test_throws ArgumentError T_.kalman_filter_diffuse(ssm, y; diffuse_idx=[1], a0=[0.0, 0.0])
        @test_throws ArgumentError T_.kalman_filter_diffuse(ssm, y; diffuse_idx=[1], P_star0=zeros(2, 2))
        @test_throws ArgumentError T_.kalman_filter_diffuse(ssm, y; diffuse_idx=[2])   # out of range (r=1)
        @test_throws ArgumentError T_.kalman_filter_diffuse(ssm, y; diffuse_idx=[0])   # out of range
        @test_throws ArgumentError T_.kalman_filter_diffuse(ssm, y; diffuse_idx=[1, 1])  # duplicate
    end
end
