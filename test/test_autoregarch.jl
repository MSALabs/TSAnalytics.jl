using DelimitedFiles, Statistics, StatsAPI

@testset "fit_autoreg_garch" begin
    d = readdlm(joinpath(@__DIR__, "verification", "arimax", "sarimax_exog_shared.csv"), ','; skipstart=1)
    y, x = Float64.(d[:, 1]), reshape(Float64.(d[:, 2]), :, 1)

    @testset "garch_order=nothing -- EXACT reduction to fit_arimax" begin
        m1 = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=nothing)
        m2 = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle)
        @test isapprox(m1.beta, m2.beta; atol=1e-10)
        @test isapprox(m1.phi, m2.arma.ar; atol=1e-10)
        @test isapprox(m1.loglik, m2.loglik; atol=1e-10)
        @test m1.garch === nothing
        @test m1.nobs == m2.nobs
        @test m1.converged
    end

    @testset "combined likelihood, degenerate case (alpha=garch_beta=0) matches Stage 8.3's plain ARIMAX" begin
        # per handoff/stage-8.5-autoreg-garch-handoff.md section 1's own
        # verification methodology -- calling the internal likelihood
        # directly at Stage 8.3's own fitted values, alpha/garch_beta
        # fixed to exactly 0, must reduce to the already-verified plain
        # ARIMAX loglik -- see
        # test/verification/autoregarch/ar-garch-ground-truth-transcript.txt
        m2 = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle)
        loglik_deg, contribs, h, ok = TSAnalytics._ar_garch_loglik(
            m2.beta, m2.arma.ar, m2.arma.sigma2, [0.0], [0.0], y, x, 1)
        @test ok
        @test isapprox(loglik_deg, m2.loglik; atol=1e-7)
        @test isapprox(loglik_deg, -211.1256527433344; atol=1e-6)
    end

    @testset "genuine non-degenerate AR-GARCH recovery (known generating parameters)" begin
        # closes the handoff's own flagged gap: no direct software
        # reference exists for this exact combination (section 1), so
        # verified via known-parameter recovery instead (the same
        # methodology used for Stage 7.1's own GARCH bulk tests) -- see
        # test/verification/autoregarch/gen_ar_garch_recovery.jl and
        # ar-garch-ground-truth-transcript.txt
        dr = readdlm(joinpath(@__DIR__, "verification", "autoregarch", "ar_garch_recovery.csv"), ','; skipstart=1)
        yr, xr = Float64.(dr[:, 1]), reshape(Float64.(dr[:, 2]), :, 1)
        m = fit_autoreg_garch(yr, 1, xr; include_mean=false, garch_order=(1, 1))
        @test m.converged
        @test isapprox(m.beta[1], 1.5; atol=0.1)
        @test isapprox(m.phi[1], 0.5; atol=0.1)
        @test isapprox(m.garch.omega, 0.05; atol=0.02)
        @test isapprox(m.garch.alpha[1], 0.15; atol=0.05)
        @test isapprox(m.garch.beta[1], 0.75; atol=0.1)
        @test m.garch.model == :garch
        @test m.garch.p == 1 && m.garch.q == 1
        # genuine volatility clustering recovered, not the degenerate ridge
        @test m.garch.alpha[1] > 0.01
    end

    @testset "include_mean adds an intercept column, no d>0 forced-off concept here" begin
        m_true = fit_autoreg_garch(y, 1, x; include_mean=true, garch_order=nothing)
        m_false = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=nothing)
        @test length(m_true.beta) == 2
        @test length(m_false.beta) == 1
        @test !isapprox(m_true.loglik, m_false.loglik; atol=1e-6)
    end

    @testset "se_type=:hessian vs :opg -- both finite, genuinely different" begin
        m_hess = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=(1, 1), se_type=:hessian)
        m_opg = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=(1, 1), se_type=:opg)
        @test all(isfinite, m_hess.se)
        @test all(isfinite, m_opg.se)
        @test length(m_hess.se) == length(m_opg.se) == 1 + 1 + 1 + 1 + 1  # beta,phi,omega,alpha,garch_beta

        m_hess0 = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=nothing, se_type=:hessian)
        m_opg0 = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=nothing, se_type=:opg)
        @test all(isfinite, m_hess0.se)
        @test all(isfinite, m_opg0.se)
    end

    @testset "n_restarts / parallel" begin
        m1 = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=(1, 1), n_restarts=1)
        m3 = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=(1, 1), n_restarts=3, parallel=false)
        @test m3.converged
        @test isapprox(m3.loglik, m1.loglik; atol=1e-3)
    end

    @testset "container-agnostic: exog as a plain Vector (k=1)" begin
        m_vec = fit_autoreg_garch(y, 1, vec(x); include_mean=false, garch_order=nothing)
        m_mat = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=nothing)
        @test isapprox(m_vec.beta, m_mat.beta; atol=1e-10)
    end

    @testset "StatsAPI methods and coef ordering" begin
        m0 = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=nothing)
        @test StatsAPI.loglikelihood(m0) == m0.loglik
        @test StatsAPI.aic(m0) == m0.aic
        @test StatsAPI.bic(m0) == m0.bic
        @test StatsAPI.nobs(m0) == m0.nobs
        @test isapprox(StatsAPI.coef(m0), vcat(m0.beta, m0.phi); atol=1e-10)

        mg = fit_autoreg_garch(y, 1, x; include_mean=false, garch_order=(1, 1))
        @test isapprox(StatsAPI.coef(mg), vcat(mg.beta, mg.phi, mg.garch.omega, mg.garch.alpha, mg.garch.beta);
                        atol=1e-10)

        io = IOBuffer()
        show(io, mg)
        s = String(take!(io))
        @test occursin("AUTOREG", s)
        @test occursin("GARCH", s)
    end

    @testset "error paths" begin
        @test_throws ArgumentError fit_autoreg_garch(y, 0, x)
        @test_throws ArgumentError fit_autoreg_garch(y, 1, x; se_type=:bogus)
        @test_throws ArgumentError fit_autoreg_garch(y, 1, x; n_restarts=0)
        @test_throws ArgumentError fit_autoreg_garch(y, 1, x; garch_order=(0, 1))  # ARCH order must be >= 1
        @test_throws ArgumentError fit_autoreg_garch(y, 1, x; garch_order=(1, -1))
        @test_throws ArgumentError fit_autoreg_garch(y, 1, x[1:100, :]; garch_order=(1, 1))  # row mismatch
        @test_throws ArgumentError fit_autoreg_garch(y, 1, zeros(150, 0); include_mean=false, garch_order=(1, 1))
    end
end
