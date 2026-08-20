using DelimitedFiles, Statistics, StatsAPI

@testset "fit_arimax / fit_sarimax" begin
    d = readdlm(joinpath(@__DIR__, "verification", "arimax", "sarimax_exog_shared.csv"), ','; skipstart=1)
    y, x = Float64.(d[:, 1]), reshape(Float64.(d[:, 2]), :, 1)

    @testset "model=:mle -- dual-verified (R + Python, AR(1) + 1 exog, n=150)" begin
        m = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle)
        @test m.converged
        @test m.model == :mle
        @test m.method == :ml
        @test isapprox(m.beta[1], 1.894806442; atol=1e-3)      # R & Python agree
        @test isapprox(m.arma.ar[1], 0.476774167; atol=1e-3)   # R
        @test isapprox(m.arma.ar[1], 0.47677706; atol=1e-3)    # Python
        @test isapprox(m.loglik, -211.1256527; atol=1e-2)
        @test m.nobs_diffuse === nothing
        @test m.beta_filtered === nothing
        @test StatsAPI.loglikelihood(m) == m.loglik
        @test StatsAPI.nobs(m) == length(y) == 150
        @test isapprox(StatsAPI.coef(m), vcat(m.beta, m.arma.ar); atol=1e-10)

        io = IOBuffer()
        show(io, m)
        s = String(take!(io))
        @test occursin("ARIMAX", s)
        @test occursin("model=mle", s)
    end

    @testset "model=:tvss -- freely-estimated Q_beta, dual case" begin
        m = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss)
        @test m.converged
        @test m.model == :tvss
        @test isapprox(m.arma.ar[1], 0.475909212; atol=1e-3)
        @test isapprox(m.Q_beta[1], 0.000109371580; atol=1e-5)
        @test isapprox(m.arma.sigma2, 0.979298992; atol=1e-2)
        @test isapprox(m.loglik, -213.74793442697535; atol=1e-2)
        @test isapprox(m.beta_filtered[1, 1:5], [2.0, 1.778, 1.458, 1.925, 2.133]; atol=1e-2)
        @test isapprox(m.beta_filtered[1, end-4:end], [1.871, 1.875, 1.864, 1.869, 1.875]; atol=1e-2)
        @test m.beta === nothing

        # beta genuinely varies -- this is the actual point of model=:tvss
        @test std(m.beta_filtered[1, :]) > 0.01

        io = IOBuffer()
        show(io, m)
        s = String(take!(io))
        @test occursin("model=tvss", s)
        @test occursin("Q_beta", s)
    end

    @testset "model=:tvss(Q_beta=0) -- point estimates converge, likelihood does NOT" begin
        m_mle = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle)
        m_tvss0 = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss, Q_beta=[0.0])

        @test m_tvss0.converged
        @test m_tvss0.Q_beta == [0.0]
        @test isapprox(m_tvss0.arma.ar[1], 0.47465807; atol=1e-3)
        # POINT ESTIMATE convergence -- final filtered beta close to :mle's fixed value
        @test isapprox(m_tvss0.beta_filtered[1, end], m_mle.beta[1]; atol=0.02)

        # LIKELIHOOD must NOT match -- this is expected and correct, not a bug:
        # model=:tvss's likelihood is a genuinely different mathematical object
        # for the early (diffuse-phase) observations, even after Q_beta=0
        # collapses beta back to an effectively fixed coefficient -- see
        # fit_arimax's own docstring and
        # handoff/stage-8.3-arimax-sarimax-handoff-v3.md section 1. Do NOT
        # "fix" this into an equality assertion without re-reading that section.
        @test !isapprox(m_tvss0.loglik, m_mle.loglik; atol=0.5)
        @test isapprox(m_tvss0.loglik, -213.76718527012807; atol=1e-2)
    end

    @testset "model=:tvss -- nobs_diffuse populated, :mle does not have it" begin
        m_mle = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle)
        m_tvss = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss)
        @test m_mle.nobs_diffuse === nothing
        @test m_tvss.nobs_diffuse !== nothing && m_tvss.nobs_diffuse >= 1
        # O(d)-bounded per Stage 8.2 -- a single diffuse beta state resolves
        # in a handful of observations, not scaling with n=150
        @test m_tvss.nobs_diffuse <= 5
    end

    @testset "fit_sarimax extends fit_arimax exactly as 6.7 extends 6.6" begin
        m_arimax = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle)
        m_sarimax = fit_sarimax(y, (1, 0, 0), (0, 0, 0, 4), x; include_mean=false, model=:mle)
        @test isapprox(m_sarimax.beta, m_arimax.beta; atol=1e-8)
        @test isapprox(m_sarimax.loglik, m_arimax.loglik; atol=1e-8)
        @test isapprox(m_sarimax.arma.phi, m_arimax.arma.ar; atol=1e-8)

        m_sarimax_tvss = fit_sarimax(y, (1, 0, 0), (0, 0, 0, 4), x; include_mean=false, model=:tvss, Q_beta=[0.0])
        m_arimax_tvss = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss, Q_beta=[0.0])
        @test isapprox(m_sarimax_tvss.loglik, m_arimax_tvss.loglik; atol=1e-6)
        @test isapprox(m_sarimax_tvss.beta_filtered, m_arimax_tvss.beta_filtered; atol=1e-6)

        io = IOBuffer()
        show(io, m_sarimax)
        s = String(take!(io))
        @test occursin("SARIMAX", s)
    end

    @testset "model and method are independent, both apply correctly" begin
        # model=:tvss + method=:css_ml -- a real, sensible combination that
        # would have been ambiguous under a single shared `method` naming
        m = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss, method=:css_ml)
        @test m.model == :tvss
        @test m.method == :css_ml
        @test m.converged

        m2 = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle, method=:css_ml)
        @test m2.model == :mle
        @test m2.method == :css_ml
        @test m2.converged
        @test isapprox(m2.beta[1], 1.894806442; atol=1e-2)  # css_ml converges to the same optimum
    end

    @testset "se_type=:hessian vs :opg -- genuinely different, both computed" begin
        m_hess = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle, se_type=:hessian)
        m_opg = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle, se_type=:opg)
        @test all(isfinite, m_hess.se)
        @test all(isfinite, m_opg.se)
        @test !isapprox(m_hess.se, m_opg.se; atol=1e-4)

        m_hess_tvss = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss, se_type=:hessian)
        m_opg_tvss = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss, se_type=:opg)
        @test all(isfinite, m_hess_tvss.se)
        @test all(isfinite, m_opg_tvss.se)
    end

    @testset "include_mean silently ignored when d>0, matches fit_arima's own convention" begin
        m_d0_true = fit_arimax(y, (1, 0, 0), x; include_mean=true, model=:mle)
        m_d0_false = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle)
        @test length(m_d0_true.beta) == 2   # exog column + intercept
        @test length(m_d0_false.beta) == 1
        @test !isapprox(m_d0_true.loglik, m_d0_false.loglik; atol=1e-6)  # genuinely different model at d=0

        m_d1_true = fit_arimax(y, (1, 1, 0), x; include_mean=true, model=:mle)
        m_d1_false = fit_arimax(y, (1, 1, 0), x; include_mean=false, model=:mle)
        @test length(m_d1_true.beta) == length(m_d1_false.beta) == 1  # constant column silently dropped
        @test isapprox(m_d1_true.beta, m_d1_false.beta; atol=1e-8)    # bit-identical fits
        @test isapprox(m_d1_true.loglik, m_d1_false.loglik; atol=1e-8)
        @test StatsAPI.nobs(m_d1_true) == length(y) - 1
    end

    @testset "n_restarts / parallel for model=:tvss" begin
        m1 = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss, n_restarts=1)
        m3 = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss, n_restarts=3, parallel=false)
        @test m3.converged
        @test isapprox(m3.loglik, m1.loglik; atol=1e-4)  # same optimum found from multiple starts
    end

    @testset "container-agnostic: exog as a plain Vector (k=1), y as a generic iterable" begin
        m_vec = fit_arimax(y, (1, 0, 0), vec(x); include_mean=false, model=:mle)
        m_mat = fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:mle)
        @test isapprox(m_vec.beta, m_mat.beta; atol=1e-10)

        m_tuple_y = fit_arimax(Tuple(y), (1, 0, 0), x; include_mean=false, model=:mle)
        @test isapprox(m_tuple_y.beta, m_mat.beta; atol=1e-10)
    end

    @testset "error paths" begin
        @test_throws ArgumentError fit_arimax(y, (1, 0, 0), x; model=:bogus)
        @test_throws ArgumentError fit_arimax(y, (1, 0, 0), x; method=:bogus)
        @test_throws ArgumentError fit_arimax(y, (1, 0, 0), x; se_type=:bogus)
        @test_throws ArgumentError fit_arimax(y, (1, 0, 0), x; model=:mle, Q_beta=[0.0])
        @test_throws ArgumentError fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss, Q_beta=[-1.0])
        @test_throws ArgumentError fit_arimax(y, (1, 0, 0), x; include_mean=false, model=:tvss, Q_beta=[0.0, 0.0])  # wrong length (k=1)
        @test_throws ArgumentError fit_arimax(y, (-1, 0, 0), x)
        @test_throws ArgumentError fit_arimax(y, (1, 0, 0), x[1:100, :])  # row mismatch
        @test_throws ArgumentError fit_arimax(y, (1, 0, 0), zeros(150, 0); include_mean=false)  # no columns, no mean
        @test_throws ArgumentError fit_sarimax(y, (1, 0, 0), (1, 0, 0, 1), x)  # s<2 with P>0
        @test_throws ArgumentError fit_arimax(y, (1, 0, 0), x; n_restarts=0)
    end
end
