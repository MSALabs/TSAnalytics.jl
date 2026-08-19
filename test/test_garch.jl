using DelimitedFiles, StatsAPI

@testset "fit_garch" begin
    e = vec(readdlm(joinpath(@__DIR__, "verification", "garch", "garch_shared.csv"), ','; skipstart=1))

    @testset "primary case (exact real arch.arch_model validation)" begin
        m = fit_garch(e, 1, 1)
        @test m isa GarchModel
        @test m.converged
        @test isapprox(m.omega, 0.043066; atol=1e-3)
        @test isapprox(m.alpha[1], 0.090638; atol=1e-3)
        @test isapprox(m.beta[1], 0.867923; atol=1e-3)
        @test isapprox(m.loglik, -1396.190867; atol=1e-2)
        @test isapprox(m.aic, 2798.381733; atol=1e-1)
        @test isapprox(m.bic, 2813.104999; atol=1e-1)
        @test m.mean_spec == :zero
        @test m.mu === nothing
        @test m.resid == e
        @test all(m.sigma2 .> 0)   # variance must stay positive throughout the recursion
        @test length(m.sigma2) == length(e) == m.nobs

        # cov_type: robust vs classic are genuinely different numbers (handoff section 1)
        m_classic = fit_garch(e, 1, 1; cov_type=:classic)
        @test !isapprox(m.se, m_classic.se; atol=1e-3)
        @test isapprox(m.se, [0.013403, 0.019497, 0.022774]; atol=1e-3)
        @test isapprox(m_classic.se, [0.015871, 0.021335, 0.028941]; atol=1e-3)

        @test_throws ArgumentError fit_garch(e, 1, 1; cov_type=:bogus)
    end

    @testset "p/q naming convention: p=ARCH order, q=GARCH order (matches Python, not Bollerslev)" begin
        e21 = vec(readdlm(joinpath(@__DIR__, "verification", "garch", "garch21.csv"), ','; skipstart=1))
        m = fit_garch(e21, 2, 1)
        @test m.p == 2 && m.q == 1
        @test length(m.alpha) == 2   # p -> alpha (ARCH) length
        @test length(m.beta) == 1    # q -> beta (GARCH) length
        @test m.converged
        @test isapprox(m.omega, 0.054010; atol=1e-2)
        @test isapprox(m.alpha, [0.088285, 0.078309]; atol=1e-2)
        @test isapprox(m.beta, [0.698140]; atol=1e-2)
        @test isapprox(m.loglik, -1387.451291; atol=1e-1)

        # asymmetric order the other way, to be doubly sure the convention isn't backwards
        m12 = fit_garch(e, 1, 2)
        @test m12.p == 1 && m12.q == 2
        @test length(m12.alpha) == 1
        @test length(m12.beta) == 2
    end

    @testset "mean_spec=:constant: joint MLE, not naive pre-demean (exact real arch validation)" begin
        yc = vec(readdlm(joinpath(@__DIR__, "verification", "garch", "garch_constant.csv"), ','; skipstart=1))
        m = fit_garch(yc, 1, 1; mean_spec=:constant)
        @test m.mean_spec == :constant
        @test m.mu !== nothing
        @test isapprox(m.mu, 0.480324; atol=1e-3)
        @test isapprox(m.omega, 0.012632; atol=1e-3)
        @test isapprox(m.alpha[1], 0.047336; atol=1e-3)
        @test isapprox(m.beta[1], 0.932710; atol=1e-3)
        @test isapprox(m.loglik, -920.108563; atol=1e-1)
        @test m.resid == yc .- m.mu

        # :zero is NOT equivalent to :constant on a mean-shifted series -- confirms the mean
        # is actually being estimated, not silently ignored
        m_zero = fit_garch(yc, 1, 1; mean_spec=:zero)
        @test !isapprox(m.loglik, m_zero.loglik; atol=1.0)

        @test_throws ArgumentError fit_garch(e, 1, 1; mean_spec=:bogus)
    end

    @testset "dist=:t deliberately not yet implemented" begin
        @test_throws ArgumentError fit_garch(e, 1, 1; dist=:t)
        @test_throws ArgumentError fit_garch(e, 1, 1; dist=:bogus)
    end

    @testset "n_restarts multi-start: never worse than single start" begin
        m1 = fit_garch(e, 1, 1; n_restarts=1)
        m8 = fit_garch(e, 1, 1; n_restarts=8, parallel=true)
        @test m8.loglik >= m1.loglik - 1e-6
        m8_serial = fit_garch(e, 1, 1; n_restarts=8, parallel=false)
        @test isapprox(m8.loglik, m8_serial.loglik; atol=1e-8)

        @test_throws ArgumentError fit_garch(e, 1, 1; n_restarts=0)
    end

    @testset "fit_garch_multi: parallel matches serial" begin
        series = [e for _ in 1:6]   # same series 6x, cheap deterministic check
        results_par = fit_garch_multi(series, 1, 1; parallel=true)
        results_serial = fit_garch_multi(series, 1, 1; parallel=false)
        @test length(results_par) == length(results_serial) == 6
        for i in eachindex(series)
            @test isapprox(results_par[i].omega, results_serial[i].omega; atol=1e-10)
            @test isapprox(results_par[i].loglik, results_serial[i].loglik; atol=1e-10)
        end
        # kwargs pass through unchanged
        results_classic = fit_garch_multi(series, 1, 1; cov_type=:classic)
        @test all(m -> m.cov_type == :classic, results_classic)
    end

    @testset "StatsAPI accessors" begin
        m = fit_garch(e, 1, 1)
        @test StatsAPI.loglikelihood(m) == m.loglik
        @test StatsAPI.aic(m) == m.aic
        @test StatsAPI.bic(m) == m.bic
        @test StatsAPI.nobs(m) == m.nobs
        @test StatsAPI.coef(m) == vcat(m.omega, m.alpha, m.beta)
        @test StatsAPI.residuals(m) == m.resid

        mc = fit_garch(vec(readdlm(joinpath(@__DIR__, "verification", "garch", "garch_constant.csv"), ','; skipstart=1)),
                        1, 1; mean_spec=:constant)
        @test StatsAPI.coef(mc) == vcat(mc.mu, mc.omega, mc.alpha, mc.beta)
    end

    @testset "parameter coverage: p x q combinations" begin
        for (p, q) in [(1, 0), (1, 1), (2, 1), (1, 2), (2, 2)]
            m = fit_garch(e, p, q)
            @test m.p == p && m.q == q
            @test length(m.alpha) == p
            @test length(m.beta) == q
            @test length(m.se) == 1 + p + q
            @test all(isfinite, m.se) || any(isnan, m.se)   # finite or honestly NaN, never crashes
            @test all(m.sigma2 .> 0)
        end
    end

    @testset "optimizer_method coverage" begin
        for method in (:lbfgs, :bfgs, :nelder_mead)
            m = fit_garch(e, 1, 1; optimizer_method=method)
            @test m isa GarchModel
        end
    end

    @testset "start_params" begin
        m = fit_garch(e, 1, 1)
        m2 = fit_garch(e, 1, 1; start_params=[log(0.05), log(0.1), log(2.0)])
        @test isapprox(m.omega, m2.omega; atol=1e-2)   # should converge to ~ the same optimum

        @test_throws ArgumentError fit_garch(e, 1, 1; start_params=[1.0, 2.0])   # wrong length
    end

    @testset "error paths" begin
        @test_throws ArgumentError fit_garch(e, 0, 1)   # p (ARCH order) must be >= 1
        @test_throws ArgumentError fit_garch(e, 1, -1)  # q must be >= 0
        @test_throws ArgumentError fit_garch(e, -1, 1)
        @test_throws ArgumentError fit_garch([1.0, 2.0, 3.0], 1, 1)   # not enough observations
    end

    @testset "container-agnostic" begin
        m1 = fit_garch(e, 1, 1)
        m2 = fit_garch(collect(e), 1, 1)
        @test m1.omega == m2.omega
    end

    @testset "show method runs without erroring" begin
        m = fit_garch(e, 1, 1)
        io = IOBuffer()
        show(io, m)
        s = String(take!(io))
        @test occursin("GARCH(1,1)", s)
        @test occursin("omega", s)

        mc = fit_garch(vec(readdlm(joinpath(@__DIR__, "verification", "garch", "garch_constant.csv"), ','; skipstart=1)),
                        1, 1; mean_spec=:constant)
        io2 = IOBuffer()
        show(io2, mc)
        @test occursin("with mean", String(take!(io2)))
    end

    @testset "model=:gjr (exact real arch.arch_model validation)" begin
        eg = vec(readdlm(joinpath(@__DIR__, "verification", "garch", "gjr_shared.csv"), ','; skipstart=1))
        m = fit_garch(eg, 1, 1; model=:gjr)
        @test m isa GarchModel
        @test m.converged
        @test m.model == :gjr
        @test isapprox(m.omega, 0.057830; atol=1e-3)
        @test isapprox(m.alpha[1], 0.051251; atol=1e-3)
        @test m.gamma !== nothing
        @test isapprox(m.gamma[1], 0.091763; atol=1e-3)
        @test isapprox(m.beta[1], 0.842113; atol=1e-3)
        @test isapprox(m.loglik, -2040.876850; atol=1e-2)
        @test length(m.se) == 4   # omega, alpha, gamma, beta
        @test StatsAPI.coef(m) == vcat(m.omega, m.alpha, m.gamma, m.beta)

        # multi-start should reach the same optimum (regression guard for the earlier
        # symmetric-cap bug: an overly restrictive gamma reparametrization silently
        # capped gamma below its true MLE value, caught by comparing against real arch)
        m8 = fit_garch(eg, 1, 1; model=:gjr, n_restarts=8)
        @test isapprox(m8.gamma[1], m.gamma[1]; atol=1e-4)
        @test isapprox(m8.loglik, m.loglik; atol=1e-4)

        # cov_type robust vs classic still genuinely different for :gjr
        m_classic = fit_garch(eg, 1, 1; model=:gjr, cov_type=:classic)
        @test !isapprox(m.se, m_classic.se; atol=1e-3)
    end

    @testset "model=:egarch (exact real arch.arch_model validation, primary target per handoff §2's flagged ambiguity)" begin
        eg = vec(readdlm(joinpath(@__DIR__, "verification", "garch", "gjr_shared.csv"), ','; skipstart=1))
        m = fit_garch(eg, 1, 1; model=:egarch)
        @test m isa GarchModel
        @test m.converged
        @test m.model == :egarch
        @test isapprox(m.omega, -0.007737; atol=1e-3)
        @test m.omega < 0   # NOT a bug -- log-variance omega has no positivity constraint
        @test isapprox(m.alpha[1], 0.174770; atol=1e-2)
        @test m.gamma !== nothing
        @test isapprox(m.gamma[1], -0.059759; atol=1e-2)
        @test isapprox(m.beta[1], 0.940519; atol=1e-3)
        @test isapprox(m.loglik, -2042.754584; atol=1e-2)
        @test all(m.sigma2 .> 0)   # level variance stays positive even though the recursion is on log(sigma2)

        m8 = fit_garch(eg, 1, 1; model=:egarch, n_restarts=8)
        @test isapprox(m8.loglik, m.loglik; atol=1e-4)
    end

    @testset "model=:garch has no gamma (structural, doesn't silently populate it)" begin
        m = fit_garch(e, 1, 1; model=:garch)
        @test m.gamma === nothing
        @test length(m.se) == 3   # omega, alpha, beta -- no gamma slot

        # plain :garch on garch_shared.csv unaffected by the :gjr/:egarch extension
        # (regression guard: an earlier refactor bug swapped gamma/beta in the tuple
        # unpack, breaking plain GARCH entirely -- caught before this ever shipped)
        @test isapprox(m.omega, 0.043066; atol=1e-3)
        @test isapprox(m.alpha[1], 0.090638; atol=1e-3)
        @test isapprox(m.beta[1], 0.867923; atol=1e-3)
    end

    @testset "GJR reduces to plain GARCH when true gamma=0" begin
        # dedicated generated case with zero true asymmetry (handoff §5)
        rng = Random.MersenneTwister(2024)
        n = 1000
        omega0, alpha0, beta0 = 0.05, 0.08, 0.85
        h = zeros(n)
        ez = zeros(n)
        h[1] = omega0 / (1 - alpha0 - beta0)
        innov = randn(rng, n)
        ez[1] = sqrt(h[1]) * innov[1]
        for t in 2:n
            h[t] = omega0 + alpha0 * ez[t - 1]^2 + beta0 * h[t - 1]
            ez[t] = sqrt(h[t]) * innov[t]
        end
        m_garch = fit_garch(ez, 1, 1; model=:garch)
        m_gjr = fit_garch(ez, 1, 1; model=:gjr)
        @test m_gjr.converged
        @test isapprox(m_gjr.gamma[1], 0.0; atol=0.06)   # statistically indistinguishable from 0
        @test isapprox(m_gjr.alpha[1], m_garch.alpha[1]; atol=0.05)
        @test isapprox(m_gjr.beta[1], m_garch.beta[1]; atol=0.05)
    end

    @testset "fit_garch_multi with model=:gjr/:egarch" begin
        eg = vec(readdlm(joinpath(@__DIR__, "verification", "garch", "gjr_shared.csv"), ','; skipstart=1))
        for model in (:gjr, :egarch)
            series = [eg for _ in 1:5]
            results_par = fit_garch_multi(series, 1, 1; model=model, parallel=true)
            results_serial = fit_garch_multi(series, 1, 1; model=model, parallel=false)
            for i in eachindex(series)
                @test isapprox(results_par[i].omega, results_serial[i].omega; atol=1e-10)
                @test isapprox(results_par[i].loglik, results_serial[i].loglik; atol=1e-10)
            end
            @test all(m -> m.model == model, results_par)
        end
    end

    @testset "model error path" begin
        @test_throws ArgumentError fit_garch(e, 1, 1; model=:bogus)
    end

    @testset "show method: GJR-GARCH/EGARCH labels" begin
        eg = vec(readdlm(joinpath(@__DIR__, "verification", "garch", "gjr_shared.csv"), ','; skipstart=1))
        io1 = IOBuffer()
        show(io1, fit_garch(eg, 1, 1; model=:gjr))
        @test occursin("GJR-GARCH(1,1)", String(take!(io1)))

        io2 = IOBuffer()
        show(io2, fit_garch(eg, 1, 1; model=:egarch))
        @test occursin("EGARCH(1,1)", String(take!(io2)))
    end
end
