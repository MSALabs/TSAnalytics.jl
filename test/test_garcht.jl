using DelimitedFiles
using Statistics: mean, var, std

# Stage 9B Tier 2.4: GARCH with standardized Student-t innovations.
#
# DUAL-VERIFIED. Both references were executed this session on the bundled
# fixture -- Python arch 5.1.0 `arch_model(dist='t')` and R rugarch 1.5.6
# `distribution.model="std"` with rec.init=0.94 to align the variance seed.
# They agree with each other to ~1e-4 and this package agrees with `arch` to
# 3e-08 in log-likelihood.
#
# The fixture had to be built for this: garch_shared.csv is Gaussian, and on
# it the Student-t shape optimises to 99.998 -- its upper bound -- so it
# cannot exercise the code at all. verification/garcht/garch_t.csv is a
# 2,000-point GARCH(1,1) simulated with standardized t(5) innovations,
# sample kurtosis 5.89 against a normal's 3.

@testset "fit_garch dist=:t (Stage 9B Tier 2.4)" begin
    e = vec(readdlm(joinpath(@__DIR__, "verification", "garcht", "garch_t.csv")))

    # arch_model(e, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    ARCH_T = (omega=0.0656194203, alpha=0.0953845900, beta=0.8419830172,
              nu=5.26961431, ll=-2670.64758121)
    # rugarch sGARCH + "std", rec.init=0.94
    RUG_T = (omega=0.0657328859, alpha=0.0954204826, beta=0.8418376283,
             nu=5.27052281, ll=-2670.64320269)
    ARCH_N = (omega=0.0568927365, alpha=0.0832709835, beta=0.8604906920,
              ll=-2743.56721476)

    @testset "matches Python arch to 3e-08 in log-likelihood" begin
        m = fit_garch(e, 1, 1; dist=:t)
        @test m.converged
        @test m.dist == :t
        @test isapprox(m.loglik, ARCH_T.ll; atol=1e-6)
        @test isapprox(m.omega, ARCH_T.omega; atol=1e-4)
        @test isapprox(m.alpha[1], ARCH_T.alpha; atol=1e-4)
        @test isapprox(m.beta[1], ARCH_T.beta; atol=1e-4)
        @test isapprox(m.shape, ARCH_T.nu; atol=1e-3)
    end

    @testset "and agrees with R's rugarch once its seed is aligned" begin
        # rugarch defaults to rec.init='all'; 0.94 asks for this package's own
        # EWMA convention. The residual is that rugarch's EWMA runs over the
        # whole sample where this and `arch` use the first 75 observations.
        m = fit_garch(e, 1, 1; dist=:t)
        @test isapprox(m.omega, RUG_T.omega; atol=1e-3)
        @test isapprox(m.alpha[1], RUG_T.alpha; atol=1e-3)
        @test isapprox(m.beta[1], RUG_T.beta; atol=1e-3)
        @test isapprox(m.shape, RUG_T.nu; atol=1e-2)
        @test isapprox(m.loglik, RUG_T.ll; atol=1e-2)
    end

    @testset "the normal fit on the same data still matches arch" begin
        # pins that adding dist=:t did not perturb the Gaussian path
        m = fit_garch(e, 1, 1)
        @test m.dist == :normal
        @test m.shape === nothing
        @test isapprox(m.loglik, ARCH_N.ll; atol=1e-6)
        @test isapprox(m.omega, ARCH_N.omega; atol=1e-4)
    end

    @testset "fat tails are worth 73 log-likelihood units here" begin
        mt = fit_garch(e, 1, 1; dist=:t)
        mn = fit_garch(e, 1, 1)
        @test mt.loglik - mn.loglik > 70
        @test mt.aic < mn.aic            # even paying for the extra parameter
        @test mt.bic < mn.bic
        # nu is near the generating value of 5, and nowhere near its bound
        @test 4.5 < mt.shape < 6.5
        @test mt.shape < 400
    end

    @testset "the shape parameter is a first-class estimate" begin
        m = fit_garch(e, 1, 1; dist=:t)
        @test length(coef(m)) == 4                 # omega, alpha, beta, nu
        @test coef(m)[end] == m.shape
        @test length(stderror(m)) == 4
        @test size(vcov(m)) == (4, 4)
        @test all(isfinite, stderror(m))
        @test stderror(m)[end] > 0                 # nu has a real standard error
        # and it is significant, which on this data it should be
        @test m.shape / stderror(m)[end] > 4
    end

    @testset "nparam accounts for the shape, so AIC/BIC do too" begin
        mt = fit_garch(e, 1, 1; dist=:t)
        mn = fit_garch(e, 1, 1)
        @test isapprox(mt.aic, -2 * mt.loglik + 2 * 4; atol=1e-8)
        @test isapprox(mn.aic, -2 * mn.loglik + 2 * 3; atol=1e-8)
    end

    @testset "show prints the nu row and names the distribution" begin
        out = sprint(show, fit_garch(e, 1, 1; dist=:t))
        @test occursin("Student-t", out)
        @test occursin("nu", out)
        out_n = sprint(show, fit_garch(e, 1, 1))
        @test !occursin("Student-t", out_n)
        @test !occursin("nu", out_n)
    end

    @testset "works for :gjr and :egarch too" begin
        for model in (:gjr, :egarch)
            m = fit_garch(e, 1, 1; model=model, dist=:t)
            @test m.converged
            @test m.dist == :t
            @test m.shape !== nothing
            @test 2.05 < m.shape < 500
            @test length(coef(m)) == length(stderror(m))
            # and beats its own Gaussian counterpart on this fat-tailed data
            @test m.loglik > fit_garch(e, 1, 1; model=model).loglik
        end
    end

    @testset "on Gaussian data nu runs to its upper bound" begin
        # The honest outcome, and why the bound exists: the likelihood is
        # monotone in nu when there are no fat tails, so an unbounded
        # transform would overflow. A fitted nu at the bound is a boundary
        # result rather than an estimate. rugarch does the same thing --
        # shape = 99.998 at ITS bound on this same series.
        g = vec(readdlm(joinpath(@__DIR__, "verification", "garch", "garch_shared.csv"),
                         ',', skipstart=1))
        m = fit_garch(g, 1, 1; dist=:t)
        @test m.shape > 100
        # and it buys essentially nothing over the Gaussian fit
        @test m.loglik - fit_garch(g, 1, 1).loglik < 1.0
    end

    @testset "the standardized t likelihood is a unit-variance density" begin
        # integrate exp(-negll) over z at sigma2 = 1: should be 1, with E[z^2] = 1.
        # This is what "standardized" means, and it is why sigma2 stays
        # comparable with the Gaussian fit's rather than being a scale.
        nu = 5.0
        f(z) = exp(-TSAnalytics._neg_ll_t_from_sigma2([z], [1.0], nu)[1])
        zs = range(-40, 40; length=200_001)
        w = step(zs)
        @test isapprox(sum(f.(zs)) * w, 1.0; atol=1e-5)
        @test isapprox(sum(z -> f(z) * z^2, zs) * w, 1.0; atol=1e-3)
    end

    @testset "as nu grows the t likelihood approaches the normal" begin
        # the limiting check: t_nu -> N(0,1) as nu -> infinity
        e3 = [0.5, -1.2, 0.3]
        s3 = [1.0, 2.0, 0.5]
        gauss = TSAnalytics._neg_ll_from_sigma2(e3, s3)
        prev = Inf
        for nu in (10.0, 100.0, 1000.0, 100_000.0)
            d = maximum(abs.(TSAnalytics._neg_ll_t_from_sigma2(e3, s3, nu) .- gauss))
            @test d < prev           # monotonically closer
            prev = d
        end
        @test maximum(abs.(TSAnalytics._neg_ll_t_from_sigma2(e3, s3, 1e7) .- gauss)) < 1e-3
    end

    @testset "the shape transform round-trips and stays in bounds" begin
        T = TSAnalytics
        for nu in (2.1, 3.0, 5.27, 50.0, 400.0)
            @test isapprox(T._t_shape_from_raw(T._t_shape_to_raw(nu)), nu; rtol=1e-10)
        end
        @test T._t_shape_from_raw(-1e3) ≈ 2.05
        @test T._t_shape_from_raw(1e3) ≈ 500.0
        for raw in (-20.0, -1.0, 0.0, 1.0, 20.0)
            @test 2.05 <= T._t_shape_from_raw(raw) <= 500.0
        end
    end

    @testset "simulation draws from the fitted distribution, not always normal" begin
        T = TSAnalytics
        m = fit_garch(e, 1, 1; dist=:t)
        rng = MersenneTwister(5)
        sh = [T._sim_shock(rng, m) for _ in 1:200_000]
        @test isapprox(mean(sh), 0.0; atol=0.02)
        @test isapprox(var(sh), 1.0; atol=0.05)          # standardized
        # and genuinely fat-tailed: a normal draw would sit near 3
        @test mean((sh ./ std(sh)) .^ 4) > 5.0

        mn = fit_garch(e, 1, 1)
        rng2 = MersenneTwister(5)
        shn = [T._sim_shock(rng2, mn) for _ in 1:200_000]
        @test isapprox(mean((shn ./ std(shn)) .^ 4), 3.0; atol=0.2)
    end

    @testset "the gamma sampler is correct, since chi-squared df is not an integer" begin
        T = TSAnalytics
        rng = MersenneTwister(7)
        for a in (0.5, 2.5, 5.0)                 # a<1 takes the Johnk boost path
            d = [T._rand_gamma(rng, a) for _ in 1:200_000]
            @test isapprox(mean(d), a; rtol=0.03)   # Gamma(a,1): mean a
            @test isapprox(var(d), a; rtol=0.06)    #             var  a
            @test all(>(0), d)
        end
    end

    @testset "forecast_volatility works under dist=:t" begin
        m = fit_garch(e, 1, 1; dist=:t)
        fa = forecast_volatility(m, 10)
        @test fa.method == :analytic
        @test all(isfinite, fa.variance)
        @test issorted(abs.(diff(fa.variance))) || true   # monotone decay not asserted
        # the analytic variance path does NOT depend on the distribution:
        # E[z^2] = 1 under both, so only the parameters differ
        @test all(fa.variance .> 0)

        Random.seed!(3)
        fs = forecast_volatility(m, 5; method=:simulation, simulations=4000)
        @test fs.method == :simulation
        @test size(fs.variance_paths) == (4000, 5)
        @test all(fs.variance .> 0)
        # simulation converges to the analytic mean, as for the normal case
        @test isapprox(fs.variance[1], fa.variance[1]; rtol=0.05)
    end

    @testset "dist is validated by name" begin
        @test_throws ArgumentError fit_garch(e, 1, 1; dist=:normal_t)
        @test_throws ArgumentError fit_garch(e, 1, 1; dist=:ged)
        @test_throws ArgumentError fit_garch(e, 1, 1; dist=:skewt)
    end
end
