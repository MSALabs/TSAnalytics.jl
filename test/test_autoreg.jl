using DelimitedFiles

# Stage 9B Tier 2.4, final item: regression with autoregressive errors --
# SAS PROC AUTOREG's basic tier, the Cochrane-Orcutt/Prais-Winsten family.
#
# VERIFICATION STRATEGY, because no single function matches this. `orcutt`
# implements Cochrane-Orcutt but will not install on this R version (tried),
# and `nlme::gls` fits by ML/REML rather than feasible GLS, so its rho
# differs by construction.
#
# AT A FIXED rho the two ARE the same estimator, so that is the comparison:
# autoreg with phi pinned reproduces
# nlme::gls(y ~ x, correlation=corAR1(value=rho, fixed=TRUE), method="ML")
# on both beta and its standard errors, to 5e-11, at three values of rho.
# The AR half is separately R-verified through ar_yw, and the OLS starting
# point against R's lm. Both components have a reference; the composition
# does not, and that is stated rather than implied.

@testset "autoreg — regression with AR errors (Stage 9B Tier 2.4)" begin
    d = readdlm(joinpath(@__DIR__, "verification", "autoreg", "ar1reg.csv"), ','; skipstart=1)
    y = d[:, 1]
    X = hcat(ones(size(d, 1)), d[:, 2])

    # nlme::gls(..., corAR1(value=rho, fixed=TRUE), method="ML")
    GLS_REF = Dict(
        0.5    => ([1.6005033984, 2.2651289397], [0.1482446792, 0.1324492191]),
        0.7    => ([1.5940058880, 2.1572699701], [0.2377242966, 0.1937205685]),
        0.8460 => ([1.5762755975, 1.9361099702], [0.4650218593, 0.2849522151]),
    )

    @testset "GLS at a fixed phi matches nlme::gls exactly (rho = $rho)" for rho in
            (0.5, 0.7, 0.8460)
        b, se = GLS_REF[rho]
        m = autoreg(y, X; phi=[rho])
        @test isapprox(m.beta, b; atol=1e-9)
        @test isapprox(m.se, se; atol=1e-9)
        @test m.method == :fixed
        @test m.iterations == 0
        @test m.phi == [rho]
        @test m.converged
    end

    @testset "the OLS starting point matches R's lm" begin
        # R: summary(lm(y ~ x))  ->  beta 1.6012368503 2.3206766674
        #                            se   0.0988494490 0.0918007124
        ob, _, ose = TSAnalytics._ols(X, y; method=:qr)
        @test isapprox(ob, [1.6012368503, 2.3206766674]; atol=1e-9)
        @test isapprox(ose, [0.0988494490, 0.0918007124]; atol=1e-9)
    end

    @testset "iterated feasible GLS lands near nlme's ML estimate" begin
        # Not identical: feasible GLS and ML are different estimators. Close
        # is the expected and correct outcome, and asserting equality would
        # be asserting something false.
        m = autoreg(y, X)
        @test m.converged
        @test m.order == 1
        @test m.method == :yw
        @test isapprox(m.phi[1], 0.6866288094; atol=0.01)      # nlme ML rho
        @test isapprox(m.beta, [1.5948583878, 2.1693207540]; atol=0.01)
        @test 1 < m.iterations <= 50
    end

    @testset "this is why OLS standard errors mislead under autocorrelation" begin
        # The whole motivation. OLS beta stays unbiased; its standard errors
        # do not, and with positively autocorrelated errors they are too
        # SMALL, so t-statistics are inflated.
        m = autoreg(y, X)
        _, _, ose = TSAnalytics._ols(X, y; method=:qr)
        @test m.se[2] > ose[2]                        # corrected is larger
        @test m.se[2] / ose[2] > 1.8                  # measured 2.05x
        t_ols = 2.3206766674 / ose[2]
        t_cor = m.beta[2] / m.se[2]
        @test t_ols > 2 * t_cor / 1.3                 # 25.3 vs 11.5
        @test t_cor < t_ols
    end

    @testset "beta stays close to OLS while its standard error does not" begin
        # the textbook pairing: correcting serial correlation moves the
        # standard errors a lot and the point estimates comparatively little
        m = autoreg(y, X)
        ob, _, ose = TSAnalytics._ols(X, y; method=:qr)
        @test abs(m.beta[1] - ob[1]) < 0.05
        @test abs(m.beta[2] - ob[2]) < 0.2
        @test m.se[2] / ose[2] > 1.8
    end

    @testset "Prais-Winsten keeps every observation" begin
        # not Cochrane-Orcutt, which discards the first `order` rows
        m = autoreg(y, X)
        @test m.n == length(y)
        @test length(m.nu) == length(y)
        @test length(m.resid) == length(y)
        @test all(isfinite, m.resid)
    end

    @testset "the residuals are whiter than the regression errors" begin
        # nu is autocorrelated by construction; eps should not be
        m = autoreg(y, X)
        @test abs(acf(m.nu, [1]).values[1]) > 0.5         # errors are correlated
        @test abs(acf(m.resid, [1]).values[1]) < 0.15     # innovations much less
        @test ljungbox_test(m.resid, 10).pvalue > 0.01
    end

    @testset "method=:ols is a genuinely different AR step" begin
        a = autoreg(y, X; method=:yw)
        b = autoreg(y, X; method=:ols)
        @test a.method == :yw && b.method == :ols
        @test !isapprox(a.phi, b.phi; atol=1e-6)
        # but they agree on the broad answer
        @test isapprox(a.beta, b.beta; atol=0.05)
    end

    @testset "vcov is consistent with se, and symmetric" begin
        m = autoreg(y, X)
        @test size(m.vcov) == (2, 2)
        @test isapprox(sqrt.(diag(m.vcov)), m.se; rtol=1e-12)
        @test isapprox(m.vcov, transpose(m.vcov); atol=1e-12)
        @test all(diag(m.vcov) .> 0)
    end

    @testset "higher AR orders fit and stay stationary" begin
        for ord in 1:3
            m = autoreg(y, X; order=ord)
            @test length(m.phi) == ord
            @test TSAnalytics._ar_stationary(m.phi)
            @test all(isfinite, m.beta)
            @test all(isfinite, m.se)
        end
    end

    @testset "a non-stationary fixed phi is refused, not silently used" begin
        # the GLS transform needs the stationary covariance to exist
        @test_throws ArgumentError autoreg(y, X; phi=[1.0])
        @test_throws ArgumentError autoreg(y, X; phi=[1.5])
        @test_throws DimensionMismatch autoreg(y, X; order=2, phi=[0.5])
    end

    @testset "the stationary covariance helper is correct" begin
        T = TSAnalytics
        # AR(1): gamma0 = 1/(1-phi^2) with unit innovation variance
        for phi in (0.0, 0.5, -0.7, 0.9)
            G = T._ar_stationary_gamma([phi], 1)
            @test isapprox(G[1, 1], 1 / (1 - phi^2); rtol=1e-10)
        end
        # AR(2): symmetric, positive definite, gamma1/gamma0 = phi1/(1-phi2)
        G2 = T._ar_stationary_gamma([0.5, 0.2], 2)
        @test isapprox(G2, transpose(G2); atol=1e-12)
        @test isapprox(G2[1, 2] / G2[1, 1], 0.5 / (1 - 0.2); rtol=1e-10)
        @test all(eigvals(Symmetric(G2)) .> 0)
    end

    @testset "zero autocorrelation reduces to OLS" begin
        # phi = 0 makes the GLS transform the identity, so beta and se must
        # be OLS's exactly -- the cheapest check that the transform is right
        m = autoreg(y, X; phi=[0.0])
        ob, _, ose = TSAnalytics._ols(X, y; method=:qr)
        @test isapprox(m.beta, ob; atol=1e-10)
        @test isapprox(m.se, ose; atol=1e-10)
    end

    @testset "error paths" begin
        @test_throws ArgumentError autoreg(y, X; method=:bogus)
        @test_throws ArgumentError autoreg(y, X; order=0)
        @test_throws ArgumentError autoreg(y, X; maxiter=0)
        @test_throws DimensionMismatch autoreg(y, X[1:50, :])
        @test_throws ArgumentError autoreg(y[1:3], X[1:3, :]; order=2)
    end

    @testset "show reports the order, method and iteration count" begin
        out = sprint(show, autoreg(y, X))
        @test occursin("AR(1) errors", out)
        @test occursin("yw", out)
        @test occursin("iteration", out)
        @test occursin("phi", out)
    end

    @testset "a single-column X works without reshaping" begin
        m = autoreg(y, d[:, 2])
        @test length(m.beta) == 1
        @test all(isfinite, m.se)
    end
end
