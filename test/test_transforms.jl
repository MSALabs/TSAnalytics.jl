using DelimitedFiles

@testset "boxcox / boxcox_inv" begin
    x = [1.0, 2.0, 4.0, 8.0]

    @test boxcox(x, 0.0) == log.(x)
    for lam in (-1.0, -0.5, 0.0, 0.3, 1.0, 1.5, 2.0)
        y = boxcox(x, lam)
        @test isapprox(boxcox_inv(y, lam), x; atol=1e-8)
    end

    @test isapprox(round.(boxcox([1.0, 2.0, 4.0], 0.5), digits=4), [0.0, 0.8284, 2.0]; atol=1e-4)

    @test_throws ArgumentError boxcox([1.0, -1.0], 0.5)
    @test_throws ArgumentError boxcox([1.0, 0.0], 0.5)
end

@testset "guerrero_lambda (R forecast::BoxCox.lambda-verified)" begin
    vdir = joinpath(@__DIR__, "verification", "transforms")
    xg1 = vec(readdlm(joinpath(vdir, "guerrero_case1.csv"), ','; skipstart=1))
    xg2 = vec(readdlm(joinpath(vdir, "guerrero_case2.csv"), ','; skipstart=1))
    air = _load_column(TSAnalytics.AIR_PASSENGERS, "passengers")

    # BoxCox.lambda(xg1, method="guerrero") -> -0.9999242 (boundary case,
    # see ground-truth-transcript.txt: objective is flat at lambda=-1,
    # R's own optimize() converges to -1 too under a tighter tolerance)
    @test isapprox(guerrero_lambda(xg1, 1), -1.0; atol=1e-3)

    # BoxCox.lambda(ts(xg2, frequency=12), method="guerrero") -> 0.1574706
    @test isapprox(guerrero_lambda(xg2, 12), 0.1574706; atol=1e-3)

    # BoxCox.lambda(AirPassengers, method="guerrero") -> -0.2947156
    @test isapprox(guerrero_lambda(air, 12), -0.2947156; atol=1e-3)

    # length(x) <= 2*period short-circuits to 1.0, matching BoxCox.lambda exactly
    @test guerrero_lambda(abs.(randn(20)) .+ 1.0, 12) == 1.0

    # bounds are respected
    lam = guerrero_lambda(air, 12; bounds=(-1.0, 2.0))
    @test -1.0 <= lam <= 2.0

    @test_throws ArgumentError guerrero_lambda([1.0, -1.0], 12)
    @test_throws ArgumentError guerrero_lambda(air, 0)
    @test_throws ArgumentError guerrero_lambda(air, 12; bounds=(2.0, -1.0))
end

@testset "boxcox_lambda — MLE Box-Cox (Stage 9B Tier 1.4)" begin
    yp = vec(readdlm(joinpath(@__DIR__, "verification", "transforms", "bc_y.csv")))

    @testset "unbounded, matches scipy.stats.boxcox" begin
        # scipy.stats.boxcox(yp) -> lambda = -1.836495441613, executed this
        # session. Widening the bounds removes the only difference between
        # the two, which is what makes this a real cross-language check
        # rather than a restatement of our own answer.
        @test isapprox(boxcox_lambda(yp; bounds=(-5.0, 5.0)), -1.836495441613; atol=1e-5)
    end

    @testset "bounded by default, like R's BoxCox.lambda" begin
        lam = boxcox_lambda(yp)
        @test -1.0 <= lam <= 2.0
        @test isapprox(lam, -1.0; atol=1e-5)        # the optimum is outside, so clamp
        @test lam > boxcox_lambda(yp; bounds=(-5.0, 5.0))
    end

    @testset "it maximises the profile the plot draws" begin
        prof = boxcox_profile_plot(yp, range(-1.0, 2.0, length=601))
        grid_best = prof.lambdas[argmax(prof.loglik)]
        @test isapprox(boxcox_lambda(yp), grid_best; atol=6e-3)
    end

    @testset "answers a different question from guerrero_lambda" begin
        air = Float64.(readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)[:, 2])
        gl = guerrero_lambda(air, 12)
        ml = boxcox_lambda(air)
        @test -1.0 <= gl <= 2.0
        @test -1.0 <= ml <= 2.0
        # both are legitimate; they target variance stabilisation and
        # normality respectively, and need not agree
        @test isfinite(gl) && isfinite(ml)
    end

    @testset "a known answer: an already-normal series wants lambda near 1" begin
        x = 100.0 .+ 5.0 .* randn(MersenneTwister(11), 4000)
        @test isapprox(boxcox_lambda(x), 1.0; atol=0.15)
    end

    @testset "a log-normal series wants lambda near 0" begin
        x = exp.(randn(MersenneTwister(12), 4000))
        @test isapprox(boxcox_lambda(x), 0.0; atol=0.1)
    end

    @testset "error paths" begin
        @test_throws ArgumentError boxcox_lambda([1.0, -2.0, 3.0])
        @test_throws ArgumentError boxcox_lambda([1.0, 0.0, 3.0])
        @test_throws ArgumentError boxcox_lambda([5.0])
        @test_throws ArgumentError boxcox_lambda(yp; bounds=(2.0, -1.0))
    end
end
