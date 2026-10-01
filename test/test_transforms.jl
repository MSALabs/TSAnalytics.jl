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

@testset "boxcox_inv bias correction (Stage 9B Tier 2.4)" begin
    # R 4.6.0 forecast::InvBoxCox(fc, lambda=, biasadj=TRUE, fvar=se^2),
    # executed this session across five lambda values -- see
    # verification/boxcoxbias/invboxcox.R. Agreement to 4e-11.
    CASES = [
        (0.3,  [1.0, 1.5, 2.0], [0.2, 0.3, 0.4],
         [2.3977901641, 3.4505898523, 4.7907106623],
         [2.4176535146, 3.5022871557, 4.8955074580]),
        (0.0,  [2.0, 2.5], [0.5, 0.5],
         [7.3890560989, 12.1824939607],
         [8.3126881113, 13.7053057058]),
        (1.0,  [3.0, 4.0], [0.1, 0.2],
         [4.0, 5.0],
         [4.0, 5.0]),
        (-0.5, [1.2, 1.8], [0.3, 0.3],
         [6.25, 100.0],
         [8.88671875, 775.0]),
        (0.75, [5.0, 6.0], [0.4, 0.6],
         [7.9846915911, 9.7084579221],
         [7.9917694341, 9.7229002562]),
    ]

    @testset "matches R's InvBoxCox at lambda = $lam" for (lam, fc, se, rn, ra) in CASES
        @test isapprox(boxcox_inv(fc, lam), rn; atol=1e-8)
        @test isapprox(boxcox_inv(fc, lam; fvar=se .^ 2), ra; atol=1e-8)
    end

    @testset "lambda = 1 means no transform, so no bias to correct" begin
        # the (1 - lambda) factor vanishes; this is the degenerate case that
        # catches a sign or placement error in the formula
        fc = [3.0, 4.0]
        @test boxcox_inv(fc, 1.0; fvar=[0.1, 0.2]) == boxcox_inv(fc, 1.0)
        @test isapprox(boxcox_inv(fc, 1.0), [4.0, 5.0]; atol=1e-12)
    end

    @testset "lambda = 0 reduces to the log-normal mean correction" begin
        # exp(y) * (1 + fvar/2), the familiar closed form
        for (y, v) in ((2.0, 0.5), (0.0, 0.25), (-1.0, 1.0))
            @test isapprox(boxcox_inv([y], 0; fvar=v)[1], exp(y) * (1 + v / 2); rtol=1e-12)
        end
    end

    @testset "the correction is upward, and grows with the variance" begin
        # the median of a right-skewed back-transform sits below the mean, so
        # correcting can only raise the forecast for lambda < 1
        fc = [1.0, 1.5, 2.0]
        plain = boxcox_inv(fc, 0.3)
        small = boxcox_inv(fc, 0.3; fvar=fill(0.01, 3))
        large = boxcox_inv(fc, 0.3; fvar=fill(0.25, 3))
        @test all(small .> plain)
        @test all(large .> small)
        # zero variance is no correction at all
        @test boxcox_inv(fc, 0.3; fvar=zeros(3)) == plain
    end

    @testset "a negative lambda makes the correction large, not subtle" begin
        # lambda = -0.5 with sd 0.3 turns 100.0 into 775.0 -- a 7.75x factor.
        # Worth pinning: it is the case where quoting a corrected mean without
        # looking at it would be most misleading.
        @test isapprox(boxcox_inv([1.8], -0.5; fvar=[0.09])[1], 775.0; atol=1e-8)
        @test boxcox_inv([1.8], -0.5; fvar=[0.09])[1] / boxcox_inv([1.8], -0.5)[1] > 7.0
    end

    @testset "fvar accepts a scalar or a vector" begin
        fc = [1.0, 1.5, 2.0]
        @test boxcox_inv(fc, 0.3; fvar=0.09) == boxcox_inv(fc, 0.3; fvar=fill(0.09, 3))
    end

    @testset "round-trip on data is unaffected" begin
        x = [1.0, 2.0, 4.0, 8.0]
        for lam in (-0.5, 0.0, 0.3, 1.0, 1.5)
            @test isapprox(boxcox_inv(boxcox(x, lam), lam), x; atol=1e-9)
        end
    end

    @testset "error paths" begin
        @test_throws DimensionMismatch boxcox_inv([1.0, 2.0], 0.3; fvar=[0.1])
        @test_throws ArgumentError boxcox_inv([1.0], 0.3; fvar=[-0.1])   # a variance
        @test_throws ArgumentError boxcox_inv([1.0], 0.3; fvar=-1.0)
    end

    @testset "end to end: log-forecast a real series and correct it" begin
        y = dataset("cardox").value
        train = log.(y[1:240])
        f = forecast(fit_sarima(train, (0, 1, 1), (0, 1, 1, 12)), 12)
        med = boxcox_inv(f.point, 0.0)                  # median
        mu = boxcox_inv(f.point, 0.0; fvar=f.se .^ 2)   # mean
        @test all(mu .>= med)
        @test all(isfinite, mu)
        # the gap widens with the horizon, because the forecast variance does
        @test (mu[12] - med[12]) > (mu[1] - med[1])
        # on a tight fit the correction is small -- it is not always material,
        # and saying so is the point
        @test maximum((mu .- med) ./ med) < 0.01
    end
end
