using Statistics: mean

# Stage 9B Tier 2.4: forecast combination (Bates & Granger 1969;
# Montgomery section 7.5).
#
# NO PACKAGE REFERENCE EXISTS for this, and that is checked rather than
# assumed: R's `forecast` has no combination function, and
# `forecastHybrid::hybridModel` averages point forecasts but recomputes
# intervals from combined residuals, which needs the fitted objects rather
# than their forecasts -- a different operation. So this is verified against
# its own definition (the weighted mean, and the independence-assumed
# variance) plus the properties a combination must have.

@testset "combine_forecasts (Stage 9B Tier 2.4)" begin
    y = dataset("cardox").value
    train, test = y[1:240], y[241:252]
    f1 = forecast(fit_sarima(train, (1, 1, 1), (0, 1, 1, 12)), 12)
    f2 = seasonal_naive(train, 12, 12)
    f3 = drift(train, 12)

    @testset "equal weights give the plain mean" begin
        c = combine_forecasts([f1, f2, f3])
        @test isapprox(c.point, (f1.point .+ f2.point .+ f3.point) ./ 3; atol=1e-12)
        @test c.horizon == 12
        @test c.levels == f1.levels
        @test c.model_name == "Combination(3, equal weights)"
    end

    @testset "weights are normalised, so scale does not matter" begin
        a = combine_forecasts([f1, f2]; weights=[3, 1])
        b = combine_forecasts([f1, f2]; weights=[0.75, 0.25])
        c = combine_forecasts([f1, f2]; weights=[300, 100])
        @test isapprox(a.point, b.point; atol=1e-12)
        @test isapprox(a.point, c.point; atol=1e-12)
        @test a.model_name == "Combination(2, weighted)"
    end

    @testset "degenerate weights reproduce a single component" begin
        c = combine_forecasts([f1, f2]; weights=[1, 0])
        @test isapprox(c.point, f1.point; atol=1e-12)
        @test isapprox(c.se, f1.se; atol=1e-12)
        @test isapprox(c.lower, f1.lower; atol=1e-10)
    end

    @testset "equal weighting is order-invariant" begin
        a = combine_forecasts([f1, f2, f3])
        b = combine_forecasts([f3, f1, f2])
        @test isapprox(a.point, b.point; atol=1e-12)
        @test isapprox(a.se, b.se; atol=1e-12)
    end

    @testset "the point forecast lies inside the components' range" begin
        c = combine_forecasts([f1, f2, f3])
        lo = min.(f1.point, f2.point, f3.point)
        hi = max.(f1.point, f2.point, f3.point)
        @test all(lo .<= c.point .<= hi)
    end

    @testset "it beats the AVERAGE component but not the BEST one" begin
        # The honest result, and the reason the docstring says the case for
        # combining is robustness rather than optimality. Two of these three
        # are weak benchmarks, and averaging a good forecast with poor ones
        # drags it down.
        c = combine_forecasts([f1, f2, f3])
        rs = [rmse(test, f.point) for f in (f1, f2, f3)]
        rc = rmse(test, c.point)
        @test rc < mean(rs)              # better than the average component
        @test rc > minimum(rs)           # worse than the best -- not a bug
    end

    @testset "two similar models have correlated errors and diversify little" begin
        # Both SARIMAs on the same series, so their errors are strongly
        # positively correlated; the combination lands BETWEEN them rather
        # than below both. Pinned because it is the counterexample to the
        # usual claim that combination always helps.
        a = forecast(fit_sarima(train, (1, 1, 1), (0, 1, 1, 12)), 12)
        b = forecast(fit_sarima(train, (0, 1, 2), (1, 1, 0, 12)), 12)
        c = combine_forecasts([a, b])
        ra, rb, rc = rmse(test, a.point), rmse(test, b.point), rmse(test, c.point)
        @test min(ra, rb) <= rc <= max(ra, rb)
    end

    @testset "the standard error assumes independence, so it is a lower bound" begin
        c = combine_forecasts([f1, f2])
        @test isapprox(c.se, sqrt.(0.25 .* f1.se .^ 2 .+ 0.25 .* f2.se .^ 2); atol=1e-12)
        # with equal weights over k identical forecasts it shrinks by sqrt(k),
        # which is exactly the independence assumption showing itself -- and
        # is wrong, since identical forecasts are perfectly correlated
        c2 = combine_forecasts([f1, f1])
        @test isapprox(c2.point, f1.point; atol=1e-12)
        @test isapprox(c2.se, f1.se ./ sqrt(2); atol=1e-12)
        @test all(c2.se .< f1.se)
    end

    @testset "intervals are rebuilt from the combined point and se" begin
        c = combine_forecasts([f1, f2, f3])
        z80 = TSAnalytics._confidence_z(1 - 80 / 100)
        z95 = TSAnalytics._confidence_z(1 - 95 / 100)
        @test isapprox(c.lower[:, 1], c.point .- z80 .* c.se; atol=1e-12)
        @test isapprox(c.upper[:, 2], c.point .+ z95 .* c.se; atol=1e-12)
        # NOT the average of the component bounds, which would correspond to
        # no distribution at all
        @test !isapprox(c.lower[:, 1], (f1.lower[:, 1] .+ f2.lower[:, 1] .+ f3.lower[:, 1]) ./ 3;
                         atol=1e-6)
        @test all(c.lower[:, 2] .<= c.lower[:, 1])   # 95% outside 80%
        @test all(c.upper[:, 2] .>= c.upper[:, 1])
    end

    @testset "it composes with the scoring rules" begin
        c = combine_forecasts([f1, f2, f3])
        ia = interval_accuracy(test, c)
        @test length(ia.coverage) == 2
        @test ia.crps > 0
        @test all(0 .<= ia.coverage .<= 1)
    end

    @testset "error paths" begin
        @test_throws ArgumentError combine_forecasts([f1])             # need >= 2
        @test_throws ArgumentError combine_forecasts(Forecast[])
        @test_throws DimensionMismatch combine_forecasts([f1, forecast(
            fit_sarima(train, (1, 1, 1), (0, 1, 1, 12)), 6)])          # horizon mismatch
        @test_throws DimensionMismatch combine_forecasts([f1, f2]; weights=[1, 2, 3])
        @test_throws ArgumentError combine_forecasts([f1, f2]; weights=[1, -1])
        @test_throws ArgumentError combine_forecasts([f1, f2]; weights=[0, 0])
        # levels must match: combining intervals quoted at different levels
        # is not a meaningful operation
        f90 = forecast(fit_sarima(train, (1, 1, 1), (0, 1, 1, 12)), 12; level=[90.0])
        @test_throws ArgumentError combine_forecasts([f1, f90])
    end
end
