using DelimitedFiles
using Statistics: mean

# Stage 9B Tier 2.3. Before this, every prediction interval the package
# emitted was unfalsifiable -- `accuracy` scored point forecasts only.
#
# VERIFICATION STANDARD IS DIFFERENT HERE, and worth stating. Neither R's
# `scoringRules` nor Python's `properscoring` is reachable in this
# environment (checked, not assumed), and R's `forecast` has no Winkler,
# pinball or CRPS function. So there is no package output to match.
#
# Instead each rule is checked against its own definition:
#   * crps_normal's closed form vs NUMERICAL INTEGRATION of
#     CRPS(F,y) = int (F(x) - 1{x>=y})^2 dx, at seven (y, mu, sigma)
#     combinations, via scipy.integrate.quad. Agreement to 5e-13, which is
#     _chisq_ccdf's own tolerance rather than any limit of the formula.
#   * pinball_loss against the median identity (tau=0.5 is half the MAE).
#   * winkler_score against hand computation.
#   * crps_ensemble against crps_normal, which it must converge to.
#   * crps_normal against |y - mu| as the distribution degenerates.
# That is arguably a stronger check than reproducing another
# implementation, but it is a different one.

@testset "distributional forecast accuracy (Stage 9B Tier 2.3)" begin

    @testset "crps_normal matches the numerically integrated definition" begin
        # scipy.integrate.quad on (F(x) - 1{x>=y})^2, executed this session
        REF = [(0.0, 0.0, 1.0, 0.233694977255),
               (1.0, 0.0, 1.0, 0.602441357628),
               (-2.5, 0.0, 1.0, 1.939818690811),
               (3.0, 2.0, 0.5, 0.726395910843),
               (10.0, 7.0, 2.5, 1.870038294716),
               (0.5, 0.5, 0.1, 0.023369497726),
               (-5.0, 1.0, 3.0, 4.358375465058)]
        for (y, mu, s, r) in REF
            @test isapprox(crps_normal([y], mu, s), r; atol=1e-10)
        end
    end

    @testset "crps_normal properties" begin
        # scales linearly in sigma at a fixed standardised residual
        @test isapprox(crps_normal([0.0], 0.0, 2.0), 2 * crps_normal([0.0], 0.0, 1.0); rtol=1e-10)
        # symmetric in the residual
        @test isapprox(crps_normal([1.5], 0.0, 1.0), crps_normal([-1.5], 0.0, 1.0); rtol=1e-10)
        # minimised at the mean, and increasing away from it
        c = [crps_normal([y], 0.0, 1.0) for y in (0.0, 0.5, 1.0, 2.0, 4.0)]
        @test issorted(c)
        @test c[1] == minimum(c)
        # strictly positive even on a perfect point forecast -- the
        # distribution still has spread, and CRPS charges for it
        @test crps_normal([0.0], 0.0, 1.0) > 0
        # reduces to the absolute error as the distribution collapses
        @test isapprox(crps_normal([3.0], 2.5, 1e-8), 0.5; atol=1e-6)
        # vectorised mu/sigma average per-observation scores
        @test isapprox(crps_normal([0.0, 1.0], [0.0, 0.0], [1.0, 1.0]),
                        (0.233694977255 + 0.602441357628) / 2; atol=1e-10)
        @test isapprox(crps_normal([0.0, 1.0], 0.0, 1.0),
                        crps_normal([0.0, 1.0], [0.0, 0.0], [1.0, 1.0]); rtol=1e-12)
    end

    @testset "pinball_loss at the median is half the MAE" begin
        y = [3.0, -0.5, 2.0, 7.0, 2.0]
        q = [2.5, 0.0, 2.0, 8.0, 1.25]
        @test isapprox(pinball_loss(y, q, 0.5), 0.5 * mae(y, q); rtol=1e-12)
    end

    @testset "pinball_loss has the asymmetry the right way round" begin
        # at tau = 0.9 under-forecasting (y above q) must cost MORE
        under = pinball_loss([10.0], [8.0], 0.9)    # y > q, weight tau
        over  = pinball_loss([8.0], [10.0], 0.9)    # y < q, weight 1-tau
        @test under > over
        @test isapprox(under, 0.9 * 2; rtol=1e-12)
        @test isapprox(over, 0.1 * 2; rtol=1e-12)
        # and the reverse at tau = 0.1
        @test pinball_loss([10.0], [8.0], 0.1) < pinball_loss([8.0], [10.0], 0.1)
        # zero only on a perfect forecast
        @test pinball_loss([1.0, 2.0], [1.0, 2.0], 0.3) == 0.0
    end

    @testset "pinball_loss is minimised at the true quantile" begin
        # the defining property: the expected loss at level tau is minimised
        # by the tau quantile of the predictive distribution
        x = sort(randn(MersenneTwister(21), 20_000))
        for tau in (0.1, 0.5, 0.9)
            qstar = quantile(x, tau)
            best = pinball_loss(x, fill(qstar, length(x)), tau)
            for delta in (-0.3, -0.1, 0.1, 0.3)
                @test pinball_loss(x, fill(qstar + delta, length(x)), tau) >= best
            end
        end
    end

    @testset "winkler_score matches hand computation" begin
        @test winkler_score([1.0], [0.0], [2.0], 95.0) == 2.0        # inside: the width
        # outside by 1 at alpha = 0.05: width 2 + (2/0.05)*1 = 42
        @test isapprox(winkler_score([3.0], [0.0], [2.0], 95.0), 42.0; atol=1e-6)
        # outside by 1 at alpha = 0.20: width 2 + (2/0.2)*1 = 12
        @test isapprox(winkler_score([3.0], [0.0], [2.0], 80.0), 12.0; atol=1e-6)
        # symmetric in which side was missed
        @test isapprox(winkler_score([3.0], [0.0], [2.0], 95.0),
                        winkler_score([-1.0], [0.0], [2.0], 95.0); rtol=1e-12)
        # boundary counts as inside
        @test winkler_score([2.0], [0.0], [2.0], 95.0) == 2.0
        @test winkler_score([0.0], [0.0], [2.0], 95.0) == 2.0
    end

    @testset "winkler_score punishes a narrow interval that misses" begin
        # the failure mode coverage alone cannot see: shrinking an interval
        # improves nothing once the actual falls outside it
        y = [5.0]
        wide   = winkler_score(y, [0.0], [10.0], 95.0)   # covers, width 10
        narrow = winkler_score(y, [0.0], [1.0], 95.0)    # misses by 4, width 1
        @test narrow > wide
        # and a needlessly wide interval is still penalised for its width
        @test winkler_score(y, [-100.0], [100.0], 95.0) > wide
    end

    @testset "interval_coverage" begin
        @test interval_coverage([1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]) == 1.0
        @test interval_coverage([1.0, 9.0], [0.0, 0.0], [5.0, 5.0]) == 0.5
        @test interval_coverage([9.0], [0.0], [5.0]) == 0.0
        @test interval_coverage([5.0], [0.0], [5.0]) == 1.0      # boundary is covered
        @test interval_coverage([0.0], [0.0], [5.0]) == 1.0
        # perfect coverage from a useless interval -- why it is never reported alone
        @test interval_coverage([1.0, 2.0], [-1e9, -1e9], [1e9, 1e9]) == 1.0
    end

    @testset "crps_ensemble converges to crps_normal" begin
        analytic = crps_normal([2.3], 2.0, 0.5)
        rng = MersenneTwister(31)
        coarse = crps_ensemble([2.3], reshape(2.0 .+ 0.5 .* randn(rng, 500), :, 1))
        fine   = crps_ensemble([2.3], reshape(2.0 .+ 0.5 .* randn(rng, 20_000), :, 1))
        @test abs(fine - analytic) < abs(coarse - analytic)
        @test isapprox(fine, analytic; atol=0.01)
    end

    @testset "crps_ensemble properties" begin
        # a degenerate ensemble reduces to the absolute error
        @test isapprox(crps_ensemble([3.0], fill(2.5, 500, 1)), 0.5; atol=1e-10)
        # zero on a perfect degenerate forecast
        @test isapprox(crps_ensemble([2.0], fill(2.0, 100, 1)), 0.0; atol=1e-12)
        # one column per horizon, averaged
        P = hcat(fill(1.0, 50), fill(5.0, 50))
        @test isapprox(crps_ensemble([1.0, 6.0], P), (0.0 + 1.0) / 2; atol=1e-10)
    end

    @testset "interval_accuracy scores a real forecast" begin
        y = dataset("cardox").value
        train, test = y[1:240], y[241:252]
        f = forecast(fit_sarima(train, (1, 1, 1), (0, 1, 1, 12)), 12)
        ia = interval_accuracy(test, f)

        @test ia.level == [80.0, 95.0]
        @test length(ia.coverage) == 2 && length(ia.winkler) == 2
        @test all(0 .<= ia.coverage .<= 1)
        @test ia.coverage[2] >= ia.coverage[1]      # 95% covers at least as much as 80%
        @test ia.winkler[2] >= ia.winkler[1]        # and is wider, so scores higher
        @test ia.crps > 0
        # CRPS sits below the RMSE and near the MAE in magnitude, as it should
        @test ia.crps < rmse(test, f.point)

        # coverage lands near nominal on this well-specified fit
        @test 0.6 <= ia.coverage[1] <= 1.0
        @test 0.7 <= ia.coverage[2] <= 1.0
    end

    @testset "a bad forecast scores worse on every distributional measure" begin
        y = dataset("cardox").value
        train, test = y[1:240], y[241:252]
        good = interval_accuracy(test, forecast(fit_sarima(train, (1,1,1), (0,1,1,12)), 12))
        bad  = interval_accuracy(test, seasonal_naive(train, 12, 12))
        @test bad.crps > good.crps
        @test bad.winkler[1] > good.winkler[1]
        @test bad.winkler[2] > good.winkler[2]

        # and here is why coverage is never reported alone: the benchmark's
        # 95% interval covers MORE than the good model's, by being far wider,
        # while scoring worse on Winkler. Coverage alone would rank it first.
        @test bad.coverage[2] >= good.coverage[2]
        @test bad.winkler[2] > good.winkler[2]
    end

    @testset "error paths" begin
        @test_throws DimensionMismatch pinball_loss([1.0, 2.0], [1.0], 0.5)
        @test_throws ArgumentError pinball_loss(Float64[], Float64[], 0.5)
        @test_throws ArgumentError pinball_loss([1.0], [1.0], 0.0)
        @test_throws ArgumentError pinball_loss([1.0], [1.0], 1.0)
        @test_throws ArgumentError pinball_loss([1.0], [1.0], 1.5)

        @test_throws ArgumentError crps_normal([1.0], 0.0, 0.0)      # sigma must be > 0
        @test_throws ArgumentError crps_normal([1.0], 0.0, -1.0)
        @test_throws ArgumentError crps_normal(Float64[], 0.0, 1.0)
        @test_throws DimensionMismatch crps_normal([1.0, 2.0], [0.0], 1.0)

        @test_throws DimensionMismatch winkler_score([1.0, 2.0], [0.0], [2.0], 95.0)
        @test_throws ArgumentError winkler_score([1.0], [0.0], [2.0], 0.0)
        @test_throws ArgumentError winkler_score([1.0], [0.0], [2.0], 100.0)
        @test_throws ArgumentError winkler_score([1.0], [2.0], [0.0], 95.0)   # lower > upper

        @test_throws DimensionMismatch interval_coverage([1.0, 2.0], [0.0], [2.0])
        @test_throws ArgumentError crps_ensemble([1.0], fill(1.0, 1, 1))      # < 2 draws
        @test_throws DimensionMismatch crps_ensemble([1.0, 2.0], fill(1.0, 10, 1))
    end

    @testset "container-agnostic" begin
        y = (3.0, -0.5, 2.0, 7.0)
        q = (2.5, 0.0, 2.0, 8.0)
        @test isapprox(pinball_loss(y, q, 0.5), pinball_loss(collect(y), collect(q), 0.5); rtol=1e-12)
        @test isapprox(crps_normal(y, 0.0, 1.0), crps_normal(collect(y), 0.0, 1.0); rtol=1e-12)
    end
end
