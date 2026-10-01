using DelimitedFiles

# Stage 9B Tier 2.4. Ground truth from real R 4.6.0 `ar()` and `spec.ar()`,
# executed this session on a 400-point AR(2) with phi = (0.75, -0.4) --
# see verification/arspectral/arspectral.R. Every reported quantity matches:
# the AIC-selected order, the coefficients, var.pred, the whole AIC sweep,
# the partial autocorrelations, and the spectrum to 5e-11.

@testset "ar_yw and spec_ar (Stage 9B Tier 2.4)" begin
    z = vec(readdlm(joinpath(@__DIR__, "verification", "arspectral", "ar2.csv")))

    @testset "ar_yw matches R's ar() with AIC order selection" begin
        f = ar_yw(z)
        @test f.n == 400
        @test f.order == 2                       # R: 2, and the truth is 2
        @test f.order_max == 26                  # R's floor(10*log10(400))
        @test isapprox(f.ar, [0.7302353258, -0.5094353084]; atol=1e-9)
        @test isapprox(f.var_pred, 1.0126696887; atol=1e-9)
        @test isapprox(f.aic_by_order[1:6],
            [222.83698591, 118.18500267, 0.0, 1.88608591, 3.43709657, 5.01333381];
            atol=1e-7)
        @test isapprox(f.partialacf[1:4],
            [0.4837804719, -0.5094353084, 0.0168743793, 0.0334939339]; atol=1e-9)
    end

    @testset "the AIC vector is shifted so its minimum is zero at the chosen order" begin
        f = ar_yw(z)
        @test minimum(f.aic_by_order) == 0.0
        @test argmin(f.aic_by_order) - 1 == f.order
        @test length(f.aic_by_order) == f.order_max + 1
        @test all(>=(0), f.aic_by_order)
    end

    @testset "a fixed order matches R's ar(aic=FALSE)" begin
        f = ar_yw(z; order=4, order_max=4)
        @test f.order == 4
        @test isapprox(f.ar,
            [0.7382665411, -0.5042818625, -0.0078720018, 0.0334939339]; atol=1e-9)
        @test isapprox(f.var_pred, 1.0163658338; atol=1e-9)
    end

    @testset "the last coefficient at order k is the lag-k partial autocorrelation" begin
        # a structural identity of the Durbin-Levinson recursion, and the same
        # one that makes R's ar.burg a reference for pacf(method=:burg)
        for k in 1:6
            fk = ar_yw(z; order=k, order_max=k)
            @test isapprox(fk.ar[end], fk.partialacf[k]; atol=1e-12)
        end
        @test isapprox([ar_yw(z; order=k, order_max=k).ar[end] for k in 1:6],
                        pacf(z, 1:6; method=:ywm).values; atol=1e-9)
    end

    @testset "Yule-Walker always returns a stationary fit" begin
        # the reason it is the natural estimator for a spectral density: a
        # non-stationary AR has no spectrum. True even on a random walk,
        # where the estimate is badly biased but still inside the unit circle.
        for s in (cumsum(randn(MersenneTwister(8), 300)),
                   randn(MersenneTwister(9), 300),
                   z)
            f = ar_yw(s)
            f.order == 0 && continue
            # stationarity via the companion matrix's spectral radius
            p = f.order
            C = zeros(p, p)
            C[1, :] = f.ar
            p > 1 && (C[2:end, 1:end-1] = Matrix(1.0I, p - 1, p - 1))
            @test maximum(abs.(eigvals(C))) < 1.0
        end
    end

    @testset "var_pred carries R's small-sample adjustment" begin
        # R: var.pred <- EA * n.obs/(n.obs - (m + 1))
        f = ar_yw(z; order=2, order_max=2)
        acov = TSAnalytics._acovf(z .- sum(z) / length(z), 2; demean=false, adjusted=false)
        dl = TSAnalytics._durbin_levinson_full(acov, 2)
        raw = dl.v[3]
        @test isapprox(f.var_pred, raw * 400 / (400 - 3); rtol=1e-10)
        @test f.var_pred > raw          # the adjustment inflates
    end

    @testset "spec_ar matches R's spec.ar" begin
        s = spec_ar(z; n_freq=9)
        @test s.kind == :spec_ar
        @test isapprox(s.freq, collect(range(0.0, 0.5; length=9)); atol=1e-12)
        @test isapprox(s.spec,
            [1.6678997458, 2.1250518203, 4.3283698744, 4.4279897990, 1.3085323160,
             0.5285503717, 0.3021474102, 0.2225697683, 0.2018828836]; atol=1e-8)
    end

    @testset "spec_ar's peak is where the theory puts it, at the FITTED coefficients" begin
        # A pseudo-cyclical AR(2) peaks at cos(2*pi*f) = phi1(1-phi2)/(-4*phi2),
        # the textbook formula. Evaluate it at the FITTED phi, not the true
        # phi: this testset is about the spectrum formula, and Yule-Walker is
        # biased at n=400 (fitted phi2 = -0.509 against a true -0.4), so the
        # true-phi peak sits at 0.136 while the fitted spectrum peaks at 0.159.
        # Asserting against the true value would be testing the estimator's
        # bias under the guise of testing the spectrum.
        s = spec_ar(z; n_freq=2001)
        f_peak = s.freq[argmax(s.spec)]
        p1, p2 = ar_yw(z).ar
        theory_fitted = acos(p1 * (1 - p2) / (-4 * p2)) / (2pi)
        @test isapprox(f_peak, theory_fitted; atol=1e-3)

        # separately, and loosely: the peak is still in the right
        # neighbourhood of the generating process, which is what says the
        # whole pipeline recovers real structure rather than noise
        theory_true = acos(0.75 * (1 - (-0.4)) / (-4 * (-0.4))) / (2pi)
        @test abs(f_peak - theory_true) < 0.05
    end

    @testset "df and bandwidth are NaN, because neither applies" begin
        # the smoothness comes from the AR assumption, not a Daniell kernel.
        # R's spec.ar does not return them at all.
        s = spec_ar(z; n_freq=16)
        @test isnan(s.df)
        @test isnan(s.bandwidth)
    end

    @testset "frequency scales both axes, as R's xfreq does" begin
        a = spec_ar(z; n_freq=9)
        b = spec_ar(z; n_freq=9, frequency=12)
        @test isapprox(b.freq, a.freq .* 12; atol=1e-12)
        @test isapprox(b.spec, a.spec ./ 12; rtol=1e-10)
    end

    @testset "order 0 gives a flat spectrum" begin
        # white noise selects order 0 often enough to be worth pinning; when
        # it does, the spectrum must be constant rather than erroring
        wn = randn(MersenneTwister(12), 500)
        f = ar_yw(wn; order=0, order_max=1)
        @test f.order == 0
        @test isempty(f.ar)
        s = spec_ar(wn; order=0, order_max=1, n_freq=16)
        @test all(≈(s.spec[1]), s.spec)
        @test s.spec[1] > 0
    end

    @testset "it is smooth where the raw periodogram is not" begin
        # the point of a parametric spectrum: no span to choose, and the
        # result has far less point-to-point variation than the periodogram
        s = spec_ar(z; n_freq=200)
        pg = periodogram(z)
        rough(v) = sum(abs.(diff(log.(v)))) / (length(v) - 1)
        @test rough(s.spec) < rough(pg.spec) / 5
        # and both put the peak in the same place
        @test abs(s.freq[argmax(s.spec)] - pg.freq[argmax(pg.spec)]) < 0.03
    end

    @testset "error paths" begin
        @test_throws ArgumentError ar_yw([1.0])
        @test_throws ArgumentError ar_yw(z; order_max=0)
        @test_throws ArgumentError ar_yw(z; order_max=400)
        @test_throws ArgumentError ar_yw(z; order=-1)
        @test_throws ArgumentError ar_yw(z; order=5, order_max=3)
        @test_throws ArgumentError ar_yw(fill(1.0, 100))          # zero variance
        @test_throws ArgumentError ar_yw([1.0, NaN, 3.0])
        @test_throws ArgumentError spec_ar(z; n_freq=1)
        @test_throws ArgumentError spec_ar(z; frequency=0)
    end

    @testset "show prints the order and how it was chosen" begin
        out = sprint(show, ar_yw(z))
        @test occursin("AR(2)", out)
        @test occursin("Yule-Walker", out)
        @test occursin("by AIC", out)
    end

    @testset "container-agnostic" begin
        @test isapprox(ar_yw(Tuple(z)).ar, ar_yw(z).ar; atol=1e-12)
        @test isapprox(spec_ar(Tuple(z); n_freq=9).spec, spec_ar(z; n_freq=9).spec; atol=1e-12)
    end
end
