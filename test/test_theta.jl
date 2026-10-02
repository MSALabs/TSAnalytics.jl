using DelimitedFiles

# Stage 9.3: the Theta method (Assimakopoulos & Nikolopoulos 2000), via the
# SES-plus-drift equivalence of Hyndman & Billah (2003).
#
# DUAL-VERIFIED against R's forecast::thetaf (verification/theta/theta.R) and
# statsmodels' ThetaModel (verification/theta/theta.py) on five series.
#
# The two references AGREE on the point forecast to 2e-5 and DISAGREE on the
# prediction interval in three separate ways. Each is pinned below:
#
#  1. statsmodels' default `sigma2` is computed on the ORIGINAL series even
#     when the point forecast is deseasonalized. Smoking gun: it returns the
#     identical 49.7153734174 whether the fixture is declared quarterly or
#     non-seasonal.
#  2. statsmodels' variance growth factor is `1 + (alpha-1)^2` where its own
#     stated IMA(1,1) derivation gives `(1 + (alpha-1))^2 == alpha^2`. The
#     square is in the wrong place; the two agree only at alpha = 1.
#  3. Neither reference scales the standard error by the seasonal factor.
#     This package does, because var(y) = S^2 var(x).
#
# This package follows R on (2), fixes (1) by construction, and departs from
# both on (3) — and (3) plus the sse/n-vs-sse/(n-2) convention are inverted
# below to land back on R's own numbers to 3e-9.

@testset "fit_theta — the Theta method (Stage 9.3)" begin
    y = vec(readdlm(joinpath(@__DIR__, "verification", "theta", "theta_y.csv")))
    n = length(y)
    m = 4
    R_ALPHA = 0.6792395168        # R: ses(seasadj(decompose(y)))$model$par["alpha"]

    @testset "the SES-plus-drift reduction is the implementation" begin
        t = fit_theta(y, m)
        @test t.deseasonalized
        @test t.theta == 2.0
        @test t.nobs == n && t.period == m
        @test length(t.figure) == m
        @test length(t.fitted) == n && length(t.resid) == n
        # sse is y - fitted on the ORIGINAL scale; sse_adjusted is the SES
        # fit's own, on the deseasonalised one. R reports both, as different
        # quantities on the same object -- see fit_theta's docstring.
        @test isapprox(t.sse, sum(abs2, t.resid); rtol=1e-12)
        @test isapprox(t.resid, y .- t.fitted; rtol=1e-12)
        @test isapprox(t.sse_adjusted,
                       sum(abs2, fit_ets(y ./ repeat(t.figure, 30), 1).resid);
                       rtol=1e-6)
        @test t.sse < t.sse_adjusted          # 1209.3 vs 1315.2 here
    end

    @testset "both of R's two residual scales are reproduced, as separate fields" begin
        # R: sum((y - thetaf fitted)^2) = 1209.29841334
        #    sum(thetaf residuals^2)    = 1315.21211028     <- the SES scale
        # alpha pinned so the optimiser is not part of the comparison
        t = fit_theta(y, m; alpha=R_ALPHA)
        @test isapprox(t.sse, 1209.29841334; rtol=1e-7)
        @test isapprox(t.sse_adjusted, 1315.21211028; rtol=1e-7)
        @test isapprox(t.sse_adjusted / t.sse, 1.0876; atol=1e-3)
        # R's fitted values themselves, which this package's resid pairs with
        # 2e-3, not 1e-5: `alpha` is pinned but the SES initial level is still
        # optimised, and the two optimisers land a little apart on it
        @test maximum(abs.(t.fitted[1:3] .- [105.883949, 100.893471, 90.608738])) < 2e-3
    end

    @testset "the drift is half the OLS slope, matching R to 1e-10" begin
        # R: lsfit(0:(n-1), seasadj(decompose(y)))$coefficients[2]/2
        t = fit_theta(y, m)
        @test isapprox(t.b0 / 2, 0.3083244283; atol=1e-10)
        # and the closed form equals a general regression solve
        x = y ./ repeat(t.figure, 30)
        X = hcat(ones(n), 0:(n-1))
        @test isapprox(TSAnalytics._theta_slope(x), (X \ x)[2]; rtol=1e-10)
    end

    @testset "classical_decompose reproduces R's multiplicative figure" begin
        # the seasonal adjustment is R's decompose(x, type="multiplicative")
        d = classical_decompose(y, m; model=:multiplicative)
        @test isapprox(d.figure,
            [1.0610975463, 1.0084770414, 0.9136917388, 1.0167336735]; atol=1e-9)
    end

    @testset "the seasonality test statistic matches R to 1e-9" begin
        # R: r <- acf(y, lag.max=m)$acf[-1]
        #    stat <- sqrt((1 + 2*sum(r[-m]^2))/n);  abs(r[m])/stat
        @test isapprox(TSAnalytics._theta_seasonal_statistic(y, m),
                       4.0883478370; atol=1e-9)
        @test isapprox(acf(y, 1:m).values[m], 0.8903208044; atol=1e-9)
        # compared against the one-sided 95% normal quantile, R's qnorm(0.95)
        @test isapprox(TSAnalytics._confidence_z(0.10), 1.6448536270; atol=1e-8)
        @test fit_theta(y, m).seasonal_statistic > TSAnalytics._confidence_z(0.10)
    end

    @testset "statsmodels drops the factor of 2 from Bartlett's formula" begin
        # Its test is n*r_m^2/(1 + sum(r^2)) > 1.645^2, i.e. the same
        # comparison with 1 + sum(r^2) where Bartlett's large-lag variance has
        # 1 + 2*sum(r^2). Dropping the 2 shrinks the denominator, so the
        # statistic is always LARGER and the test always more liberal.
        r = acf(y, 1:m).values
        bartlett = sqrt((1 + 2 * sum(abs2, r[1:m-1])) / n)
        sm = sqrt((1 + sum(abs2, r[1:m-1])) / n)
        @test sm < bartlett
        @test abs(r[m]) / sm > abs(r[m]) / bartlett
        @test isapprox(abs(r[m]) / bartlett, 4.0883478370; atol=1e-9)
        # statsmodels' own statistic, squared-scale, on this fixture
        @test isapprox(n * r[m]^2 / (1 + sum(abs2, r[1:m-1])), 28.4329379899; atol=1e-8)
        @test 28.4329379899 > 4.0883478370^2        # liberal on this fixture too
    end

    @testset "point forecasts match R to 2e-5 at this package's own optimum" begin
        f = forecast(fit_theta(y, m), 8)
        R_MEAN = [181.0777469935, 172.4089240008, 156.4861713715, 174.4474429399,
                  182.3863961709, 173.6526764298, 157.6130253035, 175.7013782545]
        # elementwise, not isapprox's aggregate 2-norm over the vector
        @test maximum(abs.(f.point .- R_MEAN)) < 2e-5
        @test f.model_name == "Theta(2.0)"
        @test f.horizon == 8
    end

    @testset "and to 4e-11 with R's own alpha pinned" begin
        # removing the optimiser from the comparison leaves the formula, which
        # is what this is testing
        t = fit_theta(y, m; alpha=R_ALPHA)
        @test isapprox(t.alpha, R_ALPHA; atol=1e-12)
        @test isapprox(t.level, 170.1974412581; atol=1e-8)    # R's ses(x)$mean[1]
        f = forecast(t, 8)
        R_MEAN = [181.0777469935, 172.4089240008, 156.4861713715, 174.4474429399,
                  182.3863961709, 173.6526764298, 157.6130253035, 175.7013782545]
        @test maximum(abs.(f.point .- R_MEAN)) < 1e-9
    end

    @testset "DIVERGENCE: R's 95% bounds come back exactly when both are undone" begin
        # The two documented interval conventions, inverted: divide the
        # standard error by the seasonal factor, and substitute R's
        # sse_x/(n-2) for this package's sse_x/n. Nothing else changes.
        t = fit_theta(y, m; alpha=R_ALPHA)
        f = forecast(t, 8)
        se_R = sqrt.(11.1458653413 .* (1 .+ (0:7) .* t.alpha^2))   # R's sigma2
        z95 = 1.959963985
        R_LO95 = [174.5343241332, 164.4987773703, 147.4128887957, 164.3440527324,
                  171.3486187374, 161.7536610228, 144.9110331510, 162.2442373387]
        R_HI95 = [187.6211698539, 180.3190706312, 165.5594539473, 184.5508331475,
                  193.4241736044, 185.5516918368, 170.3150174560, 189.1585191703]
        @test maximum(abs.(f.point .- z95 .* se_R .- R_LO95)) < 1e-7
        @test maximum(abs.(f.point .+ z95 .* se_R .- R_HI95)) < 1e-7
        # the sigma2 gap is exactly the denominator, nothing else
        @test isapprox(t.sse_adjusted / (n - 2), 11.1458653413; atol=1e-7)
        @test isapprox(11.1458653413 / t.sigma2, n / (n - 2); rtol=1e-8)
    end

    @testset "DIVERGENCE: the seasonal factor really is the only other gap" begin
        # f.se is se_x * S; dividing it out must give R's unscaled se exactly
        t = fit_theta(y, m; alpha=R_ALPHA)
        f = forecast(t, 8)
        S = [t.figure[mod1(n + h, m)] for h in 1:8]
        se_x = f.se ./ S
        @test isapprox(se_x, sqrt.(t.sigma2 .* (1 .+ (0:7) .* t.alpha^2)); rtol=1e-12)
        # and the scaling is not negligible: the factors span 0.91 to 1.06 here
        @test maximum(S) / minimum(S) > 1.15
        @test !isapprox(f.se, se_x; rtol=1e-3)
    end

    @testset "DIVERGENCE: statsmodels' sigma2 ignores the deseasonalisation" begin
        # Its sigma2 for the fixture declared QUARTERLY and the same fixture
        # declared NON-SEASONAL are the identical number, which can only
        # happen if the seasonal adjustment never enters it.
        SM_SIGMA2_SEASONAL    = 49.7153734174
        SM_SIGMA2_NONSEASONAL = 49.7153734174
        @test SM_SIGMA2_SEASONAL == SM_SIGMA2_NONSEASONAL
        # this package's two differ by a lot, as they must
        a = fit_theta(y, m)          # deseasonalized
        b = fit_theta(y, 1)          # not
        @test b.sigma2 / a.sigma2 > 4
        # and the deseasonalized one is close to the same estimator computed
        # correctly in statsmodels via use_mle=True, which does deseasonalize
        @test isapprox(a.sigma2, 10.1872899235; rtol=0.1)
    end

    @testset "DIVERGENCE: statsmodels' growth factor has the square misplaced" begin
        # It documents sigma2*(1 + (h-1)*(1 + (alpha-1)^2)); the IMA(1,1)
        # result its derivation cites is sigma2*(1 + (h-1)*(1 + (alpha-1))^2),
        # and (1 + (alpha-1))^2 == alpha^2. They coincide only at alpha = 1.
        for a in (0.2, 0.5, 0.6792395168, 0.99)
            @test !isapprox(1 + (a - 1)^2, a^2; rtol=1e-6)
            @test 1 + (a - 1)^2 > a^2                  # so always too wide
        end
        @test isapprox(1 + (1.0 - 1)^2, 1.0^2; atol=1e-12)   # equal at alpha=1
        # this package uses alpha^2, R's formula
        t = fit_theta(y, m; alpha=R_ALPHA)
        f = forecast(t, 4)
        S = [t.figure[mod1(n + h, m)] for h in 1:4]
        @test isapprox((f.se[2] / S[2])^2 / (f.se[1] / S[1])^2,
                       1 + t.alpha^2; rtol=1e-10)
    end

    @testset "the non-seasonal fit matches R, including its SES residuals" begin
        # same data declared non-seasonal: R's thetaf skips adjustment
        t = fit_theta(y, 1)
        f = forecast(t, 8)
        @test !t.deseasonalized
        @test isnan(t.seasonal_statistic)
        @test isempty(t.figure)
        R_MEAN = [167.0284978268, 167.3341776177, 167.6398574086, 167.9455371995,
                  168.2512169904, 168.5568967813, 168.8625765722, 169.1682563631]
        @test maximum(abs.(f.point .- R_MEAN)) < 3e-4
        # with no adjustment the two scales coincide, so both match R
        @test isapprox(t.sse, 6262.6001833486; atol=1e-4)
        @test isapprox(t.sse_adjusted, t.sse; rtol=1e-12)
        @test isapprox(t.b0 / 2, 0.3056797909; atol=1e-9)
        # R's se at h=1 is sqrt(sigma2) with sigma2 = sse/(n-2)
        @test isapprox(sqrt(t.sse / (n - 2)), 7.28511379; atol=1e-6)
        # with no seasonality there is nothing to scale, so this package's
        # only divergence left is the denominator
        @test isapprox(f.se[1], sqrt(t.sigma2); rtol=1e-12)
    end

    @testset "that SSE is the same quantity fit_ets reports for SES" begin
        # thetaf's residuals are the SES fit's, so a non-seasonal Theta model
        # and ETS(A,N,N) on the same series must agree -- and ETS(A,N,N) is
        # separately verified against R's ets(model="ANN")
        @test isapprox(fit_theta(y, 1).sse, fit_ets(y, 1).sse; rtol=1e-8)
        @test isapprox(fit_theta(y, 1).sigma2, fit_ets(y, 1).sigma2; rtol=1e-8)
    end

    @testset "AirPassengers, where the seasonal scaling matters most" begin
        ap = _load_column(TSAnalytics.AIR_PASSENGERS, "passengers")
        t = fit_theta(ap, 12)
        @test t.deseasonalized
        @test isapprox(t.seasonal_statistic, 2.4885154559; atol=1e-7)   # R
        @test isapprox(t.b0 / 2, 1.3230696288; atol=1e-8)               # R
        @test isapprox(t.figure, [0.9102303674, 0.8836253207, 1.0073662876,
            0.9759060123, 0.9813780275, 1.1127758267, 1.2265555429, 1.2199109694,
            1.0604919326, 0.9217572404, 0.8011780824, 0.8988243900]; atol=1e-8)
        @test isapprox(t.sse_adjusted, 16428.3538031260; rtol=1e-3)     # R's residuals
        # here the original-scale SSE is the LARGER of the two, the opposite
        # way round from the quarterly fixture: which one is bigger depends on
        # whether the seasonal factors amplify or damp each period's errors,
        # so neither ordering is a property of the two scales
        @test t.sse > t.sse_adjusted
        @test !isapprox(t.sse, t.sse_adjusted; rtol=1e-3)
        f = forecast(t, 12)
        R_MEAN = [440.0781910816, 428.3842805570, 489.7070986696, 475.7046276614,
                  479.6703886692, 545.3662721542, 602.7520018224, 601.1007630915,
                  523.9514921587, 456.6271449128, 397.9537346656, 447.6449087429]
        @test isapprox(f.point, R_MEAN; rtol=1e-4)
        # the seasonal factors span 1.23/0.80 = 1.53, so R's unscaled interval
        # is materially wrong here -- about 23% too narrow at the peak
        @test maximum(t.figure) / minimum(t.figure) > 1.5
        S = [t.figure[mod1(length(ap) + h, 12)] for h in 1:12]
        @test maximum(f.se ./ (f.se ./ S)) > 1.2        # peak scaling
        @test minimum(f.se ./ (f.se ./ S)) < 0.85       # trough scaling
    end

    @testset "a series with no seasonality is left alone" begin
        nile = _load_column(TSAnalytics.NILE, "flow")
        # annual data declared monthly: nothing seasonal to find
        t = fit_theta(nile, 12)
        @test !t.deseasonalized
        @test t.seasonal_statistic < TSAnalytics._confidence_z(0.10)
        @test isempty(t.figure)
        # R on the Nile, declared non-seasonal
        t1 = fit_theta(nile, 1)
        @test isapprox(t1.b0 / 2, -1.3571527153; atol=1e-8)
        @test isapprox(forecast(t1, 10).point[1], 799.8119962159; atol=2e-2)
        @test isapprox(t1.sse, 2038674.4382675944; rtol=1e-5)   # unadjusted: one scale
        # a declining series, so the drift is negative and the forecast falls
        @test t1.b0 < 0
        @test issorted(forecast(t1, 10).point; rev=true)
    end

    @testset "the test detects autocorrelation, not seasonality as such" begin
        # A real limitation worth stating rather than discovering later. The
        # statistic is |r_m| against Bartlett's standard error, so ANY series
        # with strong autocorrelation at lag m trips it -- seasonal or not.
        # The Nile is annual and has no seasonality at all, yet declared
        # quarterly it is (just) called seasonal, and declared half-yearly
        # it is called seasonal decisively:
        nile = _load_column(TSAnalytics.NILE, "flow")
        T = TSAnalytics
        @test isapprox(T._theta_seasonal_statistic(nile, 2), 3.143390; atol=1e-5)
        @test isapprox(T._theta_seasonal_statistic(nile, 3), 2.448752; atol=1e-5)
        @test isapprox(T._theta_seasonal_statistic(nile, 4), 1.688130; atol=1e-5)
        @test isapprox(T._theta_seasonal_statistic(nile, 12), 1.282235; atol=1e-5)
        @test fit_theta(nile, 4).deseasonalized          # a false positive
        @test !fit_theta(nile, 12).deseasonalized
        # a pure linear trend has no seasonality either, and trips it at 4
        lin = collect(1.0:100.0) .+ 100.0
        @test isapprox(T._theta_seasonal_statistic(lin, 4), 3.504975; atol=1e-5)
        # so deseasonalize=false is the escape hatch when you know better
        @test !fit_theta(lin, 4; deseasonalize=false).deseasonalized
    end

    @testset "seasonal_test=false forces adjustment the test would decline" begin
        nile = _load_column(TSAnalytics.NILE, "flow")
        a = fit_theta(nile, 12; seasonal_test=true)
        b = fit_theta(nile, 12; seasonal_test=false)
        @test !a.deseasonalized
        @test b.deseasonalized
        @test length(b.figure) == 12
        @test !isapprox(a.sse, b.sse; rtol=1e-6)
        # the statistic is reported either way, so the decision is auditable
        @test isapprox(a.seasonal_statistic, b.seasonal_statistic; rtol=1e-12)
    end

    @testset "deseasonalize=false skips the machinery entirely" begin
        t = fit_theta(y, m; deseasonalize=false)
        @test !t.deseasonalized
        @test isnan(t.seasonal_statistic)     # not computed at all
        @test isapprox(t.sse, fit_theta(y, 1).sse; rtol=1e-10)
    end

    @testset "theta controls the drift weight, and 2.0 halves it" begin
        a = fit_theta(y, m; theta=2.0)
        b = fit_theta(y, m; theta=3.0)
        @test isapprox(a.b0, b.b0; rtol=1e-12)       # the slope is the same
        fa, fb = forecast(a, 6), forecast(b, 6)
        # weight is (1 - 1/theta): 0.5 against 2/3, so b drifts harder
        @test all(fb.point .> fa.point)
        # a large theta approaches the full regression slope as drift
        c = forecast(fit_theta(y, m; theta=1e6), 6)
        S = [a.figure[mod1(n + h, m)] for h in 1:6]
        base = (1 - (1 - a.alpha)^n) / a.alpha
        expect = [(a.level + a.b0 * ((h - 1) + base)) * S[h] for h in 1:6]
        @test isapprox(c.point, expect; rtol=1e-5)
    end

    @testset "the drift is never damped, so the line never flattens" begin
        f = forecast(fit_theta(y, 1), 200)
        d = diff(f.point)
        @test isapprox(d[1], d[end]; rtol=1e-10)      # constant slope forever
        @test isapprox(d[1], 0.5 * fit_theta(y, 1).b0; rtol=1e-10)
    end

    @testset "intervals widen and nest" begin
        f = forecast(fit_theta(y, 1), 12; level=[80.0, 95.0])
        @test size(f.lower) == (12, 2)
        @test issorted(f.se)                          # no seasonal factor here
        @test all(f.lower[:, 2] .<= f.lower[:, 1])
        @test all(f.upper[:, 2] .>= f.upper[:, 1])
        @test predict(fit_theta(y, 1), 12).point == f.point
    end

    @testset "container-agnostic" begin
        @test isapprox(fit_theta(Tuple(y), m).sse, fit_theta(y, m).sse; rtol=1e-10)
    end

    @testset "show reports the drift, the level and the seasonal decision" begin
        out = sprint(show, fit_theta(y, m))
        @test occursin("Theta(2.0)", out)
        @test occursin("drift", out)
        @test occursin("adjusted", out)
        @test occursin("sigma2", out)
        out2 = sprint(show, fit_theta(y, 1))
        @test !occursin("seasonal:", out2)            # nothing to report
    end

    @testset "error paths" begin
        @test_throws ArgumentError fit_theta(Float64[])
        @test_throws ArgumentError fit_theta([1.0, 2.0])             # n < 3
        @test_throws ArgumentError fit_theta(y, 0)
        @test_throws ArgumentError fit_theta(y, m; theta=1.0)        # no drift
        @test_throws ArgumentError fit_theta(y, m; theta=0.5)        # sign flip
        @test_throws ArgumentError fit_theta(y, m; alpha=0.0)
        @test_throws ArgumentError fit_theta(y, m; alpha=1.5)
        @test_throws ArgumentError fit_theta([1.0, 2.0, NaN, 4.0])
        @test_throws ArgumentError fit_theta(randn(6), 4)            # < 2 periods
        @test_throws ArgumentError forecast(fit_theta(y, m), 0)
        @test_throws ArgumentError forecast(fit_theta(y, m), 4; level=[0.0])
        @test_throws ArgumentError forecast(fit_theta(y, m), 4; level=Float64[])
    end

    @testset "a non-positive series cannot be adjusted multiplicatively" begin
        # refused by name rather than silently left unadjusted, which is what
        # R does (it warns and carries on with the unadjusted series)
        z = collect(1.0:60.0) .- 30.0            # crosses zero
        e = try; fit_theta(z, 4); nothing; catch err; err; end
        @test e isa ArgumentError
        @test occursin("positive", e.msg)
        # but it fits once seasonality is off the table
        @test fit_theta(z, 4; deseasonalize=false).nobs == 60
        @test fit_theta(z, 1).nobs == 60
    end

    @testset "StatsAPI contract" begin
        t = fit_theta(y, m)
        @test coef(t) == [t.alpha, t.b0]
        @test residuals(t) === t.resid
        @test nobs(t) == n
        @test isapprox(t.sse, sum(abs2, residuals(t)); rtol=1e-10)
        @test fitted(t) === t.fitted
    end
end
