using DelimitedFiles

# Stage 9A: the linear ETS family.
#
# DUAL-VERIFIED, contrary to the handoff's premise. It recorded that
# forecast::ets() "could not be obtained -- CRAN has been unreachable" and
# scoped the stage as single-verified against Python. R's forecast 9.0.2 is
# in fact installed, so every quantity below is checked against BOTH R's
# ets() (verification/ets/ets.R, etsfc.R) and the statsmodels figures the
# handoff shipped.
#
# Three of the handoff's other premises also needed correcting, and each
# correction is pinned as a test here:
#
#  1. The log-likelihood gap between R and statsmodels is EXACTLY the
#     concentration constant n/2*(log(n) - log(2pi) - 1) = 116.976881 at
#     n = 120, statsmodels minus R. Mind the sign: the same expression
#     written n/2*(log(2pi) - log(n) + 1) is its negative.
#  2. statsmodels' damped fits sit at its phi = 0.98 BOUND with worse SSE
#     than R's interior optima, so R is the better damped reference.
#  3. The proposed reduction onto holt_winters cannot hold as written: the
#     parameterisations differ (classical vs innovations) AND holt_winters
#     scores n-m observations where fit_ets scores n. Matching both, it
#     holds to 1e-13.

@testset "fit_ets — the linear ETS family (Stage 9A)" begin
    y = vec(readdlm(joinpath(@__DIR__, "verification", "ets", "ets_y.csv")))
    m = 4
    n = length(y)

    SPECS = [(:none, :none, "ETS(A,N,N)"), (:add, :none, "ETS(A,A,N)"),
             (:none, :add,  "ETS(A,N,A)"), (:add, :add,  "ETS(A,A,A)"),
             (:damped, :none, "ETS(A,Ad,N)"), (:damped, :add, "ETS(A,Ad,A)")]

    # R: ets(ts(y, frequency=4), model=, damped=)
    R_SSE = Dict("ETS(A,N,N)" => 6262.600320, "ETS(A,A,N)" => 5022.881071,
                 "ETS(A,N,A)" => 621.610410,  "ETS(A,A,A)" => 418.728158,
                 "ETS(A,Ad,N)" => 5002.512708, "ETS(A,Ad,A)" => 412.363734)
    # statsmodels, as shipped with the handoff
    SM_SSE = Dict("ETS(A,N,N)" => 6262.60017, "ETS(A,A,N)" => 5022.88055,
                  "ETS(A,N,A)" => 617.96059,  "ETS(A,A,A)" => 418.67036,
                  "ETS(A,Ad,N)" => 5044.65441, "ETS(A,Ad,A)" => 420.31440)
    # R's length(par): free smoothing parameters plus free initial states
    R_NP = Dict("ETS(A,N,N)" => 2, "ETS(A,A,N)" => 4, "ETS(A,N,A)" => 6,
                "ETS(A,A,A)" => 8, "ETS(A,Ad,N)" => 5, "ETS(A,Ad,A)" => 9)

    @testset "all six fit, converge, and are labelled correctly" begin
        for (tr, se, lab) in SPECS
            f = fit_ets(y, m; trend=tr, seasonal=se)
            @test f.converged
            @test TSAnalytics.notation(f) == lab
            @test f.trend == tr && f.seasonal == se
            @test all(isfinite, f.fitted)
            @test length(f.fitted) == n && length(f.resid) == n
            @test isapprox(f.sse, sum(abs2, f.resid); rtol=1e-12)
        end
    end

    @testset "free parameter counts match R's length(par) exactly" begin
        # nparams additionally counts sigma2, as every other model here does
        for (tr, se, lab) in SPECS
            f = fit_ets(y, m; trend=tr, seasonal=se)
            @test f.nparams - 1 == R_NP[lab]
        end
    end

    @testset "every model matches or beats BOTH references" begin
        for (tr, se, lab) in SPECS
            f = fit_ets(y, m; trend=tr, seasonal=se)
            @test f.sse <= R_SSE[lab] * (1 + 1e-6)
            @test f.sse <= SM_SSE[lab] * (1 + 1e-6)
        end
    end

    @testset "the two undamped non-seasonal fits reproduce both to 1e-6" begin
        # where all three optimisers reach the same optimum, agreement should
        # be essentially exact rather than merely as-good
        for lab in ("ETS(A,N,N)", "ETS(A,A,N)")
            tr = lab == "ETS(A,N,N)" ? :none : :add
            f = fit_ets(y, m; trend=tr, seasonal=:none)
            @test isapprox(f.sse, R_SSE[lab]; rtol=1e-6)
            @test isapprox(f.sse, SM_SSE[lab]; rtol=1e-6)
        end
    end

    @testset "statsmodels' damped fits are at its phi bound and worse" begin
        # The handoff flagged phi = 0.98 as "worth investigating rather than
        # an error". It is statsmodels' upper bound: R finds interior optima
        # with better SSE, and so does this package.
        for lab in ("ETS(A,Ad,N)", "ETS(A,Ad,A)")
            @test R_SSE[lab] < SM_SSE[lab]            # R beats statsmodels
        end
        f = fit_ets(y, m; trend=:damped, seasonal=:add)
        @test f.sse < R_SSE["ETS(A,Ad,A)"]            # and this beats R
        @test f.phi < 0.98 - 1e-6                     # interior, not pinned
        f2 = fit_ets(y, m; trend=:damped, seasonal=:none)
        @test f2.sse < R_SSE["ETS(A,Ad,N)"]
        @test f2.phi < 0.98 - 1e-6
    end

    @testset "the R-to-statsmodels loglik gap IS the concentration constant" begin
        # R reports -(n/2)log(SSE); statsmodels (and this package) report the
        # full Gaussian. The difference is a pure constant.
        R_LL = Dict("ETS(A,N,N)" => -524.54104587, "ETS(A,A,N)" => -511.30553800)
        SM_LL = Dict("ETS(A,N,N)" => -407.564164,  "ETS(A,A,N)" => -394.328651)
        # statsmodels MINUS R, written in the form whose sign is obvious:
        # n/2*(log(2pi) - log(n) + 1) is the NEGATIVE of this quantity, and
        # quoting it alongside a positive 116.976881 is the sign slip this
        # line exists to pin down.
        const_expected = n / 2 * (log(n) - log(2pi) - 1)
        @test isapprox(const_expected, 116.976881; atol=1e-5)
        @test const_expected > 0
        @test isapprox(42 * (log(84) - log(2pi) - 1), 66.903469; atol=1e-5)
        for lab in keys(R_LL)
            @test isapprox(SM_LL[lab] - R_LL[lab], const_expected; atol=1e-4)
        end
        # and this package reports statsmodels' convention
        f = fit_ets(y, m; trend=:none, seasonal=:none)
        @test isapprox(f.loglik, SM_LL["ETS(A,N,N)"]; atol=1e-4)
        @test isapprox(f.loglik, -n / 2 * (log(2pi) + log(f.sigma2) + 1); rtol=1e-12)
    end

    @testset "AICc selects the true generating process" begin
        # the fixture was simulated from ETS(A,A,A)
        f = auto_ets(y, m)
        @test f.trend == :add
        @test f.seasonal == :add
        @test TSAnalytics.notation(f) == "ETS(A,A,A)"
    end

    @testset "the AICc ranking matches the handoff's expected order" begin
        # A,A,A < A,Ad,A < A,N,A < A,A,N < A,Ad,N < A,N,N
        fits = Dict(lab => fit_ets(y, m; trend=tr, seasonal=se) for (tr, se, lab) in SPECS)
        order = ["ETS(A,A,A)", "ETS(A,Ad,A)", "ETS(A,N,A)",
                 "ETS(A,A,N)", "ETS(A,Ad,N)", "ETS(A,N,N)"]
        aiccs = [fits[l].aicc for l in order]
        @test issorted(aiccs)
    end

    @testset "parallel and serial auto_ets select the IDENTICAL model" begin
        a = auto_ets(y, m; parallel=true)
        b = auto_ets(y, m; parallel=false)
        @test TSAnalytics.notation(a) == TSAnalytics.notation(b)
        @test isapprox(a.aicc, b.aicc; rtol=1e-10)
        @test isapprox(a.sse, b.sse; rtol=1e-10)
    end

    @testset "REDUCTION 1: ETS(A,A,A) is classical Holt-Winters" begin
        # The handoff's version of this test passes the same alpha/beta/gamma
        # to both and expects equality. It cannot hold: the parameterisations
        # differ AND holt_winters scores n-m observations where fit_ets scores
        # n. (It also omits trend=:additive, so it would have compared a
        # trended model against an untrended one.)
        #
        # Matching both -- the classical->innovations map and the window --
        # the two agree to 1e-13, and since holt_winters is independently
        # verified against R's HoltWinters this is a real check.
        T = TSAnalytics
        a, bstar, gstar = 0.4, 0.1, 0.3
        hw = holt_winters(y, m; trend=:additive, seasonal=:additive,
                           alpha=a, beta=bstar, gamma=gstar)
        l0, b0, fig = T._hw_heuristic_init(y, m, true, :additive)

        yw = y[(m+1):end]
        nw = length(yw)
        f = zeros(nw); r = zeros(nw); lv = zeros(nw); td = zeros(nw); sn = zeros(nw)
        sse = T._ets_recursion!(f, r, lv, td, sn, yw,
                                 a, a * bstar, gstar * (1 - a), 1.0,
                                 l0, b0, copy(fig), :add, :add)
        @test isapprox(sse, hw.sse; rtol=1e-10)
        @test isapprox(f, hw.fitted; rtol=1e-10)
        @test maximum(abs.(f .- hw.fitted)) < 1e-12

        # and the map matters: the same numbers do NOT agree
        f2 = zeros(nw); r2 = zeros(nw)
        sse_wrong = T._ets_recursion!(f2, r2, lv, td, sn, yw, a, bstar, gstar, 1.0,
                                       l0, b0, copy(fig), :add, :add)
        @test !isapprox(sse_wrong, hw.sse; rtol=1e-3)
    end

    @testset "REDUCTION 2: phi = 1 collapses damped onto undamped" begin
        a = fit_ets(y, m; trend=:damped, fixed=(alpha=0.3, beta=0.1, phi=1.0),
                     initial=:heuristic)
        b = fit_ets(y, m; trend=:add, fixed=(alpha=0.3, beta=0.1), initial=:heuristic)
        @test isapprox(a.sse, b.sse; atol=1e-10)
        @test isapprox(a.fitted, b.fitted; atol=1e-10)
        @test isapprox(a.loglik, b.loglik; atol=1e-10)
    end

    @testset "forecasts match R's at R's own parameters" begin
        # pinned, so this tests the forecast formula rather than the optimiser
        f = fit_ets(y, m; trend=:add, seasonal=:add,
                     fixed=(alpha=0.41340948, beta=0.07631261, gamma=0.36512365))
        fc = forecast(f, 8)
        R_PT = [178.03488322, 173.85611609, 169.15203550, 178.86778674,
                185.31186113, 181.13309400, 176.42901341, 186.14476465]
        @test isapprox(fc.point, R_PT; atol=1e-6)
        @test fc.model_name == "ETS(A,A,A)"
        @test fc.horizon == 8
    end

    @testset "intervals match R once its sigma2 convention is applied" begin
        # This package stores sigma2 = sse/n (ML, matching statsmodels and
        # fit_arima); R's ets uses sse/(n - np). Substituting R's value into
        # this package's own interval arithmetic reproduces R's bound to all
        # eight printed decimals, so the whole gap is the denominator.
        f = fit_ets(y, m; trend=:add, seasonal=:add,
                     fixed=(alpha=0.41340948, beta=0.07631261, gamma=0.36512365))
        fc = forecast(f, 8)
        z95 = TSAnalytics._confidence_z(0.05)
        lo_under_R = fc.point[1] - z95 * sqrt(3.7386442686)
        @test isapprox(lo_under_R, 174.24518033; atol=1e-7)
        # and the ratio of the two conventions is exactly sqrt(n/(n-np))
        @test isapprox(sqrt(3.7386442686 / f.sigma2), sqrt(120 / 112); rtol=1e-3)
    end

    @testset "forecast shape and interval nesting" begin
        f = fit_ets(y, m; trend=:add, seasonal=:add)
        fc = forecast(f, 12; level=[80.0, 95.0])
        @test length(fc.point) == 12
        @test size(fc.lower) == (12, 2)
        @test issorted(fc.se)                              # widens with horizon
        @test all(fc.lower[:, 2] .<= fc.lower[:, 1])       # 95% outside 80%
        @test all(fc.upper[:, 2] .>= fc.upper[:, 1])
        @test isapprox(fc.se[1], sqrt(f.sigma2); rtol=1e-12)   # h=1 is sigma
        @test predict(f, 12).point == fc.point             # exact alias
    end

    @testset "a damped forecast converges where an undamped one does not" begin
        d = forecast(fit_ets(y, m; trend=:damped), 200)
        u = forecast(fit_ets(y, m; trend=:add), 200)
        @test abs(d.point[200] - d.point[199]) < abs(u.point[200] - u.point[199])
        # and the undamped one keeps a constant slope
        du = diff(u.point)
        @test isapprox(du[end], du[1]; rtol=1e-8)
    end

    @testset "a seasonal forecast repeats with the period" begin
        fc = forecast(fit_ets(y, m; seasonal=:add), 12)
        d = diff(fc.point)
        @test isapprox(d[1], d[1+m]; atol=1e-6)
    end

    @testset "fixed pins parameters and removes them from the fit" begin
        f = fit_ets(y, m; trend=:add, seasonal=:add,
                     fixed=(alpha=0.4, beta=0.1, gamma=0.3))
        @test f.alpha == 0.4 && f.beta == 0.1 && f.gamma == 0.3
        # three fewer free parameters than the unpinned fit
        g = fit_ets(y, m; trend=:add, seasonal=:add)
        @test f.nparams == g.nparams - 3
    end

    @testset "the admissible region cannot fit worse than the traditional one" begin
        for (tr, se, _) in SPECS
            t = fit_ets(y, m; trend=tr, seasonal=se, constraint=:traditional)
            a = fit_ets(y, m; trend=tr, seasonal=se, constraint=:admissible)
            @test a.sse <= t.sse * (1 + 1e-6)         # strictly wider feasible set
        end
    end

    @testset "multiplicative TREND is still refused, naming the limitation" begin
        # seasonal=:mul is now supported (see the Stage 9.1 testsets below);
        # multiplicative trend is not, and is the one R also excludes by
        # default via allow.multiplicative.trend=FALSE
        for bad in (:mul, :multiplicative)
            e1 = try; fit_ets(y, m; trend=bad); nothing; catch e; e; end
            @test e1 isa ArgumentError
            @test occursin("non-linear", e1.msg)
        end
        # a misspelled seasonal is corrected rather than silently accepted
        e2 = try; fit_ets(y, m; seasonal=:multiplicative); nothing; catch e; e; end
        @test e2 isa ArgumentError
        @test occursin(":mul", e2.msg)
        # the forbidden trio: additive error with multiplicative seasonal
        e3 = try; fit_ets(y, m; error=:add, seasonal=:mul); nothing; catch e; e; end
        @test e3 isa ArgumentError
        @test occursin("unstable", e3.msg)
    end

    @testset "StatsAPI contract" begin
        f = fit_ets(y, m; trend=:add, seasonal=:add)
        @test length(coef(f)) == 3                 # alpha, beta, gamma
        @test length(residuals(f)) == nobs(f) == n
        @test isfinite(loglikelihood(f)) && isfinite(aic(f)) && isfinite(bic(f))
        @test isapprox(aic(f), -2 * loglikelihood(f) + 2 * f.nparams; rtol=1e-12)
        @test f.aicc > f.aic                       # the correction is positive
    end

    @testset "show prints the notation and the fitted parameters" begin
        out = sprint(show, fit_ets(y, m; trend=:damped, seasonal=:add))
        @test occursin("ETS(A,Ad,A)", out)
        @test occursin("alpha", out) && occursin("phi", out)
        @test occursin("AICc", out)
    end

    @testset "edge cases" begin
        @test_throws ArgumentError fit_ets(Float64[], 4)
        @test_throws ArgumentError fit_ets(randn(3), 4; seasonal=:add)   # < 2 cycles
        @test_throws ArgumentError fit_ets(randn(50), 1; seasonal=:add)  # period 1
        @test_throws ArgumentError fit_ets(y, m; constraint=:bogus)
        @test_throws ArgumentError fit_ets(y, m; initial=:bogus)
        @test_throws ArgumentError fit_ets(y, m; fixed=(bogus=0.5,))
        @test_throws ArgumentError fit_ets(y, m; fixed=(beta=0.1,))      # no trend
        @test_throws ArgumentError fit_ets(y, m; fixed=(gamma=0.1,))     # no seasonal
        @test_throws ArgumentError fit_ets(y, m; fixed=(phi=0.9,))       # not damped
        @test_throws ArgumentError forecast(fit_ets(y, m), 0)
        # a constant series still fits rather than erroring
        c = fit_ets(fill(5.0, 60), 4; trend=:add, seasonal=:add)
        @test all(isfinite, c.fitted)
    end

    @testset "container-agnostic" begin
        a = fit_ets(Tuple(y), m; trend=:add, seasonal=:add)
        b = fit_ets(y, m; trend=:add, seasonal=:add)
        @test isapprox(a.sse, b.sse; rtol=1e-10)
    end
end

# A SECOND dataset, from the public catalogue rather than the simulated
# fixture: log Johnson & Johnson quarterly earnings, 84 points, period 4.
# Every number below is R's forecast::ets() on log(dataset("jj").value).
# This matters because the fixture was simulated FROM an ETS(A,A,A), so it
# is the friendliest possible case; jj is real data with a near-exact
# log-linear trend, and it is where R's own damped fit hits the phi bound.
@testset "fit_ets — a real series, verified against R (log jj)" begin
    y = log.(dataset("jj").value)
    m = 4
    @test length(y) == 84

    # R: ets(ts(y, frequency=4), model=, damped=) -> sum(residuals^2), length(par)
    R_JJ = Dict(
        (:none, :none)   => (2.77338384, 2),
        (:add, :none)    => (1.97367382, 4),
        (:none, :add)    => (0.98210307, 6),
        (:add, :add)     => (0.63189412, 8),
        (:damped, :none) => (1.89294469, 5),
        (:damped, :add)  => (0.67535675, 9),
    )

    @testset "match or beat R on all six, with R's parameter counts" begin
        for ((tr, se), (sse, np)) in R_JJ
            f = fit_ets(y, m; trend=tr, seasonal=se)
            @test f.converged
            @test f.sse <= sse * (1 + 1e-7)
            @test f.nparams - 1 == np
        end
    end

    @testset "simple exponential smoothing agrees with R to 1e-7" begin
        # the one model with no local optima to get stuck in
        f = fit_ets(y, m)
        @test isapprox(f.sse, 2.77338384; atol=1e-7)
        # alpha agrees to 2.6e-5 rather than to the SSE's 1e-7: the objective
        # is flat in alpha near the optimum, so the two optimisers stop at
        # slightly different points on an indistinguishable SSE. The SSE is
        # the quantity to hold tightly; alpha follows it loosely.
        @test isapprox(f.alpha, 0.50010785; atol=1e-4)   # R's alpha
    end

    @testset "R's own damped fit hits the phi bound here, and so does this" begin
        # Both cap phi at 0.98 -- those are R's default bounds (0.8, 0.98),
        # adopted deliberately. R returns 0.97995483 for ETS(A,Ad,A) on this
        # series, so a phi of exactly 0.98 here is agreement with R, NOT the
        # optimiser failure that the same number represents in statsmodels
        # on the simulated fixture.
        f = fit_ets(y, m; trend=:damped, seasonal=:add)
        @test isapprox(f.phi, 0.98; atol=1e-8)
        @test isapprox(f.phi, 0.97995483; atol=1e-4)      # R's value
        @test f.sse <= 0.67535675                          # and a better SSE
    end

    @testset "auto_ets agrees with R's automatic selection" begin
        # R: ets(y)$method == "ETS(A,A,A)"
        @test TSAnalytics.notation(auto_ets(y, m)) == "ETS(A,A,A)"
    end

    @testset "a near-deterministic slope drives beta to its lower bound" begin
        # log(jj) has an almost exactly constant log-linear growth rate, so
        # the slope needs no updating: beta goes to ~0 and the trend stays at
        # b0. Legitimate, and the traditional region permits it.
        f = fit_ets(y, m; trend=:add, seasonal=:add)
        @test f.beta < 1e-6
        @test isapprox(f.trend_component[end], f.trend_component[1]; rtol=1e-4)
        # and it still forecasts an upward line
        fc = forecast(f, 8)
        @test issorted(diff(fc.point)[1:1])
        @test fc.point[5] > fc.point[1]            # one year on, higher
    end

    @testset "taking logs is how multiplicative seasonality is handled" begin
        # jj in levels has seasonal amplitude growing with the level -- the
        # multiplicative case this family cannot represent. On the log scale
        # it is additive, which is the documented workaround, and it fits
        # far better in-sample relative to the series' own scale.
        lv = dataset("jj").value
        a = fit_ets(lv, 4; trend=:add, seasonal=:add)
        b = fit_ets(log.(lv), 4; trend=:add, seasonal=:add)
        rel_a = a.sse / sum(abs2, lv .- mean(lv))
        rel_b = b.sse / sum(abs2, log.(lv) .- mean(log.(lv)))
        @test rel_b < rel_a
    end
end
