using DelimitedFiles

# Stage 9.1, second half: the multiplicative-error ETS forms.
#
# R's forecast::ets searches 15 models by default. Stage 9A built the 6 with an
# additive error; these are the 6 with a multiplicative error and an
# additive-or-absent seasonal -- Hyndman, Koehler, Ord & Snyder (2008)
# "class 2" -- plus the 3 multiplicative-SEASONAL forms (class 3), which fit
# here but do not forecast yet.
#
# The headline finding, and the reason the class-2 variance is asserted against
# SIMULATION rather than against R: R's prediction intervals for the NO-TREND
# seasonal models apply the seasonal variance increment one period early. Six
# million simulated paths from R's own fitted ETS(M,N,A) state agree with this
# package to 0.09% at every horizon, while R is +25.6% too wide at h = 4 and
# +13.4% at h = 8. The same happens for additive-error ETS(A,N,A), where four
# million paths give 2.5808*sigma2 at h = 4 against R's 3.0536*sigma2. R's
# trended seasonal models agree with this package exactly.

@testset "fit_ets — multiplicative error, class 2 and 3 (Stage 9.1)" begin
    y = vec(readdlm(joinpath(@__DIR__, "verification", "ets", "ets_y.csv")))
    n = length(y)
    m = 4
    @test all(>(0), y)            # multiplicative models need this

    MSPECS = [(:none, :none, "ETS(M,N,N)"), (:add, :none, "ETS(M,A,N)"),
              (:damped, :none, "ETS(M,Ad,N)"), (:none, :add, "ETS(M,N,A)"),
              (:add, :add, "ETS(M,A,A)"), (:damped, :add, "ETS(M,Ad,A)"),
              (:none, :mul, "ETS(M,N,M)"), (:add, :mul, "ETS(M,A,M)"),
              (:damped, :mul, "ETS(M,Ad,M)")]
    # R: ets(ts(y, frequency=4), model=, damped=) -> sum(residuals^2), length(par)
    R_SSE = Dict("ETS(M,N,N)" => 0.59796463, "ETS(M,A,N)" => 0.50452764,
                 "ETS(M,Ad,N)" => 0.49293180, "ETS(M,N,A)" => 0.05474691,
                 "ETS(M,A,A)" => 0.04114805, "ETS(M,Ad,A)" => 0.04048134,
                 "ETS(M,N,M)" => 0.06014784, "ETS(M,A,M)" => 0.04500660,
                 "ETS(M,Ad,M)" => 0.04431354)
    R_NP = Dict("ETS(M,N,N)" => 2, "ETS(M,A,N)" => 4, "ETS(M,Ad,N)" => 5,
                "ETS(M,N,A)" => 6, "ETS(M,A,A)" => 8, "ETS(M,Ad,A)" => 9,
                "ETS(M,N,M)" => 6, "ETS(M,A,M)" => 8, "ETS(M,Ad,M)" => 9)

    @testset "all nine fit, are labelled right, and match or beat R" begin
        for (tr, se, lab) in MSPECS
            f = fit_ets(y, m; error=:mul, trend=tr, seasonal=se)
            @test f.converged
            @test TSAnalytics.notation(f) == lab
            @test f.error === :mul
            @test f.sse <= R_SSE[lab] * (1 + 1e-6)
            @test f.nparams - 1 == R_NP[lab]
            @test all(isfinite, f.fitted) && all(>(0), f.fitted)
        end
    end

    @testset "the residuals are RELATIVE, as R's are for an M-error model" begin
        # eps = (y - yhat)/yhat, which is why sse is order 0.5 here and order
        # 6000 for the additive-error counterpart on the same series
        f = fit_ets(y, m; error=:mul)
        @test isapprox(f.resid, (y .- f.fitted) ./ f.fitted; rtol=1e-12)
        @test isapprox(f.sse, sum(abs2, f.resid); rtol=1e-12)
        @test f.sse < 1.0
        @test fit_ets(y, m).sse > 6000
    end

    @testset "the likelihood carries the multiplicative-error Jacobian" begin
        # dy/deps = yhat, so a term sum(log|yhat|) enters. R reports
        # -(n/2)log(SSE) - sum(log|yhat|); its own decomposition was checked
        # against that identity to 1e-13 on all nine models.
        f = fit_ets(y, m; error=:mul)
        jac = sum(log ∘ abs, f.fitted)
        @test isapprox(f.loglik, -n/2 * (log(2pi) + log(f.sigma2) + 1) - jac; rtol=1e-12)
        # R's convention, rebuilt from this package's own fit: the two differ
        # by the same concentration constant as everywhere else
        r_style = -n/2 * log(f.sse) - jac
        @test isapprox(f.loglik - r_style, n/2 * (log(n) - log(2pi) - 1); atol=1e-8)
        # without the Jacobian an M-error aic would not be comparable with an
        # A-error one, which is the whole reason it is there
        @test jac > 500                          # 558.02 on this fixture
        @test !isapprox(f.loglik, -n/2 * (log(2pi) + log(f.sigma2) + 1); rtol=1e-6)
    end

    @testset "class 2 shares class 1's point forecasts by construction" begin
        # mu*eps == e makes the state recursions identical, so at the SAME
        # parameters the two produce the same fitted path. A direct check on
        # the recursion rather than on an optimum.
        T = TSAnalytics
        fa = zeros(n); ra = zeros(n); lva = zeros(n); tda = zeros(n); sna = zeros(n)
        fm = zeros(n); rm = zeros(n); lvm = zeros(n); tdm = zeros(n); snm = zeros(n)
        l0, b0, fig = T._ets_heuristic_init(y, m, :add, :add)
        s0 = fig .- sum(fig) / m
        T._ets_recursion!(fa, ra, lva, tda, sna, y, 0.3, 0.1, 0.1, 1.0, l0, b0,
                          copy(s0), :add, :add, :add)
        T._ets_recursion!(fm, rm, lvm, tdm, snm, y, 0.3, 0.1, 0.1, 1.0, l0, b0,
                          copy(s0), :add, :add, :mul)
        @test isapprox(fa, fm; rtol=1e-12)        # identical fitted values
        @test !isapprox(ra, rm; rtol=1e-3)        # but different residuals
        @test isapprox(rm, ra ./ fa; rtol=1e-12)
    end

    @testset "class 2 variances: verified against 6M simulated paths, not R" begin
        # R is wrong for the no-trend seasonal branch (see the header), so the
        # reference here is simulation of R's own fitted ETS(M,N,A) forward
        # from R's own final state.
        T = TSAnalytics
        al, ga, s2 = 0.6639718582, 0.3360278632, 0.000480236015
        lN = 162.3040278682
        sN = reverse([9.3207620383, 1.5259037204, 7.9398375980, 13.5639715170])
        mu = T._ets_propagate(lN, 0.0, sN, al, 0.0, ga, 1.0, :none, :add, 8)
        psi = T._ets_propagate(al, 0.0, vcat(zeros(m - 1), ga), al, 0.0, ga, 1.0,
                               :none, :add, 8)
        v = T._ets_class2_var(mu, psi, s2)
        SIM = [14.84768, 20.46687, 25.57300, 32.51346,
               47.76255, 52.83867, 57.41418, 65.09467]      # 6,000,000 paths
        @test maximum(abs.(v .- SIM) ./ SIM) < 0.002          # 0.2%, MC noise
        R_VAR = [14.85349, 20.47010, 25.58158, 40.83442,
                 47.26610, 52.31558, 58.14004, 73.79743]
        @test isapprox(R_VAR[4] / SIM[4], 1.2559; atol=1e-3)  # +25.6% at h=m
        @test isapprox(R_VAR[8] / SIM[8], 1.1337; atol=1e-3)  # +13.4% at h=2m
        @test maximum(abs.(R_VAR[1:3] .- SIM[1:3]) ./ SIM[1:3]) < 0.002
        # psi carries gamma at index m, which is the crux: the seasonal state a
        # shock revises is next USED m steps later
        @test isapprox(psi[m], al + ga; rtol=1e-10)
        @test isapprox(psi[m - 1], al; rtol=1e-10)
    end

    @testset "class 2 collapses to the ETS(M,N,N) closed form" begin
        # var_h = mu^2[(1+s2)(1+alpha^2 s2)^(h-1) - 1] -- the one case with a
        # textbook closed form, and the one where R agrees, to 7e-9
        T = TSAnalytics
        al, s2, lN = 0.30673475, 0.0050674968, 165.40363457
        v = T._ets_class2_var(fill(lN, 8), fill(al, 8), s2)
        closed = [lN^2 * ((1 + s2) * (1 + al^2 * s2)^(h - 1) - 1) for h in 1:8]
        @test maximum(abs.(v .- closed) ./ closed) < 1e-12
        # and R's own ETS(M,N,N) lower bounds follow from it
        R_LO = [142.32607490, 141.25957680, 140.23775881, 139.25538218,
                138.30815639, 137.39251502, 136.50545546, 135.64442149]
        z = 1.959963984540054
        @test maximum(abs.(lN .- z .* sqrt.(closed) .- R_LO)) < 1e-6
    end

    @testset "a multiplicative error scales the interval with the level" begin
        # the point of an M-error model: the error is proportional to the
        # series, so the variance compounds rather than merely accumulating
        f = fit_ets(y, m; error=:mul, trend=:add)
        fc = forecast(f, 24)
        @test issorted(fc.se)
        @test fc.se[24] / fc.se[1] > 1.5
        @test all(isfinite, fc.se)
        @test all(fc.lower[:, 2] .< fc.lower[:, 1])
        # compounding vs accumulating: relative to its own h=1 width, the
        # M-error interval grows faster than the A-error one on this series
        fa = forecast(fit_ets(y, m; trend=:add), 24)
        @test fc.se[24] / fc.se[1] > fa.se[24] / fa.se[1]
    end

    @testset "class 3 fits but does not forecast, and says why" begin
        for tr in (:none, :add, :damped)
            f = fit_ets(y, m; error=:mul, trend=tr, seasonal=:mul)
            @test f.converged
            @test f.seasonal === :mul
            e = try; forecast(f, 4); nothing; catch err; err; end
            @test e isa ArgumentError
            @test occursin("does not forecast", e.msg)
            @test occursin("class 3", e.msg)
            @test occursin("forecast::ets", e.msg)      # points at the reference
            @test_throws ArgumentError predict(f, 4)
        end
    end

    @testset "multiplicative seasonal factors average one and stay positive" begin
        # the normalisation that replaces sum-to-zero: an additive seasonal
        # sums to 0, a multiplicative one averages 1
        f = fit_ets(y, m; error=:mul, seasonal=:mul)
        @test length(f.s0) == m
        @test isapprox(sum(f.s0), Float64(m); rtol=1e-8)
        @test all(>(0), f.s0)
        a = fit_ets(y, m; error=:mul, seasonal=:add)
        @test isapprox(sum(a.s0), 0.0; atol=1e-8)
    end

    @testset "auto_ets searches the 12 forecastable models, not 15" begin
        # R considers 15 by default. The three class-3 forms are excluded here
        # because selecting a model that cannot be forecast would be worse
        # than not considering it.
        a = auto_ets(y, m)
        @test a.seasonal !== :mul
        @test forecast(a, 4) isa TSAnalytics.Forecast      # whatever wins, forecasts
        # R's own pick on this fixture is ETS(A,A,A), and so is this
        @test TSAnalytics.notation(a) == "ETS(A,A,A)"
        # additive_only restricts the search back to Stage 9A's six
        b = auto_ets(y, m; additive_only=true)
        @test b.error === :add
        @test TSAnalytics.notation(b) == "ETS(A,A,A)"
        # parallel and serial still select the identical model over the larger
        # space, not merely an equally good one
        @test TSAnalytics.notation(auto_ets(y, m; parallel=false)) ==
              TSAnalytics.notation(a)
    end

    @testset "a non-positive series cannot take a multiplicative component" begin
        z = collect(1.0:60.0) .- 30.0
        for kw in ((error=:mul,), (error=:mul, seasonal=:mul))
            e = try; fit_ets(z, 4; kw...); nothing; catch err; err; end
            @test e isa ArgumentError
            @test occursin("positive", e.msg)
        end
        # auto_ets drops them rather than failing
        a = auto_ets(z, 4)
        @test a.error === :add
    end

    @testset "error must be spelled :add or :mul" begin
        @test_throws ArgumentError fit_ets(y, m; error=:multiplicative)
        @test_throws ArgumentError fit_ets(y, m; error=:bogus)
        e = try; fit_ets(y, m; error=:mult); nothing; catch err; err; end
        @test occursin(":mul", e.msg)
    end

    @testset "show and notation carry the error letter" begin
        out = sprint(show, fit_ets(y, m; error=:mul, trend=:damped, seasonal=:add))
        @test occursin("ETS(M,Ad,A)", out)
        out2 = sprint(show, fit_ets(y, m; error=:mul, seasonal=:mul))
        @test occursin("ETS(M,N,M)", out2)
        @test TSAnalytics.notation(fit_ets(y, m)) == "ETS(A,N,N)"
    end

    @testset "AirPassengers: the case multiplicative error exists for" begin
        # seasonal amplitude grows with the level, so an M-error model should
        # beat its A-error counterpart on AICc without needing logs
        ap = _load_column(TSAnalytics.AIR_PASSENGERS, "passengers")
        a = fit_ets(ap, 12; trend=:add, seasonal=:add)
        b = fit_ets(ap, 12; error=:mul, trend=:add, seasonal=:add)
        @test b.aicc < a.aicc
        @test all(isfinite, forecast(b, 12).se)
        # and auto_ets over the larger space picks a multiplicative error here,
        # which is the practical payoff of this stage
        @test auto_ets(ap, 12).error === :mul
    end
end
