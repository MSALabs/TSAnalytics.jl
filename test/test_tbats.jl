using DelimitedFiles

# Stage 9.4: TBATS (De Livera, Hyndman & Snyder 2011).
#
# DUAL-VERIFIED against R's forecast::tbats and Python's tbats 1.1.3.
# Reference values: test/verification/tbats/tbats.R and tbats.py.
#
# The model itself is pinned exactly: at R's own parameters and seed states the
# recursion reproduces R's fitted[1], its SSE, its variance, its "likelihood"
# (which is n*log(SSE), not a log-likelihood) and its AIC. The point forecast
# reproduces R's to 5e-7 and Python's to 1e-5.
#
# Two things do NOT agree, and both are checked rather than described:
#
#  1. R's prediction intervals for a SEASONAL TBATS model are far too narrow.
#     It uses c_j = alpha for every shock weight, where the correct weight is
#     w'F^(j-1)g. At h = 8 R's standard error is 2.640 against 7.31 from this
#     package, from Python's tbats, and from 3,000,000 simulated paths of the
#     model's own recursion -- R is 2.8 times too narrow. R's NON-seasonal
#     bats intervals are correct, which is how the bug was localised.
#
#  2. This package estimates the seed states by exact least squares; both
#     references use a heuristic regression seed. At identical smoothing
#     parameters that gives this package a strictly lower SSE -- 660.96
#     against their 672.21 -- so its fits do not reproduce theirs exactly,
#     and are better in-sample rather than merely different.

@testset "fit_tbats — trigonometric seasonality (Stage 9.4)" begin
    T = TSAnalytics
    y = vec(readdlm(joinpath(@__DIR__, "verification", "ets", "ets_y.csv")))
    n = length(y)
    # R: tbats(ts(y, frequency=4), use.box.cox=FALSE, use.trend=TRUE,
    #          use.damped.trend=FALSE, use.arma.errors=FALSE)
    R_AL, R_BE = 0.1871849, 0.2087534
    R_G1, R_G2 = -0.0001006233, -0.0001288012
    R_SEED = [100.65110696, -0.35456946, 7.49575713, -0.45597546]
    R_SSE = 672.2125
    R_MEAN = [181.22098258, 176.34227824, 172.37521555, 183.40033966,
              193.51382212, 188.63511779, 184.66805509, 195.69317920]

    @testset "the recursion reproduces R exactly at R's own parameters" begin
        f = Vector{Float64}(undef, n)
        r = similar(f)
        sse = T._tbats_recursion!(f, r, y, R_SEED, R_AL, R_BE, 1.0, [R_G1], [R_G2],
                                   Float64[], Float64[], [4.0], [1], true)
        @test isapprox(f[1], 107.79229463; atol=1e-7)      # R's fitted[1]
        @test isapprox(sse, R_SSE; atol=1e-3)
        @test isapprox(sse, 672.212547; atol=1e-5)
        @test isapprox(sse / n, 5.601771; atol=1e-6)       # R's `variance`
        # R calls n*log(SSE) the "likelihood" and builds its AIC from it
        @test isapprox(n * log(sse), 781.2689; atol=1e-3)
        @test isapprox(n * log(sse) + 2 * 8, 797.2689; atol=1e-3)   # R's AIC, np=8
    end

    @testset "Python's tbats agrees with R on the parameters and the seed" begin
        # Python: TBATS(seasonal_periods=[4], use_box_cox=False, use_trend=True,
        #               use_damped_trend=False, use_arma_errors=False)
        PY_SEED = [100.65110696, -0.35456946, 7.49575713, -0.45597546]
        @test maximum(abs.(PY_SEED .- R_SEED)) < 1e-8      # identical seeds
        @test isapprox(0.18719151, R_AL; atol=1e-5)        # Python's alpha
        @test isapprox(0.20888675, R_BE; atol=3e-4)        # Python's beta
        @test isapprox(672.107863, R_SSE; rtol=2e-4)       # Python's SSE
    end

    @testset "the seed-state layout is R's, confirmed on a k=5 monthly fit" begin
        # R: tbats(AirPassengers) selects k = 5, giving 2 + 2*5 = 12 seeds laid
        # out [l, b, s_1..s_5, sstar_1..sstar_5], and its fitted[1] is
        # l + b + sum of the five s seeds -- not of all ten.
        SEED12 = [118.60545327, -1.86866678, -45.57133214, 19.45531982,
                  -3.67671051, 4.10798308, 2.46842645, 5.73619660, 16.35274054,
                  -8.39207418, -6.40882327, -5.67949331]
        @test length(SEED12) == T._tbats_nstate(true, [5], 0, 0)
        @test isapprox(SEED12[1] + SEED12[2] + sum(SEED12[3:7]), 93.52047318;
                       atol=1e-7)                           # R's fitted[1]
        @test !isapprox(sum(SEED12), 93.52047318; atol=1e-3) # not all ten
    end

    @testset "the forecast formula reproduces R's point forecast" begin
        f = Vector{Float64}(undef, n)
        r = similar(f)
        sse = T._tbats_recursion!(f, r, y, R_SEED, R_AL, R_BE, 1.0, [R_G1], [R_G2],
                                   Float64[], Float64[], [4.0], [1], true)
        m = T.TBATSModel(nothing, R_AL, R_BE, nothing, [R_G1], [R_G2],
                         Float64[], Float64[], [4.0], [1], R_SEED, copy(f),
                         copy(r), sse, sse / n, 0.0, 0.0, 0.0, 8, n, true)
        fc = forecast(m, 8)
        @test maximum(abs.(fc.point .- R_MEAN)) < 1e-5
        # Python's mean on the same model, which differs from R's only in the
        # last couple of digits
        PY_MEAN = [181.22188773, 176.34336693, 172.37679620, 183.40174024,
                   193.51473420, 188.63621339, 184.66964267, 195.69458671]
        @test maximum(abs.(fc.point .- PY_MEAN)) < 2e-3
    end

    @testset "DIVERGENCE: R's seasonal TBATS intervals are ~2.8x too narrow" begin
        f = Vector{Float64}(undef, n)
        r = similar(f)
        sse = T._tbats_recursion!(f, r, y, R_SEED, R_AL, R_BE, 1.0, [R_G1], [R_G2],
                                   Float64[], Float64[], [4.0], [1], true)
        m = T.TBATSModel(nothing, R_AL, R_BE, nothing, [R_G1], [R_G2],
                         Float64[], Float64[], [4.0], [1], R_SEED, copy(f),
                         copy(r), sse, sse / n, 0.0, 0.0, 0.0, 8, n, true)
        fc = forecast(m, 8)
        # Python's tbats, an independent implementation
        PY_SE = [2.366622, 2.545497, 2.920511, 3.498452,
                 4.254137, 5.156907, 6.181906, 7.311177]
        # 3,000,000 simulated paths of the model's own recursion
        SIM_SE = [2.36616, 2.54584, 2.92043, 3.49711,
                  4.25493, 5.15399, 6.17872, 7.30789]
        @test maximum(abs.(fc.se .- PY_SE) ./ PY_SE) < 1e-3
        @test maximum(abs.(fc.se .- SIM_SE) ./ SIM_SE) < 2e-3
        # R's own, from (mean - lower)/z on its 95% bound
        R_SE = [2.366806, 2.407814, 2.448146, 2.487837,
                2.526810, 2.565096, 2.602831, 2.640038]
        @test isapprox(fc.se[1], R_SE[1]; rtol=1e-4)       # h=1 agrees
        @test fc.se[8] / R_SE[8] > 2.7                      # and then it does not
        # R's sequence is exactly the alpha-only formula, ignoring beta and gamma
        alpha_only = [sqrt(sse / n * (1 + (h - 1) * R_AL^2)) for h in 1:8]
        @test maximum(abs.(R_SE .- alpha_only) ./ alpha_only) < 1e-3
        # whereas the correct weight at j=1 is w'g = alpha + phi*beta + gamma1
        @test isapprox(sqrt(sse / n * (1 + (R_AL + R_BE + R_G1)^2)), PY_SE[2];
                       rtol=1e-3)
    end

    @testset "R's NON-seasonal bats intervals are correct, which localises it" begin
        # R: bats(ts(y, frequency=1), use.box.cox=FALSE, use.trend=TRUE, ...)
        #    alpha=0.14131613 beta=0.01940015, se/sqrt(variance) per horizon
        al, be = 0.14131613, 0.01940015
        R_RATIO = [1.00000, 1.01283, 1.02872, 1.04789,
                   1.07052, 1.09672, 1.12659, 1.16018]
        # the correct local-linear-trend factor, sqrt(1 + sum((al + j*be)^2))
        correct = [sqrt(1 + sum((al .+ (1:(h-1)) .* be) .^ 2)) for h in 1:8]
        @test maximum(abs.(R_RATIO .- correct)) < 1e-5
        # so R includes beta without seasonality and drops it with seasonality
        @test correct[8] > 1.15
    end

    @testset "the seed states come from exact least squares, unlike the references" begin
        x0 = T._tbats_seed(y, R_AL, R_BE, 1.0, [R_G1], [R_G2], Float64[],
                            Float64[], [4.0], [1], true)
        f = Vector{Float64}(undef, n)
        r = similar(f)
        mine = T._tbats_recursion!(f, r, y, x0, R_AL, R_BE, 1.0, [R_G1], [R_G2],
                                    Float64[], Float64[], [4.0], [1], true)
        theirs = T._tbats_recursion!(f, r, y, R_SEED, R_AL, R_BE, 1.0, [R_G1],
                                      [R_G2], Float64[], Float64[], [4.0], [1], true)
        @test mine < theirs                       # 660.96 against 672.21
        @test isapprox(mine, 660.957556; atol=1e-4)
        # it really is the minimum over x0: perturbing it cannot do better
        for j in 1:length(x0)
            for d in (-0.2, 0.2)
                xp = copy(x0)
                xp[j] += d
                worse = T._tbats_recursion!(f, r, y, xp, R_AL, R_BE, 1.0, [R_G1],
                                             [R_G2], Float64[], Float64[], [4.0],
                                             [1], true)
                @test worse >= mine - 1e-9
            end
        end
    end

    @testset "the forecastability condition is what keeps the fit sane" begin
        # Without it the optimiser reaches gamma1 = -0.477 and an SSE of 297,
        # less than half R's -- a divergent fit, not a better one.
        @test !T._tbats_admissible(0.303069, 0.238929, 1.0, [-0.4770987],
                                    [0.44601004], [4.0], [1], true)
        # the fitted parameters, and R's, are admissible
        @test T._tbats_admissible(R_AL, R_BE, 1.0, [R_G1], [R_G2], [4.0], [1], true)
        m = fit_tbats(y, 4; k=1)
        @test T._tbats_admissible(m.alpha, m.beta, 1.0, m.gamma1, m.gamma2,
                                   m.periods, m.k, true)
        # the spectral radius is strictly inside the unit circle
        F, g, w = T._tbats_matrices(m.alpha, m.beta, 1.0, m.gamma1, m.gamma2,
                                     m.periods, m.k, true)
        @test maximum(abs, eigvals(F - g * transpose(w))) < 1
    end

    @testset "a fitted model matches or beats both references in-sample" begin
        m = fit_tbats(y, 4; k=1)
        @test m.converged
        @test m.sse <= R_SSE
        @test m.sse <= 672.107863          # Python's
        @test m.nobs == n
        @test m.k == [1]
        @test m.periods == [4.0]
        @test length(m.seed_states) == 4
        @test m.lambda === nothing
        @test isempty(m.ar) && isempty(m.ma)
        @test isapprox(m.sigma2, m.sse / n; rtol=1e-12)
        @test isapprox(m.sse, sum(abs2, m.resid); rtol=1e-10)
    end

    @testset "trigonometric seasonality is what ETS cannot do" begin
        # a non-integer period: there is no integer number of seasonal states
        # to hold, only an angle to rotate by
        m = fit_tbats(y, 4.3; k=2)
        @test m.periods == [4.3]
        @test m.converged
        @test all(isfinite, forecast(m, 8).point)
        # fit_ets cannot be asked for this at all
        @test_throws Exception fit_ets(y, 4.3; seasonal=:add)
    end

    @testset "two seasonal periods at once" begin
        # the headline capability: a short and a long cycle together
        nn = 400
        t = 1:nn
        z = 100 .+ 0.05 .* t .+ 3 .* sin.(2pi .* t ./ 7) .+
            6 .* sin.(2pi .* t ./ 30.4) .+ 0.4 .* randn(MersenneTwister(11), nn)
        m = fit_tbats(z, [7, 30.4]; k=[2, 2])
        @test length(m.periods) == 2
        @test m.gamma1 isa Vector && length(m.gamma1) == 2
        @test length(m.seed_states) == 2 + 2 * 4
        @test m.converged
        fc = forecast(m, 20)
        @test all(isfinite, fc.point) && all(isfinite, fc.se)
        @test issorted(fc.se)
        # it beats a single-period fit on the same data, since there really
        # are two cycles
        m1 = fit_tbats(z, 7; k=2)
        @test m.sse < m1.sse
    end

    @testset "more harmonics means more states and more parameters" begin
        # period 7 admits k up to 3; period 4 admits only 1
        a = fit_tbats(y, 7; k=1)
        b = fit_tbats(y, 7; k=2)
        @test length(a.seed_states) == 4
        @test length(b.seed_states) == 6        # two more states per harmonic
        @test b.nparams > a.nparams
        @test a.converged && b.converged
        # NOTE deliberately not asserted: that the richer model fits better.
        # Adding a harmonic changes the dynamics rather than just adding a
        # dimension -- gamma1/gamma2 are shared across a period's harmonics --
        # so the models are not nested and a worse SSE is not by itself
        # evidence of a stuck optimiser. What the multi-start fixed was a much
        # larger gap than this.
        @test all(isfinite, forecast(b, 8).point)
    end

    @testset "the Nyquist harmonic is excluded, as R excludes it" begin
        # For an even integer period the j = m/2 harmonic rotates by pi, so
        # its pair stops mixing and sstar is unidentified. Admitting it leaves
        # the optimiser a direction the data cannot inform.
        @test T._tbats_kmax(4) == 1           # R picks k = 1 for quarterly
        @test T._tbats_kmax(12) == 5          # R picks k = 5 for monthly
        @test T._tbats_kmax(7) == 3           # odd: floor(7/2) stands
        @test T._tbats_kmax(4.3) == 2         # non-integer never hits lam = pi
        @test T._tbats_kmax(365.25) == 182
        @test T._tbats_default_k([12.0], 144) == [5]
        e = try; fit_tbats(y, 4; k=2); nothing; catch err; err; end
        @test e isa ArgumentError
        @test occursin("unidentified", e.msg)
        # and the reason it matters: the rotation block degenerates to -I.
        # With a trend and k = 2 the state is [l, b, s1, s2, sstar1, sstar2],
        # so the j = 2 (Nyquist) harmonic sits at 4 and its pair at 6.
        F, _, w = T._tbats_matrices(0.1, 0.01, 1.0, [0.001], [0.001], [4.0], [2], true)
        @test size(F) == (6, 6)
        @test isapprox(F[4, 4], -1.0; atol=1e-12)   # cos(pi)
        @test isapprox(F[4, 6], 0.0; atol=1e-12)    # sin(pi): the pair stops mixing
        @test isapprox(F[6, 4], 0.0; atol=1e-12)
        @test w[6] == 0.0                            # sstar is never measured
        @test w[4] == 1.0                            # while s is
        # the j = 1 harmonic does rotate, which is the contrast
        @test isapprox(F[3, 5], 1.0; atol=1e-12)    # sin(pi/2)
    end

    @testset "Box-Cox is applied to the fit and undone in the forecast" begin
        m0 = fit_tbats(y, 4; k=1)
        ml = fit_tbats(y, 4; k=1, lambda=0.5)
        @test ml.lambda == 0.5
        @test m0.lambda === nothing
        # the fitted values come back on the original scale either way
        @test isapprox(mean(ml.fitted), mean(y); rtol=0.1)
        fc = forecast(ml, 8)
        @test all(isfinite, fc.point)
        @test all(fc.lower[:, 2] .< fc.point .< fc.upper[:, 2])
        # lambda=:auto picks one through guerrero_lambda
        ma = fit_tbats(y, 4; k=1, lambda=:auto)
        @test ma.lambda !== nothing
        @test isapprox(ma.lambda, guerrero_lambda(y, 4); rtol=1e-10)
        # lambda=1 is the identity transform up to a shift, so the SSE on the
        # transformed scale is the untransformed one
        m1 = fit_tbats(y, 4; k=1, lambda=1.0)
        @test isapprox(m1.sse, m0.sse; rtol=1e-6)
    end

    @testset "trend and damping behave as in fit_ets" begin
        nt = fit_tbats(y, 4; k=1, trend=false)
        @test nt.beta === nothing
        @test nt.phi === nothing
        @test length(nt.seed_states) == 3            # l, s, sstar
        d = fit_tbats(y, 4; k=1, damped=true)
        @test d.phi !== nothing
        @test 0.8 <= d.phi <= 1.0
        @test d.beta !== nothing                     # damping implies a trend
        # a damped forecast flattens where an undamped one does not
        fu = forecast(fit_tbats(y, 4; k=1), 200)
        fd = forecast(d, 200)
        @test abs(fd.point[200] - fd.point[196]) <= abs(fu.point[200] - fu.point[196])
    end

    @testset "intervals nest and widen; predict is the same function" begin
        m = fit_tbats(y, 4; k=1)
        fc = forecast(m, 12; level=[80.0, 95.0])
        @test size(fc.lower) == (12, 2)
        @test issorted(fc.se)
        @test all(fc.lower[:, 2] .<= fc.lower[:, 1])
        @test all(fc.upper[:, 2] .>= fc.upper[:, 1])
        @test isapprox(fc.se[1], sqrt(m.sigma2); rtol=1e-10)
        @test predict(m, 12).point == fc.point
        @test occursin("TBATS", fc.model_name)
    end

    @testset "ARMA errors are refused, naming the reason" begin
        e = try; fit_tbats(y, 4; k=1, arma=(1, 0)); nothing; catch err; err; end
        @test e isa ArgumentError
        @test occursin("not implemented", e.msg)
        @test occursin("forecastability", e.msg)
        @test_throws ArgumentError fit_tbats(y, 4; k=1, arma=(0, 1))
        @test fit_tbats(y, 4; k=1, arma=(0, 0)).converged
    end


    @testset "the stability margin keeps fits off the boundary" begin
        # A plain radius < 1 is not enough: the optimiser rides the boundary,
        # because a nearly non-stationary seasonal fits the sample better.
        # Before the margin existed the quarterly fit reached a radius of
        # 0.9999999459 -- on the edge -- for an SSE of 657.3881.
        m = fit_tbats(y, 4; k=1)
        F, g, w = T._tbats_matrices(m.alpha, m.beta, 1.0, m.gamma1, m.gamma2,
                                     m.periods, m.k, true)
        r = maximum(abs, eigvals(F - g * transpose(w)))
        @test r <= 1 - T.TBATS_STABILITY_MARGIN + 1e-12
        @test T.TBATS_STABILITY_MARGIN == 1e-5
        # R's own optimum sits comfortably inside the margin, which is how the
        # margin was chosen rather than guessed
        F2, g2, w2 = T._tbats_matrices(R_AL, R_BE, 1.0, [R_G1], [R_G2], [4.0],
                                        [1], true)
        rr = maximum(abs, eigvals(F2 - g2 * transpose(w2)))
        @test isapprox(rr, 0.9999280; atol=1e-6)
        @test rr < 1 - T.TBATS_STABILITY_MARGIN
    end

    @testset "the region is spectral, not a rule about signs" begin
        # At gamma1 = +0.004101 the admissible gamma2 runs from negative values
        # up to about +0.0003, so neither sign is uniformly allowed or
        # forbidden -- which is why a sign heuristic cannot replace the
        # eigenvalue computation.
        ok_neg = T._tbats_admissible(0.134896, 0.224082, 1.0, [0.004101],
                                      [-0.001], [4.0], [1], true)
        ok_zero = T._tbats_admissible(0.134896, 0.224082, 1.0, [0.004101],
                                       [0.0], [4.0], [1], true)
        bad_pos = T._tbats_admissible(0.134896, 0.224082, 1.0, [0.004101],
                                       [0.004], [4.0], [1], true)
        @test ok_neg && ok_zero && !bad_pos
        # and at EQUAL magnitudes the sign does decide, for a quarterly single
        # harmonic -- the observation that first localised the start-grid bug
        @test T._tbats_admissible(0.1871849, 0.2087534, 1.0, [-1e-3], [-1e-3],
                                   [4.0], [1], true)
        @test !T._tbats_admissible(0.1871849, 0.2087534, 1.0, [-1e-3], [1e-3],
                                    [4.0], [1], true)
    end

    @testset "a multi-period fit actually leaves its starting point" begin
        # Regression test for a bug in this implementation, not in a reference:
        # with gamma2 > 0 in the start grid, every start for a two-period model
        # was inadmissible, the objective was 1e10 everywhere the simplex could
        # reach, and the fit returned alpha, beta and both gammas exactly as
        # the start values -- with converged = true.
        nn = 365
        t = 1:nn
        z = 100 .+ 0.02 .* t .+ 4 .* sin.(2pi .* t ./ 7) .+
            6 .* sin.(2pi .* t ./ 30.4) .+ 0.5 .* randn(MersenneTwister(3), nn)
        m = fit_tbats(z, [7, 30.4]; k=[2, 2])
        @test m.converged
        @test !(m.alpha ≈ 0.1)                 # the first start's alpha
        @test !all(g -> g ≈ -1e-4, m.gamma1)   # the first start's gamma1
        # and the fit recovers something near the true noise level
        @test isapprox(sqrt(m.sigma2), 0.5; rtol=0.5)
    end

    @testset "StatsAPI contract and show" begin
        m = fit_tbats(y, 4; k=1)
        @test length(coef(m)) == 4                  # alpha, beta, gamma1, gamma2
        @test residuals(m) === m.resid
        @test fitted(m) === m.fitted
        @test nobs(m) == n
        @test isfinite(loglikelihood(m)) && isfinite(aic(m)) && isfinite(bic(m))
        @test isapprox(aic(m), -2 * loglikelihood(m) + 2 * m.nparams; rtol=1e-12)
        out = sprint(show, m)
        @test occursin("TBATS", out)
        @test occursin("alpha", out) && occursin("gamma1", out)
        @test occursin("SSE", out)
    end

    @testset "error paths" begin
        @test_throws ArgumentError fit_tbats(y[1:4], 4)
        @test_throws ArgumentError fit_tbats(y, 1)              # period must be > 1
        @test_throws ArgumentError fit_tbats(y, 0.5)
        @test_throws ArgumentError fit_tbats(y, Float64[])
        @test_throws ArgumentError fit_tbats(y, 4; k=0)
        @test_throws ArgumentError fit_tbats(y, 4; k=2)         # Nyquist harmonic
        @test_throws DimensionMismatch fit_tbats(y, [4, 12]; k=[1])
        @test_throws ArgumentError fit_tbats([1.0, 2.0, NaN, 4.0, 5.0], 4; k=1)
        @test_throws ArgumentError forecast(fit_tbats(y, 4; k=1), 0)
        @test_throws ArgumentError fit_tbats(y .- 200, 4; k=1, lambda=:auto)
    end

    @testset "container-agnostic" begin
        @test isapprox(fit_tbats(Tuple(y), 4; k=1).sse, fit_tbats(y, 4; k=1).sse;
                       rtol=1e-10)
    end
end
