using DelimitedFiles

# Stage 11.3 -- Markov-switching regression and autoregression.
# handoff/stage 11/docs/stage-11.3-markov-switching-handoff.md §6.
# Primary reference statsmodels 0.15.0 (BSD); the canonical likelihood is
# also E-views' (statsmodels' own test file records -181.26339). MSwM (GPL,
# a different model, non-reproducible under its defaults) supplies no
# expected value.

const _MSFX = joinpath(@__DIR__, "..", "handoff", "stage 11")
const _RGNP = vec(Float64.(readdlm(joinpath(_MSFX, "fixtures", "hamilton_rgnp.csv"), ','; skipstart=1)[:, 2]))
const _HAM = [0.7546643166415282, 0.09591504844786301, -0.3588026223152828, 1.163522028189969,
              0.5913638295527929, 0.013479889902093485, -0.057529763871299266,
              -0.24699182841463502, -0.21292759281746598]

@testset "MS: Hamilton's model at fixed parameters (tier 1)" begin
    @test length(_RGNP) == 135
    @test sum(_RGNP) ≈ 100.5207128600 atol=1e-9
    m = fit_ms_autoregression(_RGNP, 4; params=_HAM)
    @test loglikelihood(m) ≈ -181.26339428124544 atol=1e-9
    @test nobs(m) == 131 && dof(m) == 9 && !m.estimated
    @test coefnames(m) == ["p[0->0]", "p[1->0]", "const[0]", "const[1]", "sigma2",
                           "ar.L1", "ar.L2", "ar.L3", "ar.L4"]
    F = readdlm(joinpath(_MSFX, "ground-truth", "sm_ar4_filtered.csv"), ','; skipstart=1)
    S = readdlm(joinpath(_MSFX, "ground-truth", "sm_ar4_smoothed.csv"), ','; skipstart=1)
    @test regime_probabilities(m; kind=:filtered) ≈ F atol=1e-10
    @test regime_probabilities(m) ≈ S atol=1e-10
    # the ground-truth files' own sums, tightly; the handoff's printed sums
    # (96.6874124793, 93.2939799393) come from its 10-digit parameter table
    # and sit 1e-7 away
    @test sum(m.filtered[:, 2]) ≈ sum(F[:, 2]) atol=1e-9
    @test sum(m.smoothed[:, 2]) ≈ sum(S[:, 2]) atol=1e-9
    @test sum(m.filtered[:, 2]) ≈ 96.6874124793 atol=1e-6
    @test sum(m.smoothed[:, 2]) ≈ 93.2939799393 atol=1e-6
    # at t = T there is no future information
    @test m.filtered[end, :] ≈ m.smoothed[end, :] atol=1e-12
    @test expected_durations(m) ≈ [4.07604793, 10.42589261] atol=1e-7
    @test m.initial ≈ [0.28106914, 0.71893086] atol=1e-8
    # the steady state is P's eigenvector for 1; P is left-stochastic
    @test m.P * m.initial ≈ m.initial atol=1e-14
    @test vec(sum(m.P; dims=1)) ≈ [1.0, 1.0] atol=1e-15
    @test m.P ≈ [0.7546643166415282 0.09591504844786301; 0.24533568335847178 0.904084951552137] atol=1e-15
    @test aic(m) ≈ 380.5267885624909 atol=1e-8
    @test bic(m) ≈ 406.40356447130125 atol=1e-8
    @test_throws ArgumentError regime_probabilities(m; kind=:predicted)
end

@testset "MS: the joint-state map" begin
    st = TSAnalytics._ms_states(2, 4)
    @test size(st.digits) == (32, 5)
    # decoding round-trips and every window appears once
    @test length(unique(eachrow(st.digits))) == 32
    for j in 1:32
        @test 1 + sum(st.digits[j, m] * 2^(5 - m) for m in 1:5) == j
        # a transition i -> j is permitted iff j's lags are i's window shifted
        @test length(st.pred[j]) == 2 && length(st.succ[j]) == 2
        @test all(st.digits[i, 1:4] == st.digits[j, 2:5] for i in st.pred[j])
        @test all(st.digits[j, 1:4] == st.digits[i, 2:5] for i in st.succ[j])
    end
    st3 = TSAnalytics._ms_states(3, 1)
    @test size(st3.digits) == (9, 2) && all(length.(st3.pred) .== 3)
    # the joint filtered distribution's marginal sums to one at every t
    m = fit_ms_autoregression(_RGNP, 4; params=_HAM)
    @test all(abs.(sum(m.filtered; dims=2) .- 1) .< 1e-12)
    @test all(abs.(sum(m.smoothed; dims=2) .- 1) .< 1e-12)
end

@testset "MS: the transition parameterisation" begin
    s3 = TSAnalytics._MSSpec(3, 0, true, 0, true, false, false, false, 1)
    @test TSAnalytics._ms_names(s3)[1:6] == ["p[0->0]", "p[1->0]", "p[2->0]", "p[0->1]", "p[1->1]", "p[2->1]"]
    # statsmodels' transform_params on these logits (closed form here; the
    # reference root-finds the inverse for k > 2)
    w = vcat([0.3, -0.5, 0.2, 0.4, -1.0, 0.7], zeros(4))
    θ = TSAnalytics._ms_to_natural(s3, w)
    @test θ[1:6] ≈ [0.3513716852892231, 0.3071958857184984, 0.28839620365111995,
                    0.3883257680168782, 0.18632372322584756, 0.47548495534876745] atol=1e-15
    @test TSAnalytics._ms_to_unconstrained(s3, θ)[1:6] ≈ w[1:6] atol=1e-12
    for k in (2, 3, 4)
        s = TSAnalytics._MSSpec(k, 0, true, 0, true, false, false, false, 1)
        wk = vcat(randn(MersenneTwister(k), k * (k - 1)), zeros(k + 1))
        θk = TSAnalytics._ms_to_natural(s, wk)
        @test TSAnalytics._ms_to_unconstrained(s, θk)[1:(k * (k - 1))] ≈ wk[1:(k * (k - 1))] atol=1e-12
        P = TSAnalytics._ms_P_from_probs(θk[1:(k * (k - 1))], k)
        @test vec(sum(P; dims=1)) ≈ ones(k) && all(>(0), P)        # left-stochastic
    end
    # k = 2: exp(u) / (1 + exp(u))
    s2 = TSAnalytics._MSSpec(2, 0, true, 0, true, false, false, false, 1)
    θ2 = TSAnalytics._ms_to_natural(s2, [0.5, -1.2, 0.0, 0.0, 0.0])
    @test θ2[1:2] ≈ [0.62245933, 0.23147522] atol=1e-8
end

@testset "MS: the fitted optimum (tier 2)" begin
    # among the reference's global-max hits: llf spread 2.4e-8, parameter
    # deviation 3.4e-5 -- so llf to 1e-6, parameters to 1e-4
    m = canonicalise(fit_ms_autoregression(_RGNP, 4; rng=MersenneTwister(1)))
    @test m.estimated && m.converged
    @test loglikelihood(m) ≈ -181.2633942812 atol=1e-6
    @test loglikelihood(m) >= -181.2633942812 - 1e-6
    @test coef(m) ≈ _HAM atol=1e-4
    @test expected_durations(m) ≈ [4.07604793, 10.42589261] atol=1e-2
    # the restart search reports what it found
    @test m.n_modes >= 1 && maximum(m.mode_logliks) ≈ loglikelihood(m)
    @test occursin("distinct optim", sprint(show, m)) && occursin("expected durations", sprint(show, m))
end

@testset "MS: canonicalise" begin
    # a random start returned exactly the relabelled optimum; for k = 2 the
    # swap is (a, b) -> (1 - b, 1 - a), NOT (b, a), with the means swapped
    swapped = vcat(1 - _HAM[2], 1 - _HAM[1], _HAM[4], _HAM[3], _HAM[5:end])
    @test swapped[1:2] ≈ [0.904084951552137, 0.24533568335847178]
    ms = fit_ms_autoregression(_RGNP, 4; params=swapped)
    @test loglikelihood(ms) ≈ -181.26339428124544 atol=1e-9           # same likelihood
    c = canonicalise(ms)
    @test coef(c) ≈ _HAM atol=1e-14
    @test regime_probabilities(c) ≈ regime_probabilities(ms)[:, [2, 1]] atol=1e-12
    @test coef(canonicalise(c)) == coef(c)                               # idempotent
    # the naive swap is wrong: it gives a different likelihood
    naive = vcat(_HAM[2], _HAM[1], _HAM[4], _HAM[3], _HAM[5:end])
    @test abs(loglikelihood(fit_ms_autoregression(_RGNP, 4; params=naive)) + 181.2633942812) > 1
end

@testset "MS: reductions" begin
    # 1. order 0 is the switching regression: statsmodels' MarkovRegression
    # fit, whose MarkovAutoregression(order=0).loglike agrees exactly -- and
    # whose own .fit() at order 0 is broken (defect D10)
    pr = [0.6869395711146447, 0.0898901839790639, -0.4868492498774824, 1.1042777396883068,
          0.6947516222138681]
    @test loglikelihood(fit_ms_regression(_RGNP; params=pr)) ≈ -191.28811083263912 atol=1e-10
    @test loglikelihood(fit_ms_autoregression(_RGNP, 0; params=pr)) ≈ -191.28811083263912 atol=1e-10
    mr = fit_ms_regression(_RGNP; rng=MersenneTwister(2))
    @test loglikelihood(mr) >= -191.28811083263912 - 1e-6
    # 2. k = 1 is least squares: the conditional AR(4) with a constant
    m1 = fit_ms_autoregression(_RGNP, 4; k=1, n_starts=1)
    D = hcat(ones(131), [_RGNP[(5 - l):(135 - l)] for l in 1:4]...)
    e = _RGNP[5:end] .- D * (D \ _RGNP[5:end])
    σ2 = sum(abs2, e) / 131
    @test loglikelihood(m1) ≈ -131 / 2 * (log(2pi) + log(σ2) + 1) atol=1e-6
    @test coef(m1)[2] ≈ σ2 rtol=1e-5
    # and the regression with k = 1 is OLS on a constant
    m1r = fit_ms_regression(_RGNP; k=1, n_starts=1)
    @test coef(m1r)[1] ≈ mean(_RGNP) atol=1e-6
    # 3. identical regimes: a series with no switching gives flat smoothed
    # probabilities near the steady state
    z = 1.0 .+ 0.5 .* randn(MersenneTwister(30), 300)
    mz = fit_ms_regression(z; rng=MersenneTwister(3))
    c = coef(mz)
    @test abs(c[3] - c[4]) < 0.5                       # the two means barely differ
    @test loglikelihood(mz) - (-300 / 2 * (log(2pi) + log(var(z) * 299 / 300) + 1)) < 3
end

@testset "MS: EM is a warm start, not the estimator" begin
    em = fit_ms_autoregression(_RGNP, 4; em_iter=0, n_starts=1)
    w5 = fit_ms_autoregression(_RGNP, 4; em_iter=5, n_starts=1)
    @test loglikelihood(w5) ≈ -181.2633942812 atol=1e-5
    @test loglikelihood(w5) >= loglikelihood(em) - 1e-6
    # pure EM: many steps, no gradient polish, from the same start
    s = fit_ms_autoregression(_RGNP, 4; params=_HAM).spec
    st = TSAnalytics._ms_states(2, 4)
    θ = TSAnalytics._ms_start(s, _RGNP, zeros(135, 0))
    for _ in 1:200
        θ = TSAnalytics._ms_em_step(s, θ, st, _RGNP, zeros(135, 0), ones(135, 1))
    end
    lem = TSAnalytics._ms_filter(s, θ, st, _RGNP, zeros(135, 0), ones(135, 1))[1]
    @test isfinite(lem) && lem <= -181.2633942812 + 1e-6
    # the EM M-step weights by smoothed probabilities, which reach zero:
    # _ols must accept that (Stage 11.0e)
    @test minimum(regime_probabilities(fit_ms_autoregression(_RGNP, 4; params=_HAM))) < 1e-3
end

@testset "MS: a switching variance -- the better optimum is reachable" begin
    # the reference's default fit is a local optimum (-182.0369); its best
    # over 50 restarts is -180.67728501727882
    m = fit_ms_autoregression(_RGNP, 4; switching_variance=true, n_starts=20, rng=MersenneTwister(4))
    @test dof(m) == 10
    @test loglikelihood(m) >= -180.67728501727882 - 1e-4
    @test coefnames(m)[5:6] == ["sigma2[0]", "sigma2[1]"]
end

@testset "MS: time-varying transition probabilities" begin
    # our own ground truth: statsmodels MarkovAutoregression(rgnp, order=1,
    # exog_tvtp=[1, (t mod 17) < 3]).loglike at these parameters
    t = 0:134
    Z = hcat(ones(135), Float64.((t .% 17) .< 3))
    p = [1.2, -2.0, -0.8, 1.5, -0.3, 1.1, 0.6, 0.1]
    m = fit_ms_autoregression(_RGNP, 1; exog_tvtp=Z, params=p)
    @test coefnames(m)[1:4] == ["p[0->0].tvtp0", "p[1->0].tvtp0", "p[0->0].tvtp1", "p[1->0].tvtp1"]
    @test loglikelihood(m) ≈ -191.2166000879302 atol=1e-9
    @test sum(m.smoothed[:, 2]) ≈ 92.73589168626401 atol=1e-9
    @test sum(m.filtered[:, 2]) ≈ 92.81469375902552 atol=1e-9
    @test m.P === nothing
    @test_throws ArgumentError expected_durations(m)
    # the transition matrix genuinely varies with the covariate
    u = TSAnalytics._ms_unpack(m.spec, coef(m))
    P0 = TSAnalytics._ms_P_tvtp(u.tr, 2, 2, [1.0, 0.0]); P1 = TSAnalytics._ms_P_tvtp(u.tr, 2, 2, [1.0, 1.0])
    @test !(P0 ≈ P1) && vec(sum(P1; dims=1)) ≈ [1.0, 1.0]
end

@testset "MS: refusals and log-space safety" begin
    e = try fit_ms_autoregression(_RGNP, 4; switching_trend=false) catch err; err end
    @test e isa ArgumentError && occursin("nothing switches", sprint(showerror, e))
    @test_throws ArgumentError fit_ms_regression(_RGNP; trend=false)
    @test_throws ArgumentError fit_ms_regression(_RGNP; k=0)
    @test_throws ArgumentError fit_ms_autoregression(_RGNP, -1)
    @test_throws ArgumentError fit_ms_regression(_RGNP; params=[0.5])
    @test_throws ArgumentError fit_ms_regression(_RGNP; initial=[0.5, 0.6])
    @test_throws ArgumentError fit_ms_regression(vcat(_RGNP, NaN))
    # a well-separated 2000-point series: no underflow in log space
    rng = MersenneTwister(5)
    reg = [isodd(fld(t, 100)) ? 2 : 1 for t in 1:2000]
    y = [r == 1 ? -3.0 : 3.0 for r in reg] .+ 0.3 .* randn(rng, 2000)
    m = fit_ms_regression(y; params=[0.99, 0.01, -3.0, 3.0, 0.09])
    @test isfinite(loglikelihood(m)) && all(isfinite, m.smoothed)
    @test mean(abs.(m.smoothed[:, 2] .- (reg .== 2))) < 0.01
end
