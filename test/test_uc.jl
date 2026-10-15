using DelimitedFiles

# Stage 11.1 -- unobserved components. handoff/stage 11/docs/
# stage-11.1-unobserved-components-handoff.md §7. Primary reference:
# statsmodels 0.15.0 UnobservedComponents (BSD, transcribable). R's StructTS
# is GPL and serves only as an independent check.

const _UCFX = joinpath(@__DIR__, "..", "handoff", "stage 11", "fixtures")
_ucfx(f) = Float64.(readdlm(joinpath(_UCFX, f), ','; skipstart=1))
const _NILE = vec(_ucfx("nile.csv"))

@testset "UC: local level log-likelihood at fixed parameters (tier 1)" begin
    # rows 1 and 2: statsmodels' approximate-diffuse prior (a0 = 0, P0 = 1e6),
    # the first period left out (one per diffuse state). Row 3: R's prior,
    # a0 = y[1], P0 = 1e6 var(y)/100. Rows 1 and 3 differ ONLY in the prior,
    # by 0.0079 -- the pair that catches an initialisation bug.
    y = _NILE
    m1 = uc_local_level(y; params=[15098.577154, 1469.146619])
    @test loglikelihood(m1) ≈ -632.537695 atol=1e-6
    m2 = uc_local_level(y; params=[15108.3155, 1463.5471])
    @test loglikelihood(m2) ≈ -632.53768559 atol=1e-8
    P0r = fill(1e6 * var(y) / 100, 1, 1)
    @test P0r[1] ≈ 2.86379e8 rtol=1e-5
    m3 = uc_local_level(y; params=[15098.577154, 1469.146619], initial_state=[1120.0],
                        initial_cov=P0r)
    @test loglikelihood(m3) ≈ -632.5456059039 atol=1e-9
    @test loglikelihood(m1) - loglikelihood(m3) ≈ 0.0079 atol=1e-4
    @test m1.burn == 1 && m3.burn == 1 && nobs(m1) == 99
    @test !m1.estimated && coef(m1) == [15098.577154, 1469.146619]
    @test coefnames(m1) == ["sigma2.irregular", "sigma2.level"]
end

@testset "UC: filtered and smoothed state against R's KalmanRun/tsSmooth" begin
    # uc_R_filter.csv: R's own filtered and smoothed level at its fitted
    # variances and its prior. R prints 7 significant digits, hence 1e-7.
    y = _NILE
    R = _ucfx("uc_R_filter.csv")                  # resid, filt, smooth
    m = uc_local_level(y; params=[15098.577154, 1469.146619], initial_state=[1120.0],
                       initial_cov=fill(1e6 * var(y) / 100, 1, 1))
    d = uc_decomposition(m)
    @test maximum(abs.(d.level .- R[:, 3])) < 1e-7
    # for a random walk the filtered level at t is the prediction for t+1
    filt = fitted(m)[2:end]
    @test maximum(abs.(filt .- R[1:(end - 1), 2])) < 1e-7
    # R's first innovation is identically zero (its prior mean is y[1])
    @test R[1, 1] == 0 && residuals(m)[1] == 0
    @test isnan(d.level_se[1]) && all(>(0), d.level_se[2:end]) && all(isfinite, d.level_se[2:end])
end

@testset "UC: fitted local level, tier-2 tolerances" begin
    # The Nile optimum is identified to ~3 significant figures: four
    # converged optimisers spread 1.4e-4 relative; R's point is 3.8e-3 away on
    # sigma2.level for a log-likelihood difference of 9.7e-6. So: variances to
    # 1e-2 relative, log-likelihood to 1e-5.
    m = uc_local_level(_NILE)
    p = coef(m)
    @test p[1] ≈ 15108.3155 rtol=1e-2
    @test p[2] ≈ 1463.5471 rtol=1e-2
    @test loglikelihood(m) ≈ -632.53768559 atol=1e-5
    @test m.converged && m.estimated
    # one-sided: our optimum is no worse than the reference's parameters
    @test loglikelihood(m) >= loglikelihood(uc_local_level(_NILE; params=[15098.577154, 1469.146619])) - 1e-9
    @test loglikelihood(m) >= -632.53768559 - 1e-6
    se = stderror(m)
    @test length(se) == 2 && all(>(0), se)
end

@testset "UC: state-space matrices, component by component" begin
    # §2.2 local linear trend
    m = fit_uc(_NILE; level=:stochastic, trend=:stochastic, params=[1.0, 2.0, 3.0])
    @test m.T == [1.0 1.0; 0.0 1.0] && m.Z == [1.0 0.0] && m.R == [1.0 0.0; 0.0 1.0]
    @test m.Q == [2.0 0.0; 0.0 3.0] && m.H == fill(1.0, 1, 1)
    # §2.3 dummy seasonal s = 4 with a local linear trend: k_states 5, R is 5x3
    m = uc_bsm(_NILE, 4; params=[1.0, 2.0, 3.0, 4.0])
    @test m.T == [1 1 0 0 0; 0 1 0 0 0; 0 0 -1 -1 -1; 0 0 1 0 0; 0 0 0 1 0]
    @test m.Z == [1.0 0 1 0 0]
    @test size(m.R) == (5, 3) && m.R' == [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0]
    @test coefnames(m) == ["sigma2.irregular", "sigma2.level", "sigma2.trend", "sigma2.seasonal"]
    @test m.burn == 5
    # §2.4 trigonometric s = 4, 2 harmonics: blocks [0 1; -1 0] and [-1 0; 0 -1]
    m = fit_uc(_NILE; level=:stochastic, seasonal=4, seasonal_kind=:trig, params=[1.0, 2.0, 3.0])
    @test m.T[2:5, 2:5] ≈ [0 1 0 0; -1 0 0 0; 0 0 -1 0; 0 0 0 -1] atol=1e-15
    @test m.Z == [1.0 1 0 1 0]
    @test coefnames(m)[3] == "sigma2.freq_seasonal_4(2)"
    @test size(m.R) == (5, 5) && all(m.Q[i, i] == 3.0 for i in 2:5)
    @test m.burn == 5                                    # statsmodels counts all 5
    # §2.5 damped stochastic cycle
    m = fit_uc(_NILE; level=:stochastic, cycle=:stochastic, damped_cycle=true,
               params=[1.0, 2.0, 3.0, pi / 3, 0.8])
    @test m.T[2:3, 2:3] ≈ 0.8 .* [cos(pi / 3) sin(pi / 3); -sin(pi / 3) cos(pi / 3)]
    @test m.Z == [1.0 1 0]
    @test coefnames(m)[3:5] == ["sigma2.cycle", "frequency.cycle", "damping.cycle"]
    # §2.6 AR(2): a companion block, one shock, stationary prior
    m = fit_uc(_NILE; level=:stochastic, ar=2, params=[1.0, 2.0, 3.0, -0.0772, -0.9185])
    @test m.T[2:3, 2:3] == [-0.0772 -0.9185; 1.0 0.0]
    @test size(m.R) == (3, 2) && m.Z == [1.0 1 0]
    @test m.burn == 1                                    # the AR state is not diffuse
    @test coefnames(m) == ["sigma2.irregular", "sigma2.level", "sigma2.ar", "ar.L1", "ar.L2"]
    # exog: fixed coefficients, beta.x1
    X = reshape(collect(1.0:100.0), :, 1)
    m = fit_uc(_NILE; level=:stochastic, exog=X, params=[1.0, 2.0, 0.5])
    @test coefnames(m)[end] == "beta.x1" && m.burn == 1
end

@testset "UC: partitioned initialisation (Stage 11.0c)" begin
    # level + trend + seasonal(4) + damped cycle + AR(1): 8 states, the first
    # 7 approximate diffuse at 1e6, the AR state stationary -- the reference's
    # own blocks, measured in s11/uc_init.py
    m = fit_uc(_NILE; level=:stochastic, trend=:stochastic, seasonal=4, cycle=:stochastic,
               damped_cycle=true, ar=1,
               params=[1.0, 1.0, 1.0, 1.0, 1.0, pi / 4, 0.9, 2.0, 0.5])
    L = m.layout
    @test L.r == 8 && L.ar == 8:8 && m.burn == 7
    a0, P0, didx = TSAnalytics._resolve_init(m.T, m.R * m.Q * m.R', TSAnalytics._uc_init(L, 1e6))
    @test all(P0[i, i] == 1e6 for i in 1:7)
    @test P0[8, 8] ≈ 2.0 / (1 - 0.5^2)                  # AR(1) stationary variance
    @test all(iszero, P0[1:7, 8]) && isempty(didx) && all(iszero, a0)
    # guards
    T = [1.0 0.0; 0.0 0.5]; RQR = [1.0 0.0; 0.0 1.0]
    @test_throws ArgumentError TSAnalytics._resolve_init(T, RQR, TSAnalytics.SSMInit([1:1 => :approximate_diffuse]))
    @test_throws ArgumentError TSAnalytics._resolve_init(T, RQR,
        TSAnalytics.SSMInit([1:2 => :approximate_diffuse, 2:2 => :stationary]))
    @test_throws ArgumentError TSAnalytics._resolve_init(T, RQR,
        TSAnalytics.SSMInit([1:1 => :stationary, 2:2 => :stationary]))   # unit root
    @test_throws ArgumentError TSAnalytics._resolve_init(T, RQR,
        TSAnalytics.SSMInit([1:2 => :known]))                             # no P0_known
    # a diffuse state may load a stationary one (T[1,2]); only the reverse is refused
    _, Pd, _ = TSAnalytics._resolve_init([1.0 0.3; 0.0 0.5], RQR,
        TSAnalytics.SSMInit([1:1 => :approximate_diffuse, 2:2 => :stationary]))
    @test Pd[2, 2] ≈ 1 / (1 - 0.25)
    # a stationary block driven from outside it is refused
    @test_throws ArgumentError TSAnalytics._resolve_init([1.0 0.0; 0.3 0.5], RQR,
        TSAnalytics.SSMInit([1:1 => :approximate_diffuse, 2:2 => :stationary]))
    _, Pk, _ = TSAnalytics._resolve_init(T, RQR,
        TSAnalytics.SSMInit([1:1 => :known, 2:2 => :exact_diffuse]; P0_known=[7.0 0; 0 0]))
    @test Pk[1, 1] == 7.0
    # stationary_cov is unchanged: it now wraps the plain-matrix core
    ssm = build_statespace([0.5, -0.2], [0.3])
    Q0, ok = stationary_cov(ssm)
    Q1, ok1 = TSAnalytics._stationary_cov(ssm.T, ssm.R * ssm.R')
    @test ok && ok1 && Q0 == Q1
end

@testset "UC: the UKgas boundary fixture (tier 3)" begin
    # log10(UKgas), BSM. The level variance sits on its zero boundary and the
    # rest are weakly identified: R and statsmodels differ 11.5x on the trend
    # variance and both are right. Assert only what both agree on.
    y = vec(_ucfx("ukgas.csv"))                         # already log10
    m = uc_bsm(y, 4)
    p = Dict(zip(coefnames(m), coef(m)))
    @test p["sigma2.level"] < 1e-6
    @test p["sigma2.seasonal"] ≈ 6.245776e-04 rtol=0.2
    @test p["sigma2.irregular"] ≈ 3.433252e-04 rtol=0.2
    @test loglikelihood(m) >= 172.46517259 - 1e-6
    # NO CLIPPING: the boundary estimate is a small positive number, not 0
    # (R's StructTS reports exactly 0 via pmax(coef * vx, 0) -- defect D3 --
    # so a boundary and a clipped negative look the same there)
    @test p["sigma2.level"] > 0
    @test all(>(0), coef(m))
    # and the convergence flag is reported, not hidden
    @test m.estimated
    @test occursin("optimiser", sprint(show, m))
    # the information matrix at the estimate is still invertible here
    @test all(isfinite, stderror(m))
end

@testset "UC: reductions onto things already verified" begin
    y = _NILE
    # 1. deterministic level + irregular = OLS on a constant
    m = fit_uc(y; level=:deterministic, init_kappa=1e12)
    d = uc_decomposition(m)
    @test maximum(abs.(d.level .- mean(y))) < 1e-6
    @test coef(m)[1] ≈ var(y) rtol=1e-6                 # residual variance, ddof = 1
    # 2. sigma2.level -> 0: a constant mean
    m = uc_local_level(y; fixed=Dict("sigma2.level" => 1e-12))
    d = uc_decomposition(m)
    @test maximum(d.level) - minimum(d.level) < 1e-6 * mean(y)
    # 3. sigma2.irregular -> 0: the level is the series
    m = uc_local_level(y; fixed=Dict("sigma2.irregular" => 1e-12))
    @test maximum(abs.(uc_decomposition(m).level .- y)) < 1e-4
    # 4. dummy and full-harmonic trig seasonals span the same space (odd
    # period: no degenerate sine state, so the burn counts agree too)
    s5 = 100 .+ 10 .* sin.(2pi .* (1:100) ./ 5) .+ randn(MersenneTwister(5), 100)
    md = fit_uc(s5; level=:stochastic, seasonal=5, seasonal_stochastic=false)
    mt = fit_uc(s5; level=:stochastic, seasonal=5, seasonal_kind=:trig, seasonal_stochastic=false)
    @test loglikelihood(md) ≈ loglikelihood(mt) atol=1e-4
    @test md.burn == mt.burn == 5
    # 5. an AR(p) component alone, no level, no irregular, is fit_arma
    x = vec(_ucfx("nile.csv")) .- mean(y)
    fit_ar_uc = fit_uc(x; ar=2, irregular=false)
    fa = fit_arma(x, (2, 0); include_mean=false)
    @test loglikelihood(fit_ar_uc) ≈ fa.loglik atol=1e-4
    @test fit_ar_uc.burn == 0
end

@testset "UC: specification refusals" begin
    y = _NILE
    e = try fit_uc(y; level=:deterministic, irregular=false) catch err; err end
    @test e isa ArgumentError && occursin("no disturbance", sprint(showerror, e))
    @test_throws ArgumentError fit_uc(y; irregular=false)
    @test_throws ArgumentError fit_uc(y; level=true)          # no Boolean shortcut
    @test_throws ArgumentError fit_uc(y; level=:local)
    @test_throws ArgumentError fit_uc(y; trend=:stochastic)   # a trend needs a level
    @test_throws ArgumentError fit_uc(y; level=:stochastic, seasonal=1)
    @test_throws ArgumentError fit_uc(y; level=:stochastic, seasonal=4, seasonal_kind=:trig, harmonics=3)
    @test_throws ArgumentError fit_uc(y; level=:stochastic, seasonal=4, harmonics=2)
    @test_throws ArgumentError fit_uc(y; level=:stochastic, ar=-1)
    @test_throws ArgumentError fit_uc(y; level=:stochastic, damped_cycle=true)
    @test_throws ArgumentError fit_uc(y; level=:stochastic, exog=ones(5))
    @test_throws ArgumentError fit_uc(y; level=:stochastic, params=[1.0])
    @test_throws ArgumentError fit_uc(y; level=:stochastic, fixed=Dict("sigma2.trend" => 1.0))
    @test_throws ArgumentError fit_uc(y; level=:stochastic, initial_state=[1.0])
    @test_throws ArgumentError fit_uc(y[1:3]; level=:stochastic, trend=:stochastic, seasonal=4)
end

@testset "UC: decomposition sums to the series; standard errors" begin
    y = vec(_ucfx("ukgas.csv"))
    m = uc_bsm(y, 4; params=[3.433252e-04, 2e-10, 1.503e-06, 6.245776e-04])
    d = uc_decomposition(m)
    @test d.names == [:level, :trend, :seasonal, :irregular] && d.period == 4
    @test d.level .+ d.seasonal .+ d.irregular ≈ y rtol=1e-12
    # standard errors: NaN over the burn (the prior dominates and the
    # smoothed variance is cancellation noise), positive and finite after
    b = m.burn
    @test b == 5 && all(isnan, d.level_se[1:b]) && all(isnan, d.seasonal_se[1:b])
    @test all(>(0), d.seasonal_se[(b + 1):end]) && all(isfinite, d.level_se[(b + 1):end])
    @test all(isfinite, d.trend_se[(b + 1):end])
    @test d.cycle === nothing && d.ar === nothing && d.regression === nothing
    # with an AR component and a regressor the sum still closes
    X = reshape(sin.(1:length(y)), :, 1)
    m2 = fit_uc(y; level=:stochastic, ar=1, exog=X, params=[1e-4, 1e-4, 1e-3, 0.5, 0.01])
    d2 = uc_decomposition(m2)
    @test d2.level .+ d2.ar .+ d2.regression .+ d2.irregular ≈ y rtol=1e-12
    @test d2.regression ≈ 0.01 .* X[:, 1]
end

@testset "UC: missing data" begin
    y = copy(_NILE)
    ym = copy(y); ym[[20, 21, 50, 77]] .= NaN
    m = uc_local_level(ym; params=[15098.577154, 1469.146619])
    @test nobs(m) == 95
    @test isfinite(loglikelihood(m)) && loglikelihood(m) > loglikelihood(uc_local_level(y; params=[15098.577154, 1469.146619]))
    d = uc_decomposition(m)
    @test all(isfinite, d.level)
    @test d.level_se[20] > d.level_se[10]               # less information in the gap
    @test isnan(fitted(m)[50]) && isnan(residuals(m)[50])
    mf = uc_local_level(ym)
    @test mf.converged && all(>(0), coef(mf))
end

@testset "UC: forecasting" begin
    m = uc_local_level(_NILE)
    f = forecast(m, 10)
    @test f isa Forecast && length(f.point) == 10
    @test all(==(f.point[1]), f.point)                  # a random walk's forecast is flat
    @test issorted(f.se) && f.se[10] > f.se[1]          # a diffuse state's variance grows
    @test f.point[1] ≈ uc_decomposition(m).level[end] rtol=1e-10
    p = coef(m)
    @test f.se[2]^2 - f.se[1]^2 ≈ p[2] rtol=1e-8        # one more step adds sigma2.level
    @test predict(m, 10).point == f.point                # exact alias
    # a deterministic level adds no forecast variance beyond the irregular
    md = fit_uc(_NILE; level=:deterministic)
    fd = forecast(md, 5)
    @test fd.se[5] ≈ fd.se[1] rtol=1e-6
    # regressors need their future values
    X = reshape(collect(1.0:100.0), :, 1)
    mx = fit_uc(_NILE; level=:stochastic, exog=X, params=[15000.0, 1500.0, 0.5])
    @test_throws ArgumentError forecast(mx, 3)
    fx = forecast(mx, [101.0, 102.0, 103.0], 3)
    @test fx.point[2] - fx.point[1] ≈ 0.5 rtol=1e-10
    @test_throws ArgumentError forecast(m, [1.0], 1)
end

@testset "UC: show reports the specification and convergence" begin
    m = uc_local_level(_NILE)
    s = sprint(show, m)
    @test occursin("stochastic level + irregular", s)
    @test occursin("optimiser converged", s)
    @test occursin("left out for the diffuse initialisation", s)
    s2 = sprint(show, uc_local_level(_NILE; params=[1.0, 1.0]))
    @test occursin("not estimated", s2)
end

@testset "UC: several starts escape a flat ridge, deterministically under threads" begin
    # log(jj), BSM with a trigonometric seasonal: from the single default start
    # the level and seasonal log-variances slid towards -Inf (llf 56.35, not
    # converged). statsmodels 0.15 reaches 80.5249346684875 (bfgs; lbfgs
    # 80.52492703547085) at [7.75980e-04, 1.20238e-03, 6.26000e-06, 2.61360e-04].
    jj = log.(dataset("jj").value)
    kw = (level=:stochastic, trend=:stochastic, seasonal=4, seasonal_kind=:trig)
    m = fit_uc(jj; kw...)
    @test m.loglik >= 80.5249346684875 - 1e-9
    @test m.converged
    @test m.params ≈ [7.75980e-04, 1.20238e-03, 6.26000e-06, 2.61360e-04] rtol = 1e-2
    # our likelihood at statsmodels' printed optimum (8-decimal parameters)
    @test fit_uc(jj; kw..., params=[7.75980e-04, 1.20238e-03, 6.26000e-06, 2.61360e-04]).loglik ≈
          80.5249346684875 atol = 1e-5
    # the starts run on threads; the answer must not depend on scheduling
    @test all(_ -> fit_uc(jj; kw...).params == m.params, 1:4)
end

@testset "UC: Tsay (2010) Example 11.3, log J&J, reproduced" begin
    # Tsay, Analysis of Financial Time Series 3rd ed., printed p. 606: local
    # level + dummy seasonal (s = 4) on log quarterly J&J EPS, S-PLUS/SsfPack.
    # Standard deviations sigma_eta = 7.269655e-02, sigma_omega = 2.931691e-02;
    # statsmodels gives 7.269789203e-02 and 2.931638067e-02 (1.8e-5 from Tsay).
    # The irregular is at its boundary in all three and is not asserted.
    y = log.(dataset("jj").value)
    m = fit_uc(y; level=:stochastic, seasonal=4)
    sd = Dict(zip(m.layout.names, sqrt.(m.params)))
    @test sd["sigma2.level"] ≈ 7.269655e-02 rtol = 1e-5
    @test sd["sigma2.seasonal"] ≈ 2.931691e-02 rtol = 1e-5
    @test sd["sigma2.level"] ≈ 7.269789203e-02 rtol = 5e-5
    @test sd["sigma2.seasonal"] ≈ 2.931638067e-02 rtol = 5e-5
    @test sd["sigma2.irregular"] < 1e-4
    @test m.converged
end

@testset "UC: a variance fixed at exactly zero" begin
    # Holding a variance at 0 leaves the likelihood finite but makes its
    # ForwardDiff derivative NaN; unguarded, the NaN slope broke the line
    # search (AssertionError). The free fit puts the irregular at the
    # boundary anyway, so both reach the same optimum.
    y = log.(dataset("jj").value)
    free = fit_uc(y; level=:stochastic, seasonal=4)
    zero = fit_uc(y; level=:stochastic, seasonal=4, fixed=Dict("sigma2.irregular" => 0.0))
    @test zero.params[1] == 0.0
    @test zero.converged
    @test zero.loglik ≈ free.loglik atol = 1e-6
    @test zero.params[2:3] ≈ free.params[2:3] rtol = 1e-4
end
