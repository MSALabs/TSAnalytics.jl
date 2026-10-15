using RecipesBase
using DelimitedFiles
using Statistics: median, quantile

# Stage 11 plot data and recipes. handoff/stage 11/docs/
# stage-11-plot-recipes-handoff.md §7: test the data, not the rendering.
# (All thirteen were rendered with GR and inspected once by hand.)

const _P11FX = joinpath(@__DIR__, "..", "handoff", "stage 11", "fixtures")
_p11(f) = Float64.(readdlm(joinpath(_P11FX, f), ','; skipstart=1))
_rec(x; kw...) = RecipesBase.apply_recipe(Dict{Symbol,Any}(kw...), x)
_subplots(rs) = [get(r.plotattributes, :subplot, 1) for r in rs]

const _P11_NILE = vec(_p11("nile.csv")[:, end])
const _P11_LYNX = log10.(_p11("fixture_lynx.csv")[:, 2])
const _P11_GNP = Float64.(readdlm(joinpath(_P11FX, "hamilton_rgnp.csv"), ','; skipstart=1)[:, 2])
const _P11_YA = standardize(_p11("fixtureA.csv"))
const _P11_SMA = (lam=[0.7009068155913395, 0.6042910774398366, -0.5753739335367198,
                       0.5237349698498732, 0.6650726532699288],
                  h=[0.14812824410505457, 0.36549749235533713, 0.42432314355129,
                     0.5221441574104723, 0.23249862054434853],
                  A=0.6482477536949357,
                  r2=[0.9108626308095913, 0.6770641675569274, 0.6137905314487733,
                      0.5085776162194486, 0.8201144425650966])

@testset "StateDecomposition: the exponential-smoothing form is unchanged" begin
    y = 100 .+ 10 .* sin.(2π .* (1:96) ./ 12) .+ (1:96) ./ 4
    d = state_decomposition(fit_ets(y, 12; error=:add, trend=:add, seasonal=:add))
    @test d.level_se === nothing && d.slope_se === nothing
    @test d.seasonal_se == [nothing]
    @test isempty(d.extra)
    @test d.fitted === nothing
    rs = _rec(d)
    # four panels, one series each, no band, no legend
    @test length(rs) == 4
    @test _subplots(rs) == [1, 2, 3, 4]
    @test all(!haskey(r.plotattributes, :fillrange) for r in rs)
    @test all(!haskey(r.plotattributes, :label) for r in rs)
    @test [r.plotattributes[:ylabel] for r in rs] == ["observed", "level", "slope", "season"]
    @test rs[1].args[2] == d.observed
end

@testset "StateDecomposition for UC: components, bands, fitted" begin
    m = uc_local_level(_P11_NILE)
    d = state_decomposition(m)
    u = uc_decomposition(m)
    @test d.level == u.level
    @test isequal(d.level_se, u.level_se)
    @test d.slope === nothing
    @test first.(d.extra) == ["irregular"]
    @test d.extra[1][3] === nothing           # the irregular gets no band
    @test d.observed == _P11_NILE
    @test all(isnan, d.fitted[1:m.burn])
    @test d.fitted[(m.burn + 1):end] == m.fitted[(m.burn + 1):end]
    rs = _rec(d)
    # observed + fitted, level band + line, irregular
    @test length(rs) == 5
    @test _subplots(rs) == [1, 1, 2, 2, 3]
    band = rs[3]
    @test band.plotattributes[:label] == "95% interval, smoothed component"
    # band arithmetic by hand, from the smoother's covariance, at three periods
    for t in (2, 50, 100)
        se = sqrt(m.V[t][1, 1])
        @test band.args[2][t] ≈ m.alpha[1, t] - 1.96 * se rtol = 1e-12
        @test band.plotattributes[:fillrange][t] ≈ m.alpha[1, t] + 1.96 * se rtol = 1e-12
    end
    @test isnan(band.args[2][1])                 # the diffuse period has no band
    @test rs[2].plotattributes[:label] == "one-step fitted"
    @test rs[2].plotattributes[:ylabel] == "observed"
    @test !haskey(rs[5].plotattributes, :fillrange)
    # the full BSM: level, slope, seasonal (all banded), irregular
    gas = log.(vec(_p11("ukgas.csv")[:, end]))
    db = state_decomposition(uc_bsm(gas, 4))
    @test db.slope !== nothing && db.slope_se !== nothing
    @test length(db.seasonal) == 1 && db.seasonal_se[1] !== nothing
    @test db.periods == [4.0]
    rb = _rec(db)
    @test count(r -> haskey(r.plotattributes, :fillrange), rb) == 3
    @test maximum(_subplots(rb)) == 5
end

@testset "diagnostic_plot and ts_display dispatch on the new models" begin
    m = uc_local_level(_P11_NILE)
    r = diagnostic_plot(m)
    @test length(r.std_resid) == 100 - m.burn
    s = fit_setar(_P11_LYNX, 2; thDelay=1, warn=false)
    @test diagnostic_plot(s).std_resid ≈ diagnostic_plot(residuals(s)).std_resid
    l = fit_lstar(_P11_LYNX, 2; thDelay=1)
    @test sum(abs2, residuals(l)) ≈ l.ssr rtol = 1e-10
    @test length(fitted(l)) == 112
    @test length(diagnostic_plot(l).std_resid) == 112
    @test ts_display(s).x == _P11_LYNX
    @test ts_display(l).x == _P11_LYNX
end

@testset "FactorPathResult: orientation, band, overlay" begin
    m = fit_dfm(_P11_YA, 1; params=(reshape(_P11_SMA.lam, 5, 1), fill(_P11_SMA.A, 1, 1), _P11_SMA.h))
    r = factor_paths(m)
    n = size(_P11_YA, 1)
    @test size(r.factors) == (n, 1)              # observations × factors
    @test size(r.se) == (n, 1)
    @test r.factors == factors(m)[1]
    @test r.se == factors(m)[2]
    @test size(r.series) == size(_P11_YA)
    rs = _rec(r)
    @test length(rs) == 2
    for t in (1, 100, n)
        @test rs[1].args[2][t] ≈ r.factors[t, 1] - 1.96 * r.se[t, 1] rtol = 1e-12
        @test rs[1].plotattributes[:fillrange][t] ≈ r.factors[t, 1] + 1.96 * r.se[t, 1] rtol = 1e-12
    end
    ro = _rec(r; overlay_series=true)
    @test length(ro) == 5 + 1
    @test count(x -> get(x.plotattributes, :color, nothing) == :gray, ro) == 5
end

@testset "LoadingsResult and FactorR2Result" begin
    m = fit_dfm(_P11_YA, 1; params=(reshape(_P11_SMA.lam, 5, 1), fill(_P11_SMA.A, 1, 1), _P11_SMA.h))
    L = loadings_plot(m)
    @test L.loadings == m.Lambda
    @test length(L.series_names) == 5
    rl = _rec(L)
    @test rl[1].plotattributes[:seriestype] == :heatmap
    a = maximum(abs, m.Lambda)
    @test rl[1].plotattributes[:clims] == (-a, a)     # centred at zero
    R = factor_r2_plot(m)
    # statsmodels' coefficients_of_determination at its own parameters
    @test vec(R.per_factor) ≈ _P11_SMA.r2 rtol = 1e-6
    @test R.total == series_r2(m)
    # one factor: the per-factor R² regresses on f with a free slope, the joint
    # one uses Λ itself, so least squares can only do at least as well
    @test all(R.total .<= vec(R.per_factor) .+ 1e-12)
    rr = _rec(R)
    @test length(rr) == 2                             # one bar series per factor, one marker
    @test rr[1].args[2] == R.per_factor[:, 1]
end

@testset "FactorSelectionResult" begin
    YB = standardize(_p11("fixtureB.csv"))
    r = factor_selection_plot(YB, 4)
    ic = bai_ng_ic(YB, 4)
    @test r.IC1 == ic.IC1 && r.IC2 == ic.IC2 && r.IC3 == ic.IC3
    @test r.r == 1:4
    @test length(r.eigenvalues) == 6
    @test sum(r.eigenvalues) ≈ 6 rtol = 1e-12          # trace of a correlation matrix
    @test issorted(r.eigenvalues; rev=true)
    # fixture B is built with two factors, and the criteria keep falling
    @test issorted(r.IC2; rev=true)
    rs = _rec(r)
    @test length(rs) == 4
    @test occursin("not asymptotic", rs[2].plotattributes[:title])
end

@testset "FactorResidualResult: the five-number summary" begin
    m = fit_dfm(_P11_YA, 1; params=(reshape(_P11_SMA.lam, 5, 1), fill(_P11_SMA.A, 1, 1), _P11_SMA.h))
    r = factor_residual_plot(m)
    E = _P11_YA .- r.residuals
    @test E ≈ m.factors * m.Lambda' rtol = 1e-12
    for i in 1:5
        e = r.residuals[:, i]
        @test r.median[i] ≈ median(e)
        @test r.q1[i] ≈ quantile(e, 0.25)
        @test r.q3[i] ≈ quantile(e, 0.75)
        @test r.whisker_low[i] >= r.q1[i] - 1.5 * (r.q3[i] - r.q1[i])
        @test r.whisker_high[i] <= r.q3[i] + 1.5 * (r.q3[i] - r.q1[i])
        @test length(r.outliers[i]) == count(x -> x < r.whisker_low[i] || x > r.whisker_high[i], e)
    end
    rs = _rec(r)
    @test rs[1].plotattributes[:seriestype] == :shape
    @test length(rs) == (any(!isempty, r.outliers) ? 3 : 2)
end

@testset "RegimeProbabilityResult" begin
    m = canonicalise(fit_ms_autoregression(_P11_GNP, 4; rng=MersenneTwister(1)))
    r = regime_plot(m)
    @test size(r.filtered) == size(r.smoothed) == (131, 2)
    @test r.t == 5:135
    @test r.y == _P11_GNP[5:end]
    @test r.smoothed == regime_probabilities(m)
    @test r.filtered == regime_probabilities(m; kind=:filtered)
    # the last filtered and smoothed coincide (0.92771603 on the handoff's fit)
    @test r.filtered[end, :] ≈ r.smoothed[end, :] rtol = 1e-12
    @test r.durations ≈ [4.07604793, 10.42589261] rtol = 1e-4
    rs = _rec(r)
    @test length(rs) == 4
    @test _subplots(rs) == [1, 1, 2, 2]
    @test occursin("4.08", rs[1].plotattributes[:title])
    rsh = _rec(r; mode=:shaded, regime=1)
    @test rsh[1].plotattributes[:seriestype] == :vspan
    e = rsh[1].args[1]
    @test iseven(length(e)) && issorted(e)
    # every period inside a span has Pr > 0.5, every one outside has Pr <= 0.5
    inside(t) = any(e[2i - 1] < t < e[2i] for i in 1:(length(e) ÷ 2))
    @test all(inside(r.t[i]) == (r.smoothed[i, 1] > 0.5) for i in eachindex(r.t))
    @test_throws ArgumentError _rec(r; mode=:bars)
    @test_throws ArgumentError _rec(r; mode=:shaded, regime=3)
end

@testset "MS residuals and per-regime diagnostics" begin
    m = canonicalise(fit_ms_autoregression(_P11_GNP, 4; rng=MersenneTwister(1)))
    e = residuals(m)
    @test length(e) == 131
    @test fitted(m) .+ e ≈ _P11_GNP[5:end]
    E, pooled = TSAnalytics._ms_conditional_resid(m)
    @test pooled ≈ vec(sum(m.smoothed .* E; dims=2)) rtol = 1e-10
    # k = 1 is least squares on the AR(4): the residual is the OLS residual
    m1 = fit_ms_autoregression(_P11_GNP, 4; k=1)
    y = _P11_GNP
    X = hcat(ones(131), [y[(5 - l):(end - l)] for l in 1:4]...)
    ols = y[5:end] .- X * (X \ y[5:end])
    @test residuals(m1) ≈ ols atol = 1e-5
    d1 = diagnostic_plot(m; regime=1)
    @test length(d1.std_resid) == 131
    @test diagnostic_plot(m).std_resid ≈ diagnostic_plot(e).std_resid
    @test_throws ArgumentError diagnostic_plot(m; regime=3)
    @test ts_display(m).x == _P11_GNP
end

@testset "ThresholdRegimeResult: the boundary observation" begin
    s = fit_setar(_P11_LYNX, 2; thDelay=1, warn=false)
    r = threshold_regime_plot(s)
    @test r.t == 3:114
    @test r.y == _P11_LYNX[3:end]
    @test r.z == _P11_LYNX[1:112]
    @test count(==(1), r.regime) == 78
    @test r.regime[63] == 1 && r.z[63] == r.thresholds[1]
    rs = _rec(r)
    @test length(rs) == 1 + 2 + 1
    low = rs[2]
    @test low.plotattributes[:label] == "low"
    @test r.t[63] in low.args[1]                 # drawn in the low colour
    @test !(r.t[63] in rs[3].args[1])
    @test rs[end].args[1] == [3.3100557377508912]
end

@testset "ThresholdPhaseResult: two and three regimes" begin
    s = fit_setar(_P11_LYNX, 2; thDelay=1, warn=false)
    r = threshold_phase_plot(s)
    @test r.lagged == _P11_LYNX[2:113]
    @test r.y == _P11_LYNX[3:end]
    @test r.fitted == fitted(s)
    @test threshold_phase_plot(s; lag=2).lagged == _P11_LYNX[1:112]
    @test_throws ArgumentError threshold_phase_plot(s; lag=3)
    @test length(_rec(r)) == 4
    s3 = fit_setar(_P11_LYNX, 2; thDelay=1, nthresh=2, warn=false)
    r3 = threshold_phase_plot(s3)
    @test r3.nregimes == 3
    rs = _rec(r3)
    @test length(rs) == 6                        # tsDyn draws none for nthresh = 2
    @test [x.plotattributes[:label] for x in rs[1:2:end]] == ["low", "middle", "high"]
end

@testset "ThresholdProfileResult" begin
    s = fit_setar(_P11_LYNX, 2; thDelay=1, warn=false)
    p = threshold_profile(s)
    @test length(p.candidates) == 75
    @test p.chosen == 3.3100557377508912
    @test minimum(p.values) ≈ 4.3481912792005826 rtol = 1e-12
    @test p.conditional_on === nothing
    a = threshold_profile(s; criterion=:aic)
    @test a.chosen == p.chosen                   # a monotone transform of the SSR
    @test minimum(a.values) ≈ -358.373987591728 rtol = 1e-12
    s2 = fit_setar(_P11_LYNX, 2; thDelay=1, nthresh=2, warn=false)
    p2 = threshold_profile(s2)
    @test p2.chosen in s2.thresholds
    @test p2.conditional_on in s2.thresholds && p2.conditional_on != p2.chosen
    g = fit_setar(_P11_LYNX, 2; thDelay=1, threshold=3.3100557377508912, warn=false)
    @test_throws ArgumentError threshold_profile(g)
    @test_throws ArgumentError threshold_profile(s; criterion=:bic)
    rs = _rec(p)
    @test issorted(rs[1].args[1])
    @test rs[2].args[1] == [p.chosen]
end

@testset "TransitionFunctionResult" begin
    l = fit_lstar(_P11_LYNX, 2; thDelay=1)
    r = transition_plot(l)
    @test r.G == l.G
    @test r.z == _P11_LYNX[1:112]
    @test r.G ≈ 1 ./ (1 .+ exp.(-l.gamma .* (r.z .- l.th))) rtol = 1e-12
    @test !r.gamma_at_bound
    rs = _rec(r)
    @test length(rs) == 3
    @test !occursin("bound", rs[1].plotattributes[:title])
    w = @test_logs (:warn, r"abrupt") match_mode = :any fit_lstar(_p11("fixture_setar_sim.csv")[:, 2], 1; gamma_max=5.0)
    rw = _rec(transition_plot(w))
    @test occursin("grid bound", rw[2].plotattributes[:title])
end

@testset "ThresholdTest: bootstrap draws and recipe" begin
    t = setar_test(_P11_LYNX, 2; thDelay=0:1, nboot=20, rng=MersenneTwister(3))
    @test size(t.boot) == (20, 3)
    @test t.pvalues == [mean(t.boot[:, i] .>= t.F[i]) for i in 1:3]
    @test t.critical[1, :] ≈ [quantile(t.boot[:, 1], q) for q in (0.90, 0.95, 0.975, 0.99)]
    rs = _rec(t)
    @test length(rs) == 9
    @test _subplots(rs) == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    @test rs[1].plotattributes[:seriestype] == :histogram
    @test rs[3].args[1] == [t.F[1]]
    # no chi-squared reference curve: histograms and vertical lines only
    @test all(x -> x.plotattributes[:seriestype] in (:histogram, :vline), rs)
end
