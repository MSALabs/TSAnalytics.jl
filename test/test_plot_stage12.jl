using RecipesBase
using DelimitedFiles

# Stage 12 plot data and recipes. handoff/stage 12/docs/
# stage-12-plot-recipes-handoff.md §7: test the data, not the rendering --
# except the stem mark, the one rendering choice worth asserting.
# (All nine were rendered with GR and inspected once by hand.)

const _P12FX = joinpath(@__DIR__, "..", "handoff", "stage 12", "fixtures")
_p12cp(f) = Float64.(vec(readdlm(joinpath(_P12FX, f * ".csv"), ','; skipstart=1)))
_p12im(f) = Float64.(readdlm(joinpath(_P12FX, f), ','; skipstart=1)[:, end])
_rec12(x; kw...) = RecipesBase.apply_recipe(Dict{Symbol,Any}(kw...), x)
_st12(r) = get(r.plotattributes, :seriestype, nothing)
_sp12(r) = get(r.plotattributes, :subplot, 1)

const _P12_F1 = _p12im("intermittent.csv")
const _P12_F3 = _p12im("fx_dense_p150_n120.csv")
const _P12_FX0 = _p12cp("fx0_regen")
const _P12_FXA = _p12cp("fxA_var")

@testset "RecipesBase stays soft; the 29 earlier recipes are still there" begin
    @test !isdefined(Main, :Plots)
    @test count(m -> m.module === TSAnalytics, methods(RecipesBase.apply_recipe)) == 29 + 9
end

@testset "demand_intervals(model): the smoothed levels" begin
    for m in (croston(_P12_F1), croston(_P12_F1; method=:sba), croston(_P12_F1; method=:optimized),
              croston(_P12_F1; alpha=0.3))
        d = demand_intervals(m)
        raw = demand_intervals(_P12_F1)
        @test (d.y, d.demands, d.intervals, d.positions) == (raw.y, raw.demands, raw.intervals, raw.positions)
        @test d.method === m.method
        # the paths end at the model's levels, bit for bit
        @test d.demand_levels[end] === m.demand_level
        @test d.interval_levels[end] === m.interval_level
        @test length(d.demand_levels) == length(d.interval_levels) == 22
        # and follow the recursion, started at the first value
        α = m.alpha_demand
        @test d.demand_levels[1] == d.demands[1]
        @test all(d.demand_levels[k] ≈ α * d.demands[k] + (1 - α) * d.demand_levels[k - 1] for k in 2:22)
    end
    t = demand_intervals(tsb(_P12_F1; alpha_d=0.2, alpha_p=0.3))
    @test t.demand_levels[end] === tsb(_P12_F1; alpha_d=0.2, alpha_p=0.3).demand_level
    @test isempty(t.interval_levels)
    for m in (adida(_P12_F1), imapa(_P12_F1))
        d = demand_intervals(m)
        @test isempty(d.demand_levels) && isempty(d.interval_levels) && d.method === m.method
    end
    raw = demand_intervals([0, 2, 0, 0, 1])
    @test raw.method === nothing && isempty(raw.demand_levels)
    z = demand_intervals(croston(zeros(5)))
    @test isempty(z.demands) && isempty(z.demand_levels)
end

@testset "DemandIntervals recipe: stems, not a line" begin
    m = croston(_P12_F1)
    d = demand_intervals(m)
    rs = _rec12(d)
    @test _sp12.(rs) == [1, 2, 2, 3, 3]
    # the one rendering assertion: the series is drawn as stems
    @test _st12(rs[1]) === :sticks
    @test rs[1].args == (collect(1:60), _P12_F1)
    @test all(_st12(r) !== :line && _st12(r) !== :path for r in rs)
    @test rs[2].args == (d.positions, d.demands)
    @test _st12(rs[3]) === :steppost && rs[3].args == (d.positions, d.demand_levels)
    @test rs[5].args[2] == d.interval_levels
    # without a model: no level series
    @test _sp12.(_rec12(demand_intervals(_P12_F1))) == [1, 2, 3]
    @test _sp12.(_rec12(demand_intervals(tsb(_P12_F1; alpha_d=0.2, alpha_p=0.2)))) == [1, 2, 2, 3]
    # a long series is thinned for display, keeping the largest stem per bin
    long = zeros(20_000); long[17] = 9.0; long[12_345] = 11.0; long[3:7:end] .= 1.0
    s = _rec12(demand_intervals(long))[1]
    @test length(s.args[1]) == 5_000
    @test maximum(s.args[2]) == 11.0 && 12_345 in s.args[1] && 17 in s.args[1]
end

@testset "DemandIntervals recipe: the interval histogram" begin
    d = demand_intervals(croston(_P12_F1))
    rs = _rec12(d; histogram=true)
    @test _st12.(rs) == [:histogram, :vline]
    @test rs[1].args == (d.intervals,)
    @test rs[2].args == ([d.interval_levels[end]],)
    raw = demand_intervals(_P12_F1)
    @test _rec12(raw; histogram=true)[2].args == ([sum(raw.intervals) / length(raw.intervals)],)
    @test_throws ArgumentError _rec12(demand_intervals(zeros(4)); histogram=true)
end

@testset "IntermittentForecast is drawn as stems" begin
    fc = forecast(croston(_P12_F1), 6)
    rs = _rec12(fc)
    @test length(rs) == 1
    @test _st12(rs[1]) === :sticks
    @test rs[1].args == (collect(1:6), fc.point)
end

@testset "ADI/CV² quadrant: F3 just inside the boundary" begin
    c = adi_cv2(_P12_F3)
    @test c.adi ≈ 1.3187 atol = 5e-5
    @test c.cv2 ≈ 0.2545 atol = 5e-5
    @test c.adi < c.adi_cutoff && c.cv2 < c.cv2_cutoff && c.quadrant === :smooth
    rs = _rec12(c)
    @test _st12.(rs) == [:vline, :hline, :scatter]
    @test rs[1].args == ([1.32],) && rs[2].args == ([0.49],)
    @test rs[3].args == ([c.adi], [c.cv2])
    ann = rs[1].plotattributes[:annotations]
    @test last.(ann) == ["smooth", "intermittent", "erratic", "lumpy"]
    # each label sits in its own quadrant
    for (x, y, nm) in ann
        q = x > 1.32 ? (y > 0.49 ? "lumpy" : "intermittent") : (y > 0.49 ? "erratic" : "smooth")
        @test q == nm
    end
    # custom cutoffs move the crosshair
    @test _rec12(adi_cv2(_P12_F3; adi_cutoff=1.2, cv2_cutoff=0.3))[1].args == ([1.2],)
end

@testset "ChangePointResult recipe: segment means, breaks, confidence bars" begin
    cp = changepoints(_P12_FX0)
    @test cp.breakpoints == [100, 200]
    bp = breakpoints(_P12_FX0)
    ci = break_confint(bp)
    @test ci == [99 100 101; 199 200 201]
    rs = _rec12(cp; confint=ci)
    @test _st12.(rs) == [:line, :path, :vline, :path]
    @test rs[1].args == (collect(1:300), _P12_FX0)
    # the step function is the segment means, between the breaks
    sx, sy = rs[2].args
    @test isequal(sx, [0.5, 100.5, NaN, 100.5, 200.5, NaN, 200.5, 300.5, NaN])
    @test sy[1:3:end] == cp.segment_means && sy[2:3:end] == cp.segment_means
    @test rs[3].args == ([100.5, 200.5],)
    # the confidence bars are break_confint's exact integers
    bx, by = rs[4].args
    @test isequal(bx, [99.5, 101.5, NaN, 199.5, 201.5, NaN])
    @test all(==(minimum(_P12_FX0)), filter(!isnan, by))
    # a confint for other breaks is refused, not drawn
    @test_throws ArgumentError _rec12(cp; confint=[96 97 98; 202 203 204])
    @test_throws ArgumentError _rec12(cp; confint=[99 100 101])
    # no breaks: the series and one mean
    none = changepoints(_P12_FX0; penalty=1e6)
    rn = _rec12(none)
    @test _st12.(rn) == [:line, :path]
    @test rn[2].args[2][1] ≈ sum(_P12_FX0) / 300
    @test rn[1].plotattributes[:title] == "No change points"
end

@testset "penalty_path and the penalty-path keyword" begin
    cp = changepoints(_P12_FX0)
    p = penalty_path(cp)
    @test length(p.penalties) == 60
    @test p.penalties[1] ≈ cp.penalty / 10 && p.penalties[end] ≈ cp.penalty * 10
    # an optimal segmentation can only lose breaks as the penalty grows
    @test issorted(p.counts; rev=true)
    @test p.counts == length.(p.breakpoints)
    # at the fitted penalty, the fitted segmentation
    for r in (cp, changepoints(_P12_FX0; search=:binseg), changepoints(_P12_FX0; search=:exact),
              changepoints(_p12cp("fxB_weak"); penalty=:mbic, search=:exact),
              changepoints(_P12_FXA; cost=:var, penalty=:bic))
        @test penalty_path(r; penalties=[r.penalty]).breakpoints[1] == r.breakpoints
    end
    # the MBIC per-segment term is kept: fxB_weak's 139, not PELT-with-MBIC's 187
    mb = changepoints(_p12cp("fxB_weak"); penalty=:mbic, search=:exact)
    @test mb.breakpoints == [139]
    @test penalty_path(mb; penalties=[mb.penalty]).breakpoints[1] == [139]
    @test penalty_path(cp; penalties=[1e6]).counts == [0]
    @test_throws ArgumentError penalty_path(cp; penalties=[-1.0])
    rs = _rec12(cp; penalty_path=true)
    @test _st12.(rs) == [:steppre, :vline]
    @test rs[1].args == (p.penalties, p.counts)
    @test rs[2].args == ([cp.penalty],)
end

@testset "BreakpointsResult recipe: BIC and RSS against m" begin
    bp = breakpoints(_P12_FX0)
    rs = _rec12(bp)
    @test _sp12.(rs) == [1, 1, 2, 2]
    @test rs[1].args == (collect(0:5), bp.bic)
    @test rs[3].args == (collect(0:5), bp.rss)
    @test rs[2].args == ([2],) && rs[4].args == ([2],)
    @test argmin(bp.bic) - 1 == 2
    @test minimum(bp.bic) ≈ 838.563795 atol = 1e-6
    @test _rec12(breakpoints(_P12_FXA))[2].args == ([0],)
end

@testset "diagnostic_plot dispatches" begin
    bp = breakpoints(_P12_FX0)
    e = vcat([let r = (a + 1):b, X = ones(b - a, 1)
                  _P12_FX0[r] .- X * (X \ _P12_FX0[r])
              end for (a, b) in ((0, 100), (100, 200), (200, 300))]...)
    @test diagnostic_plot(bp).std_resid ≈ diagnostic_plot(e).std_resid atol = 1e-12
    @test length(diagnostic_plot(bp).std_resid) == 300
    m = croston(_P12_F1)
    @test diagnostic_plot(m).std_resid == diagnostic_plot(residuals(m)[2:end]).std_resid
    @test diagnostic_plot(m) isa DiagnosticPlotResult
end

@testset "FStatsResult recipe: no boundary unless simulated and labelled" begin
    fs = fstats(_P12_FX0)
    rs = _rec12(fs)
    @test _st12.(rs) == [:line, :vline, :scatter]
    @test all(_st12(r) !== :hline for r in rs)
    @test rs[1].args == (fs.dates, fs.stats)
    @test rs[2].args == ([first(fs.dates), last(fs.dates)],)
    @test rs[3].args == ([fs.argmax], [fs.supF])
    rb = _rec12(fs; boundary=(value=9.1, reps=1000))
    @test _st12(rb[end]) === :hline && rb[end].args == ([9.1],)
    @test rb[end].plotattributes[:label] == "simulated 95.0% critical value, 1000 replications"
    @test _rec12(fs; boundary=(value=9.1, reps=500, level=0.9))[end].plotattributes[:label] ==
          "simulated 90.0% critical value, 500 replications"
end

@testset "efp_boundary: closed-form critical values" begin
    o = efp(_P12_FXA)
    # the Kolmogorov 5% point, 1.358, and Brown-Durbin-Evans' 0.948
    @test efp_boundary(o)[1] ≈ 1.3580986 atol = 1e-6
    r = efp(_P12_FXA; type=:rec_cusum)
    br = efp_boundary(r)
    @test br[1] ≈ 0.9478982 atol = 1e-6
    @test br[end] ≈ 3 * br[1]
    @test length(br) == size(r.process, 1)
    # the boundary is where the p-value equals alpha
    for e in (o, r, efp(_P12_FXA; type=:score_cusum), efp(_P12_FXA; type=:re)), α in (0.01, 0.05, 0.1)
        b = efp_boundary(e; alpha=α)
        k = size(e.process, 2)
        p = e.limit === :motion ? TSAnalytics._p_motion_max(b[1], k)[1] : TSAnalytics._p_bridge_max(b[1], k)[1]
        @test p ≈ α rtol = 1e-9
        # crossing the boundary is rejecting
        m = size(e.process, 1) - 1
        crosses = any(abs(e.process[j + 1, i]) > b[j + 1] for j in 0:m, i in 1:k)
        @test crosses == (e.pvalue < α)
    end
    # the variance-change fixture: Score-CUSUM crosses, OLS-CUSUM does not
    s = efp(_P12_FXA; type=:score_cusum)
    @test s.pvalue ≈ 0.00038111 atol = 5e-9
    @test o.pvalue ≈ 0.503186 atol = 5e-7
    @test maximum(abs, s.process) > efp_boundary(s)[1]
    @test maximum(abs, o.process) < efp_boundary(o)[1]
    @test_throws ArgumentError efp_boundary(o; alpha=1.0)
end

@testset "EFPResult recipe: process and boundary" begin
    s = efp(_P12_FXA; type=:score_cusum)
    rs = _rec12(s)
    @test _st12.(rs) == [:line, :line, :line, :line, :hline]
    t = collect(0:300) ./ 300
    @test rs[1].args == (t, s.process[:, 1]) && rs[2].args == (t, s.process[:, 2])
    b = efp_boundary(s)
    @test rs[3].args == (t, b) && rs[4].args == (t, -b)
    @test rs[3].plotattributes[:linestyle] === :dash
    r = efp(_P12_FXA; type=:rec_cusum)
    rr = _rec12(r; alpha=0.1)
    @test rr[2].args[2] == efp_boundary(r; alpha=0.1)
end

@testset "LagFeatureMap" begin
    y = sin.(1:60) .+ (1:60) ./ 10
    d = lag_feature_map(lag_regression(y, [1, 2, 4]; strategy=:direct, horizon=3))
    @test d.strategy === :direct
    # origin-anchored: step h uses the requested lags plus h - 1
    @test d.used == BitMatrix([1 1 0 1 0 0; 0 1 1 0 1 0; 0 0 1 1 0 1])
    r = lag_feature_map(lag_regression(y, [1, 3]); horizon=4)
    @test r.used == BitMatrix(repeat([1 0 1], 4, 1))
    @test size(lag_feature_map(lag_regression(y, [2])).used) == (6, 2)
    @test_throws ArgumentError lag_feature_map(lag_regression(y, [2]); horizon=0)
    rs = _rec12(d)
    @test length(rs) == 1 && _st12(rs[1]) === :heatmap
    @test rs[1].args == (collect(1:6), collect(1:3), Float64.(d.used))
end

@testset "CVWindowLayout" begin
    b = cv_window_layout(blocked_window_split(12; window_length=4); kind="blocked")
    @test b.layout == [2 2 2 2 1 1 1 1 1 1 1 1; 1 1 1 1 2 2 2 2 1 1 1 1; 1 1 1 1 1 1 1 1 2 2 2 2]
    # the middle fold trains on both sides of its test block
    @test b.layout[2, 1] == 1 && b.layout[2, end] == 1
    e = cv_window_layout(expanding_window_split(10; initial_window=6, fh=1:2))
    folds = expanding_window_split(10; initial_window=6, fh=1:2)
    for (i, (tr, te)) in enumerate(folds)
        @test findall(==(1), e.layout[i, :]) == tr
        @test findall(==(2), e.layout[i, :]) == te
    end
    @test_throws ArgumentError cv_window_layout(Tuple{Vector{Int},Vector{Int}}[])
    rs = _rec12(b)
    @test [r.plotattributes[:label] for r in rs] == ["train", "test"]
    @test length(rs[1].args[1]) == 3 * count(==(1), b.layout)
    @test length(rs[2].args[1]) == 3 * count(==(2), b.layout)
    @test rs[1].plotattributes[:title] == "blocked"
end
