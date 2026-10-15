using DelimitedFiles
using Statistics: mean, std

# Stage 12.1 -- intermittent demand. handoff/stage 12/docs/
# stage-12.1-intermittent-demand-handoff.md. Primary reference statsforecast
# 2.1.1 (Apache-2.0); forecast::croston (GPL-3) agrees with it to 17 digits
# and is an independent check only. The partx fixture is GPL data and is
# deliberately absent (charter §4.1).

const _IM_FX = joinpath(@__DIR__, "..", "handoff", "stage 12", "fixtures")
_imfx(f) = Float64.(readdlm(joinpath(_IM_FX, f), ','; skipstart=1)[:, end])
const _IM_F = [_imfx("intermittent.csv"), _imfx("fx_sparse_p010_n120.csv"),
               _imfx("fx_dense_p150_n120.csv")]
const _IM_GT = joinpath(@__DIR__, "..", "handoff", "stage 12", "ground-truth")

# handoff §3.1, rows: classic, optimized, SBA, TSB(0.2,0.2), ADIDA, IMAPA; cols F1, F2, F3
const _IM_POINT = [0.65909246562518586 0.11168100226728453 1.7150503549950999;
                   0.67243689556737085 0.12378068088891507 1.7852955852683219;
                   0.62613784234392655 0.1060969521539203 1.6292978372453448;
                   0.2447982165028067 0.10398268223642748 1.7508977992762016;
                   0.42114174223778827 0.11809029046769849 1.7486544426757562;
                   0.29789731076415754 0.12038743043325673 1.7486544426757562]
# §3.3, in-sample MASE with the naive-1 denominator
const _IM_MASE = [1.0885456588355449 0.75327412615990097 0.76547402721288704;
                  1.0003334636707115 0.84657935076697566 0.76526682256932366;
                  1.060205332415507 0.74061041985190545 0.76995408408447064;
                  0.93462033008444145 0.89314620347791829 0.78723985113066774;
                  0.86465563418349412 0.81819563743934332 0.77289356516392216;
                  0.88322692682056458 0.83499151570749741 0.78032048148946886]
_im_models(y) = [croston(y), croston(y; method=:optimized), croston(y; method=:sba),
                 tsb(y; alpha_d=0.2, alpha_p=0.2), adida(y), imapa(y)]

@testset "fixtures as documented" begin
    @test [length(y) for y in _IM_F] == [60, 120, 120]
    @test [count(!=(0), y) for y in _IM_F] == [22, 15, 91]
    @test [sum(y) for y in _IM_F] == [30, 15, 187]
end

@testset "Croston classic, exact" begin
    for j in 1:3
        m = croston(_IM_F[j])
        @test m.forecast === _IM_POINT[1, j]               # all 17 digits, as R and statsforecast
        @test m.method === :classic
        @test m.alpha_demand == m.alpha_interval == 0.1
    end
    @test croston(_IM_F[1]).forecast == 0.65909246562518586
end

@testset "Croston optimized, including both bounds" begin
    for j in 1:3
        @test croston(_IM_F[j]; method=:optimized).forecast === _IM_POINT[2, j]
    end
    o = [croston(y; method=:optimized) for y in _IM_F]
    @test o[1].alpha_demand ≈ 0.2406879687366284 atol = 1e-12
    @test o[1].alpha_interval ≈ 0.29999999999967963 atol = 1e-12    # upper bound binds
    @test o[2].alpha_demand ≈ 0.20000000000000001 atol = 1e-12      # constant demands: tie, midpoint
    @test o[2].alpha_interval ≈ 0.2623181302768971 atol = 1e-12
    @test o[3].alpha_demand ≈ 0.10000000000032039 atol = 1e-12      # lower bound binds
    @test o[3].alpha_interval ≈ 0.13774151423786604 atol = 1e-12
    @test all(0.1 <= m.alpha_demand <= 0.3 && 0.1 <= m.alpha_interval <= 0.3 for m in o)
    # the golden-section tie-break: a constant series returns the midpoint exactly
    @test TSAnalytics._ses_golden(ones(10)) == 0.2
    @test TSAnalytics._ses_golden(ones(10), 0.1, 0.5) == 0.3
    # the bound is real: the unconstrained interval optimum on F1 is above 0.3
    iv = TSAnalytics._imt_intervals(_IM_F[1])
    @test TSAnalytics._ses_sse(0.317, iv) < TSAnalytics._ses_sse(0.3, iv)
    @test occursin("alpha in [0.1, 0.3]", sprint(show, o[1]))
end

@testset "SBA: 1 - alpha/2, not a hardcoded 0.95" begin
    for j in 1:3
        @test croston(_IM_F[j]; method=:sba).forecast === _IM_POINT[3, j]
        @test croston(_IM_F[j]; method=:sba).forecast ≈ 0.95 * _IM_POINT[1, j] rtol = 1e-15
    end
    # D24: statsforecast multiplies by 0.95 whatever alpha is; the factor follows alpha here
    y = _IM_F[1]
    for α in (0.05, 0.2, 0.5)
        @test croston(y; method=:sba, alpha=α).forecast ≈ (1 - α / 2) * croston(y; alpha=α).forecast rtol = 1e-14
        @test !(croston(y; method=:sba, alpha=α).forecast ≈ 0.95 * croston(y; alpha=α).forecast)
    end
    @test croston(y; method=:sba, alpha=0.1).forecast ≈ 0.95 * croston(y).forecast rtol = 1e-15
    @test croston(_IM_F[2]; method=:sba).demand_level == croston(_IM_F[2]).demand_level
end

@testset "TSB" begin
    for j in 1:3
        m = tsb(_IM_F[j]; alpha_d=0.2, alpha_p=0.2)
        @test m.forecast === _IM_POINT[4, j]
        @test m.forecast ≈ m.probability_level * m.demand_level rtol = 1e-15
        @test 0 < m.probability_level < 1
    end
    @test_throws UndefKeywordError tsb(_IM_F[1])                     # no defaults, no optimisation
    @test_throws ArgumentError tsb(_IM_F[1]; alpha_d=0.0, alpha_p=0.2)
end

@testset "ADIDA" begin
    for j in 1:3
        m = adida(_IM_F[j])
        @test m.forecast === _IM_POINT[5, j]
    end
    @test [adida(y).aggregation for y in _IM_F] == [2, 8, 1]
    @test adida(_IM_F[3]; aggregation=2).aggregation == 2
    @test_throws ArgumentError adida(_IM_F[1]; aggregation=0)
    # buckets aligned to the end: the dropped remainder is at the start
    y = [9.0, 0, 1, 0, 2]
    @test TSAnalytics._imt_chunk_forecast(y, 2) == TSAnalytics._imt_chunk_forecast(y[2:end], 2)
end

@testset "IMAPA, and ADIDA = IMAPA when the width rounds to 1" begin
    for j in 1:3
        @test imapa(_IM_F[j]).forecast === _IM_POINT[6, j]
    end
    @test imapa(_IM_F[3]).forecast === adida(_IM_F[3]).forecast      # F3: mean interval 1.3187
    @test imapa(_IM_F[3]).aggregation == 1
    @test imapa(_IM_F[1]).forecast ≈
          mean(adida(_IM_F[1]; aggregation=k).forecast for k in 1:2) rtol = 1e-15
    # eight levels on F2: the sum must follow numpy's pairwise order to match exactly
    @test imapa(_IM_F[2]).aggregation == 8
    v = [TSAnalytics._imt_chunk_forecast(_IM_F[2], k) / k for k in 1:8]
    @test TSAnalytics._pairwise_sum(v) / 8 === _IM_POINT[6, 2]
    @test TSAnalytics._pairwise_sum(collect(1.0:200.0)) == sum(1.0:200.0)
end

@testset "flat in h: all 18 cells" begin
    for y in _IM_F, m in _im_models(y)
        f = forecast(m, 6)
        @test f.point == fill(m.forecast, 6)
    end
end

@testset "demand and interval levels" begin
    lv = [(1.2978605425231977, 1.9691630692396169), (1.0, 8.9540743698441609),
          (1.9398661738758045, 1.1310840922110108)]
    for j in 1:3
        m = croston(_IM_F[j])
        @test m.demand_level ≈ lv[j][1] atol = 1e-15
        @test m.interval_level ≈ lv[j][2] atol = 1e-15
        @test m.demand_level / m.interval_level === m.forecast       # demand / interval, exactly
        @test m.demand_level > 0 && m.interval_level >= 1
    end
end

@testset "fitted values, rtol 1e-12 against R" begin
    for (j, f) in enumerate(("Rfitted_F1.csv", "Rfitted_F2.csv", "Rfitted_F3.csv"))
        R = readdlm(joinpath(_IM_GT, f), ','; skipstart=1)[:, 1]
        r = [x == "NA" ? NaN : Float64(x) for x in R]
        fit = fitted(croston(_IM_F[j]))
        @test isnan(fit[1]) && isnan(r[1])
        @test length(fit) == length(r)
        @test fit[2:end] ≈ r[2:end] rtol = 1e-12
        # §3.4: R's O(n^2) per-prefix recomputation differs by a few ulp
        @test maximum(abs.((fit[2:end] .- r[2:end]) ./ max.(abs.(r[2:end]), 1e-300))) < 1e-14
    end
    @test fitted(croston(_IM_F[2]))[2:10] == zeros(9)                 # before the first demand
end

@testset "in-sample MASE, all six methods" begin
    for j in 1:3
        y = _IM_F[j]
        for (i, m) in enumerate(_im_models(y))
            v = mase(y[2:end], fitted(m)[2:end], y)
            if i == 6 && j == 1
                # IMAPA on F1: a prefix mean interval is exactly a half, and the
                # reference rounds it half away from zero in its fitted path only
                @test abs(v - _IM_MASE[i, j]) ≈ 0.0015 atol = 1e-4
            else
                @test v ≈ _IM_MASE[i, j] atol = 1e-12
            end
        end
    end
    # the naive denominators
    @test sum(abs.(diff(_IM_F[1]))) / 59 ≈ 0.77966101694915257 atol = 1e-15
    @test sum(abs.(diff(_IM_F[3]))) / 119 ≈ 1.3277310924369747 atol = 1e-15
    # D32: forecast::accuracy() on a bare vector gives 1.3400472874157467 -- not ours
    @test mase(_IM_F[1][2:end], fitted(croston(_IM_F[1]))[2:end], _IM_F[1]) != 1.3400472874157467
end

@testset "cfe and spis" begin
    @test cfe([2.0, 0.0, 3.0], [1.0, 1.0, 1.0]) == 2.0               # under-forecast: positive
    @test cfe([0.0, 0.0], [1.0, 1.0]) == -2.0
    @test spis([2.0, 0.0, 3.0], [1.0, 1.0, 1.0], [0.0, 4.0, 2.0]) == 2.0   # utilsforecast, run here
    @test_throws ArgumentError cfe([1.0], [1.0, 2.0])
    @test_throws ArgumentError spis([1.0], [1.0], zeros(3))
    @test_throws ArgumentError spis([1.0], [1.0, 2.0], [1.0])
    # SBA removes Croston's upward bias: lower total forecast, CFE moves up
    y = _IM_F[1]
    c, s = fitted(croston(y))[2:end], fitted(croston(y; method=:sba))[2:end]
    @test cfe(y[2:end], s) > cfe(y[2:end], c)
    @test sum(s) < sum(c)
    @test spis(y[2:end], c, y) > 0
    @test spis(y[2:end], s, y) ≈ sum(abs.(y[2:end] .- s)) / mean(y)
    @test cfe(y, y) == 0.0
    @test spis(y, y, y) == 0.0
end

@testset "ADI/CV² classification, with F3 on the boundary" begin
    c = [adi_cv2(y) for y in _IM_F]
    @test c[1].adi == 2.5
    @test c[1].cv2 ≈ 0.18158730158730163 atol = 1e-14
    @test c[2].adi ≈ 7.7333333333333334 atol = 1e-14
    @test c[2].cv2 == 0.0
    @test c[3].adi ≈ 1.3186813186813187 atol = 1e-14
    @test c[3].cv2 ≈ 0.25450541908547586 atol = 1e-14
    @test [x.method for x in c] == [:sba, :sba, :croston]              # F3 just below 1.32
    @test [x.quadrant for x in c] == [:intermittent, :intermittent, :smooth]
    @test adi_cv2(_IM_F[3]; adi_cutoff=1.3).method === :sba
    @test adi_cv2([1.0, 9, 1, 9]).quadrant === :erratic
    @test adi_cv2([0.0, 1, 0, 0, 9, 0, 0, 1]).quadrant === :lumpy
    @test_throws ArgumentError adi_cv2([0.0, 3.0, 0.0])
    @test croston(_IM_F[3]).classification === :croston
    @test occursin("use SBA", sprint(show, c[1]))
end

@testset "reduction: no zeros is simple exponential smoothing" begin
    y = [1.0, 2, 3, 2, 1, 2]
    m = croston(y)
    @test m.interval_level == 1.0
    @test m.forecast ≈ 1.39241 atol = 1e-12                              # both references
    @test m.forecast ≈ TSAnalytics._ses_level(y, 0.1) rtol = 1e-15
    hw = holt_winters(y; alpha=0.1)                                       # the existing SES
    @test m.forecast == hw.level[end]                                     # exactly
    z = _IM_F[3][_IM_F[3] .> 0]
    @test croston(z).forecast ≈ TSAnalytics._ses_level(z, 0.1) rtol = 1e-15
    @test croston(z).interval_level == 1.0
    @test fitted(croston(y))[2] == y[1]
end

@testset "reductions: alpha -> 0, never-zero TSB, ADIDA at width 1" begin
    y = _IM_F[1]
    @test croston(y; method=:sba, alpha=1e-12).forecast ≈ croston(y; alpha=1e-12).forecast rtol = 1e-11
    z = [2.0, 3, 1, 4, 2, 5]
    t = tsb(z; alpha_d=0.3, alpha_p=0.3)
    @test t.probability_level == 1.0
    @test t.forecast ≈ TSAnalytics._ses_level(z, 0.3) rtol = 1e-15
    a1 = adida(z; aggregation=1)
    @test a1.forecast ≈ TSAnalytics._ses_level(z, TSAnalytics._ses_golden(z)) rtol = 1e-15
    @test adida(z).aggregation == 1
    @test imapa(z).forecast == adida(z).forecast
    @test croston(z; method=:optimized).interval_level == 1.0
    @test tsb(z; alpha_d=0.3, alpha_p=0.3).demand_level ≈ TSAnalytics._ses_level(z, 0.3)
    @test adida(_IM_F[3]; aggregation=1).forecast === adida(_IM_F[3]).forecast
    @test length(fitted(a1 === nothing ? adida(z) : adida(z))) == 6
end

@testset "edge cases we answer (references agree)" begin
    # the thirteen-case table, handoff §4; both references give these numbers
    @test croston(zeros(10)).forecast == 0.0
    @test croston([0.0, 0, 0, 5, 0, 0, 0, 0, 0, 0]).forecast == 1.25
    @test croston([0.0, 0, 0, 0, 0, 0, 0, 0, 0, 5]).forecast == 0.5
    @test croston([3.0]).forecast == 3.0
    @test croston([0.0, 3.0]).forecast == 1.5
    @test croston([3.0, 4.0]).forecast ≈ 3.1 rtol = 1e-15
    @test croston([1.0, 2, 3, 2, 1, 2]).forecast ≈ 1.39241 atol = 1e-12
    @test croston([0.0, 0, 0, 5, 0, 0, 0, 0, 0, 0]; method=:sba).forecast == 0.95 * 1.25
    @test tsb([0.0, 0, 0, 5, 0, 0, 0, 0, 0, 0]; alpha_d=0.2, alpha_p=0.2).forecast > 0
    @test adida([0.0, 0, 0, 5, 0, 0, 0, 0, 0, 0]).aggregation == 4
    @test imapa([0.0, 3.0]).forecast > 0
    @test croston([0.0, 0, 0, 5, 0, 0, 0, 0, 0, 0]).demand_level == 5.0
    @test croston([0.0, 0, 0, 5, 0, 0, 0, 0, 0, 0]).interval_level == 4.0
    @test croston(Int[0, 0, 2, 0, 3]).forecast ≈ croston([0.0, 0, 2, 0, 3]).forecast
    @test croston(zeros(10); method=:optimized).forecast == 0.0
    @test croston(zeros(10); method=:sba).forecast == 0.0
    @test tsb(zeros(5); alpha_d=0.1, alpha_p=0.1).forecast == 0.0
    @test adida(zeros(5)).forecast == 0.0
    @test imapa(zeros(5)).forecast == 0.0
    @test croston([3.0, 4.0]; method=:optimized).forecast > 0
end

@testset "refusals: single arrival at t = 1, negatives, SBA on one value" begin
    # both references return 5 here -- the empirical rate is 0.5 (handoff §5.2)
    e = try croston([5.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); catch err; err end
    @test e isa ArgumentError
    @test occursin("first period", e.msg) && occursin("0.5", e.msg)
    @test_throws ArgumentError croston([3.0, 0.0])                    # references: 3
    for f in (y -> croston(y), y -> croston(y; method=:sba), y -> croston(y; method=:optimized),
              y -> tsb(y; alpha_d=0.2, alpha_p=0.2), adida, imapa)
        @test_throws ArgumentError f([5.0, 0, 0, 0])
        @test_throws ArgumentError f([1.0, -1, 0, 2])                 # R refuses; statsforecast: 1 (D25)
    end
    e2 = try croston([1.0, -1, 0, 2]); catch err; err end
    @test occursin("negative", e2.msg)
    @test_throws ArgumentError croston([3.0]; method=:sba)             # statsforecast: 2.85
    @test_throws ArgumentError croston(Float64[])
    @test_throws ArgumentError croston([1.0, NaN, 0.0])
    @test_throws ArgumentError croston(_IM_F[1]; method=:tsb)
    @test_throws ArgumentError croston(_IM_F[1]; alpha=0.0)
end

@testset "all zeros, through an explicit branch" begin
    for m in (croston(zeros(8)), croston(zeros(8); method=:sba), tsb(zeros(8); alpha_d=0.1, alpha_p=0.1),
              adida(zeros(8)), imapa(zeros(8)))
        @test m.forecast == 0.0
        @test fitted(m) == zeros(8)
    end
    @test croston(zeros(8)).classification === nothing
    @test forecast(croston(zeros(8)), 3).point == zeros(3)
end

@testset "the 2.5 tie-break, both values pinned" begin
    y = _IM_F[1]
    @test mean(TSAnalytics._imt_intervals(y)) == 2.5
    @test adida(y).aggregation == 2                                   # half to even
    @test adida(y).forecast === 0.42114174223778827
    @test adida(y; aggregation=3).forecast ≈ 0.50681015732311729 atol = 1e-15    # half away
    @test adida(y; aggregation=3).forecast / adida(y).forecast ≈ 1.2034 atol = 1e-4
    # the same rule in IMAPA's fitted path reproduces statsforecast's fitted MASE
    fa = TSAnalytics._imapa_fitted(y; mode=RoundNearestTiesAway)
    @test mase(y[2:end], fa[2:end], y) ≈ 0.88322692682056458 atol = 1e-12
    @test fitted(imapa(y)) != fa
    @test isnan(fa[1])
end

@testset "IntermittentForecast carries no invented intervals" begin
    f = forecast(croston(_IM_F[1]), 4)
    @test f isa IntermittentForecast
    @test !(f isa Forecast)
    @test fieldnames(IntermittentForecast) == (:point, :horizon, :method)
    @test f.horizon == 4 && f.method === :classic
    @test occursin("no prediction intervals", sprint(show, f))
    @test predict(croston(_IM_F[1]), 4).point == f.point
    @test_throws ArgumentError forecast(croston(_IM_F[1]), 0)
    @test_throws ArgumentError fitted(croston(_IM_F[1]; method=:optimized) |> m -> IntermittentModel(
        m.method, m.y, m.forecast, m.demand_level, m.interval_level, m.probability_level,
        m.alpha_demand, m.alpha_interval, m.alpha_probability, m.aggregation, nothing, nothing))
end

@testset "demand_intervals decomposition" begin
    for y in _IM_F
        d = demand_intervals(y)
        @test sum(d.demands) == sum(y)
        @test sum(d.intervals) == d.positions[end]
        @test length(d.demands) == length(d.intervals) == count(!=(0), y)
        @test all(>(0), d.demands)
    end
    d = demand_intervals([0, 0, 3, 0, 2, 0])
    @test d.demands == [3.0, 2.0] && d.intervals == [3, 2] && d.positions == [3, 5]
    @test_throws ArgumentError demand_intervals([1.0, -2.0])
end

@testset "show reports the method and the classification" begin
    s = sprint(show, croston(_IM_F[1]))
    @test occursin("Croston (classic)", s) && occursin("SBA", s)
    @test occursin("TSB", sprint(show, tsb(_IM_F[1]; alpha_d=0.2, alpha_p=0.2)))
    @test occursin("width 2", sprint(show, adida(_IM_F[1])))
    @test occursin("widths 1:2", sprint(show, imapa(_IM_F[1])))
    @test occursin("use Croston", sprint(show, adi_cv2(_IM_F[3])))
    @test nobs(croston(_IM_F[1])) == 60
end

@testset "performance" begin
    y = repeat(_IM_F[3], 20)                                          # n = 2400
    croston(y)
    t = @elapsed for _ in 1:20
        croston(y)
    end
    @test t / 20 < 0.05
    @test (@allocated TSAnalytics._ses_level(y, 0.1)) == 0
    @test (@allocated TSAnalytics._ses_sse(0.2, y)) == 0
    @test @elapsed(croston(_IM_F[2]; method=:optimized)) < 2.0
    @test @elapsed(imapa(_IM_F[2])) < 5.0
    @inferred TSAnalytics._ses_golden(y)
end
