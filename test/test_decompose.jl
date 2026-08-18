using DelimitedFiles

# All exact-value reference numbers below are from
# handoff/verification/stage-3.1-verification-transcript.txt (full-precision
# R and Python output, both independently re-confirmed against real
# `statsmodels` before this test file was written, not transcribed blind).

@testset "classical_decompose core matrix (dual R/Python-verified)" begin
    monthly = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))
    period7 = vec(readdlm(TSAnalytics.PERIOD7, ','; skipstart=1, header=false))
    mult_monthly = vec(readdlm(TSAnalytics.MULT_MONTHLY, ','; skipstart=1, header=false))
    @test length(monthly) == 48
    @test length(period7) == 42
    @test length(mult_monthly) == 48

    # Test A: additive, period=12, default filt
    rA = classical_decompose(monthly, 12; model=:additive)
    @test isapprox(rA.trend[20:22], [108.9275811674, 109.5358364628, 110.0988327046]; atol=1e-8)
    @test isapprox(rA.figure, [0.2343823573, 4.0499392659, 7.5916921053, 10.3677757378, 8.7504329202,
                                5.425297929, 0.1571542507, -4.4342414936, -8.1166662104, -9.9850852427,
                                -8.2512346792, -5.7894469403]; atol=1e-8)
    @test isapprox(rA.resid[20:22], [-1.4056433751, 1.3862244787, 0.1604762377]; atol=1e-6)
    @test rA.model == :additive && rA.period == 12
    @test rA.observed == monthly

    # Test B: multiplicative, period=12, default filt
    rB = classical_decompose(mult_monthly, 12; model=:multiplicative)
    @test isapprox(rB.trend[20:22], [119.0536744743, 120.2202139926, 121.2733792572]; atol=1e-8)
    @test isapprox(rB.figure, [0.9924956216, 1.06908335, 1.12839746, 1.145179751, 1.136424675,
                                1.080189378, 1.006225387, 0.924443821, 0.869526824, 0.8451125186,
                                0.8668342655, 0.9360869476]; atol=1e-6)
    @test isapprox(rB.resid[20:22], [0.9994059512, 0.9796567386, 1.003249884]; atol=1e-6)

    # Test C: additive, period=7 (odd), default filt
    rC = classical_decompose(period7, 7; model=:additive)
    @test isapprox(rC.trend[18:20], [55.18281308, 55.51575702, 55.92701432]; atol=1e-6)
    @test isapprox(rC.figure, [-0.1570750794, 3.584702401, 5.007779109, 2.172663151, -2.023778523,
                                -5.012454003, -3.571837055]; atol=1e-6)
    @test isapprox(rC.resid[18:20], [0.5920624838, -0.2974022533, 0.4125665693]; atol=1e-6)

    # Test D: additive, period=12, custom uniform filt (differs from A)
    rD = classical_decompose(monthly, 12; model=:additive, filt=fill(1/12, 12))
    @test isapprox(rD.trend[20:22], [109.261923, 109.80975, 110.3879154]; atol=1e-5)
    @test isapprox(rD.figure, [0.2693406892, 4.076198672, 7.61712525, 10.39672772, 8.761812918,
                                5.173787184, 0.1725707695, -4.397525596, -8.077804913, -9.95524764,
                                -8.253324695, -5.783660363]; atol=1e-5)
    @test !isapprox(rD.trend[20], rA.trend[20]; atol=1e-3)  # genuinely different filters

    # Test E: multiplicative, period=7 (odd), default filt. Fully deterministic
    # formula (no randn call), so it's reproduced inline rather than needing a
    # CSV -- no cross-language RNG risk since there's no randomness involved.
    t5 = 0:41
    y5 = (50 .+ 0.3 .* t5) .* (1 .+ 0.1 .* sin.(2π .* t5 ./ 7))
    rE = classical_decompose(y5, 7; model=:multiplicative)
    @test isapprox(rE.trend[18:20], [55.06885218, 55.36885218, 55.69230712]; atol=1e-6)
    @test isapprox(rE.figure, [0.9993828396, 1.077769662, 1.097639659, 1.043968027, 0.9571496342,
                                0.9026306354, 0.9214595424]; atol=1e-6)
    @test isapprox(rE.resid[18:20], [1.000010062, 1.000000141, 1.000001372]; atol=1e-6)

    # Test C2: additive, period=7, custom uniform filt -- degenerate case
    # (odd-period default IS uniform), must exactly match C
    rC2 = classical_decompose(period7, 7; model=:additive, filt=fill(1/7, 7))
    @test isapprox(rC2.trend, rC.trend; nans=true, atol=1e-12)
    @test isapprox(rC2.figure, rC.figure; atol=1e-12)

    # Test B2: multiplicative, period=12, custom uniform filt
    rB2 = classical_decompose(mult_monthly, 12; model=:multiplicative, filt=fill(1/12, 12))
    @test isapprox(rB2.trend[20:22], [119.6486649, 120.7917631, 121.7549954]; atol=1e-5)
    @test isapprox(rB2.figure, [0.9929826526, 1.069750809, 1.129545449, 1.146079719, 1.137399126,
                                 1.076826005, 1.006477277, 0.9244792892, 0.8690931649, 0.8447939508,
                                 0.8667171135, 0.9358554438]; atol=1e-6)

    # Test E2: multiplicative, period=7, custom uniform filt -- degenerate,
    # must exactly match E
    rE2 = classical_decompose(y5, 7; model=:multiplicative, filt=fill(1/7, 7))
    @test isapprox(rE2.trend, rE.trend; nans=true, atol=1e-12)
    @test isapprox(rE2.figure, rE.figure; atol=1e-12)
end

@testset "classical_decompose Python-exclusive features (two_sided, extrapolate_trend)" begin
    monthly = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))
    rA = classical_decompose(monthly, 12; model=:additive)

    # Test F: two_sided=false -- causal filter, differs from centered A
    rF = classical_decompose(monthly, 12; model=:additive, two_sided=false)
    @test isapprox(rF.trend[20:22], [105.9380310439, 106.4278387408, 106.9764548579]; atol=1e-8)
    @test all(isnan, rF.trend[1:5])   # causal filter has no history at series start
    @test !isapprox(rF.trend[20], rA.trend[20]; atol=1e-3)

    # Test G: extrapolate_trend=2 -- no NaN at all
    rG = classical_decompose(monthly, 12; model=:additive, extrapolate_trend=2)
    @test !any(isnan, rG.trend)
    @test isapprox(rG.trend[1:8],
                    [101.0722217344, 101.4435261065, 101.8148304786, 102.1861348507,
                     102.5574392228, 102.9287435949, 103.2853406442, 103.7007669848]; atol=1e-8)

    # Test H: extrapolate_trend=:freq (= period-1 = 11) -- more points in the
    # fit than G -> a different extrapolated line
    rH = classical_decompose(monthly, 12; model=:additive, extrapolate_trend=:freq)
    @test !any(isnan, rH.trend)
    @test isapprox(rH.trend[1:8],
                    [100.6354262657, 101.0561320013, 101.4768377369, 101.8975434726,
                     102.3182492082, 102.7389549438, 103.2853406442, 103.7007669848]; atol=1e-8)
    @test rG.trend[1] != rH.trend[1]

    # Middle/unaffected trend values are untouched by extrapolation
    @test isapprox(rG.trend[20:22], rA.trend[20:22]; atol=1e-10)
    @test isapprox(rH.trend[20:22], rA.trend[20:22]; atol=1e-10)

    # defaults unaffected by the Python-exclusive features existing at all
    rA2 = classical_decompose(monthly, 12; model=:additive, two_sided=true, extrapolate_trend=0)
    @test isapprox(rA2.trend[20:22], rA.trend[20:22]; atol=1e-10)
    @test isequal(rA2.trend, rA.trend)
end

@testset "classical_decompose validation (matches both languages' verified error behavior)" begin
    monthly = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))

    # Test I: internal NaN rejected, matching both R's and statsmodels'
    # actual (verified, not assumed) behavior
    y_nan = copy(monthly); y_nan[5] = NaN
    @test_throws ArgumentError classical_decompose(y_nan, 12)
    y_inf = copy(monthly); y_inf[5] = Inf
    @test_throws ArgumentError classical_decompose(y_inf, 12)

    # Test J: multiplicative + non-positive rejected -- Julia matches
    # Python's stricter behavior, not R's documented silent failure
    y_neg_full = repeat([1.0, -2.0, 3.0], 16)
    @test_throws ArgumentError classical_decompose(y_neg_full, 12; model=:multiplicative)
    y_zero_full = repeat([1.0, 0.0, 3.0], 16)
    @test_throws ArgumentError classical_decompose(y_zero_full, 12; model=:multiplicative)
    # ...but the SAME series is fine under :additive (no positivity requirement)
    r_neg_additive = classical_decompose(y_neg_full, 12; model=:additive)
    @test isfinite(r_neg_additive.figure[1])

    @test_throws ArgumentError classical_decompose(monthly, 12; model=:bogus)
    @test_throws ArgumentError classical_decompose(monthly, 1)          # period must be >= 2
    @test_throws ArgumentError classical_decompose(monthly[1:20], 12)   # < 2 full periods
end

@testset "classical_decompose exhaustive parameter coverage" begin
    monthly = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))
    mult_monthly = vec(readdlm(TSAnalytics.MULT_MONTHLY, ','; skipstart=1, header=false))
    period7 = vec(readdlm(TSAnalytics.PERIOD7, ','; skipstart=1, header=false))

    # every model x period-parity combination (period parity affects the
    # filter's odd/even branch, a genuinely distinct code path)
    cases = [
        (monthly, 12, :additive),
        (mult_monthly, 12, :multiplicative),
        (period7, 7, :additive),
    ]
    for (data, period, model) in cases
        r = classical_decompose(data, period; model=model)
        @test length(r.trend) == length(data)
        @test length(r.seasonal) == length(data)
        @test length(r.resid) == length(data)
        @test length(r.figure) == period
        @test r.period == period
        @test r.model == model
    end

    # every two_sided value
    for two_sided in (true, false)
        r = classical_decompose(monthly, 12; two_sided=two_sided)
        @test length(r.trend) == 48
    end

    # every extrapolate_trend "shape": 0, positive Int, :freq
    for ep in (0, 1, 2, 5, :freq)
        r = classical_decompose(monthly, 12; extrapolate_trend=ep)
        if ep == 0
            @test any(isnan, r.trend)
        else
            @test !any(isnan, r.trend)
        end
    end

    # seasonal is figure tiled cyclically -- true for every period/model combo
    for (data, period, model) in cases
        r = classical_decompose(data, period; model=model)
        for i in eachindex(r.seasonal)
            @test r.seasonal[i] == r.figure[mod1(i, period)]
        end
    end

    # container-agnostic
    r1 = classical_decompose(monthly, 12)
    r2 = classical_decompose(collect(Float32.(monthly)), 12)
    @test isapprox(r1.trend, r2.trend; nans=true, atol=1e-4)
end

@testset "_extrapolate_trend! direct verification" begin
    # Isolates the extrapolation helper itself against the exact reference
    # numbers (test G/H above already check it end-to-end; this checks it
    # directly, with a hand-constructed case verifying the corrected
    # index arithmetic specifically).
    trend = [NaN, NaN, 10.0, 12.0, 14.0, 16.0, NaN, NaN]
    TSAnalytics._extrapolate_trend!(trend, 2)
    # front0=2 (0-idx), npoints=2 -> front_last0=min(2+2,5)=4, fit x=[2,3],y=[10,12] -> k=2,n=6
    # extrapolate i0=0,1: trend[1]=6, trend[2]=8 (Julia 1-indexed storage)
    @test isapprox(trend[1:2], [6.0, 8.0]; atol=1e-10)
    # back0=5, back_first0=max(2,5-2)=3, fit x=[3,4],y=[12,14] -> k=2,n=6
    # extrapolate i0=6,7: trend[7]=18, trend[8]=20
    @test isapprox(trend[7:8], [18.0, 20.0]; atol=1e-10)
    @test trend[3:6] == [10.0, 12.0, 14.0, 16.0]  # untouched

    @test_throws ArgumentError TSAnalytics._extrapolate_trend!(fill(NaN, 5), 2)
end
