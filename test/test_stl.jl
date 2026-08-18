using DelimitedFiles

@testset "stl_decompose matched-parameter agreement (exact R/Python validation)" begin
    # handoff/verification/stage-3.2-transcript.txt section 3(e): both R
    # stl() and Python STL() run on identical data with fully explicit,
    # matched parameters -- independently re-confirmed against real
    # statsmodels.tsa.seasonal.STL in this session, not just transcribed.
    monthly = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))
    @test length(monthly) == 48

    r = stl_decompose(monthly, 12; seasonal_window=7, seasonal_degree=0,
                       trend_window=19, trend_degree=1,
                       low_pass_window=13, low_pass_degree=1,
                       robust=false, inner=2)

    @test isapprox(r.trend[20:24],
                    [108.958049582, 109.5046508904, 110.0658508548, 110.6210295545, 111.1650202872];
                    atol=1e-6)
    @test isapprox(r.seasonal[1:12],
                    [-0.3166825021, 3.842917142, 7.690724169, 10.31761029, 8.135563195,
                     5.096057761, 0.3674976123, -4.393395326, -7.932231049, -9.59928204,
                     -8.046860888, -5.281630535]; atol=1e-6)
    @test isapprox(r.resid[20:24],
                    [-1.5116130889, 1.3232460943, -0.0290988953, -0.1609326636, -0.7954689598];
                    atol=1e-6)
    @test r.period == 12
    @test r.observed == monthly
    @test isapprox(r.observed, r.trend .+ r.seasonal .+ r.resid; atol=1e-8)  # decomposition identity
end

@testset "stl_decompose auto-formula defaults" begin
    monthly = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))

    # trend_window auto-formula: nextodd(ceil(1.5*period/(1-1.5/s_window)))
    # for period=12, s_window=7: 1.5*12/(1-1.5/7) = 18/(1-0.214285...) = 18/0.785714... = 22.9091
    # ceil -> 23, already odd -> 23
    @test TSAnalytics._nextodd(ceil(Int, 1.5*12/(1-1.5/7))) == 23

    r_auto = stl_decompose(monthly, 12; seasonal_window=7)
    r_explicit = stl_decompose(monthly, 12; seasonal_window=7, trend_window=23)
    @test isapprox(r_auto.trend, r_explicit.trend; atol=1e-10)

    # low_pass_window auto-formula: nextodd(period) = nextodd(12) = 13
    @test TSAnalytics._nextodd(12) == 13
    r_lp_auto = stl_decompose(monthly, 12; seasonal_window=7)
    r_lp_explicit = stl_decompose(monthly, 12; seasonal_window=7, low_pass_window=13)
    @test isapprox(r_lp_auto.trend, r_lp_explicit.trend; atol=1e-10)

    # low_pass_degree follows trend_degree by default (R's dynamic coupling)
    r_deg0 = stl_decompose(monthly, 12; seasonal_window=7, trend_degree=0)
    r_deg0_explicit = stl_decompose(monthly, 12; seasonal_window=7, trend_degree=0, low_pass_degree=0)
    @test isapprox(r_deg0.trend, r_deg0_explicit.trend; atol=1e-10)
end

@testset "stl_decompose non-exact-multiple period (exact statsmodels validation)" begin
    # Regression test for a real bug: _stl_cycle_subseries!'s one-point-after
    # extension used a fixed `period+n+i` index, correct only when every
    # phase has the SAME subseries length (true for every prior exact-match
    # test here: period=12,n=48 and period=7,n=42 both divide evenly). Found
    # via mstl_decompose's period=168,n=500 (subseries lengths 2 and 3 in the
    # same call) -- this n=20,period=8 case (subseries lengths 2 and 3 too)
    # is the smallest reproduction, independently verified against real
    # statsmodels rather than just re-confirming the fix doesn't crash.
    y = [11.789, 10.437, 10.096, 8.137, 9.723, 9.645, 9.917, 9.373, 9.956, 9.523,
         8.686, 10.885, 10.881, 11.71, 10.05, 9.595, 9.455, 8.454, 10.982, 8.899]
    r = stl_decompose(y, 8; seasonal_window=7, inner=3)

    @test isapprox(r.trend,
                    [9.678130531239564, 9.68706001692203, 9.699111019928242, 9.71429111966731,
                     9.731842826189865, 9.751577051881243, 9.773508237383362, 9.797789152142672,
                     9.831186609426194, 9.889559308598983, 9.947699606576386, 10.000637178877447,
                     10.050611220059535, 10.086978355250192, 10.118530588535256, 10.14719645536704,
                     10.173556093696105, 10.197751352005573, 10.219574695883185, 10.23861029820903];
                    atol=1e-6)
    @test isapprox(r.seasonal,
                    [1.9330081891144935, 0.7823676756320924, -0.15992455220314875, -0.8818850556055269,
                     -0.013833000023667295, -0.11051350206196589, 0.1409117703954048, -0.4259766313898452,
                     0.49233860791804324, -0.4478719510173138, -0.07826048329538045, -0.6136601081386038,
                     0.8354885600965141, 1.6295209324458435, -0.0603267454791427, -0.5422501998154431,
                     -0.8788353998556336, -1.692791535687953, 0.2249006567480924, -0.6243681832797181];
                    atol=1e-6)
    @test isapprox(r.observed, r.trend .+ r.seasonal .+ r.resid; atol=1e-8)

    # every phase's subseries length must actually vary in this test, or it
    # wouldn't reproduce the bug -- guard the test's own premise
    lengths = [length(i:8:20) for i in 1:8]
    @test length(unique(lengths)) > 1
end

@testset "stl_decompose second matched-parameter combo (exact statsmodels validation)" begin
    # A fully independent parameter set (different s/t/l windows and degrees,
    # inner=3) from the "matched-parameter agreement" testset above, so degree=1
    # seasonal smoothing and degree=0 trend/low-pass smoothing both get an
    # exact-value check, not just finiteness. Generated directly against real
    # statsmodels.tsa.seasonal.STL in this session (not transcribed).
    monthly = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))
    r = stl_decompose(monthly, 12; seasonal_window=11, seasonal_degree=1,
                       trend_window=27, trend_degree=0,
                       low_pass_window=15, low_pass_degree=1, inner=3)

    @test isapprox(r.trend[20:24],
                    [109.05485579050567, 109.57317105650822, 110.09353033820146,
                     110.61635670234425, 111.1410584479628]; atol=1e-6)
    @test isapprox(r.seasonal[1:12],
                    [-0.589628808870994, 3.769573821985424, 8.090665181304598, 10.53448516261923,
                     7.771345261475801, 4.899603853909164, 0.901988231662049, -4.564525261430473,
                     -7.797297199707199, -9.17515374564158, -8.32305412438095, -5.63161394411634];
                    atol=1e-6)
    @test isapprox(r.resid[20:24],
                    [-1.534218327960886, 1.3245187739607758, -0.2096857652676789,
                     -0.11383892975344168, -0.9411470409735045]; atol=1e-6)
    @test isapprox(r.observed, r.trend .+ r.seasonal .+ r.resid; atol=1e-8)
end

@testset "stl_decompose odd-period matched combo (exact statsmodels validation)" begin
    # period=7 with its own distinct window/degree/inner set -- exercises the
    # odd-period path with exact numbers, not just the finiteness check the
    # earlier period=7 coverage used.
    period7 = vec(readdlm(TSAnalytics.PERIOD7, ','; skipstart=1, header=false))
    r = stl_decompose(period7, 7; seasonal_window=9, seasonal_degree=0,
                       trend_window=17, trend_degree=1,
                       low_pass_window=9, low_pass_degree=0, inner=4)

    @test isapprox(r.trend[10:14],
                    [52.73580834878437, 53.0123794227175, 53.29165265252992,
                     53.57940815587033, 53.88556592748009]; atol=1e-6)
    @test isapprox(r.seasonal[1:7],
                    [-0.03236672400422741, 3.436961564877171, 5.074870646400431, 2.2088387394028706,
                     -2.043893462635601, -4.986212529294717, -3.6580917519441374]; atol=1e-6)
    @test isapprox(r.resid[10:14],
                    [-0.4014864976591497, 0.13213801260018698, 0.3568775407418556,
                     -0.08025334655407335, -0.344115901538693]; atol=1e-6)
    @test isapprox(r.observed, r.trend .+ r.seasonal .+ r.resid; atol=1e-8)
end

@testset "stl_decompose robust/outer semantics" begin
    # Verified directly against R's stl.R (.Fortran() call takes only numeric
    # ni/no counts, no robust flag) and Python's _stl.pyx (`robust` only picks
    # inner/outer *defaults*): the outer/robustness loop is gated by the
    # *effective* outer count, not by `robust` itself -- so `robust=false`
    # with an explicit `outer>0` runs the outer loop too, and `robust=true`
    # with an explicit `outer=0` does NOT.
    monthly = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))

    @test_throws ArgumentError stl_decompose(monthly, 12; outer=-1)  # outer must be >= 0, checked regardless of robust

    # every combination below now succeeds (the outer loop is implemented)
    for (robust, outer) in ((true, nothing), (false, 1), (false, 5), (true, 1), (true, 5))
        r = stl_decompose(monthly, 12; seasonal_window=7, robust=robust, outer=outer)
        @test all(isfinite, r.trend)
        @test all(isfinite, r.seasonal)
        @test all(w -> 0 <= w <= 1, r.weights)
    end

    # robust=false, outer=0/nothing -- the outer loop skipped entirely -- must
    # match each other, and `weights` must be all-ones (no reweighting happened)
    r1 = stl_decompose(monthly, 12; seasonal_window=7, robust=false, outer=0)
    r2 = stl_decompose(monthly, 12; seasonal_window=7)  # outer defaults to nothing -> 0
    @test isapprox(r1.trend, r2.trend; atol=1e-10)
    @test all(==(1.0), r1.weights)
    @test all(==(1.0), r2.weights)

    # robust=true with explicit outer=0 must match robust=false with the same
    # matched inner/outer exactly -- confirmed bit-identical against real
    # statsmodels -- since `robust` itself has no computational effect.
    r3 = stl_decompose(monthly, 12; seasonal_window=7, robust=true, outer=0, inner=2)
    r4 = stl_decompose(monthly, 12; seasonal_window=7, robust=false, outer=0, inner=2)
    @test r3.trend == r4.trend
    @test r3.seasonal == r4.seasonal
    @test r3.weights == r4.weights == ones(length(monthly))
end

@testset "stl_decompose outer/robustness loop (exact statsmodels validation)" begin
    # handoff/stage-3.2-transcript.txt section 3(f): monthly_outlier.csv is
    # monthly.csv with two injected outliers (0-indexed y[10]+=30, y[30]-=25)
    # -- large enough to actually engage robustness reweighting. Deliberately
    # targets Python's *fixed* bisquare/median computation (see _stl_robustness_weights
    # docstring), not R's documented-buggy original -- independently
    # re-generated against real statsmodels in this session, not transcribed.
    monthly_outlier = vec(readdlm(TSAnalytics.MONTHLY_OUTLIER, ','; skipstart=1, header=false))
    @test length(monthly_outlier) == 48

    r = stl_decompose(monthly_outlier, 12; seasonal_window=7, seasonal_degree=0,
                       trend_window=19, trend_degree=1, low_pass_window=13, low_pass_degree=1,
                       robust=true, inner=2, outer=15)

    @test isapprox(r.trend[10:12],
                    [104.55139049302393, 104.92959741986593, 105.31459037951534]; atol=1e-6)
    @test isapprox(r.trend[30:32],
                    [113.98800792373589, 114.51875131166825, 115.0404297383129]; atol=1e-6)
    @test isapprox(r.observed, r.trend .+ r.seasonal .+ r.resid; atol=1e-8)

    # the two injected outliers (1-indexed positions 11 and 31) must be fully
    # excluded (weight 0); a comfortably-non-outlying point must keep weight 1
    @test r.weights[11] == 0.0
    @test r.weights[31] == 0.0
    @test r.weights[1] > 0.5

    # `outer` runs `outer` reweighting passes on top of the first uniform-weight
    # pass, so results must generally differ from outer=0 on data with real
    # outliers (the whole point of robust=true) -- a sanity check that `outer`
    # isn't silently a no-op now that it's implemented
    r_norobust = stl_decompose(monthly_outlier, 12; seasonal_window=7, seasonal_degree=0,
                                trend_window=19, trend_degree=1, low_pass_window=13,
                                low_pass_degree=1, inner=2, outer=0)
    @test !isapprox(r.trend, r_norobust.trend; atol=1e-3)
    @test all(==(1.0), r_norobust.weights)

    # every `outer` value: more passes should keep converging, not diverge or NaN
    for outer in (1, 5, 15, 30)
        ro = stl_decompose(monthly_outlier, 12; seasonal_window=7, robust=true, outer=outer)
        @test all(isfinite, ro.trend)
        @test all(w -> 0 <= w <= 1, ro.weights)
    end
end

@testset "stl_decompose parameter coverage and error paths" begin
    monthly = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))
    period7 = vec(readdlm(TSAnalytics.PERIOD7, ','; skipstart=1, header=false))

    # every seasonal_degree / trend_degree / low_pass_degree value
    for sdeg in (0, 1), tdeg in (0, 1), ldeg in (0, 1)
        r = stl_decompose(monthly, 12; seasonal_window=7, seasonal_degree=sdeg,
                           trend_degree=tdeg, low_pass_degree=ldeg)
        @test all(isfinite, r.trend)
        @test all(isfinite, r.seasonal)
    end

    # every inner value
    for inner in (1, 2, 5)
        r = stl_decompose(monthly, 12; seasonal_window=7, inner=inner)
        @test all(isfinite, r.trend)
    end

    # odd period works too
    r7 = stl_decompose(period7, 7; seasonal_window=7)
    @test length(r7.trend) == length(period7)
    @test r7.period == 7

    # explicit trend_window / low_pass_window / seasonal_window variety, crossed
    # with seasonal_degree/trend_degree so the window sweep isn't degree=default-only
    for sw in (5, 7, 9), tw in (13, 19, 25), lw in (13, 15, 17)
        r = stl_decompose(monthly, 12; seasonal_window=sw, seasonal_degree=sw % 2,
                           trend_window=tw, trend_degree=tw % 3 == 0 ? 1 : 0, low_pass_window=lw)
        @test all(isfinite, r.trend)
        @test all(isfinite, r.seasonal)
    end

    # decomposition identity holds across every combination tried above
    r_check = stl_decompose(monthly, 12; seasonal_window=9, seasonal_degree=0,
                             trend_degree=0, inner=3)
    @test isapprox(r_check.observed, r_check.trend .+ r_check.seasonal .+ r_check.resid; atol=1e-8)

    # error paths
    @test_throws ArgumentError stl_decompose(monthly, 1)                     # period < 2
    @test_throws ArgumentError stl_decompose(monthly[1:20], 12)              # < 2 full periods
    @test_throws ArgumentError stl_decompose(monthly, 12; seasonal_degree=2) # only 0/1 supported
    @test_throws ArgumentError stl_decompose(monthly, 12; trend_degree=2)
    @test_throws ArgumentError stl_decompose(monthly, 12; low_pass_degree=2)
    @test_throws ArgumentError stl_decompose(monthly, 12; inner=0)
    y_nan = copy(monthly); y_nan[5] = NaN
    @test_throws ArgumentError stl_decompose(y_nan, 12)

    # seasonal_window: must be odd and >= 3 (R silently auto-corrects via its
    # Fortran source; this package deliberately errors instead, matching Python)
    @test_throws ArgumentError stl_decompose(monthly, 12; seasonal_window=6)   # even
    @test_throws ArgumentError stl_decompose(monthly, 12; seasonal_window=1)   # < 3
    @test_throws ArgumentError stl_decompose(monthly, 12; seasonal_window=2)   # even and < 3
    @test stl_decompose(monthly, 12; seasonal_window=3) isa TSAnalytics.STLDecomposition  # minimum allowed

    # trend_window: odd, >= 3, and > period, when given explicitly
    @test_throws ArgumentError stl_decompose(monthly, 12; trend_window=20)     # even
    @test_throws ArgumentError stl_decompose(monthly, 12; trend_window=11)     # odd but <= period (12)
    @test_throws ArgumentError stl_decompose(monthly, 12; trend_window=1)      # odd but < 3
    @test stl_decompose(monthly, 12; trend_window=13) isa TSAnalytics.STLDecomposition  # 13 > 12, minimal valid

    # low_pass_window: odd, >= 3, and > period, when given explicitly
    @test_throws ArgumentError stl_decompose(monthly, 12; low_pass_window=14)  # even
    @test_throws ArgumentError stl_decompose(monthly, 12; low_pass_window=11)  # odd but <= period (12)
    @test_throws ArgumentError stl_decompose(monthly, 12; low_pass_window=1)   # odd but < 3
    @test stl_decompose(monthly, 12; low_pass_window=13) isa TSAnalytics.STLDecomposition  # 13 > 12, minimal valid

    # container-agnostic
    r1 = stl_decompose(monthly, 12; seasonal_window=7)
    r2 = stl_decompose(Float32.(monthly), 12; seasonal_window=7)
    @test isapprox(r1.trend, r2.trend; atol=1e-3)
end

@testset "stl_decompose parallel keyword" begin
    # Stage 3.3's threading retrofit: the cycle-subseries step is provably
    # embarrassingly parallel (see _stl_cycle_subseries! docstring), so
    # parallel=true/false must be numerically IDENTICAL, never just close --
    # this is a correctness invariant, not a performance-only knob. Only
    # actually exercises the Threads.@threads branch when this test process
    # has more than one thread (CI sets JULIA_NUM_THREADS=4 for exactly this
    # reason); under the default single-threaded `julia`, parallel=true still
    # runs (falls back to the sequential loop via the nthreads()>1 guard), so
    # this equivalence check is meaningful either way.
    monthly = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))
    period7 = vec(readdlm(TSAnalytics.PERIOD7, ','; skipstart=1, header=false))
    monthly_outlier = vec(readdlm(TSAnalytics.MONTHLY_OUTLIER, ','; skipstart=1, header=false))

    # period=12 (>= 4, the threading guard's floor) -- non-robust
    r_par = stl_decompose(monthly, 12; seasonal_window=7, robust=false)
    r_seq = stl_decompose(monthly, 12; seasonal_window=7, robust=false, parallel=false)
    @test r_par.trend == r_seq.trend
    @test r_par.seasonal == r_seq.seasonal

    # period=7 (odd, still >= 4)
    r_par7 = stl_decompose(period7, 7; seasonal_window=7)
    r_seq7 = stl_decompose(period7, 7; seasonal_window=7, parallel=false)
    @test r_par7.trend == r_seq7.trend

    # robust=true/outer>0 -- cycle-subseries runs once per outer pass, every
    # pass must still match exactly under threading
    r_par_rob = stl_decompose(monthly_outlier, 12; seasonal_window=7, robust=true, outer=5)
    r_seq_rob = stl_decompose(monthly_outlier, 12; seasonal_window=7, robust=true, outer=5, parallel=false)
    @test r_par_rob.trend == r_seq_rob.trend
    @test r_par_rob.weights == r_seq_rob.weights

    # period < 4 -- below the threading guard's floor, so parallel=true is a
    # guaranteed no-op (always the sequential branch); still must equal
    # parallel=false explicitly, not just "not crash"
    small = vec(readdlm(TSAnalytics.MONTHLY, ','; skipstart=1, header=false))[1:12]
    r_par2 = stl_decompose(small, 2; seasonal_window=3)
    r_seq2 = stl_decompose(small, 2; seasonal_window=3, parallel=false)
    @test r_par2.trend == r_seq2.trend

    # default (unspecified) must equal explicit parallel=true
    r_default = stl_decompose(monthly, 12; seasonal_window=7)
    @test r_default.trend == r_par.trend
end

@testset "_nextodd" begin
    @test TSAnalytics._nextodd(4) == 5
    @test TSAnalytics._nextodd(5) == 5
    @test TSAnalytics._nextodd(4.1) == 5
    @test TSAnalytics._nextodd(12) == 13
    @test TSAnalytics._nextodd(23.0) == 23
end
