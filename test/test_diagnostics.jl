using DelimitedFiles

@testset "diagnostics" begin
    # chi-square upper tail sanity checks against well-known textbook values
    @test isapprox(TSAnalytics._chisq_ccdf(3.841, 1), 0.05; atol=1e-3)
    @test isapprox(TSAnalytics._chisq_ccdf(5.991, 2), 0.05; atol=1e-3)
    @test isapprox(TSAnalytics._chisq_ccdf(9.488, 4), 0.05; atol=1e-3)

    # seed=3 (the original choice) deterministically gives a white-noise
    # p-value of ~0.0396 -- just past the 0.05 threshold by chance, on
    # every platform (Julia's Xoshiro RNG is seed-reproducible across
    # OSes), which is why this was misdiagnosed as "flaky" for a long
    # time rather than fixed: it never actually varied run to run, it was
    # simply always wrong. seed=6 gives a comfortable p=0.52, nowhere
    # near either boundary.
    Random.seed!(6)
    n = 1000

    # White noise: Ljung-Box should NOT reject (high p-value)
    wn = randn(n)
    lb_wn = ljungbox_test(wn, 10)
    @test lb_wn.pvalue > 0.05

    # Strongly autocorrelated AR(1): Ljung-Box SHOULD reject
    e = randn(n)
    ar1 = zeros(n)
    for t in 2:n
        ar1[t] = 0.8*ar1[t-1] + e[t]
    end
    lb_ar1 = ljungbox_test(ar1, 10)
    @test lb_ar1.pvalue < 0.01

    # fitdf reduces degrees of freedom
    lb_fit = ljungbox_test(wn, 10; fitdf=2)
    @test lb_fit.df == 8
    @test_throws ArgumentError ljungbox_test(wn, 2; fitdf=5)

    # QS test: seasonal series (period 12) should show strong residual seasonality
    t = 1:n
    seasonal = sin.(2π .* t ./ 12) .+ 0.1 .* randn(n)
    qs_seasonal = qs_test(seasonal, 12)
    @test qs_seasonal.pvalue < 0.01

    # QS test: white noise should not show seasonality at period 12
    qs_wn = qs_test(wn, 12)
    @test qs_wn.pvalue > 0.05
end

@testset "ljungbox_test lags semantics x boxpierce (exact statsmodels validation)" begin
    fixture = joinpath(@__DIR__, "fixtures", "ar1_ref_series.csv")
    y = vec(readdlm(fixture, ','; skipstart=1, header=false))
    @test length(y) == 200

    # statsmodels.stats.diagnostic.acorr_ljungbox(y, lags=h, boxpierce=True) --
    # Integer input IS cumulative, matching Python's per-lag row exactly.
    r5 = ljungbox_test(y, 5; boxpierce=true)
    @test isapprox(r5.statistic, 163.171707; atol=1e-4)
    @test isapprox(r5.bp_statistic, 160.064683; atol=1e-4)

    r10 = ljungbox_test(y, 10; boxpierce=true)
    @test isapprox(r10.statistic, 216.153792; atol=1e-4)
    @test isapprox(r10.bp_statistic, 210.481973; atol=1e-4)

    # Vector input sums EXACTLY those lags -- independently computed from
    # statsmodels' own acf() values applied to the exact-set formula (not
    # something statsmodels' acorr_ljungbox can produce directly, since it
    # only offers cumulative rows) -- deliberately different from r10 above.
    r_vec = ljungbox_test(y, [5, 10]; boxpierce=true)
    @test isapprox(r_vec.statistic, 16.641913; atol=1e-4)
    @test isapprox(r_vec.bp_statistic, 15.889696; atol=1e-4)
    @test r_vec.statistic != r10.statistic
    @test r_vec.lags == [5, 10]

    # default lags: min(10, n÷5) = min(10, 40) = 10
    r_default = ljungbox_test(y)
    @test length(r_default.lags) == 10
    @test isapprox(r_default.statistic, r10.statistic; atol=1e-8)  # same as cumulative h=10

    # boxpierce=false (default): fields are nothing, not silently zero
    @test ljungbox_test(y, 10).bp_statistic === nothing
    @test ljungbox_test(y, 10).bp_pvalue === nothing

    # Ljung-Box's (n+2)/(n-k) scaling always inflates it relative to Box-Pierce
    for lags in (5, 10, [5, 10], [3, 7, 12])
        r = ljungbox_test(y, lags; boxpierce=true)
        @test r.statistic > r.bp_statistic
    end

    # backward-compatible 4-positional-arg LjungBoxTest constructor
    manual = LjungBoxTest(1.0, 0.5, [1, 2], 2)
    @test manual.bp_statistic === nothing && manual.bp_pvalue === nothing
end

@testset "ljungbox_test parameter validation and edge cases" begin
    Random.seed!(8)
    y = randn(500)

    # every lags "shape" runs and gives a finite result
    for lags in (nothing, 5, 10, [3, 7], [1, 2, 3, 4, 5])
        r = lags === nothing ? ljungbox_test(y) : ljungbox_test(y, lags)
        @test isfinite(r.statistic)
        @test isfinite(r.pvalue)
    end

    # both boxpierce values run
    for boxpierce in (true, false)
        r = ljungbox_test(y, 10; boxpierce=boxpierce)
        @test isfinite(r.statistic)
        @test (r.bp_statistic !== nothing) == boxpierce
    end

    # fitdf reduces df across several values
    for fitdf in (0, 1, 3, 5)
        r = ljungbox_test(y, 10; fitdf=fitdf)
        @test r.df == 10 - fitdf
    end

    @test_throws ArgumentError ljungbox_test(y, 5; fitdf=5)   # df == 0
    @test_throws ArgumentError ljungbox_test(y, 5; fitdf=10)  # df < 0

    # container-agnostic
    @test ljungbox_test(1:100).statistic == ljungbox_test(collect(1.0:100)).statistic
end

@testset "qs_test clipping (thin wrapper over ljungbox_test)" begin
    # Reconstructs the handoff's own worked example: a period-24 seasonal
    # pattern sampled at lag 12 sits at a trough, giving a genuinely
    # negative lag-12 autocorrelation -- exactly the case clipping matters.
    Random.seed!(7)
    n = 240
    t = 0:n-1
    y = sin.(2π .* t ./ 24) .+ 0.3 .* randn(n)

    rho12 = acf(y, [12]).values[1]
    @test rho12 < 0   # confirms this construction actually exercises the clip

    r = qs_test(y, 12)
    lb_unclipped = ljungbox_test(y, [12, 24]; clip_negative=false)
    # clipping the negative lag-12 contribution to zero can only reduce the statistic
    @test r.statistic < lb_unclipped.statistic

    # wrapper must EXACTLY match calling ljungbox_test directly with
    # clip_negative=true -- confirms real delegation, not parallel logic
    lb_clipped = ljungbox_test(y, [12, 24]; clip_negative=true)
    @test isapprox(r.statistic, lb_clipped.statistic; atol=1e-10)
    @test isapprox(r.pvalue, lb_clipped.pvalue; atol=1e-10)
    @test r.period == 12

    @test_throws ArgumentError qs_test(y, 1)   # period must be >= 2
    @test_throws ArgumentError qs_test(y[1:20], 12)  # series too short relative to period
end

@testset "clip_negative general behavior and parameter coverage" begin
    Random.seed!(9)
    y = randn(300)  # white noise -- clipping shouldn't spuriously inflate significance

    # clipped statistic can never exceed the unclipped one, term-by-term,
    # across every lags "shape"
    for lags in (5, 10, [3, 7], [1, 2, 3, 4, 5])
        r_clipped = ljungbox_test(y, lags; clip_negative=true)
        r_unclipped = ljungbox_test(y, lags; clip_negative=false)
        @test r_clipped.statistic <= r_unclipped.statistic
    end

    # clip_negative composes with boxpierce and fitdf independently
    r_combo = ljungbox_test(y, 10; clip_negative=true, boxpierce=true, fitdf=2)
    @test isfinite(r_combo.statistic)
    @test r_combo.bp_statistic !== nothing
    @test r_combo.df == 8

    # every period value for qs_test runs and gives a finite result
    for period in (2, 4, 12, 24)
        r = qs_test(y, period)
        @test isfinite(r.statistic)
        @test isfinite(r.pvalue)
        @test r.period == period
    end

    # container-agnostic
    @test qs_test(1:100, 4).statistic == qs_test(collect(1.0:100), 4).statistic
end

@testset "jarque_bera_test (exact statsmodels validation)" begin
    fixture = joinpath(@__DIR__, "fixtures", "ar1_ref_series.csv")
    y = vec(readdlm(fixture, ','; skipstart=1, header=false))

    # statsmodels.stats.stattools.jarque_bera(y) -- exact match expected
    # (no cross-language discrepancy in this formula, unlike most of Stage 2)
    r = jarque_bera_test(y)
    @test isapprox(r.statistic, 1.6772996836; atol=1e-8)
    @test isapprox(r.skewness, 0.1145669775; atol=1e-8)
    @test isapprox(r.kurtosis, 2.6142881999; atol=1e-8)
    @test isapprox(r.pvalue, 0.4322937946; atol=1e-6)
    @test r.n == 200

    # Normal data: should NOT reject normality
    Random.seed!(1)
    y_normal = randn(2000)
    r_normal = jarque_bera_test(y_normal)
    @test r_normal.pvalue > 0.05
    @test r_normal.n == 2000

    # Strongly skewed/heavy-tailed data (squared normal ~ chi-sq(1)):
    # should reject normality
    y_skewed = randn(MersenneTwister(1), 2000).^2
    r_skewed = jarque_bera_test(y_skewed)
    @test r_skewed.pvalue < 0.01
    @test r_skewed.skewness > 1.0   # chi-sq(1) has skewness = sqrt(8) ~= 2.83

    # Symmetric heavy-tailed (Student's t, 3 df, via ratio of normals):
    # near-zero skewness, excess kurtosis -> should reject normality
    tdist = randn(MersenneTwister(2), 3000) ./ sqrt.(sum(randn(MersenneTwister(3), 3000, 3).^2; dims=2)[:] ./ 3)
    r_t = jarque_bera_test(tdist)
    @test r_t.pvalue < 0.01
    @test r_t.kurtosis > 3.0   # heavier tails than normal

    @test_throws ArgumentError jarque_bera_test([1.0])   # n < 2
    @test_throws ArgumentError jarque_bera_test(Float64[])

    # container-agnostic
    @test jarque_bera_test(1:100).statistic == jarque_bera_test(collect(1.0:100)).statistic
end

@testset "durbin_watson_test (exact statsmodels/R validation)" begin
    # ground truth: test/verification/durbinwatson/gen_and_verify.py (real
    # statsmodels.stats.stattools.durbin_watson) + fit_r.R (real R
    # lmtest::dwtest, exact p-values, cross-checked separately below)
    d_ar1 = readdlm(joinpath(@__DIR__, "verification", "durbinwatson", "durbinwatson_ar1.csv"), ','; skipstart=1)
    x_ar1, y_ar1 = d_ar1[:, 1], d_ar1[:, 2]
    _, resid_ar1, _ = TSAnalytics._ols(hcat(ones(length(x_ar1)), x_ar1), y_ar1)

    d_wn = readdlm(joinpath(@__DIR__, "verification", "durbinwatson", "durbinwatson_wn.csv"), ','; skipstart=1)
    x_wn, y_wn = d_wn[:, 1], d_wn[:, 2]
    _, resid_wn, _ = TSAnalytics._ols(hcat(ones(length(x_wn)), x_wn), y_wn)

    # Case A: AR(1) residuals (phi=0.6) -- strong positive autocorrelation.
    # statsmodels: 0.6363488878642116; R lmtest::dwtest: 0.6363489 (both agree)
    rA = durbin_watson_test(resid_ar1)
    @test isapprox(rA.statistic, 0.6363488878642116; atol=1e-8)
    @test 0 <= rA.statistic <= 4
    @test rA.pvalue < 0.001   # R's exact p-value here is 2.95e-12; :approx gives 4.6e-12, same order
    @test rA.alternative == :greater
    @test rA.method == :approx
    @test rA.n == 100

    # Case B: white-noise residuals -- no strong autocorrelation.
    # statsmodels: 1.7993979646727234; R lmtest::dwtest: 1.799398 (both agree)
    rB = durbin_watson_test(resid_wn)
    @test isapprox(rB.statistic, 1.7993979646727234; atol=1e-8)
    @test 0 <= rB.statistic <= 4
    @test rB.pvalue > 0.1    # R's exact p-value here is 0.1564; :approx gives 0.1579, close
    @test rB.pvalue < 0.2

    # alternative: :greater, :less, :two_sided all covered
    rA_less = durbin_watson_test(resid_ar1; alternative=:less)
    @test rA_less.pvalue > 0.99   # data shows POSITIVE autocorrelation; testing for NEGATIVE should fail hard
    rA_two = durbin_watson_test(resid_ar1; alternative=:two_sided)
    @test rA_two.pvalue < 0.001
    @test isapprox(rA_two.pvalue, 2 * rA.pvalue; atol=1e-10)   # two-sided is 2x the smaller one-sided tail here

    # statistic()/pvalue() generic HypothesisTest accessors
    @test statistic(rA) == rA.statistic
    @test pvalue(rA) == rA.pvalue

    # container-agnostic (tsvalues interface)
    @test durbin_watson_test(resid_ar1).statistic == durbin_watson_test(collect(resid_ar1)).statistic

    # error paths
    @test_throws ArgumentError durbin_watson_test(resid_ar1; alternative=:bogus)
    @test_throws ArgumentError durbin_watson_test(resid_ar1; method=:bogus)
    @test_throws ArgumentError durbin_watson_test(resid_ar1; method=:exact)  # deliberately unimplemented
    @test_throws ArgumentError durbin_watson_test(Float64[])
    @test_throws ArgumentError durbin_watson_test([1.0])

    # _std_normal_cdf sanity against well-known textbook values
    @test isapprox(TSAnalytics._std_normal_cdf(0.0), 0.5; atol=1e-10)
    @test isapprox(TSAnalytics._std_normal_cdf(1.96), 0.975; atol=1e-4)
    @test isapprox(TSAnalytics._std_normal_cdf(-1.96), 0.025; atol=1e-4)
    @test isapprox(TSAnalytics._std_normal_cdf(1.6449), 0.95; atol=1e-4)
    @test isapprox(TSAnalytics._std_normal_cdf(-1.6449), 0.05; atol=1e-4)
end
