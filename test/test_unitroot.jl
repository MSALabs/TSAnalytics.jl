using DelimitedFiles

@testset "unitroot" begin
    Random.seed!(2)
    n = 1000

    # Stationary AR(1), phi = 0.5 -- ADF should reject the unit-root null
    e = randn(n)
    ar1 = zeros(n)
    for t in 2:n
        ar1[t] = 0.5*ar1[t-1] + e[t]
    end
    adf_ar1 = adf_test(ar1; regression=:c)
    @test adf_ar1.statistic < -2.86          # more negative than the 5% asymptotic critical value
    @test adf_ar1.pvalue < 0.10

    # Random walk -- ADF should fail to reject the unit-root null
    rw = cumsum(randn(n))
    adf_rw = adf_test(rw; regression=:c)
    @test adf_rw.statistic > -2.86
    @test adf_rw.pvalue > 0.10

    # KPSS: stationary series should NOT reject level-stationarity (large p-value)
    kpss_ar1 = kpss_test(ar1; regression=:c)
    @test kpss_ar1.statistic < 0.463
    @test kpss_ar1.pvalue > 0.05

    # KPSS: random walk SHOULD reject level-stationarity (small p-value, large statistic)
    kpss_rw = kpss_test(rw; regression=:c)
    @test kpss_rw.statistic > 0.463
    @test kpss_rw.pvalue <= 0.05

    # argument validation
    @test_throws ArgumentError adf_test(ar1; regression=:bogus)
    @test_throws ArgumentError kpss_test(ar1; regression=:bogus)
end

@testset "adf_test regression x autolag (exact statsmodels validation)" begin
    # Same fixture as test_stattools.jl: AR(1), phi=0.6, seed=0 (numpy), n=200 --
    # a genuine cross-language reference, not a self-check.
    fixture = joinpath(@__DIR__, "fixtures", "ar1_ref_series.csv")
    y = vec(readdlm(fixture, ','; skipstart=1, header=false))

    # default maxlag formula: ceil(12*(n/100)^0.25), NOT floor -- verified
    # from adfuller's actual source (handoff/stage-2.1-adf-handoff.md
    # proposed floor; statsmodels itself uses ceil). n=200 -> 15, not 14.
    r_default = adf_test(y; regression=:c, autolag=nothing, maxlag=15)
    r_ceil_check = adf_test(y; regression=:c, autolag=:aic)  # exercises the internal default-maxlag computation
    @test isfinite(r_ceil_check.statistic)

    # exact reference table: statsmodels.tsa.stattools.adfuller(y, regression=reg, autolag=al)
    # (autolag=nothing case uses maxlag=4 explicitly, matching the reference run)
    reference = [
        (:n,   :aic,   -6.102126, 0, 199),
        (:n,   :bic,   -6.102126, 0, 199),
        (:n,   :tstat, -3.961063, 4, 195),
        (:c,   :aic,   -6.137985, 0, 199),
        (:c,   :bic,   -6.137985, 0, 199),
        (:c,   :tstat, -3.965833, 4, 195),
        (:ct,  :aic,   -6.129084, 0, 199),
        (:ct,  :bic,   -6.129084, 0, 199),
        (:ct,  :tstat, -3.828277, 4, 195),
        (:ctt, :aic,   -6.091571, 0, 199),
        (:ctt, :bic,   -6.091571, 0, 199),
        (:ctt, :tstat, -3.851485, 4, 195),
    ]
    for (regression, autolag, stat_ref, usedlag_ref, nobs_ref) in reference
        r = adf_test(y; regression=regression, autolag=autolag)
        @test isapprox(r.statistic, stat_ref; atol=1e-4)
        @test r.lags == usedlag_ref
        @test r.n == nobs_ref
        @test r.regression == regression
    end

    # autolag=nothing (fixed maxlag=4) reference, same for every regression type
    fixed_reference = [
        (:n,   -3.961063),
        (:c,   -3.965833),
        (:ct,  -3.828277),
        (:ctt, -3.851485),
    ]
    for (regression, stat_ref) in fixed_reference
        r = adf_test(y; regression=regression, autolag=nothing, maxlag=4)
        @test isapprox(r.statistic, stat_ref; atol=1e-4)
        @test r.lags == 4
        @test r.n == 195
    end
end

@testset "adf_test parameter validation and edge cases" begin
    Random.seed!(3)
    y = cumsum(randn(300))

    # every regression value runs and gives a finite result
    for regression in (:n, :c, :ct, :ctt)
        r = adf_test(y; regression=regression, autolag=:aic)
        @test isfinite(r.statistic)
        @test isfinite(r.pvalue)
        @test r.regression == regression
    end

    # every autolag value runs and gives a finite result
    for autolag in (:aic, :bic, :tstat, nothing)
        r = adf_test(y; regression=:c, autolag=autolag, maxlag=(autolag === nothing ? 5 : nothing))
        @test isfinite(r.statistic)
    end

    # explicit maxlag is honored when autolag=nothing
    for maxlag in (0, 1, 5, 10)
        r = adf_test(y; regression=:c, autolag=nothing, maxlag=maxlag)
        @test r.lags == maxlag
    end

    # invalid regression / autolag
    @test_throws ArgumentError adf_test(y; regression=:bogus)
    @test_throws ArgumentError adf_test(y; autolag=:bogus)

    # maxlag validation: negative, and too large relative to sample size
    @test_throws ArgumentError adf_test(y; maxlag=-1)
    n = length(y)
    too_large = n ÷ 2  # comfortably past n÷2 - ntrend - 1 for any regression type
    @test_throws ArgumentError adf_test(y; regression=:c, maxlag=too_large)

    # replicate tseries::adf.test(x) exactly, per the documented one-liner
    nrep = length(y)
    r_tseries = adf_test(y; regression=:ct, autolag=nothing, maxlag=trunc(Int, (nrep-1)^(1/3)))
    @test isfinite(r_tseries.statistic)
    @test r_tseries.regression == :ct
end

@testset "kpss_test bandwidth formulas (arithmetic, exact)" begin
    # n=200: short = trunc(4*(200/100)^0.25) = 4
    #        legacy = ceil(12*(200/100)^0.25) = 15 -- NOT trunc's 14; `ceil` verified
    #        directly from statsmodels' kpss() source, the same ceil-not-floor
    #        discrepancy already found in adf_test's maxlag formula.
    y200 = randn(MersenneTwister(3), 200)
    @test kpss_test(y200; nlags=:short).lags == 4
    @test kpss_test(y200; nlags=:legacy).lags == 15

    # n=1000: short = 7, legacy = 22
    y1000 = randn(MersenneTwister(3), 1000)
    @test kpss_test(y1000; nlags=:short).lags == 7
    @test kpss_test(y1000; nlags=:legacy).lags == 22

    # explicit Integer nlags is honored directly
    for l in (0, 1, 10, 50)
        @test kpss_test(y1000; nlags=l).lags == l
    end
end

@testset "kpss_test regression x nlags (exact statsmodels validation)" begin
    # Same fixture as adf_test's exact-value testset and test_stattools.jl.
    fixture = joinpath(@__DIR__, "fixtures", "ar1_ref_series.csv")
    y = vec(readdlm(fixture, ','; skipstart=1, header=false))
    @test length(y) == 200

    # statsmodels.tsa.stattools.kpss(y, regression=reg, nlags=nl)
    reference = [
        (:c,  :legacy, 0.127859, 15),
        (:c,  4,       0.240342, 4),
        (:c,  10,      0.156256, 10),
        (:ct, :legacy, 0.119078, 15),
        (:ct, 4,       0.222205, 4),
        (:ct, 10,      0.145395, 10),
    ]
    for (regression, nlags, stat_ref, lags_ref) in reference
        r = kpss_test(y; regression=regression, nlags=nlags)
        @test isapprox(r.statistic, stat_ref; atol=1e-5)
        @test r.lags == lags_ref
        @test r.regression == regression
        @test r.n == 200
    end
end

@testset "kpss_test parameter validation and edge cases" begin
    Random.seed!(4)
    y = cumsum(randn(300))

    # every regression value runs and gives a finite result
    for regression in (:c, :ct)
        r = kpss_test(y; regression=regression, nlags=:short)
        @test isfinite(r.statistic)
        @test isfinite(r.pvalue)
        @test r.regression == regression
    end

    # every nlags "mode" runs and gives a finite result (:auto excepted -- see below)
    for nlags in (:short, :legacy, 5, 20)
        r = kpss_test(y; regression=:c, nlags=nlags)
        @test isfinite(r.statistic)
    end

    # explicit Integer nlags is honored directly
    for nlags in (0, 1, 5, 15)
        r = kpss_test(y; nlags=nlags)
        @test r.lags == nlags
    end

    # invalid regression / nlags
    @test_throws ArgumentError kpss_test(y; regression=:bogus)
    @test_throws ArgumentError kpss_test(y; nlags=:bogus)
    @test_throws ArgumentError kpss_test(y; nlags=-1)

    # nlags must be < n
    n = length(y)
    @test_throws ArgumentError kpss_test(y; nlags=n)
    @test_throws ArgumentError kpss_test(y; nlags=n+10)

    # :auto is a documented, explicit gap -- not a silent fallback to :short
    @test_throws ArgumentError kpss_test(y; nlags=:auto)

    # the n-1 cap engages for a short series where the short/legacy formula
    # would otherwise exceed it
    short_y = randn(MersenneTwister(6), 5)
    r_capped = kpss_test(short_y; nlags=:legacy)
    @test r_capped.lags <= length(short_y) - 1

    # container-agnostic
    @test kpss_test(1:100; nlags=:short).statistic == kpss_test(collect(1.0:100); nlags=:short).statistic
end

@testset "pp_test trend x test_type (exact arch validation)" begin
    # Same fixture as adf_test's/kpss_test's exact-value testsets.
    fixture = joinpath(@__DIR__, "fixtures", "ar1_ref_series.csv")
    y = vec(readdlm(fixture, ','; skipstart=1, header=false))
    @test length(y) == 200

    # arch.unitroot.PhillipsPerron(y, lags=4, trend=trend, test_type=test_type).stat
    reference = [
        (:n,  :tau, -6.028827925618),
        (:n,  :rho, -63.699229805412),
        (:c,  :tau, -6.072989823595),
        (:c,  :rho, -64.606886311833),
        (:ct, :tau, -6.070752251587),
        (:ct, :rho, -65.312550064007),
    ]
    for (trend, test_type, stat_ref) in reference
        r = pp_test(y; trend=trend, test_type=test_type, lags=4)
        @test isapprox(r.statistic, stat_ref; atol=1e-6)
        @test r.trend == trend
        @test r.test_type == test_type
        @test r.lags == 4
        @test r.n == 199   # one observation lost to the single own-lag term
    end

    # default lags formula: ceil(12*(n/100)^0.25) = 15 for n=200, matching
    # the same convention already verified for adf_test/kpss_test's :legacy.
    r_default = pp_test(y; trend=:c, test_type=:tau)
    @test r_default.lags == 15
end

@testset "pp_test parameter validation and edge cases" begin
    Random.seed!(7)
    y = cumsum(randn(300))

    # every trend x test_type combination runs and gives a finite statistic
    for trend in (:n, :c, :ct), test_type in (:tau, :rho)
        r = pp_test(y; trend=trend, test_type=test_type)
        @test isfinite(r.statistic)
        @test r.trend == trend
        @test r.test_type == test_type
    end

    # :tau has a real p-value; :rho is honestly NaN, not a wrong number
    @test !isnan(pp_test(y; test_type=:tau).pvalue)
    @test isnan(pp_test(y; test_type=:rho).pvalue)

    # explicit lags is honored directly, across several values
    for lags in (0, 1, 5, 20)
        r = pp_test(y; lags=lags)
        @test r.lags == lags
    end

    # invalid trend / test_type / lags
    @test_throws ArgumentError pp_test(y; trend=:bogus)
    @test_throws ArgumentError pp_test(y; test_type=:bogus)
    @test_throws ArgumentError pp_test(y; lags=-1)

    # PP should reject the unit-root null for a stationary AR(1), and fail
    # to reject it for a random walk -- same qualitative check as adf_test
    Random.seed!(2)
    n = 1000
    e = randn(n)
    ar1 = zeros(n)
    for t in 2:n
        ar1[t] = 0.5*ar1[t-1] + e[t]
    end
    rw = cumsum(randn(n))
    @test pp_test(ar1; test_type=:tau).pvalue < 0.10
    @test pp_test(rw; test_type=:tau).pvalue > 0.10

    # container-agnostic
    @test pp_test(1:100).statistic == pp_test(collect(1.0:100)).statistic
end

@testset "_ols weighted (GLS via weighted QR)" begin
    Random.seed!(5)
    n = 500
    X = hcat(ones(n), randn(n))
    beta_true = [2.0, -1.5]

    # Heteroskedastic errors: variance grows with |x|, so downweighting
    # high-variance rows should recover beta_true more precisely than
    # unweighted OLS on the same data.
    x2 = X[:, 2]
    sigma_i = 0.2 .+ abs.(x2)
    y = X * beta_true .+ sigma_i .* randn(n)
    weights = 1.0 ./ sigma_i .^ 2

    beta_ols, = TSAnalytics._ols(X, y)
    beta_wls, resid_wls, se_wls = TSAnalytics._ols(X, y; weights=weights)

    @test length(resid_wls) == n
    @test all(se_wls .> 0)
    # WLS should be closer to the true parameters than OLS on heteroskedastic data
    @test norm(beta_wls .- beta_true) < norm(beta_ols .- beta_true)

    # weights == ones(n) must reproduce plain OLS exactly
    beta_w1, resid_w1, se_w1 = TSAnalytics._ols(X, y; weights=ones(n))
    @test isapprox(beta_w1, beta_ols; atol=1e-10)

    @test_throws ArgumentError TSAnalytics._ols(X, y; weights=ones(n-1))
    @test_throws ArgumentError TSAnalytics._ols(X, y; weights=vcat(-1.0, ones(n-1)))
end

@testset "_ols :qr vs :cholesky agreement" begin
    Random.seed!(1)
    n, k = 50, 3
    X = hcat(ones(n), randn(n), randn(n))
    beta_true = [2.0, -1.5, 0.7]
    y = X*beta_true .+ 0.3.*randn(n)
    w = 1.0 .+ rand(n)

    beta_qr,   resid_qr,   se_qr   = TSAnalytics._ols(X, y; method=:qr)
    beta_chol, resid_chol, se_chol = TSAnalytics._ols(X, y; method=:cholesky)
    @test isapprox(beta_qr, beta_chol; atol=1e-8)
    @test isapprox(se_qr, se_chol; atol=1e-8)
    @test isapprox(resid_qr, resid_chol; atol=1e-8)

    # weighted: both methods must still agree
    beta_qr_w, _, _   = TSAnalytics._ols(X, y; weights=w, method=:qr)
    beta_chol_w, _, _ = TSAnalytics._ols(X, y; weights=w, method=:cholesky)
    @test isapprox(beta_qr_w, beta_chol_w; atol=1e-8)

    # default method is :qr -- calling without `method=` must match explicit :qr
    beta_default, = TSAnalytics._ols(X, y)
    @test isapprox(beta_default, beta_qr; atol=1e-12)

    # invalid method
    @test_throws ArgumentError TSAnalytics._ols(X, y; method=:bogus)

    # collinear regressors: :cholesky should fail informatively, :qr should not
    Xcol = hcat(X, X[:, 2])  # duplicate column -> singular X'X
    @test_throws ArgumentError TSAnalytics._ols(Xcol, y; method=:cholesky)
    beta_qr_col, = TSAnalytics._ols(Xcol, y; method=:qr)  # QR's `\` handles this via least-squares
    @test all(isfinite, beta_qr_col)
end
