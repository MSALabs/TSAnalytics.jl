using DelimitedFiles

# Stage 10.4a: the Engle-Granger cointegration test, and the MacKinnon
# N = 1..12 tables it finally needs.
#
# statsmodels 0.15.0 is the reference for the test (all nine of its coint
# configurations) and the table source (BSD; transcribed by script). urca
# 1.3.3's independent MacKinnon (1995) tables are the second reference for
# the tables, at the looser tolerance two vintages of a response surface
# warrant. The hand-built R Engle-Granger (lm + ur.df) is the second reference
# for the statistic at a fixed lag, where it agrees exactly.

if !isdefined(@__MODULE__, :REF_R)
    include(joinpath(@__DIR__, "fixtures", "var", "refs.jl"))
end

_cfix(name) = readdlm(joinpath(@__DIR__, "fixtures", "var", name), ','; skipstart=1)
const _TREND_OF = Dict("n" => :none, "c" => :const, "ct" => :trend, "ctt" => :quadratic)

@testset "MacKinnon tables: N = 1..12 vs statsmodels (exact) and urca (vintage)" begin
    grid = Float64.(REF_PY_COINT["mackinnon_stats_grid"])
    TP, TR = REF_PY_COINT["mackinnon_tables"], REF_R_COINT["mackinnon_tables"]
    for reg in (:n, :c, :ct, :ctt), N in 1:12
        key = "$(reg)_N$N"
        py, r = TP[key], TR[key]
        # critical values: statsmodels has them for N <= 12 except :n beyond 1
        if py["crit_n250"] isa AbstractString
            @test reg === :n && N > 1
            @test occursin("IndexError", py["crit_n250"])
            @test_throws ArgumentError adf_critical_values_response_surface(reg; nobs=250, N=N)
        else
            c = collect(adf_critical_values_response_surface(reg; nobs=250, N=N))
            a = collect(adf_critical_values_response_surface(reg; N=N))
            @test c ≈ Float64.(py["crit_n250"]) atol=1e-12
            @test a ≈ Float64.(py["crit_asym"]) atol=1e-12
            @test all(isapprox.(c, Float64.(r["crit_n250"]); atol=1e-3))   # urca, 1995 vintage
            @test issorted(c)                                       # 1% < 5% < 10%
        end
        # p-values: statsmodels has them for N <= 6, for every regression
        if py["pvals"][1] isa AbstractString
            @test N > 6
            @test_throws ArgumentError adf_pvalue_response_surface(-3.0, reg; N=N)
        else
            pv = [adf_pvalue_response_surface(s, reg; N=N) for s in grid]
            @test pv ≈ Float64.(py["pvals"]) rtol=1e-12 atol=1e-15
            @test all(isapprox.(pv, Float64.(r["pvals"]); atol=2e-2))      # urca, 1995 vintage
            @test issorted(pv)                                      # monotone in the statistic
        end
    end
    @test_throws ArgumentError adf_pvalue_response_surface(-3.0, :c; N=0)
    @test_throws ArgumentError adf_critical_values_response_surface(:c; N=13)
    # N = 1 is untouched by the extension: the pre-existing doctest values
    @test round(adf_pvalue_response_surface(-3.5, :c), digits=6) == 0.007987
    @test round.(collect(adf_critical_values_response_surface(:c; nobs=100)); digits=5) ==
          [-3.4975, -2.89091, -2.58243]
end

@testset "the normal-CDF tail no longer underflows (found here)" begin
    # MacKinnon's p-value is Φ(polynomial); for coint3_r1 the polynomial is
    # about -10, and the old Φ computed 1 - erf after forming erf = 1 - Q,
    # which cancelled to exactly 0 below z ~ -8.3. R's pnorm, for reference:
    f = TSAnalytics._std_normal_cdf
    @test f(-12.0) ≈ 1.776482112077679e-33 rtol=1e-12
    @test f(-10.0) ≈ 7.619853024160527e-24 rtol=1e-12
    @test f(-8.0) ≈ 6.220960574271785e-16 rtol=1e-12
    @test f(-3.0) ≈ 1.349898031630095e-03 rtol=1e-12
    @test f(0.5) ≈ 0.6914624612740130 rtol=1e-14
    @test f(3.0) ≈ 0.9986501019683699 rtol=1e-14
    @test f(0.0) == 0.5
    @test f(-40.0) == 0.0                     # 3.6e-350: below floatmin, an honest 0
    for z in -9.0:0.5:9.0
        @test f(z) + f(-z) ≈ 1.0 atol=1e-15
    end
end

@testset "coint_test: every statsmodels configuration" begin
    for (key, v) in REF_PY_COINT
        (v isa Dict && haskey(v, "fx")) || continue
        tr = _TREND_OF[v["trend"]]
        Y = _cfix(v["fx"])
        al = v["autolag"] === nothing ? nothing : Symbol(v["autolag"])
        if tr === :none
            # statsmodels answers with p = 0.0668 and NaN critical values; we refuse
            @test all(isnan, Float64.(v["crit"]))
            @test_throws ArgumentError coint_test(Y; trend=tr, lags=v["maxlag"], autolag=al)
            continue
        end
        t = coint_test(Y; trend=tr, lags=v["maxlag"], autolag=al)
        @test t.statistic ≈ v["stat"] atol=1e-9
        @test t.pvalue ≈ v["p"] rtol=1e-9                         # same table, exactly
        @test collect(t.critical_values.values) ≈ Float64.(v["crit"]) atol=1e-12
        @test t.N == v["N"] == size(Y, 2)
        @test t.nobs == 250
        @test t.pvalue_available
        @test t.autolag === al
    end
end

@testset "coint_test: fixed lag 4, exact agreement with the R hand-built EG" begin
    for (fx, tr, stat) in [("coint2_yes.csv",   :const, -5.278983513620),
                           ("coint2_no.csv",    :const, -1.258083608751),
                           ("coint3_r1.csv",    :const, -6.753428704193),
                           ("coint2_trend.csv", :trend, -5.857085337938)]
        Y = _cfix(fx)
        t = coint_test(Y; trend=tr, lags=4, autolag=nothing)
        @test t.statistic ≈ stat atol=1e-9
        @test t.lags == 4
        @test t.nobs == 250
        @test t.N == size(Y, 2)
    end
    t = coint_test(_cfix("coint2_yes.csv"); trend=:const, lags=4, autolag=nothing)
    @test t.pvalue ≈ 4.893991e-05 rtol=1e-6
    t = coint_test(_cfix("coint2_no.csv"); trend=:const, lags=4, autolag=nothing)
    @test t.pvalue ≈ 8.420281e-01 rtol=1e-6
    # the cointegrating residual matches R's lm residual
    for (key, fx, tr) in [("eg_yes_c", "coint2_yes.csv", :const), ("eg_no_c", "coint2_no.csv", :const),
                          ("eg_r1_c", "coint3_r1.csv", :const), ("eg_trend_ct", "coint2_trend.csv", :trend)]
        t = coint_test(_cfix(fx); trend=tr, lags=4, autolag=nothing)
        @test t.resid[1:3] ≈ Float64.(REF_R_COINT[key]["resid_head"]) atol=1e-10
    end
end

@testset "autolag: references disagree only about the search" begin
    # R's ur.df(selectlag="AIC", lags=12) and statsmodels' autolag="aic" pick
    # different lags on three of four fixtures. coint_test delegates to
    # adf_test's selection, which is statsmodels'.
    for (key, fx, tr, pykey) in [("eg_yes_c", "coint2_yes.csv", :const, "eg_yes_c"),
                                 ("eg_no_c", "coint2_no.csv", :const, "eg_no_c"),
                                 ("eg_r1_c", "coint3_r1.csv", :const, "eg_r1_c"),
                                 ("eg_trend_ct", "coint2_trend.csv", :trend, "eg_trend_ct")]
        t = coint_test(_cfix(fx); trend=tr)
        @test t.statistic ≈ REF_PY_COINT[pykey]["stat"] atol=1e-9
        if key == "eg_yes_c"
            @test t.statistic ≈ REF_R_COINT[key]["adf_stat"] atol=1e-9    # agree here
        else
            @test !isapprox(t.statistic, REF_R_COINT[key]["adf_stat"]; atol=1e-3)
        end
    end
end

@testset "critical values: statsmodels is the source, urca the check" begin
    for (fx, tr, cv) in [("coint2_yes.csv",   :const, (-3.940964, -3.360779, -3.061527)),
                         ("coint3_r1.csv",    :const, (-4.352246, -3.775224, -3.477197)),
                         ("coint2_trend.csv", :trend, (-4.390198, -3.818960, -3.524870))]
        t = coint_test(_cfix(fx); trend=tr, lags=4, autolag=nothing)
        @test t.critical_values.levels == (1.0, 5.0, 10.0)
        @test critical_values(t) === t.critical_values
        @test all(isapprox.(t.critical_values.values, cv; atol=1e-6))
    end
    t = coint_test(_cfix("coint2_yes.csv"); trend=:const, lags=4, autolag=nothing)
    @test all(isapprox.(t.critical_values.values, (-3.94079, -3.36075, -3.06169); atol=1e-3))   # urca
    # at nobs = n - 1, which is what statsmodels' coint evaluates (the ADF
    # regression's sample), not n
    @test collect(t.critical_values.values) ≈
          collect(adf_critical_values_response_surface(:c; nobs=249, N=2)) atol=1e-14
    @test !(collect(t.critical_values.values) ≈ collect(adf_critical_values_response_surface(:c; nobs=250, N=2)))
    for (tr, N, cv) in [(:c, 2,  (-3.94079, -3.36075, -3.06169)), (:c, 3,  (-4.35143, -3.77464, -3.47662)),
                        (:c, 6,  (-5.34928, -4.77490, -4.47784)), (:c, 7,  (-5.63167, -5.05720, -4.75968)),
                        (:c, 12, (-6.84022, -6.25833, -5.95702)), (:ct, 2,  (-4.38955, -3.81875, -3.52496)),
                        (:ct, 12, (-7.05554, -6.47148, -6.16932)), (:ctt, 2,  (-4.77512, -4.20769, -3.91584)),
                        (:ctt, 12, (-7.26683, -6.68053, -6.37730))]
        got = adf_critical_values_response_surface(tr; nobs=250, N=N)
        @test all(isapprox.((got.p1, got.p5, got.p10), cv; atol=1e-3))
    end
end

@testset "coverage: refuse rather than return NaN" begin
    Y3 = _cfix("coint3_r1.csv")
    Y7 = hcat(Y3, cumsum(randn(MersenneTwister(1), 250, 4); dims=1))
    t = coint_test(Y7; trend=:const, lags=4, autolag=nothing)
    @test t.N == 7
    @test !t.pvalue_available
    @test t.pvalue === nothing
    @test_throws ArgumentError pvalue(t)
    e = try pvalue(t) catch err; err end
    @test occursin("critical_values", sprint(showerror, e))
    @test all(isfinite, t.critical_values.values)
    @test occursin("use critical_values", sprint(show, t))
    for N in 7:12
        Yn = hcat(Y3, cumsum(randn(MersenneTwister(N), 250, N - 3); dims=1))
        tn = coint_test(Yn; trend=:trend, lags=2, autolag=nothing)
        @test tn.N == N && !tn.pvalue_available && all(isfinite, tn.critical_values.values)
    end
    Y13 = hcat(Y3, cumsum(randn(MersenneTwister(2), 250, 10); dims=1))
    @test_throws ArgumentError coint_test(Y13; trend=:const, lags=4, autolag=nothing)
    e = try coint_test(Y13; lags=4) catch err; err end
    @test occursin("12", sprint(showerror, e))

    Y2 = _cfix("coint2_yes.csv")
    @test_throws ArgumentError coint_test(Y2; trend=:none, lags=4, autolag=nothing)
    e = try coint_test(Y2; trend=:none, lags=4, autolag=nothing) catch err; err end
    @test occursin("single series", lowercase(sprint(showerror, e)))
    @test occursin("NaN", sprint(showerror, e))

    for fx in ("coint2_yes.csv", "coint2_no.csv", "coint3_r1.csv", "coint3_r2.csv", "coint2_trend.csv")
        for tr in (:const, :trend, :quadratic)
            t = coint_test(_cfix(fx); trend=tr, lags=4, autolag=nothing)
            @test !isnan(t.statistic)
            @test all(!isnan, t.critical_values.values)
            @test t.pvalue_available && !isnan(t.pvalue) && 0 <= t.pvalue <= 1
        end
    end
end

@testset "size and power" begin
    for (fx, tr) in [("coint2_yes.csv", :const), ("coint3_r1.csv", :const),
                     ("coint3_r2.csv", :const), ("coint2_trend.csv", :trend)]
        t = coint_test(_cfix(fx); trend=tr, lags=4, autolag=nothing)
        @test t.statistic < t.critical_values.values[2]
        @test pvalue(t) < 0.05
    end
    t = coint_test(_cfix("coint2_no.csv"); trend=:const, lags=4, autolag=nothing)
    @test t.statistic > t.critical_values.values[3]
    @test t.pvalue > 0.5
    # the tiny p-values are real now, not zero
    @test 0 < coint_test(_cfix("coint3_r1.csv")).pvalue < 1e-20
end

@testset "the test is not symmetric in column order" begin
    Y = _cfix("coint2_yes.csv")
    a = coint_test(Y;            trend=:const, lags=4, autolag=nothing)
    b = coint_test(Y[:, [2, 1]]; trend=:const, lags=4, autolag=nothing)
    @test !isapprox(a.statistic, b.statistic; atol=1e-3)
    @test length(a.beta) == 2
    @test length(coint_test(Y; trend=:trend, lags=4, autolag=nothing).beta) == 3
    @test length(coint_test(_cfix("coint3_r1.csv"); trend=:quadratic, lags=4, autolag=nothing).beta) == 5
end

@testset "cointegrating regression by-products" begin
    Y = _cfix("coint2_yes.csv")            # columns (x, z), z = 2 + 1.5x + stationary
    t = coint_test(Y[:, [2, 1]]; trend=:const, lags=4, autolag=nothing)
    @test t.beta[1] ≈ 2.0 atol=0.6
    @test t.beta[2] ≈ 1.5 atol=0.1
    @test length(t.resid) == 250
    @test abs(sum(t.resid) / 250) < 1e-8
    X = hcat(ones(250), Y[:, 1])
    @test t.beta ≈ X \ Y[:, 2] rtol=1e-10
    @test t.resid ≈ Y[:, 2] - X * t.beta atol=1e-10
    # and the statistic IS adf_test on that residual, no constant
    @test t.statistic ≈ adf_test(t.resid; regression=:n, maxlag=4, autolag=nothing).statistic atol=1e-12
end

@testset "lags and autolag" begin
    Y = _cfix("coint2_yes.csv")
    @test coint_test(Y; trend=:const, lags=4, autolag=nothing).lags == 4
    # the default is adf_test's own Schwert rule (ceil, as adfuller's source
    # has it) -- not a second rule. The handoff's floor() would give 15.
    d = coint_test(Y; trend=:const, autolag=nothing)
    @test d.lags == ceil(Int, 12 * (250 / 100)^0.25) == 16
    for al in (:aic, :bic, :tstat)
        t = coint_test(Y; trend=:const, autolag=al)
        @test 0 <= t.lags <= d.lags
        @test t.autolag === al
        @test t.statistic ≈ adf_test(t.resid; regression=:n, autolag=al).statistic atol=1e-12
    end
    @test_throws ArgumentError coint_test(Y; trend=:const, autolag=:bogus)
    @test_throws ArgumentError coint_test(Y; trend=:const, lags=-1)
    @test_throws ArgumentError coint_test(Y; trend=:const, lags=200)
end

@testset "refusals" begin
    Y = _cfix("coint2_yes.csv")
    @test occursin("no error", REF_PY_COINT["err_one_series"])     # statsmodels runs a plain ADF
    @test_throws ArgumentError coint_test(Y[:, 1:1]; trend=:const)
    @test_throws ArgumentError coint_test(Y[:, 1]; trend=:const)
    e = try coint_test(Y[:, 1:1]) catch err; err end
    @test occursin("adf_test", sprint(showerror, e))
    e = try coint_test(Y[:, 1]) catch err; err end
    @test occursin("adf_test", sprint(showerror, e))
    @test_throws ArgumentError coint_test(Y; trend=:bogus)
    @test_throws ArgumentError coint_test(Y; trend=:both)           # fit_var's word, not this test's
    Yn = copy(Y); Yn[5, 1] = NaN
    @test_throws ArgumentError coint_test(Yn; trend=:const)
    Yi = copy(Y); Yi[5, 2] = Inf
    @test_throws ArgumentError coint_test(Yi; trend=:const)
    @test_throws ArgumentError coint_test(Y[1:6, :]; trend=:const, lags=4)
end

@testset "container-agnostic" begin
    Y = _cfix("coint2_yes.csv")
    a = coint_test(Y; lags=4, autolag=nothing)
    b = coint_test((x=Y[:, 1], z=Y[:, 2]); lags=4, autolag=nothing)
    c = coint_test([Y[:, 1], Y[:, 2]]; lags=4, autolag=nothing)
    @test a.statistic == b.statistic == c.statistic
end

@testset "HypothesisTest contract and show" begin
    t = coint_test(_cfix("coint2_yes.csv"); trend=:const, lags=4, autolag=nothing)
    @test t isa HypothesisTest
    @test statistic(t) === t.statistic
    @test 0.0 <= pvalue(t) <= 1.0
    @test critical_values(t).levels == (1.0, 5.0, 10.0)
    s = sprint(show, t)
    @test occursin("Engle-Granger", s)
    @test occursin("4 (fixed)", s)
    @test occursin("first column is the dependent variable", s)
    @test occursin("5%: -3.3608", s)
    @test isconcretetype(CointTest)
    @test (@inferred coint_test(_cfix("coint2_yes.csv"); lags=4, autolag=nothing)) isa CointTest
    # critical_values on a test without them is nothing, not an error
    @test critical_values(ljungbox_test(randn(MersenneTwister(1), 100), 5)) === nothing
end
