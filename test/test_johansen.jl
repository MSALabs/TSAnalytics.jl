using DelimitedFiles

# Stage 10.5: the Johansen cointegration rank tests.
#
# Statistics DUAL-VERIFIED against urca 1.3.3 (ca.jo) and statsmodels 0.15.0
# (coint_johansen), which agree with each other once the two mappings are
# right: urca's K = lags + 1, and urca orders r = K-1 first.
# Critical values transcribed from Osterwald-Lenum (1992), Tables 0, 1, 1*, 2,
# 2*; the 1* and 2* transcriptions equal urca's ecdet="const"/"trend" values
# exactly. For :const this package uses Table 1 where urca's ecdet="none" uses
# Table 1.1* -- see johansen_test's docstring and charter §5.13 item 18.

if !isdefined(@__MODULE__, :REF_R)
    include(joinpath(@__DIR__, "fixtures", "var", "refs.jl"))
end

_jfix(name) = readdlm(joinpath(@__DIR__, "fixtures", "var", name), ','; skipstart=1)
_jparse(x) = x isa AbstractString ? eval(Meta.parse(x)) : x
_jvec(x) = Float64.(_jparse(x))
_jmat(x) = (v = _jparse(x); Float64[v[i][j] for i in eachindex(v), j in eachindex(v[1])])
const _JFX = Dict("yes" => "coint2_yes.csv", "no" => "coint2_no.csv", "r1" => "coint3_r1.csv",
                  "r2" => "coint3_r2.csv", "US" => "uschange.csv", "E5" => "econ5.csv")
const _JDET_R = Dict("none" => :const, "const" => :const_restricted, "trend" => :trend_restricted)

@testset "johansen_test: statistics vs statsmodels (det_order -1 and 0)" begin
    for (key, v) in REF_PY_JOHANSEN
        (v isa Dict && haskey(v, "fx")) || continue
        d = parse(Int, string(v["det_order"]))
        d == 1 && continue                       # sm detrends; not this package's :trend model
        det = d == -1 ? :none : :const
        k = parse(Int, string(v["k_ar_diff"]))
        t = johansen_test(_jfix(v["fx"]); lags=k, deterministic=det)
        @test t.trace ≈ _jvec(v["lr1_trace"]) atol=1e-9
        @test t.maxeig ≈ _jvec(v["lr2_maxeig"]) atol=1e-9
        @test t.eigenvalues ≈ _jvec(v["eig"]) atol=1e-12
        @test t.K == parse(Int, string(v["K"]))
        @test t.lags == k
        @test t.ranks == collect(0:(t.K - 1))
    end
end

@testset "johansen_test: statistics, V and W vs urca::ca.jo (reversed order)" begin
    for (key, v) in REF_R_JOHANSEN
        (v isa Dict && haskey(v, "trace")) || continue
        parts = split(key, "_")
        Y = _jfix(_JFX[parts[1]])
        lags = parse(Int, parts[2][2:end]) - 1           # urca's K = lags + 1
        det = _JDET_R[parts[3]]
        t = johansen_test(Y; lags=lags, deterministic=det)
        # urca orders r = K-1 first; r = 0 first here
        @test t.trace ≈ reverse(_jvec(v["trace"]["teststat"])) atol=1e-9
        @test t.maxeig ≈ reverse(_jvec(v["eigen"]["teststat"])) atol=1e-9
        @test t.trace[1] == maximum(t.trace)
        V = _jmat(v["V"]); W = _jmat(v["W"])
        nb = normalize_beta(t, t.K)
        @test size(nb, 1) == size(V, 1)                  # K + 1 rows when restricted
        @test nb ≈ V[:, 1:t.K] atol=1e-9
        # alpha rescaled with the same normalization reproduces urca's W
        aw = t.alpha .* transpose(t.beta[1, :])
        @test aw ≈ W[:, 1:t.K] atol=1e-9
        # critical values: the restricted cases ARE urca's tables
        cvR = reverse(_jmat(v["trace"]["cval"]); dims=1)
        cvE = reverse(_jmat(v["eigen"]["cval"]); dims=1)
        if det === :const
            # urca's ecdet="none" uses Osterwald-Lenum Table 1.1*: 5% = 8.18, 17.95, 31.52
            @test cvR[end, 2] == 8.18
            @test t.trace_crit[end, 2] == 3.76
            @test !(t.trace_crit ≈ cvR)
        else
            @test t.trace_crit == cvR
            @test t.maxeig_crit == cvE
        end
    end
    # spec = "transitory" reparameterizes the short run only: identical statistics
    a, b = REF_R_JOHANSEN["yes_K2_none_LR"], REF_R_JOHANSEN["yes_K2_none_TR"]
    @test _jvec(a["trace"]["teststat"]) ≈ _jvec(b["trace"]["teststat"]) atol=1e-10
end

@testset "the handoff's statistics, explicitly" begin
    cases = [("coint2_yes.csv", 1, [62.018519, 3.591564], [58.426955, 3.591564], [0.20989747, 0.01437775]),
             ("coint2_yes.csv", 2, [49.639450, 3.607281], [46.032168, 3.607281], [0.17002944, 0.01449825]),
             ("coint2_no.csv",  1, [5.146879, 2.104971],  [3.041908, 2.104971],  [0.01219084, 0.00845187]),
             ("coint3_r1.csv",  1, [90.840085, 4.481459, 0.459450], [86.358626, 4.022008, 0.459450],
                                   [0.29405664, 0.01608698, 0.00185091]),
             ("coint3_r2.csv",  1, [195.893040, 93.160336, 0.513648], [102.732704, 92.646688, 0.513648],
                                   [0.33916083, 0.31173088, 0.00206902])]
    for (fx, k, tr, me, lam) in cases
        t = johansen_test(_jfix(fx); lags=k, deterministic=:const)
        @test t.trace ≈ tr atol=1e-6
        @test t.maxeig ≈ me atol=1e-6
        @test t.eigenvalues ≈ lam atol=1e-8
        T = t.nobs
        @test T == 250 - k - 1
        for (i, r) in enumerate(t.ranks)
            @test t.trace[i] ≈ -T * sum(log1p.(-t.eigenvalues[(r+1):end])) rtol=1e-13
            @test t.maxeig[i] ≈ -T * log1p(-t.eigenvalues[r+1]) rtol=1e-13
        end
        @test t.trace[end] == t.maxeig[end]
    end
end

@testset "eigenvalues are real, sorted and in [0, 1)" begin
    for fx in ("coint2_yes.csv", "coint3_r1.csv", "coint3_r2.csv", "uschange.csv", "econ5.csv", "coint2_no.csv")
        for det in (:none, :const, :const_restricted, :trend, :trend_restricted), k in (1, 2)
            t = johansen_test(_jfix(fx); lags=k, deterministic=det)
            @test eltype(t.eigenvalues) === Float64
            @test all(0.0 .<= t.eigenvalues .< 1.0)
            @test issorted(t.eigenvalues; rev=true)
            @test length(t.eigenvalues) == t.K == length(t.trace)
            @test all(isfinite, t.trace) && all(isfinite, t.maxeig)
            @test issorted(t.trace; rev=true)
        end
    end
end

@testset "lags counts DIFFERENCED lags (urca's K = lags + 1)" begin
    Y = _jfix("coint2_yes.csv")
    t1 = johansen_test(Y; lags=1)
    t2 = johansen_test(Y; lags=2)
    @test t1.trace ≈ [62.018519, 3.591564] atol=1e-6
    @test t2.trace ≈ [49.639450, 3.607281] atol=1e-6
    @test t1.nobs == 248 && t2.nobs == 247
    @test occursin("K must be at least K=2", REF_R_JOHANSEN["err_K1"])
    @test occursin("no error", REF_PY_JOHANSEN["err_k0"])          # statsmodels accepts k_ar_diff = 0
    @test_throws ArgumentError johansen_test(Y; lags=0)
end

@testset "deterministic: five cases, and the names that mislead" begin
    Y = _jfix("coint2_yes.csv")
    c  = johansen_test(Y; lags=1, deterministic=:const)
    cr = johansen_test(Y; lags=1, deterministic=:const_restricted)
    n  = johansen_test(Y; lags=1, deterministic=:none)
    tr = johansen_test(Y; lags=1, deterministic=:trend)
    trr = johansen_test(Y; lags=1, deterministic=:trend_restricted)
    @test c.trace  ≈ [62.018519, 3.591564] atol=1e-6      # urca ecdet="none" = sm det_order=0
    @test cr.trace ≈ [62.064072, 3.634837] atol=1e-6      # urca ecdet="const"
    @test n.trace  ≈ [18.443286, 0.861099] atol=1e-6      # sm det_order=-1
    @test trr.trace ≈ [62.379909, 3.842107] atol=1e-6     # urca ecdet="trend"
    @test !isapprox(c.trace, cr.trace; atol=1e-3)
    @test !isapprox(c.trace, n.trace; atol=1.0)
    @test !isapprox(tr.trace, trr.trace; atol=1e-3)
    # a restricted term adds a row to beta and a (zero) eigenvalue that is dropped
    @test size(cr.beta) == (3, 2) && size(trr.beta) == (3, 2)
    @test size(c.beta) == (2, 2) && size(tr.beta) == (2, 2)
    @test length(cr.eigenvalues) == cr.K == length(cr.trace)
    # how large the gap gets on real data
    E = _jfix("econ5.csv")
    ec  = johansen_test(E; lags=1, deterministic=:const)
    ecr = johansen_test(E; lags=1, deterministic=:const_restricted)
    @test ec.trace[1]  ≈ 133.709914 atol=1e-4
    @test abs(ecr.trace[1] - ec.trace[1]) > 70
    # the unrestricted trend is a genuinely different model from statsmodels'
    # det_order = 1, which detrends the data instead
    @test !isapprox(tr.trace, _jvec(REF_PY_JOHANSEN["yes_d1"]["lr1_trace"]); atol=1e-3)
end

@testset "critical values: Osterwald-Lenum (1992), by deterministic case" begin
    Y = _jfix("coint3_r1.csv")
    t = johansen_test(Y; lags=1)
    @test t.trace_crit[:, 2] == [29.68, 15.41, 3.76]          # Table 1
    @test t.maxeig_crit[:, 2] == [20.97, 14.07, 3.76]
    @test size(t.trace_crit) == (3, 3) && size(t.maxeig_crit) == (3, 3)
    @test all(t.trace_crit[:, 1] .< t.trace_crit[:, 2] .< t.trace_crit[:, 3])   # 10 < 5 < 1 %
    # the smallest block varies with the deterministic case, as the theory requires
    sb = Dict(d => johansen_test(Y; lags=1, deterministic=d).trace_crit[end, 2] for d in
              (:none, :const, :const_restricted, :trend, :trend_restricted))
    @test sb[:none] == 3.84 && sb[:const] == 3.76 && sb[:trend] == 3.74
    @test sb[:const_restricted] == 9.24 && sb[:trend_restricted] == 12.25
    # Table 1* and 2* transcriptions equal urca's (5 series, from ca.jo)
    US = _jfix("uschange.csv")
    @test johansen_test(US; lags=1, deterministic=:const_restricted).trace_crit[:, 2] ==
          [76.07, 53.12, 34.91, 19.96, 9.24]
    # statsmodels' independently simulated Tables agree with O-L 0 and 1 to within
    # simulation error (largest gap 7.5%, at p-r = 1 of Table 0)
    sj = REF_PY_JOHANSEN["sj_tables"]
    for (d, det) in ((-1, :none), (0, :const)), K in 1:11
        sm = _jvec(sj["n$(K)_p$(d)"]["trace"])
        ol = collect(TSAnalytics._OL_TABLES[(det, :trace)][K])
        @test all(abs.(sm .- ol) ./ ol .< 0.08)
    end
    # and urca's "none" values are Table 1.1*, which is not shipped:
    # 10/5/1% = 6.50 / 8.18 / 11.65 at p - r = 1
    @test _jmat(REF_R_JOHANSEN["yes_K2_none_LR"]["trace"]["cval"])[1, :] == [6.5, 8.18, 11.65]
    cv = critical_values(t)
    @test cv.levels == (10.0, 5.0, 1.0)
    @test cv.trace === t.trace_crit && cv.maxeig === t.maxeig_crit
end

@testset "Osterwald-Lenum transcription: internal consistency" begin
    for det in (:none, :const, :const_restricted, :trend, :trend_restricted)
        tt = TSAnalytics._OL_TABLES[(det, :trace)]
        tm = TSAnalytics._OL_TABLES[(det, :maxeig)]
        @test length(tt) == 11 == length(tm)
        @test tt[1] == tm[1]                       # at p - r = 1 the two statistics coincide
        for i in 1:11
            @test issorted(collect(tt[i])) && issorted(collect(tm[i]))
            @test all(tt[i] .>= tm[i])             # trace >= max-eigenvalue
        end
        for j in 1:3
            @test issorted([tt[i][j] for i in 1:11]) && issorted([tm[i][j] for i in 1:11])
        end
    end
    # restricting a term into the cointegrating space raises every quantile
    for i in 1:11, j in 1:3
        @test TSAnalytics._OL_TABLES[(:const, :trace)][i][j] < TSAnalytics._OL_TABLES[(:const_restricted, :trace)][i][j]
        @test TSAnalytics._OL_TABLES[(:trend, :trace)][i][j] < TSAnalytics._OL_TABLES[(:trend_restricted, :trace)][i][j]
    end
end

@testset "coint_rank: the point of the exercise" begin
    t1 = johansen_test(_jfix("coint3_r1.csv"); lags=1)
    @test coint_rank(t1; statistic=:trace) == 1
    @test coint_rank(t1; statistic=:maxeig) == 1
    t2 = johansen_test(_jfix("coint3_r2.csv"); lags=1)
    @test coint_rank(t2; statistic=:trace) == 2
    @test coint_rank(t2; statistic=:maxeig) == 2
    t0 = johansen_test(_jfix("coint2_no.csv"); lags=1)
    @test coint_rank(t0) == 0 && coint_rank(t0; statistic=:maxeig) == 0
    ty = johansen_test(_jfix("coint2_yes.csv"); lags=1)
    @test coint_rank(ty) == 1
    # statsmodels' select_coint_rank agrees on coint3_r1
    @test coint_rank(t1) == parse(Int, string(REF_PY_JOHANSEN["rank_trace"]["rank"]))
    @test coint_rank(t1; statistic=:maxeig) == parse(Int, string(REF_PY_JOHANSEN["rank_maxeig"]["rank"]))
    @test_throws ArgumentError coint_rank(ty; level=2.5)
    @test_throws ArgumentError coint_rank(ty; statistic=:bogus)
    for t in (t0, t1, t2, ty)
        @test coint_rank(t; level=1.0) <= coint_rank(t; level=5.0) <= coint_rank(t; level=10.0)
    end
    # every hypothesis rejected -> rank K (stationary levels)
    S = randn(MersenneTwister(4), 300, 2)
    @test coint_rank(johansen_test(S; lags=1)) == 2
end

@testset "beta normalization, and the invariance that proves it is only a normalization" begin
    t = johansen_test(_jfix("coint3_r1.csv"); lags=1)
    V = normalize_beta(t, 3; scheme=:first_unit)
    @test V[1, :] ≈ [1.0, 1.0, 1.0] atol=1e-12
    @test V[:, 1] ≈ [1.0, -0.49650458, -0.97773003] atol=1e-7
    @test V[:, 2] ≈ [1.0, -1.42210910, -0.24676673] atol=1e-7
    @test V[:, 3] ≈ [1.0,  0.21454880,  0.21404288] atol=1e-7
    B2 = normalize_beta(t, 2; scheme=:identity)
    @test B2[1:2, 1:2] ≈ I(2) atol=1e-10
    @test B2[3, :] ≈ [-1.36982691, -0.7897145] atol=1e-7
    @test B2 ≈ _jmat(REF_R_JOHANSEN["cajorls_r1fix_r2"]["beta"]) atol=1e-7
    @test normalize_beta(t, 1; scheme=:identity) ≈ normalize_beta(t, 1; scheme=:first_unit) atol=1e-12
    @test vec(normalize_beta(t, 1)) ≈ vec(_jmat(REF_R_JOHANSEN["cajorls_r1fix_r1"]["beta"])) atol=1e-9
    @test_throws ArgumentError normalize_beta(t, 0)
    @test_throws ArgumentError normalize_beta(t, 4)
    @test_throws ArgumentError normalize_beta(t, 1; scheme=:bogus)

    base = johansen_test(_jfix("coint3_r1.csv"); lags=1)
    scaled = johansen_test(_jfix("coint3_r1.csv") .* [1.0 100.0 0.01]; lags=1)
    @test scaled.trace ≈ base.trace rtol=1e-9
    @test scaled.maxeig ≈ base.maxeig rtol=1e-9
    @test scaled.eigenvalues ≈ base.eigenvalues rtol=1e-9
    @test !isapprox(scaled.beta, base.beta; atol=1e-3)
    # the normalized vector rescales exactly as the data did
    @test normalize_beta(scaled, 1)[:, 1] ≈ normalize_beta(base, 1)[:, 1] ./ [1.0, 100.0, 0.01] rtol=1e-7
    # reordering the series permutes the eigenvalues' vectors but not the statistics
    perm = johansen_test(_jfix("coint3_r1.csv")[:, [3, 1, 2]]; lags=1)
    @test perm.trace ≈ base.trace rtol=1e-9
end

@testset "alpha * beta' is the long-run impact matrix, of rank r" begin
    t = johansen_test(_jfix("coint3_r1.csv"); lags=1)
    r = coint_rank(t)
    Pi = t.alpha[:, 1:r] * t.beta[:, 1:r]'
    @test size(Pi) == (3, 3)
    @test rank(Pi; rtol=1e-8) == r
    # and is normalization-free
    B = normalize_beta(t, r)
    A = t.alpha[:, 1:r] .* transpose(t.beta[1, 1:r])
    @test A * B' ≈ Pi atol=1e-12
    @test A[:, 1] ≈ [0.0369289, -0.23427842, 0.96076186] atol=1e-6      # urca's W
    # beta' S11 beta = I (the raw eigen normalization)
    @test all(t.beta[1, :] .> 0)
end

@testset "no p-value, and coverage refusals" begin
    t = johansen_test(_jfix("coint2_yes.csv"); lags=1)
    @test !t.pvalue_available
    @test !hasfield(JohansenTest, :pvalue)
    @test_throws ArgumentError pvalue(t)
    e = try pvalue(t) catch err; err end
    @test occursin("critical_values", sprint(showerror, e))

    Y12 = cumsum(randn(MersenneTwister(11), 250, 12); dims=1)
    @test_throws ArgumentError johansen_test(Y12; lags=1)
    e = try johansen_test(Y12; lags=1) catch err; err end
    @test occursin("11", sprint(showerror, e))
    t11 = johansen_test(Y12[:, 1:11]; lags=1)
    @test t11.K == 11 && size(t11.trace_crit) == (11, 3)
    @test occursin("no error", REF_PY_JOHANSEN["err_13col"])         # statsmodels: NaN cvt
    @test occursin("no error", REF_PY_JOHANSEN["err_d2"])
    @test occursin("dimnames", REF_R_JOHANSEN["err_13col"])          # urca errors at 13
    for fx in ("coint2_yes.csv", "coint2_no.csv", "coint3_r1.csv", "coint3_r2.csv", "uschange.csv", "econ5.csv")
        for det in (:none, :const, :const_restricted, :trend, :trend_restricted)
            t = johansen_test(_jfix(fx); lags=1, deterministic=det)
            @test all(!isnan, t.trace) && all(!isnan, t.maxeig)
            @test all(!isnan, t.trace_crit) && all(!isnan, t.maxeig_crit)
        end
    end
end

@testset "refusals" begin
    Y = _jfix("coint2_yes.csv")
    @test_throws ArgumentError johansen_test(Y[:, 1:1]; lags=1)
    @test_throws ArgumentError johansen_test(Y[:, 1]; lags=1)
    @test_throws ArgumentError johansen_test(Y; lags=0)
    @test_throws ArgumentError johansen_test(Y; lags=-1)
    @test_throws ArgumentError johansen_test(Y; lags=200)
    @test_throws ArgumentError johansen_test(Y; lags=1, deterministic=:bogus)
    @test_throws ArgumentError johansen_test(Y; lags=1, deterministic=:both)
    Yn = copy(Y); Yn[9, 2] = NaN
    @test_throws ArgumentError johansen_test(Yn; lags=1)
    @test_throws UndefKeywordError johansen_test(Y)                 # lags is required
end

@testset "names, container-agnostic, contract, show" begin
    Y = _jfix("coint3_r1.csv")
    nt = (f=Y[:, 1], g=Y[:, 2], h=Y[:, 3])
    t = johansen_test(nt; lags=1)
    @test t.names == [:f, :g, :h]
    @test t.trace == johansen_test(Y; lags=1).trace
    @test t isa HypothesisTest
    @test statistic(t) === t.trace
    @test isconcretetype(JohansenTest)
    @test (@inferred johansen_test(_jfix("coint2_yes.csv"); lags=1)) isa JohansenTest
    s = sprint(show, t)
    @test occursin("Johansen", s) && occursin("trace", lowercase(s))
    @test occursin("Osterwald-Lenum", s)
    @test occursin("f, g, h", s)
    @test occursin("rank (trace, 5%): 1", s)
    @test occursin("Table 1", s)
end

@testset "performance: one factorization for both auxiliary regressions" begin
    E = _jfix("econ5.csv")
    johansen_test(E; lags=1)
    @test (@allocated johansen_test(E; lags=1)) < 400_000
    johansen_test(E; lags=4)
    t1 = minimum(@elapsed(for _ in 1:50; johansen_test(E; lags=1); end) for _ in 1:3)
    t4 = minimum(@elapsed(for _ in 1:50; johansen_test(E; lags=4); end) for _ in 1:3)
    @test t4 < 8 * t1
end
