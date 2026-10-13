using DelimitedFiles

# Stage 10.8: multivariate residual diagnostics for a fitted VAR.
#
# Statistics and degrees of freedom are asserted against vars 1.6.1 for all
# five tests on seven fixtures; the portmanteau and multivariate JB also
# against statsmodels 0.15.0, which agrees with vars on both. p-values are
# NEVER asserted against vars: it computes every one as 1 - CDF, which
# underflows to 0 in ordinary use. They are asserted against statsmodels
# where it implements the test, and otherwise against this package's own
# upper-tail functions evaluated at the reference statistic.

if !isdefined(@__MODULE__, :REF_R)
    include(joinpath(@__DIR__, "fixtures", "var", "refs.jl"))
end

_dfix(name) = readdlm(joinpath(@__DIR__, "fixtures", "var", name), ','; skipstart=1)

const DIAG_CASES = [("A_p2", "var2_p1.csv", 2), ("A_p1", "var2_p1.csv", 1),
                    ("B_p2", "var3_p2.csv", 2), ("US_p2", "uschange.csv", 2),
                    ("E5_p2", "econ5.csv", 2), ("SH_p2", "var2_short.csv", 2),
                    ("C_p2", "climhyd.csv", 2)]
_ri(x) = x isa Integer ? Int(x) : parse(Int, string(x))
# a model whose names are the reference's, so per-equation results key alike
function _named_fit(key, fx, p)
    nm = Symbol.(String.(REF_R_DIAG[key]["names"]))
    Y = _dfix(fx)
    fit_var(NamedTuple{Tuple(nm)}(Tuple(eachcol(Y))), p)
end

@testset "portmanteau_test: Hosking, vs vars and statsmodels" begin
    for (key, fx, p) in DIAG_CASES
        R, P = REF_R_DIAG[key], REF_PY_DIAG[key]
        m = fit_var(_dfix(fx), p)
        lags = _ri(R["lags.pt"])
        @test lags == _ri(P["lags"])
        a = portmanteau_test(m; lags=lags, adjusted=false)
        b = portmanteau_test(m; lags=lags, adjusted=true)
        @test a.statistic ≈ R["pt_asymptotic"]["statistic"] rtol=1e-11
        @test b.statistic ≈ R["pt_adjusted"]["statistic"] rtol=1e-11
        @test a.statistic ≈ P["whiteness_adj0"]["stat"] rtol=1e-11
        @test b.statistic ≈ P["whiteness_adj1"]["stat"] rtol=1e-11
        K = length(m.names)
        @test a.df == b.df == R["pt_asymptotic"]["parameter"] == P["whiteness_adj0"]["df"]
        @test a.df == K^2 * (lags - p)
        @test b.statistic > a.statistic                  # every weight T/(T-i) > 1
        @test !a.adjusted && b.adjusted
        @test a.nobs == m.T && a.K == K && a.lags == lags
        # p-values against statsmodels, which takes a real upper tail
        @test a.pvalue ≈ P["whiteness_adj0"]["p"] rtol=1e-9
        @test b.pvalue ≈ P["whiteness_adj1"]["p"] rtol=1e-9
        @test 0.0 <= a.pvalue <= 1.0
    end
end

@testset "portmanteau_test: definition, monotonicity, lag structure" begin
    m = fit_var(_dfix("var3_p2.csv"), 2)
    E = residuals(m); T = m.T
    C0 = (E' * E) ./ T
    Ci(i) = (E[(i+1):T, :]' * E[1:(T-i), :]) ./ T
    direct = T * sum(tr(Ci(i)' * inv(C0) * Ci(i) * inv(C0)) for i in 1:6)
    @test portmanteau_test(m; lags=6).statistic ≈ direct rtol=1e-12
    direct_adj = T^2 * sum(tr(Ci(i)' * inv(C0) * Ci(i) * inv(C0)) / (T - i) for i in 1:6)
    @test portmanteau_test(m; lags=6, adjusted=true).statistic ≈ direct_adj rtol=1e-12
    # adding a lag adds a non-negative term
    for l in 3:9
        @test portmanteau_test(m; lags=l).statistic <= portmanteau_test(m; lags=l + 1).statistic
    end
    # invariant to the units of each series: C0 normalizes them away
    s = [1.0, 100.0, 0.01]
    ms = fit_var(_dfix("var3_p2.csv") .* s', 2)
    @test portmanteau_test(ms; lags=8).statistic ≈ portmanteau_test(m; lags=8).statistic rtol=1e-9
end

@testset "normality_test: multivariate JB, vs vars and statsmodels" begin
    for (key, fx, p) in DIAG_CASES
        R, P = REF_R_DIAG[key], REF_PY_DIAG[key]
        m = fit_var(_dfix(fx), p)
        t = normality_test(m)
        K = length(m.names)
        @test t.statistic ≈ R["jb_mul"]["statistic"] rtol=1e-11
        @test t.statistic ≈ P["normality"]["stat"] rtol=1e-11
        @test t.df == 2K == R["jb_mul"]["parameter"] == P["normality"]["df"]
        @test t.skewness_statistic ≈ R["jb_skew"]["statistic"] rtol=1e-11
        @test t.kurtosis_statistic ≈ R["jb_kurt"]["statistic"] rtol=1e-11
        @test t.skewness_df == K == R["jb_skew"]["parameter"]
        @test t.kurtosis_df == K == R["jb_kurt"]["parameter"]
        # the ONLY independent check on the two halves, which only vars reports
        @test t.skewness_statistic + t.kurtosis_statistic ≈ t.statistic rtol=1e-14
        @test t.orthogonalization === :cholesky
        @test t.univariate === nothing
        if P["normality"]["p"] > 0
            @test t.pvalue ≈ P["normality"]["p"] rtol=1e-9
        end
    end
end

@testset "normality_test: per-equation JB" begin
    for (key, fx, p) in DIAG_CASES
        R = REF_R_DIAG[key]
        m = _named_fit(key, fx, p)
        u = normality_test(m; univariate=true).univariate
        @test u isa Dict{Symbol,JarqueBeraTest}
        @test Set(keys(u)) == Set(m.names)
        for nm in m.names
            ref = R["jb_uni"][String(nm)]
            @test u[nm].statistic ≈ ref["statistic"] rtol=1e-11
            @test ref["parameter"] == 2
            @test u[nm].pvalue ≈ TSAnalytics._chisq_ccdf(ref["statistic"], 2) rtol=1e-9
            @test u[nm].statistic ≈ jarque_bera_test(residuals(m)[:, findfirst(==(nm), m.names)]).statistic
        end
    end
end

@testset "standardized residuals have identity covariance" begin
    for (key, fx, p) in DIAG_CASES
        m = fit_var(_dfix(fx), p)
        W = TSAnalytics.standardized_residuals(m)
        @test size(W) == size(residuals(m))
        @test (W' * W) ./ m.T ≈ I(length(m.names)) atol=1e-10
        @test maximum(abs.(sum(W; dims=1))) < 1e-8 * m.T
    end
end

@testset "arch_lm_test(::VARModel): multivariate, vs vars" begin
    for (key, fx, p) in DIAG_CASES
        R = REF_R_DIAG[key]
        m = fit_var(_dfix(fx), p)
        q = _ri(R["lags.multi"])
        t = arch_lm_test(m; lags=q)
        K = length(m.names)
        @test t isa MultivariateARCHLMTest
        @test t.statistic ≈ R["arch_mul"]["statistic"] rtol=1e-11
        @test t.df == R["arch_mul"]["parameter"]
        @test t.df == q * (K * (K + 1) ÷ 2)^2              # grows as K⁴
        @test t.pvalue ≈ TSAnalytics._chisq_ccdf(R["arch_mul"]["statistic"], t.df) rtol=1e-9
        @test t.nobs == m.T - q
        @test arch_lm_test(m, q).statistic == t.statistic    # positional lags, like the univariate method
    end
end

@testset "arch_lm_test(::VARModel): per-equation, vs vars" begin
    for (key, fx, p) in DIAG_CASES
        R = REF_R_DIAG[key]
        m = _named_fit(key, fx, p)
        q = _ri(R["lags.single"])
        u = arch_lm_test(m; lags=q, univariate=true).univariate
        @test u isa Dict{Symbol,ARCHLMTest}
        for nm in m.names
            ref = R["arch_uni"][String(nm)]
            @test u[nm].statistic ≈ ref["statistic"] rtol=1e-11
            @test u[nm].lags == q == ref["parameter"]
        end
    end
    # the univariate method is untouched by the new one
    @test arch_lm_test(randn(MersenneTwister(4), 300), 3) isa ARCHLMTest
end

@testset "bg_test and es_test, vs vars" begin
    for (key, fx, p) in DIAG_CASES
        R = REF_R_DIAG[key]
        m = fit_var(_dfix(fx), p)
        K = length(m.names)
        h = _ri(R["lags.bg"])
        b = bg_test(m; lags=h)
        @test b.statistic ≈ R["bg"]["statistic"] rtol=1e-11
        @test b.df == R["bg"]["parameter"] == h * K^2
        @test b.pvalue ≈ TSAnalytics._chisq_ccdf(R["bg"]["statistic"], b.df) rtol=1e-9
        f = es_test(m; lags=h)
        @test f.statistic ≈ R["es"]["statistic"] rtol=1e-11
        # the denominator df is the one quantity with a single reference
        @test [f.df1, f.df2] == Int.(R["es"]["parameter"])
        @test f.df1 == h * K^2
        @test f.pvalue ≈ TSAnalytics._f_ccdf(R["es"]["statistic"], f.df1, f.df2) rtol=1e-9
        @test b.lags == f.lags == h && b.nobs == f.nobs == m.T
    end
end

@testset "bg_test: the auxiliary regression, by hand" begin
    m = fit_var(_dfix("var2_p1.csv"), 2)
    U = residuals(m); T, K, h = m.T, 2, 3
    Z0, _, _ = TSAnalytics.build_var_Z(m.Y, 2, :const)
    L = zeros(T, K * h)
    for l in 1:h; L[(l+1):T, ((l-1)*K+1):(l*K)] .= U[1:(T-l), :]; end
    Z = hcat(Z0, L)
    Ee = U - Z * (Z \ U)
    @test bg_test(m; lags=h).statistic ≈ T * (K - tr((U' * U) \ (Ee' * Ee))) rtol=1e-10
    # LM >= 0: adding regressors cannot raise the residual covariance
    for hh in 1:6
        @test bg_test(m; lags=hh).statistic >= 0
    end
end

@testset "invariants and power" begin
    m = fit_var(_dfix("var3_p2.csv"), 2)
    E = residuals(m); T = m.T
    C0 = (E' * E) ./ T
    @test issymmetric(round.(C0, digits=12))
    @test portmanteau_test(m; lags=3).statistic < portmanteau_test(m; lags=4).statistic

    rng = MersenneTwister(99)
    W = randn(rng, 400, 3)
    mw = fit_var(W, 1)
    @test portmanteau_test(mw; lags=8).pvalue > 0.01
    @test normality_test(mw).pvalue > 0.01
    @test bg_test(mw; lags=4).pvalue > 0.01
    @test arch_lm_test(mw; lags=2).pvalue > 0.01
    # under-fitting a true VAR(2) leaves structure the tests should find
    mu = fit_var(_dfix("var3_p2.csv"), 1)
    @test portmanteau_test(mu; lags=8).pvalue < portmanteau_test(m; lags=8).pvalue
    @test bg_test(mu; lags=4).pvalue < bg_test(m; lags=4).pvalue
    # heavy tails are found
    Wt = W ./ sqrt.(rand(MersenneTwister(5), 400) .+ 0.05)   # scale-mixture: fat tails
    @test normality_test(fit_var(Wt, 1)).pvalue < 1e-6
    # conditional heteroskedasticity is found
    n = 600; Ya = zeros(n, 2); s2 = ones(n)
    ra = MersenneTwister(6)
    for t in 2:n
        s2[t] = 0.2 + 0.7 * Ya[t-1, 1]^2
        Ya[t, :] = sqrt(s2[t]) .* randn(ra, 2)
    end
    @test arch_lm_test(fit_var(Ya, 1); lags=2).pvalue < 1e-6
end

@testset "p-values survive where vars reports exactly 0" begin
    mU = fit_var(_dfix("uschange.csv"), 2)
    t = normality_test(mU)
    @test REF_R_DIAG["US_p2"]["jb_mul"]["p.value"] == 0          # vars
    @test t.pvalue ≈ 9.282207020757581e-55 rtol=1e-6             # statsmodels' tail
    @test t.pvalue > 0.0

    mE = fit_var(_dfix("econ5.csv"), 2)
    tE = normality_test(mE)
    @test REF_R_DIAG["E5_p2"]["jb_mul"]["p.value"] == 2.220446049250313e-16   # the 1 - x floor
    @test tE.pvalue ≈ 1.8815926771844177e-16 rtol=1e-6
    @test tE.pvalue != 2.220446049250313e-16

    mC = fit_var(_dfix("climhyd.csv"), 2)
    @test REF_R_DIAG["C_p2"]["bg"]["p.value"] == 0
    @test REF_R_DIAG["C_p2"]["es"]["p.value"] == 0
    @test 1e-24 < bg_test(mC; lags=3).pvalue < 1e-22             # 4.3e-23
    @test 1e-33 < es_test(mC; lags=3).pvalue < 1e-31             # 4.9e-32
    # climhyd's JB = 1850.9 on 12 df has a tail near 1e-391, below
    # floatmin(Float64) ~ 2.2e-308: no Float64 holds it, so 0.0 is the honest
    # answer here -- statsmodels returns 0.0 too. (The handoff's test asked
    # for > 0 here, which no double can satisfy.)
    tC = normality_test(mC)
    @test tC.statistic ≈ 1850.9228705709647 rtol=1e-11
    @test tC.pvalue == 0.0
    @test REF_PY_DIAG["C_p2"]["normality"]["p"] == 0.0
    # and the skewness half alone (270.06 on 6 df) IS representable
    @test 0.0 < tC.skewness_pvalue < 1e-50
end

@testset "refusals" begin
    m = fit_var(_dfix("var2_p1.csv"), 2)
    @test occursin("no error", REF_R_DIAG["err_pt_le_p"])           # vars runs with df <= 0
    @test occursin("larger than the number of lags", REF_PY_DIAG["err_lags_le_p"])
    @test occursin("df=1192", REF_PY_DIAG["err_lags_huge"])         # statsmodels: lags 300 on T 198
    @test_throws ArgumentError portmanteau_test(m; lags=2)
    @test_throws ArgumentError portmanteau_test(m; lags=1)
    @test_throws ArgumentError portmanteau_test(m; lags=0)
    e = try portmanteau_test(m; lags=2) catch err; err end
    @test occursin("lags", lowercase(sprint(showerror, e)))
    @test occursin("p = 2", sprint(showerror, e))
    @test_throws ArgumentError portmanteau_test(m; lags=300)
    @test_throws ArgumentError portmanteau_test(m; lags=m.T)
    @test portmanteau_test(m; lags=m.T - 1) isa PortmanteauTest

    @test occursin("subscript out of bounds", REF_R_DIAG["err_arch_zero"])
    @test_throws ArgumentError arch_lm_test(m; lags=0)
    @test_throws ArgumentError arch_lm_test(m; lags=-1)
    mc = fit_var(_dfix("climhyd.csv"), 2)
    @test_throws ArgumentError arch_lm_test(mc; lags=25)            # 21*25+1 = 526 > T-25
    e = try arch_lm_test(mc; lags=25) catch err; err end
    @test occursin("observations", lowercase(sprint(showerror, e)))
    @test arch_lm_test(mc; lags=2) isa MultivariateARCHLMTest

    @test_throws ArgumentError bg_test(m; lags=0)
    @test_throws ArgumentError es_test(m; lags=0)
    @test_throws ArgumentError bg_test(m; lags=200)
    ms = fit_var(_dfix("var2_short.csv"), 2)
    @test_throws ArgumentError es_test(ms; lags=12)
    @test_throws ArgumentError normality_test(m; orthogonalization=:bogus)
    @test_throws ArgumentError normality_test(m; orthogonalization=:doornik_hansen)
end

@testset "var_diagnostics" begin
    m = fit_var(_dfix("var2_p1.csv"), 2)
    d = var_diagnostics(m)
    @test keys(d) == (:portmanteau, :bg, :es, :normality, :arch)
    @test d.portmanteau.lags == max(m.p + 1, min(10, m.T ÷ 4)) == 10
    @test d.portmanteau.adjusted
    @test d.bg.lags == 5 && d.arch.lags == 5
    @test d.normality.univariate !== nothing && d.arch.univariate !== nothing
    d2 = var_diagnostics(m; pt_lags=8, bg_lags=3, arch_lags=2)
    @test d2.portmanteau.statistic ≈ REF_R_DIAG["A_p2"]["pt_adjusted"]["statistic"] rtol=1e-11
    @test d2.bg.lags == 3 && d2.es.lags == 3 && d2.arch.lags == 2
    # a short series: the default pt_lags still exceeds p
    ms = fit_var(_dfix("var2_short.csv"), 2)
    @test var_diagnostics(ms; bg_lags=2, arch_lags=1).portmanteau.lags == 7 == max(3, min(10, 28 ÷ 4))
end

@testset "HypothesisTest contract and show" begin
    m = fit_var(_dfix("var2_p1.csv"), 2)
    for t in (portmanteau_test(m; lags=8), normality_test(m), arch_lm_test(m; lags=5),
              bg_test(m; lags=5), es_test(m; lags=5))
        @test t isa HypothesisTest
        @test statistic(t) === t.statistic
        @test pvalue(t) === t.pvalue
        @test 0.0 <= pvalue(t) <= 1.0
        @test !isempty(sprint(show, t))
        @test isconcretetype(typeof(t))
    end
    @test occursin("ortmanteau", sprint(show, portmanteau_test(m; lags=8)))
    @test occursin("adjusted", sprint(show, portmanteau_test(m; lags=8, adjusted=true)))
    @test occursin("skewness", sprint(show, normality_test(m)))
    s = sprint(show, normality_test(m; univariate=true))
    @test occursin("y1", s) && occursin("y2", s)
    @test occursin("Breusch-Godfrey", sprint(show, bg_test(m)))
    @test occursin("(df = 20, 364)", sprint(show, es_test(m)))
    @test occursin("ARCH", sprint(show, arch_lm_test(m; univariate=true)))
    # a p-value below 1e-4 prints in scientific form rather than as 0
    @test occursin("e-", sprint(show, normality_test(fit_var(_dfix("uschange.csv"), 2))))
end

@testset "type stability and performance" begin
    mC = fit_var(_dfix("climhyd.csv"), 2)
    @test (@inferred portmanteau_test(mC; lags=8)) isa PortmanteauTest
    @test (@inferred normality_test(mC)) isa MultivariateNormalityTest
    @test (@inferred bg_test(mC; lags=3)) isa BreuschGodfreyTest
    @test (@inferred es_test(mC; lags=3)) isa EdgertonShukurTest
    @test (@inferred arch_lm_test(mC; lags=2)) isa MultivariateARCHLMTest

    # C_0 is factorized ONCE: cost near-linear in lags
    portmanteau_test(mC; lags=4); portmanteau_test(mC; lags=16)
    t4  = minimum(@elapsed(for _ in 1:100; portmanteau_test(mC; lags=4);  end) for _ in 1:3)
    t16 = minimum(@elapsed(for _ in 1:100; portmanteau_test(mC; lags=16); end) for _ in 1:3)
    @test t16 < 6 * t4
    portmanteau_test(mC; lags=8)
    @test (@allocated portmanteau_test(mC; lags=8)) < 60_000
end
