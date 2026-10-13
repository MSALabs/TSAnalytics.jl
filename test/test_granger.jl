using DelimitedFiles

# Stage 10.2: Granger causality.
#
# DUAL-VERIFIED, but the grouping matters and is the item's central structural
# fact. There are TWO different tests, each with two references that agree
# exactly with each other:
#
#   system test   : vars::causality  ==  statsmodels test_causality
#   pairwise test : lmtest::grangertest == statsmodels grangercausalitytests
#                                          ["ssr_ftest"]
#
# The F statistics coincide only when |caused| = 1; the p-values never do,
# because df2 differs by a factor of K. So this package exposes both rather
# than choosing, as two methods of one name.
#
# Two reference defects bound what can be asserted against what:
#
#  1. vars::causality computes 1 - pf(...) and 1 - pchisq(...) instead of the
#     upper tail, so its p-values underflow to EXACTLY ZERO in routine use.
#     Four of the fifteen reference cases do this. The statistic is asserted
#     against vars; the tail is asserted against statsmodels and against
#     _f_ccdf directly.
#  2. statsmodels' test_inst_causality raises IndexError whenever BOTH sets
#     have more than one member, so vars is the only reference for a general
#     block. Verified to reproduce on 0.14.1, the version installed here.

if !isdefined(@__MODULE__, :REF_R_GRANGER)
    include(joinpath(@__DIR__, "fixtures", "var", "refs.jl"))
end

_gfix(name) = readdlm(joinpath(@__DIR__, "fixtures", "var", name), ','; skipstart=1)

@testset "Stage 10.2 — Granger causality" begin
    T = TSAnalytics
    A2 = _gfix("var2_p1.csv")
    B3 = _gfix("var3_p2.csv")
    SH = _gfix("var2_short.csv")
    US = _gfix("uschange.csv")
    E5 = _gfix("econ5.csv")
    CL = _gfix("climhyd.csv")

    @testset "_f_ccdf — the new primitive, tested before anything uses it" begin
        # The F upper tail via the regularized incomplete beta with swapped
        # parameters at the COMPLEMENTARY argument, formed directly. R's
        # pf(lower.tail=FALSE) at d1=5, d2=800 is the truth column.
        R_TAIL = [(5.0, 1.618120e-04), (20.0, 7.980745e-19), (50.0, 4.093317e-45),
                  (100.0, 6.611145e-82), (200.0, 3.364102e-138)]
        for (f, tail) in R_TAIL
            @test isapprox(T._f_ccdf(f, 5, 800), tail; rtol=1e-6)
        end
        # and this is why `1 - _f_cdf` is not acceptable: it flatlines to
        # exactly zero from about f = 20, which is the vars failure mode
        @test 1 - T._f_cdf(5.0, 5, 800) > 0
        for (f, _) in R_TAIL[2:end]
            @test 1 - T._f_cdf(f, 5, 800) == 0.0
            @test T._f_ccdf(f, 5, 800) > 0
        end
        # consistency with the existing CDF where both are well conditioned
        for f in (0.5, 1.0, 2.0, 3.0)
            @test isapprox(T._f_ccdf(f, 4, 50) + T._f_cdf(f, 4, 50), 1.0; atol=1e-12)
        end
        @test T._f_ccdf(0.0, 3, 10) == 1.0
        @test T._f_ccdf(-1.0, 3, 10) == 1.0
        # monotone decreasing in f
        @test T._f_ccdf(1.0, 3, 20) > T._f_ccdf(2.0, 3, 20) > T._f_ccdf(10.0, 3, 20)
    end

    @testset "_chisq_ccdf was already right, and beats vars in the same regime" begin
        # vars reports 0 for this; the true upper tail is 5.229214e-38
        @test isapprox(T._chisq_ccdf(184.728185617809, 5), 5.229214e-38; rtol=1e-5)
        @test T._chisq_ccdf(184.728185617809, 5) > 0
    end

    @testset "the vars p-value defect, on the case where it actually bites" begin
        # CORRECTING THE HANDOFF: §4.2 says "the same 0.000000e+00 appears in
        # four of the fifteen reference cases above (uschange cause Income and
        # cause Savings, uschange cause {Income,Production}, and every climhyd
        # case)". That is wrong about the uschange cases -- their p-values are
        # 0.890, 0.967 and 0.497, nowhere near underflow -- and climhyd is not
        # among the fifteen at all. Checked: NONE of the fifteen underflows.
        zeros_in_vars = [k for (k, v) in REF_R_GRANGER
                         if v isa Dict && haskey(v, "granger_p") && v["granger_p"] == 0.0]
        @test isempty(zeros_in_vars)
        @test isapprox(REF_R_GRANGER["US_p2_cause_Income"]["granger_p"], 0.8899812692;
                       rtol=1e-8)
        @test isapprox(REF_R_GRANGER["US_p2_cause_Savings"]["granger_p"], 0.9669600021;
                       rtol=1e-8)

        # The DEFECT ITSELF is real, and climhyd is where it bites. This
        # package reproduces the handoff's figures for that case exactly, and
        # the `1 - CDF` form vars uses returns exactly zero at the same point.
        mc = fit_var(CL, 2)
        t = granger_test(mc; cause=1)
        @test isapprox(statistic(t), 13.359738021985; rtol=1e-11)
        @test (t.df1, t.df2) == (10, 2634)
        @test isapprox(pvalue(t), 3.664479e-23; rtol=1e-6)
        @test 1 - T._f_cdf(statistic(t), t.df1, t.df2) == 0.0    # what vars reports
        @test pvalue(t) > 0                                      # what is true

        i = instantaneous_causality(mc; cause=1)
        @test isapprox(statistic(i), 184.728185617809; rtol=1e-11)
        @test i.df == 5
        @test isapprox(pvalue(i), 5.229214e-38; rtol=1e-5)

        # And note what df2 = 2634 implies: K*T - K*(Kp+ndet) = 6*452 - 6*13,
        # so T = 452 and climhyd has 454 rows. The handoff's own tables say
        # 453, which would give 2634 - 6 = 2628. Its computed numbers were
        # therefore generated from the real 454-row file, confirming that the
        # row count was a documentation slip and not a different fixture.
        @test size(CL, 1) == 454
        @test mc.T == 452
        @test t.df2 == 6 * 452 - 6 * (6 * 2 + 1)
    end

    # (reference key, data, p, cause as indices)
    SYS = [("A_p1_cause_y2", A2, 1, [2]), ("A_p1_cause_y1", A2, 1, [1]),
           ("A_p2_cause_y2", A2, 2, [2]), ("A_p2_cause_y1", A2, 2, [1]),
           ("A_p4_cause_y2", A2, 4, [2]), ("B_p2_cause_x1", B3, 2, [1]),
           ("B_p2_cause_x2", B3, 2, [2]), ("B_p2_cause_x3", B3, 2, [3]),
           ("B_p2_cause_x1x2", B3, 2, [1, 2]), ("B_p2_cause_x2x3", B3, 2, [2, 3]),
           ("SH_p2_cause_y2", SH, 2, [2]), ("E5_p2_cause_gnp", E5, 2, [2])]

    @testset "system test against vars — $key" for (key, Y, p, cause) in SYS
        m = fit_var(Y, p)
        t = granger_test(m; cause=cause)
        r = REF_R_GRANGER[key]
        @test t.kind === :system
        @test isapprox(statistic(t), r["granger_stat"]; rtol=1e-10)
        @test t.df1 == r["granger_df1"]
        @test t.df2 == r["granger_df2"]
        @test t.nobs == r["obs"]
        @test t.lags == p
        # df1 = p * |cause| * |caused|, df2 = K*T - K*(Kp+ndet) -- the SYSTEM
        # residual degrees of freedom, not one equation's
        K = size(Y, 2)
        @test t.df1 == p * length(cause) * (K - length(cause))
        @test t.df2 == K * m.T - K * (K * p + 1)
        # the p-value: only comparable with vars where vars has not underflowed
        if r["granger_p"] > 0
            @test isapprox(pvalue(t), r["granger_p"]; rtol=1e-8)
        else
            @test pvalue(t) > 0          # this package returns the real tail
            @test pvalue(t) < 1e-10
        end
        # the chi-square form is df1 * F, as both references report
        @test isapprox(t.wald, t.df1 * statistic(t); rtol=1e-12)
    end

    @testset "system test against statsmodels — $key" for (key, Y, p, cause) in SYS
        haskey(REF_PY_GRANGER, key) || continue
        m = fit_var(Y, p)
        t = granger_test(m; cause=cause)
        q = REF_PY_GRANGER[key]
        @test isapprox(statistic(t), q["f_stat"]; rtol=1e-10)
        @test [t.df1, t.df2] == Int.(q["f_df"])
        # statsmodels computes the tail correctly, so this one IS comparable
        @test isapprox(pvalue(t), q["f_p"]; rtol=1e-8)
        @test isapprox(t.wald, q["wald_stat"]; rtol=1e-10)
        @test isapprox(t.wald_pvalue, q["wald_p"]; rtol=1e-8)
        @test t.df1 == q["wald_df"]
    end

    @testset "the two references agree with each other on the system test" begin
        # charter §5.11: this is what makes a disagreement during
        # implementation a bug rather than a convention dispute
        for (key, _, _, _) in SYS
            haskey(REF_PY_GRANGER, key) || continue
            @test isapprox(REF_R_GRANGER[key]["granger_stat"],
                           REF_PY_GRANGER[key]["f_stat"]; rtol=1e-9)
            @test REF_R_GRANGER[key]["granger_df1"] == Int(REF_PY_GRANGER[key]["f_df"][1])
            @test REF_R_GRANGER[key]["granger_df2"] == Int(REF_PY_GRANGER[key]["f_df"][2])
        end
    end

    @testset "uschange blocks, where vars is the reference that underflows" begin
        m = fit_var(US, 2)
        for (key, cause) in (("US_p2_cause_Income", [2]),
                             ("US_p2_cause_Savings", [4]),   # Consumption, Income, Production, Savings, Unemployment
                             ("US_p2_cause_IncProd", [2, 3]))
            t = granger_test(m; cause=cause)
            r = REF_R_GRANGER[key]
            @test isapprox(statistic(t), r["granger_stat"]; rtol=1e-9)
            @test t.df1 == r["granger_df1"]
            @test t.df2 == r["granger_df2"]
            if r["granger_p"] > 0
                @test isapprox(pvalue(t), r["granger_p"]; rtol=1e-8)
            end
            @test 0 <= pvalue(t) <= 1
        end
    end

    @testset "PAIRWISE test against lmtest, bit-identical" begin
        PW = [("lm_A_p1_y2_to_y1", A2[:, 1], A2[:, 2], 1),
              ("lm_A_p2_y2_to_y1", A2[:, 1], A2[:, 2], 2),
              ("lm_A_p4_y2_to_y1", A2[:, 1], A2[:, 2], 4),
              ("lm_A_p2_y1_to_y2", A2[:, 2], A2[:, 1], 2),
              ("lm_B_p2_x1_to_x3", B3[:, 3], B3[:, 1], 2),
              ("lm_US_p2_Inc_to_Cons", US[:, 1], US[:, 2], 2)]
        for (key, y, x, lags) in PW
            t = granger_test(y, x; lags=lags)
            r = REF_R_GRANGER[key]
            @test t.kind === :pairwise
            @test isapprox(statistic(t), r["F"]; rtol=1e-9)
            @test t.df2 == r["res_df_unrestricted"]
            @test isapprox(pvalue(t), r["p"]; rtol=1e-8)
            @test t.df1 == lags
            # df2 = T - 2*lags - 1, ONE equation's degrees of freedom
            @test t.df2 == (length(y) - lags) - 2 * lags - 1
        end
    end

    @testset "CONTRAST: the two tests' statistics coincide only when |caused|=1" begin
        # charter §5.11 / handoff §3.3, asserted both ways round.
        # k = 2: the system test's default caused is one variable, so the
        # statistics agree exactly -- but the p-values still differ, because
        # df2 differs by a factor of K.
        m2 = fit_var(A2, 2)
        sys = granger_test(m2; cause=:y2)
        pw = granger_test(A2[:, 1], A2[:, 2]; lags=2)
        @test isapprox(statistic(sys), statistic(pw); rtol=1e-10)
        @test sys.df2 == 386
        @test pw.df2 == 193
        @test sys.df2 == 2 * pw.df2                    # the factor of K
        @test !isapprox(pvalue(sys), pvalue(pw); rtol=1e-4)
        @test isapprox(pvalue(sys), 0.4821580341; rtol=1e-7)
        @test isapprox(pvalue(pw), 0.4828206979; rtol=1e-7)

        # k = 3: |caused| = 2, and now even the statistics diverge
        m3 = fit_var(B3, 2)
        sys3 = granger_test(m3; cause=1)
        pw3 = granger_test(B3[:, 3], B3[:, 1]; lags=2)
        @test isapprox(statistic(sys3), 3.242043205406; rtol=1e-9)
        @test isapprox(statistic(pw3), 3.931001499002; rtol=1e-9)
        @test !isapprox(statistic(sys3), statistic(pw3); rtol=1e-3)
    end

    @testset "INSTANTANEOUS causality against vars" begin
        for (key, Y, p, cause) in SYS
            r = REF_R_GRANGER[key]
            haskey(r, "inst_stat") || continue
            m = fit_var(Y, p)
            t = instantaneous_causality(m; cause=cause)
            @test isapprox(statistic(t), r["inst_stat"]; rtol=1e-9)
            @test t.df == r["inst_df"]
            @test t.df == length(cause) * (size(Y, 2) - length(cause))
            if r["inst_p"] > 1e-10
                @test isapprox(pvalue(t), r["inst_p"]; rtol=1e-7)
            elseif r["inst_p"] > 0
                # far out in the tail, _chisq_ccdf and R's pchisq agree to
                # about 0.1% rather than to 7 figures -- both are computing an
                # incomplete gamma to their own tolerance, and at p ~ 1.8e-14
                # the absolute difference is 2e-17. Measured agreement on the
                # log scale is 3.4e-5, which is the only scale on which such a
                # number means anything.
                @test isapprox(log(pvalue(t)), log(r["inst_p"]); rtol=1e-4)
                @test isapprox(pvalue(t), r["inst_p"]; rtol=2e-3)
            else
                @test pvalue(t) > 0
            end
        end
    end

    @testset "instantaneous causality against statsmodels where it can run" begin
        # it manages (1, K-1) and (K-1, 1) but not a general block
        for (key, Y, p, cause) in SYS
            haskey(REF_PY_GRANGER, key) || continue
            haskey(REF_PY_GRANGER[key], "inst_stat") || continue
            length(cause) == 1 || continue           # the case it can do
            m = fit_var(Y, p)
            t = instantaneous_causality(m; cause=cause)
            @test isapprox(statistic(t), REF_PY_GRANGER[key]["inst_stat"]; rtol=1e-9)
            @test isapprox(pvalue(t), REF_PY_GRANGER[key]["inst_p"]; rtol=1e-6)
        end
    end

    @testset "the block case statsmodels cannot do — vars is the only reference" begin
        # cause {x1,x2} against caused {x3} on k=3: statsmodels raises
        # IndexError as soon as both sets have more than one member, so this
        # is checked against vars alone, and that is stated rather than
        # implied.
        m = fit_var(B3, 2)
        t = instantaneous_causality(m; cause=[1, 2])
        @test isapprox(statistic(t), 9.212257359217128; rtol=1e-9)
        @test t.df == 2
        @test isapprox(pvalue(t), 0.009990419655878102; rtol=1e-7)
        # climhyd: k=6, where the block cases are the ones that error in
        # statsmodels, and where vars' p-value underflows to zero as well
        mc = fit_var(CL, 2)
        tc = instantaneous_causality(mc; cause=[1, 2, 3])
        @test tc.df == 3 * 3
        @test statistic(tc) > 0
        @test pvalue(tc) >= 0
        gc = granger_test(mc; cause=[1, 2, 3, 4, 5])
        @test gc.df1 == 2 * 5 * 1
        @test pvalue(gc) > 0                          # not the 0 vars reports
    end

    @testset "instantaneous causality is symmetric, as its hypothesis is" begin
        # contemporaneous correlation has no direction, so swapping the two
        # sets must return the identical statistic
        m = fit_var(B3, 2)
        a = instantaneous_causality(m; cause=[1], caused=[2, 3])
        b = instantaneous_causality(m; cause=[2, 3], caused=[1])
        @test isapprox(statistic(a), statistic(b); rtol=1e-12)
        @test a.df == b.df
        @test isapprox(pvalue(a), pvalue(b); rtol=1e-12)
    end

    @testset "the duplication-matrix pseudo-inverse closed form" begin
        # charter §7 item 4: D'D is exactly diagonal with entries 1 and 2, so
        # D+ needs no SVD. Both references call pinv and pay for one.
        for K in (2, 3, 5, 8)
            Dp = T._dupl_pinv(K)
            nv = K * (K + 1) ÷ 2
            @test size(Dp) == (nv, K * K)
            # rebuild D explicitly and check the Moore-Penrose identities
            vidx = T._vech_indices(K)
            D = zeros(K * K, nv)
            for (r, (i, j)) in enumerate(vidx)
                D[(j - 1) * K + i, r] = 1.0
                i == j || (D[(i - 1) * K + j, r] = 1.0)
            end
            @test isapprox(Dp * D, I(nv); atol=1e-14)        # D+ D = I
            DtD = D' * D
            @test isapprox(DtD, diagm(diag(DtD)); atol=0.0)  # exactly diagonal
            @test sort(unique(round.(diag(DtD); digits=12))) == (K == 1 ? [1.0] : [1.0, 2.0])
            @test isapprox(Dp, inv(DtD) * D'; atol=1e-14)    # the closed form
            # vec(A) == D * vech(A) for a symmetric A
            A = rand(MersenneTwister(K), K, K); A = A + A'
            @test isapprox(D * [A[i, j] for (i, j) in vidx], vec(A); atol=1e-14)
        end
    end

    @testset "vech ordering matches statsmodels' util.vech" begin
        # lower triangle, column by column
        @test T._vech_indices(3) == [(1,1), (2,1), (3,1), (2,2), (3,2), (3,3)]
        @test length(T._vech_indices(5)) == 15
    end

    @testset "the restriction is applied by indexing, not by a dense matrix" begin
        # charter §6.1: R is a sparse 0/1 selection, so R Σa R' is Σa[idx,idx].
        # Checked by building R densely here and confirming the same answer.
        m = fit_var(B3, 2)
        K = 3; p = 2; nz = K * p + 1
        cause_ind = [1, 2]; caused_ind = [3]
        idx = T._granger_indices(m, cause_ind, caused_ind)
        @test length(idx) == p * length(cause_ind) * length(caused_ind)
        @test allunique(idx)
        @test all(1 .<= idx .<= K * nz)
        Rm = zeros(length(idx), K * nz)
        for (r, i) in enumerate(idx)
            Rm[r, i] = 1.0
        end
        Sa = vcov(m)
        @test isapprox(Rm * Sa * Rm', Sa[idx, idx]; atol=1e-14)
        vb = vec(Matrix{Float64}(transpose(coef(m))))
        @test isapprox(Rm * vb, vb[idx]; atol=1e-14)
        # and the restricted positions really are the cause lags in the caused
        # equation: position (j-1)*nz + (l-1)*K + c
        @test idx == [(3 - 1) * nz + (l - 1) * K + c for l in 1:p for c in cause_ind]
    end

    @testset "REFUSALS: where both references wave nonsense through" begin
        m = fit_var(B3, 2)
        # vars accepts cause = every variable (warns and uses the first);
        # nothing is left to be caused
        e = try; granger_test(m; cause=[1, 2, 3]); nothing; catch err; err; end
        @test e isa ArgumentError
        @test occursin("nothing is left", e.msg)
        # statsmodels accepts overlapping cause and caused sets
        e2 = try; granger_test(m; cause=[1], caused=[1, 2]); nothing; catch err; err; end
        @test e2 isa ArgumentError
        @test occursin("both", e2.msg)
        # a repeated restriction is rank deficient
        e3 = try; granger_test(m; cause=[1, 1]); nothing; catch err; err; end
        @test e3 isa ArgumentError
        @test occursin("twice", e3.msg)
        # unknown names and out-of-range indices
        e4 = try; granger_test(m; cause=:nope); nothing; catch err; err; end
        @test e4 isa ArgumentError
        @test occursin("no variable named", e4.msg)
        @test occursin("y1, y2, y3", e4.msg)          # names what IS available
        @test_throws ArgumentError granger_test(m; cause=9)
        @test_throws ArgumentError granger_test(m; cause=Int[])
        # the same guards on the instantaneous test
        @test_throws ArgumentError instantaneous_causality(m; cause=[1, 2, 3])
        @test_throws ArgumentError instantaneous_causality(m; cause=[1], caused=[1])
        # and on the pairwise test
        @test_throws DimensionMismatch granger_test(A2[:, 1], A2[1:100, 2]; lags=2)
        @test_throws ArgumentError granger_test(A2[:, 1], A2[:, 2]; lags=0)
        @test_throws ArgumentError granger_test(A2[1:6, 1], A2[1:6, 2]; lags=2)
        yn = copy(A2[:, 1]); yn[3] = NaN
        @test_throws ArgumentError granger_test(yn, A2[:, 2]; lags=1)
    end

    @testset "HypothesisTest contract and printing" begin
        m = fit_var(A2, 1)
        t = granger_test(m; cause=:y2)
        @test t isa TSAnalytics.HypothesisTest
        @test statistic(t) === t.statistic
        @test pvalue(t) === t.pvalue
        i = instantaneous_causality(m; cause=:y2)
        @test i isa TSAnalytics.HypothesisTest
        @test statistic(i) === i.statistic
        out = sprint(show, t)
        @test occursin("Granger causality", out)
        @test occursin("do not Granger-cause", out)        # R's own wording
        @test occursin("system", out)
        @test occursin("df = (1, 392)", out)
        outp = sprint(show, granger_test(A2[:, 1], A2[:, 2]; lags=2))
        @test occursin("pairwise", outp)
        outi = sprint(show, i)
        @test occursin("instantaneous causality", outi)
        @test occursin("H0:", outi)
    end

    @testset "named variables and symbol input" begin
        # names flow through from the container, and the test reports them
        d = (gdp = B3[:, 1], infl = B3[:, 2], rate = B3[:, 3])
        m = fit_var(d, 2)
        t = granger_test(m; cause=:gdp)
        @test t.cause == [:gdp]
        @test t.caused == [:infl, :rate]
        @test occursin("gdp do not Granger-cause infl rate", sprint(show, t))
        # and by index gives the identical answer
        @test isapprox(statistic(t), statistic(granger_test(m; cause=1)); rtol=1e-14)
        # a mixed vector of symbols
        t2 = granger_test(m; cause=[:gdp, :infl], caused=[:rate])
        @test isapprox(statistic(t2), 3.743411847221183; rtol=1e-9)
    end

    @testset "explicit caused reproduces the default when it is everything else" begin
        m = fit_var(B3, 2)
        a = granger_test(m; cause=[1])
        b = granger_test(m; cause=[1], caused=[2, 3])
        @test isapprox(statistic(a), statistic(b); rtol=1e-14)
        @test a.df1 == b.df1 && a.df2 == b.df2
    end

    @testset "type stability and allocation on the hot path" begin
        m = fit_var(B3, 2)
        @test (@inferred granger_test(m; cause=[1])) isa GrangerTest
        @test (@inferred instantaneous_causality(m; cause=[1])) isa
              InstantaneousCausalityTest
        @test (@inferred granger_test(A2[:, 1], A2[:, 2]; lags=2)) isa GrangerTest
        @test T._f_ccdf(2.0, 3, 50) isa Float64
        @test (@inferred T._dupl_pinv(4)) isa Matrix{Float64}
    end
end
