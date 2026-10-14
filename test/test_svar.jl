using DelimitedFiles

# Stage 10.7: structural VARs.
#
# References: vars 1.6.1 (SVAR with estmethod = "scoring", BQ) and
# statsmodels 0.15.0 (SVAR, free elements marked "E"). The structural
# estimates are numerically optimized MLEs, so the references agree with each
# other only to their optimizer tolerance (~5e-9 at K = 2, ~1e-5 at K = 3);
# the EXACT checks carry the weight: P P' = Σ for every just-identified
# scheme, the analytic optimum of the two-parameter A-model, Blanchard-Quah's
# lower-triangular long-run matrix, and vars' standard errors and LR
# statistic, which are closed-form at the optimum and match to 1e-13.
#
# One handoff premise is corrected here: A = [1 0; missing 1] with B = I is
# OVER-identified (one free parameter for three distinct elements of Σ) --
# rejected by the likelihood ratio, LR = 31.8 on 2 df -- and statsmodels' SVAR
# llf is the reduced form's whatever the restrictions, which is what made it
# look just-identified.

if !isdefined(@__MODULE__, :REF_R)
    include(joinpath(@__DIR__, "fixtures", "var", "refs.jl"))
end

_sfx(name) = readdlm(joinpath(@__DIR__, "fixtures", "var", name), ','; skipstart=1)
_sparse(x) = x isa AbstractString ? eval(Meta.parse(x)) : x
_smat(x) = (v = _sparse(x); Float64[v[i][j] for i in eachindex(v), j in eachindex(v[1])])
const _V2 = fit_var(_sfx("var2_p1.csv"), 2)
const _V3 = fit_var(_sfx("var3_p2.csv"), 2)
_chol(S) = Matrix(cholesky(Hermitian(S)).L)

@testset "every just-identified decomposition satisfies P P' = Σ, exactly" begin
    K2 = [(_V2, fit_svar(_V2; A=[1.0 0.0; missing 1.0], B=[missing 0.0; 0.0 missing])),
          (_V2, fit_svar(_V2; B=[missing 0.0; missing missing])),
          (_V2, fit_svar(_V2; A=[missing 0.0; missing missing])),
          (_V3, fit_svar(_V3; A=[1.0 0.0 0.0; missing 1.0 0.0; missing missing 1.0],
                              B=[missing 0.0 0.0; 0.0 missing 0.0; 0.0 0.0 missing])),
          (_V3, fit_svar(_V3; B=[missing 0.0 0.0; missing missing 0.0; missing missing missing])),
          (_V2, blanchard_quah(_V2)), (_V3, blanchard_quah(_V3))]
    for (m, s) in K2
        @test s.P * s.P' ≈ m.sigma_df atol=1e-10
        @test s.lr_test === nothing
        @test s.converged
        @test s.P ≈ s.A \ s.B atol=1e-14
    end
    # the recursive schemes reproduce the Cholesky factor exactly
    L2 = _chol(_V2.sigma_df); L3 = _chol(_V3.sigma_df)
    @test K2[1][2].P ≈ L2 atol=1e-10
    @test K2[2][2].B ≈ L2 atol=1e-10
    @test K2[4][2].P ≈ L3 atol=1e-9
    @test K2[5][2].B ≈ L3 atol=1e-9
    # and the free element of the recursive A is the normalized off-diagonal
    @test K2[1][2].A[2, 1] ≈ -L2[2, 1] / L2[1, 1] atol=1e-12
end

@testset "a just-identified model's likelihood is the reduced form's, at Σ_df" begin
    for s in (fit_svar(_V2; A=[1.0 0.0; missing 1.0], B=[missing 0.0; 0.0 missing]),
              fit_svar(_V2; B=[missing 0.0; missing missing]), blanchard_quah(_V2))
        K, T = 2, _V2.T
        exact = -K * T / 2 * log(2pi) - T / 2 * logdet(_V2.sigma_df) - K * T / 2
        @test loglikelihood(s) ≈ exact rtol=1e-12
        # it differs from the VAR's own loglik only through the Σ divisor
        @test loglikelihood(s) - loglikelihood(_V2) ≈
              -T / 2 * (logdet(_V2.sigma_df) - logdet(_V2.sigma_mle)) rtol=1e-10
    end
    s3 = fit_svar(_V3; B=[missing 0.0 0.0; missing missing 0.0; missing missing missing])
    @test loglikelihood(s3) ≈ -3 * _V3.T / 2 * (log(2pi) + 1) - _V3.T / 2 * logdet(_V3.sigma_df) rtol=1e-12
end

@testset "the A-model with B = I is over-identified, and vars rejects it" begin
    R = REF_R_SVAR["A_k2_scoring"]
    a = fit_svar(_V2; A=[1.0 0.0; missing 1.0])
    @test a.svar_type === :A
    @test a.B == I(2)
    @test count(a.A_mask) == 1
    # the optimum is analytic here: a = -Σ₁₂/Σ₁₁
    @test a.A[2, 1] ≈ -_V2.sigma_df[2, 1] / _V2.sigma_df[1, 1] atol=1e-11   # scoring stops at a 1e-10 step
    @test a.A[2, 1] ≈ _smat(R["A"])[2, 1] atol=1e-7                    # vars, scoring
    @test a.A[2, 1] ≈ _smat(REF_PY_SVAR["A_k2"]["A"])[2, 1] atol=1e-7   # statsmodels
    @test a.A[1, 1] == 1.0 && a.A[1, 2] == 0.0 && a.A[2, 2] == 1.0
    # vars' estmethod = "direct" stops short; we must not match it
    @test _smat(REF_R_SVAR["A_k2_direct"]["A"])[2, 1] == -0.2937500000000004
    @test !isapprox(a.A[2, 1], -0.29375; atol=1e-6)
    # it cannot reproduce Σ: the structural variances are pinned at one
    @test !(a.P * a.P' ≈ _V2.sigma_df)
    # the over-identification test is the likelihood ratio against the
    # just-identified model, 2(l_u - l_r)
    ju = fit_svar(_V2; A=[1.0 0.0; missing 1.0], B=[missing 0.0; 0.0 missing])
    @test a.lr_test.df == 2 == R["LR"]["par"]
    @test a.lr_test.statistic ≈ 2 * (ju.loglik - a.loglik) rtol=1e-8
    # vars (and Lutkepohl's short form) drop tr(Σ_r⁻¹Σ_u) - K, which is zero
    # only when the structural variances are free. Here they are pinned at
    # one, |Σ_r| = 1, and the short form is -T ln|Σ_u| -- unit-dependent.
    # vars' 121.6 is that short form; this package no longer matches it.
    Sr = a.P * a.P'
    @test R["LR"]["stat"] ≈ _V2.T * (logdet(Sr) - logdet(_V2.sigma_df)) rtol=1e-8
    @test !isapprox(a.lr_test.statistic, R["LR"]["stat"]; rtol=0.01)
    # rescaling the data changes the restriction's truth (variances fixed
    # at one), not the arithmetic: the full statistic stays a valid LR
    a100 = fit_svar(fit_var(100 .* _V2.Y, _V2.p); A=[1.0 0.0; missing 1.0])
    j100 = fit_svar(fit_var(100 .* _V2.Y, _V2.p); A=[1.0 0.0; missing 1.0], B=[missing 0.0; 0.0 missing])
    @test a100.lr_test.statistic ≈ 2 * (j100.loglik - a100.loglik) rtol=1e-6
    # vars prints p = 0 (its 1 - CDF again)
    @test R["LR"]["p"] == 0
    @test 0 < a.lr_test.pvalue < 1e-6
    # its standard error is the expected-information one, as vars reports
    @test a.stderr_A[2, 1] ≈ _smat(R["Ase"])[2, 1] rtol=1e-12
    @test a.stderr_A[1, 1] == 0.0 && a.stderr_A[1, 2] == 0.0 && a.stderr_A[2, 2] == 0.0
    @test all(iszero, a.stderr_B)
    # statsmodels' llf is the reduced form's whatever the restrictions --
    # which is what made this model look just-identified
    @test REF_PY_SVAR["A_k2"]["llf"] ≈ loglikelihood(_V2) rtol=1e-12
    @test REF_PY_SVAR["B_k2"]["llf"] ≈ loglikelihood(_V2) rtol=1e-12
    @test loglikelihood(a) < loglikelihood(fit_svar(_V2; A=[1.0 0.0; missing 1.0], B=[missing 0.0; 0.0 missing]))
end

@testset "B-model and AB-model vs both references" begin
    b = fit_svar(_V2; B=[1.0 0.0; missing 1.0])
    @test b.svar_type === :B && b.A == I(2)
    @test b.B[2, 1] ≈ _V2.sigma_df[2, 1] / _V2.sigma_df[1, 1] atol=1e-11
    @test b.B[2, 1] ≈ _smat(REF_R_SVAR["B_k2_scoring"]["B"])[2, 1] atol=1e-7
    @test b.B[2, 1] ≈ _smat(REF_PY_SVAR["B_k2"]["B"])[2, 1] atol=1e-7
    @test b.stderr_B[2, 1] ≈ _smat(REF_R_SVAR["B_k2_scoring"]["Bse"])[2, 1] rtol=1e-12
    @test b.lr_test.df == 2
    # AB with B fixed at I is the A-model (statsmodels' AB_k2)
    ab = fit_svar(_V2; A=[1.0 0.0; missing 1.0], B=[1.0 0.0; 0.0 1.0])
    @test ab.svar_type === :AB
    @test ab.A ≈ fit_svar(_V2; A=[1.0 0.0; missing 1.0]).A atol=1e-12
    @test ab.A[2, 1] ≈ _smat(REF_PY_SVAR["AB_k2"]["A"])[2, 1] atol=1e-7
end

@testset "trivariate A-model vs both references" begin
    a3 = fit_svar(_V3; A=[1.0 0.0 0.0; missing 1.0 0.0; missing missing 1.0])
    @test a3.A ≈ _smat(REF_R_SVAR["A_k3_scoring"]["A"]) atol=1e-6    # optimizer tolerance
    @test a3.A ≈ _smat(REF_PY_SVAR["A_k3"]["A"]) atol=1e-4
    @test all(iszero, triu(a3.A, 1))
    @test a3.stderr_A ≈ _smat(REF_R_SVAR["A_k3_scoring"]["Ase"]) rtol=1e-9
    @test count(a3.A_mask) == 3 && a3.lr_test.df == 3
    # scoring and quasi-Newton reach the same optimum
    m3 = fit_svar(_V3; A=[1.0 0.0 0.0; missing 1.0 0.0; missing missing 1.0], method=:mle)
    @test m3.A ≈ a3.A atol=1e-6
    @test m3.method === :mle && a3.method === :scoring
    # a tighter tolerance does not move the answer materially
    t3 = fit_svar(_V3; A=[1.0 0.0 0.0; missing 1.0 0.0; missing missing 1.0], tol=1e-14, maxiter=1000)
    @test t3.A ≈ a3.A atol=1e-9
    # statsmodels' structural IRF (P = A⁻¹) agrees
    sm = _sparse(REF_PY_SVAR["A_k3"]["irf_orth"])
    r = irf(a3, 4)
    for h in 1:5
        @test r.orth[h] ≈ Float64[sm[h][i][j] for i in 1:3, j in 1:3] atol=1e-4
    end
end

@testset "blanchard_quah: the long-run matrix is lower triangular, exactly" begin
    bq = blanchard_quah(_V2)
    @test bq.svar_type === :bq
    @test all(iszero, triu(bq.lrim, 1))
    @test bq.lrim ≈ _smat(REF_R_SVAR["BQ_k2"]["LRIM"]) atol=1e-12
    @test bq.B ≈ _smat(REF_R_SVAR["BQ_k2"]["B"]) atol=1e-12
    @test bq.lrim ≈ lr_effects(_V2) * bq.B atol=1e-12
    @test bq.B * bq.B' ≈ _V2.sigma_df atol=1e-12
    @test bq.A == I(2)
    @test bq.stderr_A === nothing && bq.stderr_B === nothing
    @test bq.method === :closed_form
    bq3 = blanchard_quah(_V3)
    @test all(iszero, triu(bq3.lrim, 1))
    @test bq3.lrim ≈ _smat(REF_R_SVAR["BQ_k3"]["LRIM"]) atol=1e-12
    @test bq3.B ≈ _smat(REF_R_SVAR["BQ_k3"]["B"]) atol=1e-12
    @test !all(iszero, triu(bq3.B, 1))                # BQ is NOT recursive in B
    @test bq3.lrim ≈ lr_effects(_V3) * bq3.B atol=1e-12
    # the long-run cumulative structural response IS the long-run matrix
    @test irf(bq, 400; cumulative=true).orth[end] ≈ bq.lrim atol=1e-9
    # an unstable VAR has no long-run matrix; vars::BQ computes one anyway
    ex = fit_var(_sfx("var2_explosive.csv"), 1)
    @test_throws ArgumentError blanchard_quah(ex)
    e = try blanchard_quah(ex) catch err; err end
    @test occursin("stable", lowercase(sprint(showerror, e)))
end

@testset "structural IRF and FEVD reuse 10.3, with P for the Cholesky factor" begin
    bq = blanchard_quah(_V2)
    r = irf(bq, 4)
    Phi = ma_rep(_V2, 4)
    for h in 1:5
        @test r.orth[h] ≈ Phi[h] * bq.P atol=1e-12
        @test r.nonorth[h] == Phi[h]
    end
    @test r.orth[1] ≈ bq.P atol=1e-14
    @test r.factorization === :blanchard_quah
    bi = REF_R_SVAR["BQ_irf"]
    for h in 1:5
        @test r.orth[h] ≈ Float64[bi[["y1", "y2"][j]][h][i] for i in 1:2, j in 1:2] atol=1e-12
    end
    f = fevd(bq, 4)
    bf = REF_R_SVAR["BQ_fevd"]
    for i in 1:2, h in 1:4
        @test sum(f.decomp[i, h, :]) ≈ 1.0 atol=1e-13
        @test f.decomp[i, h, :] ≈ Float64.(bf[["y1", "y2"][i]][h]) atol=1e-12
    end
    # a recursive just-identified model's structural IRF IS the orthogonalized IRF
    s = fit_svar(_V2; A=[1.0 0.0; missing 1.0], B=[missing 0.0; 0.0 missing])
    rs, ro = irf(s, 6), irf(_V2, 6)
    @test rs.factorization === :structural
    for h in 1:7
        @test rs.orth[h] ≈ ro.orth[h] atol=1e-9
    end
    @test fevd(s, 6).decomp ≈ fevd(_V2, 6).decomp atol=1e-9
    # selection and cumulation work as for a VAR
    @test irf(bq, 4; impulse=2).orth[3] == r.orth[3][:, [2]]
    @test irf(bq, 4; cumulative=true).orth[5] ≈ sum(r.orth) atol=1e-12
    @test_throws ArgumentError irf(bq, 4; stderr=true)
    @test_throws ArgumentError irf(bq, 4; boot=true)
    @test_throws ArgumentError irf(bq, 0)
    @test_throws ArgumentError fevd(bq, 0)
end

@testset "structural shocks" begin
    s = fit_svar(_V2; B=[missing 0.0; missing missing])
    E = TSAnalytics.structural_shocks(s)
    @test size(E) == size(residuals(_V2))
    q = size(coef(_V2), 2)
    @test (E' * E) ./ (_V2.T - q) ≈ I(2) atol=1e-10
    @test residuals(s) === residuals(_V2)
    @test nobs(s) == nobs(_V2)
end

@testset "identification is checked BEFORE optimizing" begin
    v = _V2
    Aall = Matrix{Union{Missing,Float64}}(fill(missing, 2, 2))
    @test_throws ArgumentError fit_svar(v; A=Aall)          # 4 > 3: under-identified
    e = try fit_svar(v; A=Aall) catch err; err end
    @test occursin("identified", lowercase(sprint(showerror, e)))
    @test occursin("not identified", REF_R_SVAR["err_underid"])
    @test occursin("Singular matrix", REF_PY_SVAR["err_allfree"])   # statsmodels, from deep inside

    @test_throws ArgumentError fit_svar(v; A=Union{Missing,Float64}[1.0 0.0; 0.0 1.0])
    e = try fit_svar(v; A=[1.0 0.0; 0.0 1.0]) catch err; err end
    @test occursin("free", lowercase(sprint(showerror, e)))
    @test occursin("No parameters", REF_R_SVAR["err_noNA"])

    Asame = [1.0 0.0; missing 1.0]; Bsame = [1.0 0.0; missing 1.0]
    @test_throws ArgumentError fit_svar(v; A=Asame, B=Bsame)
    e = try fit_svar(v; A=Asame, B=Bsame) catch err; err end
    @test occursin("same position", lowercase(sprint(showerror, e)))
    @test occursin("same positions", REF_R_SVAR["err_both"])

    @test_throws ArgumentError fit_svar(v)
    @test occursin("A array not given", REF_PY_SVAR["err_noA"])
    @test_throws ArgumentError fit_svar(v; A=[1.0 0.0 0.0; missing 1.0 0.0])
    @test_throws ArgumentError fit_svar(v; A=[1.0 0.0; NaN 1.0])
    e = try fit_svar(v; A=[1.0 0.0; NaN 1.0]) catch err; err end
    @test occursin("missing", sprint(showerror, e))
    @test occursin("zero-size array", REF_PY_SVAR["err_nan_marker"])
    @test_throws ArgumentError fit_svar(v; A=[1.0 0.0; missing 1.0], method=:bogus)
    @test_throws ArgumentError fit_svar(v; A=[1.0 0.0; missing 1.0], maxiter=0)
    @test_throws ArgumentError fit_svar(v; A=[1.0 0.0; missing 1.0], tol=0.0)
    # a structurally singular A is refused by name, not by a LinAlg error
    @test_throws ArgumentError fit_svar(v; A=[missing 0.0; missing 0.0])
end

@testset "contract, show, type stability" begin
    s = fit_svar(_V2; A=[1.0 0.0; missing 1.0])
    @test s isa TimeSeriesModel
    @test isconcretetype(SVARModel)
    str = sprint(show, s)
    @test occursin("SVAR", str) || occursin("Structural VAR", str)
    @test occursin("A-model", str)
    @test occursin("converged", lowercase(str))
    @test occursin("over-identified", str)
    @test occursin("LR test", str)
    @test occursin("just-identified", sprint(show, fit_svar(_V2; B=[missing 0.0; missing missing])))
    @test occursin("Blanchard", sprint(show, blanchard_quah(_V2)))
    @test (@inferred blanchard_quah(_V2)) isa SVARModel
    @test fit_svar(_V2; B=[missing 0.0; missing missing]) isa SVARModel
end
