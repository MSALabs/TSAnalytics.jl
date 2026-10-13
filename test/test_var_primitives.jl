using DelimitedFiles

# Stage 10.0: the primitives Stage 10 is built on — multi-response least
# squares (`_mols`), regressor construction (`build_var_Z`), the companion
# matrix, and the `tsmatrix`/`tsnames` container accessors.
#
# Built to `handoff/stage 10/stage-10.0-primitives-handoff.md`. An earlier
# version of this item was built from the stage charter before that handoff
# existed, and it missed the most important requirement: refusing a
# rank-deficient regressor matrix. Both references proceed silently there, and
# so did this package — `fit_var` on a series with one column duplicated
# returned coefficients of order 3.9e14 and a log-likelihood of +2798.9. The
# rank-deficiency testset below exists because that happened.
#
# References: ref_R_prim.json / ref_py_prim.json, regenerated on this machine
# (R 4.6.0, statsmodels 0.14.1) and diffed against the bundled copies before
# use — 1,670 values, worst 4.6e-12, no structural differences.

if !isdefined(@__MODULE__, :REF_R_PRIM)
    include(joinpath(@__DIR__, "fixtures", "var", "refs.jl"))
end

const VARFIX = joinpath(@__DIR__, "fixtures", "var")
_varfix(name) = readdlm(joinpath(VARFIX, name), ','; skipstart=1)
_pm(x) = reduce(vcat, [reshape(Float64.(r), 1, :) for r in x])

"The references put the constant FIRST in Z; this package puts it LAST.
Move row 1 of a const-first coefficient matrix to the end."
_const_last(B) = vcat(B[2:end, :], B[1:1, :])

@testset "Stage 10.0 — multivariate primitives" begin
    T = TSAnalytics
    A2 = _varfix("var2_p1.csv")
    B3 = _varfix("var3_p2.csv")
    US = _varfix("uschange.csv")

    # ------------------------------------------------------------------ _mols

    @testset "_mols: var2_p1 p=2 against R's lm and statsmodels" begin
        Z, X, _ = T.build_var_Z(A2, 2, :const)
        @test size(Z) == (198, 5)
        @test size(X) == (198, 2)
        B, E, se, ZtZinv = T._mols(Z, X)
        # handoff §3.4, R's lm(cbind(...)): rows [const, y1.l1, y2.l1, y1.l2, y2.l2]
        Bref = [ 0.414053755176  -0.234406837193
                 0.448249746032   0.220514094415
                 0.070329548445   0.397525035175
                 0.014232369339   0.013417589992
                -0.111386497811  -0.029056864880]
        SEref = [0.104260453320  0.081759879602
                 0.077607372675  0.060858832319
                 0.099412416939  0.077958103778
                 0.080462771256  0.063098003901
                 0.094266992601  0.073923119649]
        @test isapprox(B, _const_last(Bref); atol=1e-11)
        @test isapprox(se, _const_last(SEref); atol=1e-11)
        @test isapprox(E[1:2, :], [-1.702224863368 -0.565195879071;
                                   -1.571550257039 -0.005319281061]; atol=1e-11)
        Sig = (E' * E) ./ (198 - 5)
        @test isapprox(Sig, [1.011750538636 0.297273347693;
                             0.297273347693 0.622178089485]; atol=1e-11)
        # se is the diagonal of kron(Σ, (Z'Z)^-1), equation by equation
        @test isapprox(vec(se), sqrt.(diag(kron(Sig, ZtZinv))); atol=1e-12)
    end

    @testset "_mols: every fixture, against both references — $key" for key in
            ("mols_A_p1", "mols_A_p2", "mols_B_p2", "mols_US_p2")
        Y, p = key == "mols_A_p1" ? (A2, 1) : key == "mols_A_p2" ? (A2, 2) :
               key == "mols_B_p2" ? (B3, 2) : (US, 2)
        Z, X, _ = T.build_var_Z(Y, p, :const)
        B, E, se, _ = T._mols(Z, X)
        r = REF_R_PRIM[key]
        q = REF_PY_PRIM[key]
        @test size(Z) == Tuple(Int.(r["Z_dim"]))
        @test size(Z, 1) == r["T"] == q["T"]
        # both references build Z constant-first; compare after reordering
        @test isapprox(B, _const_last(_pm(r["B"])); rtol=1e-9)
        @test isapprox(se, _const_last(_pm(r["se"])); rtol=1e-8)
        @test isapprox(B, _const_last(_pm(q["B"])); rtol=1e-9)
        @test isapprox((E' * E) ./ size(Z, 1), _pm(r["sigma_mle"]); rtol=1e-9)
        @test isapprox((E' * E) ./ (size(Z, 1) - size(Z, 2)), _pm(r["sigma_df"]); rtol=1e-9)
        @test isapprox(E[1:size(_pm(r["resid_head"]), 1), :], _pm(r["resid_head"]); rtol=1e-8)
        # statsmodels confirms its multi-response solve IS its VAR fit
        @test q["B_matches_fit"] == true
    end

    @testset "_mols: a correct multi-response solve IS the VAR fit" begin
        for (Y, p) in ((A2, 1), (A2, 2), (B3, 2), (US, 2))
            Z, X, _ = T.build_var_Z(Y, p, :const)
            B, _, se, _ = T._mols(Z, X)
            m = fit_var(Y, p)
            @test isapprox(permutedims(B), coef(m); atol=1e-11)
            @test isapprox(permutedims(se), stderror(m); atol=1e-11)
        end
    end

    @testset "_mols: column-by-column equivalence with _ols, one factorization" begin
        X = [ones(60) collect(1.0:60.0) randn(MersenneTwister(5), 60)]
        Y = X * [1.0 -2.0; 0.05 0.02; 0.4 0.9] .+ 0.2 .* randn(MersenneTwister(6), 60, 2)
        B, E, se, XtXi = T._mols(X, Y)
        for j in 1:2
            b, e, s = T._ols(X, Y[:, j])
            @test isapprox(B[:, j], b; atol=1e-12)          # the same answer...
            @test isapprox(E[:, j], e; atol=1e-12)
            # with shared regressors the per-equation sigma^2 IS Σ_df[j,j], so
            # the standard errors coincide exactly too
            @test isapprox(se[:, j], s; atol=1e-12)
        end
        @test isapprox(XtXi, inv(X' * X); rtol=1e-9)
        # ...from one factorization rather than K. Asserted, not claimed.
        Xb = [ones(400) randn(MersenneTwister(14), 400, 8)]
        Yb = randn(MersenneTwister(15), 400, 6)
        T._mols(Xb, Yb)
        for j in 1:6; T._ols(Xb, Yb[:, j]); end
        together = @allocated T._mols(Xb, Yb)
        separate = @allocated for j in 1:6
            T._ols(Xb, view(Yb, :, j))
        end
        @test together < separate
    end

    @testset "_mols: weights, and QR/Cholesky agreement" begin
        X = [ones(100) randn(MersenneTwister(9), 100, 3)]
        Y = X * randn(MersenneTwister(10), 4, 3) .+ randn(MersenneTwister(11), 100, 3)
        a = T._mols(X, Y; method=:qr)
        b = T._mols(X, Y; method=:cholesky)
        @test isapprox(a[1], b[1]; rtol=1e-8)
        @test isapprox(a[3], b[3]; rtol=1e-7)
        # the weighted reduction mirrors _ols's, equation by equation
        w = 0.5 .+ rand(MersenneTwister(12), 100)
        Bw, Ew, sew, _ = T._mols(X, Y; weights=w)
        for j in 1:3
            bw, ew, sw = T._ols(X, Y[:, j]; weights=w)
            @test isapprox(Bw[:, j], bw; atol=1e-11)
            @test isapprox(Ew[:, j], ew; atol=1e-11)    # residuals on the ORIGINAL scale
            @test isapprox(sew[:, j], sw; atol=1e-11)
        end
        @test !isapprox(Bw, a[1]; rtol=1e-6)            # weights did something
        @test_throws ArgumentError T._mols(X, Y; weights=ones(99))
        @test_throws ArgumentError T._mols(X, Y; weights=vcat(0.0, ones(99)))
    end

    @testset "REDUCTION: a k=1 VAR(p) with a constant is stats::ar.ols" begin
        # The charter's §5.9 reduction, and why the internal primitive must
        # handle k=1 even though the public fit_var refuses it.
        # R: ar.ols(z, order.max=p, aic=FALSE, demean=FALSE, intercept=TRUE)
        z = vec(_varfix("var1_k1.csv"))
        @test length(z) == 200
        R_AR = Dict(1 => [0.6484308401025832],
                    2 => [0.673140896685106, -0.03945717137380811],
                    3 => [0.6687833427540003, -0.059036329136739174, 0.028860901589552235])
        R_INT = Dict(1 => 0.32794476411707674, 2 => 0.33998905461259055,
                     3 => 0.3237308512151343)
        R_VARPRED = Dict(1 => 0.9538204266778918, 2 => 0.9560758286522902,
                         3 => 0.9343819545156791)
        for p in 1:3
            Z, X, cn = T.build_var_Z(reshape(z, :, 1), p, :const)
            B, E, se, _ = T._mols(Z, X)
            Tn = size(Z, 1)
            @test size(B) == (p + 1, 1)                 # k = 1 is supported
            @test isapprox(B[1:p, 1], R_AR[p]; atol=1e-11)
            @test isapprox(B[p+1, 1], R_INT[p]; atol=1e-11)   # const is LAST here
            @test cn[end] == :const
            # R's `var.pred` uses the ML divisor T, not the df-adjusted T - q.
            # The handoff's §3.5 lists the numbers without saying which divisor
            # produces them; measured, it is T, bit-identically.
            @test isapprox(dot(E, E) / Tn, R_VARPRED[p]; rtol=1e-12)
            @test !isapprox(dot(E, E) / (Tn - (p + 1)), R_VARPRED[p]; rtol=1e-4)
            if p == 1
                # and R's asy.se.coef uses that same ML divisor, so the
                # df-adjusted se here is larger by exactly sqrt(T/(T-q))
                f = sqrt((Tn - 2) / Tn)
                @test isapprox(se[1, 1] * f, 0.05410479122534204; rtol=1e-12)
                @test isapprox(se[2, 1] * f, 0.08644379310823685; rtol=1e-12)
            end
        end
    end

    # ------------------------------------------------------- rank deficiency

    @testset "rank deficiency is REFUSED (both references proceed silently)" begin
        # R: lm drops a column and reports NA; statsmodels' lstsq returns a
        # finite minimum-norm solution with no warning. Pinned from the
        # references' own output:
        @test occursin("no error", REF_R_PRIM["err_collinear"])
        @test occursin("NA", REF_R_PRIM["err_collinear"])
        @test occursin("no error", REF_PY_PRIM["err_collinear"])
        @test occursin("NaN in params: False", REF_PY_PRIM["err_collinear"])

        Yc = hcat(A2, A2[:, 1])                          # exact duplicate column
        Z, X, _ = T.build_var_Z(Yc, 1, :const)
        @test_throws ArgumentError T._mols(Z, X)
        e = try; T._mols(Z, X); nothing; catch err; err; end
        msg = lowercase(sprint(showerror, e))
        @test occursin("rank", msg) || occursin("collinear", msg)
        @test occursin("remove", msg)                    # names the fix
        @test_throws ArgumentError T._mols(Z, X; method=:cholesky)

        # and the regression it guards: fit_var used to accept this and
        # return coefficients of order 3.9e14 with loglik +2798.9
        ef = try; fit_var(Yc, 1); nothing; catch err; err; end
        @test ef isa ArgumentError
        @test occursin("rank deficient", ef.msg)
        @test_throws ArgumentError var_select(Yc; lag_max=4)

        # near-collinear but full rank must still work under :qr
        Yn = hcat(A2, A2[:, 1] .+ 1e-6 .* randn(MersenneTwister(1), size(A2, 1)))
        Zn, Xn, _ = T.build_var_Z(Yn, 1, :const)
        @test_nowarn T._mols(Zn, Xn)
        @test all(isfinite, T._mols(Zn, Xn)[1])
    end

    @testset "_mols refusals" begin
        X = [ones(10) randn(MersenneTwister(12), 10)]
        @test_throws ArgumentError T._mols(X, randn(10, 2); method=:bogus)
        @test_throws DimensionMismatch T._mols(X, randn(9, 2))
        @test_throws ArgumentError T._mols(randn(3, 5), randn(3, 2))   # T <= q
        @test_throws ArgumentError T._mols(X, zeros(10, 0))
    end

    # ----------------------------------------------------------- build_var_Z

    @testset "build_var_Z: layout and the trend origin" begin
        Z, X, cn = T.build_var_Z(A2, 2, :const)
        @test size(Z) == (198, 5)
        @test cn == [:y1_l1, :y2_l1, :y1_l2, :y2_l2, :const]       # det LAST
        # handoff §4.1: R and statsmodels build this identical lag block
        @test isapprox(Z[1, :], [3.918146471612, 2.169296248687,
                                 3.187181068090, 0.893489697008, 1.0]; atol=1e-11)
        @test isapprox(Z[end, :], [1.486798501843, -0.010927085209,
                                   1.073571556737, -0.350570210875, 1.0]; atol=1e-11)
        @test X == A2[3:end, :]
        @test Z[:, 1] == A2[2:end-1, 1]
        @test Z[:, 2] == A2[2:end-1, 2]
        @test Z[:, 3] == A2[1:end-2, 1]
        @test Z[:, 4] == A2[1:end-2, 2]

        # the trend is the CALENDAR index p+1:n, not 1:T
        Zb, _, cnb = T.build_var_Z(A2, 1, :both)
        @test cnb[end-1:end] == [:const, :trend]
        @test Zb[1, end] == 2.0                   # p + 1 = 2, not 1
        @test Zb[end, end] == 200.0               # n
        Zq, _, cnq = T.build_var_Z(A2, 1, :quadratic)
        @test cnq[end-2:end] == [:const, :trend, :trend2]
        @test Zq[1, end] == 4.0                   # 2^2, the raw square
        @test Zq[end, end] == 200.0^2
    end

    @testset "build_var_Z: only the LAG block may be checked against statsmodels' helper" begin
        # handoff §4.2: statsmodels' get_var_endog builds the trend as 1:T
        # while its own VAR.fit uses p+1:n, so the helper is a correct
        # reference for the lag block and a WRONG one for the trend. Checking
        # the trend against it would enshrine the bug.
        for (key, det, ndet) in (("Zlayout_n", :none, 0), ("Zlayout_c", :const, 1),
                                 ("Zlayout_ct", :both, 2), ("Zlayout_ctt", :quadratic, 3))
            Z, _, _ = T.build_var_Z(A2, 1, det)
            ref = _pm(REF_PY_PRIM[key]["head"])
            nh = size(ref, 1)
            # statsmodels puts deterministic columns first, so its lag block
            # starts at column ndet+1; this package's starts at column 1
            @test isapprox(Z[1:nh, 1:2], ref[:, (ndet+1):(ndet+2)]; atol=1e-11)
            @test size(Z, 1) == Int(REF_PY_PRIM[key]["shape"][1])
        end
        # and the helper's trend really is 1:T, which is the trap
        @test _pm(REF_PY_PRIM["Zlayout_ct"]["head"])[1, 2] == 1.0
        @test T.build_var_Z(A2, 1, :both)[1][1, end] == 2.0
    end

    @testset "build_var_Z: the quadratic origin, checked where it is identifiable" begin
        # The trend^2 coefficient is invariant to an origin shift, so it alone
        # cannot confirm the origin. The constant and trend coefficients are
        # not, and statsmodels' "ctt" fit pins all three.
        m = fit_var(A2, 1; deterministic=:quadratic)
        q = REF_PY["A_p1_ctt"]
        P = reduce(hcat, [Float64.(r) for r in q["params"]])'   # (ndet+Kp) x K, det first
        @test isapprox(m.det_coef[:, 1], P[1, :]; rtol=1e-8)    # const: origin-sensitive
        @test isapprox(m.det_coef[:, 2], P[2, :]; rtol=1e-8)    # trend: origin-sensitive
        @test isapprox(m.det_coef[:, 3], P[3, :]; rtol=1e-8)    # trend^2
        @test isapprox(m.det_coef[:, 3], [5.913177572457e-5, 2.879857740094e-5]; rtol=1e-8)
    end

    @testset "build_var_Z: all five deterministic specs, and its refusals" begin
        for (det, ndet) in ((:none, 0), (:const, 1), (:trend, 1), (:both, 2), (:quadratic, 3))
            Z, X, cn = T.build_var_Z(A2, 2, det)
            @test size(Z, 2) == 2 * 2 + ndet
            @test length(cn) == size(Z, 2)
            @test size(Z, 1) == size(X, 1) == 198
        end
        _, _, cnt = T.build_var_Z(A2, 1, :trend)
        @test !(:const in cnt)                    # :trend has NO constant
        @test :trend in cnt
        @test_throws ArgumentError T.build_var_Z(A2, 1, :bogus)
        @test_throws ArgumentError T.build_var_Z(A2, 0, :const)
        @test_throws ArgumentError T.build_var_Z(A2, 300, :const)
    end

    # ------------------------------------------------------------ companion

    @testset "companion: structure and eigenvalues against R" begin
        m = fit_var(A2, 2)
        C = companion(m)
        @test size(C) == (4, 4)
        @test isapprox(C[1, :], [0.448249746032, 0.070329548445,
                                 0.014232369339, -0.111386497811]; atol=1e-11)
        for (key, Y, p) in (("comp_A_p1", A2, 1), ("comp_A_p2", A2, 2),
                            ("comp_A_p4", A2, 4), ("comp_B_p2", B3, 2),
                            ("comp_US_p2", US, 2))
            mm = fit_var(Y, p)
            r = REF_R_PRIM[key]
            K = size(Y, 2)
            CC = companion(mm)
            @test size(CC) == (K * p, K * p)
            @test isapprox(CC, _pm(r["companion"]); atol=1e-9)
            mods = Float64.(r["eig_mod"])
            @test isapprox(sort(abs.(eigvals(CC)); rev=true), mods; atol=1e-9)
            @test isapprox(roots(mm), mods; atol=1e-9)
            @test isapprox(roots(mm), Float64.(r["vars_roots"]); atol=1e-9)
            # statsmodels agrees on the companion matrix itself
            @test isapprox(CC, _pm(REF_PY_PRIM[key]["companion"]); atol=1e-9)
        end
    end

    @testset "companion: the p = 1 subtlety, and the block invariants" begin
        # At p = 1 the companion matrix IS A[1] and there are no identity
        # blocks. A block assignment over the ranges 3:2 and 1:0 is an empty
        # assignment in Julia rather than an error, so a missing guard would
        # not announce itself -- hence the explicit shape assertion.
        m1 = fit_var(A2, 1)
        C1 = companion(m1)
        @test size(C1) == (2, 2)
        @test C1 == m1.A[1]
        @test companion(m1.A) == C1                # the bare-matrix method
        m = fit_var(B3, 2)
        C = companion(m)
        K = 3
        @test isapprox(C[1:K, 1:K], m.A[1]; atol=1e-14)
        @test isapprox(C[1:K, K+1:2K], m.A[2]; atol=1e-14)
        @test C[K+1:2K, 1:K] == I(K)
        @test all(iszero, C[K+1:2K, K+1:2K])
        # cached at fit time rather than rebuilt on every call
        @test companion(m) === m.C
        @test companion(m) == companion(m.A)
    end

    @testset "companion: eigenvalues are reciprocals of the characteristic roots" begin
        # charter §5.3 on the primitive, before any model inherits it
        A1 = [0.5 -0.1; 0.2 0.3]
        comp = sort(abs.(eigvals(companion([A1]))); rev=true)
        char_roots = abs.(1 ./ eigvals(A1))
        @test isapprox(sort(comp), sort(1 ./ char_roots); rtol=1e-12)
        @test all(comp .< 1)
        compx = abs.(eigvals(companion([[1.05 0.02; 0.0 1.03]])))
        @test maximum(compx) > 1
    end

    @testset "_companion refusals" begin
        @test_throws ArgumentError T._companion(Matrix{Float64}[])
        @test_throws DimensionMismatch T._companion([rand(2, 2), rand(3, 3)])
        @test_throws DimensionMismatch T._companion([rand(2, 3)])
    end

    # ------------------------------------------------------ tsmatrix/tsnames

    @testset "tsmatrix and tsnames" begin
        @test tsmatrix(A2) === A2                                 # zero copy
        @test tsmatrix((A2[:, 1], A2[:, 2])) == A2
        @test tsmatrix([A2[:, 1], A2[:, 2]]) == A2
        nt = (y1 = A2[:, 1], y2 = A2[:, 2])
        @test tsmatrix(nt) == A2
        @test tsnames(nt) == [:y1, :y2]
        @test tsnames(A2) === nothing
        @test tsnames((A2[:, 1], A2[:, 2])) === nothing
        @test eltype(tsmatrix((a = [1, 2], b = [3, 4]))) == Float64
        # ragged input is refused, not padded or truncated
        @test_throws ArgumentError tsmatrix((A2[:, 1], A2[1:end-1, 2]))
        @test_throws ArgumentError tsmatrix((y1 = A2[:, 1], y2 = A2[1:end-1, 2]))
        er = try; tsmatrix((A2[:, 1], A2[1:end-1, 2])); nothing; catch err; err; end
        @test occursin("unequal length", er.msg)
        @test occursin("nothing is padded", er.msg)
        # a single vector is NOT a multivariate container
        @test_throws ArgumentError tsmatrix(A2[:, 1])
        @test_throws ArgumentError tsmatrix(NamedTuple())
        e = try; tsmatrix("not a series"); nothing; catch err; err; end
        @test e isa ArgumentError
        @test occursin("NamedTuple of columns", e.msg)
        # and tsvalues keeps its existing behaviour untouched
        @test tsvalues(A2[:, 1]) == A2[:, 1]
    end

    @testset "tsmatrix accepts what dataset() returns for a multivariate entry" begin
        d = dataset("econ5")
        M = tsmatrix(d)
        @test size(M) == (161, 5)
        @test tsnames(d) == [:unemp, :gnp, :consum, :govinv, :prinv]
        @test M[:, 1] == d.unemp
        @test size(tsmatrix(dataset("climhyd"))) == (454, 6)
    end

    @testset "tsmatrix exists because tsvalues' contract is vector-shaped" begin
        # tsvalues documents AbstractVector{<:Real}, and its collect(Float64, x)
        # fallback on a matrix returns a MATRIX -- so it silently violates its
        # own stated return type rather than erroring. Pinned, so the
        # motivation cannot quietly stop being true.
        v = tsvalues([1.0 2.0; 3.0 4.0])
        @test v isa AbstractMatrix
        @test !(v isa AbstractVector)
    end

    # --------------------------------------------- type stability, allocation

    @testset "type stability and allocations" begin
        Z, X, _ = T.build_var_Z(US, 2, :const)
        @test (@inferred T.build_var_Z(US, 2, :const)) isa Tuple
        @test eltype(Z) === Float64 && eltype(X) === Float64
        B, E, se, ZtZinv = T._mols(Z, X)
        @test typeof(B) === Matrix{Float64}
        @test typeof(ZtZinv) === Matrix{Float64}
        m = fit_var(US, 2)
        @test (@inferred companion(m.A)) isa Matrix{Float64}
        companion(m.A)                                       # warm up
        @test (@allocated companion(m.A)) < 10 * 10 * 8 * 3
        # build_var_Z allocates Z and X and essentially nothing else
        T.build_var_Z(US, 2, :const)
        a = @allocated T.build_var_Z(US, 2, :const)
        @test a < 3 * (size(Z, 1) * size(Z, 2) + size(X, 1) * size(X, 2)) * 8
    end
end
