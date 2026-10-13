using DelimitedFiles

# Stage 10.0: the primitives Stage 10 is built on — multi-response least
# squares, `tsmatrix`/`tsnames`, and the companion matrix.
#
# A new roadmap row rather than something smuggled into `fit_var`: these are
# Stage 1-level primitives arriving late, and writing them separately is what
# keeps a loop of single-response `_ols` calls from being baked into every
# multivariate model.
#
# The headline verification is the k=1 reduction. A one-variable VAR(p) with a
# constant IS `stats::ar.ols(demean=FALSE, intercept=TRUE)`, so the primitive
# can be checked against R before any VAR code exists. Reference values from
# R 4.6.0, regenerated in test/verification/var/.

const VARFIX = joinpath(@__DIR__, "fixtures", "var")
_varfix(name) = readdlm(joinpath(VARFIX, name), ','; skipstart=1)

@testset "Stage 10.0 — multivariate primitives" begin
    T = TSAnalytics

    @testset "tsmatrix — the multivariate container accessor" begin
        M = [1.0 4.0; 2.0 5.0; 3.0 6.0]
        @test tsmatrix(M) === M                     # zero copy for a real matrix
        @test tsmatrix((a = [1.0, 2.0], b = [3.0, 4.0])) == [1.0 3.0; 2.0 4.0]
        @test tsmatrix([[1.0, 2.0], [3.0, 4.0]]) == [1.0 3.0; 2.0 4.0]
        @test tsmatrix(([1.0, 2.0], [3.0, 4.0])) == [1.0 3.0; 2.0 4.0]
        # integer input is widened, not rejected
        @test tsmatrix((a = [1, 2], b = [3, 4])) == [1.0 3.0; 2.0 4.0]
        @test eltype(tsmatrix((a = [1, 2], b = [3, 4]))) == Float64
        # rows are time, columns are variables -- the orientation both
        # references use and every bundled multivariate CSV carries
        @test size(tsmatrix((a = zeros(10), b = zeros(10)))) == (10, 2)
    end

    @testset "tsmatrix accepts what dataset() returns for a multivariate entry" begin
        # the point of the NamedTuple method: no unpacking at the call site
        d = dataset("econ5")
        M = tsmatrix(d)
        @test size(M) == (161, 5)
        @test tsnames(d) == [:unemp, :gnp, :consum, :govinv, :prinv]
        @test M[:, 1] == d.unemp
        @test M[:, 5] == d.prinv
        @test size(tsmatrix(dataset("climhyd"))) == (454, 6)
    end

    @testset "tsmatrix exists because tsvalues' contract is vector-shaped" begin
        # `tsvalues` documents `AbstractVector{<:Real}`, and its fallback is
        # `collect(Float64, x)` -- which on a matrix returns a MATRIX. So it
        # silently violates its own stated return type today rather than
        # erroring. That is the concrete reason for a separate name, and it is
        # asserted here so the motivation cannot quietly stop being true.
        A = [1.0 2.0; 3.0 4.0]
        v = tsvalues(A)
        @test v isa AbstractMatrix              # NOT the documented vector
        @test !(v isa AbstractVector)
        # tsmatrix is the honest route for the same input
        @test tsmatrix(A) === A
    end

    @testset "tsnames" begin
        @test tsnames((x = [1.0], y = [2.0])) == [:x, :y]
        @test tsnames(rand(3, 2)) === nothing   # a bare matrix carries none
        @test tsnames([[1.0], [2.0]]) === nothing
        @test tsnames(1:5) === nothing
    end

    @testset "tsmatrix refusals" begin
        @test_throws DimensionMismatch tsmatrix((a = [1.0, 2.0], b = [3.0]))
        @test_throws DimensionMismatch tsmatrix([[1.0, 2.0], [3.0]])
        @test_throws ArgumentError tsmatrix(NamedTuple())
        @test_throws ArgumentError tsmatrix(Vector{Float64}[])
        e = try; tsmatrix("not a series"); nothing; catch err; err; end
        @test e isa ArgumentError
        @test occursin("NamedTuple of columns", e.msg)   # names the fix
    end

    @testset "multi-response _ols agrees with a loop of the single-response one" begin
        # same answer, one factorization instead of K
        X = [ones(60) collect(1.0:60.0) randn(MersenneTwister(5), 60)]
        Btrue = [1.0 -2.0; 0.05 0.02; 0.4 0.9]
        Y = X * Btrue .+ 0.2 .* randn(MersenneTwister(6), 60, 2)
        B, E, se, XtXi = T._ols(X, Y)
        @test size(B) == (3, 2)
        @test size(E) == (60, 2)
        @test size(se) == (3, 2)
        @test size(XtXi) == (3, 3)
        for j in 1:2
            b, e, s = T._ols(X, Y[:, j])
            @test isapprox(B[:, j], b; rtol=1e-12)
            @test isapprox(E[:, j], e; rtol=1e-12)
            # with shared regressors the per-equation sigma^2 IS Sigma_df[j,j],
            # so the standard errors coincide exactly too
            @test isapprox(se[:, j], s; rtol=1e-12)
        end
        @test isapprox(E, Y - X * B; rtol=1e-14)
        @test isapprox(XtXi, inv(X' * X); rtol=1e-9)
        @test issymmetric(round.(XtXi; digits=12))
    end

    @testset "se follows from kron(Sigma, (X'X)^-1), not a per-equation refit" begin
        X = [ones(80) randn(MersenneTwister(7), 80, 2)]
        Y = X * [0.5 1.5; 1.0 -1.0; -0.5 0.25] .+ randn(MersenneTwister(8), 80, 2)
        B, E, se, XtXi = T._ols(X, Y)
        k, K = size(B)
        sigma = (E' * E) ./ (size(X, 1) - k)
        # the full covariance of vec(B) is the Kronecker product; se is its
        # diagonal, reshaped. 10.1 stores the whole thing because the analytic
        # IRF standard errors consume it whole.
        V = kron(sigma, XtXi)
        @test size(V) == (k * K, k * K)
        for j in 1:K, i in 1:k
            @test isapprox(se[i, j], sqrt(V[(j-1)*k + i, (j-1)*k + i]); rtol=1e-12)
            @test isapprox(se[i, j], sqrt(XtXi[i, i] * sigma[j, j]); rtol=1e-12)
        end
    end

    @testset "REDUCTION: a k=1 VAR(p) with a constant is stats::ar.ols" begin
        # The charter's §5.9 reduction, and the reason the internal primitive
        # must handle k=1 even though the public fit_var refuses it.
        # R: ar.ols(z, order.max=p, aic=FALSE, demean=FALSE, intercept=TRUE)
        z = vec(_varfix("var1_k1.csv"))
        n = length(z)
        @test n == 200

        R_AR = Dict(1 => [0.6484308401025832],
                    2 => [0.6731408966851060, -0.0394571713738081])
        R_INT = Dict(1 => 0.3279447641170767, 2 => 0.3399890546125905)
        R_VARPRED = Dict(1 => 0.9538204266778918, 2 => 0.9560758286522902)
        # R's asy.se.coef is ordered [intercept, ar1, ar2, ...]
        R_SE = Dict(1 => [0.0864437931082369, 0.0541047912253420],
                    2 => [0.0900277123064169, 0.0710201209648068, 0.0710881059842270])

        for p in 1:2
            Tn = n - p
            Z = Matrix{Float64}(undef, Tn, p + 1)
            for j in 1:p, t in 1:Tn
                Z[t, j] = z[p + t - j]
            end
            Z[:, end] .= 1.0
            Y = reshape(z[(p+1):n], Tn, 1)
            B, E, se, _ = T._ols(Z, Y)
            @test size(B) == (p + 1, 1)                  # k = 1 is supported
            @test isapprox(B[1:p, 1], R_AR[p]; atol=1e-14)
            @test isapprox(B[end, 1], R_INT[p]; atol=1e-14)

            # R's `var.pred` and `asy.se.coef` use the ML divisor T, where
            # `_ols` uses the df-adjusted T - k. Measured, not assumed: the
            # two are related by exactly sqrt((T - k) / T), and the charter's
            # §5.9 numbers are the ML ones.
            sse = sum(abs2, E)
            @test isapprox(sse / Tn, R_VARPRED[p]; atol=1e-14)
            f = sqrt((Tn - p - 1) / Tn)
            mine_in_R_order = vcat(se[end, 1], se[1:p, 1]) .* f
            @test isapprox(mine_in_R_order, R_SE[p]; atol=1e-14)
            # and the df-adjusted figure is the LARGER one, by that factor
            @test se[1, 1] > R_SE[p][2]
            @test isapprox(sse / (Tn - p - 1) / (sse / Tn), 1 / f^2; rtol=1e-12)
        end
    end

    @testset "_ols QR and Cholesky agree where both are well conditioned" begin
        X = [ones(100) randn(MersenneTwister(9), 100, 3)]
        Y = X * randn(MersenneTwister(10), 4, 3) .+ randn(MersenneTwister(11), 100, 3)
        q = T._ols(X, Y; method=:qr)
        c = T._ols(X, Y; method=:cholesky)
        @test isapprox(q[1], c[1]; rtol=1e-9)
        @test isapprox(q[3], c[3]; rtol=1e-9)
        @test isapprox(q[4], c[4]; rtol=1e-7)
    end

    @testset "_ols refusals, multi-response" begin
        X = [ones(10) randn(MersenneTwister(12), 10)]
        @test_throws ArgumentError T._ols(X, randn(10, 2); method=:bogus)
        @test_throws DimensionMismatch T._ols(X, randn(9, 2))
        # fewer observations than regressors per equation
        @test_throws ArgumentError T._ols(randn(3, 5), randn(3, 2))
        # collinear regressors under Cholesky: refused by name, not a raw
        # PosDefException
        Xc = [ones(20) ones(20)]
        e = try; T._ols(Xc, randn(20, 2); method=:cholesky); nothing; catch err; err; end
        @test e isa ArgumentError
        @test occursin(":qr", e.msg)
    end

    @testset "_companion structure" begin
        A1 = [0.5 -0.1; 0.2 0.3]
        C1 = T._companion([A1])
        @test C1 == A1                                   # p = 1 is just A1
        A2 = [0.1 0.0; 0.0 0.2]
        C2 = T._companion([A1, A2])
        @test size(C2) == (4, 4)
        @test C2[1:2, 1:2] == A1
        @test C2[1:2, 3:4] == A2
        @test C2[3:4, 1:2] == I(2)                       # the shift block
        @test all(iszero, C2[3:4, 3:4])
        # three lags, three variables
        A = [rand(MersenneTwister(13 + j), 3, 3) .* 0.1 for j in 1:3]
        C3 = T._companion(A)
        @test size(C3) == (9, 9)
        @test C3[4:6, 1:3] == I(3)
        @test C3[7:9, 4:6] == I(3)
        @test all(iszero, C3[4:9, 7:9])
    end

    @testset "_companion eigenvalues are the reciprocals of the characteristic roots" begin
        # Charter §5.3, the stage's most dangerous convention trap, checked
        # here on the primitive before any model can inherit it. For p = 1,
        # det(I - A z) = 0 has roots 1/lambda for each eigenvalue lambda of A.
        A1 = [0.5 -0.1; 0.2 0.3]
        comp = sort(abs.(eigvals(T._companion([A1]))); rev=true)
        char_roots = sort(abs.(1 ./ eigvals(A1)); rev=true)
        @test isapprox(comp, sort(1 ./ char_roots; rev=true); rtol=1e-12)
        @test all(comp .< 1)                             # this A is stable
        # a deliberately explosive one: the companion modulus crosses 1 while
        # the characteristic root crosses it the other way
        Aex = [1.05 0.02; 0.0 1.03]
        compx = sort(abs.(eigvals(T._companion([Aex]))); rev=true)
        @test maximum(compx) > 1
        @test minimum(abs.(1 ./ eigvals(Aex))) < 1
    end

    @testset "_companion refusals" begin
        @test_throws ArgumentError T._companion(Matrix{Float64}[])
        @test_throws DimensionMismatch T._companion([rand(2, 2), rand(3, 3)])
        @test_throws DimensionMismatch T._companion([rand(2, 3)])
    end

    @testset "the primitive is one factorization, not K" begin
        # The performance argument for the item, asserted rather than claimed:
        # solving K equations together allocates far less than K separate
        # single-response solves on the same regressors.
        X = [ones(400) randn(MersenneTwister(14), 400, 8)]
        Y = randn(MersenneTwister(15), 400, 6)
        T._ols(X, Y)                                     # warm up
        for j in 1:6; T._ols(X, Y[:, j]); end
        together = @allocated T._ols(X, Y)
        separate = @allocated for j in 1:6
            T._ols(X, view(Y, :, j))
        end
        @test together < separate
    end
end
