using DelimitedFiles

# Stage 10.6: the VECM.
#
# THREE agreeing references once the deterministic specification is matched:
# statsmodels 0.15.0 VECM (seven configurations: beta, alpha, gamma, Sigma,
# llf, all three standard-error blocks, var_rep, predict, residuals),
# urca::cajorls + vars::vec2var (five configurations), and tsDyn 11.0.5.2
# VECM for both estimators (ML and 2OLS). Two notes the tests encode:
# - cajorls' lag coefficients are ca.jo's spec = "longrun" parameterization,
#   Γ*_i = Γ_i + Π, not a disagreement (asserted as that identity);
# - tsDyn's 2OLS runs the cointegrating regression WITHOUT an intercept when
#   include = "const": the constant goes in the error-correction equation.

if !isdefined(@__MODULE__, :REF_R)
    include(joinpath(@__DIR__, "fixtures", "var", "refs.jl"))
end

_vfx(name) = readdlm(joinpath(@__DIR__, "fixtures", "var", name), ','; skipstart=1)
_vparse(x) = x isa AbstractString ? eval(Meta.parse(x)) : x
_vmat(x) = (v = _vparse(x); Float64[v[i][j] for i in eachindex(v), j in eachindex(v[1])])
const _VDET = Dict("n" => :none, "co" => :const, "ci" => :const_restricted)
const _VRFX = Dict("yes" => "coint2_yes.csv", "r1" => "coint3_r1.csv", "r2" => "coint3_r2.csv")

@testset "fit_vecm: every statsmodels configuration" begin
    for (key, v) in REF_PY_VECM
        (v isa Dict && haskey(v, "fx")) || continue
        k = parse(Int, string(v["k_ar_diff"])); r = parse(Int, string(v["coint_rank"]))
        m = fit_vecm(_vfx(v["fx"]); lags=k, rank=r, deterministic=_VDET[v["deterministic"]])
        K = length(m.names)
        @test m.beta[1:K, :] ≈ _vmat(v["beta"]) atol=1e-12
        @test m.alpha ≈ _vmat(v["alpha"]) atol=1e-11
        @test reduce(hcat, m.gamma) ≈ _vmat(v["gamma"]) atol=1e-11
        @test m.sigma_mle ≈ _vmat(v["sigma_u"]) atol=1e-12
        @test loglikelihood(m) ≈ _vparse(v["llf"]) atol=1e-9
        @test m.nobs == parse(Int, string(v["nobs"]))
        @test m.stderr_alpha ≈ _vmat(v["stderr_alpha"]) atol=1e-12
        @test m.stderr_gamma ≈ _vmat(v["stderr_gamma"]) atol=1e-12
        @test m.stderr_beta[1:K, :] ≈ _vmat(v["stderr_beta"]) atol=1e-12
        @test all(iszero, m.stderr_beta[1:r, :])           # the normalized block is not estimated
        rh = _vmat(v["resid_head"])
        @test residuals(m)[1:size(rh, 1), :] ≈ rh atol=1e-11
        vr = _vparse(v["var_rep"]); A = var_rep(m)
        @test length(A) == k + 1
        for j in eachindex(A)
            @test A[j] ≈ Float64[vr[j][a][b] for a in 1:K, b in 1:K] atol=1e-11
        end
        pr = _vmat(v["predict"])
        @test forecast(m, size(pr, 1)).point ≈ pr atol=1e-10
        @test m.rank == r && m.lags == k && m.method === :ml
    end
end

@testset "fit_vecm: urca/cajorls, vars::vec2var and tsDyn" begin
    for (key, v) in REF_R_VECM
        (v isa Dict && haskey(v, "cajorls_beta")) || continue
        Y = _vfx(_VRFX[split(key, "_")[1]])
        lags = parse(Int, string(v["K"])) - 1
        r = parse(Int, string(v["r"]))
        m = fit_vecm(Y; lags=lags, rank=r, deterministic=:const)
        K = length(m.names)
        @test m.beta ≈ _vmat(v["cajorls_beta"]) atol=1e-12
        cf = _vmat(v["cajorls_coef"])                 # rows: ect.., constant, x.dl1 …
        @test cf[1:r, :] ≈ transpose(m.alpha) atol=1e-11
        @test cf[r + 1, :] ≈ m.det_coef[:, 1] atol=1e-11
        # cajorls' lag block is ca.jo's spec = "longrun" form: Γ*_i = Γ_i + Π
        Pi = m.alpha * transpose(m.beta[1:K, :])
        for i in 1:lags
            @test transpose(cf[(r + 1 + (i - 1) * K + 1):(r + 1 + i * K), :]) ≈ m.gamma[i] + Pi atol=1e-11
        end
        @test m.sigma_df ≈ _vmat(v["cajorls_sigma"]) atol=1e-12          # df-adjusted divisor
        A = _vparse(v["vec2var_A"])
        for j in 1:(lags + 1)
            @test var_rep(m)[j] ≈ _vmat(A["A$j"]) atol=1e-11
        end
        @test m.det_coef[:, 1] ≈ vec(_vmat(v["vec2var_deterministic"])) atol=1e-11
        rn = String.(_vparse(v["names"]))
        fc = _vparse(v["vec2var_fcst"]); f = forecast(m, 4)
        for i in 1:K, s in 1:4
            @test f.point[s, i] ≈ fc[rn[i]][s][1] atol=1e-10
        end
        # tsDyn: ML always; 2OLS at rank 1 (it switches to ML silently above that)
        ml = v["tsdyn_ML"]
        @test hcat(m.alpha, m.det_coef, reduce(hcat, m.gamma)) ≈ _vmat(ml["coef"]) atol=1e-11
        @test normalize_beta(m, r; scheme=:identity) ≈ _vmat(ml["beta"]) atol=1e-12
        if r == 1
            m2 = fit_vecm(Y; lags=lags, rank=1, deterministic=:const, method=:twostep)
            ts = v["tsdyn_2OLS"]
            @test m2.beta ≈ _vmat(ts["beta"]) atol=1e-12
            @test hcat(m2.alpha, m2.det_coef, reduce(hcat, m2.gamma)) ≈ _vmat(ts["coef"]) atol=1e-11
        else
            # tsDyn's "2OLS" entry here is ML in disguise -- it switched estimator
            @test _vmat(v["tsdyn_2OLS"]["coef"]) ≈ _vmat(ml["coef"]) atol=1e-12
        end
    end
end

@testset "the handoff's values, explicitly" begin
    for (fx, k, beta, alpha) in [("coint3_r1.csv", 1, [1.0, -0.496504585, -0.9777300348],
                                                      [0.0369289001, -0.23427842, 0.9607618645]),
                                 ("coint2_yes.csv", 1, [1.0, -0.6673249283], [-0.1220914208, 0.6066110024]),
                                 ("coint2_yes.csv", 2, [1.0, -0.6655228742], [-0.1881544488, 0.4920002608])]
        m = fit_vecm(_vfx(fx); lags=k, rank=1)
        @test vec(normalize_beta(m, 1)) ≈ beta atol=1e-9
        @test vec(m.alpha) ≈ alpha atol=1e-8           # the handoff prints 8 decimals
    end
    m = fit_vecm(_vfx("coint3_r1.csv"); lags=1, rank=1)
    @test vec(m.det_coef) ≈ [0.0395842162, 0.0545899489, 0.0262256932] atol=1e-9
    @test length(m.gamma) == 1
    @test minimum(eigvals(Hermitian(m.sigma_df))) > 0
end

@testset "rank 2, and parameter recovery" begin
    # coint3_r2's DGP: d1 = tr + e, d2 = 2tr + e, d3 = -tr + e, so the
    # relations are d1 + d3 and d2 + 2 d3
    m = fit_vecm(_vfx("coint3_r2.csv"); lags=1, rank=2)
    B = normalize_beta(m, 2; scheme=:identity)
    @test B[1:2, 1:2] ≈ I(2) atol=1e-12
    @test B[3, :] ≈ [1.0059150205, 1.9908711456] atol=1e-8      # the references
    @test B[3, :] ≈ [1.0, 2.0] atol=0.02                        # and the TRUTH
    # over-specified rank: the references agree on a second vector the data
    # do not identify. The numbers are reproducible; their meaning is not.
    mo = fit_vecm(_vfx("coint3_r1.csv"); lags=1, rank=2)
    @test normalize_beta(mo, 2; scheme=:identity)[3, :] ≈ [-1.3698269062, -0.7897145025] atol=1e-8
    @test all(isfinite, mo.stderr_beta)
end

@testset "johansen_test and fit_vecm agree on beta for the same deterministic" begin
    for (fx, k, r) in [("coint2_yes.csv", 1, 1), ("coint3_r1.csv", 1, 1),
                       ("coint3_r2.csv", 1, 2), ("coint2_yes.csv", 2, 1)]
        Y = _vfx(fx)
        for det in (:none, :const, :const_restricted, :trend, :trend_restricted)
            j = johansen_test(Y; lags=k, deterministic=det)
            m = fit_vecm(Y; lags=k, rank=r, deterministic=det)
            @test normalize_beta(j, r; scheme=:identity) ≈ normalize_beta(m, r; scheme=:identity) atol=1e-10
            if r == 1
                @test normalize_beta(j, 1; scheme=:first_unit) ≈ normalize_beta(m, 1; scheme=:first_unit) atol=1e-10
            else
                # first-unit at rank > 1 belongs to the eigenvector basis: refused on a VECM
                @test_throws ArgumentError normalize_beta(m, r; scheme=:first_unit)
                @test normalize_beta(m, 1; scheme=:first_unit)[1] == 1.0
            end
            @test size(m.beta, 1) == size(j.beta, 1)
        end
    end
end

@testset "deterministic: :const is 'co', :const_restricted is 'ci'" begin
    Y = _vfx("coint2_yes.csv")
    n  = fit_vecm(Y; lags=1, rank=1, deterministic=:none)
    co = fit_vecm(Y; lags=1, rank=1, deterministic=:const)
    ci = fit_vecm(Y; lags=1, rank=1, deterministic=:const_restricted)
    @test normalize_beta(n, 1)[2] ≈ -0.5761832355 atol=1e-9
    @test normalize_beta(co, 1)[2] ≈ -0.6673249283 atol=1e-9
    @test normalize_beta(ci, 1)[2] ≈ -0.6673216872 atol=1e-9
    @test loglikelihood(n)  ≈ -662.848601 atol=1e-5
    @test loglikelihood(co) ≈ -642.403440 atol=1e-5
    @test loglikelihood(ci) ≈ -642.425077 atol=1e-5
    @test normalize_beta(co, 1)[2] != normalize_beta(ci, 1)[2]
    @test abs(loglikelihood(co) - loglikelihood(ci)) > 0.01
    @test size(ci.beta) == (3, 1) && size(co.beta) == (2, 1)
    @test size(co.det_coef) == (2, 1) && size(ci.det_coef) == (2, 0) && size(n.det_coef) == (2, 0)
    # the restricted constant enters the levels VAR's intercept as alpha * beta_det
    _, dc = TSAnalytics._vecm_var_det(ci)
    @test dc ≈ ci.alpha * ci.beta[3, 1] atol=1e-14
    # trends: both forms fit, and differ
    t  = fit_vecm(Y; lags=1, rank=1, deterministic=:trend)
    tr = fit_vecm(Y; lags=1, rank=1, deterministic=:trend_restricted)
    @test size(t.det_coef) == (2, 2) && size(tr.det_coef) == (2, 1)
    @test !isapprox(loglikelihood(t), loglikelihood(tr); atol=1e-3)
end

@testset "method = :twostep is a different estimator, and rank 1 only" begin
    m1 = fit_vecm(_vfx("coint3_r1.csv"); lags=1, rank=1, method=:ml)
    m2 = fit_vecm(_vfx("coint3_r1.csv"); lags=1, rank=1, method=:twostep)
    @test vec(m2.beta) ≈ [1.0, -0.4930964735, -0.9654865132] atol=1e-9
    @test !isapprox(m1.beta, m2.beta; atol=1e-3)
    @test m2.method === :twostep
    @test m2.stderr_beta === nothing                      # no valid OLS SE for a static I(1) regression
    my = fit_vecm(_vfx("coint2_yes.csv"); lags=1, rank=1, method=:twostep)
    @test vec(my.beta) ≈ [1.0, -0.5738885256] atol=1e-9
    # step one is the static regression of the first series on the others,
    # no intercept (the constant is unrestricted, so it is not in the relation)
    Y = Float64.(_vfx("coint2_yes.csv"))
    @test my.beta[2] ≈ -(Y[:, 2:2] \ Y[:, 1])[1] atol=1e-12
    # a restricted constant goes into the static regression instead
    mc = fit_vecm(Y; lags=1, rank=1, deterministic=:const_restricted, method=:twostep)
    b = hcat(Y[:, 2], ones(250)) \ Y[:, 1]
    @test mc.beta ≈ [1.0, -b[1], -b[2]] atol=1e-12
    @test_throws ArgumentError fit_vecm(_vfx("coint3_r2.csv"); lags=1, rank=2, method=:twostep)
    e = try fit_vecm(_vfx("coint3_r2.csv"); lags=1, rank=2, method=:twostep) catch err; err end
    @test occursin("rank", lowercase(sprint(showerror, e)))
    @test occursin("ml", lowercase(sprint(showerror, e)))
end

@testset "standard errors: by hand" begin
    m = fit_vecm(_vfx("coint3_r1.csv"); lags=1, rank=1)
    W, X0, X1, T, _ = TSAnalytics._vecm_design(Float64.(_vfx("coint3_r1.csv")), 1, :const)
    Z = hcat(X1 * m.beta, W)
    S = residuals(m)' * residuals(m) / T
    se = sqrt.(diag(inv(Z' * Z)) * diag(S)')            # q x K, ML Sigma
    @test transpose(se[1:1, :]) ≈ m.stderr_alpha rtol=1e-12
    @test transpose(se[2:4, :]) ≈ m.stderr_gamma rtol=1e-12
    @test transpose(se[5:5, :]) ≈ m.stderr_det rtol=1e-12
    # beta: Lütkepohl Prop. 7.6, the levels partialled on W
    X1p = X1 - W * (W \ X1)
    cb = kron(inv(X1p[:, 2:3]' * X1p[:, 2:3]), inv(m.alpha' * inv(S) * m.alpha))
    @test m.stderr_beta[2:3, 1] ≈ sqrt.(diag(cb)) rtol=1e-12
end

@testset "var_rep: the VECM is a levels VAR, and its residuals must match" begin
    for (fx, k, r, det) in [("coint3_r1.csv", 1, 1, :const), ("coint2_yes.csv", 2, 1, :const_restricted),
                            ("coint3_r2.csv", 1, 2, :trend), ("coint2_yes.csv", 3, 1, :trend_restricted),
                            ("coint3_r1.csv", 2, 1, :none)]
        Y = Float64.(_vfx(fx))
        m = fit_vecm(Y; lags=k, rank=r, deterministic=det)
        A = var_rep(m)
        K = length(m.names)
        @test length(A) == k + 1
        @test A[1] ≈ I(K) + m.alpha * m.beta[1:K, :]' + m.gamma[1] atol=1e-12
        @test A[end] ≈ -m.gamma[end] atol=1e-14
        # the levels VAR reproduces the VECM's own residuals
        vd, dc = TSAnalytics._vecm_var_det(m)
        n = size(Y, 1); p = k + 1
        E = zeros(n - p, K)
        for t in (p + 1):n
            yh = zeros(K)
            vd !== :none && (yh .+= dc[:, 1])
            vd === :both && (yh .+= dc[:, 2] .* t)
            for j in 1:p; yh .+= A[j] * Y[t - j, :]; end
            E[t - p, :] = Y[t, :] - yh
        end
        @test E ≈ residuals(m) atol=1e-9
        # a VECM has K - r unit roots BY CONSTRUCTION
        rts = roots(m)
        @test count(x -> abs(x - 1) < 1e-8, rts) == K - r
        @test !is_stable(m)
        @test_throws ArgumentError lr_effects(m)
        @test_throws ArgumentError lr_effect_stderr(m)
    end
    e = try lr_effects(fit_vecm(_vfx("coint3_r1.csv"); lags=1, rank=1)) catch err; err end
    @test occursin("unit root", sprint(showerror, e))
end

@testset "forecast, irf, fevd delegate to the VAR recursions" begin
    m = fit_vecm(_vfx("coint3_r1.csv"); lags=1, rank=1)
    V = TSAnalytics._vecm_as_var(m)
    f = forecast(m, 6)
    @test f.point == forecast(V, 6).point
    @test predict(m, 6).point == f.point
    @test typeof(predict(m, 6)) === typeof(f) === VARForecast
    @test occursin("VECM", f.model_name)
    @test all(diff(f.se[:, 1]) .> 0)                    # I(1) levels: intervals keep widening
    r = irf(m, 20)
    @test r.orth == irf(V, 20).orth
    # responses converge to a permanent effect rather than dying out
    @test maximum(abs.(r.nonorth[21] - r.nonorth[20])) < 1e-4
    @test maximum(abs.(r.nonorth[21])) > 0.1
    @test fevd(m, 6).decomp == fevd(V, 6).decomp
    @test_throws ArgumentError irf(m, 4; stderr=true)
    @test_throws ArgumentError irf(m, 4; boot=true)
end

@testset "refusals: statsmodels accepts four nonsense configurations" begin
    Y = _vfx("coint2_yes.csv")
    @test occursin("beta=[]", REF_PY_VECM["err_r0"])
    @test occursin("no error", REF_PY_VECM["err_rK"])
    @test occursin("no error", REF_PY_VECM["err_k0"])
    @test occursin("in the interval of 1 to 1", REF_R_VECM["err_r0"])
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=0)
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=2)
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=3)
    @test_throws ArgumentError fit_vecm(Y; lags=0, rank=1)
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=1, deterministic=:bogus)
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=1, deterministic=:both)
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=1, method=:bogus)
    @test_throws ArgumentError fit_vecm(Y[:, 1:1]; lags=1, rank=1)
    @test_throws ArgumentError fit_vecm(Y[:, 1]; lags=1, rank=1)
    Yn = copy(Y); Yn[11, 1] = NaN
    @test_throws ArgumentError fit_vecm(Yn; lags=1, rank=1)
    e = try fit_vecm(Y; lags=1, rank=2) catch err; err end
    @test occursin("rank", lowercase(sprint(showerror, e)))
    @test occursin("1:1", sprint(showerror, e))
    @test_throws UndefKeywordError fit_vecm(Y; lags=1)
    @test_throws ArgumentError fit_vecm(Y[1:5, :]; lags=1, rank=1)
end

@testset "StatsAPI contract, names, show, type stability" begin
    Y = _vfx("coint3_r1.csv")
    m = fit_vecm((f=Y[:, 1], g=Y[:, 2], h=Y[:, 3]); lags=1, rank=1)
    @test m isa TimeSeriesModel
    @test m.names == [:f, :g, :h]
    @test size(coef(m)) == (3, 1 + 3 + 1)
    @test coefnames(m) == [:ect1, :df_l1, :dg_l1, :dh_l1, :const]
    @test size(residuals(m)) == (m.nobs, 3)
    @test nobs(m) == m.nobs == 248
    @test loglikelihood(m) isa Float64
    @test dof(m) == 3 * 1 + 1 * 2 + 9 + 3
    @test aic(m) ≈ -2loglikelihood(m) + 2dof(m)
    @test bic(m) ≈ -2loglikelihood(m) + log(248) * dof(m)
    @test isconcretetype(VECMModel)
    @test (@inferred fit_vecm(_vfx("coint2_yes.csv"); lags=1, rank=1)) isa VECMModel
    s = sprint(show, m)
    @test occursin("VECM", s) && occursin("rank=1", s)
    @test occursin("f, g, h", s)
    @test occursin("unrestricted constant", s)
end

@testset "performance" begin
    E = _vfx("econ5.csv")
    fit_vecm(E; lags=1, rank=1)
    @test (@allocated fit_vecm(E; lags=1, rank=1)) < 600_000
    fit_vecm(E; lags=4, rank=1)
    t1 = minimum(@elapsed(for _ in 1:30; fit_vecm(E; lags=1, rank=1); end) for _ in 1:3)
    t4 = minimum(@elapsed(for _ in 1:30; fit_vecm(E; lags=4, rank=1); end) for _ in 1:3)
    @test t4 < 8 * t1
end
