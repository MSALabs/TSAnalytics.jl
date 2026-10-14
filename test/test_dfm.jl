using DelimitedFiles
using LinearAlgebra: Diagonal, svd, det

# Stage 11.2 -- dynamic factor models. handoff/stage 11/docs/
# stage-11.2-dynamic-factor-handoff.md §6. Primary reference statsmodels
# 0.15.0 DynamicFactor (BSD, transcribable; full-precision values from
# handoff/stage 11/ground-truth/sm_results.json). dfms (GPL) is a check only.

const _DFMFX = joinpath(@__DIR__, "..", "handoff", "stage 11", "fixtures")
_dfmfx(f) = Float64.(readdlm(joinpath(_DFMFX, f), ','; skipstart=1))
# both likelihood targets are for data standardised with ddof = 1
const _YA = standardize(_dfmfx("fixtureA.csv"))
const _YB = standardize(_dfmfx("fixtureB.csv"))

const _SM_A = (lam=[0.7009068155913395, 0.6042910774398366, -0.5753739335367198,
                    0.5237349698498732, 0.6650726532699288],
               h=[0.14812824410505457, 0.36549749235533713, 0.42432314355129,
                  0.5221441574104723, 0.23249862054434853],
               A=0.6482477536949357, llf=-1049.5334081297972,
               aic=2121.0668162595944, bic=2157.348307291623,
               r2=[0.9108626308095913, 0.6770641675569274, 0.6137905314487733,
                   0.5085776162194486, 0.8201144425650966])
const _SM_B = (lam=[0.5533928213458822 0.4309722523368841; 0.6757507498964977 0.12316763253210221;
                    -0.021867285934118467 -0.6705529300459949; 0.03628674218233418 0.587948019498846;
                    0.6662154932967631 0.29179316358890744; 0.5238410868614163 -0.5266630210571038],
               A=[0.5309760294445562 -0.03867147247037333; 0.20951816279826058 0.6166771974292626],
               h=[0.15894097178603864, 0.30764156470260096, 0.1856257988091492, 0.36829583094348023,
                  0.1629034340031904, 0.21139302091552234],
               llf=-1514.940610472312)

# Orthogonal Procrustes, reflections allowed: the G minimising |X G - Y|.
# Fixture B's recovered rotation has det = -1, so a helper that assumes a
# proper rotation fails on exactly this fixture.
function _procrustes(X, Y)
    F = svd(X' * Y)
    return F.U * F.Vt
end

@testset "DFM: fixture A likelihood at fixed parameters (tier 1)" begin
    # statsmodels -1049.5334081297972; dfms' own C++ filter -1049.53340812981
    m = fit_dfm(_YA, 1; params=(reshape(_SM_A.lam, 5, 1), fill(_SM_A.A, 1, 1), _SM_A.h))
    @test loglikelihood(m) ≈ _SM_A.llf atol=1e-10
    @test loglikelihood(m) ≈ -1049.53340812981 atol=1e-9
    @test !m.estimated
    # the likelihood is for standardised data: DynamicFactor is not
    # scale-invariant (raw -1180.14 against standardised -1049.53)
    raw = fit_dfm(_dfmfx("fixtureA.csv"), 1; params=(reshape(_SM_A.lam, 5, 1), fill(_SM_A.A, 1, 1), _SM_A.h))
    @test abs(loglikelihood(raw) - loglikelihood(m)) > 10
end

@testset "DFM: fixture B likelihood at fixed parameters" begin
    m = fit_dfm(_YB, 2; params=(_SM_B.lam, _SM_B.A, _SM_B.h))
    @test loglikelihood(m) ≈ _SM_B.llf atol=1e-9
    @test size(factors(m)[1]) == (250, 2)               # observations x factors
end

@testset "DFM: fixture A fitted optimum against statsmodels" begin
    m = fit_dfm(_YA, 1)
    @test m.converged && m.estimated && m.normalisation === :lower
    # r = 1: sign is the only indeterminacy and the positive-diagonal
    # normalisation fixes it the way statsmodels' optimum landed
    @test vec(m.Lambda) ≈ _SM_A.lam rtol=1e-4
    @test m.H ≈ _SM_A.h rtol=1e-4
    @test m.A[1][1] ≈ _SM_A.A rtol=1e-4
    @test loglikelihood(m) ≈ _SM_A.llf atol=1e-6
    @test loglikelihood(m) >= _SM_A.llf - 1e-6
    @test aic(m) ≈ _SM_A.aic atol=1e-5
    @test bic(m) ≈ _SM_A.bic atol=1e-5
    @test dof(m) == 11 && nobs(m) == 200             # diagonal H: N r + p r^2 + N
    # statsmodels' coefficients_of_determination
    @test vec(factor_r2(m)) ≈ _SM_A.r2 rtol=1e-4
    # and they track the true loadings' magnitude [1.0, 0.8, -0.6, 0.5, 0.9]
    @test sortperm(vec(factor_r2(m)); rev=true) == [1, 5, 2, 3, 4]
    @test all(0 .< series_r2(m) .< 1)
end

@testset "DFM: fixture B -- the identification, and alignment to statsmodels" begin
    m = fit_dfm(_YB, 2)
    @test m.converged
    @test m.Lambda[1, 2] == 0 && m.Lambda[1, 1] > 0 && m.Lambda[2, 2] > 0
    # one parameter fewer than DynamicFactor (r(r-1)/2 = 1), the same maximum
    @test dof(m) == 21
    @test loglikelihood(m) >= _SM_B.llf - 1e-6
    @test loglikelihood(m) ≈ _SM_B.llf atol=1e-5
    # statsmodels' loadings, rotated onto ours by Procrustes: a reflection
    G = _procrustes(_SM_B.lam, m.Lambda)
    @test det(G) ≈ -1 atol=1e-8
    @test _SM_B.lam * G ≈ m.Lambda rtol=1e-3
    @test G' * _SM_B.A * G ≈ m.A[1] rtol=1e-3
    @test sort(m.H) ≈ sort(_SM_B.h) rtol=1e-3
    @test m.H ≈ _SM_B.h rtol=1e-3
    @test size(factor_r2(m)) == (6, 2) && length(series_r2(m)) == 6
end

@testset "DFM: the normalisation makes the fit unique" begin
    # 20 random starts on fixture B: every converged fit gives the same
    # loadings -- no rotation, no sign flip. The test that fails if either
    # reference's non-identification is copied.
    rng = MersenneTwister(112)
    ref = fit_dfm(_YB, 2)
    nconv = 0
    for _ in 1:20
        Λ0 = randn(rng, 6, 2) .* 0.5
        Λ0[1, 2] = 0.0; Λ0[1, 1] = abs(Λ0[1, 1]) + 0.1; Λ0[2, 2] = abs(Λ0[2, 2]) + 0.1
        A0 = 0.3 .* randn(rng, 2, 2)
        h0 = 0.2 .+ rand(rng, 6)
        m = fit_dfm(_YB, 2; start=(Λ0, A0, h0))
        m.converged || continue
        nconv += 1
        @test m.Lambda ≈ ref.Lambda atol=1e-5
    end
    @test nconv == 20
end

@testset "DFM: rotate_factors preserves the likelihood" begin
    m = fit_dfm(_YB, 2)
    for θ in (0.3, 1.9)
        G = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        mr = rotate_factors(m, G)
        @test loglikelihood(mr) ≈ loglikelihood(m) atol=1e-10
        @test mr.normalisation === :rotated
        @test mr.Lambda * mr.factors' ≈ m.Lambda * m.factors' rtol=1e-8
    end
    Gref = [1.0 0.0; 0.0 -1.0]                           # a reflection
    @test loglikelihood(rotate_factors(m, Gref)) ≈ loglikelihood(m) atol=1e-10
    @test_throws ArgumentError rotate_factors(m, [1.0 0.5; 0.0 1.0])
    @test_throws ArgumentError rotate_factors(m, Matrix(1.0I, 3, 3))
end

@testset "DFM: missing data" begin
    Ym = standardize(_dfmfx("fixtureA_miss.csv"))
    @test count(isnan, Ym) == 12
    # bookkeeping: missing cells per period {0: 193, 1: 5, 2: 1, 5: 1}, and
    # the periods with any missing (0-based in the handoff) 5,40,41,42,100,150,199
    nm = [count(isnan, r) for r in eachrow(Ym)]
    @test count(==(0), nm) == 193 && count(==(1), nm) == 5 && count(==(2), nm) == 1 && count(==(5), nm) == 1
    @test findall(>(0), nm) .- 1 == [5, 40, 41, 42, 100, 150, 199]
    m = fit_dfm(Ym, 1; params=(reshape(_SM_A.lam, 5, 1), fill(_SM_A.A, 1, 1), _SM_A.h))
    @test loglikelihood(m) ≈ -1035.8439176455618 atol=1e-9       # statsmodels loglike
    mf = fit_dfm(Ym, 1)
    @test mf.converged
    # the fitted likelihood rises (12 fewer density terms); per contributing
    # observation it is stable: -1.04953341 complete, -1.04840752 here
    @test loglikelihood(mf) ≈ -1035.826629949621 atol=1e-5          # statsmodels fit
    @test loglikelihood(mf) / 988 ≈ -1.04840752 atol=1e-6
    @test all(isfinite, mf.factors)
end

@testset "DFM: refusals" begin
    e = try fit_dfm(_YA, 5) catch err; err end
    @test e isa ArgumentError && occursin("r < N", sprint(showerror, e))
    @test_throws ArgumentError fit_dfm(_YA, 0)
    @test_throws ArgumentError fit_dfm(_YA[:, 1:1], 1)
    e = try fit_dfm(_YA, 1; p=0) catch err; err end
    @test e isa ArgumentError && occursin("p must be >= 1", sprint(showerror, e))
    @test_throws ArgumentError fit_dfm(_YA, 1; params=(ones(4, 1), fill(0.5, 1, 1), ones(5)))
    # a zero idiosyncratic variance is a degenerate likelihood: refused
    @test_throws ArgumentError fit_dfm(_YA, 1; params=(reshape(_SM_A.lam, 5, 1), fill(_SM_A.A, 1, 1), zeros(5)))
end

@testset "DFM: Bai-Ng information criteria (dfms ICr values)" begin
    ic = bai_ng_ic(_YB, 4)
    @test ic.IC1 ≈ [-0.47458253, -1.26002227, -1.37817140, -1.62434149] atol=1e-8
    @test ic.IC2[1:2] ≈ [-0.47053491, -1.25192703] atol=1e-8
    @test ic.IC3[1:2] ≈ [-0.47770195, -1.26626110] atol=1e-8
    # they keep decreasing past the true r = 2 at N = 6 -- not a selector here
    @test argmin(ic.IC1) == 4
    @test_throws ArgumentError bai_ng_ic(_YB, 6)
end

@testset "DFM: the stationary prior is recomputed from the parameters" begin
    Λ = reshape(_SM_A.lam, 5, 1)
    l1 = loglikelihood(fit_dfm(_YA, 1; params=(Λ, fill(0.3, 1, 1), _SM_A.h)))
    l2 = loglikelihood(fit_dfm(_YA, 1; params=(Λ, fill(0.9, 1, 1), _SM_A.h)))
    @test l1 != l2
    # by hand: P0 = 1/(1 - A^2) for the factor AR(1)
    ssm = TimeVaryingSSM{Float64}([fill(0.9, 1, 1)], [Λ], [ones(1, 1)], [ones(1, 1)],
                                  [Matrix(Diagonal(_SM_A.h))], 1)
    @test l2 ≈ kalman_filter(ssm, _YA, [0.0], fill(1 / (1 - 0.81), 1, 1))[1] rtol=1e-12
    # a VAR(2) factor process fits and stays stationary
    m2 = fit_dfm(_YB, 1; p=2)
    @test maximum(abs, eigvals(TSAnalytics._companion(m2.A))) < 1
end

@testset "DFM: forecasting and show" begin
    m = fit_dfm(_YA, 1)
    f = forecast(m, 8)
    @test f isa VARForecast && size(f.point) == (8, 5)
    @test predict(m, 8).point == f.point
    @test all(f.se[8, :] .>= f.se[1, :])
    # the long-run forecast decays to the mean (zero) with the factor's AR
    @test maximum(abs, f.point[8, :]) < maximum(abs, f.point[1, :])
    # the long-run variance is the factor's stationary variance plus H
    fl = forecast(m, 400)
    @test fl.se[400, :] .^ 2 ≈ vec(m.Lambda .^ 2) ./ (1 - m.A[1][1]^2) .+ m.H rtol=1e-6
    s = sprint(show, m)
    @test occursin("lower-triangular", s) && occursin("optimiser converged", s)
end
