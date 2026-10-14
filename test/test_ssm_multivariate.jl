using DelimitedFiles
using LinearAlgebra: Diagonal

# Stage 11.0a/b/d -- the sequential multivariate filter and smoother, and the
# stationary VAR reparameterisation. handoff/stage 11/docs/
# stage-11.0-primitives-handoff.md §4 and §6.3. 11.0c (partitioned
# initialisation) is tested in test_uc.jl, 11.0e (_ols weights) in
# test_unitroot.jl.

const _MVFX = joinpath(@__DIR__, "..", "handoff", "stage 11", "fixtures")
_mvfx(f) = Float64.(readdlm(joinpath(_MVFX, f), ','; skipstart=1))
# the fixtures are raw; the references' likelihoods are for data standardised
# with ddof = 1 (DynamicFactor is not scale-invariant: raw -1180.14 vs
# standardised -1049.53 on fixture A)
_mvstd(Y) = (Y .- [mean(filter(!isnan, c)) for c in eachcol(Y)]') ./
            [std(filter(!isnan, c)) for c in eachcol(Y)]'

# statsmodels DynamicFactor(z, k_factors=1, factor_order=1).fit() on fixture A,
# full precision (ground-truth/sm_results.json). NB the parameter table printed
# in the handoff's §6.1 is from a different fit and gives -1049.53340809606.
const _LAM_A = [0.7009068155913395, 0.6042910774398366, -0.5753739335367198,
                0.5237349698498732, 0.6650726532699288]
const _H_A = [0.14812824410505457, 0.36549749235533713, 0.42432314355129,
              0.5221441574104723, 0.23249862054434853]
const _A_A = 0.6482477536949357

function _dfm_ssm(lam::AbstractMatrix, A::AbstractMatrix, h::AbstractVector)
    r = size(A, 1)
    TimeVaryingSSM{Float64}([Matrix(A)], [Matrix(lam)], [Matrix(1.0I, r, r)],
                            [Matrix(1.0I, r, r)], [Matrix(Diagonal(h))], r)
end
_statP0(A) = TSAnalytics._stationary_cov(A, Matrix(1.0I, size(A)...))[1]

# The conventional filter -- matrix F_t, all N series at once -- as a TEST
# HELPER only (handoff §4.3): an independent check, not package code.
function _conventional_filter(ssm, Y, a0, P0)
    n, N = size(Y)
    a, P = copy(a0), copy(P0)
    ll = 0.0
    for t in 1:n
        Z, H, T, R, Q = ssm.Z[1], ssm.H[1], ssm.T[1], ssm.R[1], ssm.Q[1]
        obs = findall(!isnan, Y[t, :])
        if !isempty(obs)
            Zo = Z[obs, :]
            v = Y[t, obs] .- Zo * a
            F = Zo * P * Zo' .+ H[obs, obs]
            Fi = inv(F)
            ll += -0.5 * (length(obs) * log(2pi) + logdet(F) + dot(v, Fi * v))
            K = P * Zo' * Fi
            a = a .+ K * v
            P = P .- K * Zo * P
        end
        a = T * a
        P = T * P * T' .+ R * Q * R'
    end
    return ll
end

@testset "multivariate filter: N = 1 reduces to the univariate method" begin
    # Not bit-exact, and cannot be: the univariate method folds T into the
    # gain (K = T P z / f, a' = T a + K v) where the sequential form updates
    # then transitions (a + k v, then T). Same algebra, different rounding --
    # so a relative tolerance at the level of one ulp per step.
    Y = _mvstd(_mvfx("fixtureA.csv"))
    y = Y[:, 1]
    s1 = _dfm_ssm(fill(_LAM_A[1], 1, 1), fill(_A_A, 1, 1), [_H_A[1]])
    P0 = fill(1 / (1 - _A_A^2), 1, 1)
    l1, v1, F1, ok1 = kalman_filter(s1, y, [0.0], P0)
    lm, vm, Fm, okm = kalman_filter(s1, reshape(y, :, 1), [0.0], P0)
    @test ok1 && okm
    @test lm ≈ l1 rtol=1e-13
    @test size(vm) == (200, 1) && size(Fm) == (200, 1)
    @test vm[:, 1] ≈ v1 rtol=1e-12
    @test Fm[:, 1] ≈ F1 rtol=1e-12
    # with a missing value too
    ym = copy(y); ym[[5, 6, 90]] .= NaN
    @test kalman_filter(s1, reshape(ym, :, 1), [0.0], P0)[1] ≈ kalman_filter(s1, ym, [0.0], P0)[1] rtol=1e-13
    # the smoother: all seven returns
    a1, V1, e1, ev1, ep1, epv1, c1 = kalman_smoother(s1, y, [0.0], P0)
    am, Vm, em, evm, epm, epvm, cm = kalman_smoother(s1, reshape(y, :, 1), [0.0], P0)
    @test c1 && cm
    @test am ≈ a1 rtol=1e-12
    @test [V[1, 1] for V in Vm] ≈ [V[1, 1] for V in V1] rtol=1e-12
    @test vec(em) ≈ e1 rtol=1e-11
    @test [V[1, 1] for V in evm] ≈ ev1 rtol=1e-12
    @test epm[:, 1] ≈ ep1 rtol=1e-11
    @test epvm[:, 1] ≈ epv1 rtol=1e-12
end

@testset "multivariate filter: fixtures A and B against statsmodels and dfms" begin
    # fixture A: statsmodels -1049.5334081297972; dfms' own C++ filter
    # -1049.53340812981 -- two independent implementations within 1.4e-10
    Y = _mvstd(_mvfx("fixtureA.csv"))
    ssm = _dfm_ssm(reshape(_LAM_A, 5, 1), fill(_A_A, 1, 1), _H_A)
    ll, v, F, ok = kalman_filter(ssm, Y, [0.0], fill(1 / (1 - _A_A^2), 1, 1))
    @test ok && size(v) == (200, 5)
    @test ll ≈ -1049.5334081297972 atol=1e-9
    @test ll ≈ -1049.53340812981 atol=1e-9
    @test all(>(0), F)
    # fixture B, r = 2: statsmodels' loglike at its own fitted parameters
    YB = _mvstd(_mvfx("fixtureB.csv"))
    lamB = [0.5533928213458822 0.4309722523368841; 0.6757507498964977 0.12316763253210221;
            -0.021867285934118467 -0.6705529300459949; 0.03628674218233418 0.587948019498846;
            0.6662154932967631 0.29179316358890744; 0.5238410868614163 -0.5266630210571038]
    AB = [0.5309760294445562 -0.03867147247037333; 0.20951816279826058 0.6166771974292626]
    hB = [0.15894097178603864, 0.30764156470260096, 0.1856257988091492, 0.36829583094348023,
          0.1629034340031904, 0.21139302091552234]
    sB = _dfm_ssm(lamB, AB, hB)
    llB, = kalman_filter(sB, YB, zeros(2), _statP0(AB))
    @test llB ≈ -1514.940610472312 atol=1e-9
end

@testset "multivariate filter: agreement with the conventional form" begin
    Y = _mvstd(_mvfx("fixtureA.csv"))
    P0 = fill(1 / (1 - _A_A^2), 1, 1)
    ssm = _dfm_ssm(reshape(_LAM_A, 5, 1), fill(_A_A, 1, 1), _H_A)
    @test kalman_filter(ssm, Y, [0.0], P0)[1] ≈ _conventional_filter(ssm, Y, [0.0], P0) rtol=1e-12
    Ym = _mvstd(_mvfx("fixtureA_miss.csv"))
    @test kalman_filter(ssm, Ym, [0.0], P0)[1] ≈ _conventional_filter(ssm, Ym, [0.0], P0) rtol=1e-12
end

@testset "multivariate filter: as the measurement variances collapse" begin
    # statsmodels' start parameters on fixture A, with every idiosyncratic
    # variance set to s2. The true log-likelihood scales like 1/s2 (one
    # series nearly determines the next): ~-1.37e8 at 1e-6, so ~-1.4e14 at
    # 1e-12. statsmodels' conventional filter tracks that; its UNIVARIATE
    # filter agrees at 1e-6 and then reports -696.17 at 1e-12, because it
    # silently drops near-degenerate updates. The handoff (§2.5) read the
    # -696.17 as graceful degradation and the -1.4e14 as garbage; measured,
    # it is the other way round. This filter computes the true value and
    # refuses f = 0 outright.
    Y = _mvstd(_mvfx("fixtureA.csv"))
    p = [12.919803043241036, 11.937820487098975, -11.651677436222098, 10.82468894190494,
         12.565283642625342, 0.9339618663417418, 0.9428877832391263, 0.9453560458269605,
         0.9521530107679921, 0.9372656802813816, 0.5905385254421762]
    P0 = fill(1 / (1 - p[11]^2), 1, 1)
    ll(h) = kalman_filter(_dfm_ssm(reshape(p[1:5], 5, 1), fill(p[11], 1, 1), h), Y, [0.0], P0)[1]
    @test ll(p[6:10]) ≈ -1699.4011064899162 rtol=1e-12       # statsmodels, both filters
    @test ll(fill(1e-2, 5)) ≈ -13472.637488057357 rtol=1e-11
    @test ll(fill(1e-6, 5)) ≈ -1.3736870220394245e8 rtol=1e-7
    @test ll(fill(1e-12, 5)) < -1e13                          # not -696.17
    e = try ll(zeros(5)) catch err; err end
    @test e isa ArgumentError
    @test occursin("period 1, series 3", sprint(showerror, e))
    # the optimiser's path gets the sentinel instead of an exception
    s0 = _dfm_ssm(reshape(p[1:5], 5, 1), fill(p[11], 1, 1), zeros(5))
    ls, vs, Fs, oks = TSAnalytics._kf_mv(s0, Y, [0.0], P0, false)
    @test ls == -Inf && !oks && isempty(vs)
end

@testset "multivariate filter: missing data" begin
    Y = _mvstd(_mvfx("fixtureA_miss.csv"))
    @test count(isnan, Y) == 12 && count(r -> all(isnan, r), eachrow(Y)) == 1
    ssm = _dfm_ssm(reshape(_LAM_A, 5, 1), fill(_A_A, 1, 1), _H_A)
    P0 = fill(1 / (1 - _A_A^2), 1, 1)
    ll, v, F, ok = kalman_filter(ssm, Y, [0.0], P0)
    # statsmodels DynamicFactor.loglike at the fixture-A parameters, on the
    # 12-NaN fixture standardised with nan-aware ddof = 1
    @test ll ≈ -1035.8439176455618 atol=1e-9
    @test count(isnan, v) == 12 && count(isnan, F) == 12
    # a wholly missing period is prediction-only: the smoothed state passes
    # through it, and the filter's state there is just T a
    t0 = findfirst(r -> all(isnan, r), collect(eachrow(Y)))
    @test t0 == 151                 # the handoff's "t = 150" is a 0-based Python index
    al, V = kalman_smoother(ssm, Y, [0.0], P0)
    @test all(isfinite, al) && V[t0][1, 1] > V[t0 - 2][1, 1]
    # fewer observations, fewer density terms: the likelihood rises
    Yc = _mvstd(_mvfx("fixtureA.csv"))
    @test ll > kalman_filter(ssm, Yc, [0.0], P0)[1]
end

@testset "multivariate filter: invariances that need no reference" begin
    YB = _mvstd(_mvfx("fixtureB.csv"))
    lam = [0.55 0.43; 0.68 0.12; -0.02 -0.67; 0.04 0.59; 0.67 0.29; 0.52 -0.53]
    A = [0.53 -0.04; 0.21 0.62]; h = [0.16, 0.31, 0.19, 0.37, 0.16, 0.21]
    s = _dfm_ssm(lam, A, h)
    l0 = kalman_filter(s, YB, zeros(2), _statP0(A))[1]
    # permuting the series (and Z, H to match) -- the series do enter in order
    perm = [4, 1, 6, 2, 5, 3]
    sp = _dfm_ssm(lam[perm, :], A, h[perm])
    @test kalman_filter(sp, YB[:, perm], zeros(2), _statP0(A))[1] ≈ l0 rtol=1e-12
    # an orthogonal rotation of the state
    U = [cos(0.7) -sin(0.7); sin(0.7) cos(0.7)]
    sr = TimeVaryingSSM{Float64}([U * A * U'], [lam * U'], [U], [Matrix(1.0I, 2, 2)],
                                 [Matrix(Diagonal(h))], 2)
    @test kalman_filter(sr, YB, zeros(2), U * _statP0(A) * U')[1] ≈ l0 rtol=1e-12
end

@testset "multivariate smoother: the disturbance identity, and refusals" begin
    YB = _mvstd(_mvfx("fixtureB.csv"))
    lam = [0.55 0.43; 0.68 0.12; -0.02 -0.67; 0.04 0.59; 0.67 0.29; 0.52 -0.53]
    A = [0.53 -0.04; 0.21 0.62]; h = [0.16, 0.31, 0.19, 0.37, 0.16, 0.21]
    s = _dfm_ssm(lam, A, h)
    al, V, eta, etav, eps, epsv, ok = kalman_smoother(s, YB, zeros(2), _statP0(A))
    @test ok && size(al) == (2, 250) && size(eta) == (2, 250) && size(eps) == (250, 6)
    # alpha_{t+1} = T alpha_t + R eta_t, exactly -- this catches a
    # mis-ordered series loop that a state-only comparison would miss
    @test maximum(abs.(al[:, 2:end] .- (A * al[:, 1:(end - 1)] .+ eta[:, 1:(end - 1)]))) < 1e-12
    # and the observation equation closes: y = Z alpha + eps
    @test maximum(abs.(YB' .- (lam * al .+ eps'))) < 1e-12
    @test all(v -> all(>=(-1e-12), diag(v)), V) && all(>(0), epsv)
    # a dense H is refused rather than silently diagonalised
    sd = TimeVaryingSSM{Float64}([A], [lam], [Matrix(1.0I, 2, 2)], [Matrix(1.0I, 2, 2)],
                                 [Matrix(Diagonal(h)) .+ 0.01], 2)
    e = try kalman_filter(sd, YB, zeros(2), _statP0(A)) catch err; err end
    @test e isa ArgumentError && occursin("not diagonal", sprint(showerror, e))
    @test_throws ArgumentError kalman_smoother(sd, YB, zeros(2), _statP0(A))
    @test_throws ArgumentError kalman_filter(s, YB[:, 1:5], zeros(2), _statP0(A))
end

@testset "multivariate filter: the vector method is untouched; cost grows with N" begin
    # every existing call site passes a Vector and still dispatches to the
    # univariate method
    s1 = _dfm_ssm(fill(0.7, 1, 1), fill(0.5, 1, 1), [0.3])
    @test which(kalman_filter, (typeof(s1), Vector{Float64}, Vector{Float64}, Matrix{Float64})).file ==
          which(kalman_filter, (typeof(s1), Vector{Float64}, Vector{Float64}, Matrix{Float64})).file
    @test occursin("timevaryingssm",
                   string(which(kalman_filter, (typeof(s1), Vector{Float64}, Vector{Float64}, Matrix{Float64})).file))
    @test occursin("multivariate",
                   string(which(kalman_filter, (typeof(s1), Matrix{Float64}, Vector{Float64}, Matrix{Float64})).file))
    # no allocation inside the time loop: doubling n adds only the output
    # arrays (v and F, n x N each)
    YB = _mvstd(_mvfx("fixtureB.csv"))
    lam = [0.55 0.43; 0.68 0.12; -0.02 -0.67; 0.04 0.59; 0.67 0.29; 0.52 -0.53]
    s = _dfm_ssm(lam, [0.53 -0.04; 0.21 0.62], [0.16, 0.31, 0.19, 0.37, 0.16, 0.21])
    P0 = _statP0([0.53 -0.04; 0.21 0.62])
    kalman_filter(s, YB, zeros(2), P0)
    b125 = @allocated kalman_filter(s, YB[1:125, :], zeros(2), P0)
    b250 = @allocated kalman_filter(s, YB, zeros(2), P0)
    @test b250 - b125 <= 2 * 125 * 6 * 8 + 1024
end

@testset "stationary VAR reparameterisation (Stage 11.0d)" begin
    rng = MersenneTwister(110)
    for (k, p) in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2))
        maxrho = 0.0; ok_cov = true; rt_A = 0.0; rt_u = 0.0
        for _ in 1:300
            u = 1.5 .* randn(rng, p * k^2)
            A = TSAnalytics._constrain_stationary(u, k, p)
            C = TSAnalytics._companion(A)
            maxrho = max(maxrho, maximum(abs, eigvals(C)))
            # the stationary covariance exists for every draw
            Rk = zeros(k * p, k); Rk[1:k, :] = Matrix(1.0I, k, k)
            ok_cov &= TSAnalytics._stationary_cov(C, Rk * Rk'; maxiter=200)[2]
            u2 = TSAnalytics._unconstrain_stationary(A)
            A2 = TSAnalytics._constrain_stationary(u2, k, p)
            rt_A = max(rt_A, maximum(maximum.(abs, A2 .- A)))
            rt_u = max(rt_u, maximum(abs, u2 .- u))
        end
        @test maxrho < 1                                  # every draw, not usually
        @test ok_cov
        @test rt_A < 1e-12                                # coefficients round-trip
        # u itself to 1e-9: large |u| sits where the map flattens (P -> ±1),
        # which amplifies the error in u while the coefficients stay exact
        @test rt_u < 1e-9
    end
    # k = 1, p = 1 is the familiar u / sqrt(1 + u^2), exactly
    for u in (-3.0, -0.4, 0.0, 0.37, 2.5)
        @test TSAnalytics._constrain_stationary([u], 1, 1)[1][1] == u / sqrt(1 + u^2)
    end
    # differentiable: ForwardDiff against central differences
    u0 = randn(MersenneTwister(3), 8)
    f(u) = sum(sum, TSAnalytics._constrain_stationary(u, 2, 2))
    g = TSAnalytics.ForwardDiff.gradient(f, u0)
    gfd = [(f(u0 .+ 1e-6 .* (1:8 .== j)) - f(u0 .- 1e-6 .* (1:8 .== j))) / 2e-6 for j in 1:8]
    @test all(isfinite, g) && maximum(abs.(g .- gfd)) < 1e-7
    # an explosive VAR has no preimage: refused
    @test_throws ArgumentError TSAnalytics._unconstrain_stationary([fill(1.2, 1, 1)])
end
