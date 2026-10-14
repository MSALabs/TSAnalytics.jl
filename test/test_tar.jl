# Stage 11.4: threshold autoregressions (SETAR, LSTAR, Hansen's test).
# Targets from handoff/stage 11/docs/stage-11.4-tar-star-handoff.md, measured
# with tsDyn 11.0.5.2 (and aeon 1.6.0 for the matched sample).
using DelimitedFiles


const TAR_FIX = joinpath(@__DIR__, "..", "handoff", "stage 11", "fixtures")
_tar_csv(f) = readdlm(joinpath(TAR_FIX, f), ','; skipstart=1)
const LYNX10 = log10.(Float64.(_tar_csv("fixture_lynx.csv")[:, 2]))
const SUNH = Float64.(_tar_csv("sunspot_hansen.csv")[:, 1])
const SETARC = Float64.(_tar_csv("fixture_setar_sim.csv")[:, 2])

@testset "SETAR lynx: threshold exact, SSR, coefficients" begin
    f = fit_setar(LYNX10, 2; thDelay=1, warn=false)
    @test f.thresholds == [3.3100557377508912]
    @test f.searched
    @test f.ssr ≈ 4.3481912792005826 rtol = 1e-13
    ref = [0.58843692925212931, 1.2642792839182719, -0.42842921164134479,
           1.1656919479037287, 1.5992540700908249, -1.0115754904952012]
    @test coef(f) ≈ ref rtol = 1e-12
    for i in 1:6
        @test coef(f)[i] ≈ ref[i] rtol = 1e-12
    end
    @test coefnames(f) == ["const.L", "phiL.1", "phiL.2", "const.H", "phiH.1", "phiH.2"]
    @test nobs(f) == 112
    @test count(==(1), f.regime) == 78
    @test count(==(2), f.regime) == 34
    @test length(f.resid) == 112
    @test sum(abs2, f.resid) ≈ f.ssr rtol = 1e-12
    @test f.fitted .+ f.resid ≈ LYNX10[3:end] rtol = 1e-13
    @test dof(f) == 7
end

@testset "SETAR lynx: standard errors, both denominators" begin
    f = fit_setar(LYNX10, 2; thDelay=1, warn=false)
    se_ref = [0.14330661378560067, 0.065256287325703141, 0.077486946473903501,
              0.87660576968756576, 0.10896581100312897, 0.26501113872650156]
    @test f.se_tsdyn ≈ se_ref rtol = 1e-10
    for i in 1:6
        @test f.se_tsdyn[i] ≈ se_ref[i] rtol = 1e-10
        # ours: SSR/(rows − k) = SSR/106; tsDyn's SSR/108
        @test f.se[i] ≈ se_ref[i] * sqrt(108 / 106) rtol = 1e-10
    end
    @test stderror(f) == f.se
    ic = tsdyn_ic(f)
    @test ic.mse ≈ 0.038142028764917388 rtol = 1e-12
    @test ic.aic ≈ -358.373987591728 rtol = 1e-12
    @test ic.bic ≈ -339.22059845296656 rtol = 1e-12
end

@testset "the four-way denominator table" begin
    f = fit_setar(LYNX10, 2; thDelay=1, warn=false)
    s108 = f.se_tsdyn[1]
    tab = [(108, 0.14330661379), (107, 0.14397471351), (106, 0.14465224531), (105, 0.14533943325)]
    for (df, v) in tab
        @test s108 * sqrt(108 / df) ≈ v rtol = 1e-9
    end
    # only 108 = n − p reproduces tsDyn; ours is 106 = rows − k
    @test f.se_tsdyn[1] ≈ 0.14330661379 rtol = 1e-9
    @test f.se[1] ≈ 0.14465224531 rtol = 1e-9
    @test !(f.se[1] ≈ 0.14330661379)
    @test f.se[1] > f.se_tsdyn[1]
end

@testset "SETAR lynx thDelay = 0" begin
    f = fit_setar(LYNX10, 2; thDelay=0, warn=false)
    @test f.thresholds[1] == 2.5575072019056577
    @test f.thresholds[1] ≈ 2.557507201906 atol = 1e-12
    @test f.ssr ≈ 4.565530806743 rtol = 1e-11
    ref = [0.405942732118, 1.245677428926, -0.333928504230, 1.180869464937,
           1.547698349232, -0.956274108919]
    for i in 1:6
        @test coef(f)[i] ≈ ref[i] atol = 1e-11
    end
    @test f.thDelay == 0
end

@testset "SETAR sunspot m = 11" begin
    @test length(SUNH) == 289
    @test sum(SUNH) ≈ 3151.20015794 rtol = 1e-11
    @test SUNH[1] ≈ (sqrt(5 + 1) - 1) * 2 rtol = 1e-14
    f = fit_setar(SUNH, 11; thDelay=1, warn=false)
    @test f.thresholds[1] ≈ 7.4233751915118 rtol = 1e-13
    @test f.thresholds[1] ≈ 7.423375191512 atol = 1e-12
    @test f.ssr ≈ 907.442712049961 rtol = 1e-11
    @test length(coef(f)) == 24
    @test dof(f) == 25
    @test nobs(f) == 278
    @test count(==(1), f.regime) / 278 ≈ 0.309352517986 rtol = 1e-11
    @test count(==(2), f.regime) / 278 ≈ 0.690647482014 rtol = 1e-11
    @test tsdyn_ic(f).aic ≈ 380.674883408 rtol = 1e-10
    @test all(isfinite, f.se)
    @test all(>(0), f.se)
    @test all(f.se .> f.se_tsdyn)
end

@testset "SETAR fixture C: DGP recovery within 1.3 se" begin
    @test length(SETARC) == 1000
    @test mean(SETARC) ≈ 0.435436080662 rtol = 1e-11
    @test std(SETARC) ≈ 1.28768121011 rtol = 1e-10
    f = fit_setar(SETARC, 1; thDelay=0, warn=false)
    @test f.thresholds[1] ≈ 0.0127793045 atol = 1e-10
    @test f.ssr ≈ 1001.7766997942632 rtol = 1e-12
    ref = [0.23751762454458322, 0.44953311723945383, -0.27843809089464733, 0.88205356007858393]
    se_ref = [0.079993633065496755, 0.078034597295509919, 0.065272771619280689, 0.042680558818975908]
    @test coef(f) ≈ ref rtol = 1e-11
    @test f.se_tsdyn ≈ se_ref rtol = 1e-10
    @test count(==(1), f.regime) == 385
    @test count(==(2), f.regime) == 614
    @test tsdyn_ic(f).aic ≈ 11.775123330175814 rtol = 1e-11
    truth = [0.3, 0.5, -0.2, 0.9]
    for i in 1:4
        @test abs(coef(f)[i] - truth[i]) < 1.3 * f.se_tsdyn[i]
    end
    @test abs(coef(f)[1] - truth[1]) < 0.79 * f.se_tsdyn[1]
    # the threshold fixed at its true value
    g = fit_setar(SETARC, 1; thDelay=0, threshold=0.0, warn=false)
    @test coef(g) ≈ [0.230389423843, 0.444148537350, -0.263473337001, 0.874406039907] atol = 1e-11
    @test g.ssr ≈ 1004.11661592 rtol = 1e-10
    @test g.ssr > f.ssr
end

@testset "aeon-matched sample, bit-exact" begin
    f = fit_setar(LYNX10[2:end], 2; thDelay=0, warn=false)
    @test f.thresholds[1] ≈ 2.557507201906 atol = 1e-12
    @test f.ssr ≈ 4.562802243613 atol = 1e-12
    @test coef(f)[1] ≈ 0.416800274876 atol = 1e-11
    @test coef(f)[2] ≈ 1.237891018066 atol = 1e-11
    @test coef(f)[3] ≈ -0.332005825545 atol = 1e-11
    @test coef(f)[4] ≈ 1.180869464937 atol = 1e-11
    @test coef(f)[5] ≈ 1.547698349232 atol = 1e-11
    @test coef(f)[6] ≈ -0.956274108919 atol = 1e-11
    @test count(==(1), f.regime) == 30
    @test count(==(2), f.regime) == 81
end

@testset "incremental search = naive search, exact" begin
    for (y, m, d) in ((LYNX10, 2, 1), (LYNX10, 2, 0), (SETARC, 1, 0))
        f = fit_setar(y, m; thDelay=d, warn=false)
        naive = [(c, TSAnalytics._setar_fit_at(y, m, d, [c], :const,
                                                [collect(1:m), collect(1:m)]).ssr)
                 for (c, _) in f.profile]
        @test first.(naive) == first.(f.profile)
        @test maximum(abs.(last.(naive) .- last.(f.profile)) ./ last.(naive)) < 1e-10
        @test naive[argmin(last.(naive))][1] == f.thresholds[1]
    end
end

@testset "candidate grid: lynx, ties included" begin
    _, _, z, _ = TSAnalytics._tar_embed(LYNX10, 2, 1)
    @test length(z) == 112
    @test length(unique(z)) == 108
    full = TSAnalytics._tar_grid(z, 0.15)
    ts = TSAnalytics._tar_grid(z, 0.15; grid=:tsdyn)
    @test length(full) == 75
    @test length(ts) == 74
    @test issubset(ts, full)
    @test length(setdiff(full, ts)) == 1
    @test issorted(full)
    @test allunique(full)
    @test all(in(z), full)
    f1 = fit_setar(LYNX10, 2; thDelay=1, warn=false)
    f2 = fit_setar(LYNX10, 2; thDelay=1, grid=:tsdyn, warn=false)
    @test f1.thresholds == f2.thresholds
    @test f1.ssr == f2.ssr
    @test length(f1.profile) == 75
end

@testset "candidate grid: ties vs unique differ on a tied series" begin
    # 200 values on 45 levels, heavily tied in the middle
    z = Float64[]
    for (i, v) in enumerate(-2.2:0.1:2.2)
        append!(z, fill(round(v; digits=1), max(1, round(Int, 12 * exp(-v^2)))))
    end
    sort!(z)
    gt = TSAnalytics._tar_grid(z, 0.15)
    gu = TSAnalytics._tar_grid(unique(z), 0.15)
    @test length(unique(z)) == 45
    @test gt != gu
    @test length(gt) < length(gu)
    # the ties-grid trims by mass: its ends are nearer the centre
    @test first(gt) > first(gu)
    @test last(gt) < last(gu)
    @test all(in(z), gt)
    @test allunique(gt)
    @test issorted(gt)
end

@testset "boundary: z ≤ th inclusive" begin
    f = fit_setar(LYNX10, 2; thDelay=1, warn=false)
    _, _, z, _ = TSAnalytics._tar_embed(LYNX10, 2, 1)
    th = f.thresholds[1]
    @test count(==(th), z) == 1
    @test count(<=(th), z) == 78
    @test count(<(th), z) == 77
    @test f.regime[63] == 1
    @test z[63] == th
    @test all(f.regime[z .<= th] .== 1)
    @test all(f.regime[z .> th] .== 2)
    @test count(==(1), f.regime) == count(<=(th), z)
    @test count(==(2), f.regime) == 34
    @test length(f.regime) == 112
end

@testset "thDelay indexing: d = 0 uses y_{t−1}" begin
    _, lag, z0, _ = TSAnalytics._tar_embed(LYNX10, 2, 0)
    _, _, z1, _ = TSAnalytics._tar_embed(LYNX10, 2, 1)
    @test z0 == lag(1)
    @test z1 == lag(2)
    @test z0 == LYNX10[2:113]
    @test z1 == LYNX10[1:112]
    @test_throws ArgumentError fit_setar(LYNX10, 2; thDelay=2)
    @test_throws ArgumentError fit_setar(LYNX10, 2; thDelay=-1)
    f = fit_setar(LYNX10, 3; thDelay=2, warn=false)
    @test f.thDelay == 2
end

@testset "bit-exact threshold trap" begin
    exact = fit_setar(LYNX10, 2; thDelay=1, threshold=3.3100557377508912, warn=false)
    trunc = fit_setar(LYNX10, 2; thDelay=1, threshold=3.31005573775, warn=false)
    @test exact.ssr ≈ 4.3481912792005826 rtol = 1e-13
    @test trunc.ssr ≈ 4.4217191153 rtol = 1e-10
    @test count(==(1), exact.regime) == 78
    @test count(==(1), trunc.regime) == 77
    @test !exact.searched
    # show prints the threshold so it round-trips
    s = sprint(show, fit_setar(LYNX10, 2; thDelay=1, warn=false))
    @test occursin("3.3100557377508912", s)
    @test parse(Float64, "3.3100557377508912") == exact.thresholds[1]
    @test occursin("no standard error for the threshold", s)
end

@testset "include = variants" begin
    th = 3.3100557377508912
    fs = Dict(inc => fit_setar(LYNX10, 2; thDelay=1, threshold=th, include=inc, warn=false)
              for inc in (:none, :const, :trend, :both))
    @test length(coef(fs[:none])) == 4
    @test length(coef(fs[:const])) == 6
    @test length(coef(fs[:trend])) == 6
    @test length(coef(fs[:both])) == 8
    @test coefnames(fs[:none]) == ["phiL.1", "phiL.2", "phiH.1", "phiH.2"]
    @test "trend.L" in coefnames(fs[:trend])
    @test !("const.L" in coefnames(fs[:trend]))
    @test "const.H" in coefnames(fs[:both]) && "trend.H" in coefnames(fs[:both])
    # nested models: more columns never raise the SSR
    @test fs[:both].ssr <= fs[:const].ssr
    @test fs[:both].ssr <= fs[:trend].ssr
    @test fs[:const].ssr <= fs[:none].ssr
    @test_throws ArgumentError fit_setar(LYNX10, 2; include=:common)
end

@testset "per-regime orders mL ≠ mH" begin
    f = fit_setar(LYNX10, 3; thDelay=1, mL=1, mH=3, warn=false)
    @test f.ssr ≈ 4.97708484151 rtol = 1e-10
    @test f.thresholds[1] ≈ 2.406540180434 atol = 1e-12
    @test coefnames(f) == ["const.L", "phiL.1", "const.H", "phiH.1", "phiH.2", "phiH.3"]
    @test dof(f) == 7
    @test f.lags == [[1], [1, 2, 3]]
    @test nobs(f) == 111
    @test length(predict(f, 3)) == 3
    @test_throws ArgumentError fit_setar(LYNX10, 2; mL=3)
    # an order of 0 is a constant-only regime, which is allowed
    @test coefnames(fit_setar(LYNX10, 2; mH=0, warn=false))[end] == "const.H"
    @test all(isfinite, f.se)
end

@testset "non-contiguous ML / MH" begin
    f = fit_setar(LYNX10, 3; thDelay=1, ML=[1, 3], MH=[1, 2], warn=false)
    @test f.ssr ≈ 4.11329623954 rtol = 1e-10
    @test coefnames(f) == ["const.L", "phiL.1", "phiL.3", "const.H", "phiH.1", "phiH.2"]
    @test f.lags == [[1, 3], [1, 2]]
    @test dof(f) == 7
    @test length(f.resid) == 111
    # the prediction uses exactly the named lags
    p = predict(f, 1)[1]
    n = length(LYNX10)
    r = LYNX10[n - 1] <= f.thresholds[1] ? 1 : 2
    b = coef(f)
    want = r == 1 ? b[1] + b[2] * LYNX10[n] + b[3] * LYNX10[n - 2] :
                    b[4] + b[5] * LYNX10[n] + b[6] * LYNX10[n - 1]
    @test p ≈ want rtol = 1e-14
    @test_throws ArgumentError fit_setar(LYNX10, 2; ML=[1, 3])
    @test_throws ArgumentError fit_setar(LYNX10, 2; MH=[0, 1])
end

@testset "select_setar: tsDyn's table" begin
    s = select_setar(LYNX10, 2; thDelay=0:1, grid=:tsdyn)
    @test length(s.table) == 592
    @test s.best.thDelay == 1
    @test s.best.mL == 2
    @test s.best.mH == 2
    @test s.best.th == 3.3100557377508912
    @test s.best.value ≈ -32.07652679559 rtol = 1e-11
    top = [-32.0765267956, -30.8942818150, -30.3664513768, -28.8191039294, -28.6478076597]
    ths = [3.31005573775, 3.32613095671, 3.26387267687, 3.21431389742, 3.21031851983]
    for i in 1:5
        @test s.table[i].value ≈ top[i] atol = 1e-9
        @test s.table[i].th ≈ ths[i] atol = 1e-10
        @test (s.table[i].thDelay, s.table[i].mL, s.table[i].mH) == (1, 2, 2)
    end
    @test issorted(last.(s.table))
    # the full grid adds one candidate per delay: 75 × 2 × 2 × 2
    sf = select_setar(LYNX10, 2; thDelay=0:1)
    @test length(sf.table) == 600
    @test sf.best.th == s.best.th
    # the SSR criterion at the full order agrees with fit_setar
    ss = select_setar(LYNX10, 2; thDelay=1, mL=2, mH=2, criterion=:ssr)
    @test ss.best.value ≈ 4.3481912792005826 rtol = 1e-12
    @test ss.best.th == 3.3100557377508912
    @test_throws ArgumentError select_setar(LYNX10, 2; criterion=:bic)
    @test_throws ArgumentError select_setar(LYNX10, 2; thDelay=0:2)
end

@testset "nthresh = 2: sequential search" begin
    f = fit_setar(LYNX10, 2; thDelay=1, nthresh=2, warn=false)
    @test f.thresholds[1] ≈ 2.611723308007 atol = 1e-12
    @test f.thresholds[2] == 3.3100557377508912
    ref = [0.572916292592, 1.398050235753, -0.572948456361,
           1.561316851385, 1.214974024205, -0.699594817932,
           1.165691947904, 1.599254070091, -1.011575490495]
    @test length(coef(f)) == 9
    for i in 1:9
        @test coef(f)[i] ≈ ref[i] atol = 1e-11
    end
    @test f.ssr ≈ 4.08380041426 rtol = 1e-10
    @test [count(==(r), f.regime) for r in 1:3] == [40, 38, 34]
    # the high regime is the one-threshold fit's high regime
    f1 = fit_setar(LYNX10, 2; thDelay=1, warn=false)
    @test coef(f)[7:9] ≈ coef(f1)[4:6] rtol = 1e-12
    @test dof(f) == 11
    @test tsdyn_ic(f).aic ≈ -357.525427817 rtol = 1e-10
    se_ref = [0.23852440947, 0.10699944188, 0.14534970297, 0.48419192919, 0.08129938952,
              0.18588175008, 0.86158769351, 0.10709900051, 0.26047094790]
    @test f.se_tsdyn ≈ se_ref rtol = 1e-9
    @test coefnames(f)[4] == "const.M"
end

@testset "nthresh = 2: given thresholds and refusals" begin
    g = fit_setar(LYNX10, 2; thDelay=1, nthresh=2, threshold=[3.3100557377508912, 2.611723308007342],
                  warn=false)
    @test g.thresholds == [2.611723308007342, 3.3100557377508912]
    @test g.ssr ≈ 4.08380041426 rtol = 1e-10
    @test dof(g) == 9
    @test_throws ArgumentError fit_setar(LYNX10, 2; nthresh=2, threshold=3.3)
    @test_throws ArgumentError fit_setar(LYNX10, 2; nthresh=3)
    @test_throws ArgumentError fit_setar(LYNX10, 2; threshold=[2.6, 3.3])
end

@testset "LSTAR lynx: SSR, coefficients, transition" begin
    f = fit_lstar(LYNX10, 2; thDelay=1)
    # tsDyn stops short on the flat γ profile; we may only do better
    @test f.ssr <= 4.33764323215
    @test f.ssr ≈ 4.33764323215 rtol = 1e-5
    @test f.converged
    @test !f.gamma_at_bound
    @test f.th ≈ 3.339198526369 atol = 1e-2
    @test f.gamma ≈ 11.153834351444 rtol = 0.05
    @test coef(f)[1:3] ≈ [0.489101416212, 1.246539926533, -0.366432799971] atol = 0.02
    # high-regime levels are phiL + phiH
    @test f.phiL .+ f.phiH ≈ [-0.534974408877, 1.669806789842, -0.621041599748] atol = 0.05
    @test length(coef(f)) == 8
    @test coefnames(f)[end-1:end] == ["gamma", "th"]
    @test nobs(f) == 112
    @test minimum(f.G) < 1e-7
    @test 0.99 < maximum(f.G) < 1
    @test mean(f.G) ≈ 0.273310796644 atol = 2e-3
    @test dof(f) == 8
    # the profile at tsDyn's parameters reproduces its SSR exactly
    Y, lag, z, _ = TSAnalytics._tar_embed(LYNX10, 2, 1)
    X = hcat(ones(112), lag(1), lag(2))
    θR = [0.489101416212, 1.246539926533, -0.366432799971, -1.024075825089,
          0.423266863310, -0.254608799777, 11.153834351444, 3.339198526369]
    @test TSAnalytics._lstar_ssr_full(Y, X, z, θR) ≈ 4.33764323215 rtol = 1e-11
end

@testset "LSTAR: additive vs convex form" begin
    Y, lag, z, _ = TSAnalytics._tar_embed(LYNX10, 2, 1)
    X = hcat(ones(112), lag(1), lag(2))
    θR = [0.489101416212, 1.246539926533, -0.366432799971, -1.024075825089,
          0.423266863310, -0.254608799777, 11.153834351444, 3.339198526369]
    G = TSAnalytics._lstar_G(z, θR[7], θR[8])
    additive = sum(abs2, Y .- X * θR[1:3] .- (X * θR[4:6]) .* G)
    convex = sum(abs2, Y .- ((1 .- G) .* (X * θR[1:3]) .+ G .* (X * θR[4:6])))
    @test additive ≈ 4.33764323215 rtol = 1e-11
    @test convex ≈ 290.77 rtol = 1e-4
    @test convex / additive > 60
    @test extrema(G)[1] ≈ 3.40372245713e-9 rtol = 1e-8
    @test extrema(G)[2] ≈ 0.996447296207 rtol = 1e-10
    @test mean(G) ≈ 0.273310796644 rtol = 1e-10
    # γ multiplies the raw z − c, no division by sd(z)
    @test G[1] ≈ 1 / (1 + exp(-θR[7] * (z[1] - θR[8]))) rtol = 1e-14
    @test all(0 .< G .< 1)
end

@testset "LSTAR standard errors, tsDyn's convention" begin
    Y, lag, z, _ = TSAnalytics._tar_embed(LYNX10, 2, 1)
    X = hcat(ones(112), lag(1), lag(2))
    θR = [0.489101416212, 1.246539926533, -0.366432799971, -1.024075825089,
          0.423266863310, -0.254608799777, 11.153834351444, 3.339198526369]
    se = TSAnalytics._lstar_se(Y, X, z, θR, 114)
    ref = [0.20491524525, 0.06787121417, 0.10430101686, 2.43009870400,
           0.17214631591, 0.58542260103, 10.00479258390, 0.09274925250]
    # tsDyn's Hessian is numerical (optim): agreement is to its accuracy
    for i in 1:8
        @test se[i] ≈ ref[i] rtol = 2e-4
    end
    f = fit_lstar(LYNX10, 2; thDelay=1)
    @test all(isfinite, stderror(f))
    @test length(stderror(f)) == 8
end

@testset "LSTAR: γ at the bound is reported" begin
    # a SETAR series with an abrupt switch: γ runs to the grid maximum
    w = @test_logs (:warn, r"effectively abrupt") match_mode = :any fit_lstar(SETARC, 1; gamma_max=5.0)
    @test w.gamma_at_bound
    @test w.gamma >= 5.0 * (1 - 1e-6)
    @test occursin("effectively abrupt", sprint(show, w))
    f = fit_lstar(LYNX10, 2; thDelay=1)
    @test !occursin("effectively abrupt", sprint(show, f))
end

@testset "setar_test: SSRs, F statistics, n = N − m" begin
    t = setar_test(LYNX10, 2; thDelay=0:1, nboot=5, rng=MersenneTwister(1))
    @test t.ssr[1] ≈ 5.78258084172 rtol = 1e-10
    @test t.ssr[2] ≈ 4.34819127920 rtol = 1e-10
    @test t.ssr[3] ≈ 4.08380041426 rtol = 1e-10
    @test t.F[1] ≈ 36.94677181537 rtol = 1e-10
    @test t.F[2] ≈ 46.58979102185 rtol = 1e-10
    @test t.F[3] ≈ 7.25103429878 rtol = 1e-9
    @test t.nobs == 112
    @test t.F[1] / ((t.ssr[1] - t.ssr[2]) / t.ssr[2]) ≈ 112 rtol = 1e-12
    @test statistic(t) == t.F
    @test pvalue(t) == t.pvalues
    @test all(0 .<= t.pvalues .<= 1)
    @test occursin("no asymptotic p-value", sprint(show, t))
end

@testset "setar_test: reproducibility, monotone critical values" begin
    a = setar_test(LYNX10, 2; thDelay=0:1, nboot=20, rng=MersenneTwister(42))
    b = setar_test(LYNX10, 2; thDelay=0:1, nboot=20, rng=MersenneTwister(42))
    @test a.pvalues == b.pvalues
    @test a.critical == b.critical
    @test size(a.critical) == (3, 4)
    @test all(issorted(a.critical[i, :]) for i in 1:3)
    # lynx is clearly nonlinear: no bootstrap draw reaches the observed 1-vs-2 F
    @test a.pvalues[1] == 0
    @test a.pvalues[2] == 0
    @test a.nboot == 20
    @test_throws ArgumentError setar_test(LYNX10, 2; nboot=0)
end

@testset "reduction: LSTAR with γ → ∞ is SETAR" begin
    s = fit_setar(LYNX10, 2; thDelay=1, warn=false)
    Y, lag, z, _ = TSAnalytics._tar_embed(LYNX10, 2, 1)
    X = hcat(ones(112), lag(1), lag(2))
    th = s.thresholds[1]
    c = (th + minimum(z[z .> th])) / 2      # between the regimes, off the tie
    for γ in (1e4, 1e6)
        ssr, b, G = TSAnalytics._lstar_profile(Y, X, z, γ, c)
        @test ssr ≈ s.ssr rtol = 1e-8
        @test b[1:3] ≈ coef(s)[1:3] rtol = 1e-7
        @test b[1:3] .+ b[4:6] ≈ coef(s)[4:6] rtol = 1e-7
    end
    @test TSAnalytics._lstar_profile(Y, X, z, 1.0, c)[1] > s.ssr
    @test TSAnalytics._lstar_profile(Y, X, z, 1e6, c)[3] == (z .> th)
end

@testset "reduction: one regime is the pooled AR" begin
    # a threshold above every z puts all rows in the low regime
    _, lag, z, _ = TSAnalytics._tar_embed(LYNX10, 2, 0)
    Y = LYNX10[3:end]
    X = hcat(ones(112), lag(1), lag(2))
    b = X \ Y
    @test b ≈ [1.057600456442, 1.384237711639, -0.747775720384] atol = 1e-11
    @test sum(abs2, Y .- X * b) ≈ 5.78258084172 rtol = 1e-10
    # tsDyn's linear() uses yet another variance, (SSR/n) · rows/(rows − k):
    # backed out exactly, the implied divisor is 109 · 114/112 = 110.946...
    se = sqrt.(sum(abs2, Y .- X * b) / 114 * 112 / 109 .* diag(inv(X'X)))
    @test se ≈ [0.12083699374, 0.06333183622, 0.06338507067] rtol = 1e-9
    # the incremental profile starts from the pooled fit: a cut with an empty
    # high regime is refused, not silently fitted
    @test_throws ArgumentError fit_setar(LYNX10, 2; threshold=maximum(z) + 1, warn=false)
    # and SETAR at its threshold improves on it by 24.8%
    s = fit_setar(LYNX10, 2; thDelay=1, warn=false)
    @test 1 - s.ssr / 5.78258084172 ≈ 0.248 atol = 1e-3
    @test 114 * log(5.78258084172 / 114) + 2 * 3 ≈ -333.873712237 rtol = 1e-10
    @test tsdyn_ic(s).aic < -333.873712237
end

@testset "supplied vs searched threshold: same SSR, k = 6 vs 7" begin
    s = fit_setar(LYNX10, 2; thDelay=1, warn=false)
    g = fit_setar(LYNX10, 2; thDelay=1, threshold=s.thresholds[1], warn=false)
    @test g.ssr == s.ssr
    @test coef(g) == coef(s)
    @test dof(s) == 7
    @test dof(g) == 6
    @test tsdyn_ic(s).aic ≈ -358.373987591728 rtol = 1e-12
    @test tsdyn_ic(g).aic ≈ -360.373987591728 rtol = 1e-12
    @test isempty(g.profile)
    @test occursin("(given)", sprint(show, g))
end

@testset "refusals" begin
    @test_throws ArgumentError fit_setar(LYNX10, 2; thDelay=2)
    @test_throws ArgumentError fit_setar(LYNX10, 2; trim=0.5)
    @test_throws ArgumentError fit_setar(LYNX10, 2; trim=0.0)
    @test_throws ArgumentError fit_setar(LYNX10, 0)
    @test_throws ArgumentError fit_setar(vcat(LYNX10[1:10], NaN, LYNX10[12:end]), 2)
    @test_throws ArgumentError fit_setar(LYNX10[1:6], 2; trim=0.45)
    @test_throws ArgumentError fit_setar(LYNX10, 2; grid=:half)
    @test_throws ArgumentError fit_lstar(LYNX10, 2; thDelay=2)
    @test_throws ArgumentError fit_lstar(LYNX10, 2; trim=0.5)
    @test_throws ArgumentError predict(fit_setar(LYNX10, 2; warn=false), 0)
    @test_throws ArgumentError TSAnalytics._tar_grid(collect(1.0:3.0), 0.4)
    @test_throws ArgumentError select_setar(LYNX10, 2; criterion=:pooled)
end

@testset "unit-root warning in the high regime" begin
    f = @test_logs (:warn, r"possible unit root in the high regime") fit_setar(LYNX10, 2; thDelay=1)
    rts = TSAnalytics._setar_regime_roots(f)
    @test minimum(rts["high"]) ≈ 0.994262021909 rtol = 1e-10
    @test minimum(rts["low"]) ≈ 1.52777874135 rtol = 1e-10
    @test_logs fit_setar(LYNX10, 2; thDelay=1, warn=false)
    @test predict(f, 5) ≈ [3.3485758177, 2.9490750890, 2.4946750617, 2.4789330144, 2.6537089156] atol = 1e-10
    @test length(predict(f, 1)) == 1
end

@testset "performance: the incremental search scales" begin
    f = fit_setar(SETARC, 1; warn=false)            # compile
    t = @elapsed fit_setar(SETARC, 1; warn=false)
    @test t < 1.0
    a = @allocated fit_setar(SETARC, 1; warn=false)
    @test a < 5_000_000
    # 700 in-range values; tsDyn's truncated grid has the handoff's 699
    @test length(f.profile) == 700
    @test length(TSAnalytics._tar_grid(SETARC[1:999], 0.15; grid=:tsdyn)) == 699
    @test f.thresholds[1] ≈ 0.012779304500 atol = 1e-12
    @inferred TSAnalytics._tar_grid(collect(1.0:100.0), 0.15)
end
