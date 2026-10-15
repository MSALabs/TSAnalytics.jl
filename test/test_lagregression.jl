using DelimitedFiles
using LinearAlgebra: rank, I
using Statistics: mean, std

# Stage 12.3 -- lag-feature regression ("the ML reduction").
# handoff/stage 12/docs/stage-12.3-lag-feature-regression-handoff.md.
# References: mlforecast 1.1.0 (Apache-2.0), sktime 1.2.0 (BSD-3), forecastML
# 0.9.0 (MIT) -- all permissive. Figures from handoff/stage 12/ground-truth/,
# and (rolling features, Fourier) from mlforecast run on this machine.

const _LR_FX = joinpath(@__DIR__, "..", "handoff", "stage 12", "fixtures")
_lrfx(f) = readdlm(joinpath(_LR_FX, f), ','; skipstart=1)
const _LR_A = Float64.(_lrfx("fixture_a_ar2.csv")[:, 3])
const _LR_B = Float64.(_lrfx("fixture_b_seasonal.csv")[:, 3])
const _LR_C = _lrfx("fixture_c_exog.csv")
const _LR_CY = Float64.(_LR_C[1:132, 3])
const _LR_CX = Float64.(_LR_C[:, 4:5])

# core_ols.txt: arx-style OLS, mlforecast, sktime, forecastML all agree to 2e-15
const _LR_ACOEF = [1.010348499228958, 0.6026972101854623, -0.35837911855381488]
const _LR_AREC = [1.2806577561017716, 1.2243195830708189, 1.2892814985203096,
                  1.3486242885217865, 1.3611090285019425, 1.3473663296846201]

@testset "injected OLS = arx, fixture A" begin
    r = lag_regression(_LR_A, 2)
    m = arx(_LR_A, 2)
    @test r.strategy === :recursive
    @test r.names[1] == m.names == ["const", "y.L1", "y.L2"]
    @test r.nobs == [118]
    @test r.models[1] ≈ _LR_ACOEF atol = 1e-14
    @test m.coef ≈ _LR_ACOEF atol = 1e-14
    @test m.sigma2 ≈ 0.22541381721778347 rtol = 1e-14
    fr = forecast(r, 6)
    @test fr.point ≈ forecast(m, 6).point atol = 1e-14
    @test fr.point ≈ _LR_AREC atol = 1e-14
    for h in 1:6
        @test fr.point[h] ≈ _LR_AREC[h] atol = 2e-15
    end
    @test isempty(fr.levels)                       # no intervals from an injected model
    @test all(isnan, fr.se)
    @test predict(r, 6).point == fr.point
end

@testset "injected OLS = arx: lag subsets, trend, seasonal" begin
    # final_checks.txt: lags=[1,3] -- arx vs mlforecast 6.661e-16 coefficients
    r = lag_regression(_LR_A, [1, 3])
    m = arx(_LR_A, [1, 3])
    @test r.names[1] == ["const", "y.L1", "y.L3"]
    @test r.models[1] ≈ m.coef atol = 1e-14
    @test forecast(r, 6).point ≈ forecast(m, 6).point atol = 1e-14
    for (tr, seas) in ((:ct, false), (:c, true), (:n, false), (:t, true))
        kw = (; trend=tr, seasonal=seas, period=seas ? 12 : nothing)
        rr = lag_regression(_LR_B, 2; kw...)
        mm = arx(_LR_B, 2; kw...)
        @test rr.names[1] == mm.names
        @test rr.models[1] ≈ mm.coef rtol = 1e-12
        @test forecast(rr, 12).point ≈ forecast(mm, 12).point rtol = 1e-12
    end
end

@testset "recursive against mlforecast, sktime, forecastML" begin
    f = forecast(lag_regression(_LR_A, 2), 6).point
    mlf = [1.2806577561017727, 1.2243195830708207, 1.2892814985203114, 1.3486242885217883,
           1.3611090285019443, 1.3473663296846217]
    skt = [1.2806577561017729, 1.2243195830708209, 1.2892814985203114, 1.3486242885217883,
           1.3611090285019443, 1.3473663296846217]
    @test maximum(abs, f .- mlf) < 2e-15
    @test maximum(abs, f .- skt) < 2e-15
    r = lag_regression(_LR_A, 2)
    @test r.models[1][1] ≈ 1.0103484992289584 atol = 1e-15        # mlforecast intercept_
    @test r.models[1][2] ≈ 0.60269721018546285 atol = 1e-15       # lag1
    @test r.models[1][3] ≈ -0.35837911855381488 atol = 1e-15      # lag2
    @test r.models[1][1] ≈ 1.010348499228958 atol = 1e-15         # forecastML (Intercept)
    for h in 1:6
        @test f[h] ≈ mlf[h] atol = 2e-15
        @test f[h] ≈ skt[h] atol = 2e-15
    end
end

@testset "a non-OLS injected function works" begin
    ridge(λ) = ((X, y) -> (X' * X + λ * I) \ (X' * y))
    r0 = lag_regression(_LR_A, 2; fit=ridge(0.0))
    @test forecast(r0, 6).point ≈ _LR_AREC atol = 1e-12            # λ = 0 is OLS
    r1 = lag_regression(_LR_A, 2; fit=ridge(50.0))
    @test norm(r1.models[1]) < norm(r0.models[1])                   # shrinkage
    @test !(forecast(r1, 6).point ≈ _LR_AREC)
    # a model that is not a vector of coefficients: a constant (mean) predictor
    meanfit = (X, y) -> mean(y)
    meanpred = (m, X) -> fill(m, size(X, 1))
    rm = lag_regression(_LR_A, 2; fit=meanfit, predict=meanpred)
    @test forecast(rm, 6).point ≈ fill(mean(_LR_A[3:end]), 6)
    @test rm.models[1] isa Float64
    # the direct strategy refits the injected model per horizon
    rd = lag_regression(_LR_A, 2; strategy=:direct, horizon=3, fit=ridge(0.0))
    @test length(rd.models) == 3
    @test forecast(rd, 3).point ≈ forecast(lag_regression(_LR_A, 2; strategy=:direct, horizon=3), 3).point atol = 1e-12
    @test occursin("injected regressor", sprint(show, r1))
    @test occursin("least squares", sprint(show, lag_regression(_LR_A, 2)))
end

@testset "the injected contract and refusals" begin
    # forecastML rejects a fitted object; so do we
    @test_throws ArgumentError lag_regression(_LR_A, 2; fit=[1.0, 2.0])
    @test_throws ArgumentError lag_regression(_LR_A, 2; predict=3.0)
    badpred = (m, X) -> X * m .+ 0.0 |> x -> x[1:end-1]
    @test_throws ArgumentError lag_regression(_LR_A, 2; predict=badpred)
    matpred = (m, X) -> reshape(X * m, :, 1)
    @test_throws ArgumentError lag_regression(_LR_A, 2; predict=matpred)
    @test_throws ArgumentError lag_regression(_LR_A, 2; strategy=:dirrec)
    @test_throws ArgumentError lag_regression(_LR_A, 2; strategy=:multioutput)
    @test_throws ArgumentError lag_regression(_LR_A, 2; strategy=:direct)          # no horizon
    @test_throws ArgumentError lag_regression(_LR_A, 2; horizon=3)                  # recursive + horizon
    @test_throws ArgumentError lag_regression(_LR_A, 2; strategy=:direct, horizon=0)
    @test_throws ArgumentError lag_regression(_LR_A, 0)
    @test_throws ArgumentError lag_regression(_LR_A, [0, 1])
    @test_throws ArgumentError lag_regression(_LR_A, 2; trend=:quadratic)
    @test_throws ArgumentError lag_regression(_LR_A, 2; seasonal=true)
    @test_throws ArgumentError lag_regression(vcat(_LR_A[1:5], NaN, _LR_A[7:end]), 2)
    @test_throws ArgumentError lag_regression(_LR_A[1:3], 5)
    @test_throws ArgumentError forecast(lag_regression(_LR_A, 2), 0)
end

@testset "direct strategy: sktime windows_identical=False, every horizon" begin
    # strategies.txt: windows_identical=False -- each horizon on its own rows
    d = lag_regression(_LR_A, 2; strategy=:direct, horizon=6)
    target = [1.2806577561017729, 1.2287056917420438, 1.3146799275256662, 1.3142036491594953,
              1.3056022702505263, 1.3839921465580807]
    f = forecast(d, 6).point
    for h in 1:6
        @test f[h] ≈ target[h] atol = 1e-15
    end
    @test d.nobs == [118, 117, 116, 115, 114, 113]
    @test d.names[6] == ["const", "y.L6", "y.L7"]                  # origin-anchored
    # the h = 1 model is the recursive model
    @test d.models[1] ≈ lag_regression(_LR_A, 2).models[1] atol = 1e-15
    @test_throws ArgumentError forecast(d, 7)
end

@testset "direct strategy: same_rows = sktime's make_reduction default" begin
    ds = lag_regression(_LR_A, 2; strategy=:direct, horizon=6, same_rows=true)
    target = [1.2578665415283894, 1.2136302750556682, 1.3070101870553992, 1.3084783854924888,
              1.3031988298866279, 1.3839921465580807]
    f = forecast(ds, 6).point
    for h in 1:6
        @test f[h] ≈ target[h] atol = 1e-14
    end
    @test ds.nobs == fill(113, 6)
    # strategies.txt: h=1 coefficients int=1.005639629996196, -0.3810578737131684, 0.6182...
    @test ds.models[1] ≈ [1.005639629996196, 0.61823215648324448, -0.3810578737131684] atol = 1e-14
    # h = fh_max is where every truncation, and forecastML, agree (2.22e-16)
    @test f[6] ≈ 1.38399214655808045 atol = 4.5e-16      # 2 ulp of the printed figure
    @test f[6] == forecast(lag_regression(_LR_A, 2; strategy=:direct, horizon=6), 6).point[6]
    # fixture B: same pattern
    db = forecast(lag_regression(_LR_B, 2; strategy=:direct, horizon=6), 6).point
    @test db ≈ [16.640279114134177, 17.268917602737965, 16.028342799094084, 13.2325695354236,
                9.731659655738234, 6.5097384142679067] atol = 1e-12
end

@testset "direct differs from recursive" begin
    d = forecast(lag_regression(_LR_A, 2; strategy=:direct, horizon=6, same_rows=true), 6).point
    rc = forecast(lag_regression(_LR_A, 2), 6).point
    @test maximum(abs, d .- rc) ≈ 5.79e-2 atol = 5e-4
    @test d[1] != rc[1]
    # with each horizon on its own rows, h = 1 coincides; later steps do not
    d2 = forecast(lag_regression(_LR_A, 2; strategy=:direct, horizon=6), 6).point
    @test d2[1] ≈ rc[1] atol = 1e-15
    @test maximum(abs, d2[2:end] .- rc[2:end]) > 1e-3
    @test d2[6] != rc[6]
    @test length(d2) == 6
end

# OLS fits by hand, for the identities that are asserted rather than built
_lr_ols(X, y) = X \ y

@testset "DirRec = Direct and multi-output = Direct under OLS (identities)" begin
    y = _LR_A
    n, H = length(y), 6
    rows = 3:(n - H + 1)                     # origins with all H targets (same_rows)
    X0 = hcat(ones(length(rows)), y[rows .- 1], y[rows .- 2])
    direct = lag_regression(y, 2; strategy=:direct, horizon=H, same_rows=true)
    # multi-output: one least-squares fit of the H-column target -- column-wise identical
    Y = reduce(hcat, [y[rows .+ (h - 1)] for h in 1:H])
    B = X0 \ Y
    @test maximum(abs, B .- reduce(hcat, direct.models)) < 1e-13
    # DirRec: horizon h also gets the in-sample predictions of horizons 1..h-1
    Xa = copy(X0)
    preds_dirrec = Float64[]
    xo = [1.0, y[n], y[n - 1]]
    xaug = copy(xo)
    for h in 1:H
        b = Xa \ Y[:, h]
        push!(preds_dirrec, dot(xaug, b))
        Xa = hcat(Xa, Xa * b)
        push!(xaug, preds_dirrec[end])
    end
    @test maximum(abs, preds_dirrec .- forecast(direct, H).point) < 1e-12
    @test size(Xa, 2) == 3 + H                # it really did grow, as sktime's n_features_in_
    @test forecast(direct, H).point ≈ [1.2578665415283894, 1.2136302750556682, 1.3070101870553992,
                                       1.3084783854924888, 1.3031988298866279,
                                       1.3839921465580807] atol = 1e-14
end

@testset "past-only regressors and the sample bookkeeping" begin
    m = arx(_LR_CY, 1; exog=_LR_CX[1:132, 1], past_exog=_LR_CX[1:132, 2], past_lags=1)
    @test m.names == ["const", "y.L1", "x1", "past1.L1"]
    @test m.nobs == 131
    # exog_py.txt (e): mlforecast transform_exog(lags=[1]) with x_known contemporaneous
    f = forecast(m, 6; newexog=_LR_CX[133:138, 1], newpast=_LR_CX[133:138, 2]).point
    target = [-3.2315233566935162, 1.090407250567387, 1.7924879588059655, 2.7644426438931902,
              3.7447949003299863, 3.17055308218232]
    for h in 1:6
        @test f[h] ≈ target[h] atol = 1e-13
    end
    # the past-only lag lengthens the hold-back exactly like y's lags
    m3 = arx(_LR_CY, 1; past_exog=_LR_CX[1:132, 2], past_lags=3)
    @test m3.nobs == 132 - 3
    @test m3.names == ["const", "y.L1", "past1.L1", "past1.L2", "past1.L3"]
    # h <= smallest past lag needs no future values
    @test length(forecast(m3, 1).point) == 1
    @test_throws ArgumentError arx(_LR_CY, 1; past_exog=_LR_CX[1:132, 2], past_lags=0)
    @test_throws ArgumentError arx(_LR_CY, 1; past_exog=_LR_CX[1:100, 2])
end

@testset "known-future regressors required at forecast, refused by name" begin
    m = arx(_LR_CY, 1; exog=_LR_CX[1:132, :])
    @test m.coef ≈ [0.0018631539340603354, 0.38423957639220185, 2.0268793035159347,
                    1.5035550438207026] atol = 1e-14                 # mlforecast, 1.110e-15
    f = forecast(m, 6; newexog=_LR_CX[133:138, :]).point
    target = [-4.8391651439814414, 1.0561926855297166, 2.7677991988183308, 4.9844189934247147,
              6.1885089853832485, 3.9936585139939313]
    @test f ≈ target atol = 1e-13
    @test forecast(m, _LR_CX[133:138, :], 6).point == f             # positional form
    e1 = try forecast(m, 6); catch e; e end
    @test e1 isa ArgumentError && occursin("needs 6 future rows", e1.msg) && occursin("newexog", e1.msg)
    # mlforecast silently accepts 3 of 6 rows (exog_py.txt case d); we refuse
    e2 = try forecast(m, 6; newexog=_LR_CX[133:135, :]); catch e; e end
    @test e2 isa ArgumentError && occursin("has 3 rows", e2.msg) && occursin("columns 1, 2", e2.msg)
    e3 = try forecast(m, 2; newexog=_LR_CX[133:134, 1]); catch e; e end
    @test e3 isa ArgumentError && occursin("1 column", e3.msg)
    @test_throws ArgumentError forecast(arx(_LR_CY, 2), 2; newexog=_LR_CX[133:134, :])
    bad = copy(_LR_CX[133:138, :]); bad[4, 2] = NaN
    @test_throws ArgumentError forecast(m, 6; newexog=bad)
    @test length(forecast(m, 3; newexog=_LR_CX[133:138, :]).point) == 3   # extra rows are fine
end

@testset "arx + fourier_terms now forecasts" begin
    F, nm = fourier_terms(length(_LR_B), 12, 2)
    Ff, _ = fourier_terms(length(_LR_B), 12, 2; h=6)
    m = arx(_LR_B, 2; exog=F)
    @test m.names == ["const", "y.L1", "y.L2", "x1", "x2", "x3", "x4"]
    f = forecast(m, 6; newexog=Ff)
    # mlforecast, the same four columns as known-future regressors (run on this machine)
    target = [17.104113552609292, 19.03651181480809, 19.592531224668051, 18.689102750370871,
              16.627252010125314, 13.972759647487813]
    @test f.point ≈ target rtol = 1e-13
    @test size(f.lower) == (6, 2)                                    # arx keeps its intervals
    @test all(f.lower[:, 2] .< f.point .< f.upper[:, 2])
    # the regressors' future values are known: the se is arx's psi-weight se
    @test f.se ≈ forecast(arx(_LR_B, 2; exog=F), 6; newexog=Ff).se
    @test_throws ArgumentError forecast(m, 6)
    # and through lag_regression
    r = lag_regression(_LR_B, 2; exog=F)
    @test forecast(r, 6; newexog=Ff).point ≈ f.point rtol = 1e-12
    @test_throws ArgumentError forecast(r, 6)
    @test length(nm) == 4
    @test size(Ff) == (6, 4)
end

# the leakage probe: perturb one observation, rebuild the design, and see
# which rows' FEATURES move -- parameterised by strategy (handoff §6.2)
function _lr_probe(y, t0; strategy=:recursive, H=1, features=RollingFeature[])
    d0 = lag_regression(y, 2; strategy=strategy, horizon=strategy === :direct ? H : nothing,
                        features=features)
    y1 = copy(y); y1[t0] += 1000
    d1 = lag_regression(y1, 2; strategy=strategy, horizon=strategy === :direct ? H : nothing,
                        features=features)
    moved = Int[]
    for h in eachindex(d0.names)
        reach = maximum(_lr_reach(d0, h))
        X0 = TSAnalytics._arx_design(y, [1, 2], reach, :c, false, nothing, nothing, nothing, Int[];
                                     shift=strategy === :direct ? h - 1 : 0,
                                     extra=[(TSAnalytics._rf_name(f, strategy === :direct ? h - 1 : 0),
                                             TSAnalytics._rf_eval(f, strategy === :direct ? h - 1 : 0))
                                            for f in features])[1]
        X1 = TSAnalytics._arx_design(y1, [1, 2], reach, :c, false, nothing, nothing, nothing, Int[];
                                     shift=strategy === :direct ? h - 1 : 0,
                                     extra=[(TSAnalytics._rf_name(f, strategy === :direct ? h - 1 : 0),
                                             TSAnalytics._rf_eval(f, strategy === :direct ? h - 1 : 0))
                                            for f in features])[1]
        r = t0 - reach                       # the design row whose TARGET is y[t0]
        1 <= r <= size(X0, 1) && X0[r, :] != X1[r, :] && push!(moved, h)
    end
    return moved
end
_lr_reach(d, h) = (2 + (d.strategy === :direct ? h - 1 : 0),
                   (isempty(d.features) ? 0 : maximum(TSAnalytics._rf_reach, d.features)) +
                   (d.strategy === :direct ? h - 1 : 0))

@testset "leakage probe: recursive and direct" begin
    for t0 in (20, 60, 100)
        @test isempty(_lr_probe(_LR_A, t0))
        @test isempty(_lr_probe(_LR_A, t0; strategy=:direct, H=6))
        @test isempty(_lr_probe(_LR_A, t0; features=[rolling_feature(:mean, 1, 3)]))
        @test isempty(_lr_probe(_LR_A, t0; strategy=:direct, H=4,
                                features=[rolling_feature(:max, 1, 4), rolling_feature(:expanding_mean, 1)]))
    end
    # and the probe can see a leak: a hand-built lag-0 window is y_t itself
    y = _LR_A
    leak = [mean(y[t:t]) for t in 3:length(y)]                 # sktime's mean over [[0,1]]
    @test leak == y[3:end]
    X = hcat(ones(length(leak)), leak)
    b = X \ y[3:end]
    @test 1 - sum(abs2, y[3:end] .- X * b) / sum(abs2, y[3:end] .- mean(y[3:end])) ≈ 1 atol = 1e-12
end

@testset "leakage probe: the multi-output form (perturb the first target)" begin
    # in multi-output, the row with origin t legitimately has y[t] as a feature
    # and targets y[t+1..t+H]: the right probe perturbs y[t+1], not y[t]
    y = _LR_A
    H = 6
    rows = 3:(length(y) - H + 1)
    feats(v) = hcat(ones(length(rows)), v[rows .- 1], v[rows .- 2])
    for t0 in (30, 70)
        y1 = copy(y); y1[t0] += 1000           # = the first target of origin row t0 - 1 ...
        r = findfirst(==(t0), rows)            # row whose h = 1 target is y[t0]
        @test feats(y)[r, :] == feats(y1)[r, :]                    # clean
        # the naive probe (target == y_t) would flag the NEXT row, which uses y[t0] legitimately
        @test feats(y)[r + 1, :] != feats(y1)[r + 1, :]
    end
    @test length(rows) == 113
    @test size(feats(y)) == (113, 3)
end

@testset "lag 0 rejected at construction" begin
    e = try rolling_feature(:mean, 0, 1); catch err; err end
    @test e isa ArgumentError && occursin("leak", e.msg)
    @test_throws ArgumentError rolling_feature(:mean, -1, 3)
    @test_throws ArgumentError rolling_feature(:median, 1, 3)
    @test_throws ArgumentError rolling_feature(:std, 1, 1)
    @test_throws ArgumentError rolling_feature(:expanding_mean, 1, 3)
    @test_throws ArgumentError rolling_feature(:mean, 1, 0)
    e2 = try arx(_LR_A, [0, 1]); catch err; err end
    @test e2 isa ArgumentError && occursin("leak", e2.msg)
    @test rolling_feature(:mean, 1, 3) == RollingFeature(:mean, 1, 3)
end

@testset "rolling features: mlforecast, recomputed under recursion" begin
    # mlforecast 1.1.0, lags=[1,2], lag_transforms={1:[RollingMean(3)]}, LinearRegression
    r = lag_regression(_LR_A, 2; features=[rolling_feature(:mean, 1, 3)])
    @test r.nobs == [117]
    @test r.models[1] ≈ [1.0255590684077869, 0.60233039890791973, -0.36717520348224542,
                         0.0029322768363984919] atol = 1e-13
    @test forecast(r, 6).point ≈ [1.2841478577721368, 1.2320682520180051, 1.3001449216672922,
                                  1.3600211755923197, 1.3711645589973378, 1.3560274381431845] atol = 1e-14
    # four transforms at once, including std (n-1) and an expanding mean
    feats = [rolling_feature(:mean, 1, 3), rolling_feature(:std, 1, 3),
             rolling_feature(:max, 1, 3), rolling_feature(:expanding_mean, 1)]
    r2 = lag_regression(_LR_A, 2; features=feats)
    @test r2.names[1] == ["const", "y.L1", "y.L2", "roll_mean(L1,w3)", "roll_std(L1,w3)",
                          "roll_max(L1,w3)", "exp_mean(L1)"]
    @test r2.models[1] ≈ [1.7911084470276581, 0.58303909332442061, -0.2326214190710576,
                          -0.78655752969263015, -1.1033230404901653, 0.48472437706115135,
                          -0.1856900229238537] atol = 1e-12
    @test forecast(r2, 6).point ≈ [1.3274967358289194, 1.3167358851871704, 1.5056731563214458,
                                   1.6391803667006266, 1.5939344307321286, 1.5672815066032597] atol = 1e-13
    # recomputed, not frozen: freezing the last observed window gives a different path
    frozen = let b = r.models[1], yy = copy(_LR_A), w = mean(_LR_A[end-2:end]), out = Float64[]
        for h in 1:6
            v = b[1] + b[2] * yy[end] + b[3] * yy[end - 1] + b[4] * w
            push!(out, v); push!(yy, v)
        end
        out
    end
    @test frozen[1] ≈ forecast(r, 6).point[1] atol = 1e-14           # step 1 is the same
    @test abs(frozen[4] - forecast(r, 6).point[4]) > 1e-6            # later steps are not
    @test r2.nobs == [117]
end

@testset "rank check on every fixture design" begin
    for (y, lags) in ((_LR_A, 2), (_LR_B, 2), (_LR_CY, 1))
        X = TSAnalytics._arx_design(y, collect(1:lags), lags, :c, false, nothing, nothing, nothing, Int[])[1]
        @test rank(X) == size(X, 2)
    end
    XB = TSAnalytics._arx_design(_LR_B, [1, 2], 2, :c, true, 12, nothing, nothing, Int[])[1]
    @test rank(XB) == size(XB, 2)
    XC = TSAnalytics._arx_design(_LR_CY, [1], 1, :c, false, nothing, _LR_CX[1:132, :], nothing, Int[])[1]
    @test rank(XC) == size(XC, 2)
    # a deterministic sinusoid's lags span two dimensions (handoff §6.3): refused
    s = sin.(2π .* (1:100) ./ 12)
    @test_throws ArgumentError arx(s, 6; trend=:n)
    @test_throws ArgumentError lag_regression(s, 6; trend=:n)
    # D33: a dummy for every month plus a constant is rank-deficient; arx refuses
    # where mlforecast returns a silent minimum-norm solution
    D = Float64[mod1(t, 12) == k for t in 1:length(_LR_B), k in 1:12]
    @test_throws ArgumentError arx(_LR_B, 2; exog=D)
    @test_throws ArgumentError lag_regression(_LR_B, 2; exog=D)
end

@testset "blocked cross-validation (forecastML create_windows)" begin
    b = blocked_window_split(117; window_length=20)
    @test length(b) == 6
    @test [length(f[1]) for f in b] == [97, 97, 97, 97, 97, 100]      # 97 = 117 - 20
    @test [length(f[2]) for f in b] == [20, 20, 20, 20, 20, 17]
    @test all(sort(vcat(f...)) == 1:117 for f in b)                   # train and block partition
    @test b[2][1] == vcat(1:20, 41:117)                               # trains on both sides
    # skip=10: forecastML's blocks start at 4, 34, 64, 94 (its rows begin at 4)
    @test [first(f[2]) + 3 for f in blocked_window_split(117; window_length=20, skip=10)] == [4, 34, 64, 94]
    @test length(blocked_window_split(117; window_length=20, include_partial=false)) == 5
    @test_throws ArgumentError blocked_window_split(10; window_length=0)
    @test_throws ArgumentError blocked_window_split(10; window_length=10)
    @test_throws ArgumentError blocked_window_split(10; window_length=3, skip=-1)
end

@testset "mdape and rmsse" begin
    @test mdape([100.0, 200.0, 300.0], [110.0, 190.0, 0.0]) ≈ 10.0
    @test mdape([100.0, 200.0, 300.0], [110.0, 190.0, 0.0]; as_percentage=false) ≈ 0.1
    @test mape([100.0, 200.0, 300.0], [110.0, 190.0, 0.0]) > 30          # the mean is dragged
    @test mdape([1.0, 2.0], [1.0, 2.0]) == 0.0
    @test_throws ArgumentError mdape([1.0], [1.0, 2.0])
    @test rmsse([3.0, 5.0], [2.0, 5.0], [1.0, 2.0, 4.0]) ≈ sqrt(0.5 / 2.5)
    @test rmsse([3.0, 5.0], [2.0, 5.0], [1.0, 2.0, 4.0, 5.0, 7.0]; sp=2) ≈
          sqrt(0.5 / (sum(abs2, [4.0, 5, 7] .- [1.0, 2, 4]) / 3))
    @test_throws ArgumentError rmsse([1.0], [1.0], [1.0])
    @test_throws ArgumentError rmsse([1.0], [1.0, 2.0], [1.0, 2.0, 3.0])
    @test_throws ArgumentError rmsse([1.0], [1.0], [1.0, 2.0]; sp=0)
end

@testset "tscv refit policy (mlforecast cross_validation)" begin
    fitf(tr) = lag_regression(tr, 2)
    fcf(m, tr, h) = forecast(m, h; history=tr)
    e1 = tscv(_LR_A, fitf, fcf; h=1:3, initial=108, step_length=3)
    # cv_py.txt: cutoffs 2018-12 .. 2019-09 (t = 108, 111, 114, 117), h = 3
    mlf = [0.6732143146 1.770665724 1.953082237; 1.499333354 1.3845637 1.294849728;
           1.66049942 1.454267695 1.276193671; 1.412964122 1.259164938 1.255034281]
    pred = [_LR_A[108 + 3(i - 1) + j] - e1[i, j] for i in 1:4, j in 1:3]
    @test pred ≈ mlf atol = 1e-9
    e0 = tscv(_LR_A, fitf, fcf; h=1:3, initial=108, step_length=3, refit=false)
    e2 = tscv(_LR_A, fitf, fcf; h=1:3, initial=108, step_length=3, refit=2)
    @test maximum(abs, e0 .- e1) ≈ 0.05061 atol = 5e-6
    @test maximum(abs, e2 .- e1) ≈ 0.02185 atol = 5e-6
    @test e0[1, :] == e1[1, :]                                         # first fold is fitted
    @test_throws ArgumentError tscv(_LR_A, fitf, fcf; h=1, initial=108, refit=0)
    # the one-argument form is unchanged
    @test tscv(_LR_A, (tr, h) -> forecast(arx(tr, 2), h); h=1:3, initial=108, step_length=3) ≈ e1 atol = 1e-12
end

@testset "performance" begin
    lag_regression(_LR_A, 2); forecast(lag_regression(_LR_A, 2), 6)       # compile
    t = @elapsed for _ in 1:50
        forecast(lag_regression(_LR_A, 2), 6)
    end
    @test t / 50 < 0.01
    t2 = @elapsed lag_regression(_LR_A, 2; strategy=:direct, horizon=24)
    @test t2 < 1.0
    a = @allocated forecast(lag_regression(_LR_A, 2), 6)
    @test a < 1_000_000
    @inferred TSAnalytics._arx_lagset(3, "lags")
    @test length(forecast(lag_regression(_LR_A, 2), 200).point) == 200
end

@testset "regressors under lag_regression, both strategies" begin
    yC, X = _LR_CY, _LR_CX
    # recursive with both kinds = arx with both kinds
    r = lag_regression(yC, 1; exog=X[1:132, 1], past_exog=X[1:132, 2], past_lags=1)
    m = arx(yC, 1; exog=X[1:132, 1], past_exog=X[1:132, 2], past_lags=1)
    @test r.names[1] == m.names
    @test r.models[1] ≈ m.coef atol = 1e-13
    fr = forecast(r, 6; newexog=X[133:138, 1], newpast=X[133:138, 2]).point
    @test fr ≈ forecast(m, 6; newexog=X[133:138, 1], newpast=X[133:138, 2]).point atol = 1e-13
    # direct: past-only lags shift past the horizon, so no future past values are needed
    d = lag_regression(yC, 1; strategy=:direct, horizon=4, exog=X[1:132, 1],
                       past_exog=X[1:132, 2], past_lags=1)
    @test d.names[3] == ["const", "y.L3", "x1", "past1.L3"]
    fd = forecast(d, 4; newexog=X[133:136, 1]).point
    @test length(fd) == 4
    @test all(isfinite, fd)
    # direct h = 1 is the recursive one-step model
    @test fd[1] ≈ fr[1] atol = 1e-13
    @test d.models[1] ≈ r.models[1] atol = 1e-13
    # known-future values are still required at every horizon
    @test_throws ArgumentError forecast(d, 4)
    @test_throws ArgumentError forecast(d, 4; newexog=X[133:134, 1])
    # the direct model's design rows shrink by one per horizon, as y's do
    @test d.nobs == [131, 130, 129, 128]
end

@testset "forecast from another history" begin
    r = lag_regression(_LR_A[1:100], 2)
    full = forecast(r, 3; history=_LR_A).point
    # the same coefficients, applied from the end of the longer series
    b = r.models[1]
    yy = copy(_LR_A)
    hand = Float64[]
    for h in 1:3
        v = b[1] + b[2] * yy[end] + b[3] * yy[end - 1]
        push!(hand, v); push!(yy, v)
    end
    @test full ≈ hand atol = 1e-14
    @test full != forecast(lag_regression(_LR_A, 2), 3).point        # not refitted
    @test forecast(r, 3; history=_LR_A[1:100]).point == forecast(r, 3).point
    @test_throws ArgumentError forecast(r, 3; history=_LR_A[1:2])
    rx = lag_regression(_LR_CY, 1; exog=_LR_CX[1:132, :])
    @test_throws ArgumentError forecast(rx, 2; history=_LR_CY, newexog=_LR_CX[133:134, :])
    d = lag_regression(_LR_A[1:100], 2; strategy=:direct, horizon=3)
    @test length(forecast(d, 3; history=_LR_A).point) == 3
end

@testset "tscv against mlforecast, prediction by prediction" begin
    fitf(tr) = lag_regression(tr, 2)
    fcf(m, tr, h) = forecast(m, h; history=tr)
    e = tscv(_LR_A, fitf, fcf; h=1:3, initial=108, step_length=3)
    mlf = [0.6732143146, 1.770665724, 1.953082237, 1.499333354, 1.3845637, 1.294849728,
           1.66049942, 1.454267695, 1.276193671, 1.412964122, 1.259164938, 1.255034281]
    k = 0
    for i in 1:4, j in 1:3
        k += 1
        @test _LR_A[108 + 3(i - 1) + j] - e[i, j] ≈ mlf[k] atol = 1e-9
    end
    @test size(e) == (4, 3)
end

@testset "display and accessors" begin
    r = lag_regression(_LR_A, 2; features=[rolling_feature(:mean, 1, 3)])
    s = sprint(show, r)
    @test occursin("strategy = recursive", s)
    @test occursin("roll_mean(L1,w3)", s)
    d = lag_regression(_LR_A, 2; strategy=:direct, horizon=3)
    @test occursin("horizons 1:3", sprint(show, d))
    @test occursin("118/117/116", sprint(show, d))
    @test nobs(r) == 117
    @test nobs(d) == 118
    f = forecast(r, 4)
    @test occursin("lag regression (recursive", f.model_name)
    @test occursin("4 steps ahead", sprint(show, f))
    m = arx(_LR_CY, 1; exog=_LR_CX[1:132, 1], past_exog=_LR_CX[1:132, 2])
    @test m.exog == reshape(_LR_CX[1:132, 1], :, 1)
    @test m.past_lags == [1]
end
