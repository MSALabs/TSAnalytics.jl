using DelimitedFiles, LinearAlgebra, StatsAPI

@testset "arx matched cases (exact statsmodels validation)" begin
    # handoff/stage-5.1-arx-handoff.md section 6, independently re-verified
    # against real statsmodels.tsa.ar_model.AutoReg in this session (not
    # just transcribed) -- including re-deriving the exact se/sigma2/loglik/
    # aic/bic formulas from scratch, which caught a real bug the handoff's
    # own proposed code had (see arx's docstring "Standard errors do NOT
    # come directly from _ols" note).
    y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1, header=false))
    @test length(y) == 100
    x1 = vec(readdlm(TSAnalytics.ARX_EXOG_X1, ','; skipstart=1, header=false))
    y2 = vec(readdlm(TSAnalytics.ARX_EXOG_Y2, ','; skipstart=1, header=false))

    # Case A: trend=:n
    mA = arx(y, 2; trend=:n)
    @test isapprox(mA.coef, [0.5410303763560437, 0.25630630950482997]; atol=1e-9)
    @test mA.names == ["y.L1", "y.L2"]
    @test isapprox(sqrt.(diag(mA.vcov)), [0.09785471177309289, 0.09857897662162088]; atol=1e-8)
    @test isapprox(mA.sigma2, 0.9895445224634959; atol=1e-9)
    @test isapprox(mA.loglik, -138.54096077201433; atol=1e-6)
    @test isapprox(mA.aic, 283.08192154402866; atol=1e-6)
    @test isapprox(mA.bic, 290.8368239800404; atol=1e-6)
    @test mA.nobs == 98

    # Case B: trend=:ct
    mB = arx(y, 2; trend=:ct)
    @test isapprox(mB.coef,
                    [0.07877254380859917, -0.0008544483456752027, 0.5386482617920445, 0.253652866319518];
                    atol=1e-9)
    @test mB.names == ["const", "trend", "y.L1", "y.L2"]

    # Case C: exog (single regressor, y2 = y + 0.5*x1)
    mC = arx(y2, 2; trend=:c, exog=x1)
    @test isapprox(mC.coef,
                    [0.008939226282289953, 0.46309424275362754, 0.3280948131256025, 0.6628465276267352];
                    atol=1e-9)
    @test mC.names == ["const", "y.L1", "y.L2", "x1"]  # exog after lags, matching AutoReg's own order

    # Case D: lags=[1,3], a genuine subset skipping lag 2
    mD = arx(y, [1, 3]; trend=:c)
    @test isapprox(mD.coef, [0.02622096843720014, 0.6772447645380137, 0.07250312505600517]; atol=1e-9)
    @test mD.names == ["const", "y.L1", "y.L3"]

    # Case E: hold_back=10 -- flagged in the handoff as needing careful
    # re-verification of the nobs/indexing arithmetic; confirmed exact
    mE = arx(y, 2; trend=:c, hold_back=10)
    @test mE.nobs == 90
    @test isapprox(mE.coef, [-0.007517213709204509, 0.512751032062308, 0.2761702698395734]; atol=1e-9)

    # decomposition/consistency identity: coef[1] is y.L1 (paired with the
    # lag-1 predictor y[2:end-1]), coef[2] is y.L2 (paired with y[1:end-2])
    @test isapprox(y[3:end], mA.resid .+ mA.coef[1] .* y[2:end-1] .+ mA.coef[2] .* y[1:end-2]; atol=1e-8)
end

@testset "arx seasonal dummies (exact statsmodels validation)" begin
    # Real, substantive bug the handoff's own proposed code had: AutoReg's
    # seasonal dummies start at season 2 (season 1 is the implicit
    # baseline), named "s(2,period)" etc -- not a dummy for every season
    # starting at 1 named "seasonal.$s". Confirmed by execution: these are
    # ALGEBRAICALLY DIFFERENT parameterizations (different reference
    # category), not just a naming difference.
    y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1, header=false))
    m = arx(y, 1; trend=:c, seasonal=true, period=4)
    @test isapprox(m.coef,
                    [-0.027925308153237908, 0.053428425967817515, -0.025972527704870033,
                     0.2167558786094829, 0.7230130172844552]; atol=1e-9)
    @test m.names == ["const", "s(2,4)", "s(3,4)", "s(4,4)", "y.L1"]
end

@testset "arx StatsAPI contract" begin
    y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1, header=false))
    m = arx(y, 2; trend=:c)
    @test m isa TSAnalytics.UnivariateModel
    @test StatsAPI.coef(m) == m.coef
    @test StatsAPI.vcov(m) == m.vcov
    @test isapprox(StatsAPI.stderror(m), sqrt.(diag(m.vcov)); atol=1e-12)
    @test StatsAPI.residuals(m) == m.resid
    @test StatsAPI.nobs(m) == m.nobs
    @test StatsAPI.loglikelihood(m) == m.loglik
    @test StatsAPI.aic(m) == m.aic
    @test StatsAPI.bic(m) == m.bic
end

@testset "arx show / CoefTable" begin
    y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1, header=false))
    m = arx(y, 2; trend=:ct)
    io = IOBuffer()
    show(io, m)
    s = String(take!(io))
    @test occursin("AR(2)", s)
    @test occursin("ct trend", s)
    @test occursin("Coef.", s)
    @test occursin("Std. Error", s)
    @test occursin("Pr(>|z|)", s)
    @test occursin("Lower 95%", s)
    @test occursin("AIC", s)
    @test occursin("BIC", s)
    @test occursin("Log-likelihood", s)
    for nm in m.names
        @test occursin(nm, s)
    end

    # trend=:n omits the "with ... trend" clause entirely
    m_n = arx(y, 2; trend=:n)
    io = IOBuffer()
    show(io, m_n)
    @test !occursin("trend", String(take!(io)))
end

@testset "arx parameter coverage and error paths" begin
    y = vec(readdlm(TSAnalytics.AR2_ARX, ','; skipstart=1, header=false))
    x1 = vec(readdlm(TSAnalytics.ARX_EXOG_X1, ','; skipstart=1, header=false))

    # every trend value
    for trend in (:n, :c, :t, :ct)
        m = arx(y, 2; trend=trend)
        @test all(isfinite, m.coef)
        @test m.trend == trend
        nextra = trend == :n ? 0 : trend == :ct ? 2 : 1
        @test length(m.names) == nextra + 2
    end

    # lags as Integer vs equivalent explicit Vector give identical results
    m_int = arx(y, 3; trend=:c)
    m_vec = arx(y, [1, 2, 3]; trend=:c)
    @test m_int.coef == m_vec.coef
    @test m_int.names == m_vec.names

    # every hold_back >= maxlag
    for hb in (2, 5, 10, 20)
        m = arx(y, 2; trend=:c, hold_back=hb)
        @test m.nobs == length(y) - hb
    end

    # method=:qr vs :cholesky agree
    m_qr = arx(y, 2; trend=:c, method=:qr)
    m_chol = arx(y, 2; trend=:c, method=:cholesky)
    @test isapprox(m_qr.coef, m_chol.coef; atol=1e-8)
    @test isapprox(m_qr.vcov, m_chol.vcov; atol=1e-8)

    # exog as a Matrix (single column) matches exog as a Vector
    m_vec_exog = arx(y, 2; trend=:c, exog=x1)
    m_mat_exog = arx(y, 2; trend=:c, exog=reshape(x1, :, 1))
    @test m_vec_exog.coef == m_mat_exog.coef
    @test m_vec_exog.names == m_mat_exog.names

    # exog with multiple columns
    x2 = reverse(x1)
    m_multi = arx(y, 2; trend=:c, exog=hcat(x1, x2))
    @test m_multi.names[end-1:end] == ["x1", "x2"]
    @test length(m_multi.coef) == 5  # const + y.L1 + y.L2 + x1 + x2

    # every seasonal period value
    for period in (3, 4, 6)
        m = arx(y, 1; trend=:c, seasonal=true, period=period)
        @test length(m.names) == 1 + (period - 1) + 1  # const + (period-1) dummies + y.L1
        @test all(isfinite, m.coef)
    end

    # container-agnostic
    m1 = arx(y, 2; trend=:c)
    m2 = arx(Float32.(y), 2; trend=:c)
    @test isapprox(m1.coef, m2.coef; atol=1e-4)
    # a plain AbstractRange, trend=:n avoids the deliberate singular-design
    # case exercised separately below (trend=:c's constant column plus two
    # lags of a perfectly linear range are exactly collinear)
    m3 = arx(1.0:50.0, 2; trend=:n)
    @test all(isfinite, m3.coef)

    # singular/collinear design -> clear ArgumentError, not a raw
    # SingularException: _ols's QR-based solve can still return SOME beta
    # for a rank-deficient X (unlike inv(X'X), which arx needs for vcov)
    @test_throws ArgumentError arx(1.0:50.0, 2; trend=:c)

    # error paths
    @test_throws ArgumentError arx(y, 2; trend=:bogus)
    @test_throws ArgumentError arx(y, 2; seasonal=true)                 # missing period
    @test_throws ArgumentError arx(y, 1; seasonal=true, period=1)       # period < 2
    @test_throws ArgumentError arx(y, 2; hold_back=1)                   # hold_back < maxlag
    @test_throws ArgumentError arx(y, Int[])                            # empty lags
    @test_throws ArgumentError arx(y, 0)                                # lag < 1
    @test_throws ArgumentError arx(y, [1, 0, 2])                        # a lag < 1 in a subset
    y_nan = copy(y); y_nan[5] = NaN
    @test_throws ArgumentError arx(y_nan, 2)
    @test_throws ArgumentError arx(y, 2; exog=x1[1:50])                 # length mismatch
    x1_nan = copy(x1); x1_nan[3] = NaN
    @test_throws ArgumentError arx(y, 2; exog=x1_nan)
end
