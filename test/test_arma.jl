using DelimitedFiles

@testset "fit_arma — ARMA(1,1), dual-verified real fit (exact R+Python ground truth)" begin
    # handoff/stage-6.5-arma-mle-handoff.md §3. Re-verified this session
    # directly against real R stats::arima() and Python statsmodels
    # ARIMA on the exact bundled arma11_fit.csv, not just transcribed --
    # see the handoff's own "Status" header for the two real bugs this
    # verification caught in the handoff's own proposed sketch (missing
    # +1-for-sigma2 in AIC/BIC, and partrans not being ForwardDiff-safe).
    y = vec(readdlm(joinpath(@__DIR__, "verification", "arma", "arma11_fit.csv"), ',', skipstart=1))

    m_ml = fit_arma(y, (1, 1); include_mean=false, method=:ml)
    @test m_ml.converged
    @test isapprox(m_ml.ar[1], 0.5465817922; atol=1e-3)
    @test isapprox(m_ml.ma[1], 0.2717023914; atol=1e-3)
    @test isapprox(m_ml.loglik, -280.5177637; atol=1e-2)
    @test isapprox(m_ml.aic, 567.0355274458; atol=1e-2)
    # R's actual BIC (via stats::BIC()), confirmed this session to require
    # k=3 (ar1, ma1, sigma2) -- NOT k=2 as the handoff's own §6 sketch had it
    @test isapprox(m_ml.bic, 576.9304795454; atol=1e-2)
    @test m_ml.mean === nothing

    m_cssml = fit_arma(y, (1, 1); include_mean=false, method=:css_ml)
    @test m_cssml.converged
    @test isapprox(m_cssml.ar[1], 0.5465783235; atol=1e-3)
    @test isapprox(m_cssml.ma[1], 0.2717063783; atol=1e-3)

    m_hess = fit_arma(y, (1, 1); include_mean=false, se_type=:hessian)
    @test isapprox(m_hess.se[1], 0.0878534233; atol=1e-2)
    @test isapprox(m_hess.se[2], 0.1074810562; atol=1e-2)

    m_opg = fit_arma(y, (1, 1); include_mean=false, se_type=:opg)
    @test isapprox(m_opg.se[1], 0.081; atol=1e-2)
    @test isapprox(m_opg.se[2], 0.092; atol=1e-2)
    @test !isapprox(m_hess.se[1], m_opg.se[1]; atol=1e-4)  # genuinely different, not a bug

    @test_throws ArgumentError fit_arma(y, (1, 1); method=:bogus)
    @test_throws ArgumentError fit_arma(y, (1, 1); se_type=:bogus)

    io = IOBuffer()
    show(io, m_ml)
    s = String(take!(io))
    @test occursin("ARMA(1,1)", s)
    @test occursin("ar1", s) && occursin("ma1", s)
    @test occursin("hessian", s)
    @test occursin("Log-likelihood", s) && occursin("AIC", s) && occursin("BIC", s)
end

@testset "fit_arma — order (0,0), independent R verification (real gap found while implementing 6.6)" begin
    # fit_arma originally rejected (0,0) as "nothing to fit" -- discovered
    # to be wrong while implementing Stage 6.6 (ARIMA(0,d,0), e.g. a pure
    # random walk after differencing, is common and R fully supports
    # order=c(0,0,0) too). Verified directly against a fresh R run.
    y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "d1_clean.csv"), ',', skipstart=1))

    m_mean = fit_arma(y, (0, 0); include_mean=true)
    @test m_mean.converged
    @test isapprox(m_mean.mean, 26.9981946596; atol=1e-4)
    @test isapprox(m_mean.loglik, -532.9856934583; atol=1e-3)
    @test isapprox(m_mean.aic, 1069.9713869165; atol=1e-3)
    @test length(m_mean.se) == 1
    @test isapprox(m_mean.se[1], 0.6900313664; atol=1e-4)

    m_nomean = fit_arma(y, (0, 0); include_mean=false)
    @test m_nomean.converged
    @test isapprox(m_nomean.loglik, -714.2170291442; atol=1e-3)
    @test isapprox(m_nomean.aic, 1430.4340582883; atol=1e-3)
    @test isempty(m_nomean.se)  # zero free parameters -> nothing to report
end

@testset "fit_arma — joint mean estimation (independent R verification, beyond the handoff's own ground truth)" begin
    # The handoff's own ground truth only covers include_mean=false. Its
    # §6 sketch pre-demeans with the sample mean and fits ARMA on the
    # residual -- verified this session, via a fresh R run on a
    # mean-shifted version of the same series, that this does NOT match
    # R's actual include.mean=TRUE behavior (which jointly optimizes the
    # mean with AR/MA). fit_arma implements joint estimation instead.
    y = vec(readdlm(joinpath(@__DIR__, "verification", "arma", "arma11_fit.csv"), ',', skipstart=1))
    y_shifted = y .+ 5.0

    m = fit_arma(y_shifted, (1, 1); include_mean=true, method=:ml)
    @test m.converged
    @test m.mean !== nothing
    @test isapprox(m.ar[1], 0.5424673554; atol=1e-3)
    @test isapprox(m.ma[1], 0.2741658630; atol=1e-3)
    @test isapprox(m.mean, 4.8917354742; atol=1e-3)
    @test isapprox(m.loglik, -280.3599737306; atol=1e-2)
    @test isapprox(m.aic, 568.7199474612; atol=1e-2)
    @test length(m.se) == 3  # ar1, ma1, mean
    @test isapprox(m.se[1], 0.0883356276; atol=1e-2)
    @test isapprox(m.se[2], 0.1075022657; atol=1e-2)
    @test isapprox(m.se[3], 0.1919046163; atol=1e-2)

    # joint estimation genuinely differs from a naive demean-then-fit:
    # the include_mean=false fit on the UNSHIFTED series gives different
    # ar1/ma1 than this include_mean=true fit on the SHIFTED series would
    # if it just subtracted the sample mean once
    m_nomean = fit_arma(y, (1, 1); include_mean=false)
    @test !isapprox(m.ar[1], m_nomean.ar[1]; atol=1e-4)

    io = IOBuffer()
    show(io, m)
    @test occursin("with mean", String(take!(io)))
end

@testset "fit_arma parameter coverage and error paths" begin
    y = vec(readdlm(joinpath(@__DIR__, "verification", "arma", "arma11_fit.csv"), ',', skipstart=1))

    # every order combination
    for order in ((1, 0), (0, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2))
        m = fit_arma(y, order; include_mean=false)
        @test length(m.ar) == order[1]
        @test length(m.ma) == order[2]
        @test isfinite(m.loglik)
        @test m.order == order
    end

    # optimizer_method coverage
    for opt in (:lbfgs, :bfgs, :nelder_mead)
        m = fit_arma(y, (1, 1); include_mean=false, optimizer_method=opt)
        @test isfinite(m.loglik)
    end

    # start_params override (no mean)
    m_default = fit_arma(y, (1, 1); include_mean=false)
    m_explicit = fit_arma(y, (1, 1); include_mean=false, start_params=[0.0, 0.0])
    @test isapprox(m_default.ar, m_explicit.ar; atol=1e-6)

    # start_params override (with mean) must have length p+q+1
    m_mean_explicit = fit_arma(y, (1, 1); include_mean=true, start_params=[0.0, 0.0, sum(y)/length(y)])
    @test isfinite(m_mean_explicit.loglik)
    @test_throws ArgumentError fit_arma(y, (1, 1); include_mean=true, start_params=[0.0, 0.0])  # wrong length

    # container-agnostic (range, Float32)
    m_f32 = fit_arma(Float32.(y), (1, 1); include_mean=false)
    @test isfinite(m_f32.loglik)

    # aic/bic consistency: k always counts sigma2
    m = fit_arma(y, (2, 1); include_mean=true)
    k_expected = 2 + 1 + 1 + 1  # p + q + mean + sigma2
    @test isapprox(m.aic, -2*m.loglik + 2*k_expected; atol=1e-8)
    @test isapprox(m.bic, -2*m.loglik + k_expected*log(m.nobs); atol=1e-8)

    # order (0,0): a legitimate white-noise(+mean) model, not an error --
    # confirmed against real R's arima(order=c(0,0,0)) directly (see the
    # dedicated testset below)
    m00 = fit_arma(y, (0, 0); include_mean=false)
    @test isempty(m00.ar) && isempty(m00.ma) && isempty(m00.se)
    @test m00.converged

    # error paths
    @test_throws ArgumentError fit_arma(y, (-1, 1))               # negative order
    @test_throws ArgumentError fit_arma(y[1:2], (1, 1))           # not enough observations
    @test_throws ArgumentError fit_arma(y, (1, 1); start_params=[0.0])  # wrong start_params length
end

@testset "fit_arma internal helpers — TSAnalytics.-prefixed, per project convention" begin
    y = vec(readdlm(joinpath(@__DIR__, "verification", "arma", "arma11_fit.csv"), ',', skipstart=1))
    yc = y .- sum(y)/length(y)

    # _css_start_values returns a raw (partrans-space) vector of length p+q
    raw = TSAnalytics._css_start_values(yc, 1, 1)
    @test length(raw) == 2
    phi = TSAnalytics.partrans(raw[1:1])
    @test isapprox(phi[1], 0.5465783235; atol=1e-2)  # close to R's CSS-ML ar1

    # _hessian_se / _opg_se agree with the public se_type= dispatch
    m_hess = fit_arma(y, (1, 1); include_mean=false, se_type=:hessian)
    params_hat = vcat(m_hess.ar, m_hess.ma)
    se_direct = TSAnalytics._hessian_se(params_hat, y, 1, 1, false)
    @test isapprox(m_hess.se, se_direct; atol=1e-10)
end
