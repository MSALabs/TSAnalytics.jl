using DelimitedFiles, StatsAPI

@testset "fit_sarima — SARIMA(1,0,0)(1,1,0)[12], dual-verified real fit (exact R+Python ground truth)" begin
    # handoff/stage-6.7-sarima-handoff.md §1-2, §7. Re-verified this
    # session directly against real R stats::arima(seasonal=...) and
    # Python statsmodels SARIMAX on the exact bundled sarima_shared.csv.
    y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "sarima_shared.csv"), ',', skipstart=1))

    m = fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); include_mean=false, method=:ml)
    @test m.converged
    @test isapprox(m.phi[1], 0.3424276441; atol=1e-3)
    @test isapprox(m.Phi[1], 0.3242885717; atol=1e-3)
    @test isapprox(m.loglik, -121.334298; atol=1e-2)
    @test isapprox(m.aic, 248.6685961; atol=1e-2)
    # headline finding, extending Stage 6.6: nobs = n - D*s = 84, NOT
    # Python's full n = 96
    @test StatsAPI.nobs(m) == length(y) - 1 * 12 == 84

    m_hess = fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); include_mean=false, se_type=:hessian)
    m_opg = fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); include_mean=false, se_type=:opg)
    @test isapprox(m_hess.se[1], 0.1048658295; atol=1e-2)  # R
    @test isapprox(m_hess.se[2], 0.1105468955; atol=1e-2)  # R
    @test isapprox(m_opg.se[1], 0.13195281; atol=1e-2)     # Python
    @test isapprox(m_opg.se[2], 0.12930198; atol=1e-2)     # Python
    @test !isapprox(m_hess.se, m_opg.se; atol=1e-3)        # genuinely different, not a bug

    # P=Q=D=0 reduces exactly to fit_arima's ARIMA(1,0,0)
    m_nonseasonal = fit_sarima(y, (1, 0, 0), (0, 0, 0, 12); include_mean=false)
    m_direct = fit_arima(y, (1, 0, 0); include_mean=false)
    @test isapprox(m_nonseasonal.phi, m_direct.arma.ar; atol=1e-8)
    @test isapprox(m_nonseasonal.loglik, m_direct.arma.loglik; atol=1e-8)

    @test_throws ArgumentError fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); method=:bogus)

    io = IOBuffer()
    show(io, m)
    s = String(take!(io))
    @test occursin("ARIMA(1,0,0)(1,1,0)[12]", s)
    @test occursin("ar1", s) && occursin("sar1", s)
end

@testset "fit_sarima — full polynomial block (p,q,P,Q all > 0), closing the handoff's own flagged gap" begin
    # The handoff's own ground truth (above) has q=Q=0, so the
    # theta/Theta parameter-unpacking indices (raw[p+q+1:p+q+P],
    # raw[p+q+P+1:p+q+P+Q]) are never exercised by it -- flagged
    # explicitly in the handoff §7 as the highest-risk untested
    # arithmetic in this stage. Closed here with a dedicated synthetic
    # SARIMA(1,0,1)(1,0,1)_4 series, verified against fresh real R and
    # Python fits (not just one reference).
    y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "sarima_full.csv"), ',', skipstart=1))

    m = fit_sarima(y, (1, 0, 1), (1, 0, 1, 4); include_mean=false)
    @test m.converged
    @test isapprox(m.phi[1], 0.3172665537; atol=1e-2)    # R
    @test isapprox(m.theta[1], 0.4069508635; atol=1e-2)  # R
    @test isapprox(m.Phi[1], 0.8599145050; atol=1e-2)    # R
    @test isapprox(m.Theta[1], -0.7597378821; atol=1e-2) # R
    @test isapprox(m.phi[1], 0.31724708; atol=1e-2)      # Python
    @test isapprox(m.theta[1], 0.40697284; atol=1e-2)    # Python
    @test isapprox(m.Phi[1], 0.86001889; atol=1e-2)      # Python
    @test isapprox(m.Theta[1], -0.75986672; atol=1e-2)   # Python
    @test isapprox(m.loglik, -291.3972474151; atol=1e-1)
    @test isapprox(m.aic, 592.7944948303; atol=1e-1)
    @test length(m.se) == 4
    @test isapprox(m.se, [0.1371657073, 0.1364037718, 0.1118915840, 0.1449169720]; atol=1e-2)
end

@testset "fit_sarima — combined d>0 AND D>0 (independent verification)" begin
    # Also confirms include_mean is silently forced off for D>0 alone
    # (d=0), not just d>0 -- verified directly against real R on
    # sarima_shared.csv (order=(1,0,0), seasonal=(1,1,0,12)):
    # include.mean=TRUE and FALSE give bit-identical coefficients.
    y_dD = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "sarima_dD.csv"), ',', skipstart=1))

    m = fit_sarima(y_dD, (1, 1, 0), (1, 1, 0, 4))  # include_mean=true (default)
    @test m.converged
    @test m.mean === nothing  # silently forced off: d>0 AND D>0
    @test isapprox(m.phi[1], 0.3274163693; atol=1e-2)
    @test isapprox(m.Phi[1], 0.4435180303; atol=1e-2)
    @test isapprox(m.loglik, -182.8764091079; atol=1e-1)
    @test isapprox(m.aic, 371.7528182158; atol=1e-1)
    @test StatsAPI.nobs(m) == length(y_dD) - 1 - 1 * 4 == 145

    # D>0 alone (d=0) also forces include_mean off -- verified on the
    # primary handoff series directly
    y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "sarima_shared.csv"), ',', skipstart=1))
    m_D_true = fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); include_mean=true)
    m_D_false = fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); include_mean=false)
    @test m_D_true.mean === nothing
    @test m_D_true.phi == m_D_false.phi
    @test m_D_true.Phi == m_D_false.Phi
    @test m_D_true.loglik == m_D_false.loglik

    # method=:css_ml converges to essentially the same answer as :ml
    m_css = fit_sarima(y_dD, (1, 1, 0), (1, 1, 0, 4); method=:css_ml)
    @test m_css.converged
    @test isapprox(m_css.phi[1], m.phi[1]; atol=1e-2)
    @test isapprox(m_css.Phi[1], m.Phi[1]; atol=1e-2)
end

@testset "fit_sarima parameter coverage and error paths" begin
    y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "sarima_shared.csv"), ',', skipstart=1))

    # optimizer_method coverage
    for opt in (:lbfgs, :bfgs, :nelder_mead)
        m = fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); include_mean=false, optimizer_method=opt)
        @test isfinite(m.loglik)
    end

    # start_params override
    m_default = fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); include_mean=false)
    m_explicit = fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); include_mean=false, start_params=[0.0, 0.0])
    @test isapprox(m_default.phi, m_explicit.phi; atol=1e-6)
    @test_throws ArgumentError fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); start_params=[0.0])  # wrong length

    # container-agnostic (Float32)
    m_f32 = fit_sarima(Float32.(y), (1, 0, 0), (1, 1, 0, 12); include_mean=false)
    @test isfinite(m_f32.loglik)

    # StatsAPI delegates
    m = fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); include_mean=false)
    @test StatsAPI.loglikelihood(m) == m.loglik
    @test StatsAPI.aic(m) == m.aic
    @test StatsAPI.bic(m) == m.bic
    @test StatsAPI.coef(m) == vcat(m.phi, m.theta, m.Phi, m.Theta)

    # (0,0,0)(0,0,0,s) -- degenerate but legitimate (pure white noise + mean)
    m00 = fit_sarima(y, (0, 0, 0), (0, 0, 0, 4); include_mean=true)
    @test m00.converged
    @test isempty(m00.phi) && isempty(m00.theta) && isempty(m00.Phi) && isempty(m00.Theta)
    @test m00.mean !== nothing

    # error paths
    @test_throws ArgumentError fit_sarima(y, (-1, 0, 0), (1, 1, 0, 12))
    @test_throws ArgumentError fit_sarima(y, (1, 0, 0), (-1, 1, 0, 12))
    @test_throws ArgumentError fit_sarima(y, (1, 0, 0), (1, 1, 0, 0))     # s must be >= 1
    @test_throws ArgumentError fit_sarima(y, (1, 0, 0), (1, 1, 0, 1))    # P>0 needs s>=2
    @test_throws ArgumentError fit_sarima(y, (1, 0, 0), (1, 1, 0, 12); se_type=:bogus)
end
