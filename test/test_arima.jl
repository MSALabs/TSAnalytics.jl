using DelimitedFiles, StatsAPI

@testset "fit_arima — ARIMA(1,1,1), dual-verified real fit (exact R+Python ground truth)" begin
    # handoff/stage-6.6-arima-handoff.md §3, §8. Re-verified this session
    # directly against real R stats::arima(order=c(1,1,1)) and Python
    # statsmodels ARIMA(order=(1,1,1)) on the exact bundled d1_clean.csv,
    # not just transcribed.
    y1 = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "d1_clean.csv"), ',', skipstart=1))

    m = fit_arima(y1, (1, 1, 1); include_mean=false, method=:ml)
    @test m.arma.converged
    @test isapprox(m.arma.ar[1], 0.5828351408; atol=1e-3)
    @test isapprox(m.arma.ma[1], 0.01726218017; atol=1e-2)
    @test isapprox(m.arma.loglik, -203.9508758; atol=1e-2)
    @test isapprox(m.arma.aic, 413.9017516; atol=1e-2)
    # the headline finding: R's nobs = n - d = 149, NOT Python's full n = 150
    @test StatsAPI.nobs(m) == length(y1) - 1 == 149
    @test m.d == 1

    m_hess = fit_arima(y1, (1, 1, 1); include_mean=false, se_type=:hessian)
    m_opg = fit_arima(y1, (1, 1, 1); include_mean=false, se_type=:opg)
    @test isapprox(m_hess.arma.se[1], 0.1309278732; atol=1e-2)  # R
    @test isapprox(m_hess.arma.se[2], 0.1733885098; atol=1e-2)  # R
    @test isapprox(m_opg.arma.se[1], 0.10513078; atol=1e-2)     # Python
    @test isapprox(m_opg.arma.se[2], 0.15201031; atol=1e-2)     # Python
    @test !isapprox(m_hess.arma.se, m_opg.arma.se; atol=1e-3)   # genuinely different, not a bug

    @test_throws ArgumentError fit_arima(y1, (1, 1, 1); method=:bogus)

    io = IOBuffer()
    show(io, m)
    s = String(take!(io))
    @test occursin("ARIMA(1,1,1)", s)
    @test occursin("ar1", s) && occursin("ma1", s)
end

@testset "fit_arima — near-unit-root stress case (both R and Python warn on it too)" begin
    # A genuine hard case, kept deliberately rather than avoided: both R
    # and Python report convergence warnings here. Tolerances are looser
    # than the clean case, matching what the handoff itself expects.
    y2 = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "d1_series.csv"), ',', skipstart=1))

    m = fit_arima(y2, (1, 1, 1); include_mean=false)
    @test m.arma.converged
    @test isapprox(m.arma.ar[1], 0.9989472225; atol=1e-2)
    @test isapprox(m.arma.ma[1], -0.983617654; atol=1e-2)
    @test isapprox(m.arma.loglik, -126.2521509; atol=1e-1)
    @test StatsAPI.nobs(m) == length(y2) - 1 == 99
end

@testset "fit_arima — include_mean silently ignored when d>0 (verified against real R)" begin
    # handoff §5's flagged-as-needing-verification interaction, confirmed
    # directly: R's arima(..., include.mean=TRUE) with d=1 produces
    # BIT-IDENTICAL fitted coefficients to include.mean=FALSE (no
    # intercept term appears at all) -- this project's own design
    # decision (parent stage-6-arima-handoff.md §4.3) matches that
    # exactly rather than inventing a third convention.
    y1 = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "d1_clean.csv"), ',', skipstart=1))

    m_true = fit_arima(y1, (1, 1, 1); include_mean=true)
    m_false = fit_arima(y1, (1, 1, 1); include_mean=false)
    @test m_true.arma.mean === nothing  # silently forced off, not estimated
    @test m_true.arma.ar == m_false.arma.ar
    @test m_true.arma.ma == m_false.arma.ma
    @test m_true.arma.loglik == m_false.arma.loglik

    # d=0: include_mean behaves exactly like fit_arma's own default (has
    # a real effect, unlike the d>0 case above)
    m_d0_true = fit_arima(y1, (1, 0, 1); include_mean=true)
    @test m_d0_true.arma.mean !== nothing
end

@testset "fit_arima — d=0 reduces exactly to fit_arma (thin-wrapper regression guard)" begin
    # The single most important structural test per the handoff §8: if
    # this stage duplicates fit_arma's logic instead of delegating,
    # THIS is what catches it.
    y1 = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "d1_clean.csv"), ',', skipstart=1))

    for order in ((1, 1), (2, 1), (1, 2), (0, 1), (1, 0))
        p, q = order
        m_wrapped = fit_arima(y1, (p, 0, q); include_mean=false)
        m_direct = fit_arma(y1, order; include_mean=false)
        @test m_wrapped.arma.ar == m_direct.ar
        @test m_wrapped.arma.ma == m_direct.ma
        @test m_wrapped.arma.loglik == m_direct.loglik
        @test m_wrapped.arma.se == m_direct.se
        @test StatsAPI.nobs(m_wrapped) == m_direct.nobs == length(y1)
    end
end

@testset "fit_arima parameter coverage and error paths" begin
    y1 = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "d1_clean.csv"), ',', skipstart=1))

    # every d value
    for d in (0, 1, 2)
        m = fit_arima(y1, (1, d, 1); include_mean=false)
        @test m.d == d
        @test StatsAPI.nobs(m) == length(y1) - d
        @test isfinite(m.arma.loglik)
    end

    # order (0,d,0): pure random walk after differencing -- a legitimate
    # model, not an error (verified against real R's arima(order=c(0,1,0)))
    m_rw = fit_arima(y1, (0, 1, 0))
    @test m_rw.arma.converged
    @test isempty(m_rw.arma.ar) && isempty(m_rw.arma.ma)

    # StatsAPI delegates
    m = fit_arima(y1, (1, 1, 1); include_mean=false)
    @test StatsAPI.loglikelihood(m) == m.arma.loglik
    @test StatsAPI.aic(m) == m.arma.aic
    @test StatsAPI.bic(m) == m.arma.bic
    @test StatsAPI.coef(m) == vcat(m.arma.ar, m.arma.ma)

    # container-agnostic (Float32)
    m_f32 = fit_arima(Float32.(y1), (1, 1, 1); include_mean=false)
    @test isfinite(m_f32.arma.loglik)

    # struct fields
    @test m isa TSAnalytics.ArimaModel
    @test m.original_y == y1
    @test length(m.original_y) == length(y1)

    # error paths
    @test_throws ArgumentError fit_arima(y1, (-1, 1, 1))
    @test_throws ArgumentError fit_arima(y1, (1, -1, 1))
    @test_throws ArgumentError fit_arima(y1, (1, 1, -1))
    @test_throws ArgumentError fit_arima(y1, (1, 1, 1); se_type=:bogus)
end
