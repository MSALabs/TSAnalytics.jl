# test/test_gaussianssm.jl
#
# Extracted from gaussianssm-project-brief.md section 6, as a real file
# rather than markdown-embedded code. The algorithm this tests against
# was independently verified in Python (see verification/) to reproduce
# all three ground-truth log-likelihoods to full floating-point
# precision -- see python-verification-of-src.txt in this bundle. That
# means a failure here points at the Julia translation specifically
# (indexing, syntax), not the underlying math.

using Test, DelimitedFiles, LinearAlgebra

@testset "GaussianSSM — dual-verified against R and Python" begin
    y1 = vec(readdlm(joinpath(@__DIR__, "..", "verification", "ar1_shared.csv"), ',', skipstart=1))
    ssm1 = build_statespace([0.6], Float64[])
    loglik1, sigma2_1, v1, F1, converged1 = kalman_filter(ssm1, y1)
    @test converged1
    @test isapprox(loglik1, -141.527779252; atol=1e-6)

    y2 = vec(readdlm(joinpath(@__DIR__, "..", "verification", "ar2_shared.csv"), ',', skipstart=1))
    ssm2 = build_statespace([0.5, 0.1], Float64[])
    loglik2, sigma2_2, v2, F2, converged2 = kalman_filter(ssm2, y2)
    @test converged2
    @test isapprox(loglik2, -143.256444123; atol=1e-6)

    y3 = vec(readdlm(joinpath(@__DIR__, "..", "verification", "ma1_shared.csv"), ',', skipstart=1))
    ssm3 = build_statespace(Float64[], [0.4])
    loglik3, sigma2_3, v3, F3, converged3 = kalman_filter(ssm3, y3)
    @test converged3
    @test isapprox(loglik3, -128.114020042; atol=1e-6)

    # stationary_cov sanity: must be symmetric positive definite for a
    # genuinely stationary model
    Q0 = stationary_cov(ssm1)
    @test isapprox(Q0, Q0'; atol=1e-10)
    @test all(eigvals(Q0) .> 0)

    # non-stationary input should be caught, not crash: |phi| > 1 has no
    # valid stationary covariance
    ssm_bad = build_statespace([1.5], Float64[])
    loglik_bad, sigma2_bad, v_bad, F_bad, converged_bad = kalman_filter(ssm_bad, y1)
    @test !converged_bad
    @test loglik_bad == -Inf

    # combined_ar_ma degenerate case: no seasonal terms should reduce to
    # the plain ar/ma vectors exactly, unchanged
    ar, ma = combined_ar_ma([0.5, 0.1], Float64[], [0.4], Float64[], 12)
    @test isapprox(ar, [0.5, 0.1]; atol=1e-10)
    @test isapprox(ma, [0.4]; atol=1e-10)
end

# NOT YET COVERED, flagged rather than silently skipped (per the project
# brief's "high risk" flag on combined_ar_ma's seasonal case):
# a genuine seasonal SARIMA combined_ar_ma test, verified against its own
# dedicated R/Python fixed-coefficient comparison. Do this before trusting
# combined_ar_ma for any seasonal model, using the exact same technique
# as the AR(1)/AR(2)/MA(1) cases above (arima(..., seasonal=list(...),
# fixed=..., transform.pars=FALSE, optim.control=list(maxit=0)) in R;
# SARIMAX(..., seasonal_order=..., concentrate_scale=True).loglike(...)
# in Python).
