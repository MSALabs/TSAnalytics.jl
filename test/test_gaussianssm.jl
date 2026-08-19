# Extracted from the standalone GaussianSSM project brief's own test
# section, as a real file rather than markdown-embedded code. The
# algorithm this tests against was independently verified in Python
# (see test/verification/gaussianssm/) to reproduce all three
# ground-truth log-likelihoods to full floating-point precision -- see
# python-verification-of-src.txt in that bundle. That means a failure
# here points at the Julia translation specifically (indexing, syntax),
# not the underlying math. `build_statespace`/`kalman_filter`/etc. are
# deliberately unexported (see `handoff/stage-6-arima-handoff.md` §1.4 --
# their signatures will change at Stage 8), so every call below goes
# through the `TSAnalytics.` prefix, matching this project's existing
# convention for unexported internals (e.g. `TSAnalytics._ols`).

using Test, DelimitedFiles, LinearAlgebra

@testset "GaussianSSM — dual-verified against R and Python" begin
    y1 = vec(readdlm(joinpath(@__DIR__, "verification", "gaussianssm", "ar1_shared.csv"), ',', skipstart=1))
    ssm1 = TSAnalytics.build_statespace([0.6], Float64[])
    loglik1, sigma2_1, v1, F1, converged1 = TSAnalytics.kalman_filter(ssm1, y1)
    @test converged1
    @test isapprox(loglik1, -141.527779252; atol=1e-6)

    y2 = vec(readdlm(joinpath(@__DIR__, "verification", "gaussianssm", "ar2_shared.csv"), ',', skipstart=1))
    ssm2 = TSAnalytics.build_statespace([0.5, 0.1], Float64[])
    loglik2, sigma2_2, v2, F2, converged2 = TSAnalytics.kalman_filter(ssm2, y2)
    @test converged2
    @test isapprox(loglik2, -143.256444123; atol=1e-6)

    y3 = vec(readdlm(joinpath(@__DIR__, "verification", "gaussianssm", "ma1_shared.csv"), ',', skipstart=1))
    ssm3 = TSAnalytics.build_statespace(Float64[], [0.4])
    loglik3, sigma2_3, v3, F3, converged3 = TSAnalytics.kalman_filter(ssm3, y3)
    @test converged3
    @test isapprox(loglik3, -128.114020042; atol=1e-6)

    # stationary_cov sanity: must be symmetric positive definite for a
    # genuinely stationary model. Returns (Q0, converged) since the
    # doubling-iteration swap (handoff §3.2) -- kron's SingularException
    # became an explicit flag instead.
    Q0, sc_converged = TSAnalytics.stationary_cov(ssm1)
    @test sc_converged
    @test isapprox(Q0, Q0'; atol=1e-10)
    @test all(eigvals(Q0) .> 0)

    # non-stationary input should be caught, not crash: |phi| > 1 has no
    # valid stationary covariance
    ssm_bad = TSAnalytics.build_statespace([1.5], Float64[])
    loglik_bad, sigma2_bad, v_bad, F_bad, converged_bad = TSAnalytics.kalman_filter(ssm_bad, y1)
    @test !converged_bad
    @test loglik_bad == -Inf
    @test isempty(v_bad) && isempty(F_bad)

    # combined_ar_ma degenerate case: no seasonal terms should reduce to
    # the plain ar/ma vectors exactly, unchanged
    ar, ma = TSAnalytics.combined_ar_ma([0.5, 0.1], Float64[], [0.4], Float64[], 12)
    @test isapprox(ar, [0.5, 0.1]; atol=1e-10)
    @test isapprox(ma, [0.4]; atol=1e-10)

    # keyword form (handoff §3.1) agrees with the positional form exactly
    ar_kw, ma_kw = TSAnalytics.combined_ar_ma(; phi=[0.5, 0.1], theta=[0.4], s=12)
    @test ar_kw == ar && ma_kw == ma

    # seasonal_poly s=0 guard (handoff §2.2): s=0 with non-empty seasonal
    # coefficients used to silently corrupt the polynomial's leading 1.0
    @test_throws ArgumentError TSAnalytics.seasonal_poly([0.5], 0)
    @test TSAnalytics.seasonal_poly(Float64[], 0) == [1.0]  # empty coefs: s irrelevant, not an error

    # ergonomic trap (handoff §2.1 side benefit): a bare `[]` literal
    # (Vector{Any}) works exactly like Float64[] for an absent polynomial
    ar_bare, ma_bare = TSAnalytics.combined_ar_ma([0.5], [], [0.4], [], 12)
    @test ar_bare == [0.5] && ma_bare == [0.4]

    # empty-series / empty-y guards (handoff §3.4)
    @test_throws ArgumentError TSAnalytics.kalman_filter(ssm1, Float64[])
    @test_throws ArgumentError TSAnalytics.kalman_smoother(ssm1, Float64[])
end
