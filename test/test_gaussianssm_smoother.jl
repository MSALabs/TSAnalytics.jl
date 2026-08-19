# Verifies TSAnalytics.kalman_smoother against statsmodels
# SARIMAX(...).smooth(params)'s own smoothed_state/smoothed_state_cov.
# Confirmed (see test/verification/gaussianssm/bulk/smoother_check/
# gen_smoother_check.py) that statsmodels' transition/selection/design
# matrices for order=(p,0,q), trend='n' match build_statespace's T/R/Z
# convention exactly, so no reindexing is needed -- alpha/V are directly
# comparable column-for-column, entry-for-entry.

using Test, DelimitedFiles, LinearAlgebra

@testset "kalman_smoother -- verified against statsmodels' own smoother" begin
    cases = [
        (name="ar1", dataset="ar1_shared", ar=[0.6], ma=Float64[], r=1),
        (name="ar2", dataset="ar2_shared", ar=[0.5, 0.1], ma=Float64[], r=2),
        (name="ma1", dataset="ma1_shared", ar=Float64[], ma=[0.4], r=2),
        (name="arma11", dataset="ar1_shared", ar=[0.4], ma=[0.3], r=2),
    ]

    for c in cases
        y = vec(readdlm(joinpath(@__DIR__, "verification", "gaussianssm", "$(c.dataset).csv"), ',', skipstart=1))
        ssm = TSAnalytics.build_statespace(c.ar, c.ma)
        alpha, V = TSAnalytics.kalman_smoother(ssm, y)

        r = c.r
        n = length(y)
        @test size(alpha) == (r, n)
        @test length(V) == n

        # alpha[1, :] must reproduce y exactly -- no observation noise in this
        # state-space form, so smoothing can't touch the observed component.
        @test isapprox(alpha[1, :], y; atol=1e-9)

        expected = readdlm(joinpath(@__DIR__, "verification", "gaussianssm", "bulk", "smoother_check", "$(c.name)_expected.csv"), ',', skipstart=1)
        # columns: alpha_1..alpha_r, then V_11, V_12, ..., V_r1, ..., V_rr (row-major from Python)
        alpha_expected = expected[:, 1:r]'
        @test isapprox(alpha, alpha_expected; atol=1e-6)

        for t in 1:n
            V_expected_flat = expected[t, r+1:end]
            V_expected = reshape(collect(V_expected_flat), r, r)'  # Python reshape(-1) was row-major
            @test isapprox(V[t], V_expected; atol=1e-6)
        end
    end
end
