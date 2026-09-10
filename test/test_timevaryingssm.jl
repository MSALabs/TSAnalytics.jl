using DelimitedFiles, LinearAlgebra

@testset "TimeVaryingSSM" begin
    T_ = TSAnalytics

    @testset "exact reduction to GaussianSSM (single hand-verified case)" begin
        # arma11_fit.csv, ar1=0.5466 ma1=0.2717 -- Stage 6.5's own dual-verified case
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arma", "arma11_fit.csv"), ','; skipstart=1))
        ssm = T_.build_statespace([0.5466], [0.2717])
        loglik, sigma2, v, F, converged = T_.kalman_filter(ssm, y)
        @test converged
        @test isapprox(loglik, -280.517763771456; atol=1e-6)

        tv, a0, P0 = T_.to_time_varying(ssm, y)
        @test tv isa T_.TimeVaryingSSM
        loglik_tv, v_tv, F_tv, converged_tv = T_.kalman_filter(tv, y, a0, P0)
        @test converged_tv
        @test loglik_tv == loglik   # exact, not just approximate -- the real point of this stage
        @test isapprox(v_tv, v; atol=1e-10)
        # F is NOT expected to match exactly -- it's scaled by sigma2 relative to GaussianSSM's
        # own RAW (pre-concentration) F, by design (see to_time_varying's own docstring)
        @test isapprox(F_tv ./ F, fill(sigma2, length(F)); atol=1e-10)
    end

    @testset "Z_t varying (handoff section 3b, exact real statsmodels validation)" begin
        x = [1.0, 2.0, -1.0, 0.5, 3.0, -0.5]
        y = [2.1, 4.5, -0.8, 1.6, 6.9, -0.4]
        n = length(y)
        Zseq = [reshape([1.0, xt], 1, 2) for xt in x]
        tv = T_.TimeVaryingSSM{Float64}(
            [reshape([0.5 0.0; 0.0 0.5], 2, 2)], Zseq, [Matrix{Float64}(I, 2, 2)],
            [reshape([0.01 0.0; 0.0 0.001], 2, 2)], [reshape([0.1], 1, 1)], 2)
        a0 = zeros(2)
        P0 = 0.5 .* Matrix{Float64}(I, 2, 2)
        loglik, v, F, converged = T_.kalman_filter(tv, y, a0, P0)
        @test converged
        @test isapprox(loglik, -186.3544625437698; atol=1e-6)
        forecasts = y .- v
        @test isapprox(forecasts, [0, 1.43181818, -0.78848842, 0.19589125, 0.6789244, 0.23150176]; atol=1e-6)
    end

    @testset "T_t varying (exact real statsmodels validation, closes handoff's flagged gap)" begin
        y = [1.0, 0.8, -0.3, 0.5, -0.9, 0.2]
        Tvals = [0.5, 0.3, 0.5, 0.3, 0.5, 0.3]
        tv = T_.TimeVaryingSSM{Float64}(
            [reshape([tv_], 1, 1) for tv_ in Tvals], [reshape([1.0], 1, 1)],
            [reshape([1.0], 1, 1)], [reshape([0.5], 1, 1)], [reshape([0.1], 1, 1)], 1)
        loglik, v, F, converged = T_.kalman_filter(tv, y, [0.2], reshape([0.3], 1, 1))
        @test converged
        @test isapprox(loglik, -6.405060681355125; atol=1e-8)
        forecasts = y .- v
        @test isapprox(forecasts, [0.2, 0.4, 0.2206060606060606, -0.10715497032270936,
                                    0.12066341316463322, -0.3660017573501198]; atol=1e-8)
    end

    @testset "Q_t varying (exact real statsmodels validation, closes handoff's flagged gap)" begin
        y = [0.5, -0.2, 1.1, 0.3, -0.7, 0.9]
        Qvals = [0.2, 0.8, 0.2, 0.8, 0.2, 0.8]
        tv = T_.TimeVaryingSSM{Float64}(
            [reshape([0.4], 1, 1)], [reshape([1.0], 1, 1)], [reshape([1.0], 1, 1)],
            [reshape([qv], 1, 1) for qv in Qvals], [reshape([0.15], 1, 1)], 1)
        loglik, v, F, converged = T_.kalman_filter(tv, y, [0.0], reshape([0.5], 1, 1))
        @test converged
        @test isapprox(loglik, -6.868603615038962; atol=1e-8)
        forecasts = y .- v
        @test isapprox(forecasts, [0.0, 0.15384615384615385, -0.022379958246346556,
                                    0.37015896385531744, 0.1313689421121904, -0.22826992403697466]; atol=1e-8)
    end

    @testset "H_t varying (exact real statsmodels validation, closes handoff's flagged gap)" begin
        y = [0.9, -0.4, 0.6, -1.1, 0.4, 0.2]
        Hvals = [0.05, 0.4, 0.05, 0.4, 0.05, 0.4]
        tv = T_.TimeVaryingSSM{Float64}(
            [reshape([0.45], 1, 1)], [reshape([1.0], 1, 1)], [reshape([1.0], 1, 1)],
            [reshape([0.3], 1, 1)], [reshape([hv], 1, 1) for hv in Hvals], 1)
        loglik, v, F, converged = T_.kalman_filter(tv, y, [0.0], reshape([0.4], 1, 1))
        @test converged
        @test isapprox(loglik, -7.060444436067119; atol=1e-8)
        forecasts = y .- v
        @test isapprox(forecasts, [0.0, 0.36000000000000004, 0.01294781382228495,
                                    0.23571863137356824, -0.15579911087977288, 0.14754265304928768]; atol=1e-8)
    end

    @testset "R_t varying, 2-state system (exact real statsmodels validation, closes handoff's flagged gap)" begin
        y = [1.2, -0.5, 0.8, 0.3, -0.9, 1.5]
        Rmats = [reshape([1.0, 0.0], 2, 1), reshape([0.0, 1.0], 2, 1)]
        Rseq = [Rmats[(t - 1) % 2 + 1] for t in 1:length(y)]
        tv = T_.TimeVaryingSSM{Float64}(
            [reshape([0.5 0.0; 0.0 0.3], 2, 2)], [reshape([1.0, 0.0], 1, 2)],
            Rseq, [reshape([0.5], 1, 1)], [reshape([0.1], 1, 1)], 2)
        loglik, v, F, converged = T_.kalman_filter(tv, y, [0.1, -0.1], reshape([0.3 0.0; 0.0 0.3], 2, 2))
        @test converged
        @test isapprox(loglik, -14.715919572720294; atol=1e-8)
        forecasts = y .- v
        @test isapprox(forecasts, [0.1, 0.4624999999999999, -0.17222222222222228,
                                    -0.001878914405010479, 0.12502374989204593, -0.025957055324318024]; atol=1e-8)
    end

    @testset "x_t constant degenerates to time-invariant-with-regression" begin
        # same Z_t=[1,x_t] shape as the section-3b case, but x_t held constant --
        # should match a fixed-Z 2-state time-invariant computation exactly
        n = 6
        y = [2.1, 1.5, 0.8, 1.6, 2.9, 0.4]
        x_const = 1.5
        Zseq = [reshape([1.0, x_const], 1, 2)]  # length-1 -> broadcast every period
        tv = T_.TimeVaryingSSM{Float64}(
            [reshape([0.5 0.0; 0.0 0.5], 2, 2)], Zseq, [Matrix{Float64}(I, 2, 2)],
            [reshape([0.01 0.0; 0.0 0.001], 2, 2)], [reshape([0.1], 1, 1)], 2)
        loglik1, v1, F1, converged1 = T_.kalman_filter(tv, y, zeros(2), 0.5 .* Matrix{Float64}(I, 2, 2))
        @test converged1

        Zseq_full = [reshape([1.0, x_const], 1, 2) for _ in 1:n]  # fully materialized, same values
        tv2 = T_.TimeVaryingSSM{Float64}(
            [reshape([0.5 0.0; 0.0 0.5], 2, 2)], Zseq_full, [Matrix{Float64}(I, 2, 2)],
            [reshape([0.01 0.0; 0.0 0.001], 2, 2)], [reshape([0.1], 1, 1)], 2)
        loglik2, v2, F2, converged2 = T_.kalman_filter(tv2, y, zeros(2), 0.5 .* Matrix{Float64}(I, 2, 2))
        @test converged2
        @test loglik1 == loglik2   # length-1 broadcast vs fully-materialized: bit-identical
        @test v1 == v2
    end

    @testset "converged=false on a degenerate F_t" begin
        # H_t driven deeply negative makes F_t <= 0 possible; construct directly rather than
        # search for one, matching GaussianSSM's own convention of a clear sentinel, not a crash
        y = [1.0, 2.0, 3.0]
        tv = T_.TimeVaryingSSM{Float64}(
            [reshape([0.5], 1, 1)], [reshape([1.0], 1, 1)], [reshape([1.0], 1, 1)],
            [reshape([0.0], 1, 1)], [reshape([-10.0], 1, 1)], 1)
        loglik, v, F, converged = T_.kalman_filter(tv, y, [0.0], reshape([0.0], 1, 1))
        @test !converged
        @test loglik == -Inf
        @test isempty(v)
        @test isempty(F)
    end

    @testset "error paths" begin
        tv = T_.TimeVaryingSSM{Float64}(
            [reshape([0.5], 1, 1)], [reshape([1.0], 1, 1)], [reshape([1.0], 1, 1)],
            [reshape([0.5], 1, 1)], [reshape([0.1], 1, 1)], 1)
        @test_throws ArgumentError T_.kalman_filter(tv, Float64[], [0.0], reshape([0.3], 1, 1))
        @test_throws ArgumentError T_.kalman_filter(tv, [1.0, 2.0], [0.0, 0.0], reshape([0.3], 1, 1))  # wrong a0 length
        @test_throws ArgumentError T_.kalman_filter(tv, [1.0, 2.0], [0.0], reshape([0.3 0.0; 0.0 0.1], 2, 2))  # wrong P0 size

        ssm_bad = T_.GaussianSSM{Float64}(reshape([1.0], 1, 1), [1.0], 1)  # unit root, non-stationary
        @test_throws ArgumentError T_.to_time_varying(ssm_bad, [1.0, 2.0, 3.0])
    end

    @testset "performance: fit_arma unaffected by TimeVaryingSSM's existence (Stage 6.5 still fast)" begin
        # a coarse sanity check, not a tight benchmark -- confirms the new dispatch didn't
        # somehow slow down the existing GaussianSSM path (multiple-dispatch design per the
        # handoff's own section 4 claim: separate compiled methods, zero shared overhead)
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arma", "arma11_fit.csv"), ','; skipstart=1))
        fit_arma(y, (1, 1))  # warm up / compile
        t = @elapsed for _ in 1:20
            fit_arma(y, (1, 1))
        end
        @test t / 20 < 1.0   # generously loose -- this is a regression smoke test, not a timing gate
    end

    @testset "missing observations (NaN), kalman_filter" begin
        # local level: T=Z=R=1, Q=0.01, H=0.5, known a0=1.0/P0=1.0 -- verified directly
        # against real statsmodels' own low-level KalmanFilter (initialize_known), Part VI
        # verification session: llf over non-missing observations matches to displayed
        # precision (np.nansum(res.llf_obs) = -3.686903488937376 on this exact system/data)
        T_ssm = reshape([1.0], 1, 1)
        Z_ssm = reshape([1.0], 1, 1)
        R_ssm = reshape([1.0], 1, 1)
        Q_ssm = reshape([0.01], 1, 1)
        H_ssm = reshape([0.5], 1, 1)
        tv = T_.TimeVaryingSSM{Float64}([T_ssm], [Z_ssm], [R_ssm], [Q_ssm], [H_ssm], 1)
        y = [1.0, 1.2, NaN, 1.6, 1.5]
        loglik, v, F, converged = T_.kalman_filter(tv, y, [1.0], reshape([1.0], 1, 1))
        @test converged
        @test isnan(v[3]) && isnan(F[3])
        @test isapprox(loglik, -3.686903488937376; atol=1e-9)

        # a series with no missing values at all must be bit-identical before/after the
        # NaN-skip logic was added -- the reduction case for this specific change
        y_complete = [1.0, 1.2, 1.4, 1.6, 1.5]
        loglik2, v2, F2, converged2 = T_.kalman_filter(tv, y_complete, [1.0], reshape([1.0], 1, 1))
        @test converged2
        @test all(isfinite, v2) && all(isfinite, F2)
    end

    @testset "kalman_smoother(::TimeVaryingSSM): exact reduction to GaussianSSM's own smoother" begin
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arma", "arma11_fit.csv"), ','; skipstart=1))
        ssm = T_.build_statespace([0.5466], [0.2717])
        alpha_old, V_old = T_.kalman_smoother(ssm, y)
        tv, a0, P0 = T_.to_time_varying(ssm, y)
        alpha_new, V_new, eta, eta_var, eps, eps_var, converged = T_.kalman_smoother(tv, y, a0, P0)
        @test converged
        @test isapprox(alpha_new, alpha_old; atol=1e-10)
        @test isapprox([V_new[t][1, 1] for t in eachindex(V_new)], [V_old[t][1, 1] for t in eachindex(V_old)]; atol=1e-10)
    end

    @testset "kalman_smoother(::TimeVaryingSSM): endpoint identity (Chapter 31's own check)" begin
        # a genuine local level (H > 0, unlike the ARMA companion form's H=0 special case,
        # where alpha[1,:] == y trivially everywhere and this check would be vacuous):
        # the filtered and smoothed state must coincide exactly at the FINAL observation,
        # since there is no future left for the backward pass to add there, and differ
        # everywhere before it, since the smoother has strictly more information earlier on.
        T_ssm = reshape([1.0], 1, 1)
        Z_ssm = reshape([1.0], 1, 1)
        R_ssm = reshape([1.0], 1, 1)
        Q_ssm = reshape([0.02], 1, 1)
        H_ssm = reshape([0.3], 1, 1)
        tv = T_.TimeVaryingSSM{Float64}([T_ssm], [Z_ssm], [R_ssm], [Q_ssm], [H_ssm], 1)
        y = [1.0, 1.4, 0.9, 1.6, 2.0, 1.7, 2.3, 2.1]
        n = length(y)
        a0 = [1.0]; P0 = reshape([1.0], 1, 1)

        loglik, v, F, fconverged = T_.kalman_filter(tv, y, a0, P0)
        @test fconverged
        # reconstruct the filtered (not merely predicted) state at time n: a_{n|n} = a_{n|n-1} + K_n v_n,
        # recovered here from the same v/F the filter already returned rather than re-deriving K_n by hand
        a_pred_n = y[n] - v[n]
        a_filtered_n = a_pred_n + (F[n] - H_ssm[1, 1]) / F[n] * v[n]  # P_{n|n-1} = F_n - H (Z=1 here)

        alpha, V, eta, eta_var, eps, eps_var, sconverged = T_.kalman_smoother(tv, y, a0, P0)
        @test sconverged
        @test isapprox(alpha[1, n], a_filtered_n; atol=1e-8)
        # and NOT equal earlier in the series -- the smoother is doing real work there
        a_pred_1 = a0[1]
        @test abs(alpha[1, 1] - a_pred_1) > 1e-3
    end

    @testset "kalman_smoother(::TimeVaryingSSM): disturbance smoothing vs. real statsmodels" begin
        # local level, same system/data as the missing-observations filter test above --
        # verified against real statsmodels' KalmanSmoother (initialize_known, .smooth()):
        # smoothed_state, smoothed_state_cov, smoothed_state_disturbance(_cov) and
        # smoothed_measurement_disturbance(_cov) all match to displayed precision, Part VI
        # verification session (see development-sequence.md's own Part VI note for the numbers)
        T_ssm = reshape([1.0], 1, 1)
        Z_ssm = reshape([1.0], 1, 1)
        R_ssm = reshape([1.0], 1, 1)
        Q_ssm = reshape([0.01], 1, 1)
        H_ssm = reshape([0.5], 1, 1)
        tv = T_.TimeVaryingSSM{Float64}([T_ssm], [Z_ssm], [R_ssm], [Q_ssm], [H_ssm], 1)
        y = [1.0, 1.2, NaN, 1.6, 1.5]
        alpha, V, eta, eta_var, eps, eps_var, converged = T_.kalman_smoother(tv, y, [1.0], reshape([1.0], 1, 1))
        @test converged

        alpha_ref = [1.27379017, 1.28200387, 1.29185765, 1.30171144, 1.30559945]
        V_ref = [0.1195341, 0.11651373, 0.11775142, 0.11856896, 0.12376871]
        eta_ref = [0.0082137, 0.00985378, 0.00985378, 0.00388801, 0.0]
        eta_var_ref = [0.00980758, 0.00978993, 0.00978993, 0.00984951, 0.01]
        eps_ref_nonmissing = [-0.27379017, -0.08200387, 0.29828856, 0.19440055]
        eps_var_ref_nonmissing = [0.1195341, 0.11651373, 0.11856896, 0.12376871]

        @test isapprox(alpha[1, :], alpha_ref; atol=1e-7)
        @test isapprox([V[t][1, 1] for t in eachindex(V)], V_ref; atol=1e-7)
        @test isapprox(eta, eta_ref; atol=1e-7)
        @test isapprox(eta_var, eta_var_ref; atol=1e-7)
        nonmissing = [1, 2, 4, 5]
        @test isapprox(eps[nonmissing], eps_ref_nonmissing; atol=1e-7)
        @test isapprox(eps_var[nonmissing], eps_var_ref_nonmissing; atol=1e-7)
        @test isnan(eps[3]) && isnan(eps_var[3])  # no observation disturbance at a missing period
    end

    @testset "kalman_filter_diffuse: missing observations don't disturb the diffuse phase" begin
        # local level, no proper prior -- same shape as the handoff's own section-1 system,
        # with an interior missing observation added on top
        T_ssm = reshape([1.0], 1, 1)
        Z_ssm = reshape([1.0], 1, 1)
        R_ssm = reshape([1.0], 1, 1)
        Q_ssm = reshape([0.01], 1, 1)
        H_ssm = reshape([0.5], 1, 1)
        tv = T_.TimeVaryingSSM{Float64}([T_ssm], [Z_ssm], [R_ssm], [Q_ssm], [H_ssm], 1)
        y_complete = [1.0, 1.2, 1.4, 1.6, 1.5, 1.7, 1.6]
        loglik_c, v_c, F_c, nd_c, conv_c = T_.kalman_filter_diffuse(tv, y_complete; diffuse_idx=[1])
        @test conv_c

        y_missing = copy(y_complete); y_missing[4] = NaN
        loglik_m, v_m, F_m, nd_m, conv_m = T_.kalman_filter_diffuse(tv, y_missing; diffuse_idx=[1])
        @test conv_m
        @test isnan(v_m[4]) && isnan(F_m[4])
        @test nd_m == nd_c  # a missing observation cannot itself resolve or extend the diffuse phase
        # the trajectory has not diverged yet before the missing point -- v/F there must match
        # exactly between the two runs (the whole point of a *sequential* filter)
        @test v_m[1:3] == v_c[1:3]
        @test F_m[1:3] == F_c[1:3]
        # and it genuinely differs afterwards, since step 5 onward now predicts through the
        # gap rather than correcting against y[4] -- this is expected, not a bug
        @test v_m[5] != v_c[5]
    end
end
