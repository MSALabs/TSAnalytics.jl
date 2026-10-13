using DelimitedFiles

# Stage 10.1: VAR and lag-order selection.
#
# DUAL-VERIFIED against vars 1.6.1 on R 4.6.0 and statsmodels 0.14.1. Every
# reference value is loaded from test/fixtures/var/refs.jl, generated from the
# reference JSONs by test/verification/var/json_to_jl.py — so the tests assert
# against the whole bundle rather than a handful of transcribed numbers, with
# no JSON dependency added (charter §6.5 keeps Project.toml unchanged).
#
# The JSON key names carry one trap worth stating here, because it would
# otherwise look like a typo. R's `A_p1_trend` is type="trend" — a trend with
# NO constant — while Python's `A_p1_trend` is trend="ct", constant AND trend.
# They are DIFFERENT MODELS under the same key: logLik -506.68 against
# -500.53. Python's `A_p1_trend` corresponds to R's `A_p1_both`. The mapping
# below is explicit for exactly that reason.

if !isdefined(@__MODULE__, :REF_R)
    include(joinpath(@__DIR__, "fixtures", "var", "refs.jl"))
end

_vfix(name) = readdlm(joinpath(@__DIR__, "fixtures", "var", name), ','; skipstart=1)

# my `deterministic` => (R key suffix, Python key suffix); `nothing` = no
# counterpart in that reference, which is charter §5.2's point
const DET_KEYS = Dict(
    :const     => ("const", "const"),
    :none      => ("none", "none"),
    :trend     => ("trend", nothing),      # R only: trend, no constant
    :both      => ("both", "trend"),       # py calls this one "trend"
    :quadratic => (nothing, "ctt"),        # py only
)

"Reorder statsmodels' params/stderr — (ndet+Kp) × K, deterministic FIRST —
into this package's K × (Kp+ndet) layout with deterministic LAST."
function _py_to_mine(M, K::Int, p::Int, ndet::Int)
    A = reduce(hcat, [Float64.(r) for r in M])'   # (ndet+Kp) × K
    out = Matrix{Float64}(undef, K, K * p + ndet)
    for j in 1:K
        for i in 1:(K * p)
            out[j, i] = A[ndet + i, j]
        end
        for d in 1:ndet
            out[j, K * p + d] = A[d, j]
        end
    end
    return out
end

_mat(x) = reduce(vcat, [reshape(Float64.(r), 1, :) for r in x])

@testset "Stage 10.1 — VAR and lag-order selection" begin
    T = TSAnalytics
    A2 = _vfix("var2_p1.csv")
    B3 = _vfix("var3_p2.csv")
    SH = _vfix("var2_short.csv")
    US = _vfix("uschange.csv")
    E5 = _vfix("econ5.csv")

    # (label, data, p, deterministic, R key, py key)
    CASES = [
        ("A_p1_const", A2, 1, :const), ("A_p2_const", A2, 2, :const),
        ("A_p1_none", A2, 1, :none),   ("B_p2_const", B3, 2, :const),
        ("B_p3_const", B3, 3, :const), ("SH_p2_const", SH, 2, :const),
        ("US_p2_const", US, 2, :const), ("E5_p2_const", E5, 2, :const),
    ]

    @testset "coefficients match R's Bcoef layout exactly — $lbl" for
            (lbl, Y, p, det) in CASES
        m = fit_var(Y, p; deterministic=det)
        r = REF_R[lbl]
        # R's Bcoef is K × (Kp + ndet) with deterministic LAST, which is
        # precisely this package's coef() layout
        @test isapprox(coef(m), _mat(r["Bcoef"]); rtol=1e-10)
        @test m.T == r["obs"]
        @test length(coefnames(m)) == size(coef(m), 2)
    end

    @testset "coefficients match statsmodels once reordered — $lbl" for
            (lbl, Y, p, det) in CASES
        m = fit_var(Y, p; deterministic=det)
        q = REF_PY[lbl]
        K = length(m.names)
        ndet = T._var_ndet(det)
        @test isapprox(coef(m), _py_to_mine(q["params"], K, p, ndet); rtol=1e-9)
        # statsmodels' `coefs` is (p, K, K) and indexes exactly like this
        # package's A — the one place the two layouts already agree
        for j in 1:p
            @test isapprox(m.A[j], _mat(q["coefs"][j]); rtol=1e-9)
        end
        @test m.T == q["nobs"]
    end

    @testset "both Σ conventions, against both references — $lbl" for
            (lbl, Y, p, det) in CASES
        m = fit_var(Y, p; deterministic=det)
        # charter §5.5: three conventions exist and all three agree across
        # references, so both are stored and both are pinned
        @test isapprox(m.sigma_df, _mat(REF_R[lbl]["Sigma_df"]); rtol=1e-9)
        @test isapprox(m.sigma_mle, _mat(REF_R[lbl]["Sigma_mle"]); rtol=1e-9)
        @test isapprox(m.sigma_df, _mat(REF_PY[lbl]["sigma_u"]); rtol=1e-9)
        @test isapprox(m.sigma_mle, _mat(REF_PY[lbl]["sigma_u_mle"]); rtol=1e-9)
        # the df version is the larger one, and by the exact factor
        nfree = length(m.names) * p + T._var_ndet(det)
        @test isapprox(m.sigma_df[1, 1] / m.sigma_mle[1, 1],
                       m.T / (m.T - nfree); rtol=1e-12)
    end

    @testset "log-likelihood against both references — $lbl" for
            (lbl, Y, p, det) in CASES
        m = fit_var(Y, p; deterministic=det)
        @test isapprox(m.loglik, REF_R[lbl]["logLik"]; rtol=1e-11)
        @test isapprox(m.loglik, REF_PY[lbl]["llf"]; rtol=1e-11)
        @test isapprox(loglikelihood(m), m.loglik; rtol=1e-14)
    end

    @testset "standard errors and t statistics — $lbl" for (lbl, Y, p, det) in CASES
        m = fit_var(Y, p; deterministic=det)
        K = length(m.names)
        ndet = T._var_ndet(det)
        # R: per-equation coef(summary(e)), keyed by equation name, in Bcoef
        # column order
        for (i, nm) in enumerate(REF_R[lbl]["Bcoef_rownames"])
            @test isapprox(m.se[i, :], Float64.(REF_R[lbl]["coef_se"][nm]); rtol=1e-8)
            @test isapprox(coef(m)[i, :], Float64.(REF_R[lbl]["coef_est"][nm]); rtol=1e-9)
        end
        @test isapprox(m.se, _py_to_mine(REF_PY[lbl]["stderr"], K, p, ndet); rtol=1e-8)
        tstat = coef(m) ./ m.se
        @test isapprox(tstat, _py_to_mine(REF_PY[lbl]["tvalues"], K, p, ndet); rtol=1e-7)
        @test isapprox(stderror(m), m.se; rtol=1e-14)
    end

    @testset "residuals and fitted values — $lbl" for (lbl, Y, p, det) in CASES
        m = fit_var(Y, p; deterministic=det)
        @test isapprox(residuals(m)[1:3, :], _mat(REF_R[lbl]["resid_head"]); rtol=1e-8)
        @test isapprox(residuals(m)[end-2:end, :], _mat(REF_R[lbl]["resid_tail"]); rtol=1e-8)
        @test isapprox(fitted(m)[1:3, :], _mat(REF_R[lbl]["fitted_head"]); rtol=1e-8)
        @test isapprox(residuals(m)[1:3, :], _mat(REF_PY[lbl]["resid_head"]); rtol=1e-8)
        # fitted + resid reconstructs the data, which is the real invariant
        @test isapprox(fitted(m) .+ residuals(m), Y[(p+1):end, :]; rtol=1e-10)
    end

    @testset "DIVERGENCE: roots are exact reciprocals between references" begin
        # Charter §5.3, the stage's most dangerous trap: getting this wrong
        # silently INVERTS a stability test. This package follows R.
        for (lbl, Y, p, det) in CASES
            m = fit_var(Y, p; deterministic=det)
            @test isapprox(roots(m), Float64.(REF_R[lbl]["roots"]); rtol=1e-9)
            @test isapprox(maximum(roots(m)), REF_R[lbl]["roots_mod_max"]; rtol=1e-9)
            # statsmodels reports characteristic-polynomial roots, whose moduli
            # are the reciprocals -- stable if > 1 there, < 1 here
            py = sort(Float64.(REF_PY[lbl]["roots"]); rev=false)
            @test isapprox(sort(1 ./ roots(m)), py; rtol=1e-8)
            @test is_stable(m) == REF_PY[lbl]["is_stable"]
        end
    end

    @testset "stability at and over the boundary" begin
        # ref_R_extra.json: the near-unit and explosive fixtures
        nu = fit_var(_vfix("var2_nearunit.csv"), 1)
        @test isapprox(roots(nu), Float64.(REF_R_EXTRA["nearunit"]["roots"]); rtol=1e-9)
        @test isapprox(maximum(roots(nu)), 0.9744941374102439; rtol=1e-10)
        @test is_stable(nu)                       # 0.974 < 1, only just
        ex = fit_var(_vfix("var2_explosive.csv"), 1)
        @test isapprox(roots(ex), Float64.(REF_R_EXTRA["explosive"]["roots"]); rtol=1e-9)
        @test isapprox(maximum(roots(ex)), 1.0404185209834451; rtol=1e-10)
        @test !is_stable(ex)                      # and this one is not
        @test occursin("NOT STABLE", sprint(show, ex))
    end

    @testset "fitted-model information criteria, against statsmodels — $lbl" for
            (lbl, Y, p, det) in CASES
        m = fit_var(Y, p; deterministic=det)
        q = REF_PY[lbl]
        @test isapprox(aic(m), q["aic"]; rtol=1e-10)
        @test isapprox(bic(m), q["bic"]; rtol=1e-10)
        @test isapprox(T.hqic(m), q["hqic"]; rtol=1e-10)
        @test isapprox(T.fpe(m), q["fpe"]; rtol=1e-9)
    end

    @testset "order selection against both references — $key" for key in
            ("sel_A_const", "sel_A_none", "sel_B_const", "sel_US_const", "sel_E5_const")
        Y, lmax, det = if key == "sel_A_const"; (A2, 8, :const)
        elseif key == "sel_A_none"; (A2, 8, :none)
        elseif key == "sel_B_const"; (B3, 8, :const)
        elseif key == "sel_US_const"; (US, 8, :const)
        else; (E5, 6, :const) end
        s = var_select(Y; lag_max=lmax, deterministic=det, parallel=false)
        r = REF_R[key]
        # R's criteria rows are ordered AIC, HQ, SC, FPE -- note HQ before SC
        @test r["rownames"] == Any["AIC(n)", "HQ(n)", "SC(n)", "FPE(n)"]
        crit = _mat(r["criteria"])
        @test isapprox(s.aic, crit[1, :]; rtol=1e-9)
        @test isapprox(s.hqic, crit[2, :]; rtol=1e-9)
        @test isapprox(s.bic, crit[3, :]; rtol=1e-9)
        @test isapprox(s.fpe, crit[4, :]; rtol=1e-8)
        @test [s.selected.aic, s.selected.hqic, s.selected.bic, s.selected.fpe] ==
              Int.(r["selection"])
        # statsmodels' ics normally run over lags 0…L, giving L+1 entries --
        # but NOT with trend="n", where a lag-0 model has no regressors at all
        # and statsmodels silently drops that row, giving L. So charter §5.4's
        # "+1 when comparing" holds only when a deterministic term is present.
        # Indexing from the end is right either way.
        q = REF_PY[key]
        nics = length(q["ics"]["aic"])
        @test nics == (det === :none ? lmax : lmax + 1)
        off = nics - lmax
        for j in 1:lmax
            @test isapprox(s.aic[j], q["ics"]["aic"][j+off]; rtol=1e-9)
            @test isapprox(s.bic[j], q["ics"]["bic"][j+off]; rtol=1e-9)
            @test isapprox(s.hqic[j], q["ics"]["hqic"][j+off]; rtol=1e-9)
            @test isapprox(s.fpe[j], q["ics"]["fpe"][j+off]; rtol=1e-8)
        end
        @test s.selected.aic == q["selected"]["aic"]
        @test s.selected.bic == q["selected"]["bic"]
        @test s.T == size(Y, 1) - lmax
    end

    @testset "CONVENTION: selection holds the sample fixed, a fitted model does not" begin
        # Charter §5.4. Both numbers are pinned so the difference can never be
        # mistaken for a bug.
        s = var_select(A2; lag_max=8, parallel=false)
        m1 = fit_var(A2, 1)
        @test isapprox(s.aic[1], -0.6050354269424555; rtol=1e-12)   # T = 192
        @test isapprox(aic(m1), -0.5819716231697688; rtol=1e-12)    # T = 199
        @test s.T == 192
        @test m1.T == 199
        @test !isapprox(s.aic[1], aic(m1); rtol=1e-3)
        # and slicing the data by hand reproduces the selection figure exactly,
        # which is what makes it a convention rather than a discrepancy
        sliced = fit_var(A2[8:end, :], 1)
        @test sliced.T == 192
        @test isapprox(aic(sliced), s.aic[1]; rtol=1e-12)
    end

    @testset "order selection recovers the true order" begin
        # var3_p2.csv is a genuine VAR(2) -- gen_fixtures.R builds it with a
        # second lag matrix B2 whose non-zero entries are 0.20, 0.15, 0.20 and
        # 0.05 -- so selection has something to find.
        s = var_select(B3; lag_max=8, parallel=false)
        @test s.selected.aic == 2
        @test s.selected.hqic == 2
        @test s.selected.fpe == 2
        # BIC picks 1, not 2, and that is not a failure: with 9 extra
        # parameters per lag on 3 variables and a log(T) penalty, BIC is not
        # paid back by B2's small coefficients. BOTH references select 1 here
        # too -- which the reference comparison above already pins, since
        # s.selected is asserted equal to R's `selection` and py's `selected`.
        @test s.selected.bic == 1
        @test Int.(REF_R["sel_B_const"]["selection"]) == [2, 2, 1, 2]   # AIC HQ SC FPE
        @test REF_PY["sel_B_const"]["selected"]["bic"] == 1
        @test REF_PY["sel_B_const"]["selected"]["aic"] == 2
    end

    @testset "all five deterministic specifications, and the :trend/:both trap" begin
        # Charter §5.2: the vocabularies are mutually incomplete, and
        # R's type="trend" is NOT statsmodels' trend="ct".
        LL = Dict(:const => -500.8313597100677, :none => -522.3719059322,
                  :trend => -506.6828693867, :both => -500.5285172459,
                  :quadratic => -497.6112301085)
        for det in (:const, :none, :trend, :both, :quadratic)
            m = fit_var(A2, 1; deterministic=det)
            @test isapprox(m.loglik, LL[det]; rtol=1e-9)
            @test size(m.det_coef, 2) == T._var_ndet(det)
        end
        # :trend has ndet=1 and :both has ndet=2 -- the heart of the trap
        @test T._var_ndet(:trend) == 1
        @test T._var_ndet(:both) == 2
        # and they differ by ~6.15 in log-likelihood, so a mix-up is not subtle
        @test isapprox(fit_var(A2, 1; deterministic=:both).loglik -
                       fit_var(A2, 1; deterministic=:trend).loglik, 6.1543521; atol=1e-5)
        # R's A_p1_trend and py's A_p1_trend really are different models
        @test isapprox(REF_R["A_p1_trend"]["logLik"], -506.6828693867; rtol=1e-9)
        @test isapprox(REF_PY["A_p1_trend"]["llf"], -500.5285172459; rtol=1e-9)
        @test isapprox(fit_var(A2, 1; deterministic=:trend).loglik,
                       REF_R["A_p1_trend"]["logLik"]; rtol=1e-10)
        @test isapprox(fit_var(A2, 1; deterministic=:both).loglik,
                       REF_PY["A_p1_trend"]["llf"]; rtol=1e-10)
        @test isapprox(fit_var(A2, 1; deterministic=:quadratic).loglik,
                       REF_PY["A_p1_ctt"]["llf"]; rtol=1e-10)
    end

    @testset "CONVENTION: the trend column is the calendar index p+1:n" begin
        # Charter §5.4b. This one is silent: an origin shift is absorbed
        # entirely by the constant, so the fit, residuals, Σ, log-likelihood
        # and every criterion are IDENTICAL either way. Only the intercept
        # moves, by exactly p × the trend coefficient. So the test must assert
        # the intercept, or it passes while being wrong.
        p = 1
        m = fit_var(A2, p; deterministic=:both)
        ref = _mat(REF_R["A_p1_both"]["Bcoef"])
        @test isapprox(coef(m), ref; rtol=1e-10)
        const_col = size(ref, 2) - 1          # (…lags…, const, trend)
        trend_col = size(ref, 2)
        @test isapprox(m.det_coef[:, 1], ref[:, const_col]; atol=1e-12)
        @test isapprox(m.det_coef[:, 2], ref[:, trend_col]; atol=1e-14)
        # the within-sample form 1:T would shift the intercept by p × trend
        wrong = m.det_coef[:, 1] .+ p .* m.det_coef[:, 2]
        @test maximum(abs.(wrong .- ref[:, const_col])) > 1e-5
        # both references agree with each other here, so there is one right
        # answer rather than a convention to choose
        @test isapprox(m.det_coef[:, 1], Float64.(REF_PY["A_p1_trend"]["intercept"]);
                       rtol=1e-9)
    end

    @testset "forecast point and intervals, against both references" begin
        for (lbl, Y, p, det) in CASES
            h = lbl == "US_p2_const" ? 8 : (lbl == "E5_p2_const" ? 4 : 5)
            m = fit_var(Y, p; deterministic=det)
            f = forecast(m, h; levels=[95.0])
            @test size(f.point) == (h, length(m.names))
            @test isapprox(f.point, _mat(REF_R[lbl]["fcst_point"]); rtol=1e-8)
            @test isapprox(f.point, _mat(REF_PY[lbl]["forecast"]); rtol=1e-8)
            @test isapprox(f.lower[:, :, 1], _mat(REF_R[lbl]["fcst_lower"]); rtol=1e-8)
            @test isapprox(f.upper[:, :, 1], _mat(REF_R[lbl]["fcst_upper"]); rtol=1e-8)
            # statsmodels' forecast_interval is (point, lower, upper)
            pi = REF_PY[lbl]["forecast_interval"]
            @test isapprox(f.lower[:, :, 1], _mat(pi[2]); rtol=1e-8)
            @test isapprox(f.upper[:, :, 1], _mat(pi[3]); rtol=1e-8)
        end
    end

    @testset "the h=1 forecast covariance IS Σ_df, as in both references" begin
        # Charter §6.7: both references use the df-adjusted Σ and ignore
        # parameter estimation uncertainty. Confirmed because forecast_cov(1)
        # equals sigma_u element for element.
        m = fit_var(A2, 1)
        f = forecast(m, 3)
        @test isapprox(f.se[1, :], sqrt.([m.sigma_df[1, 1], m.sigma_df[2, 2]]); rtol=1e-12)
        fc = _mat(REF_PY["A_p1_const"]["forecast_cov"][1])
        @test isapprox(fc, m.sigma_df; rtol=1e-9)
        # and the bands widen with the horizon
        @test all(f.se[2, :] .>= f.se[1, :])
        @test all(f.se[3, :] .>= f.se[2, :])
    end

    @testset "interval levels nest, and predict is the same function" begin
        m = fit_var(A2, 1)
        f = forecast(m, 6; levels=[80.0, 95.0])
        @test size(f.lower) == (6, 2, 2)
        @test all(f.lower[:, :, 2] .<= f.lower[:, :, 1])
        @test all(f.upper[:, :, 2] .>= f.upper[:, :, 1])
        @test predict(m, 6; levels=[80.0, 95.0]).point == f.point
        @test f.model_name == "VAR(1)"
        @test f.names == m.names
        @test occursin("VAR(1) forecast", sprint(show, f))
    end

    @testset "REDUCTION: a VAR with a diagonal A is K independent ARs" begin
        # Built here rather than assumed: generate two series that do not
        # interact, fit a VAR(1), and the off-diagonal coefficients should be
        # indistinguishable from zero while the diagonal matches each
        # series' own AR(1) fit through the already-verified _ols path.
        # n is large on purpose: the sampling standard error on an AR(1)
        # coefficient is about 1/sqrt(n), so a short series would need a
        # tolerance so wide the test would stop meaning anything.
        n = 4000
        rng = MersenneTwister(4242)
        y1 = zeros(n); y2 = zeros(n)
        for t in 2:n
            y1[t] = 0.6 * y1[t-1] + randn(rng)
            y2[t] = -0.3 * y2[t-1] + randn(rng)
        end
        Y = hcat(y1, y2)
        m = fit_var(Y, 1)
        # the off-diagonals are indistinguishable from zero: within 3 of their
        # own standard errors, which is the self-calibrating form of the claim
        @test abs(m.A[1][1, 2]) < 3 * m.se[1, 2]
        @test abs(m.A[1][2, 1]) < 3 * m.se[2, 1]
        @test abs(m.A[1][1, 2]) < 0.04
        @test abs(m.A[1][2, 1]) < 0.04
        @test isapprox(m.A[1][1, 1], 0.6; atol=0.03)
        @test isapprox(m.A[1][2, 2], -0.3; atol=0.03)
        # each diagonal entry matches that series' own univariate regression
        for (j, y) in enumerate((y1, y2))
            Z = hcat(y[1:(n-1)], ones(n - 1))
            b, _, _ = T._ols(Z, y[2:n])
            @test isapprox(m.A[1][j, j], b[1]; atol=0.02)
        end
    end

    @testset "parameter recovery from the known DGP" begin
        # gen_fixtures.R builds var2_p1.csv with
        #   A₁ = matrix(c(0.5, -0.1, 0.2, 0.3), 2, 2, byrow=TRUE)   # rows = equations
        #   ν  = c(0.4, -0.2)
        #   Σ  = matrix(c(1.0, 0.3, 0.3, 0.75), 2, 2)
        # on n = 200 after a 100-row burn-in. Read from the generator, not
        # assumed -- and the orientation matters: against A₁' the worst
        # coefficient error would be 0.34 rather than 0.11.
        m = fit_var(A2, 1)
        Atrue = [0.5 -0.1; 0.2 0.3]
        nutrue = [0.4, -0.2]
        Stru = [1.0 0.3; 0.3 0.75]

        # Self-calibrating: every coefficient is within 3 of ITS OWN standard
        # error of the truth. At n = 200 a fixed tolerance would either be so
        # tight it fails on sampling noise or so loose it proves nothing.
        for i in 1:2, j in 1:2
            @test abs(m.A[1][i, j] - Atrue[i, j]) < 3 * m.se[i, j]
        end
        for i in 1:2
            @test abs(m.det_coef[i, 1] - nutrue[i]) < 3 * m.se[i, 3]
        end
        # the wrong orientation is firmly rejected, which is what confirms the
        # generator was read correctly rather than guessed
        @test maximum(abs.(m.A[1] .- Atrue)) < maximum(abs.(m.A[1] .- transpose(Atrue)))

        # Σ: the asymptotic standard error of a variance estimate is
        # sigma_ij * sqrt(2/T), so scale the comparison that way instead of
        # picking a number
        for i in 1:2, j in 1:2
            tol = 3 * sqrt(2 / m.T) * sqrt(Stru[i, i] * Stru[j, j])
            @test abs(m.sigma_df[i, j] - Stru[i, j]) < tol
        end
    end

    @testset "REFUSALS: every input both references wave through" begin
        # Charter §5.8/§5.9. This is the honest-refusal pattern, and here it
        # is strictly better than both references.
        @test REF_R_EXTRA["err_p_too_large"] == "no error"      # R accepts it
        @test REF_PY_EXTRA["err_p_too_large"] == "no error"     # so does py
        e = try; fit_var(A2[1:8, :], 5); nothing; catch err; err; end
        @test e isa ArgumentError
        @test occursin("not enough observations", e.msg)
        @test occursin("Reduce p", e.msg)                        # names the fix

        # p = 0: R errors obscurely, py not at all
        @test occursin("dimnames", REF_R_EXTRA["err_p_zero"])
        e0 = try; fit_var(A2, 0); nothing; catch err; err; end
        @test e0 isa ArgumentError
        @test occursin("p must be at least 1", e0.msg)

        # k = 1: both references refuse, and so does this
        @test occursin("at least two variables", REF_R["k1_refusal_R"])
        @test occursin("Only gave one variable", REF_PY["k1_refusal_py"])
        e1 = try; fit_var(_vfix("var1_k1.csv"), 1); nothing; catch err; err; end
        @test e1 isa ArgumentError
        @test occursin("at least two variables", e1.msg)
        @test occursin("fit_arima", e1.msg)                      # names the route

        # NaN
        Yn = copy(A2); Yn[5, 1] = NaN
        en = try; fit_var(Yn, 1); nothing; catch err; err; end
        @test en isa ArgumentError
        @test occursin("NaN", en.msg)
        Yi = copy(A2); Yi[5, 1] = Inf
        @test_throws ArgumentError fit_var(Yi, 1)

        # an unrecognised deterministic, with the trap spelled out
        eb = try; fit_var(A2, 1; deterministic=:ct); nothing; catch err; err; end
        @test eb isa ArgumentError
        @test occursin(":quadratic", eb.msg)
        @test occursin("NO constant", eb.msg)
        @test_throws ArgumentError fit_var(A2, 1; deterministic=:c)

        # and var_select guards the same ground
        @test_throws ArgumentError var_select(A2; lag_max=0)
        @test_throws ArgumentError var_select(_vfix("var1_k1.csv"))
        @test_throws ArgumentError var_select(A2[1:12, :]; lag_max=8)
        @test_throws ArgumentError forecast(fit_var(A2, 1), 0)
        @test_throws ArgumentError forecast(fit_var(A2, 1), 4; levels=[0.0])
    end

    @testset "StatsAPI contract — the fullest of any model here" begin
        m = fit_var(A2, 2)
        K = 2; p = 2; ndet = 1
        @test size(coef(m)) == (K, K * p + ndet)
        @test size(vcov(m)) == (K * (K * p + ndet), K * (K * p + ndet))
        @test size(residuals(m)) == (m.T, K)
        @test size(fitted(m)) == (m.T, K)
        @test nobs(m) == m.T
        @test TSAnalytics.StatsAPI.dof(m) == K * (K * p + ndet)
        @test isfinite(loglikelihood(m)) && isfinite(aic(m)) && isfinite(bic(m))
        @test coefnames(m) == [:y1_l1, :y2_l1, :y1_l2, :y2_l2, :const]
        @test coefnames(fit_var(A2, 1; deterministic=:quadratic)) ==
              [:y1_l1, :y2_l1, :const, :trend, :trend2]
        @test coefnames(fit_var(A2, 1; deterministic=:trend)) == [:y1_l1, :y2_l1, :trend]
        # vcov is the full Kronecker product, and se is its diagonal reshaped
        V = vcov(m)
        for j in 1:K, i in 1:(K * p + ndet)
            @test isapprox(m.se[j, i], sqrt(V[(j-1)*(K*p+ndet) + i, (j-1)*(K*p+ndet) + i]);
                           rtol=1e-10)
        end
        @test m isa TSAnalytics.TimeSeriesModel
        @test m.converged
    end

    @testset "container-agnostic input through tsmatrix" begin
        a = fit_var(A2, 1)
        b = fit_var((y1 = A2[:, 1], y2 = A2[:, 2]), 1)
        @test isapprox(coef(a), coef(b); rtol=1e-12)
        @test b.names == [:y1, :y2]                   # names carried through
        c = fit_var([A2[:, 1], A2[:, 2]], 1)
        @test isapprox(coef(a), coef(c); rtol=1e-12)
        # a real dataset through its NamedTuple, names and all
        d = fit_var(dataset("econ5"), 2)
        @test d.names == [:unemp, :gnp, :consum, :govinv, :prinv]
        @test isapprox(coef(d), coef(fit_var(E5, 2)); rtol=1e-12)
        @test coefnames(d)[1] == :unemp_l1
    end

    @testset "QR and Cholesky agree, including on the conditioning stress case" begin
        for (Y, p) in ((A2, 1), (B3, 2), (E5, 2))
            q = fit_var(Y, p; method=:qr)
            c = fit_var(Y, p; method=:cholesky)
            @test isapprox(coef(q), coef(c); rtol=1e-6)
            @test isapprox(q.loglik, c.loglik; rtol=1e-9)
        end
        # econ5 is the fixture that would catch a conditioning mistake: values
        # of order 1e4, where reference-vs-reference agreement itself degrades
        # from ~1e-16 to ~5e-13
        m = fit_var(E5, 2)
        @test isapprox(coef(m), _mat(REF_R["E5_p2_const"]["Bcoef"]); rtol=1e-8)
        @test maximum(abs.(E5)) > 1e3
    end

    @testset "parallel and serial selection pick the identical model" begin
        for (Y, lmax) in ((A2, 8), (B3, 8), (US, 8))
            sp = var_select(Y; lag_max=lmax, parallel=true)
            ss = var_select(Y; lag_max=lmax, parallel=false)
            @test sp.selected == ss.selected
            @test isapprox(sp.aic, ss.aic; rtol=1e-12)
            @test isapprox(sp.fpe, ss.fpe; rtol=1e-12)
        end
    end

    @testset "the MA representation, and its reduction at p=1" begin
        A1 = [0.5 -0.1; 0.2 0.3]
        Phi = T._ma_rep([A1], 4)
        @test Phi[1] == I(2)
        @test isapprox(Phi[2], A1; rtol=1e-14)        # Φ₁ = A₁
        @test isapprox(Phi[3], A1 * A1; rtol=1e-14)   # Φ₂ = A₁²
        @test isapprox(Phi[5], A1^4; rtol=1e-13)
        # with two lags the recursion is Φ_i = Φ_{i-1}A₁ + Φ_{i-2}A₂
        A2m = [0.1 0.0; 0.0 0.2]
        P2 = T._ma_rep([A1, A2m], 3)
        @test isapprox(P2[3], A1 * A1 + A2m; rtol=1e-13)
        @test isapprox(P2[4], P2[3] * A1 + P2[2] * A2m; rtol=1e-13)
    end

    @testset "companion, and the forecast recursion's allocations" begin
        m = fit_var(A2, 2)
        C = companion(m)
        @test size(C) == (4, 4)
        @test C[1:2, 1:2] == m.A[1]
        @test C[3:4, 1:2] == I(2)
        @test isapprox(sort(abs.(eigvals(C)); rev=true), roots(m); rtol=1e-12)
        # type stability on the exported entry points
        @test (@inferred fit_var(A2, 1)) isa VARModel
        @test (@inferred roots(m)) isa Vector{Float64}
        @test (@inferred is_stable(m)) isa Bool
        @test eltype(m.A) == Matrix{Float64}          # concrete, not Vector{Matrix}
    end

    @testset "show methods report the load-bearing facts" begin
        out = sprint(show, fit_var(A2, 2))
        @test occursin("VAR(2)", out)
        @test occursin("K=2", out)
        @test occursin("deterministic=:const", out)
        @test occursin("stable", out)
        @test occursin("logLik", out)
        s = sprint(show, var_select(A2; lag_max=4, parallel=false))
        @test occursin("held fixed", out * s)
        @test occursin("selected", s)
        @test occursin("AIC", s)
    end
end
