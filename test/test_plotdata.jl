using DelimitedFiles

# Plot data (handoff/stage 10/plot-recipes-handoff.md): the NUMBERS each
# chart draws, checked against independent computations without a plotting
# backend. Series counts per recipe are in test_recipes.jl, per the
# handoff's split -- a chart is only as right as what it draws.

_pdx(name) = Float64.(readdlm(joinpath(@__DIR__, "fixtures", "var", name), ','; skipstart=1))
_airp() = Float64.(readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)[:, end])

@testset "state_decomposition: ETS, R's plot.ets panels" begin
    y = _airp()
    for (e, t, s) in ((:add, :add, :add), (:mul, :add, :mul), (:add, :none, :none),
                      (:add, :damped, :none), (:mul, :none, :mul), (:mul, :add, :add),
                      (:add, :none, :add))
        m = fit_ets(y, 12; error=e, trend=t, seasonal=s)
        d = state_decomposition(m)
        # the observed series is rebuilt from the model, exactly
        @test d.observed ≈ y rtol=1e-12
        @test d.level == m.level
        @test (d.slope === nothing) == (t === :none)
        t === :none || @test d.slope == m.trend_component
        @test length(d.seasonal) == (s === :none ? 0 : 1)          # at most four panels
        @test d.t == collect(1:144)
        @test startswith(d.model_name, "ETS(")
        if s !== :none
            # R's s1: the seasonal state just updated at t, i.e. the term that
            # enters observation t + m
            @test d.seasonal[1][1:132] == m.seasonal_component[13:144]
            @test d.seasonal[1][133:144] == TSAnalytics._ets_final_seasonal(m)
            @test d.periods == [12.0]
        end
    end
    @test state_decomposition(fit_ets(y, 12; error=:add, trend=:damped, seasonal=:add)).model_name ==
          "ETS(A,Ad,A)"
end

@testset "state_decomposition: TBATS, R's tbats.components" begin
    y = _airp()
    for (m, scale) in ((fit_tbats(y, [12.0]; k=3), identity),
                       (fit_tbats(y, [12.0]; k=2, lambda=0.0), log),
                       (fit_tbats(y, [4.0, 12.0]; k=[1, 1]), identity))
        d = state_decomposition(m)
        P = TSAnalytics._tbats_state_path(m)
        @test P[1, :] == m.seed_states
        @test P[end, :] == TSAnalytics._tbats_final_state(m)     # one loop for both
        @test d.observed ≈ scale.(y) rtol=1e-10                   # on the Box-Cox scale
        @test d.level == P[2:end, 1]
        @test length(d.seasonal) == length(m.periods)
        # THE identity that pins the seasonal panels: with no ARMA, the
        # one-step mean on the transformed scale is the lagged level and slope
        # plus every period's seasonal panel at t
        has_t = m.beta !== nothing
        phi = m.phi === nothing ? 1.0 : m.phi
        mean_t = P[1:end-1, 1] .+ (has_t ? phi .* P[1:end-1, 2] : 0.0) .+ sum(d.seasonal)
        fitted_t = m.lambda === nothing ? m.fitted : boxcox(m.fitted, m.lambda)
        @test mean_t ≈ fitted_t rtol=1e-10
        m.lambda === nothing || @test occursin("Box-Cox", d.model_name)
    end
end

@testset "state_decomposition: Holt-Winters" begin
    y = _airp()
    m = holt_winters(y, 12; trend=:additive, seasonal=:multiplicative)
    d = state_decomposition(m)
    @test d.t == collect(13:144)                     # after the season the initialization uses
    @test d.observed ≈ y[13:end] rtol=1e-12
    @test d.level == m.level && d.slope == m.trend_component
    @test d.seasonal == [m.seasonal_component]
end

@testset "arma_roots: inverse roots of the full polynomials" begin
    y = log.(_airp())
    m = fit_arima(y, (2, 1, 1))
    r = arma_roots(m)
    @test length(r.ar_inverse) == 2 && length(r.ma_inverse) == 1
    # each inverse AR root w solves w^p - phi_1 w^(p-1) - ... - phi_p = 0
    phi = m.arma.ar
    for w in r.ar_inverse
        @test abs(w^2 - phi[1] * w - phi[2]) < 1e-10
    end
    # an inverse MA root solves w^q + theta_1 w^(q-1) + ... = 0
    @test r.ma_inverse[1] ≈ -m.arma.ma[1] atol=1e-12
    # inside the circle iff stationary, by the package's own check
    @test all(abs.(r.ar_inverse) .< 1) == TSAnalytics._ar_stationary(phi)
    # seasonal factors are multiplied out: (1,0,0)(1,0,0)_12 has 13 AR roots
    s = fit_sarima(y, (1, 1, 0), (1, 1, 0, 12))
    rs = arma_roots(s)
    @test length(rs.ar_inverse) == 13 && isempty(rs.ma_inverse)
    # the product of the inverse roots is the constant term's sign-adjusted
    # coefficient: here the full polynomial is (1 - phi B)(1 - Phi B^12)
    @test abs(prod(rs.ar_inverse)) ≈ abs(s.phi[1] * s.Phi[1]) rtol=1e-8
    @test occursin("SARIMA", rs.model_name)
    # pure AR: no MA roots at all
    @test isempty(arma_roots(fit_arima(y, (2, 1, 0))).ma_inverse)
    am = fit_arma(randn(MersenneTwister(3), 300), (1, 1))
    @test length(arma_roots(am).ar_inverse) == 1
end

@testset "ts_display" begin
    y = _airp()
    d = ts_display(y)
    @test d.x == y && d.t == collect(1:144)
    @test d.acf.values == acf(y).values && d.acf.kind === :acf
    @test d.pacf.values == pacf(y).values && d.pacf.kind === :pacf
    d2 = ts_display(y; lags=10)
    @test d2.acf.lags == collect(1:10) && d2.pacf.lags == collect(1:10)
    @test_throws ArgumentError ts_display([1.0, 2.0])
end

@testset "seasonal_plot and lag_plot: data only, no recipe" begin
    c = seasonal_plot(1.0:30.0, 12)
    @test length(c.cycles) == 3 && c.cycle_ids == [1, 2, 3]
    @test c.cycles[1] == collect(1.0:12.0) && c.cycles[3] == collect(25.0:30.0)
    @test c.positions[3] == collect(1:6)
    @test vcat(c.cycles...) == collect(1.0:30.0)
    @test_throws ArgumentError seasonal_plot(1.0:5.0, 12)
    @test_throws ArgumentError seasonal_plot(1.0:30.0, 1)
    l = lag_plot(1.0:10.0, [1, 3])
    @test l.lags == [1, 3]
    @test l.pairs[1] == (collect(1.0:9.0), collect(2.0:10.0))
    @test l.pairs[2] == (collect(1.0:7.0), collect(4.0:10.0))
    @test length(lag_plot(randn(MersenneTwister(1), 50)).pairs) == 4
    @test_throws ArgumentError lag_plot(1.0:10.0, [0])
    @test_throws ArgumentError lag_plot(1.0:10.0, Int[])
    # the refusal is documented, in seasonal_subseries_plot's words
    for f in (seasonal_plot, lag_plot)
        doc = string(Base.Docs.doc(f))
        @test occursin("no one natural", doc) && occursin("rendering", doc)
    end
end

@testset "roots_plot" begin
    m = fit_var(_pdx("var2_p1.csv"), 2)
    r = roots_plot(m)
    @test sort(abs.(r.eigenvalues); rev=true) ≈ roots(m) atol=1e-12
    @test r.stable
    @test !roots_plot(fit_var(_pdx("var2_explosive.csv"), 1)).stable
    v = fit_vecm(_pdx("coint3_r1.csv"); lags=1, rank=1)
    rv = roots_plot(v)
    @test count(x -> abs(abs(x) - 1) < 1e-8, rv.eigenvalues) == 2      # K - r unit roots
    ev = roots_plot(fit_var(_pdx("var3_p2.csv"), 2)).eigenvalues
    @test sort(imag.(ev)) ≈ -reverse(sort(imag.(ev))) atol=1e-12       # conjugate pairs
end

@testset "diagnostic_plot(::VARModel): DiagnosticPlotResult per equation" begin
    m = fit_var(_pdx("var3_p2.csv"), 2)
    d = diagnostic_plot(m)
    @test d.names == m.names && length(d.equations) == 3
    for i in 1:3
        ref = diagnostic_plot(m.resid[:, i])
        @test d.equations[i].std_resid == ref.std_resid
        @test d.equations[i].acf == ref.acf
        @test d.equations[i].ljungbox_pvalues == ref.ljungbox_pvalues
    end
    @test length(diagnostic_plot(m; lags=8).equations[1].acf) == 8
    @test diagnostic_plot(randn(MersenneTwister(1), 200)) isa DiagnosticPlotResult
end

@testset "cointegration_plot: the error-correction term, with its ACF" begin
    v = fit_vecm(_pdx("coint3_r1.csv"); lags=1, rank=1)
    c = cointegration_plot(v)
    W, X0, X1, T, _ = TSAnalytics._vecm_design(v.Y, 1, :const)
    @test c.relations ≈ X1 * v.beta
    @test (hcat(c.relations, W) \ X0)[1, :] ≈ vec(v.alpha)      # IS what the VECM regresses on
    @test c.t == collect(3:(2 + T))
    @test length(c.acfs) == 1 && c.acfs[1].values == acf(c.relations[:, 1]).values
    Y = _pdx("coint3_r1.csv")
    @test std(c.relations[:, 1]) < 0.2 * minimum(std(Y; dims=1))   # it is stationary
    vr = fit_vecm(_pdx("coint2_yes.csv"); lags=1, rank=1, deterministic=:const_restricted)
    @test size(cointegration_plot(vr).beta) == (3, 1)
    j = johansen_test(_pdx("coint3_r2.csv"); lags=1)
    cj = cointegration_plot(_pdx("coint3_r2.csv"), j)
    @test size(cj.relations, 2) == coint_rank(j) == 2 && length(cj.acfs) == 2
    @test cj.beta ≈ normalize_beta(j, 2; scheme=:identity)
    @test size(cointegration_plot(_pdx("coint3_r2.csv"), j; rank=1).relations) == (248, 1)
    j0 = johansen_test(_pdx("coint2_no.csv"); lags=1)
    e = try cointegration_plot(_pdx("coint2_no.csv"), j0) catch err; err end
    @test e isa ArgumentError && occursin("no cointegrating relation", sprint(showerror, e))
    @test_throws ArgumentError cointegration_plot(_pdx("coint2_yes.csv"), j)
    @test_throws ArgumentError cointegration_plot(_pdx("coint3_r2.csv")[1:200, :], j)
end

@testset "min-max decimation keeps both extremes" begin
    x = collect(1:20_000); y = sin.(x ./ 50) .+ (x .== 7_777) .* 5 .- (x .== 12_345) .* 5
    xd, yd = TSAnalytics._decimate_minmax(x, y, 1000)
    @test length(xd) <= 1000
    @test maximum(yd) == maximum(y) && minimum(yd) == minimum(y)
    @test issorted(xd)
    @test TSAnalytics._decimate_minmax(1:10, collect(1.0:10.0), 100) == (collect(1:10), collect(1.0:10.0))
end
