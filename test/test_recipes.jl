using RecipesBase
using DelimitedFiles

@testset "RecipesBase recipes -- dispatch and produce series (no rendering backend needed)" begin
    Random.seed!(1)
    y = 100 .+ 10 .* sin.(2π .* (1:96) ./ 12) .+ randn(96)

    r_acf = acf(y, collect(1:10))
    rec_acf = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r_acf)
    @test !isempty(rec_acf)
    @test length(rec_acf) == 3   # confidence band + zero line + main sticks series

    r_pacf = pacf(y, collect(1:10))
    rec_pacf = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r_pacf)
    @test !isempty(rec_pacf)

    for d in (classical_decompose(y, 12), stl_decompose(y, 12), mstl_decompose(y, [12]))
        rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), d)
        @test !isempty(rec)
        @test length(rec) == 4   # observed/trend/seasonal/remainder
    end

    md = mstl_decompose(200 .+ 20 .* sin.(2π .* (1:400) ./ 24) .+ randn(MersenneTwister(2), 400), [24, 168])
    rec_md = RecipesBase.apply_recipe(Dict{Symbol,Any}(), md)
    @test length(rec_md) == 5   # observed/trend/2 seasonal panels/remainder

    dpr = diagnostic_plot(randn(MersenneTwister(3), 200))
    rec_dp = RecipesBase.apply_recipe(Dict{Symbol,Any}(), dpr)
    @test length(rec_dp) == 6   # std_resid, acf, qq scatter+line, lb scatter+hline
end

@testset "seasonal_subseries_plot" begin
    x = 100 .+ 10 .* sin.(2π .* (1:96) ./ 12) .+ (1:96) ./ 10
    r = seasonal_subseries_plot(x, 12)

    @test length(r.means) == 12
    @test length(r.values) == length(r.position) == length(r.cycle) == 96
    @test r.position[1:14] == [1,2,3,4,5,6,7,8,9,10,11,12,1,2]
    @test r.cycle[1:12] == fill(1, 12)
    @test r.cycle[13] == 2 && r.cycle[14] == 2

    # each position's mean matches a direct computation
    for p in 1:12
        idx = findall(==(p), r.position)
        @test isapprox(r.means[p], sum(r.values[idx]) / length(idx); atol=1e-10)
    end

    # container-agnostic
    @test seasonal_subseries_plot(x, 12).means == seasonal_subseries_plot(collect(x), 12).means

    @test_throws ArgumentError seasonal_subseries_plot(x, 1)
    @test_throws ArgumentError seasonal_subseries_plot(x, 200)
end

@testset "boxcox_profile_plot" begin
    Random.seed!(1)
    x = abs.(randn(200)) .+ 5.0
    profile = boxcox_profile_plot(x)
    @test length(profile.lambdas) == length(profile.loglik) == 50

    gl = guerrero_lambda(x, 12)
    peak_lam = profile.lambdas[argmax(profile.loglik)]
    @test isapprox(peak_lam, gl; atol=0.3)

    # custom lambda grid honored
    profile2 = boxcox_profile_plot(x, range(-0.5, 0.5, length=10))
    @test length(profile2.lambdas) == 10

    @test_throws ArgumentError boxcox_profile_plot([1.0, -1.0])
end

# ---------------------------------------------------------------------------
# Stage 9 and 10 recipes (handoff/stage 10/plot-recipes-handoff.md §6): every
# count is asserted, never just non-emptiness -- a recipe that silently drops
# a band still passes !isempty. The NUMBERS are in test_plotdata.jl.
# ---------------------------------------------------------------------------

_nrec(r; kw...) = length(RecipesBase.apply_recipe(Dict{Symbol,Any}(kw...), r))
_rfx(name) = Float64.(readdlm(joinpath(@__DIR__, "fixtures", "var", name), ','; skipstart=1))
_ap() = Float64.(readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)[:, end])

@testset "recipes: Forecast fan chart" begin
    y = _ap()
    fc = forecast(fit_ets(y, 12; trend=:add, seasonal=:add), 12)
    @test _nrec(fc) == 3                                # 2 levels + point
    @test _nrec(fc; history=y) == 4                     # + observed
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(:history => y), fc)
    @test rec[1].plotattributes[:label] == "95%"        # widest drawn first
    @test rec[end].args[1] == collect(145:156)          # forecast after the data
    @test rec[3].args[2] == y                           # history in full (short series)
    f3 = forecast(fit_ets(y, 12; trend=:add, seasonal=:add), 6; level=[50.0, 80.0, 95.0])
    @test _nrec(f3) == 4
    # every forecasting family shares the one recipe
    for f in (forecast(fit_arima(log.(y), (1, 1, 0)), 6), forecast(fit_theta(y, 12), 6),
              forecast(fit_tbats(y, [12.0]; k=2), 6), naive(y, 6))
        @test _nrec(f) == length(f.levels) + 1
    end
    # long histories are thinned for display, both extremes kept
    long = cumsum(randn(MersenneTwister(1), 20_000))
    rl = RecipesBase.apply_recipe(Dict{Symbol,Any}(:history => long), forecast(fit_arima(long, (1, 1, 0)), 5))
    @test length(rl[3].args[2]) <= 5_000
    @test maximum(rl[3].args[2]) == maximum(long) && minimum(rl[3].args[2]) == minimum(long)
end

@testset "recipes: VARForecast and VolatilityForecast" begin
    v = fit_var(_rfx("var2_p1.csv"), 2)
    @test _nrec(forecast(v, 8)) == 2 * (2 + 1)          # K=2, 2 levels
    @test _nrec(forecast(v, 8); history=v.Y) == 2 * (2 + 2)
    @test_throws ArgumentError _nrec(forecast(v, 8); history=v.Y[:, 1:1])
    g = fit_garch(randn(MersenneTwister(1), 500))
    @test _nrec(forecast_volatility(g, 10)) == 1        # analytic: no interval invented
    vs = forecast_volatility(g, 10; method=:simulation, simulations=200)
    @test _nrec(vs) == 2                                # simulated band + sigma
    @test _nrec(vs; history=sqrt.(g.sigma2)) == 3
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), forecast_volatility(g, 10))
    @test rec[1].args[2] ≈ sqrt.(forecast_volatility(g, 10).variance)   # sigma, not sigma^2
end

@testset "recipes: StateDecomposition, ARMA roots, ts_display" begin
    y = _ap()
    @test _nrec(state_decomposition(fit_ets(y, 12; trend=:add, seasonal=:add))) == 4
    @test _nrec(state_decomposition(fit_ets(y, 12))) == 2
    @test _nrec(state_decomposition(fit_ets(y, 12; trend=:add))) == 3
    @test _nrec(state_decomposition(fit_tbats(y, [4.0, 12.0]; k=[1, 1]))) == 5
    @test _nrec(state_decomposition(holt_winters(y, 12; trend=:additive, seasonal=:additive))) == 4
    @test _nrec(arma_roots(fit_arima(log.(y), (1, 1, 1)))) == 3      # circle + AR + MA
    @test _nrec(arma_roots(fit_arima(log.(y), (2, 1, 0)))) == 2      # no empty MA series
    @test _nrec(arma_roots(fit_arima(log.(y), (0, 1, 1)))) == 2
    @test _nrec(ts_display(y)) == 7                                  # 1 + 3 + 3
end

@testset "recipes: Stage 10" begin
    v = fit_var(_rfx("var2_p1.csv"), 2)
    @test _nrec(irf(v, 6)) == 8                                      # 2K^2, no band
    @test _nrec(irf(v, 6; stderr=true)) == 12                        # analytic band
    @test _nrec(irf(v, 6; boot=true, runs=20, rng=MersenneTwister(1))) == 12
    ra = RecipesBase.apply_recipe(Dict{Symbol,Any}(), irf(v, 6; stderr=true))
    @test occursin("analytic", ra[1].plotattributes[:label])         # labelled by method
    rb = RecipesBase.apply_recipe(Dict{Symbol,Any}(), irf(v, 6; boot=true, runs=20, rng=MersenneTwister(1)))
    @test occursin("bootstrap", rb[1].plotattributes[:label])
    # FEVD: stacked bars built by hand (core Plots' bar_position=:stack is a
    # no-op): K bars per panel, each sitting on the cumulative share below it
    f = fevd(v, 6)
    rf = RecipesBase.apply_recipe(Dict{Symbol,Any}(), f)
    @test length(rf) == 2 * 2
    @test rf[2].plotattributes[:fillrange] ≈ f.decomp[1, :, 1]
    @test rf[2].args[2] ≈ f.decomp[1, :, 1] .+ f.decomp[1, :, 2]
    @test all(rf[2].args[2] .≈ 1.0)                                  # stacks to one
    @test _nrec(roots_plot(v)) == 2
    @test _nrec(diagnostic_plot(v)) == 2 * 6                         # per equation, as DiagnosticPlotResult
    @test _nrec(cointegration_plot(fit_vecm(_rfx("coint3_r1.csv"); lags=1, rank=1))) == 5
    @test _nrec(cointegration_plot(_rfx("coint3_r2.csv"), johansen_test(_rfx("coint3_r2.csv"); lags=1))) == 10
    @test _nrec(irf(blanchard_quah(v), 6)) == 8
end
