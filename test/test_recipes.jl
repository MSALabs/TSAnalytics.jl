using RecipesBase

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
