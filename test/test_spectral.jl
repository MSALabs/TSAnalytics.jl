using DelimitedFiles, RecipesBase

@testset "periodogram / spectral_density (R spec.pgram-verified)" begin
    vdir = joinpath(@__DIR__, "verification", "spectral")
    x = vec(readdlm(joinpath(vdir, "x_case1.csv"), ','; skipstart=1))
    x2 = vec(readdlm(joinpath(vdir, "x_case2.csv"), ','; skipstart=1))

    # spec.pgram(x, taper=0.1, plot=FALSE)
    r1 = periodogram(x; taper=0.1)
    @test isapprox(collect(r1.freq[1:5]), [0.01041667, 0.02083333, 0.03125, 0.04166667, 0.05208333]; atol=1e-6)
    @test isapprox(r1.spec[1:5], [5.787371, 1.375819, 2.646642, 2.651191, 9.823574]; atol=1e-5)
    @test isapprox(r1.freq[argmax(r1.spec)], 1 / 12; atol=1e-6)
    @test isapprox(r1.df, 1.79159; atol=1e-4)
    @test isapprox(r1.bandwidth, 0.003007033; atol=1e-8)

    # spec.pgram(x, taper=0, plot=FALSE) -- this project's own default
    r1b = periodogram(x; taper=0.0)
    @test isapprox(r1b.spec[1:5], [3.935855, 0.2155094, 0.296584, 0.04881481, 1.001055]; atol=1e-5)

    # spec.pgram(x, spans=c(3,3), taper=0.1, plot=FALSE)
    r2 = spectral_density(x, [3, 3]; taper=0.1)
    @test isapprox(r2.spec[1:5], [4.488187, 3.151845, 2.974927, 4.667025, 7.085133]; atol=1e-5)
    @test isapprox(r2.df, 6.552102; atol=1e-5)
    @test isapprox(r2.bandwidth, 0.01084201; atol=1e-7)

    # spec.pgram(x, spans=c(7,7), taper=0, plot=FALSE)
    r3 = spectral_density(x, [7, 7]; taper=0.0)
    @test isapprox(r3.spec[1:5], [1.759758, 6.963694, 22.99766, 44.46989, 65.96196]; atol=1e-4)
    @test isapprox(r3.df, 18.46483; atol=1e-4)
    @test isapprox(r3.bandwidth, 0.0263866; atol=1e-6)

    # spec.pgram(x2, taper=0, plot=FALSE) -- n=97, exercises nextn/fast padding
    r4 = periodogram(x2; taper=0.0)
    @test length(r4.spec) == 50
    @test isapprox(r4.spec[1:5], [0.09683977, 1.70822, 0.7014162, 1.422045, 0.6433586]; atol=1e-5)
    @test isapprox(collect(r4.freq[1:5]), [0.01, 0.02, 0.03, 0.04, 0.05]; atol=1e-10)

    # spec.pgram(x2, spans=c(3,3), taper=0, plot=FALSE)
    r5 = spectral_density(x2, [3, 3]; taper=0.0)
    @test isapprox(r5.spec[1:5], [0.5374709, 0.9350769, 1.09186, 1.128727, 1.708311]; atol=1e-5)
    @test isapprox(r5.df, 7.094857; atol=1e-5)
    @test isapprox(r5.bandwidth, 0.01040833; atol=1e-7)

    # _nextn: smallest 5-smooth number >= n (verified empirically against R's nextn())
    @test TSAnalytics._nextn(1) == 1
    @test TSAnalytics._nextn(7) == 8
    @test TSAnalytics._nextn(10) == 10
    @test TSAnalytics._nextn(11) == 12
    @test TSAnalytics._nextn(17) == 18
    @test TSAnalytics._nextn(50) == 50
    @test TSAnalytics._nextn(96) == 96
    @test TSAnalytics._nextn(97) == 100
    @test TSAnalytics._nextn(101) == 108
    @test TSAnalytics._nextn(127) == 128
    @test TSAnalytics._nextn(151) == 160
    @test TSAnalytics._nextn(997) == 1000
    @test TSAnalytics._nextn(1000) == 1000

    # argument validation
    @test_throws ArgumentError periodogram(Float64[])
    @test_throws ArgumentError spectral_density(x, [0, 3])

    # PeriodogramResult is a real struct now, not a bare NamedTuple --
    # a bare NamedTuple recipe would apply too broadly (see recipes.jl)
    @test r1 isa TSAnalytics.PeriodogramResult
    @test r1.kind == :periodogram
    @test r2 isa TSAnalytics.PeriodogramResult
    @test r2.kind == :spectral_density
end

@testset "periodogram/spectral_density recipe -- see handoff/periodogram-recipe-handoff.md" begin
    # one series, path/log10 scale -- unlike ACF's three (fill band + hline + sticks)
    x = randn(96)
    r = periodogram(x)
    rec = RecipesBase.apply_recipe(Dict{Symbol,Any}(), r)
    @test length(rec) == 1
    @test rec[1].plotattributes[:seriestype] in (:path, :line)
    @test get(rec[1].plotattributes, :yscale, :identity) == :log10

    # real, R-verified peak case renders the right x-value at the right place --
    # reuses Stage 1.5's own exact verified case, not a separately-computed value
    Random.seed!(42)
    n = 96; t = 1:n
    xp = 5 .* sin.(2π .* t ./ 12) .+ 3 .* cos.(2π .* t ./ 12) .+ randn(n) .* 0.5
    rp = periodogram(xp; taper=0.1)
    recp = RecipesBase.apply_recipe(Dict{Symbol,Any}(), rp)
    xdata, ydata = recp[1].args
    @test isapprox(xdata[argmax(ydata)], 1 / 12; atol=1e-6)

    # edge cases
    @test_throws ArgumentError periodogram(Float64[])
    @test_throws ArgumentError periodogram([1.0])
    rmin = periodogram(randn(4))
    recmin = RecipesBase.apply_recipe(Dict{Symbol,Any}(), rmin)
    @test length(recmin[1].args[1]) > 0

    # spectral_density: df/bandwidth surfaced in the plot title, not silently dropped
    rsd = spectral_density(randn(200), [7, 7])
    recsd = RecipesBase.apply_recipe(Dict{Symbol,Any}(), rsd)
    @test occursin("df", recsd[1].plotattributes[:title])
    @test occursin("bandwidth", recsd[1].plotattributes[:title])

    # large series -- decimation must never silently corrupt the visible peak
    n2 = 50_000
    t2 = 1:n2
    xl = 5 .* sin.(2π .* t2 ./ 12) .+ randn(n2) .* 0.5
    rl = periodogram(xl)
    @test length(rl.freq) > 10_000  # confirms decimation actually engages below
    recl = RecipesBase.apply_recipe(Dict{Symbol,Any}(), rl)
    xdatal, ydatal = recl[1].args
    @test length(xdatal) == 10_000
    @test isapprox(xdatal[argmax(ydatal)], 1 / 12; atol=1e-4)

    # decimation is a no-op below the threshold
    rs = periodogram(randn(200))
    recs = RecipesBase.apply_recipe(Dict{Symbol,Any}(), rs)
    @test length(recs[1].args[1]) == length(rs.freq)

    # _decimate_local_max always preserves the exact global peak
    xv = collect(1.0:10_000.0)
    yv = randn(MersenneTwister(1), 10_000)
    yv[7331] = 100.0  # a single, narrow spike
    xd, yd = TSAnalytics._decimate_local_max(xv, yv, 500)
    @test maximum(yd) == 100.0
    @test xd[argmax(yd)] == 7331.0
end
