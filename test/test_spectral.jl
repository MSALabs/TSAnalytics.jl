using DelimitedFiles

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
end
