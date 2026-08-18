@testset "differencing" begin
    # ---------------------------------------------------------------------
    # Base.diff extension coexists with, and doesn't change, Base's own
    # single-argument method -- this is the whole point of requiring `lag`
    # as a positional argument rather than shadowing Base's method.
    # ---------------------------------------------------------------------
    y = [1.0, 3.0, 6.0, 10.0, 15.0, 21.0]  # partial sums of 1:6 -- diff = 1:5
    @test diff(y) == [2.0, 3.0, 4.0, 5.0, 6.0]          # Base.diff, unaffected
    @test diff(y, 1) == diff(y)                          # our method agrees with Base's for lag=1
    @test diff(y, 1; differences=1) == diff(y)

    M = reshape(1.0:12.0, 3, 4)
    @test diff(M; dims=2) == M[:, 2:end] .- M[:, 1:end-1]  # Base's `dims` keyword still works

    # Return type is always a concrete Vector, matching Base.diff -- even on
    # a range whose differences happen to be constant, which would otherwise
    # collapse back into a StepRangeLen under naive slicing+broadcasting.
    @test diff(1.0:12.0, 4) isa Vector{Float64}
    @test diff([1, 3, 6, 10], 1) isa Vector{Int}   # eltype preserved, like Base.diff

    # ---------------------------------------------------------------------
    # Regular repeated differencing (differences > 1) matches nested Base.diff
    # ---------------------------------------------------------------------
    @test diff(y, 1; differences=2) == diff(diff(y))
    @test diff(y, 1; differences=3) == diff(diff(diff(y)))

    # ---------------------------------------------------------------------
    # Seasonal differencing, checked against the direct longhand formula
    # ---------------------------------------------------------------------
    s = collect(1.0:24.0)
    @test diff(s, 4) == s[5:end] .- s[1:end-4]
    @test diff(s, 4; differences=2) == diff(diff(s, 4), 4)

    # ---------------------------------------------------------------------
    # External reference: internet.csv ships a pre-computed first-difference
    # column ("dinternet", with a leading NaN placeholder) -- an actual
    # reference number, not a self-check.
    # ---------------------------------------------------------------------
    net = _load_column(TSAnalytics.INTERNET, "internet")
    dnet = _load_column(TSAnalytics.INTERNET, "dinternet")
    @test isnan(dnet[1])
    @test isapprox(diff(net, 1), dnet[2:end]; atol=1e-6)
    @test isapprox(diff(net), dnet[2:end]; atol=1e-6)   # Base.diff agrees too

    # ---------------------------------------------------------------------
    # AirPassengers: seasonal (period 12) differencing on a real series
    # ---------------------------------------------------------------------
    air = _load_column(TSAnalytics.AIR_PASSENGERS, "passengers")
    seasonal_air = diff(air, 12)
    @test length(seasonal_air) == length(air) - 12
    @test isapprox(seasonal_air, air[13:end] .- air[1:end-12]; atol=1e-8)
    # airline model differencing: one seasonal + one regular difference
    airline = diff(diff(air, 12), 1)
    @test length(airline) == length(air) - 12 - 1

    # ---------------------------------------------------------------------
    # Container-agnostic: works on non-Vector iterables via tsvalues
    # ---------------------------------------------------------------------
    @test diff(1:10, 1) == diff(collect(1:10), 1)
    @test diff(Iterators.map(x -> Float64(x^1), 1:10), 2) == diff(collect(1.0:10.0), 2)

    # ---------------------------------------------------------------------
    # Error paths
    # ---------------------------------------------------------------------
    @test_throws ArgumentError diff(y, 0)
    @test_throws ArgumentError diff(y, -1)
    @test_throws ArgumentError diff(y, 1; differences=-1)
    @test_throws ArgumentError diff([1.0, 2.0, 3.0], 2; differences=2)  # lag*differences >= n

    # =======================================================================
    # diffinv -- exact inverse of diff
    # =======================================================================

    # Hand-worked example (also documents the algorithm): lag=1, differences=2
    x = [1.0, 2.0, 4.0, 7.0, 11.0]
    z = diff(x, 1; differences=2)
    @test z == [1.0, 1.0, 1.0]
    @test diffinv(z; differences=2, xi=x[1:2]) == x

    # Roundtrip: diffinv(diff(x)) recovers x exactly, across several (lag, differences)
    for (lag, differences) in ((1, 1), (1, 2), (1, 3), (4, 1), (4, 2), (12, 1))
        n = 40
        v = cumsum(randn(MersenneTwister(lag * 100 + differences), n))
        dv = diff(v, lag; differences=differences)
        seed = v[1:lag*differences]
        @test isapprox(diffinv(dv; lag=lag, differences=differences, xi=seed), v; atol=1e-8)
    end

    # Roundtrip on real data (AirPassengers, seasonal + regular)
    d_air = diff(diff(air, 12), 1)
    recovered = diffinv(diffinv(d_air; lag=1, differences=1, xi=[diff(air, 12)[1]]);
                          lag=12, differences=1, xi=air[1:12])
    @test isapprox(recovered, air; atol=1e-6)

    # xi defaults to zeros: diffinv(x) == vcat(0.0, cumsum(x)) for lag=1, differences=1
    inc = [2.0, 3.0, 4.0]
    @test diffinv(inc) == vcat(0.0, cumsum(inc))
    @test diffinv(inc; lag=1, differences=1) == vcat(0.0, cumsum(inc))

    # Output length convention: length(x) + lag*differences (xi prepended)
    @test length(diffinv(z; differences=2, xi=x[1:2])) == length(z) + 2

    # differences = 0 is an identity pass-through
    @test diffinv([5.0, 6.0]; differences=0) == [5.0, 6.0]

    # Container-agnostic
    @test diffinv(1:3; xi=[0.0]) == diffinv(collect(1.0:3.0); xi=[0.0])

    # Error paths
    @test_throws ArgumentError diffinv(z; differences=2, xi=[1.0])  # wrong xi length
    @test_throws ArgumentError diffinv(z; lag=0)
    @test_throws ArgumentError diffinv(z; differences=-1)

    # =======================================================================
    # tsdiff / tsundiff -- true aliases, identical output to diff / diffinv
    # =======================================================================
    @test tsdiff(y) == diff(y, 1)
    @test tsdiff(y; lag=1, differences=1) == diff(y, 1; differences=1)
    @test tsdiff(s; lag=4) == diff(s, 4)
    @test tsdiff(s; lag=4, differences=2) == diff(s, 4; differences=2)
    @test tsdiff(air; lag=12) == diff(air, 12)
    @test tsdiff(1:10) == diff(collect(1:10), 1)   # container-agnostic

    @test tsundiff(z; differences=2, xi=x[1:2]) == diffinv(z; differences=2, xi=x[1:2])
    @test tsundiff(inc) == diffinv(inc)
    @test tsundiff(z; lag=1, differences=2, xi=x[1:2]) == diffinv(z; lag=1, differences=2, xi=x[1:2])
    @test tsundiff(1:3; xi=[0.0]) == diffinv(1:3; xi=[0.0])  # container-agnostic

    # tsdiff/tsundiff roundtrip, mirroring the diff/diffinv roundtrip above
    @test isapprox(tsundiff(tsdiff(air; lag=12); lag=12, xi=air[1:12]), air; atol=1e-8)
end
