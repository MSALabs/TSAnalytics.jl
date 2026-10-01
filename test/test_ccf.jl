using DelimitedFiles

# Stage 9B Tier 1.3. Ground truth re-executed this session against real
# R 4.6.0 `ccf(x, y, lag.max=6, plot=FALSE)` -- see verification/ccf/ccf.R --
# independently reproducing every digit the handoff quoted. Agreement is to
# ~5e-11, which is R's own printed precision rather than any limit here.
#
# The fixture is 145 observations with `y` lagging `x` by 3, so the peak sits
# at lag -3 and the sign convention is actually tested rather than assumed.

@testset "ccf — cross-correlation" begin
    vdir = joinpath(@__DIR__, "verification", "ccf")
    x = vec(readdlm(joinpath(vdir, "ccf_x.csv")))
    y = vec(readdlm(joinpath(vdir, "ccf_y.csv")))

    # R's ccf(x, y, lag.max=6), lag => value
    R_REF = Dict(
        -6 => 0.2661575938, -5 => 0.4159572432, -4 => 0.6633372049,
        -3 => 0.9636929430, -2 => 0.6817189202, -1 => 0.4641562350,
         0 => 0.3365029144,  1 => 0.2255713144,  2 => 0.1562916927,
         3 => 0.0941916568,  4 => 0.0907835083,  5 => 0.1170574662,
         6 => 0.1431579617)

    @testset "matches R's two-sided convention at every lag" begin
        r = ccf(x, y, 6)
        @test r.lags == collect(-6:6)
        @test r.kind == :ccf
        @test r.n == length(x)
        for (i, k) in enumerate(r.lags)
            @test isapprox(r.values[i], R_REF[k]; atol=1e-9)
        end
    end

    @testset "the peak identifies which series leads" begin
        r = ccf(x, y, 6)
        # lag k estimates cor(x[t+k], y[t]), so a peak at -3 says x leads y by 3
        @test r.lags[argmax(r.values)] == -3
        @test isapprox(maximum(r.values), 0.9636929430; atol=1e-9)
    end

    @testset "ccf(x,y) at lag -k equals ccf(y,x) at lag +k" begin
        a = ccf(x, y, 6)
        b = ccf(y, x, 6)
        @test isapprox(a.values, reverse(b.values); atol=1e-12)
    end

    @testset "ccf of a series with itself reproduces acf exactly" begin
        # the shared 1/n denominator means this is an identity, not an
        # approximation -- it pins ccf's normalisation to acf's
        c = ccf(x, x, 6)
        @test isapprox(c.values[7:end], acf(x, 0:6).values; atol=1e-12)
        @test isapprox(c.values[7], 1.0; atol=1e-12)
        @test isapprox(c.values, reverse(c.values); atol=1e-12)   # symmetric
    end

    @testset "values stay inside [-1, 1]" begin
        for r in (ccf(x, y, 20), ccf(randn(300), randn(300), 25))
            @test all(-1.0 .<= r.values .<= 1.0)
        end
    end

    @testset "demean=false is a different, documented computation" begin
        a = ccf(x, y, 6)
        b = ccf(x, y, 6; demean=false)
        @test !isapprox(a.values, b.values; atol=1e-6)
    end

    @testset "band is the constant +/- z/sqrt(n)" begin
        r = ccf(x, y, 6)
        @test all(r.upper .≈ r.upper[1])
        @test all(r.lower .≈ -r.upper[1])
        # against the package's own quantile, not a hardcoded 1.95996... --
        # `_confidence_z` is an approximation accurate to ~2e-9 and has its
        # own tests; this one is about the band formula, not the quantile
        @test isapprox(r.upper[1], TSAnalytics._confidence_z(0.05) / sqrt(length(x)); atol=1e-14)
        @test isapprox(r.upper[1], 1.959963984540054 / sqrt(length(x)); atol=1e-8)
        @test ccf(x, y, 6; alpha=0.01).upper[1] > r.upper[1]   # wider at 99%
    end

    @testset "edge cases refuse rather than guess" begin
        @test_throws DimensionMismatch ccf(randn(50), randn(60), 5)
        @test_throws ArgumentError ccf(x, y, 0)
        @test_throws ArgumentError ccf(x, y, -1)
        @test_throws ArgumentError ccf(x, y, length(x))
        @test_throws ArgumentError ccf(fill(1.0, 50), randn(50), 5)   # zero variance
        @test_throws ArgumentError ccf([1.0, NaN, 3.0], randn(3), 1)
        @test_throws ArgumentError ccf(randn(3), [1.0, NaN, 3.0], 1)
    end

    @testset "container-agnostic, like every other entry point" begin
        r1 = ccf(x, y, 4)
        r2 = ccf(collect(x), Tuple(y), 4)
        @test isapprox(r1.values, r2.values; atol=1e-12)
    end
end
