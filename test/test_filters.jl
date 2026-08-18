@testset "filters" begin
    x = 1.0:10.0
    filt3 = [1/3, 1/3, 1/3]

    # Centered (sides=2) 3-term moving average
    r2 = convolution_filter(x, filt3; sides=2)
    @test isnan(r2[1]) && isnan(r2[end])
    @test isapprox(r2[2:end-1], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; atol=1e-10)

    # Causal (sides=1) 3-term moving average
    r1 = convolution_filter(x, filt3; sides=1)
    @test all(isnan, r1[1:2])
    @test isapprox(r1[3:end], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; atol=1e-10)

    # Circular wraps instead of producing NaN at the edges
    rc = convolution_filter(x, filt3; sides=2, circular=true)
    @test !any(isnan, rc)
    @test isapprox(rc[2:end-1], r2[2:end-1]; atol=1e-10)

    @test_throws ArgumentError convolution_filter(x, filt3; sides=3)

    # Container-agnostic
    @test isequal(convolution_filter(1:10, filt3), convolution_filter(collect(1.0:10.0), filt3))

    # Recursive AR(1), coefficient 0.5, constant input, zero init
    xr = ones(5)
    yr = recursive_filter(xr, [0.5])
    @test isapprox(yr, [1.0, 1.5, 1.75, 1.875, 1.9375]; atol=1e-10)

    @test_throws ArgumentError recursive_filter(xr, [0.5]; init=[0.0, 0.0])

    # Non-zero init: y[0] = 2.0 (init[1]) feeds directly into y[1]
    yr2 = recursive_filter(xr, [0.5]; init=[2.0])
    @test isapprox(yr2[1], 1.0 + 0.5*2.0; atol=1e-10)

    # AR(2), reverse-time-order coefficients: ar_coeff[1] multiplies y[i-1]
    yr3 = recursive_filter([1.0, 0.0, 0.0, 0.0], [0.5, 0.25])
    @test isapprox(yr3[1], 1.0; atol=1e-10)                      # y1 = x1
    @test isapprox(yr3[2], 0.5*yr3[1]; atol=1e-10)                # y2 = 0.5*y1
    @test isapprox(yr3[3], 0.5*yr3[2] + 0.25*yr3[1]; atol=1e-10)  # y3 = 0.5*y2 + 0.25*y1

    # Container-agnostic (both x and init)
    @test recursive_filter(1:5, [0.5]; init=[0]) == recursive_filter(collect(1.0:5.0), [0.5]; init=[0.0])
end

@testset "moving_average" begin
    x12 = 1.0:12.0

    ma3 = moving_average(x12, 3)   # odd order
    @test isnan(ma3[1]) && isnan(ma3[end])
    @test isapprox(ma3[2:end-1], collect(2.0:11.0); atol=1e-10)

    ma4c = moving_average(x12, 4)  # even order, centre=true (default)
    @test all(isnan, ma4c[1:2]) && all(isnan, ma4c[end-1:end])
    @test isapprox(ma4c[3:end-2], collect(3.0:10.0); atol=1e-10)

    ma4u = moving_average(x12, 4; centre=false)  # even order, uncentred
    @test isapprox(ma4u[2:end-2], [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]; atol=1e-10)

    @test_throws ArgumentError moving_average(x12, 0)

    # AirPassengers: order-12 trend extraction (classical/X-11-style decomposition)
    air = _load_column(TSAnalytics.AIR_PASSENGERS, "passengers")
    trend = moving_average(air, 12)
    @test count(isnan, trend) == 12   # 6 NaN at each end for an even order-12 filter
    @test all(x -> x > 0, filter(!isnan, trend))

    # Container-agnostic
    @test isequal(moving_average(1:12, 3), moving_average(collect(1.0:12.0), 3))

    # moving_average(x, order; centre=true) for even order equals a direct
    # convolution_filter call with the explicit half-weighted-endpoint filter
    # -- cross-checks the wrapper against the primitive it's built on.
    m = 4
    explicit_filt = vcat(1/(2m), fill(1/m, m-1), 1/(2m))
    @test isapprox(moving_average(x12, m), convolution_filter(x12, explicit_filt); nans=true)
end
