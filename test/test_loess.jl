using DelimitedFiles

@testset "_lowess exact statsmodels validation" begin
    # statsmodels.nonparametric.smoothers_lowess.lowess(y, x, frac=frac, it=it,
    # return_sorted=False), x = 1:200 (matching Julia's 1-indexed convention)
    fixture = joinpath(@__DIR__, "fixtures", "ar1_ref_series.csv")
    y = vec(readdlm(fixture, ','; skipstart=1, header=false))
    n = length(y)
    @test n == 200
    x = collect(1.0:n)

    reference = [
        (2/3, 3, [0.6431840001, 0.6253663304, 0.6077456656, 0.590328142, 0.5731196005],
                 [0.3259534212, 0.3406600281, 0.3549518301, 0.36879408, 0.3821670566]),
        (2/3, 0, [0.6541863946, 0.6349515919, 0.6159496262, 0.5971873315, 0.5786712098],
                 [0.3689910893, 0.3855174668, 0.4015608554, 0.4170819483, 0.4320574199]),
        (0.3, 3, [1.5385324829, 1.4930244021, 1.4473471713, 1.4015323547, 1.355614661],
                 [0.825426979, 0.8693585505, 0.9107796373, 0.9492257259, 0.9843376899]),
        (0.1, 1, [1.0860703674, 1.1051463599, 1.1215224772, 1.1377458429, 1.1565893411],
                 [1.2286553465, 1.2076953054, 1.1586993732, 1.0899128433, 1.0107700082]),
    ]
    for (frac, it, head_ref, tail_ref) in reference
        fitted = TSAnalytics._lowess(x, y; frac=frac, it=it)
        @test length(fitted) == n
        @test isapprox(fitted[1:5], head_ref; atol=1e-6)
        @test isapprox(fitted[96:100], tail_ref; atol=1e-6)
    end
end

@testset "_lowess mathematical invariants" begin
    xc = collect(1.0:50)

    # Constant input: lowess must reproduce the constant exactly, for any
    # frac/it (a weighted average of identical values is that value).
    const_y = fill(7.0, 50)
    for frac in (0.2, 0.5, 1.0), it in (0, 1, 3)
        fitted = TSAnalytics._lowess(xc, const_y; frac=frac, it=it)
        @test all(isapprox.(fitted, 7.0; atol=1e-10))
    end

    # Perfectly linear input: a local LINEAR fit of any nonzero weights on
    # exactly-linear data recovers the line exactly, for any frac/it.
    lin_y = 2.0 .* xc .+ 3.0
    for frac in (0.2, 0.5, 1.0), it in (0, 1, 3)
        fitted = TSAnalytics._lowess(xc, lin_y; frac=frac, it=it)
        @test isapprox(fitted, lin_y; atol=1e-8)
    end
end

@testset "_lowess robustness (it > 0 downweights outliers)" begin
    Random.seed!(11)
    n = 60
    x = collect(1.0:n)
    y = 2.0 .* x .+ 5.0 .+ 0.5 .* randn(n)
    y[30] += 40.0  # a single large outlier

    fitted_robust = TSAnalytics._lowess(x, y; frac=0.4, it=3)
    fitted_plain = TSAnalytics._lowess(x, y; frac=0.4, it=0)
    # the fit at (and near) the outlier's own position should be pulled
    # toward the underlying trend less by the robust pass than the
    # non-robust one, since the outlier gets downweighted after iterating
    @test abs(fitted_robust[30] - (2.0*30+5.0)) < abs(fitted_plain[30] - (2.0*30+5.0))
end

@testset "_lowess parameter coverage and error paths" begin
    Random.seed!(12)
    n = 40
    x = collect(1.0:n)
    y = sin.(2π .* x ./ 12) .+ 0.2 .* randn(n)

    # every frac value runs and gives a finite result
    for frac in (0.1, 0.3, 2/3, 1.0)
        fitted = TSAnalytics._lowess(x, y; frac=frac)
        @test all(isfinite, fitted)
        @test length(fitted) == n
    end

    # every it value runs and gives a finite result
    for it in (0, 1, 2, 3, 5)
        fitted = TSAnalytics._lowess(x, y; it=it)
        @test all(isfinite, fitted)
    end

    @test_throws ArgumentError TSAnalytics._lowess(x, y[1:end-1])  # length mismatch
    @test_throws ArgumentError TSAnalytics._lowess(x, y; frac=0.0)
    @test_throws ArgumentError TSAnalytics._lowess(x, y; frac=1.5)
    @test_throws ArgumentError TSAnalytics._lowess(x, y; it=-1)

    # frac=1.0 (all points as neighbors) still runs without error
    fitted_full = TSAnalytics._lowess(x, y; frac=1.0, it=0)
    @test all(isfinite, fitted_full)
end

@testset "_lowess degree/k/xout (exact statsmodels validation where possible)" begin
    # Fixed, hardcoded data (not Julia's own RNG): numpy's np.random.seed(5)
    # stream doesn't match Julia's Random.seed!(5) stream, so generating this
    # inline in Julia would silently compare against the wrong numbers --
    # the same cross-language RNG pitfall documented in this session's
    # memory. y = np.sin(2*np.pi*x/12) + 0.1*np.random.randn(30), seed=5.
    n = 30
    x = collect(1.0:n)
    y = [0.5441227487, 0.8329383886, 1.2430771187, 0.8408161908, 0.5109609842,
         0.1582481117, -0.5909232405, -0.9251890696, -0.9812396774, -0.8990123996,
         -0.6192764612, -0.0204876511, 0.4641171053, 0.9263725640, 0.8335211471,
         0.7960075000, 0.6151391009, 0.1857331007, -0.6511179558, -0.8015406527,
         -1.0980607885, -0.9517107193, -0.5871879183, -0.0422507929, 0.5996439827,
         0.9372675309, 1.0059144243, 0.8296943159, 0.5003288843, -0.0105930442]

    # off-grid xout, including points OUTSIDE the observed range -- exact
    # match against statsmodels.nonparametric.lowess's `xvals` argument,
    # the same off-grid capability `stl_decompose` needs for cycle-subseries
    # extension.
    xout = [0.0, 31.0, 15.5]
    fitted_offgrid = TSAnalytics._lowess(x, y; frac=0.5, it=0, xout=xout)
    @test isapprox(fitted_offgrid, [1.26635747, 0.95753164, 0.23223551]; atol=1e-6)

    # explicit k is equivalent to the frac that produces the same k
    # (k = trunc(Int, frac*n) = trunc(Int, 0.5*30) = 15)
    fitted_k = TSAnalytics._lowess(x, y; k=15, it=0, xout=xout)
    @test isapprox(fitted_k, fitted_offgrid; atol=1e-10)

    # degree=0: hand-verified weighted-mean case (Python's lowess() has no
    # degree=0 mode to validate against directly, so this is independently
    # computed from the same tricube-weight formula, not from _lowess's
    # own code).
    x7 = collect(1.0:7)
    y7 = [1.0, 2.0, 3.0, 10.0, 5.0, 6.0, 7.0]
    fitted0 = TSAnalytics._lowess(x7, y7; k=5, degree=0, it=0, xout=[4.0])
    @test isapprox(fitted0[1], 6.564273789649416; atol=1e-10)

    # degree=0 on constant data still reproduces the constant exactly
    const_y = fill(3.0, 20)
    xc = collect(1.0:20)
    @test all(isapprox.(TSAnalytics._lowess(xc, const_y; k=7, degree=0, it=0), 3.0; atol=1e-10))

    # degree=2 on a perfectly quadratic series recovers it exactly (any
    # nonzero weights, any span -- same invariant as the linear case for
    # degree=1)
    xq = collect(1.0:30)
    quad_y = 0.5 .* xq .^ 2 .- 2.0 .* xq .+ 1.0
    for kk in (5, 15, 30)
        fitted_q = TSAnalytics._lowess(xq, quad_y; k=kk, degree=2, it=0)
        @test isapprox(fitted_q, quad_y; atol=1e-6)
    end

    # invalid degree / k
    @test_throws ArgumentError TSAnalytics._lowess(x, y; degree=3)
    @test_throws ArgumentError TSAnalytics._lowess(x, y; k=0)
end
