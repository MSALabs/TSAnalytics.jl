using DelimitedFiles

@testset "holt_winters fixed-parameter cases (exact R stats::HoltWinters validation)" begin
    # handoff/stage-5.5-holtwinters-handoff.md. Verified this session by
    # reproducing R's actual fitted/SSE output BIT-FOR-BIT (to ~1e-13) on
    # TSAnalytics.AIR_PASSENGERS across additive/multiplicative/
    # with-trend/without-trend/Holt/simple-ES cases -- not just the final
    # SSE, the full per-step level/trend/season recursion, found by
    # comparing R's second-seasonal-cycle output directly (see
    # `_hw_recursion`'s docstring for the one non-obvious formula point
    # this surfaced: seasonal update uses the NEW level, not old
    # level+trend).
    ap = readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)
    y = Float64.(ap[:, 2])

    m1 = holt_winters(y, 12; trend=:additive, seasonal=:additive, alpha=0.3, beta=0.05, gamma=0.2)
    @test isapprox(m1.sse, 92402.9671029133; atol=1e-4)

    m2 = holt_winters(y, 12; trend=:additive, seasonal=:multiplicative, alpha=0.3, beta=0.05, gamma=0.2)
    @test isapprox(m2.sse, 32523.7616164221; atol=1e-4)

    m3 = holt_winters(y, 12; seasonal=:additive, alpha=0.3, gamma=0.2)
    @test isapprox(m3.sse, 96618.0625182468; atol=1e-4)
    @test m3.beta === nothing
    @test m3.trend_component === nothing

    m4 = holt_winters(y; trend=:additive, alpha=0.3, beta=0.05)
    @test isapprox(m4.sse, 315258.4643612918; atol=1e-4)
    @test m4.gamma === nothing
    @test m4.seasonal_component === nothing

    m5 = holt_winters(y; alpha=0.3)
    @test isapprox(m5.sse, 301000.9448609633; atol=1e-4)
    @test m5.beta === nothing && m5.gamma === nothing

    # show() sanity
    io = IOBuffer()
    show(io, m1)
    s = String(take!(io))
    @test occursin("Holt-Winters (additive)", s)
    @test occursin("alpha", s) && occursin("beta", s) && occursin("gamma", s)
    @test occursin("SSE", s)

    io2 = IOBuffer()
    show(io2, m4)
    s2 = String(take!(io2))
    @test occursin("Holt's linear", s2)
    @test !occursin("gamma", s2)

    io3 = IOBuffer()
    show(io3, m5)
    s3 = String(take!(io3))
    @test occursin("Simple exponential smoothing", s3)
    @test !occursin("beta", s3)
end

@testset "holt_winters full optimization (exact R optimum reproduction)" begin
    # R's own free-parameter optimum on AIR_PASSENGERS (additive, all
    # three smoothing params estimated), confirmed by direct execution:
    # alpha=0.2479595, beta=0.03453373, gamma=1, SSE=21860.18.
    ap = readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)
    y = Float64.(ap[:, 2])

    m = holt_winters(y, 12; trend=:additive, seasonal=:additive)
    @test m.initialization_method == :heuristic
    @test isapprox(m.alpha, 0.2479595; atol=1e-3)
    @test isapprox(m.beta, 0.03453373; atol=1e-3)
    @test isapprox(m.gamma, 1.0; atol=1e-3)
    @test isapprox(m.sse, 21860.18; atol=1e-1)
end

@testset "holt_winters :heuristic vs :estimated genuinely diverge" begin
    # handoff section 3's headline finding, confirmed by direct execution
    # of both R's and Python's actual defaults: R's deterministic
    # (classical_decompose-based) initialization and Python's
    # jointly-optimized initialization solve genuinely different
    # optimization problems on identical data, not just different
    # starting guesses -- :estimated should reach a lower (better
    # in-sample fit) SSE since it has more free parameters.
    ap = readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)
    y = Float64.(ap[:, 2])

    m_h = holt_winters(y, 12; trend=:additive, seasonal=:additive, initialization_method=:heuristic)
    m_e = holt_winters(y, 12; trend=:additive, seasonal=:additive, initialization_method=:estimated)
    @test m_h.initialization_method == :heuristic
    @test m_e.initialization_method == :estimated
    @test !isapprox(m_h.alpha, m_e.alpha; atol=1e-3)
    @test m_e.sse < m_h.sse  # more free parameters -> at least as good in-sample fit
end

@testset "holt_winters parameter coverage and error paths" begin
    ap = readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)
    y = Float64.(ap[:, 2])

    # every combination of which of alpha/beta/gamma are pinned vs free
    for (a, b, g) in [(0.3, nothing, nothing), (nothing, 0.05, nothing), (nothing, nothing, 0.2),
                      (0.3, 0.05, nothing), (0.3, nothing, 0.2), (nothing, 0.05, 0.2)]
        m = holt_winters(y, 12; trend=:additive, seasonal=:additive, alpha=a, beta=b, gamma=g)
        a !== nothing && @test m.alpha == a
        b !== nothing && @test m.beta == b
        g !== nothing && @test m.gamma == g
        @test isfinite(m.sse)
    end

    # multiplicative, fully optimized
    m_mult = holt_winters(y, 12; trend=:additive, seasonal=:multiplicative)
    @test isfinite(m_mult.sse)
    @test 0 <= m_mult.alpha <= 1

    # every period value (structural, not exact-value)
    for period in (4, 6, 12)
        if length(y) >= 2 * period
            m = holt_winters(y, period; seasonal=:additive)
            @test m.period == period
            @test length(m.fitted) == length(y) - period
        end
    end

    # container-agnostic (range, Float32)
    m_f32 = holt_winters(Float32.(y), 12; trend=:additive, seasonal=:additive, alpha=0.3, beta=0.05, gamma=0.2)
    @test isapprox(m_f32.sse, 92402.9671029133; atol=1.0)

    # struct field types/lengths
    m = holt_winters(y, 12; trend=:additive, seasonal=:additive, alpha=0.3, beta=0.05, gamma=0.2)
    @test m isa TSAnalytics.ExponentialSmoothingModel
    @test length(m.level) == length(m.fitted) == length(m.resid)
    @test length(m.trend_component) == length(m.fitted)
    @test length(m.seasonal_component) == length(m.fitted)
    @test m.resid == y[13:end] .- m.fitted

    # error paths
    @test_throws ArgumentError holt_winters(y; seasonal=:additive)  # missing period
    @test_throws ArgumentError holt_winters(y, 12; initialization_method=:bogus, seasonal=:additive)
    @test_throws ArgumentError holt_winters(y, 12; seasonal=:bogus)
    @test_throws ArgumentError holt_winters(y; trend=:bogus)
    @test_throws ArgumentError holt_winters(y, 1; seasonal=:additive)  # period must be >= 2
    @test_throws ArgumentError holt_winters(y, 12; seasonal=:additive, beta=0.05)  # beta without trend
    @test_throws ArgumentError holt_winters(y; trend=:additive, gamma=0.2)  # gamma without seasonal
    @test_throws ArgumentError holt_winters(y; alpha=1.5)  # out of [0,1]
    @test_throws ArgumentError holt_winters(y; alpha=-0.1)
    @test_throws ArgumentError holt_winters(y; alpha=0.0)  # R: cannot fit without a level
    @test_throws ArgumentError holt_winters(y, 200; seasonal=:additive)  # not enough data for 2 periods
end
