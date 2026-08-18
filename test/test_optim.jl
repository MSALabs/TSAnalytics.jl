@testset "_optimize (Rosenbrock + quadratic, exact minima)" begin
    # handoff/stage-4.1-optimizer-handoff.md section 4: Optim.jl's own
    # README example, an independently-published check.
    rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

    r = TSAnalytics._optimize(rosenbrock, [0.0, 0.0])
    @test r.converged
    @test isapprox(r.minimizer, [1.0, 1.0]; atol=1e-6)
    @test isapprox(r.minimum, 0.0; atol=1e-10)
    @test r.method == :lbfgs
    @test r.iterations >= 1

    # every method value
    r_lbfgs = TSAnalytics._optimize(rosenbrock, [0.0, 0.0]; method=:lbfgs)
    @test isapprox(r_lbfgs.minimizer, [1.0, 1.0]; atol=1e-6)

    r_bfgs = TSAnalytics._optimize(rosenbrock, [0.0, 0.0]; method=:bfgs)
    @test r_bfgs.converged
    @test isapprox(r_bfgs.minimizer, [1.0, 1.0]; atol=1e-6)
    @test r_bfgs.method == :bfgs

    r_nm = TSAnalytics._optimize(rosenbrock, [0.0, 0.0]; method=:nelder_mead, iterations=5000)
    @test isapprox(r_nm.minimizer, [1.0, 1.0]; atol=1e-3)  # gradient-free: less precise
    @test r_nm.method == :nelder_mead

    # invalid method
    @test_throws ArgumentError TSAnalytics._optimize(rosenbrock, [0.0, 0.0]; method=:bogus)

    # second independent check: a quadratic with a known closed-form minimum
    quad(x) = (x[1] - 3.0)^2 + (x[2] + 2.0)^2 + 5.0
    r_quad = TSAnalytics._optimize(quad, [0.0, 0.0])
    @test isapprox(r_quad.minimizer, [3.0, -2.0]; atol=1e-6)
    @test isapprox(r_quad.minimum, 5.0; atol=1e-8)

    r_quad_bfgs = TSAnalytics._optimize(quad, [0.0, 0.0]; method=:bfgs)
    @test isapprox(r_quad_bfgs.minimizer, [3.0, -2.0]; atol=1e-6)

    r_quad_nm = TSAnalytics._optimize(quad, [0.0, 0.0]; method=:nelder_mead)
    @test isapprox(r_quad_nm.minimizer, [3.0, -2.0]; atol=1e-3)
end

@testset "_optimize parameter coverage" begin
    rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
    quad(x) = (x[1] - 3.0)^2 + (x[2] + 2.0)^2 + 5.0

    # autodiff=false (finite-difference fallback) must still converge, for
    # every gradient-based method
    for method in (:lbfgs, :bfgs)
        r = TSAnalytics._optimize(quad, [0.0, 0.0]; method=method, autodiff=false)
        @test r.converged
        @test isapprox(r.minimizer, [3.0, -2.0]; atol=1e-5)
    end

    # autodiff is silently irrelevant for :nelder_mead (gradient-free) --
    # both settings must produce the same result
    r_ad_true = TSAnalytics._optimize(quad, [0.0, 0.0]; method=:nelder_mead, autodiff=true)
    r_ad_false = TSAnalytics._optimize(quad, [0.0, 0.0]; method=:nelder_mead, autodiff=false)
    @test r_ad_true.minimizer == r_ad_false.minimizer

    # g_tol variety: looser tolerance stops earlier (fewer or equal iterations)
    r_loose = TSAnalytics._optimize(rosenbrock, [0.0, 0.0]; g_tol=1e-2)
    r_tight = TSAnalytics._optimize(rosenbrock, [0.0, 0.0]; g_tol=1e-12)
    @test r_loose.iterations <= r_tight.iterations
    @test isapprox(r_loose.minimizer, [1.0, 1.0]; atol=1e-1)  # loose tol -> loose accuracy
    @test isapprox(r_tight.minimizer, [1.0, 1.0]; atol=1e-8)

    # iterations: a genuinely too-low cap must NOT silently claim convergence
    r_capped = TSAnalytics._optimize(rosenbrock, [0.0, 0.0]; iterations=1)
    @test r_capped.iterations <= 1
    @test !r_capped.converged

    # a generous cap on an easy problem converges comfortably under it
    r_generous = TSAnalytics._optimize(quad, [0.0, 0.0]; iterations=10_000)
    @test r_generous.converged
    @test r_generous.iterations < 10_000
end

@testset "OptimResult struct" begin
    r = TSAnalytics._optimize(x -> sum(abs2, x), [1.0, 2.0, 3.0])
    @test r isa TSAnalytics.OptimResult
    @test r.minimizer isa Vector{Float64}
    @test r.minimum isa Float64
    @test r.converged isa Bool
    @test r.iterations isa Int
    @test r.method isa Symbol
    @test isapprox(r.minimizer, [0.0, 0.0, 0.0]; atol=1e-6)
    @test isapprox(r.minimum, 0.0; atol=1e-10)
end
