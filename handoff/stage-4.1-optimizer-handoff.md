# Handoff: Stage 4.1 — Numerical Optimizer Wrapper

Status: **done.** `_optimize`/`OptimResult` (`src/optim.jl`,
`test/test_optim.jl`) built per this handoff's section 2/3 design, with
one correction to section 1's core API claim, found by running it rather
than trusting the doc: `autodiff=AutoForwardDiff()` (the `ADTypes.jl`
interface this handoff proposed, describing `autodiff=:forward` as
deprecated) throws `MethodError: promote_objtype` on the actual
`Optim.jl` version this package's own `julia = "1.9"` compat floor
resolves (`v1.11.x` -- the `ADTypes` interface is a `v2.x`-only feature
needing a newer Julia than this package supports). `autodiff=:forward`
is what actually works, confirmed by execution on real Rosenbrock/
quadratic test problems, both gradient-based methods, `autodiff=true`
and `false`. `ForwardDiff.jl` turned out to already be a direct
dependency of `Optim.jl` itself, so -- keeping to the "no unused
dependencies" policy -- neither it nor the now-unnecessary `ADTypes.jl`
were added to `Project.toml`; only `Optim = "1"` was. All of section 4's
hand-verified test values pass exactly, plus broader parameter coverage
(`g_tol`, `iterations` including a deliberately-too-low cap that must
correctly report `converged=false`, every `method` value, `autodiff`
true/false) added per this project's standing "test every parameter"
practice.

For a fresh Claude Code session picking this up with no prior context.
This is the first stage where the right move is *depending on* something
rather than building it natively -- `Optim.jl` is general numerical
infrastructure (same category as `LinearAlgebra`/BLAS), not a statistical
algorithm this package should own. The "reference, never port" policy
covers statistical algorithms; it does not mean reimplementing a mature
optimizer.

## Where this fits

- **Depends on:** nothing.
- **Everything from Stage 6 onward depends on this** -- every MLE-fit
  model (ARIMA, SARIMAX, ETS, GARCH) calls into this wrapper. Getting the
  interface right once now matters more than the two-item length of
  Stage 4 suggests.

---

## 1. Verified current `Optim.jl` API -- one real gotcha found

**The commonly-shown `autodiff = :forward` syntax is deprecated.**
Current `Optim.jl` (docs confirmed dated June 2026) requires the
`ADTypes.jl`-based interface:

```julia
using Optim
using ADTypes: AutoForwardDiff

optimize(f, x0, LBFGS(); autodiff = AutoForwardDiff())
```

Requires `ForwardDiff.jl` loaded alongside (`DifferentiationInterface.jl`
dispatches to it under the hood). Many tutorials and even some package
READMEs still show the old `:forward` symbol syntax -- **do not use it**,
it no longer works on current Optim.jl and will error with a confusing
method-not-found message rather than a clear deprecation notice.

Confirmed result-object API (from Optim.jl's own README example):
```julia
result = optimize(rosenbrock, zeros(2), BFGS())
Optim.minimizer(result)    # the argmin
Optim.minimum(result)      # the objective value there
Optim.converged(result)    # Bool
Optim.iterations(result)   # Int
```

Convergence/iteration control via `Optim.Options(...)`:
```julia
optimize(f, x0, LBFGS(), Optim.Options(g_tol=1e-8, iterations=1000); autodiff=AutoForwardDiff())
```

---

## 2. Proposed Julia wrapper API

```julia
_optimize(objective, x0::Vector{Float64};
          method::Symbol=:lbfgs,
          autodiff::Bool=true,
          g_tol::Float64=1e-8,
          iterations::Int=1000) -> OptimResult
```

Design notes:
- **`method::Symbol` over passing an `Optim.jl` type directly**: keeps
  every call site in this package decoupled from `Optim.jl`'s own type
  hierarchy -- if the backend ever needs to change, only this one wrapper
  function needs to know about it, not every model-fitting call site.
  `:lbfgs` (default), `:bfgs`, `:nelder_mead` (gradient-free fallback,
  useful for objectives `ForwardDiff` can't differentiate through)
  covers what's actually needed for this roadmap's model types.
- **`autodiff::Bool=true`, not exposing the AD backend choice**: for the
  problem sizes in this package (ARIMA/GARCH-style models, typically a
  handful to a few dozen parameters), `ForwardDiff` is the right choice
  -- reverse-mode (`AutoZygote`/`AutoEnzyme`) only pays off at much
  larger parameter counts. Not worth exposing that knob until a model
  genuinely needs it.
- **Own result type (`OptimResult`), not `Optim.jl`'s raw result
  object**: same decoupling reasoning as `method`. Every model-fitting
  function in this package should return/consume the same small struct
  regardless of what optimizer backend produced it.
- **Non-convergence is surfaced, not hidden**: `OptimResult.converged`
  is a real field callers are expected to check, not silently ignored --
  matches the project's existing pattern of `SarimaFit.converged` from
  the very first X13ArimaSeats Phase 1 sketch.

### `OptimResult`

```julia
struct OptimResult
    minimizer::Vector{Float64}
    minimum::Float64
    converged::Bool
    iterations::Int
    method::Symbol
end
```

---

## 3. Implementation

```julia
using Optim
using ADTypes: AutoForwardDiff

"""
    _optimize(objective, x0; method=:lbfgs, autodiff=true, g_tol=1e-8, iterations=1000)

Thin wrapper around Optim.jl, giving every model-fitting call site in
this package one consistent interface regardless of backend. Uses the
current (non-deprecated) `AutoForwardDiff()` AD interface -- see the
Stage 4.1 handoff doc for why the more commonly-shown `autodiff=:forward`
syntax should NOT be used (deprecated on current Optim.jl).

- `method`: `:lbfgs` (default), `:bfgs`, or `:nelder_mead` (gradient-free,
  for objectives ForwardDiff can't handle).
- `autodiff`: use ForwardDiff-based gradients (default `true`) rather
  than Optim.jl's finite-difference fallback. Ignored for `:nelder_mead`.
- `g_tol`/`iterations`: passed through to `Optim.Options`.
"""
function _optimize(objective, x0::Vector{Float64};
                    method::Symbol=:lbfgs, autodiff::Bool=true,
                    g_tol::Float64=1e-8, iterations::Int=1000)
    method in (:lbfgs, :bfgs, :nelder_mead) ||
        throw(ArgumentError("method must be :lbfgs, :bfgs, or :nelder_mead"))

    optim_method = method == :lbfgs ? LBFGS() :
                   method == :bfgs  ? BFGS()  : NelderMead()
    opts = Optim.Options(g_tol=g_tol, iterations=iterations)

    result = if autodiff && method != :nelder_mead
        optimize(objective, x0, optim_method, opts; autodiff=AutoForwardDiff())
    else
        optimize(objective, x0, optim_method, opts)
    end

    return OptimResult(Optim.minimizer(result), Optim.minimum(result),
                        Optim.converged(result), Optim.iterations(result), method)
end
```

---

## 4. Hand-verified test values

Rosenbrock's function has a well-known, extensively-documented minimum
at `(1,1)` with value `0` -- Optim.jl's own README uses exactly this
example, giving a trustworthy, independently-published check rather than
one this project derived itself:

```julia
using Test

@testset "_optimize" begin
    rosenbrock(x) = (1.0 - x[1])^2 + 100.0*(x[2] - x[1]^2)^2

    r = _optimize(rosenbrock, [0.0, 0.0])
    @test r.converged
    @test isapprox(r.minimizer, [1.0, 1.0]; atol=1e-6)
    @test isapprox(r.minimum, 0.0; atol=1e-10)

    # gradient-free method should also find it, just less precisely/slower
    r_nm = _optimize(rosenbrock, [0.0, 0.0]; method=:nelder_mead, iterations=5000)
    @test isapprox(r_nm.minimizer, [1.0, 1.0]; atol=1e-3)

    # invalid method
    @test_throws ArgumentError _optimize(rosenbrock, [0.0, 0.0]; method=:bogus)

    # A simple quadratic with a known closed-form minimum, as a second
    # independent check beyond Rosenbrock
    quad(x) = (x[1]-3.0)^2 + (x[2]+2.0)^2 + 5.0
    r_quad = _optimize(quad, [0.0, 0.0])
    @test isapprox(r_quad.minimizer, [3.0, -2.0]; atol=1e-6)
    @test isapprox(r_quad.minimum, 5.0; atol=1e-8)
end
```

---

## 5. What to do with this

1. Add `Optim.jl` and `ADTypes.jl` (and `ForwardDiff.jl`) to
   `Project.toml`'s `[deps]` -- genuine, deliberate dependencies, same
   tier as `LinearAlgebra`.
2. Implement `_optimize`/`OptimResult` per section 3.
3. Run the tests in section 4.
4. Update `development-sequence.md`'s Stage 4.1 row: mark implemented,
   note the deprecated-syntax gotcha so it doesn't get reintroduced later
   by copying an outdated tutorial example.
