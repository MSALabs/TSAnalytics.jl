export OptimResult

"""
    OptimResult

Result of `_optimize` -- a small, backend-independent struct every
model-fitting function in this package returns/consumes, regardless of
which `Optim.jl` method actually produced it.

- `minimizer`: the argmin found.
- `minimum`: the objective value there.
- `converged`: **not just cosmetic** -- callers are expected to check
  this rather than silently trust a non-converged fit.
- `iterations`: how many the optimizer actually used.
- `method`: which of `:lbfgs`/`:bfgs`/`:nelder_mead` produced this result.
"""
struct OptimResult
    minimizer::Vector{Float64}
    minimum::Float64
    converged::Bool
    iterations::Int
    method::Symbol
end

"""
_optimize(objective, x0; method=:lbfgs, autodiff=true, g_tol=1e-8, iterations=1000) -> OptimResult

Thin wrapper around `Optim.jl`, giving every model-fitting call site in
this package one consistent interface regardless of backend -- see
`handoff/stage-4.1-optimizer-handoff.md`.

- `method`: `:lbfgs` (default), `:bfgs`, or `:nelder_mead` (gradient-free,
  for objectives `ForwardDiff` can't differentiate through).
- `autodiff`: use `ForwardDiff`-based gradients (default `true`) rather
  than `Optim.jl`'s finite-difference fallback. Ignored for
  `:nelder_mead` (gradient-free by construction).
- `g_tol`/`iterations`: passed through to `Optim.Options`.

!!! note "`autodiff=:forward`, not `AutoForwardDiff()`"
    The handoff doc that scoped this function proposed the newer
    `ADTypes.jl`-based interface (`autodiff=AutoForwardDiff()`), citing
    current `Optim.jl` docs as the source and describing the older
    `autodiff=:forward` `Symbol` syntax as deprecated. **Verified
    directly by running it, not by trusting the doc claim**: under this
    package's own `julia = "1.9"` compat floor, `Pkg` resolves `Optim.jl`
    to `v1.11.x` (the `v2.x` line needing the `ADTypes` interface requires
    a newer Julia than this package supports) -- and `v1.11.x` does not
    support `autodiff=AutoForwardDiff()` at all (`MethodError:
    promote_objtype` -- it expects a `Symbol` in that argument position,
    not an `ADTypes` object). `autodiff=:forward` is confirmed to work
    correctly, converging exactly as expected. `ForwardDiff` itself is
    already a direct dependency of `Optim.jl`, so it doesn't need to be
    (and isn't) added as a separate dependency here -- keeping `ADTypes`
    out too, since it's genuinely unused by the code path this package
    can actually run.
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
        optimize(objective, x0, optim_method, opts; autodiff=:forward)
    else
        optimize(objective, x0, optim_method, opts)
    end

    return OptimResult(Optim.minimizer(result), Optim.minimum(result),
                        Optim.converged(result), Optim.iterations(result), method)
end
