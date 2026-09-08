export boxcox, boxcox_inv, guerrero_lambda

"""
    boxcox(x, lambda) -> Vector{Float64}

Box-Cox power transform:
```
y_t = (x_t^lambda - 1) / lambda    if lambda != 0
y_t = log(x_t)                      if lambda == 0
```
`x` must be strictly positive (the transform is undefined otherwise,
matching every reference implementation -- no silent `NaN` production).
Corroborated by fpp3 §3.1, structurally alongside decomposition (already
this project's own Stage 3.1), not a separate later chapter.

# Examples
```jldoctest
julia> using TSAnalytics

julia> boxcox([1.0, 2.0, 4.0], 0.0) == log.([1.0, 2.0, 4.0])
true

julia> round.(boxcox([1.0, 2.0, 4.0], 0.5), digits=4)
3-element Vector{Float64}:
 0.0
 0.8284
 2.0
```
"""
function boxcox(x, lambda::Real)
    xv = Float64.(collect(tsvalues(x)))
    all(>(0), xv) || throw(ArgumentError("boxcox: x must be strictly positive"))
    return lambda == 0 ? log.(xv) : (xv .^ lambda .- 1) ./ lambda
end

"""
    boxcox_inv(y, lambda) -> Vector{Float64}

Exact inverse of [`boxcox`](@ref):
```
x_t = exp(y_t)                  if lambda == 0
x_t = (lambda*y_t + 1)^(1/lambda)  if lambda != 0
```

# Examples
```jldoctest
julia> using TSAnalytics

julia> x = [1.0, 2.0, 4.0];

julia> isapprox(boxcox_inv(boxcox(x, 0.5), 0.5), x; atol=1e-10)
true
```
"""
function boxcox_inv(y, lambda::Real)
    yv = Float64.(collect(tsvalues(y)))
    return lambda == 0 ? exp.(yv) : (lambda .* yv .+ 1) .^ (1 / lambda)
end

"_boxcox_cv(lambda, subseries) -- Guerrero's (1993) coefficient of
variation of the rescaled range `s_i/m_i^(1-lambda)` across subseries
`i`, the exact quantity Guerrero's method minimizes over `lambda`."
function _boxcox_cv(lambda::Real, subseries::Vector{Vector{Float64}})
    rescaled = Float64[]
    for s in subseries
        length(s) < 2 && continue
        m = sum(s) / length(s)
        sd = sqrt(sum((s .- m) .^ 2) / (length(s) - 1))
        m > 0 || continue
        push!(rescaled, sd / m^(1 - lambda))
    end
    length(rescaled) < 2 && return Inf
    mu = sum(rescaled) / length(rescaled)
    mu == 0 && return Inf
    sd = sqrt(sum((rescaled .- mu) .^ 2) / (length(rescaled) - 1))
    return sd / mu
end

"_golden_section_min(f, lo, hi; tol=1e-6) -> Float64 -- bounded 1-D
minimization via golden-section search. Stage 4.1's `_optimize` only
wraps `Optim.jl`'s *unconstrained* methods (`LBFGS`/`BFGS`/`NelderMead`)
-- no bounded solver is wired in there for this project to reuse --
so `guerrero_lambda`'s bounded search over `bounds` uses this
self-contained, textbook golden-section implementation directly rather
than extending the shared optimizer infrastructure for one narrow use."
function _golden_section_min(f, lo::Real, hi::Real; tol::Real=1e-6, maxiter::Integer=200)
    invphi = (sqrt(5) - 1) / 2
    a, b = Float64(lo), Float64(hi)
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in 1:maxiter
        (b - a) < tol && break
        if fc < fd
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
        end
    end
    return (a + b) / 2
end

"""
    guerrero_lambda(x, period=1; bounds=(-1.0, 2.0)) -> Float64

Guerrero's (1993, *Journal of Forecasting*) automatic method for
selecting a [`boxcox`](@ref) `lambda`, matching R's `forecast`
package's own default (`forecast::BoxCox.lambda(method="guerrero")`,
whose real source -- `forecast:::guerrero`/`forecast:::guer.cv` --
this was verified directly against, not merely fpp3's textbook
description of it):

1. Let `period_eff = max(2, period)` -- the non-seasonal case
   (`period=1`) still splits into subseries of length 2, matching
   `forecast`'s own `nonseasonal.length=2` default, *not* two
   full-series halves.
2. Split `x` into `nyr = n ÷ period_eff` subseries of length
   `period_eff`, using the **last** `nyr*period_eff` observations
   (trimming any remainder from the *front*, matching
   `x[(nobsf-nobst+1):nobsf]` in `guer.cv`'s own source).
3. For each subseries `i`, compute the sample mean `m_i` and standard
   deviation `s_i`.
4. For a candidate `lambda`, compute the rescaled variability
   `s_i / m_i^(1-lambda)` for each subseries.
5. Compute the coefficient of variation of these rescaled values across
   all subseries.
6. Return the `lambda` (bounded to `bounds`, `(-1, 2)` by default) that
   **minimizes** this coefficient of variation -- the transform that
   makes the subseries' variability most *consistent*, Box-Cox's actual
   variance-stabilization goal.

Matching `forecast::BoxCox.lambda` exactly, returns `1.0` immediately
(no transform) when `length(x) <= 2*period` -- too little data for the
subseries split to be meaningful.

**Verification**: independently confirmed against real
`forecast::BoxCox.lambda(method="guerrero")` output (R 4.6.0,
`forecast` package) on three cases, including the real
`AirPassengers` series (`test/verification/transforms/`) --
agreement to 5 significant figures.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); x = abs.(randn(96)) .+ 5.0;

julia> lam = guerrero_lambda(x, 12);

julia> -1.0 <= lam <= 2.0
true
```
"""
function guerrero_lambda(x, period::Integer=1; bounds::Tuple{<:Real,<:Real}=(-1.0, 2.0))
    xv = Float64.(collect(tsvalues(x)))
    n = length(xv)
    all(>(0), xv) || throw(ArgumentError("guerrero_lambda: x must be strictly positive"))
    period >= 1 || throw(ArgumentError("guerrero_lambda: period must be >= 1"))
    bounds[1] < bounds[2] || throw(ArgumentError("guerrero_lambda: bounds must satisfy lo < hi"))

    n <= 2 * period && return 1.0

    period_eff = max(2, period)
    nyr = n ÷ period_eff
    nyr >= 2 || throw(ArgumentError(
        "guerrero_lambda: need at least 2*period observations, got n=$n, period=$period"))
    nobst = nyr * period_eff
    trimmed = xv[(n-nobst+1):n]
    subseries = [trimmed[((i-1)*period_eff+1):(i*period_eff)] for i in 1:nyr]

    return _golden_section_min(lam -> _boxcox_cv(lam, subseries), bounds[1], bounds[2])
end

export boxcox_profile_plot

"""
    boxcox_profile_plot(x, lambdas=range(-1.0, 2.0, length=50)) -> (lambdas, loglik)

Box-Cox profile log-likelihood across candidate `lambda` values -- the
classic by-eye Box-Cox selection view (Box & Cox 1964's own derivation,
as `MASS::boxcox()` computes it), adapted to a plain intercept-only
model since [`boxcox`](@ref) isn't tied to a regression here:

    loglik(lambda) = -n/2 * log(RSS(lambda)/n) + (lambda-1) * sum(log(x))

`RSS(lambda)` is the sum of squared deviations of `boxcox(x, lambda)`
from its own mean; the `(lambda-1)*sum(log(x))` term is the transform's
log-Jacobian correction, needed to make log-likelihoods comparable
across different `lambda` (without it, the profile would spuriously
favor extreme `lambda`).

**This ignores trend/seasonality entirely** (no covariates, unlike a
real `MASS::boxcox()` regression call) -- it agrees closely with
[`guerrero_lambda`](@ref) on i.i.d.-ish data (verified: both landing at
the same `lambda=-1` boundary on the same synthetic series), but a
strongly trending/seasonal real series (e.g. `AirPassengers`) can show
genuine, expected disagreement between the two criteria, since only
[`guerrero_lambda`](@ref) accounts for the seasonal subseries structure
at all.

Returns a plain `(lambdas, loglik)` NamedTuple, plottable directly via
`plot(lambdas, loglik)` -- deliberately no dedicated result type/recipe;
unlike [`periodogram`](@ref), there's no natural fixed display beyond
"y against x", so a recipe would add a type without adding a real
opinionated rendering choice worth encoding.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); x = abs.(randn(200)) .+ 5.0;

julia> profile = boxcox_profile_plot(x);

julia> length(profile.lambdas) == length(profile.loglik) == 50
true

julia> gl = guerrero_lambda(x, 12);

julia> peak_lam = profile.lambdas[argmax(profile.loglik)];

julia> isapprox(peak_lam, gl; atol=0.3)
true
```
"""
function boxcox_profile_plot(x, lambdas=range(-1.0, 2.0, length=50))
    xv = Float64.(collect(tsvalues(x)))
    all(>(0), xv) || throw(ArgumentError("boxcox_profile_plot: x must be strictly positive"))
    n = length(xv)
    logsum = sum(log, xv)
    ll = map(lambdas) do lam
        y = boxcox(xv, lam)
        rss = sum(abs2, y .- sum(y) / n)
        -n / 2 * log(rss / n) + (lam - 1) * logsum
    end
    return (lambdas=collect(lambdas), loglik=ll)
end
