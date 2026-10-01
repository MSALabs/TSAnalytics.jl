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
    boxcox_inv(y, lambda; fvar=nothing) -> Vector{Float64}

Inverse of [`boxcox`](@ref):
```
x_t = exp(y_t)                     if lambda == 0
x_t = (lambda*y_t + 1)^(1/lambda)  if lambda != 0
```

Exact on *data*. On a **forecast** it is not what most people want, and
`fvar` is the fix.

## Why a back-transformed forecast needs correcting

Back-transforming a point forecast gives the **median** of the forecast
distribution on the original scale, not its mean. Box-Cox is a
non-linear transform, so `E[g(Y)] != g(E[Y])` -- the transform does not
commute with taking an expectation, and what survives it is the
quantile, because quantiles *are* preserved by any monotone map.

For a right-skewed back-transform (which log and any `lambda < 1` are)
the median sits **below** the mean, so an uncorrected back-transformed
forecast is biased low. The bias grows with the forecast variance, so it
is worst exactly where it matters most: far out.

Pass `fvar` -- the forecast variance on the **transformed** scale, i.e.
`f.se.^2` from a [`Forecast`](@ref) of the transformed series -- to get
the mean instead:

    x_t * (1 + 0.5 * fvar_t * (1 - lambda) / x_t^(2*lambda))

fpp3 §5.6. This is a second-order Taylor correction, not an exact
expectation; it is the standard one and the same formula R's
`forecast::InvBoxCox(biasadj=TRUE, fvar=)` applies, verified against it
directly.

At `lambda = 0` the formula reduces to `exp(y)*(1 + fvar/2)`, the
familiar log-normal mean correction, since `x^0 = 1`.

!!! note "R's `biasadj` flag has no counterpart here"
    R takes `biasadj=TRUE` *and* `fvar=`, and errors if the flag is set
    without the variance -- so the flag carries no information its own
    source does not already have. Supplying `fvar` is the request here;
    there is no separate switch to forget to set.

**Whether to correct is a real choice, not an oversight to fix.** If you
want the value the series is equally likely to fall above or below --
and for a skewed quantity that is often the more useful summary -- the
uncorrected median is the right answer. Correct when you need a mean:
when the forecasts will be **summed or aggregated**, since medians do
not add, or when feeding a calculation that assumes an expectation.

# Examples
```jldoctest
julia> using TSAnalytics

julia> x = [1.0, 2.0, 4.0];

julia> isapprox(boxcox_inv(boxcox(x, 0.5), 0.5), x; atol=1e-10)
true
```

```jldoctest
julia> using TSAnalytics

julia> round.(boxcox_inv([1.0, 1.5], 0.3), digits=8)
2-element Vector{Float64}:
 2.39779016
 3.45058985

julia> round.(boxcox_inv([1.0, 1.5], 0.3; fvar=[0.04, 0.09]), digits=8)
2-element Vector{Float64}:
 2.41765351
 3.50228716
```

See also [`boxcox`](@ref), [`boxcox_lambda`](@ref), [`guerrero_lambda`](@ref).
"""
function boxcox_inv(y, lambda::Real; fvar=nothing)
    yv = Float64.(collect(tsvalues(y)))
    out = lambda == 0 ? exp.(yv) : (lambda .* yv .+ 1) .^ (1 / lambda)
    fvar === nothing && return out

    v = fvar isa Real ? fill(Float64(fvar), length(out)) :
                        Float64.(collect(tsvalues(fvar)))
    length(v) == length(out) || throw(DimensionMismatch(
        "boxcox_inv: fvar must be a scalar or have the same length as y " *
        "(got $(length(v)) for $(length(out)))"))
    all(>=(0), v) || throw(ArgumentError("boxcox_inv: fvar is a variance, so it must be non-negative"))

    # one formula covers both cases: at lambda=0, out^(2*lambda) is 1 and this
    # collapses to the log-normal mean correction exp(y)*(1 + fvar/2)
    return out .* (1 .+ 0.5 .* v .* (1 - lambda) ./ out .^ (2 * lambda))
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

export boxcox_lambda

"""
    boxcox_lambda(x; bounds=(-1.0, 2.0)) -> Float64

Box-Cox `lambda` by **maximum likelihood** -- the value maximising the
profile log-likelihood [`boxcox_profile_plot`](@ref) traces out,

    -n/2 * log(RSS(lambda)/n) + (lambda - 1) * sum(log(x))

which is the same objective `scipy.stats.boxcox` maximises. `x` must be
strictly positive.

This is the sibling of [`guerrero_lambda`](@ref) and they answer
different questions. Guerrero's method minimises the coefficient of
variation *across seasonal blocks*, so it targets variance
stabilisation and needs a `period`; this maximises a Gaussian
likelihood, so it targets normality and does not. On a series whose
variance grows with its level they tend to agree; on one that is merely
skewed they need not.

!!! warning "`bounds` is why this can differ from scipy"
    The search is **bounded**, `[-1, 2]` by default, matching
    [`guerrero_lambda`](@ref)'s own default and R's
    `forecast::BoxCox.lambda` (`lower=-1, upper=2`).
    `scipy.stats.boxcox` is **unbounded**.

    When the unconstrained optimum lies outside the interval the two
    genuinely disagree, and this returns the boundary. On
    `test/verification/transforms/bc_y.csv` scipy's MLE is
    `-1.836495441613`; bounded on `[-1, 2]` the answer is `-1.0`.
    Neither is wrong -- a lambda of `-1.8` is a reciprocal-squared
    transform that few people intend, which is why both R and this
    package bound it. Widen `bounds` to reproduce scipy.

```jldoctest
julia> using TSAnalytics

julia> x = Float64[2, 3, 5, 8, 13, 21, 34, 55, 89, 144];

julia> lam = boxcox_lambda(x);

julia> -1.0 <= lam <= 2.0
true
```

See also [`boxcox`](@ref), [`boxcox_inv`](@ref), [`guerrero_lambda`](@ref),
[`boxcox_profile_plot`](@ref).
"""
function boxcox_lambda(x; bounds::Tuple{<:Real,<:Real}=(-1.0, 2.0))
    xv = Float64.(collect(tsvalues(x)))
    all(>(0), xv) || throw(ArgumentError("boxcox_lambda: x must be strictly positive"))
    length(xv) >= 2 || throw(ArgumentError("boxcox_lambda: need at least 2 observations"))
    bounds[1] < bounds[2] || throw(ArgumentError("boxcox_lambda: bounds must satisfy lo < hi"))

    n = length(xv)
    logsum = sum(log, xv)
    # minimise the negative profile log-likelihood; the profile is unimodal in
    # lambda for positive data, which is what makes a golden section safe here
    # (the same routine guerrero_lambda uses, for the same reason)
    negll(lam) = begin
        yt = boxcox(xv, lam)
        rss = sum(abs2, yt .- sum(yt) / n)
        rss <= 0 && return Inf
        -(-n / 2 * log(rss / n) + (lam - 1) * logsum)
    end
    return _golden_section_min(negll, Float64(bounds[1]), Float64(bounds[2]))
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
