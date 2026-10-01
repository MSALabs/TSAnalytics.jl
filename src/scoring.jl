export pinball_loss, crps_normal, crps_ensemble, winkler_score, interval_coverage,
       interval_accuracy

"""
    pinball_loss(actual, predicted_quantile, tau) -> Float64

Mean quantile ("pinball") loss at level `tau`:

    L = mean(tau * (y - q)      where y >= q,
             (1 - tau) * (q - y) where y <  q)

This is the scoring rule a quantile forecast is **supposed** to be
evaluated with: it is minimised in expectation by the true `tau`
quantile, so reporting it is what makes a quantile forecast falsifiable.
Lower is better.

At `tau = 0.5` it is exactly **half** the mean absolute error — the
median minimises MAE, and the pinball loss at the median is the same
objective scaled by `1/2`. That identity is the cheapest check that an
implementation has the asymmetry the right way round.

```jldoctest
julia> using TSAnalytics

julia> y = [3.0, -0.5, 2.0, 7.0];

julia> q = [2.5, 0.0, 2.0, 8.0];

julia> round(pinball_loss(y, q, 0.5), digits=6) == round(0.5 * mae(y, q), digits=6)
true
```

See also [`winkler_score`](@ref), [`crps_normal`](@ref), [`accuracy`](@ref).
"""
function pinball_loss(actual, predicted_quantile, tau::Real)
    a = Float64.(collect(tsvalues(actual)))
    q = Float64.(collect(tsvalues(predicted_quantile)))
    length(a) == length(q) || throw(DimensionMismatch(
        "pinball_loss: actual and predicted_quantile must have the same length " *
        "(got $(length(a)) and $(length(q)))"))
    isempty(a) && throw(ArgumentError("pinball_loss: inputs must be non-empty"))
    0 < tau < 1 || throw(ArgumentError("pinball_loss: tau must be in (0, 1), got $tau"))

    s = 0.0
    @inbounds for i in eachindex(a)
        d = a[i] - q[i]
        s += d >= 0 ? tau * d : (tau - 1) * d
    end
    return s / length(a)
end

"""
    crps_normal(actual, mu, sigma) -> Float64

Mean continuous ranked probability score for Gaussian predictive
distributions `N(mu, sigma)`:

    CRPS = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],
    z = (y - mu) / sigma

the closed form given by Gneiting & Raftery (2007) for a normal
predictive distribution. `mu` and `sigma` may each be a scalar or a
vector the same length as `actual`. Lower is better.

CRPS scores the **whole predictive distribution** rather than a point or
an interval, in the units of the data, and it reduces to the absolute
error as the distribution collapses to a point — which makes it directly
comparable with [`mae`](@ref) and is the cheapest sanity check on it.

Every interval this package emits is Gaussian (see [`forecast`](@ref)),
so this is the scoring rule that applies to them.

!!! note "Verified against the definition, not another package"
    Neither R's `scoringRules` nor Python's `properscoring` is reachable
    in this project's environment, so there is no package to match
    against. Instead the closed form is checked against **numerical
    integration of the definition**,

        CRPS(F, y) = integral (F(x) - 1{x >= y})^2 dx

    at seven `(y, mu, sigma)` combinations, agreeing to machine
    precision (max difference `1.8e-15`). That is arguably a stronger
    check than reproducing another implementation, but it is a different
    one, and worth knowing which you have.

```jldoctest
julia> using TSAnalytics

julia> round(crps_normal([0.0], 0.0, 1.0), digits=10)
0.2336949773
```

See also [`crps_ensemble`](@ref), [`pinball_loss`](@ref).
"""
function crps_normal(actual, mu, sigma)
    a = Float64.(collect(tsvalues(actual)))
    n = length(a)
    isempty(a) && throw(ArgumentError("crps_normal: inputs must be non-empty"))
    m = _broadcast_like(mu, n, "mu")
    s = _broadcast_like(sigma, n, "sigma")
    all(>(0), s) || throw(ArgumentError("crps_normal: sigma must be strictly positive"))

    total = 0.0
    inv_sqrt_pi = 1 / sqrt(pi)
    @inbounds for i in 1:n
        z = (a[i] - m[i]) / s[i]
        total += s[i] * (z * (2 * _std_normal_cdf(z) - 1) + 2 * _std_normal_pdf(z) - inv_sqrt_pi)
    end
    return total / n
end

function _broadcast_like(v, n::Int, name::AbstractString)
    v isa Real && return fill(Float64(v), n)
    out = Float64.(collect(tsvalues(v)))
    length(out) == n || throw(DimensionMismatch(
        "crps_normal: $name must be a scalar or have length $n, got $(length(out))"))
    return out
end

_std_normal_pdf(z::Real) = exp(-z^2 / 2) / sqrt(2pi)

# Standard normal CDF from the chi-squared tail already in diagnostics.jl:
# Z^2 ~ ChiSq(1), so _chisq_ccdf(z^2, 1) is the two-sided tail P(|Z| > |z|)
# and half of it is the one-sided tail. The same identity arx.jl already uses
# for its two-sided p-values -- reusing verified machinery rather than taking
# a SpecialFunctions dependency for one `erf` call.
_std_normal_cdf(z::Real) = z >= 0 ? 1 - _chisq_ccdf(z^2, 1) / 2 : _chisq_ccdf(z^2, 1) / 2

"""
    crps_ensemble(actual, paths) -> Float64

CRPS from a **simulated ensemble** rather than a parametric
distribution, using the energy form

    CRPS = mean|x_i - y| - (1/(2n^2)) * sum_ij |x_i - x_j|

`paths` is a matrix whose columns are horizons and rows are simulation
draws -- exactly the shape [`forecast_volatility`](@ref)'s
`variance_paths` has -- or a plain vector for a single horizon.
`actual` has one entry per horizon.

This is what to reach for when the predictive distribution is not
Gaussian and only draws from it are available, which is the EGARCH case:
`forecast_volatility` has no closed form there and simulates instead, so
a parametric score would be assuming away the reason simulation was
needed.

Cost is `O(n^2)` per horizon in the number of draws, which at the
default `simulations = 1000` is a million absolute differences per
horizon -- fine, but worth knowing before raising the draw count to
score rather than to forecast.

See also [`crps_normal`](@ref).
"""
function crps_ensemble(actual, paths)
    a = Float64.(collect(tsvalues(actual)))
    P = paths isa AbstractMatrix ? Float64.(paths) : reshape(Float64.(collect(tsvalues(paths))), :, 1)
    isempty(a) && throw(ArgumentError("crps_ensemble: actual must be non-empty"))
    size(P, 2) == length(a) || throw(DimensionMismatch(
        "crps_ensemble: paths must have one column per horizon -- got $(size(P, 2)) " *
        "columns for $(length(a)) actuals"))
    size(P, 1) >= 2 || throw(ArgumentError("crps_ensemble: need at least 2 draws"))

    total = 0.0
    n = size(P, 1)
    for h in eachindex(a)
        col = view(P, :, h)
        term1 = 0.0
        @inbounds for i in 1:n
            term1 += abs(col[i] - a[h])
        end
        term1 /= n
        # the pairwise term is symmetric with a zero diagonal, so only the
        # strict upper triangle is computed and doubled
        term2 = 0.0
        @inbounds for i in 1:n, j in (i+1):n
            term2 += abs(col[i] - col[j])
        end
        total += term1 - term2 / (n^2)
    end
    return total / length(a)
end

"""
    winkler_score(actual, lower, upper, level) -> Float64

Mean Winkler interval score at the given `level` (a percentage, e.g.
`95.0`, matching [`forecast`](@ref)'s own convention). With
`alpha = 1 - level/100`:

    W = (u - l)                            if l <= y <= u
        (u - l) + (2/alpha) * (l - y)      if y < l
        (u - l) + (2/alpha) * (y - u)      if y > u

fpp3 §5.9. Lower is better. A narrow interval scores well **only** while
the actuals stay inside it, so this is the measure that stops an
interval from being made to look good by shrinking it — the failure mode
[`interval_coverage`](@ref) alone cannot see.

```jldoctest
julia> using TSAnalytics

julia> winkler_score([1.0], [0.0], [2.0], 95.0)   # inside: just the width
2.0

julia> round(winkler_score([3.0], [0.0], [2.0], 95.0), digits=6)   # outside by 1
42.0
```

The penalty is `2/alpha` per unit outside, so at 95 % a miss by `1`
costs `40` on top of the width — steep on purpose, because an interval
that excludes the outcome has failed at the one thing it was for.

See also [`interval_coverage`](@ref), [`interval_accuracy`](@ref).
"""
function winkler_score(actual, lower, upper, level::Real)
    a = Float64.(collect(tsvalues(actual)))
    l = Float64.(collect(tsvalues(lower)))
    u = Float64.(collect(tsvalues(upper)))
    length(a) == length(l) == length(u) || throw(DimensionMismatch(
        "winkler_score: actual, lower and upper must have the same length " *
        "(got $(length(a)), $(length(l)), $(length(u)))"))
    isempty(a) && throw(ArgumentError("winkler_score: inputs must be non-empty"))
    0 < level < 100 || throw(ArgumentError("winkler_score: level must be in (0, 100), got $level"))
    all(l .<= u) || throw(ArgumentError("winkler_score: every lower bound must be <= its upper bound"))

    alpha = 1 - level / 100
    s = 0.0
    @inbounds for i in eachindex(a)
        w = u[i] - l[i]
        s += if a[i] < l[i]
            w + 2 / alpha * (l[i] - a[i])
        elseif a[i] > u[i]
            w + 2 / alpha * (a[i] - u[i])
        else
            w
        end
    end
    return s / length(a)
end

"""
    interval_coverage(actual, lower, upper) -> Float64

Fraction of `actual` values falling inside `[lower, upper]`, in `[0, 1]`.

Compare it against the nominal level: a 95 % interval covering `0.72` of
the actuals is badly optimistic, and one covering `1.00` of twelve points
is either conservative or lucky and the two cannot be told apart at that
sample size.

**Coverage alone cannot be trusted**, which is why
[`interval_accuracy`](@ref) reports it beside the Winkler score: an
arbitrarily wide interval has perfect coverage and is useless, and only
the score penalises width.

Bounds are inclusive, matching the convention that a point exactly on
the boundary is covered.

See also [`winkler_score`](@ref).
"""
function interval_coverage(actual, lower, upper)
    a = Float64.(collect(tsvalues(actual)))
    l = Float64.(collect(tsvalues(lower)))
    u = Float64.(collect(tsvalues(upper)))
    length(a) == length(l) == length(u) || throw(DimensionMismatch(
        "interval_coverage: actual, lower and upper must have the same length " *
        "(got $(length(a)), $(length(l)), $(length(u)))"))
    isempty(a) && throw(ArgumentError("interval_coverage: inputs must be non-empty"))
    return count(i -> l[i] <= a[i] <= u[i], eachindex(a)) / length(a)
end

"""
    interval_accuracy(actual, f::Forecast) -> NamedTuple

Score every interval a [`Forecast`](@ref) carries, in one call:

```julia
(level = [80.0, 95.0],
 coverage = [...],        # one per level
 winkler = [...],         # one per level
 crps = ...)              # one number, from the Gaussian predictive distribution
```

The counterpart of [`accuracy`](@ref), which covers point forecasts
only. **Without this, every prediction interval this package emits is
unfalsifiable** — you can see that an interval is wide, but not whether
it was wide in the right places.

`crps` uses [`crps_normal`](@ref) with the forecast's own `point` and
`se`, which is the predictive distribution the intervals were built
from.

See also [`accuracy`](@ref), [`winkler_score`](@ref), [`interval_coverage`](@ref).
"""
function interval_accuracy(actual, f::Forecast)
    a = Float64.(collect(tsvalues(actual)))
    length(a) == f.horizon || throw(DimensionMismatch(
        "interval_accuracy: actual must have one entry per forecast step -- got " *
        "$(length(a)) for horizon=$(f.horizon)"))
    cov = [interval_coverage(a, view(f.lower, :, j), view(f.upper, :, j))
            for j in eachindex(f.levels)]
    wink = [winkler_score(a, view(f.lower, :, j), view(f.upper, :, j), f.levels[j])
             for j in eachindex(f.levels)]
    return (level=copy(f.levels), coverage=cov, winkler=wink,
            crps=crps_normal(a, f.point, f.se))
end
