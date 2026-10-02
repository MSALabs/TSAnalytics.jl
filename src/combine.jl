export combine_forecasts

"""
    combine_forecasts(forecasts; weights=nothing) -> Forecast

Combine several forecasts of the same series over the same horizon into
one. With `weights=nothing` the weights are **equal**, which is the
version that is hard to beat in practice.

```julia
combine_forecasts([f_sarima, f_ets, f_naive])
combine_forecasts([f1, f2]; weights=[0.7, 0.3])
```

`weights` must be non-negative and are normalised to sum to `1`, so
`[2, 1]` and `[2/3, 1/3]` mean the same thing.

## What combining does and does not buy

Bates & Granger (1969) started this, and the finding that has held up
across forecasting competitions since is that **equal weights are very
hard to improve on** — estimated optimal weights must themselves be
estimated, and that error routinely costs more than the optimality
gains. Hence the default.

**It does not generally beat the best component**, and it is worth being
blunt about that, because the opposite is often implied. Measured on the
cardox hold-out:

| | RMSE |
|---|---|
| Best of three components | `0.431` |
| Mean of the three components | `1.323` |
| Equal-weight combination | **`0.965`** |

The combination beats the *average* component comfortably and the *best*
component not at all — two of those three were weak benchmarks, and
averaging a good forecast with poor ones drags it down. Nor does
pairing two comparable models fix it: two SARIMAs on this series give
`0.431` and `0.215`, and their combination `0.266`, landing between them
rather than below both, because forecasts from similar models fitted to
the same data have strongly **positively correlated** errors and so
diversify little.

So the case for combining is **robustness, not optimality**: you get
close to the better component without having had to know in advance
which one that was, and relative accuracy is unstable enough that
in-sample ranking is a poor guide to out-of-sample ranking. Combine when
you cannot confidently pick; pick when you can.

If you supply weights, you should be able to say where they came from.

## What the interval does, and what it does not

The combined point forecast is the weighted mean. The combined standard
error is the **weighted root-mean-square** of the components',

    se = sqrt(sum(w_i^2 * se_i^2))

which is the variance of the weighted mean **assuming the component
forecast errors are independent**. They are not: forecasts of the same
series from models fitted to the same data are strongly positively
correlated, so this **understates** the true combined uncertainty.

!!! warning "The combined interval is optimistic, by construction"
    Doing better needs the covariance between the component errors,
    which needs a holdout history of their joint errors — not something
    a `Forecast` carries. Treat the combined point forecast as the
    useful output and the combined interval as a lower bound on the
    spread.

    Both references behave the same way. R's `forecast` has no
    combination function at all; `forecastHybrid::hybridModel` averages
    point forecasts and recomputes intervals from the combined
    residuals, which needs the fitted objects rather than their
    forecasts.

Levels must match across inputs, and the returned intervals are rebuilt
from the combined point forecast and standard error at those levels
rather than averaged — averaging interval *bounds* would not correspond
to any distribution.

```jldoctest
julia> using TSAnalytics

julia> y = Float64[1:40;] .+ repeat([0.0, 2.0, -1.0, 0.5], 10);

julia> f1 = forecast(fit_arima(y, (1, 1, 0)), 4);

julia> f2 = naive(y, 4);

julia> c = combine_forecasts([f1, f2]);

julia> c.horizon
4

julia> c.model_name
"Combination(2, equal weights)"

julia> isapprox(c.point, (f1.point .+ f2.point) ./ 2; atol=1e-10)
true
```

See also [`accuracy`](@ref), [`interval_accuracy`](@ref), [`forecast`](@ref).
"""
function combine_forecasts(forecasts::AbstractVector{Forecast};
                            weights::Union{Nothing,AbstractVector{<:Real}}=nothing)
    k = length(forecasts)
    k >= 2 || throw(ArgumentError(
        "combine_forecasts: need at least 2 forecasts to combine, got $k"))

    h = forecasts[1].horizon
    all(f -> f.horizon == h, forecasts) || throw(DimensionMismatch(
        "combine_forecasts: every forecast must share the same horizon -- got " *
        string([f.horizon for f in forecasts])))
    lv = forecasts[1].levels
    all(f -> f.levels == lv, forecasts) || throw(ArgumentError(
        "combine_forecasts: every forecast must share the same `levels` -- " *
        "combining intervals quoted at different levels is not meaningful"))

    w = if weights === nothing
        fill(1 / k, k)
    else
        length(weights) == k || throw(DimensionMismatch(
            "combine_forecasts: got $(length(weights)) weights for $k forecasts"))
        all(>=(0), weights) || throw(ArgumentError("combine_forecasts: weights must be non-negative"))
        s = sum(weights)
        s > 0 || throw(ArgumentError("combine_forecasts: weights must not all be zero"))
        Float64.(weights) ./ s
    end

    point = zeros(h)
    var = zeros(h)
    for (i, f) in enumerate(forecasts)
        @. point += w[i] * f.point
        # independence assumption -- see the docstring; this is a lower bound
        @. var += w[i]^2 * f.se^2
    end
    se = sqrt.(var)

    z = [_confidence_z(1 - l / 100) for l in lv]
    lower = reduce(hcat, [point .- zi .* se for zi in z])
    upper = reduce(hcat, [point .+ zi .* se for zi in z])

    name = weights === nothing ? "Combination($k, equal weights)" : "Combination($k, weighted)"
    return Forecast(point, se, copy(lv), lower, upper, h, name)
end
