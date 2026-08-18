export classical_decompose, ClassicalDecomposition

"""
    ClassicalDecomposition

Result of [`classical_decompose`](@ref). `trend` has `NaN` at the edges
unless `extrapolate_trend > 0` was requested. `figure` is the one-period
seasonal figure (length `period`); `seasonal` is `figure` tiled across
the full series length.
"""
struct ClassicalDecomposition
    observed::Vector{Float64}
    trend::Vector{Float64}
    seasonal::Vector{Float64}
    resid::Vector{Float64}
    figure::Vector{Float64}
    model::Symbol
    period::Int
end

"""
    _extrapolate_trend!(trend, npoints)

Linear-least-squares extrapolation of `trend`'s `NaN` edges, using the
`npoints` closest defined values on each side. Transcribed directly from
`statsmodels.tsa.seasonal._extrapolate_trend`'s actual source (read via
`inspect.getsource`, not from its docstring or any secondhand
description -- those describe the index arithmetic slightly differently
from what the code actually does). Works internally in the same
0-indexed coordinate system the Python source uses for the regression
itself (so the fitted line is bit-identical to it), translating only the
*storage* indices to Julia's 1-based convention -- deliberately not a
"cleaned up" 1-indexed re-derivation of the formula, which is exactly
where an earlier draft of this function introduced a subtle off-by-one
in the case where `npoints` exceeds the available defined span.
"""
function _extrapolate_trend!(trend::Vector{Float64}, npoints::Integer)
    n = length(trend)
    front = findfirst(!isnan, trend)
    back = findlast(!isnan, trend)
    front === nothing && throw(ArgumentError("trend is entirely NaN"))

    front0 = front - 1
    back0 = back - 1

    front_last0 = min(front0 + npoints, back0)
    mf = front_last0 - front0
    xs_f = collect(Float64, front0:(front0 + mf - 1))
    ys_f = trend[front:(front + mf - 1)]
    Xf = hcat(xs_f, ones(mf))
    kf, nf = Xf \ ys_f
    for i0 in 0:(front0 - 1)
        trend[i0 + 1] = kf * i0 + nf
    end

    back_first0 = max(front0, back0 - npoints)
    mb = back0 - back_first0
    xs_b = collect(Float64, back_first0:(back_first0 + mb - 1))
    ys_b = trend[(back_first0 + 1):(back_first0 + mb)]
    Xb = hcat(xs_b, ones(mb))
    kb, nb = Xb \ ys_b
    for i0 in (back0 + 1):(n - 1)
        trend[i0 + 1] = kb * i0 + nb
    end

    return trend
end

"""
    classical_decompose(x, period::Integer; model::Symbol=:additive,
                         filt::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         two_sided::Bool=true,
                         extrapolate_trend::Union{Integer,Symbol}=0) -> ClassicalDecomposition

Classical (moving-average) seasonal decomposition. Matches both R's
`stats::decompose()` and Python's `statsmodels.tsa.seasonal.
seasonal_decompose()` exactly when `two_sided=true` and
`extrapolate_trend=0` (R's only supported mode) -- both references' exact
source were read directly and cross-checked against each other and
against this implementation on shared data; see
`handoff/stage-3.1-classical-decompose-handoff.md` and
`handoff/verification/stage-3.1-verification-transcript.txt` for the full
dual-verified numeric matrix. `two_sided=false` and `extrapolate_trend`
extend beyond R's capability (no equivalent exists there at all), matching
Python's.

- `model`: `:additive` or `:multiplicative`.
- `filt`: custom trend filter coefficients; `nothing` (default) computes
  the standard "2×period" filter for even `period` ([`moving_average`](@ref)'s
  own default), uniform for odd -- identical formula in both references.
- `two_sided`: `true` (default) uses a centered trend filter; `false`
  uses a causal (past-only) one -- Python-only capability, no R
  equivalent (R's `filter()` call is always centered).
- `extrapolate_trend`: `0` (default) leaves edge `NaN`s, matching R's
  only behavior. An `Integer > 0` or `:freq` (shorthand for `period - 1`)
  linearly extrapolates the trend's edges, eliminating the `NaN`s --
  Python-only capability, no R equivalent.

Validation is deliberately **stricter than R's**: rejects non-finite
input and non-positive input under `:multiplicative` (matching Python's
checks). R's `decompose()` has neither check and would silently produce
`Inf`/`NaN`/garbage instead of erroring -- verified directly, not
assumed; not a behavior worth inheriting.

`x` accepts anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); t = 0:47; y = (100 .+ 0.5 .* t) .+ 10 .* sin.(2π .* t ./ 12) .+ randn(48);

julia> r = classical_decompose(y, 12; model=:additive);

julia> length(r.figure)
12

julia> all(isnan, r.trend[1:5])   # edge NaN with the default extrapolate_trend=0
true

julia> r2 = classical_decompose(y, 12; model=:additive, extrapolate_trend=2);

julia> any(isnan, r2.trend)
false

julia> y_neg = repeat([1.0, -1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2);

julia> classical_decompose(y_neg, 12; model=:multiplicative)
ERROR: ArgumentError: multiplicative model requires strictly positive values (R's decompose() has no such check and would silently misbehave; this does not)
```
"""
function classical_decompose(x, period::Integer; model::Symbol=:additive,
                              filt::Union{Nothing,AbstractVector{<:Real}}=nothing,
                              two_sided::Bool=true,
                              extrapolate_trend::Union{Integer,Symbol}=0)
    model in (:additive, :multiplicative) ||
        throw(ArgumentError("model must be :additive or :multiplicative"))
    y = tsvalues(x)
    n = length(y)
    any(!isfinite, y) && throw(ArgumentError(
        "classical_decompose: missing/non-finite values not supported (verified to match both " *
        "R's and statsmodels' actual behavior, not just Python's)"))
    period >= 2 || throw(ArgumentError("period must be >= 2"))
    n >= 2*period || throw(ArgumentError(
        "need at least 2 full periods ($(2*period) observations), got $n"))
    if model == :multiplicative
        all(>(0), y) || throw(ArgumentError(
            "multiplicative model requires strictly positive values " *
            "(R's decompose() has no such check and would silently misbehave; this does not)"))
    end

    f = filt === nothing ?
        (iseven(period) ? vcat(1/(2period), fill(1/period, period-1), 1/(2period)) :
                           fill(1/period, period)) :
        filt

    sides = two_sided ? 2 : 1
    trend = convolution_filter(y, f; sides=sides)

    ep = extrapolate_trend === :freq ? period - 1 : extrapolate_trend
    if ep isa Integer && ep > 0
        trend = _extrapolate_trend!(trend, ep + 1)
    end

    detrended = model == :additive ? y .- trend : y ./ trend

    figure = zeros(period)
    for i in 1:period
        idxs = i:period:n
        vals = filter(!isnan, detrended[idxs])
        figure[i] = sum(vals) / length(vals)
    end
    figure = model == :additive ? figure .- sum(figure)/period : figure ./ (sum(figure)/period)

    seasonal = [figure[mod1(i, period)] for i in 1:n]

    resid = model == :additive ? detrended .- seasonal : y ./ seasonal ./ trend

    return ClassicalDecomposition(y, trend, seasonal, resid, figure, model, period)
end
