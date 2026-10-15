# ---------------------------------------------------------------------------
# Intermittent demand (Stage 12.1,
# handoff/stage 12/docs/stage-12.1-intermittent-demand-handoff.md).
#
# Series that are mostly zero with occasional demands. Croston (1972)
# smooths the non-zero demands and the intervals between them separately
# and divides; SBA (Syntetos & Boylan 2005) removes the ratio's bias; TSB
# (Teunter, Syntetos & Babai 2011) smooths the probability of a demand
# instead of the interval; ADIDA (Nikolopoulos et al. 2011) and IMAPA
# (Petropoulos & Kourentzes 2015) aggregate to buckets first.
#
# Every point forecast agrees with statsforecast (and, for Croston, with R's
# forecast::croston) exactly: the simple exponential smoothing starts at the
# first value, as both do, and the bounded alpha search is the same golden-
# section search with the same stopping rule.
# ---------------------------------------------------------------------------

export croston, tsb, adida, imapa, IntermittentModel, IntermittentForecast
export demand_intervals, DemandIntervals, adi_cv2, ADICV2Result

# simple exponential smoothing started at the first value: the one-step
# forecast after the whole of x, and the in-sample one-step forecasts
function _ses_level(x::AbstractVector{Float64}, α::Real)
    f = x[1]
    for i in 2:length(x)
        f = α * x[i - 1] + (1 - α) * f
    end
    return α * x[end] + (1 - α) * f
end

function _ses_sse(α::Real, x::AbstractVector{Float64})
    length(x) < 2 && return 0.0
    f = x[1]
    s = 0.0
    for i in 2:length(x)
        f = α * x[i - 1] + (1 - α) * f
        s += (x[i] - f)^2
    end
    return s
end

# golden-section search for the alpha minimising the one-step SSE on
# [lo, hi], stopped when the bracket is below 1e-12 and returning its
# midpoint; an exact tie between the two interior points stops the search
# (so a constant series returns the midpoint of [lo, hi], 0.2)
function _ses_golden(x::AbstractVector{Float64}, lo::Real=0.1, hi::Real=0.3)
    gr = (sqrt(5.0) + 1.0) / 2.0
    a, b = Float64(lo), Float64(hi)
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc, fd = _ses_sse(c, x), _ses_sse(d, x)
    while abs(b - a) >= 1e-12
        if fc < fd
            b, d, fd = d, c, fc
            c = b - (b - a) / gr
            fc = _ses_sse(c, x)
        elseif fd < fc
            a, c, fc = c, d, fd
            d = a + (b - a) / gr
            fd = _ses_sse(d, x)
        else
            break
        end
    end
    return (a + b) / 2
end

_imt_demand(y) = y[y .> 0]
function _imt_intervals(y)
    idx = findall(!=(0), y)
    return Float64.(diff(vcat(0, idx)))
end

# the forecast of y aggregated into buckets of width k, aligned to the end
# of the series (the incomplete bucket at the start is dropped)
function _imt_chunk_forecast(y::AbstractVector{Float64}, k::Integer)
    n = length(y)
    lost = n % k
    nchunks = (n - lost) ÷ k
    nchunks == 0 && return y[end]
    sums = [sum(view(y, (lost + (i - 1) * k + 1):(lost + i * k))) for i in 1:nchunks]
    nchunks == 1 && return sums[1]
    return _ses_level(sums, _ses_golden(sums))
end

function _imt_check(y, who)
    yv = collect(Float64, tsvalues(y))
    isempty(yv) && throw(ArgumentError("$who: y is empty"))
    any(!isfinite, yv) && throw(ArgumentError("$who: y contains missing/non-finite values"))
    any(<(0), yv) && throw(ArgumentError(
        "$who: demand cannot be negative (R's forecast::croston refuses too; statsforecast " *
        "counts a negative as an arrival that contributes no demand)"))
    nz = findall(!=(0), yv)
    if length(nz) == 1 && nz[1] == 1 && length(yv) > 1
        throw(ArgumentError(
            "$who: a single demand at the first period followed by $(length(yv) - 1) zero" *
            "$(length(yv) == 2 ? "" : "s") gives no inter-arrival rate to estimate -- the " *
            "references return the demand itself ($(yv[1])), against an empirical rate of " *
            "$(round(yv[1] / length(yv), sigdigits=4)) per period"))
    end
    return yv
end

"""
    IntermittentModel

A fitted intermittent-demand model, from [`croston`](@ref), [`tsb`](@ref),
[`adida`](@ref) or [`imapa`](@ref): the `method`, the flat point
`forecast`, the smoothed `demand_level` and `interval_level` (Croston; the
`probability_level` for TSB), the smoothing constants used, the aggregation
level (ADIDA/IMAPA), the in-sample `fitted` one-step forecasts where they
are computed (`nothing` otherwise), and the [`adi_cv2`](@ref)
classification of the series.
"""
struct IntermittentModel <: UnivariateModel
    method::Symbol
    y::Vector{Float64}
    forecast::Float64
    demand_level::Float64
    interval_level::Float64
    probability_level::Float64
    alpha_demand::Float64
    alpha_interval::Float64
    alpha_probability::Float64
    aggregation::Int
    fitted::Union{Nothing,Vector{Float64}}
    classification::Union{Nothing,Symbol}
end

"""
    IntermittentForecast

Point forecasts from an [`IntermittentModel`](@ref) -- flat in the
horizon -- and nothing else. **There are no prediction intervals**: neither
R's `forecast::croston` nor `statsforecast` computes any (one passes
`PI = FALSE`, the other raises unless wrapped in conformal intervals,
which give negative lower bounds on a count series), and a [`Forecast`](@ref)
with invented ones would be worse than none.
"""
struct IntermittentForecast
    point::Vector{Float64}
    horizon::Int
    method::Symbol
end

_imt_class(y) = (count(!=(0), y) >= 2 ? adi_cv2(y).method : nothing)

# Croston's in-sample one-step forecasts in one pass: the demand and
# interval levels change only at an arrival, and are carried through runs of
# zeros (R recomputes the whole method on every prefix -- the same numbers to
# a few ulp, at O(n^2))
function _croston_fitted(y, αd, αi, factor)
    n = length(y)
    fit = fill(NaN, n)
    zd = zi = 0.0               # last demand, last interval
    fd = fi = NaN               # the SES fits before the last arrival
    k = 0
    last = 0
    for t in 2:n
        if y[t - 1] != 0
            k += 1
            d = y[t - 1]; iv = (t - 1) - last; last = t - 1
            fd = k == 1 ? d : αd * zd + (1 - αd) * fd
            fi = k == 1 ? iv : αi * zi + (1 - αi) * fi
            zd, zi = d, iv
        end
        fit[t] = if k == 0
            0.0
        else
            dl = αd * zd + (1 - αd) * fd
            il = αi * zi + (1 - αi) * fi
            factor * (il == 0 ? dl : dl / il)
        end
    end
    return fit
end

# Optimised Croston's in-sample forecasts: the alphas are re-chosen on every
# prefix of the demand and interval sequences (the reference does the same,
# and warns that it is expensive -- it is O(n^2) golden-section searches)
function _croston_opt_fitted(y)
    n = length(y)
    fit = fill(NaN, n)
    d = Float64[]; iv = Float64[]
    dl = il = NaN
    last = 0
    for t in 2:n
        if y[t - 1] != 0
            push!(d, y[t - 1]); push!(iv, (t - 1) - last); last = t - 1
            dl = _ses_level(d, _ses_golden(d))
            il = _ses_level(iv, _ses_golden(iv))
            il == 0 && (il = 1.0)
        end
        fit[t] = isempty(d) ? 0.0 : dl / il
    end
    return fit
end

# ADIDA's in-sample forecasts: at each prefix, the width is the rounded
# running mean interval (updated at each arrival; 1 before the first), and
# the forecast is the bucketed forecast of the prefix divided by it
function _adida_fitted(y)
    n = length(y)
    fit = fill(NaN, n)
    k = 1
    s = 0.0; c = 0; last = 0
    for t in 2:n
        if y[t - 1] != 0
            c += 1; s += (t - 1) - last; last = t - 1
            k = max(Int(round(s / c)), 1)
        end
        fit[t] = _imt_chunk_forecast(view(y, 1:(t - 1)), k) / k
    end
    return fit
end

# IMAPA's in-sample forecasts: the per-prefix average over widths
# 1..min(rounded mean interval, prefix length); zero before the first demand.
# Rounded half to even like the forecast; statsforecast's fitted path rounds
# half away from zero instead (its D26 inconsistency) -- `mode` reproduces it
function _imapa_fitted(y; mode=RoundNearest)
    n = length(y)
    fit = fill(NaN, n)
    for t in 2:n
        pre = view(y, 1:(t - 1))
        iv = _imt_intervals(pre)
        if isempty(iv)
            fit[t] = 0.0
            continue
        end
        K = min(max(Int(round(mean(iv), mode)), 1), t - 1)
        fit[t] = _pairwise_sum([_imt_chunk_forecast(pre, k) / k for k in 1:K]) / K
    end
    return fit
end

"""
    croston(y; method=:classic, alpha=0.1) -> IntermittentModel

Croston's (1972) method for intermittent demand: simple exponential
smoothing of the non-zero demands, divided by the same smoothing of the
intervals between them, a flat forecast of demand per period.

- `:classic`: both smoothed with `alpha` (0.1, the value both references
  hardcode). Identical to R's `forecast::croston` and `statsforecast`'s
  `CrostonClassic`, to all 17 digits.
- `:sba`: Syntetos & Boylan's (2005) bias correction, the classic forecast
  times `1 − alpha/2`. `statsforecast` multiplies by a literal `0.95`,
  which is `1 − alpha/2` only at its hardcoded `alpha = 0.1`; here the
  factor follows `alpha`. (Syntetos & Boylan's later SBJ factor
  `1 − alpha/(2 − alpha)` is not offered.) A single observation is refused:
  there is no bias to correct.
- `:optimized`: each of the two smoothings gets its own `alpha`, chosen by a
  golden-section search for the smallest one-step squared error **within
  `[0.1, 0.3]`** -- `statsforecast`'s `CrostonOptimized`. The bound binds
  often (on two of four test series one `alpha` ends at it), so this is a
  restricted search, not a free optimum.

Simple exponential smoothing starts from the first value, as in both
references -- the reason they agree exactly. Refused: negative values, and
a single demand at the first period followed only by zeros, for which both
references return the demand itself (5 for `[5, 0, …, 0]`, against an
empirical rate of 0.5). An all-zero series forecasts zero.

```jldoctest
julia> using TSAnalytics

julia> m = croston([0, 0, 3, 0, 0, 0, 2, 0, 4, 0]);

julia> round(m.forecast, digits = 6), m.method
(1.006689, :classic)

julia> forecast(m, 3).point == fill(m.forecast, 3)
true
```
"""
function croston(y; method::Symbol=:classic, alpha::Real=0.1)
    method in (:classic, :sba, :optimized) || throw(ArgumentError(
        "croston: method must be :classic, :sba or :optimized (TSB, ADIDA and IMAPA are tsb, adida, imapa)"))
    0 < alpha <= 1 || throw(ArgumentError("croston: alpha must be in (0, 1]"))
    yv = _imt_check(y, "croston")
    method === :sba && length(yv) == 1 && throw(ArgumentError(
        "croston: SBA on a single observation -- there is no bias to correct (statsforecast " *
        "returns 0.95 of it)"))
    if all(==(0), yv)
        return IntermittentModel(method, yv, 0.0, 0.0, NaN, NaN, alpha, alpha, NaN, 1,
                                 zeros(length(yv)), nothing)
    end
    d = _imt_demand(yv)
    iv = _imt_intervals(yv)
    αd, αi = method === :optimized ? (_ses_golden(d), _ses_golden(iv)) : (Float64(alpha), Float64(alpha))
    dl = _ses_level(d, αd)
    il = _ses_level(iv, αi)
    factor = method === :sba ? 1 - αi / 2 : 1.0
    fc = factor * (il == 0 ? dl : dl / il)
    fit = method === :optimized ? _croston_opt_fitted(yv) : _croston_fitted(yv, αd, αi, factor)
    return IntermittentModel(method, yv, fc, dl, il, NaN, αd, αi, NaN, 1, fit, _imt_class(yv))
end

"""
    tsb(y; alpha_d, alpha_p) -> IntermittentModel

Teunter, Syntetos & Babai's (2011) method: the probability of a demand in
a period, smoothed over **every** period with `alpha_p`, times the size of a
demand, smoothed over the non-zero demands with `alpha_d`. Unlike Croston
there is no interval -- so the forecast decays during a long run of zeros
instead of waiting for the next demand, which is what makes it suited to
items going obsolete. Both constants are required: `statsforecast`'s
docstring says they are optimised as in its optimised Croston, but its code
takes them as given, and so does this.

```jldoctest
julia> using TSAnalytics

julia> round(tsb([0, 0, 3, 0, 0, 0, 2, 0, 4, 0]; alpha_d = 0.2, alpha_p = 0.2).forecast, digits = 6)
0.925203
```
"""
function tsb(y; alpha_d::Real, alpha_p::Real)
    (0 < alpha_d <= 1 && 0 < alpha_p <= 1) || throw(ArgumentError("tsb: alpha_d and alpha_p must be in (0, 1]"))
    yv = _imt_check(y, "tsb")
    if all(==(0), yv)
        return IntermittentModel(:tsb, yv, 0.0, 0.0, NaN, 0.0, alpha_d, NaN, alpha_p, 1,
                                 zeros(length(yv)), nothing)
    end
    d = _imt_demand(yv)
    p = Float64.(yv .!= 0)
    pl = _ses_level(p, alpha_p)
    dl = _ses_level(d, alpha_d)
    # in-sample: the probability path at every period, times the demand path
    # carried between arrivals
    n = length(yv)
    fit = fill(NaN, n)
    fp = p[1]; fd = NaN; zd = 0.0; k = 0
    for t in 2:n
        fp = t == 2 ? p[1] : alpha_p * p[t - 2] + (1 - alpha_p) * fp
        pt = alpha_p * p[t - 1] + (1 - alpha_p) * fp
        if yv[t - 1] != 0
            k += 1
            fd = k == 1 ? yv[t - 1] : alpha_d * zd + (1 - alpha_d) * fd
            zd = yv[t - 1]
        end
        dt = k == 0 ? 0.0 : alpha_d * zd + (1 - alpha_d) * fd
        fit[t] = pt * dt
    end
    return IntermittentModel(:tsb, yv, pl * dl, dl, NaN, pl, alpha_d, NaN, alpha_p, 1, fit,
                             _imt_class(yv))
end

"""
    adida(y; aggregation=nothing) -> IntermittentModel

The aggregate–disaggregate approach (Nikolopoulos et al. 2011): sum the
series into non-overlapping buckets as wide as the mean interval between
demands (`round` of it, half to even), forecast the bucket sums by simple
exponential smoothing with `alpha` chosen in `[0.1, 0.3]`, and divide by the
width. Buckets are aligned to the end of the series; an incomplete bucket
at the start is dropped. `aggregation` fixes the width instead.

The rounding rule matters when the mean interval is exactly a half:
`statsforecast` rounds half to even for the forecast but half away from
zero for its fitted values, so the same series gets two widths -- on a
series with mean interval 2.5, widths 2 and 3 give forecasts 20% apart.
This uses half to even throughout, matching `statsforecast`'s forecast.
"""
function adida(y; aggregation::Union{Nothing,Integer}=nothing)
    yv = _imt_check(y, "adida")
    if all(==(0), yv)
        return IntermittentModel(:adida, yv, 0.0, NaN, NaN, NaN, NaN, NaN, NaN, 1,
                                 zeros(length(yv)), nothing)
    end
    k = aggregation === nothing ? Int(round(mean(_imt_intervals(yv)))) : Int(aggregation)
    k >= 1 || throw(ArgumentError("adida: aggregation must be >= 1"))
    return IntermittentModel(:adida, yv, _imt_chunk_forecast(yv, k) / k, NaN, NaN, NaN, NaN, NaN,
                             NaN, k, aggregation === nothing ? _adida_fitted(yv) : nothing,
                             _imt_class(yv))
end

# Summation in the order of numpy's pairwise sum (sequential below 8 terms,
# eight interleaved accumulators up to 128, halves beyond), so that a mean of
# the per-level forecasts agrees with statsforecast's to the last bit -- with
# eight or more levels a plain left-to-right sum can differ by an ulp.
function _pairwise_sum(v::AbstractVector{Float64})
    n = length(v)
    if n < 8
        s = 0.0
        for x in v
            s += x
        end
        return s
    elseif n <= 128
        r = v[1:8]
        m = n - n % 8
        for i in 9:8:m, j in 0:7
            r[j + 1] += v[i + j]
        end
        s = ((r[1] + r[2]) + (r[3] + r[4])) + ((r[5] + r[6]) + (r[7] + r[8]))
        for i in (m + 1):n
            s += v[i]
        end
        return s
    else
        n2 = n ÷ 2
        n2 -= n2 % 8
        return _pairwise_sum(view(v, 1:n2)) + _pairwise_sum(view(v, (n2 + 1):n))
    end
end

"""
    imapa(y) -> IntermittentModel

Multiple temporal aggregation (Petropoulos & Kourentzes 2015): the
[`adida`](@ref) forecast at every bucket width from 1 to the rounded mean
interval, averaged. When the mean interval rounds to 1 it is ADIDA at width
1, exactly.
"""
function imapa(y)
    yv = _imt_check(y, "imapa")
    if all(==(0), yv)
        return IntermittentModel(:imapa, yv, 0.0, NaN, NaN, NaN, NaN, NaN, NaN, 1,
                                 zeros(length(yv)), nothing)
    end
    K = Int(round(mean(_imt_intervals(yv))))
    fc = _pairwise_sum([_imt_chunk_forecast(yv, k) / k for k in 1:K]) / K
    return IntermittentModel(:imapa, yv, fc, NaN, NaN, NaN, NaN, NaN, NaN, K, _imapa_fitted(yv),
                             _imt_class(yv))
end

"""
    forecast(m::IntermittentModel, h) -> IntermittentForecast

The flat point forecast, `h` times. No intervals -- see
[`IntermittentForecast`](@ref).
"""
function forecast(m::IntermittentModel, h::Integer)
    h >= 1 || throw(ArgumentError("forecast: h must be >= 1"))
    return IntermittentForecast(fill(m.forecast, h), Int(h), m.method)
end
StatsAPI.predict(m::IntermittentModel, h::Integer) = forecast(m, h)
StatsAPI.fitted(m::IntermittentModel) = m.fitted === nothing ? throw(ArgumentError(
    "fitted: not computed for $(m.method) (statsforecast recomputes the whole method per " *
    "prefix for it, and calls that 'very expensive')")) : copy(m.fitted)
StatsAPI.residuals(m::IntermittentModel) = m.y .- StatsAPI.fitted(m)
StatsAPI.nobs(m::IntermittentModel) = length(m.y)

const _IMT_NAME = Dict(:classic => "Croston (classic)", :sba => "Croston (SBA)",
                       :optimized => "Croston (optimised, alpha in [0.1, 0.3])", :tsb => "TSB",
                       :adida => "ADIDA", :imapa => "IMAPA")

function Base.show(io::IO, m::IntermittentModel)
    println(io, _IMT_NAME[m.method], ", n = ", length(m.y), ", ", count(!=(0), m.y), " demands")
    print(io, "  forecast per period: ", round(m.forecast; sigdigits=8))
    m.method in (:classic, :sba, :optimized) &&
        print(io, "\n  demand level ", round(m.demand_level; sigdigits=8), " (alpha ",
              round(m.alpha_demand; sigdigits=6), "), interval level ",
              round(m.interval_level; sigdigits=8), " (alpha ", round(m.alpha_interval; sigdigits=6), ")")
    m.method === :tsb && print(io, "\n  probability ", round(m.probability_level; sigdigits=8),
                               " x demand ", round(m.demand_level; sigdigits=8))
    m.method in (:adida, :imapa) && print(io, "\n  aggregation ",
                                          m.method === :adida ? "width " : "widths 1:", m.aggregation)
    m.classification === nothing ||
        print(io, "\n  ADI/CV² classification: ", m.classification === :sba ? "SBA" : "Croston")
end

function Base.show(io::IO, f::IntermittentForecast)
    println(io, _IMT_NAME[f.method], " forecast, ", f.horizon, " steps ahead (flat)")
    print(io, "  ", round(f.point[1]; sigdigits=8), " per period; no prediction intervals (no reference computes any)")
end

"""
    DemandIntervals

An intermittent series split into its non-zero `demands`, the `intervals`
between them (the first counts from the start of the series) and the
`positions` of the demands, built by [`demand_intervals`](@ref). Built from
a fitted model, it also carries the smoothed `demand_levels` and
`interval_levels` after each demand, and the model's `method` (otherwise
empty and `nothing`). `plot(d)` draws the decomposition.
"""
struct DemandIntervals
    y::Vector{Float64}
    demands::Vector{Float64}
    intervals::Vector{Int}
    positions::Vector{Int}
    demand_levels::Vector{Float64}
    interval_levels::Vector{Float64}
    method::Union{Nothing,Symbol}
end

"""
    demand_intervals(y) -> DemandIntervals

The decomposition Croston's method smooths: the non-zero demands and the
intervals between them. The intervals sum to the position of the last
demand, and the demands to the series total.

```jldoctest
julia> using TSAnalytics

julia> d = demand_intervals([0, 0, 3, 0, 2, 0]);

julia> d.demands, d.intervals
([3.0, 2.0], [3, 2])
```
"""
function demand_intervals(y)
    yv = collect(Float64, tsvalues(y))
    any(<(0), yv) && throw(ArgumentError("demand_intervals: demand cannot be negative"))
    pos = findall(!=(0), yv)
    return DemandIntervals(yv, yv[pos], diff(vcat(0, pos)), pos, Float64[], Float64[], nothing)
end

"""
    ADICV2Result

The Syntetos–Boylan–Croston classification of an intermittent series,
from [`adi_cv2`](@ref): the average demand interval `adi`, the squared
coefficient of variation of the non-zero demands `cv2`, the recommended
`method` (`:sba` or `:croston`), and the `quadrant`.
"""
struct ADICV2Result
    adi::Float64
    cv2::Float64
    method::Symbol
    quadrant::Symbol
    adi_cutoff::Float64
    cv2_cutoff::Float64
end

"""
    adi_cv2(y; adi_cutoff=1.32, cv2_cutoff=0.49) -> ADICV2Result

Classify an intermittent series by its average demand interval (ADI, the
mean of the intervals [`demand_intervals`](@ref) returns) and the squared
coefficient of variation of its non-zero demands (CV², sample standard
deviation). Above either cutoff SBA is recommended, otherwise Croston:
the rule of R's `tsintermittent::idclass`, whose cutoffs 1.32 and 0.49 come
from Syntetos, Boylan & Croston (2005). Neither `forecast` nor
`statsforecast` classifies at all.

The four quadrant names -- `:smooth` (both below), `:intermittent` (ADI
above), `:erratic` (CV² above), `:lumpy` (both above) -- are the paper's
vocabulary, not `idclass`'s, which decides only SBA versus Croston.

```jldoctest
julia> using TSAnalytics

julia> c = adi_cv2([0, 0, 3, 0, 0, 0, 2, 0, 4, 0]);

julia> round(c.adi, digits = 4), c.method, c.quadrant
(3.0, :sba, :intermittent)
```
"""
function adi_cv2(y; adi_cutoff::Real=1.32, cv2_cutoff::Real=0.49)
    d = demand_intervals(y)
    length(d.demands) >= 2 || throw(ArgumentError(
        "adi_cv2: needs at least two demands (CV² of one value is undefined)"))
    adi = mean(d.intervals)
    cv2 = (std(d.demands) / mean(d.demands))^2
    hi_a, hi_c = adi > adi_cutoff, cv2 > cv2_cutoff
    quadrant = hi_a ? (hi_c ? :lumpy : :intermittent) : (hi_c ? :erratic : :smooth)
    return ADICV2Result(adi, cv2, (hi_a || hi_c) ? :sba : :croston, quadrant,
                        Float64(adi_cutoff), Float64(cv2_cutoff))
end

function Base.show(io::IO, r::ADICV2Result)
    print(io, "ADI = ", round(r.adi; sigdigits=6), ", CV² = ", round(r.cv2; sigdigits=6),
          " -> ", r.quadrant, ", use ", r.method === :sba ? "SBA" : "Croston",
          " (cutoffs ", r.adi_cutoff, ", ", r.cv2_cutoff, ")")
end
