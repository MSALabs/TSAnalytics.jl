export fourier_terms

"""
    fourier_terms(n, period, K; h=nothing) -> (terms, names)

Fourier (harmonic) regressors for a seasonal pattern: `K` sine/cosine
pairs at frequencies `1/period, 2/period, ..., K/period`, evaluated at
times `1:n`, or at `n+1:n+h` when `h` is given.

Returns a named tuple: `terms`, an `n × 2K` matrix (see the Nyquist note
below for when it is narrower), and `names`, the matching column labels
`["S1-12", "C1-12", "S2-12", ...]` in R's `forecast::fourier` format.

`period` and `K` may each be a vector, for several seasonal periods at
once — the analogue of R's `msts`. Frequencies duplicated across periods
are kept only once.

## Why reach for these instead of seasonal dummies

Seasonal dummies cost `period - 1` parameters. For monthly data that is
eleven, which is tolerable; for weekly data it is **fifty-one**, and for
half-hourly data with a daily cycle it is forty-seven. Fourier terms cost
`2K` for whatever `K` you choose, so a large period stops being a reason
to give up on modelling the season at all.

The trade is that they impose a **smooth** seasonal shape. `K` controls
how wiggly: `K=1` is a single sine wave, higher `K` admits sharper
features, and at `K = period/2` the two parameterisations span the same
space and cost the same. Choose `K` by information criterion, which is
what both references do.

They are also deterministic, so the `h` form extrapolates exactly — the
pattern repeats with the period and there is no uncertainty in what next
January's regressor is. That is what makes them usable with
[`fit_arimax`](@ref), which needs future regressor values to forecast.

!!! note "At `K = period/2` the sine column is dropped"
    The highest resolvable harmonic has `sin(2π·t/2) = sin(π t) = 0` for
    every integer `t` — a column of zeros, carrying no information and
    making the design matrix singular. It is dropped, so `K = 6` on
    monthly data gives **11 columns, not 12**, and `names` ends
    `"C6-12"` with no `"S6-12"`.

    R does the same, by the same reasoning. It is the one place the
    column count is not `2K`, and the reason to read `names` rather than
    assume positions.

`K > period/2` is rejected: those frequencies alias onto lower ones and
carry nothing new.

```jldoctest
julia> using TSAnalytics

julia> f = fourier_terms(48, 12, 2);

julia> f.names
4-element Vector{String}:
 "S1-12"
 "C1-12"
 "S2-12"
 "C2-12"

julia> round.(f.terms[1, :], digits=10)
4-element Vector{Float64}:
 0.5
 0.8660254038
 0.8660254038
 0.5

julia> size(fourier_terms(48, 12, 6).terms)   # Nyquist sine dropped
(48, 11)
```

Verified against real `forecast::fourier` (R 4.6.0) on `K=2`/`m=12`,
`K=6`/`m=12` (the Nyquist case), `K=1`/`m=4`, and the `h` form — see
`test/verification/fourier/fourier.R`.

See also [`fit_arimax`](@ref), [`mstl_decompose`](@ref), [`arx`](@ref).
"""
function fourier_terms(n::Integer, period, K; h::Union{Nothing,Integer}=nothing)
    periods = period isa Real ? [Float64(period)] : Float64.(collect(period))
    Ks = K isa Integer ? [Int(K)] : Int.(collect(K))
    length(periods) == length(Ks) || throw(ArgumentError(
        "fourier_terms: got $(length(periods)) period(s) and $(length(Ks)) order(s) -- " *
        "they must match"))
    n >= 1 || throw(ArgumentError("fourier_terms: n must be >= 1, got $n"))
    all(>(1), periods) || throw(ArgumentError("fourier_terms: every period must be > 1"))
    all(>=(0), Ks) || throw(ArgumentError("fourier_terms: every K must be >= 0"))
    any(2 .* Ks .> periods) && throw(ArgumentError(
        "fourier_terms: K must not exceed period/2 -- harmonics above that alias " *
        "onto lower ones and carry no new information (got K=$Ks for period=$periods)"))

    times = if h === nothing
        collect(1:n)
    else
        h >= 1 || throw(ArgumentError("fourier_terms: h must be >= 1, got $h"))
        collect((n+1):(n+h))
    end

    # frequencies k/m, deduplicated across periods but keeping first appearance
    freqs = Float64[]
    labels = String[]
    for j in eachindex(periods)
        Ks[j] == 0 && continue
        for k in 1:Ks[j]
            f = k / periods[j]
            any(≈(f), freqs) && continue      # already covered by an earlier period
            push!(freqs, f)
            push!(labels, "S$k-$(round(Int, periods[j]))")
            push!(labels, "C$k-$(round(Int, periods[j]))")
        end
    end
    isempty(freqs) && return (terms=Matrix{Float64}(undef, length(times), 0), names=String[])

    cols = Vector{Vector{Float64}}()
    names = String[]
    for (j, f) in enumerate(freqs)
        # at the Nyquist harmonic 2f is an integer, so sin(2*pi*f*t) = sin(pi*k*t)
        # is identically zero for integer t -- a zero column, dropped
        if abs(2f - round(2f)) > eps(Float64)
            push!(cols, [sinpi(2f * t) for t in times])
            push!(names, labels[2j-1])
        end
        push!(cols, [cospi(2f * t) for t in times])
        push!(names, labels[2j])
    end

    return (terms=reduce(hcat, cols), names=names)
end
