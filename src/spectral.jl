export periodogram, spectral_density

"_nextn(n, factors=(2,3,5)) -> Int -- smallest m >= n whose prime
factorization uses only the given factors, matching R's own `nextn`
exactly (confirmed by direct execution across a range of values --
`7->8, 97->100, 101->108, 151->160`, etc. -- not just its documented
description, since R's own implementation is C, not inspectable
Julia-side)."
function _nextn(n::Integer, factors::NTuple{K,<:Integer}=(2, 3, 5)) where {K}
    m = n
    while true
        k = m
        for f in factors
            while k % f == 0
                k ÷= f
            end
        end
        k == 1 && return m
        m += 1
    end
end

"_spec_taper(x, p) -- R's `stats:::spec.taper`: split-cosine-bell
taper, fraction `p` (`0 <= p <= 0.5`) tapered at *each* end."
function _spec_taper(x::AbstractVector{<:Real}, p::Real)
    0 <= p <= 0.5 || throw(ArgumentError("taper p must be between 0 and 0.5, got $p"))
    n = length(x)
    m = floor(Int, n * p)
    m == 0 && return Float64.(x)
    w = 0.5 .* (1 .- cos.(π .* (1:2:(2m - 1)) ./ (2m)))
    return vcat(w, ones(n - 2m), reverse(w)) .* x
end

"""
    periodogram(x; taper=0.0, detrend=true, demean=false, pad=0, fast=true, xfreq=1.0)
        -> (freq=Vector{Float64}, spec=Vector{Float64}, df, bandwidth)

Raw periodogram estimate of the spectral density, matching R base's
`stats::spec.pgram(x, spans=NULL, ...)` exactly (algorithm read
directly from R's own real source -- `stats:::spec.pgram`/
`stats:::spec.taper`/`stats::nextn` -- this session, not reconstructed
from documentation).

**`taper` defaults to `0.0` here, not R's own `0.1` default** --
matching Shumway & Stoffer's own companion package `astsa::mvspec`
instead, which *deliberately* overrides base R's tapering default
(confirmed directly from its documentation: *"the script does not taper
by default; this forces the user to do 'conscious tapering'"*). Two
real, different design philosophies from two authoritative sources on
the same computation -- this project follows the more actively-taught
convention and documents the divergence explicitly rather than picking
one silently. Pass `taper=0.1` explicitly to match base R's own default
behavior.

- **`detrend=true`** (R's own default): removes a linear trend (which
  also removes the mean) via the closed-form OLS-on-centered-time
  formula R itself uses, not a general regression call -- `t` is a
  known, deterministic centered index, so the fit reduces to
  `x - mean(x) - (sum(x.*t)/sum(t.^2))*t` directly.
- **`demean`**: only consulted when `detrend=false`; subtracts the mean
  alone.
- **`pad`**: fraction of additional zeros appended (`pad=1` doubles the
  series length with zeros) *before* any `fast`-driven padding.
- **`fast=true`** (R's own default): further zero-pads to `_nextn`
  of the (possibly already-`pad`-padded) length -- the smallest
  5-smooth number at least that large -- matching R's own default
  FFT-length choice exactly, not just computing the FFT at the raw
  padded length.
- **`xfreq`**: the sampling frequency (R's own `frequency(x)` for a
  `ts` object; `1.0` for a plain vector, this project's own
  container-agnostic convention -- see [`tsvalues`](@ref)).

The DC term (frequency `0`) is excluded from the returned `freq`/`spec`
entirely, matching R's own convention exactly (`pgram[2:(Nspec+1),...]`
in the real source).

Verified directly against real base R `spec.pgram` on a series with a
known, injected period-12 component -- peak frequency recovered exactly
at `1/12` (R: `0.08333333333`), spectrum values matching to full
displayed precision (`5.787370716, 1.375818839, ...`) -- see
`test/verification/spectral/`.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(42); n = 96; t = 1:n;

julia> x = 5 .* sin.(2π .* t ./ 12) .+ 3 .* cos.(2π .* t ./ 12) .+ randn(n) .* 0.5;

julia> result = periodogram(x; taper=0.1);

julia> peak_freq = result.freq[argmax(result.spec)];

julia> isapprox(peak_freq, 1/12; atol=1e-6)
true
```
"""
function periodogram(x; taper::Real=0.0, detrend::Bool=true, demean::Bool=false,
                      pad::Real=0, fast::Bool=true, xfreq::Real=1.0)
    xv = Float64.(collect(tsvalues(x)))
    N0 = length(xv)
    N0 == 0 && throw(ArgumentError("periodogram: x must be non-empty"))

    if detrend
        t = (1:N0) .- (N0 + 1) / 2
        sumt2 = N0 * (N0^2 - 1) / 12
        mu = sum(xv) / N0
        xv = xv .- mu .- (sum(xv .* t) / sumt2) .* t
    elseif demean
        xv = xv .- sum(xv) / N0
    end

    xv = _spec_taper(xv, taper)
    u2 = 1 - (5 / 8) * taper * 2
    u4 = 1 - (93 / 128) * taper * 2

    N = N0
    if pad > 0
        xv = vcat(xv, zeros(round(Int, N0 * pad)))
        N = length(xv)
    end
    NewN = fast ? _nextn(N) : N
    xv = vcat(xv, zeros(NewN - N))
    N = length(xv)

    Nspec = N ÷ 2
    Nspec >= 1 || throw(ArgumentError("periodogram: series too short after padding to produce any spectral estimate"))
    freq = (1:Nspec) .* (xfreq / N)

    xfft = fft(xv)
    pgram_full = abs2.(xfft) ./ (N0 * xfreq)
    pgram_full[1] = 0.5 * (pgram_full[2] + pgram_full[N])

    spec = pgram_full[2:(Nspec+1)] ./ u2

    df = (2 / (u4 / u2^2)) * (N0 / N)
    bandwidth = sqrt(1 / 12) * xfreq / N

    return (freq=freq, spec=spec, df=df, bandwidth=bandwidth)
end

"_modified_daniell_kernel(m::Integer) -> Vector{Float64} -- symmetric
kernel of length `2m+1` (indices `-m` to `m`, stored in natural order),
matching R's `kernel(\"modified.daniell\", m)` exactly: interior
coefficients `1/(2m)`, the two endpoints `1/(4m)` (half-weighted, the
\"modified\" part of the name)."
function _modified_daniell_kernel(m::Integer)
    m == 0 && return [1.0]
    c = fill(1 / (2m), 2m + 1)
    c[1] = c[end] = 1 / (4m)
    return c
end

"_kernel_convolve(k1, k2) -- composes two symmetric kernels via full
convolution, matching R's `kernapply.tskernel` (how a multi-element
`spans` vector builds a single combined Daniell kernel, e.g.
`spans=(7,7)` convolves two `m=3` modified-Daniell kernels)."
function _kernel_convolve(k1::Vector{Float64}, k2::Vector{Float64})
    n1, n2 = length(k1), length(k2)
    out = zeros(n1 + n2 - 1)
    for i in 1:n1, j in 1:n2
        out[i+j-1] += k1[i] * k2[j]
    end
    return out
end

"_modified_daniell_kernel(ms::AbstractVector{<:Integer}) -- composes one
kernel per element of `ms`, where each element is already a half-width
(the caller applies `spans .÷ 2` once, matching `spec.pgram`'s own
`kernel(\"modified.daniell\", spans %/% 2)` -- this method must NOT
divide by 2 again)."
function _modified_daniell_kernel(ms::AbstractVector{<:Integer})
    k = _modified_daniell_kernel(ms[1])
    for i in 2:length(ms)
        k = _kernel_convolve(k, _modified_daniell_kernel(ms[i]))
    end
    return k
end

"_kernel_df/_kernel_bandwidth -- R's `stats:::df.kernel`/
`stats:::bandwidth.kernel`, applied to a full symmetric kernel vector
(length `2m+1`) rather than R's own half-stored `tskernel` object."
_kernel_df(k::Vector{Float64}) = 2 / sum(k .^ 2)
function _kernel_bandwidth(k::Vector{Float64})
    m = (length(k) - 1) ÷ 2
    idx = -m:m
    return sqrt(sum((1 / 12 .+ idx .^ 2) .* k))
end

"_kernapply_circular(x, k) -- circular convolution of `x` with a
symmetric kernel `k` (length `2m+1`, indices `-m..m`), matching R's
`kernapply(x, kernel, circular=TRUE)` -- used to smooth the raw
periodogram across frequency, wrapping at the boundary. `k`'s symmetry
(`k[m+1+i] == k[m+1-i]`, always true for a Daniell-family kernel) means
the lag sign convention doesn't affect the result, so this doesn't need
to replicate R's own FFT-shifted-array convolution mechanics exactly to
agree with it -- verified directly against real R output regardless
(`test/verification/spectral/`)."
function _kernapply_circular(x::AbstractVector{<:Real}, k::Vector{Float64})
    n = length(x)
    m = (length(k) - 1) ÷ 2
    y = Vector{Float64}(undef, n)
    for i in 1:n
        s = 0.0
        for j in 1:length(k)
            lag = j - m - 1
            s += k[j] * x[mod1(i - lag, n)]
        end
        y[i] = s
    end
    return y
end

"""
    spectral_density(x, spans; taper=0.0, detrend=true, demean=false,
                      pad=0, fast=true, xfreq=1.0)
        -> (freq=Vector{Float64}, spec=Vector{Float64}, df, bandwidth)

Smoothed periodogram via modified-Daniell kernel smoothing, matching R
base's `stats::spec.pgram(x, spans=spans, ...)` exactly -- the same
algorithm [`periodogram`](@ref) implements, plus circular kernel
smoothing of the raw periodogram across frequency (Durbin & Koopman's
and Shumway & Stoffer's own recommended way to reduce a raw
periodogram's characteristic high variance).

**`spans`**: a vector of odd-ish span widths, matching R's own
convention directly (e.g. `spans=[7,7]`, the exact usage pattern found
in `astsa::mvspec(soi, spans=c(7,7), ...)`) -- internally converted to
kernel half-widths via `spans .÷ 2` (matching `spec.pgram`'s own
`kernel("modified.daniell", spans %/% 2)` exactly), then composed into
one kernel via repeated convolution when `spans` has more than one
element.

All other keywords carry [`periodogram`](@ref)'s own exact meaning.
`df`/`bandwidth` are computed from the composed kernel (matching R's
`df.kernel`/`bandwidth.kernel`, further corrected by the same
taper/padding factors `periodogram` itself applies) -- the values a
caller needs to construct the standard confidence intervals both R and
the reference textbooks use for a spectral estimate.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(42); n = 96; t = 1:n;

julia> x = 5 .* sin.(2π .* t ./ 12) .+ 3 .* cos.(2π .* t ./ 12) .+ randn(n) .* 0.5;

julia> result = spectral_density(x, [3, 3]; taper=0.1);

julia> peak_freq = result.freq[argmax(result.spec)];

julia> isapprox(peak_freq, 1/12; atol=1e-6)
true
```
"""
function spectral_density(x, spans::AbstractVector{<:Integer}; taper::Real=0.0,
                           detrend::Bool=true, demean::Bool=false, pad::Real=0,
                           fast::Bool=true, xfreq::Real=1.0)
    all(>=(1), spans) || throw(ArgumentError("spectral_density: every span must be >= 1"))

    xv = Float64.(collect(tsvalues(x)))
    N0 = length(xv)
    N0 == 0 && throw(ArgumentError("spectral_density: x must be non-empty"))

    if detrend
        t = (1:N0) .- (N0 + 1) / 2
        sumt2 = N0 * (N0^2 - 1) / 12
        mu = sum(xv) / N0
        xv = xv .- mu .- (sum(xv .* t) / sumt2) .* t
    elseif demean
        xv = xv .- sum(xv) / N0
    end

    xv = _spec_taper(xv, taper)
    u2 = 1 - (5 / 8) * taper * 2
    u4 = 1 - (93 / 128) * taper * 2

    N = N0
    if pad > 0
        xv = vcat(xv, zeros(round(Int, N0 * pad)))
        N = length(xv)
    end
    NewN = fast ? _nextn(N) : N
    xv = vcat(xv, zeros(NewN - N))
    N = length(xv)

    Nspec = N ÷ 2
    Nspec >= 1 || throw(ArgumentError("spectral_density: series too short after padding to produce any spectral estimate"))
    freq = (1:Nspec) .* (xfreq / N)

    xfft = fft(xv)
    pgram_full = abs2.(xfft) ./ (N0 * xfreq)
    pgram_full[1] = 0.5 * (pgram_full[2] + pgram_full[N])

    kern = _modified_daniell_kernel(collect(spans) .÷ 2)
    length(kern) - 1 <= 2 * (N - 1) || throw(ArgumentError("spectral_density: spans too wide for series length"))
    pgram_smoothed = _kernapply_circular(pgram_full, kern)

    spec = pgram_smoothed[2:(Nspec+1)] ./ u2

    df = (_kernel_df(kern) / (u4 / u2^2)) * (N0 / N)
    bandwidth = _kernel_bandwidth(kern) * xfreq / N

    return (freq=freq, spec=spec, df=df, bandwidth=bandwidth)
end
