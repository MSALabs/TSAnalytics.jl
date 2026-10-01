export ar_yw, spec_ar, ARYWFit

"""
    ARYWFit

Result of [`ar_yw`](@ref). Fields:

- `ar`: the fitted AR coefficients, length `order`.
- `order`: the order used — chosen by AIC unless one was given.
- `var_pred`: the innovation variance, with R's small-sample adjustment
  `v * n/(n - (order+1))` applied.
- `aic_by_order`: AIC at each order `0:order_max`, **shifted so the
  minimum is `0`**, matching R's `ar()` `aic` field. The selected order is where
  this is zero. Named this way rather than `aic` because that is
  `StatsAPI.aic`, and a field of the same name would shadow it.
- `partialacf`: the partial autocorrelations at orders `1:order_max`.
- `n`: observations used.
- `order_max`: the largest order considered.
"""
struct ARYWFit
    ar::Vector{Float64}
    order::Int
    var_pred::Float64
    aic_by_order::Vector{Float64}
    partialacf::Vector{Float64}
    n::Int
    order_max::Int
end

function Base.show(io::IO, f::ARYWFit)
    print(io, "AR($(f.order)) by Yule-Walker, n=$(f.n)")
    f.order > 0 && print(io, "\n  coefficients : ", round.(f.ar, digits=6))
    print(io, "\n  var_pred     : ", round(f.var_pred, digits=6))
    print(io, "\n  order        : ", f.order,
           f.order_max > 0 ? "  (chosen from 0:$(f.order_max) by AIC)" : "")
end

"""
    ar_yw(y; order=nothing, order_max=nothing, demean=true) -> ARYWFit

Fit an autoregression by **Yule-Walker**: solve the sample
autocovariances for the AR coefficients via the Durbin-Levinson
recursion, rather than by least squares or maximum likelihood.

With `order=nothing` the order is selected by **AIC** over
`0:order_max`, exactly as R's `ar()` does, using

    AIC(m) = n*log(v_m) + 2m

where `v_m` is the order-`m` prediction error variance. `order_max`
defaults to `min(n-1, floor(10*log10(n)))`, R's own default.

This is the estimator behind [`spec_ar`](@ref), and the classical
counterpart to [`arx`](@ref), which fits by conditional least squares
instead. The two differ, and not only numerically:

| | [`ar_yw`](@ref) | [`arx`](@ref) |
|---|---|---|
| Estimator | Yule-Walker, from autocovariances | Conditional least squares |
| Stationarity | **Guaranteed** by construction | Not guaranteed |
| Exogenous regressors | No | Yes |
| Order selection | AIC, built in | Caller's job |
| Matches | R's `ar()` / `ar.yw()` | `statsmodels`' `AutoReg` |

Yule-Walker always returns a stationary fit, which is why it is the
natural estimator for a spectral density — a non-stationary AR has no
spectrum to compute. The cost is bias, which is appreciable near a unit
root.

`demean=true` subtracts the sample mean first, matching R. The
autocovariances use the `1/n` denominator, also matching R's
`acf(type="covariance")`.

Reads from the shipped AR(2) fixture rather than a seeded draw, because
Julia's RNG stream is not stable across versions and a doctest that
depends on it fails on whichever version is not the one it was written
on.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> z = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                                "verification", "arspectral", "ar2.csv")));

julia> f = ar_yw(z);

julia> f.order
2

julia> round.(f.ar, digits=6)
2-element Vector{Float64}:
  0.730235
 -0.509435
```

Those are R's `ar(z)` coefficients to all six digits — see
`test/verification/arspectral/`.

See also [`spec_ar`](@ref), [`arx`](@ref), [`pacf`](@ref).
"""
function ar_yw(y; order::Union{Nothing,Integer}=nothing,
                order_max::Union{Nothing,Integer}=nothing, demean::Bool=true)
    yv = Float64.(collect(tsvalues(y)))
    any(isnan, yv) && throw(ArgumentError("ar_yw: NaN present (no missing-data policy implemented yet)"))
    n = length(yv)
    n >= 2 || throw(ArgumentError("ar_yw: need at least 2 observations, got $n"))

    omax = if order_max !== nothing
        Int(order_max)
    elseif order !== nothing
        Int(order)
    else
        min(n - 1, floor(Int, 10 * log10(n)))     # R's own default
    end
    omax >= 1 || throw(ArgumentError("ar_yw: order_max must be >= 1, got $omax"))
    omax < n || throw(ArgumentError("ar_yw: order_max must be < n (got $omax for n=$n)"))
    if order !== nothing
        order >= 0 || throw(ArgumentError("ar_yw: order must be >= 0, got $order"))
        order <= omax || throw(ArgumentError("ar_yw: order=$order exceeds order_max=$omax"))
    end

    acov = _acovf(yv, omax; demean=demean, adjusted=false)
    acov[1] > 0 || throw(ArgumentError("ar_yw: the series has zero variance"))
    dl = _durbin_levinson_full(acov, omax)

    # AIC(m) = n*log(v_m) + 2m over m = 0:omax, as R's ar.yw computes it
    aics = [n * log(dl.v[m+1]) + 2 * m for m in 0:omax]
    sel = order === nothing ? (argmin(aics) - 1) : Int(order)

    ar = sel == 0 ? Float64[] : collect(dl.phi[sel, 1:sel])
    # R: var.pred <- EA * n.obs/(n.obs - (m + 1))
    var_pred = dl.v[sel+1] * n / (n - (sel + 1))

    return ARYWFit(ar, sel, var_pred, aics .- minimum(aics),
                    collect(dl.pacf), n, omax)
end

"""
    spec_ar(y; n_freq=500, order=nothing, order_max=nothing, frequency=1) -> PeriodogramResult

Spectral density estimated **parametrically**, from a fitted
autoregression rather than from the periodogram:

    S(f) = var_pred / (xfreq * |1 - sum_k phi_k exp(-2*pi*i*f*k)|^2)

evaluated on `n_freq` points spanning `0` to `0.5` (scaled by
`frequency`). The AR fit comes from [`ar_yw`](@ref), with its order
chosen by AIC unless you pass one. Matches R's `stats::spec.ar`.

## Why a parametric spectrum

A raw [`periodogram`](@ref) is a noisy estimator no matter how long the
series: its variance does not fall with `n`, which is why
[`spectral_density`](@ref) smooths it with a Daniell kernel. That
smoothing is a bias-variance trade you set by hand through the span.

This takes the other route — assume the series is an AR of some order,
estimate it, and read the spectrum off the fitted model. The result is
**smooth by construction** and needs no span. The assumption is doing
the work the smoothing otherwise would, and the order selection decides
how much structure the spectrum is allowed to have.

Neither is strictly better. The parametric estimate is sharper when the
AR assumption is close to right, and will confidently show a peak that
is an artefact of the chosen order when it is not. Reading both is the
usual advice, and cheap:

```@example
using TSAnalytics, Random
Random.seed!(4)
z = zeros(500)
for t in 3:500
    z[t] = 0.75z[t-1] - 0.4z[t-2] + randn()
end
pa = spec_ar(z; n_freq=64)
pg = periodogram(z)
println("AR order chosen    : ", ar_yw(z).order)
println("AR peak at freq    : ", round(pa.freq[argmax(pa.spec)], digits=4))
println("periodogram peak at: ", round(pg.freq[argmax(pg.spec)], digits=4))
```

Returns a [`PeriodogramResult`](@ref) tagged `kind = :spec_ar`, so the
existing plot recipe serves it. Its `df` and `bandwidth` are `NaN`: both
are Daniell-smoothing quantities with no analogue here, since the
smoothness comes from the AR assumption rather than a kernel. R's
`spec.ar` does not return them either.

`frequency` scales the returned `freq` axis, as R's `xfreq` does — pass
`12` for monthly data to read cycles per year rather than per
observation. It also divides the density, again matching R.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> z = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                                "verification", "arspectral", "ar2.csv")));

julia> s = spec_ar(z; n_freq=9);

julia> length(s.freq) == 9 && s.kind == :spec_ar
true

julia> round(s.spec[1], digits=8)
1.66789975
```

Verified against real R `spec.ar` — see `test/verification/arspectral/`.

See also [`periodogram`](@ref), [`spectral_density`](@ref), [`ar_yw`](@ref).
"""
function spec_ar(y; n_freq::Integer=500, order::Union{Nothing,Integer}=nothing,
                  order_max::Union{Nothing,Integer}=nothing, frequency::Real=1)
    n_freq >= 2 || throw(ArgumentError("spec_ar: n_freq must be >= 2, got $n_freq"))
    frequency > 0 || throw(ArgumentError("spec_ar: frequency must be > 0, got $frequency"))

    fit = ar_yw(y; order=order, order_max=order_max)
    freqs = collect(range(0.0, 0.5; length=n_freq))

    spec = Vector{Float64}(undef, n_freq)
    if fit.order == 0
        fill!(spec, fit.var_pred / frequency)
    else
        @inbounds for (i, f) in enumerate(freqs)
            cs = 0.0
            sn = 0.0
            for k in 1:fit.order
                cs += fit.ar[k] * cospi(2 * f * k)
                sn += fit.ar[k] * sinpi(2 * f * k)
            end
            spec[i] = fit.var_pred / (frequency * ((1 - cs)^2 + sn^2))
        end
    end

    # df and bandwidth are Daniell-smoothing quantities and have no analogue
    # here -- the smoothness comes from the AR assumption, not from a kernel.
    # R's spec.ar does not return them either; NaN says "not applicable"
    # rather than inviting a reader to use a zero.
    return PeriodogramResult(freqs .* frequency, spec, NaN, NaN, :spec_ar)
end
