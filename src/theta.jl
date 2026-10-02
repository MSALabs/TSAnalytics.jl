export fit_theta, ThetaModel

"""
    ThetaModel

A fitted Theta model, returned by [`fit_theta`](@ref).

## Fields

- `alpha`: the SES smoothing parameter, fitted on the deseasonalised series.
- `b0`: the OLS slope of the deseasonalised series on `0:(n-1)`. The drift
  the forecast actually applies is `(1 - 1/theta) * b0`, so `b0/2` for the
  standard `theta = 2`.
- `level`: the final SES level — the forecast before drift and reseasonalising.
- `theta`: the theta parameter, `2.0` for the standard method.
- `figure`: the one-period multiplicative seasonal figure (length `period`),
  empty when the series was not deseasonalised.
- `deseasonalized`: whether the seasonal adjustment was applied.
- `seasonal_statistic`: the seasonality test statistic, `|r_m| / se(r_m)`.
  Reported whether or not the test gated the decision, and `NaN` when
  `period == 1`.
- `fitted`, `resid`, `sse`: one-step SES fitted values on the original scale,
  the residuals against them (`resid == y - fitted`, always), and their sum of
  squares. **These exclude the drift term**, matching R's `thetaf`; see
  [`fit_theta`](@ref).
- `sse_adjusted`: the same sum of squares on the **deseasonalised** scale,
  which `sigma2` is built from. Equal to `sse` when no adjustment was applied.
- `sigma2`: the residual variance on the deseasonalised scale,
  `sse_adjusted/n`. The forecast's interval scales it back up by the seasonal
  factor; see [`fit_theta`](@ref) on why.
- `nobs`, `period`.
"""
struct ThetaModel
    alpha::Float64
    b0::Float64
    level::Float64
    theta::Float64
    figure::Vector{Float64}
    deseasonalized::Bool
    seasonal_statistic::Float64
    fitted::Vector{Float64}
    resid::Vector{Float64}
    sse::Float64
    sse_adjusted::Float64
    sigma2::Float64
    nobs::Int
    period::Int
end

function Base.show(io::IO, m::ThetaModel)
    print(io, "Theta(", round(m.theta, digits=4), ")  n=", m.nobs)
    m.period > 1 && print(io, "  period=", m.period)
    println(io)
    print(io, "  alpha = ", round(m.alpha, digits=6),
              "   drift = ", round((1 - 1 / m.theta) * m.b0, digits=6))
    println(io, "   level = ", round(m.level, digits=4))
    if m.period > 1
        print(io, "  seasonal: ", m.deseasonalized ? "adjusted" : "not adjusted",
                  " (statistic ", round(m.seasonal_statistic, digits=4),
                  " vs ", round(_confidence_z(0.10), digits=4), ")")
        println(io)
    end
    print(io, "  SSE = ", round(m.sse, digits=4))
    m.deseasonalized && print(io, "   SSE(adj) = ", round(m.sse_adjusted, digits=4))
    print(io, "   sigma2 = ", round(m.sigma2, digits=5))
end

"""
    _theta_seasonal_statistic(y, period) -> Float64

The Theta method's seasonality test statistic, `|r_m| / se(r_m)`, where
`se(r_m)` is **Bartlett's large-lag standard error** under the hypothesis
that the process is `MA(m-1)`:

    se(r_m) = sqrt((1 + 2*sum(r[1:m-1].^2)) / n)

Compared against `_confidence_z(0.10)` (`1.6449`, the one-sided 95% normal
quantile), which is what R's `thetaf` does.

!!! note "`statsmodels` drops the factor of 2"
    Its `ThetaModel._test_seasonality` computes
    `n*r_m^2 / (1 + sum(r[1:m-1].^2))` against `1.645^2`, which is the same
    comparison with `1 + sum(r^2)` in place of `1 + 2*sum(r^2)`. Bartlett's
    formula carries the 2, so `statsmodels`' test is **liberal** — it
    declares seasonality slightly more often than it should. Verified: on
    the bundled fixture R's statistic is `4.0883478370` and the
    factor-of-2-free version is larger.
"""
function _theta_seasonal_statistic(y::AbstractVector{<:Real}, period::Integer)
    n = length(y)
    r = acf(y, 1:period).values
    se = sqrt((1 + 2 * sum(abs2, view(r, 1:(period - 1)))) / n)
    return abs(r[period]) / se
end

"""
    fit_theta(y, period=1; theta=2.0, deseasonalize=true, seasonal_test=true,
              alpha=nothing) -> ThetaModel

Fit the **Theta method** of Assimakopoulos & Nikolopoulos (2000) — the
method that won the M3 forecasting competition.

The idea is to fit two lines to the series after rescaling its local
curvature by a factor `theta`: `theta = 0` flattens it to the linear
regression line, `theta = 2` doubles the curvature. Forecasting each and
averaging gives, as Hyndman & Billah (2003) showed, something much simpler
than the original description suggests: **simple exponential smoothing
plus half the regression slope as drift**. That equivalence is what is
implemented here, because it is exact, not an approximation.

    yhat[n+h] = level + (1 - 1/theta) * b0 * (h - 1 + (1 - (1-alpha)^n)/alpha)

with `level` and `alpha` from SES and `b0` the OLS slope on `0:(n-1)`. For
the standard `theta = 2` the drift weight is `1/2`.

## Seasonality

With `period > 1` the series is seasonally adjusted first, by classical
**multiplicative** decomposition ([`classical_decompose`](@ref)), and the
forecast is reseasonalised afterwards. `seasonal_test=true` gates that on
Bartlett's test of the lag-`period` autocorrelation; `seasonal_test=false`
always adjusts. `deseasonalize=false` skips it entirely.

The series must be strictly positive to be adjusted multiplicatively; a
series that is not is rejected rather than silently left unadjusted.

## `fitted` excludes the drift

`fitted` is the one-step SES fitted value, reseasonalised — **not** the
model's own one-step forecast, which would include the drift term. This
matches R's `thetaf` exactly (to `1e-6`), and keeps [`accuracy`](@ref) on
these residuals comparable with R's.

!!! warning "R's `thetaf` returns `fitted` and `residuals` on different scales"
    Its `fitted` is reseasonalised, as above. Its `residuals` are the SES
    model's, on the **deseasonalised** scale, so on a seasonal series
    `y - fitted` and `residuals` are two different vectors: on the bundled
    fixture their sums of squares are `1209.29841334` and `1315.21211028`,
    9% apart. Both are reproduced here — `sse` is the first and
    `sse_adjusted` the second — but they are kept clearly labelled rather
    than silently interchangeable, and `resid` is always `y - fitted`.

    Anything computed from R's `residuals` on a seasonal Theta fit, an
    in-sample RMSE most obviously, is therefore on the deseasonalised
    scale even though the fitted values it appears to pair with are not.

## Verified against both references, which disagree with each other

Point forecasts agree with R's `forecast::thetaf` and `statsmodels`'
`ThetaModel` to `2e-5`, the residual being their different SES optima.
**The prediction intervals do not agree, in three separate ways**, and
this implementation follows R on the one they differ on and neither on the
third:

1. **`statsmodels`' `sigma2` ignores the deseasonalisation.** Its
   `sigma2` property fits a `SARIMAX(0,1,1)` with drift to
   `self.model._y` — the *original* series — while its point forecast is
   built on the deseasonalised one. On `AirPassengers` that is `993.23`
   against `104.71` for the same estimator on the deseasonalised series, a
   factor of 9.5. The giveaway: `sigma2` comes out **identical**
   (`49.7153734174`) whether the fixture is declared quarterly or
   non-seasonal, which can only happen if the seasonal adjustment never
   enters it. Passing `use_mle=True` takes a different path and does not
   have the problem.

2. **`statsmodels`' variance growth factor has an algebra slip.** It
   documents the variance as following from the model's IMA(1,1) structure
   and then computes `sigma2 * (1 + (h-1)*(1 + (alpha-1)^2))`. The IMA(1,1)
   result is `sigma2 * (1 + (h-1)*(1 + (alpha-1))^2)` — the square sits
   outside — which simplifies to `sigma2 * (1 + (h-1)*alpha^2)`, R's
   formula and the standard SES one. The two agree only at `alpha = 1`; at
   the fixture's `alpha = 0.68` the intervals widen 2.4 times too fast.
   This package uses `alpha^2`.

3. **Neither reference scales the standard error by the seasonal factor,
   and this one does.** If `y = x * S` with the error additive on `x`,
   then `var(y) = S^2 * var(x)`, so the interval has to be scaled along
   with the point forecast. R and `statsmodels` both apply an unscaled
   `sigma2` from the deseasonalised fit to the reseasonalised mean. On
   `AirPassengers`, where the factors run `0.80` to `1.23`, that makes
   their intervals about 23% too narrow at the seasonal peak and 20% too
   wide in the trough.

`sigma2` itself is `sse_x/n`, the ML convention used everywhere else here;
R's `ses` uses `sse_x/(n-2)`, which on the fixture is larger by exactly
`n/(n-2) = 1.0169491525`. Same `n`-versus-`n-k` choice as
[`fit_ets`](@ref) and [`arx`](@ref).

**Both of those are pinned by reproducing R exactly rather than asserted
loosely.** Undo the two conventions — divide the standard error by the
seasonal factor and substitute `sse_x/(n-2)` — and R's 95% bounds come
back to `3e-9`, with `alpha` pinned to R's value so the optimiser plays no
part. Left free, this package's SES reaches a slightly different optimum
(`0.67934` against R's `0.67924`), which by itself moves the bound about
`1.5e-3`; the point forecasts still agree to `2e-5`. Pass `alpha` to pin
it, as the verification does.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                                 "verification", "theta", "theta_y.csv")));

julia> m = fit_theta(y, 4);

julia> m.deseasonalized
true

julia> round(m.b0 / 2, digits=8)      # R's thetaf: 0.30832443
0.30832443
```

See also [`fit_ets`](@ref), [`holt_winters`](@ref), [`forecast`](@ref).
"""
function fit_theta(y, period::Integer=1; theta::Real=2.0,
                   deseasonalize::Bool=true, seasonal_test::Bool=true,
                   alpha::Union{Nothing,Real}=nothing)
    yv = Float64.(collect(tsvalues(y)))
    n = length(yv)
    n >= 3 || throw(ArgumentError("fit_theta: need at least 3 observations, got $n"))
    any(isnan, yv) && throw(ArgumentError("fit_theta: series contains NaN"))
    period >= 1 || throw(ArgumentError("fit_theta: period must be >= 1, got $period"))
    alpha === nothing || 0 < alpha <= 1 || throw(ArgumentError(
        "fit_theta: alpha must be in (0, 1], got $alpha"))
    theta > 1 || throw(ArgumentError(
        "fit_theta: theta must be > 1, got $theta — theta = 1 is the series " *
        "itself and carries no drift, and theta < 1 reverses its sign"))

    # --- seasonal adjustment -------------------------------------------------
    stat = NaN
    adjust = false
    if period > 1 && deseasonalize
        n >= 2 * period || throw(ArgumentError(
            "fit_theta: need at least 2 full periods ($(2 * period)) to " *
            "deseasonalize, got $n — pass deseasonalize=false"))
        stat = _theta_seasonal_statistic(yv, period)
        adjust = !seasonal_test || stat > _confidence_z(0.10)
    end

    figure = Float64[]
    seasonal_tile = Float64[]
    x = yv
    if adjust
        all(>(0), yv) || throw(ArgumentError(
            "fit_theta: multiplicative seasonal adjustment needs a strictly " *
            "positive series; pass deseasonalize=false, or model the series " *
            "on a scale where it is positive"))
        d = classical_decompose(yv, period; model=:multiplicative)
        figure = d.figure
        seasonal_tile = d.seasonal
        x = yv ./ seasonal_tile
    end

    # --- SES, and the regression slope --------------------------------------
    # ETS(A,N,N) is SES, already R-verified; `alpha` pins it and leaves only
    # the initial level to optimise, exactly as R's ses(..., alpha=) does
    ses = alpha === nothing ? fit_ets(x, 1) :
          fit_ets(x, 1; fixed=(alpha=Float64(alpha),))
    b0 = _theta_slope(x)

    # classical_decompose's `seasonal` is `figure` already tiled to length n
    fitted = adjust ? ses.fitted .* seasonal_tile : copy(ses.fitted)
    resid = yv .- fitted

    return ThetaModel(ses.alpha, b0, ses.level[end], Float64(theta), figure,
                      adjust, stat, fitted, resid, sum(abs2, resid), ses.sse,
                      ses.sigma2, n, period)
end

"""
    _theta_slope(x) -> Float64

The OLS slope of `x` on `0:(n-1)`, in closed form rather than through a
regression solve: the regressor is a known arithmetic sequence, so the
normal equations reduce to

    b = sum((t - tbar) * (x - xbar)) / sum((t - tbar)^2)

with `sum((t - tbar)^2) = n*(n^2 - 1)/12`. Matches R's
`lsfit(0:(n-1), x)\$coefficients[2]` to `1e-10`.
"""
function _theta_slope(x::AbstractVector{<:Real})
    n = length(x)
    xbar = sum(x) / n
    tbar = (n - 1) / 2
    num = zero(eltype(x)) + 0.0
    @inbounds for t in 1:n
        num += (t - 1 - tbar) * (x[t] - xbar)
    end
    return num / (n * (n^2 - 1) / 12)
end

"""
    forecast(m::ThetaModel, horizon; level=[80.0, 95.0]) -> Forecast
    predict(m::ThetaModel, horizon; level=[80.0, 95.0]) -> Forecast

Forecast `horizon` steps ahead from a fitted [`ThetaModel`](@ref).

    yhat[n+h] = (level + (1 - 1/theta) * b0 * (h - 1 + (1-(1-alpha)^n)/alpha)) * S[h]

and the interval from `sigma2 * (1 + (h-1)*alpha^2) * S[h]^2`, the IMA(1,1)
h-step variance with the seasonal factor carried through. `S[h]` is the
seasonal figure at the cycle position of observation `n+h`, or `1` when the
series was not deseasonalised.

The drift grows linearly in `h` and is never damped, so like
[`fit_ets`](@ref)'s undamped trend this extrapolates a straight line
indefinitely — the Theta method has no mechanism for flattening a long
horizon.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                                 "verification", "theta", "theta_y.csv")));

julia> f = forecast(fit_theta(y, 4), 8);

julia> f.model_name
"Theta(2.0)"

julia> round(f.point[1], digits=4)      # R's thetaf: 181.0777
181.0778
```

See also [`fit_theta`](@ref), [`accuracy`](@ref).
"""
function StatsAPI.predict(model::ThetaModel, horizon::Integer;
                           level::Vector{<:Real}=[80.0, 95.0])
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    isempty(level) && throw(ArgumentError("level must be non-empty"))
    all(0 .< level .< 100) || throw(ArgumentError("level entries must be in (0, 100)"))

    n, alpha = model.nobs, model.alpha
    # the drift's h-independent part: sum of the SES weights over the sample
    base = alpha > 0 ? (1 - (1 - alpha)^n) / alpha : Float64(n)
    w = (1 - 1 / model.theta) * model.b0
    point = [model.level + w * ((h - 1) + base) for h in 1:horizon]
    se = [sqrt(model.sigma2 * (1 + (h - 1) * alpha^2)) for h in 1:horizon]

    if model.deseasonalized
        m = model.period
        s = [model.figure[mod1(n + h, m)] for h in 1:horizon]
        point .*= s
        se .*= s          # var(y) = S^2 var(x); see fit_theta's docstring
    end

    z = [_confidence_z(1 - l / 100) for l in level]
    lower = reduce(hcat, [point .- zi .* se for zi in z])
    upper = reduce(hcat, [point .+ zi .* se for zi in z])
    name = string("Theta(", round(model.theta, digits=4), ")")
    return Forecast(point, se, Float64.(level), lower, upper, horizon, name)
end

forecast(model::ThetaModel, horizon::Integer; level::Vector{<:Real}=[80.0, 95.0]) =
    StatsAPI.predict(model, horizon; level=level)

StatsAPI.coef(m::ThetaModel) = [m.alpha, m.b0]
StatsAPI.residuals(m::ThetaModel) = m.resid
StatsAPI.fitted(m::ThetaModel) = m.fitted
StatsAPI.nobs(m::ThetaModel) = m.nobs
