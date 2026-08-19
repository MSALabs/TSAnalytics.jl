export fit_arima, ArimaModel

"""
    ArimaModel <: UnivariateModel

Result of [`fit_arima`](@ref): a non-seasonal ARIMA(p,d,q) model --
`d`-times-differenced `y`, fit as a stationary ARMA(p,q) via
[`fit_arma`](@ref) (Stage 6.5) and wrapped with the differencing order
and the original (undifferenced) series.

- `arma`: the underlying [`ArmaModel`](@ref) fit on the differenced
  series -- reused directly, not duplicated. All of `ArmaModel`'s own
  fields (`ar`, `ma`, `mean`, `se`, `loglik`, `sigma2`, `aic`, `bic`,
  `converged`, ...) are reachable through it.
- `d`: the differencing order actually applied.
- `original_y`: the undifferenced series, kept for future forecast
  re-integration (`undifference`/`diffinv`).
"""
struct ArimaModel <: UnivariateModel
    arma::ArmaModel
    d::Int
    original_y::Vector{Float64}
end

StatsAPI.loglikelihood(m::ArimaModel) = m.arma.loglik
StatsAPI.aic(m::ArimaModel) = m.arma.aic
StatsAPI.bic(m::ArimaModel) = m.arma.bic
StatsAPI.coef(m::ArimaModel) = vcat(m.arma.ar, m.arma.ma)

"""
    StatsAPI.nobs(m::ArimaModel) -> Int

**`n - d`**, matching R's `stats::arima()` reported `nobs`/`n.used`
exactly (verified directly, on two series, including one with
convergence warnings) -- **not** Python's `statsmodels ARIMA`, whose
`nobs` is the full original `n`. This is architecturally real, not a
display quirk: `statsmodels` uses diffuse state augmentation internally
and genuinely retains all `n` observations in its likelihood; this
package differences first (matching the scope boundary decided when
Stage 6 vs. Stage 8 was split -- diffuse initialization is Stage 8's
job) and genuinely only has `n-d` effective observations to compute a
stationary likelihood on. Every information criterion (`aic`/`bic`)
inherits this convention through [`ArmaModel`](@ref)'s own `nobs`.
"""
StatsAPI.nobs(m::ArimaModel) = m.arma.nobs

"""
    fit_arima(y, order::Tuple{Int,Int,Int}; include_mean=true, method=:ml,
              se_type=:hessian, optimizer_method=:lbfgs, start_params=nothing) -> ArimaModel

Fit ARIMA(p,d,q) by differencing `y` `d` times (`diff(y, 1;
differences=d)`, Stage 1.1) and delegating the stationary ARMA(p,q) fit
entirely to [`fit_arma`](@ref) (Stage 6.5) -- this function adds no new
fitting logic of its own; `order=(p, 0, q)` produces results identical
to calling `fit_arma(y, (p, q))` directly. Every keyword matches
`fit_arma`'s own signature exactly, since this function's whole job is
differencing plus delegation, not introducing new fitting options.

Verified end-to-end against real R `stats::arima(order=c(p,d,q))` and
real Python `statsmodels.tsa.arima.model.ARIMA(order=(p,d,q))` on two
series (one clean, one with genuine convergence warnings on both
references -- kept as a documented hard case, not avoided) -- see
`handoff/stage-6.6-arima-handoff.md` §1-3. Coefficients and
log-likelihood match both references closely; `nobs` matches R's `n-d`
convention specifically -- see the `nobs` field/docstring note below.

**`include_mean` is silently ignored (forced to `false`) whenever
`d > 0`**, matching R's actual `stats::arima()` behavior exactly
(confirmed directly: `include.mean=TRUE` and `include.mean=FALSE`
produce bit-identical fitted coefficients when `d>0`) -- a mean term
only makes sense on a stationary (here: differenced) series, and R
drops it rather than erroring. This is a deliberate match to the
reference's own convention, not an oversight; it only has any effect
when `d == 0`, in which case it behaves exactly like
[`fit_arma`](@ref)'s own `include_mean`.

# Examples
```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> y = vec(readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test", "verification", "arima", "d1_clean.csv"), ','; skipstart=1));

julia> m = fit_arima(y, (1, 1, 1); include_mean=false);

julia> m.arma.converged
true

julia> round(m.arma.ar[1], digits=4)
0.5828

julia> m.arma.nobs  # n - d = 150 - 1 -- StatsAPI.nobs(m) reads this same field
149
```
"""
function fit_arima(y, order::Tuple{Int,Int,Int};
                    include_mean::Bool=true,
                    method::Symbol=:ml,
                    se_type::Symbol=:hessian,
                    optimizer_method::Symbol=:lbfgs,
                    start_params::Union{Nothing,Vector{Float64}}=nothing)
    p, d, q = order
    p >= 0 && d >= 0 && q >= 0 || throw(ArgumentError("order must be non-negative: got ($p, $d, $q)"))

    yv = Float64.(collect(tsvalues(y)))
    yd = diff(yv, 1; differences=d)

    mean_flag = d > 0 ? false : include_mean

    arma = fit_arma(yd, (p, q); include_mean=mean_flag, method=method,
                     se_type=se_type, optimizer_method=optimizer_method,
                     start_params=start_params)

    return ArimaModel(arma, d, yv)
end

function Base.show(io::IO, m::ArimaModel)
    p, q = length(m.arma.ar), length(m.arma.ma)
    print(io, "ARIMA(", p, ",", m.d, ",", q, ")")
    println(io, m.arma.mean !== nothing ? " with mean" : "", ", n=", m.arma.nobs,
            " (", m.arma.method, ", se: ", m.arma.se_type, ")")
    show(io, m.arma)
end
