export autoreg, AutoRegModel

"""
    AutoRegModel

Result of [`autoreg`](@ref): a linear regression whose **errors** follow
an autoregression. Fields:

- `beta`: regression coefficients, in the column order of `X`.
- `se`: their standard errors, from the GLS covariance at the fitted `phi`.
- `vcov`: the full GLS covariance matrix of `beta`.
- `phi`: the fitted AR coefficients of the error process, length `order`.
- `sigma2`: innovation variance of the error process.
- `resid`: the AR residuals (`epsilon`), i.e. what is left after both the
  regression and the AR structure.
- `nu`: the regression errors (`y - X*beta`), which are the series the AR
  was fitted to.
- `iterations`: feasible-GLS iterations actually used.
- `converged`: whether `phi` stopped moving by more than `tol`.
- `n`, `order`, `method`.

**`se` describes `beta` only.** The AR coefficients are a nuisance
parameter here, estimated but not inferred on, which is what makes this
the *basic* tier rather than the full maximum-likelihood one -- see
[`autoreg`](@ref).
"""
struct AutoRegModel
    beta::Vector{Float64}
    se::Vector{Float64}
    vcov::Matrix{Float64}
    phi::Vector{Float64}
    sigma2::Float64
    resid::Vector{Float64}
    nu::Vector{Float64}
    iterations::Int
    converged::Bool
    n::Int
    order::Int
    method::Symbol
end

function Base.show(io::IO, m::AutoRegModel)
    print(io, "Regression with AR(", m.order, ") errors, n=", m.n,
           " (", m.method, ", ", m.iterations, " iteration",
           m.iterations == 1 ? "" : "s", ")")
    print(io, "\n  beta   : ", round.(m.beta, digits=6))
    print(io, "\n  se     : ", round.(m.se, digits=6))
    print(io, "\n  phi    : ", round.(m.phi, digits=6))
    print(io, "\n  sigma2 : ", round(m.sigma2, digits=6))
    m.converged || print(io, "\nWARNING: phi had not settled within maxiter")
end

"""
    autoreg(y, X; order=1, method=:yw, maxiter=50, tol=1e-8) -> AutoRegModel

Linear regression with **autoregressive errors**:

    y_t = X_t' beta + nu_t,     nu_t = sum_i phi_i nu_{t-i} + eps_t

fitted by **feasible GLS**: OLS, estimate the AR structure of the
residuals, transform, re-estimate, repeat. This is SAS `PROC AUTOREG`'s
basic tier — the Cochrane-Orcutt/Prais-Winsten family — and the
classical counterpart to [`fit_arimax`](@ref), which fits the same model
by exact maximum likelihood.

| | [`autoreg`](@ref) | [`fit_arimax`](@ref) |
|---|---|---|
| Estimator | Feasible GLS, iterated | Exact ML via the Kalman filter |
| Error structure | AR(`order`) | Full ARIMA |
| First observation | **Kept** (Prais-Winsten scaling) | Kept |
| Inference on `phi` | No — a nuisance parameter | Yes, `se` covers it |
| Cost | A few OLS solves | An optimiser run |

Reach for `autoreg` when the regression coefficients are what you care
about and the serial correlation is a nuisance to be corrected for
rather than modelled. Reach for `fit_arimax` when the error dynamics
matter in their own right, or when you want a likelihood.

## Why not just use OLS

OLS `beta` stays unbiased under autocorrelated errors — but its standard
errors do not. With positively autocorrelated errors the usual OLS
standard errors are **too small**, often badly, so `t`-statistics are
inflated and a regressor can look significant when it is not. That is
the whole motivation, and it is visible in the worked example below:
correcting for AR(1) errors with `phi = 0.69` roughly **doubles** the
standard error on the slope.

## Method

`method=:yw` estimates the AR coefficients from the current residuals by
Yule-Walker ([`ar_yw`](@ref) with a fixed order), which guarantees a
stationary error process at every iteration — the property that keeps
the GLS transform well defined. `method=:ols` uses the least-squares AR
estimate instead, which is less biased but can wander outside the
stationary region on short samples; it is rejected there rather than
producing an undefined transform.

The transform is **Prais-Winsten**, not plain Cochrane-Orcutt: the first
`order` observations are kept and rescaled by the stationary covariance
rather than discarded. Dropping them is the more commonly taught
version, and it throws away real information — on a 200-point series
with `order=1` that is only one observation, but it is also the
observation carrying the level.

!!! note "Verified by holding `phi` fixed"
    There is no single R or Python function that matches this. `orcutt`
    implements Cochrane-Orcutt but is not installable on this R version,
    and `nlme::gls` fits by ML/REML rather than feasible GLS, so its
    `phi` differs by construction.

    **At a fixed `phi` the two are the same estimator**, so that is what
    is checked: `autoreg` with `phi` pinned reproduces
    `nlme::gls(..., correlation=corAR1(value=rho, fixed=TRUE))`'s `beta`
    and standard errors to `1e-9` at three values of `rho`. The AR half
    is separately R-verified through [`ar_yw`](@ref). Both components
    have a reference; their composition does not.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> d = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                             "verification", "autoreg", "ar1reg.csv"), ','; skipstart=1);

julia> y = d[:, 1]; X = hcat(ones(size(d, 1)), d[:, 2]);

julia> m = autoreg(y, X);

julia> m.order
1

julia> round(m.phi[1], digits=4)
0.6877

julia> round.(m.beta, digits=4)
2-element Vector{Float64}:
 1.5948
 2.1684
```

(`nlme::gls`'s ML estimate on the same data is `rho = 0.6866`, `beta =
[1.5949, 2.1693]` — close, and not identical, because feasible GLS and
maximum likelihood are different estimators. Asserting equality would be
asserting something false.)

See also [`fit_arimax`](@ref), [`ar_yw`](@ref), [`arx`](@ref),
[`durbin_watson_test`](@ref).
"""
function autoreg(y, X; order::Integer=1, method::Symbol=:yw,
                  maxiter::Integer=50, tol::Real=1e-8,
                  phi::Union{Nothing,AbstractVector{<:Real}}=nothing)
    method in (:yw, :ols) ||
        throw(ArgumentError("autoreg: method must be :yw or :ols, got :$method"))
    order >= 1 || throw(ArgumentError("autoreg: order must be >= 1, got $order"))
    maxiter >= 1 || throw(ArgumentError("autoreg: maxiter must be >= 1, got $maxiter"))

    yv = Float64.(collect(tsvalues(y)))
    Xm = X isa AbstractMatrix ? Float64.(X) : reshape(Float64.(collect(tsvalues(X))), :, 1)
    n = length(yv)
    size(Xm, 1) == n || throw(DimensionMismatch(
        "autoreg: X must have one row per observation (got $(size(Xm, 1)) for n=$n)"))
    k = size(Xm, 2)
    n > k + order || throw(ArgumentError(
        "autoreg: not enough observations ($n) for $k regressors and an AR($order) error"))
    any(isnan, yv) || any(isnan, Xm) &&
        throw(ArgumentError("autoreg: NaN present (no missing-data policy implemented yet)"))

    if phi !== nothing
        length(phi) == order || throw(DimensionMismatch(
            "autoreg: a fixed phi must have length order=$order, got $(length(phi))"))
        ph = Float64.(collect(phi))
        _ar_stationary(ph) || throw(ArgumentError(
            "autoreg: the supplied phi is not stationary, so the GLS transform is undefined"))
        beta, vc, s2, eps, nu = _autoreg_gls(yv, Xm, ph)
        return AutoRegModel(beta, _vcov_to_se(vc), vc, ph, s2, eps, nu, 0, true,
                             n, Int(order), :fixed)
    end

    beta, = _ols(Xm, yv; method=:qr)
    ph = zeros(order)
    iters = 0
    conv = false
    vc = zeros(k, k)
    s2 = NaN
    eps = Float64[]
    nu = yv .- Xm * beta
    for it in 1:maxiter
        iters = it
        ph_new = _autoreg_ar_step(nu, Int(order), method)
        _ar_stationary(ph_new) || throw(ArgumentError(
            "autoreg: the AR($order) estimate left the stationary region at iteration $it " *
            (method === :ols ? "-- try method=:yw, which cannot" : "-- the series may not support this order")))
        delta = maximum(abs.(ph_new .- ph))
        ph = ph_new
        beta, vc, s2, eps, nu = _autoreg_gls(yv, Xm, ph)
        if delta < tol
            conv = true
            break
        end
    end

    return AutoRegModel(beta, _vcov_to_se(vc), vc, ph, s2, eps, nu, iters, conv,
                         n, Int(order), method)
end

"_autoreg_ar_step(nu, order, method) -- the AR coefficients of the
current regression residuals, at a FIXED order (no AIC selection: the
order is the user's model choice here, not something to re-pick inside a
GLS loop, which would make the iteration non-monotone)."
function _autoreg_ar_step(nu::Vector{Float64}, order::Int, method::Symbol)
    if method === :yw
        return ar_yw(nu; order=order, order_max=order).ar
    end
    # :ols -- regress nu[t] on its own lags, no intercept (the residuals are
    # already centred by the regression)
    n = length(nu)
    A = [nu[t-j] for t in (order+1):n, j in 1:order]
    b = nu[(order+1):n]
    return A \ b
end

"_ar_stationary(phi) -- whether the AR polynomial's roots lie outside the
unit circle, via the companion matrix's spectral radius. The GLS
transform needs the stationary covariance to exist, so this is a
precondition rather than a diagnostic."
function _ar_stationary(phi::AbstractVector{<:Real})
    p = length(phi)
    p == 0 && return true
    C = zeros(p, p)
    C[1, :] = phi
    p > 1 && (C[2:end, 1:end-1] = Matrix(1.0LinearAlgebra.I, p - 1, p - 1))
    return maximum(abs.(LinearAlgebra.eigvals(C))) < 1
end

"_autoreg_gls(y, X, phi) -- one GLS step at a given `phi`, by the
**Prais-Winsten** transform: differencing rows `order+1:n` by the AR
polynomial, and rescaling the first `order` rows by the Cholesky factor
of the stationary covariance so they are kept rather than discarded.

Returns `(beta, vcov, sigma2, eps, nu)`. `vcov` is `sigma2 * inv(Z'Z)`
on the transformed design `Z`, which is the GLS covariance of `beta`."
function _autoreg_gls(yv::Vector{Float64}, Xm::Matrix{Float64}, phi::Vector{Float64})
    n = length(yv)
    k = size(Xm, 2)
    p = length(phi)

    # stationary covariance of the first p errors, from the Yule-Walker
    # equations, then its inverse Cholesky factor as the row weights
    G = _ar_stationary_gamma(phi, p)
    L = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(G)).L
    W = inv(L)                       # W' W = inv(G): whitens the first p rows

    Z = Matrix{Float64}(undef, n, k)
    w = Vector{Float64}(undef, n)
    # first p rows: whitened jointly
    Z[1:p, :] = W * Xm[1:p, :]
    w[1:p] = W * yv[1:p]
    # remaining rows: filtered by the AR polynomial
    for t in (p+1):n
        w[t] = yv[t] - sum(phi[j] * yv[t-j] for j in 1:p)
        for c in 1:k
            Z[t, c] = Xm[t, c] - sum(phi[j] * Xm[t-j, c] for j in 1:p)
        end
    end

    beta = Z \ w
    r = w .- Z * beta
    dof = n - k
    s2 = sum(abs2, r) / dof
    vc = Matrix(s2 .* inv(LinearAlgebra.Symmetric(Z' * Z)))

    nu = yv .- Xm * beta
    eps = similar(nu)
    eps[1:p] = (W * nu[1:p])
    for t in (p+1):n
        eps[t] = nu[t] - sum(phi[j] * nu[t-j] for j in 1:p)
    end
    return beta, vc, s2, eps, nu
end

"_ar_stationary_gamma(phi, p) -- the `p x p` stationary autocovariance
matrix of an AR(`p`) process with unit innovation variance, built from
the autocovariance sequence the Yule-Walker equations give."
function _ar_stationary_gamma(phi::Vector{Float64}, p::Int)
    # gamma[0..p-1] from the Yule-Walker system, unit innovation variance
    A = zeros(p + 1, p + 1)
    for i in 0:p
        A[i+1, i+1] += 1.0
        for j in 1:p
            idx = abs(i - j)
            A[i+1, idx+1] -= phi[j]
        end
    end
    rhs = zeros(p + 1)
    rhs[1] = 1.0
    g = A \ rhs
    return [g[abs(i - j)+1] for i in 1:p, j in 1:p]
end
