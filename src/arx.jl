export arx, ARXModel

"""
    ARXModel <: UnivariateModel

Result of [`arx`](@ref). `names` labels each entry of `coef`/`vcov`'s
rows-and-columns in the same order the design matrix was built:
trend/seasonal terms first, then lags (`"y.L1"`, `"y.L2"`, ...), then any
`exog` columns (`"x1"`, `"x2"`, ...) -- matching `statsmodels.tsa.ar_model.AutoReg`'s
own column order and naming exactly (verified by execution, not assumed).
"""
struct ARXModel <: UnivariateModel
    coef::Vector{Float64}
    vcov::Matrix{Float64}
    names::Vector{String}
    resid::Vector{Float64}
    sigma2::Float64
    loglik::Float64
    aic::Float64
    bic::Float64
    nobs::Int
    lags::Vector{Int}
    trend::Symbol
end

StatsAPI.coef(m::ARXModel) = m.coef
StatsAPI.vcov(m::ARXModel) = m.vcov
StatsAPI.stderror(m::ARXModel) = sqrt.(_diagvec(m.vcov))
StatsAPI.residuals(m::ARXModel) = m.resid
StatsAPI.nobs(m::ARXModel) = m.nobs
StatsAPI.loglikelihood(m::ARXModel) = m.loglik
StatsAPI.aic(m::ARXModel) = m.aic
StatsAPI.bic(m::ARXModel) = m.bic

"""
    arx(y, lags; trend=:c, seasonal=false, period=nothing, exog=nothing,
        hold_back=nothing, method=:qr) -> ARXModel

Autoregressive model with optional exogenous regressors ("AR-X"), fit by
conditional least squares. Argument names follow Python's
`statsmodels.tsa.ar_model.AutoReg` (the fuller-featured reference -- R's
`stats::ar.ols` has no exogenous-regressor support at all, and R's
`stats::ar()` is a different function defaulting to Yule-Walker
estimation, not CLS; see `handoff/stage-5.1-arx-handoff.md` for the full
comparison). Verified exactly against real `AutoReg` output -- every
coefficient, standard error, `sigma2`, log-likelihood, AIC, and BIC --
across `trend` values, an `exog` regressor, a lag *subset* (not just
`1:p`), and explicit `hold_back`.

- `lags`: an `Integer` (fits lags `1:lags`) or an `AbstractVector` (an
  arbitrary lag subset, e.g. `[1,3]`, skipping lag 2 entirely) -- matches
  `AutoReg` exactly, a real capability R's `ar.ols` lacks.
- `trend`: `:n` (none) / `:c` (constant, default) / `:t` (linear trend,
  no constant) / `:ct` (constant + trend).
- `seasonal`: add `period-1` deterministic seasonal dummy columns,
  named `"s(2,period)"`, `"s(3,period)"`, ... `"s(period,period)"` --
  season `1` is the implicit baseline/reference category, not season
  `period`. Matches `AutoReg` exactly, confirmed by execution: its own
  dummy columns and naming start at season `2`, not `1` (the handoff's
  original proposal built a dummy for every season `1:(period-1)`
  instead, which is a *different*, equally-valid-but-non-matching
  parameterization -- same underlying model, different reference
  category, so genuinely different coefficient values, not just a
  cosmetic naming difference). Requires `period`.
- `exog`: optional exogenous regressor vector/matrix -- the "X" in AR-X.
  Must have the same number of rows as `y`.
- `hold_back`: number of initial observations to exclude from the fit;
  defaults to `maximum(lags)`, and must be at least that (matching
  `AutoReg`'s own validation, confirmed by execution:
  `hold_back must be >= lags if lags is an int or max(lags) if lags is array_like`).
- `method`: `:qr` (default) or `:cholesky`, passed to `_ols`.

!!! note "Standard errors do NOT come directly from `_ols`"
    `_ols`'s own `se` uses the degrees-of-freedom-adjusted variance
    estimator (`sigma2 = SSR/(n-k)`, the usual "unbiased" OLS
    convention) -- but `AutoReg`'s actual reported standard errors use
    the *conditional-MLE* convention (`sigma2 = SSR/n`), confirmed by
    reproducing `AutoReg`'s exact `bse` from a from-scratch computation:
    the two differ by a factor of `sqrt(n/(n-k))`, about 1% for typical
    AR-X sample sizes -- small, but not negligible, and silently wrong
    had the handoff's original "just use `_ols`'s `se`" proposal been
    followed as written. `arx` computes `vcov`/`stderror` itself here
    (`sigma2 * inv(X'X)`, `sigma2` on the `n`-denominator convention)
    rather than trusting `_ols`'s own `se` field.

`y` and `exog` both accept anything [`tsvalues`](@ref) does.

# Examples
```jldoctest
julia> using TSAnalytics, Random

julia> Random.seed!(1); y = cumsum(randn(100)) .* 0.1 .+ sin.(2π .* (1:100) ./ 12);

julia> m = arx(y, 2);

julia> length(m.coef) == 3  # const + 2 lags
true

julia> m.nobs == length(y) - 2
true
```
"""
function arx(y, lags::Union{Integer,AbstractVector{<:Integer}};
             trend::Symbol=:c, seasonal::Bool=false, period::Union{Nothing,Integer}=nothing,
             exog=nothing, hold_back::Union{Nothing,Integer}=nothing,
             method::Symbol=:qr)
    trend in (:n, :c, :t, :ct) || throw(ArgumentError("trend must be :n, :c, :t, or :ct"))
    seasonal && period === nothing && throw(ArgumentError("seasonal=true requires period"))
    seasonal && period !== nothing && period < 2 && throw(ArgumentError("period must be >= 2"))

    lagset = lags isa Integer ? collect(1:lags) : collect(Int, lags)
    isempty(lagset) && throw(ArgumentError("lags must be non-empty"))
    all(>=(1), lagset) || throw(ArgumentError("every lag must be >= 1"))
    maxlag = maximum(lagset)

    yv = tsvalues(y)
    n = length(yv)
    any(!isfinite, yv) && throw(ArgumentError("arx: missing/non-finite values not supported in y"))

    hb = hold_back === nothing ? maxlag : hold_back
    hb >= maxlag || throw(ArgumentError("hold_back must be >= maximum(lags) ($maxlag)"))
    n > hb || throw(ArgumentError("not enough observations after hold_back=$hb (n=$n)"))

    resp = collect(Float64, yv[(hb+1):n])
    nobs = length(resp)

    cols = Vector{Vector{Float64}}()
    names = String[]

    if trend in (:c, :ct)
        push!(cols, ones(nobs)); push!(names, "const")
    end
    if trend in (:t, :ct)
        push!(cols, collect(Float64, (hb+1):n)); push!(names, "trend")
    end
    if seasonal
        # Seasons 2:period get a dummy, season 1 is the implicit baseline --
        # matches AutoReg's actual column choice and naming exactly (verified
        # by execution: 's(2,4)','s(3,4)','s(4,4)' for period=4, not a dummy
        # for every season starting at 1, which the handoff's original
        # 1:(period-1)/"seasonal.$s" proposal would have built instead).
        for s in 2:period
            push!(cols, Float64[mod1(t, period) == s ? 1.0 : 0.0 for t in (hb+1):n])
            push!(names, "s($s,$period)")
        end
    end
    for lag in lagset
        push!(cols, collect(Float64, yv[(hb+1-lag):(n-lag)])); push!(names, "y.L$lag")
    end
    if exog !== nothing
        exv = tsvalues(exog)
        ex_mat = exv isa AbstractMatrix ? exv : reshape(collect(exv), :, 1)
        size(ex_mat, 1) == n || throw(ArgumentError("exog must have the same length as y"))
        any(!isfinite, ex_mat) && throw(ArgumentError("arx: missing/non-finite values not supported in exog"))
        for j in 1:size(ex_mat, 2)
            push!(cols, collect(Float64, ex_mat[(hb+1):n, j])); push!(names, "x$j")
        end
    end

    X = reduce(hcat, cols)
    # `_ols`'s QR-based solve can still return SOME beta for a rank-deficient
    # X (least-squares via QR doesn't require full column rank the way
    # inv(X'X) does), and relying on inv() to throw is NOT portable: exact
    # singularity detection via LU pivoting is BLAS-implementation-dependent
    # -- confirmed the hard way, via a real cross-platform CI failure: a
    # design matrix that inv() cleanly throws SingularException on under
    # OpenBLAS (Linux/Windows) came back as "just barely invertible" under
    # Apple's Accelerate framework (macOS, Apple Silicon runners) for the
    # exact same input, so a try/catch around inv() silently passed there
    # instead of raising. Checking `rank(X)` (SVD-based, with an
    # appropriately scaled tolerance) up front is the numerically robust,
    # portable way to catch this -- checked before calling `_ols` at all,
    # not opportunistically after.
    rank(X) == size(X, 2) ||
        throw(ArgumentError("arx: design matrix is singular/collinear -- check for redundant " *
                             "regressors (e.g. trend plus lags of a perfectly linear series, or " *
                             "an exog column that duplicates another regressor)"))

    beta, resid, = _ols(X, resp; method=method)

    k = length(beta)
    sigma2 = sum(abs2, resid) / nobs  # MLE (n-denominator) convention -- see docstring note
    vc = Matrix(sigma2 .* inv(Symmetric(X' * X)))
    loglik = -0.5 * nobs * (log(2π) + log(sigma2) + 1)
    aic_val = -2*loglik + 2*(k+1)
    bic_val = -2*loglik + (k+1)*log(nobs)

    return ARXModel(beta, vc, names, resid, sigma2, loglik, aic_val, bic_val, nobs, lagset, trend)
end

function Base.show(io::IO, m::ARXModel)
    se = sqrt.(_diagvec(m.vcov))
    z = m.coef ./ se
    p = _chisq_ccdf.(z .^ 2, 1)  # two-sided normal p-value via Z^2 ~ ChiSq(1), see arx's docstring
    zcrit = _confidence_z(0.05)  # precise 0.975 quantile (1.959964...), not the coarse 1.96 literal
    ci_lo = m.coef .- zcrit .* se
    ci_hi = m.coef .+ zcrit .* se
    ct = StatsBase.CoefTable(
        hcat(m.coef, se, z, p, ci_lo, ci_hi),
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower 95%", "Upper 95%"],
        m.names, 4, 3,
    )
    println(io, "AR(", maximum(m.lags), ")",
            m.trend == :n ? "" : " with $(m.trend) trend", ", n=", m.nobs)
    println(io)
    println(io, ct)
    println(io)
    print(io, "Log-likelihood: ", round(m.loglik, digits=2),
          "   AIC: ", round(m.aic, digits=2),
          "   BIC: ", round(m.bic, digits=2))
end
