export fit_var, var_select, VARModel, VARSelection, VARForecast
export roots, is_stable, companion

"""
    VARModel <: TimeSeriesModel

A fitted vector autoregression, returned by [`fit_var`](@ref).

## Fields

- `Y`: the data, `n × K`, **rows = time, columns = variables**.
- `names`: the `K` variable names, `[:y1, …]` when the input carried none.
- `p`: the lag order; `deterministic`: which deterministic terms were included.
- `A`: `p` matrices, each `K × K`. `A[j][i, m]` is the coefficient on
  variable `m` at lag `j` in equation `i` — so `A[j]` is directly the
  matrix the companion form and the impulse-response recursion want.
- `det_coef`: `K × ndet`, columns in the order `(const, trend, trend²)`.
- `sigma_df`: `E'E / (T - Kp - ndet)`, the degrees-of-freedom-adjusted
  residual covariance — this feeds standard errors and forecast intervals.
- `sigma_mle`: `E'E / T` — this feeds the log-likelihood and the
  information criteria. **Both are stored** because both references
  report both and they are not interchangeable.
- `se`: `K × (Kp + ndet)`, laid out like `coef(m)`.
- `vcov_full`: `K(Kp+ndet) × K(Kp+ndet)`, `kron(sigma_df, (Z'Z)⁻¹)`, kept
  whole rather than as a diagonal because the analytic impulse-response
  standard errors consume the whole thing.
- `resid`, `fitted`: `T × K`.
- `loglik`, `T` (effective observations, `n - p`).
- `C`: the `Kp × Kp` companion matrix, built once at fit time and cached,
  because [`roots`](@ref), the impulse-response recursion and the
  structural decomposition all want it.
- `converged`: always `true` — this is OLS, and the field exists for
  contract uniformity.

## Coefficient layout

Neither reference's flat layout is used. `vars` returns `K × (Kp + ndet)`
with the deterministic terms **last**; `statsmodels` returns
`(ndet + Kp) × K` with them **first**, transposed. Both were verified to
agree with each other once reordered (maximum absolute difference `2.5e-16`
at `K = 2`), so there is no right answer to inherit — and separating `A`
from `det_coef` removes the ordering question from every model built on
top of this one.
"""
struct VARModel <: TimeSeriesModel
    Y::Matrix{Float64}
    names::Vector{Symbol}
    p::Int
    deterministic::Symbol
    A::Vector{Matrix{Float64}}
    det_coef::Matrix{Float64}
    sigma_df::Matrix{Float64}
    sigma_mle::Matrix{Float64}
    se::Matrix{Float64}
    vcov_full::Matrix{Float64}
    resid::Matrix{Float64}
    fitted::Matrix{Float64}
    loglik::Float64
    T::Int
    C::Matrix{Float64}
    converged::Bool
end

"""
    VARSelection

The lag-order selection table returned by [`var_select`](@ref): the four
information criteria across candidate lags, and the lag each one picks.

`T` is held **fixed** at `n - lag_max` across every candidate, which is
what makes the criteria comparable with one another. A fitted
[`VARModel`](@ref)'s own `aic` uses its own larger sample, so the two
numbers differ by construction — see [`var_select`](@ref).
"""
struct VARSelection
    lags::Vector{Int}
    aic::Vector{Float64}
    bic::Vector{Float64}
    hqic::Vector{Float64}
    fpe::Vector{Float64}
    selected::NamedTuple{(:aic, :bic, :hqic, :fpe),NTuple{4,Int}}
    T::Int
    deterministic::Symbol
end

"""
    VARForecast

A multivariate forecast: `horizon × K` points with `horizon × K × levels`
interval bands.

Deliberately **not** a generalised [`Forecast`](@ref). That type is
univariate by construction — its `point` is a vector and its `lower`/
`upper` are `horizon × levels` — and nine model families already return
it, so widening it would be a breaking change to all of them. The field
names match `Forecast`'s wherever they mean the same thing, so the two
read alike.

- `point`: `horizon × K`
- `se`: `horizon × K`
- `levels`: the requested percentages
- `lower`, `upper`: `horizon × K × length(levels)`
- `names`: the `K` variable names
- `horizon`, `model_name`
"""
struct VARForecast
    point::Matrix{Float64}
    se::Matrix{Float64}
    levels::Vector{Float64}
    lower::Array{Float64,3}
    upper::Array{Float64,3}
    names::Vector{Symbol}
    horizon::Int
    model_name::String
end

"""
    _var_ndet(deterministic) -> Int

How many deterministic columns a specification contributes.

The five values span both references' vocabularies, which are **mutually
incomplete** — R has no equivalent of `statsmodels`' `"ctt"`, and
`statsmodels` has no equivalent of R's `type="trend"` (a trend with *no*
constant). The trap this keyword exists to remove: R's `type="trend"` and
`statsmodels`' `trend="ct"` sound equivalent and are not. `[V]` On the
bundled fixture they differ by `6.15` in log-likelihood, and the mismatch
propagates into order selection.

| `deterministic` | ndet | columns | R `type=` | `statsmodels` `trend=` |
|---|---|---|---|---|
| `:none` | 0 | — | `"none"` | `"n"` |
| `:const` | 1 | 1 | `"const"` | `"c"` |
| `:trend` | 1 | t | `"trend"` | *none* |
| `:both` | 2 | 1, t | `"both"` | `"ct"` |
| `:quadratic` | 3 | 1, t, t² | *none* | `"ctt"` |
"""
function _var_ndet(deterministic::Symbol)
    deterministic === :none && return 0
    deterministic === :const && return 1
    deterministic === :trend && return 1
    deterministic === :both && return 2
    deterministic === :quadratic && return 3
    throw(ArgumentError(
        "fit_var: deterministic must be :none, :const, :trend, :both or " *
        ":quadratic, got :$deterministic. Note :trend means a trend with NO " *
        "constant (R's type=\"trend\"); :both is constant + trend " *
        "(statsmodels' trend=\"ct\")."))
end

"""
    build_var_Z(Y, p, deterministic) -> (Z, X, colnames)

Build the VAR regressor matrix once, with its column names.

Internal and unexported, but named without an underscore to match the
Stage 10.0 handoff, which specifies it as a primitive the later items
reuse.

    Z = [ y_{t-1}' … y_{t-p}' | d_t' ]   for t = p+1 … n     (T × (Kp + ndet))
    X = [ y_t' ]                                              (T × K)

Lag blocks first — ascending lag outside, variable index inside — and the
deterministic terms **last**, in the order `(const, trend, trend²)`.

`[V]` Both references build the identical lag block, verified on its first
and last rows. They put the deterministic terms *first*; this package puts
them last so that `coef(m)` reads in R's order, and nothing downstream
depends on the choice.

The quadratic term is the raw square `t²` with `t = p+1 … n`, not a
centred or scaled variant. The trend²-coefficient alone cannot confirm
that, because it is invariant to an origin shift; the constant and trend
coefficients are not, and the test suite checks all three against
`statsmodels`' `"ctt"` fit.

!!! warning "Do not use `statsmodels`' `get_var_endog` as a reference for the trend"
    `[V]` Its helper builds the trend as `1:T`, while its own
    `VAR.fit(trend="ct")` returns coefficients corresponding to `p+1:n`.
    The helper and the fit path disagree with each other. The helper is a
    correct reference for the lag block and a wrong one for the
    deterministic columns, so a test written against it would enshrine
    the bug. The test suite compares only the lag block against it.

`[V]` **The trend column runs `p+1 : n`** — the calendar index of the
estimation sample, not a within-sample counter `1 : T`. Both references
do this and agree with each other on both deterministic coefficients to
`9.5e-16`, so there is one right answer rather than a convention to pick.

It is worth knowing *why* this is easy to get wrong: an origin shift is
absorbed entirely by the constant, so the fit, the residuals, the fitted
values, `Σ`, the log-likelihood and every information criterion are
**identical** under either form. Only the reported intercept moves, by
exactly `p ×` the trend coefficient. `[V]` On `var2_p1.csv` at `p = 1`
with `:both`, the calendar form matches the reference intercept to
`1.97e-15` and the within-sample form is out by `4.92e-4`.
"""
function build_var_Z(Y::AbstractMatrix{<:Real}, p::Integer, deterministic::Symbol)
    n, K = size(Y)
    ndet = _var_ndet(deterministic)          # validates `deterministic` first
    p >= 1 || throw(ArgumentError("build_var_Z: p must be at least 1, got $p"))
    p < n || throw(ArgumentError(
        "build_var_Z: p = $p leaves no observations from $n rows"))
    Tn = n - p
    Z = Matrix{Float64}(undef, Tn, K * p + ndet)
    @inbounds for j in 1:p
        off = (j - 1) * K
        for m in 1:K
            col = off + m
            for t in 1:Tn
                Z[t, col] = Y[p + t - j, m]
            end
        end
    end
    if ndet > 0
        c = K * p
        if deterministic === :const || deterministic === :both ||
           deterministic === :quadratic
            c += 1
            @inbounds for t in 1:Tn
                Z[t, c] = 1.0
            end
        end
        if deterministic !== :none && deterministic !== :const
            c += 1
            @inbounds for t in 1:Tn
                Z[t, c] = Float64(p + t)        # the CALENDAR index
            end
            if deterministic === :quadratic
                c += 1
                @inbounds for t in 1:Tn
                    Z[t, c] = Float64((p + t)^2)
                end
            end
        end
    end
    X = Matrix{Float64}(undef, Tn, K)
    @inbounds for m in 1:K, t in 1:Tn
        X[t, m] = Y[p + t, m]
    end
    return Z, X, _var_colnames(K, p, deterministic)
end

"""
    _var_colnames(K, p, deterministic; names=nothing) -> Vector{Symbol}

Column labels for `build_var_Z`'s `Z`, in its order:
`[:y1_l1, :y2_l1, …, :y1_l2, …, :const, :trend, :trend2]`.
"""
function _var_colnames(K::Integer, p::Integer, deterministic::Symbol;
                       names::Union{Nothing,AbstractVector{Symbol}}=nothing)
    nm = names === nothing ? [Symbol("y", i) for i in 1:K] : names
    out = Symbol[]
    for j in 1:p, v in nm
        push!(out, Symbol(v, "_l", j))
    end
    deterministic in (:const, :both, :quadratic) && push!(out, :const)
    deterministic in (:trend, :both, :quadratic) && push!(out, :trend)
    deterministic === :quadratic && push!(out, :trend2)
    return out
end

"""
    _var_ic(sigma_mle, T, K, p, ndet) -> (aic, bic, hqic, fpe)

The four information criteria, in both references' shared convention.

`[V]` Verified to `1e-12` against `vars::VARselect` and `statsmodels`'
`select_order` at `p = 1, 2, 3, 8`, with `nfree = Kp + ndet` **per
equation** and `Σ̃` the **maximum-likelihood** divisor — not the
df-adjusted one:

    aic  = log det Σ̃ + (2 / T)              · K · nfree
    bic  = log det Σ̃ + (log T / T)          · K · nfree
    hqic = log det Σ̃ + (2 log log T / T)    · K · nfree
    fpe  = ((T + nfree) / (T − nfree))^K · det Σ̃

`fpe` is the one criterion needing `det` rather than `logdet`, because
the reference reports the determinant itself; `[V]` on `econ5` its value
is of order `4.9e7`, which is comfortable in `Float64` but worth knowing.
"""
function _var_ic(sigma_mle::AbstractMatrix, T::Integer, K::Integer,
                 p::Integer, ndet::Integer)
    nfree = K * p + ndet
    npar = K * nfree
    ld = logdet(cholesky(Hermitian(Matrix(sigma_mle))))
    aic = ld + (2 / T) * npar
    bic = ld + (log(T) / T) * npar
    hqic = ld + (2 * log(log(T)) / T) * npar
    fpe = ((T + nfree) / (T - nfree))^K * det(sigma_mle)
    return aic, bic, hqic, fpe
end

"""
    fit_var(Y, p; deterministic=:const, method=:qr) -> VARModel

Fit a vector autoregression of order `p` by least squares.

`Y` is anything [`tsmatrix`](@ref) accepts — a matrix with **rows = time
and columns = variables**, a `NamedTuple` of columns (so
`fit_var(dataset("econ5"), 2)` works directly), or a collection of column
vectors. Variable names come from [`tsnames`](@ref) when the container
carries them.

    y_t = ν + Π t + A₁ y_{t-1} + … + A_p y_{t-p} + u_t

`deterministic` selects the deterministic terms and spans **both**
references' vocabularies, which are mutually incomplete; see
`_var_ndet`'s table. The spelling is deliberately neither R's `type=` nor
`statsmodels`' `trend=`, because both are ambiguous: R's `type="trend"`
means a trend with *no* constant, while `statsmodels`' `trend="ct"` means
constant *and* trend, and they differ by `6.15` in log-likelihood on the
bundled fixture.

## What it refuses

Both references silently accept several inputs that cannot produce a
meaningful fit. `[V]` `n = 8, K = 2, p = 5` — eleven parameters per
equation from three observations — returns without complaint from both.
This refuses, by name:

- `K = 1`. Both references refuse this too, and the univariate path is
  far better served by [`arx`](@ref) or [`fit_arima`](@ref).
- `p < 1`. `[V]` R errors obscurely (`length of 'dimnames' [2] not equal
  to array extent`); `statsmodels` does not error at all.
- `p` too large for the sample.
- `NaN` in the input. `[V]` R says `NAs in y.`; `statsmodels` raises
  `LinAlgError: SVD did not converge` and LAPACK writes to stderr.
- An unrecognised `deterministic`.

## Out of scope, stated rather than hidden

Exogenous regressors (`vars::VAR(exogen=)`, `statsmodels VAR(exog=)`),
restricted VARs (`vars::restrict`) and Bayesian shrinkage priors are not
implemented. The regressor construction here would carry exogenous
columns untouched, so it is a clean later addition rather than a
rewrite.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> m = fit_var(Y, 1);

julia> size(m.A[1])
(2, 2)

julia> round(m.loglik, digits=6)               # vars::logLik: -500.83136
-500.83136

julia> is_stable(m)
true
```

See also [`var_select`](@ref), [`roots`](@ref), [`forecast`](@ref).
"""
function fit_var(Y, p::Integer; deterministic::Symbol=:const, method::Symbol=:qr)
    Ym = Float64.(tsmatrix(Y))
    nms = tsnames(Y)
    n, K = size(Ym)
    ndet = _var_ndet(deterministic)        # validates `deterministic` first

    K >= 2 || throw(ArgumentError(
        "fit_var: a VAR needs at least two variables, got $K. Both references " *
        "refuse this too, and a single series is better served by arx or " *
        "fit_arima."))
    p >= 1 || throw(ArgumentError(
        "fit_var: p must be at least 1, got $p. (R errors obscurely here and " *
        "statsmodels does not error at all.)"))
    any(isnan, Ym) && throw(ArgumentError(
        "fit_var: the input contains NaN and there is no missing-data policy yet."))
    any(isinf, Ym) && throw(ArgumentError("fit_var: the input contains Inf."))

    Tn = n - p
    nfree = K * p + ndet
    Tn > nfree || throw(ArgumentError(
        "fit_var: not enough observations -- p=$p on $n rows leaves $Tn usable, " *
        "against $nfree parameters per equation. Reduce p, or supply more data. " *
        "(Both references accept this silently.)"))

    names = nms === nothing ? [Symbol("y", i) for i in 1:K] : copy(nms)
    length(names) == K || throw(DimensionMismatch(
        "fit_var: $(length(names)) names for $K variables"))

    Z, X, _ = build_var_Z(Ym, p, deterministic)
    B, E, se_zk, ZtZ_inv = _mols(Z, X; method=method)  # one factorization, K solves

    A = [Matrix{Float64}(transpose(view(B, ((j-1)*K+1):(j*K), :))) for j in 1:p]
    det_coef = ndet == 0 ? Matrix{Float64}(undef, K, 0) :
               Matrix{Float64}(transpose(view(B, (K*p+1):(K*p+ndet), :)))

    sigma_df = (E' * E) ./ (Tn - nfree)
    sigma_mle = (E' * E) ./ Tn
    loglik = -Tn * K / 2 * log(2pi) -
             Tn / 2 * logdet(cholesky(Hermitian(Matrix(sigma_mle)))) - Tn * K / 2
    vcov_full = kron(sigma_df, ZtZ_inv)
    fittedv = X - E

    return VARModel(Ym, names, Int(p), deterministic, A, det_coef,
                    sigma_df, sigma_mle, Matrix{Float64}(transpose(se_zk)),
                    vcov_full, E, fittedv, loglik, Tn, _companion(A), true)
end

"""
    var_select(Y; lag_max=10, deterministic=:const, parallel=true) -> VARSelection

Fit every VAR order from 1 to `lag_max` and report the four information
criteria, with the lag each one selects.

## The sample is held fixed, and that is the whole subtlety

`[V]` Both `vars::VARselect` and `statsmodels`' `select_order` estimate
**every** candidate `p` on the **same** `T = n - lag_max` observations,
so the criteria are comparable across `p`. A single `fit_var(Y, p)` uses
all `n - p` observations available to it, which is more. On
`var2_p1.csv` with `lag_max = 8`:

    var_select(Y; lag_max=8).aic[1]   = -0.6050354269   (T = 192)
    aic(fit_var(Y, 1))                = -0.5819716232   (T = 199)

Neither is wrong and both references report both, so this is a
difference to understand rather than a bug to find. If you want the
selection figure for one order, slice the data yourself:
`aic(fit_var(Y[lag_max - p + 1:end, :], p))` reproduces it exactly.

`[V]` Note also that `statsmodels`' `select_order(L)` returns **L+1**
values, for lags `0…L`, where `vars::VARselect(lag.max=L)` returns **L**,
for lags `1…L`. This returns `1…lag_max`, matching R; a lag-0 model is
not a VAR.

The candidates are independent fits, so they run on threads when Julia
has more than one. **The parallel and serial paths select the identical
model**, not merely an equally good one, which the test suite asserts.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var3_p2.csv"), ','; skipstart=1);

julia> s = var_select(Y; lag_max=8);

julia> s.selected.aic        # the series really is a VAR(2)
2

julia> s.T                   # held fixed at n - lag_max
242
```

See also [`fit_var`](@ref).
"""
function var_select(Y; lag_max::Integer=10, deterministic::Symbol=:const,
                    parallel::Bool=true, method::Symbol=:qr)
    Ym = Float64.(tsmatrix(Y))
    n, K = size(Ym)
    ndet = _var_ndet(deterministic)
    K >= 2 || throw(ArgumentError("var_select: a VAR needs at least two variables, got $K"))
    lag_max >= 1 || throw(ArgumentError("var_select: lag_max must be at least 1, got $lag_max"))
    any(isnan, Ym) && throw(ArgumentError("var_select: the input contains NaN"))
    Tfix = n - lag_max
    Tfix > K * lag_max + ndet || throw(ArgumentError(
        "var_select: not enough observations -- lag_max=$lag_max on $n rows " *
        "leaves $Tfix usable, against $(K * lag_max + ndet) parameters per " *
        "equation at the largest candidate. Reduce lag_max."))

    lags = collect(1:Int(lag_max))
    aics = Vector{Float64}(undef, lag_max)
    bics = similar(aics); hqics = similar(aics); fpes = similar(aics)

    function one!(j)
        # hold the sample fixed: every candidate sees exactly Tfix rows
        sub = view(Ym, (lag_max - j + 1):n, :)
        Z, X, _ = build_var_Z(sub, j, deterministic)
        _, E, _, _ = _mols(Z, X; method=method)
        sig = (E' * E) ./ Tfix
        a, b, h, f = _var_ic(sig, Tfix, K, j, ndet)
        aics[j] = a; bics[j] = b; hqics[j] = h; fpes[j] = f
        return nothing
    end

    if parallel && Threads.nthreads() > 1 && lag_max >= 4
        # A failure inside a threaded loop surfaces as a CompositeException of
        # TaskFailedExceptions. Unwrap it, so a rank-deficient input raises the
        # same ArgumentError on many threads as on one.
        try
            Threads.@threads for j in lags
                one!(j)
            end
        catch e
            inner = e
            while true
                if inner isa CompositeException && !isempty(inner.exceptions)
                    inner = first(inner.exceptions)
                elseif inner isa TaskFailedException
                    inner = inner.task.exception
                else
                    break
                end
            end
            throw(inner)
        end
    else
        for j in lags
            one!(j)
        end
    end

    sel = (aic = lags[argmin(aics)], bic = lags[argmin(bics)],
           hqic = lags[argmin(hqics)], fpe = lags[argmin(fpes)])
    return VARSelection(lags, aics, bics, hqics, fpes, sel, Tfix, deterministic)
end

"""
    companion(m::VARModel) -> Matrix{Float64}
    companion(A::AbstractVector{<:AbstractMatrix}) -> Matrix{Float64}

The `Kp × Kp` companion matrix: the VAR(p) restacked as a VAR(1) on `Kp`
states. Exported because the impulse-response recursion and the
structural decomposition both consume it.

On a fitted model this returns the matrix cached at fit time. Given the
coefficient matrices directly it builds one. At `p = 1` the companion
matrix **is** `A[1]` — there is no identity block.
"""
companion(m::VARModel) = m.C
companion(A::AbstractVector{<:AbstractMatrix}) = _companion(A)

"""
    roots(m::VARModel) -> Vector{Float64}

The **moduli of the companion-matrix eigenvalues**, largest first. The
process is stable when every one is below 1.

!!! warning "The two references mean different things by `roots`"
    `[V]` This is the stage's most dangerous convention trap, because
    getting it wrong silently *inverts* a stability test.

    `vars::roots()` returns these companion moduli (stable if `< 1`).
    `statsmodels`' `.roots` returns the roots of the characteristic
    polynomial (stable if `|·| > 1`). They are exact reciprocals —
    verified on five fixtures with `max|R − 1/py|` between `1.1e-16` and
    `9.3e-15`.

    This package follows R, because "the largest root is 0.97, close to
    the boundary" is the reading practitioners expect. To compare against
    `statsmodels`, take `1 ./ roots(m)`.

See also [`is_stable`](@ref), [`companion`](@ref).
"""
roots(m::VARModel) = sort!(abs.(eigvals(companion(m))); rev=true)

"""
    is_stable(m::VARModel) -> Bool

Whether every companion eigenvalue modulus is below 1 — see
[`roots`](@ref) for why the comparison runs that way round and not the
other.
"""
is_stable(m::VARModel) = maximum(roots(m)) < 1

"""
    _ma_rep(A, h) -> Vector{Matrix{Float64}}

The first `h+1` matrices of the moving-average representation,
`Φ₀ = I` and `Φ_i = Σ_{j=1}^{min(i,p)} Φ_{i-j} A_j`.

Shared between the forecast error variance accumulation here and the
impulse responses in 10.3, so it is written once.
"""
function _ma_rep(A::AbstractVector{<:AbstractMatrix}, h::Integer)
    p = length(A)
    K = size(A[1], 1)
    Phi = [zeros(Float64, K, K) for _ in 0:h]
    Phi[1] .= I(K)
    @inbounds for i in 1:h
        acc = Phi[i+1]
        for j in 1:min(i, p)
            mul!(acc, Phi[i-j+1], A[j], 1.0, 1.0)
        end
    end
    return Phi
end

"""
    forecast(m::VARModel, h; levels=[80.0, 95.0]) -> VARForecast
    predict(m::VARModel, h; levels=[80.0, 95.0]) -> VARForecast

Forecast `h` steps ahead from a fitted [`VARModel`](@ref).

The point forecast is the deterministic recursion

    ŷ_{n+h} = d_{n+h}'Θ + Σ_{j=1}^{p} A_j ŷ_{n+h-j}      (ŷ = y for h ≤ 0)

and the error covariance accumulates through the moving-average
representation, `Σ_h = Σ_{i=0}^{h-1} Φ_i Σ Φ_i'`.

`[V]` Both references use the **df-adjusted** `Σ` here and **ignore
parameter estimation uncertainty** — confirmed because `forecast_cov(1)`
equals `sigma_u` element for element, and R's `predict` intervals match
`statsmodels`' `forecast_interval` to `1.1e-15`. This package follows
them, so the intervals are conditional on the estimated coefficients and
are therefore slightly too narrow — the same honesty as the
[`fit_ets`](@ref) `sigma2` note.

Deterministic terms are evaluated at the forecast dates: the trend
column continues as `n+1, n+2, …`, matching the calendar indexing the fit
used.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> f = forecast(fit_var(Y, 1), 5);

julia> size(f.point)
(5, 2)

julia> size(f.lower)
(5, 2, 2)
```

See also [`fit_var`](@ref), [`VARForecast`](@ref).
"""
function StatsAPI.predict(m::VARModel, h::Integer;
                          levels::Vector{<:Real}=[80.0, 95.0])
    h >= 1 || throw(ArgumentError("forecast(::VARModel): horizon must be >= 1, got $h"))
    isempty(levels) && throw(ArgumentError("levels must be non-empty"))
    all(0 .< levels .< 100) || throw(ArgumentError("levels must be in (0, 100)"))

    K = length(m.names)
    n = size(m.Y, 1)
    p = m.p
    point = Matrix{Float64}(undef, h, K)
    hist = Matrix{Float64}(undef, p + h, K)       # the last p actuals, then forecasts
    @inbounds for m_ in 1:K, t in 1:p
        hist[t, m_] = m.Y[n - p + t, m_]
    end
    yh = Vector{Float64}(undef, K)
    @inbounds for s in 1:h
        fill!(yh, 0.0)
        # deterministic part, evaluated at the forecast date n + s
        if m.deterministic !== :none
            c = 0
            if m.deterministic !== :trend
                c += 1
                for i in 1:K
                    yh[i] += m.det_coef[i, c]
                end
            end
            if m.deterministic !== :const
                c += 1
                tt = Float64(n + s)
                for i in 1:K
                    yh[i] += m.det_coef[i, c] * tt
                end
                if m.deterministic === :quadratic
                    c += 1
                    for i in 1:K
                        yh[i] += m.det_coef[i, c] * tt^2
                    end
                end
            end
        end
        for j in 1:p
            mul!(yh, m.A[j], view(hist, p + s - j, :), 1.0, 1.0)
        end
        for i in 1:K
            hist[p + s, i] = yh[i]
            point[s, i] = yh[i]
        end
    end

    Phi = _ma_rep(m.A, h - 1)
    se = Matrix{Float64}(undef, h, K)
    acc = zeros(Float64, K, K)
    tmp = Matrix{Float64}(undef, K, K)
    @inbounds for s in 1:h
        mul!(tmp, Phi[s], m.sigma_df)
        mul!(acc, tmp, transpose(Phi[s]), 1.0, 1.0)
        for i in 1:K
            se[s, i] = sqrt(max(acc[i, i], 0.0))
        end
    end

    zs = [_confidence_z(1 - l / 100) for l in levels]
    L = length(levels)
    lower = Array{Float64,3}(undef, h, K, L)
    upper = Array{Float64,3}(undef, h, K, L)
    @inbounds for q in 1:L, i in 1:K, s in 1:h
        lower[s, i, q] = point[s, i] - zs[q] * se[s, i]
        upper[s, i, q] = point[s, i] + zs[q] * se[s, i]
    end
    return VARForecast(point, se, Float64.(levels), lower, upper,
                       copy(m.names), Int(h), "VAR($(m.p))")
end

forecast(m::VARModel, h::Integer; levels::Vector{<:Real}=[80.0, 95.0]) =
    StatsAPI.predict(m, h; levels=levels)

# --------------------------------------------------------------------------
# StatsAPI contract
# --------------------------------------------------------------------------

"""
    coef(m::VARModel) -> Matrix{Float64}

The stacked coefficients, `K × (Kp + ndet)`, in this package's own order:
lag blocks first, deterministic terms last — R's order, since it reads
naturally. [`coefnames`](@ref) gives the matching column labels.
"""
StatsAPI.coef(m::VARModel) = hcat(reduce(hcat, m.A), m.det_coef)

"""
    coefnames(m::VARModel) -> Vector{Symbol}

Column labels for [`coef`](@ref): `[:y1_l1, :y2_l1, …, :const, :trend]`.
"""
StatsAPI.coefnames(m::VARModel) =
    _var_colnames(length(m.names), m.p, m.deterministic; names=m.names)

StatsAPI.vcov(m::VARModel) = m.vcov_full
StatsAPI.stderror(m::VARModel) = m.se
StatsAPI.residuals(m::VARModel) = m.resid
StatsAPI.fitted(m::VARModel) = m.fitted
StatsAPI.nobs(m::VARModel) = m.T
StatsAPI.loglikelihood(m::VARModel) = m.loglik
StatsAPI.dof(m::VARModel) = length(m.names) * (length(m.names) * m.p +
                                               _var_ndet(m.deterministic))

"""
    aic(m::VARModel)

The information criteria of a **fitted** model, computed on its own
`T = n - p` observations — which is what both references report for a
fitted model, and which differs from [`var_select`](@ref)'s figure for
the same order because that one holds the sample fixed. See
[`var_select`](@ref).
"""
StatsAPI.aic(m::VARModel) =
    _var_ic(m.sigma_mle, m.T, length(m.names), m.p, _var_ndet(m.deterministic))[1]
StatsAPI.bic(m::VARModel) =
    _var_ic(m.sigma_mle, m.T, length(m.names), m.p, _var_ndet(m.deterministic))[2]

"""
    hqic(m::VARModel)
    fpe(m::VARModel)

The Hannan-Quinn criterion and the final prediction error, the two
criteria `StatsAPI` has no generic for. Both references report both.
"""
hqic(m::VARModel) =
    _var_ic(m.sigma_mle, m.T, length(m.names), m.p, _var_ndet(m.deterministic))[3]
fpe(m::VARModel) =
    _var_ic(m.sigma_mle, m.T, length(m.names), m.p, _var_ndet(m.deterministic))[4]

function Base.show(io::IO, m::VARModel)
    K = length(m.names)
    println(io, "VAR(", m.p, ")  K=", K, "  T=", m.T,
            "  deterministic=:", m.deterministic)
    println(io, "  variables: ", join(string.(m.names), ", "))
    r = roots(m)
    println(io, "  companion roots (moduli, max first): ",
            join(string.(round.(r[1:min(end, 4)], digits=6)), ", "),
            length(r) > 4 ? ", …" : "", is_stable(m) ? "   stable" : "   NOT STABLE")
    print(io, "  logLik = ", round(m.loglik, digits=4),
          "   AIC = ", round(StatsAPI.aic(m), digits=6),
          "   BIC = ", round(StatsAPI.bic(m), digits=6))
end

function Base.show(io::IO, s::VARSelection)
    println(io, "VAR lag-order selection  (T=", s.T,
            " held fixed, deterministic=:", s.deterministic, ")")
    println(io, "  lag      AIC         BIC        HQIC         FPE")
    for (i, l) in enumerate(s.lags)
        println(io, lpad(l, 5), "  ", lpad(round(s.aic[i], digits=6), 11),
                " ", lpad(round(s.bic[i], digits=6), 11),
                " ", lpad(round(s.hqic[i], digits=6), 11),
                " ", lpad(round(s.fpe[i], digits=8), 13))
    end
    print(io, "  selected: AIC ", s.selected.aic, "   BIC ", s.selected.bic,
          "   HQIC ", s.selected.hqic, "   FPE ", s.selected.fpe)
end

function Base.show(io::IO, f::VARForecast)
    println(io, f.model_name, " forecast  horizon=", f.horizon,
            "  K=", length(f.names), "  levels=", f.levels)
    println(io, "  variables: ", join(string.(f.names), ", "))
    hh = min(f.horizon, 5)
    for s in 1:hh
        println(io, "   h=", s, "  ",
                join([string(round(f.point[s, i], digits=6)) for i in 1:length(f.names)], "  "))
    end
    f.horizon > hh && print(io, "   …")
end
