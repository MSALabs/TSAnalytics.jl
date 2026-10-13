export portmanteau_test, normality_test, bg_test, es_test, var_diagnostics
export PortmanteauTest, MultivariateNormalityTest, MultivariateARCHLMTest,
       BreuschGodfreyTest, EdgertonShukurTest

# ---------------------------------------------------------------------------
# Stage 10.8 -- multivariate residual diagnostics for a fitted VAR.
#
# Primary source: Lütkepohl (2005), "New Introduction to Multiple Time Series
# Analysis", §4.4 (portmanteau and LM tests for autocorrelation, including
# the Edgerton-Shukur F form) and §4.5 (multivariate Jarque-Bera); the
# multivariate ARCH-LM test is his §16.5.1. vars::serial.test / normality.test
# / arch.test and statsmodels' test_whiteness / test_normality are the
# numerical references.
#
# Every p-value comes from this package's own upper-tail functions. vars
# computes all ten of its diagnostic p-values as 1 - CDF, which underflows to
# exactly zero on ordinary data; its statistics are right, its tails are not.
# ---------------------------------------------------------------------------

"""
    PortmanteauTest <: HypothesisTest

Multivariate portmanteau test for residual autocorrelation, returned by
[`portmanteau_test`](@ref). `adjusted` records which form; `df = K²(lags − p)`.
"""
struct PortmanteauTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    df::Int
    lags::Int
    adjusted::Bool
    nobs::Int
    K::Int
end

"""
    portmanteau_test(m::VARModel; lags, adjusted=false) -> PortmanteauTest

Hosking's multivariate portmanteau test of the null that the VAR's
residual vector is white noise up to lag `lags`:

    Q_h = T Σ_{i=1}^{h} tr(C_i' C_0⁻¹ C_i C_0⁻¹),     C_i = T⁻¹ Σ_t u_t u_{t-i}'

asymptotically χ² with `K²(h − p)` degrees of freedom (Lütkepohl 2005,
§4.4.1). `adjusted=true` weights lag `i` by `T/(T − i)`, the small-sample
form; every weight exceeds one, so the adjusted statistic is always the
larger. `[V]` Both forms match `vars::serial.test` (`"PT.asymptotic"`,
`"PT.adjusted"`) and `statsmodels`' `test_whiteness(adjusted=…)` to
`≤ 4.2e-12` on seven fixtures.

`C_0` is factorized once and each lag costs two triangular solves:
`tr(C_i' C_0⁻¹ C_i C_0⁻¹) = ‖L⁻¹ C_i L⁻ᵀ‖²_F` with `C_0 = LLᵀ`.

**Refusals.** `lags` must exceed `p`, or the degrees of freedom are not
positive — `[V]` `vars` runs anyway and reports `df ≤ 0`, while
`statsmodels` refuses correctly. And `lags` must leave observations:
`statsmodels` accepts `lags = 300` on `T = 198` and reports `df = 1192`.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> t = portmanteau_test(fit_var(Y, 2); lags=8);

julia> round(t.statistic; digits=6), t.df
(14.920132, 24)
```

See also [`var_diagnostics`](@ref), [`bg_test`](@ref).
"""
function portmanteau_test(m::VARModel; lags::Integer, adjusted::Bool=false)
    K, p, T = length(m.names), m.p, m.T
    lags > p || throw(ArgumentError(
        "portmanteau_test: lags must exceed the VAR order p = $p, got $lags -- " *
        "the degrees of freedom K²(lags - p) would not be positive. (vars runs " *
        "anyway and reports df <= 0.)"))
    lags < T || throw(ArgumentError(
        "portmanteau_test: lags = $lags leaves no observations at T = $T. " *
        "(statsmodels accepts this and reports a statistic.)"))
    E = m.resid
    C0 = (E' * E) ./ T
    L = cholesky(Hermitian(C0)).L
    Ci = Matrix{Float64}(undef, K, K)
    q = 0.0
    for i in 1:lags
        mul!(Ci, transpose(view(E, (i + 1):T, :)), view(E, 1:(T - i), :))
        Ci ./= T
        M = L \ Ci
        M = transpose(L \ transpose(M))             # L⁻¹ C_i L⁻ᵀ
        tr_i = sum(abs2, M)
        q += adjusted ? tr_i * T / (T - i) : tr_i
    end
    stat = T * q
    df = K^2 * (lags - p)
    return PortmanteauTest(stat, _chisq_ccdf(stat, df), df, Int(lags), adjusted, T, K)
end

function Base.show(io::IO, t::PortmanteauTest)
    println(io, t.adjusted ? "Portmanteau test (adjusted, Hosking)" :
                             "Portmanteau test (asymptotic, Hosking)")
    println(io, "  lags          : ", t.lags)
    println(io, "  K, T          : ", t.K, ", ", t.nobs)
    println(io, "  statistic     : ", round(t.statistic, digits=4), "  (df = ", t.df, ")")
    print(io,   "  p-value       : ", _fmt_p(t.pvalue))
end

"""
    MultivariateNormalityTest <: HypothesisTest

Multivariate Jarque-Bera test, returned by [`normality_test`](@ref): the
joint statistic (`df = 2K`) and its skewness and kurtosis halves (each
`df = K`), which sum to it exactly. `univariate` holds a
[`JarqueBeraTest`](@ref) per equation when requested.
"""
struct MultivariateNormalityTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    df::Int
    skewness_statistic::Float64
    skewness_pvalue::Float64
    skewness_df::Int
    kurtosis_statistic::Float64
    kurtosis_pvalue::Float64
    kurtosis_df::Int
    univariate::Union{Nothing,Dict{Symbol,JarqueBeraTest}}
    orthogonalization::Symbol
    nobs::Int
end

"""
    standardized_residuals(m::VARModel) -> Matrix{Float64}

The VAR's residuals, centred and orthogonalized by the Cholesky factor of
their own covariance: `w_t = L⁻¹(u_t − ū)` with `LLᵀ = T⁻¹ Σ (u_t − ū)(u_t − ū)ᵀ`.
Their sample covariance is the identity. This is what the multivariate
Jarque-Bera test computes moments of.
"""
function standardized_residuals(m::VARModel)
    Ec = m.resid .- mean(m.resid; dims=1)
    S = (Ec' * Ec) ./ size(Ec, 1)
    L = cholesky(Hermitian(S)).L
    return Matrix(transpose(L \ transpose(Ec)))
end

"""
    normality_test(m::VARModel; univariate=false, orthogonalization=:cholesky)
        -> MultivariateNormalityTest

Multivariate Jarque-Bera test of the null that the VAR's residuals are
jointly normal (Lütkepohl 2005, §4.5). With `w_t` the
standardized residuals (`TSAnalytics.standardized_residuals`), `b₁` and `b₂` the vectors of their third
and fourth sample moments:

    s₃ = T b₁'b₁ / 6               (skewness half, df K)
    s₄ = T (b₂ − 3)'(b₂ − 3) / 24  (kurtosis half, df K)
    JB = s₃ + s₄                   (df 2K)

`[V]` Matches `vars::normality.test` and `statsmodels`' `test_normality`
to `≤ 1.1e-11` on seven fixtures; the two halves, which only `vars`
reports, sum to the joint statistic exactly — the identity the test suite
leans on, since it is the only independent check available for them.

The statistic is **not** invariant to how the residuals are
orthogonalized. Only the Cholesky standardization is implemented, which
is both references' default; the variable order therefore matters, as it
does for orthogonalized impulse responses. Any other `orthogonalization`
is refused rather than silently ignored.

`univariate=true` adds the ordinary [`jarque_bera_test`](@ref) of each
equation's residuals, keyed by variable name, which says *which*
equation fails. `[V]` Those match `vars`' per-equation results.

**p-values.** `[V]` `vars` reports `0.000000e+00` for `uschange`'s
`JB = 282.12` (df 10), whose true tail is `9.28e-55`; this package's tail
function returns the latter. Where the true tail is below
`floatmin(Float64) ≈ 2.2e-308` — `climhyd`'s `JB = 1850.9` on 12 df is
about `1e-391` — no `Float64` can hold it and `0.0` is the honest answer.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> t = normality_test(fit_var(Y, 2));

julia> round(t.statistic; digits=6), t.df
(1.940443, 4)

julia> t.skewness_statistic + t.kurtosis_statistic ≈ t.statistic
true
```

See also [`var_diagnostics`](@ref).
"""
function normality_test(m::VARModel; univariate::Bool=false,
                        orthogonalization::Symbol=:cholesky)
    orthogonalization === :cholesky || throw(ArgumentError(
        "normality_test: orthogonalization must be :cholesky, got :$orthogonalization. " *
        "It is the references' default and the only one implemented; the " *
        "statistic depends on the choice, so another is not silently substituted."))
    K, T = length(m.names), m.T
    W = standardized_residuals(m)
    b1 = vec(mean(W .^ 3; dims=1))
    b2 = vec(mean(W .^ 4; dims=1))
    s3 = T * sum(abs2, b1) / 6
    s4 = T * sum(abs2, b2 .- 3) / 24
    jb = s3 + s4
    uni = nothing
    if univariate
        uni = Dict{Symbol,JarqueBeraTest}(m.names[i] => jarque_bera_test(m.resid[:, i])
                                          for i in 1:K)
    end
    return MultivariateNormalityTest(jb, _chisq_ccdf(jb, 2K), 2K,
                                     s3, _chisq_ccdf(s3, K), K,
                                     s4, _chisq_ccdf(s4, K), K,
                                     uni, :cholesky, T)
end

function Base.show(io::IO, t::MultivariateNormalityTest)
    println(io, "Multivariate Jarque-Bera normality test (Cholesky-standardized residuals)")
    println(io, "  joint         : ", round(t.statistic, digits=4), "  (df = ", t.df,
            ")  p = ", _fmt_p(t.pvalue))
    println(io, "  skewness      : ", round(t.skewness_statistic, digits=4), "  (df = ",
            t.skewness_df, ")  p = ", _fmt_p(t.skewness_pvalue))
    print(io,   "  kurtosis      : ", round(t.kurtosis_statistic, digits=4), "  (df = ",
            t.kurtosis_df, ")  p = ", _fmt_p(t.kurtosis_pvalue))
    if t.univariate !== nothing
        for (k, u) in sort!(collect(t.univariate); by=first)
            print(io, "\n    ", rpad(string(k), 12), " JB = ", round(u.statistic, digits=4),
                  "  p = ", _fmt_p(u.pvalue))
        end
    end
end

"""
    MultivariateARCHLMTest <: HypothesisTest

Multivariate ARCH-LM test, returned by `arch_lm_test(::VARModel)`.
`df = lags · (K(K+1)/2)²`; `univariate` holds the per-equation
[`ARCHLMTest`](@ref)s when requested. A separate type from the univariate
`ARCHLMTest` because it carries different fields, not because it is a
different idea.
"""
struct MultivariateARCHLMTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    df::Int
    lags::Int
    nobs::Int
    K::Int
    univariate::Union{Nothing,Dict{Symbol,ARCHLMTest}}
end

"""
    arch_lm_test(m::VARModel; lags=5, univariate=false) -> MultivariateARCHLMTest

Multivariate ARCH-LM test for conditional heteroskedasticity in a VAR's
residuals (Lütkepohl 2005, §16.5.1): regress `vech(u_t u_t')` on a
constant and `lags` of its own lags, and form

    LM = ½ T K(K+1) R²_m,    R²_m = 1 − 2/(K(K+1)) · tr(Ω̂ Ω̂₀⁻¹)

with `Ω̂` the auxiliary regression's residual covariance, `Ω̂₀` the
covariance of `vech(u_t u_t')` about its mean, and `T` the auxiliary
sample. Asymptotically χ² with `lags · (K(K+1)/2)²` degrees of freedom.
`[V]` Matches `vars::arch.test`'s multivariate statistic on five fixtures.

That degrees-of-freedom count grows as `K⁴`: at `K = 6` the auxiliary
regression has 21 responses and `21·lags + 1` regressors, so it runs out
of observations quickly. An infeasible regression is **refused** by name
— `[V]` `vars` does not check. The regression is one multi-response
least-squares solve (`_mols`), not 21.

`univariate=true` adds the ordinary [`arch_lm_test`](@ref) of each
equation's residuals at the same `lags`.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> t = arch_lm_test(fit_var(Y, 2); lags=5);

julia> round(t.statistic; digits=6), t.df
(53.369468, 45)
```
"""
function arch_lm_test(m::VARModel; lags::Integer=5, univariate::Bool=false)
    lags >= 1 || throw(ArgumentError("arch_lm_test: lags must be at least 1, got $lags"))
    K, Tm = length(m.names), m.T
    nv = K * (K + 1) ÷ 2
    vidx = _vech_indices(K)
    V = Matrix{Float64}(undef, Tm, nv)
    @inbounds for t in 1:Tm, (r, (i, j)) in enumerate(vidx)
        V[t, r] = m.resid[t, i] * m.resid[t, j]
    end
    Ta = Tm - lags
    nreg = 1 + lags * nv
    Ta > nreg || throw(ArgumentError(
        "arch_lm_test: not enough observations -- the auxiliary regression has " *
        "$nv responses and $nreg regressors at lags = $lags, but only $Ta " *
        "observations remain. Reduce lags. (vars does not check.)"))
    Z = Matrix{Float64}(undef, Ta, nreg)
    Z[:, 1] .= 1.0
    for l in 1:lags, c in 1:nv
        @views Z[:, 1 + (l - 1) * nv + c] .= V[(lags + 1 - l):(Tm - l), c]
    end
    X = V[(lags + 1):Tm, :]
    _, Eaux, _, _ = _mols(Z, X)
    Om = (Eaux' * Eaux) ./ Ta
    Xc = X .- mean(X; dims=1)
    Om0 = (Xc' * Xc) ./ Ta
    R2 = 1 - (2 / (K * (K + 1))) * tr(Om / Om0)
    stat = 0.5 * Ta * K * (K + 1) * R2
    df = lags * nv^2
    uni = nothing
    if univariate
        uni = Dict{Symbol,ARCHLMTest}(m.names[i] => arch_lm_test(m.resid[:, i], lags)
                                      for i in 1:K)
    end
    return MultivariateARCHLMTest(stat, _chisq_ccdf(stat, df), df, Int(lags), Ta, K, uni)
end

arch_lm_test(m::VARModel, lags::Integer; univariate::Bool=false) =
    arch_lm_test(m; lags=lags, univariate=univariate)

function Base.show(io::IO, t::MultivariateARCHLMTest)
    println(io, "Multivariate ARCH-LM test")
    println(io, "  lags          : ", t.lags)
    println(io, "  K, T          : ", t.K, ", ", t.nobs)
    println(io, "  statistic     : ", round(t.statistic, digits=4), "  (df = ", t.df, ")")
    print(io,   "  p-value       : ", _fmt_p(t.pvalue))
    if t.univariate !== nothing
        for (k, u) in sort!(collect(t.univariate); by=first)
            print(io, "\n    ", rpad(string(k), 12), " LM = ", round(u.statistic, digits=4),
                  "  p = ", _fmt_p(u.pvalue))
        end
    end
end

"""
    BreuschGodfreyTest <: HypothesisTest

Breusch-Godfrey LM test for residual autocorrelation in a VAR, returned
by [`bg_test`](@ref). `df = lags · K²`.
"""
struct BreuschGodfreyTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    df::Int
    lags::Int
    nobs::Int
end

"""
    EdgertonShukurTest <: HypothesisTest

Edgerton-Shukur F test for residual autocorrelation in a VAR, returned
by [`es_test`](@ref), with degrees of freedom `(df1, df2)`.
"""
struct EdgertonShukurTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    df1::Int
    df2::Int
    lags::Int
    nobs::Int
end

"""
    _bg_aux(m, lags) -> (Σ̃_u, Σ̃_e, T, nz)

The Breusch-Godfrey auxiliary regression (Lütkepohl 2005, §4.4.4): the
VAR residuals on the VAR's own regressors plus `lags` lags of the
residuals, presample residuals set to zero. Returns the original and the
auxiliary residual covariances (both divided by `T`, so the divisor
cancels in every ratio that uses them) and the number of original
regressors per equation.
"""
function _bg_aux(m::VARModel, lags::Integer, who::AbstractString)
    lags >= 1 || throw(ArgumentError("$who: lags must be at least 1, got $lags"))
    K, T = length(m.names), m.T
    Z0, _, _ = build_var_Z(m.Y, m.p, m.deterministic)
    nz = size(Z0, 2)
    T > nz + K * lags || throw(ArgumentError(
        "$who: not enough observations -- the auxiliary regression needs " *
        "$(nz + K * lags) regressors per equation at lags = $lags, but T = $T. " *
        "Reduce lags."))
    Z = hcat(Z0, zeros(Float64, T, K * lags))
    U = m.resid
    for l in 1:lags, c in 1:K
        @views Z[(l + 1):T, nz + (l - 1) * K + c] .= U[1:(T - l), c]
    end
    _, Ee, _, _ = _mols(Z, U)
    return (U' * U) ./ T, (Ee' * Ee) ./ T, T, nz
end

"""
    bg_test(m::VARModel; lags=5) -> BreuschGodfreyTest

Breusch-Godfrey LM test of the null of no residual autocorrelation up to
lag `lags` (Lütkepohl 2005, §4.4.4):

    LM_h = T (K − tr(Σ̃_u⁻¹ Σ̃_e))

with `Σ̃_u` the VAR's residual covariance and `Σ̃_e` that of the auxiliary
regression of the residuals on the original regressors and `lags` of
their own lags (presample values zero). Asymptotically χ²(`lags·K²`).
`[V]` Matches `vars::serial.test(type="BG")` on five fixtures; there is
no `statsmodels` counterpart.

`[V]` `vars` reports `p = 0.000000e+00` for `climhyd`'s `LM = 321.6` on
108 df through its `1 − CDF` computation; the tail is representable and
is reported here. See [`es_test`](@ref) for the small-sample F form.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> t = bg_test(fit_var(Y, 2); lags=5);

julia> round(t.statistic; digits=6), t.df
(15.553824, 20)
```
"""
function bg_test(m::VARModel; lags::Integer=5)
    Su, Se, T, _ = _bg_aux(m, lags, "bg_test")
    K = length(m.names)
    stat = T * (K - tr(Su \ Se))
    df = Int(lags) * K^2
    return BreuschGodfreyTest(stat, _chisq_ccdf(stat, df), df, Int(lags), T)
end

"""
    es_test(m::VARModel; lags=5) -> EdgertonShukurTest

Edgerton-Shukur's small-sample F form of the [`bg_test`](@ref) statistic
(Lütkepohl 2005, eq. 4.4.23), a Rao-type F approximation:

    R²_r = 1 − |Σ̃_e| / |Σ̃_u|,     m = K·lags,   n = regressors per equation
    r    = √((K²m² − 4) / (K² + m² − 5)),        q = Km/2 − 1
    N    = T − n − m − (K − m + 1)/2
    LMF  = (1 − (1 − R²_r)^{1/r}) / (1 − R²_r)^{1/r} · (N r − q) / (K m)

with degrees of freedom `(lags·K², ⌊N r − q⌋)`. The denominator degrees
of freedom are the one quantity here with a single reference, and they
change the p-value without changing the statistic. `[V]` The formula
reproduces all five of `vars`' recorded pairs: `(20, 364)`, `(45, 666)`,
`(100, 736)`, `(108, 2391)` and `(8, 36)`.

`[V]` `vars` reports `p = 0` for `climhyd`'s `F = 3.73` on `(108, 2391)`
— the tail is about `1e-31`, and is reported here.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> t = es_test(fit_var(Y, 2); lags=5);

julia> round(t.statistic; digits=6), (t.df1, t.df2)
(0.746057, (20, 364))
```
"""
function es_test(m::VARModel; lags::Integer=5)
    Su, Se, T, nz = _bg_aux(m, lags, "es_test")
    K = length(m.names)
    mm = K * Int(lags)
    R2 = 1 - det(Se) / det(Su)
    r = sqrt((K^2 * mm^2 - 4) / (K^2 + mm^2 - 5))
    q = K * mm / 2 - 1
    N = T - nz - mm - (K - mm + 1) / 2
    w = (1 - R2)^(1 / r)
    stat = (1 - w) / w * (N * r - q) / (K * mm)
    df1 = Int(lags) * K^2
    df2 = floor(Int, N * r - q)
    return EdgertonShukurTest(stat, _f_ccdf(stat, df1, df2), df1, df2, Int(lags), T)
end

function Base.show(io::IO, t::BreuschGodfreyTest)
    println(io, "Breusch-Godfrey LM test for residual autocorrelation (VAR)")
    println(io, "  lags          : ", t.lags)
    println(io, "  statistic     : ", round(t.statistic, digits=4), "  (df = ", t.df, ")")
    print(io,   "  p-value       : ", _fmt_p(t.pvalue))
end

function Base.show(io::IO, t::EdgertonShukurTest)
    println(io, "Edgerton-Shukur F test for residual autocorrelation (VAR)")
    println(io, "  lags          : ", t.lags)
    println(io, "  F statistic   : ", round(t.statistic, digits=4),
            "  (df = ", t.df1, ", ", t.df2, ")")
    print(io,   "  p-value       : ", _fmt_p(t.pvalue))
end

"""
    var_diagnostics(m::VARModel; pt_lags=nothing, bg_lags=5, arch_lags=5) -> NamedTuple

The standard residual checks for a fitted VAR in one call:

- `portmanteau`: [`portmanteau_test`](@ref), adjusted form
- `bg`, `es`: [`bg_test`](@ref) and [`es_test`](@ref) at `bg_lags`
- `normality`: [`normality_test`](@ref), with the per-equation tests
- `arch`: `arch_lm_test(m; lags=arch_lags)`, with the per-equation tests

`pt_lags` defaults to `max(p + 1, min(10, T ÷ 4))`. Neither reference has
a sensible default, and the portmanteau test **needs `lags > p`**; its
power depends on the choice, so state it when reporting the result.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> d = var_diagnostics(fit_var(Y, 2));

julia> keys(d)
(:portmanteau, :bg, :es, :normality, :arch)

julia> d.portmanteau.lags
10
```
"""
function var_diagnostics(m::VARModel; pt_lags::Union{Nothing,Integer}=nothing,
                         bg_lags::Integer=5, arch_lags::Integer=5)
    ptl = pt_lags === nothing ? max(m.p + 1, min(10, m.T ÷ 4)) : pt_lags
    return (portmanteau = portmanteau_test(m; lags=ptl, adjusted=true),
            bg = bg_test(m; lags=bg_lags),
            es = es_test(m; lags=bg_lags),
            normality = normality_test(m; univariate=true),
            arch = arch_lm_test(m; lags=arch_lags, univariate=true))
end
