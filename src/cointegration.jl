export coint_test, CointTest

# ---------------------------------------------------------------------------
# Stage 10.4a -- the Engle-Granger residual-based cointegration test.
#
# Primary sources: Engle & Granger (1987), Econometrica 55(2), for the
# two-step procedure; MacKinnon (1994, 2010) for the response surfaces,
# whose N > 1 rows are the cointegration case. statsmodels' `coint` is the
# numerical reference; urca has no Engle-Granger test of its own, so the
# second reference is the procedure hand-built from `lm` + `ur.df` in
# test/verification/coint/eg_fixed.R, which agrees exactly at a fixed lag.
# ---------------------------------------------------------------------------

const _COINT_TREND_REG = Dict(:none => :n, :const => :c, :trend => :ct, :quadratic => :ctt)

"""
    CointTest <: HypothesisTest

The Engle-Granger cointegration test returned by [`coint_test`](@ref).

- `statistic`: the ADF `t` statistic on the cointegrating residual.
- `pvalue`: the MacKinnon response-surface p-value, or `nothing` when the
  tables do not cover the case — then `pvalue(t)` throws, naming
  [`critical_values`](@ref). `pvalue_available` says which.
- `critical_values`: `(levels = (1.0, 5.0, 10.0), values = (…))`.
- `N` (number of series), `trend`, `lags` (the lag actually used),
  `autolag`, `nobs` (series length).
- `beta`: the cointegrating regression's coefficients, deterministic terms
  first — `[const, x₂, …]` for `trend = :const`; `resid`: the residual the
  ADF regression ran on.
"""
struct CointTest <: HypothesisTest
    statistic::Float64
    pvalue::Union{Nothing,Float64}
    critical_values::NamedTuple{(:levels, :values),Tuple{NTuple{3,Float64},NTuple{3,Float64}}}
    N::Int
    trend::Symbol
    lags::Int
    autolag::Union{Nothing,Symbol}
    nobs::Int
    beta::Vector{Float64}
    resid::Vector{Float64}
    pvalue_available::Bool
end

function pvalue(t::CointTest)
    t.pvalue === nothing && throw(ArgumentError(
        "pvalue: MacKinnon's p-value surfaces cover at most 6 series and this " *
        "test has N = $(t.N). The critical values exist to N = 12 -- use " *
        "critical_values(t)."))
    return t.pvalue
end

"""
    coint_test(Y; trend=:const, lags=nothing, autolag=:aic) -> CointTest

The Engle-Granger (1987) two-step test of the null of **no
cointegration**: regress the **first column** of `Y` on the others (and
the deterministic terms), then run an augmented Dickey-Fuller test with no
deterministic terms on that regression's residual. A residual with no unit
root means some linear combination of the series is stationary.

!!! warning "The first column is the dependent variable"
    The statistic is not symmetric in the ordering of the columns —
    swapping them gives a different number, and can give a different
    conclusion. Put the variable you would normalize the cointegrating
    relation on first.

`trend` sets the deterministic terms of the cointegrating regression:
`:none`, `:const` (default), `:trend` (constant and linear trend) or
`:quadratic` (constant, trend and trend²) — `statsmodels`' `"n"`, `"c"`,
`"ct"`, `"ctt"`. (Unlike [`fit_var`](@ref)'s `deterministic`, `:trend`
here *includes* the constant; a trend without one is not a case
MacKinnon tabulates.)

## Lags: pass them explicitly to reproduce a reference

`lags` is the ADF regression's lag order — a maximum when `autolag` is
`:aic`, `:bic` or `:tstat`, the exact order when `autolag = nothing`. The
default is [`adf_test`](@ref)'s own Schwert rule, `⌈12 (n/100)^{1/4}⌉`,
and the selection *is* `adf_test`'s, so the package has one lag-selection
implementation rather than two.

`[V]` At a fixed lag the statistic is unambiguous: `statsmodels`'
`coint` and an Engle-Granger built by hand from R's `lm` and
`urca::ur.df` agree to `1.8e-15`. With each selecting its own lag they
land on different orders on three of four fixtures — `coint3_r1` gives
`−9.57` in R and `−12.27` in Python. The test is not what they disagree
about; the search is. An autolag-selected statistic will therefore not in
general match either reference, and an explicit `lags` with
`autolag = nothing` reproduces both.

## Tables, and what is refused

p-values are MacKinnon's (1994) response surfaces, which cover **N ≤ 6**
series; critical values are MacKinnon (2010) at sample size `n − 1`,
covering **N ≤ 12** — both transcribed from `statsmodels` (BSD) and
checked against `urca`'s independent 1995-vintage tables, which agree to
about `5e-4` on critical values and `2e-2` on p-values.

- `N` between 7 and 12: the critical values are returned and `pvalue`
  throws, naming them.
- `N > 12`: refused — nothing is tabulated.
- `trend = :none`: refused. MacKinnon's no-constant critical values are
  for a **single series** only. `[V]` `statsmodels` returns a p-value
  with `NaN` critical values beside it here; no field of this result is
  ever `NaN`.
- A single series: refused, naming [`adf_test`](@ref). `[V]` `statsmodels`
  silently turns that into a plain ADF test and reports `N = 1` critical
  values, which is not a cointegration test.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "coint2_yes.csv"), ','; skipstart=1);

julia> t = coint_test(Y; lags=4, autolag=nothing);

julia> round(t.statistic; digits=6), t.lags
(-5.278984, 4)

julia> t.critical_values.levels
(1.0, 5.0, 10.0)

julia> pvalue(t) < 0.001
true
```

See also [`adf_test`](@ref), [`critical_values`](@ref).
"""
function coint_test(Y; trend::Symbol=:const, lags::Union{Nothing,Integer}=nothing,
                    autolag::Union{Nothing,Symbol}=:aic)
    if Y isa AbstractVector{<:Real}
        throw(ArgumentError(
            "coint_test: a cointegration test needs at least two series, got a " *
            "single vector. For a unit-root test on one series use adf_test. " *
            "(statsmodels silently runs a plain ADF here.)"))
    end
    haskey(_COINT_TREND_REG, trend) || throw(ArgumentError(
        "coint_test: trend must be :none, :const, :trend or :quadratic, got :$trend"))
    autolag === nothing || autolag in (:aic, :bic, :tstat) || throw(ArgumentError(
        "coint_test: autolag must be :aic, :bic, :tstat or nothing, got :$autolag"))
    Ym = Float64.(tsmatrix(Y))
    n, N = size(Ym)
    N >= 2 || throw(ArgumentError(
        "coint_test: a cointegration test needs at least two series, got $N. " *
        "For a unit-root test on one series use adf_test. (statsmodels silently " *
        "runs a plain ADF here and reports N = 1 critical values.)"))
    any(!isfinite, Ym) && throw(ArgumentError("coint_test: the input contains NaN or Inf."))
    N <= 12 || throw(ArgumentError(
        "coint_test: MacKinnon's critical values cover at most 12 series, got $N. " *
        "Nothing is tabulated beyond that, so no test can be reported."))
    trend === :none && throw(ArgumentError(
        "coint_test: trend = :none is tabulated by MacKinnon for a single series " *
        "only, so there are no critical values for a $N-series cointegration " *
        "test. Use :const. (statsmodels returns a p-value with NaN critical " *
        "values here.)"))

    reg = _COINT_TREND_REG[trend]
    ndet = trend === :const ? 1 : trend === :trend ? 2 : 3
    X = Matrix{Float64}(undef, n, ndet + N - 1)
    @inbounds for t in 1:n
        X[t, 1] = 1.0
        ndet >= 2 && (X[t, 2] = t)
        ndet >= 3 && (X[t, 3] = Float64(t)^2)
    end
    X[:, (ndet + 1):end] .= view(Ym, :, 2:N)
    n > size(X, 2) + 1 || throw(ArgumentError(
        "coint_test: not enough observations -- $n rows for $(size(X, 2)) " *
        "regressors in the cointegrating regression."))
    beta, resid, _ = _ols(X, view(Ym, :, 1))

    adf = adf_test(resid; regression=:n, maxlag=lags, autolag=autolag)
    stat = adf.statistic
    cv = adf_critical_values_response_surface(reg; nobs=n - 1, N=N)
    pv = N <= 6 ? adf_pvalue_response_surface(stat, reg; N=N) : nothing
    return CointTest(stat, pv, (levels=(1.0, 5.0, 10.0), values=(Float64(cv.p1), Float64(cv.p5), Float64(cv.p10))),
                     N, trend, adf.lags, autolag, n, collect(beta), collect(resid),
                     pv !== nothing)
end

function Base.show(io::IO, t::CointTest)
    println(io, "Engle-Granger cointegration test (null: no cointegration)")
    println(io, "  series (N)          : ", t.N, "   first column is the dependent variable")
    println(io, "  trend               : ", t.trend)
    println(io, "  ADF lags            : ", t.lags,
            t.autolag === nothing ? " (fixed)" : " (selected by $(t.autolag))")
    println(io, "  n                   : ", t.nobs)
    println(io, "  test statistic      : ", round(t.statistic, digits=4))
    cv = t.critical_values
    println(io, "  critical values     : ",
            join(("$(Int(l))%: $(round(v, digits=4))" for (l, v) in zip(cv.levels, cv.values)), ", "))
    print(io,   "  p-value             : ",
          t.pvalue === nothing ? "not tabulated for N > 6 -- use critical_values" : _fmt_p(t.pvalue))
end

# ---------------------------------------------------------------------------
# Stage 10.4b -- the Phillips-Ouliaris residual-based tests, Pu and Pz.
#
# Primary source: Phillips & Ouliaris (1990), "Asymptotic Properties of
# Residual Based Tests for Cointegration", Econometrica 58(1), 165-193:
# the statistics are their (12)-(14) and §3 remark (f); the critical values
# are their Appendix B, Tables IIIa-IIIc (Pu) and IVa-IVc (Pz), transcribed
# from the paper. urca::ca.po is the numerical check -- and on Pz with
# demeaned or detrended data, a reference this package deliberately does not
# follow.
# ---------------------------------------------------------------------------

export phillips_ouliaris_test, PhillipsOuliarisTest, reject

# Phillips & Ouliaris (1990), Appendix B. Rows: n = 1 ... 5 regressors
# (m = n + 1 = 2 ... 6 series). Columns: sizes 0.15, 0.125, 0.10, 0.075,
# 0.05, 0.025, 0.01. "standard" = no deterministic term, "demeaned" =
# constant, "demeaned and detrended" = constant and trend.
const _PO_SIZES = (15.0, 12.5, 10.0, 7.5, 5.0, 2.5, 1.0)
const _PO_TABLES = Dict(
    (:Pu, :none) => (                                  # Table IIIa
        (17.2146, 18.6785, 20.3933, 22.7588, 25.9711, 31.8337, 38.3413),
        (22.9102, 24.6299, 26.7022, 29.4114, 32.9392, 39.2236, 46.4097),
        (28.9811, 31.0664, 33.5359, 36.5407, 40.1220, 46.3395, 55.7341),
        (34.5226, 36.4575, 39.2826, 41.8969, 46.2691, 53.3683, 63.2149),
        (39.7187, 41.7669, 44.3725, 47.6970, 51.8614, 59.6040, 69.4939)),
    (:Pu, :const) => (                                 # Table IIIb
        (24.1833, 25.8456, 27.8536, 30.3123, 33.7130, 39.9288, 48.0021),
        (29.3836, 31.4238, 33.6955, 36.4757, 40.5252, 46.6707, 53.8731),
        (35.1077, 37.4543, 39.6949, 42.8111, 46.7281, 53.9710, 63.4128),
        (40.5469, 42.5683, 45.3308, 48.6675, 53.2502, 61.2555, 71.5214),
        (45.3177, 47.6684, 50.3537, 53.5654, 57.7855, 65.8230, 76.7705)),
    (:Pu, :trend) => (                                 # Table IIIc
        (36.9055, 38.8150, 41.2488, 44.2416, 48.8439, 56.0886, 65.1714),
        (41.2115, 43.4320, 46.1061, 49.3671, 53.8300, 60.8745, 69.2629),
        (46.9643, 49.2906, 52.0015, 55.4625, 60.2384, 68.4051, 78.3470),
        (51.9689, 54.3205, 57.3667, 60.8175, 65.8706, 74.4712, 84.5480),
        (56.0522, 58.6310, 61.6155, 65.3514, 70.7416, 79.0043, 91.0392)),
    (:Pz, :none) => (                                  # Table IVa
        (30.0137, 31.7517, 33.9267, 36.6646, 40.8217, 47.2452, 55.1911),
        (56.7679, 59.1613, 62.1436, 65.6162, 71.2751, 79.5177, 89.6679),
        (92.7621, 95.7974, 99.2664, 103.8454, 109.7426, 119.3793, 131.5716),
        (135.2724, 138.9636, 143.0775, 148.4109, 155.8019, 166.3516, 180.4845),
        (186.4277, 190.6337, 195.6202, 201.9621, 210.2910, 224.0976, 237.7723)),
    (:Pz, :const) => (                                 # Table IVb
        (42.5452, 44.8266, 47.5877, 50.7511, 55.2202, 61.4556, 71.9273),
        (74.1493, 76.8850, 80.2034, 84.4027, 89.7619, 97.8734, 109.4525),
        (113.5617, 116.6933, 120.3035, 125.4579, 132.2207, 142.5992, 153.4504),
        (160.8156, 164.7394, 168.8572, 174.2575, 182.0749, 194.7555, 209.8054),
        (215.2089, 219.5757, 225.2303, 232.4652, 241.3316, 255.5091, 270.5018)),
    (:Pz, :trend) => (                                 # Table IVc
        (66.2417, 68.8271, 71.9586, 75.7349, 81.3812, 90.2944, 102.0167),
        (106.6198, 109.9751, 113.4929, 118.3710, 124.3933, 133.6963, 145.8644),
        (154.8402, 158.6619, 163.1050, 168.7736, 175.9902, 188.1265, 201.0905),
        (210.3150, 214.3858, 219.5098, 225.6645, 234.2865, 247.3640, 264.4988),
        (273.3064, 277.9294, 284.0100, 291.2705, 301.0949, 315.7299, 335.9054)),
)

"""
    PhillipsOuliarisTest <: HypothesisTest

A Phillips-Ouliaris residual-based cointegration test, returned by
[`phillips_ouliaris_test`](@ref). `kind` is `:Pu` or `:Pz`; `m` the
number of series; `bandwidth` the Bartlett lag used for the long-run
covariance. `critical_values` is `(levels = (10.0, 5.0, 1.0), values)` —
**ascending**, since this is a right-tail test, the opposite of the
MacKinnon tables' order; use [`reject`](@ref) rather than indexing by
position. There is no p-value: `pvalue(t)` throws.
"""
struct PhillipsOuliarisTest <: HypothesisTest
    statistic::Float64
    critical_values::NamedTuple{(:levels, :values),Tuple{NTuple{3,Float64},NTuple{3,Float64}}}
    kind::Symbol
    deterministic::Symbol
    m::Int
    bandwidth::Int
    nobs::Int
    beta::Vector{Float64}
    resid::Vector{Float64}
    pvalue_available::Bool
end

function pvalue(t::PhillipsOuliarisTest)
    throw(ArgumentError(
        "pvalue: the Phillips-Ouliaris tests have critical values only -- Phillips " *
        "& Ouliaris (1990) tabulate quantiles, not a distribution function -- so " *
        "use critical_values(t), or reject(t; level). (tseries::po.test reports " *
        "the table bounds 0.01 and 0.15 as if they were p-values.)"))
end

function _po_bandwidth(bw, T::Int)
    if bw === :short
        return trunc(Int, 4 * (T / 100)^0.25)
    elseif bw === :long
        return trunc(Int, 12 * (T / 100)^0.25)
    elseif bw isa Integer
        0 <= bw < T || throw(ArgumentError(
            "phillips_ouliaris_test: bandwidth must be in 0:$(T - 1), got $bw"))
        return Int(bw)
    end
    throw(ArgumentError(
        "phillips_ouliaris_test: bandwidth must be :short, :long or an Integer, got $bw"))
end

_po_det(n::Int, det::Symbol) = det === :none ? zeros(n, 0) :
                               det === :const ? ones(n, 1) : hcat(ones(n), collect(1.0:n))

_resid_on(Y::AbstractVecOrMat, X::AbstractMatrix) =
    size(X, 2) == 0 ? Y : Y - X * (X \ Y)

"""
    phillips_ouliaris_test(Y; kind=:Pz, deterministic=:const, bandwidth=:short)
        -> PhillipsOuliarisTest

The Phillips & Ouliaris (1990) residual-based tests of the null of **no
cointegration** among the columns of `Y` (2 to 6 series):

- `kind = :Pu` — the variance ratio (their (12)): `T ω̂₁₁.₂ / (T⁻¹ Σ û²)`,
  the long-run conditional variance of the first series given the others
  against the residual variance of the cointegrating regression. Like
  [`coint_test`](@ref) it depends on which series is first.
- `kind = :Pz` — the multivariate trace statistic (their (14)):
  `T tr(Ω̂ M_zz⁻¹)`, **invariant to the normalization** — no series is
  singled out. That is why it is the default.

Both are right-tailed: a **large** value rejects. `Ω̂` is the Bartlett-
weighted long-run covariance of the residuals of a first-order VAR in the
levels, with `bandwidth` lags: `:short` is `⌊4 (T/100)^{1/4}⌋` and `:long`
`⌊12 (T/100)^{1/4}⌋`, with `T = n − 1` (5 and 15 at `n = 250`) — `urca`'s
rules — or an integer; `0` gives the uncorrected statistic.

## `deterministic` is not a detail

`:none`, `:const` or `:trend` (constant and trend) — `urca`'s
`demean = "none" | "constant" | "trend"`, the paper's "standard",
"demeaned" and "demeaned and detrended". The choice changes the answer
qualitatively: `[V]` on `coint2_trend.csv`, whose cointegrating relation
contains a trend, `Pu` is `3.19` with `:none` (no rejection at 10%),
`50.24` with `:const` (rejects at 5%) and `170.0` with `:trend` (rejects
at 1%). The default is `:const`; choose deliberately.

## `Pz` with a deterministic term follows the paper, not `urca`

With a constant or trend, Phillips & Ouliaris build `M_zz` from the
**demeaned** (or detrended) data (§3, remark (f)), which is what their
Tables IVb and IVc were simulated under. `[V]` `urca::ca.po` uses the raw
`z'z` there. The consequence is not cosmetic: `urca`'s statistic is not
invariant to adding a constant to the data (`71.86` becomes `38.84` on
`coint2_yes.csv` after adding 1000 to both series), and in 4000
simulations of independent random walks at `n = 500` it rejected **0.25%**
of the time at the nominal 5% with a constant and **0%** with a trend,
against 4.3% and 4.2% for the paper's statistic. `Pu`, and `Pz` with
`:none`, agree with `urca` to `1e-10`.

## No p-value

The paper tabulates quantiles only, at seven sizes from 15% to 1%;
`critical_values` reports the 10%, 5% and 1% points and [`reject`](@ref)
compares against them. `pvalue(t)` throws. `[V]` `tseries::po.test`
reports `0.01` or `0.15` — the bounds of its table — as though they were
p-values; nothing here is censored that way.

`m > 6` is refused: the tables stop at five regressors.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "coint2_yes.csv"), ','; skipstart=1);

julia> t = phillips_ouliaris_test(Y; kind=:Pu, deterministic=:const);

julia> round(t.statistic; digits=6), t.bandwidth
(143.609807, 5)

julia> t.critical_values
(levels = (10.0, 5.0, 1.0), values = (27.8536, 33.713, 48.0021))

julia> reject(t; level=1.0)
true
```

See also [`coint_test`](@ref), [`reject`](@ref).
"""
function phillips_ouliaris_test(Y; kind::Symbol=:Pz, deterministic::Symbol=:const,
                                bandwidth::Union{Integer,Symbol}=:short)
    kind in (:Pu, :Pz) || throw(ArgumentError(
        "phillips_ouliaris_test: kind must be :Pu or :Pz, got :$kind"))
    deterministic in (:none, :const, :trend) || throw(ArgumentError(
        "phillips_ouliaris_test: deterministic must be :none, :const or :trend, got :$deterministic"))
    Y isa AbstractVector{<:Real} && throw(ArgumentError(
        "phillips_ouliaris_test: a cointegration test needs at least two series, " *
        "got a single vector. For a unit-root test on one series use pp_test."))
    z = Float64.(tsmatrix(Y))
    n, m = size(z)
    m >= 2 || throw(ArgumentError(
        "phillips_ouliaris_test: a cointegration test needs at least two series, got $m."))
    m <= 6 || throw(ArgumentError(
        "phillips_ouliaris_test: Phillips & Ouliaris (1990) tabulate critical values " *
        "for at most five regressors, i.e. six series; got $m."))
    any(!isfinite, z) && throw(ArgumentError(
        "phillips_ouliaris_test: the input contains NaN or Inf."))
    T = n - 1
    l = _po_bandwidth(bandwidth, T)
    T > m + 2 + l || throw(ArgumentError(
        "phillips_ouliaris_test: not enough observations -- $n rows for $m series " *
        "with bandwidth $l."))

    # residuals of the first-order VAR in levels (their (14)), with the
    # deterministic terms fitted alongside
    zl = z[2:n, :]
    D = _po_det(T, deterministic)
    xi = _resid_on(zl, hcat(z[1:T, :], D))
    Om = (xi' * xi) ./ T
    G = Matrix{Float64}(undef, m, m)
    for s in 1:l
        w = 1 - s / (l + 1)
        mul!(G, transpose(view(xi, (s + 1):T, :)), view(xi, 1:(T - s), :))
        Om .+= (w / T) .* (G .+ transpose(G))
    end

    # the cointegrating regression of the first series on the rest
    Xc = hcat(_po_det(n, deterministic), z[:, 2:m])
    beta = Xc \ z[:, 1]
    u = z[:, 1] - Xc * beta

    stat = if kind === :Pu
        w21 = Om[2:m, 1]
        w112 = Om[1, 1] - dot(w21, Om[2:m, 2:m] \ w21)
        T * w112 / (sum(abs2, u) / T)
    else
        Zm = _resid_on(zl, D)                      # demeaned / detrended: remark 3(f)
        Mzz = (Zm' * Zm) ./ T
        T * tr(Mzz \ Om)
    end

    row = _PO_TABLES[(kind, deterministic)][m - 1]
    cv = (row[3], row[5], row[7])                  # 10%, 5%, 1%
    return PhillipsOuliarisTest(stat, (levels=(10.0, 5.0, 1.0), values=cv), kind,
                                deterministic, m, l, n, beta, u, false)
end

"""
    reject(t; level=5.0) -> Bool

Whether a test reported with critical values rejects its null at `level`
percent, comparing in the test's own direction — left tail for
[`coint_test`](@ref) (a statistic **below** the critical value rejects),
right tail for [`phillips_ouliaris_test`](@ref) (**above**). `level` must
be one of the levels in `critical_values(t)`; it is matched by value,
never by position, because the two families order their levels oppositely.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "coint2_no.csv"), ','; skipstart=1);

julia> reject(phillips_ouliaris_test(Y); level=10.0)   # independent random walks
false
```
"""
function reject(t::Union{CointTest,PhillipsOuliarisTest}; level::Real=5.0)
    cv = t.critical_values
    i = findfirst(==(Float64(level)), cv.levels)
    i === nothing && throw(ArgumentError(
        "reject: level must be one of $(cv.levels), got $level"))
    return t isa CointTest ? t.statistic < cv.values[i] : t.statistic > cv.values[i]
end

const _PO_DET_NAME = Dict(:none => "none", :const => "constant (demeaned)",
                          :trend => "constant and trend (detrended)")

function Base.show(io::IO, t::PhillipsOuliarisTest)
    println(io, "Phillips-Ouliaris cointegration test, ", t.kind,
            t.kind === :Pz ? " (trace, normalization-invariant)" : " (variance ratio)")
    println(io, "  null                : no cointegration (a large value rejects)")
    println(io, "  series (m)          : ", t.m)
    println(io, "  deterministic       : ", _PO_DET_NAME[t.deterministic])
    println(io, "  bandwidth           : ", t.bandwidth)
    println(io, "  n                   : ", t.nobs)
    println(io, "  test statistic      : ", round(t.statistic, digits=4))
    cv = t.critical_values
    print(io,   "  critical values     : ",
          join(("$(Int(l))%: $(round(v, digits=4))" for (l, v) in zip(cv.levels, cv.values)), ", "),
          "   (Phillips & Ouliaris 1990; no p-value)")
end

# ---------------------------------------------------------------------------
# Stage 10.5 -- the Johansen procedure: trace and maximum-eigenvalue tests.
#
# Primary sources: Johansen (1988, 1991) and Johansen & Juselius (1990) for
# the reduced-rank regression; Osterwald-Lenum (1992), Oxford Bulletin of
# Economics and Statistics 54(3), 461-472, for the critical values,
# transcribed from its Tables 0, 1*, 1, 2* and 2 (90/95/99% columns). urca's
# ca.jo and statsmodels' coint_johansen are the numerical references for the
# statistics, which they compute identically once the mappings are right.
# ---------------------------------------------------------------------------

export johansen_test, JohansenTest, coint_rank, normalize_beta

# Osterwald-Lenum (1992). Index [p - r] (1 ... 11); each entry the 90%, 95%
# and 99% quantiles, i.e. the 10%, 5% and 1% critical values.
const _OL_TABLES = Dict(
    # Table 0 -- no deterministic terms
    (:none, :maxeig) => (
        (2.86, 3.84, 6.51), (9.52, 11.44, 15.69), (15.59, 17.89, 22.99), (21.58, 23.80, 28.82),
        (27.62, 30.04, 35.17), (33.62, 36.36, 41.00), (38.98, 41.51, 47.15), (44.99, 47.99, 53.90),
        (50.65, 53.69, 59.78), (56.09, 59.06, 65.21), (61.96, 65.30, 72.36)),
    (:none, :trace) => (
        (2.86, 3.84, 6.51), (10.47, 12.53, 16.31), (21.63, 24.31, 29.75), (36.58, 39.89, 45.58),
        (55.44, 59.46, 66.52), (78.36, 82.49, 90.45), (104.77, 109.99, 119.80), (135.24, 141.20, 152.32),
        (169.45, 175.77, 187.31), (206.05, 212.67, 226.40), (248.45, 255.27, 269.81)),
    # Table 1* -- constant restricted to the cointegrating space
    (:const_restricted, :maxeig) => (
        (7.52, 9.24, 12.97), (13.75, 15.67, 20.20), (19.77, 22.00, 26.81), (25.56, 28.14, 33.24),
        (31.66, 34.40, 39.79), (37.45, 40.30, 46.82), (43.25, 46.45, 51.91), (48.91, 52.00, 57.95),
        (54.35, 57.42, 63.71), (60.25, 63.57, 69.94), (66.02, 69.74, 76.63)),
    (:const_restricted, :trace) => (
        (7.52, 9.24, 12.97), (17.85, 19.96, 24.60), (32.00, 34.91, 41.07), (49.65, 53.12, 60.16),
        (71.86, 76.07, 84.45), (97.18, 102.14, 111.01), (126.58, 131.70, 143.09), (159.48, 165.58, 177.20),
        (196.37, 202.92, 215.74), (236.54, 244.15, 257.68), (282.45, 291.40, 307.64)),
    # Table 1 -- unrestricted constant
    (:const, :maxeig) => (
        (2.69, 3.76, 6.65), (12.07, 14.07, 18.63), (18.60, 20.97, 25.52), (24.73, 27.07, 32.24),
        (30.90, 33.46, 38.77), (36.76, 39.37, 45.10), (42.32, 45.28, 51.57), (48.33, 51.42, 57.69),
        (53.98, 57.12, 62.80), (59.62, 62.81, 69.09), (65.38, 68.83, 75.95)),
    (:const, :trace) => (
        (2.69, 3.76, 6.65), (13.33, 15.41, 20.04), (26.79, 29.68, 35.65), (43.95, 47.21, 54.46),
        (64.84, 68.52, 76.07), (89.48, 94.15, 103.18), (118.50, 124.24, 133.57), (150.53, 156.00, 168.36),
        (186.39, 192.89, 204.95), (225.85, 233.13, 247.18), (269.96, 277.71, 293.44)),
    # Table 2* -- unrestricted constant, trend restricted to the cointegrating space
    (:trend_restricted, :maxeig) => (
        (10.49, 12.25, 16.26), (16.85, 18.96, 23.65), (23.11, 25.54, 30.34), (29.12, 31.46, 36.65),
        (34.75, 37.52, 42.36), (40.91, 43.97, 49.51), (46.32, 49.42, 54.71), (52.16, 55.50, 62.46),
        (57.87, 61.29, 67.88), (63.18, 66.23, 73.73), (69.26, 72.72, 79.23)),
    (:trend_restricted, :trace) => (
        (10.49, 12.25, 16.26), (22.76, 25.32, 30.45), (39.06, 42.44, 48.45), (59.14, 62.99, 70.05),
        (83.20, 87.31, 96.58), (110.42, 114.90, 124.75), (141.01, 146.76, 158.49), (176.67, 182.82, 196.08),
        (215.17, 222.21, 234.41), (256.72, 263.42, 279.07), (303.13, 310.81, 327.45)),
    # Table 2 -- unrestricted constant and trend
    (:trend, :maxeig) => (
        (2.57, 3.74, 6.40), (14.84, 16.87, 21.47), (21.53, 23.78, 28.83), (27.76, 30.33, 35.68),
        (33.74, 36.41, 41.58), (39.50, 42.48, 48.17), (45.49, 48.45, 54.48), (51.14, 54.25, 60.81),
        (57.01, 60.29, 66.91), (62.69, 66.10, 72.96), (68.22, 71.68, 78.51)),
    (:trend, :trace) => (
        (2.57, 3.74, 6.40), (16.06, 18.17, 23.46), (31.42, 34.55, 40.49), (50.74, 54.64, 61.24),
        (73.40, 77.74, 85.78), (100.14, 104.94, 114.36), (130.84, 136.61, 146.99), (164.34, 170.80, 182.51),
        (201.95, 208.97, 222.46), (244.12, 250.84, 263.94), (288.08, 295.99, 312.58)),
)

const _JOHANSEN_DET = (:none, :const, :const_restricted, :trend, :trend_restricted)

"""
    JohansenTest <: HypothesisTest

The Johansen cointegration rank tests returned by [`johansen_test`](@ref).
Everything is ordered **`r = 0` first**, with the rank of each row in
`ranks` so position is never load-bearing.

- `trace`, `maxeig`: the statistics for `H₀: rank ≤ r` against `rank ≤ K`
  and against `rank = r + 1`.
- `trace_crit`, `maxeig_crit`: `K × 3`, columns the 10%, 5% and 1% points
  of Osterwald-Lenum (1992) for the matching deterministic case.
- `eigenvalues`: `K` of them, real and in `[0, 1)` by construction.
- `beta`: the eigenvectors, `K₁ × K` with `K₁ = K + 1` when a deterministic
  term is restricted to the cointegrating space (its coefficient is the
  last row); normalized `β'S₁₁β = I`, sign fixed so the first element is
  positive. Use [`normalize_beta`](@ref) for a conventional form.
- `alpha`: the matching adjustment coefficients `S₀₁β`, so that
  `alpha[:, 1:r] * beta[:, 1:r]'` is the rank-`r` impact matrix.
- `lags` counts **differenced** lags (`statsmodels`' `k_ar_diff`; `urca`'s
  `K` is `lags + 1`); `nobs` is the effective sample.
"""
struct JohansenTest <: HypothesisTest
    trace::Vector{Float64}
    maxeig::Vector{Float64}
    ranks::Vector{Int}
    trace_crit::Matrix{Float64}
    maxeig_crit::Matrix{Float64}
    eigenvalues::Vector{Float64}
    beta::Matrix{Float64}
    alpha::Matrix{Float64}
    K::Int
    lags::Int
    deterministic::Symbol
    nobs::Int
    names::Vector{Symbol}
    pvalue_available::Bool
end

statistic(t::JohansenTest) = t.trace

function pvalue(t::JohansenTest)
    throw(ArgumentError(
        "pvalue: the Johansen tests are reported with critical values only -- " *
        "Osterwald-Lenum (1992) tabulates quantiles, and neither reference reports " *
        "a p-value either. Use critical_values(t), or coint_rank(t)."))
end

critical_values(t::JohansenTest) =
    (levels=(10.0, 5.0, 1.0), trace=t.trace_crit, maxeig=t.maxeig_crit)

"""
    _johansen_rrr(Y, lags, deterministic) -> (eigenvalues, V, S01, T)

The reduced-rank regression at the heart of the procedure. With `ΔY_t`
regressed on `W_t = [ΔY_{t-1} … ΔY_{t-lags}, unrestricted deterministic]`
giving residuals `R₀`, and the lagged levels `Y_{t-1}` (augmented by a
restricted deterministic term) on the same `W_t` giving `R₁`, it solves

    | λ S₁₁ − S₁₀ S₀₀⁻¹ S₀₁ | = 0,    S_ij = R_i'R_j / T

Both auxiliary regressions share `W`, so they are one multi-response
least-squares solve (`_mols`). The eigenproblem is put in **symmetric-
definite** form and handed to `eigen(A, B)`, which returns real
eigenvalues by construction; `statsmodels` uses a non-symmetric form,
gets complex eigenvalues back and discards the imaginary parts with a
`ComplexWarning` on every call.
"""
function _johansen_rrr(Y::Matrix{Float64}, lags::Int, deterministic::Symbol)
    n, K = size(Y)
    dY = diff(Y; dims=1)                     # row s is ΔY_{s+1}
    T = n - lags - 1
    restricted = deterministic in (:const_restricted, :trend_restricted)
    nun = deterministic === :const ? 1 : deterministic === :trend_restricted ? 1 :
          deterministic === :trend ? 2 : 0
    W = Matrix{Float64}(undef, T, K * lags + nun)
    K1 = K + restricted
    X = Matrix{Float64}(undef, T, K + K1)
    @inbounds for i in 1:T
        t = lags + 1 + i                     # calendar time of ΔY_t, t = lags+2 … n
        for j in 1:lags, c in 1:K
            W[i, (j - 1) * K + c] = dY[t - 1 - j, c]
        end
        if nun >= 1
            W[i, K * lags + 1] = 1.0
        end
        if nun == 2
            W[i, K * lags + 2] = t
        end
        for c in 1:K
            X[i, c] = dY[t - 1, c]
            X[i, K + c] = Y[t - 1, c]
        end
        if restricted
            X[i, 2K + 1] = deterministic === :const_restricted ? 1.0 : Float64(t)
        end
    end
    _, R, _, _ = _mols(W, X)
    R0 = view(R, :, 1:K)
    R1 = view(R, :, (K + 1):(K + K1))
    S00 = (R0' * R0) ./ T
    S01 = (R0' * R1) ./ T
    S11 = (R1' * R1) ./ T
    A = transpose(S01) * (cholesky(Hermitian(S00)) \ S01)
    F = eigen(Hermitian((A + A') ./ 2), Hermitian(S11))
    ord = sortperm(F.values; rev=true)
    lam = F.values[ord]
    V = F.vectors[:, ord]
    for j in 1:size(V, 2)                    # a deterministic sign: first element positive
        V[1, j] < 0 && (V[:, j] .*= -1)
    end
    return lam, V, Matrix(S01), T
end

"""
    johansen_test(Y; lags, deterministic=:const) -> JohansenTest

Johansen's likelihood-ratio tests for the **cointegration rank** of the
`K` series in `Y`: the trace statistic `−T Σ_{i>r} log(1 − λᵢ)` for
`H₀: rank ≤ r` against `rank ≤ K`, and the maximum-eigenvalue statistic
`−T log(1 − λ_{r+1})` against `rank = r + 1`, for `r = 0 … K−1`, with
`λ` the eigenvalues of the reduced-rank regression (see `_johansen_rrr`).
`[V]` With the mappings below, `urca::ca.jo` and `statsmodels`'
`coint_johansen` agree with each other and with this to `≤ 2.2e-11`.

`lags` is **required**, and counts **differenced** lags — `statsmodels`'
`k_ar_diff`. `urca`'s `K` is `lags + 1` (it counts the levels VAR's
order). `lags = 0` is refused: `[V]` `statsmodels` silently accepts it.

## `deterministic`: five cases, one shared between the references

| `deterministic` | meaning | `urca` `ecdet` | `statsmodels` `det_order` | Osterwald-Lenum |
|---|---|---|---|---|
| `:none` | no deterministic term | — | `-1` | Table 0 |
| `:const` | unrestricted constant (a drift in the data) | `"none"` | `0` | Table 1 |
| `:const_restricted` | constant only inside the cointegrating relation | `"const"` | — | Table 1* |
| `:trend` | unrestricted constant and trend | — | — | Table 2 |
| `:trend_restricted` | unrestricted constant, trend inside the relation | `"trend"` | — | Table 2* |

`:const` is the common default and is `urca`'s `ecdet = "none"` **despite
the name**: "none" there means no deterministic term *inside the
cointegrating relation*, and an unrestricted constant is still fitted. A
restricted term adds a row to `β` and an eigenvalue that is zero by
construction; it is dropped from the test sequence. `statsmodels`'
`det_order = 1` detrends the data rather than fitting this package's
`:trend` model, so the two are not compared.

## Critical values: Osterwald-Lenum (1992), and which table

Each case uses the Osterwald-Lenum table its own decision table assigns
(p. 466). Two consequences worth knowing:

- **`:const` uses Table 1, so it agrees with `statsmodels`' values, not
  `urca`'s.** `urca`'s `ecdet = "none"` values (5% trace for `p − r = 1, 2,
  3`: `8.18, 17.95, 31.52`) are Osterwald-Lenum's `Table 1.1*`, the
  case where the model has an unrestricted constant but the data's
  constant lies in the cointegrating space only — a table the paper
  itself calls superfluous under Johansen's (1992) testing procedure.
  Table 1's `3.76, 15.41, 29.68` is the matching case, and `statsmodels`'
  independently simulated `3.84, 15.49, 29.80` agrees with it.
- The smallest-block value near `3.84` under `:const` and `:trend` is the
  asymptotic theory, not a defect: the last block is χ²(1)-like in exactly
  those cases.

The paper tabulates `p − r ≤ 11`, so **`K ≤ 11`**; more is refused, never
answered with `NaN` — `[V]` which `statsmodels` does silently for 13
series and for `det_order = 2`.

## No p-value

`pvalue(t)` throws, naming [`critical_values`](@ref) and
[`coint_rank`](@ref). MacKinnon-Haug-Michelis (1999) response surfaces
would give real p-values — something neither reference has — but they
are a separate, paper-sourced task, and interpolating three tabulated
quantiles into a pseudo-p-value is not done.

`urca`'s `spec = "transitory"` reparameterizes the short-run part only;
the eigenvalues and statistics are identical (`[V]` to `5e-12`), so it is
not a separate option here.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "coint3_r1.csv"), ','; skipstart=1);

julia> t = johansen_test(Y; lags=1);

julia> round.(t.trace; digits=4)
3-element Vector{Float64}:
 90.8401
  4.4815
  0.4595

julia> t.trace_crit[:, 2]          # 5% points, r = 0 first
3-element Vector{Float64}:
 29.68
 15.41
  3.76

julia> coint_rank(t)
1
```

See also [`coint_rank`](@ref), [`normalize_beta`](@ref), [`coint_test`](@ref).
"""
function johansen_test(Y; lags::Integer, deterministic::Symbol=:const)
    deterministic in _JOHANSEN_DET || throw(ArgumentError(
        "johansen_test: deterministic must be one of $(join(string.(':', _JOHANSEN_DET), ", ")), " *
        "got :$deterministic"))
    lags >= 1 || throw(ArgumentError(
        "johansen_test: lags counts differenced lags and must be at least 1, got $lags " *
        "(urca's K = lags + 1 must be at least 2; statsmodels silently accepts 0)."))
    Y isa AbstractVector{<:Real} && throw(ArgumentError(
        "johansen_test: a cointegration test needs at least two series, got a single vector."))
    Ym = Float64.(tsmatrix(Y))
    nms = tsnames(Y)
    n, K = size(Ym)
    K >= 2 || throw(ArgumentError(
        "johansen_test: a cointegration test needs at least two series, got $K."))
    K <= 11 || throw(ArgumentError(
        "johansen_test: Osterwald-Lenum (1992) tabulates critical values for at most " *
        "11 series, got $K. (statsmodels returns NaN critical values beyond its 12; " *
        "urca warns and then errors at 13.)"))
    any(!isfinite, Ym) && throw(ArgumentError("johansen_test: the input contains NaN or Inf."))
    T = n - lags - 1
    nreg = K * lags + 2
    T > nreg + K + 1 || throw(ArgumentError(
        "johansen_test: not enough observations -- lags = $lags on $n rows leaves " *
        "$T, against about $nreg regressors per auxiliary equation. Reduce lags."))

    lam_all, V, S01, T = _johansen_rrr(Ym, Int(lags), deterministic)
    lam = clamp.(lam_all[1:K], 0.0, prevfloat(1.0))
    l1 = log1p.(-lam)
    trace = [-T * sum(view(l1, (r + 1):K)) for r in 0:(K - 1)]
    maxeig = [-T * l1[r + 1] for r in 0:(K - 1)]
    tt = _OL_TABLES[(deterministic, :trace)]
    tm = _OL_TABLES[(deterministic, :maxeig)]
    trace_crit = [tt[K - r][j] for r in 0:(K - 1), j in 1:3]
    maxeig_crit = [tm[K - r][j] for r in 0:(K - 1), j in 1:3]
    beta = V[:, 1:K]
    alpha = S01 * beta
    names = nms === nothing ? [Symbol("y", i) for i in 1:K] : collect(Symbol, nms)
    return JohansenTest(trace, maxeig, collect(0:(K - 1)), trace_crit, maxeig_crit,
                        lam, beta, alpha, K, Int(lags), deterministic, T, names, false)
end

"""
    coint_rank(t::JohansenTest; statistic=:trace, level=5.0) -> Int

The cointegration rank chosen by Johansen's sequential procedure: test
`r = 0, 1, …` in turn and stop at the first `r` the statistic does **not**
reject at `level` percent (10, 5 or 1). If every hypothesis is rejected
the rank is `K` — the levels are stationary.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "coint3_r2.csv"), ','; skipstart=1);

julia> coint_rank(johansen_test(Y; lags=1))          # one common trend, three series
2
```
"""
function coint_rank(t::JohansenTest; statistic::Symbol=:trace, level::Real=5.0)
    statistic in (:trace, :maxeig) || throw(ArgumentError(
        "coint_rank: statistic must be :trace or :maxeig, got :$statistic"))
    j = findfirst(==(Float64(level)), (10.0, 5.0, 1.0))
    j === nothing && throw(ArgumentError(
        "coint_rank: level must be 10, 5 or 1 (the Osterwald-Lenum columns), got $level"))
    stat = statistic === :trace ? t.trace : t.maxeig
    crit = statistic === :trace ? t.trace_crit : t.maxeig_crit
    for r in 0:(t.K - 1)
        stat[r + 1] <= crit[r + 1, j] && return r
    end
    return t.K
end

"""
    normalize_beta(t, r; scheme=:first_unit) -> Matrix{Float64}

The first `r` cointegrating vectors in a conventional normalization:

- `:first_unit` — each column divided by its first element, so the first
  row is all ones. This is `urca::ca.jo`'s `V`.
- `:identity` — the leading `r × r` block made the identity,
  `β_r (β_r[1:r, :])⁻¹`. This is `urca::cajorls`'s `β`, and what most
  applied work reports for `r > 1`.

The two coincide at `r = 1`. **The test statistics do not depend on the
normalization; β does** — `[V]` the references differ on it (`statsmodels`'
`evec` is not normalized at all, and `cajorls` switches scheme with the
rank), which is not a disagreement about the estimate. Beyond the first
relation the vectors are only identified up to a rotation, so an
over-specified `r` still returns numbers; they mean nothing more than the
rank you asserted.
"""
function normalize_beta(t::JohansenTest, r::Integer; scheme::Symbol=:first_unit)
    return _normalize_beta(t.beta, r, scheme, size(t.beta, 2))
end

function _normalize_beta(beta::AbstractMatrix, r::Integer, scheme::Symbol, K::Integer)
    1 <= r <= K || throw(ArgumentError("normalize_beta: r must be in 1:$K, got $r"))
    B = beta[:, 1:r]
    if scheme === :first_unit
        return B ./ transpose(B[1, :])
    elseif scheme === :identity
        return B / B[1:r, :]
    end
    throw(ArgumentError("normalize_beta: scheme must be :first_unit or :identity, got :$scheme"))
end

const _JO_DET_NAME = Dict(:none => "none (Osterwald-Lenum Table 0)",
                          :const => "unrestricted constant (Table 1)",
                          :const_restricted => "restricted constant (Table 1*)",
                          :trend => "unrestricted constant and trend (Table 2)",
                          :trend_restricted => "restricted trend (Table 2*)")

function Base.show(io::IO, t::JohansenTest)
    println(io, "Johansen cointegration rank tests (null: rank <= r)")
    println(io, "  series (K)          : ", t.K, "   (", join(string.(t.names), ", "), ")")
    println(io, "  differenced lags    : ", t.lags)
    println(io, "  deterministic       : ", _JO_DET_NAME[t.deterministic])
    println(io, "  T                   : ", t.nobs)
    println(io, "  r    trace      10%      5%      1%     max-eig    10%      5%      1%")
    for i in 1:t.K
        print(io, "  ", rpad(string(t.ranks[i]), 3), lpad(string(round(t.trace[i], digits=3)), 9))
        for j in 1:3
            print(io, lpad(string(t.trace_crit[i, j]), 8))
        end
        print(io, lpad(string(round(t.maxeig[i], digits=3)), 11))
        for j in 1:3
            print(io, lpad(string(t.maxeig_crit[i, j]), 8))
        end
        i < t.K && println(io)
    end
    print(io, "\n  critical values: Osterwald-Lenum (1992); no p-value. rank (trace, 5%): ",
          coint_rank(t))
end
