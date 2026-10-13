export granger_test, instantaneous_causality, GrangerTest, InstantaneousCausalityTest

"""
    GrangerTest <: HypothesisTest

A Granger causality test. `H0` is that `cause` does **not** Granger-cause
`caused`.

`kind` records **which test this is**, and the distinction is not
cosmetic — see [`granger_test`](@ref):

- `:system` — the VAR-based Wald/F test, `vars::causality` and
  `statsmodels`' `test_causality`. Its `df2` is the *system* residual
  degrees of freedom, `K·T − K·(Kp + ndet)`.
- `:pairwise` — the single-equation F test, `lmtest::grangertest` and
  `statsmodels`' `grangercausalitytests[...]["ssr_ftest"]`. Its `df2` is
  one equation's, `T − 2·lags − 1`.

`statistic` is the F form and `wald` the chi-square form, `df1 ·
statistic`. Both references report both for the system test, and their
p-values differ because one refers to `F(df1, df2)` and the other to
`χ²(df1)`.
"""
struct GrangerTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    df1::Int
    df2::Int
    wald::Float64
    wald_pvalue::Float64
    cause::Vector{Symbol}
    caused::Vector{Symbol}
    kind::Symbol
    lags::Int
    nobs::Int
end

"""
    InstantaneousCausalityTest <: HypothesisTest

A test for instantaneous causality: `H0` is that the residual covariance
has no cross-block correlation between `cause` and `caused`. A Wald
statistic on the off-diagonal block of `vech(Σ)`, asymptotically
`χ²(|cause| · |caused|)`.

This says nothing about *direction* — it is a statement about
contemporaneous correlation, which is symmetric — so swapping `cause` and
`caused` gives the identical statistic, and the test suite asserts that.
"""
struct InstantaneousCausalityTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    df::Int
    cause::Vector{Symbol}
    caused::Vector{Symbol}
    nobs::Int
end

"""
    _var_indices(m, want; who="granger_test") -> Vector{Int}

Resolve variable names (or indices) against a fitted model's `names`,
refusing anything that does not appear. Duplicates are refused too:
`[V]` `vars` accepts a repeated `cause` and `statsmodels` accepts
overlapping cause and caused sets, both silently, and a repeated
restriction makes the restriction matrix rank-deficient.
"""
function _var_indices(m::VARModel, want; who::AbstractString="granger_test")
    items = want isa Union{Symbol,Integer} ? [want] : collect(want)
    isempty(items) && throw(ArgumentError("$who: the variable set is empty"))
    out = Int[]
    for w in items
        i = if w isa Integer
            1 <= w <= length(m.names) || throw(ArgumentError(
                "$who: variable index $w is outside 1:$(length(m.names))"))
            Int(w)
        else
            j = findfirst(==(Symbol(w)), m.names)
            if j === nothing
                have = join(string.(m.names), ", ")
                throw(ArgumentError(
                    "$who: no variable named :$w. The model has $have."))
            end
            j
        end
        i in out && throw(ArgumentError(
            "$who: variable :$(m.names[i]) is named twice; a repeated " *
            "restriction is rank deficient. (Both references accept this silently.)"))
        push!(out, i)
    end
    return out
end

"""
    _granger_indices(m, cause_ind, caused_ind) -> Vector{Int}

The positions in `vec(B)` of the coefficients the null restricts: every
lag of every `cause` variable in every `caused` equation.

The restriction matrix `R` is a sparse 0/1 selection, so it is never
materialized. `R Σ_a R'` is just `Σ_a[idx, idx]`, which turns an
`O(df1 · (K(Kp+ndet))²)` pair of dense multiplications into an indexing
operation — the same "exploit the structure rather than call the general
routine" point that applies to the duplication matrix in
`_dupl_pinv`.

`vec(B)` stacks equation by equation, matching `vcov(m) =
kron(Σ_df, (Z'Z)⁻¹)`: position `(j-1)·nz + i` is regressor `i` of
equation `j`.
"""
function _granger_indices(m::VARModel, cause_ind::Vector{Int}, caused_ind::Vector{Int})
    K = length(m.names)
    nz = K * m.p + _var_ndet(m.deterministic)
    idx = Int[]
    for j in caused_ind, l in 1:m.p, c in cause_ind
        push!(idx, (j - 1) * nz + (l - 1) * K + c)
    end
    return idx
end

"""
    _vech_indices(K) -> Vector{Tuple{Int,Int}}

The `(i, j)` pairs of `vech` for a `K × K` symmetric matrix: the lower
triangle taken column by column, matching `statsmodels`' `util.vech`.
"""
function _vech_indices(K::Integer)
    out = Tuple{Int,Int}[]
    for j in 1:K, i in j:K
        push!(out, (i, j))
    end
    return out
end

"""
    _dupl_pinv(K) -> Matrix{Float64}

The Moore-Penrose pseudo-inverse `D⁺` of the `K² × K(K+1)/2` duplication
matrix, in **closed form** rather than from an SVD.

`D'D` is exactly diagonal — `1` for a diagonal element of the matrix
being vectorized and `2` for an off-diagonal one, because each
off-diagonal appears twice in `vec`. So `D⁺ = (D'D)⁻¹D'` needs no
factorization at all.

`[V]` Both references call `np.linalg.pinv(duplication_matrix(k))` and
pay for an SVD. Verified for `k = 2, 3, 5, 8`: the off-diagonal mass of
`D'D` is exactly `0.0` and `max|pinv(D) − closed form| = 2.22e-16`. This
is faster *and* exact.
"""
function _dupl_pinv(K::Integer)
    vidx = _vech_indices(K)
    nv = length(vidx)
    Dp = zeros(Float64, nv, K * K)
    @inbounds for (r, (i, j)) in enumerate(vidx)
        if i == j
            Dp[r, (j - 1) * K + i] = 1.0
        else
            Dp[r, (j - 1) * K + i] = 0.5      # (D'D)^-1 = 1/2 off the diagonal
            Dp[r, (i - 1) * K + j] = 0.5
        end
    end
    return Dp
end

"""
    granger_test(m::VARModel; cause, caused=nothing) -> GrangerTest
    granger_test(y, x; lags) -> GrangerTest

Granger causality. `H0` is that `cause` does not Granger-cause `caused`.

**These are two different tests, and which you get depends on what you
pass.** Julia's dispatch is doing real work here rather than overloading
a name for convenience.

## The system test — `granger_test(m::VARModel; cause, caused)`

A Wald test on the fitted VAR, restricting every lag of every `cause`
variable in every `caused` equation to zero. `caused` defaults to every
variable not named in `cause`.

    df1 = p · |cause| · |caused|
    df2 = K·T − K·(Kp + ndet)        <- the SYSTEM residual degrees of freedom
    F   = Wald / df1

`[V]` Agrees with `vars::causality` and `statsmodels`' `test_causality`
on all 15 reference cases — `F` to `≤ 7.6e-12`, `df` identically, and
p-values to `≤ 7.5e-15` wherever `vars` does not underflow (see below).

## The pairwise test — `granger_test(y, x; lags)`

A single-equation F test comparing restricted and unrestricted OLS for
`y` on its own lags with and without the lags of `x`, so `df2` is one
equation's `T − 2·lags − 1`. `[V]` Bit-identical to
`lmtest::grangertest`, and to `statsmodels`'
`grangercausalitytests[...]["ssr_ftest"]`.

## When the two coincide, and when they do not

`[V]` The F **statistics** are identical when `|caused| = 1` — which for
the system test's default `caused` means `K = 2` — and differ otherwise.
**The p-values never coincide**, because `df2` differs by a factor of
`K`: on `var2_p1` at `p = 2`, `0.4821580341` against `0.4828206979` for
the same `F` of `0.730863702158`.

So pick deliberately. The system test is the one to use when a VAR is
the model; the pairwise test is what `lmtest` users will expect.

!!! warning "`vars` reports p = 0 for any moderately strong rejection"
    `[V]` `vars::causality` computes `1 - pf(...)` instead of the upper
    tail, so its p-values underflow to **exactly zero** in ordinary use —
    on `climhyd` an `F` of `13.36` on `(10, 2634)` df is reported as `0`
    against a true `3.66e-23`. Four of the fifteen reference cases do
    this. This package uses `_f_ccdf` and returns the tail, so
    its p-values are **not** comparable with `vars` there; they agree
    with `statsmodels`, which gets it right.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> m = fit_var(Y, 1);

julia> t = granger_test(m; cause = :y1);      # vars: F 17.504579, df (1, 392)

julia> (t.df1, t.df2, t.kind)
(1, 392, :system)

julia> round(statistic(t), digits=6)
17.504579
```

See also [`instantaneous_causality`](@ref), [`fit_var`](@ref).
"""
function granger_test(m::VARModel; cause, caused=nothing)
    K = length(m.names)
    cause_ind = _var_indices(m, cause)
    caused_ind = if caused === nothing
        setdiff(1:K, cause_ind)
    else
        _var_indices(m, caused)
    end
    isempty(caused_ind) && throw(ArgumentError(
        "granger_test: `cause` names every variable, so nothing is left to be " *
        "caused. (vars accepts this silently.)"))
    ov = intersect(cause_ind, caused_ind)
    isempty(ov) || throw(ArgumentError(
        "granger_test: :$(m.names[first(ov)]) appears in both `cause` and " *
        "`caused`. A variable cannot Granger-cause itself in this test. " *
        "(statsmodels accepts overlapping sets silently.)"))

    ndet = _var_ndet(m.deterministic)
    nz = K * m.p + ndet
    B = transpose(StatsAPI.coef(m))             # nz × K, equations in columns
    vecB = vec(Matrix{Float64}(B))
    idx = _granger_indices(m, cause_ind, caused_ind)
    df1 = length(idx)
    df2 = K * m.T - K * nz
    df2 > 0 || throw(ArgumentError(
        "granger_test: no residual degrees of freedom -- K*T = $(K * m.T) " *
        "against K*(Kp+ndet) = $(K * nz)."))

    rb = vecB[idx]
    V = m.vcov_full[idx, idx]
    F = try
        cholesky(Hermitian(Matrix(V)))
    catch e
        e isa PosDefException && throw(ArgumentError(
            "granger_test: the restricted covariance block is not positive " *
            "definite (collinear restrictions). Check for duplicated variables " *
            "or a rank-deficient VAR."))
        rethrow()
    end
    wald = dot(rb, F \ rb)
    fstat = wald / df1
    return GrangerTest(fstat, _f_ccdf(fstat, df1, df2), df1, df2,
                       wald, _chisq_ccdf(wald, df1),
                       m.names[cause_ind], m.names[caused_ind],
                       :system, m.p, m.T)
end

function granger_test(y, x; lags::Integer)
    yv = Float64.(collect(tsvalues(y)))
    xv = Float64.(collect(tsvalues(x)))
    length(yv) == length(xv) || throw(DimensionMismatch(
        "granger_test: y has $(length(yv)) observations and x has $(length(xv))"))
    lags >= 1 || throw(ArgumentError("granger_test: lags must be at least 1, got $lags"))
    any(isnan, yv) && throw(ArgumentError("granger_test: y contains NaN"))
    any(isnan, xv) && throw(ArgumentError("granger_test: x contains NaN"))
    n = length(yv)
    Tn = n - lags
    df2 = Tn - 2 * lags - 1
    df2 > 0 || throw(ArgumentError(
        "granger_test: not enough observations -- $n points at lags=$lags leaves " *
        "$Tn usable against $(2 * lags + 1) parameters. Reduce lags."))

    # one unrestricted design; the restricted fit is a view of its first columns
    Z = Matrix{Float64}(undef, Tn, 2 * lags + 1)
    @inbounds for l in 1:lags, t in 1:Tn
        Z[t, l] = yv[lags + t - l]
    end
    @inbounds for t in 1:Tn
        Z[t, lags + 1] = 1.0
    end
    @inbounds for l in 1:lags, t in 1:Tn
        Z[t, lags + 1 + l] = xv[lags + t - l]
    end
    target = yv[(lags+1):n]
    _, eu, _ = _ols(Z, target)
    _, er, _ = _ols(view(Z, :, 1:(lags + 1)), target)
    rssu = dot(eu, eu)
    rssr = dot(er, er)
    fstat = ((rssr - rssu) / lags) / (rssu / df2)
    wald = lags * fstat
    return GrangerTest(fstat, _f_ccdf(fstat, lags, df2), Int(lags), df2,
                       wald, _chisq_ccdf(wald, lags),
                       [:x], [:y], :pairwise, Int(lags), Tn)
end

"""
    instantaneous_causality(m::VARModel; cause, caused=nothing)
        -> InstantaneousCausalityTest

Test for instantaneous (contemporaneous) causality between `cause` and
`caused`: a Wald statistic on the off-diagonal block of `vech(Σ)`,
asymptotically `χ²(|cause| · |caused|)`.

    Cs     = vech(Σ)[idx]
    middle = (M[idx, idx])⁻¹ / 2       with M = D⁺ (Σ ⊗ Σ) D⁺'
    Wald   = T · Cs' · middle · Cs

`D⁺` comes from `_dupl_pinv`'s closed form rather than an SVD,
and the selection matrix `C` is applied by indexing rather than
multiplication.

The hypothesis is about contemporaneous correlation, which is symmetric,
so this is **not** a directional statement: swapping `cause` and `caused`
returns the identical statistic.

!!! note "This package is the only one that handles the general block case"
    `[V]` `statsmodels`' `test_inst_causality` raises `IndexError`
    whenever **both** sets have more than one member —
    `var_model.py` does `sig_mask[causing_ind, caused_ind] = 1` with two
    fancy index arrays, which only broadcast when one has length 1.
    Verified failing at (2,4), (3,3) and (4,2) on `k = 6`, (2,3) on
    `k = 5` and (2,2) on `k = 4`, and working at (1, k−1) and (k−1, 1).
    It always errors rather than returning a wrong answer — checked
    deliberately. The bug reproduces identically on 0.14.1 and 0.15.0.

    So `vars::causality` is the only reference for a general block, and
    this package agrees with it there.

```jldoctest
julia> using TSAnalytics, DelimitedFiles

julia> Y = readdlm(joinpath(dirname(pathof(TSAnalytics)), "..", "test",
                            "fixtures", "var", "var2_p1.csv"), ','; skipstart=1);

julia> t = instantaneous_causality(fit_var(Y, 1); cause = :y2);

julia> t.df
1

julia> round(statistic(t), digits=6)        # vars: 25.783697
25.783697
```

See also [`granger_test`](@ref).
"""
function instantaneous_causality(m::VARModel; cause, caused=nothing)
    K = length(m.names)
    cause_ind = _var_indices(m, cause)
    caused_ind = caused === nothing ? setdiff(1:K, cause_ind) : _var_indices(m, caused)
    isempty(caused_ind) && throw(ArgumentError(
        "instantaneous_causality: `cause` names every variable, so nothing is " *
        "left to be caused."))
    ov = intersect(cause_ind, caused_ind)
    isempty(ov) || throw(ArgumentError(
        "instantaneous_causality: :$(m.names[first(ov)]) appears in both sets."))

    Sig = m.sigma_df
    vidx = _vech_indices(K)
    # the vech positions coupling the two blocks, in either direction, which
    # is what statsmodels builds its sig_mask to find
    sel = Int[]
    for (r, (i, j)) in enumerate(vidx)
        if (i in cause_ind && j in caused_ind) || (i in caused_ind && j in cause_ind)
            push!(sel, r)
        end
    end
    isempty(sel) && throw(ArgumentError(
        "instantaneous_causality: the two sets share no covariance entry."))

    Dp = _dupl_pinv(K)
    M = Dp * kron(Sig, Sig) * transpose(Dp)
    Cs = [Sig[i, j] for (i, j) in vidx][sel]
    Vb = M[sel, sel]
    F = try
        cholesky(Hermitian(Matrix(Vb)))
    catch e
        e isa PosDefException && throw(ArgumentError(
            "instantaneous_causality: the covariance block is not positive " *
            "definite; Σ may be singular."))
        rethrow()
    end
    wald = m.T * dot(Cs, (F \ Cs)) / 2
    df = length(cause_ind) * length(caused_ind)
    return InstantaneousCausalityTest(wald, _chisq_ccdf(wald, df), df,
                                      m.names[cause_ind], m.names[caused_ind], m.T)
end

"""
    _fmt_p(p) -> String

Format a p-value for display without destroying a small one.

`round(p, digits=10)` prints `0.0` for anything below `5e-11`, which would
have this package's own output reproduce — cosmetically — the exact
failure it exists to avoid: `vars` reporting `p = 0` for a real tail of
`3.7e-23`. Small values switch to scientific notation instead.
"""
_fmt_p(p::Real) = p == 0 ? "0" : (p < 1e-4 ? string(round(p; sigdigits=6)) :
                                  string(round(p; digits=10)))

function Base.show(io::IO, t::GrangerTest)
    kindname = t.kind === :system ? "system (VAR Wald/F)" : "pairwise (single-equation F)"
    println(io, "Granger causality test -- ", kindname)
    if t.kind === :system
        println(io, "  H0: ", join(string.(t.cause), " "), " do not Granger-cause ",
                join(string.(t.caused), " "))
    else
        println(io, "  H0: x does not Granger-cause y")
    end
    println(io, "  F = ", round(t.statistic, digits=6),
            "   df = (", t.df1, ", ", t.df2, ")   p = ", _fmt_p(t.pvalue))
    print(io, "  chi2 = ", round(t.wald, digits=6), "   p = ",
          _fmt_p(t.wald_pvalue), "   lags = ", t.lags, "   T = ", t.nobs)
end

function Base.show(io::IO, t::InstantaneousCausalityTest)
    println(io, "Instantaneous causality test")
    println(io, "  H0: no instantaneous causality between ",
            join(string.(t.cause), " "), " and ", join(string.(t.caused), " "))
    print(io, "  chi2 = ", round(t.statistic, digits=6), "   df = ", t.df,
          "   p = ", _fmt_p(t.pvalue), "   T = ", t.nobs)
end
