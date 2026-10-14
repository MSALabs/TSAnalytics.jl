# Handoff: Stage 10.0 — Multivariate primitives

**Status: implemented ✅ (2026-10-13).** Build this first. Nothing else in Stage 10 should
start before it.

For a fresh Claude Code session with no prior context. **Read
`stage-10-overall-handoff.md` first** — the stage charter carries the
conventions (§5), architecture decisions (§6), reference environment
(§3) and the performance and test policy (§7, §8). This document cites
them rather than repeating them.

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every reference number below is `[V]`. **Julia was not installed in the
sandbox that produced this document** — re-verify any package-internal
claim before building on it.

---

## 1. Why this is a roadmap row of its own

`development-sequence.md` has no 10.0. It should: four things are needed
by **every** item in the stage, none of them exist, and each is a
Stage-1-shaped primitive rather than part of any one model.

| Primitive | Needed by | Exists? |
|---|---|---|
| Multi-response least squares | 10.1, 10.5, 10.6, 10.8 | `[S]` No — `_ols` is `y::AbstractVector` only |
| `tsmatrix` / `tsnames` | every item | `[S]` No — `tsvalues` is vector-shaped by contract |
| `build_var_Z` (regressor construction) | 10.1, 10.2, 10.5, 10.6, 10.8 | No |
| `companion` matrix | 10.1 (roots), 10.3 (IRF `G`), 10.7 | `[S]` No `k`-variate version |

Written inside `fit_var` instead, each would be re-implemented or
re-derived by the next item. The multi-response solve in particular has
a performance consequence that is invisible once buried: a loop of
`_ols` calls re-factorizes the shared regressor matrix `K` times.

---

## 2. Scope

### In

- `_mols(Z, X; method, weights)` — least squares with a **matrix**
  response, one factorization and `K` solves.
- `tsmatrix(x)` → `AbstractMatrix{<:Real}`, with methods for
  `AbstractMatrix`, a `Tuple`/`Vector` of vectors, and a `NamedTuple` of
  columns.
- `tsnames(x)` → `Union{Nothing, Vector{Symbol}}`.
- `build_var_Z(Y, p, deterministic)` → `(Z, X, colnames)`.
- `companion(A)` → `Kp × Kp`.
- Rank-deficiency detection with an honest refusal (§6) — where both
  references silently proceed.

### Out

- Extending `_ols`'s own exported behaviour. `_mols` is a sibling, not a
  replacement; the univariate path keeps its exact current semantics so
  nothing in Stages 1–9 changes. A later refactor can unify them once
  there is a reason.
- Pivoted / rank-revealing QR with column dropping (R's `lm` does this;
  §6). Refuse instead, same deferral as `stage-1.4-ols-cholesky-handoff.md`.
- Sparse `Z`. Not useful here — `Z` is dense by construction.

---

## 3. `_mols` — multi-response least squares

### 3.1 Both references agree, and both equal a full VAR fit

`[V]` R's `lm(cbind(y1, y2) ~ Z - 1)` against statsmodels'
`get_var_endog` + `lstsq`, and both against `VAR().fit()`:

| Fixture | B shape | max abs diff | Σ_df diff | equals `fit().params`? |
|---|---|---|---|---|
| `var2_p1` p=1 | 3 × 2 | 5.83e-16 | 2.2e-16 | **yes** |
| `var2_p1` p=2 | 5 × 2 | 6.66e-16 | 5.6e-17 | **yes** |
| `var3_p2` p=2 | 7 × 3 | 5.00e-16 | 2.2e-16 | **yes** |
| `uschange` p=2 | 11 × 5 | 5.33e-15 | 1.8e-15 | **yes** |

So `_mols` has a clean dual reference, and the last column is the useful
part: a correct `_mols` **is** the VAR fit, so 10.1 becomes assembly
rather than estimation.

### 3.2 Signature and contract

```julia
"""
    _mols(Z, X; method=:qr, weights=nothing) -> (B, E, se, ZtZ_inv)

Least squares with a MATRIX response: solves X = Z*B + E for
B of size (q, K), where Z is (T, q) and X is (T, K).

The point of the matrix form: all K columns of X share one regressor
matrix, so this does ONE factorization of Z and K back-substitutions.
Calling `_ols` in a loop over columns re-factorizes Z K times for an
identical answer.

Returns the coefficient matrix `B`, the residual matrix `E` (T x K),
the standard errors `se` (q x K, column j being equation j's), and
`ZtZ_inv`, which callers need for the full parameter covariance
`kron(Sigma, ZtZ_inv)`.

`method=:qr` (default) factorizes Z directly; `:cholesky` factorizes
Z'Z, faster for well-conditioned problems and losing accuracy with
cond(Z)^2 rather than cond(Z) -- the same tradeoff and the same default
as `_ols`, for the same reason.

Throws an `ArgumentError` naming rank deficiency when Z is not full
column rank. Both R's `lm` and statsmodels' `lstsq` proceed silently
there -- R by dropping a column and returning NA coefficients, Python
by returning a minimum-norm solution -- and neither behaviour is safe
to inherit. See the Stage 10.0 handoff, section 6.
"""
function _mols(Z::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real};
               method::Symbol=:qr,
               weights::Union{Nothing,AbstractVector{<:Real}}=nothing)
```

Mirror `_ols`'s own argument names, error style and the weighted
reduction (scale rows by `sqrt.(weights)`), so the two read as siblings.
`[S]` `_ols` is at `src/unitroot.jl:51`; `_diagvec` at `:99`.

### 3.3 Implementation

```julia
    T, q = size(Z); K = size(X, 2)
    size(X, 1) == T || throw(ArgumentError("_mols: Z and X must have the same number of rows"))
    dof = T - q
    dof > 0 || throw(ArgumentError("_mols: not enough observations ($T) for $q regressors"))

    Zw, Xw = _apply_weights(Z, X, weights)       # identical reduction to _ols's

    local B, ZtZ_inv
    if method === :qr
        F = qr(Zw)
        rank_ok(F, q) || throw(ArgumentError(
            "_mols: Z is not full column rank (rank < $q) -- collinear regressors; " *
            "drop a redundant series or reduce the lag order"))
        B = F \ Xw                               # ONE factorization, K solves
        Rinv = F.R \ Matrix{Float64}(I, q, q)
        ZtZ_inv = Rinv * Rinv'
    else
        ZtZ = Zw' * Zw
        C = try
            cholesky(Hermitian(Matrix(ZtZ)))
        catch e
            e isa PosDefException && throw(ArgumentError(
                "_mols: method=:cholesky failed -- Z'Z is not positive definite " *
                "(likely collinear regressors); try method=:qr instead"))
            rethrow()
        end
        B = C \ (Zw' * Xw)
        ZtZ_inv = C \ Matrix{Float64}(I, q, q)
    end

    E  = X - Z * B                               # ORIGINAL scale, one GEMM
    Ew = Xw - Zw * B
    Sigma = (Ew' * Ew) ./ dof
    se = similar(B)
    d = _diagvec(ZtZ_inv)
    @inbounds for j in 1:K, i in 1:q
        se[i, j] = sqrt(Sigma[j, j] * d[i])
    end
    return B, E, se, ZtZ_inv
```

Three things to get right:

1. **`F \ Xw` with a matrix right-hand side is the whole optimization.**
   Julia's QR solve already does one factorization and `K`
   back-substitutions, so the naive-looking expression is the fast one.
   Do **not** write `hcat([_ols(Z, X[:,j])[1] for j in 1:K]...)`.
2. **`se[i,j] = sqrt(Sigma[j,j] * ZtZ_inv[i,i])`** — the `(i,j)`
   standard error pairs equation `j`'s residual variance with regressor
   `i`'s diagonal. The full covariance is `kron(Sigma, ZtZ_inv)` and
   `se` is the square root of its diagonal, reshaped. Returning
   `ZtZ_inv` rather than the Kronecker product keeps `_mols` from
   building a `qK × qK` matrix that only 10.1 and 10.3 want.
3. **`Sigma` here is df-adjusted** (`/dof`). The ML version is the
   caller's business — 10.1 stores both (charter §5.5).

### 3.4 Reference values

`[V]` `var2_p1`, p=2, `:const`. `B` rows are
`[const, y1.l1, y2.l1, y1.l2, y2.l2]`, columns are equations:

```
B = [ 0.414053755176  -0.234406837193
      0.448249746032   0.220514094415
      0.070329548445   0.397525035175
      0.014232369339   0.013417589992
     -0.111386497811  -0.029056864880]

se = [0.104260453320  0.081759879602
      0.077607372675  0.060858832319
      0.099412416939  0.077958103778
      0.080462771256  0.063098003901
      0.094266992601  0.073923119649]

Sigma_df = [1.011750538636  0.297273347693
            0.297273347693  0.622178089485]

E[1:2, :] = [-1.702224863368  -0.565195879071
             -1.571550257039  -0.005319281061]
```

`[V]` `var3_p2`, p=2, `:const`, `B` (7 × 3):

```
[ 0.031130809530   0.124867937320  -0.065545311252
  0.290077426458  -0.045247308605  -0.125708121200
  0.068287895814   0.557104755017   0.205459393486
 -0.030404069817   0.099593849045   0.508486961624
  0.191143749358  -0.000107435400   0.244293832209
 -0.019570758286   0.093802396263  -0.199763544710
  0.137891937175   0.049493870786   0.162354789018]
```

`[V]` `uschange`, p=2, `Sigma_df` first row:
`[0.369574475216, 0.199516463793, 0.379682807764, -1.899479226109, -0.092305049392]`.

### 3.5 The `k = 1` reduction

`fit_var` refuses `k = 1` (charter §5.9), but `_mols` must handle it,
and `stats::ar.ols` is exactly that case. `[V]` On `var1_k1.csv`
(n = 200), `demean=FALSE, intercept=TRUE`:

| p | `ar` | `intercept` | `var.pred` |
|---|---|---|---|
| 1 | 0.6484308401025832 | 0.32794476411707674 | 0.9538204266778918 |
| 2 | 0.673140896685106, −0.03945717137380811 | 0.33998905461259055 | 0.9560758286522902 |
| 3 | 0.6687833427540003, −0.059036329136739174, 0.028860901589552235 | 0.3237308512151343 | 0.9343819545156791 |

`[V]` and at p = 1 the asymptotic standard errors are `0.05410479122534204`
on `ar`, `0.08644379310823685` on the intercept.

---

## 4. `build_var_Z` — regressor construction

### 4.1 Layout, verified

`[V]` R and statsmodels build the identical lag block. First row of `Z`
for `var2_p1`, p=2:

```
R : [1.0, 3.918146471612, 2.169296248687, 3.187181068090, 0.893489697008]
py: [1.0, 3.918146471612, 2.169296248687, 3.187181068090, 0.893489697008]
```

Ordering is `[deterministic | lag 1 all variables | lag 2 all variables | …]`
— ascending lag outside, variable index inside. `[V]` R's own column
names confirm it: `const, y1.l1, y2.l1, y1.l2, y2.l2` and, for
`uschange`, `const, Consumption.l1, Income.l1, …, Unemployment.l2`.

`[V]` `Z` has `T = n − p` rows: `[198, 5]` for `var2_p1` p=2, and the
last row is
`[1.0, 1.486798501843, −0.010927085209, 1.073571556737, −0.350570210875]`.

**Decision:** deterministic columns **last** in TSAnalytics (matching
R's reading order and 10.1's `coef(m)`), in the order
`(const, trend, trend²)`.

### 4.2 A trap: statsmodels' own helper disagrees with its own fit

`[V]` **`util.get_var_endog(Y, p, trend="ct")` builds the trend as
`1:T`, but `VAR.fit(p, trend="ct")` returns coefficients corresponding
to `p+1:n`.** The helper and the fit path disagree.

```
get_var_endog(trend='ct') rows 1-3:    [1, 1, 3.18718107, 0.8934897]
                                       [1, 2, 3.91814647, 2.16929625]
                                       [1, 3, 0.56654113, 0.94355565]

lstsq on that Z        -> const [ 0.36909826 -0.18545125]
VAR.fit(trend='ct')    -> const [ 0.36860606 -0.18500209]     <- differs
R  VAR(type='both')    -> const [ 0.36860606 -0.18500209]     <- matches fit

explicit trend = 1:T    -> |const - R| = 4.922e-04
explicit trend = p+1:n  -> |const - R| = 3.89e-16
```

The trend **coefficient** is identical either way
(`0.000492205472`, `-0.000449155817`); the whole discrepancy lands in
the constant, and equals exactly `p ×` the trend coefficient.

**Consequences:**

- Build the trend as `p+1 : n` (the estimation sample's calendar index).
  This confirms `stage-10.1-var-handoff.md` §6.1 from a second
  direction.
- **Do not use `get_var_endog` as the reference for the deterministic
  columns.** It is a correct reference for the lag block and wrong for
  the trend. A test that compared against it would enshrine the bug.
- Add this to the charter §5.10 list.

### 4.3 Signature

```julia
build_var_Z(Y::AbstractMatrix{<:Real}, p::Integer, deterministic::Symbol)
    -> (Z::Matrix{Float64}, X::Matrix{Float64}, colnames::Vector{Symbol})
```

`deterministic ∈ (:none, :const, :trend, :both, :quadratic)` per charter
§5.2. `X = Y[p+1:end, :]`. Build with `@views` into `Y` and `@inbounds`;
one allocation for `Z` itself.

For the quadratic term use `t^2` with `t = p+1 … n` directly, not
`(t - mean)^2` or a scaled variant — `[V]` statsmodels' `ctt` third
column starts at 1 for `t = 1`, consistent with a raw square, and the
first row of its `ctt` Z is `[1, 1, 1, 3.187181, 0.89349]`. Re-verify
the origin for `ctt` the same way §4.2 did for `ct`; the 10.1 reference
`A_p1_ctt` has the coefficients to check against
(`det_coef[:,3] = [5.913177572457e-5, 2.879857740094e-5]`).

---

## 5. `companion` — and its one real subtlety

### 5.1 Structure and references

```
C = [ A_1  A_2  …  A_p ]      (K   × Kp)
    [ I    0    …  0   ]
    [ 0    I    …  0   ]       (K(p-1) × Kp)
    [ …                ]
```

`[V]` Verified against statsmodels' `util.comp_matrix(coefs)` and
against a hand construction from R's `Bcoef`, on five cases:

| Case | shape | max abs diff | eigenvalue-modulus diff | `vars::roots()` == sorted moduli |
|---|---|---|---|---|
| `var2_p1` p=1 | 2 × 2 | 2.50e-16 | 1.1e-16 | **0.0e+00** |
| `var2_p1` p=2 | 4 × 4 | 8.33e-16 | 8.2e-16 | **0.0e+00** |
| `var3_p2` p=2 | 6 × 6 | 4.44e-16 | 6.7e-16 | **0.0e+00** |
| `var2_p1` p=4 | 8 × 8 | 4.79e-16 | 1.7e-15 | **0.0e+00** |
| `uschange` p=2 | 10 × 10 | 1.42e-14 | 9.4e-15 | **0.0e+00** |

The last column is the useful one: `vars::roots()` **is** the sorted
companion eigenvalue moduli, exactly. That settles charter §5.3's
convention choice with a construction rather than an assertion.

`[V]` Reference companion for `var2_p1` p=2:

```
[ 0.448249746032  0.070329548445  0.014232369339 -0.111386497811
  0.220514094415  0.397525035175  0.013417589992 -0.029056864880
  1.0             0.0             0.0             0.0
  0.0             1.0             0.0             0.0            ]

eigenvalue moduli: [0.498760872427, 0.498760872427,
                    0.065920201718, 0.065920201718]
```

`[V]` Other eigenvalue-modulus sets to assert:

```
var3_p2 p=2 : [0.794367666844, 0.679227662595, 0.525274720722,
               0.456971502603, 0.104228668830, 0.104228668830]
var2_p1 p=4 : [0.642903193503, 0.642903193503, 0.614614306781,
               0.614614306781, 0.546525304394, 0.546525304394,
               0.465958036074, 0.131815551172]
uschange p=2: [0.650747126958, 0.650747126958, 0.426864696068,
               0.426864696068, 0.407386546795, 0.407386546795,
               0.358538490557, 0.358538490557, 0.236786554235,
               0.179159927807]
```

### 5.2 The subtlety: `p = 1`

At `p = 1` the companion matrix **is** `A_1` and there are no identity
blocks. `[V]` Confirmed: shape 2 × 2 for `var2_p1` p=1. A loop that
unconditionally writes `C[(K+1):(K*p), 1:(K*(p-1))] = I` must be guarded
— at `p = 1` the ranges are `3:2` and `1:0`, which in Julia is an empty
assignment rather than an error, so the bug would not announce itself.
Assert the `p = 1` shape explicitly.

### 5.3 Implementation

```julia
function companion(A::Vector{Matrix{Float64}})
    p = length(A); K = size(A[1], 1)
    C = zeros(K*p, K*p)
    @inbounds for j in 1:p
        C[1:K, (j-1)*K+1 : j*K] .= A[j]
    end
    if p > 1
        @inbounds for i in 1:(K*(p-1))
            C[K+i, i] = 1.0
        end
    end
    return C
end
```

Write the sub-diagonal identity element by element rather than with a
block assignment from `I(K*(p-1))` — it avoids materializing a second
matrix, and the `p = 1` guard becomes explicit instead of relying on
empty-range semantics. Cache `C` on the model (10.1), since 10.3's `G`
construction and 10.7 both want it.

---

## 6. Rank deficiency — where both references proceed silently

`[V]` With a duplicated column (`cbind(Y, Y[,1])`), `p = 1`, `:const`:

```
R  (lm inside vars::VAR) : no error; Bcoef contains NA
py (np.linalg.lstsq)     : no error; no NaN -- a minimum-norm solution
```

R drops a column and reports `NA` coefficients, which at least signals
something. statsmodels returns a perfectly finite minimum-norm solution
to a rank-deficient system, with no warning — and every downstream
quantity (standard errors, IRF, FEVD, Granger tests) is then
meaningless while looking fine.

**Decision: `_mols` refuses.** Check the rank from the QR factorization
(compare `abs.(diag(F.R))` against `q * eps() * maximum(abs, diag(F.R))`,
the conventional tolerance) and throw an `ArgumentError` naming
collinearity and suggesting the fix. `[S]` This matches `_ols`'s own
`:cholesky` branch, which already throws an `ArgumentError` rather than
letting `PosDefException` escape (`src/unitroot.jl:182–189`).

Pivoted QR with column dropping stays deferred, same as
`stage-1.4-ols-cholesky-handoff.md` §1.6: the bookkeeping is real and no
caller needs it.

---

## 7. `tsmatrix` and `tsnames`

Per charter §6.2. `[S]` `src/interface.jl` documents
`tsvalues(x) -> AbstractVector{<:Real}` with exactly two methods, and
the fallback `collect(Float64, x)` on a matrix **returns a matrix**, so
today it silently violates its own documented return contract rather
than erroring. Do not overload `tsvalues`; add siblings.

```julia
tsmatrix(x::AbstractMatrix{<:Real}) = x                      # zero copy
tsmatrix(x::NamedTuple)                                       # columns -> matrix
tsmatrix(x::Union{Tuple,AbstractVector{<:AbstractVector}})    # vector of series
tsmatrix(x) = error(...)                                      # explicit, not a silent collect

tsnames(x) = nothing
tsnames(x::NamedTuple) = collect(Symbol, keys(x))
```

Requirements:

- Ragged input is refused by name, not padded or truncated.
- A `NamedTuple` of columns is the shape `dataset()` already returns for
  multivariate entries (`[S]` `src/dataset_catalogue.jl`), so it must
  work without the caller unpacking.
- Column-major: build the matrix by columns.
- Names default to `[:y1, …, :yK]` when the container carries none.
- Keep the docstring's promise honest — state the return type and mean
  it, which is the specific thing `tsvalues` got wrong.

---

## 8. Test battery

Exact values all `[V]`, from `fixtures/ref_R_prim.json` and
`fixtures/ref_py_prim.json`. Suggested file: `test/test_mvprimitives.jl`.
Target **≥ 150 assertions**.

### 8.1 `_mols` against both references

```julia
@testset "_mols: var2_p1 p=2 vs R lm(cbind(...)) and statsmodels" begin
    Y = read_fixture("var2_p1.csv")
    Z, X, cn = build_var_Z(Y, 2, :const)
    @test size(Z) == (198, 5)
    @test size(X) == (198, 2)
    B, E, se, ZtZinv = _mols(Z, X)

    @test B ≈ [ 0.414053755176  -0.234406837193
                0.448249746032   0.220514094415
                0.070329548445   0.397525035175
                0.014232369339   0.013417589992
               -0.111386497811  -0.029056864880] atol=1e-11
    @test se ≈ [0.104260453320  0.081759879602
                0.077607372675  0.060858832319
                0.099412416939  0.077958103778
                0.080462771256  0.063098003901
                0.094266992601  0.073923119649] atol=1e-11
    @test E[1:2, :] ≈ [-1.702224863368  -0.565195879071
                       -1.571550257039  -0.005319281061] atol=1e-10
    T, q = size(Z)
    @test (E' * E) ./ (T - q) ≈ [1.011750538636  0.297273347693
                                 0.297273347693  0.622178089485] atol=1e-11

    # se must be the sqrt diagonal of kron(Sigma, ZtZinv), reshaped
    Sig = (E'*E) ./ (T - q)
    @test vec(se) ≈ sqrt.(diag(kron(Sig, ZtZinv))) atol=1e-12
end

@testset "_mols: every fixture" begin
    for (fx, p, tol) in [("var2_p1.csv",1,1e-11), ("var2_p1.csv",2,1e-11),
                         ("var3_p2.csv",2,1e-11), ("uschange.csv",2,1e-9)]
        Y = read_fixture(fx); Z, X, _ = build_var_Z(Y, p, :const)
        B, E, se, _ = _mols(Z, X)
        ref = load_ref_prim(fx, p)
        @test B  ≈ ref.B  atol=tol
        @test se ≈ ref.se atol=tol
    end
end

@testset "_mols: a correct multi-response solve IS the VAR fit" begin
    # both references confirm B == VAR().fit().params exactly
    for (fx, p) in [("var2_p1.csv",1), ("var2_p1.csv",2),
                    ("var3_p2.csv",2), ("uschange.csv",2)]
        Y = read_fixture(fx)
        Z, X, _ = build_var_Z(Y, p, :const)
        B, = _mols(Z, X)
        m = fit_var(Y, p; deterministic=:const)
        # B rows are [const, lags...]; coef(m) is K x (Kp+ndet) with det LAST
        @test permutedims(B)[:, 2:end]  ≈ coef(m)[:, 1:end-1] atol=1e-11
        @test permutedims(B)[:, 1]      ≈ coef(m)[:, end]     atol=1e-11
    end
end

@testset "_mols: column-by-column equivalence with _ols" begin
    Y = read_fixture("var3_p2.csv"); Z, X, _ = build_var_Z(Y, 2, :const)
    B, E, se, _ = _mols(Z, X)
    for j in 1:3
        b, e, s = TSAnalytics._ols(Z, X[:, j])
        @test B[:, j]  ≈ b atol=1e-12       # same answer...
        @test E[:, j]  ≈ e atol=1e-12
        @test se[:, j] ≈ s atol=1e-12
    end
    # ...but one factorization. K separate _ols calls must be slower.
    t_m = @elapsed for _ in 1:500; _mols(Z, X); end
    t_l = @elapsed for _ in 1:500; for j in 1:3; TSAnalytics._ols(Z, X[:,j]); end; end
    @test t_m < t_l
end

@testset "_mols: weights, and the QR/Cholesky agreement" begin
    Y = read_fixture("var2_p1.csv"); Z, X, _ = build_var_Z(Y, 2, :const)
    w = 1.0 .+ rand(MersenneTwister(5), size(Z,1))
    for kw in ((), (weights=w,))
        a = _mols(Z, X; kw...)
        b = _mols(Z, X; method=:cholesky, kw...)
        @test a[1] ≈ b[1] rtol=1e-8
        @test a[3] ≈ b[3] rtol=1e-7
    end
    # weighted _mols == weighted _ols per column
    Bw, = _mols(Z, X; weights=w)
    for j in 1:2
        bw, = TSAnalytics._ols(Z, X[:, j]; weights=w)
        @test Bw[:, j] ≈ bw atol=1e-11
    end
end

@testset "_mols: k = 1 reduction onto stats::ar.ols" begin
    z = read_fixture("var1_k1.csv")                  # 200 x 1
    for (p, ar, intercept, varpred) in
            [(1, [0.6484308401025832], 0.32794476411707674, 0.9538204266778918),
             (2, [0.673140896685106, -0.03945717137380811], 0.33998905461259055, 0.9560758286522902),
             (3, [0.6687833427540003, -0.059036329136739174, 0.028860901589552235],
                 0.3237308512151343, 0.9343819545156791)]
        Z, X, _ = build_var_Z(z, p, :const)
        B, E, se, _ = _mols(Z, X)
        @test B[1:p, 1] ≈ ar atol=1e-11              # lag coefficients
        @test B[p+1, 1] ≈ intercept atol=1e-11       # const is LAST in our layout
        T, q = size(Z)
        @test dot(E, E) / (T - q) ≈ varpred rtol=1e-9
    end
    # p = 1 asymptotic standard errors
    Z, X, _ = build_var_Z(z, 1, :const)
    _, _, se, _ = _mols(Z, X)
    @test se[1, 1] ≈ 0.05410479122534204 rtol=1e-7
    @test se[2, 1] ≈ 0.08644379310823685 rtol=1e-7
end
```

### 8.2 `build_var_Z`

```julia
@testset "build_var_Z: layout and the trend origin" begin
    Y = read_fixture("var2_p1.csv")
    Z, X, cn = build_var_Z(Y, 2, :const)
    @test size(Z) == (198, 5)
    @test cn == [:y1_l1, :y2_l1, :y1_l2, :y2_l2, :const]      # det LAST
    @test Z[1, :] ≈ [3.918146471612, 2.169296248687,
                     3.187181068090, 0.893489697008, 1.0] atol=1e-11
    @test Z[end, :] ≈ [1.486798501843, -0.010927085209,
                       1.073571556737, -0.350570210875, 1.0] atol=1e-11
    @test X == Y[3:end, :]
    # lag block: ascending lag outside, variable inside
    @test Z[:, 1] == Y[2:end-1, 1]
    @test Z[:, 2] == Y[2:end-1, 2]
    @test Z[:, 3] == Y[1:end-2, 1]
    @test Z[:, 4] == Y[1:end-2, 2]

    # the trend runs p+1:n, NOT 1:T -- statsmodels' own get_var_endog gets
    # this wrong relative to its own fit path (handoff section 4.2)
    Zb, _, cnb = build_var_Z(Y, 1, :both)
    @test cnb[end-1:end] == [:const, :trend]
    @test Zb[1, end] == 2.0                   # p + 1 = 2, not 1
    @test Zb[end, end] == 200.0               # n
    Zq, _, cnq = build_var_Z(Y, 1, :quadratic)
    @test cnq[end-2:end] == [:const, :trend, :trend2]
    @test Zq[1, end] == 4.0                   # 2^2
    @test Zq[end, end] == 200.0^2
end

@testset "build_var_Z: all five deterministic specs" begin
    Y = read_fixture("var2_p1.csv")
    for (d, ndet) in [(:none,0), (:const,1), (:trend,1), (:both,2), (:quadratic,3)]
        Z, X, cn = build_var_Z(Y, 2, d)
        @test size(Z, 2) == 2*2 + ndet
        @test length(cn) == size(Z, 2)
        @test size(Z, 1) == size(X, 1) == 198
    end
    # :trend has NO constant -- R's type="trend", which statsmodels cannot express
    Zt, _, cnt = build_var_Z(Y, 1, :trend)
    @test !(:const in cnt)
    @test :trend in cnt
    @test_throws ArgumentError build_var_Z(Y, 1, :bogus)
    @test_throws ArgumentError build_var_Z(Y, 0, :const)
    @test_throws ArgumentError build_var_Z(Y, 300, :const)
end
```

### 8.3 `companion`

```julia
@testset "companion: structure and eigenvalues" begin
    Y = read_fixture("var2_p1.csv")
    m2 = fit_var(Y, 2; deterministic=:const)
    C = companion(m2.A)
    @test size(C) == (4, 4)
    @test C ≈ [ 0.448249746032  0.070329548445  0.014232369339 -0.111386497811
                0.220514094415  0.397525035175  0.013417589992 -0.029056864880
                1.0             0.0             0.0             0.0
                0.0             1.0             0.0             0.0           ] atol=1e-11
    @test sort(abs.(eigvals(C)); rev=true) ≈
          [0.498760872427, 0.498760872427, 0.065920201718, 0.065920201718] atol=1e-10

    # p = 1: the companion IS A_1, with no identity blocks (easy to get wrong,
    # because the guarded ranges are EMPTY rather than an error in Julia)
    m1 = fit_var(Y, 1; deterministic=:const)
    C1 = companion(m1.A)
    @test size(C1) == (2, 2)
    @test C1 == m1.A[1]

    for (fx, p, mods) in [
        ("var3_p2.csv", 2, [0.794367666844, 0.679227662595, 0.525274720722,
                            0.456971502603, 0.104228668830, 0.104228668830]),
        ("var2_p1.csv", 4, [0.642903193503, 0.642903193503, 0.614614306781,
                            0.614614306781, 0.546525304394, 0.546525304394,
                            0.465958036074, 0.131815551172]),
        ("uschange.csv", 2, [0.650747126958, 0.650747126958, 0.426864696068,
                             0.426864696068, 0.407386546795, 0.407386546795,
                             0.358538490557, 0.358538490557, 0.236786554235,
                             0.179159927807])]
        m = fit_var(read_fixture(fx), p; deterministic=:const)
        C = companion(m.A)
        K = size(m.A[1], 1)
        @test size(C) == (K*p, K*p)
        @test sort(abs.(eigvals(C)); rev=true) ≈ mods atol=1e-9
        # and this IS what vars::roots() returns -- charter 5.3
        @test roots(m) ≈ mods atol=1e-9
    end
end

@testset "companion: block structure invariants" begin
    m = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    C = companion(m.A); K = 3
    @test C[1:K, 1:K]       ≈ m.A[1] atol=1e-14
    @test C[1:K, K+1:2K]    ≈ m.A[2] atol=1e-14
    @test C[K+1:2K, 1:K]    == I(K)
    @test all(iszero, C[K+1:2K, K+1:2K])
end
```

### 8.4 Rank deficiency

```julia
@testset "rank deficiency is refused (both references proceed silently)" begin
    Y = read_fixture("var2_p1.csv")
    Yc = hcat(Y, Y[:, 1])                      # exact duplicate column
    Z, X, _ = build_var_Z(Yc, 1, :const)
    @test_throws ArgumentError _mols(Z, X)
    e = try _mols(Z, X) catch err; err end
    msg = lowercase(sprint(showerror, e))
    @test occursin("rank", msg) || occursin("collinear", msg)
    @test_throws ArgumentError _mols(Z, X; method=:cholesky)

    # near-collinear but full rank must still work
    Yn = hcat(Y, Y[:, 1] .+ 1e-6 .* randn(MersenneTwister(1), size(Y,1)))
    Zn, Xn, _ = build_var_Z(Yn, 1, :const)
    @test_nowarn _mols(Zn, Xn)                 # :qr handles it
end
```

### 8.5 `tsmatrix` / `tsnames`

```julia
@testset "tsmatrix and tsnames" begin
    Y = read_fixture("var2_p1.csv")
    @test tsmatrix(Y) === Y                                   # zero copy
    @test tsmatrix((Y[:,1], Y[:,2])) ≈ Y atol=0
    @test tsmatrix([Y[:,1], Y[:,2]]) ≈ Y atol=0
    nt = (y1 = Y[:,1], y2 = Y[:,2])
    @test tsmatrix(nt) ≈ Y atol=0
    @test tsnames(nt) == [:y1, :y2]
    @test tsnames(Y) === nothing
    @test tsnames((Y[:,1], Y[:,2])) === nothing

    # ragged input is refused, not padded
    @test_throws ArgumentError tsmatrix((Y[:,1], Y[1:end-1, 2]))
    @test_throws ArgumentError tsmatrix((y1 = Y[:,1], y2 = Y[1:end-1, 2]))
    # a single vector is NOT a multivariate container
    @test_throws ArgumentError tsmatrix(Y[:, 1])
    # and tsvalues keeps its existing behaviour untouched
    @test tsvalues(Y[:, 1]) === Y[:, 1]
end
```

### 8.6 Type stability and allocations

```julia
@testset "type stability and allocations" begin
    Y = read_fixture("uschange.csv")
    Z, X, _ = build_var_Z(Y, 2, :const)
    @test (@inferred build_var_Z(Y, 2, :const)) isa Tuple
    @test eltype(Z) === Float64 && eltype(X) === Float64
    B, E, se, ZtZinv = _mols(Z, X)
    @test typeof(B) === Matrix{Float64}
    @test typeof(ZtZinv) === Matrix{Float64}

    m = fit_var(Y, 2; deterministic=:const)
    @test (@inferred companion(m.A)) isa Matrix{Float64}
    companion(m.A)                                   # warm up
    @test (@allocated companion(m.A)) < 10 * 10 * 8 * 3

    # build_var_Z allocates Z and X and essentially nothing else
    build_var_Z(Y, 2, :const)
    a = @allocated build_var_Z(Y, 2, :const)
    @test a < 3 * (size(Z,1)*size(Z,2) + size(X,1)*size(X,2)) * 8
end
```

---

## 9. Checklist

1. **Re-establish Julia** and re-verify every `[S]` claim before coding.
2. `tsmatrix` / `tsnames` in `src/interface.jl`, next to `tsvalues`,
   with an honest return-type docstring.
3. `_mols` next to `_ols` in `src/unitroot.jl`, sharing its weighted
   reduction and error style.
4. `build_var_Z` and `companion` in a new `src/varcore.jl`, included
   after `arx.jl` (for `_ols`/`_diagvec`).
5. Run the battery. Target ≥ 150 assertions in
   `test/test_mvprimitives.jl`.
6. Copy `scripts/ref_R_prim.R` and `scripts/ref_py_prim.py` to
   `test/verification/mvprimitives/`.
7. `@code_warntype` on all four entry points.
8. **Add 10.0 as a row in `development-sequence.md`'s Stage 10 table**,
   before 10.1, with the dependency note from §1.
9. **Add to the charter §5.10**: statsmodels' `get_var_endog` disagrees
   with its own `fit` on the trend column (§4.2), and both references
   proceed silently on rank-deficient input (§6).
10. No manual chapter — these are internal primitives. `tsmatrix` does
    belong in the container-agnosticism section of the existing manual,
    wherever `tsvalues` is documented.

---

## 10. Bundle additions for this item

```
fixtures/
  ref_R_prim.json      # lm(cbind(...)) coefficients, SEs, Sigma, Z rows, companion
                       #   matrices and eigenvalues for 5 cases, ar.ols at p=1,2,3
  ref_py_prim.json     # statsmodels counterparts, plus get_var_endog's Z layout
                       #   under all four trend settings and the collinear-input result
scripts/
  ref_R_prim.R         # regenerates ref_R_prim.json
  ref_py_prim.py       # regenerates ref_py_prim.json
```
