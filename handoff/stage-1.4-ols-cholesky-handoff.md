# Handoff: Stage 1.4 — `_ols` with both QR (default) and Cholesky methods

Status: implemented ✅ (`src/unitroot.jl`, `test/test_unitroot.jl`).
Before implementing, independently verified in Julia (not just trusted
from this doc) that `cholesky(Hermitian(XtX))` on a singular `X'X`
(duplicate-column case) throws exactly `PosDefException`, and that QR's
`\` on the same collinear `X` returns a finite (if numerically extreme --
coefficients on the order of 1e14 that cancel, the known signature of an
unpivoted least-squares solve on a rank-deficient system) result rather
than erroring -- both match section 1's requirements exactly. One caveat
on section 4's transcript: the specific `beta` values shown there don't
reproduce under Julia's `Random.seed!(1)` (same cross-language RNG
mismatch as the Stage 1.3 handoff -- numpy and Julia don't share a stream
even with "the same" seed) -- but the actual `@testset` code in that
section never hardcodes those values, only compares `:qr` against
`:cholesky` relatively, so this doesn't affect anything: the structural
QR/Cholesky agreement (~1e-15/1e-16, both weighted and unweighted) was
independently confirmed in Julia directly. No deltas from the proposed
implementation otherwise.

For a fresh Claude Code session picking this up with no prior context.
Covers extending the existing `_ols` helper (currently QR-only, with GLS
via weights already added) to also support a Cholesky-based path,
following GLM.jl's `cholpred`/`qrpred` dispatch pattern as a validated
reference — **not as a dependency**. No part of GLM.jl is imported or
called; its source is read to understand a correct, battle-tested design,
then rebuilt natively for `_ols`'s own (smaller, non-GLM-shaped)
requirements.

## Where this fits

- **Already built:** `_ols(X, y; weights=nothing)` in `unitroot.jl`,
  QR-only, with weighted least squares (GLS) via the standard
  weighted-QR reduction.
- **This handoff adds:** a `method::Symbol=:qr` keyword, `:qr` (default,
  unchanged behavior) or `:cholesky` (new).
- **Used by:** `adf_test`, `kpss_test` today; `pacf`'s `:ols` method
  (Stage 1.3 handoff) if that's landed; will matter more once anything
  does many repeated regressions at larger `p` (e.g. `pacf`'s `:ols`
  path loops a regression per lag up to `maxlag`).

---

## 1. Requirements

1. `_ols` must support **both** `:qr` and `:cholesky` as explicit,
   user-selectable methods — not just internally-chosen.
2. **`:qr` stays the default** — existing call sites and existing tests
   must not change behavior unless `method=:cholesky` is passed
   explicitly.
3. Both methods must support the existing `weights=` (GLS) keyword,
   producing numerically equivalent results to each other (verified
   below) and to the existing QR-only implementation.
4. Both methods return the same `(beta, resid, se)` tuple shape — callers
   shouldn't need to know which method was used.
5. `:cholesky` must fail with a clear, actionable error (not a raw LAPACK
   exception) when `X'X` isn't positive definite (e.g. collinear
   regressors) — and that error message should point the caller at
   `:qr` as the fix, since QR handles that case more gracefully.
6. **Explicitly out of scope for this pass:** pivoted/rank-deficient
   handling (GLM.jl's `dropcollinear`) for either method. GLM.jl's own
   code shows both `DensePredQR` and `DensePredChol` support a pivoted
   variant (`QRPivoted` / `CholeskyPivoted`) — worth knowing that
   precedent exists if this becomes a real requirement later, but it
   adds real complexity (permutation bookkeeping) for no current caller
   that needs it. Same deferral as the original `_ols` handoff.

---

## 2. What GLM.jl actually does (verified from source, not memory)

The dispatch pattern this is modeled on, from `GLM.jl/src/lm.jl`:

```julia
if method === :cholesky
    fit!(LinearModel(LmResp(y, weightsvec), cholpred(X, dropcollinear, weightsvec), f))
elseif method === :qr
    fit!(LinearModel(LmResp(y, weightsvec), qrpred(X, dropcollinear, weightsvec), f))
```

**`DensePredChol`** (`GLM.jl/src/linpred.jl`, verified from source):

```julia
function DensePredChol(X::StridedMatrix, pivot::Bool)
    F = Hermitian(float(X'X))
    T = eltype(F)
    F = pivot ? cholesky!(F, Val(true), tol=-one(T), check=false) : cholesky!(F)
    DensePredChol(AbstractMatrix{T}(X), zeros(T, size(X,2)), zeros(T, size(X,2)),
                   zeros(T, size(X,2)), F, similar(X, T), similar(cholfactors(F)))
end
```

i.e.: form the normal-equations matrix `X'X` explicitly, wrap in
`Hermitian` (so Julia's `cholesky` dispatches correctly and treats it as
symmetric positive definite), factorize. Solving:

```julia
function delbeta!(p::DensePredChol{T,<:Cholesky}, r::Vector{T}) where {T<:BlasReal}
    ldiv!(p.chol, mul!(p.delbeta, transpose(p.X), r))
    return p
end
```

i.e.: `beta = (X'X) \ (X'r)`, using the precomputed factorization for the
solve. For weighted regression, GLM's docstring for `DensePredChol`
confirms: *"chol: a Cholesky object created from X'X, possibly using row
weights"* — weights are baked into the normal equations before
factorizing, the direct analogue of `_ols`'s existing weighted-QR
reduction (scale rows by `√weights` first, in either method's case).

**The tradeoff, stated directly in GLM.jl's own docs** (not a guess):

> Currently there are two dense predictor types, DensePredQR and
> DensePredChol... The Cholesky version is faster but somewhat less
> accurate than the QR version.

Why, precisely: QR factorizes `X` directly, so its numerical error scales
with `cond(X)`. Cholesky factorizes `X'X`, and **`cond(X'X) = cond(X)^2`**
— squaring the condition number is the actual mechanism behind "less
accurate." For well-conditioned regressions (the common case — a
constant, a trend, a handful of lag terms) this rarely matters in
practice; it matters when regressors are nearly collinear.

Speed: for `n >> p` (the typical case — many observations, few
parameters), both methods are `O(np^2)`-dominated, but Cholesky's
`BLAS syrk` (symmetric rank-`p` update to form `X'X`) has a smaller
constant factor than a full QR factorization — hence "faster" in
practice, not just asymptotically.

---

## 3. Implementation

Rebuilt for `_ols`'s actual shape (no `LinPred`/`LmResp` abstraction,
no mutability/scratch-buffer machinery GLM needs for IRLS reuse across
GLM iterations — `_ols` is a single one-shot solve, so none of that
applies):

```julia
using LinearAlgebra: qr, dot, I, cholesky, Hermitian, PosDefException

"""
    _ols(X, y; weights=nothing, method::Symbol=:qr)

Fit y = X*beta + e by (weighted) least squares. `method=:qr` (default)
factorizes X directly via QR -- the more numerically robust choice,
matching GLM.jl's own stated rationale for defaulting to it. `method=
:cholesky` instead factorizes the normal equations X'X -- faster in
practice for well-conditioned problems, but loses accuracy proportional
to cond(X)^2 rather than cond(X), per GLM.jl's own documented tradeoff
(verified from its source, not reimplemented from it -- see the Stage 1.4
handoff doc for the exact quote and reasoning).

GLS is implemented identically for both methods via the standard
weighted reduction: scale rows of X and y by sqrt(weights), then solve
the resulting OLS problem with the chosen method.

Returns `(beta, residuals, se)` -- `residuals` are on the original
(unweighted) scale; `se` uses the usual `sigma^2*(X'X)^-1` sandwich (or
its weighted analogue), not HAC.

`:cholesky` throws an `ArgumentError` (not a raw `PosDefException`) if
X'X isn't positive definite -- e.g. collinear regressors -- and points
the caller at `:qr`, which handles that case more gracefully.

Not yet implemented: pivoted/rank-deficient handling for either method
(GLM.jl's `dropcollinear`) -- deferred until a concrete caller needs it;
see the Stage 1.4 handoff doc for why.
"""
function _ols(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real};
              weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
              method::Symbol=:qr)
    method in (:qr, :cholesky) || throw(ArgumentError("method must be :qr or :cholesky"))
    n, k = size(X)

    if weights !== nothing
        length(weights) == n || throw(ArgumentError("weights must have length n"))
        all(>(0), weights) || throw(ArgumentError("weights must be positive"))
        sw = sqrt.(weights)
        Xw = X .* sw
        yw = y .* sw
    else
        Xw = X
        yw = y
    end

    dof = n - k
    dof > 0 || throw(ArgumentError("not enough observations for the requested regression"))

    local beta, XtX_inv
    if method === :qr
        F = qr(Xw)
        beta = F \ yw
        R = F.R
        Rinv = R \ Matrix{Float64}(I(k))
        XtX_inv = Rinv * Rinv'
    else # :cholesky
        XtX = Xw' * Xw
        Xty = Xw' * yw
        C = try
            cholesky(Hermitian(Matrix(XtX)))
        catch e
            e isa PosDefException &&
                throw(ArgumentError("_ols: method=:cholesky failed -- X'X is not positive definite " *
                                     "(likely collinear regressors); try method=:qr instead"))
            rethrow()
        end
        beta = C \ Xty
        XtX_inv = C \ Matrix{Float64}(I(k))
    end

    resid = y - X * beta          # residuals on the ORIGINAL scale
    wresid = yw - Xw * beta       # weighted residuals, for sigma2
    sigma2 = dot(wresid, wresid) / dof
    se = sqrt.(sigma2 .* _diagvec(XtX_inv))
    return beta, resid, se
end
```

`_diagvec` already exists from the original `_ols` implementation --
reuse as-is.

---

## 4. Hand-verified test values

Verified in Python/numpy that QR and Cholesky agree to machine precision
(`~1e-15`), both weighted and unweighted, before writing this up:

```
X: [1, x1, x2] with n=50, k=3, seed=1
beta_true = [2.0, -1.5, 0.7]
QR       beta = [2.02615592, -1.49274416, 0.67615492]
Cholesky beta = [2.02615592, -1.49274416, 0.67615492]
max |QR - Cholesky| (weighted)   = 4.44e-16
max |QR - Cholesky| (unweighted) = 1.11e-15
```

```julia
using Test, Random

@testset "_ols :qr vs :cholesky agreement" begin
    Random.seed!(1)
    n, k = 50, 3
    X = hcat(ones(n), randn(n), randn(n))
    beta_true = [2.0, -1.5, 0.7]
    y = X*beta_true .+ 0.3.*randn(n)
    w = 1.0 .+ rand(n)

    beta_qr,   resid_qr,   se_qr   = _ols(X, y; method=:qr)
    beta_chol, resid_chol, se_chol = _ols(X, y; method=:cholesky)
    @test isapprox(beta_qr, beta_chol; atol=1e-8)
    @test isapprox(se_qr, se_chol; atol=1e-8)

    # weighted: both methods must still agree
    beta_qr_w, _, _   = _ols(X, y; weights=w, method=:qr)
    beta_chol_w, _, _ = _ols(X, y; weights=w, method=:cholesky)
    @test isapprox(beta_qr_w, beta_chol_w; atol=1e-8)

    # default method is :qr -- calling without `method=` must match explicit :qr
    beta_default, = _ols(X, y)
    @test isapprox(beta_default, beta_qr; atol=1e-12)

    # invalid method
    @test_throws ArgumentError _ols(X, y; method=:bogus)

    # collinear regressors: :cholesky should fail informatively, :qr should not
    Xcol = hcat(X, X[:, 2])  # duplicate column -> singular X'X
    @test_throws ArgumentError _ols(Xcol, y; method=:cholesky)
    beta_qr_col, = _ols(Xcol, y; method=:qr)  # QR's `\` handles this via least-squares
    @test all(isfinite, beta_qr_col)
end
```

---

## 5. What to do with this

1. Add the `method::Symbol=:qr` keyword to the existing `_ols` in
   `unitroot.jl` per section 3.
2. Run the tests in section 4 — the QR/Cholesky agreement check and the
   collinearity-handling check are the two that actually matter; the rest
   are regression guards.
3. No changes needed to `adf_test`/`kpss_test` themselves — they don't
   need to expose `method=` unless there's a reason to; `_ols`'s own
   default (`:qr`) stays what they use unless someone deliberately wants
   `:cholesky`'s speed for a large-`p` case later (e.g. `pacf`'s `:ols`
   method, if it ever needs to be faster at large `maxlag`).
4. Update `development-sequence.md`'s Stage 1.4 row: mark the Cholesky
   addition, and note the pivoting deferral explicitly so it isn't
   silently forgotten (same pattern as the original `_ols` handoff).
