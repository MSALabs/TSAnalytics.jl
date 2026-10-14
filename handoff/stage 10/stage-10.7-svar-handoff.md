# Handoff: Stage 10.7 — Structural VAR

**Status: implemented ✅ (2026-10-13).** §4.1–4.2 and §5's identification rule were corrected in the build — see the overall handoff §5.13 item 22.

For a fresh Claude Code session with no prior context. **Read
`stage-10-overall-handoff.md` first**, then
`stage-10.3-irf-fevd-handoff.md` — this item reuses its entire MA and
standard-error machinery with a non-Cholesky `P`.

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every reference number below is `[V]`. **Julia was not installed in the
sandbox that produced this document** — re-verify any package-internal
claim before building on it.

---

## 1. Where this fits

- **Depends on:** 10.1 (`VARModel`), 10.3 (the MA representation, the
  `G` matrices, the IRF/FEVD path), `_optimize` (`src/optim.jl:57`),
  10.0's `companion`.
- **Feeds into:** nothing. The last item in the stage.
- **Reference size:** `[V]` `vars/R/SVAR.R` is **282 lines**;
  statsmodels' `svar_model.py` is **882** total, with `fit` 94,
  `_estimate_svar` 78, `loglike` 50. The largest single item in
  Stage 10 after 10.3.
- **The roadmap's note that Python support is weak is correct** — but
  not for the reason it suggests. statsmodels' SVAR works; its API has a
  trap that makes it look broken (§3.1).

---

## 2. Scope

### In

- **A-model**: `A y_t = … + ε_t`, `A` partly restricted, `B = I`.
- **B-model**: `y_t = … + B ε_t`, `B` partly restricted, `A = I`.
- **AB-model**: both partly restricted.
- **Blanchard-Quah**: long-run restrictions, `R` only.
- Structural IRF and FEVD, by delegating to 10.3 with `P = A⁻¹B`.
- Standard errors for the free elements of `A` and `B`.
- An over-identification likelihood-ratio test.
- Identification checking **before** optimization (§5).

### Out

- Sign-restriction identification (penalty-function or
  rejection-sampling). Neither reference ships it.
- External-instrument / proxy SVAR.
- `vars::SVEC` (structural VECM). A later row once 10.6 lands;
  `[V]` it is present in `vars` 1.6.1 and confirmed callable, but not
  exercised here.
- Narrative or event-based restrictions.

---

## 3. The references, and what agrees

### 3.1 statsmodels' SVAR works — the free-parameter marker is a string

`[V]` Passing `np.nan` to mark free parameters produces
`ValueError: zero-size array to reduction operation maximum which has no
identity`, which reads like a broken implementation. It is not. `[S]`
`svar_model.py SVAR.__init__`:

```python
A = A.astype("U")
A_mask = np.logical_or(A == "E", A == "e")
```

**Free parameters are marked with the string `"E"`**, in an object or
string array. `A.astype("U")` on a float array turns `nan` into
`"nan"`, which never equals `"E"`, so the mask is empty and the
optimizer is handed zero parameters.

```python
A = np.array([["1","0"],["E","1"]], dtype=object)      # correct
A = np.array([[1.0, 0.0],[np.nan, 1.0]])               # silently zero free params
```

Record this in the verification script, because the next person will
reach for `np.nan` too.

### 3.2 Once called correctly, the two references agree

`[V]` Just-identified A-model on `var2_p1`, p=2:

| Reference | `A[2,1]` |
|---|---|
| `vars::SVAR(estmethod="scoring")` | **−0.29382078758060526** |
| statsmodels `svar_type="A"` | **−0.29382078291590424** |
| difference | 4.66e-09 |
| `vars::SVAR(estmethod="direct")` | −0.2937500000000004 |

Trivariate A-model on `var3_p2`, p=2:

```
R  (scoring)  A = [ 1.0           0.0           0.0
                   -0.3176524896  1.0           0.0
                    0.2061129275 -0.1741739493  1.0 ]
py            A = [ 1.0           0.0           0.0
                   -0.3176465794  1.0           0.0
                    0.2061126777 -0.1741650884  1.0 ]
max|diff| = 8.86e-06
```

B-model, `var2_p1` p=2: R `B[2,1] = 0.2938207914`, statsmodels
`0.2938207727` — 1.9e-8. AB-model with `B = I` reproduces the A-model's
`A` exactly in statsmodels.

**Agreement is to optimizer tolerance, not to machine precision**,
because these are numerically optimized MLEs. Assert with
`atol=1e-5`, and make the structural checks in §4 carry the weight —
they are exact.

### 3.3 Do not use `vars`' `estmethod="direct"`

`[V]` It returns `−0.2937500000000004` against the correct
`−0.293820788` — wrong in the fourth decimal, and the trailing
`...0004` on a suspiciously round `−0.29375` shows it stopped on a grid
point. `[V]` R also emits *"one-dimensional optimization by Nelder-Mead
is unreliable: use 'Brent' or optimize() directly"*.

**`estmethod="scoring"` is the reference.** Say so in the verification
script, and do not reproduce the `direct` path.

---

## 4. Exact structural checks — these do the real verifying

Because §3.2's agreement is only to optimizer tolerance, the tests that
pin correctness are the ones with exact answers.

### 4.1 A just-identified SVAR does not change the likelihood

`[V]` statsmodels' SVAR log-likelihood for every just-identified model
on `var2_p1` p=2 is **−496.03769350**, and `[V]`
`stage-10.1-var-handoff.md` §7.5 records the plain VAR(2)
log-likelihood as **−496.03769349818344**. Identical.

```julia
@test loglikelihood(svar) ≈ loglikelihood(var) atol=1e-7
```

That holds for the A-model, the B-model and the AB-model — `[V]` all
three report the same value. It is the single best check available: a
just-identified structural model is a reparameterization, not a
different fit.

### 4.2 A just-identified recursive SVAR reproduces the Cholesky factor

`[V]` For the lower-triangular A-model with `B = I`:

```
A^-1 B                  = [1.0           0.0
                           0.2938207876  1.0]
chol(Sigma.U) lower     = [10.0          0.0
                            2.9382078758 10.0]
```

The two are proportional, not equal — because `[S]` `vars`' `Sigma.U` is
**`A⁻¹BB'A'⁻¹ × 100`** (`vars/R/SVAR.R:256` and `:278`, where the `*100`
is explicit). Against the VAR's own `sigma_df`, the identity to assert
is:

```julia
P = A \ B
@test P * P' ≈ var.sigma_df  atol=1e-9      # exact, no optimizer involved
```

That is the defining property of any SVAR decomposition, and it holds
for every identification scheme. **Make it the first test.**

And for the recursive case specifically, `P` must equal the Cholesky
factor up to column signs:

```julia
L = cholesky(Hermitian(var.sigma_df)).L
@test abs.(P) ≈ abs.(L) atol=1e-7
```

### 4.3 Blanchard-Quah's long-run matrix is lower triangular by construction

`[V]` `vars::BQ` on `var2_p1` p=2:

```
LRIM = [1.7871223406  0.0
        1.0736811371  1.1791967059]
B    = [1.0046922922  0.0484142194
        0.2599992736  0.7447002532]
```

and on `var3_p2` p=2:

```
LRIM = [2.3683347141  0.0           0.0
        1.6382078972  2.4942601615  0.0
        1.6770695856  1.5949182158  2.7714602003]
B    = [ 0.9685684656 -0.2929475729 -0.2978983464
         0.4292713958  0.6329456651 -0.4131906819
         0.2618396484  0.5107735591  0.9122489877]
```

The upper triangle of `LRIM` is **exactly zero** — that *is* the
identifying restriction, and asserting it needs no reference:

```julia
@test all(iszero, triu(bq.lrim, 1))
@test bq.lrim ≈ lr_effects(var) * bq.B atol=1e-9   # LRIM = (I - sum A_j)^-1 B
@test bq.B * bq.B' ≈ var.sigma_df atol=1e-9        # still a valid decomposition
```

`[V]` There is **no statsmodels equivalent of `BQ`**, so these exact
identities plus `vars`' values are the whole verification.

---

## 5. Identification must be checked before optimizing

`[V]` `vars::SVAR` checks first and refuses with messages that name the
problem:

| Case | `vars::SVAR` | statsmodels |
|---|---|---|
| all elements free | *"Model is not identified, choose different settings for 'Amat' and/or 'Bmat'."* | `LinAlgError: Singular matrix` |
| no free elements | *"No parameters provided for optimisation, i.e. neither 'Amat' nor 'Bmat' do contain NA elements."* | — |
| free at the same position in A and B | *"SVAR not identified. Free parameters at the same positions in Amat and Bmat."* | — |
| `A` not supplied for type A | — | clean `ValueError: SVAR of type A or AB but A array not given.` |
| `np.nan` as the free marker | — | `ValueError: zero-size array to reduction...` (§3.1) |
| bad `svar_type` | — | clean `ValueError: svar_type must be one of: 'A', 'B', 'AB'` |

**Follow `vars` here.** The counting condition — the number of free
parameters must equal `K(K+1)/2` for exact identification, and may
exceed it for over-identification but never fall short — is cheap to
check and turns a `LinAlgError` from deep inside an optimizer into a
sentence the caller can act on.

```julia
nfree = count(ismissing, A) + count(ismissing, B)
nfree == 0 && throw(ArgumentError("svar: no free parameters -- mark them with `missing`"))
nfree <  K*(K+1)÷2 && throw(ArgumentError(
    "svar: $nfree free parameters is fewer than the $(K*(K+1)÷2) needed to identify " *
    "a K=$K system; the model is under-identified"))
any(i -> ismissing(A[i]) && ismissing(B[i]), eachindex(A)) && throw(ArgumentError(
    "svar: free parameters at the same position in A and B -- not identified"))
```

**Use `missing`, not `NaN`, as the free marker.** It is the Julia idiom,
it cannot be confused with a computed value, and `[V]` statsmodels'
`np.nan` trap is a direct demonstration of what the alternative costs.
Accept a `Matrix{Union{Missing,Float64}}`.

---

## 6. Proposed Julia API

```julia
struct SVARModel <: TimeSeriesModel
    var::VARModel
    svar_type::Symbol                  # :A | :B | :AB | :bq
    A::Matrix{Float64}; B::Matrix{Float64}
    A_mask::BitMatrix; B_mask::BitMatrix
    stderr_A::Matrix{Float64}; stderr_B::Matrix{Float64}
    P::Matrix{Float64}                 # A \ B -- what 10.3 consumes
    lrim::Union{Nothing,Matrix{Float64}}     # :bq only
    loglik::Float64
    lr_test::Union{Nothing,NamedTuple}       # over-identified only: (; statistic, df, pvalue)
    converged::Bool
    iterations::Int
end

fit_svar(m::VARModel; A = nothing, B = nothing,
         method::Symbol = :scoring,          # :scoring | :mle
         maxiter::Integer = 200, tol::Real = 1e-10) -> SVARModel

blanchard_quah(m::VARModel) -> SVARModel     # svar_type = :bq

irf(s::SVARModel, h::Integer; kwargs...)     # delegates to 10.3 with P = s.P
fevd(s::SVARModel, h::Integer)
```

`irf` and `fevd` must be **methods on 10.3's functions**, not copies.
`[S]` statsmodels does exactly this — `svar_ma_rep` differs from
`orth_ma_rep` only in which `P` it multiplies by. If 10.3 is written
with `P` as an argument rather than hard-coded to the Cholesky factor,
10.7 adds no new recursion at all. **Write 10.3 that way.**

---

## 7. Test battery

Suggested file: `test/test_svar.jl`. Target **≥ 170 assertions**.
References in `fixtures/ref_R_svar.json` and `fixtures/ref_py_svar.json`.

### 7.1 The exact identities first

```julia
@testset "every SVAR decomposition satisfies P P' = Sigma" begin
    v  = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    v3 = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    A2 = [1.0 0.0; missing 1.0]
    B2 = [1.0 0.0; missing 1.0]
    A3 = [1.0 0.0 0.0; missing 1.0 0.0; missing missing 1.0]
    for (m, s) in [(v, fit_svar(v; A=A2)), (v, fit_svar(v; B=B2)),
                   (v, fit_svar(v; A=A2, B=Matrix{Union{Missing,Float64}}(I,2,2))),
                   (v3, fit_svar(v3; A=A3)), (v, blanchard_quah(v)),
                   (v3, blanchard_quah(v3))]
        @test s.P * s.P' ≈ m.sigma_df atol=1e-8
        @test s.converged
    end
end

@testset "a just-identified SVAR does not change the likelihood" begin
    v = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    @test loglikelihood(v) ≈ -496.03769349818344 atol=1e-9
    for s in (fit_svar(v; A=[1.0 0.0; missing 1.0]),
              fit_svar(v; B=[1.0 0.0; missing 1.0]),
              blanchard_quah(v))
        @test loglikelihood(s) ≈ loglikelihood(v) atol=1e-7
        @test s.lr_test === nothing              # just-identified: no LR test
    end
    v3 = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    @test loglikelihood(fit_svar(v3; A=[1.0 0.0 0.0; missing 1.0 0.0; missing missing 1.0])) ≈
          loglikelihood(v3) atol=1e-6
end

@testset "a recursive A-model reproduces the Cholesky factor" begin
    v = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    s = fit_svar(v; A=[1.0 0.0; missing 1.0])
    L = cholesky(Hermitian(v.sigma_df)).L
    @test abs.(s.P) ≈ abs.(Matrix(L)) atol=1e-7
    # and the free element is the normalized off-diagonal
    @test s.A[2,1] ≈ -L[2,1]/L[1,1] atol=1e-7
end
```

### 7.2 Against both references

```julia
@testset "A-model and B-model vs vars(scoring) and statsmodels" begin
    v = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    a = fit_svar(v; A=[1.0 0.0; missing 1.0])
    @test a.A[2,1] ≈ -0.29382078758060526 atol=1e-5     # vars scoring
    @test a.A[2,1] ≈ -0.29382078291590424 atol=1e-5     # statsmodels
    @test a.A[1,1] == 1.0 && a.A[1,2] == 0.0 && a.A[2,2] == 1.0
    @test a.B ≈ I(2) atol=1e-12
    # vars' estmethod="direct" gives -0.29375 -- wrong in the 4th decimal.
    # We must NOT match it.
    @test !isapprox(a.A[2,1], -0.29375; atol=1e-6)

    b = fit_svar(v; B=[1.0 0.0; missing 1.0])
    @test b.B[2,1] ≈ 0.2938207914 atol=1e-5             # vars
    @test b.B[2,1] ≈ 0.2938207727 atol=1e-5             # statsmodels
    @test b.A ≈ I(2) atol=1e-12

    v3 = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    a3 = fit_svar(v3; A=[1.0 0.0 0.0; missing 1.0 0.0; missing missing 1.0])
    @test a3.A ≈ [ 1.0           0.0           0.0
                  -0.3176524896  1.0           0.0
                   0.2061129275 -0.1741739493  1.0] atol=1e-4   # optimizer tolerance
    @test all(iszero, triu(a3.A, 1))
end
```

### 7.3 Blanchard-Quah

```julia
@testset "blanchard_quah: the long-run matrix is lower triangular" begin
    v = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    bq = blanchard_quah(v)
    @test bq.svar_type == :bq
    @test all(iszero, triu(bq.lrim, 1))                  # the RESTRICTION, exact
    @test bq.lrim ≈ [1.7871223406 0.0
                     1.0736811371 1.1791967059] atol=1e-7
    @test bq.B ≈ [1.0046922922 0.0484142194
                  0.2599992736 0.7447002532] atol=1e-7
    # LRIM = (I - sum A_j)^-1 B, which needs no reference
    @test bq.lrim ≈ lr_effects(v) * bq.B atol=1e-8
    @test bq.B * bq.B' ≈ v.sigma_df atol=1e-9

    v3 = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    bq3 = blanchard_quah(v3)
    @test all(iszero, triu(bq3.lrim, 1))
    @test bq3.lrim ≈ [2.3683347141 0.0          0.0
                      1.6382078972 2.4942601615 0.0
                      1.6770695856 1.5949182158 2.7714602003] atol=1e-6
    @test bq3.B ≈ [ 0.9685684656 -0.2929475729 -0.2978983464
                    0.4292713958  0.6329456651 -0.4131906819
                    0.2618396484  0.5107735591  0.9122489877] atol=1e-6
    # BQ is NOT recursive in B -- that is the whole point
    @test !all(iszero, triu(bq3.B, 1))
end
```

### 7.4 Structural IRF and FEVD delegate to 10.3

```julia
@testset "structural IRF reuses 10.3, with P instead of the Cholesky factor" begin
    v = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    bq = blanchard_quah(v)
    r = irf(bq, 4)
    Phi = ma_rep(v, 4)
    for h in 1:5
        @test r.structural[h] ≈ Phi[h] * bq.P atol=1e-9   # Psi_i = Phi_i P
    end
    @test r.structural[1] ≈ bq.P atol=1e-12
    # and against vars' own svar irf
    ref = load_ref_svar_irf()
    for h in 1:5; @test r.structural[h] ≈ ref[h] atol=1e-6; end

    f = fevd(bq, 4)
    for i in 1:2, h in 1:4
        @test sum(f.decomp[i, h, :]) ≈ 1.0 atol=1e-12
    end
    ref_f = load_ref_svar_fevd()
    @test f.decomp ≈ ref_f atol=1e-6

    # a recursive A-model's structural IRF must equal the ORTHOGONALISED IRF
    a = fit_svar(v; A=[1.0 0.0; missing 1.0])
    ra = irf(a, 4); ro = irf(v, 4)
    for h in 1:5; @test abs.(ra.structural[h]) ≈ abs.(ro.orth[h]) atol=1e-6; end
end
```

### 7.5 Over-identification and the LR test

```julia
@testset "over-identified models get an LR test; just-identified do not" begin
    v = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    # K=3 needs K(K+1)/2 = 6 free parameters for exact identification
    A_just = [1.0 0.0 0.0; missing 1.0 0.0; missing missing 1.0]   # 3 free + 3 unit diag
    s = fit_svar(v; A=A_just)
    @test s.lr_test === nothing
    @test count(s.A_mask) == 3

    # fewer restrictions than needed -> over-identified -> an LR test appears
    B_over = Matrix{Union{Missing,Float64}}(I, 3, 3)
    B_over[2,1] = missing; B_over[3,1] = missing
    A_over = [1.0 0.0 0.0; missing 1.0 0.0; missing missing 1.0]
    so = fit_svar(v; A=A_over, B=B_over)
    @test so.lr_test !== nothing
    @test so.lr_test.df > 0
    @test 0.0 <= so.lr_test.pvalue <= 1.0
    @test so.lr_test.statistic >= 0.0
    # the LR statistic is T*(log det Sigma_structural - log det Sigma_VAR)
    @test loglikelihood(so) <= loglikelihood(v) + 1e-8
end
```

### 7.6 Identification refusals

```julia
@testset "identification is checked BEFORE optimizing" begin
    v = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)

    # all free: under-identified. vars says so; statsmodels gives LinAlgError.
    Aall = Matrix{Union{Missing,Float64}}(fill(missing, 2, 2))
    @test_throws ArgumentError fit_svar(v; A=Aall)
    e = try fit_svar(v; A=Aall) catch err; err end
    @test occursin("identif", lowercase(sprint(showerror, e)))

    # no free parameters at all
    @test_throws ArgumentError fit_svar(v; A=Matrix{Union{Missing,Float64}}(I,2,2))
    e = try fit_svar(v; A=Matrix{Union{Missing,Float64}}(I,2,2)) catch err; err end
    @test occursin("free", lowercase(sprint(showerror, e)))

    # free at the same position in A and B
    Asame = [1.0 0.0; missing 1.0]; Bsame = [1.0 0.0; missing 1.0]
    @test_throws ArgumentError fit_svar(v; A=Asame, B=Bsame)
    e = try fit_svar(v; A=Asame, B=Bsame) catch err; err end
    @test occursin("same position", lowercase(sprint(showerror, e)))

    # neither A nor B
    @test_throws ArgumentError fit_svar(v)
    # wrong shape
    @test_throws ArgumentError fit_svar(v; A=[1.0 0.0 0.0; missing 1.0 0.0])
    # NaN must NOT be accepted as a free marker -- `missing` is the marker
    @test_throws ArgumentError fit_svar(v; A=[1.0 0.0; NaN 1.0])
    @test_throws ArgumentError fit_svar(v; method=:bogus, A=[1.0 0.0; missing 1.0])
    @test_throws ArgumentError fit_svar(v; A=[1.0 0.0; missing 1.0], maxiter=0)
end
```

### 7.7 Standard errors, convergence, contract

```julia
@testset "standard errors and convergence" begin
    v = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    s = fit_svar(v; A=[1.0 0.0; missing 1.0])
    @test s.stderr_A[2,1] > 0.0
    @test s.stderr_A[1,1] == 0.0 && s.stderr_A[1,2] == 0.0   # restricted: no se
    @test s.stderr_A[2,2] == 0.0
    @test s.converged && s.iterations > 0
    ref = load_ref_svar("A_k2_scoring")
    @test s.stderr_A[2,1] ≈ ref.Ase[2,1] rtol=1e-3           # vars' Ase

    # a tighter tolerance must not move the answer materially
    t = fit_svar(v; A=[1.0 0.0; missing 1.0], tol=1e-14, maxiter=1000)
    @test t.A[2,1] ≈ s.A[2,1] atol=1e-9
end

@testset "contract, show, type stability" begin
    v = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    s = fit_svar(v; A=[1.0 0.0; missing 1.0])
    @test s isa TimeSeriesModel
    @test isconcretetype(SVARModel)
    @test nobs(s) == nobs(v)
    @test residuals(s) == residuals(v)       # the reduced form is unchanged
    str = sprint(show, s)
    @test occursin("SVAR", str)
    @test occursin("A", str)
    @test occursin("converged", lowercase(str))
    @test occursin("Blanchard", sprint(show, blanchard_quah(v)))
end
```

---

## 8. Performance and numerics

Per charter §7. For this item specifically:

1. **Delegate the MA recursion.** If 10.3 takes `P` as an argument,
   10.7 adds no recursion. Verify that before writing anything here.
2. **The likelihood is concentrated.** Only the free elements of `A`
   and `B` are optimized; `Σ` follows. `[V]` The free-parameter count is
   3 for the trivariate A-model — a tiny problem, so the optimizer
   choice matters less than the starting point.
3. **Start from the Cholesky factor.** For a recursive scheme it is the
   answer; for anything else it is a feasible point satisfying
   `PP' = Σ`. `[V]` `vars`' `direct` method's failure to reach the
   optimum (§3.3) is what a poor start plus Nelder-Mead costs.
4. **Reuse `_optimize`** (`src/optim.jl:57`) rather than calling `Optim`
   directly, so the convergence reporting matches the rest of the
   package. `[S]` That is the pattern every GARCH and ETS fit already
   follows.
5. **Analytic scoring where it is cheap.** `vars`' `scoring` method
   beats its `direct` method decisively; the score of the concentrated
   likelihood with respect to the free elements is available in closed
   form. If the first implementation uses a numerical gradient, say so
   in the docstring and leave the analytic score as a flagged
   improvement rather than an unstated gap.

---

## 9. Checklist

1. **Re-establish Julia** and re-verify every `[S]` claim before coding.
2. **Confirm 10.3 exposes `P` as an argument.** If it does not, change
   10.3 first — otherwise 10.7 duplicates the MA recursion.
3. Implement identification checking (§5) **before** the estimator, and
   test it first (§7.6).
4. Then the exact identities (§7.1): `PP' = Σ`, the unchanged
   likelihood, the Cholesky reproduction. Those must pass before any
   reference comparison.
5. Then the A, B and AB estimators, then Blanchard-Quah.
6. Structural IRF and FEVD by delegation.
7. Run the battery. Target ≥ 170 assertions in `test/test_svar.jl`.
8. Copy `scripts/ref_svar.R` and `scripts/ref_py_svar.py` to
   `test/verification/svar/`. **Keep the `"E"`-marker comment** in the
   Python script (§3.1).
9. `@code_warntype` on `fit_svar` and `blanchard_quah`.
10. **Add to the charter §5.10**: statsmodels' SVAR marks free
    parameters with the string `"E"` and gives an opaque empty-array
    error for `np.nan`; `vars`' `estmethod="direct"` does not reach the
    optimum and R warns about it; `vars`' `Sigma.U` is scaled by 100
    (`SVAR.R:278`).
11. Manual section: the three identification schemes, the
    Blanchard-Quah long-run restriction, and the honest statement that
    identification is an assumption the data cannot check — which is
    the thing users most need told.

---

## 10. Bundle additions

```
fixtures/
  ref_R_svar.json    # A-model (scoring AND direct), B-model, just-identified A,
                     #   Blanchard-Quah for k=2 and k=3 with LRIM, trivariate A-model,
                     #   SVAR irf and fevd, Sigma.U, Ase/Bse, and vars' three
                     #   identification error messages
  ref_py_svar.json   # statsmodels SVAR for types A, B and AB (k=2) and A (k=3), with
                     #   llf, A, B, sigma_u and svar_irfs; plus the "E"-vs-np.nan
                     #   demonstration and its error strings
scripts/
  ref_svar.R         # regenerates ref_R_svar.json
  ref_py_svar.py     # regenerates ref_py_svar.json -- note the "E" marker comment
```

Fixtures are 10.1's `var2_p1.csv` and `var3_p2.csv`; no new data.
