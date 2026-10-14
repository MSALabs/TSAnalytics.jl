# Handoff: Stage 10.6 — VECM

**Status: implemented ✅ (2026-10-13).**

For a fresh Claude Code session with no prior context. **Read
`stage-10-overall-handoff.md` first**, then
`stage-10.5-johansen-handoff.md` — this item consumes its rank and
cointegrating vectors, and inherits its deterministic-vocabulary table.

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every reference number below is `[V]`. **Julia was not installed in the
sandbox that produced this document** — re-verify any package-internal
claim before building on it.

---

## 1. A correction, and why it changes the difficulty

`stage-10-dependency-list.md` §4.5 recorded that VECM has *"three
references, three normalizations"* and rated 10.6 the hardest item in
the stage. **That was wrong, and the reason was a deterministic
mismatch, not a normalization difference.**

`[V]` Once the deterministic specification is matched —
`urca::ca.jo(ecdet="none")` + `cajorls` ↔ statsmodels
`VECM(deterministic="co")` ↔ `tsDyn::VECM(estim="ML", include="const")` —
**all three agree to machine precision on every fixture**:

| Fixture | k | r | max‖β_urca − β_sm(co)‖ | max‖α_urca − α_sm‖ |
|---|---|---|---|---|
| `coint2_yes` | 1 | 1 | 2.11e-15 | 8.73e-14 |
| `coint2_yes` | 2 | 1 | 3.11e-15 | 3.11e-14 |
| `coint3_r1` | 1 | 1 | 1.39e-15 | 3.32e-14 |
| `coint3_r1` | 1 | 2 | 2.66e-15 | — |
| `coint3_r2` | 1 | 2 | 9.77e-15 | — |

`[V]` And `tsDyn::VECM(estim="ML")` matches too: on `coint3_r1`,
β = `[1, −0.496504585, −0.9777300348]`, identical to `urca`'s to the
last printed digit.

**So 10.6 has three agreeing references, not three conventions.**
Revised difficulty: **moderate**, not hard. The remaining work is the
deterministic vocabulary (§3), the two genuinely different estimators
(§4), and inference on β.

- **Depends on:** 10.5 (rank and β), 10.0 (`_mols`, `build_var_Z`),
  `_chisq_ccdf`.
- **Feeds into:** nothing structurally. A VECM forecast and its IRF
  reuse 10.1's `VARForecast` and 10.3's machinery through the
  VAR representation (§6).

---

## 2. Scope

### In

- `fit_vecm(Y; lags, rank, deterministic, method)` → `VECMModel`.
- ML (reduced-rank regression, Johansen's) and two-step OLS
  (Engle-Granger's), both named.
- α, β, Γ (the short-run coefficients), Σ, log-likelihood.
- Standard errors for α, β and Γ.
- β normalization schemes, inherited from 10.5's `normalize_beta`.
- The VAR representation, so 10.1's forecast and 10.3's IRF apply.
- `forecast` / `predict` as exact aliases, per charter §2 item 5.
- Full StatsAPI contract.

### Out

- Restricted-β / restricted-α likelihood-ratio tests
  (`urca::blrtest`, `ablrtest`, `bh5lrtest`, `bh6lrtest`). The natural
  next row.
- Threshold VECM (`tsDyn::TVECM`).
- Seasonal cointegration.
- VECM residual diagnostics — 10.8's tests apply to the residuals
  directly; note that in the manual rather than duplicating them.

---

## 3. The deterministic vocabulary — eleven options in statsmodels, three in urca

`[V]` statsmodels' `deterministic` accepts
`'' | 'n' | 'nc' | 'co' | 'ci' | 'lo' | 'li' | 'coli' | 'cili' | 'colo' | 'cilo'`
— constant/linear-trend × outside/inside the cointegration relation, in
combination. `urca` offers `ecdet ∈ {"none", "const", "trend"}`.

`[V]` The mapping that matters, on `coint2_yes`, k=1, r=1:

| statsmodels | β | α | log-likelihood |
|---|---|---|---|
| `"n"` (none) | 1, −0.5761832355 | −0.14561855, 0.0279822964 | −662.848601 |
| `"co"` (constant **outside**) | 1, **−0.6673249283** | −0.1220914208, 0.6066110024 | −642.403440 |
| `"ci"` (constant **inside**) | 1, −0.6673216872 | −0.1222695053, 0.6063340546 | −642.425077 |

`[V]` `urca::cajorls` from `ca.jo(ecdet="none")` gives
β = `1, −0.6673249283` — **`"co"` exactly**. The name `ecdet="none"`
again means "no deterministic term *inside* the cointegration relation",
not "no deterministic term" — the same trap as 10.5 §4.

`[V]` Note how close `"co"` and `"ci"` are (β differs at the sixth
decimal, log-likelihood by 0.022) while being different models. A test
must distinguish them, so assert to `1e-8`, not `1e-4`.

**Decision:** reuse 10.5's five-symbol vocabulary
(`:none`, `:const`, `:const_restricted`, `:trend`,
`:trend_restricted`) with the same meanings, so a rank from
`johansen_test` and a fit from `fit_vecm` cannot silently disagree. Map
each to statsmodels and `urca` in the docstring. **A test must assert
that `johansen_test` and `fit_vecm` with the same `deterministic` give
the same β** (§7.3) — that is the invariant that makes the shared
vocabulary worth having.

---

## 4. Two estimators, genuinely different

`[V]` On `coint3_r1`, k=1, r=1:

| Estimator | β |
|---|---|
| `urca::cajorls` (Johansen ML) | 1, −0.496504585, −0.9777300348 |
| statsmodels `"co"` (ML) | 1, −0.496504585, −0.9777300348 |
| `tsDyn::VECM(estim="ML")` | 1, −0.496504585, −0.9777300348 |
| `tsDyn::VECM(estim="2OLS")` | 1, **−0.4930964735**, **−0.9654865132** |

And on `coint2_yes`: ML `1, −0.6673249283` vs 2OLS
`1, −0.5738885256`.

This is **not** a normalization difference — it is Engle-Granger's
two-step estimator versus Johansen's reduced-rank MLE. Both are
legitimate; 2OLS is consistent but not efficient.

`[V]` **`tsDyn` refuses 2OLS beyond rank 1** and silently switches:
*"Estimation of more than 1 coint relationship is not possible with
estim '2OLS'. Switched to Johansen 'ML'"* — a warning, not an error.

**Decisions:**

- `method = :ml` (default) and `:twostep`.
- `:twostep` with `rank > 1` **throws**, rather than silently switching
  estimator the way `tsDyn` does. Silently returning a different
  estimator's answer under the name the caller asked for is the worst
  of the options.
- The docstring states that `:ml` is what all three references compute
  by default and what the Johansen rank test assumes.

---

## 5. Reference values to assert

`[V]` `coint3_r1.csv`, k=1, r=1, `:const` (= `ecdet="none"` = `"co"`):

```
beta  = [1.0, -0.496504585, -0.9777300348]
alpha = [0.0369289001, -0.23427842, 0.9607618645]      (urca's ect1 row / sm's alpha)

urca cajorls coef, rows [ect1, constant, f.dl1, g.dl1, h.dl1], cols = equations:
[ 0.0369289001  -0.2342784174   0.9607618645
  0.0395842162   0.0545899489   0.0262256932
 -0.0497500753  -0.0257458406   0.7555020091
  0.0731949114   0.1205277722  -0.3301676918
  0.0503959055   0.1339971869  -0.7765036294]
```

`[V]` `coint2_yes.csv`, k=2, r=1, `:const`:
β = `[1.0, -0.6655228742]`, α = `[-0.1881544488, 0.4920002608]`.

`[V]` `coint3_r2.csv`, k=1, r=2, `:const` — identity-normalized β:

```
beta = [1.0          0.0
        0.0          1.0
        1.0059150205 1.9908711456]
```

The DGP is `d1 = tr + ε`, `d2 = 2tr + ε`, `d3 = −tr + ε`, so the true
relations are `d1 + d3` and `d2 + 2d3` — and the fitted third row
`[1.0059, 1.9909]` recovers exactly that. **Assert the recovery**, not
only the reference agreement.

`[V]` `coint3_r1.csv`, k=1, **r=2** (over-specified; true rank is 1) —
identity-normalized β third row `[-1.3698269062, -0.7897145025]`,
reproduced by all three references. The second cointegrating vector is
not identified by the data; the references agree because they compute
the same eigenvector, not because it means anything. **Assert the
numbers and say so in a comment** — this is the case where a user will
get a confident number from a misspecified rank.

---

## 6. The VAR representation — reuse, don't rebuild

`[V]` `vars::vec2var(j, r)` converts a Johansen fit into a levels VAR,
and `statsmodels`' `VECMResults.var_rep` does the same. Both are
recorded in the reference files.

Once `var_rep` exists, **10.1's forecast and 10.3's IRF/FEVD apply
unchanged**. Build `var_rep` and delegate; do not write a second
forecast recursion. The conversion is:

```
A_1 = I + alpha*beta' + Gamma_1
A_j = Gamma_j - Gamma_{j-1}        j = 2 … k
A_{k+1} = -Gamma_k
```

A test must assert that the VECM's own residuals equal the residuals of
its VAR representation (§7.6) — that is the check that the conversion
is right, and it needs no reference at all.

**Note the rank-deficiency trap:** the implied levels VAR has a unit
root by construction (that is the point of a VECM), so
`is_stable(var_rep)` is **false** and `lr_effects` must refuse — exactly
the honest-refusal case 10.3 §7.9 introduces. Assert it.

---

## 7. Test battery

Suggested file: `test/test_vecm.jl`. Target **≥ 200 assertions**.
References in `fixtures/ref_R_vecm.json` and `fixtures/ref_py_vecm.json`.

### 7.1 β and α against three references

```julia
@testset "fit_vecm: ML, vs urca/cajorls, statsmodels and tsDyn" begin
    cases = [("coint3_r1.csv", 1, 1, [1.0, -0.496504585, -0.9777300348],
                                      [0.0369289001, -0.23427842, 0.9607618645]),
             ("coint2_yes.csv", 1, 1, [1.0, -0.6673249283],
                                      [-0.1220914208, 0.6066110024]),
             ("coint2_yes.csv", 2, 1, [1.0, -0.6655228742],
                                      [-0.1881544488, 0.4920002608])]
    for (fx, k, r, beta, alpha) in cases
        m = fit_vecm(read_fixture(fx); lags=k, rank=r, deterministic=:const, method=:ml)
        b = normalize_beta(m, r; scheme=:first_unit)
        @test vec(b) ≈ beta  atol=1e-9
        @test vec(m.alpha) ≈ alpha atol=1e-9
        @test m.rank == r && m.lags == k
        @test m.method == :ml
    end
end

@testset "fit_vecm: rank 2, and parameter recovery" begin
    # coint3_r2's DGP has relations d1 + d3 and d2 + 2 d3
    m = fit_vecm(read_fixture("coint3_r2.csv"); lags=1, rank=2, deterministic=:const)
    B = normalize_beta(m, 2; scheme=:identity)
    @test B[1:2, 1:2] ≈ I(2) atol=1e-10
    @test B[3, :] ≈ [1.0059150205, 1.9908711456] atol=1e-8   # the references
    @test B[3, :] ≈ [1.0, 2.0] atol=0.02                     # and the TRUTH

    # over-specified rank: the references agree, but the second vector is
    # not identified by the data. Assert the numbers and remember why.
    mo = fit_vecm(read_fixture("coint3_r1.csv"); lags=1, rank=2, deterministic=:const)
    Bo = normalize_beta(mo, 2; scheme=:identity)
    @test Bo[3, :] ≈ [-1.3698269062, -0.7897145025] atol=1e-8
end
```

### 7.2 Γ and the full coefficient block

```julia
@testset "fit_vecm: short-run coefficients" begin
    m = fit_vecm(read_fixture("coint3_r1.csv"); lags=1, rank=1, deterministic=:const)
    # urca's cajorls reports [ect1, constant, f.dl1, g.dl1, h.dl1] x equations
    @test m.det_coef ≈ [0.0395842162, 0.0545899489, 0.0262256932] atol=1e-9
    @test m.gamma[1] ≈ [-0.0497500753 -0.0257458406  0.7555020091
                         0.0731949114  0.1205277722 -0.3301676918
                         0.0503959055  0.1339971869 -0.7765036294]' atol=1e-9
    @test length(m.gamma) == 1
    @test size(m.sigma_df) == (3, 3)
    @test isposdef(Hermitian(m.sigma_df))
end
```

### 7.3 The invariant that justifies the shared vocabulary

```julia
@testset "johansen_test and fit_vecm agree on beta for the same deterministic" begin
    for (fx, k, r) in [("coint2_yes.csv",1,1), ("coint3_r1.csv",1,1),
                       ("coint3_r2.csv",1,2), ("coint2_yes.csv",2,1)]
        Y = read_fixture(fx)
        for det in (:none, :const, :const_restricted)
            j = johansen_test(Y; lags=k, deterministic=det)
            m = fit_vecm(Y; lags=k, rank=r, deterministic=det, method=:ml)
            @test normalize_beta(j, r; scheme=:first_unit) ≈
                  normalize_beta(m, r; scheme=:first_unit) atol=1e-10
        end
    end
end

@testset "deterministic: :const is 'co', and :const_restricted is 'ci'" begin
    Y = read_fixture("coint2_yes.csv")
    n  = fit_vecm(Y; lags=1, rank=1, deterministic=:none)
    co = fit_vecm(Y; lags=1, rank=1, deterministic=:const)
    ci = fit_vecm(Y; lags=1, rank=1, deterministic=:const_restricted)
    @test vec(normalize_beta(n,  1))[2] ≈ -0.5761832355 atol=1e-8
    @test vec(normalize_beta(co, 1))[2] ≈ -0.6673249283 atol=1e-9
    @test vec(normalize_beta(ci, 1))[2] ≈ -0.6673216872 atol=1e-9
    @test loglikelihood(n)  ≈ -662.848601 atol=1e-5
    @test loglikelihood(co) ≈ -642.403440 atol=1e-5
    @test loglikelihood(ci) ≈ -642.425077 atol=1e-5
    # co and ci differ at the 6th decimal -- the test must be tight enough to see it
    @test vec(normalize_beta(co,1))[2] != vec(normalize_beta(ci,1))[2]
    @test abs(loglikelihood(co) - loglikelihood(ci)) > 0.01
end
```

### 7.4 Two-step OLS is a different estimator

```julia
@testset "method=:twostep differs from :ml, and refuses rank > 1" begin
    m1 = fit_vecm(read_fixture("coint3_r1.csv"); lags=1, rank=1,
                  deterministic=:const, method=:ml)
    m2 = fit_vecm(read_fixture("coint3_r1.csv"); lags=1, rank=1,
                  deterministic=:const, method=:twostep)
    @test vec(normalize_beta(m2, 1)) ≈ [1.0, -0.4930964735, -0.9654865132] atol=1e-7
    @test !isapprox(normalize_beta(m1,1), normalize_beta(m2,1); atol=1e-3)

    my = fit_vecm(read_fixture("coint2_yes.csv"); lags=1, rank=1,
                  deterministic=:const, method=:twostep)
    @test vec(normalize_beta(my, 1)) ≈ [1.0, -0.5738885256] atol=1e-7

    # tsDyn silently switches estimator here; we throw instead
    @test_throws ArgumentError fit_vecm(read_fixture("coint3_r2.csv");
                                        lags=1, rank=2, method=:twostep)
    e = try fit_vecm(read_fixture("coint3_r2.csv"); lags=1, rank=2, method=:twostep)
        catch err; err end
    @test occursin("rank", lowercase(sprint(showerror, e)))
    @test occursin("ml", lowercase(sprint(showerror, e)))
end
```

### 7.5 Standard errors

```julia
@testset "standard errors for alpha, beta and gamma" begin
    m = fit_vecm(read_fixture("coint3_r1.csv"); lags=1, rank=1, deterministic=:const)
    # statsmodels reports all three; see ref_py_vecm.json r1_k1_r1_* entries
    ref = load_ref_vecm("coint3_r1.csv", 1, 1, "co")
    @test m.stderr_alpha ≈ ref.stderr_alpha rtol=1e-6
    @test m.stderr_beta  ≈ ref.stderr_beta  rtol=1e-6
    @test m.stderr_gamma ≈ ref.stderr_gamma rtol=1e-6
    # beta's first element is normalized to 1, so its standard error is 0
    @test m.stderr_beta[1, 1] == 0.0
    @test all(>=(0.0), m.stderr_alpha)
end
```

### 7.6 The VAR representation — a check needing no reference

```julia
@testset "var_rep: the VECM is a levels VAR, and its residuals must match" begin
    m = fit_vecm(read_fixture("coint3_r1.csv"); lags=1, rank=1, deterministic=:const)
    A = var_rep(m)                           # Vector of (lags+1) K x K matrices
    @test length(A) == m.lags + 1
    # A_1 = I + alpha*beta' + Gamma_1 ;  A_2 = -Gamma_1
    @test A[1] ≈ I(3) + m.alpha * m.beta' + m.gamma[1] atol=1e-10
    @test A[2] ≈ -m.gamma[1] atol=1e-12

    # the implied VAR reproduces the VECM's own residuals
    @test residuals(m) ≈ residuals_from_var_rep(m) atol=1e-10

    # a VECM has a unit root BY CONSTRUCTION, so the levels VAR is not stable
    # and the long-run effect does not exist
    @test maximum(roots_from_var_rep(m)) ≈ 1.0 atol=1e-6
    @test !is_stable_var_rep(m)
    @test_throws ArgumentError lr_effects_from_var_rep(m)

    # and the reference agrees on var_rep
    ref = load_ref_vecm("coint3_r1.csv", 1, 1, "co")
    for j in 1:length(A); @test A[j] ≈ ref.var_rep[j] atol=1e-9; end
end

@testset "forecast delegates to 10.1's recursion" begin
    m = fit_vecm(read_fixture("coint3_r1.csv"); lags=1, rank=1, deterministic=:const)
    f = forecast(m, 4)
    ref = load_ref_vecm("coint3_r1.csv", 1, 1, "co")
    @test f.point ≈ ref.predict atol=1e-8
    @test predict(m, 4).point == f.point         # exact aliases, charter 2.5
    @test typeof(predict(m, 4)) === typeof(f)
end
```

### 7.7 Refusals — statsmodels accepts four nonsense configurations

`[V]` Verbatim:

| Case | `urca::cajorls` | statsmodels `VECM` |
|---|---|---|
| `rank = 0` | *"Please, provide a cointegration rank 'r' in the interval of 1 to 1, in accordance with 'z@V'."* | **no error** — returns an **empty** β |
| `rank = K` | (refused by the same range check) | **no error** — β = identity |
| `rank = K + 1` | refused | **no error** — same as `rank = K` |
| `k_ar_diff = 0` | — | **no error** — β = `[1, −0.577717327]` |
| bad `deterministic` | — | clean `ValueError` listing the eleven options |

```julia
@testset "refusals" begin
    Y = read_fixture("coint2_yes.csv")
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=0)      # sm returns empty beta
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=2)      # rank = K: no cointegration
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=3)      # rank > K
    @test_throws ArgumentError fit_vecm(Y; lags=0, rank=1)      # sm accepts this
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=1, deterministic=:bogus)
    @test_throws ArgumentError fit_vecm(Y; lags=1, rank=1, method=:bogus)
    @test_throws ArgumentError fit_vecm(Y[:, 1:1]; lags=1, rank=1)
    Yn = copy(Y); Yn[11, 1] = NaN
    @test_throws ArgumentError fit_vecm(Yn; lags=1, rank=1)
    e = try fit_vecm(Y; lags=1, rank=2) catch err; err end
    @test occursin("rank", lowercase(sprint(showerror, e)))
    @test occursin("1", sprint(showerror, e))      # the valid range must be named
end
```

### 7.8 StatsAPI contract, show, type stability

```julia
@testset "StatsAPI contract" begin
    m = fit_vecm(read_fixture("coint3_r1.csv"); lags=1, rank=1, deterministic=:const)
    @test m isa TimeSeriesModel
    @test size(coef(m), 1) == 3
    @test size(residuals(m)) == (m.nobs, 3)
    @test nobs(m) == m.nobs
    @test loglikelihood(m) isa Float64
    @test aic(m) isa Float64 && bic(m) isa Float64
    @test isconcretetype(VECMModel)
    @test (@inferred fit_vecm(read_fixture("coint2_yes.csv"); lags=1, rank=1)) isa VECMModel
    s = sprint(show, m)
    @test occursin("VECM", s)
    @test occursin("rank", lowercase(s))
    @test occursin("1", s)
end

@testset "performance: one factorization for the reduced-rank regression" begin
    E = read_fixture("econ5.csv")
    fit_vecm(E; lags=1, rank=1, deterministic=:const)       # warm up
    @test (@allocated fit_vecm(E; lags=1, rank=1, deterministic=:const)) < 600_000
    t1 = @elapsed for _ in 1:30; fit_vecm(E; lags=1, rank=1, deterministic=:const); end
    t4 = @elapsed for _ in 1:30; fit_vecm(E; lags=4, rank=1, deterministic=:const); end
    @test t4 < 8 * t1
end
```

---

## 8. Proposed Julia API

```julia
struct VECMModel <: TimeSeriesModel
    Y::Matrix{Float64}; names::Vector{Symbol}
    lags::Int                          # DIFFERENCED lags, as in 10.5
    rank::Int
    deterministic::Symbol              # 10.5's five-symbol vocabulary
    method::Symbol                     # :ml | :twostep
    alpha::Matrix{Float64}             # K x r
    beta::Matrix{Float64}              # K x r (raw; use normalize_beta)
    gamma::Vector{Matrix{Float64}}     # `lags` matrices, each K x K
    det_coef::Vector{Float64}
    sigma_df::Matrix{Float64}; sigma_mle::Matrix{Float64}
    stderr_alpha::Matrix{Float64}; stderr_beta::Matrix{Float64}
    stderr_gamma::Matrix{Float64}
    resid::Matrix{Float64}; loglik::Float64; nobs::Int
end

fit_vecm(Y; lags::Integer, rank::Integer,
         deterministic::Symbol = :const, method::Symbol = :ml) -> VECMModel

var_rep(m::VECMModel) -> Vector{Matrix{Float64}}
normalize_beta(m::VECMModel, r::Integer; scheme::Symbol = :first_unit)
forecast(m::VECMModel, h::Integer; levels=[80.0,95.0]) -> VARForecast
predict(m::VECMModel, h::Integer; kwargs...)                      # EXACT alias
```

`normalize_beta` is the same function 10.5 exposes, with a second
method — one implementation, two model types.

---

## 9. Checklist

1. **Re-establish Julia** and re-verify every `[S]` claim before coding.
2. Implement in `src/vecm.jl`, after `src/cointegration.jl` (10.5) and
   whatever holds `VARModel`.
3. Get §7.1 and §7.3 green before anything else — β against three
   references, and the Johansen/VECM invariant.
4. `var_rep` next, with the reference-free residual check (§7.6).
5. Then `:twostep`, standard errors, forecast.
6. Run the battery. Target ≥ 200 assertions in `test/test_vecm.jl`.
7. Copy `scripts/ref_vecm.R` and `scripts/ref_py_vecm.py` to
   `test/verification/vecm/`.
8. **Correct `stage-10-dependency-list.md` §4.5 and the charter**: VECM
   has three agreeing references once the deterministic specification
   is matched; the "three normalizations" reading was a deterministic
   mismatch. Revise the difficulty from hard to moderate.
9. **Add to the charter §5.10**: statsmodels' `VECM` accepts
   `coint_rank = 0` (returning an empty β), `rank ≥ K`, and
   `k_ar_diff = 0`; `tsDyn` silently switches estimator when 2OLS is
   asked for with rank > 1.
10. Manual section: the deterministic table from §3, the ML-vs-2OLS
    distinction, and the point that a VECM's levels VAR is
    non-stationary by construction.

---

## 10. Bundle additions

```
fixtures/
  ref_R_vecm.json    # cajorls beta/coef/sigma and vec2var A/forecast for 5 configurations,
                     #   tsDyn 2OLS and ML for each, urca's error messages
  ref_py_vecm.json   # statsmodels VECM for 7 configurations (deterministic n/co/ci,
                     #   k = 1 and 2, rank 1 and 2) with alpha, beta, gamma, all three
                     #   standard-error blocks, llf, var_rep, predict, and error strings
scripts/
  ref_vecm.R         # regenerates ref_R_vecm.json
  ref_py_vecm.py     # regenerates ref_py_vecm.json
```

Fixtures shared with 10.4 and 10.5.
