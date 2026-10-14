# Handoff: Stage 10.5 — Cointegration: the Johansen procedure

**Status: implemented ✅ (2026-10-13).** §5's table decision was reversed on reading the paper — see the overall handoff §5.13 item 18.

For a fresh Claude Code session with no prior context. **Read
`stage-10-overall-handoff.md` first** — the charter carries the
conventions (§5, including the Johansen traps at §5.6 and the reference
defect list at §5.10), the architecture decisions (§6), the reference
environment (§3) and the performance and test policy (§7, §8).

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every reference number below is `[V]`. **Julia was not installed in the
sandbox that produced this document** — re-verify any package-internal
claim before building on it.

---

## 1. Where this fits

- **Depends on:** 10.0 (`_mols`, `tsmatrix`, `build_var_Z`),
  `LinearAlgebra.eigen(A, B)` — the **generalized symmetric**
  eigenproblem — and new Osterwald-Lenum critical-value tables (§5).
- **The roadmap's "10.5 depends on 10.1" is wrong.** Johansen is a
  reduced-rank regression on `ΔY` against lagged differences and
  levels. It never constructs a `VARModel`. Verified by building it:
  nothing in the procedure below touches 10.1.
- **Feeds into:** 10.6 (VECM), which takes the rank and the
  cointegrating vectors from here.
- **Difficulty:** moderate-to-hard, and `[V]` the reason is not the
  mathematics — statsmodels' `coint_johansen` is **130 lines** — but
  that the two references disagree about almost everything surrounding
  the statistics (§3, §4, §5).

---

## 2. Scope

### In

- The trace and maximum-eigenvalue statistics for `r = 0 … K−1`.
- Eigenvalues `λ`, cointegrating vectors `β`, adjustment coefficients
  `α`.
- Five deterministic specifications spanning both references'
  vocabularies (§4).
- Osterwald-Lenum critical values for `K ≤ 12`.
- Rank selection from either statistic at a chosen level.
- Honest refusals: no p-value, and nothing outside the table (§6).

### Out

- VECM estimation and its inference — 10.6.
- Restricted-β and restricted-α likelihood-ratio tests
  (`urca::blrtest`, `ablrtest`, `bh5lrtest`, `bh6lrtest`). A natural
  follow-on row; note as unimplemented.
- `urca::cajolst` (Johansen with a level shift at an unknown date).
- MacKinnon-Haug-Michelis p-values — see §6.2 for why this is tempting
  and why it is a separate, paper-sourced task.

---

## 3. The statistics agree exactly, once the two mappings are right

`[V]` With the mapping established below, `urca::ca.jo` and
statsmodels' `coint_johansen` agree to `≤ 2.2e-11` on trace,
max-eigenvalue and every eigenvalue:

| Fixture | urca | statsmodels | trace | max‖diff‖ |
|---|---|---|---|---|
| `coint2_yes` | K=2, `ecdet="none"` | k=1, d=0 | 62.018519, 3.591564 | 2.14e-11 |
| `coint2_yes` | K=3, `ecdet="none"` | k=2, d=0 | 49.639450, 3.607281 | 7.96e-12 |
| `coint2_no` | K=2, `ecdet="none"` | k=1, d=0 | 5.146879, 2.104971 | 8.88e-16 |
| `coint3_r1` | K=2, `ecdet="none"` | k=1, d=0 | 90.840085, 4.481459, 0.459450 | 1.19e-12 |
| `coint3_r2` | K=2, `ecdet="none"` | k=1, d=0 | 195.893040, 93.160336, 0.513648 | 1.25e-10 |

Max-eigenvalue on the same cases: `58.426955, 3.591564` /
`46.032168, 3.607281` / `3.041908, 2.104971` /
`86.358626, 4.022008, 0.459450` / `102.732704, 92.646688, 0.513648`.

Eigenvalues, `coint3_r1`: `0.29405664, 0.01608698, 0.00185091`.

### 3.1 Mapping one: lag order

`[V]` **`urca`'s `K` is statsmodels' `k_ar_diff` + 1.** `K = 2` ↔
`k_ar_diff = 1`; `K = 3` ↔ `k_ar_diff = 2`. `urca` counts the levels-VAR
order, statsmodels counts the differenced lags. `[V]` `urca` refuses
`K = 1` cleanly (*"K must be at least K=2."*) while statsmodels
**silently accepts `k_ar_diff = 0`** and returns numbers.

### 3.2 Mapping two: output order is reversed

`[V]` `urca` returns the statistics ordered `r ≤ K−1` first; statsmodels
returns `r = 0` first. Same numbers, opposite order. A test written
against one and checked against the other passes on the wrong rows.

**Decision: `r = 0` first** (statsmodels' order, and the order the
hypotheses are stated in), with the rank index stored explicitly in the
result so position is never load-bearing.

---

## 4. The deterministic vocabularies are incompatible, and this is worse than the VAR case

`[V]` Only **one** specification is shared. Verified on `coint2_yes`:

| `urca` `ecdet` | statsmodels `det_order` | Same? | evidence |
|---|---|---|---|
| `"none"` (unrestricted constant) | `0` | **yes** | trace 62.018519 both, 2.1e-11 |
| `"const"` (constant **restricted** to the cointegration space) | — | no equivalent | urca 62.064072 vs sm `d=1` 60.812438, diff 1.29 |
| `"trend"` (trend restricted) | — | no equivalent | — |
| — | `-1` (no deterministic at all) | no urca equivalent | sm 18.443286 vs urca `"none"` 62.018519 |
| — | `1` (constant **and** unrestricted trend) | no urca equivalent | — |

The names are actively misleading: `ecdet="none"` still includes an
unrestricted constant, which is why it maps to `det_order=0` and not
`-1`. `[V]` On `econ5` the `ecdet="const"`-vs-`det_order=0` gap reaches
**76.77** on the trace statistic — not a rounding matter, a different
model.

`[V]` **A restricted deterministic term adds an eigenvalue.** With
`ecdet="const"`, `urca` returns `K+1` eigenvalues, the last numerically
zero:

```
coint2_yes  ecdet="const"  P=2  lambda = [0.209905, 0.014550, 0.0]
uschange    ecdet="const"  P=5  lambda = [0.428748, 0.384232, 0.271750,
                                          0.208981, 0.143728, -0.0]
econ5       ecdet="const"  P=5  lambda = [0.461018, 0.314280, 0.195556,
                                          0.063977, 0.043703, -0.0]
coint2_yes  ecdet="none"   P=2  lambda = [0.209897, 0.014378]     (K only)
```

The augmented eigenvalue must be dropped from the test sequence, and the
`-0.0` shows the reference is not even consistent about its sign.

**Decision: `deterministic::Symbol` with five values, every one mapped
explicitly in the docstring:**

| TSAnalytics | meaning | `urca` | statsmodels |
|---|---|---|---|
| `:none` | no deterministic term | — | `det_order = -1` |
| `:const` | unrestricted constant | `ecdet="none"` | `det_order = 0` |
| `:const_restricted` | constant in the cointegration space | `ecdet="const"` | — |
| `:trend` | unrestricted constant + trend | — | `det_order = 1` |
| `:trend_restricted` | trend in the cointegration space | `ecdet="trend"` | — |

Do not reuse the charter §5.2 VAR vocabulary verbatim here: the
restricted/unrestricted distinction has no VAR analogue, and collapsing
it would reproduce exactly the confusion `ecdet="none"` causes. Say in
the docstring that `:const` is the common default and corresponds to
`ecdet="none"` **despite the name**.

`[S]` `urca` also has a `spec` argument (`"longrun"` vs
`"transitory"`), which changes how the lagged levels enter. The
reference set includes a `"transitory"` case
(`yes_K2_none_TR` in `ref_R_johansen.json`); decide whether to expose it
and, if not, say so.

---

## 5. Critical values: the tables disagree, and one of them is better

`[V]` Both references ship a table, and they are not the same table.
5% trace values:

| K (or n−r) | `urca` (Osterwald-Lenum 1992) | statsmodels (`coint_tables.c_sjt`) |
|---|---|---|
| 1 | 8.18 (`ecdet="none"`) | 3.8415 (`det_order=0`) **and** 3.8415 (`det_order=1`) |
| 2 | 17.95 | 15.4943 |
| 3 | 31.52 | 29.7961 |
| 5 (`uschange`) | 9.24, 19.96, 34.91, 53.12, 76.07 | 3.8415, 15.4943, 29.7961, 47.8545, 69.8189 |

`[V]` The deciding observation: statsmodels' `n−r = 1` entry is
**3.8415 for both `det_order = 0` and `det_order = 1`** — identical
across two different deterministic specifications — and 3.8415 is
exactly the χ²(1) 95% point. Osterwald-Lenum's smallest-block value
*does* vary with the specification (8.18 unrestricted constant, 9.24
restricted constant, 12.25 restricted trend), which is what the
asymptotic theory requires, since the limiting distribution depends on
the deterministic terms.

**Decision: ship Osterwald-Lenum (1992)**, matching `urca` and the bulk
of published applied work. Record in the docstring that statsmodels'
values differ and why this package does not follow them.

`[V]` Table coverage, established by making each reference fail:

- statsmodels `c_sja`/`c_sjt`: `n = 1 … 12`, `det_order ∈ {−1, 0, 1}`.
  `n = 13` and `det_order = 2` both return **`[nan, nan, nan]`
  silently** — `det_order = 2` with no warning at all, 13 variables with
  only a `HypothesisTestWarning`.
- `urca`: `K ≤ 12`. At 13 columns it warns *"Too many variables,
  critical values cannot be computed."* and then **errors** with
  *"length of 'dimnames' [2] not equal to array extent"*.

### 5.1 Licensing

Charter §4. statsmodels is BSD-3 but ships the table this item
deliberately does **not** use. `urca` is GPL and must not be copied.
The Osterwald-Lenum values must come from the published source —
Osterwald-Lenum (1992), *Oxford Bulletin of Economics and Statistics*
54(3), Tables 0–2* — cited in the source comment, with `urca` used as
the check. The spot values in §5's table and in `ref_R_johansen.json`
are what to verify against.

---

## 6. Two honest refusals

### 6.1 No p-value

`[V]` Neither reference reports one. `urca` gives critical values at
10%, 5% and 1%; statsmodels gives `cvt`/`cvm` and nothing else.

This collides with `[S]` `src/abstract.jl:64`'s
`pvalue(t::HypothesisTest) = t.pvalue`. Per charter §6.4: add
`critical_values(t)` to `abstract.jl`, have `pvalue` throw an
`ArgumentError` naming it, and **do not store `NaN`** — storing `NaN`
is precisely the failure mode §5 criticises statsmodels for.

10.4a and 10.4b need the same extension; whichever item lands first
makes the change.

### 6.2 Outside the table

Refuse `K > 12` by name. Refuse an unsupported deterministic
specification. **Never return `NaN` critical values**, which is what
statsmodels does for `det_order = 2` and for 13 variables.

*A note on the temptation:* MacKinnon-Haug-Michelis (1999) response
surfaces would give genuine p-values for the Johansen statistics —
something **neither reference has**, and therefore real
differentiation. But those coefficients are in neither installed
package, so they are a separate paper-sourced task. Do not improvise
them, and do not interpolate the three Osterwald-Lenum points into a
pseudo-p-value.

---

## 7. The procedure, and why the eigenproblem form matters

### 7.1 Construction

With `ΔY` the first differences, `k = K_levels − 1` differenced lags:

```
R_0 = residuals of regressing ΔY_t  on [ΔY_{t-1} … ΔY_{t-k}, deterministic]
R_1 = residuals of regressing Y_{t-1} on the same regressors
S_ij = R_i' R_j / T          i, j in {0, 1}
```

then solve the **generalized symmetric** eigenproblem

```
| λ S_11 − S_10 S_00^{-1} S_01 | = 0
```

for `λ_1 ≥ … ≥ λ_K`, and

```
trace(r)  = −T * sum_{i=r+1}^{K} log(1 − λ_i)
maxeig(r) = −T * log(1 − λ_{r+1})
```

`[V]` Both the `R_0`/`R_1` regressions are multi-response with a shared
regressor matrix — **this is `_mols`' job** (10.0). Do not write a
second solve.

### 7.2 Use `eigen(A, B)`, not an explicit inverse

`[V]` statsmodels emits `ComplexWarning: Casting complex values to real
discards the imaginary part` from `vecm.py:717–718` on **every** call —
its eigenvalues come back complex from a non-symmetric formulation and
are truncated.

`LinearAlgebra.eigen(Hermitian(S10 * (S00 \ S01)), Hermitian(S11))`
solves the symmetric-definite generalized problem and returns **real
eigenvalues by construction**. Already available; no new dependency.
Form `S00 \ S01` with a Cholesky solve rather than `inv(S00)`.

A test should assert the eigenvalues are real and in `[0, 1)` — `[V]`
every reference value above is, and `urca`'s `-0.0` for the augmented
eigenvalue under a restricted deterministic term shows the reference
does not guarantee it.

### 7.3 β normalization differs between references — and within one

`[V]` This is the trap 10.6 inherits.

`urca`'s `V` (the cointegrating vectors) is normalized so the **first
element of each column is 1**:

```
coint3_r1, K=2, ecdet="none":
V = [ 1.0          1.0          1.0
     -0.49650458  -1.4221091    0.2145488
     -0.97773003  -0.24676673   0.21404288]
W = [ 0.0369289   -0.01001952  -0.00355161
     -0.23427842   0.01197295  -0.00419933
      0.96076186  -0.01445675  -0.00167233]
```

statsmodels' `evec` is **not** normalized that way:

```
evec = [ 2.04341035  -0.13381102  -0.1327328
        -1.01456261   0.19029387  -0.02847766
        -1.99790368   0.03302011  -0.02841051]
```

`[V]` And `urca::cajorls` changes normalization again depending on the
rank — at `r = 1` it uses the first-element-is-1 form, at `r = 2` an
identity-block form:

```
cajorls(r=1)  beta = [ 1.0, -0.49650458, -0.97773003]'
cajorls(r=2)  beta = [ 1.0   0.0
                       0.0   1.0
                      -1.36982691  -0.7897145 ]
```

**Decision:** store β in the raw eigenvector form **and** provide
`normalize_beta(t, r; scheme=:first_unit | :identity)`. Document that
the *statistics are invariant* to normalization while β is not, and
assert that invariance in a test — it is the check that separates a
normalization difference from a bug.

---

## 8. Proposed Julia API

```julia
struct JohansenTest <: HypothesisTest
    trace::Vector{Float64}            # r = 0 … K-1, r = 0 FIRST
    maxeig::Vector{Float64}
    ranks::Vector{Int}                # 0:(K-1), so position is never load-bearing
    trace_crit::Matrix{Float64}       # (K) x 3, levels 10/5/1 %
    maxeig_crit::Matrix{Float64}
    eigenvalues::Vector{Float64}      # real by construction
    beta::Matrix{Float64}             # raw eigenvectors, K x K
    alpha::Matrix{Float64}            # adjustment coefficients
    K::Int
    lags::Int                         # DIFFERENCED lags -- statsmodels' k_ar_diff
    deterministic::Symbol
    nobs::Int
    pvalue_available::Bool            # always false; see section 6.1
end

johansen_test(Y; lags::Integer, deterministic::Symbol = :const) -> JohansenTest

coint_rank(t::JohansenTest; statistic::Symbol = :trace, level::Real = 5.0) -> Int
normalize_beta(t::JohansenTest, r::Integer; scheme::Symbol = :first_unit) -> Matrix{Float64}
critical_values(t::JohansenTest) -> NamedTuple
```

`lags` counts **differenced** lags, like statsmodels, and the docstring
states that `urca`'s `K` is `lags + 1`. Make `lags` required — `[V]`
statsmodels' acceptance of `k_ar_diff = 0` shows what a permissive
default costs.

---

## 9. Test battery

Suggested file: `test/test_johansen.jl`. Target **≥ 200 assertions**.
References in `fixtures/ref_R_johansen.json` and
`fixtures/ref_py_johansen.json`.

### 9.1 Statistics, both forms, five fixtures

```julia
@testset "johansen_test: trace and maxeig vs both references" begin
    cases = [
     ("coint2_yes.csv", 1, [62.018519, 3.591564],  [58.426955, 3.591564],
                           [0.20989747, 0.01437775]),
     ("coint2_yes.csv", 2, [49.639450, 3.607281],  [46.032168, 3.607281],
                           [0.17002944, 0.01449825]),
     ("coint2_no.csv",  1, [5.146879, 2.104971],   [3.041908, 2.104971],
                           [0.01219084, 0.00845187]),
     ("coint3_r1.csv",  1, [90.840085, 4.481459, 0.459450],
                           [86.358626, 4.022008, 0.459450],
                           [0.29405664, 0.01608698, 0.00185091]),
     ("coint3_r2.csv",  1, [195.893040, 93.160336, 0.513648],
                           [102.732704, 92.646688, 0.513648],
                           [0.33916083, 0.31173088, 0.00206902])]
    for (fx, k, tr, me, lam) in cases
        t = johansen_test(read_fixture(fx); lags=k, deterministic=:const)
        @test t.trace  ≈ tr  atol=1e-6
        @test t.maxeig ≈ me  atol=1e-6
        @test t.eigenvalues ≈ lam atol=1e-8
        @test t.ranks == 0:(t.K-1)
        @test t.lags == k
        # r = 0 FIRST -- urca orders the other way (section 3.2)
        @test t.trace[1] == maximum(t.trace)
        # the trace statistic is the running tail sum of -T log(1-lambda)
        T = t.nobs
        for (i, r) in enumerate(t.ranks)
            @test t.trace[i] ≈ -T * sum(log1p.(-lam[(r+1):end])) atol=1e-5
            @test t.maxeig[i] ≈ -T * log1p(-lam[r+1]) atol=1e-5
        end
    end
end

@testset "eigenvalues are real and in [0,1) by construction" begin
    for (fx, k) in [("coint2_yes.csv",1), ("coint3_r1.csv",1), ("coint3_r2.csv",1),
                    ("uschange.csv",1), ("econ5.csv",1), ("coint2_no.csv",1)]
        t = johansen_test(read_fixture(fx); lags=k, deterministic=:const)
        @test eltype(t.eigenvalues) === Float64
        @test all(0.0 .<= t.eigenvalues .< 1.0)
        @test issorted(t.eigenvalues; rev=true)
        # statsmodels gets complex eigenvalues and truncates; we must not
        @test length(t.eigenvalues) == t.K
    end
end
```

### 9.2 The two mappings

```julia
@testset "lags counts DIFFERENCED lags (urca's K = lags + 1)" begin
    Y = read_fixture("coint2_yes.csv")
    t1 = johansen_test(Y; lags=1, deterministic=:const)   # urca K = 2
    t2 = johansen_test(Y; lags=2, deterministic=:const)   # urca K = 3
    @test t1.trace ≈ [62.018519, 3.591564] atol=1e-6
    @test t2.trace ≈ [49.639450, 3.607281] atol=1e-6
    @test t1.trace != t2.trace
    # statsmodels silently accepts k_ar_diff = 0; we refuse
    @test_throws ArgumentError johansen_test(Y; lags=0, deterministic=:const)
end

@testset "deterministic: :const is urca's ecdet='none', despite the name" begin
    Y = read_fixture("coint2_yes.csv")
    c  = johansen_test(Y; lags=1, deterministic=:const)
    cr = johansen_test(Y; lags=1, deterministic=:const_restricted)
    n  = johansen_test(Y; lags=1, deterministic=:none)
    @test c.trace  ≈ [62.018519, 3.591564] atol=1e-6      # urca ecdet="none"
    @test cr.trace ≈ [62.064072, 3.634837] atol=1e-6      # urca ecdet="const"
    @test n.trace  ≈ [18.443286, 0.861099] atol=1e-6      # sm det_order=-1
    # all three genuinely differ
    @test !isapprox(c.trace, cr.trace; atol=1e-3)
    @test !isapprox(c.trace, n.trace;  atol=1.0)

    # a RESTRICTED deterministic term augments the eigenvalue count,
    # and the extra one must be dropped from the test sequence
    @test length(cr.eigenvalues) == cr.K                   # not K+1 in OUR output
    @test length(cr.trace) == cr.K

    # econ5 shows how large the gap gets: 76.77 on the trace statistic
    E = read_fixture("econ5.csv")
    ec  = johansen_test(E; lags=1, deterministic=:const)
    ecr = johansen_test(E; lags=1, deterministic=:const_restricted)
    @test ec.trace[1]  ≈ 133.709914 atol=1e-4              # sm det_order=0
    @test ecr.trace[1] ≈ 210.478611 atol=1e-4              # urca ecdet="const"
    @test abs(ecr.trace[1] - ec.trace[1]) > 70
end
```

### 9.3 Critical values: Osterwald-Lenum, not statsmodels'

```julia
@testset "critical values are Osterwald-Lenum" begin
    t = johansen_test(read_fixture("coint2_yes.csv"); lags=1, deterministic=:const)
    # 5% column, r = 0 then r = 1
    @test t.trace_crit[:, 2] ≈ [17.95, 8.18] atol=1e-4
    # NOT statsmodels' values
    @test !isapprox(t.trace_crit[:, 2], [15.4943, 3.8415]; atol=1e-2)

    t3 = johansen_test(read_fixture("coint3_r1.csv"); lags=1, deterministic=:const)
    @test t3.trace_crit[:, 2] ≈ [31.52, 17.95, 8.18] atol=1e-4

    tU = johansen_test(read_fixture("uschange.csv"); lags=1, deterministic=:const_restricted)
    @test tU.trace_crit[:, 2] ≈ [76.07, 53.12, 34.91, 19.96, 9.24] atol=1e-4

    # the smallest-block value MUST vary with the deterministic specification.
    # statsmodels gives 3.8415 (the chi2(1) quantile) for both det_order 0 and 1,
    # which is the reason this package does not use its table.
    a = johansen_test(read_fixture("coint2_yes.csv"); lags=1, deterministic=:const)
    b = johansen_test(read_fixture("coint2_yes.csv"); lags=1, deterministic=:const_restricted)
    c = johansen_test(read_fixture("coint2_yes.csv"); lags=1, deterministic=:trend_restricted)
    @test a.trace_crit[end, 2] ≈ 8.18  atol=1e-4
    @test b.trace_crit[end, 2] ≈ 9.24  atol=1e-4
    @test c.trace_crit[end, 2] ≈ 12.25 atol=1e-4
    @test a.trace_crit[end, 2] != b.trace_crit[end, 2] != c.trace_crit[end, 2]

    @test size(t.trace_crit) == (2, 3)
    @test size(t.maxeig_crit) == (2, 3)
    # levels are 10, 5, 1 -- ASCENDING critical values, right-tail test
    @test all(t.trace_crit[:, 1] .<= t.trace_crit[:, 2] .<= t.trace_crit[:, 3])
end
```

### 9.4 Rank selection — the point of the exercise

```julia
@testset "coint_rank" begin
    # one relation by construction
    t1 = johansen_test(read_fixture("coint3_r1.csv"); lags=1, deterministic=:const)
    @test coint_rank(t1; statistic=:trace,  level=5.0) == 1
    @test coint_rank(t1; statistic=:maxeig, level=5.0) == 1
    # TWO relations by construction (one common trend among three series)
    t2 = johansen_test(read_fixture("coint3_r2.csv"); lags=1, deterministic=:const)
    @test coint_rank(t2; statistic=:trace,  level=5.0) == 2
    @test coint_rank(t2; statistic=:maxeig, level=5.0) == 2
    # independent random walks: rank 0
    t0 = johansen_test(read_fixture("coint2_no.csv"); lags=1, deterministic=:const)
    @test coint_rank(t0; statistic=:trace,  level=5.0) == 0
    @test coint_rank(t0; statistic=:maxeig, level=5.0) == 0
    # a bivariate cointegrated pair: rank 1
    ty = johansen_test(read_fixture("coint2_yes.csv"); lags=1, deterministic=:const)
    @test coint_rank(ty) == 1

    @test_throws ArgumentError coint_rank(ty; level=2.5)
    @test_throws ArgumentError coint_rank(ty; statistic=:bogus)
    # a stricter level can only lower or hold the rank
    @test coint_rank(t2; level=1.0) <= coint_rank(t2; level=5.0) <= coint_rank(t2; level=10.0)
end
```

### 9.5 β, α and normalization invariance

```julia
@testset "beta normalization, and the invariance that proves it is only a normalization" begin
    t = johansen_test(read_fixture("coint3_r1.csv"); lags=1, deterministic=:const)
    @test size(t.beta) == (3, 3)
    @test size(t.alpha, 1) == 3

    # first-element-unit scheme reproduces urca's V
    V = normalize_beta(t, 3; scheme=:first_unit)
    @test V[1, :] ≈ [1.0, 1.0, 1.0] atol=1e-12
    @test V[:, 1] ≈ [1.0, -0.49650458, -0.97773003] atol=1e-7
    @test V[:, 2] ≈ [1.0, -1.42210910, -0.24676673] atol=1e-7
    @test V[:, 3] ≈ [1.0,  0.21454880,  0.21404288] atol=1e-7

    # identity-block scheme reproduces cajorls at r = 2
    B2 = normalize_beta(t, 2; scheme=:identity)
    @test B2[1:2, 1:2] ≈ I(2) atol=1e-10
    @test B2[3, :] ≈ [-1.36982691, -0.7897145] atol=1e-7
    # and at r = 1 the two schemes coincide
    @test normalize_beta(t, 1; scheme=:identity) ≈
          normalize_beta(t, 1; scheme=:first_unit) atol=1e-10

    # THE invariance check: statistics do not depend on normalization
    scaled = johansen_test(read_fixture("coint3_r1.csv") .* [1.0 100.0 0.01];
                           lags=1, deterministic=:const)
    base   = johansen_test(read_fixture("coint3_r1.csv"); lags=1, deterministic=:const)
    @test scaled.trace  ≈ base.trace  atol=1e-6      # scale-invariant
    @test scaled.maxeig ≈ base.maxeig atol=1e-6
    @test scaled.eigenvalues ≈ base.eigenvalues atol=1e-8
    @test !isapprox(scaled.beta, base.beta; atol=1e-3)   # beta is NOT
end

@testset "alpha * beta' is the long-run impact matrix, and has rank r" begin
    t = johansen_test(read_fixture("coint3_r1.csv"); lags=1, deterministic=:const)
    r = coint_rank(t)
    B = normalize_beta(t, r; scheme=:first_unit)
    Pi = t.alpha[:, 1:r] * B'
    @test size(Pi) == (3, 3)
    @test rank(Pi; rtol=1e-8) == r
    # urca's W, the adjustment coefficients
    @test t.alpha[:, 1] ≈ [0.0369289, -0.23427842, 0.96076186] atol=1e-6
end
```

### 9.6 No p-value, and coverage refusals

```julia
@testset "no p-value is reported, and none is faked" begin
    t = johansen_test(read_fixture("coint2_yes.csv"); lags=1, deterministic=:const)
    @test !t.pvalue_available
    @test_throws ArgumentError pvalue(t)
    e = try pvalue(t) catch err; err end
    @test occursin("critical_values", sprint(showerror, e))
    cv = critical_values(t)
    @test cv.levels == (10.0, 5.0, 1.0)
    @test size(cv.trace) == (2, 3)
end

@testset "coverage: K <= 12, and never a NaN" begin
    # statsmodels returns all-NaN critical values for 13 variables (with only a
    # warning) and for det_order = 2 (with no warning at all).
    Y13 = cumsum(randn(MersenneTwister(11), 250, 13); dims=1)
    @test_throws ArgumentError johansen_test(Y13; lags=1, deterministic=:const)
    e = try johansen_test(Y13; lags=1) catch err; err end
    @test occursin("12", sprint(showerror, e))
    @test_nowarn johansen_test(Y13[:, 1:12]; lags=1, deterministic=:const)

    @test_throws ArgumentError johansen_test(read_fixture("coint2_yes.csv");
                                             lags=1, deterministic=:bogus)
    for fx in ("coint2_yes.csv","coint2_no.csv","coint3_r1.csv","coint3_r2.csv",
               "uschange.csv","econ5.csv")
        t = johansen_test(read_fixture(fx); lags=1, deterministic=:const)
        @test all(!isnan, t.trace) && all(!isnan, t.maxeig)
        @test all(!isnan, t.trace_crit) && all(!isnan, t.maxeig_crit)
        @test all(!isnan, t.eigenvalues)
    end
end

@testset "refusals" begin
    Y = read_fixture("coint2_yes.csv")
    @test_throws ArgumentError johansen_test(Y[:, 1:1]; lags=1)     # one series
    @test_throws ArgumentError johansen_test(Y; lags=0)
    @test_throws ArgumentError johansen_test(Y; lags=-1)
    @test_throws ArgumentError johansen_test(Y; lags=200)
    Yn = copy(Y); Yn[9, 2] = NaN
    @test_throws ArgumentError johansen_test(Yn; lags=1)
end
```

### 9.7 Contract, printing, performance

```julia
@testset "contract, show, type stability, performance" begin
    t = johansen_test(read_fixture("coint3_r1.csv"); lags=1, deterministic=:const)
    @test t isa HypothesisTest
    @test isconcretetype(JohansenTest)
    @test (@inferred johansen_test(read_fixture("coint2_yes.csv"); lags=1)) isa JohansenTest
    s = sprint(show, t)
    @test occursin("Johansen", s)
    @test occursin("trace", lowercase(s))
    @test occursin("Osterwald", s) || occursin("10%", s)   # the table must be identifiable

    # the two auxiliary regressions share a regressor matrix: ONE factorization
    E = read_fixture("econ5.csv")
    johansen_test(E; lags=1, deterministic=:const)         # warm up
    a = @allocated johansen_test(E; lags=1, deterministic=:const)
    @test a < 400_000
    # cost grows with lags, not quadratically
    t1 = @elapsed for _ in 1:50; johansen_test(E; lags=1, deterministic=:const); end
    t4 = @elapsed for _ in 1:50; johansen_test(E; lags=4, deterministic=:const); end
    @test t4 < 8 * t1
end
```

---

## 10. Checklist

1. **Re-establish Julia** and re-verify every `[S]` claim before coding.
2. Extend `HypothesisTest` per charter §6.4 (`critical_values`, a
   refusing `pvalue`) if 10.4a/b has not already.
3. Transcribe the **Osterwald-Lenum (1992)** tables from the paper,
   citing it, and verify against the `[V]` values in §5 and §9.3. Do not
   copy from `urca`.
4. Implement `johansen_test`, `coint_rank` and `normalize_beta` in
   `src/cointegration.jl` alongside 10.4a/b, reusing `_mols` and
   `build_var_Z`.
5. Use `eigen(A, B)` on the symmetric-definite problem (§7.2); assert
   real eigenvalues in `[0,1)`.
6. Run the battery. Target ≥ 200 assertions in `test/test_johansen.jl`.
7. Copy `scripts/ref_jo.R` and `scripts/ref_py_jo.py` to
   `test/verification/johansen/`.
8. `@code_warntype` on all three entry points; record the benchmark
   table for `K ∈ {2, 5, 12}`, `lags ∈ {1, 4}`.
9. **Fix the roadmap**: 10.5's `Depends on` column says 10.1. It does
   not. Change it to 10.0.
10. **Add to the charter §5.10**: statsmodels accepts `k_ar_diff = 0`,
    returns all-NaN critical values for `det_order = 2` with no warning
    and for 13 variables with only a warning; and its `n−r = 1` trace
    critical value does not vary with `det_order`.
11. Manual section. Lead with the deterministic-vocabulary table from
    §4 — `ecdet="none"` meaning "unrestricted constant" is the single
    most common source of wrong Johansen results in practice.

---

## 11. Bundle additions

```
fixtures/
  ref_R_johansen.json   # 11 ca.jo configurations (trace and eigen, three ecdet values,
                        #   both spec values, K = 2 and 3) with teststat, cval, lambda,
                        #   V, W, Vorg; cajorls at r = 1 and r = 2; urca's error messages
  ref_py_johansen.json  # 10 coint_johansen configurations with lr1/lr2/cvt/cvm/eig/evec,
                        #   the full c_sja/c_sjt tables for n = 1..13 x det_order -1..2
                        #   (NaN pattern included), select_coint_rank, error strings
scripts/
  ref_jo.R              # regenerates ref_R_johansen.json
  ref_py_jo.py          # regenerates ref_py_johansen.json
```

Fixtures are shared with 10.4a/b (`coint2_yes`, `coint2_no`,
`coint3_r1`, `coint3_r2`, `coint2_trend`) plus `uschange` and `econ5`.
