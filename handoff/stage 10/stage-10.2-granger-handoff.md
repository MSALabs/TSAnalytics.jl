# Handoff: Stage 10.2 — Granger causality

**Status: pending.**

For a fresh Claude Code session with no prior context. **Read
`stage-10-overall-handoff.md` first** — it carries the stage-wide
conventions (§5), architecture decisions (§6), the reference-environment
recipe (§3) and the performance and test policy (§7, §8). This document
cites them rather than repeating them.

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every reference number below is `[V]`. **Julia was not installed in the
sandbox that produced this document** — re-verify any package-internal
claim before building on it.

---

## 1. Where this fits

- **Depends on:** 10.1 (`VARModel`) and **one new numerical primitive**,
  `_f_ccdf` — see §4, which is the most important section in this
  document.
- **Feeds into:** nothing structurally. 10.2 is a leaf. It is scheduled
  fourth (after 10.1, 10.3, 10.8) because it is cheap once `VARModel`
  exists and because it exposes the `vcov_full` path that 10.3 also
  needs.
- **Reference size:** `[V]` `vars/R/causality.R` is **173 lines**;
  statsmodels' `VARResults.test_causality` is **129**. Most of both is
  restriction-matrix bookkeeping and result objects, not mathematics.

---

## 2. Scope

### In

- **System Granger test** on a fitted VAR: `H₀: the `cause` block does
  not Granger-cause the rest`. Wald on a restriction matrix, reported as
  an F statistic (and a χ² variant).
- **Pairwise Granger test** on two series: the single-equation F test,
  which is a genuinely *different* test, not a special case (§5).
- **Instantaneous causality**: Wald χ² on the off-diagonal block of Σ.
- Blocks of more than one causing variable, and more than one caused.
- `_f_ccdf` as a new primitive in `src/diagnostics.jl`.

### Out

- Toda-Yamamoto (causality in possibly-integrated systems). No reference
  in either installed package; note it as unimplemented.
- Frequency-domain causality (Breitung-Candelon).
- Causality on a VECM. Belongs with 10.6.
- Bootstrap p-values for the causality tests.

---

## 3. Two different tests, each with two agreeing references

This is the central structural fact of the item, and it corrects what
`stage-10-dependency-list.md` §4.2 said. That note claimed "10.2 must
pick a convention" because three references gave different p-values. The
truth, once the references are grouped properly, is cleaner:

### 3.1 The system test — `vars::causality` ≡ statsmodels `test_causality`

`[V]` Agreement on **all 15 cases tested**, across k = 2, 3, 5 and
p = 1, 2, 4: F statistic to `≤ 7.6e-12`, p-value to `≤ 7.5e-15`,
identical `df2`.

| Case | F | df | p |
|---|---|---|---|
| `var2_p1`, p=1, cause y2 | 0.023490551143 | (1, 392) | 0.8782671595 |
| `var2_p1`, p=1, cause y1 | 17.504579420963 | (1, 392) | 0.0000354015 |
| `var2_p1`, p=2, cause y2 | 0.730863702158 | (2, 386) | 0.4821580341 |
| `var2_p1`, p=2, cause y1 | 7.712480653203 | (2, 386) | 0.0005196436 |
| `var2_p1`, p=4, cause y2 | 0.829165909264 | (4, 374) | 0.5072142974 |
| `var3_p2`, p=2, cause x1 | 3.242043205406 | (4, 723) | 0.0119237127 |
| `var3_p2`, p=2, cause x2 | 2.084475336178 | (4, 723) | 0.0811331989 |
| `var3_p2`, p=2, cause x3 | 3.465874581908 | (4, 723) | 0.0081329307 |
| `var3_p2`, p=2, cause {x1,x2} | 3.743411847221 | (4, 723) | 0.0050422805 |
| `var3_p2`, p=2, cause {x2,x3} | 2.082120007689 | (4, 723) | 0.0814386404 |
| `var2_short`, p=2, cause y2 | 0.992318413455 | (2, 46) | 0.3785116479 |
| `uschange`, p=2, cause Income | 0.451416306491 | (8, 870) | 0.8899812692 |
| `uschange`, p=2, cause {Income,Production} | 0.948732438079 | (12, 870) | 0.4970171780 |
| `uschange`, p=2, cause Savings | 0.297268765099 | (8, 870) | 0.9669600021 |
| `econ5`, p=2, cause gnp | 3.066973695481 | (8, 740) | 0.0020925493 |

`df1 = p · |cause| · |caused|`; `df2 = K·T − K·(Kp + ndet)`. `[V]` For
`var2_p1` p=2: `2·198 − 2·5 = 386`. **This is the *system* residual
degrees of freedom**, not a single equation's.

### 3.2 The pairwise test — `lmtest::grangertest` ≡ `grangercausalitytests`

`[V]` A single-equation F test comparing restricted and unrestricted
OLS on **one** equation, so `df2 = T − 2p − 1`.

| Case | F | res_df | p |
|---|---|---|---|
| `var2_p1`, p=1, y2→y1 | 0.023490551143 | 196 | 0.8783459725 |
| `var2_p1`, p=2, y2→y1 | 0.730863702158 | 193 | 0.4828206979 |
| `var2_p1`, p=4, y2→y1 | 0.829165909264 | 187 | 0.5081242019 |
| `var2_p1`, p=2, y1→y2 | 7.712480653203 | 193 | 0.0005992823 |
| `uschange`, p=2, Income→Consumption | 1.006147346109 | 180 | 0.3676717052 |
| `var3_p2`, p=2, x1→x3 | 3.931001499002 | 243 | 0.0208844791 |

`[V]` statsmodels' `grangercausalitytests` returns four variants, and
its `ssr_ftest` **is** `lmtest::grangertest` — `var2_p1`, lag 2:
`ssr_ftest = (0.7308637022, 4.8282e-01)`, matching the table above
exactly. `params_ftest` is bit-identical to `ssr_ftest`. The other two
are χ² forms:

| lag | ssr_ftest | ssr_chi2test | lrtest |
|---|---|---|---|
| 1 | 0.0234905511, p 8.7835e-01 | 0.0238501004, p 8.7727e-01 | 0.0238486713, p 8.7727e-01 |
| 2 | 0.7308637022, p 4.8282e-01 | 1.4995959899, p 4.7246e-01 | 1.4939457430, p 4.7380e-01 |
| 4 | 0.8291659093, p 5.0812e-01 | 3.4762891597, p 4.8149e-01 | 3.4458208664, p 4.8616e-01 |

### 3.3 When the two tests coincide, and when they do not

`[V]` The F **statistics** are identical when `k = 2` (so `|caused| = 1`)
and diverge otherwise:

| Case | System F (df2) | Pairwise F (res_df) | Same? |
|---|---|---|---|
| k=2, p=1, \|caused\|=1 | 0.023490551143 (392) | 0.023490551143 (196) | **yes** |
| k=2, p=2, \|caused\|=1 | 0.730863702158 (386) | 0.730863702158 (193) | **yes** |
| k=3, p=2, \|caused\|=2 | 3.242043205406 (723) | 3.931001499002 (243) | no |
| k=5, p=2, \|caused\|=4 | 0.451416306491 (870) | 1.006147346109 (180) | no |

Even where the statistic coincides, the **p-value never does**, because
`df2` differs by a factor of `K`. `0.4821580341` versus `0.4828206979`.

**Decision: expose both, as two methods of one name.** Julia's multiple
dispatch is exactly the right tool, and it keeps users from reaching for
the wrong one:

- `granger_test(m::VARModel; cause, caused)` — the system test.
- `granger_test(y, x; lags)` — the pairwise test.

Document in both docstrings that they are different tests, that the
statistics coincide only in the bivariate case, and that the p-values
never do. A test asserts both behaviours (§7.4).

---

## 4. A new primitive is required: `_f_ccdf` — and a reference bug

**Read this section before writing any code.** It is the one place where
10.2 cannot be built from what the package already has, and it is also
where both references can be beaten.

### 4.1 What the package has

`[S]` `src/diagnostics.jl`:

- `_f_cdf(x, d1, d2) = _beta_inc_reg(d1/2, d2/2, d1*x/(d1*x + d2))` at
  line 126 — the **lower** tail.
- `_beta_inc_reg(a, b, x)` at line 86 — the regularized incomplete beta
  function, continued fraction, tolerance `1e-14`.
- `_chisq_ccdf(x, df)` at line 13 — already an **upper** tail, via the
  regularized upper incomplete gamma.

There is **no `_f_ccdf`**. The only existing consumer, `[S]`
`diagnostics.jl:596`, needs the CDF. Granger's p-value needs the
complement.

### 4.2 Why `1 - _f_cdf(...)` is not acceptable

`[V]` **`vars::causality` computes its p-values as `1 − CDF`, and they
underflow to exactly zero in routine use.** The lines, read from source:

```
vars/R/causality.R:135    PVAL <- 1 - pf(STATISTIC, PARAMETER1, PARAMETER2)
vars/R/causality.R:167    PVAL <- 1 - pchisq(STATISTIC, PARAMETER)
```

`[V]` This is not a pathological-input problem. On `climhyd` (k=6,
p=2) — an ordinary dataset with an ordinary strong rejection:

```
Granger F = 13.359738021985, df = (10, 2634)
  vars reports:  1 - pf(F, 10, 2634)              = 0.000000e+00
  true tail:     pf(F, 10, 2634, lower.tail=FALSE) = 3.664479e-23
```

And for the instantaneous χ²:

```
chi2 = 184.728185617809, df = 5
  vars reports:  1 - pchisq(chi2, 5)               = 0.000000e+00
  true tail:     pchisq(chi2, 5, lower.tail=FALSE) = 5.229214e-38
```

`[V]` statsmodels gets both right (it calls the survival function
directly), so this is a `vars` defect, not a convention. `[V]` The same
`0.000000e+00` appears in four of the fifteen reference cases above
(`uschange` cause Income and cause Savings, `uschange` cause
{Income,Production}, and every `climhyd` case) — so a test that compared
TSAnalytics against `vars` on p-values alone would "fail" where
TSAnalytics is correct.

**This is the fourth bug this project has found in a reference
implementation**, after the three in R's `forecast` during Stage 9. Log
it in the charter §5 when 10.2 lands.

### 4.3 The fix

The F upper tail is the regularized incomplete beta with **swapped
parameters at the complementary argument**:

```
P(F_{d1,d2} > f) = I_{1-x}(d2/2, d1/2)   where  x = d1·f / (d1·f + d2)
```

and crucially `1 − x = d2 / (d1·f + d2)`, which must be formed
**directly** rather than as `1 - x` — for large `f`, `x → 1` and the
subtraction loses every significant digit.

```julia
"""
    _f_ccdf(f, d1, d2)

Upper tail P(X > f) for X ~ F(d1, d2), via the regularized incomplete
beta function with swapped parameters:

    P(F > f) = I_{d2/(d1 f + d2)}(d2/2, d1/2)

The complementary argument `d2/(d1 f + d2)` is formed directly rather
than as `1 - d1 f/(d1 f + d2)`: for large `f` the latter rounds to 1.0
and the entire tail is lost. That is precisely the failure mode in
`vars::causality`, which computes `1 - pf(...)` and reports p = 0 for
any F beyond roughly 1e-16 of tail -- on real data, from F of about 13
upward (verified: F = 13.3597, df = (10, 2634), true tail 3.66e-23,
reported 0). This function returns the tail.
"""
_f_ccdf(f::Real, d1::Real, d2::Real) =
    f <= 0 ? 1.0 : _beta_inc_reg(d2 / 2, d1 / 2, d2 / (d1 * f + d2))
```

`[V]` The identity verified against R's `pf(lower.tail=FALSE)` at
`d1 = 5, d2 = 800`, agreeing to every printed digit down to `1e-138`:

| F | `_f_ccdf` form | R `pf(lower=FALSE)` | R `1 - pf(...)` |
|---|---|---|---|
| 5 | 1.618120e-04 | 1.618120e-04 | 1.618120e-04 |
| 20 | 7.980745e-19 | 7.980745e-19 | **0.000000e+00** |
| 50 | 4.093317e-45 | 4.093317e-45 | **0.000000e+00** |
| 100 | 6.611145e-82 | 6.611145e-82 | **0.000000e+00** |
| 200 | 3.364102e-138 | 3.364102e-138 | **0.000000e+00** |

*(The `_f_ccdf` column was computed with scipy's `betainc`, whose own
relative error here is ~1e-7; R is the truth column. TSAnalytics'
`_beta_inc_reg` runs a continued fraction to `1e-14`, so it should do
better — assert with `rtol=1e-6` and tighten once measured.)*

**Also add a consistency test** between the new `_f_ccdf` and the
existing `_f_cdf`: they must sum to 1 in the regime where both are well
conditioned, and `_f_ccdf` must keep going where `1 - _f_cdf` flatlines.

### 4.4 `_chisq_ccdf` is already correct

`[S]` It computes the upper incomplete gamma directly, so it has no
analogous problem. `[V]` A test should confirm it beats `vars` in the
same regime: `_chisq_ccdf(184.728185617809, 5)` must return
`5.229214e-38`, not `0.0`.

---

## 5. A real bug in statsmodels' instantaneous causality

`[V]` **`VARResults.test_inst_causality` raises `IndexError` whenever
both the causing set and the caused set have more than one member.**
`vars::causality` handles every case.

Verified across three k values:

| k | \|causing\| | \|caused\| | statsmodels | R (`vars`) |
|---|---|---|---|---|
| 6 (`climhyd`) | 1 | 5 | 184.728185617809, df 5 | identical |
| 6 | 2 | 4 | `IndexError: shape mismatch: ... shapes (2,) (4,)` | 200.262649874447, df 8 |
| 6 | 3 | 3 | `IndexError: index 3 is out of bounds for axis 0 with size 3` | 173.881015172060, df 9 |
| 6 | 4 | 2 | `IndexError: shape mismatch: ... shapes (4,) (2,)` | 151.384179873536, df 8 |
| 6 | 5 | 1 | 175.715857769361, df 5 | identical |
| 5 (`uschange`) | 2 | 3 | `IndexError: shape mismatch: ... shapes (2,) (3,)` | 146.084802756021, df 6 |
| 4 (`uschange[:,1:4]`) | 2 | 2 | `IndexError: index 2 is out of bounds for axis 0 with size 2` | 118.341954206478, df 4 |

The cause, `[S]` from `statsmodels/tsa/vector_ar/var_model.py:2337`:

```python
sig_mask[causing_ind, caused_ind] = 1
```

Two fancy index arrays that only broadcast correctly when one of them
has length 1. When both are longer, numpy either refuses (shape
mismatch) or walks off the end.

**It always errors — it never returns a silently wrong answer.** That
was checked deliberately at `|causing| = |caused|`, where equal-length
broadcasting *would* have selected only the paired diagonal; it raises
instead. Good news, and worth stating in the handoff so nobody re-checks
it.

**Consequence for 10.2:** statsmodels can only verify the instantaneous
test for singleton causing or singleton caused sets. **`vars` is the
reference for the general block case**, and the correct `df` is
`|causing| · |caused|` — `[V]` confirmed by R across the whole sweep
(1·5=5, 2·4=8, 3·3=9, 4·2=8, 5·1=5).

`[V]` Reference values for the instantaneous test, agreeing with
statsmodels wherever it runs (`≤ 3.9e-14`):

| Case | χ² | df |
|---|---|---|
| `var2_p1`, p=1 | 25.783697306122 | 1 |
| `var2_p1`, p=2 | 24.374577969097 | 1 |
| `var2_p1`, p=4 | 23.736058320371 | 1 |
| `var3_p2`, p=2, cause x1 | 37.509236560536 | 2 |
| `var3_p2`, p=2, cause x2 | 34.796460089299 | 2 |
| `var3_p2`, p=2, cause x3 | 9.212257359217 | 2 |
| `var2_short`, p=2 | 2.903515865352 | 1 |
| `uschange`, p=2, cause Income | 84.431865745024 | 4 |
| `uschange`, p=2, cause Savings | 83.506033769775 | 4 |
| `uschange`, p=2, cause {Income,Production} | 146.084802756021 | 6 |
| `econ5`, p=2, cause gnp | 70.438733835801 | 4 |

Note the symmetry, `[V]` and assert it: instantaneous causality between
`{x1}` and `{x2,x3}` equals that between `{x2,x3}` and `{x1}` —
`37.509236560536` in both directions. The Granger test is **not**
symmetric, and a test should pin that contrast.

---

## 6. Proposed Julia API

```julia
struct GrangerTest <: HypothesisTest
    statistic::Float64              # the F statistic
    pvalue::Float64
    df1::Int
    df2::Int
    wald::Float64                   # chi-square form = df1 * statistic
    wald_pvalue::Float64
    cause::Vector{Symbol}
    caused::Vector{Symbol}
    kind::Symbol                    # :system | :pairwise
    lags::Int
    nobs::Int
end

struct InstantaneousCausalityTest <: HypothesisTest
    statistic::Float64              # Wald chi-square
    pvalue::Float64
    df::Int
    cause::Vector{Symbol}
    caused::Vector{Symbol}
    nobs::Int
end

# the system test: H0 is that `cause` does not Granger-cause `caused`
granger_test(m::VARModel;
             cause,                         # Symbol, or a vector of them
             caused = nothing) -> GrangerTest     # default: everything else

# the pairwise single-equation test -- a DIFFERENT test, see section 3
granger_test(y, x; lags::Integer) -> GrangerTest

instantaneous_causality(m::VARModel; cause, caused = nothing)
    -> InstantaneousCausalityTest
```

`statistic(t)` and `pvalue(t)` come free from `[S]` `src/abstract.jl:57,64`.
Both types need a `show` method printing a compact summary in the
package's existing style; `[V]` R's own header strings are a good model
for the wording:

```
Granger causality H0: x1 x2 do not Granger-cause x3
H0: No instantaneous causality between: x1 x2 and x3
```

### 6.1 Computation — the system test

With `B` the stacked coefficients and `Σ_a = vcov(m)` the full
`K(Kp+ndet)` covariance that 10.1 stores:

```
R·vec(B) = 0        R is the (df1 × K(Kp+ndet)) restriction matrix
                    selecting the `cause` lag coefficients in the
                    `caused` equations
Wald = (R vec(B))' (R Σ_a R')^{-1} (R vec(B))
F    = Wald / df1
df1  = p · |cause| · |caused|
df2  = K·T − K·(Kp + ndet)
p    = _f_ccdf(F, df1, df2)
```

**`R` is a sparse 0/1 selection matrix — never materialize it densely
and never form `R Σ_a R'` by two dense multiplications.** Collect the
index vector `idx` of the restricted positions and take
`Σ_a[idx, idx]` as a view. That turns an `O(df1 · (K(Kp+ndet))²)`
multiplication into an indexing operation. This is the same
"exploit the structure instead of calling the general routine" point as
the duplication matrix in the charter §7 item 4.

```julia
idx = _granger_indices(m, cause_ind, caused_ind)   # Vector{Int}, length df1
rb  = @view vecB[idx]
V   = @view Σa[idx, idx]
wald = dot(rb, cholesky(Hermitian(Matrix(V))) \ rb)
```

Use a Cholesky solve, not `inv(V)` — `V` is symmetric positive definite
by construction, and the solve is both faster and better conditioned.
Throw an `ArgumentError` naming collinearity if the factorization fails,
following `_ols`'s own precedent (`[S]` `src/unitroot.jl:182–189`).

### 6.2 Computation — instantaneous causality

Wald on the off-diagonal block of `vech(Σ)` with the duplication-matrix
pseudo-inverse. `[S]` statsmodels' own form at `var_model.py:2340–2351`:

```
Cs     = C · vech(Σ)
middle = (C D⁺ (Σ ⊗ Σ) D⁺' C')^{-1} / 2
Wald   = T · Cs' · middle · Cs
df     = |cause| · |caused|
```

Two performance notes, both verified in the charter §7 item 4:

- `D⁺` has a **closed form**. `D'D` is exactly diagonal with entries 1
  and 2, so `D⁺ = (D'D)⁻¹D'`. `[V]` Verified for k = 2, 3, 5, 8:
  off-diagonal mass of `D'D` exactly `0.0`, and
  `max|pinv(D) − closed form| = 2.22e-16`. **Do not call a pseudo-inverse.**
  Both references do; both are wasting an SVD.
- `C` is again a 0/1 selection; index into `D⁺ (Σ⊗Σ) D⁺'` rather than
  multiplying by `C`.

### 6.3 Computation — the pairwise test

Two OLS fits on one equation, restricted and unrestricted, on the same
sample:

```
F = ((RSS_r − RSS_u) / df1) / (RSS_u / df2)
df1 = lags,  df2 = T − 2·lags − 1
p   = _f_ccdf(F, df1, df2)
```

Reuse `_ols` directly — this one genuinely is a single-response
regression, so no multi-response path is needed. Both fits share the
unrestricted regressor matrix's leading columns, so build the
unrestricted `Z` once and take a view for the restricted fit.

---

## 7. Test battery

Exact values all `[V]`. The full reference set is in
`fixtures/ref_R_granger.json` and `fixtures/ref_py_granger.json`;
generating scripts in `scripts/`. Suggested file:
`test/test_granger.jl`. Target **≥ 180 assertions**.

### 7.1 `_f_ccdf` — the new primitive, tested before anything uses it

```julia
@testset "_f_ccdf: upper tail of the F distribution" begin
    # verified against R's pf(lower.tail=FALSE), d1=5, d2=800
    @test TSAnalytics._f_ccdf(5.0,   5, 800) ≈ 1.618120e-04   rtol=1e-6
    @test TSAnalytics._f_ccdf(20.0,  5, 800) ≈ 7.980745e-19   rtol=1e-6
    @test TSAnalytics._f_ccdf(50.0,  5, 800) ≈ 4.093317e-45   rtol=1e-6
    @test TSAnalytics._f_ccdf(100.0, 5, 800) ≈ 6.611145e-82   rtol=1e-6
    @test TSAnalytics._f_ccdf(200.0, 5, 800) ≈ 3.364102e-138  rtol=1e-6

    # consistency with the existing lower tail where both are well conditioned
    for f in (0.5, 1.0, 2.0, 3.5, 5.0)
        @test TSAnalytics._f_ccdf(f, 5, 800) + TSAnalytics._f_cdf(f, 5, 800) ≈ 1.0 atol=1e-12
    end
    # and it must keep going where 1 - cdf flatlines -- THE point of the primitive
    @test 1 - TSAnalytics._f_cdf(100.0, 5, 800) == 0.0
    @test TSAnalytics._f_ccdf(100.0, 5, 800) > 0.0

    @test TSAnalytics._f_ccdf(0.0,  5, 800) == 1.0
    @test TSAnalytics._f_ccdf(-1.0, 5, 800) == 1.0
    @test TSAnalytics._f_ccdf(Inf,  5, 800) ≈ 0.0 atol=1e-300
end

@testset "_chisq_ccdf keeps the tail too (vars loses it)" begin
    # vars computes 1 - pchisq(...) and reports exactly 0 here
    @test TSAnalytics._chisq_ccdf(184.728185617809, 5) ≈ 5.229214e-38 rtol=1e-6
    @test TSAnalytics._chisq_ccdf(84.587559378014, 3)  ≈ 3.181884e-18 rtol=1e-6
    @test TSAnalytics._chisq_ccdf(184.728185617809, 5) > 0.0
end
```

### 7.2 System Granger test — bivariate, against both references

```julia
@testset "granger_test (system): var2_p1" begin
    Y = read_fixture("var2_p1.csv")

    m1 = fit_var(Y, 1; deterministic=:const)
    g = granger_test(m1; cause=:y2)
    @test g.statistic ≈ 0.023490551143 atol=1e-10
    @test g.pvalue    ≈ 0.8782671595   atol=1e-9
    @test (g.df1, g.df2) == (1, 392)
    @test g.cause == [:y2] && g.caused == [:y1]
    @test g.kind == :system

    g = granger_test(m1; cause=:y1)
    @test g.statistic ≈ 17.504579420963 atol=1e-9
    @test g.pvalue    ≈ 0.0000354015    atol=1e-10
    @test (g.df1, g.df2) == (1, 392)

    m2 = fit_var(Y, 2; deterministic=:const)
    g = granger_test(m2; cause=:y2)
    @test g.statistic ≈ 0.730863702158 atol=1e-10
    @test g.pvalue    ≈ 0.4821580341   atol=1e-9
    @test (g.df1, g.df2) == (2, 386)
    # the chi-square form is exactly df1 * F
    @test g.wald ≈ 2 * 0.730863702158 atol=1e-10
    @test g.wald ≈ 1.461727404316654  atol=1e-10
    @test g.wald_pvalue ≈ 0.48149294395118214 atol=1e-10

    g = granger_test(m2; cause=:y1)
    @test g.statistic ≈ 7.712480653203 atol=1e-9
    @test g.pvalue    ≈ 0.0005196436   atol=1e-10

    m4 = fit_var(Y, 4; deterministic=:const)
    g = granger_test(m4; cause=:y2)
    @test g.statistic ≈ 0.829165909264 atol=1e-10
    @test g.pvalue    ≈ 0.5072142974   atol=1e-9
    @test (g.df1, g.df2) == (4, 374)
end
```

### 7.3 System Granger test — blocks of more than one cause

```julia
@testset "granger_test (system): var3_p2, k=3, blocks" begin
    Y3 = read_fixture("var3_p2.csv")
    m = fit_var(Y3, 2; deterministic=:const)

    for (cause, F, p) in [(:x1, 3.242043205406, 0.0119237127),
                          (:x2, 2.084475336178, 0.0811331989),
                          (:x3, 3.465874581908, 0.0081329307)]
        g = granger_test(m; cause=cause)
        @test g.statistic ≈ F atol=1e-9
        @test g.pvalue    ≈ p atol=1e-9
        @test (g.df1, g.df2) == (4, 723)       # p * 1 * 2 = 4
    end

    g = granger_test(m; cause=[:x1, :x2])
    @test g.statistic ≈ 3.743411847221 atol=1e-9
    @test g.pvalue    ≈ 0.0050422805   atol=1e-9
    @test (g.df1, g.df2) == (4, 723)           # p * 2 * 1 = 4
    @test g.caused == [:x3]

    g = granger_test(m; cause=[:x2, :x3])
    @test g.statistic ≈ 2.082120007689 atol=1e-9
    @test g.pvalue    ≈ 0.0814386404   atol=1e-9

    # Granger causality is NOT symmetric -- assert the contrast
    @test granger_test(m; cause=:x1).statistic !=
          granger_test(m; cause=[:x2,:x3]).statistic
end

@testset "granger_test (system): real data" begin
    U = read_fixture("uschange.csv")
    m = fit_var(U, 2; deterministic=:const)

    g = granger_test(m; cause=:Income)
    @test g.statistic ≈ 0.451416306491 atol=1e-9
    @test g.pvalue    ≈ 0.8899812692   atol=1e-9
    @test (g.df1, g.df2) == (8, 870)           # 2 * 1 * 4 = 8

    g = granger_test(m; cause=[:Income, :Production])
    @test g.statistic ≈ 0.948732438079 atol=1e-9
    @test g.pvalue    ≈ 0.4970171780   atol=1e-9
    @test (g.df1, g.df2) == (12, 870)          # 2 * 2 * 3 = 12

    g = granger_test(m; cause=:Savings)
    @test g.statistic ≈ 0.297268765099 atol=1e-9
    @test g.pvalue    ≈ 0.9669600021   atol=1e-9

    E = read_fixture("econ5.csv")
    g = granger_test(fit_var(E, 2; deterministic=:const); cause=:gnp)
    @test g.statistic ≈ 3.066973695481 rtol=1e-9   # rtol: references differ at 7.6e-12
    @test g.pvalue    ≈ 0.0020925493   atol=1e-9
    @test (g.df1, g.df2) == (8, 740)
end

@testset "granger_test (system): short sample" begin
    S = read_fixture("var2_short.csv")
    g = granger_test(fit_var(S, 2; deterministic=:const); cause=:y2)
    @test g.statistic ≈ 0.992318413455 atol=1e-10
    @test g.pvalue    ≈ 0.3785116479   atol=1e-9
    @test (g.df1, g.df2) == (2, 46)            # 2*28 - 2*5 = 46
end
```

### 7.4 Pairwise test — and the contrast with the system test

```julia
@testset "granger_test (pairwise): vs lmtest::grangertest" begin
    Y = read_fixture("var2_p1.csv")
    y1, y2 = Y[:, 1], Y[:, 2]

    g = granger_test(y1, y2; lags=1)
    @test g.kind == :pairwise
    @test g.statistic ≈ 0.023490551143 atol=1e-10
    @test g.pvalue    ≈ 0.8783459725   atol=1e-9
    @test (g.df1, g.df2) == (1, 196)

    g = granger_test(y1, y2; lags=2)
    @test g.statistic ≈ 0.730863702158 atol=1e-10
    @test g.pvalue    ≈ 0.4828206979   atol=1e-9
    @test (g.df1, g.df2) == (2, 193)

    g = granger_test(y1, y2; lags=4)
    @test g.statistic ≈ 0.829165909264 atol=1e-10
    @test g.pvalue    ≈ 0.5081242019   atol=1e-9
    @test (g.df1, g.df2) == (4, 187)

    g = granger_test(y2, y1; lags=2)
    @test g.statistic ≈ 7.712480653203 atol=1e-9
    @test g.pvalue    ≈ 0.0005992823   atol=1e-9

    U = read_fixture("uschange.csv")
    g = granger_test(U[:,1], U[:,2]; lags=2)       # Income -> Consumption
    @test g.statistic ≈ 1.006147346109 atol=1e-9
    @test g.pvalue    ≈ 0.3676717052   atol=1e-9
    @test (g.df1, g.df2) == (2, 180)

    Y3 = read_fixture("var3_p2.csv")
    g = granger_test(Y3[:,3], Y3[:,1]; lags=2)     # x1 -> x3
    @test g.statistic ≈ 3.931001499002 atol=1e-9
    @test g.pvalue    ≈ 0.0208844791   atol=1e-9
    @test (g.df1, g.df2) == (2, 243)
end

@testset "system and pairwise are DIFFERENT tests (section 3.3)" begin
    Y  = read_fixture("var2_p1.csv")
    Y3 = read_fixture("var3_p2.csv")

    # k=2: the F statistics coincide, the p-values never do
    sys = granger_test(fit_var(Y, 2; deterministic=:const); cause=:y2)
    pw  = granger_test(Y[:,1], Y[:,2]; lags=2)
    @test sys.statistic ≈ pw.statistic atol=1e-10      # same statistic
    @test sys.df2 == 386 && pw.df2 == 193              # different df
    @test sys.df2 == 2 * pw.df2                        # exactly K times
    @test !isapprox(sys.pvalue, pw.pvalue; atol=1e-5)  # different p
    @test sys.pvalue ≈ 0.4821580341 atol=1e-9
    @test pw.pvalue  ≈ 0.4828206979 atol=1e-9

    # k=3 with |caused| > 1: even the statistics diverge
    sys3 = granger_test(fit_var(Y3, 2; deterministic=:const); cause=:x1)
    pw3  = granger_test(Y3[:,3], Y3[:,1]; lags=2)
    @test !isapprox(sys3.statistic, pw3.statistic; atol=1e-3)
    @test sys3.statistic ≈ 3.242043205406 atol=1e-9
    @test pw3.statistic  ≈ 3.931001499002 atol=1e-9
end
```

### 7.5 Instantaneous causality

```julia
@testset "instantaneous_causality: vs vars (and statsmodels where it runs)" begin
    Y = read_fixture("var2_p1.csv")
    for (p, chi2, pv) in [(1, 25.783697306122, 3.819007e-07),
                          (2, 24.374577969097, 7.930840e-07),
                          (4, 23.736058320371, 1.104929e-06)]
        t = instantaneous_causality(fit_var(Y, p; deterministic=:const); cause=:y2)
        @test t.statistic ≈ chi2 atol=1e-10
        @test t.pvalue    ≈ pv   rtol=1e-6
        @test t.df == 1
    end

    Y3 = read_fixture("var3_p2.csv")
    m3 = fit_var(Y3, 2; deterministic=:const)
    for (cause, chi2, pv) in [(:x1, 37.509236560536, 7.160985e-09),
                              (:x2, 34.796460089299, 2.779999e-08),
                              (:x3,  9.212257359217, 9.990420e-03)]
        t = instantaneous_causality(m3; cause=cause)
        @test t.statistic ≈ chi2 atol=1e-9
        @test t.pvalue    ≈ pv   rtol=1e-6
        @test t.df == 2                                  # 1 * 2
    end

    # SYMMETRY: {x1} vs {x2,x3} equals {x2,x3} vs {x1}
    @test instantaneous_causality(m3; cause=:x1).statistic ≈
          instantaneous_causality(m3; cause=[:x2,:x3]).statistic atol=1e-10
    @test instantaneous_causality(m3; cause=[:x1,:x2]).statistic ≈
          instantaneous_causality(m3; cause=:x3).statistic atol=1e-10

    S = read_fixture("var2_short.csv")
    t = instantaneous_causality(fit_var(S, 2; deterministic=:const); cause=:y2)
    @test t.statistic ≈ 2.903515865352 atol=1e-10
    @test t.pvalue    ≈ 8.838658e-02   rtol=1e-6
end
```

### 7.6 The block case statsmodels cannot do — `vars` is the only reference

```julia
@testset "instantaneous_causality: blocks statsmodels cannot compute" begin
    # statsmodels raises IndexError whenever BOTH sets have >1 member.
    # vars is the reference here. df = |cause| * |caused| throughout.
    U = read_fixture("uschange.csv")
    t = instantaneous_causality(fit_var(U, 2; deterministic=:const);
                                cause=[:Income, :Production])
    @test t.statistic ≈ 146.084802756021 atol=1e-8
    @test t.df == 6                                      # 2 * 3
    @test t.pvalue > 0.0            # vars reports exactly 0 here; we must not
    @test t.pvalue < 1e-25

    # singleton cases, where statsmodels agrees with vars
    t = instantaneous_causality(fit_var(U, 2; deterministic=:const); cause=:Income)
    @test t.statistic ≈ 84.431865745024 atol=1e-9
    @test t.df == 4
    t = instantaneous_causality(fit_var(U, 2; deterministic=:const); cause=:Savings)
    @test t.statistic ≈ 83.506033769775 atol=1e-9
    @test t.df == 4

    E = read_fixture("econ5.csv")
    t = instantaneous_causality(fit_var(E, 2; deterministic=:const); cause=:gnp)
    @test t.statistic ≈ 70.438733835801 atol=1e-9
    @test t.df == 4
    @test t.pvalue ≈ 1.833851e-14 rtol=1e-5     # statsmodels value; R gives 1.831868e-14

    # the full k=6 sweep: df must be |cause| * |caused| at every split
    C = read_fixture("climhyd.csv")             # 454 x 6
    mc = fit_var(C, 2; deterministic=:const)
    nm = mc.names
    for (n, chi2, df) in [(1, 184.728185617809, 5), (2, 200.262649874447, 8),
                          (3, 173.881015172060, 9), (4, 151.384179873536, 8),
                          (5, 175.715857769361, 5)]
        t = instantaneous_causality(mc; cause=nm[1:n])
        @test t.statistic ≈ chi2 atol=1e-8
        @test t.df == df
        @test t.df == n * (6 - n)
        @test t.pvalue > 0.0                    # vars reports 0 for all five
    end
end
```

### 7.7 The p-value tail — where this package beats `vars`

```julia
@testset "p-value tails survive where vars reports exactly 0" begin
    C = read_fixture("climhyd.csv")
    mc = fit_var(C, 2; deterministic=:const)

    g = granger_test(mc; cause=mc.names[1])
    @test g.statistic ≈ 13.359738021985 atol=1e-8
    @test (g.df1, g.df2) == (10, 2634)
    # vars: 1 - pf(...) = 0.0 exactly.  True tail:
    @test g.pvalue ≈ 3.664479e-23 rtol=1e-5
    @test g.pvalue > 0.0

    g5 = granger_test(mc; cause=mc.names[1:5])
    @test g5.statistic ≈ 12.335643785349 atol=1e-8
    @test g5.pvalue ≈ 3.585106e-21 rtol=1e-5

    t = instantaneous_causality(mc; cause=mc.names[1])
    @test t.pvalue ≈ 5.229214e-38 rtol=1e-5
    @test t.pvalue > 0.0
end
```

### 7.8 Reduction and internal-consistency checks

```julia
@testset "reduction: Wald, F and chi-square are one quantity" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    g = granger_test(m; cause=:y2)
    @test g.wald ≈ g.df1 * g.statistic atol=1e-12
    @test g.pvalue      ≈ TSAnalytics._f_ccdf(g.statistic, g.df1, g.df2) atol=1e-14
    @test g.wald_pvalue ≈ TSAnalytics._chisq_ccdf(g.wald, g.df1)         atol=1e-14
    # chi-square p is always the smaller of the two (no small-sample correction)
    @test g.wald_pvalue < g.pvalue
end

@testset "reduction: a zero-restriction VAR has F = 0" begin
    # fit a VAR, zero the cause block by construction, re-test
    Y = read_fixture("var2_p1.csv")
    m = fit_var(Y, 2; deterministic=:const)
    m0 = @set_cause_block_zero(m, :y2)      # test helper, see note below
    @test granger_test(m0; cause=:y2).statistic ≈ 0.0 atol=1e-12
    @test granger_test(m0; cause=:y2).pvalue ≈ 1.0 atol=1e-12
end

@testset "caused defaults to everything else" begin
    m3 = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    @test granger_test(m3; cause=:x1).statistic ≈
          granger_test(m3; cause=:x1, caused=[:x2,:x3]).statistic atol=1e-14
    # an explicit subset is a different, legitimate test
    @test !isapprox(granger_test(m3; cause=:x1, caused=:x2).statistic,
                    granger_test(m3; cause=:x1).statistic; atol=1e-4)
    @test granger_test(m3; cause=:x1, caused=:x2).df1 == 2      # p * 1 * 1
end

@testset "name and index forms agree" begin
    m3 = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    @test granger_test(m3; cause=:x1).statistic ≈
          granger_test(m3; cause=1).statistic atol=1e-14
    @test granger_test(m3; cause=[:x1,:x2]).statistic ≈
          granger_test(m3; cause=[1,2]).statistic atol=1e-14
end
```

*(`@set_cause_block_zero` is a small test-only helper that rebuilds a
`VARModel` with the named variable's lag coefficients set to zero in the
other equations. Write it in the test file, not in `src/`.)*

### 7.9 Refusals — where both references wave nonsense through

`[V]` The verbatim reference behaviour:

| Case | `vars` | statsmodels |
|---|---|---|
| `cause` = every variable | `subscript out of bounds` | **no error** |
| unknown variable name/index | `Argument cause does not match variables names.` | `IndexError: list index out of range` |
| `cause` omitted entirely | **no error** — warns and silently uses the first variable | n/a |
| `cause` and `caused` overlap | n/a | **no error** |
| invalid `kind` | n/a | `ValueError: kind bogus not recognized` |

```julia
@testset "refusals" begin
    m  = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    m3 = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)

    # cause = all variables leaves nothing to be caused.
    # vars: "subscript out of bounds". statsmodels: no error at all.
    @test_throws ArgumentError granger_test(m; cause=[:y1, :y2])
    e = try granger_test(m; cause=[:y1,:y2]) catch err; err end
    @test occursin("caused", lowercase(sprint(showerror, e)))

    # overlapping cause and caused: statsmodels accepts this silently
    @test_throws ArgumentError granger_test(m3; cause=:x1, caused=[:x1, :x2])

    # unknown names and out-of-range indices
    @test_throws ArgumentError granger_test(m; cause=:nope)
    @test_throws ArgumentError granger_test(m; cause=0)
    @test_throws ArgumentError granger_test(m; cause=3)

    # empty cause: vars silently defaults to the first variable
    @test_throws ArgumentError granger_test(m; cause=Symbol[])
    @test_throws ArgumentError granger_test(m; cause=Int[])

    # duplicate entries
    @test_throws ArgumentError granger_test(m3; cause=[:x1, :x1])

    # pairwise: lags must be feasible for the sample
    y1, y2 = read_fixture("var2_p1.csv")[:,1], read_fixture("var2_p1.csv")[:,2]
    @test_throws ArgumentError granger_test(y1, y2; lags=0)
    @test_throws ArgumentError granger_test(y1, y2; lags=-1)
    @test_throws ArgumentError granger_test(y1, y2; lags=100)   # 2*100+1 > n
    @test_throws ArgumentError granger_test(y1, y2[1:end-1]; lags=2)   # ragged
    @test_throws ArgumentError granger_test([1.0, NaN, 3.0], [1.0, 2.0, 3.0]; lags=1)

    # instantaneous: same guards
    @test_throws ArgumentError instantaneous_causality(m; cause=[:y1,:y2])
    @test_throws ArgumentError instantaneous_causality(m; cause=:nope)
end
```

### 7.10 HypothesisTest contract, printing, type stability, allocations

```julia
@testset "HypothesisTest contract and show" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    g = granger_test(m; cause=:y2)
    t = instantaneous_causality(m; cause=:y2)

    @test g isa HypothesisTest
    @test t isa HypothesisTest
    @test statistic(g) === g.statistic
    @test pvalue(g)    === g.pvalue
    @test statistic(t) === t.statistic
    @test pvalue(t)    === t.pvalue
    @test 0.0 <= pvalue(g) <= 1.0
    @test 0.0 <= pvalue(t) <= 1.0

    s = sprint(show, g)
    @test occursin("Granger", s)
    @test occursin("y2", s) && occursin("y1", s)
    s = sprint(show, t)
    @test occursin("instantaneous", lowercase(s))
end

@testset "type stability and allocations" begin
    m  = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    mU = fit_var(read_fixture("uschange.csv"), 2; deterministic=:const)

    @test isconcretetype(GrangerTest)
    @test isconcretetype(InstantaneousCausalityTest)
    @test (@inferred granger_test(m; cause=:y2)) isa GrangerTest
    @test (@inferred instantaneous_causality(m; cause=:y2)) isa InstantaneousCausalityTest
    @test (@inferred TSAnalytics._f_ccdf(2.0, 3, 100)) isa Float64

    # the restriction must be an index view, not a dense R * Sigma * R'
    granger_test(mU; cause=:Income)                     # warm up
    @test (@allocated granger_test(mU; cause=:Income)) < 10_000

    # cost must scale with the restriction size, not with K^2(Kp+1)^2
    t_small = @elapsed for _ in 1:200; granger_test(m;  cause=:y2);     end
    t_big   = @elapsed for _ in 1:200; granger_test(mU; cause=:Income); end
    @test t_big < 50 * t_small
end
```

### 7.11 Cross-reference guards

```julia
@testset "convention guards" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    g = granger_test(m; cause=:y2)
    # df1 = p * |cause| * |caused|;  df2 = K*T - K*(Kp + ndet)
    @test g.df1 == 2 * 1 * 1
    @test g.df2 == 2 * m.T - 2 * (2*2 + 1)
    @test g.df2 == 386
    # the SYSTEM df, not a single equation's
    @test g.df2 != m.T - (2*2 + 1)
    # p-values must never be exactly 0 for a finite statistic (the vars bug)
    C = fit_var(read_fixture("climhyd.csv"), 2; deterministic=:const)
    for n in 1:5
        @test granger_test(C; cause=C.names[1:n]).pvalue > 0.0
        @test instantaneous_causality(C; cause=C.names[1:n]).pvalue > 0.0
    end
end
```

---

## 8. Performance notes specific to 10.2

Per charter §7. The three that matter here:

1. **The restriction matrix is an index vector, not a matrix.** `R` is
   `df1 × K(Kp+ndet)` and 0/1. Forming it densely and computing
   `R Σ_a R'` costs `O(df1 · (K(Kp+ndet))²)`; taking `Σ_a[idx, idx]`
   costs `O(df1²)`. On `uschange` (K=5, p=2) that is a 55×55 covariance
   and an 8-element restriction — a 50× difference for the same answer.
   Both references build the dense matrix.
2. **Cholesky solve, not `inv`.** `Σ_a[idx,idx]` is symmetric positive
   definite. `cholesky(Hermitian(V)) \ rb` beats `inv(V) * rb` and is
   better conditioned. Catch `PosDefException` and rethrow as an
   `ArgumentError` naming collinearity, as `_ols` does.
3. **`D⁺` in closed form** for the instantaneous test. `[V]` `D'D` is
   exactly diagonal with entries 1 and 2 (off-diagonal mass `0.0` for
   k = 2, 3, 5, 8), so `D⁺ = (D'D)⁻¹D'` and
   `max|pinv(D) − closed form| = 2.22e-16`. Both references call a
   pseudo-inverse they do not need.

Benchmark table to record on completion, for `k ∈ {2, 5, 6}` and the
block sizes exercised in §7.6, against `vars::causality` and
statsmodels.

---

## 9. Checklist

1. **Re-establish Julia** and re-verify every `[S]` claim before coding.
2. Add **`_f_ccdf`** to `src/diagnostics.jl`, next to `_f_cdf`, with the
   §4.3 docstring. Test it (§7.1) **before** anything consumes it.
3. Implement `GrangerTest`, `InstantaneousCausalityTest`,
   `granger_test` (both methods) and `instantaneous_causality` in a new
   `src/granger.jl`, included after `arx.jl` (for `_ols`) and after
   whatever file 10.1 puts `VARModel` in.
4. `show` methods for both types, in the package's existing compact
   style.
5. Run the battery. Target ≥ 180 assertions in `test/test_granger.jl`.
6. Copy `scripts/ref_R_granger.R` and `scripts/ref_py_granger.py` to
   `test/verification/granger/`.
7. `@code_warntype` on the three exported entry points; record the
   benchmark table (§8).
8. Add the manual section — Granger causality belongs in the same
   chapter as VAR. Say plainly that Granger causality is predictive
   precedence, not causation, and that the system and pairwise tests
   answer different questions.
9. Mark 10.2 `✅ built` in `development-sequence.md` with Stage-9-level
   detail.
10. **Record the two reference bugs in the charter §5**: `vars`'
    `1 − CDF` p-value underflow (§4.2) and statsmodels'
    `test_inst_causality` `IndexError` for multi-member blocks (§5).
    These are the fourth and fifth reference defects this project has
    documented; they belong in the running list.

---

## 10. Bundle additions for this item

```
fixtures/
  ref_R_granger.json     # 15 system tests + 15 instantaneous + 6 pairwise, from vars 1.6.1 / lmtest 0.9.40
  ref_py_granger.json    # statsmodels 0.15.0 counterparts, incl. its own error strings
  climhyd.csv            # 454 x 6, Shumway & Stoffer -- the k=6 block sweep and the tail cases
  us4.csv                # uschange[:, 1:4] -- the k=4, |cause|=2, |caused|=2 case
scripts/
  ref_R_granger.R        # regenerates ref_R_granger.json
  ref_py_granger.py      # regenerates ref_py_granger.json
```
