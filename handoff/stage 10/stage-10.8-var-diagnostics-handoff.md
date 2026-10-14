# Handoff: Stage 10.8 — Multivariate residual diagnostics

**Status: implemented ✅ (2026-10-13).**

For a fresh Claude Code session with no prior context. **Read
`stage-10-overall-handoff.md` first** — the stage charter carries the
conventions (§5, including the running list of reference defects at
§5.10), the architecture decisions (§6), the reference environment (§3)
and the performance and test policy (§7, §8).

Built against TSAnalytics.jl HEAD
`4bb66f38027be3b4fb538de408449f174128c8e0`.

Tagging: `[V]` executed, `[S]` read from source, `[U]` not re-verified.
Every reference number below is `[V]`. **Julia was not installed in the
sandbox that produced this document** — re-verify any package-internal
claim before building on it.

---

## 1. Why this is a roadmap row of its own

`development-sequence.md` has no 10.8. It needs one: *"is the residual
vector white?"* is the first question anyone asks of a fitted VAR, and
`[S]` the package has nothing multivariate. A grep across `src/` finds
no Hosking, no Li-McLeod, no multivariate Jarque-Bera, no multivariate
ARCH-LM. `ljungbox_test` (`src/diagnostics.jl:236–245`) takes a single
series.

**Build this third, right after 10.1 and 10.3.** A VAR without residual
diagnostics is not shippable, and this item is small.

- **Depends on:** 10.1 (`VARModel`), `_chisq_ccdf`
  (`src/diagnostics.jl:13`), and `_f_ccdf` — the primitive 10.2
  introduces (`stage-10.2-granger-handoff.md` §4.3). If 10.2 has not
  landed, add `_f_ccdf` here instead; §4 explains why it is mandatory.
- **Feeds into:** nothing structurally. A leaf, like 10.2.
- **Reference size:** `[S]` `vars/R/internal.R` holds `.pt.multi`
  (~45 lines), `.bgserial` (~56), `.arch.multi`, `.jb.multi`,
  `.matlag1/.matlag2`. All small, pure linear algebra.

---

## 2. Scope

### In

| Test | What it asks | References |
|---|---|---|
| Hosking portmanteau | residual vector white up to lag `h`? | `vars::serial.test(type="PT.asymptotic")`, statsmodels `test_whiteness(adjusted=false)` |
| Li-McLeod / small-sample adjusted portmanteau | same, finite-sample corrected | `vars::serial.test(type="PT.adjusted")`, statsmodels `test_whiteness(adjusted=true)` |
| Breusch-Godfrey LM | serial correlation, LM form | `vars::serial.test(type="BG")` — **R only** |
| Edgerton-Shukur F | the same LM in a small-sample F form | `vars::serial.test(type="ES")` — **R only** |
| Multivariate Jarque-Bera | residuals jointly normal? | `vars::normality.test`, statsmodels `test_normality` |
| JB skewness-only and kurtosis-only | which half fails | `vars::normality.test` — **R only** |
| Multivariate ARCH-LM | conditional heteroskedasticity | `vars::arch.test` — **R only** |
| Univariate JB and ARCH-LM per equation | which equation fails | `vars::normality.test`/`arch.test` with `multivariate.only=FALSE` |

### Out

- Multivariate normality tests other than JB (Mardia, Doornik-Hansen —
  `vars` uses the Doornik-Hansen *orthogonalization* inside its JB; see
  §5.3).
- Stability / structural-break tests (`vars::stability`, which wraps
  `strucchange`). A separate item if ever wanted.
- Residual plots. The IRF/FEVD recipes are 10.3's; a VAR residual panel
  would reuse `[S]` `src/diagnosticplot.jl`'s existing pattern and is
  worth a later row of its own.

---

## 3. What agrees, and what has only one reference

### 3.1 Two tests with a clean dual reference

`[V]` **Hosking portmanteau: `vars` PT.asymptotic ≡ statsmodels
`test_whiteness(adjusted=False)`, and PT.adjusted ≡ `adjusted=True`.**
Seven fixtures, K ∈ {2, 3, 5, 6}, identical df, worst difference
`4.2e-12`:

| Fixture | K | lags | PT.asymptotic | df | PT.adjusted | max diff |
|---|---|---|---|---|---|---|
| `var2_p1` p=1 | 2 | 8 | 15.985587414 | 28 | 16.367486277 | 2.5e-14 |
| `var2_p1` p=2 | 2 | 8 | 14.920131694 | 24 | 15.289383840 | 2.0e-14 |
| `var3_p2` p=2 | 3 | 8 | 58.592777973 | 54 | 59.982269267 | 7.1e-15 |
| `uschange` p=2 | 5 | 10 | 266.245960140 | 200 | 275.048373466 | 1.8e-12 |
| `econ5` p=2 | 5 | 6 | 194.706132010 | 100 | 199.219495487 | 4.2e-12 |
| `var2_short` p=2 | 2 | 4 | 4.627194437 | 8 | 5.207845203 | 8.9e-15 |
| `climhyd` p=2 | 6 | 8 | 367.405871419 | 216 | 370.921973365 | 1.7e-13 |

`df = K²(h − p)` — `[V]` check: K=2, h=8, p=2 → 4·6 = 24. ✓

`[V]` **Multivariate Jarque-Bera: `vars::normality.test` ≡ statsmodels
`test_normality`.** Seven fixtures, worst difference `1.05e-11`,
`df = 2K`:

| Fixture | JB | df |
|---|---|---|
| `var2_p1` p=1 | 1.982033145 | 4 |
| `var2_p1` p=2 | 1.940443127 | 4 |
| `var3_p2` p=2 | 8.600401802 | 6 |
| `uschange` p=2 | 282.122726829 | 10 |
| `econ5` p=2 | 97.308004668 | 10 |
| `var2_short` p=2 | 2.562616200 | 4 |
| `climhyd` p=2 | 1850.922870571 | 12 |

### 3.2 Four tests with only `vars`

statsmodels has no multivariate ARCH test, no Breusch-Godfrey for a VAR,
and does not split JB into its skewness and kurtosis halves. For those,
`vars` is the only software reference — so verify them **structurally**
as well: the skewness and kurtosis components must sum to the full JB
statistic, and their df must sum to the full df.

`[V]` `var2_p1` p=2: `jb_skew = 1.111294059` (df 2) and
`jb_kurt = 0.829149068` (df 2), summing to `1.940443127` (df 4) — the
reported JB exactly. Assert that identity on every fixture; it is a real
check that costs nothing.

`[V]` Reference values, `var2_p1` p=2, `lags.bg = lags.multi = 5`,
`lags.single = 4`:

```
bg        (Breusch-Godfrey LM)   stat = 15.553823556   df = 20          p = 7.438899e-01
es        (Edgerton-Shukur F)    stat =  0.746057268   df = (20, 364)   p = 7.774361e-01
arch_mul  (multivariate ARCH-LM) stat = 53.369468157   df = 45          p = 1.835001e-01
jb_uni    y1 = 0.262207074 (p 8.7713e-01),  y2 = 1.258163434 (p 5.3308e-01)
arch_uni  y1 = 0.382425551 (p 9.8389e-01),  y2 = 1.768510494 (p 7.7824e-01)
```

`[V]` `var3_p2` p=2, same lag settings:

```
bg        stat =  45.569851060   df = 45          p = 4.482481e-01
es        stat =   0.979915496   df = (45, 666)   p = 5.116529e-01
arch_mul  stat = 177.763848409   df = 180         p = 5.331398e-01
jb_skew   stat =   3.313788431   df = 3
jb_kurt   stat =   5.286613371   df = 3           (sum = 8.600401802 = jb_mul)
jb_uni    x1 = 1.860096970, x2 = 0.835292836, x3 = 7.731162484 (p 2.0951e-02)
arch_uni  x1 = 3.412578844, x2 = 6.728881673, x3 = 10.244134712 (p 3.6510e-02)
```

`[V]` `uschange` p=2, `lags.pt=10, lags.bg=4, lags.multi=4, lags.single=4`:

```
bg        stat =  150.324883239   df = 100         p = 8.514125e-04
es        stat =    1.503478581   df = (100, 736)  p = 1.965738e-03
jb_skew   stat =   16.764027225   df = 5           p = 4.969766e-03
jb_kurt   stat =  265.358699604   df = 5
arch_mul  stat = 1198.142848744   df = 900         p = 8.352452e-11
jb_uni    Consumption 39.335608075, Income 72.718379713, Production 37.255460020,
          Savings 128.731046507, Unemployment 30.455037263
arch_uni  Consumption 6.772822741, Income 4.805773243, Production 20.030889440,
          Savings 23.506183275, Unemployment 20.735659049
```

`[V]` `climhyd` p=2 (K=6), `lags.pt=8, lags.bg=3, lags.multi=2`:

```
bg        stat =  321.623555216   df = 108          p -> 0 reported by vars (see section 4)
es        stat =    3.731117359   df = (108, 2391)  p -> 0 reported by vars
jb_skew   stat =  270.059874481   df = 6
jb_kurt   stat = 1580.862996090   df = 6            (sum = 1850.922870571 = jb_mul)
arch_mul  stat =  960.019370587   df = 882          p = 3.428409e-02
```

`[V]` `var2_short` p=2 (n = 30, T = 28), `lags.pt=4, lags.bg=2, lags.multi=2`:

```
pt_asym   4.627194437 (df 8)      pt_adj   5.207845203 (df 8)
bg        6.403518808 (df 8)      es       0.581349581 (df (8, 36))
jb_mul    2.562616200 (df 4)      arch_mul 18.689489651 (df 18)
```

---

## 4. `_f_ccdf` is mandatory here too, and `vars`' p-values are unusable

`[V]` **Every p-value in the whole `vars` diagnostics suite is computed
as `1 − CDF` and underflows.** `[S]` Ten occurrences in
`vars/R/internal.R`:

```
:480  .arch.uni     PVAL <- 1 - pchisq(STATISTIC, df = PARAMETER)
:512  .arch.multi   PVAL <- 1 - pchisq(STATISTIC, df = PARAMETER)
:534  .jb.uni       PVAL <- 1 - pchisq(STATISTIC, df = 2)
:555  .jb.multi     PVAL <- 1 - pchisq(STATISTIC, df = PARAMETER)
:563  .jb.multi     PVAL <- 1 - pchisq(STATISTIC, df = PARAMETER)
:571  .jb.multi     PVAL <- 1 - pchisq(STATISTIC, df = PARAMETER)
:617  .pt.multi     PVAL <- 1 - pchisq(STATISTIC, df = PARAMETER)
:629  .pt.multi     PVAL <- 1 - pchisq(STATISTIC, df = PARAMETER)
:668  .bgserial     PVAL <- 1 - pchisq(STATISTIC, df = PARAMETER)
:686  (ES F test)   PVAL <- 1 - pf(LMFh.stat, PARAMETER1, PARAMETER2)
```

This is the same defect 10.2 found in `vars::causality` (charter §5.10
item 1), and it is **package-wide**, not one function. `[V]` It bites on
ordinary data:

| Case | `vars` reports | true tail |
|---|---|---|
| `uschange` multivariate JB = 282.122726829, df 10 | `0.000000e+00` | `9.282e-55` |
| `econ5` multivariate JB = 97.308004668, df 10 | `2.220e-16` (the `1 − x` floor) | `1.882e-16` |
| `climhyd` BG = 321.623555216, df 108 | `0.000000e+00` | finite |
| `climhyd` ES F = 3.731117359, df (108, 2391) | `0.000000e+00` | finite |

**Consequences:**

1. Take **every** p-value from TSAnalytics' own `_chisq_ccdf`
   (`src/diagnostics.jl:13`, already an upper tail) and `_f_ccdf`
   (10.2 §4.3).
2. **Assert statistics against `vars`, never p-values.** Where
   statsmodels also implements the test, compare p-values with it; for
   the four R-only tests, compare against `pchisq(..., lower.tail=FALSE)`
   computed directly in the verification script rather than against
   `vars`' reported value. `scripts/ref_R_diag.R` records `vars`'
   reported p-values as-is — they are there to document the defect, not
   to assert against.
3. A test must confirm TSAnalytics stays positive where `vars` reports
   zero (§7.6).

If 10.2 has not landed yet, add `_f_ccdf` in this item — the ES test
needs it, and the `climhyd` case above shows `1 − pf` failing on real
data at an F of only 3.73.

---

## 5. Definitions to implement, with the structural checks that pin them

### 5.1 Hosking portmanteau

```
C_0 = E'E / T
C_i = E_{(i)}' E_{(-i)} / T          (E lagged i periods against itself)
Q_h = T * sum_{i=1}^{h} tr( C_i' C_0^-1 C_i C_0^-1 )
df  = K^2 (h - p)
```

`[S]` `vars`' `.pt.multi` is exactly this, and worth reading for the
indexing — it drops the first `i` rows of both matrices before the cross
product.

The adjusted form rescales each term by `T / (T - i)`:

```
Q_h^adj = T * sum_{i=1}^{h} [ T/(T-i) ] * tr( C_i' C_0^-1 C_i C_0^-1 )
```

`[V]` Structural check, `var2_p1` p=2: adjusted (15.289383840) is larger
than asymptotic (14.920131694) on every fixture, since every weight
exceeds 1. Assert the inequality — it catches a sign or reciprocal error
immediately.

**Performance:** factorize `C_0` **once** (Cholesky — it is symmetric
positive definite) and reuse the factor across all `h` lags. The trace
is `sum(abs2, L \ C_i / L')`-shaped; do not form `C_0^-1` explicitly and
do not refactorize per lag. That is the difference between `O(h K³)` and
`O(h K³ + K³)` with a much better constant, and `[S]` `vars` inverts
`C_0` once but then does four dense `K × K` products per lag where two
triangular solves suffice.

### 5.2 Multivariate Jarque-Bera

Standardize the residuals by the Cholesky factor of `C_0`, then compute
skewness and kurtosis of each standardized series:

```
Es = P^-1 E'          (P = chol(C_0)', lower)
b1 = column means of Es^3 ;  b2 = column means of Es^4
s_skew = T * b1' b1 / 6
s_kurt = T * (b2 - 3)' (b2 - 3) / 24
JB     = s_skew + s_kurt          df = 2K   (each half has df K)
```

`[V]` The identity `JB = s_skew + s_kurt` holds exactly on every
fixture — `1.111294059 + 0.829149068 = 1.940443127` for `var2_p1` p=2,
`3.313788431 + 5.286613371 = 8.600401802` for `var3_p2`,
`270.059874481 + 1580.862996090 = 1850.922870571` for `climhyd`.
**Assert it.** It is the only independent check available for the two
halves, which `vars` alone reports.

`[S]` One caution: `vars` offers both a Cholesky and a
Doornik-Hansen orthogonalization for the standardization step. The
reference values above are the **default**. Check `vars`'
`normality.test` signature before implementing, and state which
orthogonalization TSAnalytics uses in the docstring — the statistic is
not invariant to the choice, and agreement with statsmodels
(`[V]` 1.05e-11) confirms the default is the one to match.

### 5.3 Multivariate ARCH-LM

Regress `vech(e_t e_t')` on `q` of its own lags and form the LM
statistic:

```
df = q * (K(K+1)/2)^2
```

`[V]` Check: K=2, q=5 → `5 * 3² = 45` ✓ (`var2_p1`). K=3, q=5 →
`5 * 6² = 180` ✓ (`var3_p2`). K=5, q=4 → `4 * 15² = 900` ✓
(`uschange`). K=6, q=2 → `2 * 21² = 882` ✓ (`climhyd`).

**That df formula grows as `q K⁴`.** At K=6, q=2 it is already 882 on
452 observations. The regression has `K(K+1)/2 = 21` responses and
`21q + 1 = 43` regressors — which is exactly `_mols`' job (10.0), so
reuse it rather than writing another solve. And **refuse when the
regression is infeasible**: `T - q` rows against `21q + 1` regressors
runs out fast, and `[V]` `vars` does not check (§6).

### 5.4 Breusch-Godfrey and Edgerton-Shukur

Auxiliary regression of the VAR residuals on the original regressors
plus `h` lags of the residuals.

```
BG (LM) : stat = T * (K - tr(Sigma_r^-1 Sigma_e)) ... df = h K^2
ES (F)  : a small-sample F transform of the same LM, df = (h K^2, denom)
```

`[V]` Both df patterns confirmed: `var2_p1` p=2, h=5, K=2 → BG df = 20,
ES df = (20, 364). `var3_p2` h=5, K=3 → 45 and (45, 666). `uschange`
h=4, K=5 → 100 and (100, 736). `climhyd` h=3, K=6 → 108 and (108, 2391).

`[S]` Read `vars`' `.bgserial` (`R/internal.R:639–694`) for the exact
denominator df — it is the one quantity here with no second reference,
and getting it wrong changes the p-value without changing the statistic.
Record the formula in the docstring with the four `[V]` df values above
as the check.

---

## 6. Refusals — the two references fail in opposite directions

`[V]` Verbatim behaviour:

| Case | `vars` | statsmodels |
|---|---|---|
| portmanteau lags ≤ p | **no error** — runs with `df = K²(h−p) ≤ 0` | `ValueError: The whiteness test can only be used when nlags is larger than the number of lags included in the model (2).` |
| portmanteau lags = 0 | `non-conformable arguments` | same clear `ValueError` |
| portmanteau lags = 300 (T = 198) | `non-conformable arguments` | **no error** — returns `df = 1192`, stat 352.70187812995357 |
| `arch.test(lags.multi = 0)` | `subscript out of bounds` | n/a |

statsmodels has the right check for `lags ≤ p` and none for
`lags > T`; `vars` has neither, and fails with internal messages where
it does fail. **TSAnalytics refuses all four by name**, which is
strictly better than either.

```julia
@testset "refusals" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)

    # lags must exceed p, or df <= 0. vars runs anyway.
    @test_throws ArgumentError portmanteau_test(m; lags=2)
    @test_throws ArgumentError portmanteau_test(m; lags=1)
    @test_throws ArgumentError portmanteau_test(m; lags=0)
    e = try portmanteau_test(m; lags=2) catch err; err end
    @test occursin("lags", lowercase(sprint(showerror, e)))

    # lags must leave observations. statsmodels returns df=1192 on T=198.
    @test_throws ArgumentError portmanteau_test(m; lags=300)
    @test_throws ArgumentError portmanteau_test(m; lags=m.T)

    @test_throws ArgumentError arch_lm_test(m; lags=0)
    @test_throws ArgumentError arch_lm_test(m; lags=-1)
    # the ARCH regression must be feasible: K(K+1)/2 responses, 21q+1 regressors
    mc = fit_var(read_fixture("climhyd.csv"), 2; deterministic=:const)
    @test_throws ArgumentError arch_lm_test(mc; lags=25)        # 21*25+1 = 526 > T-25
    e = try arch_lm_test(mc; lags=25) catch err; err end
    @test occursin("observations", lowercase(sprint(showerror, e)))

    @test_throws ArgumentError bg_test(m; lags=0)
    @test_throws ArgumentError normality_test(m; orthogonalization=:bogus)
end
```

---

## 7. Test battery

Exact values all `[V]`, from `fixtures/ref_R_diag.json` and
`fixtures/ref_py_diag.json`. Suggested file: `test/test_vardiag.jl`.
Target **≥ 200 assertions**.

### 7.1 Portmanteau, both forms, all seven fixtures

```julia
@testset "portmanteau_test: Hosking, vs vars and statsmodels" begin
    for (fx, p, lags, asym, adj, df) in [
            ("var2_p1.csv",    1,  8,  15.985587414,  16.367486277,  28),
            ("var2_p1.csv",    2,  8,  14.920131694,  15.289383840,  24),
            ("var3_p2.csv",    2,  8,  58.592777973,  59.982269267,  54),
            ("uschange.csv",   2, 10, 266.245960140, 275.048373466, 200),
            ("econ5.csv",      2,  6, 194.706132010, 199.219495487, 100),
            ("var2_short.csv", 2,  4,   4.627194437,   5.207845203,   8),
            ("climhyd.csv",    2,  8, 367.405871419, 370.921973365, 216)]
        m = fit_var(read_fixture(fx), p; deterministic=:const)
        a = portmanteau_test(m; lags=lags, adjusted=false)
        b = portmanteau_test(m; lags=lags, adjusted=true)
        @test a.statistic ≈ asym rtol=1e-9
        @test b.statistic ≈ adj  rtol=1e-9
        @test a.df == df == b.df
        K = size(m.A[1], 1)
        @test a.df == K^2 * (lags - p)             # the df formula
        @test b.statistic > a.statistic            # every weight T/(T-i) > 1
        @test 0.0 <= a.pvalue <= 1.0
    end
end
```

### 7.2 Multivariate Jarque-Bera, and the component identity

```julia
@testset "normality_test: multivariate JB" begin
    for (fx, p, jb, df) in [("var2_p1.csv",    1,    1.982033145,  4),
                            ("var2_p1.csv",    2,    1.940443127,  4),
                            ("var3_p2.csv",    2,    8.600401802,  6),
                            ("uschange.csv",   2,  282.122726829, 10),
                            ("econ5.csv",      2,   97.308004668, 10),
                            ("var2_short.csv", 2,    2.562616200,  4),
                            ("climhyd.csv",    2, 1850.922870571, 12)]
        m = fit_var(read_fixture(fx), p; deterministic=:const)
        t = normality_test(m)
        K = size(m.A[1], 1)
        @test t.statistic ≈ jb rtol=1e-9
        @test t.df == df == 2K
        # the ONLY independent check on the two halves, which only vars reports
        @test t.skewness_statistic + t.kurtosis_statistic ≈ t.statistic rtol=1e-11
        @test t.skewness_df == K && t.kurtosis_df == K
    end
end

@testset "normality_test: skewness and kurtosis halves" begin
    for (fx, p, sk, ku) in [("var2_p1.csv",  2,   1.111294059,    0.829149068),
                            ("var3_p2.csv",  2,   3.313788431,    5.286613371),
                            ("uschange.csv", 2,  16.764027225,  265.358699604),
                            ("climhyd.csv",  2, 270.059874481, 1580.862996090),
                            ("var2_short.csv", 2, 1.640497438,    0.922118762)]
        t = normality_test(fit_var(read_fixture(fx), p; deterministic=:const))
        @test t.skewness_statistic ≈ sk rtol=1e-9
        @test t.kurtosis_statistic ≈ ku rtol=1e-9
    end
end

@testset "normality_test: per-equation JB" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    u = normality_test(m; univariate=true).univariate
    @test u[:y1].statistic ≈ 0.262207074 rtol=1e-8
    @test u[:y2].statistic ≈ 1.258163434 rtol=1e-8
    @test all(t -> t.df == 2, values(u))

    m3 = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    u3 = normality_test(m3; univariate=true).univariate
    @test u3[:x1].statistic ≈ 1.860096970 rtol=1e-8
    @test u3[:x2].statistic ≈ 0.835292836 rtol=1e-8
    @test u3[:x3].statistic ≈ 7.731162484 rtol=1e-8

    mU = fit_var(read_fixture("uschange.csv"), 2; deterministic=:const)
    uU = normality_test(mU; univariate=true).univariate
    @test uU[:Consumption].statistic  ≈  39.335608075 rtol=1e-8
    @test uU[:Income].statistic       ≈  72.718379713 rtol=1e-8
    @test uU[:Production].statistic   ≈  37.255460020 rtol=1e-8
    @test uU[:Savings].statistic      ≈ 128.731046507 rtol=1e-8
    @test uU[:Unemployment].statistic ≈  30.455037263 rtol=1e-8
end
```

### 7.3 Multivariate ARCH-LM

```julia
@testset "arch_lm_test: multivariate" begin
    for (fx, p, q, stat, df) in [("var2_p1.csv",    2, 5,   53.369468157,  45),
                                 ("var3_p2.csv",    2, 5,  177.763848409, 180),
                                 ("uschange.csv",   2, 4, 1198.142848744, 900),
                                 ("climhyd.csv",    2, 2,  960.019370587, 882),
                                 ("var2_short.csv", 2, 2,   18.689489651,  18)]
        m = fit_var(read_fixture(fx), p; deterministic=:const)
        t = arch_lm_test(m; lags=q)
        K = size(m.A[1], 1)
        @test t.statistic ≈ stat rtol=1e-8
        @test t.df == df
        @test t.df == q * (K*(K+1) ÷ 2)^2          # the df formula
    end
end

@testset "arch_lm_test: per-equation" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    u = arch_lm_test(m; lags=4, univariate=true).univariate
    @test u[:y1].statistic ≈ 0.382425551 rtol=1e-8
    @test u[:y2].statistic ≈ 1.768510494 rtol=1e-8
    @test all(t -> t.df == 4, values(u))

    m3 = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)
    u3 = arch_lm_test(m3; lags=4, univariate=true).univariate
    @test u3[:x1].statistic ≈  3.412578844 rtol=1e-8
    @test u3[:x2].statistic ≈  6.728881673 rtol=1e-8
    @test u3[:x3].statistic ≈ 10.244134712 rtol=1e-8
end
```

### 7.4 Breusch-Godfrey and Edgerton-Shukur

```julia
@testset "bg_test and es_test" begin
    for (fx, p, h, bg, bgdf, es, esdf) in [
            ("var2_p1.csv",    2, 5,  15.553823556,  20, 0.746057268, (20,  364)),
            ("var3_p2.csv",    2, 5,  45.569851060,  45, 0.979915496, (45,  666)),
            ("uschange.csv",   2, 4, 150.324883239, 100, 1.503478581, (100, 736)),
            ("climhyd.csv",    2, 3, 321.623555216, 108, 3.731117359, (108, 2391)),
            ("var2_short.csv", 2, 2,   6.403518808,   8, 0.581349581, (8,   36))]
        m = fit_var(read_fixture(fx), p; deterministic=:const)
        K = size(m.A[1], 1)
        b = bg_test(m; lags=h)
        @test b.statistic ≈ bg rtol=1e-8
        @test b.df == bgdf == h * K^2
        f = es_test(m; lags=h)
        @test f.statistic ≈ es rtol=1e-8
        @test (f.df1, f.df2) == esdf
        @test f.df1 == h * K^2
    end
end
```

### 7.5 Structural and cross-test invariants

```julia
@testset "invariants" begin
    m = fit_var(read_fixture("var3_p2.csv"), 2; deterministic=:const)

    # the portmanteau's own building blocks
    E = residuals(m); T = m.T
    C0 = (E' * E) ./ T
    @test issymmetric(round.(C0, digits=12))
    @test isposdef(Hermitian(C0))
    # lag-0 term is excluded; including it would add tr(I) = K per lag
    @test portmanteau_test(m; lags=3).statistic <
          portmanteau_test(m; lags=4).statistic      # monotone in lags

    # JB standardisation: the standardised residuals have identity covariance
    Es = standardized_residuals(m)                   # test-visible helper
    @test (Es' * Es) ./ T ≈ I(3) atol=1e-10

    # white noise must NOT be rejected; a deliberately mis-specified model must be
    rng = MersenneTwister(99)
    W = randn(rng, 400, 3)
    mw = fit_var(W, 1; deterministic=:const)
    @test portmanteau_test(mw; lags=8).pvalue > 0.01
    @test normality_test(mw).pvalue > 0.01
    # under-fitting a true VAR(2) leaves structure the test should find
    mu = fit_var(read_fixture("var3_p2.csv"), 1; deterministic=:const)
    @test portmanteau_test(mu; lags=8).pvalue < portmanteau_test(m; lags=8).pvalue
end
```

### 7.6 p-value tails — where this package beats `vars`

```julia
@testset "p-values survive where vars reports exactly 0" begin
    mU = fit_var(read_fixture("uschange.csv"), 2; deterministic=:const)
    t = normality_test(mU)
    @test t.statistic ≈ 282.122726829 rtol=1e-9
    # vars: 1 - pchisq(282.12, 10) = 0.0 exactly. True tail:
    @test t.pvalue ≈ 9.282e-55 rtol=1e-3
    @test t.pvalue > 0.0

    mE = fit_var(read_fixture("econ5.csv"), 2; deterministic=:const)
    tE = normality_test(mE)
    @test tE.statistic ≈ 97.308004668 rtol=1e-9
    @test tE.pvalue ≈ 1.882e-16 rtol=1e-3      # vars reports 2.220e-16, the 1-x floor
    @test tE.pvalue != 2.220446049250313e-16

    # and the F-based ES test, where 1 - pf also underflows
    mC = fit_var(read_fixture("climhyd.csv"), 2; deterministic=:const)
    @test es_test(mC; lags=3).pvalue > 0.0       # vars reports 0.000000e+00
    @test bg_test(mC; lags=3).pvalue > 0.0       # likewise
    @test normality_test(mC).pvalue > 0.0
end
```

### 7.7 Contract, printing, type stability, performance

```julia
@testset "HypothesisTest contract and show" begin
    m = fit_var(read_fixture("var2_p1.csv"), 2; deterministic=:const)
    for t in (portmanteau_test(m; lags=8), normality_test(m),
              arch_lm_test(m; lags=5), bg_test(m; lags=5), es_test(m; lags=5))
        @test t isa HypothesisTest
        @test statistic(t) === t.statistic
        @test 0.0 <= pvalue(t) <= 1.0
        s = sprint(show, t)
        @test !isempty(s)
    end
    @test occursin("ortmanteau", sprint(show, portmanteau_test(m; lags=8)))
end

@testset "type stability and performance" begin
    mC = fit_var(read_fixture("climhyd.csv"), 2; deterministic=:const)
    @test (@inferred portmanteau_test(mC; lags=8)) isa PortmanteauTest
    @test (@inferred normality_test(mC)) isa MultivariateNormalityTest

    # C_0 is factorized ONCE, so cost is near-linear in lags, not quadratic
    portmanteau_test(mC; lags=4)
    t4  = @elapsed for _ in 1:100; portmanteau_test(mC; lags=4);  end
    t16 = @elapsed for _ in 1:100; portmanteau_test(mC; lags=16); end
    @test t16 < 6 * t4
    # and it must not allocate a K x K inverse per lag
    portmanteau_test(mC; lags=8)
    @test (@allocated portmanteau_test(mC; lags=8)) < 60_000
end
```

---

## 8. Proposed Julia API

```julia
struct PortmanteauTest <: HypothesisTest
    statistic::Float64; pvalue::Float64; df::Int
    lags::Int; adjusted::Bool; nobs::Int; K::Int
end

struct MultivariateNormalityTest <: HypothesisTest
    statistic::Float64; pvalue::Float64; df::Int
    skewness_statistic::Float64; skewness_pvalue::Float64; skewness_df::Int
    kurtosis_statistic::Float64; kurtosis_pvalue::Float64; kurtosis_df::Int
    univariate::Union{Nothing,Dict{Symbol,JarqueBeraTest}}
    orthogonalization::Symbol
end

struct ArchLMTest <: HypothesisTest
    statistic::Float64; pvalue::Float64; df::Int
    lags::Int; univariate::Union{Nothing,Dict{Symbol,ArchLMTest}}
end

struct BreuschGodfreyTest <: HypothesisTest
    statistic::Float64; pvalue::Float64; df::Int; lags::Int
end

struct EdgertonShukurTest <: HypothesisTest
    statistic::Float64; pvalue::Float64; df1::Int; df2::Int; lags::Int
end

portmanteau_test(m::VARModel; lags::Integer, adjusted::Bool=false)
normality_test(m::VARModel; univariate::Bool=false, orthogonalization::Symbol=:cholesky)
arch_lm_test(m::VARModel; lags::Integer=5, univariate::Bool=false)
bg_test(m::VARModel; lags::Integer=5)
es_test(m::VARModel; lags::Integer=5)

# one call that runs the standard set, for the manual's "check your VAR" section
var_diagnostics(m::VARModel; pt_lags=nothing, bg_lags=5, arch_lags=5) -> NamedTuple
```

`pt_lags` defaults to `max(p+1, min(10, T ÷ 4))` — stated in the
docstring, since neither reference has a sensible default and `vars`
will happily run with `df ≤ 0`.

`JarqueBeraTest` already exists `[S]` (`src/diagnostics.jl:445` computes
a univariate JB); reuse that type for the per-equation results rather
than introducing a second one.

---

## 9. Checklist

1. **Re-establish Julia** and re-verify every `[S]` claim before coding.
2. Add `_f_ccdf` if 10.2 has not landed (§4).
3. Implement the five tests in a new `src/vardiagnostics.jl`, included
   after whatever file holds `VARModel` and after `diagnostics.jl` (for
   `_chisq_ccdf`, `_f_ccdf`, `_beta_inc_reg` and the existing
   `JarqueBeraTest`).
4. `show` methods for all five, house style.
5. Run the battery. Target ≥ 200 assertions in `test/test_vardiag.jl`.
6. Copy `scripts/ref_R_diag.R` and `scripts/ref_py_diag.py` to
   `test/verification/vardiag/`.
7. `@code_warntype` on all five entry points; confirm `C_0` is
   factorized once (§7.7).
8. **Add 10.8 as a row in `development-sequence.md`'s Stage 10 table.**
9. **Add to the charter §5.10**: the `1 − CDF` defect is package-wide in
   `vars` (ten sites in `R/internal.R`), not confined to
   `causality.R`; and `vars` runs the portmanteau test with
   `df ≤ 0` when `lags ≤ p`, where statsmodels correctly refuses.
10. Manual section: a "checking a fitted VAR" walkthrough using
    `var_diagnostics`, with the honest note that the portmanteau test
    needs `lags > p` and that its power depends on the choice.

---

## 10. Bundle additions for this item

```
fixtures/
  ref_R_diag.json      # 7 fixtures x {PT.asymptotic, PT.adjusted, BG, ES, JB multivariate,
                       #   JB skewness, JB kurtosis, per-equation JB, multivariate ARCH-LM,
                       #   per-equation ARCH-LM}, plus vars' own error messages
  ref_py_diag.json     # statsmodels test_whiteness (both adjustments) and test_normality
                       #   for the same 7 fixtures, plus its error messages
scripts/
  ref_R_diag.R         # regenerates ref_R_diag.json
  ref_py_diag.py       # regenerates ref_py_diag.json
```

**Note on the recorded p-values.** `ref_R_diag.json` stores `vars`'
reported p-values verbatim, several of which are `0.000000e+00` through
the `1 − CDF` defect (§4). They are documentation of the defect. Assert
statistics and df against this file; take p-values from
`ref_py_diag.json` where statsmodels implements the test, and otherwise
compute the tail directly in the verification script.
