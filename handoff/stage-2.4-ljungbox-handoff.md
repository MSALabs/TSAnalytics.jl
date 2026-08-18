# Handoff: Stage 2.4 — Ljung-Box / Box-Pierce Test (R/Python-competitive)

Status: implemented ✅ (`src/diagnostics.jl`, `test/test_diagnostics.jl`).
The headline finding (Python's vector-lags semantics differ from this
package's, R defaults to Box-Pierce) was independently re-verified, not
just trusted: `acorr_ljungbox(y, lags=[5,10])` on the shared reference
series (`test/fixtures/ar1_ref_series.csv`) confirmed the two-cumulative-
rows behavior directly, and the exact-set `{5,10}` sum this package
actually computes (`lb_stat≈16.64`) was independently derived (via
`statsmodels`' own `acf()` values fed through the exact-set formula by
hand, since `acorr_ljungbox` has no way to produce that number itself)
and confirmed against the Julia implementation to 1e-4.

One deliberate structural deviation from section 5's proposed API: kept
Julia's multiple-dispatch pattern (three small methods -- no-arg,
`Integer`, `AbstractVector`) rather than collapsing into the doc's single
`lags::Union{Nothing,Integer,AbstractVector{<:Integer}}=nothing`
function with internal `if`/`elseif` branching. Behaviorally identical;
this matches the *existing* code's own already-established two-method
structure (which predates this handoff) more closely than introducing a
new style would have.

For a fresh Claude Code session picking this up with no prior context.
Same treatment as 2.1–2.3. **Headline finding: R's `Box.test()` defaults
to the older, weaker `"Box-Pierce"` statistic — not Ljung-Box.** Both
Wikipedia and statsmodels' own docs note Box-Pierce has worse
finite-sample properties. Worth knowing before assuming "R's default" is
the one to match.

## Where this fits

- **Depends on:** Stage 1.3 (`acf`/autocorrelation).
- **Replaces:** `ljungbox_test(x, lags::Integer; fitdf::Integer=0)` /
  the `AbstractVector` overload, in `diagnostics.jl`. The core Ljung-Box
  math is already correct (confirmed in the Stage 1.3 handoff, where its
  output matched `acf(...; qstat=true)` exactly) — this pass adds the
  missing Box-Pierce option, a sensible auto-lag default, and — most
  importantly — **documents a real semantic difference** between this
  package's vector-lags behavior and what a lag list means in Python
  (see section 4).
- **Also used by:** `qs_test`, which depends specifically on the
  "sum over exactly these lags" behavior (not cumulative) — this handoff
  keeps that behavior, just documents it far more clearly than before.

---

## 1. Requirements

Same four as 2.1–2.3, plus a fifth specific to this test: **the
"different lag semantics" behavior must be documented prominently enough
that a Python user passing a lag vector doesn't silently get a different
statistic than they expect.** This isn't a naming-comfort issue, it's a
correctness-of-understanding issue.

---

## 2. Verified reference: R `stats::Box.test`

```r
Box.test(x, lag = 1, type = c("Box-Pierce", "Ljung-Box"), fitdf = 0)
```

- `lag`: **a single integer**, not a vector — R has no way to test an
  arbitrary lag set at all, only "cumulative through lag `h`."
- `type`: `"Box-Pierce"` (**default**) or `"Ljung-Box"`. Formulas,
  confirmed from R's own docs:
  - Box-Pierce: `Q = n * Σ_{k=1}^{h} ρ_k²`
  - Ljung-Box: `Q = n(n+2) * Σ_{k=1}^{h} ρ_k²/(n-k)`
- `fitdf`: degrees of freedom to subtract — **this name already matches
  the existing Julia build exactly**, no change needed here.

Wikipedia, discussing this exact test: *"Simulation studies have shown
that the distribution for the Ljung–Box statistic is closer to a
χ² distribution than is the distribution for the Box–Pierce statistic
for all sample sizes including small ones."* R's default is, by this
evidence, the statistically weaker choice.

**Also worth knowing**: Wikipedia's own cross-reference table lists
Julia's `HypothesisTests.jl` as already implementing both tests. Per this
project's "reference, never port" policy, worth reading as a design/
validation reference (not a dependency) before finalizing this
implementation — particularly for anything about return-value shape or
edge-case handling that isn't obvious from the R/Python docs alone.

## 3. Verified reference: Python `statsmodels.stats.diagnostic.acorr_ljungbox`

```python
acorr_ljungbox(x, lags=None, boxpierce=False, model_df=0, period=None, return_df=True, auto_lag=False)
```

| Arg | Meaning |
|---|---|
| `lags` | `None` (default) → `min(10, nobs // 5)` — **note this default itself changed across statsmodels versions** (older versions used `min(nobs//2-2, 40)`); using the current formula. Int or list also accepted — see section 4 for what a list actually means. |
| `boxpierce` | `False` (**default**) — Ljung-Box is computed either way; Box-Pierce is only *additionally* returned if `True`. Python's *effective* default is Ljung-Box-only, matching the statistically preferred choice, unlike R. |
| `model_df` | same concept as R's `fitdf`, different name. |
| `period` | seasonal-aware default lag count — not implemented here, noted as a possible future addition, not attempted now. |
| `auto_lag` | automatic optimal-lag selection — same shape of feature as ADF's `autolag`; **not implemented here**, flagged rather than faked. |

Docs state directly: *"Ljung-Box and Box-Pierce statistic differ in their
scaling of the autocorrelation function. Ljung-Box test has better
finite-sample properties."* — the same conclusion Wikipedia states about
R's default, from the other reference's own documentation. Two
independent sources agreeing Ljung-Box is the better default is strong
grounds for that being this package's default too.

---

## 4. Critical finding: what a lag **vector** means differs across all three

Verified directly (not just from docs) — `acorr_ljungbox(y, lags=[5,10])`
on real data:
```
      lb_stat       bp_stat
5   76.924095     76.112467
10  82.581350     81.571554
```
Lag 10's statistic is **larger** than lag 5's — because Python reports
**two separate cumulative statistics**, one through lag 5 (`Σρ₁²...ρ₅²`)
and one through lag 10 (`Σρ₁²...ρ₁₀²`), as two rows. It is **not**
"the statistic using exactly lags 5 and 10."

**The existing Julia build's vector behavior computes something
genuinely different**: given `lags=[5,10]`, it sums `ρ₅²` and `ρ₁₀²`
*only* — a single combined statistic over exactly that lag set, with no
contribution from lags 1-4 or 6-9 at all. This is real, intentional, and
what `qs_test` needs (testing *only* the seasonal lags `s` and `2s`,
deliberately excluding everything between). **It is not a bug** — but it
means a Python user who passes a lag vector expecting Python's per-lag
cumulative-row semantics will silently get a different number.

**Resolution: keep the existing behavior, document the difference
loudly.** Rewriting it would break `qs_test`; silently matching Python's
semantics would lose a genuinely useful capability neither reference
offers. The fix here is documentation, not a behavior change.

---

## 5. Proposed Julia API

```julia
ljungbox_test(x, lags::Union{Nothing,Integer,AbstractVector{<:Integer}}=nothing;
              fitdf::Integer=0, boxpierce::Bool=false)
```

Design notes:
- **`lags`**: now has a sensible default (`nothing` → `min(10, n÷5)`,
  matching Python's current default) — the existing build required it as
  a mandatory argument, which is worse ergonomics than either reference.
  - `Integer h`: cumulative through lag `h` — matches R's single-lag call
    and Python's single-int call exactly (already correct in the
    existing build, confirmed by tracing through: `1:h` fed to the
    vector path sums every lag from 1 to h, which *is* the cumulative
    statistic).
  - `AbstractVector`: sum over **exactly** those lags — the Julia-only
    generalization from section 4, now documented prominently rather
    than left implicit.
- **`fitdf`**: unchanged, already matches R exactly.
- **`boxpierce::Bool=false`**: new, matches Python's argument name and
  default exactly. When `true`, **both** statistics are computed and
  returned (matching Python's "compute both, report both" design, which
  is more informative than R's exclusive `type=` choice) — populated via
  the same optional-field extension pattern already used for
  `ACFResult`'s `qstat`/`pvalues` in the Stage 1.3 work.

### `LjungBoxTest` — extended, not replaced

```julia
struct LjungBoxTest <: HypothesisTest
    statistic::Float64      # Ljung-Box, always
    pvalue::Float64
    lags::Vector{Int}
    df::Int
    bp_statistic::Union{Nothing,Float64}   # populated iff boxpierce=true
    bp_pvalue::Union{Nothing,Float64}
end

# keep the old 4-positional-arg call sites working
LjungBoxTest(statistic, pvalue, lags, df) = LjungBoxTest(statistic, pvalue, lags, df, nothing, nothing)
```

---

## 6. Implementation

```julia
"""
    ljungbox_test(x, lags=nothing; fitdf=0, boxpierce=false)

Ljung-Box (and optionally Box-Pierce) portmanteau test of the null
hypothesis that `x` is white noise. Ljung-Box is always the primary
statistic returned -- R's `Box.test()` actually defaults to the older,
weaker Box-Pierce statistic (both Wikipedia and statsmodels' own docs
note Ljung-Box has better finite-sample properties), so this
deliberately does NOT default-match R here; see the Stage 2.4 handoff
doc for the full comparison.

- `lags`: `nothing` (default) computes `min(10, n÷5)`, matching Python's
  current `acorr_ljungbox` default (R's default is a single fixed
  `lag=1`, unusually small in practice). An `Integer` gives the
  cumulative statistic through that lag (matches both references'
  single-lag behavior). An `AbstractVector` sums **exactly** the given
  lags -- **not** the same as passing a list to Python's
  `acorr_ljungbox`, which instead reports one cumulative statistic per
  listed lag. See the handoff doc section 4 for a worked example of the
  difference; this Julia-specific behavior is what `qs_test` depends on.
- `fitdf`: degrees of freedom to subtract (e.g. `p+q` for ARMA(p,q)
  residuals) -- matches R's argument name exactly.
- `boxpierce`: if `true`, also compute the Box-Pierce statistic
  (`Q = n*Σρ_k²`, no `(n-k)` denominator, no `n+2` factor), populating
  `bp_statistic`/`bp_pvalue`. Matches Python's argument name and default
  (`false`) exactly.
"""
function ljungbox_test(x, lags::Union{Nothing,Integer,AbstractVector{<:Integer}}=nothing;
                        fitdf::Integer=0, boxpierce::Bool=false)
    y = tsvalues(x)
    n = length(y)
    ls = if lags === nothing
        1:min(10, n ÷ 5)
    elseif lags isa Integer
        1:lags
    else
        lags
    end

    rho = acf(y, collect(ls); bartlett=false).values  # reuse Stage 1.3's acf; bands unused here
    Q = n * (n + 2) * sum(rho[i]^2 / (n - ls[i]) for i in eachindex(ls))
    df = length(ls) - fitdf
    df > 0 || throw(ArgumentError("degrees of freedom must be positive; reduce fitdf or add lags"))
    pval = _chisq_ccdf(Q, df)

    bp_stat, bp_pval = if boxpierce
        Qbp = n * sum(rho[i]^2 for i in eachindex(ls))
        (Qbp, _chisq_ccdf(Qbp, df))
    else
        (nothing, nothing)
    end

    return LjungBoxTest(Q, pval, collect(ls), df, bp_stat, bp_pval)
end
```

Note: `acf(y, collect(ls); bartlett=false).values` returns autocorrelation
values *at exactly the requested lags* (per Stage 1.3's `acf` design,
which already takes an arbitrary lag vector) — so `rho[i]` corresponds to
`ls[i]`, not to lag `i`. The `sum(... for i in eachindex(ls))` loop
already accounts for this correctly by indexing `rho` and `ls` together,
matching the existing (correct) build's approach.

---

## 7. Hand-verified test values

```python
# From the verification above:
acorr_ljungbox(y, lags=[5,10], boxpierce=True)
#     lb_stat       lb_pvalue    bp_stat      bp_pvalue
# 5   76.924095   3.688773e-15  76.112467   5.449956e-15
# 10  82.581350   1.562991e-13  81.571554   2.468330e-13
```

These are Python's *cumulative* per-lag values — useful as a check that,
if Julia's `Integer`-input path is called separately for `h=5` and
`h=10`, each individually matches the corresponding Python row (since
`Integer` input *is* cumulative in both). **Not** a check for the
vector-input path, which computes something different by design.

```julia
using Test, Random

@testset "ljungbox_test" begin
    Random.seed!(4)
    n = 300
    e = randn(n)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.5*y[t-1] + e[t]
    end

    # Integer input IS cumulative -- structurally verify monotonic growth
    r5 = ljungbox_test(y, 5)
    r10 = ljungbox_test(y, 10)
    @test r10.statistic > r5.statistic   # more terms in the sum -> larger Q

    # boxpierce=true populates both statistics; Box-Pierce and Ljung-Box
    # should be close but not identical (different scaling)
    r_both = ljungbox_test(y, 10; boxpierce=true)
    @test r_both.bp_statistic !== nothing
    @test r_both.statistic != r_both.bp_statistic
    @test r_both.statistic > r_both.bp_statistic  # LB's (n+2)/(n-k) scaling
                                                    # inflates it relative to BP

    # boxpierce=false (default): fields are nothing, not silently zero
    r_default = ljungbox_test(y, 10)
    @test r_default.bp_statistic === nothing

    # Vector input sums EXACTLY those lags -- deliberately NOT equal to
    # the cumulative Integer-input statistic at the max lag
    r_vec = ljungbox_test(y, [5, 10])
    @test r_vec.statistic != r10.statistic

    # default lags formula: min(10, n÷5) = min(10, 60) = 10
    r_auto = ljungbox_test(y)
    @test length(r_auto.lags) == 10

    @test_throws ArgumentError ljungbox_test(y, 5; fitdf=10)  # df <= 0
end
```

---

## 8. What to do with this

1. Add the `lags=nothing` default, the `boxpierce` keyword, and the
   `LjungBoxTest` field extension per sections 5-6.
2. Run the tests in section 7 — the "vector input ≠ cumulative Integer
   input at max lag" assertion is the one that most directly guards
   against accidentally "fixing" the intentional vector-lags behavior
   into matching Python's semantics later.
3. Worth a quick read of `HypothesisTests.jl`'s Ljung-Box/Box-Pierce
   implementation for validated design patterns before finalizing, per
   the "reference, never port" policy — not a dependency, just a check.
4. Update `development-sequence.md`'s Stage 2.4 row: note the R
   default-statistic finding, the `boxpierce` addition, and — most
   importantly — the vector-lags semantic difference, since that's the
   one most likely to confuse someone later if it isn't written down.

**Next in sequence:** Stage 2.5 (QS test) — already built and marked ✅.
Given `qs_test` is really a special case of `ljungbox_test`'s
exact-lag-set behavior (testing exactly `{s, 2s}`), this next handoff
should mostly focus on whether `qs_test` should just become a thin
wrapper calling `ljungbox_test(x, [s, 2s])` instead of duplicating the
chi-square logic — worth checking when we get there.
