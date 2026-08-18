# Handoff: Stage 2.6 — Jarque-Bera Normality Test (built fresh)

Status: implemented ✅ (`src/diagnostics.jl`, `test/test_diagnostics.jl`).
This doc's "no cross-language discrepancy" claim was independently
re-verified against real `statsmodels.stats.stattools.jarque_bera`
(installed in this environment), not just this doc's own hand
transcription -- run on the shared reference series
(`test/fixtures/ar1_ref_series.csv`, not this doc's own test data),
matching to 10 decimal places (`JB=1.6772996836`,
`skew=0.1145669775`, `kurt=2.6142881999`). Implemented exactly as
proposed in section 5, no deltas.

For a fresh Claude Code session picking this up with no prior context.
This is the sixth and final item in Stage 2 (the diagnostics table
originally listed six rows; 2.1–2.5 are done, this closes the stage
out). Unlike most of Stage 2, **the underlying formula genuinely agrees
between R and Python** — no default-behavior archaeology needed this
time, mostly a build-fresh-and-verify pass.

## Where this fits

- **Depends on:** Stage 1.4 (`_ols`, indirectly — this test is commonly
  applied to regression/ARMA residuals, though it works on any vector).
- **New territory:** not built anywhere in earlier work — this stage
  starts from the reference review, same as 2.3 (Phillips-Perron) did.

---

## 1. Verified reference: R `tseries::jarque.bera.test`

```r
jarque.bera.test(x)
```

No other arguments — R's version is minimal. From the package source
(`tseries/R/test.R`, the same file as `adf.test`/`kpss.test`/`pp.test`):
biased (population, `n`-denominator) central moments are computed
directly:
```r
m1 <- sum(x)/n
m2 <- sum((x-m1)^2)/n
m3 <- sum((x-m1)^3)/n
m4 <- sum((x-m1)^4)/n   # (pattern continues, matching the formula below)
```
Result is `"htest"`-classed, degrees of freedom fixed at 2 (chi-squared).

## 2. Verified reference: Python `statsmodels.stats.stattools.jarque_bera`

```python
jarque_bera(resids, axis=0)
```

`axis` is a NumPy-multidimensional-array concern with no Julia
equivalent needed for a 1D vector — not worth carrying into the Julia
signature. Exact source (confirmed, not recalled):

```python
skew = stats.skew(resids, axis=axis)          # scipy's biased skewness estimator
kurtosis = 3 + stats.kurtosis(resids, axis=axis)  # scipy's biased excess kurtosis + 3
n = resids.shape[axis]
jb = (n / 6.) * (skew**2 + (1/4.) * (kurtosis - 3)**2)
jb_pv = stats.chi2.sf(jb, 2)
return jb, jb_pv, skew, kurtosis
```

**Returns more than R does** — `skew` and `kurtosis` themselves, not just
the test statistic and p-value. Genuinely useful diagnostic info, worth
exposing directly in the Julia return type rather than discarding it
after computing it internally.

## 3. The formulas actually agree — verified numerically, not assumed

Transcribed the biased-moment formula in Python and confirmed it
reproduces `statsmodels`' actual output exactly on a heavy-tailed test
series (Student's t, 3 df, n=200, seed=9):
```
statsmodels: JB=59.185671  pv≈0.0  skew=-0.276907  kurt=5.606832
hand transcription: JB=59.185671  skew=-0.276907  kurt=5.606832
```
Both R and Python use the same biased (population, `n`-denominator)
skewness/kurtosis estimators and the same JB formula
`JB = n·(S²/6 + (K-3)²/24)` with 2 degrees of freedom. **No cross-language
discrepancy found this time** — worth stating plainly, per the same
"don't only report problems" principle from the Stage 2.2 (KPSS) handoff.

---

## 4. Proposed Julia API

```julia
jarque_bera_test(x) -> JarqueBeraTest
```

Design notes:
- **No keyword arguments** — R has none; Python's only extra (`axis`) has
  no Julia analogue for a 1D vector. Simplicity here is a feature, not an
  oversight — there's nothing to add.
- **Return type exposes `skewness`/`kurtosis` directly**, matching
  Python's richer return (R only gives the statistic/p-value) — this is
  "matching the union of both languages' capability," the same principle
  used throughout Stage 2, just resolved in Python's favor here since R
  simply computes less.

### `JarqueBeraTest`

```julia
struct JarqueBeraTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    skewness::Float64
    kurtosis::Float64
    n::Int
end
```

---

## 5. Implementation

```julia
"""
    jarque_bera_test(x) -> JarqueBeraTest

Jarque-Bera test of the null hypothesis that `x` is drawn from a normal
distribution, based on sample skewness and kurtosis:

    JB = n * (S^2/6 + (K-3)^2/24)

where `S` and `K` are the *biased* (population, `n`-denominator) skewness
and kurtosis estimators -- confirmed to match both R's
`tseries::jarque.bera.test` (verified from source) and Python's
`statsmodels.stats.stattools.jarque_bera` (verified numerically against
actual `statsmodels` output) exactly; unlike most of this project's other
diagnostic tests, no cross-language discrepancy was found here. `JB` is
asymptotically chi-squared with 2 degrees of freedom under the null.

Returns `skewness` and `kurtosis` directly (matching Python's richer
return value, which R's version doesn't provide) -- useful diagnostic
info on its own, not just an intermediate computation.

`x` accepts anything [`tsvalues`](@ref) does. Commonly applied to
regression or ARMA residuals, but works on any vector.

!!! note "Reliability at small n"
    Both references note the chi-squared approximation is asymptotic;
    SciPy's docs specifically recommend n > 2000 for the test to be
    reliable. Treat results on short series with proportionate caution --
    this isn't unique to the Julia implementation, it's inherent to the
    test.
"""
function jarque_bera_test(x)
    y = tsvalues(x)
    n = length(y)
    n >= 2 || throw(ArgumentError("jarque_bera_test: need at least 2 observations"))

    m1 = sum(y) / n
    c = y .- m1
    m2 = sum(abs2, c) / n
    m3 = sum(c.^3) / n
    m4 = sum(c.^4) / n

    skewness = m3 / m2^1.5
    kurtosis = m4 / m2^2

    JB = (n / 6) * (skewness^2 + (kurtosis - 3)^2 / 4)
    pval = _chisq_ccdf(JB, 2)

    return JarqueBeraTest(JB, pval, skewness, kurtosis, n)
end
```

`_chisq_ccdf` — the same shared chi-squared tail helper already used by
`ljungbox_test`/`qs_test`, reused rather than reimplemented.

---

## 6. Hand-verified test values

From the verification transcript above (exact, not approximate — this
formula reproduces `statsmodels` to full floating-point precision, unlike
the interpolated-p-value tests elsewhere in this project):

```julia
using Test, Random

@testset "jarque_bera_test" begin
    Random.seed!(1)
    # Normal data: should NOT reject normality
    y_normal = randn(2000)
    r_normal = jarque_bera_test(y_normal)
    @test r_normal.pvalue > 0.05

    # Strongly skewed/heavy-tailed data (squared normal ~ chi-sq(1)):
    # should reject normality
    y_skewed = randn(2000).^2
    r_skewed = jarque_bera_test(y_skewed)
    @test r_skewed.pvalue < 0.01
    @test r_skewed.skewness > 1.0   # chi-sq(1) has skewness = sqrt(8) ~= 2.83

    # returned n matches input length
    @test r_normal.n == 2000

    @test_throws ArgumentError jarque_bera_test([1.0])  # n < 2
end
```

**For exact cross-language validation** (this test is unusually easy to
validate exactly, unlike the p-value-interpolated tests elsewhere):
pipe identical data through Julia and `statsmodels.stats.stattools.
jarque_bera` and compare `statistic`/`skewness`/`kurtosis` directly —
given the formula match confirmed in section 3, this should agree to
near machine precision, not just approximately.

---

## 7. What to do with this

1. Implement `jarque_bera_test` per section 5 — this is a short,
   self-contained addition; no dependency on anything not already built.
2. Run the tests in section 6.
3. Given how exactly the formula matches both references, this is a good
   candidate for the exact-cross-language CSV-based validation mentioned
   throughout this project's other handoffs — worth actually doing here
   specifically because the payoff (near-machine-precision agreement) is
   higher than for the p-value-interpolated tests where exact matching
   isn't expected anyway.
4. Update `development-sequence.md`'s Stage 2.6 row: mark implemented,
   note that this was the one Stage 2 test with **no** cross-language
   discrepancy found — worth keeping that fact visible rather than losing
   it among the other five stages' bug reports.

---

## Stage 2 is now genuinely complete (all six rows)

Updated summary table:

| Stage | Nature of the finding |
|---|---|
| 2.1 ADF | R itself is internally inconsistent (two packages, two different capabilities); a real missing feature (autolag search) |
| 2.2 KPSS | Mostly already correct; one real gap (Python's `:auto` is a different algorithm, not a formula) |
| 2.3 PP | Built fresh; formula-verified against real `arch` package source to ~1e-13 before writing Julia |
| 2.4 Ljung-Box | R's *default* is the statistically weaker test; a subtle vector-semantics difference worth documenting loudly |
| 2.5 QS | A genuine correctness bug (missing clipping), found only by checking the actual defining reference |
| 2.6 Jarque-Bera | Built fresh; **the one case where R and Python formulas agree exactly**, verified to full precision |

Six stages, five with real substantive findings and one clean pass — a
reasonable batting average for "assume there's something to find,
verify rather than trust" as a working method. Worth carrying the same
discipline into Stage 3.
