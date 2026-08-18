# Handoff: Stage 2.2 — KPSS Test (R/Python-competitive)

Status: implemented ✅ (`src/unitroot.jl`, `test/test_unitroot.jl`). This
doc's core claim held up -- the existing math and critical-value table
really were already correct, confirmed by exact-value validation against
real `statsmodels.tsa.stattools.kpss` output. But **one delta from
section 5's proposed code**, the same shape as Stage 2.1's bug: this
doc's `:legacy` formula (`trunc(12*(n/100)^0.25)`, giving `14` for
`n=200`) does not match Python's actual `nlags="legacy"` behavior --
`kpss`'s real source (`statsmodels.tsa.stattools`, read directly, not
its docs) uses `ceil`, giving `15` for `n=200`, plus a `min(nlags, n-1)`
safety cap this doc didn't mention at all. Since the whole point of
`:legacy` is "matches Python's `legacy` exactly," implementing it with
`trunc` as originally proposed would have defeated that goal. Whether
`ceil` is *also* what R's `lshort=FALSE` computes could not be checked
(no R available in this environment) -- the doc's "matches R exactly"
claim for this specific formula is unconfirmed, not denied.

`:short` (the R-only formula, no Python equivalent to cross-check
against) was left as `trunc`/`floor`, unchanged from the pre-existing
build, since nothing contradicts it. Also added, beyond section 5: a
hard `nlags < n` validation for explicit integers (matching `kpss`'s own
validation) and the same `n-1` cap applied to `:short` too, for
robustness.

Validated exactly (atol=1e-5) against real `statsmodels` output across
`regression`×`nlags` combinations, using the same AR(1) reference series
as Stage 2.1/1.3 (`test/fixtures/ar1_ref_series.csv`).

For a fresh Claude Code session picking this up with no prior context.
Same treatment as the Stage 2.1 (ADF) handoff — verify both references'
actual argument names and default behavior, rename for comfort, close any
real functionality gaps found, flag what can't be closed yet rather than
hide it.

## Where this fits

- **Depends on:** Stage 1.4 (`_ols`).
- **Replaces:** `kpss_test(x; regression=:level, lags=:auto)` in
  `unitroot.jl`. Good news up front: **the core math and the
  critical-value table in the existing build are already correct** — this
  is much more a naming/completeness pass than Stage 2.1 was.

---

## 1. Requirements

Same four as Stage 2.1: names should mirror a real reference, support the
union of both languages' capability, document disagreements explicitly
rather than picking one silently, and don't fake support for something
not actually implemented (see section 3's `:auto` discussion — this
requirement is why the default below is *not* simply "match Python").

---

## 2. Verified reference: R `tseries::kpss.test`

```r
kpss.test(x, null = c("Level", "Trend"), lshort = TRUE)
```

- `null`: `"Level"` (default) or `"Trend"`.
- `lshort`: **boolean**, not a formula choice with more options. `TRUE`
  (default) → truncation lag `trunc(4*(n/100)^0.25)`. `FALSE` →
  `trunc(12*(n/100)^0.25)`. R offers no other option — no direct integer
  override, no data-dependent method.
- Newey-West/Bartlett estimator for `sigma^2` — matches what's already
  built.
- p-values: *"interpolated from Table 1 of Kwiatkowski, Phillips, Schmidt
  & Shin (1992)"* — **matches** what's already built. No discrepancy here,
  worth confirming positively rather than assuming a gap exists just
  because ADF had one.

## 3. Verified reference: Python `statsmodels.tsa.stattools.kpss`

```python
kpss(x, regression: Literal["c","ct"]="c", nlags: Literal["auto","legacy"]|int="auto", store=False)
```

- `regression`: `"c"` (default, level-stationarity) / `"ct"`
  (trend-stationarity) — same concept as R's `null`, different name and
  value convention (matches Python's own `adfuller`'s `regression`
  naming, which is a nice internal-to-Python consistency, and one worth
  mirroring in Julia across `adf_test`/`kpss_test` too).
- `nlags`: **three-way**, not R's boolean:
  - `"auto"` (**default**) → *"lags is calculated using the
    data-dependent method of Hobijn et al. (1998)"* — a genuinely
    different algorithm from any fixed formula, not just a different
    constant. Cites Andrews (1991) and Newey & West (1994) alongside it —
    this is the same general family of automatic HAC-bandwidth-selection
    methods, not a simple `n`-based formula.
  - `"legacy"` → `int(12*(n/100)^0.25)` — **this is R's `lshort=FALSE`
    formula**, exactly.
  - explicit `int` → direct override (R has no equivalent to this).
- p-values: same Kwiatkowski et al. (1992) Table 1 as R — confirmed
  matching, not a discrepancy.

**The real finding**: Python's actual default (`nlags="auto"`) is the
Hobijn data-dependent method, which is **not** what the existing Julia
build computes (it computes R's `lshort=TRUE` short formula
unconditionally). This is the same shape of gap as Stage 2.1's
`autolag='AIC'` — a genuinely different default algorithm, not yet
implemented, not something to silently paper over by mapping our formula
to Python's `"auto"` name.

---

## 4. Proposed Julia API

```julia
kpss_test(x; regression::Symbol=:c, nlags::Union{Symbol,Int}=:short)
```

Design notes:
- **`regression`**: matches Python's argument name (chosen over R's
  `null` for the same reason as `adf_test` — and for internal
  consistency, since `adf_test` already uses `regression` for the same
  concept). `:c` (default, matches Python's default and is R's `"Level"`)
  / `:ct` (matches Python's `"ct"`, R's `"Trend"`).
- **`nlags`**: four values, a strict superset of both references:
  - `:short` (**default here** — see below for why this isn't `:auto`) →
    `trunc(4*(n/100)^0.25)`, matching R's actual default (`lshort=TRUE`)
    exactly.
  - `:legacy` → `trunc(12*(n/100)^0.25)`, matching both R's
    `lshort=FALSE` *and* Python's `"legacy"` exactly — this one string
    covers both references at once.
  - `:auto` → **not yet implemented**. Calling it should throw a clear
    `ArgumentError` pointing at this gap, *not* silently fall back to
    `:short` — a silent fallback would misrepresent what was actually
    computed as matching Python's default when it doesn't.
  - explicit `Int` → direct override, matching Python's int option.
- **Why the default is `:short`, not `:auto`**: `:auto` isn't
  implemented, so defaulting to it would mean either an error on default
  use (bad ergonomics) or silently using a different formula while
  claiming the `:auto` name (dishonest). `:short` is fully implemented,
  matches a real reference's actual default (R's), and is clearly
  documented as not matching Python's default — the honest choice given
  what's actually built right now.

### For Python-default-equivalence later

The Hobijn et al. (1998) method, from the exact citation in Python's
docs (*"Generalizations of the KPSS-test for stationarity,"* Statistica
Neerlandica), is the concrete reference to implement `:auto` against
when there's a real need for it — flagged as a known gap rather than
attempted here without enough time to verify it carefully against
`statsmodels`' actual output.

---

## 5. Implementation

The core KPSS math is unchanged from the existing build (already
verified correct) — this is a signature/argument change plus adding the
`:legacy` bandwidth option and the `:auto` guard.

```julia
"""
    kpss_test(x; regression::Symbol=:c, nlags::Union{Symbol,Int}=:short)

KPSS test of the null hypothesis that `x` is (trend-)stationary, against
the alternative of a unit root. Argument names follow Python's
`statsmodels.tsa.stattools.kpss` (`regression`) for consistency with
`adf_test`; see the Stage 2.2 handoff doc for the full comparison against
both `tseries::kpss.test` (R) and `statsmodels.tsa.stattools.kpss`
(Python), including one genuine functionality gap (`nlags=:auto`, see
below) and one confirmed non-discrepancy (both references use the same
Kwiatkowski et al. 1992 critical-value table, already implemented here).

- `regression`: `:c` (constant only, **default** -- matches Python's
  default and R's `null="Level"`) or `:ct` (constant + trend -- matches
  R's `null="Trend"`).
- `nlags`: `:short` (**default here** -- matches R's actual default,
  `lshort=TRUE`: `trunc(4*(n/100)^0.25)`) / `:legacy` (matches R's
  `lshort=FALSE` *and* Python's `"legacy"` exactly:
  `trunc(12*(n/100)^0.25)`) / an explicit `Int`. `:auto` (Python's
  default -- the Hobijn et al. 1998 data-dependent method) is **not yet
  implemented** and throws `ArgumentError` rather than silently
  substituting a different formula under that name.

!!! note "Default mismatch with Python, by necessity, not oversight"
    This function's default (`:short`) matches R's default, not Python's
    (`:auto`/Hobijn, unimplemented). Pass `nlags=:legacy` explicitly if
    you want the one bandwidth choice that's identical across both
    references.
"""
function kpss_test(x; regression::Symbol=:c, nlags::Union{Symbol,Int}=:short)
    regression in (:c, :ct) || throw(ArgumentError("regression must be :c or :ct"))
    y = tsvalues(x)
    n = length(y)

    l = if nlags isa Int
        nlags >= 0 || throw(ArgumentError("nlags must be >= 0"))
        nlags
    elseif nlags === :short
        trunc(Int, 4 * (n/100)^0.25)
    elseif nlags === :legacy
        trunc(Int, 12 * (n/100)^0.25)
    elseif nlags === :auto
        throw(ArgumentError("nlags=:auto (Hobijn et al. 1998 data-dependent method) " *
                             "is not yet implemented -- use :short, :legacy, or an explicit Int. " *
                             "See the Stage 2.2 handoff doc for the reference to implement against."))
    else
        throw(ArgumentError("nlags must be :short, :legacy, :auto, or an Int"))
    end

    X = regression == :ct ? hcat(ones(n), collect(1.0:n)) : reshape(ones(n), n, 1)
    _, resid, _ = _ols(X, y)

    S = cumsum(resid)
    numerator = sum(abs2, S) / n^2

    gamma0 = sum(abs2, resid) / n
    lrv = gamma0
    for k in 1:l
        w = 1 - k / (l + 1)
        gk = dot(view(resid, 1:n-k), view(resid, 1+k:n)) / n
        lrv += 2 * w * gk
    end

    stat = numerator / lrv
    pval = _interp_pvalue_right(stat, _KPSS_CRIT[regression == :ct ? :trend : :level])

    return KPSSTest(stat, pval, regression, l, n)
end
```

`_KPSS_CRIT`, `_interp_pvalue_right`, `dot` usage — all unchanged from the
existing build; only the dict key convention needs a small adapter
(`regression == :ct ? :trend : :level`) since `_KPSS_CRIT` was originally
keyed by `:level`/`:trend`, not `:c`/`:ct` — keep the internal dict keys
as-is, just translate at the lookup site, to avoid an unnecessary
renaming ripple through code that doesn't need to change.

---

## 6. Hand-verified test values

The short/legacy bandwidth formulas are simple enough to verify by direct
computation rather than needing a Python round-trip:

```
n=200: short = trunc(4*(200/100)^0.25) = trunc(4*1.1892) = trunc(4.7568) = 4
n=200: legacy = trunc(12*(200/100)^0.25) = trunc(12*1.1892) = trunc(14.270) = 14
n=1000: short = trunc(4*(1000/100)^0.25) = trunc(4*1.7783) = trunc(7.113) = 7
n=1000: legacy = trunc(12*(1000/100)^0.25) = trunc(12*1.7783) = trunc(21.339) = 21
```

```julia
using Test, Random

@testset "kpss_test bandwidth formulas" begin
    Random.seed!(3)
    y = randn(200)

    r_short = kpss_test(y; nlags=:short)
    @test r_short.lags == 4

    r_legacy = kpss_test(y; nlags=:legacy)
    @test r_legacy.lags == 14

    r_int = kpss_test(y; nlags=10)
    @test r_int.lags == 10

    @test_throws ArgumentError kpss_test(y; nlags=:auto)
    @test_throws ArgumentError kpss_test(y; regression=:bogus)
    @test_throws ArgumentError kpss_test(y; nlags=-1)

    # regression argument rename: :c/:ct must behave like the old :level/:trend
    r_c = kpss_test(y; regression=:c)
    r_ct = kpss_test(y; regression=:ct)
    @test r_c.regression == :c
    @test r_ct.regression == :ct
end
```

Structural checks already validated in the earlier build (stationary
series shouldn't reject; a random walk should) remain valid unchanged —
just update the call sites to the new argument names when porting those
tests over.

---

## 7. What to do with this

1. Rename `kpss_test`'s arguments per section 5; keep the underlying
   Newey-West/critical-value logic as-is (already correct).
2. Add the `:legacy` bandwidth path and the `:auto` guard.
3. Run the bandwidth-formula tests in section 6 — these are exact
   arithmetic checks, not statistical ones, so they should pass
   deterministically on first try if the formula transcription is right.
4. Update `development-sequence.md`'s Stage 2.2 row: note the argument
   rename and the `:auto`/Hobijn gap explicitly.
5. If `:auto` ever becomes worth implementing, the Hobijn et al. (1998)
   paper (via the "Generalizations of the KPSS-test for stationarity"
   citation) is the concrete target — verify against `statsmodels`'
   actual `nlags="auto"` output on a real series before trusting it,
   same discipline as everything else in this project.

**Next in sequence:** Stage 2.3 (Phillips-Perron) — not yet built at all
(unlike 2.1/2.2, which existed and needed revision), so this one starts
from the requirements/reference-review stage with no prior draft to
correct.
