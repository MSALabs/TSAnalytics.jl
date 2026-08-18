# Handoff: Stage 2.1 — Augmented Dickey-Fuller Test (R/Python-competitive)

Status: implemented ✅ (`src/unitroot.jl`, `test/test_unitroot.jl`) — but
**not as a drop-in of section 5's code**, which this doc itself flagged
as an incomplete draft (a non-functional `:tstat` stub, no real `:ctt`
critical-value table, "fiddly" indexing asked to be re-derived). All
three were addressed properly, plus two bugs found beyond what this doc
caught, both verified directly from `adfuller`'s actual source (available
locally via `python`, not just its docs):

1. **`maxlag`'s default formula is `ceil`, not `floor`.** This doc (and
   the pre-existing Julia build) used `floor(12*(n/100)^0.25)`; the real
   source uses `ceil`, plus a sample-size cap
   (`min(n÷2 - ntrend - 1, maxlag)`) neither this doc nor the old code
   had at all. For `n=200` this is the difference between `maxlag=14`
   and the actual `maxlag=15`.
2. **The `:tstat` significance threshold is `norm.ppf(0.95) ≈ 1.6449`
   (one-sided), not `1.96`** (two-sided) — confirmed from `_autolag`'s
   source, not guessed.
3. **AIC/BIC/t-stat search candidates must share one fixed sample size**
   (trimmed to `maxlag`, not each candidate's own natural `p`-dependent
   size) for the criteria to be comparable at all — this doc's `fit_at`
   didn't do this; the implemented version fits every search candidate
   at a common `window=maxp`, then re-fits only the final chosen lag at
   its own natural size, matching `adfuller`'s explicit re-run exactly.
4. `:ctt`'s critical values are extracted directly from
   `statsmodels.tsa.adfvalues.mackinnoncrit(regression="ctt")`, not
   approximated by falling back to `:trend`'s table.

Validated exactly (atol=1e-4) against real `statsmodels.tsa.stattools.
adfuller` output across all 12 `regression`×`autolag` combinations
(`:n`/`:c`/`:ct`/`:ctt` × `:aic`/`:bic`/`:tstat`), matching on the test
statistic, chosen lag (`usedlag`), and `nobs` simultaneously — not just
the statistic alone, per this doc's own section 7 recommendation. Uses
the same AR(1) reference series as `handoff/stage-1.3-acf-pacf-handoff.md`
(`test/fixtures/ar1_ref_series.csv`), so the two stages' exact-value
tests share one fixture rather than each inventing their own. The
`ADFTest.deterministic` struct field was also renamed to `.regression`
for consistency with the new argument name (a deliberate breaking change,
acceptable pre-release).

For a fresh Claude Code session picking this up with no prior context.
Covers replacing the existing `adf_test` (built earlier without this level
of verification) with one whose **argument names and default behavior**
are genuinely comfortable for an R or Python user to pick up, not just
mathematically equivalent under different names.

## Where this fits

- **Depends on:** Stage 1.1 (differencing), Stage 1.4 (`_ols`, now with
  `:qr`/`:cholesky`).
- **Replaces:** `adf_test(x; deterministic=:constant, lags=:auto)` in
  `unitroot.jl` — the existing argument names (`deterministic`, `lags`)
  don't actually match either R or Python; they were invented names
  loosely modeling Python's `regression`/`maxlag` concepts. This handoff
  corrects that, and closes a real functionality gap found along the way
  (see section 3).

---

## 1. Requirements

1. Argument names should mirror a reference implementation's actual
   names, not an invented equivalent — an R or Python user reading the
   Julia signature should recognize it immediately.
2. Must support the **union** of what R and Python each offer (see
   section 3 for why neither alone is "the" reference here), not just
   whichever was found first.
3. Default behavior should be documented against **both** references
   explicitly, including where they disagree — silently picking one
   default without flagging the disagreement risks someone assuming
   Julia matches both when it can only match one at a time.
4. `:aic`/`:bic` automatic lag-length selection (Python's actual default
   behavior) is in scope, not deferred — it's a real, commonly-relied-on
   feature, not a nice-to-have.

---

## 2. Verified reference: R — two different packages, worth knowing both

**`tseries::adf.test`** — the "obvious" R reference, but more rigid than
expected. Verified from actual source (`tseries/R/test.R`):

```r
adf.test(x, alternative = c("stationary", "explosive"),
         k = trunc((length(x)-1)^(1/3)))
```

The source shows the regression is **always**:
```r
lm(yt ~ xt1 + 1 + tt + yt1)   # always both a constant (1) AND a trend (tt)
```
**There is no way to request "no constant" or "constant only, no trend"
from `tseries::adf.test` at all.** It always fits constant+trend,
regardless of what you might expect from statsmodels' `regression=`
flexibility.

- `k`: number of augmenting lags, default `trunc((length(x)-1)^(1/3))` —
  the Said-Dickey/Banerjee et al. (1993) cube-root rule, **not** the
  Schwert (1989) rule the earlier build actually implemented.
- `alternative`: `"stationary"` (default) or `"explosive"` — from the
  source, this does **not** change the regression or the test statistic
  at all; it only affects how the conclusion is worded/interpreted.
  Effectively cosmetic for computational purposes.
- p-values: *"interpolated from Table 4.2, p. 103 of Banerjee, Dolado,
  Galbraith & Hendry (1993)"* — a **different** critical-value table than
  MacKinnon's, which is what the existing build actually approximates.
  Since `tseries::adf.test` only ever fits constant+trend, it only ever
  needs one table; MacKinnon's tables (used by both the existing build
  and by statsmodels) cover all four regression types.

**`fUnitRoots::adfTest`** — the R package that actually has the
type-flexibility resembling what the earlier build assumed R offered:
```r
adfTest(x, lags = 0, type = c("nc", "c", "ct"))
```
`type`: `"nc"` (no constant) / `"c"` (constant only, default) / `"ct"`
(constant + trend) — three options, not Python's four (`ctt` — quadratic
trend — has no equivalent in either R package).

**Practical implication:** there is no single "the R behavior" for this
test — `tseries::adf.test` and `fUnitRoots::adfTest` genuinely disagree
on whether type-flexibility exists at all. Document both rather than
picking one and calling it "the R default."

## 3. Verified reference: Python `statsmodels.tsa.stattools.adfuller`

```python
adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)
```

| Arg | Meaning |
|---|---|
| `maxlag` | default `12*(nobs/100)^(1/4)` (Schwert 1989) — but see `autolag` below, this is a **ceiling**, not necessarily the lag actually used |
| `regression` | `'c'` (constant only, **default**) / `'ct'` (constant+trend) / `'ctt'` (constant+linear+quadratic trend) / `'n'` (neither) — four options |
| `autolag` | `'AIC'` (**default**) / `'BIC'` / `'t-stat'` / `None`. If `'AIC'` or `'BIC'`: search lags `0..maxlag`, pick the one minimizing that criterion. If `'t-stat'`: start at `maxlag`, drop a lag at a time until the last lag's t-statistic is significant at 5%. If `None`: just use `maxlag` lags directly, no search. |
| `store`/`regresults` | return extra diagnostic detail — lower priority |

**The critical finding**: Python's actual default (`autolag='AIC'`) does
a genuine **search** over candidate lag orders and picks the
AIC-minimizing one. **The earlier Julia build does not do this at all**
— it computes the Schwert formula once and uses that value directly as a
fixed lag count, which is actually equivalent to Python's `autolag=None,
maxlag=<schwert formula>` — a real, non-default Python behavior, not
Python's actual default. This is a genuine functionality gap, not just a
naming one.

---

## 4. Proposed Julia API

```julia
adf_test(x; regression::Symbol=:c, maxlag::Union{Nothing,Int}=nothing,
          autolag::Union{Nothing,Symbol}=:aic)
```

Design notes:
- **`regression`**: matches Python's argument name exactly (not R's —
  neither R package's name is a clean match: `tseries` has no such
  argument, `fUnitRoots` calls it `type`). Four options matching Python's
  exactly: `:c` (default), `:ct`, `:ctt`, `:n`. This is a strict superset
  of both R packages' capability (`tseries` offers none of this
  flexibility; `fUnitRoots` offers 3 of the 4 options, missing `:ctt`).
- **`maxlag`**: matches Python's name exactly (not R's `k` — but
  documented as the R-equivalent argument). Default `nothing` →
  internally computes the Schwert formula as the ceiling, matching
  Python's default computation.
- **`autolag`**: matches Python's concept and default (`:aic`), with
  Julia symbols instead of strings: `:aic` (default) / `:bic` / `:tstat`
  / `nothing` (use `maxlag` directly, no search — **this is how to
  replicate `tseries::adf.test`'s behavior**: pass
  `regression=:ct, autolag=nothing, maxlag=<cube-root formula>`
  explicitly, since that's `tseries`'s fixed behavior).
- **No `alternative` argument** — per section 2, it's cosmetic in R and
  doesn't change any computed value; not worth adding surface area for a
  labeling-only option. Document this explicitly so it isn't seen as a
  silently-dropped feature.
- **Docstring should state both references' defaults explicitly**,
  including that they disagree on regression-type flexibility and on the
  default lag formula, per requirement 3.

### Helper for exact R (`tseries`) replication

Worth a documented one-liner, since "how do I get R's exact behavior" is
a reasonable question:
```julia
# Replicates tseries::adf.test(x) exactly:
n = length(tsvalues(x))
adf_test(x; regression=:ct, autolag=nothing, maxlag=trunc(Int, (n-1)^(1/3)))
```

---

## 5. Implementation

Extends the existing `adf_test`/`_ols` machinery — the regression-building
logic mostly already exists (the earlier build's `:none`/`:constant`/
`:trend` cases map to `:n`/`:c`/`:ct`; `:ctt` is new), the new work is the
`autolag` search loop.

```julia
"""
    adf_test(x; regression::Symbol=:c, maxlag=nothing, autolag=:aic)

Augmented Dickey-Fuller test of the null hypothesis that `x` has a unit
root. Argument names and defaults follow Python's
`statsmodels.tsa.stattools.adfuller` (the more flexible of the two common
references) -- see the Stage 2.1 handoff doc for a full comparison
against both `tseries::adf.test` and `fUnitRoots::adfTest` in R, which
disagree with each other (and with Python) on regression-type
flexibility and on the default lag-count formula.

- `regression`: `:n` (no constant, no trend) / `:c` (constant only,
  **default**, matches Python's default) / `:ct` (constant + trend) /
  `:ctt` (constant + linear + quadratic trend). `tseries::adf.test`
  offers none of this flexibility (always constant+trend);
  `fUnitRoots::adfTest` offers `:n`/`:c`/`:ct` under the name `type`, not
  `:ctt`.
- `maxlag`: ceiling on the number of augmenting lags. `nothing` (default)
  computes the Schwert (1989) rule `floor(12*(n/100)^0.25)`, matching
  Python's default. R's `tseries::adf.test` uses a **different** default
  formula (`trunc((n-1)^(1/3))`, Said-Dickey/Banerjee et al. 1993) --
  pass `maxlag` explicitly with that formula for exact R replication.
- `autolag`: `:aic` (default, matches Python's default) / `:bic` /
  `:tstat` / `nothing` (use `maxlag` lags directly, no search -- this is
  what `tseries::adf.test` effectively always does).

!!! note "p-value accuracy"
    As before: approximate, via linear interpolation among MacKinnon's
    asymptotic critical values -- not the finite-sample response-surface
    p-value, and not Banerjee et al.'s table (which is what
    `tseries::adf.test` specifically uses for its one fixed regression
    type). Tracked as a known limitation.
"""
function adf_test(x; regression::Symbol=:c, maxlag::Union{Nothing,Int}=nothing,
                   autolag::Union{Nothing,Symbol}=:aic)
    regression in (:n, :c, :ct, :ctt) ||
        throw(ArgumentError("regression must be :n, :c, :ct, or :ctt"))
    autolag === nothing || autolag in (:aic, :bic, :tstat) ||
        throw(ArgumentError("autolag must be :aic, :bic, :tstat, or nothing"))
    y = tsvalues(x)
    n0 = length(y)
    maxp = maxlag === nothing ? floor(Int, 12 * (n0/100)^0.25) : maxlag
    maxp >= 0 || throw(ArgumentError("maxlag must be >= 0"))

    # Build the ADF regression at a given lag order p; returns
    # (tstat_on_gamma, aic, bic, tstat_on_last_lag_coeff)
    function fit_at(p::Int)
        dy = diff(y)
        nobs = length(dy) - p
        resp = dy[(p+1):end]
        ylag = y[(p+1):(n0-1)]
        cols = Vector{Vector{Float64}}()
        push!(cols, ylag)
        for i in 1:p
            push!(cols, dy[(p+1-i):(end-i)])
        end
        if regression in (:c, :ct, :ctt)
            pushfirst!(cols, ones(nobs))
        end
        if regression in (:ct, :ctt)
            push!(cols, collect(1.0:nobs))
        end
        if regression == :ctt
            push!(cols, collect(1.0:nobs).^2)
        end
        X = reduce(hcat, cols)
        beta, resid, se = _ols(X, resp)
        k = size(X, 2)
        rss = sum(abs2, resid)
        aic = nobs*log(rss/nobs) + 2*k
        bic = nobs*log(rss/nobs) + k*log(nobs)
        gamma_idx = regression == :n ? 1 : 2
        tstat = beta[gamma_idx] / se[gamma_idx]
        tstat_last_lag = p > 0 ? beta[end - (regression in (:ct,:ctt) ? (regression==:ctt ? 2 : 1) : 0)] / se[end - (regression in (:ct,:ctt) ? (regression==:ctt ? 2 : 1) : 0)] : NaN
        return (tstat=tstat, aic=aic, bic=bic, nobs=nobs, p=p)
    end

    chosen_p = if autolag === nothing
        maxp
    elseif autolag in (:aic, :bic)
        results = [fit_at(p) for p in 0:maxp]
        crit = autolag == :aic ? getfield.(results, :aic) : getfield.(results, :bic)
        results[argmin(crit)].p
    else # :tstat
        p = maxp
        while p > 0
            # significance of the LAST lag's own coefficient at 5%
            # (simplified: stop when p=0 reached if never significant)
            r = fit_at(p)
            # NOTE: full t-stat-on-last-lag implementation deferred --
            # see "what to verify" below; using maxp directly here is a
            # placeholder that should be replaced with the real
            # drop-one-at-a-time loop before relying on :tstat.
            break
        end
        p
    end

    final = fit_at(chosen_p)
    pval = _interp_pvalue_left(final.tstat, _ADF_CRIT[regression == :n ? :none : regression == :c ? :constant : :trend])
    # NOTE: :ctt has no separate critical-value table in the existing
    # _ADF_CRIT dict -- falls back to :trend's table as an approximation.
    # Flag this explicitly; a proper :ctt table is a follow-up.

    return ADFTest(final.tstat, pval, chosen_p, regression, final.nobs)
end
```

**Known incomplete pieces in this draft, flagged rather than hidden:**

1. **`:tstat` autolag is a stub**, not a real implementation — the actual
   algorithm (start at `maxlag`, test significance of the last lag's
   coefficient, drop and retry) needs the coefficient's own t-stat at
   each candidate `p`, which `fit_at` computes but doesn't return in a
   convenient form yet. Finish this before relying on `autolag=:tstat`.
2. **`:ctt` has no dedicated critical-value table** in the existing
   `_ADF_CRIT` dict (which only has `:none`/`:constant`/`:trend` from the
   earlier build). Needs a fourth entry with MacKinnon's asymptotic
   critical values for the constant+quadratic-trend case before `:ctt`
   p-values are trustworthy — currently falls back to the `:trend` table,
   which is wrong, just not obviously wrong.
3. The `gamma_idx`/`tstat_last_lag` indexing is fiddly (column order
   shifts depending on which of constant/trend/trend² are present) —
   worth a careful re-derivation with a concrete worked example (like the
   original `_ols` column-alignment check) rather than trusting the
   indexing arithmetic above blind.

---

## 6. Hand-verified test values

AIC-search behavior verified against `statsmodels` directly:

```python
from statsmodels.tsa.stattools import adfuller
import numpy as np
np.random.seed(0)
n = 200
e = np.random.randn(n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.6*y[t-1] + e[t]

stat, pval, usedlag, nobs, crit, icbest = adfuller(y, regression='c', autolag='AIC')
print(stat, usedlag)
```
```
-8.79... , usedlag = 0
```
(AIC search on this particular AR(1) series selects 0 augmenting lags —
worth confirming the Julia port selects the same `usedlag` on the same
data, which is a stronger check than just the test statistic matching,
since it validates the search loop itself, not just the regression math.)

Fixed-lag structural checks (already validated in the earlier `_ols`
work, still applicable):
```julia
@testset "adf_test regression types" begin
    Random.seed!(2)
    n = 1000
    e = randn(n)
    ar1 = zeros(n)
    for t in 2:n
        ar1[t] = 0.5*ar1[t-1] + e[t]
    end

    # :ctt should run without erroring and produce a finite statistic
    r = adf_test(ar1; regression=:ctt, autolag=nothing, maxlag=4)
    @test isfinite(r.statistic)

    # autolag=nothing with explicit maxlag should match the old
    # fixed-lag behavior exactly
    r_fixed = adf_test(ar1; regression=:c, autolag=nothing, maxlag=4)
    @test r_fixed.lags == 4

    # invalid regression/autolag
    @test_throws ArgumentError adf_test(ar1; regression=:bogus)
    @test_throws ArgumentError adf_test(ar1; autolag=:bogus)
end
```

---

## 7. What to do with this

1. Finish the two flagged incomplete pieces (section 5, items 1-2) before
   trusting `:tstat` or `:ctt` output.
2. Carefully re-derive the column-indexing arithmetic (item 3) with a
   worked example, the way the original `_ols` handoff did for the
   lagged-difference columns — this is the most bug-prone part.
3. Run the AIC-search check against the exact `statsmodels` output above
   — matching `usedlag` is a stronger validation than matching the
   statistic alone.
4. Update `development-sequence.md`'s Stage 2.1 row: note the argument
   rename, the new `autolag` capability, and the two explicitly-flagged
   incomplete pieces so they don't get silently forgotten.

**Next in sequence:** Stage 2.2 (KPSS) — same treatment, to follow.
