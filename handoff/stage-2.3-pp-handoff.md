# Handoff: Stage 2.3 — Phillips-Perron Test (built fresh, formula-verified)

Status: implemented ✅ (`src/unitroot.jl`, `test/test_unitroot.jl`). This
doc's unusually strong verification claim (~1e-13 agreement with real
`arch` output) was independently re-checked, not just trusted: `arch`
5.1.0 was installed in this environment, and `PhillipsPerron` was run
directly on the same reference series this doc used
(`test/fixtures/ar1_ref_series.csv`, shared with Stages 1.3/2.1/2.2) --
the `trend='c'`, `test_type='tau'`/`'rho'` values matched this doc's
stated numbers (`-6.072989823595` / `-64.606886311833`) exactly, and all
six `trend`×`test_type` combinations (not just the one this doc checked)
were validated against real `arch` before being written into the test
suite.

Two deltas from section 6's proposed code:

1. **Returns a proper `PPTest <: HypothesisTest` struct**, not the bare
   `NamedTuple` this doc proposed -- every other test in this package
   (`ADFTest`, `KPSSTest`, `LjungBoxTest`, `QSTest`) follows that
   convention (`statistic`/`pvalue`/`Base.show`, per `abstract.jl`'s
   `HypothesisTest` docstring), and a `NamedTuple` return would have been
   the only exception.
2. **A real bug**, not a design choice: `cols = [rhs_y]` infers its
   element type from the first pushed column rather than being declared
   `Vector{Vector{Float64}}` up front (unlike `adf_test`'s already-fixed
   pattern from Stage 2.1). This crashes on `push!` for any
   container-agnostic input whose native type isn't already
   `Vector{Float64}` -- caught by this session's own container-agnostic
   test with an integer range (`pp_test(1:100)`), not by anything in this
   doc's own test suggestions. Fixed by matching `adf_test`'s pattern.

Also note: the `_ADF_CRIT` translation this doc's code proposed
(`trend == :n ? :none : ...`) was written against the *pre-Stage-2.1*
critical-value table key scheme. Since Stage 2.1 renamed those keys to
`:n`/`:c`/`:ct`/`:ctt` (matching `regression`'s own values), and `trend`
here uses the identical three-value subset, `_ADF_CRIT[trend]` works
directly with no translation needed at all.

For a fresh Claude Code session picking this up with no prior context.
Unlike 2.1/2.2, this test doesn't exist in any earlier draft — starts
from the reference review. The formula is dense enough to get subtly
wrong (normalization conventions, a correction-term coefficient), so
**the exact algorithm below was verified by installing the real `arch`
Python package, extracting its source, transcribing it, and confirming
the transcription reproduces `arch`'s actual output to machine precision
(~1e-13)** before any Julia was written. This is a stronger guarantee
than the hand-verification used elsewhere in this project — worth
knowing the difference when judging how much to trust this one.

## Where this fits

- **Depends on:** Stage 1.4 (`_ols`).
- **New territory:** PP is algorithmically different from ADF, not just a
  variant — it fits a plain AR(1)-plus-trend regression (no augmenting
  lagged-difference terms at all) and corrects the resulting statistic
  using a Newey-West long-run-variance adjustment, rather than adding
  lags to the regression itself. Confirmed directly from `arch`'s own
  docstring: *"Unlike the ADF test, the regression estimated includes
  only one lag of the dependent variable, in addition to trend terms."*

---

## 1. Requirements

Same as 2.1/2.2: reference-matched argument names, union of both
languages' capability, explicit disagreement documentation, no faked
support for anything not actually implemented.

---

## 2. Verified reference: R — again, two different functions

**`tseries::pp.test`** (the fuller-featured one):
```r
pp.test(x, alternative = c("stationary", "explosive"),
        type = c("Z(alpha)", "Z(t_alpha)"), lshort = TRUE)
```
- Always fits constant + trend — **same rigidity as `tseries::adf.test`**,
  no way to request a no-trend or constant-only regression.
- `type`: `"Z(alpha)"` (**default**) or `"Z(t_alpha)"` — two statistic
  variants. `Z(alpha)` is the coefficient-based statistic (matches
  Python's `test_type="rho"`); `Z(t_alpha)` is the t-stat-based one
  (matches Python's `test_type="tau"`).
- `lshort`: `TRUE` (default) → `trunc(4*(n/100)^0.25)`, `FALSE` →
  `trunc(12*(n/100)^0.25)` — same Newey-West bandwidth convention as
  `tseries::kpss.test`.
- p-values: *"interpolated from Table 4.1 and 4.2, p. 103 of Banerjee et
  al. (1993)"* — same table family as `tseries::adf.test`, **not**
  MacKinnon's tables.

**`stats::PP.test`** (base R, capital PP — a second, more rigid function,
mirroring the `adf.test`/`fUnitRoots::adfTest` split from Stage 2.1):
```r
PP.test(x, lshort = TRUE)
```
Always constant+trend, always the t-stat-based variant (no `type`
choice at all), no `alternative` argument either.

## 3. Verified reference: Python `arch.unitroot.PhillipsPerron`

```python
PhillipsPerron(y, lags=None, trend='c', test_type='tau')
```

| Arg | Meaning |
|---|---|
| `lags` | Newey-West truncation lag. `None` (default) → `ceil(12*(nobs/100)^0.25)` — **the same "long" formula as R's `lshort=FALSE`**, not R's default `lshort=TRUE` short formula. |
| `trend` | `'n'` (none) / `'c'` (constant, **default**) / `'ct'` (constant+trend) — real flexibility, unlike `tseries::pp.test`'s rigidity. No `'ctt'` option (unlike `adfuller`). |
| `test_type` | `'tau'` (t-stat based, **default**) / `'rho'` (coefficient based). **Python defaults to the opposite variant from R** — R's `tseries::pp.test` defaults to `Z(alpha)` (≈ `'rho'`), Python defaults to `'tau'` (≈ `Z(t_alpha)`). |

p-values: MacKinnon (1994/2010) surface approximation — same family as
`adfuller`, **different** from `tseries::pp.test`'s Banerjee et al. table.

**Three real discrepancies found, all in defaults, none in the underlying
math:**
1. Default lag formula: Python matches R's *non-default* `lshort=FALSE`.
2. Default statistic type: Python (`tau`) and R (`Z(alpha)`≈`rho`) default
   to *opposite* variants.
3. Default critical-value table: Banerjee et al. (R) vs. MacKinnon
   (Python) — same split as ADF.

## 4. The verified algorithm (from `arch`'s actual source, not derived)

Regression (no augmenting lags — this is the key structural difference
from ADF):
```
y_t = ρ·y_{t-1} + [μ] + [δ·t] + u_t     (OLS, plain, non-HAC standard errors)
```
Let `n` = number of regression observations (`length(y)-1`), `k` = number
of regressors (1 + trend terms), `u` = OLS residuals, `ρ̂` and `σ_ρ̂` =
coefficient and its ordinary (non-HAC) OLS standard error.

```
s²     = (u'u) / (n - k)                          # df-corrected residual variance
γ₀     = (u'u) / n                                 # biased (n-denominator) residual variance
λ²     = [Σuₜ² + 2·Σⱼ₌₁ˡ (1 - j/(l+1))·Σₜ uₜ·uₜ₋ⱼ] / n     # Newey-West long-run variance
                                                              # (identical Bartlett-kernel formula
                                                              # already used in kpss_test's lrv)
Z(tau) = sqrt(γ₀/λ²)·((ρ̂-1)/σ_ρ̂) - 0.5·((λ²-γ₀)/λ)·(n·σ_ρ̂/s)
Z(rho) = n·(ρ̂-1) - 0.5·(n²·σ_ρ̂²/s²)·(λ²-γ₀)
```

**Verified**: this exact formula, transcribed to Python and run against
`n=200` AR(1) data (φ=0.6, seed=0), `lags=4`, `trend='c'`, reproduced
`arch`'s own `PhillipsPerron` output to `~1e-13` for both `tau` and `rho`
— see section 6 for the full verification script.

---

## 5. Proposed Julia API

```julia
pp_test(x; trend::Symbol=:c, test_type::Symbol=:tau, lags::Union{Nothing,Int}=nothing)
```

Design notes:
- **`trend`**: matches Python's argument name and values (`:n`/`:c`/`:ct`)
  — chosen over R's rigid, option-less `tseries::pp.test` for the same
  reason as `adf_test`/`kpss_test`'s `regression` argument. Note: named
  `trend` here (matching `arch`'s exact name), not `regression`
  (`adfuller`'s name) — the two Python packages themselves use different
  names for the same concept; picking `arch`'s name for *this* function
  keeps it recognizable to `arch` users specifically, at the cost of
  perfect naming consistency with `adf_test` internally. Worth a
  docstring cross-reference so this inconsistency is understood as
  deliberate, not an oversight.
- **`test_type`**: `:tau` (**default**, matches `arch`'s default) /
  `:rho`. R's default is the opposite (`Z(alpha)`≈`:rho`) — documented
  explicitly, not silently chosen around.
- **`lags`**: `nothing` (default) → `ceil(12*(n/100)^0.25)`, matching
  `arch`'s default (which is R's *non-default* long formula). Pass an
  explicit `Int` for R's short-formula default
  (`trunc(4*(n/100)^0.25)`) or any other choice.
- **No `alternative` argument** — same reasoning as `adf_test`: cosmetic
  in R, doesn't change any computed value.

---

## 6. Implementation

```julia
"""
    pp_test(x; trend::Symbol=:c, test_type::Symbol=:tau, lags=nothing)

Phillips-Perron test of the null hypothesis that `x` has a unit root.
Unlike `adf_test`, the underlying regression has no augmenting lagged
difference terms -- serial correlation is instead corrected via a
Newey-West long-run variance adjustment to the test statistic. Argument
names follow `arch.unitroot.PhillipsPerron` (Python) since R's
`tseries::pp.test` offers no comparable flexibility (always fits
constant+trend); see the Stage 2.3 handoff doc for the full comparison,
including where R and Python defaults genuinely disagree (lag formula,
statistic type, and critical-value table -- three separate
discrepancies, not one).

This implementation was verified by transcribing `arch`'s own source
formula and confirming it reproduces `arch`'s actual numerical output to
~1e-13 before being ported to Julia -- see the handoff doc for the
verification script.

- `trend`: `:n` (no constant, no trend) / `:c` (constant only,
  **default**) / `:ct` (constant + trend).
- `test_type`: `:tau` (t-stat based, **default**, matches `arch`'s
  default -- R's `tseries::pp.test` defaults to the *other* variant,
  `Z(alpha)`≈`:rho`) / `:rho` (coefficient based, matches R's default).
- `lags`: Newey-West truncation lag. `nothing` (default) computes
  `ceil(12*(n/100)^0.25)`, matching `arch`'s default (itself R's
  *non-default* `lshort=FALSE` formula -- R's actual default uses
  `trunc(4*(n/100)^0.25)` instead; pass that explicitly for R-default
  equivalence).

!!! note "p-value accuracy"
    Same caveat as `adf_test`: approximate, via linear interpolation
    among MacKinnon's asymptotic critical values -- not the finite-sample
    response-surface p-value, and not Banerjee et al.'s table (which is
    what `tseries::pp.test` specifically uses).
"""
function pp_test(x; trend::Symbol=:c, test_type::Symbol=:tau, lags::Union{Nothing,Int}=nothing)
    trend in (:n, :c, :ct) || throw(ArgumentError("trend must be :n, :c, or :ct"))
    test_type in (:tau, :rho) || throw(ArgumentError("test_type must be :tau or :rho"))
    y = tsvalues(x)
    n_full = length(y)
    l = lags === nothing ? ceil(Int, 12 * (n_full/100)^0.25) : lags
    l >= 0 || throw(ArgumentError("lags must be >= 0"))

    lhs = y[2:end]
    rhs_y = y[1:end-1]
    nobs = length(lhs)

    cols = [rhs_y]
    trend in (:c, :ct) && push!(cols, ones(nobs))
    trend == :ct && push!(cols, collect(1.0:nobs))
    X = reduce(hcat, cols)
    k = size(X, 2)

    beta, u, se = _ols(X, lhs)   # existing helper -- gives beta, residuals, and OLS (non-HAC) se
    rho = beta[1]
    sigma = se[1]
    sigma2 = sigma^2

    s2 = dot(u, u) / (nobs - k)
    s = sqrt(s2)
    gamma0 = dot(u, u) / nobs

    cov = sum(abs2, u)
    for j in 1:l
        w = 1 - j/(l+1)
        gamma = dot(view(u, j+1:nobs), view(u, 1:nobs-j))
        cov += w * 2 * gamma
    end
    lam2 = cov / nobs
    lam = sqrt(lam2)

    stat_tau = sqrt(gamma0/lam2)*((rho-1)/sigma) - 0.5*((lam2-gamma0)/lam)*(nobs*sigma/s)
    stat_rho = nobs*(rho-1) - 0.5*(nobs^2 * sigma2/s2)*(lam2-gamma0)
    stat = test_type == :tau ? stat_tau : stat_rho

    # p-value: reuse the ADF critical-value table by regression type,
    # since both draw on the same MacKinnon asymptotic tau distribution
    # family for the `:tau` case. The `:rho` statistic has a DIFFERENT
    # asymptotic distribution (the "Z" / unit-root coefficient
    # distribution, not the tau/t distribution) -- using _ADF_CRIT for
    # :rho is WRONG and is flagged as a known gap below, not silently
    # assumed correct.
    crit_key = trend == :n ? :none : trend == :c ? :constant : :trend
    pval = test_type == :tau ? _interp_pvalue_left(stat, _ADF_CRIT[crit_key]) : NaN

    return (statistic=stat, pvalue=pval, lags=l, trend=trend, test_type=test_type, nobs=nobs)
end
```

**Known incomplete piece, flagged rather than hidden:** the `:rho`
statistic's asymptotic null distribution is **not** the same as `:tau`'s
(it's the Dickey-Fuller "Z" distribution for the coefficient, not the
t-distribution-like tau family) — reusing `_ADF_CRIT` for `:rho` would be
wrong, so `pvalue` is returned as `NaN` for `test_type=:rho` rather than
silently computing an incorrect number. A proper `:rho` critical-value
table (MacKinnon has one) is a follow-up.

---

## 7. Hand-verified test values

The verification transcript itself (reproducible, not just asserted):

```
Series: AR(1), φ=0.6, seed=0, n=200. lags=4, trend='c'.
arch tau stat: -6.072989823594912
arch rho stat: -64.60688631183274
my (verified) transcription: tau=-6.0729898235949245, rho=-64.60688631183295
max diff: ~1e-13 (both)
```

```julia
using Test, Random

@testset "pp_test" begin
    Random.seed!(0)
    n = 200
    e = randn(n)
    y = zeros(n)
    for t in 2:n
        y[t] = 0.6*y[t-1] + e[t]
    end
    # NOTE: as with the Stage 1.3 handoff, Julia's RNG stream differs
    # from numpy's even with "the same" seed -- exact cross-language
    # matching requires piping identical data through both, e.g. via a
    # shared CSV. Structural tests below are RNG-independent.

    r_tau = pp_test(y; trend=:c, test_type=:tau, lags=4)
    r_rho = pp_test(y; trend=:c, test_type=:rho, lags=4)
    @test isfinite(r_tau.statistic)
    @test isfinite(r_rho.statistic)
    @test r_tau.statistic != r_rho.statistic  # genuinely different statistics

    # :rho p-value is honestly NaN, not a wrong number
    @test isnan(r_rho.pvalue)
    @test !isnan(r_tau.pvalue)

    @test_throws ArgumentError pp_test(y; trend=:bogus)
    @test_throws ArgumentError pp_test(y; test_type=:bogus)
    @test_throws ArgumentError pp_test(y; lags=-1)
end
```

**For exact cross-language validation**: pipe the identical `y` (generate
in Python, save to CSV, load via `DelimitedFiles`) through both, then
compare directly against the two verified numbers above
(`-6.072989823594912` / `-64.60688631183274`) with `atol=1e-8`ish — a much
stronger check than the structural tests alone.

---

## 8. What to do with this

1. Implement `pp_test` per section 6.
2. **Verify `_ols` returns the plain OLS (non-HAC) standard error** for
   `se` — check this assumption holds; if `_ols` ever gains a HAC-se
   option later, `pp_test` must keep using the plain OLS one specifically
   (the algorithm requires it, per the verified source).
3. Run the structural tests in section 7; ideally also do the
   exact-cross-language check via a shared CSV, given how much value the
   Python-transcription verification already added here.
4. Leave `:rho`'s p-value as `NaN` until a real critical-value table is
   added — don't silently borrow `:tau`'s table for it.
5. Update `development-sequence.md`'s Stage 2.3 row: mark implemented,
   note the `:rho` p-value gap explicitly.

**Next in sequence:** Stage 2.4 (Ljung-Box) — already built and marked
✅ in earlier work, so (like 2.1/2.2) this will be a verification-and-
correction pass rather than starting fresh.
