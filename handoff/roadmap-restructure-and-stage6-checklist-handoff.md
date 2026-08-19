# Handoff: Roadmap Restructure + Stage 6 Conclusion Checklist

**Status: done, with two corrections to this doc's own claims.**

1. **The "attached alongside this doc, edits already applied" claim in
   Part 1 was checked directly and found false** — only this handoff
   `.md` itself was ever added to the repo (`git status` confirmed no
   other new/modified files at the time this was picked up);
   `development-sequence.md`/`CLAUDE.md` still had the pre-restructure
   Stage 6.9/Stage 9 structure. The restructure described in Part 1 was
   written from scratch into both files instead of adopted from
   nonexistent attachments — see the current Stage 6/8/9 sections and
   `CLAUDE.md`'s "Unscheduled but ready".
2. **Part 2's checklist was already satisfied before this handoff
   arrived**, by work done earlier in the same session that built Stage
   6.7/6.8: checklist item 1 (all-four-SARIMA-blocks-nonzero case) was
   closed during 6.7 itself (`test/verification/sarima/sarima_full.csv`,
   the bulk sweep's `full_block` case); checklist item 2 (auto_arima's
   seasonal `D`) took option (b), explicit-`D`-required, with a named
   `ArgumentError` — both already documented in `development-sequence.md`'s
   6.7/6.8 rows before this doc was read.
3. **Part 3's own "exact seed=5 script in this handoff's verification
   transcript" doesn't exist either** (checked directly — no such
   transcript file, no `lmtest`-installed-or-not claim verified beforehand
   either). Implemented from the section's *description* (`n=100`,
   `y=2+1.5x+e`, AR(1) `phi=0.6`) with a fresh, self-contained
   regeneration instead — `test/verification/durbinwatson/`, dual-checked
   against real Python `statsmodels` and real R `lmtest::dwtest` (both
   confirmed installed and reachable this session, correcting the
   assumed-unreachable claim implicitly baked into "no such transcript
   exists to check against"). See `development-sequence.md`'s new Stage
   2.7 row for the numbers.

For a fresh Claude Code session picking this up with no prior context.
Two things to do here: (1) adopt the already-restructured
`development-sequence.md`/`CLAUDE.md` (attached alongside this doc,
edits already applied — just replace your local copies, no manual
re-editing needed), and (2) **do not mark Stage 6 concluded** until two
specific re-verification items are confirmed. This second part is the
important one — read it even if you skim the rest.

---

## Part 1 — What changed in the roadmap, and why

### 1a. Full ETS moved from 6.9 to a renamed Stage 9

**Old**: Stage 6.9 = full ETS, inside "Stage 6 — ARIMA, SARIMA, and full
ETS." Stage 9 = "Deferred ETS extras (not required for v1.0)," implying
low priority.

**New**: Full ETS is now **Stage 9.1**, inside a renamed **"Stage 9 —
Full ETS (state-space layer) and extensions."** Damped trend (9.2) and
TBATS (9.4) had their dependency updated from `6.9` to `9.1` (same
requirement, new location). Theta (9.3) is unaffected — it never
depended on the ETS engine at all, only on Stage 3.2 + 5.5.

**Why, precisely** — two reasons, not one:
1. **A real technical uncertainty, not just tidiness**: local-level/
   trend ETS components are non-stationary by construction. Stage 6's
   `GaussianSSM` (6.1-6.4) only supports *stationary* initialization via
   the Lyapunov equation. Full ETS may genuinely require Stage 8's exact
   diffuse initialization to be built correctly at all -- this was
   already flagged as an open question before this restructure, and
   moving the item doesn't resolve it, just places it where the
   dependency can actually be checked once Stage 8 exists.
2. **Full ETS is popular -- say so explicitly wherever this stage is
   described.** The old "deferred extras, not required for v1.0" framing
   was honest for damped-trend/Theta/TBATS but actively misleading for
   full ETS itself, which is arguably as widely used in practice as
   `auto.arima()`. Don't let the new Stage 9's name or description imply
   this is a minor feature -- it's deferred for a dependency reason, not
   a priority reason. Stage 5.5's classical Holt-Winters (already built)
   is what v1.0 ships with in the meantime; that's a real, worth-stating
   gap (no auto model selection, no multiplicative-error models, no
   likelihood-based intervals), not a silent one.

### 1b. AutoReg (SAS `PROC AUTOREG` equivalent) -- added as two tiers, not one

Not built anywhere yet, but now on record with a specific placement:

- **Classical tier (Yule-Walker/ULS)**: noted in `CLAUDE.md` under
  "Unscheduled but ready" -- no new numbered stage, deliberately, to
  avoid roadmap churn while Stage 6/GaussianSSM work is active. Needs
  nothing beyond what's already built. Good candidate for spare-capacity
  filler work.
- **Full tier (ML + AR-GARCH)**: added as **Stage 8.5**. Needs 8.2
  (exact diffuse init, for the regression coefficients as a diffuse
  state block) and Stage 7 (GARCH, for the combined AR-GARCH case
  specifically).
- **A genuine differentiation note, not just a feature gap**: neither R
  nor Python has as clean a single equivalent as SAS's `PROC AUTOREG` --
  R's closest analogs (`nlme::gls()` + the separate `orcutt` package)
  are scattered; Python has nothing comparably integrated. Worth
  remembering this isn't just "catching up to R/Python" the way most of
  this roadmap's items are.
- **A related, still-unbuilt gap**: no Durbin-Watson test exists in
  Stage 2. It's PROC AUTOREG's signature diagnostic -- worth adding
  alongside whichever AutoReg tier gets built first, not deferred
  separately.

---

## Part 2 -- Stage 6 conclusion checklist (do not skip)

Two items were reported as "done"/"in progress" earlier but have
specific, real, unverified risk flagged against them. **Confirm both
explicitly before treating Stage 6 as concluded** -- this isn't a
formality, both were flagged as the highest-risk unverified arithmetic
in their respective handoffs.

### Checklist item 1 -- Stage 6.7 (seasonal ARIMA)

**The problem**: the only dual-verified ground truth for `fit_sarima`
had `q=Q=0` (a SARIMA(1,0,0)(1,1,0)_12 case). This means the `theta`/
`Theta` parameter-unpacking indices in the optimizer's objective
function --

```julia
theta = q > 0 ? partrans(raw[p+1:p+q]) : Float64[]
Theta = Q > 0 ? partrans(raw[p+q+P+1:p+q+P+Q]) : Float64[]
```

-- were **never exercised by real data**. This is exactly the category
of bug that's bitten this project before (off-by-one slips in index
arithmetic that look fine until the specific code path that exercises
them actually runs).

**What to do**:
1. Generate a series with a genuine `SARIMA(1,0,1)(1,1,1)_s` structure
   (all four polynomial blocks nonzero) -- synthetic, known coefficients,
   same style as the Stage 6.7 handoff's `sarima_shared.csv`.
2. Fit it with R's `arima(order=c(1,0,1), seasonal=list(order=c(1,1,1),
   period=s))` and/or Python's `SARIMAX(order=(1,0,1),
   seasonal_order=(1,1,1,s))` for ground truth.
3. Fit the same series with this project's `fit_sarima` and confirm the
   coefficients match.
4. Only once this passes: Stage 6.7 can be considered actually verified,
   not just "ran without erroring."

### Checklist item 2 -- Stage 6.8 (auto-order selection)

**The problem**: `auto_arima`'s seasonal case needs a seasonal
unit-root/strength test (Canova-Hansen or OCSB) to auto-detect `D`.
**Neither exists anywhere in this project.** The Stage 6.8 handoff
proposed a specific interim: require `D` passed explicitly for the
seasonal case, erroring clearly rather than silently guessing.

**What to do**:
1. Confirm which path was actually taken: (a) one of the two tests got
   built, or (b) the explicit-`D`-required interim was adopted.
2. If (b): confirm the error message is clear and specific (something
   like *"automatic D detection requires Canova-Hansen/OCSB, not yet
   implemented -- pass D explicitly"*), not a generic failure.
3. Update this project's own `development-sequence.md` Stage 6.8 row
   with whichever answer is actually true -- it currently just flags the
   question, not the resolution.
4. Separately (not blocking Stage 6 conclusion, but worth tracking): if
   parallelism was implemented for the `stepwise=false` exhaustive
   search path, confirm the parallel and serial versions select the
   **identical** model -- parallelism should only affect speed, never
   which model wins. Cheap, valuable regression guard specific to this
   stage's design.

---

## Part 3 — New addition: Durbin-Watson test (Stage 2.7)

Not built anywhere in this project yet. Flagged during the AutoReg
discussion as PROC AUTOREG's signature diagnostic, but it belongs in
Stage 2 by category, not tied to AutoReg specifically — any regression
residuals can be tested for autocorrelation this way, independent of
which model produced them.

### Verified references

**Python** (`statsmodels.stats.stattools.durbin_watson`, confirmed
directly from source): computes **only the statistic**, no p-value at
all:
```python
dw = sum((e[t] - e[t-1])^2 for t in 2..n) / sum(e[t]^2 for t in 1..n)
```
Genuinely minimal — the docstring itself notes `dw ~ 2*(1-r)` where `r`
is the residuals' lag-1 autocorrelation, as a rough intuition, not a
formula the function computes.

**R** (`lmtest::dwtest` — the standard implementation; not in base R,
and the `lmtest`/`car` packages hit the same CRAN-unreachable boundary as
`forecast` throughout this project, confirmed by a direct failed install
attempt, not assumed): a materially richer function. Two things worth
knowing precisely:
1. **It takes a formula or fitted model, not just a residual vector** —
   a real architectural difference, not a naming one. The exact null
   distribution of the DW statistic is a weighted sum of chi-squared
   variables whose weights depend on the eigenvalues of a matrix built
   from the regression design matrix `X`. It is **not** computable from
   residuals alone in general — Python's version sidesteps this
   entirely by not providing a p-value at all.
2. **The default `alternative` is `"greater"`** — a one-sided test
   specifically for *positive* autocorrelation, matching the classical
   econometric tradition (positive autocorrelation is the historically
   common case tested for), not a two-sided default.

### Verified statistic, real regression data

```python
# n=100, y = 2 + 1.5x + e, e[t] = 0.6*e[t-1] + innovation[t]
DW = 0.6342859899908829   # correctly low -- strong positive autocorrelation

# n=100, same y, e = white noise (no AR structure)
DW = 2.3043180529148293   # correctly near 2 -- no strong autocorrelation
```

### Proposed API

```julia
durbin_watson_test(resid::Vector{Float64}, X::Union{Nothing,Matrix{Float64}}=nothing;
                    alternative::Symbol=:greater, method::Symbol=:approx)
```

- `resid`: OLS residuals.
- `X`: optional design matrix — unused by `:approx`, but part of the
  signature now so `:exact` (Pan's or Imhof's algorithm, matching R's
  actual method) can be added later without a breaking signature change.
  Passing `method=:exact` before that exists should throw a clear,
  named error, not silently fall back to `:approx` under that name --
  same "don't fake support" discipline as every other flagged gap in
  this project.
- `alternative`: `:greater` (default, matches R's default), `:less`,
  `:two_sided`.
- `method`: `:approx` (default) uses a large-sample normal approximation
  — `(DW - 2) / sqrt(4/n)` treated as approximately standard normal.
  **This is a genuinely crude approximation, not R's exact method** —
  document that explicitly; it ignores the `X`-dependence entirely,
  unlike R's real formula. `:exact` is the honest future upgrade path.

### Implementation

```julia
"""
    durbin_watson_test(resid, X=nothing; alternative=:greater, method=:approx)

Durbin-Watson test for first-order autocorrelation in regression
residuals. The statistic itself matches Python's
`statsmodels.stats.stattools.durbin_watson` exactly (verified against
its source). The p-value uses a large-sample normal approximation by
default -- a genuinely cruder method than R's `lmtest::dwtest`, which
computes an exact p-value from the eigenvalues of a matrix built from
the regression design `X` (via Pan's or Imhof's algorithm). `:exact` is
not yet implemented; `X` is accepted now so that method can be added
without breaking this signature later.
"""
function durbin_watson_test(resid::Vector{Float64}, X::Union{Nothing,Matrix{Float64}}=nothing;
                             alternative::Symbol=:greater, method::Symbol=:approx)
    alternative in (:greater, :less, :two_sided) ||
        throw(ArgumentError("alternative must be :greater, :less, or :two_sided"))
    method in (:approx, :exact) || throw(ArgumentError("method must be :approx or :exact"))
    method == :exact && throw(ArgumentError(
        "method=:exact (Pan's/Imhof's algorithm, matching R's lmtest::dwtest) " *
        "is not yet implemented -- use :approx"))

    n = length(resid)
    dw = sum(abs2, diff(resid)) / sum(abs2, resid)

    z = (dw - 2.0) / sqrt(4.0 / n)
    pval = alternative == :greater ? _std_normal_cdf(z) :
           alternative == :less    ? 1 - _std_normal_cdf(z) :
                                      2 * min(_std_normal_cdf(z), 1 - _std_normal_cdf(z))

    return (statistic=dw, pvalue=pval, alternative=alternative, method=method, n=n)
end
```

### Test cases

```julia
using Test

@testset "durbin_watson_test" begin
    # Case A: verified statistic value, autocorrelated residuals
    # (regenerate via the exact seed=5 script in this handoff's verification
    # transcript for byte-identical resid; structural check shown here)
    resid_ar = [...]  # phi=0.6 AR(1) OLS residuals, n=100
    rA = durbin_watson_test(resid_ar)
    @test isapprox(rA.statistic, 0.6342859899908829; atol=1e-8)
    @test rA.pvalue < 0.001   # strong evidence of positive autocorrelation

    # Case B: verified statistic value, white-noise residuals
    resid_wn = [...]  # n=100
    rB = durbin_watson_test(resid_wn)
    @test isapprox(rB.statistic, 2.3043180529148293; atol=1e-8)
    @test rB.pvalue > 0.5    # no evidence of positive autocorrelation

    # DW must always lie in [0, 4]
    @test 0 <= rA.statistic <= 4
    @test 0 <= rB.statistic <= 4

    # alternative direction sanity: same data, opposite alternative should
    # give a large p-value for the case that has strong evidence the other way
    rA_less = durbin_watson_test(resid_ar; alternative=:less)
    @test rA_less.pvalue > 0.99   # data shows POSITIVE autocorrelation, so
                                   # testing for NEGATIVE should fail hard

    @test_throws ArgumentError durbin_watson_test(resid_ar; method=:exact)
    @test_throws ArgumentError durbin_watson_test(resid_ar; alternative=:bogus)
end
```

### What to do with this

1. Implement `durbin_watson_test` per the sketch above.
2. Regenerate `resid_ar`/`resid_wn` from the exact Python generating code
   in this handoff (seed=5, n=100, `y = 2 + 1.5x + e`) for byte-identical
   ground truth, rather than re-deriving new test data.
3. Leave `:exact` genuinely unimplemented rather than approximated under
   that name -- this is real, non-trivial numerical work (eigenvalues of
   an `X`-derived matrix, then Pan's or Imhof's method to get the exact
   null distribution), worth its own dedicated pass later, not squeezed
   into this addition.
4. Add to `development-sequence.md` as **Stage 2.7**, alongside the rest
   of Chapter Three's diagnostic tests.

---



## What "concluding Stage 6" should actually mean

**Update: 6.8 (auto-order selection) reported complete** — this means
implementation capacity is no longer the blocker for anything in Stage
6. It does **not** by itself confirm the two checklist items above are
resolved; "implemented" and "the specific flagged risk was re-verified"
are different claims. Confirm both explicitly before treating Stage 6 as
concluded, same as before this update.

Once both checklist items are confirmed (not just "the code runs" --
actually re-verified against fresh ground truth per the steps above):
1. Update `development-sequence.md`'s Stage 6 intro text to remove the
   "re-verification required" language on 6.7/6.8's rows and replace
   with the actual confirmed status.
2. Update `CLAUDE.md`'s "Active decisions" section -- the GaussianSSM
   2-week trial's stated bar (a working end-to-end `fit_arima()` call)
   was already reported met pending confirmation; Stage 6's full
   conclusion (through 6.8) is the natural point to formally close that
   decision out, not before.
3. Only then move attention to Stage 7 (GARCH) or Stage 8 (ARIMAX/
   SARIMAX) as the next real work, per the roadmap's own sequencing.
