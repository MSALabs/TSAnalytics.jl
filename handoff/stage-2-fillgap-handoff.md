# Handoff: Stage 2 Fill-Gap — 4 Existing-Test Improvements + 2 New Tests

For a fresh Claude Code session picking this up with no prior context.
This closes six items found by comparing Stage 2 (diagnostics) against
all six reference books, with a second, deeper research pass focused
specifically on getting real, verified technical material for each
one — not just citations. **Verification depth differs genuinely
across these six items — said explicitly per item below, not implied
uniform.**

## Where this fits

- **Depends on:** the existing `src/unitroot.jl` (ADF/KPSS/PP) and
  `src/diagnostics.jl` (Durbin-Watson) — all six items extend these
  two files, no new files needed.
- **All six preserve the project's existing discipline**: a test that
  can't be computed correctly should throw a clear error or return
  `NaN` with an explanation, never a silently wrong number. Every item
  below follows this.

---

## 1. ADF — full MacKinnon response-surface p-values (upgrades the existing linear-interpolation approximation)

**Fully verified — the exact source Python's own `adfuller()` uses**,
found directly in the installed `statsmodels.tsa.adfvalues` module, not
reconstructed from a paper description:

- **`mackinnonp(teststat, regression, N, lags)`**: for each regression
  type (`n`/`c`/`ct`/`ctt`), real tables of `tau_star` (regime cutoff),
  `tau_min`/`tau_max` (return 0.0/1.0 directly beyond these), and two
  sets of polynomial coefficients (`_tau_smallps`, `_tau_largeps`) —
  the p-value is `norm.cdf(polyval(coefficients, teststat))`, selecting
  the small-p or large-p coefficient set based on `tau_star`.
- **`mackinnoncrit(N, regression, nobs)`**: MacKinnon (2010)'s updated
  tables (`tau_c_2010`, `tau_ct_2010`, etc.) give exact critical values
  at 1%/5%/10% for *any* finite sample size via
  `polyval(coefficients, 1/nobs)` — not just the asymptotic values
  already in the current implementation.

**All real numerical coefficients for every regression type are
already extracted** (see the verification bundle's `adfvalues_source.py`
— the full, real file, not excerpted). This closes the gap directly;
no further research needed for this item.

### Design

```julia
adf_pvalue_response_surface(teststat::Float64, regression::Symbol, N::Int=1) -> Float64
adf_critical_values_response_surface(regression::Symbol, nobs::Union{Int,Nothing}=nothing; N::Int=1) -> NamedTuple
```
Add as the new default computation path in `adf_test`, keeping the
existing linear-interpolation method available under an explicit
`pvalue_method=:interpolated` fallback (matching this project's own
established pattern of preserving an older, honestly-labeled method
rather than deleting it outright when a better one is added).

---

## 2. PP `:rho` — the Z-statistic distribution tables (closes the current `NaN` p-value)

**Also fully verified, from the same file** — `adfvalues.py` contains
a second, separate set of tables specifically for this: `z_star_c`,
`z_c_smallp`, `z_c_largep` (and the `n`/`ct`/`ctt` equivalents),
explicitly commented *"The Z-statistic is used when lags are included
to account for serial correlation"* — exactly PP's `:rho` case. Same
algorithm shape as ADF's `:tau` (regime cutoff via `z_star`, then
`norm.cdf(polyval(...))`), different coefficient tables.

### Design

```julia
pp_rho_pvalue(teststat::Float64, regression::Symbol) -> Float64
```
Replace the current `return NaN` in `pp_test`'s `:rho` branch. **Keep
the existing honest docstring note** (`"!!! note ':rho's p-value is
NaN, not an approximation"`) as historical context in a comment, or
remove once this lands — team's call, but don't silently delete the
reasoning without a trace.

---

## 3. KPSS `:auto` (Hobijn et al. 1998) — a real, short, portable algorithm

**Found and verified directly** — `JimVaranelli/KPSS-autolag` on
GitHub, explicitly described as *"submitted to the statsmodels
package"* for this exact feature. The core algorithm
(`_kpss_autolag`) is genuinely short and self-contained:

```python
def _kpss_autolag(resids, nobs):
    covlags = int(nobs ** (2./9.))
    s0 = sum(resids**2) / nobs
    s1 = 0
    for i in range(1, covlags + 1):
        resids_prod = np.dot(resids[i:], resids[:nobs-i]) / (nobs/2.)
        s0 += resids_prod
        s1 += i * resids_prod
    s_hat = s1 / s0
    gamma_hat = 1.1447 * (s_hat**2) ** (1./3.)
    return min(nobs, int(gamma_hat * nobs**(1./3.)))
```
The repo's own `main()` includes real regression test cases (real
`statsmodels` datasets — `macrodata`, `sunspots`, `nile`, `randhie` —
with exact expected `lags` values, e.g. `realgdp('c')` → `lags=9`,
`sunactivity('c')` → `lags=7`) — genuine, reusable test fixtures, not
invented.

### Design

```julia
kpss_hobijn_autolag(resids::Vector{Float64}, nobs::Integer) -> Int
```
Wire into `kpss_test`'s existing `nlags=:auto` branch, which currently
throws `ArgumentError` — replace that branch's body, keep every other
`nlags` option unchanged.

---

## 4. Durbin-Watson `:exact` (Farebrother's AS 153 / "Pan's procedure") — citation-verified, not implementation-verified; said honestly

**Confirmed precisely which algorithm and paper**: R's own
`lmtest::dwtest`/`DescTools::DurbinWatsonTest` (the same function,
"integrated without logical changes") uses *"the Fortran version of
Applied Statistics Algorithm AS 153 by Farebrother (1980, 1984)"* —
citation: Farebrother, R.W. (1980), "Algorithm AS 153: Pan's Procedure
for the Tail Probabilities of the Durbin-Watson Statistic," *Applied
Statistics* 29, 224-227, with a correction in Farebrother (1984), "AS
R53."

**A real, useful, confirmed default-switching convention**, worth
matching directly: `DescTools::DurbinWatsonTest`'s own default is
`exact=NULL`, which resolves to **using the exact "pan" algorithm when
`n < 100`, falling back to the normal approximation above that** — not
an arbitrary choice, R's own actual default behavior.

**Honest gap, unlike items 1-3**: this algorithm computes the exact
null distribution of a weighted sum of chi-squared random variables via
characteristic-function inversion and numerical integration — genuinely
more involved than a portable closed-form table or a 15-line function.
**No working reference implementation was found and inspected this
session** (only the citation and general approach, via web search) —
unlike items 1-3, where the actual source code was found and read
directly. **Do not treat this as equivalent in verification depth to
the other three.** Two honest paths forward, either is acceptable:
1. Find and inspect R's actual Fortran AS 153 source directly (likely
   in `lmtest`'s own CRAN source package — try this first, CRAN may or
   may not be reachable depending on the environment) before
   implementing, matching this project's usual standard.
2. Implement via the more general Imhof (1961) method instead (a
   well-documented, more widely-implemented general technique for
   exactly this class of problem — the distribution of a quadratic
   form in normal variables), which achieves the same result via a
   different, better-documented numerical route, and cite the
   substitution explicitly rather than claim AS 153 was implemented
   when Imhof's method was used instead.

### Design (both paths converge on the same signature)

```julia
durbin_watson_pvalue_exact(dw_stat::Float64, X::Matrix{Float64}) -> Float64
```
Needs the design matrix `X` (not just residuals) — already noted as a
real requirement in the existing `:approx`-only implementation's own
documentation (R's exact method needs `X`, confirmed previously). Wire
into `durbin_watson_test`'s existing `method=:exact` (currently
"not implemented" per the source). **Default `method` should switch on
sample size** (`:exact` for `n < 100`, `:approx` otherwise), matching
`DescTools`'s real, confirmed convention above — not the current
always-`:approx` behavior.

---

## 5. ARCH-LM test — new, fully verified via real execution this session

Engle's test for conditional heteroskedasticity (from Tsay/`FinTS`'s
`ArchTest`). Constructed directly and run against both a genuine-ARCH
case and white noise, in base R this session:
```r
arch_lm_test <- function(resid, lags=4) {
  e2 <- resid^2; n <- length(e2)
  X <- sapply(1:lags, function(i) e2[(lags-i+1):(n-i)])
  y <- e2[(lags+1):n]
  r2 <- summary(lm(y ~ X))$r.squared
  stat <- (n-lags) * r2
  list(statistic=stat, p.value=1-pchisq(stat, df=lags))
}
```
**Real, verified results**: ARCH-present case: `statistic=25.345,
p=4.29e-05` (correctly rejects). White-noise case: `statistic=2.977,
p=0.562` (correctly fails to reject). Both directions confirmed
correct, not just one.

### Design
```julia
arch_lm_test(resid::Vector{Float64}, lags::Integer=4) -> ARCHLMTest
```
Matches the existing `HypothesisTest` family's shape (`statistic`,
`pvalue`, `df`) — same display/`show` conventions already established
for `ADFTest`/`KPSSTest`/etc.

---

## 6. Durbin & Koopman heteroskedasticity test — new, fully verified via real execution this session

Variance-ratio F-test on standardized residual thirds — a genuinely
different diagnostic from anything currently in Stage 2. Constructed
and verified the same way:
```r
dk_hetero_test <- function(resid) {
  n <- length(resid); h <- floor(n/3)
  stat <- sum(resid[(n-h+1):n]^2) / sum(resid[1:h]^2)
  list(statistic=stat, p.value=2*min(pf(stat,h,h), 1-pf(stat,h,h)), h=h)
}
```
**Real, verified results**: genuinely heteroskedastic case:
`statistic=13.059, p≈4.4e-16` (strongly rejects homoskedasticity).
Homoskedastic case: `statistic=1.036, p=0.901` (correctly near null,
statistic near 1 as expected when variance is genuinely constant).

### Design
```julia
dk_heteroskedasticity_test(resid::Vector{Float64}) -> DKHeteroTest
```
Same `HypothesisTest`-family conventions as item 5.

---

## Comprehensive test matrix

```julia
using Test

@testset "ADF response-surface -- matches statsmodels' own mackinnonp exactly" begin
    # run the real Python function directly during test development to
    # generate exact target values across several (teststat, regression, N)
    # combinations -- not yet done in this handoff, flagged as the first
    # concrete step, since the coefficients are verified but a specific
    # Julia-vs-Python numeric cross-check hasn't been run yet
end

@testset "PP :rho -- no longer NaN, matches statsmodels' pp_test rho pvalue" begin
    # same cross-check approach as above
end

@testset "KPSS :auto -- reproduces the real fixture cases from KPSS-autolag" begin
    # realgdp('c') -> lags=9;  sunactivity('c') -> lags=7;
    # volume('c') -> lags=5 -- real values from the verified repo, section 3
end

@testset "Durbin-Watson :exact -- default switches on n<100, matches DescTools' convention" begin
    # once implemented via either path in section 4
end

@testset "ARCH-LM -- real verified statistics from this session" begin
    # reproduce the R-verified ARCH-present (stat=25.345) and
    # white-noise (stat=2.977) cases exactly, section 5
end

@testset "DK heteroskedasticity -- real verified statistics from this session" begin
    # reproduce the heteroskedastic (stat=13.059) and
    # homoskedastic (stat=1.036) cases exactly, section 6
end
```

---

## What to do with this

1. Implement items 1-3 and 5-6 directly — all five have real,
   inspected source material or execution-verified formulas.
2. For item 4, pick one of the two honest paths in that section
   explicitly, and say in the code/commit which one was taken — don't
   let "implemented Durbin-Watson :exact" imply AS 153 specifically if
   Imhof's method was used instead.
3. Run the full test matrix; the ADF/PP cross-checks against real
   Python output are the one piece not yet done even for the
   well-verified items — do this before considering 1-2 fully closed,
   not just "coefficients copied correctly."
4. Update `development-sequence.md`'s Stage 2 table: mark 2.8 (ARCH-LM)
   and 2.9 (DK heteroskedasticity) as new entries, and record the
   verification-depth distinction for item 4 explicitly so it isn't
   later assumed to be as solid as the other five.
