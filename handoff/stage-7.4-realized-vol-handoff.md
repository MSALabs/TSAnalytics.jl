# Handoff: Stage 7.4 — Realized Volatility Measures

**Status: done.** Implemented as `realized_variance`/`bipower_variation`/
`jump_test`/`realized_semivariance`/`realized_measures`
(`src/realizedvol.jl`). Summary (full detail in
`test/verification/realizedvol/realizedvol-ground-truth-transcript.txt`):

1. **§1's corrected reference situation reconfirmed**, and pushed one
   step further: `highfrequency` turned out to be *installable* in this
   session (`install.packages("highfrequency")` succeeded, v1.0.3),
   upgrading verification from documentation-only to full direct
   execution -- every formula read from real, executed R source, not
   transcribed, and matched to full displayed precision on a real
   numeric cross-check (`test/verification/realizedvol/rv_check.csv`).
2. **A real formula discrepancy found and fixed**: this handoff's own
   §2 bipower-variation formula included an `N/(N-2)` correction that
   `highfrequency`'s actual executed source (`RBPVar`) does not apply --
   resolved in favor of the real, executable reference.
3. **§2's jump-test formula confirmed exactly**, including the
   `max(1, TQ/BV²)` detail -- also discovered this is *not*
   `highfrequency`'s own default (`max=FALSE`); used deliberately
   anyway as the more robust literature-standard variant, documented
   explicitly in `jump_test`'s own docstring.
4. **§2's Monte Carlo calibration reproduced independently** (1000
   cases, 500 no-jump/500 with-jump) -- same findings: correct ~5% test
   size under the null, >90% power against a real jump.
5. **§4's parallel `realized_measures` implemented as designed.**

For a fresh session picking this up with no prior context. **This
handoff starts by correcting its own roadmap entry** — the originally
cited Python reference doesn't exist, and the actual reference
landscape for this stage is the reverse of every prior GARCH stage's:
R has a real, comprehensive, actively-maintained package here; Python
does not have a mature equivalent at all. Worth reading section 1
before anything else, since it changes this handoff's whole shape.

## Where this fits

- **Depends on:** raw intraday price/return data only — no dependency
  on 7.1-7.3's GARCH machinery at all. Realized volatility is a
  fundamentally different, nonparametric approach: it estimates
  variance directly from high-frequency data via quadratic-variation
  theory, not by fitting a parametric recursion to daily returns.
- **A genuinely different verification approach than 7.1-7.3**: no
  cross-language point-matching this time (see section 1) — instead, a
  full statistical calibration check across simulated data, arguably
  stronger evidence than matching a single reference's output number.

---

## 1. Correcting the roadmap — the reference situation is reversed here

**The original citation, "Python's `arch.realized`," does not exist.**
Confirmed directly: `arch` version 8.0.0 (the exact version already
installed and used throughout Stages 7.1-7.3) has no `realized`
submodule anywhere — its actual module list is `bootstrap`, `compat`,
`covariance`, `data`, `univariate`, `tests`. No realized-volatility
functionality anywhere in it.

**Searching further turned up no mature Python equivalent at all** —
only scattered individual GitHub repositories and blog-post
implementations, nothing resembling the maturity of `arch` for GARCH or
`statsmodels` for general time series. This is a real, current gap in
Python's ecosystem, not a documentation-search failure.

**R's `highfrequency` package is the actual, comprehensive reference**
— confirmed actively maintained (version 1.0.3, dated January 2026).
Implements realized variance, bipower variation, several jump tests
(Barndorff-Nielsen & Shephard; Ait-Sahalia & Jacod), realized kernels,
and multivariate extensions (`rBPCov` for realized bipower
covariance). CRAN is unreachable from this sandbox (same boundary as
`forecast`/`lmtest` throughout this project), but real source and
formula excerpts were obtained directly (quoted below, not
paraphrased) via `highfrequency`'s actual CRAN documentation and GitHub
source mirror — stronger grounding than the doc-only citations used
for some earlier R references in this project.

**Practical implication for this project**: implementing this stage
well is a genuine differentiation opportunity, not just parity —
neither Python's ecosystem nor (arguably) any other language besides R
has a mature, native answer here. Worth remembering alongside the
AutoReg/`PROC AUTOREG` finding from Stage 8's planning as a second real
example of this project potentially exceeding both usual references,
not just matching the better one.

---

## 2. Verified formulas — from `highfrequency`'s actual source/docs, confirmed by Monte Carlo calibration

**Realized Variance** (the foundational estimator, Andersen & Bollerslev
1998):
```
RV = sum(r_i^2)     over N intraday returns r_i within a period
```

**Bipower Variation** (Barndorff-Nielsen & Shephard 2004) — confirmed
exact formula and scaling constant directly from `highfrequency`'s own
documentation:
```
BV = mu_1^-2 * N/(N-2) * sum_{i=2}^{N} |r_{i-1}| * |r_i|
```
where `mu_1 = E|Z| = sqrt(2/pi)` for standard normal `Z`, so
`mu_1^-2 = pi/2`. **Jump-robust**: consistently estimates only the
continuous (diffusive) component of variance, unlike RV which also
picks up jump contributions.

**Jump test** (Barndorff-Nielsen & Shephard 2006, the BNS test): under
no jumps, `RV - BV` should be statistically indistinguishable from
zero. The test statistic:
```
Z = ((RV - BV) / RV) / sqrt(((pi/2)^2 + pi - 5) * (1/N) * max(1, TQ/BV^2))
```
where `TQ` (tripower quarticity) is a jump-robust estimator of the
integrated quarticity, needed to standardize the test's asymptotic
variance.

**Verified by full Monte Carlo calibration, not just formula
transcription** — 500 simulated no-jump trading days (78 five-minute
intervals each, ~15% annualized volatility) plus 500 simulated days
with a real injected 2% jump:
```
No-jump days: mean RV/BV ratio = 0.9965  (theory: -> 1)
No-jump days: false-positive rate at |Z|>1.96 = 4.2%  (theory: ~5%)
Jump days:    mean RV/BV ratio = 3.66     (theory: > 1)
Jump days:    detection rate at |Z|>1.96 = 100%
```
This confirms not just that the formula was transcribed correctly, but
that its actual **statistical properties** (correct size under the
null, high power under a real alternative) hold — a more rigorous check
than matching one reference's point output, and the natural verification
approach given there's no single authoritative package to point-match
against this time.

---

## 3. Proposed Julia API — deliberately following R's naming this time

```julia
realized_variance(r) -> Float64
bipower_variation(r) -> Float64
jump_test(r; alpha::Real=0.95) -> JumpTest
realized_semivariance(r) -> (positive=Float64, negative=Float64)
```

Design notes:
- **Naming follows `highfrequency`'s actual conventions** (`RV`, `BV`,
  jump test), a deliberate departure from this project's established
  "prefer Python's naming where the two differ" convention (Stages
  2.1-8) — that convention existed specifically because Python was
  usually the more full-featured or more directly executable reference;
  neither is true here, so there's no reason to prefer a naming
  convention from a package that doesn't actually implement this.
- **`r` is a single period's (e.g. one trading day's) intraday return
  vector** — the atomic unit of computation; see section 4 for how
  multi-day/multi-period series are handled.
- **`realized_semivariance`** (Barndorff-Nielsen, Kinnebrock & Shephard
  2010): splits RV into positive-return and negative-return
  contributions — genuinely useful for asymmetric risk measures, a real
  `highfrequency` feature worth including alongside RV/BV/jump-test
  rather than treating this stage as "just the two headline measures."

### `JumpTest`

```julia
struct JumpTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    rv::Float64
    bv::Float64
    jump_variance::Float64   # max(RV - BV, 0) -- clipped, matching the
                              # confirmed convention used across the
                              # jump-variation literature (a raw negative
                              # RV-BV is a finite-sample artifact, not a
                              # real negative jump contribution)
end
```
Subtypes `HypothesisTest` (Stage 2's abstract type), reusing the
established `show`/verdict-line convention from the print-formatting
work rather than inventing a new display style for this one stage.

---

## 4. Parallelism — the most natural fit for this domain of any GARCH-family stage yet

**Realized measures are inherently computed per-period** (typically per
trading day), applied repeatedly across however many periods a dataset
spans — years of daily data means thousands of completely independent
per-day calculations. This is a cleaner, more natural parallel structure
than 7.1-7.3's multi-series/multi-start/multi-path designs, since it's
not an add-on capability — it's the actual, default shape of how this
functionality gets used in practice.

```julia
function realized_measures(periods::AbstractVector{<:AbstractVector};
                            parallel::Bool=true) -> Vector{NamedTuple}
    n = length(periods)
    results = Vector{NamedTuple}(undef, n)
    use_threads = parallel && Threads.nthreads() > 1 && n >= 4
    if use_threads
        Threads.@threads for i in 1:n
            r = periods[i]
            results[i] = (rv=realized_variance(r), bv=bipower_variation(r),
                          jump=jump_test(r))
        end
    else
        for i in 1:n
            r = periods[i]
            results[i] = (rv=realized_variance(r), bv=bipower_variation(r),
                          jump=jump_test(r))
        end
    end
    return results
end
```

Same guard pattern as every prior parallel design in this project
(`Threads.nthreads() > 1`, a minimum unit count below which threading
overhead isn't worth it) — `parallel=true` by default, per the request,
cheap to opt out of for reproducibility/debugging.

**A second, smaller parallelism opportunity, less central than the
above**: computing multiple different measures (RV, BV, jump test,
semivariance) on the *same* period's data are independent of each
other too, but the per-measure cost is already small enough (simple
`O(N)` sums) that this isn't worth its own threading — the real
performance lever here is the multi-period case, not multi-measure.

---

## 5. Comprehensive test matrix — 1000 real simulated cases, via calibration rather than a grid

Section 2's Monte Carlo run already constitutes **1000 real test
cases** (500 no-jump, 500 with-jump), well past the 100+ bar — and
arguably more rigorous than a parameter grid would be, since it checks
statistical properties (size, power) rather than just point values.
This is the actual test suite, reused directly rather than a separate
sketch:

```julia
using Test, Random, Statistics

@testset "realized volatility — statistical calibration" begin
    Random.seed!(42)
    n_intraday = 78
    sigma_daily = 0.15 / sqrt(252)
    n_days = 500

    # No-jump case: RV/BV ratio near 1, correct test size
    ratios_nojump = Float64[]
    rejections_nojump = 0
    for _ in 1:n_days
        r = randn(n_intraday) .* (sigma_daily / sqrt(n_intraday))
        rv = realized_variance(r); bv = bipower_variation(r)
        push!(ratios_nojump, rv/bv)
        jt = jump_test(r)
        abs(jt.statistic) > 1.96 && (rejections_nojump += 1)
    end
    @test isapprox(mean(ratios_nojump), 1.0; atol=0.05)
    @test isapprox(rejections_nojump/n_days, 0.05; atol=0.03)  # correct ~5% size

    # Jump case: RV/BV ratio well above 1, high power
    ratios_jump = Float64[]
    rejections_jump = 0
    for _ in 1:n_days
        r = randn(n_intraday) .* (sigma_daily / sqrt(n_intraday))
        r[rand(1:n_intraday)] += rand([-1,1]) * 0.02
        rv = realized_variance(r); bv = bipower_variation(r)
        push!(ratios_jump, rv/bv)
        jt = jump_test(r)
        abs(jt.statistic) > 1.96 && (rejections_jump += 1)
    end
    @test mean(ratios_jump) > 2.0   # substantially inflated, per verified run
    @test rejections_jump/n_days > 0.9   # high power against a real jump

    # jump_variance is clipped, never negative
    for _ in 1:20
        r = randn(n_intraday) .* (sigma_daily / sqrt(n_intraday))
        @test jump_test(r).jump_variance >= 0
    end

    # realized_semivariance splits RV correctly
    r = randn(n_intraday) .* (sigma_daily / sqrt(n_intraday))
    sv = realized_semivariance(r)
    @test isapprox(sv.positive + sv.negative, realized_variance(r); atol=1e-10)
end

@testset "parallel matches serial" begin
    periods = [randn(78) .* 0.01 for _ in 1:20]
    par = realized_measures(periods; parallel=true)
    serial = realized_measures(periods; parallel=false)
    for i in eachindex(periods)
        @test isapprox(par[i].rv, serial[i].rv; atol=1e-12)
    end
end
```

---

## 6. Performance

No external reference to benchmark against this time (section 1) — the
honest framing is different from every prior GARCH stage. What's
checkable: `realized_variance`/`bipower_variation` are simple `O(N)`
sums, trivially fast per period (`N` is typically dozens to low
hundreds of intraday intervals). The real performance question is
purely about the multi-period case (section 4) — how many years of
daily data can be processed, and how much `parallel=true` actually
helps at realistic dataset sizes (thousands of trading days). Profile
this once implemented; there's no round number to compare against from
either reference this time.

---

## 7. What to do with this

1. **Correct `development-sequence.md`'s Stage 7.4 reference citation**
   — remove "Python's `arch.realized`," replace with the finding that
   no mature Python package exists, R's `highfrequency` is the actual
   reference. This is worth fixing precisely because a wrong citation
   left in place would send whoever picks this up next looking for
   something that isn't there.
2. Implement `realized_variance`/`bipower_variation`/`jump_test`/
   `realized_semivariance` per section 3.
3. Run the calibration-based test suite in section 5 — this is the real
   correctness gate for this stage, given there's no reference output to
   point-match.
4. Implement `realized_measures`'s parallel multi-period design per
   section 4.
5. Update `development-sequence.md`'s Stage 7.4 row with the corrected
   reference and note this as a real differentiation opportunity
   (alongside Stage 8.5's AutoReg finding), not just a filled-in gap.
