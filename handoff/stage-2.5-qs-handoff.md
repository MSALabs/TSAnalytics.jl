# Handoff: Stage 2.5 — QS Test (verified against JDemetra+ documentation)

Status: implemented ✅ (`src/diagnostics.jl`, `test/test_diagnostics.jl`).
This doc's central claim (a real correctness bug — missing `max(0,·)`
clipping) was independently re-verified via web search directly against
JDemetra+'s own published documentation
(jdemetradocumentation.github.io), not just trusted from this doc's
transcription — both the exact formula and the "`QSTest` calls
`LjungBoxTest`" architecture claim came back confirmed, word for word.
The worked numeric example (rho at lag 12 ≈ -0.82, unclipped QS ≈ 333.5,
clipped ≈ 161.8) was also independently reconstructed by hand from the
stated formula and matched before any Julia was written.

Implemented exactly as proposed in section 5: `clip_negative` added to
`ljungbox_test` generally (not hidden as a QS-only detail), `qs_test`
rewritten as a thin wrapper with the old parallel chi-square
implementation deleted entirely. The test suite's "wrapper matches direct
`ljungbox_test(..., clip_negative=true)` call" check (this doc's own
suggested most-important test) passes, confirming real delegation.

For a fresh Claude Code session picking this up with no prior context.
Different situation from 2.1–2.4: QS isn't an R or Python function at
all — it's a diagnostic specific to X-13ARIMA-SEATS/TRAMO-SEATS output,
reimplemented in the open-source **JDemetra+** toolkit. The "compete with
R and Python" framing shifts here to "match the actual reference
implementations that define this statistic" — and doing that surfaced a
**real correctness bug** in the existing build, not just a naming gap.

## Where this fits

- **Depends on:** Stage 1.3 (`acf`), and now Stage 2.4's `ljungbox_test`
  (see the architecture change below).
- **Replaces:** `qs_test(x, period::Integer)` in `diagnostics.jl`. The
  existing implementation computes a real statistic, but omits a step the
  reference implementations both do — see section 1.

---

## 1. The bug: QS clips negative autocorrelations to zero before squaring

Verified from JDemetra+'s own documentation, exact formula:

```
QS = n(n+2) · sum_{i=1}^{k} [max(0, rho_hat_{i*l})]^2 / (n - i*l),    k=2
```

where `l` is the seasonal period and `k=2` means only the first and
second seasonal lags are used (`l` and `2l`) — this part already matches
what's built. **What's missing is the `max(0, .)`.** JDemetra+'s own docs
state it directly: *"The QS test is a variant of the Ljung-Box test
computed on seasonal lags, where we only consider positive
auto-correlations."*

This isn't a cosmetic detail. Verified numerically on a constructed
series with a genuinely negative lag-12 autocorrelation (period-24
seasonal pattern, so lag 12 sits at a trough):
```
rho at lag 12: -0.821,  rho at lag 24: 0.776
QS without clipping: 333.47
QS with clipping (correct):  161.81
```
More than a 2x difference in the statistic — enough to flip a
significance conclusion in some cases. The reasoning behind the clip
makes sense once stated: seasonality manifests as *positive*
autocorrelation at the seasonal lag; a negative correlation there isn't
evidence of seasonality and should contribute nothing to the statistic,
not be squared into a false positive contribution.

## 2. A second finding: JDemetra+ implements `QSTest` as a thin call into its general `LjungBoxTest`

Directly from the JDemetra+ core wiki: *"The test is implemented in the
class QSTest, which calls the more general
`ec.tstoolkit.stats.LjungBoxTest`."* This confirms the architectural
question flagged at the end of the Stage 2.4 handoff — `qs_test` should
be a thin wrapper over `ljungbox_test`'s machinery, not a parallel
implementation, and this is the reference's own actual design, not just
a nice-to-have simplification on this package's part.

## 3. A third finding, flagged but not resolved: differencing convention ambiguity

From the literature (Webel & Ollech via an ASA proceedings paper):
*"the QS-statistic is biased in the case of under-differencing... In JD+,
first differences are taken once and the series is mean-adjusted, whereas
in X-13ARIMA-SEATS and TRAMO/SEATS the default order of differencing is
`max{1, min(d+D, 2)}`."* **The two reference implementations don't even
agree with each other** on how much differencing to apply before
computing QS. Given that ambiguity, the responsible choice is: **don't
silently difference inside `qs_test`** (there's no single "correct"
convention to silently pick), document both conventions so the caller can
choose deliberately, and expect the series passed in to already be
differenced appropriately for the use case.

---

## 4. Proposed Julia API

Add `clip_negative::Bool=false` to `ljungbox_test` (used internally by
`qs_test`, and available generally since it's a legitimate variant
someone might want directly):

```julia
ljungbox_test(x, lags=nothing; fitdf=0, boxpierce=false, clip_negative::Bool=false)
qs_test(x, period::Integer)   # signature unchanged; now correctly clips internally
```

Design notes:
- `qs_test`'s own signature doesn't change — `period` was already a
  sensible name (JDemetra+'s own notation uses `l` for the same
  concept, `qs_test`'s `period` is clearer for a Julia audience, no
  reason to rename to match a symbol that's only meaningful in the
  original paper's notation).
- `qs_test` becomes a thin wrapper, matching JDemetra+'s own
  `QSTest`-calls-`LjungBoxTest` architecture.
- `clip_negative` added to `ljungbox_test` generally, not hidden as a
  QS-only internal detail — it's a legitimate, real variant (a
  "one-sided" portmanteau test) that isn't unique to seasonal lags, so
  exposing it is more useful than burying it.

---

## 5. Implementation

```julia
"""
    ljungbox_test(x, lags=nothing; fitdf=0, boxpierce=false, clip_negative=false)

[... existing docstring from Stage 2.4 ...]

- `clip_negative`: if `true`, autocorrelations are clamped to `max(0, rho)`
  before squaring at each lag -- a "one-sided" portmanteau variant that
  only counts positive autocorrelation as evidence against the white-noise
  null. This is what `qs_test` uses internally (see its own docstring);
  exposed here directly since it's a legitimate variant in its own right,
  not exclusively a QS implementation detail.
"""
function ljungbox_test(x, lags::Union{Nothing,Integer,AbstractVector{<:Integer}}=nothing;
                        fitdf::Integer=0, boxpierce::Bool=false, clip_negative::Bool=false)
    y = tsvalues(x)
    n = length(y)
    ls = if lags === nothing
        1:min(10, n ÷ 5)
    elseif lags isa Integer
        1:lags
    else
        lags
    end

    rho_raw = acf(y, collect(ls); bartlett=false).values
    rho = clip_negative ? max.(rho_raw, 0.0) : rho_raw

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

"""
    qs_test(x, period::Integer) -> QSTest

QS statistic for residual seasonality at seasonal period `period`, per
JDemetra+'s definition (a one-sided Ljung-Box variant restricted to the
first two seasonal lags, `period` and `2*period`, counting only positive
autocorrelation as evidence of seasonality):

    QS = n(n+2) * sum_{i=1}^{2} [max(0, rho_hat_{i*period})]^2 / (n - i*period)

Implemented as a thin wrapper over `ljungbox_test` with
`clip_negative=true`, matching JDemetra+'s own architecture (its
`QSTest` class calls the more general `LjungBoxTest` internally, per the
JDemetra+ core wiki).

!!! note "Differencing is the caller's responsibility"
    JDemetra+ and X-13ARIMA-SEATS use *different* default differencing
    conventions before computing this statistic (JDemetra+: first
    difference once; X-13/TRAMO-SEATS: `max(1, min(d+D, 2))`), and the
    literature notes the statistic is biased under under-differencing.
    Since the two references don't agree with each other, this function
    does not silently difference `x` -- pass an appropriately
    differenced series for your use case.

Used to check seasonally adjusted series or SARIMA residuals for
leftover seasonality (the same diagnostic reported in X-13ARIMA-SEATS
output).
"""
function qs_test(x, period::Integer)
    period >= 2 || throw(ArgumentError("period must be >= 2"))
    y = tsvalues(x)
    n = length(y)
    2*period < n || throw(ArgumentError("series too short relative to period for QS test"))

    lb = ljungbox_test(y, [period, 2*period]; clip_negative=true)
    return QSTest(lb.statistic, lb.pvalue, period)
end
```

`QSTest` struct is unchanged from the existing build (`statistic`,
`pvalue`, `period`) — no field extension needed here, unlike
`ACFResult`/`LjungBoxTest` in earlier stages.

---

## 6. Hand-verified test values

From the verification script above (reproducible):
```
Series: sin(2*pi*t/24) + 0.3*noise, n=240, seed=7
rho at lag 12 (= period, in a period-24 series): -0.8209
rho at lag 24 (= 2*period):                       0.7757
QS without clipping (WRONG): 333.47
QS with clipping (correct):  161.81
```

```julia
using Test, Random

@testset "qs_test clipping" begin
    Random.seed!(7)
    n = 240
    t = 0:n-1
    # period-24 pattern -> lag-12 autocorrelation is negative (a trough)
    y = sin.(2π .* t ./ 24) .+ 0.3 .* randn(n)

    r = qs_test(y, 12)
    # the correct (clipped) statistic must be LESS than what an
    # unclipped computation would give, since the lag-12 contribution
    # gets zeroed rather than squared into a positive term
    lb_unclipped = ljungbox_test(y, [12, 24]; clip_negative=false)
    @test r.statistic < lb_unclipped.statistic

    # ljungbox_test with clip_negative=true on the same lags must
    # exactly match qs_test's own statistic (confirms the wrapper is
    # actually delegating, not duplicating divergent logic)
    lb_clipped = ljungbox_test(y, [12, 24]; clip_negative=true)
    @test isapprox(r.statistic, lb_clipped.statistic; atol=1e-10)
    @test isapprox(r.pvalue, lb_clipped.pvalue; atol=1e-10)

    @test_throws ArgumentError qs_test(y, 1)   # period must be >= 2
end

@testset "clip_negative general behavior" begin
    Random.seed!(8)
    y = randn(300)  # white noise -- clipping shouldn't spuriously inflate significance
    r_clipped = ljungbox_test(y, 10; clip_negative=true)
    r_unclipped = ljungbox_test(y, 10; clip_negative=false)
    # clipped statistic can never exceed the unclipped one, term-by-term
    @test r_clipped.statistic <= r_unclipped.statistic
end
```

---

## 7. What to do with this

1. Add `clip_negative` to `ljungbox_test` per section 5 — this is a
   Stage 2.4 file change, done here since QS is what surfaced the need
   for it.
2. Rewrite `qs_test` as the thin wrapper shown — delete its old
   standalone chi-square computation entirely rather than leaving two
   parallel implementations that could drift apart.
3. Run the tests in section 6 — the "wrapper matches direct
   `ljungbox_test(..., clip_negative=true)` call" test is the most
   important one structurally, since it's what actually confirms the
   delegation is real and not just superficially similar.
4. Update `development-sequence.md`'s Stage 2.5 row: mark the clipping
   bug fix explicitly (this was a real correctness issue in shipped code,
   worth being visible in the changelog, not quietly folded in), note the
   architecture change to a thin wrapper, and note the differencing
   convention ambiguity as a caller-responsibility, not a package gap.

---

## Stage 2 complete

This closes out the "one by one" pass through 2.1–2.5. Summary of what
each pass actually found, since the pattern itself is worth carrying
into future stages:

| Stage | Nature of the finding |
|---|---|
| 2.1 ADF | R itself is internally inconsistent (two packages, two different capabilities); a real missing feature (autolag search) |
| 2.2 KPSS | Mostly already correct; one real gap (Python's `:auto` is a different algorithm, not a formula) |
| 2.3 PP | Built fresh; formula-verified against real `arch` package source to ~1e-13 before writing Julia |
| 2.4 Ljung-Box | R's *default* is the statistically weaker test; a subtle vector-semantics difference worth documenting loudly |
| 2.5 QS | A genuine correctness bug (missing clipping), found only by checking the actual defining reference instead of assuming the existing build's math was right |

Worth noting: every single stage in this pass turned up something real,
not just naming polish. That's a reasonable prior to carry into Stage 3
and beyond — assume there's a genuine discrepancy or gap to find, not
just an argument to rename.
