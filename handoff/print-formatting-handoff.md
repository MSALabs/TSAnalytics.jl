# Handoff: Print/Show Formatting (Stages 0–4)

Status: **closed.** Checked against the actual codebase before
implementing anything (per this project's standing practice) and found
most of this handoff's "current state" premises were already stale by
the time it was picked up:
- `ADFTest.deterministic` → `regression`: **already done** (the struct
  already has `regression`, not `deterministic`).
- `PPTest` NamedTuple → struct promotion: **already done** (`PPTest` is
  already a proper `struct <: HypothesisTest` with its own `show`).
- New `show` methods for all six `HypothesisTest` subtypes: **already
  existed** for all six, in an ad hoc per-type style, before this handoff
  was written.

Net new work was much smaller than the handoff scoped: only
`ACFResult` genuinely had no `show` method. Implemented (`src/stattools.jl`)
with one addition beyond the handoff's own draft -- a `kind::Symbol`
field (`:acf`/`:pacf`), since `acf`/`pacf` share the one `ACFResult` type
with nothing to tell a printed result apart otherwise (the handoff's own
proposed `show` code had a dead-code placeholder for exactly this,
`kind = r === nothing ? "ACF" : "ACF"`, flagging but not resolving it).
Both convenience constructors default `kind=:acf`; `acf`/`pacf`
themselves always pass it explicitly.

**The shared verdict-line helper (`_pvalue_qualifier`/`_print_verdict`,
sections 3 and 4) was explicitly declined by the user** -- not
implemented, and not planned. The six existing `HypothesisTest` show
methods keep their current factual, compact style (statistic/p-value,
no interpretive "reject the null..." line). Don't reintroduce this later
without asking again.

For a fresh Claude Code session picking this up with no prior context.
Covers the print-formatting design discussed after Stage 3/4: a shared
`show` template for the six `HypothesisTest` subtypes built across
Stage 2, promoting `pp_test`'s return type to match the others, and a
one-line `ACFResult` display. Scoped deliberately to what exists through
Stage 4 — `StatsBase.CoefTable`-based model-fit printing (GLM.jl-style)
is real future work, tracked separately, not needed until Stage 6
produces actual model fits with coefficients to show.

## Where this fits

- **Touches:** `abstract.jl` (new shared helper), `unitroot.jl`
  (`ADFTest` field rename, `PPTest` struct promotion), `diagnostics.jl`
  (`LjungBoxTest`, `QSTest`, `JarqueBeraTest` show methods), `stattools.jl`
  (`ACFResult` show method).
- **Depends on:** nothing new — this is entirely about existing types
  from Stages 1–2, no new numerical work.

---

## 1. Verified references

**R's `print.htest`** (verified via `print(stats:::print.htest)` on a
live R session) — the single generic every built-in R hypothesis test
shares. Genuinely compact:
```
	Box-Ljung test

data:  y
X-squared = 5.7466, df = 5, p-value = 0.3317
```
One line of `statistic = ..., parameter = ..., p-value = ...`. No
interpretive help — the reader has to know what to do with the number.

**`HypothesisTests.jl`'s actual convention** (verified via several
independent real REPL transcripts, not assumed) — the more relevant
reference since it's the sibling Julia package in this exact domain:
```
One sample t-test
-----------------
Population details:
    parameter of interest:   Mean
    value under h_0:         0
    point estimate:          -0.0130
    95% confidence interval: (-0.0759, 0.0498)

Test summary:
    outcome with 95% confidence: fail to reject h_0
    two-sided p-value:            0.6843 (not significant)

Details:
    number of observations: 1000
    t-statistic:             -0.4068
```
Two things worth adopting that R's convention lacks entirely: a
**plain-English verdict line** and a **significance qualifier** on the
p-value.

## 2. Design: adopt the verdict/qualifier, keep the existing compact shape

`HypothesisTests.jl`'s full three-section format (11+ lines) is more
than this project's tests need, given they're run repeatedly during
interactive model diagnostics where scan speed matters. Keep the
compact multi-line shape already established across Stage 2's ad hoc
`show` methods, add a shared trailing verdict line via one small helper
every test type's `show` calls at the end.

```
Augmented Dickey-Fuller test
  regression   : constant
  lags         : 4
  n            : 200
  statistic    : -3.4521
  p-value      : 0.0089
  → reject the null of a unit root (p=0.0089, significant)
```

---

## 3. Implementation: shared helper

Add to `abstract.jl` (lives alongside the `HypothesisTest` abstract type
itself, so every subtype's file can use it without a new include-order
dependency):

```julia
"""_pvalue_qualifier(p) -- plain-English significance qualifier,
matching the style HypothesisTests.jl uses."""
function _pvalue_qualifier(p::Real)
    p < 0.001 && return "extremely significant"
    p < 0.01  && return "highly significant"
    p < 0.05  && return "significant"
    return "not significant"
end

"""_print_verdict(io, pvalue, reject_msg, fail_msg; alpha=0.05) -- the
shared trailing verdict line every HypothesisTest subtype's `show`
method ends with. `reject_msg`/`fail_msg` describe what rejecting vs.
failing to reject the null means for THIS specific test (e.g. "the null
of a unit root" for adf_test), not a generic phrase, so the printed line
reads as a complete sentence."""
function _print_verdict(io::IO, pvalue::Real, reject_msg::AbstractString,
                         fail_msg::AbstractString; alpha::Real=0.05)
    qualifier = _pvalue_qualifier(pvalue)
    verdict = pvalue < alpha ? "reject $reject_msg" : "fail to reject $fail_msg"
    print(io, "  → ", verdict, " (p=", round(pvalue, digits=4), ", ", qualifier, ")")
end
```

---

## 4. Per-type changes

### `ADFTest` — field rename + new show

The struct's `deterministic` field predates the Stage 2.1 review, which
renamed the *function's* keyword to `regression` (`:n`/`:c`/`:ct`/`:ctt`)
but didn't restate the struct. Fix the inconsistency while touching this
code anyway:

```julia
struct ADFTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    lags::Int
    regression::Symbol   # renamed from `deterministic`, matches adf_test's argument name
    n::Int
end

function Base.show(io::IO, t::ADFTest)
    println(io, "Augmented Dickey-Fuller test")
    println(io, "  regression   : ", t.regression)
    println(io, "  lags         : ", t.lags)
    println(io, "  n            : ", t.n)
    println(io, "  statistic    : ", round(t.statistic, digits=4))
    println(io, "  p-value      : ", round(t.pvalue, digits=4))
    _print_verdict(io, t.pvalue, "the null of a unit root", "the null of a unit root")
end
```

### `KPSSTest` — show only (field already named correctly)

```julia
function Base.show(io::IO, t::KPSSTest)
    println(io, "KPSS test")
    println(io, "  regression   : ", t.regression)
    println(io, "  lags         : ", t.lags)
    println(io, "  n            : ", t.n)
    println(io, "  statistic    : ", round(t.statistic, digits=4))
    println(io, "  p-value      : ", round(t.pvalue, digits=4))
    _print_verdict(io, t.pvalue, "the null of stationarity", "the null of stationarity")
end
```

### `PPTest` — promote from `NamedTuple` to a proper struct

This is the real fix in this pass, not just formatting. `pp_test`
(Stage 2.3) currently returns a plain `NamedTuple`
`(statistic=..., pvalue=..., lags=..., trend=..., test_type=..., nobs=...)`
— it can't participate in this shared design and breaks the
`<: HypothesisTest` contract every other Stage 2 test honors.

```julia
struct PPTest <: HypothesisTest
    statistic::Float64
    pvalue::Float64
    lags::Int
    trend::Symbol
    test_type::Symbol
    nobs::Int
end

function Base.show(io::IO, t::PPTest)
    println(io, "Phillips-Perron test")
    println(io, "  trend        : ", t.trend)
    println(io, "  test_type    : ", t.test_type)
    println(io, "  lags         : ", t.lags)
    println(io, "  n            : ", t.nobs)
    println(io, "  statistic    : ", round(t.statistic, digits=4))
    if isnan(t.pvalue)
        println(io, "  p-value      : NA (see pp_test docstring -- :rho lacks a critical-value table)")
        print(io, "  → no verdict available for test_type=:rho")
    else
        println(io, "  p-value      : ", round(t.pvalue, digits=4))
        _print_verdict(io, t.pvalue, "the null of a unit root", "the null of a unit root")
    end
end
```

Update `pp_test`'s final line (Stage 2.3 handoff, section 6) from
```julia
return (statistic=stat, pvalue=pval, lags=l, trend=trend, test_type=test_type, nobs=nobs)
```
to
```julia
return PPTest(stat, pval, l, trend, test_type, nobs)
```
No other change to `pp_test`'s logic — this is purely a return-type
promotion.

### `LjungBoxTest` — extend for the `boxpierce` fields

```julia
function Base.show(io::IO, t::LjungBoxTest)
    println(io, "Ljung-Box test")
    println(io, "  lags         : ", t.lags)
    println(io, "  df           : ", t.df)
    println(io, "  statistic    : ", round(t.statistic, digits=4))
    println(io, "  p-value      : ", round(t.pvalue, digits=4))
    if t.bp_statistic !== nothing
        println(io, "  Box-Pierce   : ", round(t.bp_statistic, digits=4),
                " (p=", round(t.bp_pvalue, digits=4), ")")
    end
    _print_verdict(io, t.pvalue, "the null of white noise", "the null of white noise")
end
```

### `QSTest`

```julia
function Base.show(io::IO, t::QSTest)
    println(io, "QS test (residual seasonality)")
    println(io, "  period       : ", t.period)
    println(io, "  statistic    : ", round(t.statistic, digits=4))
    println(io, "  p-value      : ", round(t.pvalue, digits=4))
    _print_verdict(io, t.pvalue, "the null of no residual seasonality", "the null of no residual seasonality")
end
```

### `JarqueBeraTest`

```julia
function Base.show(io::IO, t::JarqueBeraTest)
    println(io, "Jarque-Bera test")
    println(io, "  n            : ", t.n)
    println(io, "  skewness     : ", round(t.skewness, digits=4))
    println(io, "  kurtosis     : ", round(t.kurtosis, digits=4))
    println(io, "  statistic    : ", round(t.statistic, digits=4))
    println(io, "  p-value      : ", round(t.pvalue, digits=4))
    _print_verdict(io, t.pvalue, "the null of normality", "the null of normality")
end
```

### `ACFResult` — new one-line display

Not a `HypothesisTest`, no verdict line needed — its main use is the
`RecipesBase` plotting recipe (from the earlier plotting discussion),
not reading raw numbers off a REPL dump:

```julia
function Base.show(io::IO, r::ACFResult)
    kind = r === nothing ? "ACF" : "ACF"  # placeholder if ACF/PACF ever need distinguishing
    print(io, "ACFResult: ", length(r.lags), " lags, n=", r.n)
end
```

Note: `ACFResult` is shared by both `acf` and `pacf` (Stage 1.3) with no
field distinguishing which one produced it — worth adding a `kind::Symbol`
field (`:acf`/`:pacf`) if that distinction ever matters for display or
downstream dispatch; not changed here since it's out of scope for a
formatting-only pass, but flagged so it isn't lost.

---

## 5. Hand-verified tests

String-output tests, not numerical ones — check the right pieces appear,
not exact formatting to the character (which would make the tests brittle
against trivial spacing changes):

```julia
using Test

@testset "print formatting" begin
    # verdict/qualifier helper, boundary values
    @test TSAnalytics._pvalue_qualifier(0.0001) == "extremely significant"
    @test TSAnalytics._pvalue_qualifier(0.005) == "highly significant"
    @test TSAnalytics._pvalue_qualifier(0.03) == "significant"
    @test TSAnalytics._pvalue_qualifier(0.5) == "not significant"

    io = IOBuffer()
    TSAnalytics._print_verdict(io, 0.01, "the null of X", "the null of X")
    @test occursin("reject the null of X", String(take!(io)))

    io = IOBuffer()
    TSAnalytics._print_verdict(io, 0.5, "the null of X", "the null of X")
    @test occursin("fail to reject the null of X", String(take!(io)))

    # each HypothesisTest subtype's show runs without erroring and
    # contains its own key fields -- a smoke test, not exhaustive
    y = randn(200)
    for t in (adf_test(y), kpss_test(y), pp_test(y), ljungbox_test(y, 5),
              qs_test(sin.(2π.*(1:200)./12).+0.1.*randn(200), 12), jarque_bera_test(y))
        io = IOBuffer()
        show(io, t)
        s = String(take!(io))
        @test occursin("p-value", s)
        @test occursin("statistic", s) || occursin(uppercase("statistic"), s)
    end

    # PPTest is now a proper struct, not a NamedTuple
    @test pp_test(y) isa TSAnalytics.PPTest
    @test pp_test(y) isa HypothesisTest

    # :rho case shows "no verdict" rather than a fabricated p-value line
    io = IOBuffer()
    show(io, pp_test(y; test_type=:rho))
    @test occursin("no verdict available", String(take!(io)))

    # ACFResult
    io = IOBuffer()
    show(io, acf(y, 0:5))
    @test occursin("ACFResult", String(take!(io)))
end
```

---

## 6. What to do with this

1. Add `_pvalue_qualifier`/`_print_verdict` to `abstract.jl`.
2. Rename `ADFTest.deterministic` → `ADFTest.regression`, update
   `adf_test`'s construction call site accordingly.
3. Promote `pp_test`'s return from `NamedTuple` to the `PPTest` struct.
4. Add/replace the six `show` methods and `ACFResult`'s per section 4.
5. Run the tests in section 5.
6. Update `development-sequence.md` and the memory note: this closes out
   the "revisit print/show formatting" item queued after Stage 3 —
   `StatsBase.CoefTable`/GLM.jl-style model-fit printing remains real
   future work, tracked for Stage 6 when there's an actual coefficient
   table to show.
