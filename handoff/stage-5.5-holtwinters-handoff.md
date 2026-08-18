# Handoff: Stage 5.5 — Classical Exponential Smoothing (Simple/Holt/Holt-Winters)

For a fresh Claude Code session picking this up with no prior context.
Last of the five Stage 5 handoffs. **The headline finding here is a
genuine fitting-philosophy divergence, not a parameter mismatch**: R and
Python don't converge to slightly different numbers on the same data —
they solve genuinely different optimization problems that happen to
share the same model form. Confirmed by running both on identical data
in this session, not inferred from docs.

## Where this fits

- **Depends on:** Stage 4.1 (`_optimize`), and — surprisingly directly —
  **Stage 3.1's `classical_decompose`**, which R's own initialization
  turns out to be built on. This is a genuine, verified synergy worth
  exploiting, not a coincidence to note in passing.

---

## 1. Verified reference: R `stats::HoltWinters()` — full initialization algorithm from source

```r
HoltWinters(x, alpha=NULL, beta=NULL, gamma=NULL, seasonal=c("additive","multiplicative"),
            start.periods=2, l.start=NULL, b.start=NULL, s.start=NULL,
            optim.start=c(alpha=0.3, beta=0.1, gamma=0.1), optim.control=list())
```

Confirmed by printing R's actual source (`print(stats:::HoltWinters)`).
**Seasonal case** (`gamma` not `FALSE`):
```r
wind <- start.periods * f                                  # default: 2 full cycles
st <- decompose(ts(x[1:wind], ...), seasonal)                # <-- R's own classical decompose()!
dat <- na.omit(st$trend)
cf <- coef(.lm.fit(x = cbind(1, seq_along(dat)), y = dat))   # OLS: trend ~ time
l.start <- cf[1]      # intercept  -> initial level
b.start <- cf[2]      # slope      -> initial trend
s.start <- st$figure  # the decomposition's seasonal figure, directly -> initial seasonal
```
**Non-seasonal case** (`gamma=FALSE`): much simpler —
`l.start <- x[2]` (or `x[1]` for simple exponential smoothing with no
trend either), `b.start <- x[2] - x[1]`.

**The critical point**: these initial states are computed **once,
deterministically**, before optimization ever starts. The actual
optimization (a call into compiled C, `.C(C_HoltWinters, ...)`) only
searches over `alpha`/`beta`/`gamma` — 1 to 3 free parameters — to
minimize SSE, with the initial states held fixed throughout.

**This means R's `HoltWinters()` initialization is literally built on
the same algorithm as this project's own `classical_decompose`
(Stage 3.1, already implemented and tested)** — not a coincidental
similarity, R's C source calls `decompose()` directly. Reusing
`classical_decompose` for this stage's initialization isn't just
convenient, it's the actual reference algorithm.

## 2. Verified reference: Python `statsmodels.tsa.holtwinters.ExponentialSmoothing`

```python
ExponentialSmoothing(endog, trend=None, damped_trend=False, seasonal=None,
                      seasonal_periods=None, initialization_method='estimated',
                      initial_level=None, initial_trend=None, initial_seasonal=None,
                      use_boxcox=False, bounds=None, ...)
```

Confirmed via `inspect.signature`. **`initialization_method='estimated'`
is the default**, and — this is the real finding — it treats
`initial_level`/`initial_trend`/`initial_seasonal` as **free parameters,
optimized jointly with the smoothing parameters**, not fixed ahead of
time the way R does it.

## 3. The divergence — verified by running both on identical data, not assumed

Same series (trend + period-12 seasonality + noise, seed=0, n=48), full
additive Holt-Winters, both languages' actual defaults:

| | R (`HoltWinters`, seasonal="additive") | Python (`ExponentialSmoothing`, default `'estimated'`) |
|---|---|---|
| alpha | `0.3346` | `1.49e-08` (essentially 0) |
| beta | `0.0054` | `0.0` (exactly) |
| gamma | `1.0000` | `0.0` (exactly) |
| SSE | `57.35` | `37.09` (lower — better in-sample fit) |

**Not a close-but-different result — genuinely different fitted
models.** Python's `gamma=0` means the seasonal component is never
updated after initialization at all (all the seasonal fitting happens
through the jointly-optimized `initial_seasonal` values instead). Python
achieves the lower SSE precisely *because* it has more effective free
parameters (initial states are optimized too, not fixed) — this isn't R
being wrong or Python being wrong, it's two different, both-legitimate
optimization problems.

**Tried a third configuration for completeness**: Python's
`initialization_method='legacy-heuristic'` (intended to approximate
older/more classical behavior) gives yet a **third** distinct result
(`alpha=0.205, gamma=0.573, SSE=144.6` — worse than both defaults).
There is no clean "set this Python flag to match R" answer — confirmed
empirically, not assumed from the option's name.

---

## 4. Design decision: default to R's deterministic initialization

Reasons, stated explicitly:
1. **Directly reuses already-built, already-tested code**
   (`classical_decompose`, Stage 3.1) — genuine implementation
   leverage, not just "R happened to do it this way."
2. **Deterministic and fast**: fixed initial states mean the
   optimizer only searches 1-3 dimensions (`alpha`/`beta`/`gamma`),
   not the much larger joint space Python's default searches — fewer
   local-minima risks, faster convergence, more predictable behavior.
3. **Still offer Python's approach as an option** (`initialization_method
   = :heuristic` for R-style, `:estimated` for Python-style) — both are
   legitimate, and someone specifically wanting to match Python's
   numbers (or benefit from its potentially-better in-sample fit) should
   be able to.

---

## 5. Proposed Julia API

```julia
holt_winters(y, period::Union{Nothing,Integer}=nothing;
             trend::Union{Nothing,Symbol}=nothing,       # nothing | :additive
             seasonal::Union{Nothing,Symbol}=nothing,    # nothing | :additive | :multiplicative
             initialization_method::Symbol=:heuristic,   # :heuristic (R-style, default) | :estimated (Python-style)
             alpha::Union{Nothing,Real}=nothing, beta::Union{Nothing,Real}=nothing,
             gamma::Union{Nothing,Real}=nothing) -> ExponentialSmoothingModel
```

Design notes:
- **`trend`/`seasonal` as `Union{Nothing,Symbol}`**, matching Python's
  `None`-or-string pattern more directly than R's `beta=FALSE`-style
  boolean-as-sentinel convention — `nothing` reads more clearly in
  Julia than a boolean doing double duty as both "off" and "unset."
  `period` required only when `seasonal !== nothing`.
- **`initialization_method`**: the one genuinely new design surface
  this stage needs, given section 3's finding. `:heuristic` (default)
  = R's `decompose()`+OLS approach via `classical_decompose`.
  `:estimated` = Python's joint-optimization approach.
- **`alpha`/`beta`/`gamma`**: matches both references' naming and
  optional-override pattern (`nothing` = estimate it, a value = fix it)
  exactly — both R and Python support pinning individual smoothing
  parameters this way.

### `ExponentialSmoothingModel`

```julia
struct ExponentialSmoothingModel <: UnivariateModel
    alpha::Float64
    beta::Union{Nothing,Float64}
    gamma::Union{Nothing,Float64}
    level::Vector{Float64}
    trend_component::Union{Nothing,Vector{Float64}}
    seasonal_component::Union{Nothing,Vector{Float64}}
    fitted::Vector{Float64}
    resid::Vector{Float64}
    sse::Float64
    seasonal_type::Union{Nothing,Symbol}
    period::Union{Nothing,Int}
    initialization_method::Symbol
end
```

---

## 6. Implementation sketch

```julia
"""
    holt_winters(y, period=nothing; trend=nothing, seasonal=nothing,
                 initialization_method=:heuristic, alpha=nothing, beta=nothing, gamma=nothing)

Exponential smoothing (Simple / Holt / Holt-Winters), fit by SSE
minimization. `initialization_method=:heuristic` (default) computes
fixed initial level/trend/seasonal states via `classical_decompose`
(Stage 3.1) plus OLS on the extracted trend -- matches R's
`stats::HoltWinters()` exactly (verified from its actual C-backed
source, which itself calls `decompose()`). `:estimated` instead jointly
optimizes the initial states alongside the smoothing parameters,
matching Python's `statsmodels.tsa.holtwinters.ExponentialSmoothing`
default -- see the Stage 5.5 handoff doc for why these are genuinely
different fitting philosophies, not just different starting guesses
(verified: identical data produces meaningfully different fitted
parameters and different SSE under each).
"""
function holt_winters(y, period::Union{Nothing,Integer}=nothing;
                       trend::Union{Nothing,Symbol}=nothing,
                       seasonal::Union{Nothing,Symbol}=nothing,
                       initialization_method::Symbol=:heuristic,
                       alpha::Union{Nothing,Real}=nothing,
                       beta::Union{Nothing,Real}=nothing,
                       gamma::Union{Nothing,Real}=nothing)
    initialization_method in (:heuristic, :estimated) ||
        throw(ArgumentError("initialization_method must be :heuristic or :estimated"))
    seasonal !== nothing && period === nothing &&
        throw(ArgumentError("seasonal smoothing requires period"))
    yv = tsvalues(y)

    # --- initial states ---
    if initialization_method == :heuristic
        if seasonal !== nothing
            # R's exact algorithm: decompose the first 2*period points,
            # OLS the extracted trend for level/trend start, take the
            # figure directly for seasonal start
            wind = 2 * period
            decomp = classical_decompose(yv[1:wind], period; model=seasonal)
            valid = .!isnan.(decomp.trend)
            idx = findall(valid)
            X = hcat(ones(length(idx)), Float64.(idx))
            coefs, = _ols(X, decomp.trend[idx])
            l0, b0 = coefs[1], coefs[2]
            s0 = decomp.figure
        else
            l0 = trend === nothing ? yv[1] : yv[2]
            b0 = trend === nothing ? nothing : yv[2] - yv[1]
            s0 = nothing
        end
        # optimize only alpha/beta/gamma via _optimize (Stage 4.1),
        # holding l0/b0/s0 fixed -- mirrors R's C-level SSE search
    else # :estimated
        # optimize alpha/beta/gamma AND l0/b0/s0 jointly via _optimize
        # -- mirrors Python's default
    end

    # (SSE-minimizing recursion + optimizer wiring elided here -- the
    # Holt-Winters update equations themselves are standard and
    # low-risk; the actual complexity in this stage is entirely in the
    # initialization-method fork above, already fully specified)
end
```

**Deliberately incomplete**: the recursion + optimizer wiring itself is
standard textbook Holt-Winters (level/trend/seasonal update equations)
and genuinely low-risk — not worth writing out in full here when the
real content of this handoff is the initialization-method design that's
now fully specified above. Implement the recursion directly from any
standard reference (Hyndman & Athanasopoulos's textbook, already used
elsewhere in this project, has the exact equations) once the
initialization fork is in place.

---

## 7. Comprehensive test matrix

Series: trend + period-12 seasonality + noise, seed=0, n=48 (same data
as section 3's verification).

| Case | Call | Verified value |
|---|---|---|
| A | `holt_winters(y; seasonal=:additive, period=12)` (default `:heuristic`) | Initial states from `classical_decompose`; final `alpha=0.3346, beta=0.0054, gamma=1.0, SSE=57.35` (R match target) |
| B | same, `initialization_method=:estimated` | `alpha~0, beta=0, gamma=0, SSE=37.09` (Python match target) |
| C | `holt_winters(y; trend=:additive)` (no seasonal — Holt's linear) | `alpha=1, beta=1` per R's verified output (boundary case, expected given non-seasonal model on seasonal data) |
| D | `holt_winters(y)` (no trend, no seasonal — simple ES) | `alpha~1` per R's verified output |

```julia
using Test, Random

@testset "holt_winters" begin
    Random.seed!(0)
    n = 48
    t = 0:(n-1)
    y = 100 .+ 0.5 .* t .+ 10 .* sin.(2π .* t ./ 12) .+ randn(n)
    # NOTE: RNG differs from R/numpy -- verify against section 3's exact
    # values via a shared-CSV cross-language check, same technique as
    # Stage 3.1/3.2, before trusting exact-value assertions here.

    mA = holt_winters(y, 12; seasonal=:additive, initialization_method=:heuristic)
    @test mA.initialization_method == :heuristic
    @test 0 <= mA.alpha <= 1

    mB = holt_winters(y, 12; seasonal=:additive, initialization_method=:estimated)
    @test mB.initialization_method == :estimated
    # heuristic and estimated should NOT converge to the same fit --
    # per section 3, this is expected, not a bug
    @test !isapprox(mA.alpha, mB.alpha; atol=1e-3)
    @test !isapprox(mA.sse, mB.sse; atol=1e-3)

    mC = holt_winters(y; trend=:additive)  # Holt's linear, no seasonal
    @test mC.seasonal_component === nothing
    @test mC.gamma === nothing

    mD = holt_winters(y)  # simple ES
    @test mD.trend_component === nothing
    @test mD.seasonal_component === nothing

    @test_throws ArgumentError holt_winters(y; seasonal=:additive)  # missing period
    @test_throws ArgumentError holt_winters(y, 12; initialization_method=:bogus)
end
```

---

## 8. `show` — same `CoefTable`-adjacent style as `ARXModel`

Given `ExponentialSmoothingModel` doesn't have per-observation
coefficients the way `ARXModel` does (just 1-3 smoothing parameters plus
initial states), a full `CoefTable` is overkill — a compact summary
closer to R's terse `print.HoltWinters` fits better:

```julia
function Base.show(io::IO, m::ExponentialSmoothingModel)
    kind = m.seasonal_type !== nothing ? "Holt-Winters ($(m.seasonal_type))" :
           m.trend_component !== nothing ? "Holt's linear" : "Simple exponential smoothing"
    println(io, kind, " (initialization: ", m.initialization_method, ")")
    println(io, "  alpha : ", round(m.alpha, digits=4))
    m.beta !== nothing  && println(io, "  beta  : ", round(m.beta, digits=4))
    m.gamma !== nothing && println(io, "  gamma : ", round(m.gamma, digits=4))
    print(io, "  SSE   : ", round(m.sse, digits=4))
end
```

---

## 9. What to do with this

1. Implement `holt_winters`/`ExponentialSmoothingModel` per sections 5-6
   — the initialization fork is the real content; the recursion itself
   is standard and low-risk.
2. Run the tests in section 7; do the shared-CSV cross-language check
   (Stage 3.1/3.2 technique) for exact-value validation against both
   R's and Python's confirmed numbers from section 3.
3. Implement the `show` method per section 8.
4. Update `development-sequence.md`'s Stage 5.5 row: mark implemented,
   and make sure the initialization-philosophy divergence is visible
   there — it's the most consequential finding in all of Stage 5, given
   it means "the same call with different defaults" genuinely produces
   different science, not just different formatting.

---

## Stage 5 is now fully documented, all five handoffs complete

| Stage | Headline finding |
|---|---|
| 5.1 AR-X | R's `ar()` defaults to Yule-Walker not OLS; R's OLS variant has no exogenous-regressor support at all |
| 5.2 Forecast | Verified the SE formula ignores parameter uncertainty (matches both references' actual, simpler behavior) |
| 5.3 Accuracy | sMAPE's own intellectual lineage (Hyndman) publicly recommends against using it |
| 5.4 Cross-validation | R and Python solve this with genuinely different-shaped APIs, not just different defaults |
| 5.5 Exp. smoothing | R and Python use fundamentally different fitting philosophies (fixed vs. jointly-optimized initial states), confirmed to produce different models on identical data |

Every stage surfaced something substantive. Worth carrying the same
"verify by execution, don't assume convergence" discipline into
Stage 6.
