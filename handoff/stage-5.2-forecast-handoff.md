# Handoff: Stage 5.2 — Forecast Object & Prediction Intervals

Status: **done.** `forecast`/`StatsAPI.predict`/`Forecast`
(`src/forecast.jl`, `test/test_forecast.jl`) built per sections 4-6,
with one correction to this handoff's own stated environment and two
real gaps found and closed along the way:

1. **The "R unreachable" verification boundary didn't hold in this
   session.** `R`/`Rscript` weren't on `PATH`, which this handoff (and an
   earlier session's memory note) read as "R not installed" -- but R
   4.6.0 turned out to be genuinely installed (`C:\Program Files\R\R-4.6.0`),
   reachable via its full path, and CRAN turned out to be reachable too
   (`install.packages("forecast")` succeeded, pulling in `urca`/`zoo`/etc.
   as dependencies). Section 2's "documented, not executed" caveat is
   resolved: `predict.ar()`, `forecast.ar()`, and `print.forecast()`'s
   exact column format (`Point Forecast`/`Lo 80`/`Hi 80`/`Lo 95`/`Hi 95`)
   were all confirmed by direct execution on the exact Stage 5.1 model,
   and R's numbers agree exactly with Python's `AutoReg.get_prediction()`
   -- section 3's design (matching both) needed no correction.
2. **A real design gap, not just an incomplete sketch**: section 5's
   point-forecast recursion was left as a comment
   (`# full point-forecast recursion elided here`) because `ARXModel`
   (Stage 5.1) never stored the fitted series at all -- there was
   nothing to seed lagged values from. Added `y::Vector{Float64}` and
   `period::Union{Nothing,Int}` fields to `ARXModel` (verified only one
   construction site exists, safe to extend) rather than changing
   `predict`'s signature to take data separately.
3. **The recursion generalizes by construction, not by extra cases**:
   rather than hand-coding trend/seasonal logic, it walks
   `ARXModel.names` (parsing `"const"`/`"trend"`/`"s(season,period)"`/
   `"y.L<lag>"` labels) at each future time step -- so an arbitrary lag
   *subset*, `trend`/`seasonal` continuation, and combinations of both
   are handled correctly automatically, verified against freshly
   generated (not handoff-provided) Python reference numbers for
   `trend=:ct`, `lags=[1,3]`, and `seasonal=true` separately, beyond the
   handoff's own single default-`trend=:c` test case.
4. **Exogenous-regressor models are explicitly rejected** (future
   `exog` values would be needed and aren't available) -- a case this
   handoff's design never mentioned at all, closed with a clear
   `ArgumentError` rather than silently ignoring the regressor.

Documentation convention: the full docstring lives on the exported
`forecast`, not the unexported `StatsAPI.predict` extension -- matching
the precedent `ARXModel`'s own `StatsAPI.coef`/`vcov` extensions already
set (no separate docstring on the bare extension method).

For a fresh Claude Code session picking this up with no prior context.
Second of five Stage 5 handoffs. First consumer of `ARXModel` (Stage
5.1) — this is the interface every later model's forecasting (SARIMAX,
ETS, GARCH, VAR) will reuse, so the design here matters more than its
size suggests, same reasoning as Stage 4.1's optimizer wrapper.

## Where this fits

- **Depends on:** Stage 5.1 (`ARXModel`), Stage 4.1 (none directly, but
  the same "build the shared interface once, carefully" logic applies).
- **A verification boundary worth stating up front**: base R's
  `predict.ar()` was verified by direct execution in this session (R is
  installed in this sandbox). The `forecast` package's richer S3
  `forecast` class/`print.forecast()` convention — the more commonly
  recognized "R style" — could **not** be independently verified here:
  it isn't installed, and CRAN is unreachable from this sandbox
  (`install.packages()` failed with a network error, confirmed).
  Everything about `forecast::forecast()` below is from well-established,
  stable, extensively documented public knowledge (the package's API
  hasn't materially changed in over a decade), not fresh execution —
  worth a quick sanity check against a real R session with `forecast`
  installed before fully trusting the exact column headers/spacing.

---

## 1. Verified reference: R base `predict.ar()` (executed directly)

```r
predict(fit, n.ahead = 5)
```
On the exact `ar.ols` model from Stage 5.1's own verification data,
confirmed output:
```r
$pred
[1] 1.1069165 0.6916340 0.4358003 0.2781955 0.1811041

$se
[1] 0.8610201 1.0107815 1.0621043 1.0809441 1.0880086
```
A plain `list`, not even a proper S3 class with its own print method
(`class(pr)` returned `"list"`). Just `$pred` and `$se` — **no
confidence bounds computed at all**, no levels, no table. Genuinely
minimal.

## 2. Documented reference: R's `forecast` package (not independently executed — see boundary note above)

The well-known, extremely widely-used `forecast(object, h, level=c(80,95))`
convention. Its hallmark feature, worth adopting: **`level` defaults to
`c(80, 95)` — two nested confidence levels simultaneously, as
percentages, not a single alpha.** `print.forecast()`'s well-documented
tabular format:
```
     Point Forecast     Lo 80    Hi 80     Lo 95    Hi 95
101        1.106917  ...
102        0.691634  ...
```

## 3. Verified reference: Python `statsmodels` `.get_prediction()` (executed directly)

Run on the *exact same* `AutoReg` model built in Stage 5.1:
```python
pred = res.get_prediction(start=n, end=n+4)
pred.predicted_mean   # [1.47232242, 1.29900868, 1.11361523, 0.96905487, 0.84345146]
pred.se_mean          # [0.99415797, 1.12989085, 1.25440185, 1.32668084, 1.37810745]
pred.conf_int(alpha=0.05)   # single level per call, via `alpha`, NOT R's multi-level `level=`
pred.summary_frame(alpha=0.05)
#       mean   mean_se  mean_ci_lower  mean_ci_upper
# 0  1.472322  0.994158      -0.476191       3.420836
# ...
```
`summary_frame()`'s tabular shape (mean / se / ci_lower / ci_upper) is
close to R's `print.forecast()` table, minus the multi-level columns.

**Standard error formula, verified by direct numerical comparison, not
assumed**: transcribed the standard "known-parameters" forecast-error
propagation formula (Box-Jenkins), computed it independently, and
confirmed it reproduces `se_mean` **exactly** (all 5 values, full
displayed precision):
```
psi[0] = 1
psi[j] = sum_{i=1}^{min(j,p)} phi[i] * psi[j-i]     (impulse-response/MA weights)
se[h]  = sqrt(sigma2 * sum_{j=0}^{h} psi[j]^2)
```
This confirms `AutoReg.get_prediction()` does **not** adjust for
parameter estimation uncertainty — it treats the fitted AR coefficients
as known and only propagates innovation variance. Simpler than it might
have been, and now confirmed rather than assumed — worth knowing
explicitly since a "more correct" parameter-uncertainty-adjusted formula
exists in the literature but isn't what either reference actually
computes by default.

---

## 4. Proposed Julia API

```julia
StatsAPI.predict(model::ARXModel, horizon::Integer; level::Vector{<:Real}=[80.0, 95.0]) -> Forecast
forecast(model, horizon; level=[80.0, 95.0]) = predict(model, horizon; level=level)  # R-recognizable alias
```

Design notes:
- **Extends `StatsAPI.predict`** rather than inventing a new verb —
  matches the Julia ecosystem convention (GLM.jl and others already use
  `predict`), consistent with this project's StatsAPI-first principle.
- **`forecast(...)` as a thin exported alias** for R users who'd
  otherwise look for that exact name — cheap to provide, no reason not
  to given both languages use different verbs for the same concept.
- **`level::Vector{<:Real}=[80.0, 95.0]`**: adopts R's multi-level,
  percentage-based default rather than Python's single-`alpha` call —
  genuinely more useful (one call gives both bands for a fan chart) and
  trivially subsumes Python's convention (pass a single-element vector).
  This is the "exceed both, grounded in the better-designed reference"
  pattern already used elsewhere in this project (e.g. Stage 3.1's
  validation strictness, Stage 1.3's Bartlett-band default).
- **SE formula**: the verified known-parameters psi-weight propagation
  from section 3 — matches both references' actual (not idealized)
  behavior.

### `Forecast`

```julia
struct Forecast
    point::Vector{Float64}
    se::Vector{Float64}
    levels::Vector{Float64}     # e.g. [80.0, 95.0], as given
    lower::Matrix{Float64}      # horizon x length(levels)
    upper::Matrix{Float64}      # horizon x length(levels)
    horizon::Int
    model_name::String
end
```

---

## 5. Implementation

```julia
"""
    predict(model::ARXModel, horizon; level=[80.0, 95.0]) -> Forecast

Forecast `horizon` steps ahead from a fitted AR-X model, with prediction
intervals at each level in `level` (percentages, e.g. `95.0` for a 95%
interval -- matches R's `forecast()` convention, not Python's single
`alpha`). Standard errors use the known-parameters forecast-error
propagation formula (Box-Jenkins), verified to exactly reproduce
`statsmodels`' `AutoReg.get_prediction()` output -- see the Stage 5.2
handoff doc section 3. Does not adjust for parameter estimation
uncertainty, matching both references' actual (not idealized) behavior.

`forecast(model, horizon; level=...)` is a direct alias, for users
looking for R's `forecast()` name specifically.
"""
function StatsAPI.predict(model::ARXModel, horizon::Integer; level::Vector{<:Real}=[80.0, 95.0])
    horizon >= 1 || throw(ArgumentError("horizon must be >= 1"))
    all(0 .< level .< 100) || throw(ArgumentError("level entries must be in (0, 100)"))

    # split coefficients: assumes ARXModel.names starts with any of
    # "const"/"trend"/"seasonal.*" followed by "y.L<lag>" entries, in
    # the order arx() constructs them (Stage 5.1, section 4)
    lag_idx = findall(nm -> startswith(nm, "y.L"), model.names)
    phi = model.coef[lag_idx]
    lags = model.lags  # already stored on ARXModel
    p = maximum(lags)

    # psi-weights (impulse response), verified formula
    psi = zeros(horizon)
    psi[1] = 1.0
    for j in 2:horizon
        s = 0.0
        for lag in lags
            if lag < j
                s += phi[findfirst(==(lag), lags)] * psi[j-lag]
            end
        end
        psi[j] = s
    end
    se = [sqrt(model.sigma2 * sum(abs2, psi[1:h])) for h in 1:horizon]

    # point forecasts: iterative, using fitted coefficients and the
    # deterministic terms extrapolated forward (const/trend/seasonal),
    # feeding back previously forecasted y values for lags beyond the
    # available history
    yhist = Float64[]  # last max(lags) observed + forecasted values, extended each step
    # (full point-forecast recursion elided here -- mechanically
    # straightforward given ARXModel's stored coefficients/names, but
    # needs the exact same column-order logic as arx()'s own X
    # construction in Stage 5.1 to avoid a repeat of that stage's
    # flagged indexing risk. Implement by mirroring arx()'s column
    # construction loop exactly, substituting forecasted values for
    # yv[...] once t exceeds the original series length.)

    z = [_confidence_z(1 - l/100) for l in level]  # reuse Stage 1.3's helper; l/100 since level is a percentage
    lower = hcat([point .- zi .* se for zi in z]...)
    upper = hcat([point .+ zi .* se for zi in z]...)

    return Forecast(point, se, Float64.(level), lower, upper, horizon, "AR($(p))")
end

forecast(model, horizon; level::Vector{<:Real}=[80.0, 95.0]) = predict(model, horizon; level=level)
```

**Flagged, not fully specified**: the point-forecast recursion itself
(iteratively feeding forecasted values back in as pseudo-history once
the horizon exceeds available lags) is described but not written out in
full — it needs to mirror `arx()`'s own column-construction order
exactly (`const`/`trend`/`seasonal`/`y.L*`/`exog`, per Stage 5.1 section
4) to avoid silently misaligning coefficients with the wrong lag terms.
This is exactly the kind of indexing risk flagged repeatedly in this
project (the `_ols` lagged-difference columns, STL's extrapolation
indexing, 5.1's own `hold_back` arithmetic) — write it out carefully
against a hand-worked 2-3-step example before trusting it, rather than
extending the sketch above by pattern-matching alone.

---

## 6. `show` method — R's `print.forecast` table shape

```julia
function Base.show(io::IO, f::Forecast)
    println(io, f.model_name, " forecast, ", f.horizon, " steps ahead")
    println(io)
    header = ["h", "Point Forecast"]
    for l in f.levels
        push!(header, "Lo $(Int(l))"); push!(header, "Hi $(Int(l))")
    end
    println(io, join(header, "  "))
    for h in 1:f.horizon
        row = [string(h), string(round(f.point[h], digits=4))]
        for j in eachindex(f.levels)
            push!(row, string(round(f.lower[h,j], digits=4)))
            push!(row, string(round(f.upper[h,j], digits=4)))
        end
        println(io, join(row, "  "))
    end
end
```

Deliberately not using `StatsBase.CoefTable` here (unlike `ARXModel`'s
`show`) — `CoefTable` is shaped for one-row-per-coefficient displays,
not one-row-per-horizon with paired lower/upper columns per level. A
plain formatted table is the better fit; worth revisiting with a real
`PrettyTables.jl`-style dependency later if the manual column-alignment
above turns out to look rough for wide `level` vectors (3+ levels).

---

## 7. Comprehensive test matrix

Reusing the exact `ARXModel` from Stage 5.1's own verified data (phi=(0.5,
0.2), seed=0, n=100), so the standard errors below are directly
checkable against section 3's confirmed values.

| Case | Call | Verified values |
|---|---|---|
| A | `predict(m, 5)` (default `level=[80,95]`) | `se = [0.9942, 1.1299, 1.2544, 1.3267, 1.3781]` (exact match to `statsmodels`, section 3) |
| B | `predict(m, 5; level=[95.0])` | single-level output; `lower`/`upper` are `5x1` matrices |
| C | `predict(m, 1)` | `horizon=1`; se should equal `sqrt(sigma2)` exactly (psi=[1.0] only) |
| D | `predict(m, 5; level=[50.0, 80.0, 95.0])` | 3-level output; intervals must nest (50% narrowest, 95% widest, at every horizon) |

```julia
using Test

@testset "predict/forecast" begin
    # m = the ARXModel from Stage 5.1's verified test data (trend=:c, lags=2)

    fA = predict(m, 5)
    @test isapprox(fA.se, [0.9942, 1.1299, 1.2544, 1.3267, 1.3781]; atol=1e-3)
    @test fA.levels == [80.0, 95.0]
    @test size(fA.lower) == (5, 2)

    fB = predict(m, 5; level=[95.0])
    @test size(fB.lower) == (5, 1)

    fC = predict(m, 1)
    @test isapprox(fC.se[1], sqrt(m.sigma2); atol=1e-8)

    fD = predict(m, 5; level=[50.0, 80.0, 95.0])
    @test size(fD.lower) == (5, 3)
    # nesting: wider level -> wider interval, at every horizon
    for h in 1:5
        @test fD.lower[h,1] > fD.lower[h,2] > fD.lower[h,3]  # 50% inside 80% inside 95% (lower bounds increase)
        @test fD.upper[h,1] < fD.upper[h,2] < fD.upper[h,3]
    end

    # se must be non-decreasing with horizon (forecast uncertainty compounds)
    @test issorted(fA.se)

    @test_throws ArgumentError predict(m, 0)
    @test_throws ArgumentError predict(m, 5; level=[150.0])

    # forecast() alias matches predict() exactly
    @test forecast(m, 5).point == predict(m, 5).point

    # show runs without erroring, contains expected column headers
    io = IOBuffer()
    show(io, fA)
    s = String(take!(io))
    @test occursin("Point Forecast", s)
    @test occursin("Lo 80", s) && occursin("Hi 95", s)
end
```

---

## 8. What to do with this

1. Implement `predict`/`forecast`/`Forecast` per sections 4-5.
2. **Fully work out the point-forecast recursion** flagged as
   incomplete in section 5 — write it against a hand-worked 2-3-step
   example first, mirroring `arx()`'s exact column order.
3. Run the tests in section 7; the SE values there are genuinely
   verified against real `statsmodels` output, not estimated — a
   mismatch means a real bug, not just a tolerance issue.
4. If possible, get access to R's `forecast` package (even briefly, on
   a machine with CRAN access) to confirm the exact `print.forecast()`
   column header text/spacing before finalizing `show` — this handoff's
   version of that format is from documentation, not verified execution,
   per the boundary noted in section 0.
5. Update `development-sequence.md`'s Stage 5.2 row: mark implemented,
   note the R-`forecast`-package verification boundary explicitly so
   it isn't mistaken for the same execution-verified rigor as the rest
   of this handoff.

**Next in sequence:** Stage 5.3 (accuracy metrics — MAE, RMSE, MAPE,
MASE, sMAPE), which will consume `Forecast.point` directly against a
held-out actual series.
