# Handoff: Stage 5.3 — Accuracy Metrics (MAE, RMSE, MAPE, MASE, sMAPE)

Status: **done.** `mae`/`rmse`/`mape`/`smape`/`mase`/`accuracy`
(`src/accuracy.jl`, `test/test_accuracy.jl`) built per sections 4-6.
Every formula and numeric claim in this handoff was independently
re-verified by direct execution this session, not just trusted from the
prior transcription:

1. **All five `sktime` values reproduced exactly**: `mean_absolute_error`,
   `mean_squared_error(square_root=True)`,
   `mean_absolute_percentage_error(symmetric=False/True)`, and
   `mean_absolute_scaled_error` were re-run on the exact handoff test
   data (`y_train`/`y_true`/`y_pred` from section 6) and matched to full
   float precision: MAE=0.55, RMSE=0.6422616289332564,
   MAPE(frac)=0.33690476190476193, sMAPE(frac)=0.5553379953379953,
   MASE(sp=1)=0.18333333333333335. The independent seasonal case
   (`sp=4`, linear `y_train2`) also reproduced exactly: MASE=0.25.
2. **The "R could not be executed" boundary from section 2 didn't hold**
   -- same correction as Stage 5.2: R 4.6.0 is installed
   (`C:\Program Files\R\R-4.6.0`, just not on `PATH`) and CRAN is
   reachable, so `forecast::accuracy()` was run for real
   (`Rscript.exe`, full path) rather than left as documentation-only.
   Confirmed directly: R's `MAPE` column genuinely is on the percentage
   scale (`33.69048` for the section-6 test case), exactly matching
   `sktime`'s fraction (`0.33690476190476193`) times 100 -- validating
   this handoff's `as_percentage=true` R-convention default by
   execution, not just by the original documentation-based claim.
3. No discrepancies were found between this handoff's proposed
   implementation (section 5) and either reference -- unlike Stages
   5.1/5.2, nothing needed correcting; the code was implemented
   essentially as proposed, with `ArgumentError` validation added for
   mismatched `actual`/`predicted` lengths (not explicit in the
   handoff's own sketch, but consistent with `arx`/`forecast`'s existing
   error-handling conventions).
4. `smape`'s docstring carries the `!!! warning "Use with caution"`
   admonition quoting Hyndman directly, exactly as section 7 requested
   -- visible without digging, not buried in prose.

---

For a fresh Claude Code session picking this up with no prior context.
Third of five Stage 5 handoffs. **This one centers on a genuine, well-
documented scientific controversy, not just an implementation detail**:
sMAPE's own inventor's intellectual heir (Hyndman, author of both
`forecast::accuracy()` and the MASE paper) has publicly and repeatedly
recommended *against* using sMAPE at all. Worth reading section 3 before
deciding how prominently to expose it.

## Where this fits

- **Depends on:** Stage 5.2 (`Forecast.point`, consumed directly against
  a held-out actual series).
- **Verification boundary**: `sktime` was installed and its actual
  source read directly (`inspect.getsource`) for MAPE/sMAPE/MASE — high
  confidence. R's `forecast::accuracy()` itself could **not** be executed
  (not installed, CRAN unreachable, same boundary as Stage 5.2) — but the
  sMAPE finding below comes from **Hyndman's own public blog post**,
  which is arguably a stronger source than executing the function would
  have been, since it's the author directly explaining a design decision
  rather than just observing behavior.

---

## 1. Verified reference: Python `sktime.performance_metrics.forecasting`

Confirmed via `pip install sktime` (real package, real source read
directly, not docs alone):

```python
mean_absolute_error(y_true, y_pred)
mean_squared_error(y_true, y_pred, square_root=True)   # RMSE when square_root=True
mean_absolute_percentage_error(y_true, y_pred, symmetric=False, relative_to='y_true')  # MAPE
mean_absolute_percentage_error(y_true, y_pred, symmetric=True)                          # sMAPE
mean_absolute_scaled_error(y_true, y_pred, y_train=y_train, sp=1)                        # MASE
```

**Exact formulas, from source** (`_percentage_error`, the shared helper
behind both MAPE and sMAPE):
```
MAPE_i  = |y_i - yhat_i| / max(|y_i|, eps)
sMAPE_i = |y_i - yhat_i| / max((|y_i| + |yhat_i|)/2, eps)
```
Both are **fractions**, not percentages — `mean_absolute_percentage_error`
does **not** multiply by 100 internally, despite the name. Multiply by
100 yourself for the conventional "5.2%"-style number. Worth getting
right; it's an easy thing to silently omit.

**Exact MASE formula, from source** (`mean_absolute_scaled_error`):
```python
y_pred_naive = y_train[:-sp]
mae_naive = mean_absolute_error(y_train[sp:], y_pred_naive)   # in-sample seasonal-naive MAE
mae_pred  = mean_absolute_error(y_true, y_pred)                # out-of-sample MAE
MASE = mae_pred / max(mae_naive, eps)
```
`sp=1` gives the plain (non-seasonal) naive benchmark; `sp=period` gives
the seasonal-naive benchmark. **MASE genuinely needs the training series**
as a separate input from `y_true`/`y_pred` — it's not computable from the
test-set actual/predicted pair alone, unlike the other four metrics.

## 2. R reference: `forecast::accuracy()` — documented, not executed

Standard, extremely widely known columns: ME, RMSE, MAE, MPE, MAPE,
MASE, ACF1 (residual autocorrelation at lag 1), and Theil's U for
one-step out-of-sample forecasts against a random-walk benchmark. `MAPE`
here **is** reported as a percentage (unlike `sktime`'s fraction
convention) — worth getting this scaling difference right explicitly in
the Julia design rather than silently picking one.

## 3. The sMAPE finding — from Hyndman's own writing, not inference

Directly quoted from Hyndman's blog post "Errors on percentage errors"
(`robjhyndman.com/hyndsight/smape`):

> Personally, I would much prefer that either the original MAPE be used
> (when it makes sense), or the mean absolute scaled error (MASE) be
> used instead. There seems little point using the sMAPE...

And from Hyndman & Koehler (2006) itself, per an independent secondary
source's direct quote: sMAPE *"remains quite asymmetric"* despite its
name, and the authors *"explicitly recommend avoiding its use
altogether."*

**Three genuinely different published "sMAPE" formulas exist**,
confirmed from Hyndman's own post:
1. Armstrong's (1985) original "adjusted MAPE":
   `100 * mean(2|y-yhat| / (y+yhat))` -- **no absolute value on the
   denominator sum**, so it can be negative or infinite when
   `y + yhat = 0`.
2. Makridakis's (1993) "symmetric MAPE" -- close to the above, commonly
   cited with an absolute-value denominator, used in the M3/M4
   competitions.
3. `sktime`'s implementation (section 1, verified from source) -- the
   `max((|y|+|yhat|)/2, eps)` version, which avoids Armstrong's negative/
   undefined edge case entirely.

Hyndman himself: *"I can't match the published results for any
definition of sMAPE, so I'm not sure how the calculations were actually
done"* -- even the original M3 competition's own published numbers can't
be reproduced with certainty under any known formula. This is a genuinely
disputed metric, not a simple cross-language naming difference like most
earlier findings in this project.

**Design implication**: implement sMAPE (still widely requested; several
forecasting competitions use it, `sktime` supports it, users will ask for
it), matching `sktime`'s specific formula (verified, avoids the
Armstrong edge case), but with the criticism surfaced prominently in the
docstring -- not buried -- so a user reaches for MASE or plain MAPE
instead when it matters, the way the metric's own intellectual lineage
recommends.

---

## 4. Proposed Julia API

```julia
mae(actual, predicted) -> Float64
rmse(actual, predicted) -> Float64
mape(actual, predicted; as_percentage::Bool=true) -> Float64
smape(actual, predicted; as_percentage::Bool=true) -> Float64
mase(actual, predicted, train; sp::Integer=1) -> Float64
accuracy(actual, predicted, train=nothing; sp::Integer=1) -> NamedTuple
```

Design notes:
- **Separate functions, not one dispatched on a `metric=` symbol** --
  matches this project's established convention (`adf_test`/`kpss_test`,
  `convolution_filter`/`recursive_filter`) of separate functions over a
  mode switch.
- **`as_percentage::Bool=true` default** -- unlike `sktime`'s
  fraction-by-default convention, default to R's percentage convention
  (`5.2` meaning 5.2%, not `0.052`) since that's the more commonly
  expected number when someone says "MAPE" in conversation, and it's
  R's actual `accuracy()` behavior. Set `as_percentage=false` for
  `sktime`'s fraction convention directly.
- **`mase` requires `train` as a separate positional argument** --
  matches `sktime`'s `y_train` requirement exactly; it's a genuine
  mathematical necessity (section 1), not an API choice to relitigate.
- **`sp::Integer=1`** -- matches `sktime`'s naming and default (plain
  naive benchmark); pass the seasonal period for seasonal-naive scaling.
- **`accuracy(...)` convenience function** -- computes all applicable
  metrics at once (mirrors R's `accuracy()` table), returns a
  `NamedTuple` rather than a table object for simplicity; `mase`-related
  fields only populated if `train` is provided.

---

## 5. Implementation

```julia
"""
    mae(actual, predicted) -> Float64
    rmse(actual, predicted) -> Float64

Mean absolute error / root mean squared error. Both accept anything
`tsvalues` does; `actual` and `predicted` must have the same length.
"""
mae(actual, predicted) = (a = tsvalues(actual); p = tsvalues(predicted);
                           sum(abs.(a .- p)) / length(a))
rmse(actual, predicted) = (a = tsvalues(actual); p = tsvalues(predicted);
                            sqrt(sum(abs2, a .- p) / length(a)))

"""
    mape(actual, predicted; as_percentage=true) -> Float64

Mean absolute percentage error: `mean(|actual-predicted| / max(|actual|, eps))`.
`as_percentage=true` (default) scales to R's `accuracy()` convention
(e.g. `5.2` meaning 5.2%); `false` gives `sktime`'s raw-fraction
convention (`0.052`) directly.
"""
function mape(actual, predicted; as_percentage::Bool=true)
    a = tsvalues(actual); p = tsvalues(predicted)
    epsval = eps(Float64)
    v = sum(abs.(a .- p) ./ max.(abs.(a), epsval)) / length(a)
    return as_percentage ? 100*v : v
end

"""
    smape(actual, predicted; as_percentage=true) -> Float64

Symmetric MAPE: `mean(|actual-predicted| / max((|actual|+|predicted|)/2, eps))`
-- matches `sktime`'s formula exactly (one of at least three genuinely
different published "sMAPE" definitions; see the Stage 5.3 handoff doc
section 3 for why this specific one was chosen).

!!! warning "Use with caution"
    Hyndman (author of the MASE metric and R's `forecast` package) has
    publicly recommended against using sMAPE at all: *"There seems
    little point using the sMAPE... [prefer] the original MAPE... or
    MASE instead."* Provided here because it's still widely requested
    and used in forecasting competitions, not as an endorsement --
    prefer [`mase`](@ref) or [`mape`](@ref) where a choice is available.
"""
function smape(actual, predicted; as_percentage::Bool=true)
    a = tsvalues(actual); p = tsvalues(predicted)
    epsval = eps(Float64)
    v = sum(abs.(a .- p) ./ max.((abs.(a) .+ abs.(p)) ./ 2, epsval)) / length(a)
    return as_percentage ? 100*v : v
end

"""
    mase(actual, predicted, train; sp::Integer=1) -> Float64

Mean absolute scaled error: the out-of-sample MAE divided by the
in-sample MAE of the (seasonal-)naive benchmark computed on `train`.
Matches `sktime`'s `mean_absolute_scaled_error` exactly (verified from
source). Requires `train` -- the training series -- as a genuine
mathematical necessity, not an incidental API choice: MASE cannot be
computed from `actual`/`predicted` alone.

`sp=1` (default): naive benchmark (`train[t]` predicts `train[t+1]`).
`sp=period`: seasonal-naive benchmark.
"""
function mase(actual, predicted, train; sp::Integer=1)
    a = tsvalues(actual); p = tsvalues(predicted); tr = tsvalues(train)
    sp >= 1 || throw(ArgumentError("sp must be >= 1"))
    length(tr) > sp || throw(ArgumentError("train must have more than sp observations"))
    mae_naive = sum(abs.(tr[(sp+1):end] .- tr[1:(end-sp)])) / (length(tr) - sp)
    mae_pred = mae(a, p)
    return mae_pred / max(mae_naive, eps(Float64))
end

"""
    accuracy(actual, predicted, train=nothing; sp::Integer=1) -> NamedTuple

Convenience: computes MAE, RMSE, MAPE (percentage), and -- if `train` is
given -- MASE, all at once. Mirrors R's `forecast::accuracy()` table
conceptually; returns a `NamedTuple` rather than a table object.
"""
function accuracy(actual, predicted, train=nothing; sp::Integer=1)
    m = (mae=mae(actual, predicted), rmse=rmse(actual, predicted),
         mape=mape(actual, predicted))
    train === nothing && return m
    return merge(m, (mase=mase(actual, predicted, train; sp=sp),))
end
```

---

## 6. Comprehensive test matrix — verified numerically against real `sktime`

Data (from `sktime`'s own docstring example, and a second seasonal case):
```
y_train = [5, 0.5, 4, 6, 3, 5, 2]
y_true  = [3, -0.5, 2, 7, 2]
y_pred  = [2.5, 0, 2, 8, 1.25]
```

| Metric | Verified value (fraction/raw) |
|---|---|
| MAE | `0.55` |
| RMSE | `0.6422616289332564` |
| MAPE (fraction) | `0.33690476190476193` |
| sMAPE (fraction) | `0.5553379953379953` |
| MASE (sp=1) | `0.18333333333333335` |

Second case, seasonal MASE (`sp=4`, `y_train2 = 1.0:24.0`,
`y_true2=[25,27,24,30]`, `y_pred2=[24,26,25,29]`): `MASE = 0.25`.

```julia
using Test

@testset "accuracy metrics" begin
    y_train = [5.0, 0.5, 4, 6, 3, 5, 2]
    y_true  = [3.0, -0.5, 2, 7, 2]
    y_pred  = [2.5, 0.0, 2, 8, 1.25]

    @test isapprox(mae(y_true, y_pred), 0.55; atol=1e-10)
    @test isapprox(rmse(y_true, y_pred), 0.6422616289332564; atol=1e-10)
    @test isapprox(mape(y_true, y_pred; as_percentage=false), 0.33690476190476193; atol=1e-10)
    @test isapprox(smape(y_true, y_pred; as_percentage=false), 0.5553379953379953; atol=1e-10)
    @test isapprox(mase(y_true, y_pred, y_train), 0.18333333333333335; atol=1e-10)

    # as_percentage=true (Julia's default) must be exactly 100x the fraction
    @test isapprox(mape(y_true, y_pred), 100*0.33690476190476193; atol=1e-8)

    # seasonal MASE
    y_train2 = collect(1.0:24.0)
    y_true2 = [25.0, 27, 24, 30]
    y_pred2 = [24.0, 26, 25, 29]
    @test isapprox(mase(y_true2, y_pred2, y_train2; sp=4), 0.25; atol=1e-10)

    # accuracy() convenience wrapper
    acc = accuracy(y_true, y_pred, y_train)
    @test isapprox(acc.mae, 0.55; atol=1e-10)
    @test haskey(acc, :mase)
    acc_no_train = accuracy(y_true, y_pred)
    @test !haskey(acc_no_train, :mase)

    # error paths
    @test_throws ArgumentError mase(y_true, y_pred, y_train; sp=0)
    @test_throws ArgumentError mase(y_true, y_pred, [1.0, 2.0]; sp=5)  # train too short
end
```

---

## 7. What to do with this

1. Implement all five functions plus `accuracy()` per section 5.
2. Run the tests in section 6 -- these are genuine, source-verified
   numbers, not estimated; a mismatch is a real bug.
3. Get the `smape` docstring warning right -- it should be visible
   without the user having to dig, matching the seriousness of Hyndman's
   own stated objection.
4. Update `development-sequence.md`'s Stage 5.3 row: mark implemented,
   and make sure the sMAPE controversy is visible there too -- it's
   genuinely the most interesting finding of this handoff and easy to
   lose if only left in this file.

**Next in sequence:** Stage 5.4 (rolling-origin cross-validation), which
will call `accuracy()` repeatedly across expanding/sliding windows.
