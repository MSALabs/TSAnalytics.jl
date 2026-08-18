# Handoff: Stage 5.4 — Rolling-Origin Cross-Validation

For a fresh Claude Code session picking this up with no prior context.
Fourth of five Stage 5 handoffs. **R and Python solve this with
genuinely different-shaped APIs**, not just different names for the same
thing — R's `tsCV()` is a "batteries included" function that takes your
forecasting function and hands back an error matrix directly; Python's
splitters just hand back train/test indices and leave metric computation
to you. Worth designing for both shapes rather than picking one, same
approach as the Ljung-Box/Box-Pierce dual-return design in Stage 2.4.

## Where this fits

- **Depends on:** Stage 5.3 (`accuracy`/`mae`/etc., consumed by the
  higher-level convenience wrapper).
- **Verification boundary**: `sktime`'s splitters were run directly in
  this session with real index output (already installed from Stage
  5.3). R's `forecast::tsCV()` could **not** be executed — same CRAN
  boundary as Stages 5.2/5.3 — documented from its long-stable,
  well-known API (unchanged since ~2017) rather than fresh execution.

---

## 1. Verified reference: Python `sktime.split`

Two classes, confirmed via `inspect.signature` and real runs on
`y = 0:19` (20 observations):

```python
ExpandingWindowSplitter(fh=1, initial_window=10, step_length=1)
SlidingWindowSplitter(fh=1, window_length=10, step_length=1, initial_window=None, start_with_window=True)
```

Confirmed real split behavior:
- **`ExpandingWindowSplitter`**: training window **grows** each fold —
  fold 0: `train` has 10 points (indices 0-9), `test=[10]`; fold 1:
  `train` has 11 points, `test=[11]`; ... fold 9: `train` has 19 points,
  `test=[19]`.
- **`SlidingWindowSplitter`**: training window **stays fixed size**
  (`window_length=10`) and slides forward — fold 0: `train=[0..9]`,
  `test=[10]`; fold 1: `train=[1..10]`, `test=[11]` (note index 0
  dropped).
- **`fh` can be a list**, e.g. `fh=[1,2,3]`, for multi-horizon
  evaluation in one fold: confirmed `test=[10,11,12]` when
  `fh=[1,2,3]`, all from a single 10-point training window.
- **`step_length`** controls how many observations to advance the
  origin between folds — confirmed `step_length=2` produces half as
  many folds, with test origins `[10], [12], [14], ...` instead of
  every consecutive point.

## 2. Documented reference: R `forecast::tsCV()` — not executed, well-established API

```r
tsCV(y, forecastfunction, h = 1, window = NULL, xreg = NULL, initial = 0, ...)
```

- `forecastfunction`: **user-supplied** — a function taking a training
  series (and `h`) and returning a forecast object. This is the key
  structural difference from `sktime`: R's `tsCV` owns the entire
  fit-forecast-score loop internally, model-agnostic by construction,
  rather than just handing back indices for the caller to loop over.
- `h`: forecast horizon (can request multiple steps; returns a
  multi-column error matrix, one column per horizon).
- `window`: `NULL` (default) → **expanding** window using all data from
  the start — matches `ExpandingWindowSplitter`'s behavior conceptually.
  A fixed integer → **sliding** fixed-size window — matches
  `SlidingWindowSplitter`.
- `initial`: number of observations to skip before the first possible
  origin — same concept as `initial_window`, different name.
- **Return value**: a matrix of forecast **errors** (`actual -
  forecast`), one row per time point, one column per horizon, `NA`
  where a fold couldn't be computed. Not indices, not predictions — R's
  `tsCV` computes the errors for you, ready for e.g. `sqrt(colMeans(e^2,
  na.rm=TRUE))` for RMSE per horizon.

---

## 3. Proposed Julia API — both shapes, not one

### Low-level: index splitters, matching `sktime`'s flexible shape

```julia
expanding_window_split(n::Integer; initial_window::Integer, step_length::Integer=1, fh::Union{Integer,AbstractVector{<:Integer}}=1)
sliding_window_split(n::Integer; window_length::Integer, step_length::Integer=1, fh::Union{Integer,AbstractVector{<:Integer}}=1)
```
Both return a `Vector{Tuple{Vector{Int}, Vector{Int}}}` — `(train_idx,
test_idx)` pairs, 1-indexed, directly mirroring `sktime`'s confirmed
behavior (section 1) translated to Julia's indexing convention.

### High-level: `tscv`, matching R's "batteries included" shape

```julia
tscv(y, fit_forecast_fn; h::Union{Integer,AbstractVector{<:Integer}}=1,
     window::Union{Nothing,Integer}=nothing, initial::Integer=0, step_length::Integer=1)
```
Matches R's argument names directly (`h`, `window`, `initial`) since
this is the function playing R's exact role — `window=nothing` (default)
gives expanding, matching R's `window=NULL` default exactly; an integer
gives sliding. `fit_forecast_fn(train_data, h)` is the user-supplied
callable (matches R's `forecastfunction`), expected to return a vector
of `h` (or `length(h)`) point forecasts.

Design notes:
- **Both shapes exist because they serve different needs**, not because
  of indecision: the low-level splitters compose with anything (custom
  metrics, non-Julia-native models, inspecting the folds themselves);
  `tscv` is the fast path when you just want an error matrix from a
  simple model+forecast callable, matching what most R users actually
  reach for.
- `tscv` is implemented **on top of** the low-level splitters
  internally, not as a separate parallel implementation — avoids the
  two ever silently disagreeing about fold boundaries.
- Returns errors (`actual .- forecast`), matching R's return convention
  exactly, as a matrix (`n_folds x length(h)`, `NaN` where a fold
  wasn't computable) rather than a nested structure — directly usable
  with `sqrt(mean(skipmissing(e.^2)))`-style downstream code, same as R.

---

## 4. Implementation

```julia
"""
    expanding_window_split(n; initial_window, step_length=1, fh=1)

Generate `(train_idx, test_idx)` pairs for expanding-window
cross-validation: the training window grows by `step_length` each fold,
starting from `initial_window` observations. Matches `sktime`'s
`ExpandingWindowSplitter` exactly (verified against real index output --
see Stage 5.4 handoff doc section 1) and corresponds to R's
`forecast::tsCV(window=NULL)` (the default).

`fh`: an `Integer` (single horizon) or `AbstractVector` (multiple
horizons evaluated per fold, e.g. `fh=[1,2,3]`).
"""
function expanding_window_split(n::Integer; initial_window::Integer,
                                 step_length::Integer=1, fh::Union{Integer,AbstractVector{<:Integer}}=1)
    horizons = fh isa Integer ? [fh] : collect(fh)
    max_h = maximum(horizons)
    folds = Tuple{Vector{Int},Vector{Int}}[]
    origin = initial_window
    while origin + max_h <= n
        train_idx = collect(1:origin)
        test_idx = [origin + h for h in horizons]
        push!(folds, (train_idx, test_idx))
        origin += step_length
    end
    return folds
end

"""
    sliding_window_split(n; window_length, step_length=1, fh=1)

Generate `(train_idx, test_idx)` pairs for sliding (fixed-size) window
cross-validation. Matches `sktime`'s `SlidingWindowSplitter` exactly
(verified) and corresponds to R's `forecast::tsCV(window=<integer>)`.
"""
function sliding_window_split(n::Integer; window_length::Integer,
                               step_length::Integer=1, fh::Union{Integer,AbstractVector{<:Integer}}=1)
    horizons = fh isa Integer ? [fh] : collect(fh)
    max_h = maximum(horizons)
    folds = Tuple{Vector{Int},Vector{Int}}[]
    start = 1
    while start + window_length - 1 + max_h <= n
        train_idx = collect(start:(start + window_length - 1))
        test_idx = [start + window_length - 1 + h for h in horizons]
        push!(folds, (train_idx, test_idx))
        start += step_length
    end
    return folds
end

"""
    tscv(y, fit_forecast_fn; h=1, window=nothing, initial=0, step_length=1) -> Matrix{Float64}

Rolling-origin cross-validation, R-`forecast::tsCV`-style: for each
fold, `fit_forecast_fn(train_data, h)` is called and its point
forecast(s) compared against the actual held-out value(s), returning a
matrix of errors (`actual - forecast`), one row per fold, one column per
requested horizon in `h` -- matches R's return convention exactly.
`window=nothing` (default) is expanding-window (matches R's
`window=NULL` default); an `Integer` gives a fixed-size sliding window.
`initial` skips that many observations before the first possible origin
(matches R's argument name and role exactly).

Built on `expanding_window_split`/`sliding_window_split` internally.
"""
function tscv(y, fit_forecast_fn; h::Union{Integer,AbstractVector{<:Integer}}=1,
              window::Union{Nothing,Integer}=nothing, initial::Integer=0, step_length::Integer=1)
    yv = tsvalues(y)
    n = length(yv)
    horizons = h isa Integer ? [h] : collect(h)

    folds = if window === nothing
        expanding_window_split(n; initial_window=max(1, initial), step_length=step_length, fh=horizons)
    else
        sliding_window_split(n; window_length=window, step_length=step_length, fh=horizons)
    end

    errors = fill(NaN, length(folds), length(horizons))
    for (i, (train_idx, test_idx)) in enumerate(folds)
        train_data = yv[train_idx]
        fc = fit_forecast_fn(train_data, maximum(horizons))
        for (j, hh) in enumerate(horizons)
            errors[i, j] = yv[test_idx[j]] - fc[hh]
        end
    end
    return errors
end
```

---

## 5. `show`/print

No dedicated `show` method proposed here — `tscv`'s return is a plain
`Matrix{Float64}` (matching R's plain-matrix return exactly), and the
low-level splitters return plain vectors of tuples. Both display fine
via Julia's default `Matrix`/`Vector` formatting; adding a custom `show`
would just be reformatting a plain numeric matrix for no real gain,
unlike `ARXModel`/`Forecast` where the structured fields genuinely
benefited from a custom layout. Consistent with the Stage 4.1 decision
to leave `OptimResult` unadorned for the same reason — not every result
type needs one.

---

## 6. Comprehensive test matrix — verified against real `sktime` index output

Using `n=20` (indices 0-19 in the verification run; **+1 for Julia's
1-indexing** in the assertions below):

| Case | Call | Verified behavior |
|---|---|---|
| A | `expanding_window_split(20; initial_window=10)` | fold 0: `train=1:10, test=[11]`; fold 1: `train=1:11, test=[12]`; ... 10 folds total, last: `train=1:19, test=[20]` |
| B | `sliding_window_split(20; window_length=10)` | fold 0: `train=1:10, test=[11]`; fold 1: `train=2:11, test=[12]` (window slides, doesn't grow) |
| C | `expanding_window_split(20; initial_window=10, fh=[1,2,3])` | fold 0: `train=1:10, test=[11,12,13]` |
| D | `expanding_window_split(20; initial_window=10, step_length=2)` | half as many folds; test origins `11, 13, 15, ...` |

```julia
using Test

@testset "cross-validation splitters" begin
    # A: expanding, matches sktime's confirmed fold 0/1 exactly (+1 for 1-indexing)
    foldsA = expanding_window_split(20; initial_window=10)
    @test foldsA[1] == (collect(1:10), [11])
    @test foldsA[2] == (collect(1:11), [12])
    @test length(foldsA) == 10
    @test foldsA[end] == (collect(1:19), [20])

    # B: sliding, window stays fixed size, confirmed fold 0/1 exactly
    foldsB = sliding_window_split(20; window_length=10)
    @test foldsB[1] == (collect(1:10), [11])
    @test foldsB[2] == (collect(2:11), [12])
    @test all(f -> length(f[1]) == 10, foldsB)  # every training window is exactly size 10

    # C: multi-horizon fh
    foldsC = expanding_window_split(20; initial_window=10, fh=[1,2,3])
    @test foldsC[1] == (collect(1:10), [11,12,13])

    # D: step_length
    foldsD = expanding_window_split(20; initial_window=10, step_length=2)
    @test length(foldsD) == 5
    @test [f[2][1] for f in foldsD] == [11, 13, 15, 17, 19]

    # tscv end-to-end with a trivial "naive" forecaster (forecast = last training value, repeated)
    y = collect(1.0:20.0)
    naive_forecast(train, h) = fill(train[end], h)
    errs = tscv(y, naive_forecast; h=1, window=nothing, initial=10)
    @test size(errs) == (10, 1)
    # naive forecast of a linear trend: error should be constant = step size = 1.0
    @test all(isapprox.(errs, 1.0; atol=1e-10))

    # sliding-window tscv
    errs_sliding = tscv(y, naive_forecast; h=1, window=10)
    @test all(isapprox.(errs_sliding, 1.0; atol=1e-10))

    # multi-horizon tscv
    errs_multi = tscv(y, (train,h) -> fill(train[end], h); h=[1,2,3], initial=10)
    @test size(errs_multi, 2) == 3
end
```

---

## 7. What to do with this

1. Implement `expanding_window_split`/`sliding_window_split`/`tscv` per
   sections 3-4.
2. Run the tests in section 6 — the fold-boundary assertions are exact,
   verified against real `sktime` output, not estimated.
3. No `show` method needed per section 5 — confirm this holds once
   actually used in practice; revisit only if plain matrix output turns
   out to be genuinely hard to read for wide multi-horizon cases.
4. Update `development-sequence.md`'s Stage 5.4 row: mark implemented,
   note the dual-shape design (low-level splitters + R-style `tscv`
   convenience wrapper) explicitly, since it's a real design decision
   worth being visible, not just an implementation detail.

**Next in sequence:** Stage 5.5 (classical exponential smoothing —
Simple/Holt/Holt-Winters), the last of the five Stage 5 handoffs.
