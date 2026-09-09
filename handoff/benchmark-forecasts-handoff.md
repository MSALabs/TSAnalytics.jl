# Handoff: Benchmark Forecasting Methods

Adds the four benchmark forecasting methods to Stage 5. Small in code,
but it closes a real inconsistency and unblocks book chapter 22.

---

## 1. The gap, stated precisely

I have described this loosely in earlier sessions as *"naive forecasts
are missing"*. Checking the source directly gives a sharper picture,
and one correction.

**What already exists.** `mase(actual, predicted, train; sp)` computes
its own naive benchmark internally from the training data, with `sp`
selecting the seasonal period. So the *denominator logic* is present
and correct. MASE is not broken.

**What is missing.** Naive forecasting is not available as a
*forecasting method* — something a user can call, plot, put a
prediction interval around, and compare a real model against. The
package currently teaches users to write it themselves, in a docstring
example inside `tscv.jl`:

```julia
julia> naive_forecast(train, h) = fill(train[end], h);
julia> errs = tscv(y, naive_forecast; h=1, initial=10);
```

That is the inconsistency worth closing. Every accuracy comparison in
practice needs a benchmark to beat, and the package ships the metric
that requires one while asking the user to supply the method.

**The correction: there are four methods, not three.** I have
repeatedly said "naive, seasonal naive, drift". fpp3's own Table 5.2
lists **four** benchmark methods — the **mean** method is the fourth,
and it is the right benchmark for a series with no trend and no
seasonality. Implement all four.

---

## 2. Verified formulas

Taken directly from fpp3 §5.2 and §5.5, fetched this session. These are
authoritative, not recalled.

### Point forecasts

| Method | ŷ(T+h) |
|---|---|
| Mean | mean of `y` |
| Naive | `y[T]` |
| Seasonal naive | `y[T + h - m(k+1)]`, i.e. the value from the same season one or more periods back |
| Drift | `y[T] + h · (y[T] − y[1]) / (T − 1)` |

Note the drift denominator is `T − 1`, not `T`. It is the average
change per step across `T − 1` steps.

### Residual standard deviation (fpp3 eq. 5.1)

```
σ̂ = sqrt( (1 / (T − K − M)) · Σ eₜ² )
```

`K` is the number of estimated parameters, `M` the number of residuals
that cannot be computed. **This denominator differs by method and is
easy to get wrong:**

| Method | K | M | Divisor |
|---|---|---|---|
| Mean | 1 | 0 | `T − 1` |
| Naive | 0 | 1 | `T − 1` |
| Seasonal naive | 0 | m | `T − m` |
| Drift | 1 | 1 | `T − 2` |

### Multi-step forecast standard deviation (fpp3 Table 5.2)

With `m` the seasonal period and `k` the integer part of `(h − 1) / m`:

| Method | σ̂(h) |
|---|---|
| Mean | `σ̂ · sqrt(1 + 1/T)` |
| Naive | `σ̂ · sqrt(h)` |
| Seasonal naive | `σ̂ · sqrt(k + 1)` |
| Drift | `σ̂ · sqrt(h · (1 + h/(T − 1)))` |

Two details worth care. The drift expression has `T − 1` in the
denominator, matching its point-forecast formula. And the seasonal
naive standard deviation is a **step function** in `h` — constant
within each block of `m` horizons, then jumping. That is a genuinely
distinctive shape and makes a good test (section 4).

### Intervals

`ŷ(T+h) ± c · σ̂(h)`, with `c` from the normal distribution: 1.28 for
80%, 1.96 for 95%. The existing `Forecast` type already carries
`levels`, `lower` and `upper`, so this needs no new machinery.

---

## 3. API

Match the existing `Forecast` struct exactly — `point`, `se`, `levels`,
`lower`, `upper`, `horizon`, `model_name` — so these display, plot and
compare like any other forecast in the package.

```julia
mean_forecast(y, h::Integer; levels = [80.0, 95.0]) -> Forecast
naive(y, h::Integer; levels = [80.0, 95.0]) -> Forecast
seasonal_naive(y, h::Integer, m::Integer; levels = [80.0, 95.0]) -> Forecast
drift(y, h::Integer; levels = [80.0, 95.0]) -> Forecast
```

Set `model_name` to `"Mean"`, `"Naive"`, `"Seasonal naive"` and
`"Drift"` so the existing display code labels them without special
casing.

**Naming note.** `mean_forecast` rather than `mean` — `mean` would
clash with `Statistics.mean` and the collision is not worth the
brevity. Flag this in the docstring so it does not look arbitrary.

### The `tscv` integration decision

`tscv(y, fit_forecast_fn; ...)` currently calls its callback and
expects a plain `Vector` back. These functions return a `Forecast`, so
they will not slot in directly:

```julia
tscv(y, naive)                          # would fail today
tscv(y, (t, h) -> naive(t, h).point)    # works, but clumsy
```

**Recommend teaching `tscv` to accept either** — if the callback
returns a `Forecast`, use its `.point` field; otherwise treat the
return as the point forecast. Three lines, and it makes
`tscv(y, naive)` work as anyone would expect. Update the `tscv`
docstring at the same time: the inline `naive_forecast` definition in
its example should become a real call to `naive`, since that example is
currently the package's only documentation of naive forecasting.

---

## 4. Tests

### Point forecasts — exact identities, no tolerance needed

```julia
@testset "benchmark point forecasts" begin
    y = [10.0, 12, 14, 16, 18]

    @test naive(y, 3).point == [18.0, 18.0, 18.0]
    @test mean_forecast(y, 2).point == [14.0, 14.0]

    # drift: slope = (18-10)/4 = 2 per step
    @test drift(y, 3).point ≈ [20.0, 22.0, 24.0]

    ys = [1.0, 2, 3, 4, 5, 6, 7, 8]      # m = 4
    @test seasonal_naive(ys, 4, 4).point == [5.0, 6.0, 7.0, 8.0]
    @test seasonal_naive(ys, 6, 4).point == [5.0, 6.0, 7.0, 8.0, 5.0, 6.0]
end
```

### Interval shapes — the formulas made testable

```julia
@testset "naive interval grows as sqrt(h)" begin
    f = naive(randn(100), 9)
    @test f.se[4] / f.se[1] ≈ 2.0 atol=1e-10      # sqrt(4)/sqrt(1)
    @test f.se[9] / f.se[1] ≈ 3.0 atol=1e-10      # sqrt(9)/sqrt(1)
end

@testset "seasonal naive interval is a step function in h" begin
    f = seasonal_naive(randn(100), 8, 4)
    @test f.se[1] == f.se[2] == f.se[3] == f.se[4]     # k = 0 throughout
    @test f.se[5] == f.se[8]                            # k = 1 throughout
    @test f.se[5] > f.se[4]                             # jumps between blocks
    @test f.se[5] / f.se[1] ≈ sqrt(2) atol=1e-10
end

@testset "residual sd divisors follow eq. 5.1" begin
    # naive divides by T-1, drift by T-2 -- confirm they differ
    # in the documented direction on identical data
end
```

### The strongest test — an exact tie to existing code

MASE divides mean absolute error by the mean absolute error of the
naive forecast. So **the MASE of the naive forecast is exactly 1**, by
construction. This links the new functions to `mase` with no tolerance
fudging:

```julia
@testset "MASE of the naive forecast is exactly 1" begin
    y = cumsum(randn(60)) .+ 100
    train, test = y[1:50], y[51:60]

    f = naive(train, 10)
    @test mase(test, f.point, train; sp=1) != 1.0   # a real forecast, not 1

    # in-sample: naive one-step-ahead against the series itself
    onestep = train[1:end-1]
    @test mase(train[2:end], onestep, train; sp=1) ≈ 1.0 atol=1e-12
end
```

Do the same for `seasonal_naive` with `sp = m`. If either fails, the
benchmark logic inside `mase` and the new standalone methods have
diverged, which is exactly the bug this test exists to catch.

### Integration

```julia
@testset "benchmarks slot into tscv directly" begin
    y = cumsum(randn(80)) .+ 50
    for f in (naive, drift, mean_forecast)
        errs = tscv(y, f; h=1, initial=20)
        @test length(errs) > 0
    end
end
```

---

## 5. Out of scope, deliberately

**Bootstrapped prediction intervals.** fpp3 §5.5 covers these as an
alternative when normality is unreasonable — simulate future paths by
resampling residuals, then take percentiles. Genuinely useful, and a
natural companion to these four methods, but it is separate work with
its own design questions (how many paths, seeding, reproducibility).
Note it as a follow-up rather than folding it in here.

---

## 6. Checklist

- [ ] All **four** methods, including mean
- [ ] `Forecast` struct returned, `model_name` set for each
- [ ] Residual sd divisor correct per method (`T−1`, `T−1`, `T−m`, `T−2`)
- [ ] Interval formulas match fpp3 Table 5.2 exactly, including the
      `T−1` in drift
- [ ] Seasonal naive `se` is a step function, not smooth
- [ ] MASE-equals-one test passes for naive and seasonal naive
- [ ] `tscv` accepts a `Forecast`-returning callback
- [ ] `tscv` docstring updated to call `naive` rather than define it inline
- [ ] All four exported
- [ ] British-Indian spelling in docstrings, per the writing guide

---

## 7. What to do

1. Implement in `src/forecast.jl` alongside the existing `Forecast`
   machinery — not a new file; these are forecasting methods, not a
   separate concern.
2. Make the `tscv` change and update its docstring.
3. Run the test matrix. The MASE-equals-one test is the one that
   actually proves correctness rather than shape.
4. Update `development-sequence.md`: Stage 5 gains benchmark methods,
   and note the four-not-three correction so the earlier framing is not
   carried forward again.
5. Book chapter 22.4 becomes writable once this lands.
