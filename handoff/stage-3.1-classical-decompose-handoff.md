# Handoff: Stage 3.1 — Classical Decomposition (comprehensive, R+Python dual-verified)

Status: implemented ✅ (`src/decompose.jl`, `test/test_decompose.jl`). The
dual-verification claim itself was independently re-checked, not just
trusted: reran the full core matrix (A/B/C), the Python-exclusive
features (F/G/H), and the validation behaviors (I/J) directly against
real `statsmodels` using the actual `data/monthly.csv`/`mult_monthly.csv`
/`period7.csv` files this handoff shipped -- confirmed the CSVs
themselves exactly reproduce the described `numpy` generation script
(same continued-RNG-state trick across all three), then confirmed every
number in the verification transcript to full precision.

**One real bug found in this doc's own draft, before writing any
Julia**: section 4's prose claims `_extrapolate_trend`'s index arithmetic
is `front_last = min(front + npoints - 1, back)` (a hand-adjustment for
Julia's 1-indexing). Reading `statsmodels.tsa.seasonal._extrapolate_trend`'s
actual source directly (not its docstring) shows the real formula is
`min(front + npoints, back)` -- no `-1`. The `-1` adjustment isn't just
wrong, it's the wrong KIND of fix: naively shifting a constant doesn't
correctly translate 0-indexed slice-exclusive arithmetic to 1-indexed
inclusive arithmetic in general (it specifically breaks in the edge case
where `npoints` exceeds the available `back-front` span, giving one extra
point). The implemented `_extrapolate_trend!` instead keeps the
regression itself in Python's exact 0-indexed coordinate system
(identical `k`/`n` fit to the source, by construction) and translates
only the storage-index reads/writes to Julia's 1-based convention --
verified against the full-precision G/H trend arrays from this doc's own
verification transcript, and against a hand-constructed perfectly-linear
test case (extrapolating a linear sequence must reproduce the same line
exactly, a self-consistent sanity check independent of the derivation).

Test E/E2 (multiplicative, period=7) use a purely deterministic formula
with no `randn` call at all, so they're constructed inline in the test
file rather than needing a CSV -- no cross-language RNG risk since
there's no randomness involved for that specific series.

No other deltas from the proposed implementation -- section 4's
`classical_decompose` body, the API surface, and the `filt`/`figure`/
`seasonal` construction all matched on the first test run (all 214 tests
passing immediately, including all 8 core-matrix combinations at full
precision).

For a fresh Claude Code session picking this up with no prior context.
This is a deeper treatment than the Stage 1–2 handoffs: **every algorithm
claim below was verified by actually running R and Python in a sandbox
on identical shared data**, not by reading documentation alone. R's
`stats::decompose()` and Python's `statsmodels.tsa.seasonal.
seasonal_decompose()` source code were both read directly (`print()` in
R, `inspect.getsource()` in Python) and their outputs cross-checked
against each other on the same series. Where they diverge, it's called
out explicitly with real numbers from both sides — not assumed.

## Goal, per the request this responds to

Match or exceed R and Python's combined capability, with argument names
comfortable to users of either, and a test suite covering every
meaningfully distinct argument combination with **verified values from
both R and Python** wherever both support that combination — and an
honest note where only one does.

## Where this fits

- **Depends on:** Stage 1.2 (`convolution_filter`, `moving_average` —
  already built, reused directly here).
- **New territory:** first Stage 3 item; nothing built here before.

---

## 1. Verified reference: R `stats::decompose()` — exact source

```r
decompose(x, type = c("additive", "multiplicative"), filter = NULL)
```

Full source (printed directly via `print(stats::decompose)`):
```r
function (x, type = c("additive", "multiplicative"), filter = NULL) {
    type <- match.arg(type)
    l <- length(x)
    f <- frequency(x)
    if (f <= 1 || length(na.omit(x)) < 2 * f)
        stop("time series has no or less than 2 periods")
    if (is.null(filter))
        filter <- if (!f%%2) c(0.5, rep_len(1, f - 1), 0.5)/f
                  else rep_len(1, f)/f
    trend <- filter(x, filter)
    season <- if (type == "additive") x - trend else x/trend
    periods <- l%/%f
    index <- seq.int(1L, l, by = f) - 1L
    figure <- numeric(f)
    for (i in 1L:f) figure[i] <- mean(season[index + i], na.rm = TRUE)
    figure <- if (type == "additive") figure - mean(figure) else figure/mean(figure)
    seasonal <- ts(rep(figure, periods + 1)[seq_len(l)], start = start(x), frequency = f)
    structure(list(x = x, seasonal = seasonal, trend = trend,
        random = if (type == "additive") x - seasonal - trend else x/seasonal/trend,
        figure = figure, type = type), class = "decomposed.ts")
}
```

Key points:
- Default filter: even `f` -> `[0.5, 1,...,1, 0.5]/f` (the classic "2xf"
  double moving average -- **this is exactly `moving_average(x, f;
  centre=true)`, already built in Stage 1.2**). Odd `f` -> uniform
  `[1/f,...,1/f]`.
- Trend via R's own `filter()` -- always **centered** (`sides=2`, R's
  default), no option to use a one-sided filter at all.
- Seasonal figure: average the detrended series at each of the `f`
  within-period positions across all available cycles, `na.rm=TRUE`.
- Normalize: subtract mean (additive) or divide by mean (multiplicative)
  so the figure sums to zero / averages to one.
- No trend extrapolation at all -- edges stay `NA`/propagate to `NA` in
  `random` wherever `trend` is `NA`.

**Corrected assumption, found only by actually running it**: reading the
source, the `na.rm=TRUE` in the figure-averaging step suggested R
tolerates missing data gracefully. It doesn't -- verified directly:
```r
y[5] <- NA
decompose(ts(y, frequency=12))
# Error in na.omit.ts(x) : time series contains internal NAs
```
R rejects internal `NA`s too, just earlier in the function (the
`length(na.omit(x)) < 2*f` check itself triggers `na.omit.ts`'s
internal-NA rejection) and with a different error message than Python's.
**Both languages reject missing data** -- worth knowing this was a wrong
first-read assumption, corrected by execution rather than left in.

## 2. Verified reference: Python `statsmodels.tsa.seasonal.seasonal_decompose()` -- exact source

```python
seasonal_decompose(x, model='additive', filt=None, period=None,
                    two_sided=True, extrapolate_trend=0)
```

Full source read via `inspect.getsource`; key logic:
```python
if not np.all(np.isfinite(x)):
    raise ValueError("This function does not handle missing values")
if model.startswith("m") and np.any(x <= 0):
    raise ValueError("Multiplicative seasonality is not appropriate "
                      "for zero and negative values")
if filt is None:
    filt = ([0.5]+[1]*(period-1)+[0.5])/period if period%2==0 else [1/period]*period
nsides = int(two_sided) + 1
trend = convolution_filter(x, filt, nsides)
if extrapolate_trend == "freq":
    extrapolate_trend = period - 1
if extrapolate_trend > 0:
    trend = _extrapolate_trend(trend, extrapolate_trend + 1)
detrended = x/trend if model.startswith("m") else x-trend
period_averages = seasonal_mean(detrended, period)   # nanmean per position
period_averages /= or -= mean(period_averages)         # normalize
seasonal = tile(period_averages, ...)[:nobs]
resid = x/seasonal/trend  or  detrended - seasonal
```

**Default filter formula is identical to R's** (verified: same even/odd
branching, same coefficients). **Two genuine extra capabilities R has no
equivalent for**:
- `two_sided`: `False` uses a **causal** (one-sided) filter for trend
  (`nsides=1` -> `convolution_filter`'s `sides=1`, already built). R's
  `filter()` call is always centered; there's no way to ask R's
  `decompose()` for a one-sided trend at all.
- `extrapolate_trend`: `> 0` (int) or `'freq'` -> linear-least-squares
  extrapolation of the trend's edges using the `n+1` closest defined
  points, eliminating the edge `NaN`s entirely. Exact algorithm (from
  `_extrapolate_trend`'s source, verified):
  ```
  front = index of first non-NaN trend value
  back  = index of last non-NaN trend value
  front_last = min(front + npoints, back)
  fit OLS line to trend[front:front_last] vs their indices
  extrapolate trend[:front] from that line
  (symmetric for the back end)
  ```
  `extrapolate_trend='freq'` is shorthand for `extrapolate_trend =
  period - 1`.

Also stricter validation than R: explicitly rejects non-finite `x`
(matches R's behavior once actually tested, per section 1's correction)
and explicitly rejects non-positive `x` under the multiplicative model
(verified: R's `decompose()` has **no such check** -- it would silently
divide by non-positive trend values and produce `Inf`/`NaN`/garbage
rather than erroring. This is a real robustness gap in R worth not
inheriting.)

---

## 3. Proposed Julia API

```julia
classical_decompose(x, period::Integer; model::Symbol=:additive,
                     filt::Union{Nothing,AbstractVector{<:Real}}=nothing,
                     two_sided::Bool=true,
                     extrapolate_trend::Union{Int,Symbol}=0)
```

Design notes:
- **`model`**: matches Python's argument name and values (`:additive`/
  `:multiplicative`) -- R's `type` is a synonym worth noting in the
  docstring but not adopted as the primary name, for consistency with
  this package's existing convention (`adf_test`/`kpss_test`/`pp_test`
  all already use Python-style names where the two references differ).
- **`period`**: required positional, matching Python's requirement
  exactly (Python only infers it from a pandas frequency, which has no
  Julia analogue given the container-agnostic design -- always requiring
  it explicitly is simpler and matches what a non-pandas Python call
  needs anyway).
- **`filt`**: matches Python's name exactly; `nothing` (default)
  computes the same even/odd formula both references use.
- **`two_sided`**: matches Python's name/default exactly. **No R
  equivalent** -- R is always effectively `two_sided=true`.
- **`extrapolate_trend`**: matches Python's name/default/values exactly
  (`Int` or `:freq` -- using a Julia `Symbol` instead of Python's string
  `'freq'`). **No R equivalent** -- R is always effectively
  `extrapolate_trend=0`.
- **Validation**: adopt Python's stricter checks (reject non-finite `x`,
  reject non-positive `x` under `:multiplicative`) rather than R's more
  permissive-until-it-silently-breaks behavior -- this is a case of
  "exceed both, don't just match," per the request's actual goal.

### Return type

```julia
struct ClassicalDecomposition
    observed::Vector{Float64}
    trend::Vector{Float64}      # NaN at edges unless extrapolate_trend > 0
    seasonal::Vector{Float64}
    resid::Vector{Float64}
    figure::Vector{Float64}     # one-period seasonal figure, length == period
    model::Symbol
    period::Int
end
```

---

## 4. Implementation

```julia
"""
    classical_decompose(x, period; model=:additive, filt=nothing,
                         two_sided=true, extrapolate_trend=0)

Classical (moving-average) seasonal decomposition. Matches both R's
`stats::decompose()` and Python's `statsmodels.tsa.seasonal.
seasonal_decompose()` exactly when `two_sided=true` and
`extrapolate_trend=0` (R's only supported mode) -- verified by running
both on identical data; see the Stage 3.1 handoff doc for the full
dual-verified test matrix. `two_sided=false` and `extrapolate_trend`
extend beyond R's capability, matching Python's.

- `model`: `:additive` or `:multiplicative`.
- `filt`: custom trend filter coefficients; `nothing` (default) computes
  the standard "2xperiod" filter for even `period`, uniform for odd --
  identical formula in both references, and identical to
  `moving_average`'s own default (Stage 1.2).
- `two_sided`: `true` (default, matches both references when unset in R)
  uses a centered trend filter; `false` uses a causal (past-only) one --
  Python-only capability, no R equivalent.
- `extrapolate_trend`: `0` (default) leaves edge `NaN`s, matching R's
  only behavior. An `Int > 0` or `:freq` linearly extrapolates the
  trend's edges -- Python-only capability, no R equivalent.

Validation is stricter than R's: rejects non-finite input and
non-positive input under `:multiplicative` (matching Python; R has
neither check and would silently produce `Inf`/`NaN` instead).
"""
function classical_decompose(x, period::Integer; model::Symbol=:additive,
                              filt::Union{Nothing,AbstractVector{<:Real}}=nothing,
                              two_sided::Bool=true,
                              extrapolate_trend::Union{Int,Symbol}=0)
    model in (:additive, :multiplicative) ||
        throw(ArgumentError("model must be :additive or :multiplicative"))
    y = tsvalues(x)
    n = length(y)
    any(isnan, y) && throw(ArgumentError(
        "classical_decompose: missing values not supported (verified to match both " *
        "R's and statsmodels' actual behavior, not just Python's)"))
    period >= 2 || throw(ArgumentError("period must be >= 2"))
    n >= 2*period || throw(ArgumentError(
        "need at least 2 full periods ($(2*period) observations), got $n"))
    if model == :multiplicative
        all(>(0), y) || throw(ArgumentError(
            "multiplicative model requires strictly positive values " *
            "(R's decompose() has no such check and would silently misbehave; this does not)"))
    end

    f = filt === nothing ?
        (iseven(period) ? vcat(1/(2period), fill(1/period, period-1), 1/(2period)) :
                           fill(1/period, period)) :
        filt

    sides = two_sided ? 2 : 1
    trend = convolution_filter(y, f; sides=sides)

    ep = extrapolate_trend === :freq ? period - 1 : extrapolate_trend
    if ep isa Integer && ep > 0
        trend = _extrapolate_trend!(trend, ep + 1)
    end

    detrended = model == :additive ? y .- trend : y ./ trend

    figure = zeros(period)
    for i in 1:period
        idxs = i:period:n
        vals = filter(!isnan, detrended[idxs])
        figure[i] = sum(vals) / length(vals)
    end
    figure = model == :additive ? figure .- sum(figure)/period : figure ./ (sum(figure)/period)

    seasonal = [figure[mod1(i, period)] for i in 1:n]

    resid = model == :additive ? detrended .- seasonal : y ./ seasonal ./ trend

    return ClassicalDecomposition(y, trend, seasonal, resid, figure, model, period)
end

"""_extrapolate_trend!(trend, npoints) -- linear-least-squares
extrapolation of NaN edges, using the `npoints` closest defined values on
each side. Transcribed directly from statsmodels' `_extrapolate_trend`
source (read via `inspect.getsource`, not re-derived), verified
numerically to reproduce its output on real data -- see handoff doc
section 5, test G/H."""
function _extrapolate_trend!(trend::Vector{Float64}, npoints::Int)
    n = length(trend)
    front = findfirst(!isnan, trend)
    back = findlast(!isnan, trend)
    front === nothing && throw(ArgumentError("trend is entirely NaN"))

    front_last = min(front + npoints - 1, back)
    idx_f = front:front_last
    Xf = hcat(collect(Float64, idx_f), ones(length(idx_f)))
    kf, nf = Xf \ trend[idx_f]
    for i in 1:(front-1)
        trend[i] = kf*i + nf
    end

    back_first = max(front, back - npoints + 1)
    idx_b = back_first:back
    Xb = hcat(collect(Float64, idx_b), ones(length(idx_b)))
    kb, nb = Xb \ trend[idx_b]
    for i in (back+1):n
        trend[i] = kb*i + nb
    end
    return trend
end
```

**Note on indexing**: Python's `_extrapolate_trend` is 0-indexed;
`front`/`back`/`npoints` arithmetic above was carefully re-derived for
Julia's 1-indexing (`front_last = min(front + npoints - 1, back)` rather
than a naive `front + npoints`) -- worth double-checking against the
Python source line-by-line during implementation rather than trusting
this transcription blindly, given how easy off-by-one errors are in this
exact kind of index arithmetic (see the original `_ols` handoff for the
same category of risk).

---

## 5. Comprehensive dual-verified test matrix

Every combination below was run through **both** real R (`Rscript`,
R 4.3.3) and real Python (`statsmodels`) on identical shared data in this
session -- not transcribed from documentation. Datasets (regenerate with
these exact seeds for reproducibility):

```python
# monthly.csv: n=48, period=12 (even), additive-appropriate
np.random.seed(42); t=np.arange(48)
y1 = (100+0.5*t) + 10*np.sin(2*np.pi*t/12) + np.random.normal(0,1,48)

# period7.csv: n=42, period=7 (odd), additive-appropriate
t=np.arange(42)
y2 = (50+0.3*t) + 5*np.sin(2*np.pi*t/7) + np.random.normal(0,0.5,42)

# mult_monthly.csv: n=48, period=12, multiplicative-appropriate (all positive)
t=np.arange(48)
y3 = (100+1.0*t) * (1+0.15*np.sin(2*np.pi*t/12)) * (1+np.random.normal(0,0.01,48))

# period=7 multiplicative series (R test E / E2), seed=1:
t=np.arange(42); y5 = (50+0.3*t) * (1+0.1*np.sin(2*np.pi*t/7))
```

### Core matrix -- model x period parity x filter type (both R and Python verified, 8 combinations)

| Test | model | period | filt | R trend (sample) | Python trend (same indices) | Match? |
|---|---|---|---|---|---|---|
| A | additive | 12 (even) | default | `[19:22] approx 108.928,109.536,110.099` | identical to 10 digits | yes |
| B | multiplicative | 12 (even) | default | `[19:22] approx 119.054,120.220,121.273` | identical to 10 digits | yes |
| C | additive | 7 (odd) | default | `[17:20] approx 55.183,55.516,55.927` | identical to 10 digits | yes |
| D | additive | 12 (even) | custom uniform `[1/12,...]` | `[19:22] approx 109.262,109.810,110.388` | identical to 10 digits | yes |
| E | multiplicative | 7 (odd) | default | `[17:20] approx 55.069,55.369,55.692` | identical to 10 digits | yes |
| C2 | additive | 7 (odd) | custom uniform `[1/7,...]` | identical to C (odd-period default *is* uniform) | identical to C | yes (degenerate, worth noting) |
| B2 | multiplicative | 12 (even) | custom uniform `[1/12,...]` | `[19:22] approx 119.649,120.792,121.755` | identical to 10 digits | yes |
| E2 | multiplicative | 7 (odd) | custom uniform | identical to E (same degenerate case) | identical to E | yes |

Full seasonal figures also matched exactly (all 12 or 7 values, both
languages) for every row above -- truncated in this table for space; the
raw session transcript has the complete vectors if needed for exact test
assertions.

**Note on C2/E2**: for odd periods, the default filter *is* the uniform
filter -- passing a uniform custom `filt` for an odd period is
mathematically a no-op, not a meaningfully distinct test case. Included
for completeness of the requested matrix, but don't expect it to catch
bugs beyond what C/E already cover -- the genuinely distinguishing test
for custom-filt behavior is D/B2 (even period, where default != uniform).

### Python-exclusive features (no R equivalent exists -- single-source verification, clearly flagged as such)

| Test | Feature | Verified Python output |
|---|---|---|
| F | `two_sided=false` | `trend[19:22] approx 105.938,106.428,106.976` (differs from A's centered trend, as expected); `trend[0:5]` all `NaN` (causal filter has no history at series start) |
| G | `extrapolate_trend=2` | `trend[0:8]` all finite, no `NaN` at all -- `approx 101.072,101.444,101.815,...` |
| H | `extrapolate_trend=:freq` (= `period-1` = 11) | `trend[0:8] approx 100.635,101.056,101.477,...` -- different from G (more points used in the fit -> different extrapolated line), both finite |

**These cannot be R-verified because R's `decompose()` has no comparable
option at all** -- not a gap in this verification effort, a genuine
capability gap in R itself. The honest "second test case" for these rows
is: confirm that with `two_sided=true, extrapolate_trend=0` (the
defaults), Julia's output matches test A/B/C exactly -- i.e., the
Python-exclusive code paths don't silently change behavior when their
features aren't requested.

### Validation / error-handling tests (behavior verified in both languages via actual execution, not assumed)

| Test | Scenario | R behavior (verified) | Python behavior (verified) | Julia behavior (this design) |
|---|---|---|---|---|
| I | Internal `NA`/`NaN` in `x` | Errors: `"time series contains internal NAs"` (via `na.omit.ts`) | Errors: `"This function does not handle missing values"` | Errors: `ArgumentError`, matching both |
| J | `:multiplicative` with non-positive values | **No check -- silently produces `Inf`/`NaN`/garbage** (verified: R has no validation here at all) | Errors: `"Multiplicative seasonality is not appropriate for zero and negative values"` | Errors: `ArgumentError` -- **adopts Python's stricter behavior over R's silent failure**, per the "exceed both" goal |

---

## 6. Hand/machine-verified Julia test suite

```julia
using Test, Random

@testset "classical_decompose core matrix" begin
    Random.seed!(42)
    t1 = 0:47
    y1 = (100 .+ 0.5 .* t1) .+ 10 .* sin.(2π .* t1 ./ 12) .+ randn(48)

    # Test A: additive, period=12, default filt
    rA = classical_decompose(y1, 12; model=:additive)
    @test isapprox(rA.trend[20:22], [108.9275812, 109.5358365, 110.0988327]; atol=1e-6)

    # Test D: additive, period=12, custom uniform filt (differs from A)
    rD = classical_decompose(y1, 12; model=:additive, filt=fill(1/12, 12))
    @test isapprox(rD.trend[20:22], [109.261923, 109.80975, 110.3879154]; atol=1e-5)
    @test !isapprox(rD.trend[20], rA.trend[20]; atol=1e-3)  # genuinely different filters

    # Test I: missing values rejected
    y_nan = copy(y1); y_nan[5] = NaN
    @test_throws ArgumentError classical_decompose(y_nan, 12)

    # Test J: multiplicative + non-positive rejected (Julia matches Python, not R's silence)
    y_neg_full = repeat([1.0, -2.0, 3.0], 16)
    @test_throws ArgumentError classical_decompose(y_neg_full, 12; model=:multiplicative)

    # two_sided=false produces leading NaN (causal filter) -- structural check
    rF = classical_decompose(y1, 12; model=:additive, two_sided=false)
    @test all(isnan, rF.trend[1:5])
    @test !isapprox(rF.trend[20], rA.trend[20]; atol=1e-3)  # genuinely different from centered

    # extrapolate_trend removes NaN entirely
    rG = classical_decompose(y1, 12; model=:additive, extrapolate_trend=2)
    @test !any(isnan, rG.trend)

    rH = classical_decompose(y1, 12; model=:additive, extrapolate_trend=:freq)
    @test !any(isnan, rH.trend)
    @test rG.trend[1] != rH.trend[1]  # different npoints -> different extrapolated line

    # defaults unaffected by the Python-exclusive features existing
    rA2 = classical_decompose(y1, 12; model=:additive, two_sided=true, extrapolate_trend=0)
    @test isapprox(rA2.trend[20:22], rA.trend[20:22]; atol=1e-10)
end
```

Given the exact values above were read off truncated terminal output
(displayed to varying decimal places), **re-run the actual R/Python
session transcript for full-precision values** before finalizing
`atol`s tighter than shown -- the values here are accurate to the digits
displayed, treat anything beyond that as approximate until re-verified
at full precision.

---

## 7. What to do with this

1. Implement `classical_decompose` and `_extrapolate_trend!` per section
   4 -- double-check the extrapolation index arithmetic specifically
   (flagged as the highest-risk part).
2. Run the test suite in section 6.
3. For tighter numerical tolerances, re-run the R/Python verification
   scripts (reproducible from the seeds in section 5) and capture
   full-precision output rather than the truncated terminal display used
   here.
4. Update `development-sequence.md`'s Stage 3.1 row: mark implemented,
   note the two genuine Python-exclusive capabilities adopted
   (`two_sided`, `extrapolate_trend`) and the one place this
   implementation is deliberately *stricter* than R (multiplicative
   validation) rather than merely matching it.
5. `MSTL`/`STL` (Stage 3.2/3.3) will likely reuse `classical_decompose`'s
   filter-construction logic -- worth keeping that factored out
   separately rather than embedded inline, for reuse.
