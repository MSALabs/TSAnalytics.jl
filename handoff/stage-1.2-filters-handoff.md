# Handoff: Stage 1.2 — Linear Filters & Moving Average

Status: implemented ✅ (`src/filters.jl`, `test/test_filters.jl`) — the
implementation and hand-verified test values below were run against an
actual Julia 1.9 runtime and matched exactly, with one unrelated fix: the
`jldoctest` examples added to the docstrings needed `isapprox(...;
nans=true)` instead of raw vector display, since `1/3+1/3+1/3` doesn't
land on exactly `2.0` in Float64 and `NaN != NaN` breaks doctest/`==`
comparison either way — not a correctness issue, just a doctest-exactness
one. Two minor deltas from the proposed API below, for consistency with
conventions established since this doc was written: `order`/`sides` are
typed `Integer` rather than `Int`, and `init` is routed through
`tsvalues` (like `diffinv`'s `xi`) rather than a raw `collect(Float64,
...)`, for full container-agnosticism.

For a fresh Claude Code session picking this up with no prior context.
This covers one specific, self-contained piece of TSAnalytics.jl: Stage
1.2 in `development-sequence.md` — linear filtering primitives plus a
`moving_average` convenience wrapper. If you have the full
`development-sequence.md` in the repo already, this doc is the detailed
backing for that one row; read this before implementing it.

## Where this fits

- **Depends on:** Stage 0.2 (the `tsvalues`/`tsindex` container-agnostic
  interface — every function below accepts anything `tsvalues` can be
  called on: a plain `Vector`, a `TSFrame` column, `TimeSeries.values(ta)`,
  a `DataFrame` column, or any other iterable via `collect`).
- **Feeds into:** Stage 3 (classical/STL/MSTL decomposition) — `moving_average`
  is specifically the trend-extraction filter classical decomposition needs.
- **Project-wide policy in effect:** "reference, never port" — every
  function here is implemented natively from the referenced paper/textbook
  formula or the reference implementation's *documented behavior*, never
  by translating R/Python source code. The R and Python references below
  exist to (a) resolve ambiguity in the formula and (b) validate output
  numbers, not to copy from.
- **Naming convention already established in this codebase:** separate
  functions over a `method=` string switch (e.g. existing `adf_test`/
  `kpss_test` rather than one `unit_root_test(method=...)`). Follow that
  same pattern here — `convolution_filter`/`recursive_filter` as two
  functions, not one.

---

## 1. Verified reference: R `stats::filter()`

```r
filter(x, filter, method = c("convolution", "recursive"), sides = 2L, circular = FALSE, init = NULL)
```

| Argument | Meaning |
|---|---|
| `x` | univariate or multivariate time series |
| `filter` | coefficients — filter weights for convolution, AR coefficients for recursive |
| `method` | `"convolution"` (moving average) or `"recursive"` (autoregression) |
| `sides` | convolution only: `2` (default) = centered around lag 0; `1` = past values only. If filter length is even under `sides=2`, more of it sits forward in time than backward |
| `circular` | convolution only: wrap the filter around the series' ends instead of padding with `NA` |
| `init` | recursive only: initial values of the series just prior to the start, in reverse time order; defaults to zeros |

Formulas (from R's own docs):

- Convolution: `y[i] = filter[1]*x[i+o] + filter[2]*x[i+o-1] + ... + filter[p]*x[i+o-(p-1)]`, where `o` is an offset determined by `sides`.
- Recursive (note the *implied* unit coefficient at lag 0): `y[i] = x[i] + filter[1]*y[i-1] + ... + filter[p]*y[i-p]`.

## 2. Verified reference: Python

**Correction worth knowing:** an earlier draft of the roadmap cited
`pandas.rolling` as the Python equivalent of R's `filter()`. That's wrong
— `pandas.rolling` is a window-*aggregation* API (`.mean()`, `.sum()`,
`.apply()`), a different concept from linear filtering. The actual
structural match — and what `statsmodels` uses internally for exactly
this purpose (its own `seasonal_decompose`'s trend step) — is:

```python
statsmodels.tsa.filters.filtertools.convolution_filter(x, filt, nsides=2)
statsmodels.tsa.filters.filtertools.recursive_filter(x, ar_coeff, init=None)
```

`nsides` matches R's `sides` exactly (2=centered, 1=past-only), but there
is **no circular option** in the Python version — R is a strict superset
here. `recursive_filter`'s formula and `init` convention match R's
recursive case exactly; its docstring confirms it's built on
`scipy.signal.lfilter` internally.

## 3. Verified reference: R `forecast::ma()` (for `moving_average`)

```r
ma(x, order, centre = TRUE)
```

Formula: `T̂_t = (1/m) * sum(y[t+j] for j in -k:k)`, where `k = (m-1)/2`.

- **Odd `order`**: straightforward centered average.
- **Even `order`**: per R's docs, *"the observations averaged will include
  one more observation from the future than the past (k rounded up)."*
  If `centre = TRUE` (the default), *"the value from two moving averages
  (where k is rounded up and down respectively) are averaged, centering
  the moving average"* — this is the classic **"2×m" double moving
  average**, the exact technique X-11-style seasonal decomposition uses
  for even periods (e.g. monthly data, period 12). It is mathematically
  equivalent to a single `(order+1)`-length filter with half-weighted
  endpoints: `[1/(2m), 1/m, ..., 1/m, 1/(2m)]`.

This is why `moving_average` should **not** just be a thin pandas-style
rolling mean — the even-order re-centering is the whole reason it's worth
having as its own function rather than telling users to call
`convolution_filter` directly with a naive uniform filter.

---

## 4. Proposed Julia API

```julia
convolution_filter(x, filt::AbstractVector{<:Real}; sides::Int=2, circular::Bool=false)
recursive_filter(x, ar_coeff::AbstractVector{<:Real}; init::Union{Nothing,AbstractVector{<:Real}}=nothing)
moving_average(x, order::Int; centre::Bool=true)
```

Design notes:
- `x` in every case: untyped, passed through `tsvalues(x)` internally —
  works with any container per Stage 0.2's interface.
- `filt`/`ar_coeff` coefficient order: kept in the **same reverse-time-order
  convention both R and Python use**, deliberately — flipping to a more
  "intuitive" forward order would break direct validation against
  reference output numbers.
- `init` naming: matches both R's and Python's argument name exactly (a
  rare case where all three languages happen to agree).
- `sides`/`circular` naming: matches R's argument names exactly (R's
  `sides` over Python's `nsides` — shorter, and matches general
  signal-processing terminology).
- `centre` (not `center`): matches R's spelling exactly, deliberately, for
  the same direct-validation reason as the coefficient order above.
- `moving_average` is implemented as a thin wrapper that *constructs the
  right filter and calls `convolution_filter`* — no duplicated filtering
  logic.

---

## 5. Implementation

Drop this into `src/filters.jl` (or merge into wherever your local repo's
Stage 1.2 file lives):

```julia
export convolution_filter, recursive_filter, moving_average

"""
    convolution_filter(x, filt; sides::Int=2, circular::Bool=false)

Linear filtering by convolution -- a (possibly weighted, possibly
one-sided) moving average:

    y[i] = filt[1]*x[i+o] + filt[2]*x[i+o-1] + ... + filt[p]*x[i+o-(p-1)]

where `o` is an offset determined by `sides`. Matches R's
`stats::filter(x, filt, method="convolution", sides=sides,
circular=circular)` and Python's
`statsmodels.tsa.filters.filtertools.convolution_filter(x, filt,
nsides=sides)` exactly, including the reverse-time-order convention for
`filt` -- kept deliberately (rather than flipped to a perhaps more
intuitive forward order) so results validate directly against those
reference implementations without a translation step.

- `sides = 2` (default): centered moving average, `o = length(filt) ÷ 2`.
  If `length(filt)` is even, more of the filter sits forward in time than
  backward, matching R's documented convention.
- `sides = 1`: one-sided/causal -- coefficients apply to past values only
  (`o = 0`).
- `circular = true`: wrap the filter around the ends of the series
  (an R-only feature; statsmodels has no equivalent, hence `false` as the
  default to match Python's only available behaviour). When `false`,
  positions without enough data return `NaN`, matching both references.

`x` accepts anything `tsvalues` does.
"""
function convolution_filter(x, filt::AbstractVector{<:Real}; sides::Int=2, circular::Bool=false)
    sides in (1, 2) || throw(ArgumentError("sides must be 1 or 2"))
    xv = tsvalues(x)
    n = length(xv)
    p = length(filt)
    out = fill(NaN, n)
    o = sides == 2 ? p ÷ 2 : 0

    for i in 1:n
        acc = 0.0
        ok = true
        for j in 1:p
            idx = i + o - (j - 1)
            if circular
                idx = mod1(idx, n)
            elseif idx < 1 || idx > n
                ok = false
                break
            end
            acc += filt[j] * xv[idx]
        end
        out[i] = ok ? acc : NaN
    end
    return out
end

"""
    recursive_filter(x, ar_coeff; init=nothing)

Autoregressive (recursive/IIR) filtering, with an *implied* unit
coefficient at lag 0:

    y[i] = x[i] + ar_coeff[1]*y[i-1] + ar_coeff[2]*y[i-2] + ... + ar_coeff[p]*y[i-p]

Matches R's `stats::filter(x, ar_coeff, method="recursive", init=init)`
and Python's `statsmodels.tsa.filters.filtertools.recursive_filter(x,
ar_coeff, init=init)` exactly, including the reverse-time-order
convention for both `ar_coeff` and `init`.

`init` supplies the values of `y` just prior to the start of the series,
in reverse time order (`init[1]` = y at time 0, i.e. immediately before
the first observation; `init[2]` = y at time -1; ...). Defaults to zeros.
No stability check is performed -- as in R, the output may diverge if
`ar_coeff` doesn't correspond to a stable/invertible filter.

`x` accepts anything `tsvalues` does.
"""
function recursive_filter(x, ar_coeff::AbstractVector{<:Real};
                           init::Union{Nothing,AbstractVector{<:Real}}=nothing)
    xv = tsvalues(x)
    n = length(xv)
    p = length(ar_coeff)
    initv = init === nothing ? zeros(Float64, p) : collect(Float64, init)
    length(initv) == p || throw(ArgumentError("init must have the same length as ar_coeff"))

    out = Vector{Float64}(undef, n)
    yprev = copy(initv)   # yprev[k] holds y[i-k] at the start of step i

    for i in 1:n
        acc = xv[i]
        for k in 1:p
            acc += ar_coeff[k] * yprev[k]
        end
        out[i] = acc
        for k in p:-1:2
            yprev[k] = yprev[k-1]
        end
        p >= 1 && (yprev[1] = acc)
    end
    return out
end

"""
    moving_average(x, order::Int; centre::Bool=true)

Simple moving average smoother -- the convenience entry point most users
actually reach for, built directly on `convolution_filter` rather than
duplicating its logic. Matches R's `forecast::ma(x, order, centre=TRUE)`
exactly (name and formula, including the R spelling of `centre`):

    T̂ₜ = (1/m) * sum(y[t+j] for j in -k:k),   k = (m-1)/2

- **Odd `order`**: a plain centered `order`-length uniform-weight filter.
- **Even `order`, `centre=false`**: a plain `order`-length uniform-weight
  filter (asymmetric -- includes one more future observation than past,
  matching R's documented behaviour for this case).
- **Even `order`, `centre=true`** (the default, and the case that matters
  for seasonal decomposition with an even period like 12): the classic
  "2×`order`" double moving average -- a single `(order+1)`-length filter
  with half-weighted endpoints `[1/(2m), 1/m, ..., 1/m, 1/(2m)]`. This is
  the exact filter X-11-style seasonal decomposition uses for even
  periods, which is why it's the default here despite plain rolling
  means elsewhere (e.g. `pandas.Series.rolling`) not doing this.

`x` accepts anything `tsvalues` does. Positions without enough
surrounding data return `NaN`, same as `convolution_filter`.
"""
function moving_average(x, order::Int; centre::Bool=true)
    order >= 1 || throw(ArgumentError("order must be >= 1"))
    if isodd(order) || !centre
        filt = fill(1 / order, order)
    else
        filt = vcat(1 / (2*order), fill(1 / order, order - 1), 1 / (2*order))
    end
    return convolution_filter(x, filt; sides=2)
end
```

---

## 6. Hand-verified test values

No Julia runtime was available when this was written, so every expected
value below was computed independently (in Python, mirroring the exact
same algorithm) before being written as a test assertion — treat these as
trustworthy, but still run `Pkg.test()` to confirm the Julia
implementation matches.

```julia
using Test

@testset "filters" begin
    x = 1.0:10.0
    filt3 = [1/3, 1/3, 1/3]

    # Centered (sides=2) 3-term moving average
    r2 = convolution_filter(x, filt3; sides=2)
    @test isnan(r2[1]) && isnan(r2[end])
    @test isapprox(r2[2:end-1], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; atol=1e-10)

    # Causal (sides=1) 3-term moving average
    r1 = convolution_filter(x, filt3; sides=1)
    @test all(isnan, r1[1:2])
    @test isapprox(r1[3:end], [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; atol=1e-10)

    # Circular wraps instead of producing NaN at the edges
    rc = convolution_filter(x, filt3; sides=2, circular=true)
    @test !any(isnan, rc)
    @test isapprox(rc[2:end-1], r2[2:end-1]; atol=1e-10)

    @test_throws ArgumentError convolution_filter(x, filt3; sides=3)

    # Recursive AR(1), coefficient 0.5, constant input, zero init
    xr = ones(5)
    yr = recursive_filter(xr, [0.5])
    @test isapprox(yr, [1.0, 1.5, 1.75, 1.875, 1.9375]; atol=1e-10)

    @test_throws ArgumentError recursive_filter(xr, [0.5]; init=[0.0, 0.0])
end

@testset "moving_average" begin
    x12 = 1.0:12.0

    ma3 = moving_average(x12, 3)   # odd order
    @test isnan(ma3[1]) && isnan(ma3[end])
    @test isapprox(ma3[2:end-1], collect(2.0:11.0); atol=1e-10)

    ma4c = moving_average(x12, 4)  # even order, centre=true (default)
    @test all(isnan, ma4c[1:2]) && all(isnan, ma4c[end-1:end])
    @test isapprox(ma4c[3:end-2], collect(3.0:10.0); atol=1e-10)

    ma4u = moving_average(x12, 4; centre=false)  # even order, uncentred
    @test isapprox(ma4u[2:end-2], [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]; atol=1e-10)

    @test_throws ArgumentError moving_average(x12, 0)
end
```

If `data/airpassengers.csv` is already in your local repo (Stage: bundled
datasets), a good additional sanity check:

```julia
air = _load_column(TSAnalytics.AIR_PASSENGERS, "passengers")  # or however you load it locally
trend = moving_average(air, 12)
@test count(isnan, trend) == 12   # 6 NaN at each end for an even order-12 filter
@test all(x -> x > 0, filter(!isnan, trend))
```

---

## 7. What to do with this

1. Implement `src/filters.jl` as above (or adapt to match whatever's
   already in the local repo's structure).
2. Wire it into the main module file's `include(...)` list, after
   `interface.jl` (needs `tsvalues`) and in whatever position matches the
   local repo's existing convention.
3. Add the test file, run `Pkg.test()`, fix anything that doesn't match.
4. Cross-reference against R/Python directly if anything's ambiguous —
   `Rscript -e 'filter(1:10, rep(1/3,3))'` and the equivalent
   `statsmodels` call are both fast ways to double-check an edge case that
   isn't covered by the hand-verified values above.
