# Handoff: Stage 3.3 — MSTL (comprehensive, source-verified, parallel-by-default design)

Status: **done.** `mstl_decompose`/`MSTLDecomposition` (`src/mstl.jl`,
`test/test_mstl.jl`) built exactly per this handoff's design (sections 3-5)
and verified against real `statsmodels.tsa.seasonal.MSTL` to 10+
significant digits on `data/hourly_mstl.csv` (n=500, daily+weekly
seasonality, a scaled-down version of `MSTL`'s own docstring example).
The parallelization design (section 4) was retrofitted directly into
`stl_decompose`'s cycle-subseries step (`_stl_cycle_subseries!` in
`src/stl.jl`) rather than into a separate MSTL-only code path, exactly as
this handoff recommended in section 6.2 -- `stl_decompose` itself gained
a new `parallel::Bool=true` keyword as part of this.

**A real, previously-latent bug in `stl_decompose` itself was found and
fixed while building this**, not a bug in the new MSTL code: every
Stage 3.2 exact-match test used `period=12,n=48` or `period=7,n=42` --
both exact multiples of the period, where every cycle-subseries phase has
the *same* length. `_stl_cycle_subseries!`'s one-point-after extension
used a fixed `period+n+i` index, which only happens to be correct in that
exact-multiple case; the real formula (verified directly from `_stl.pyx`'s
`_ss`) depends on *each phase's own* subseries length, `(m+1)*period+i`.
`mstl_decompose`'s own reference case (`period=168` on `n=500`, giving
subseries lengths of both `2` and `3` in the same call) is what exposed
it -- confirmed via a from-scratch Python transcription of the whole
inner loop (the same debugging technique that found Stage 3.2's `k>n`
width bug) that isolated the exact wrong array index, not just "the
numbers are slightly off." Fixed in `src/stl.jl`; a small
(`n=20,period=8`) regression test with its own independently-verified
`statsmodels` reference numbers was added to `test/test_stl.jl`
specifically because this is the smallest case that reproduces it.

Also found and fixed a docs-build regression (`@ref` links to two
private, unexported helper functions that Documenter can't cross-reference
since they're not listed in `api.md`) and confirmed real Python quirk
worth deliberately not replicating: `MSTL`'s `elif self.lmbda:` treats
`lmbda=0.0` as falsy, so it silently skips the Box-Cox transform instead
of applying the mathematically-correct `log(x)` -- confirmed by executing
real `statsmodels`, not just reading source. `mstl_decompose`'s `lambda=0`
genuinely applies `log(x)`, documented explicitly as an intentional
divergence.

For a fresh Claude Code session picking this up with no prior context.
Same treatment as 3.1/3.2. **Read section 0** — same scope reality as
3.2: MSTL wraps STL, which needs a Loess engine that doesn't exist yet.
This handoff delivers full reference verification (this time from
**actual Python source**, not just docstrings -- `MSTL` is pure Python,
unlike the Cython-compiled `STL` class), a complete API design, and a
precisely-specified parallelization strategy -- proven correct from both
languages' real source, not assumed.

## 0. Scope reality check

Identical situation to 3.2: MSTL cannot run until STL can, and STL
cannot run until a Loess primitive exists. This is design + verified
reference + a from-first-principles-correct parallelization plan, not
runnable code. The parallelization design specifically is worth having
now regardless, since it constrains *how* STL's cycle-subseries loop
should be written when that work happens -- better to design the hook in
from the start than retrofit it.

---

## 1. Verified reference: R `forecast::mstl` (from actual GitHub source)

```r
mstl(x, lambda = NULL, biasadj = FALSE, iterate = 2,
     s.window = 7 + 4 * seq_len(6), ...)   # ... passed to stats::stl()
```

Core loop, from `forecast/R/mstl.R`:
```r
for (j in seq_len(iterate)) {
  for (i in seq_along(msts)) {
    deseas <- deseas + seas[[i]]
    fit <- stl(ts(deseas, frequency = msts[i]), s.window = s.window[i], ...)
    seas[[i]] <- msts(seasonal(fit), seasonal.periods = msts)
    deseas <- deseas - seas[[i]]
  }
}
trend <- msts(trendcycle(fit), seasonal.periods = msts)  # from the LAST fit only
```
- `msts` (periods) sorted ascending before processing.
- If `x` is a plain `ts` (single frequency, not multi-seasonal `msts`),
  `iterate` is forced to `1L`.
- If `x` has no seasonal periods at all, falls back to `stats::supsmu()`
  (Friedman's SuperSmoother) for trend instead of STL entirely.
- Per the algorithm's own paper (Bandura, Hyndman & Bergmeir 2021,
  arXiv:2107.13462): missing values are imputed via `na.interp` *before*
  decomposition -- R's MSTL tolerates missing data, unlike `stl()` itself.

## 2. Verified reference: Python `statsmodels.tsa.seasonal.MSTL` (full source, confirmed pure Python)

```python
class MSTL:
    def __init__(self, endog, *, periods=None, windows=None, lmbda=None,
                 iterate=2, stl_kwargs=None): ...

    def fit(self):
        num_seasons = len(self.periods)
        iterate = 1 if num_seasons == 1 else self.iterate      # <- CONFIRMED: matches R exactly

        if self.lmbda == "auto":
            y, lmbda = boxcox(self._y, lmbda=None)               # scipy's MLE-based auto-lambda
        elif self.lmbda:
            y = boxcox(self._y, lmbda=self.lmbda)
        else:
            y = self._y

        seasonal = np.zeros((num_seasons, self.nobs))
        deseas = y
        for _ in range(iterate):
            for i in range(num_seasons):
                deseas = deseas + seasonal[i]
                res = STL(endog=deseas, period=self.periods[i],
                           seasonal=self.windows[i], **self._stl_kwargs
                           ).fit(inner_iter=stl_inner_iter, outer_iter=stl_outer_iter)
                seasonal[i] = res.seasonal
                deseas = deseas - seasonal[i]

        trend = res.trend        # from the LAST fit only -- matches R exactly
        rw = res.weights         # from the LAST fit only
        resid = deseas - trend
```

Additional confirmed-from-source behavior:
- `windows` default: `tuple(7 + 4*i for i in range(1, n+1))` -- **confirmed
  identical formula to R's** `7 + 4*seq_len(6)`.
- `periods` sorted ascending (`sorted(periods)`, or zipped-and-sorted with
  `windows` if both given) -- **confirmed identical to R's** `sort(msts,
  decreasing=FALSE)`.
- If any period `>= nobs/2`: **warns and drops it** (not an error) --
  worth carrying this exact behavior into Julia rather than erroring
  outright.
- **No missing-value imputation anywhere in this source.** Python's MSTL
  does *not* do R's `na.interp` step -- it would simply inherit `STL`'s
  own hard rejection of non-finite input (confirmed in the 3.2 handoff).
  **This is a real, now source-confirmed discrepancy**: R tolerates
  missing data via imputation, Python doesn't tolerate it at all.
- Box-Cox auto-lambda: `scipy.stats.boxcox`'s MLE-based method -- **not**
  the same algorithm as R's `BoxCox.lambda()` (Guerrero's method by
  default). Manual `lambda` values will match exactly; `lambda="auto"`
  vs R's default auto-selection will generally **not** pick the same
  value.

## 3. What this confirms about parallelization -- proven from source, not assumed

**Both the `iterate` loop and the `i in range(num_seasons)` / `seq_along(msts)` loop are provably sequential**, directly visible in both languages' actual source:

```python
deseas = deseas + seasonal[i]     # add back this period's OLD seasonal estimate
res = STL(endog=deseas, ...)      # fit depends on the CURRENT state of deseas
seasonal[i] = res.seasonal        # get the NEW estimate
deseas = deseas - seasonal[i]     # remove it again, ready for period i+1
```

Period `i+1`'s STL call reads `deseas`, which was just mutated by period
`i`'s fit. **There is no way to parallelize across periods or across
`iterate` passes** -- this isn't a missed optimization opportunity in
either reference implementation, it's a genuine data dependency inherent
to the algorithm. Confirmed identically in R's source (`deseas <- deseas
+ seas[[i]]` / `deseas <- deseas - seas[[i]]`, same pattern).

**The only genuine parallelism opportunity is *inside* a single STL
call** -- specifically the cycle-subseries smoothing step (3.2 handoff,
algorithm section, step 2): splitting the detrended series into `period`
independent subseries and Loess-smoothing each one separately. Those
`period` Loess fits share no data dependency within a single pass.

---

## 4. Parallelization design: intelligent, on by default, cheap to disable

Per the request: parallel by default, but not blindly -- "intelligent"
here means detecting whether parallelism can actually help before paying
its overhead, and giving an explicit, cheap way to turn it off.

```julia
function _cycle_subseries_smooth!(smoothed, subseries, s_window, s_degree; parallel::Bool=true)
    period = length(subseries)
    use_threads = parallel && Threads.nthreads() > 1 && period >= 4
    # period >= 4 guard: for very small period counts (e.g. period=2 or 3),
    # thread-spawning overhead can exceed the actual work -- not worth it
    if use_threads
        Threads.@threads for i in 1:period
            smoothed[i] = loess_smooth(subseries[i], s_window, s_degree)
        end
    else
        for i in 1:period
            smoothed[i] = loess_smooth(subseries[i], s_window, s_degree)
        end
    end
    return smoothed
end
```

Design notes:
- **`Threads.nthreads() > 1` check**: if the user launched Julia with
  `-t 1` (the default if `-t`/`JULIA_NUM_THREADS` isn't set), spawning
  tasks via `Threads.@threads` adds pure overhead for zero benefit --
  detecting this and falling back to a plain loop is the "intelligent"
  part of "intelligent... and by default."
- **`period >= 4` guard**: thread-spawning has real overhead; for tiny
  period counts the sequential loop is very likely faster regardless of
  core count. This threshold is a reasonable starting guess, not a
  rigorously benchmarked constant -- worth revisiting with actual
  `BenchmarkTools.jl` numbers once Loess exists and this can be measured
  rather than estimated.
- **One level of parallelism, not nested**: don't also parallelize
  individual point-level Loess fits *within* each subseries chunk.
  Julia's `Threads.@threads` doesn't compose well with nested
  parallelism by default (oversubscription risk -- spawning `period`
  outer tasks that each try to spawn more inner tasks competes for the
  same thread pool). If per-point Loess fitting itself becomes a
  bottleneck later (relevant when `jump=1`, Python's default, computing
  Loess at *every* point with no interpolation shortcut), that's a
  separate, deliberate decision to make with real profiling data, not a
  default to reach for now.
- **`parallel::Bool=true` exposed at the top-level `mstl_decompose`
  call**, threaded down through to STL and into this function -- cheap,
  explicit opt-out (`parallel=false`) for reproducibility/debugging
  contexts (e.g., comparing against a single-threaded reference run,
  or diagnosing a suspected threading bug) without needing to relaunch
  Julia with different thread settings.
- **Where this actually pays off**: MSTL's own multi-seasonal use case
  is exactly where `period` tends to be large -- the Python docstring's
  own example uses `periods=(24, 24*7)`, i.e. an hourly series with
  daily (period=24) and weekly (period=168) seasonality. A period-168
  cycle-subseries smoothing step is genuinely worth parallelizing; a
  period-4 (quarterly) one probably isn't, which is exactly what the
  guard above is for.

---

## 5. Proposed Julia API

```julia
mstl_decompose(x, periods::Union{Integer,AbstractVector{<:Integer}};
               windows::Union{Nothing,Integer,AbstractVector{<:Integer}}=nothing,
               lambda::Union{Nothing,Real,Symbol}=nothing,   # nothing | :auto | a Real
               iterate::Int=2,
               parallel::Bool=true,
               stl_kwargs::NamedTuple=NamedTuple())
```

Design notes:
- **`periods`**: matches Python's naming; accepts a scalar or vector
  (Julia idiom for "one or many" rather than requiring a 1-element
  vector). Sorted ascending internally, matching both references.
- **`windows`**: matches Python's naming; `nothing` (default) computes
  `7 .+ 4 .* (1:length(periods))`, confirmed identical in both
  references.
- **`lambda`**: `nothing` (no transform, matches both defaults) / `:auto`
  (matches both languages' *concept*, but **not their exact algorithm** --
  see section 2's Box-Cox note; document this divergence explicitly
  rather than implying false equivalence) / a `Real` (manual value,
  matches both exactly).
- **`iterate`**: matches both languages' name and default (`2`); forced
  to `1` internally when `length(periods) == 1`, matching both
  references' confirmed behavior.
- **`parallel`**: new, not in either reference -- the efficiency
  requirement from this request. Threaded down to the STL cycle-subseries
  step per section 4.
- **`stl_kwargs`**: matches Python's pass-through design (`stl_kwargs`
  dict) more closely than R's `...` splat, since Julia's `NamedTuple`
  gives the same "pass anything STL accepts" flexibility with better
  discoverability than an untyped varargs splat.
- **Missing-value policy**: given the section 2 finding that Python's
  MSTL doesn't actually impute (despite the algorithm's source paper
  describing that behavior) and just inherits STL's rejection, while R's
  does impute via `na.interp` -- **recommend following Python's simpler,
  now-source-confirmed behavior** (reject, don't silently impute) as the
  default, since silent imputation is a meaningfully different and
  riskier default than the rest of this project has adopted anywhere
  else (every other stage rejects missing data outright). Note this
  explicitly as choosing Python's actual behavior over R's *documented*
  behavior, which itself may not even be what R's shipped code does --
  worth an actual R execution check (`is.na` injection test, same
  technique as the 3.1/3.2 handoffs) before finalizing, rather than
  trusting the paper's description at face value.
- **Period-too-large handling**: adopt Python's confirmed behavior
  (warn and drop periods `>= nobs/2`, don't error) -- a real, sensible
  behavior worth matching exactly since it's now verified from source on
  both sides conceptually (R's `mstl.R` has the same
  `msts[msts < n/2]` filtering with a warning, per the source snippet
  in the earlier research pass).

---

## 6. What to actually do next

1. Everything from the 3.2 handoff's section 6 (Loess primitive first,
   verify independently, then STL inner loop, then STL outer loop).
2. When writing STL's cycle-subseries smoothing step specifically,
   implement it using the `_cycle_subseries_smooth!` pattern from
   section 4 of *this* handoff from the start -- this is the one piece
   of Stage 3.3's design that actually belongs inside Stage 3.2's
   implementation, not bolted on after.
3. Once STL works, `mstl_decompose` itself is a genuinely thin wrapper
   (confirmed by both references' source being ~15 lines of actual
   logic) -- the sequential loop structure in section 3, nothing more.
4. **Verify the R missing-data-imputation claim empirically** before
   committing to "R imputes, Python doesn't" as a documented discrepancy
   -- the finding above is from the algorithm's *paper*, not from reading
   `mstl.R`'s actual source the way every other claim in this handoff
   was verified. Worth the same `NA`-injection test used in 3.1/3.2
   before treating it as confirmed rather than merely likely.
5. Update `development-sequence.md`'s Stage 3.3 row: same "designed,
   implementation pending Loess" status as 3.2, plus a note that the
   parallelization hook needs to be built into 3.2's implementation
   directly, not added retroactively.

---

## Stage 3 status (updated post-implementation)

| Stage | Status |
|---|---|
| 3.1 Classical decomposition | Done -- dual R+Python verified |
| 3.2 STL | Done -- inner loop, outer/robustness loop, and `parallel` cycle-subseries threading all implemented and verified; see `stage-3.2-stl-handoff.md`'s Status block for the two bugs (window validation, `robust`/`outer` gating, the `k>n` width formula) found and fixed along the way |
| 3.3 MSTL | Done -- see this doc's own Status block above for the real `(m+1)*period+i` indexing bug (in `stl_decompose`, not MSTL itself) this stage's own reference case exposed |

Next: whatever the roadmap's Stage 4 (optimization backbone) or later
decomposition work (`seasonal_window=:periodic`, exact Box-Cox `lambda=
:auto`) calls for -- both explicit, documented gaps, not blockers.
