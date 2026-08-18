# Handoff: Stage 3.2 — STL Decomposition (comprehensive, R+Python execution-verified)

Status: **done.** `_lowess` (`src/loess.jl`) and `stl_decompose`'s full
inner + outer/robustness loop (`src/stl.jl`, `test/test_stl.jl`) are
built and verified exactly against real R/Python output on
`monthly.csv`, `period7.csv`, and `monthly_outlier.csv`. This doc's own
scope framing (section 0, section 6) was followed literally rather than
attempting the whole thing at once -- and steps 3 and 4 each turned out
to need a real, nontrivial bug hunt (see below) that the phased approach
made much easier to isolate than debugging it inside a "do everything at
once" attempt would have been.

**Step 4 (the outer/robustness loop) closed.** Read `_stl.pyx`'s `fit()`,
`_onestp`, `_rwts`, `_ss`, and `_ess` directly (not transcribed from this
doc's own section 4 summary) to get the exact structure: `outer` passes
of bisquare reweighting (`cmad = 6*median(|resid|)`, weight
`1`/bisquare/`0` below `0.001*cmad`/between/above `0.999*cmad`) computed
from the *previous* pass's residuals, applied to the cycle-subseries and
trend Loess steps only -- confirmed the low-pass step is never
reweighted (`_onestp`'s low-pass `_ess` call hardcodes `userw=False`).
`Statistics.median` confirmed algebraically identical to `_rwts`'s
partition-based fixed computation for both even and odd `n` (the
partition's two middle indices coincide for odd `n`, doubling the same
value), so it's deliberately reused rather than hand-rolling a substitute
-- and it's Python's *documented-fixed* version, not R's original
NETLIB Fortran's documented-buggy one (per section 3(f) below).

**One more genuinely hard bug**, same shape as step 3's `k>n` one:
`_stl.pyx`'s `fit()` zeros `trend`/`season` exactly *once*, before the
first outer pass -- every subsequent pass continues refining the
*previous* pass's trend, not restarting from zero. The first
implementation attempt here reset `trend` to zero on every outer pass,
which runs without error and produces plausible-looking output, off by
~0.3 out of ~105 (an order of magnitude worse than step 3's bug, and one
that "looks reasonable" without an exact numeric target to catch it
against) -- caught immediately by testing against section 3(f)'s
pre-verified numbers rather than settling for "it runs." Fixed by
threading a `trend_init` argument through `_stl_inner` and carrying
`trend` across `stl_decompose`'s outer-loop passes; re-verified to 13+
significant digits against real `statsmodels` on both this doc's own
combo and a freshly-generated one. `STLDecomposition` gained a `weights`
field (both references expose the final robustness weights -- without it
there'd be no way to inspect which points `robust=true` actually
downweighted).

**Two more gaps closed during a test-coverage audit before step 4**
(triggered by "do we have tests for every parameter combination?", not a
user-reported bug -- both were things no existing test happened to
exercise):
1. `seasonal_window`/`trend_window`/`low_pass_window` had no
   oddness/minimum validation. Confirmed directly from R's Fortran
   (`stl.f`'s top-level `stl` subroutine: `newns = max0(3,ns);
   if(mod(newns,2).eq.0) newns=newns+1`, same for `nt`/`nl`) that R
   silently auto-corrects any window to `(odd, >= 3)`, while Python's
   `_stl.pyx` raises `ValueError` on the same bad input. Now raises here
   too (odd, `>= 3`, and `trend_window`/`low_pass_window` also
   `> period`), matching the stricter-validation precedent already used
   for `classical_decompose`.
2. `robust` was wrongly the gate for the "not yet implemented" error
   (`robust=true` threw; `robust=false` silently ignored `outer`
   entirely, which would have returned a *wrong, non-robust* answer for
   `outer > 0` had that path ever run). Confirmed from R's `stl.R` (its
   `.Fortran()` call passes only numeric `ni`/`no` iteration counts, no
   robust flag) and Python's `_stl.pyx` (`robust` only selects
   `inner`/`outer` *defaults* -- `2`/`15` in both) that the outer loop is
   actually gated by the effective outer count being `> 0`, not by
   `robust`. Verified empirically against real `statsmodels`:
   `robust=true, outer_iter=0` and `robust=false, outer_iter=0` (matched
   `inner_iter`) produce bit-identical output. Fixed so the "not yet
   implemented" throw now triggers whenever the effective outer count is
   `> 0` -- via `robust=true`'s default `outer=15`, or an explicit
   `outer` regardless of `robust` -- instead of only on `robust=true`.

`test/test_stl.jl` grew from 49 tests (before the audit) to 168 (after
step 4 landed): two fully-independent non-robust matched-parameter combos
with fresh `statsmodels` reference numbers (one on `monthly.csv`, one on
`period7.csv`, covering degree/window combinations the original section
3(e) combo didn't touch), window-validation error-path coverage across
all three window parameters, an explicit `robust`/`outer` semantics
testset (22 tests), and a dedicated outer-loop exact-value testset (17
tests) covering section 3(f)'s numbers, the `weights` field's outlier
detection and bounds, a non-robust-vs-robust divergence sanity check, and
every explicit `outer` value tried (`1`, `5`, `15`, `30`).

Both Python-side numeric claims in section 3 (the matched-parameter
agreement in 3(e) and the `robust=True` divergence in 3(f)) were
independently re-run against real `statsmodels.tsa.seasonal.STL` using
the actual `monthly.csv`/`monthly_outlier.csv` files, matching exactly --
including confirming `monthly_outlier.csv` really is `monthly.csv` with
the two described outliers injected at the described (0-indexed)
positions.

Per section 6, step 1-2: read `SeasonalTrendLoess.jl`'s actual `loess.jl`
source directly (GitHub, not a description of it). Found something worth
flagging rather than porting: it row-scales `A`/`b` by the raw tricube
weight `w` before an ordinary (unweighted) `\` solve --
`A = A.*(w.*rho); b = b.*(w.*rho)` -- which is mathematically a
*different* (non-standard) weighting scheme, not the standard weighted
least squares `_ols` already implements correctly (`sqrt(weight)` row
scaling into an unweighted solve). Cross-checked this concern against
`statsmodels`' own algorithm: its current `lowess()` is a compiled
`.pyd` with no introspectable source, but its readable pre-Cython
predecessor (`statsmodels.nonparametric.smoothers_lowess_old`, still
shipped) confirms `sqrt(weight)` scaling is the validated convention --
matching `_ols`, not `SeasonalTrendLoess.jl`. `_lowess` reuses `_ols`
directly for every local fit rather than re-deriving the weighted solve.

Also recovered from that same pre-Cython source, more precisely than
this handoff's own docs-only research could get: `frac` converts to a
neighbor count via `k = int(frac*n)` -- **truncation, not rounding** --
and the exact bisquare robustness formula (`|resid|/(6*median(|resid|))`,
weight 0 above 1). Validated `_lowess` against real `statsmodels.
nonparametric.lowess` output across 4 `frac`×`it` combinations on the
existing AR(1) reference series, plus two mathematical invariants
(constant and perfectly-linear input must be reproduced exactly,
independent of `frac`/`it`) and a structural robustness check (an
injected outlier's own fitted value is pulled toward the true trend less
under `it>0` than `it=0`).

`_lowess` was then extended (once `stl_decompose` actually needed them,
not speculatively) with `degree` 0/1/2, an explicit `k` (STL's own
window-as-point-count convention, alongside `frac`), and off-grid `xout`
evaluation (for extending each cycle-subseries one period beyond its own
extent) -- degree-1 on-grid and off-grid validated exactly against real
`statsmodels.nonparametric.lowess`'s `xvals` argument; degree=0 validated
by hand-computing the weighted-mean formula independently (Python's
`lowess()` has no degree=0 mode); degree=2 validated via the same
exact-recovery-on-noiseless-data invariant already used for degree=1.

**The one genuinely hard bug in this whole handoff's implementation**:
`_lowess_fit1`'s handling of `k` exceeding the number of available
points. This is not an edge case for STL -- it's the *normal* case for
cycle-subseries smoothing, where a subseries can easily be shorter than
`seasonal_window` (`monthly.csv`: `n=48`, `period=12` gives subseries
length 4, well under `seasonal_window=7`). First attempt copied
`SeasonalTrendLoess.jl`'s `width *= max(1, k/n)` multiplicative stretch
-- this got the full inner loop *close* (`trend[20]` off by ~0.003 out of
~109) but consistently, measurably wrong, not just floating-point noise.
Isolated the bug properly before guessing at fixes: wrote an independent
Python transcription of the *entire* inner loop (same index bookkeeping,
same low-pass composition, same `_lowess` design) -- it reproduced the
exact same wrong numbers, confirming the bug was in the algorithm design
itself, not a Julia-specific slip, before spending any more time staring
at Julia code. Then fetched and read the actual NETLIB STL Fortran source
(`stl.f`'s `stlest` subroutine, the exact routine R's `stl()` calls into,
via the `wch/r-source` GitHub mirror -- not a description of it, the
literal `.f` file). The real formula: `h = max(xs-nleft, nright-xs); if
(len .gt. n) h = h + (len-n)/2` -- **additive**, not multiplicative, and
`len`/`n` are Fortran `integer`s, so `(len-n)/2` is **truncating integer
division**, not real division (matters: `(7-4)/2` is `1`, not `1.5`).
Fixed `_lowess_fit1` to match this exactly (`width += (k-n) ÷ 2` when `k
> n`) and re-verified against the same independent Python transcription
first -- exact match to 8+ significant digits -- before touching the
Julia code again, then confirmed the fix in Julia matched the same way.

**Done**: the outer/robustness loop (`robust=true`/`outer`) landed too,
targeting section 3(f)'s numbers (Python's fixed median computation, not
R's documented-buggy original, per section 5's design decision) -- see
the Status block at the top of this doc for the implementation and bug
history. Stage 3.2 is fully complete; `seasonal_window=:periodic` (R-only,
no Loess) remains an explicit, documented, separately-tracked gap, not a
blocker.

For a fresh Claude Code session picking this up with no prior context.
Same dual-execution rigor as Stage 3.1, applied to a substantially harder
algorithm. **Read section 0 first** — this stage's scope is different
from 3.1's: STL requires a Loess smoothing engine that doesn't exist
anywhere in this project yet, so this handoff is reference-verification
and API design at full depth, but implementation is honestly scoped as
phased rather than delivered complete inline.

## 0. Scope reality check — read this before anything else

Classical decomposition (3.1) needed only `convolution_filter`, already
built. STL needs a working **Loess (locally weighted regression)
engine** as a prerequisite -- genuinely new numerical machinery, not a
composition of existing pieces. Writing a correct, from-scratch Loess
implementation *and* the full STL inner/outer-loop algorithm *and*
verifying both to the standard the rest of this project holds to is
realistically its own multi-session effort, not one handoff doc's worth
of code.

**What this handoff delivers**: full reference verification (both
languages' actual source/behavior, executed and compared, not assumed),
a complete Julia API design, the algorithm structure from Cleveland et
al. (1990), and a genuinely useful early finding (a real R/Python
numerical divergence under `robust=TRUE`, see section 3). **What it does
not deliver**: a working Julia implementation. That's flagged honestly
as the next phase, with a concrete recommended starting point (read
`SeasonalTrendLoess.jl`'s actual source, per the project's "reference,
never port" policy) rather than attempted here at lower confidence than
this project's standard.

## Where this fits

- **Depends on:** a new Loess primitive (not yet built -- this is the
  actual blocker, not Stage 1.2's filters).
- **Reference note already in `development-sequence.md`**:
  `SeasonalTrendLoess.jl` (Julia) flagged as "worth reading before
  reimplementing" -- this is exactly the moment to do that reading.

---

## 1. Verified reference: R `stats::stl()` -- full argument list and source

```r
stl(x, s.window, s.degree = 0, t.window = NULL, t.degree = 1,
    l.window = nextodd(period), l.degree = t.degree,
    s.jump = ceiling(s.window/10), t.jump = ceiling(t.window/10),
    l.jump = ceiling(l.window/10), robust = FALSE,
    inner = if (robust) 1 else 2, outer = if (robust) 15 else 0,
    na.action = na.fail)
```

Confirmed by printing R's actual source (`print(stats:::stl)`):
```r
if (is.null(t.window))
    t.window <- nextodd(ceiling(1.5 * period/(1 - 1.5/s.window)))
...
z <- .Fortran(C_stl, x, n, as.integer(period), as.integer(s.window),
    as.integer(t.window), as.integer(l.window), s.degree, t.degree,
    l.degree, nsjump=..., ntjump=..., nljump=..., ni=as.integer(inner),
    no=as.integer(outer), ...)
```
**R calls directly into compiled Fortran** -- the original NETLIB STL
routine (Cleveland et al., Bell Labs, 1990), essentially unmodified.

**R-only feature found while reading source**: `s.window` can be the
**literal string `"periodic"`**, not just a numeric window -- this
triggers a simplified mode where the seasonal component is just the mean
per within-cycle position (no Loess at all for the seasonal part):
```r
if (periodic) {
    which.cycle <- cycle(x)
    z$seasonal <- tapply(z$seasonal, which.cycle, mean)[which.cycle]
}
```
No equivalent found in Python's `STL` -- worth a Julia `:periodic` option
on `seasonal_window` matching this exactly, since it's a real, useful
mode (effectively a Loess-based STL that degrades gracefully to
classical-style fixed seasonal figures).

## 2. Verified reference: Python `statsmodels.tsa.seasonal.STL`

Constructor and `fit()` split, confirmed via docstrings (the class itself
is Cython-compiled, no introspectable signature, so docstrings are the
verified source here):
```python
STL(endog, period=None, seasonal=7, trend=None, low_pass=None,
    seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=False,
    seasonal_jump=1, trend_jump=1, low_pass_jump=1)
# then:
.fit(inner_iter=None, outer_iter=None)
#   inner_iter: 2 if robust else 5
#   outer_iter: 15 if robust else 0
```

**Explicitly documented, verified claim**: *"The original code contains
a bug that appears in the determination of the median that is used in
the robust weighting. This version matches the fixed version that uses a
correct partitioned sort to determine the median."* -- i.e., Python
knowingly diverges from the original Fortran (and therefore from R,
which calls that Fortran directly) specifically in `robust=True` mode.

## 3. Real discrepancies found -- verified by execution, not just reading docs

**(a) `seasonal_deg` default differs**: R's `s.degree` defaults to `0`
(locally constant); Python's `seasonal_deg` defaults to `1` (locally
linear). Different smoothness behavior out of the box.

**(b) `low_pass_deg` default differs in *kind*, not just value**: R ties
`l.degree` to whatever `t.degree` is set to (dynamic default); Python's
`low_pass_deg` has a fixed default of `1` regardless of `trend_deg`. If a
user changes `trend_deg` in Python expecting `low_pass_deg` to follow
(as it would in R), it won't.

**(c) Jump-parameter defaults differ substantially**: R defaults every
jump to `ceiling(window/10)` (a real interpolation shortcut, computing
Loess only every ~10% of the window and linearly interpolating between --
faster, slightly less precise). Python defaults every jump to a fixed
`1` (compute Loess at *every* point, no shortcut). **R's default output
is a coarser approximation than Python's default**, not just a
performance difference with identical results.

**(d) `inner_iter` non-robust default is very different**: R uses `2`,
Python uses `5`. More than double the smoothing passes by default.

**(e) Core algorithm agrees exactly once parameters are matched** --
this is the important positive finding, worth stating as clearly as the
discrepancies. Verified directly: ran both with identical explicit
`s.window=7, s.degree=0, t.window=19, t.degree=1, l.window=13,
l.degree=1`, all jumps `=1`, `robust=FALSE, inner=2, outer=0` on
identical data (`monthly.csv`, n=48, period=12):
```
R      trend[20:24]: 108.9580496 109.5046509 110.0658509 110.6210296 111.1650203
Python trend[20:24]: 108.958049582 109.5046508904 110.0658508548 110.6210295545 111.1650202872
```
Agreement to ~9-10 significant digits. **Every discrepancy found above
is a default-value choice, not an algorithmic disagreement.**

**(f) `robust=True` genuinely diverges, even with matched parameters** --
the one place actual algorithmic behavior differs, not just defaults.
Verified on data with two injected outliers (`monthly_outlier.csv`),
identical explicit parameters including `inner=2, outer=15` on both
sides:
```
R      trend (realigned to same positions): 104.5544593  104.9324364  105.317138
Python trend (same positions):              104.55139049 104.92959742 105.31459038
                                       diff:  ~0.0031       ~0.0028      ~0.0025
```
Small (third decimal place) but consistent and non-zero -- exactly
consistent with the documented median-computation bug fix in section 2.
**This is the one place where matching R exactly and matching Python
exactly are mutually exclusive under `robust=true`** -- a design decision
is needed here (see section 5), not just a default-value choice.

**(g) Auto-formula for `t.window`/`trend`**: verified identical --
`nextodd(ceiling(1.5*period/(1 - 1.5/s.window)))` confirmed directly in
R's source, matches Python's documented formula exactly (same formula,
Python's docstring phrasing is just the English description of the same
computation).

---

## 4. Algorithm structure (Cleveland et al. 1990, both references implement this identically per section 3(e))

**Inner loop** (repeated `inner` times):
1. **Detrend**: `y - trend` (trend initialized to 0 on the first pass).
2. **Cycle-subseries smoothing**: split the detrended series into
   `period` subseries (all Januaries, all Februaries, ...), Loess-smooth
   each independently using `s.window`/`s.degree`, extending one period
   before and after.
3. **Low-pass filter** the smoothed cycle-subseries: a length-`period`
   moving average, applied three times (twice at `period`, once at 3),
   then Loess-smoothed with `l.window`/`l.degree`.
4. **Deseasonalize**: smoothed cycle-subseries minus the low-pass result
   = the seasonal component for this pass.
5. **Detrend again**: `y - seasonal`.
6. **Trend smoothing**: Loess-smooth the deseasonalized series with
   `t.window`/`t.degree` -> updated trend for the next inner-loop pass.

**Outer loop** (only if `robust=true`, repeated `outer` times): compute
residuals `y - trend - seasonal`, derive **bisquare robustness weights**
from their magnitude (larger residuals -> smaller weight), re-run the
entire inner loop with those weights applied to every Loess fit. This is
exactly where section 3(f)'s divergence originates -- the weight/median
computation inside this loop.

---

## 5. Proposed Julia API

```julia
stl_decompose(x, period::Integer;
              seasonal_window::Union{Int,Symbol}=7,   # or :periodic (R-only feature, adopted)
              seasonal_degree::Int=1,                  # Python's default -- see note below
              trend_window::Union{Nothing,Int}=nothing,
              trend_degree::Int=1,
              low_pass_window::Union{Nothing,Int}=nothing,
              low_pass_degree::Union{Nothing,Int}=nothing,  # nothing -> follow trend_degree (R's dynamic default)
              seasonal_jump::Int=1, trend_jump::Int=1, low_pass_jump::Int=1,  # Python's defaults, not R's
              robust::Bool=false,
              inner::Union{Nothing,Int}=nothing,   # nothing -> 2 if robust else 5 (Python's default)
              outer::Union{Nothing,Int}=nothing)   # nothing -> 15 if robust else 0 (both agree)
```

Design decisions, made explicitly rather than by silent default-copying:

- **`seasonal_window`/`seasonal_degree`/etc. naming**: full words, matching
  Python's naming style (`seasonal_deg` -> `seasonal_degree`) over R's
  terse `s.degree` -- consistent with this package's established
  preference (see `adf_test`/`kpss_test`) for Python-style names when the
  two references differ, while still accepting `:periodic` as a
  `seasonal_window` value (R's feature, genuinely useful, no reason to
  drop it).
- **Jump defaults -> Python's (`1`), not R's**: R's default is a
  *precision* tradeoff (coarser approximation for speed), not just a
  style choice. Defaulting to exact computation (matching Python) and
  letting a performance-conscious user opt into the interpolation
  shortcut explicitly seems like the more defensible default for a
  package prioritizing correctness -- flagged as a judgment call, not an
  obviously-correct choice, worth revisiting if it becomes a real
  performance problem in practice.
- **`inner` default -> Python's (`5` non-robust, `2` robust), not R's**:
  same reasoning -- more smoothing passes by default is the safer choice
  precision-wise; R's leaner default reads as more of a historical
  performance concession (1990s Fortran) than a considered statistical
  choice.
- **`low_pass_degree` default -> R's *dynamic* behavior (follow
  `trend_degree`), not Python's fixed `1`**: this one goes the other way
  -- R's coupling seems like the more defensible statistical default (no
  obvious reason low-pass smoothness should be decoupled from trend
  smoothness by default), so `nothing` here means "follow
  `trend_degree`," with an explicit `Int` available to override,
  matching Python's independent-degree flexibility when wanted.
- **`robust=true` divergence (section 3(f))**: **adopt Python's fixed
  median computation**, not R's original-Fortran (buggy, per Python's
  own documentation) version. This is the one case where "match R
  exactly" and "be correct" are in tension, and correctness wins --
  consistent with the project's broader stance (see Stage 3.1's
  multiplicative-validation decision) of exceeding a reference rather
  than reproducing a known bug for parity's sake. **Document this
  explicitly in the docstring** so anyone specifically needing bit-exact
  R reproduction under `robust=true` knows why they won't get it and
  why.

---

## 6. What to actually do next (phased, not "implement all of this now")

1. **Read `SeasonalTrendLoess.jl`'s actual source** (the Julia package
   already flagged in `development-sequence.md`) -- specifically its
   Loess implementation, to understand a validated Julia approach to the
   locally-weighted regression core before writing one. Per this
   project's "reference, never port" policy: read for the algorithm and
   validated numerical tricks (weight computation, degree-0/1 local
   fitting, edge handling), implement natively, don't transcribe.
2. **Build the Loess primitive first, as its own tested unit** -- before
   attempting the STL inner/outer loop at all. It should be independently
   verifiable: Loess-smooth a known series, compare against R's `loess()`
   or Python's `statsmodels.nonparametric.lowess` directly (both are
   simpler, standalone functions worth verifying against before
   tackling STL's more complex cycle-subseries usage of it).
3. **Implement the inner loop first, `robust=false` only** -- this is
   fully specified by section 4 and directly checkable against section
   3(e)'s exact verified numbers (`monthly.csv`, matched parameters).
   Getting these to match is the real correctness gate before attempting
   the outer loop at all.
4. **Add the outer/robust loop last**, using Python's fixed
   median-computation approach per section 5's decision, and validate
   against section 3(f)'s numbers *as the target*, not R's (which are
   deliberately not being reproduced here).
5. Update `development-sequence.md`'s Stage 3.2 row to reflect this
   phased status honestly -- "API designed and reference-verified,
   implementation pending a Loess primitive" is a meaningfully different
   state than "built," and the row should say so rather than getting a
   premature checkmark.

---

## 7. Verification data (reproducible)

Same `monthly.csv` as the Stage 3.1 bundle (`n=48, period=12`, seed=42);
`monthly_outlier.csv` is identical with `y[10] += 30` and `y[30] -= 25`
injected (0-indexed) to actually engage robustness weighting for section
3(f)'s test. Regenerate both from Stage 3.1's bundle plus that two-line
modification rather than re-deriving -- keeps the two stages' fixtures
consistent.

R and Python commands used for every number in section 3, for exact
reproducibility:
```r
stl(ts(y, frequency=12), s.window=7, s.degree=0, t.window=19, t.degree=1,
    l.window=13, l.degree=1, s.jump=1, t.jump=1, l.jump=1,
    robust=FALSE, inner=2, outer=0)   # section 3(e)
# and robust=TRUE, inner=2, outer=15 on monthly_outlier.csv for 3(f)
```
```python
STL(y, period=12, seasonal=7, seasonal_deg=0, trend=19, trend_deg=1,
    low_pass=13, low_pass_deg=1, seasonal_jump=1, trend_jump=1,
    low_pass_jump=1, robust=False).fit(inner_iter=2, outer_iter=0)
# and robust=True with .fit(inner_iter=2, outer_iter=15) on the outlier data
```
