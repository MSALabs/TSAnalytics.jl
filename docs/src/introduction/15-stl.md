# STL

Between 1956 and 1965, the fourth quarter — October to December, the
run-up to the Southern Hemisphere summer — was Australia's best quarter
for beer production, comfortably ahead of every other. Between 2000 and
2010, the fourth quarter had become its *worst*. Did Australians really
stop drinking beer in summer over the course of forty-four years, or is
something else going on?

```jldoctest ch15
julia> using TSAnalytics, Dates

julia> d = dataset("aus_production");

julia> beer = d.Beer;

julia> dates = [Date(1970, 1, 1) + Dates.Day(Int(q)) for q in d.Quarter];

julia> first(dates), last(dates)
(Date("1956-01-01"), Date("2010-04-01"))

julia> length(beer)
218
```

(`dataset`'s catalogue does not yet decode `tsibbledata`'s own
quarter-serial encoding — days since 1970-01-01, one value per
quarter's first day — into a real `Date` automatically the way it does
for columns literally named `time`/`date`/`index`; decoding it by hand,
as above, is one line and worth knowing about if you work with more of
`tsibbledata`'s bundled series.)

A single number settles the summer question quickly enough. Split the
series into its first and its last decade and look at each decade's own
seasonal pattern:

```jldoctest ch15
julia> r_early = classical_decompose(beer[1:40], 4);

julia> round.(r_early.figure, digits=1)
4-element Vector{Float64}:
   6.4
 -37.9
 -20.6
  52.0

julia> r_late = classical_decompose(beer[(end - 39):end], 4);

julia> round.(r_late.figure, digits=1)
4-element Vector{Float64}:
 -18.5
  56.9
   0.0
 -38.4
```

Q4 goes from +52 (comfortably the best quarter) to -38 (comfortably the
worst); Q2 goes the other way, from -38 to +57. This is a real,
substantial reversal, not sampling noise — Australian brewing shifted
from a Christmas-and-New-Year peak towards a mid-year one as
refrigeration, drinking habits and the retail calendar all changed
across the second half of the twentieth century. Whatever the cause,
the pattern is not fixed, and any decomposition method that assumes it
is fixed is going to get something wrong somewhere in these 218
quarters.

## The obvious attempt

[Classical decomposition](14-classical-decomposition.md) is the natural
first tool, and it is worth seeing exactly where it runs into trouble
here, because that is the honest motivation for STL — not an appeal to
STL being "more flexible" in the abstract. Classical decomposition
detrends the whole series with a centred moving average, then averages
the detrended values quarter by quarter into **one** seasonal figure for
the entire span:

```jldoctest ch15
julia> r_full = classical_decompose(beer, 4);

julia> round.(r_full.figure, digits=1)
4-element Vector{Float64}:
   2.1
 -42.5
 -28.5
  68.9
```

Compare this to the two decade-level figures above. It is not close to
either. Q4's figure of +69 is an average of a decade where Q4 ran at
+52 and a decade where Q4 ran at -38 — a compromise that describes
neither era correctly, computed by a method that has no way to notice
the seasonal pattern has drifted, because a single seasonal figure
covering the whole series is the only kind of answer classical
decomposition is able to give. Whatever this compromise value is not
explaining ends up in the residual instead, misattributed as noise when
it is really a slow, structural drift in the seasonal pattern itself.

The question this chapter actually answers, then, is not "is there a
more flexible decomposition method" — obviously there could be — but
*specifically*: how do you build a decomposition whose seasonal
component is allowed to evolve, without simply refitting a fresh
classical decomposition on every short window and losing all
statistical stability in the process?

## The idea

STL — Seasonal-Trend decomposition using Loess — answers this with a
smoothing method (loess: a local weighted regression that fits a
low-degree polynomial to a window of nearby points, sliding that window
along the series) applied in a specific two-loop arrangement, due to
Cleveland, Cleveland, McRae & Terpenning (1990).

**The inner loop** runs once per pass and does five things in order:

1. **Detrend.** Subtract the current trend estimate from the series
   (all zeros, the first time through).
2. **Smooth each cycle-subseries.** This is the step that makes the
   seasonal component allowed to evolve, and it is worth being precise
   about what it actually smooths. Take *all* the Q1 values, in their
   own order across the years, as one short series, and loess-smooth
   *that* — separately from all the Q2 values, all the Q3 values, and
   all the Q4 values. The window here (`seasonal_window`) counts in
   *cycles* (years), not raw time steps: a `seasonal_window` of 7 means
   each smoothed Q1 value is a local weighted average of roughly the
   nearest seven years' worth of Q1 values, not the nearest seven
   quarters. A short window lets the seasonal figure drift quickly
   (even year to year, at the extreme); a long window holds it close to
   one fixed shape, approaching classical decomposition's assumption
   as a limiting case rather than a hard rule.
3. **Low-pass filter** the smoothed cycle-subseries (a moving average
   followed by a further loess pass) to isolate any trend that leaked
   into the seasonal estimate.
4. **Deseasonalise** the original series using this filtered result.
5. **Smooth the deseasonalised series** with loess again
   (`trend_window`) to get the next trend estimate, and go back to
   step 1.

**The outer loop** wraps all of this. After the inner loop converges,
compute a robustness weight for every observation from the size of its
residual — points the model cannot currently explain get downweighted,
in the limit to zero — and run the inner loop again with those weights
folded into every loess fit. This is what `robust=true` controls, and
it is the direct answer to a question a careful reader should already
be asking: if the seasonal component is now free to evolve, what stops
it evolving *around* a single bad observation instead of around the
genuine underlying pattern? The outer loop is that safeguard.

## Making it work

Running STL on the real Beer series with a fairly long seasonal window
— long enough to track a decades-long drift without chasing individual
outliers — recovers the same two seasonal shapes found by splitting the
series in half by hand, without ever being told where the split should
go:

```jldoctest ch15
julia> r_stl = stl_decompose(beer, 4; seasonal_window=21);

julia> round.(r_stl.seasonal[1:4], digits=1)          # STL's own first year
4-element Vector{Float64}:
   9.9
 -37.4
 -20.6
  48.5

julia> round.(r_stl.seasonal[(end - 3):end], digits=1)  # STL's own last year
4-element Vector{Float64}:
 -15.4
  51.6
  -1.0
 -35.0
```

Set `seasonal_window` shorter — down towards 7, say — and the seasonal
component would track the year-to-year wiggle far more tightly, at the
cost of being noisier itself; set it longer still and it approaches
classical decomposition's single fixed figure. There is no
context-free correct value; it is a genuine bias-variance choice the
analyst has to make, informed by how quickly the underlying process is
actually expected to change.

The Beer series does not contain a single dramatic outlier, which makes
it the wrong series for seeing the outer loop's robustness weights do
anything interesting. For that, construct a series where the answer is
known exactly: a 10-year monthly series with a linear trend, a 12-month
sine seasonal pattern, Gaussian noise, and one clearly artificial spike
of +45 injected at month 61.

```julia
using Random
Random.seed!(11)
t = 0:119
y = (50 .+ 0.4 .* t) .+ 12 .* sin.(2π .* t ./ 12) .+ 1.5 .* randn(120)
y[61] += 45.0                              # a constructed, labelled outlier

r = stl_decompose(y, 12; robust=true)
r.weights[58:64]
# 7-element Vector{Float64}:
#  0.9949
#  0.9181
#  0.9965
#  0.0       # <- the injected outlier, fully excluded
#  0.9084
#  0.6613
#  0.8857
```

The outlier's own weight goes to exactly zero, as expected. What is
more interesting, and genuinely easy to miss, is that it is not the
*only* point that does: six other months, none of them constructed and
none of them visually remarkable, are also fully excluded. STL's
bisquare weighting sets `h` — the threshold beyond which a residual is
treated as fully unreliable — to six times the *median* absolute
residual, and one enormous outlier barely moves a median. With a
typical residual of about half a unit here, `h` works out to under
three; several perfectly ordinary noise draws exceed that purely by
chance across 120 points. Robustness weighting is a genuinely more
aggressive filter than "downweight the one obvious outlier" — it
downweights anything unusually large relative to how well-behaved the
*rest* of the series is, which is both the right general behaviour and
a reasonable source of surprise the first time you see it.

!!! julia "Under the Hood"
    `stl_decompose` takes `parallel::Bool = true`, and cycle-subseries
    smoothing (step 2 of the inner loop) is the one part of STL that
    genuinely parallelises: every phase writes only to a set of output
    indices that are congruent to each other modulo the period, so no
    two phases can ever write the same location regardless of how many
    years the series spans. The outer loop, and the rest of the inner
    loop, read state the previous pass just finished writing and so
    are provably sequential — there is exactly one place in the whole
    algorithm where threading helps, and the implementation threads
    precisely that one place, not more.

## The complication

Fit the same constructed series in R's `stats::stl()` and Python's
`statsmodels.tsa.seasonal.STL`, with every shared parameter set to the
same value (`s.window`/`seasonal = 7`, `s.degree`/`seasonal_deg = 0`,
`t.degree`/`trend_deg = 1`), and the two disagree — not by
floating-point noise, but in the second decimal place:

| Configuration | max abs trend difference | max abs seasonal difference |
|---|---|---|
| `robust = TRUE`, package defaults | 0.0486 | 0.0339 |
| `robust = FALSE`, package defaults | 0.1328 | 0.1100 |

(Reproduced directly this session, on a freshly generated draw of the
series above — not carried forward from an earlier claim. An earlier,
unverified version of this investigation attributed the gap to a bug in
one package's median calculation; that story does not survive contact
with the numbers below, and is not repeated here.)

The natural first suspect is the robustness weighting itself — perhaps
one package computes it slightly differently. Test that directly: if
weighting were the cause, `robust = FALSE` should remove the
disagreement entirely, since there is no weighting step to disagree
about. Instead the gap *triples*. That hypothesis is dead on arrival.

The real cause is a parameter neither package documents as a point of
difference from the other, because from either package's own point of
view it isn't one — it is simply an unannounced default. R's `stl()`
does not evaluate its inner loess at every single point; for speed, it
evaluates every `jump`-th point exactly and linearly interpolates the
rest, with `jump` derived from the window widths (`s.window = 7`,
`period = 12` gives `s.jump = 1`, `t.jump = 3`, `l.jump = 2`, confirmed
directly from R's own computed `$jump`). Python's `STL` defaults every
jump to `1` — exact loess at every point, no interpolation — confirmed
directly from its own source documentation. Match every jump to `1` on
both sides and the robust-mode gap falls by a factor of roughly 25:

| Configuration | max abs trend difference | max abs seasonal difference |
|---|---|---|
| `robust = TRUE`, jumps matched | 0.0019 | 0.0080 |
| `robust = FALSE`, jumps matched | 0.0535 | 0.1237 |

So the jump defaults explain most of the *robust*-mode disagreement —
but plainly not the non-robust one, which barely moves. Forcing both
packages far past ordinary convergence (50 inner iterations, 10 outer,
jumps still matched) narrows the robust case further but does not close
it:

| Configuration | max abs trend difference | max abs seasonal difference |
|---|---|---|
| `robust = TRUE`, jumps matched, forced iterations | 0.0255 | 0.0232 |

A residual difference of two to twelve hundredths persists between two
careful implementations of the same published algorithm, run on
identical data with every shared parameter matched, and this chapter
does not know why. "Tracked this far and no further" is a more useful
sentence than a confident wrong explanation, and it is the only
sentence today's evidence actually supports.

Where does `stl_decompose` sit in all this? It has **no jump parameter
at all** — every loess fit is evaluated at every point, which is
Python's behaviour, not R's, and the two agree to floating-point
precision (on the order of `1e-12`) once run with identical window and
degree settings. It also *defaults* `seasonal_degree` to `1`, matching
Python's own default rather than R's default of `0` — a choice that
matters even when a user changes nothing: running both packages with
their own out-of-the-box defaults on this series (no parameters matched
at all) puts `stl_decompose` roughly `0.5` away from R on the trend and
a full `4.3` away on the seasonal component, an order of magnitude
larger than any of the matched-parameter gaps above. A reader arriving
from R and calling `stl_decompose(y, period)` with no extra arguments
will get visibly different numbers from `stl(y, period)`, and now knows
precisely why.

The generalisable lesson is not really about STL specifically: **two
implementations of one published algorithm are not the same function**,
and the difference — when there is one — usually lives in a default
nobody thought to document as a difference at all, because from inside
either codebase it just looks like "the default."

## Where this leaves you

STL handles exactly one seasonal period. Electricity demand — daily
peaks, a weekly pattern of weekday-versus-weekend, and a slower annual
cycle, all in the same series — has at least three, simultaneously, and
none of them fixed relative to each other. That is
[Chapter 16](16-multiple-seasonality.md)'s problem, and STL's own
cycle-subseries machinery turns out to be exactly the piece MSTL reuses
to solve it.

STL by itself also does not forecast: it separates a series into
components, and says nothing about what happens next. It has no notion
of a calendar — Easter, a trading-day count, a public holiday — beyond
whatever a fixed integer period can express, which rules out anything
that does not repeat at an exact, constant interval. Forecasting from a
decomposition and calendar-aware regressors are both later work, in
Parts IV and VII respectively.

!!! india "The Indian Series"
    This chapter would be a natural home for an Indian series with its
    own evolving seasonal pattern — the Index of Industrial Production
    shifting around demonetisation, say, or a retail series drifting
    around the GST rollout. No Indian dataset is bundled yet to make
    that example concrete and honest rather than invented, so this box
    stays short: the fuller Indian-calendar treatment, including the
    festival-timing problem, belongs to
    [Chapter 36](36-calendar-effects.md), and adding a suitable
    bundled Indian series is separate, flagged work rather than
    something quietly skipped here.

For the full `stl_decompose` signature — every keyword argument, and
the complete window/degree defaults table — see
[Manual: Decomposition](../manual/03-decomposition.md).
