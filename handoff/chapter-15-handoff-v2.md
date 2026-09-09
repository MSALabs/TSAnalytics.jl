# Handoff: Chapter 15 — STL (revised)

`docs/src/introduction/15-stl.md`. Target 13–14 pages.

Supersedes the earlier chapter 15 handoff, which was written before the
current requirements — maximum charts, a detailed reading after each,
and the voice guidance developed across chapters 1–14. The verified
findings carry over unchanged; the structure does not.

Reading template from chapter 1's handoff. Ten charts.

---

## 1. The correction that still governs this chapter

Across many sessions of building this package I repeatedly cited the
following as established:

> *"Python fixed a median bug in the STL Fortran that R still has; the
> numbers differ near outliers when `robust=TRUE`."*

**It was tested properly and it is wrong.** Do not write the chapter
around it. What is actually true is more interesting, and the
investigation itself is the chapter's best material.

---

## 2. Verified findings

Test setup: monthly series, `n = 120`, linear trend `50 + 0.4t`,
seasonal amplitude 12, noise `σ = 1.5`, one outlier of `+45` at index
60. Seed 11. R and Python read the identical CSV, with parameters
matched explicitly where the packages name them differently —
`s.window = seasonal = 7`, `s.degree = seasonal_deg = 0`,
`t.degree = trend_deg = 1`.

**They disagree, and not trivially:**

| configuration | max abs trend diff | max abs seasonal diff |
|---|---|---|
| `robust = TRUE`, package defaults | 0.0486 | 0.0339 |
| `robust = FALSE`, package defaults | **0.133** | **0.110** |

**The disagreement is larger without robustness**, which kills the
median-bug hypothesis outright. A robustness-weight bug would leave
`robust = FALSE` agreeing to machine precision. It disagrees three
times as much.

**The main culprit is a parameter almost nobody sets.** R's `stl`
computes loess at every *j*th point and interpolates between, for
speed, with `jump` derived from the window widths:

```
s.window = 7, period = 12  ->  t.window = 23
                               s.jump = 1
                               t.jump = 3
                               l.jump = 2
```

Python defaults every jump to **1** — exact loess at every point. So R
ships a coarser approximation by default and Python ships the exact
computation, and neither documents this as a divergence from the other.

Setting all jumps to 1 on both sides:

| configuration | max abs trend diff | max abs seasonal diff |
|---|---|---|
| `robust = TRUE`, jumps matched | 0.0019 | 0.0080 |
| `robust = FALSE`, jumps matched | 0.0535 | 0.124 |

The robust case improves by a factor of about 25.

**Something remains, and we do not know what.** With jumps matched
*and* iterations forced far past convergence (`inner = 50`,
`outer = 10`):

```
max abs trend diff     0.0254
max abs seasonal diff  0.0232
```

Not a convergence artefact either. **Write that honestly.** "We tracked
it this far and no further" is the only sentence the evidence supports.

**Where this package sits**, confirmed from source:

```julia
stl_decompose(x, period; seasonal_window=7, seasonal_degree=1,
              trend_window=nothing, trend_degree=1,
              low_pass_window=nothing, low_pass_degree=nothing,
              robust=false, inner=nothing, outer=nothing, parallel=true)
```

**No jump parameters at all** — exact loess, Python's behaviour, not
R's. And `seasonal_degree = 1`, again matching Python
(`seasonal_deg = 1`) rather than R (`s.degree = 0`). Both must be
disclosed; a reader migrating from R will get different numbers.

`STLDecomposition` carries `observed`, `trend`, `seasonal`, `resid`,
`period` and — usefully — **`weights`**, which makes the robustness
mechanism chartable.

---

## 3. The chapter, beat by beat

### Beat 1 — Three complaints (about 1 page)

**Chart 1 — the contaminated series from chapter 14, decomposed
classically, one more time.**

*Reading:* chapter 14 ended with three specific failures — a frozen
seasonal pattern, no defence against outliers, and no estimate at the
endpoints. All three are visible here. This chapter's method fixes all
three, and the interesting part is that it fixes them with one idea
rather than three patches.

### Beat 2 — Fit a line, locally (about 2 pages)

**Chart 2 — loess on a scatter of points, with the local window and
the fitted line at three different positions marked.**

*Reading:* loess fits a low-degree polynomial to the points near each
target, weighted by distance, and keeps only the value at the centre.
Slide the window along and the fitted values trace a curve. It is a
moving average from chapter 4 with two upgrades — the weights fall off
with distance instead of being flat, and a line is fitted rather than a
mean taken.

The second upgrade is what solves the endpoint problem. A one-sided
window still supports a fitted line; it just supports it less well. A
centred average has nothing to average.

### Beat 3 — The two loops (about 2.5 pages)

**Chart 3 — the inner loop as a flow: detrend, smooth each
cycle-subseries, low-pass filter, deseasonalise, smooth for trend.**

*Reading:* the crucial step is the second. Where classical
decomposition averaged all the January values into one number, STL
*smooths* them across years. If January's effect is drifting, a smooth
through the Januaries follows the drift. The seasonal window controls
how much drift is allowed — and setting it to `"periodic"` recovers
classical decomposition's frozen pattern exactly, which is worth
showing as the limiting case.

**Chart 4 — the same series with `seasonal_window` at 7, 21 and
periodic. Three panels of the seasonal component.**

*Reading:* at 7 the seasonal pattern visibly evolves. At 21 it barely
moves. Periodic is a flat line repeated. The parameter is a dial
between chapter 14's method and something fully flexible, and there is
no correct setting — only a setting appropriate to a belief about the
data.

fpp3's own defaults are worth quoting: `season(window = 11)` for a
single seasonal period, `trend(window = 21)` for monthly data.

**Chart 5 — the outer loop's robustness weights.**

*Reading, and this is the chapter's most instructive chart:* plot the
`weights` field against time. The outlier at index 60 sits at a weight
near zero; everything else sits near one. That is the outer loop
telling you, explicitly, which observations it decided to ignore.

Almost no treatment shows this, and it turns robustness from an
adjective into a mechanism the reader can see. It is also a diagnostic
in its own right — a cluster of downweighted points is worth
investigating.

### Beat 4 — Doing it (about 2 pages)

**Chart 6 — the full four-panel STL decomposition of a real bundled
series.**

Use `aus_production` or `vic_elec` at a single period.

*Reading:* observed, trend, seasonal, remainder, and the endpoints all
present — the third complaint from beat 1 answered. Compare the trend
against chapter 13's classical one if the same series was used there.

**Chart 7 — `robust = false` against `robust = true` on the
contaminated series. Two panels.**

*Reading:* without robustness the outlier distorts the seasonal
component for its own month across the whole series, exactly as
classical decomposition did. With robustness it is downweighted and the
distortion largely disappears. Chart 5 showed the mechanism; this shows
the consequence.

### Beat 5 — Two implementations, one algorithm (about 3 pages)

The `disagreement` box, and it should be told as the investigation it
was.

**Chart 8 — R's STL trend and Python's, overlaid, on identical data
with identical stated parameters.**

*Reading:* they do not coincide. The gap reaches 0.133.

Then the sequence:

1. Same data, same nominal parameters, two respected implementations,
   different numbers in the second decimal place.
2. The natural first hypothesis is robustness. Test it. **The
   disagreement is worse without robustness.** Hypothesis dead.
3. Read the defaults properly. R interpolates loess between every third
   point for the trend; Python computes it at every point. Neither
   documents this as a divergence.
4. Match the jumps. The robust-mode gap falls by a factor of 25.
5. Force convergence too. **0.0254 remains, and we do not know why.**

**Chart 9 — the gap as jumps go from R's defaults to 1.**

*Reading:* most of the disagreement is explained and some is not. Say
so.

Then disclose where this package sits — exact loess, no jump
approximation, `seasonal_degree = 1` — and what that means for someone
porting R code.

The moral generalises well past STL and is worth stating plainly:
**two implementations of one published algorithm are not the same
function, and the difference usually lives in defaults nobody reads.**

### Beat 6 — Where this leaves you (half a page)

**Chart 10 — `vic_elec` at half-hourly resolution, with STL's single
seasonal component fitted to it.**

*Reading:* STL handles one seasonal period. This series has three —
daily, weekly, annual — and the fit is visibly inadequate. One period
is not always enough.

Chapter 16.

No recap.

---

## 4. The `india` box

Placed in beat 3, after chart 4.

Indian monthly series need an evolving seasonal pattern more than most.
The festival calendar moves against the Gregorian one, so the October
effect and the November effect trade places from year to year — and a
frozen seasonal index cannot represent that at all, while a short
seasonal window can partially absorb it.

Partially, not fully. STL adapts to *drift* in a seasonal pattern; it
does not know that the drift is caused by a calendar it has never heard
of. The proper fix is a regressor built from the actual festival dates,
which is chapter 36.

One paragraph.

---

## 5. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Cleveland, Cleveland, McRae & Terpenning (1990)** | The algorithm itself; cite it properly | The full parameter-tuning guidance |
| **fpp3 §3.6** | The practical framing, the default window recommendations, and honesty about parameter choice | The `fable` syntax |
| **Shumway & Stoffer** | Loess as a general smoother, introduced before STL uses it | Kernel smoothing generally |
| **Ladiray & Quenneville** | The comparison against X-11's approach — both iterate, both use robustness weights, and the family resemblance is not accidental | X-11 mechanics |
| **Montgomery, Jennings & Kulahci** | Robustness as an operational necessity; real industrial series have real outliers | Control charts |
| **Hamilton** | Nothing; STL is outside his scope | — |

**On examples:** `aus_production`, `vic_elec`, `jj`, `aus_retail` all
bundled. The contaminated series is constructed and should be labelled.

---

## 6. Voice

**Beat 5 is a detective story and should read as one.** A false lead
that dies on contact with evidence, a real discovery in an
unglamorous place, and an honest dead end. Told flatly it is a list of
numbers; told properly it is the best passage in Part III.

**Chart 5 deserves protection.** The robustness weights are the single
most under-shown thing in every STL treatment and they make an abstract
adjective concrete.

**Do not oversell STL.** It fixes chapter 14's three complaints and
introduces its own parameters, and chart 10 shows it failing. Beat 6
should not read as an advertisement for chapter 16 so much as an honest
limit.

Avoid, beyond earlier lists:

- Deriving loess. Chart 2 does the work.
- Describing STL as "the modern method". It is from 1990.
- Any suggestion that the residual 0.0254 is unimportant because it is
  small. It is unexplained, which is a different thing.

---

## 7. Checklist

- [ ] Ten charts through `@example ch15`
- [ ] Loess shown geometrically before any formula
- [ ] `seasonal_window` dial shown across three settings, including
      periodic as the classical limiting case
- [ ] **Robustness weights plotted** — chart 5 is not optional
- [ ] Four-panel decomposition on real bundled data
- [ ] `disagreement` box told as an investigation, with the false lead
      included
- [ ] The unexplained 0.0254 residual stated, not glossed
- [ ] Package's no-jump and `seasonal_degree = 1` conventions disclosed
- [ ] `india` box on moving festivals, with its honest limit
- [ ] Chart 10 shows STL failing on multiple seasonality
- [ ] Ends by opening chapter 16, no recap
- [ ] 13–14 pages, British-Indian spelling

---

## 8. What to do

1. **Generate the Julia numbers fresh.** Section 2's figures come from R
   and Python. The chapter's Julia output has never been produced and
   must not be estimated from them.
2. Reproduce the R-versus-Python comparison during drafting so charts 8
   and 9 come from a live run.
3. Do not repeat the median-bug claim anywhere. If it survives in other
   drafts or notes, correct it.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 7.
