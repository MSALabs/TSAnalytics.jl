# Handoff: Chapter 4 — Filters and Moving Averages

`docs/src/introduction/04-filters-and-moving-averages.md`. Target 11–12 pages.

Chapter 3 removed the trend. This chapter does the opposite — it keeps
the trend and throws away everything else. The two operations are more
closely related than they look, and by the end of this chapter the
reader should see why.

Same requirements as chapters 1–3. The reading template from chapter 1's
handoff applies unchanged.

---

## 1. A disagreement box this chapter was not allocated

`all-chapters-handoff.md` gave chapter 4 only a `julia` box, on the
grounds that filtering is uncontroversial. Checking it directly, that
was wrong — **there is a genuine and dangerous disagreement here**, and
it is one most people never notice.

Verified this session on `x = [10,12,11,13,15,14,16,18,17,19]` with a
three-term equal-weight filter:

```
R  filter(x, rep(1/3,3), sides=2):
    NA  11  12  13  14  15  16  17  18  NA

numpy convolve(x, w, mode='same'):
    7.333  11  12  13  14  15  16  17  18  12

numpy convolve(x, w, mode='valid'):
    11  12  13  14  15  16  17  18
```

**R marks the endpoints as missing. numpy returns numbers.** And those
numbers are wrong in a specific, misleading way — numpy zero-pads, so
the first output is `(0 + 10 + 12)/3 = 7.333`, which is *below every
observation in the series*. The minimum of the data is 10.

Plot a numpy `mode='same'` moving average and you get pronounced dips at
both ends of the series that look like real features and are entirely
artefacts of the padding. Nothing warns you. `mode='valid'` is honest
but returns a shorter series, which then misaligns with the original if
you are not careful.

**Where this package sits**, confirmed from source:

```julia
function convolution_filter(x, filt; sides::Integer=2, circular::Bool=false)
    out = fill(NaN, n)
```

It follows R — `NaN` at the endpoints, no invented values. State this
explicitly in the chapter, the same way chapter 15 discloses the STL
jump convention. A reader arriving from numpy will get shorter usable
output and deserves to know it is deliberate.

---

## 2. The package API

Confirmed exported: **`convolution_filter`**, **`recursive_filter`**,
**`moving_average`**.

`convolution_filter(x, filt; sides=2, circular=false)` mirrors R's
`stats::filter` argument-for-argument, which makes the comparison in
beat 5 straightforward to write.

---

## 3. The chapter, beat by beat

### Beat 1 — Too much noise to see anything (about 1 page)

**Chart 1 — `cmort`, weekly cardiovascular mortality in Los Angeles,
508 observations.**

Shumway & Stoffer's series, and the right one here because the trend is
real but genuinely hard to see through the week-to-week variation.

*Reading:* there is something going on — a slow decline, possibly some
seasonality — but the week-to-week jumps are large enough that any
claim about the shape would be arguing from noise. The eye wants to
average nearby points together. That instinct is correct, and this
chapter is about what happens when you act on it.

### Beat 2 — Averaging nearby points (about 2 pages)

**Chart 2 — `cmort` with three moving averages overlaid: 5-week,
21-week, 51-week.**

*Reading, and take space:* the 5-week average still wobbles. The 51-week
average is smooth and has lost the seasonal pattern entirely, along
with a year's worth of observations from each end. The 21-week average
is somewhere in between and is arguably the most useful, though nothing
so far tells you that except taste.

The trade-off is not a nuisance to be optimised away. **A wider window
buys smoothness and pays in resolution and in observations lost at the
ends.** There is no window width that is correct in the abstract; there
is only a width appropriate to the question. Someone asking about the
annual cycle wants a narrow window. Someone asking about the decade-long
trend wants a wide one. The same data supports both.

**Chart 3 — the same filter applied to a pure straight line and to
pure noise. Two panels.**

*Reading:* this is the conceptual heart of the chapter. Applied to the
straight line, the filter returns the straight line, essentially
untouched. Applied to noise, it returns something with a fraction of
the original variance. A moving average is not a general-purpose
smoother — it is a device that passes slow movement through and
suppresses fast movement, and "trend" and "noise" are simply names for
those two speeds.

Say plainly that this is a frequency-domain idea arriving in
time-domain clothing, and that chapter 6 will give it its proper name.
Do not develop it further here.

### Beat 3 — A filter is a list of weights (about 1.5 pages)

Now generalise. A moving average is the special case where the weights
are all equal. Nothing requires that.

**Chart 4 — equal weights against a triangular weighting, same width,
same series.**

*Reading:* the weighted version is smoother at the same nominal width,
because it does not treat the observation seven weeks away as being as
informative as the one next door. This is the entire idea behind the
Henderson filters that official statistical agencies use, and chapter
38 returns to them. Here the point is narrower: the width of a filter
and its shape are separate choices, and most people only ever vary the
first.

**Chart 5 — a convolution filter and a recursive filter on the same
data. Two panels.**

*Reading:* they are not variations on a theme; they are different
operations. The convolution filter combines nearby *inputs*. The
recursive filter feeds its own *outputs* back in, so every value
depends on the entire history before it. Verified behaviour on a
short series — `recursive_filter` with coefficient 0.5 on
`[10,12,11,…]` produces `10, 17, 19.5, 22.75, …`, which is not
smoothing in any ordinary sense; it is accumulation.

**`julia` box here.** The two filter types have genuinely different
performance characteristics — one can be written as a single pass with
no state, the other cannot. Half a page on why, and on what that means
for allocation. Do not let it grow.

### Beat 4 — The even-order problem (about 2 pages)

fpp3's §3.3 handles this better than anyone and it is the right source.

**Chart 6 — a 12-term moving average on monthly data, and a 2×12
average, plotted against the original. Three panels or one overlay.**

*Reading:* an odd-length filter has a middle. A 12-term filter does
not, so its output sits half a month off from any actual observation.
On monthly data with an annual cycle this matters, because 12 is
exactly the period you most want to average over. The fix is to average
two consecutive 12-term averages, which recentres the result and
produces the 2×12 filter that classical decomposition uses throughout.

This is not a technicality. It is the reason chapter 14's decomposition
uses a filter that looks oddly specific, and meeting it here means it
will not need explaining there.

**Chart 7 — `sides=1` against `sides=2` on the same series.**

*Reading:* the centred filter uses observations from both sides, so it
cannot be computed for the most recent points — and in real-time work
those are exactly the points you care about. The trailing filter can be
computed right up to today, and it lags: its estimate of the current
level is really an estimate of the level some weeks ago. Neither is
better. The choice depends on whether you are explaining the past or
tracking the present, and getting it wrong produces confident answers
about a moment that has already gone.

### Beat 5 — What happens at the edges (about 2 pages)

**Chart 8 — the endpoints disagreement, drawn.**

Plot three lines on one set of axes: the original series, the R/Julia
convention with gaps at the ends, and the numpy `mode='same'` result
with its zero-padded values.

*Reading:* the numpy line dives at both ends. Those dips are not in the
data. They are the padding showing through, and on the ten-point
example verified for this handoff the first value came out at 7.333
when the smallest actual observation was 10.

Then the `disagreement` box proper, using section 1's material, and the
disclosure of where this package stands.

The general lesson is worth stating because it recurs: **a function
that returns a number for every input is not more useful than one that
admits it cannot compute some of them.** It is less useful, because it
has moved the problem from a place you would notice to a place you
would not.

### Beat 6 — Where this leaves you (half a page)

A filter works because nearby observations are related. That is the
assumption underneath every window width chosen in this chapter, and it
has been used without ever being measured.

Measuring it is chapter 5.

No recap.

---

## 4. The `india` box

Placed in beat 4, after chart 7.

The Reserve Bank and MoSPI publish most monthly indicators with a lag of
several weeks. Anyone monitoring conditions in real time is therefore
already looking at old data, and if they smooth it with a trailing
filter they add the filter's own lag on top. A 12-month trailing
average of a series published six weeks in arrears is describing
conditions roughly a year old, which is a perfectly reasonable thing to
do and a completely unreasonable thing to do unknowingly.

Centred filters do not have this problem and cannot be computed for
recent months. There is no arrangement that avoids the trade-off; there
is only being explicit about which one you have taken.

One paragraph.

---

## 5. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **fpp3 §3.3** | The even-order problem and the 2×m construction, explained better here than anywhere else | The `fable` syntax |
| **Shumway & Stoffer** | `cmort` as the running example, and their treatment of smoothing as a family rather than one technique | Kernel smoothing and lowess; those belong near chapter 15 |
| **Ladiray & Quenneville** | That filter *shape* is a design problem with real theory behind it — the Henderson weights are the payoff, forward-referenced to chapter 38 | The X-11 mechanics themselves |
| **Hamilton** | The lag-polynomial view, which makes convolution and recursion obviously the same algebra | The frequency-domain development; chapter 6 does that |
| **Montgomery, Jennings & Kulahci** | The industrial monitoring context — trailing filters exist because someone needs an answer today | Control-chart specifics |
| **Cowpertwait & Metcalfe** | The patient entry level | R-specific scaffolding |

**On examples:** `cmort` (508), `sunspotz`, `jj` and `soi` are all
bundled and GPL-3. No external data needed.

---

## 6. Voice

Earlier guidance applies. Specific to this chapter:

**Resist the urge to be comprehensive about filter types.** There are
many. The reader needs to understand three ideas — weights, direction,
and edges — and every additional named filter dilutes them. Henderson
gets a forward reference and nothing more.

**The frequency-domain observation in beat 2 is a trap.** It is
genuinely illuminating and it is also chapter 6's entire subject.
Two sentences, then stop. If the draft starts explaining transfer
functions, cut back.

Additional things to avoid:

- Formulae for filters the chapter does not then use.
- "Smoothing" and "filtering" used interchangeably without saying they
  are the same thing here.
- Any sentence starting "Of course,".

---

## 7. Checklist

- [ ] Eight charts through `@example ch4`
- [ ] Eight readings, five-part template, varied openings
- [ ] Chart 3 (trend versus noise) present — the conceptual core
- [ ] Even-order problem shown with an actual 12-term versus 2×12
      comparison, not described
- [ ] `sides=1` versus `sides=2` chart, framed as real-time versus
      retrospective
- [ ] `disagreement` box uses the verified numbers, including the 7.333
      against a series minimum of 10
- [ ] The package's `NaN`-at-edges convention disclosed explicitly
- [ ] `julia` box on convolution versus recursive, half a page maximum
- [ ] `india` box on publication lag plus filter lag
- [ ] Ends by opening chapter 5, no recap
- [ ] 11–12 pages
- [ ] British-Indian spelling

---

## 8. What to do

1. **Check `moving_average`'s signature and behaviour** before writing.
   Three filter functions are exported and I have only read
   `convolution_filter` closely. In particular, find out whether
   `moving_average` handles the even-order 2×m case automatically or
   leaves it to the caller — beat 4 depends on the answer.
2. Reproduce the numpy comparison in section 1 as part of drafting, so
   the numbers on the page are from a live run.
3. Choose the three window widths in chart 2 by actually trying several
   on `cmort`. The reading claims 21 weeks is the most useful; confirm
   that before asserting it, or change the number.
4. Render every chart; use the honest CI note from `10-plotting.md` if
   the local environment cannot.
5. Check against section 7.
