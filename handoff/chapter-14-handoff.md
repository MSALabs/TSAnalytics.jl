# Handoff: Chapter 14 — Classical Decomposition

`docs/src/introduction/14-classical-decomposition.md`. Target 10–11 pages.

The oldest method in the book, still the one most people meet first,
and the one whose failures motivate everything after it. This chapter
has a job beyond teaching classical decomposition: it must make the
reader want chapter 15.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

### The seasonal pattern is frozen — exactly

Run this session on log `jj`, twenty years of quarterly data:

```
classical seasonal indices
  year 1:   -0.000956   0.039625   0.111383   -0.150052
  year 2:   -0.000956   0.039625   0.111383   -0.150052
  year 20:  -0.000956   0.039625   0.111383   -0.150052

identical across all twenty years?  TRUE
```

Not approximately equal. **Bit-identical**, because the method computes
one seasonal index per quarter by averaging across all years and then
repeats it. Johnson & Johnson's fourth quarter in 1960 and in 1980 are
assigned the same seasonal effect by construction.

That is the defining limitation and the reason STL exists. It should be
demonstrated with this exact comparison rather than described.

### The endpoints do not exist

```
classical trend NAs:  4   (2 at each end, m/2 for m=4)
STL trend NAs:        0
```

The trend comes from a centred moving average, which needs
observations on both sides. At the ends there are none.

---

## 2. The chapter, beat by beat

### Beat 1 — The oldest idea in the subject (about 1 page)

**Chart 1 — `aus_production` or `jj`, raw.**

*Reading:* chapter 13 named three components. This chapter extracts
them, using nothing the reader has not already met — a moving average
from chapter 4 and an average of averages. The method predates
computers, which is worth saying, because it explains both its
simplicity and its limitations. It was designed to be done by hand.

### Beat 2 — Extract the trend (about 1.5 pages)

**Chart 2 — the series with its centred moving average overlaid.**

*Reading:* a moving average whose window equals the seasonal period
averages exactly one full cycle at each point, so the seasonality
cancels. What remains is trend-cycle. For quarterly data that means a
4-term average, and chapter 4's even-order problem arrives immediately
— a 4-term average has no middle, so a 2×4 is needed to recentre it.

This is where chapter 4's oddly specific 2×m filter pays off. Say so.

**Chart 3 — the trend estimate, with the missing endpoints marked.**

*Reading:* two observations at each end have no estimate, and the chart
should show the gap rather than quietly starting the line later. For
quarterly data with `m = 4` that is four observations lost from
eighty-four — tolerable. For monthly data it is twelve, and for a short
monthly series that is a real cost. The method is silent about the most
recent months, which are the ones anyone forecasting actually cares
about.

### Beat 3 — Extract the seasonality (about 2 pages)

Detrend, then average each season across all years.

**Chart 4 — the detrended series, with all the Q1 values highlighted,
then all the Q2 values, and so on. Or a subseries plot.**

*Reading:* this is the step where the method makes its defining
assumption. Collecting every first quarter and averaging them produces
a single number that stands for "what the first quarter does". The
average is stable and easy to compute, and it says the first quarter
does the same thing every year.

**Chart 5 — the seasonal component plotted over the full twenty years.**

*Reading:* a perfectly repeating sawtooth. Use the verified indices —
`-0.000956, 0.039625, 0.111383, -0.150052` — and point out that these
are bit-identical in year 1 and year 20. Not similar. Identical, by
construction.

Then ask the question the chart provokes: is that plausible? For
Johnson & Johnson between 1960 and 1980, through two decades of changing
product mix and distribution, is the fourth quarter's seasonal effect
really unchanged? Almost certainly not. The method cannot represent a
seasonal pattern that evolves, so it does not.

### Beat 4 — Where it breaks (about 2.5 pages)

Three failures, three charts, and this beat is why chapter 15 exists.

**Chart 6 — classical decomposition of a series with one large
outlier.**

*Reading:* the outlier contaminates the seasonal index for its own
quarter in *every* year, because the index is an average that includes
it. A single strike, flood, or data-entry error propagates across the
whole series. There is no mechanism to downweight it.

**Chart 7 — a series whose seasonal pattern genuinely changes partway
through.**

Construct one: seasonal amplitude that doubles halfway.

*Reading:* the fitted seasonal component is an average of the two
regimes, so it is too large in the first half and too small in the
second, and the error lands in the remainder as a systematic pattern
rather than as noise. Show the remainder's ACF — the structure is
visible, and Part II's tests would flag it.

**Chart 8 — the remainder from a good case and from chart 7's case.**

*Reading:* a remainder with structure is the decomposition telling you
its assumptions were violated. This is the tool from chapter 13's beat 5
doing real work.

### Beat 5 — Why it survives (about 1.5 pages)

Balance, honestly.

*Reading, no chart needed or one small one:* classical decomposition is
fast, requires no parameters beyond the period, produces a result that
anyone can reconstruct by hand, and is entirely transparent about what
it did. For a series with a stable seasonal pattern and no outliers, it
gives essentially the same answer as anything more sophisticated —
chapter 13's comparison put the gap at 0.014 on the log scale.

It is also what "seasonally adjusted" meant for most of the twentieth
century, and understanding it is necessary to read older material.

The case against is narrow and specific: fixed seasonality, no
robustness, missing endpoints. Every one of those is fixable, and the
next chapter fixes them.

### Beat 6 — Where this leaves you (half a page)

Three specific complaints, all of them about rigidity. A method that
let the seasonal pattern evolve, downweighted outliers, and estimated
the endpoints would fix all three.

That method exists and is the subject of chapter 15.

No recap.

---

## 3. The `india` box

Placed in beat 4, after chart 7.

Indian retail and industrial series have seasonal patterns that have
genuinely changed shape over the past two decades — e-commerce has
shifted festival buying earlier, and the introduction of GST in 2017
altered the timing of within-year inventory movements. A method that
forces one fixed seasonal pattern across the whole period will fit the
average of two different regimes and match neither.

For a long Indian series, evolving seasonality is not an edge case. It
is the normal situation.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **fpp3 §3.4** | The clearest step-by-step account of the method, and an honest list of its limitations | The `fable` syntax |
| **Ladiray & Quenneville** | The historical context — this is the ancestor of X-11, and the moving-average machinery is the same | X-11 itself |
| **Cowpertwait & Metcalfe** | Patience with each step; this method rewards being walked through slowly | R specifics |
| **Shumway & Stoffer** | The remainder as a diagnostic | — |
| **Montgomery, Jennings & Kulahci** | Seasonal indices as an operational output — businesses use them directly for planning | — |
| **Hamilton** | Nothing; he does not treat classical decomposition, which is itself informative about its standing in econometrics | — |

**On examples:** `jj`, `aus_production`, `aus_retail` all bundled.
Chart 7's changing-seasonality series is constructed and should be
labelled as such.

---

## 5. Voice

**Do not be dismissive.** The chapter's structure builds towards STL,
and it would be easy to write classical decomposition as a strawman.
It is a genuinely good method within its assumptions, it is still used,
and beat 5 exists to say so properly.

**The frozen-seasonality demonstration should be allowed to land.**
Bit-identical indices twenty years apart is a striking fact and one
sentence of understatement serves it better than three of emphasis.

Avoid, beyond earlier lists:

- "Naive" or "simplistic" applied to the method.
- Rushing beat 5 to get to chapter 15.

---

## 6. Checklist

- [ ] Eight charts through `@example ch14`
- [ ] The 2×m recentring connected explicitly back to chapter 4
- [ ] Missing endpoints shown as a visible gap, not silently trimmed
- [ ] The bit-identical seasonal indices demonstrated with real output
- [ ] Three distinct failure modes shown, each with its own chart
- [ ] Remainder-with-structure used as the diagnostic
- [ ] Beat 5 makes a genuine case for the method
- [ ] `india` box on seasonality that has actually changed
- [ ] Constructed series labelled as constructed
- [ ] Ends by opening chapter 15, no recap
- [ ] 10–11 pages, British-Indian spelling

---

## 7. What to do

1. Re-run the seasonal-index comparison during drafting and print the
   `all.equal` result, or the Julia equivalent, in the text. The claim
   is strong enough to deserve its evidence on the page.
2. Confirm what this package's `classical_decompose` does at the
   endpoints — `NaN`, trimming, or something else — and show it.
3. Build chart 7's changing-seasonality series so the failure is
   unmistakable; a subtle version wastes the chart.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
