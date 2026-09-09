# Handoff: Chapter 13 — Components

`docs/src/introduction/13-components.md`. Target 9–10 pages.

The shortest chapter in Part III and the one that sets up the other
three. It answers a question the reader has been assuming an answer to
since chapter 1: what are the pieces a time series is made of?

The honest answer is that this is a modelling choice rather than a fact
about the data, and that is the chapter.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

### Two methods, two different trends, same series

Run this session on log `jj`:

```
classical trend, first 6:   NA        NA        -0.465982  -0.473586  -0.452323  -0.414538
STL trend,       first 6:   -0.361477 -0.407482 -0.451935  -0.481076  -0.465506  -0.405894

max |difference| where both are defined:  0.014047
classical NAs:  4        STL NAs:  0
```

**Same data, same transform, two respected methods, two different
answers** — and one of them declines to estimate the first two and last
two observations at all.

This is the chapter's central point and it should not be buried.
"The trend" is not something the series has and the method reveals. It
is something the method constructs, and a different method constructs a
different one. Neither is wrong.

The 0.014 gap is on the log scale, which understates it — in level
terms it is a difference of about 1.4% in the trend estimate at the low
end of the series.

---

## 2. The chapter, beat by beat

### Beat 1 — Three things at once (about 1 page)

**Chart 1 — `jj`, with the three components sketched by hand over it.**

Draw the raw series and annotate: the upward sweep, the four-quarter
wiggle, the leftover jitter.

*Reading:* the reader has been describing this series in these terms
since chapter 1 without anyone defining them. Trend, seasonality and
remainder are the vocabulary everyone reaches for, and the reason they
work is that they correspond to genuinely different mechanisms — a
company growing, a calendar repeating, and everything nobody modelled.

### Beat 2 — Adding or multiplying? (about 2 pages)

**Chart 2 — an additive series and a multiplicative one, side by side.**

Construct both so the difference is unmistakable: constant seasonal
amplitude against amplitude that grows with the level.

*Reading:* in the additive case the seasonal swing is the same size in
year one and year twenty. In the multiplicative case it scales.
Real series are usually multiplicative, because most things that grow
do so proportionally — a company selling twice as much has twice the
Christmas peak, not the same Christmas peak.

**Chart 3 — the multiplicative series, logged, beside the additive
one.**

*Reading:* they now look like the same kind of object, because taking
logs converts multiplication into addition exactly. Show the identity —
`log(T × S × R) = log T + log S + log R` — and note that this is why
chapter 7 came before Part III rather than after. The transform is not
a convenience; it is what lets one decomposition machinery serve both
cases.

Worth stating the practical consequence: a multiplicative decomposition
and an additive decomposition of the logs are the same operation, and
most software offers both because users expect both, not because they
differ.

### Beat 3 — Trend, or trend-cycle? (about 1.5 pages)

**Chart 4 — a long series where the trend estimate visibly wanders
rather than climbing steadily.** `global_economy` for a country with
recessions, or `GNP23`.

*Reading:* the "trend" here is not a straight line and not close to
one. It rises, flattens, dips, rises again. What the method has
extracted is really trend *and* business cycle together, which is why
the careful name is **trend-cycle**.

Classical decomposition cannot separate them, and neither can STL. The
cycle is slow, the trend is slower, and no filter distinguishes two
things that differ only in degree. Chapter 41's unobserved components
models can, because they impose structure rather than filtering — and
that is a genuine reason to reach for them.

This distinction is skipped in most introductory treatments and it
matters. A reader who thinks "trend" means "the underlying growth path"
will misread every decomposition in the next three chapters.

### Beat 4 — Whose trend? (about 2.5 pages)

The chapter's payload.

**Chart 5 — classical and STL trends for log `jj`, overlaid.**

*Reading:* use the verified numbers. The two curves track each other
and do not coincide, differing by up to 0.014 on the log scale. More
strikingly, the classical trend simply does not exist for the first two
and last two observations, while STL's does.

**Chart 6 — the same comparison zoomed on the final two years.**

*Reading:* the end of a series is where forecasters look, and it is
exactly where the two methods diverge most and where one of them gives
up. Classical decomposition's moving average needs observations on both
sides; at the end of the series there are none to the right. STL
estimates anyway, using a one-sided fit, which is a choice with its own
consequences rather than a free lunch.

Then the `disagreement` box, framed the way this chapter needs it: not
two packages disagreeing about one quantity, but two definitions of a
quantity that has no independent existence. **There is no true trend to
be right or wrong about.** Every trend is the output of a filter, and
choosing a filter is choosing what counts as trend.

That framing is worth the space because it defuses a question readers
ask constantly — which decomposition is correct — that has no answer.

### Beat 5 — What remains (about 1 page)

**Chart 7 — the remainder from a decomposition, with its ACF beside
it.**

*Reading:* the remainder is defined by subtraction — it is whatever the
method did not assign elsewhere. That makes it the natural place to
check whether the decomposition worked, using exactly the tools of
Part II. If the remainder still has structure at the seasonal lag, the
seasonal component did not capture everything.

Note the circularity honestly: the remainder is not an estimate of
anything, it is a residue, and its properties depend entirely on the
method that produced it.

### Beat 6 — Where this leaves you (half a page)

You have vocabulary and a warning. The next three chapters give three
methods, in increasing order of flexibility, and each constructs its
components differently.

Chapter 14 starts with the oldest and simplest, which is still the one
most people meet first.

No recap.

---

## 3. The `india` box

Placed in beat 3, after chart 4.

Indian GDP growth is routinely discussed as though the trend were a
fixed rate that the economy deviates from temporarily. The trend-cycle
distinction says that is a modelling assumption, not an observation —
and the choice matters for policy. If a slowdown is cycle, it reverses
on its own. If it is trend, it does not.

No decomposition in Part III can settle that question, because none of
them separates trend from cycle. It is worth knowing which questions
your tools cannot answer before you use them to answer one.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **fpp3 §3.2** | The components framing and the additive/multiplicative contrast, presented more clearly than anywhere else | The `fable` syntax |
| **Ladiray & Quenneville** | The trend-cycle terminology and the insistence on being precise about it — official statistics has been careful about this for decades | X-11 mechanics |
| **Shumway & Stoffer** | The remainder as an object to be tested, not discarded | — |
| **Hamilton** | The observation that decomposition is identification by assumption, which is the beat 4 point stated formally | The full identification theory |
| **Montgomery, Jennings & Kulahci** | Components as separately actionable — different parts of a business respond to trend and to seasonality | — |
| **Cowpertwait & Metcalfe** | The gentle introduction to the idea itself | R specifics |

**On examples:** `jj`, `global_economy`, `GNP23`, `aus_production` all
bundled.

---

## 5. Voice

**Beat 4 is the chapter.** Everything before it is vocabulary. A
version that defines the three components and moves on has skipped the
only idea here that a reader will not already have absorbed.

**Do not resolve the "which trend is correct" question.** It has no
answer and pretending otherwise would be the one dishonest move
available in this chapter.

Avoid, beyond earlier lists:

- Calling the remainder "noise". It is a residue, and it frequently has
  structure — chapter 11 was entirely about a case where it did.
- Formal decomposition notation before chart 3's log identity.

---

## 6. Checklist

- [ ] Seven charts through `@example ch13`
- [ ] Log identity shown, connecting to chapter 7
- [ ] Trend-cycle distinction made explicit, with its own chart
- [ ] Classical and STL trends overlaid, using the verified numbers
- [ ] The endpoint difference (4 NAs versus 0) stated
- [ ] `disagreement` box framed as "no true trend exists", not as one
      method being better
- [ ] Remainder shown with its ACF, and its circularity acknowledged
- [ ] `india` box on trend versus cycle in growth debates
- [ ] Ends by opening chapter 14, no recap
- [ ] 9–10 pages, British-Indian spelling

---

## 7. What to do

1. Re-run the classical-versus-STL comparison during drafting. The
   section 1 numbers are real; the chapter's should come from its own
   run.
2. Choose the chart 4 series deliberately — the trend-cycle point needs
   a series where the extracted trend visibly wanders. Try several.
3. Render every chart; honest CI note if the environment cannot.
4. Check against section 6.
