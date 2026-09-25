# Handoff: Chapter 31 — Smoothing

`docs/src/introduction/31-smoothing.md`. Target 10–11 pages.

Chapter 30's filter estimated the state using everything observed *up
to* each moment. This chapter uses everything, including what came
after — and the difference between the two is larger and more useful
than most readers expect.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Framing

This chapter has no external disagreement box and should not invent
one. What it has instead is a **measurable gap**: the filtered and
smoothed estimates of the same state, at the same time point, differ by
an amount that can be computed and plotted. That comparison is the
chapter.

It carries a `julia` box on why the backward pass cannot be fused with
the forward one, and an `india` box on revision.

Chapter 29's export question applies here too. If the state-space API
is internal, this chapter is architecture explanation rather than
tutorial.

---

## 2. The chapter, beat by beat

### Beat 1 — What you knew then, and what you know now (about 1.5 pages)

**Chart 1 — a simulated local level: the true state, the filtered
estimate, and the smoothed estimate, all three drawn.**

Simulation matters here because the truth must be visible.

*Reading:* the filtered line lags and wobbles. The smoothed line sits
closer to the truth almost everywhere, and is visibly steadier. The gap
is widest early in the series, where the filter had least information
and the smoother had the whole future to draw on.

State the asymmetry plainly. At time 10 the filter had ten
observations. The smoother had all fifty. It would be strange if the
smoother were not better, and the interesting question is by how much
and where.

**Chart 2 — the difference between filtered and smoothed estimates,
plotted against time.**

*Reading:* large at the start, shrinking through the middle, and
**exactly zero at the final observation** — because at the last point
there is no future to add, so the two estimates coincide by
construction. That endpoint identity is a good check on any
implementation and worth stating as one.

### Beat 2 — Running backwards (about 2 pages)

**Chart 3 — the backward recursion drawn as a diagram, beneath chapter
30's forward one.**

*Reading:* the filter sweeps forward, storing what it computed at each
step. The smoother then sweeps backward, revising each estimate using
the information that arrived later. Two passes, and the second cannot
begin until the first has finished.

Explain the mechanism without the algebra: the backward pass carries a
correction term that accumulates the future's disagreement with what
the filter believed at the time, and applies it in proportion to how
uncertain the filter was.

**`julia` box here.** The two passes cannot be fused. The backward
recursion needs the forward pass's stored quantities at every step, so
the whole filtered path must be retained rather than discarded as it
goes. That is a real memory cost for long series, and it is why some
implementations offer a filter-only mode. Half a page, concrete.

### Beat 3 — Uncertainty shrinks too (about 2 pages)

**Chart 4 — filtered and smoothed uncertainty bands on the same axes.**

*Reading:* the smoothed band is narrower everywhere except at the final
point, where the two coincide. The improvement is not merely a better
point estimate; it is genuinely more information about where the state
was.

Report the ratio of the two variances at a few time points — early,
middle, and at the end. The numbers make the improvement concrete in a
way the picture alone does not.

**Chart 5 — the same comparison for a model with high observation
noise and one with low.**

*Reading:* when observations are precise, the filter is already close
and the smoother adds little. When they are noisy, the smoother adds a
great deal. The value of looking backward depends on how much the
forward pass had to guess.

### Beat 4 — What it is actually for (about 2.5 pages)

Three applications, and this is where the chapter earns its place.

**Chart 6 — a decomposition produced by smoothing: a real series with
its smoothed level and its smoothed seasonal component.**

*Reading:* this is chapter 13's decomposition problem solved by a
model rather than a filter. The components are estimates from a fitted
model with uncertainty attached, which classical decomposition and STL
never provide. Chapter 13 asked which trend was correct and answered
that none was; here the trend at least comes with an interval.

Point forward to chapter 41's unobserved components models, which are
this idea taken seriously.

**Chart 7 — smoothed residuals used to find an outlier.**

*Reading:* disturbance smoothing estimates the individual shocks that
drove the series. A large estimated observation-noise shock at one
point suggests a measurement error; a large state shock suggests a
genuine break. The two are distinguishable, which no method in Part I
could do.

Connect back to chapter 15's robustness weights — a different mechanism
answering a related question.

**Chart 8 — a missing stretch, filtered and smoothed.**

*Reading:* the filter's uncertainty grows through the gap and stays
grown until data resumes. The smoother's grows and then *shrinks again*
from the far side, because observations after the gap say something
about what happened during it. The smoothed band through a gap is
lens-shaped rather than a widening cone.

That picture is the clearest possible statement of what smoothing adds,
and if only one chart from this chapter survives, it should be this
one.

### Beat 5 — When not to (about 1 page)

*Reading, short:* smoothing uses the future, which means it cannot be
used for anything that must be computed in real time. A smoothed
estimate of last month's level is not something you could have known
last month.

That matters for anyone evaluating a forecasting method. Comparing a
forecast against a smoothed estimate of the truth is comparing against
something that used the answer. The filtered estimate is the honest
comparison, and chapter 23's discipline applies here too.

Also: revisions. If a published statistic is produced by smoothing, it
will change when more data arrives, and that is correct behaviour
rather than an error.

### Beat 6 — Where this leaves you (half a page)

You can estimate a state forwards and backwards, with uncertainty, and
handle gaps honestly.

Every model so far has had matrices that stay put. Sometimes they
should not — a regression coefficient that drifts, a seasonal pattern
that evolves, a system whose structure genuinely changes.

Chapter 32.

No recap.

---

## 3. The `india` box

Placed in beat 5, after the revision point.

Indian statistical releases are revised, sometimes substantially, and
the revisions are frequently treated in public commentary as errors or
worse. Some of them are exactly what this chapter describes — an
estimate that used the information available at the time, updated when
more arrived.

That is not a defence of every revision. Methodology changes and base
revisions are a different matter. But a quarterly figure that moves when
the next quarter is published is behaving the way a filtered estimate
behaves, and the distinction between "revised because more data
arrived" and "revised because the method changed" is worth being able
to make.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Durbin & Koopman chs. 4–5** | The definitive treatment, including disturbance smoothing, which almost no other introductory source covers | The full derivations |
| **Shumway & Stoffer ch. 6** | Smoothing presented with plots, and the filtered-versus-smoothed comparison as a picture | The EM algorithm |
| **Harvey via Durbin & Koopman** | Smoothed components as an interpretable decomposition — the beat 4 application | The structural taxonomy |
| **Hamilton ch. 13** | The formal statement of the backward recursion | The algebra |
| **Ladiray & Quenneville** | The revision problem, which official statistics has thought about harder than anyone | X-11 mechanics |
| **fpp3** | Nothing; smoothing in this sense is outside its scope | — |

**On examples:** simulated local level for beats 1–3 so truth is
visible; `gtemp_land`, `GNP23` or an Indian series for beat 4's
decomposition.

---

## 5. Voice

**Chart 8 is the chapter's best moment.** A lens-shaped uncertainty
band through a gap says everything about what smoothing adds, in one
picture, with no explanation needed. Build it carefully.

**Beat 5's warning is genuinely important.** Evaluating forecasts
against smoothed estimates is a real and common error, and it inflates
apparent accuracy. It belongs in this chapter because this is where the
reader first has a smoothed estimate to misuse.

**Do not derive the backward recursion.** The diagram and the intuition
carry it.

Avoid, beyond earlier lists:

- Presenting smoothing as strictly better than filtering. It answers a
  different question and cannot be used in real time.
- The term "fixed-interval smoothing" without explaining that it means
  what the chapter has been doing all along.

---

## 6. Checklist

- [ ] Eight charts through `@example ch31`
- [ ] True state visible in beats 1–3 via simulation
- [ ] Filtered-minus-smoothed difference shown, **zero at the final
      point** and stated as an implementation check
- [ ] Uncertainty ratio reported at several time points
- [ ] `julia` box on why the passes cannot be fused, and the memory cost
- [ ] Smoothed decomposition shown, with uncertainty, connected to
      chapter 13
- [ ] Disturbance smoothing used to distinguish measurement error from
      a genuine break
- [ ] **Lens-shaped band through a gap** — chart 8
- [ ] Real-time caveat stated, with the forecast-evaluation warning
- [ ] `india` box on revisions
- [ ] Ends by opening chapter 32, no recap
- [ ] 10–11 pages, British-Indian spelling

---

## 7. What to do

1. Confirm the export decision still governs how this chapter is
   written.
2. **Verify the endpoint identity** — filtered and smoothed estimates
   must coincide exactly at the last observation. If they do not in
   this package, that is a bug and the chapter has found it.
3. Build chart 8 first; it is the chapter's centrepiece and the rest
   can be arranged around it.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
