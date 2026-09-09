# Handoff: Chapter 8 — Stationarity

`docs/src/introduction/08-stationarity.md`. Target 11–12 pages.

Part I built tools. Part II asks whether you are entitled to use them,
and this chapter states the condition that almost all of them assume.

Same requirements as Part I. Reading template from chapter 1.

---

## 1. A disagreement box this chapter was not allocated

`all-chapters-handoff.md` listed chapter 8 as conceptual, with no box.
That was a missed opportunity. There is a demonstration here that is
more alarming than anything in Part I, and it is the reason
stationarity is worth a chapter rather than a definition.

**Granger and Newbold's spurious regression, verified this session:**

```
Two INDEPENDENT random walks, n = 100, 2000 trials
  regression significant at 5%:  75.2% of the time
  (a valid test would reject 5% of the time)

Same test on independent white noise:  6.3%
```

Two series with **no relationship whatsoever** — generated from
separate random number streams, sharing nothing — produce a
statistically significant regression three times in four. On white
noise the same procedure behaves correctly, rejecting 6.3% of the time
against a nominal 5%.

This is not a subtle bias. It is a test that is wrong far more often
than it is right, on data where the truth is known by construction.
And nothing in the output looks unusual: the t-statistic is large, the
p-value is small, the R² is often respectable.

Frame it as a `disagreement` box in the broader sense the book uses —
not two packages disagreeing, but a standard method disagreeing with
reality. It earns the format.

---

## 2. The chapter, beat by beat

### Beat 1 — Two series that appear to be related (about 1.5 pages)

**Chart 1 — two independent random walks on the same axes.**

Do not say yet that they are independent. Let the reader look.

*Reading:* they rise together for a stretch, diverge, and converge
again. Anyone shown this pair in a report would accept a claim that
they are connected. The visual impression is strong and it is entirely
false — these came from two separate random number generators and share
nothing but the fact that both accumulate their own noise.

**Chart 2 — a scatterplot of one against the other, with the fitted
line and the regression output.**

*Reading:* the line fits well. The slope is significant. The R² is
substantial. Every number a regression normally produces to reassure
you is present and reassuring. Report the actual t-statistic and
p-value from the run — the specific numbers matter more than a
description.

### Beat 2 — How often does this happen? (about 2 pages)

The obvious response is that this was a fluke. Test it.

**Chart 3 — the distribution of t-statistics from 2,000 such
regressions, with the ±1.96 critical values marked.**

*Reading, and take real space:* if the test were valid, roughly 5% of
the mass would lie outside the critical values. Instead 75.2% does.
The distribution is far wider than the t-distribution the test assumes,
so the critical values are in the wrong place — not slightly, but by a
factor that makes the procedure worse than useless.

**Chart 4 — the same experiment on white noise. Two panels, side by
side with chart 3.**

*Reading:* 6.3%, which is what a working test looks like. The
difference between the two panels is the entire content of this
chapter. The regression machinery is not broken. It is being applied to
data that violates an assumption it never states out loud.

Name the assumption now, having earned it.

### Beat 3 — What stationarity actually requires (about 2 pages)

Weak stationarity: constant mean, constant variance, and an
autocovariance that depends only on the gap between two observations
and not on where they sit. Three conditions, each with a picture.

**Chart 5 — rolling mean and rolling variance of a stationary series
and of a random walk. Four panels.**

*Reading:* for the stationary series both rolling statistics hover
around a fixed level. For the random walk the rolling mean wanders and
the rolling variance climbs steadily — its variance genuinely grows
with time, which is what "no fixed distribution to sample from" looks
like in practice. This is the most direct visual test of stationarity
available and it requires no theory at all.

Mention strict stationarity in two sentences and move on. Weak
stationarity is what every method in this book actually needs.

### Beat 4 — A gallery of failures (about 2 pages)

**Chart 6 — four non-stationary series, four different reasons. Four
panels.**

A trending series; a series with a variance that grows (`varve` or
`jj`); a series with a level shift partway through; a seasonal series.

*Reading:* each violates a different condition, and each needs a
different remedy — differencing, a transformation, an intervention
term, seasonal differencing. Lumping them together as "non-stationary"
is accurate and unhelpful. The diagnosis matters because the treatment
differs, and Part I supplied all four treatments before this chapter
explained what they were for.

### Beat 5 — The grey zone (about 2 pages)

**Chart 7 — an AR(0.95) process and a random walk, side by side.**

*Reading:* the AR(0.95) is stationary. Its coefficient is less than one,
it has a fixed mean it returns to, and every theorem in the book
applies to it. The random walk is not stationary. They are, at this
sample size, essentially indistinguishable by eye — and, as chapter 9
will show, nearly indistinguishable by test as well.

This matters because stationarity is a property of the process, not of
the data, and you only ever see the data. A finite stretch of a
strongly persistent stationary series looks exactly like a
non-stationary one. There is no procedure that reliably separates them
at ordinary sample sizes, and pretending otherwise is how the spurious
regression in beat 1 gets published.

Connect back to chapter 3 explicitly. That chapter showed two series
needing opposite treatment and promised a test would settle it. This
beat is the first hint that the test has limits.

### Beat 6 — Where this leaves you (half a page)

You know what stationarity is, why its absence is dangerous, and that
the eye cannot reliably detect it.

Chapter 9 introduces the tests. They are better than the eye. They are
not as good as you would like.

No recap.

---

## 3. The `india` box

Placed in beat 4, after chart 6.

Almost every headline Indian macroeconomic series — GDP, IIP, the CPI,
bank credit — is non-stationary in level and roughly stationary in
growth rate. That is why Indian policy discussion is conducted almost
entirely in growth terms rather than levels, and why the RBI's own
publications lead with year-on-year change.

The convention is not merely presentational. Regressing one Indian
level series on another reproduces exactly the beat 1 problem, and the
resulting "relationship" between, say, credit growth and output can be
almost entirely an artefact of both series trending upward over three
decades.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Hamilton** | The formal definition and, more importantly, the seriousness with which he treats the consequences of getting it wrong | The asymptotic apparatus |
| **Granger & Newbold via Hamilton and Tsay** | The spurious regression result itself — the single most persuasive argument for this chapter's existence | — |
| **fpp3 §9.1** | The practical, visual framing — a stationary series looks the same wherever you cut it | The `fable` machinery |
| **Shumway & Stoffer** | The weak-versus-strict distinction, handled cleanly and briefly | Ergodicity |
| **Tsay** | That returns are stationary and prices are not, which is why finance models returns | Distributional detail |
| **Cowpertwait & Metcalfe** | Rolling statistics as a first diagnostic | R specifics |
| **Montgomery, Jennings & Kulahci** | Level shifts as a distinct, practically common failure | Control charts |

**On examples:** `varve`, `jj`, `gtemp_land`, `GNP23` all bundled.
The random-walk demonstrations are simulated and should be labelled so.

---

## 5. Voice

**The spurious regression should feel like a shock.** It is one, and
readers who have run regressions on economic data will recognise
uncomfortably that they may have published one. Do not soften it with
qualifications before the reader has felt it.

**Do not define stationarity first.** Every other book does, and it
makes the concept feel like a technicality to be satisfied rather than
a warning to be heeded. The definition arrives in beat 3, after two
beats establishing why anyone should care.

Avoid, beyond earlier lists:

- Ergodicity. It is a genuinely different condition, it is not needed
  anywhere in this book, and mentioning it costs a paragraph and buys
  nothing.
- "Stationarity is a key assumption." Show it, do not assert it.

---

## 6. Checklist

- [ ] Seven charts through `@example ch8`
- [ ] Charts 1 and 2 withhold that the series are independent
- [ ] The 75.2% figure comes from a live run, not this handoff
- [ ] White-noise control shown beside it
- [ ] Definition arrives in beat 3, not beat 1
- [ ] Rolling mean and variance shown for both a stationary and a
      non-stationary series
- [ ] AR(0.95) versus random walk shown as genuinely indistinguishable
- [ ] Explicit callback to chapter 3's unresolved question
- [ ] `india` box on level versus growth
- [ ] Simulated series labelled as simulated
- [ ] Ends by opening chapter 9, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 7. What to do

1. Re-run the spurious regression experiment during drafting. The 75.2%
   figure is real but the chapter's number should come from its own
   run, and the t-statistic reported in beat 1 must come from the
   specific pair plotted in chart 1.
2. Pick the chart 1 pair deliberately. Generate several and choose one
   where the visual impression of a relationship is strong — that is
   not cherry-picking, it is illustrating the phenomenon the experiment
   quantifies.
3. Render every chart; honest CI note if the environment cannot.
4. Check against section 6.
