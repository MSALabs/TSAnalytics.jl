# Handoff: Chapter 1 — Why Model a Time Series? (revised)

`docs/src/introduction/01-why-model-a-time-series.md`

Supersedes the earlier chapter 1 handoff. Two requirements have changed
it substantially: **every example gets a chart**, and **every chart
gets a detailed reading**. Those two things together make this a
gallery chapter, and gallery chapters are longer.

---

## 1. On length — a deliberate exception

`chapter-writing-guide.md` sets 6–8 pages per chapter and warns that
anything past ten has stopped being selective. **Chapter 1 is an
exception, and the guide should be amended to say so.**

With thirteen charts and a substantial reading after each, this chapter
runs to roughly **14–16 pages**. That is justified for three reasons.
It is the gallery chapter — the one place the reader meets the whole
range of what a time series can be. It is where the reader decides
whether to trust the author, and detailed readings are how that trust
gets built. And the examples introduced here are referenced by name
throughout the rest of the book, so the investment is amortised.

No other chapter gets this licence.

---

## 2. The plotting mechanism is already working

Confirmed this session: `docs/Project.toml` already has `Plots` as a
dependency, and `docs/src/manual/10-plotting.md` already contains six
working `@example` blocks. Follow that established pattern exactly:

````markdown
```@example ch1
using TSAnalytics, Plots
jj = dataset("jj")
plot(jj; title="Johnson & Johnson quarterly earnings", legend=false)
```
````

Use a single named context — `@example ch1` — throughout the chapter so
data loaded early stays available later.

**One inherited caveat.** `10-plotting.md` carries an honest warning
that its blocks render on CI but were not eyeballed locally, because
that session hit a broken GR dependency. If the same happens here, use
the same honest note rather than shipping charts nobody has looked at.

---

## 3. The reading template

This is the new structural requirement, and it must be applied
**consistently to every example**. Five parts, in this order, roughly
150–250 words:

1. **What you are looking at.** Axes, units, the span in years, the
   number of observations, and where the data came from. Concrete.
   "Eighty-four quarterly observations, 1960 to 1980" — not "several
   decades of data".
2. **What is obvious.** What any reader sees in two seconds. Do not
   skip this because it is obvious; naming it teaches the reader what
   to look for next time.
3. **What is not obvious.** The thing you only see on a second look.
   This is where the reading earns its length, and where the author's
   experience shows.
4. **What it means for modelling.** The consequence. Which of the
   book's later tools this series will need, and why.
5. **The question it leaves open.** One sentence, pointing forward.

Do not let this template become mechanical. The headings should not
appear in the text — it is a discipline for the writer, not a
structure for the reader.

---

## 4. The chapter, beat by beat, with all thirteen charts

### Beat 1 — Two series, one question

The best opening hook in the field is fpp3's: some things can be
forecast and some cannot, and the difference is not the amount of data.

**Chart 1 — `vic_elec`, two weeks of half-hourly demand.**
Plot a fortnight, not the whole 52,608 observations. The daily cycle
and the weekday/weekend difference are both visible at that zoom and
invisible at full extent.

*Reading:* the obvious thing is the daily rhythm. The less obvious
thing is that weekends have a different shape, not merely a lower
level — the morning ramp is later and shallower. And there is a second
periodicity underneath the daily one that only becomes clear over a
year. This series has at least three seasonal periods at once, which is
why chapter 16 exists.

**Chart 2 — `gafa_stock`, daily closing prices.**

*Reading:* it trends, it wanders, and it looks like it has structure.
It has almost none that helps you. The best forecast of tomorrow's
price is today's price, and the chapter should say so plainly rather
than hedge. What makes this different from electricity is not
noisiness — it is that the mechanism generating it incorporates
information faster than a forecaster can act on it.

**Chart 3 — the two series' autocorrelation functions, side by side.**
Do not explain the ACF yet; chapter 5 does that. Just show the pictures
and let the reader see that they look nothing alike.

### Beat 2 — The obvious attempt, and why it fails

This is the chapter's most important beat, and it now carries two
charts.

**Chart 4 — a simulated AR(1) with `φ = 0.9`.**

*Reading:* it looks like a series with a wandering level. There is no
trend, no seasonality, and nothing obviously wrong with it. If handed
this without context, most people would compute a mean and a standard
error and move on.

Now do exactly that, in code, and show the confident-looking answer.

**Chart 5 — the demonstration. This is the single most important
figure in the chapter.**

Repeat the experiment five thousand times. Each time, generate a fresh
AR(1) of the same length and record its sample mean. Plot the histogram
of those five thousand means, and overlay the normal curve the textbook
standard error predicts.

The histogram will be roughly four times wider than the curve.

*Reading, and take real space here:* the textbook formula did not fail
loudly. It returned a number, the number had the right units and a
plausible magnitude, and nothing anywhere flagged a problem. The true
variability of the sample mean is about four times what `σ/√n` claims,
because for an AR(1) the variance of the mean is inflated by roughly
`(1 + φ)/(1 − φ)` — nineteen at `φ = 0.9`, and √19 ≈ 4.4.

**Run this and use the real numbers.** If the simulation gives 4.2 or
4.6, use that figure rather than the theoretical one.

Then the point the whole book rests on: this is the characteristic
failure of applying independent-data methods to dependent data. Not an
error, not a warning — a confident wrong answer. Every technique in
Part I exists to prevent some version of it.

### Beat 3 — What is actually going on

**Chart 6 — the ACF of that same AR(1).**

*Reading:* the correlation at lag 1 is about 0.9, at lag 2 about 0.81,
and it decays geometrically. Each observation carries most of the
previous one's information. That is why a hundred of them are worth far
fewer than a hundred independent draws — and it is measurable, which is
the encouraging part. Dependence is not a nuisance to be assumed away;
it is a property to be estimated, and chapter 5 shows how.

Introduce trend, seasonality and remainder here in two paragraphs.
Point at chapter 13. Do not decompose anything yet.

### Beat 4 — The gallery

Five series, each with a chart and a full reading. This is the heart of
the chapter.

**Chart 7 — `jj`, Johnson & Johnson quarterly earnings, 84
observations.**

Shumway & Stoffer's own opening figure, and worth saying plainly that
it is the best single teaching series in the field.

*Reading:* obvious — an upward trend and a strong quarterly pattern.
Less obvious, and the reason this series is famous: the seasonal swings
grow as the level grows. Early swings are a few cents, late ones are
nearly a dollar. The seasonal effect is *proportional*, not additive.
That single observation forces two decisions later — a multiplicative
decomposition in chapter 14, or a log transform in chapter 7 — and the
reader should meet the problem here, before either solution.

**Chart 8 — `jj` again, on a log scale.**

The same series, second view. The growth becomes roughly linear and the
seasonal swings become roughly constant width.

*Reading:* one transformation turned a hard problem into an easy one.
This is the strongest possible motivation for chapter 7, and showing it
costs one extra plot.

**Chart 9 — `gtemp_land` and `gtemp_ocean` on the same axes.**

*Reading:* obvious — both rise. Less obvious — land rises faster than
ocean, and the two series are far from parallel. There is no seasonality
at all here, which makes this a useful contrast with `jj`: trend without
season is a genuinely different modelling problem from trend with
season. Also worth noting the year-to-year jitter, which is real
variation and not measurement error.

**Chart 10 — `soi` and `rec` as two stacked panels, 453 observations
each.**

The Southern Oscillation Index and a fish recruitment series. The most
reused pair in Shumway & Stoffer.

*Reading:* obvious — both oscillate. Less obvious, and this is the
whole point — recruitment follows the SOI with a lag of several months.
Neither series explains itself; each explains the other, displaced in
time. Every method in the book so far looks at one series alone. This
picture is the argument for the ones that do not, and it is why
cross-correlation and Part VII exist.

**Chart 11 — `EQ5` and `EXP6` as two stacked panels, 2,048 points
each.**

An earthquake and a mining explosion, recorded as seismic traces.

*Reading, and slow down here:* the question this pair asks is not
*what happens next*. It is *which of these two things is this* — a
classification problem, on time series data, with real consequences,
since distinguishing an underground nuclear test from an earthquake is
how test-ban treaties are monitored. The two traces differ in the
relative energy of their early and late phases, which is a frequency
question rather than a level question, and points at chapter 6.

Include this. It is the single best defence against the reader
assuming that time series analysis means forecasting and nothing else.

**Chart 12 — `varve`, 634 observations** *(optional; include if the
chapter is not already too long).*

*Reading:* glacial sediment thickness, and the variance visibly grows
with the level, as with `jj` but without seasonality. A second, cleaner
case for transformation.

### Beat 5 — A cautionary tale

**Chart 13 — the corrupted airline series overlaid on the correct
one.**

This is the chapter's most quietly effective figure, and it is worth
constructing carefully. Plot both. They will look almost identical.

*Reading:* the airline passengers series is the most reproduced dataset
in this field — Box and Jenkins, dozens of textbooks, the documentation
of most forecasting packages. The copy circulating inside this project
was wrong. The second row was a shifted duplicate of the first, and the
final year was missing entirely.

It survived several rounds of analysis unnoticed, and this chart shows
why: a corrupted monthly series with a trend and a seasonal pattern
still looks exactly like a monthly series with a trend and a seasonal
pattern. Nothing in the plot says "this is wrong".

It was caught by comparing against `Testairline.spc`, the file the US
Census Bureau ships inside its own X-13ARIMA-SEATS distribution. Every
result derived from it had to be regenerated.

State the operating principle here and let it carry the remaining
forty chapters: **check against a primary source, and do not trust a
number because it came from somewhere reputable.**

Tell this in first person. It is an experience, not a fact, and it
reads better as one.

### Beat 6 — Where this leaves you

Half a page. You have series you cannot handle with the tools you
have. Part I builds the ones you need, starting with a question that
sounds trivial and is not — what a time series *is*, as far as a
computer is concerned.

No recap. No bullet summary.

---

## 5. The `india` box

In beat 1, after chart 1.

Electricity demand in Maharashtra is forecastable for the same reasons
Victorian demand is, with one complication Australian data does not
have. Diwali falls in October some years and November in others, and
industrial and retail demand move with it. A model that assumes October
is always October will be wrong in a way more data cannot fix.

One paragraph. Forward-reference chapter 36.

---

## 6. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **fpp3** | The predictability hook; `vic_elec` | The business-planning framing |
| **Shumway & Stoffer** | The gallery, and specifically `jj`, `soi`/`rec`, `EQ5`/`EXP6`, `gtemp` | The immediate move into formal notation |
| **Hamilton** | That dependence *is* the subject — beat 2's argument | The difference-equation opening |
| **Tsay** | Why financial series are the standard unforecastable case | Returns-versus-prices detail; that is Part V |
| **Montgomery, Jennings & Kulahci** | Forecasting has organisational consequences | The statistics-review chapter |
| **Cowpertwait & Metcalfe** | The gentle entry level — assume less | R-specific scaffolding |

**On covering the canonical examples:** unusually for this book, the
requirement is very nearly satisfiable here. Verified present in the
bundle, with row counts — `jj` (84), `soi` (453), `rec` (453), `EQ5`
(2,048), `EXP6` (2,048), `speech` (1,020), `varve` (634),
`gtemp_land`, `gtemp_ocean`, `gtemp_both`, `nyse` (2,000),
`vic_elec` (52,608), `gafa_stock` (5,032). All GPL-3.

Later chapters will not be this fortunate, particularly Part V, where
Tsay's CRSP series cannot be shipped.

---

## 7. Voice

The general rules are in `chapter-writing-guide.md`. Chapter 1 needs
extra care because a reader decides in two pages whether to trust the
author, and because thirteen readings in one chapter will drift into
formula if nobody is watching.

**Avoid — these are what make prose read as machine-written:**

- Groups of three. "Clear, concise, and compelling." Once is fine.
  Four times is a tell.
- "It is not just X, it is Y."
- Restating the previous paragraph to open the next.
- Summarising a section the reader has just read.
- Uniform sentence length. Vary it hard. Some sentences should be very
  short.
- Hedging every claim. If the standard error is wrong by a factor of
  four, say four.
- "In this chapter we will explore…"
- An em-dash aside in every paragraph.
- **Thirteen readings that all open the same way.** This is the
  specific risk here. Vary the entry. Sometimes start with what is
  obvious; sometimes start with the surprise; sometimes start with the
  history of the data.

**Do:**

- Have opinions. `jj` really is the best teaching series in the field.
  Say so.
- First person, sparingly, where it is genuinely the author's
  experience. Beat 5 especially.
- Be specific. "Eighty-four quarterly observations, 1960 to 1980", not
  "several decades".
- Let paragraphs end flatly. Not everything needs a closing line.
- Use Indian institutional references naturally where they fit — the
  Reserve Bank, MoSPI, the NSE alongside the Fed and the ABS.

---

## 8. Checklist

- [ ] Thirteen charts, all rendering through `@example ch1`
- [ ] Every chart followed by a reading covering all five parts of the
      section 3 template
- [ ] The five parts do **not** appear as visible headings
- [ ] Readings do not all open the same way
- [ ] Chart 5's factor is a **run** number, not the theoretical 4.4
- [ ] `jj` shown twice, linear and log
- [ ] `EQ5`/`EXP6` present, framed as classification not forecasting
- [ ] `india` box after chart 1
- [ ] Airline overlay chart, told in first person
- [ ] Ends by opening chapter 2, no recap
- [ ] 14–16 pages, and `chapter-writing-guide.md` amended to record
      this as a deliberate exception
- [ ] British-Indian spelling
- [ ] Read aloud. If a reading sounds like a template, rewrite it

---

## 9. What to do

1. **Run the beat 2 simulation first**, before writing anything. The
   whole chapter pivots on that figure, and the number must be real.
2. Write the chapter to the beats in section 4.
3. Render every chart. If the local environment cannot (the GR problem
   that affected `10-plotting.md`), say so honestly in the same way
   that page does rather than shipping unseen figures.
4. Amend `chapter-writing-guide.md` to record the length exception,
   so the two documents do not contradict each other.
5. Check against section 8.
