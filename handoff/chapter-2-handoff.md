# Handoff: Chapter 2 — What a Time Series Is

`docs/src/introduction/02-what-a-time-series-is.md`. Target 11–13 pages.

Chapter 1 showed the reader a gallery and told them their statistics
toolkit will mislead them. This chapter asks a question that sounds
trivial and is not: what is a time series, as far as a computer is
concerned? The answer determines what you are able to compute, and
getting it wrong silently limits everything downstream.

Same requirements as chapter 1 — best from each book, story structure,
a chart for every example, a detailed reading after every chart, and
prose that reads as though a person wrote it. The reading template in
section 3 of the chapter 1 handoff applies unchanged; do not repeat it,
apply it.

---

## 1. An honest note on chart count

Chapter 1 carried thirteen charts because it was a gallery of data.
This chapter is about *representation*, and representation is less
naturally visual. It supports **nine charts** that genuinely earn their
place.

**Do not pad it to thirteen.** A manufactured chart is worse than an
absent one, and the checklist in chapter 1's handoff was never meant to
set a quota. Nine well-chosen figures with substantial readings is the
right shape here.

---

## 2. The argument

A time series is not a vector of numbers. It is **values, plus an
index, plus an implied claim about the spacing between them** — and
different systems make that claim in radically different ways, with
consequences most users never see until something breaks.

The chapter's spine, and it is verifiable in the reader's own copy of
the bundled data:

> Two datasets that ship with this package store time in completely
> incompatible ways. Neither is wrong. Understanding why is the whole
> chapter.

---

## 3. The findings this chapter is built on

Both verified this session, with output.

### R's `ts` stores three numbers and computes the rest

```r
x <- ts(c(10,20,30,40,50), start=c(1960,1), frequency=4)
attributes(x)
#> $tsp
#> [1] 1960 1961    4
```

That is the entire time representation — start, end, frequency. The
time axis is *derived*: `time(x)` returns `1960.00, 1960.25, 1960.50,
1960.75, 1961.00`. There are no dates stored anywhere.

**Two consequences follow, and they are the chapter's payload.** An R
`ts` object cannot represent irregular spacing at all — the spacing is
`1/frequency` by construction, always. And `1960.25` means "second
quarter of 1960" only by convention; nothing in the object says so.

This is visible in the bundled data. `jj.csv`, converted from `astsa`,
begins:

```
"time","value"
1960,0.71
1960.25,0.63
1960.5,0.85
```

While `vic_elec.csv`, from `tsibbledata`, begins:

```
"Time","Demand","Temperature","Date","Holiday"
"2012-01-01",4382.825174,21.4,"2012-01-01",TRUE
"2012-01-01 00:30:00",4263.365526,21.05,"2012-01-01",TRUE
```

Decimal years in one, timestamps in the other, in the same package.
That is not sloppiness — it is two ecosystems' genuinely different
answers to the question this chapter asks.

### Real data is not as regular as its description

`vic_elec` is described as half-hourly. Counting the observations per
calendar day:

```
48 half-hours: 1090 days
50 half-hours:    3 days   (2012-04-01, 2013-04-07, 2014-04-06)
46 half-hours:    3 days   (2012-10-07, 2013-10-06, 2014-10-05)
```

Those six dates are the Australian daylight-saving transitions. In
April the clocks go back and the day gains an hour; in October they go
forward and it loses one.

**A representation that computes timestamps from start-plus-frequency
cannot express this.** It is not an edge case in synthetic data — it is
in a headline dataset shipped with this package, and it will silently
misalign every calculation that assumes 48.

---

## 4. The chapter, beat by beat

### Beat 1 — What did you actually need? (about 1 page)

**Chart 1 — `jj`, plotted.**

Open with a plot the reader has already seen in chapter 1, and ask a
question they have not considered: to draw this, what did the computer
need to know?

*Reading:* the values, obviously. But also that the observations are
quarterly, that the first one is the first quarter of 1960, and that
they are evenly spaced. Remove any of the three and the picture changes
or becomes impossible. That trio — values, index, spacing — is what a
time series actually is, and the rest of this chapter is about how
different systems store it.

### Beat 2 — A vector should be enough (about 1.5 pages)

The obvious attempt: surely a `Vector{Float64}` is a time series?

**Chart 2 — `jj` plotted from a bare vector, beside `jj` plotted with
its proper index.** Two panels, identical shapes, different x-axes.
One reads 1 to 84; the other reads 1960 to 1980.

*Reading:* the shape survives, which is why this mistake is easy to
make and hard to notice. What is lost is everything calendar-shaped.
You cannot ask which observation is a December. You cannot align this
series with another that starts in a different year. You cannot say
what "one year ahead" means. The vector is sufficient for plotting and
insufficient for almost everything else — and nothing about the plot
tells you so.

### Beat 3 — Index, values, spacing (about 2 pages)

Now the real content. Introduce the three components properly, then the
container problem.

**Chart 3 — the same series in four containers, four panels,
identical plots.**

`Vector`, `DataFrame`, `TSFrame`, `TimeArray`. Four types, one picture.

*Reading:* the plots are identical because the *data* is identical. The
types differ in what they let you do next, not in what they contain.
A package facing this has two options — convert everything to one
canonical type on entry, which is what R and Python largely do, or
accept all of them and dispatch. TSAnalytics.jl takes the second route,
which is what `tsvalues` and `tsindex` are for.

**`julia` box here.** This is the clearest example in the entire book of
a Julia-specific design payoff. Multiple dispatch means the package does
not need a conversion layer, a canonical type, or an `as_ts()` function
littered through the API. Show the two methods and let it speak for
itself. Keep it to half a page — the point is sharp and does not need
elaboration.

### Beat 4 — Where the tidy story breaks (about 4 pages)

The longest beat, and the chapter's substance. Four charts.

**Chart 4 — `vic_elec` at four aggregations: half-hourly, daily,
weekly, monthly. Four panels.**

*Reading, and take real space:* these are the same data. They look like
four different series. At half-hourly resolution the daily cycle
dominates and the annual pattern is invisible. At monthly resolution
the annual pattern is obvious and the daily cycle has vanished
completely — not smoothed, *gone*, because aggregation destroyed it.

Frequency is therefore not a property of the underlying phenomenon. It
is a choice made when the data was recorded, and a second choice made
whenever it is aggregated. Both choices determine what questions can
be asked. This is worth internalising before chapter 13 asks the reader
to decompose anything, because "the seasonal component" means nothing
until the frequency is settled.

**Chart 5 — the daylight-saving days. Two panels: 2012-04-01 with its
50 half-hours, and 2012-10-07 with its 46.**

*Reading:* `vic_elec` is described as half-hourly, and 1,090 of its
1,096 days have 48 observations. Six do not. Those six are the
Australian daylight-saving transitions, and they are not a data-quality
problem — the data is correct. The *description* is what is imprecise.

This is where R's `ts` representation runs out of road. An object that
stores start, end and frequency computes its timestamps as
`start + k/frequency`, so it cannot express a day with 50 half-hours.
Forced to hold this data, it would silently misalign everything after
1 April 2012. Nothing would error.

**Chart 6 — `gafa_stock` plotted against trading-day index versus
against calendar date. Two panels.**

*Reading:* against a trading-day index the series is continuous. Against
calendar dates it is full of small gaps — weekends and market holidays.
Both are legitimate views and they answer different questions. If you
want to know how the price moved over five trading sessions, the first
is right. If you want to relate the price to something measured on
calendar time — weather, a policy announcement, a festival — the second
is the only one that works. Choosing wrongly produces a plausible,
wrong answer, which by now the reader should recognise as this book's
recurring villain.

**Chart 7 — missing observations. Three panels of the same series with
a gap: plotted as a break, as an interpolation, and with the missing
points silently dropped.**

*Reading:* the third panel is the dangerous one. Dropping missing
observations does not create a hole in the picture; it creates a series
that is shorter and subtly mis-spaced, and looks entirely healthy.
Every subsequent calculation inherits the distortion. State the
package's own behaviour here plainly, whatever it is, so the reader
knows which of these three they are getting.

### Beat 5 — The two-representations problem (about 1.5 pages)

The `disagreement` box, and the chapter's sharpest moment.

**Chart 8 — decimal-year time plotted as though it were a date.**

Take `jj`'s time column — `1960, 1960.25, 1960.5` — and hand it to
something expecting dates. The result is nonsense, and visibly so.

*Reading:* two datasets ship with this package. One stores time as
decimal years because it came from R's `ts`, where the time axis is
computed from three numbers. The other stores real timestamps because
it came from a `tsibble`, where every observation carries its own.
Neither ecosystem is wrong. R's representation is compact, fast, and
makes regular spacing unrepresentable-as-anything-else, which is a
genuine feature for the regular case. The `tsibble` representation is
larger and handles the daylight-saving day without comment.

They are simply incompatible, and a package that wants to accept data
from both worlds has to know which it is holding. That is why
`tsindex` exists and why it is not a trivial function.

Worth stating the honest consequence: the `astsa`-derived datasets in
this package still carry decimal-year time. That is a real wart,
documented rather than hidden, and the reader will meet it the first
time they load one.

### Beat 6 — Where this leaves you (half a page)

**Chart 9 — a series and its first difference, two panels.**

No explanation. Just the picture, and one sentence: the second panel is
what chapter 3 is about, and it is the first thing anyone does to a
series that will not sit still.

No recap.

---

## 5. The `india` box

Placed in beat 4, after chart 6.

The Indian financial year runs April to March, not January to December.
A quarterly series from MoSPI has its Q1 in April, and a package that
assumes calendar quarters will label every observation wrongly — not
approximately, but by an entire quarter. The same problem appears in
any market whose fiscal calendar is not the Gregorian one, and it is a
pure indexing problem: the values are fine, the index is wrong, and
nothing in a plot will tell you.

One paragraph. This makes the chapter's abstract point about indices
concrete and consequential.

---

## 6. What each book contributes

Chapter 2 has a very different contribution profile from chapter 1.
Most of these books barely address representation at all.

| Book | Take | Leave |
|---|---|---|
| **fpp3** | §2.1 on `tsibble` objects — the only reference book that treats the data structure as worth a section of its own, and the source of the index-plus-key framing | The tidyverse-specific machinery |
| **Cowpertwait & Metcalfe** | The gentle introduction to R's `ts` — they explain `start`/`frequency` more patiently than anyone | The R-only worldview |
| **Montgomery, Jennings & Kulahci** | Data collection and quality as a first-class concern, which fits beat 4's missing-data material | The statistics review |
| **Shumway & Stoffer** | `jj` as the running example, and their Appendix R on data handling | Their fairly quick move past representation |
| **Tsay** | That the *choice* of what to model — prices or returns — is itself a representation decision | Financial detail; that is Part V |
| **Hamilton** | Nothing. He does not discuss data representation at all, and it is worth noticing that the most rigorous book in the field skips it entirely | — |

**On the examples requirement:** every series named here is bundled and
GPL-3 — `jj`, `vic_elec`, `gafa_stock`. Verified present with row
counts. This chapter needs no external data.

---

## 7. Voice

All of chapter 1's guidance applies. Two risks specific to this
chapter:

**This material is genuinely dry, and dry writing about dry material is
fatal.** The saving grace is that every abstraction here has a concrete
failure attached — the daylight-saving day, the decimal years, the
fiscal-year misalignment. Lead with the failure, then explain the
abstraction. Never the reverse.

**Do not let it become a comparison of data structures.** The reader
does not care about `TSFrame` versus `TimeArray` and should not be made
to. They care that their October figure was labelled wrongly. Keep the
consequences in front and the types behind.

Specific things to avoid, beyond chapter 1's list:

- Explaining multiple dispatch at length. One box, half a page, done.
- Comparative tables of container features. That belongs in the Manual.
- Any sentence beginning "It is important to note that".

---

## 8. Checklist

- [ ] Nine charts, all rendering through `@example ch2`
- [ ] Nine readings, following the five-part template, none opening the
      same way
- [ ] The `tsp` finding shown with real R output
- [ ] The daylight-saving counts stated exactly: 1090 / 3 / 3, with the
      six dates named
- [ ] `jj.csv` and `vic_elec.csv` raw heads shown side by side
- [ ] `julia` box on multiple dispatch, half a page, not more
- [ ] `india` box on the April–March fiscal year
- [ ] The package's own missing-data behaviour stated plainly
- [ ] The decimal-year wart in the `astsa` datasets acknowledged, not
      hidden
- [ ] Ends on chart 9 opening chapter 3, no recap
- [ ] 11–13 pages
- [ ] British-Indian spelling

---

## 9. What to do

1. **Verify the package's actual missing-data behaviour before writing
   beat 4's chart 7.** I have not checked what `tsvalues` does with
   `missing` or `NaN`, and the reading must describe what the code
   really does, not what would be sensible.
2. Re-run the daylight-saving count to confirm 1090 / 3 / 3 against the
   shipped `vic_elec.csv`.
3. Render every chart; use the honest CI note from
   `manual/10-plotting.md` if the local environment cannot.
4. Check against section 8.

One thing worth deciding while writing this chapter. The decimal-year
time in the `astsa` datasets is a genuine inconsistency in the package,
not merely a teaching point. The dataset handoff flagged converting
them to proper `Date`s and left the decision open. **Writing this
chapter is the moment that decision becomes unavoidable** — either
convert them and simplify the chapter, or keep them and document the
wart honestly. Both are defensible; drifting is not.
