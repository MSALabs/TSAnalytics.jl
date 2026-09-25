# Handoff: Chapter 36 — Calendar Effects

`docs/src/introduction/36-calendar-effects.md`. Target 12–13 pages.

The chapter the book's Indian thread has been pointing at since chapter
1. Every previous `india` box has forward-referenced this one, and it
carries the project's signature verified result.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

### The Diwali regressor genuinely changes the fit

Fed into a RegARIMA model as a user-defined holiday regressor, against
the same series fitted without it:

```
October seasonal factor, no Diwali regressor:  0.915164290487692
October seasonal factor, with the regressor:   0.753973303751993
```

The seasonal factor moved by a sixth. The regressor was absorbed into
the fit and materially changed the decomposition — it was not silently
ignored, which is the failure mode worth ruling out.

### Two practical requirements, both found by hitting them

**User-defined regressors must cover the forecast horizon**, not just
the historical sample. The binary errors clearly if they do not:

```
ERROR: forecasts end date, 1961.Dec, must end on or before
       user-defined regression variables end date, 1960.Dec.
```

**A RegARIMA model combined with multiplicative adjustment needs an
explicit log transform declared.** Otherwise:

```
ERROR: Multiplicative or log additive seasonal adjustment cannot be
       performed when preadjustment factors are derived from a regARIMA
       model for data which have not been log transformed
```

Both are the kind of thing that costs an afternoon the first time.

### The weekend is hardcoded in the obvious library

From `BusinessDays.jl`'s source:

```julia
@inline isweekend(dt::Dates.Date)::Bool = signbit(5 - Dates.dayofweek(dt))
```

It takes a `Date` and **no calendar argument**, so every subtype
inherits a Saturday–Sunday weekend. Markets with a Friday–Saturday
weekend cannot be represented by any calendar built on it.

### Aggregator sources genuinely disagree

Two separate cases found while assembling an Indian holiday table:
sources disagreed on a 2026 Diwali date by more than two weeks, and
three aggregators contradicted a Guru Nanak Jayanti date supplied from
elsewhere.

---

## 2. The chapter, beat by beat

### Beat 1 — October is not October (about 1.5 pages)

**Chart 1 — an Indian monthly series with October and November
highlighted across several years, and the Diwali date marked on each.**

*Reading:* the festival lands in October some years and November in
others, and the production or retail surge moves with it. A model with
a twelve-month seasonal structure sees a pattern that is not repeating
at a fixed lag, and it has no way to know why.

Every `india` box in this book so far has pointed here. Chapter 5's ACF
blurred around lag 12. Chapter 16's MSTL could not handle a period that
moves. Chapter 20's SARIMA fitted a compromise wrong in both months.
This is the chapter that fixes it, and the fix is not a better seasonal
model.

### Beat 2 — Effects a calendar can cause (about 2 pages)

**Chart 2 — the number of each weekday in each month across two years.**

*Reading:* a month with five Saturdays behaves differently from one
with four, for anything driven by trading days or working days. The
count varies by month and by year in a way that repeats only every
twenty-eight years.

This is a *deterministic* effect — the calendar is known in advance,
forever — and it is the strongest argument for a regressor rather than
a seasonal component. There is no uncertainty to model. You know how
many Tuesdays next March has.

**Chart 3 — a series with and without a trading-day adjustment.**

*Reading:* the adjusted series is visibly smoother at the monthly
frequency. Some of what looked like noise was calendar arithmetic.

**Chart 4 — Easter's date across twenty years.**

*Reading:* Easter moves between March and April, so a retail series
influenced by it has a March effect in some years and an April effect
in others. Statistical agencies have handled this for decades with a
dedicated regressor, and the machinery generalises to any moving
holiday — including Diwali.

### Beat 3 — Building the regressor (about 2.5 pages)

**Chart 5 — a Diwali proximity regressor, drawn as a monthly series.**

*Reading:* the construction is simple once the dates are known. Assign
weight to the month containing the festival, or spread it across the
days before and after if the effect is anticipatory. Show both and note
that the window length is a modelling choice with no canonical answer.

**Chart 6 — the same series decomposed with and without the regressor,
with the October seasonal factors marked.**

*Reading:* use the verified numbers — `0.915164` against `0.753974`.
The regressor absorbed the festival effect out of the seasonal
component, which is exactly what it is for. Without it, the seasonal
factor was carrying a Diwali effect that does not belong to October in
general.

**Then the two practical requirements**, from section 1. The regressor
must extend past the end of the data to cover the forecast horizon, and
a multiplicative adjustment needs an explicit log transform. Quote both
error messages — they are clear, they are the kind of thing that costs
an afternoon, and reproducing them saves the reader that afternoon.

### Beat 4 — Where the dates come from (about 2.5 pages)

The `disagreement` box, and it is about data rather than algorithms.

**Chart 7 — a table or figure of the same holiday's date across
several sources.**

*Reading:* use the verified cases. Sources disagreed on a 2026 Diwali
date by more than two weeks. Three aggregators contradicted a Guru
Nanak Jayanti date from a fourth source.

The reason is structural rather than careless. Fixed-date holidays are
computable — Republic Day is always 26 January. Easter is computable
from a known algorithm. **Diwali, Holi, Eid and most of India's actual
trading calendar are not computable at all** — they follow lunisolar or
lunar calendars, and the observed date is set by announcement, not by
arithmetic. Aggregators reconstruct them, and reconstructions differ.

The only reliable source is the exchange's or government's own
published circular. State that plainly — it is the practical lesson and
it generalises to every market with non-Gregorian holidays.

Note what a maintained table implies: it needs updating, it cannot be
extrapolated forward, and a calendar that silently returns only its
fixed holidays for an unlisted year is worse than one that raises an
error.

**Chart 8 — the weekend problem.**

*Reading:* quote the `isweekend` source line. It takes a date and
nothing else, so Saturday–Sunday is baked in for every calendar built
on that library. A Friday–Saturday market — Saudi Arabia, and
historically the UAE — cannot be represented at all.

This package therefore keeps its calendar type outside that hierarchy,
with an explicit weekend set. That is a design decision forced by a
one-line function, and it is worth showing because it is a good example
of how a small upstream choice propagates.

### Beat 5 — Whether it was worth it (about 2 pages)

**Chart 9 — the model with and without calendar regressors, compared
out of sample.**

*Reading:* use chapter 23's cross-validation. Report whatever comes
out, honestly. A calendar regressor that does not improve out-of-sample
accuracy has not earned its parameter, and the fact that the effect is
real does not guarantee that modelling it helps.

**Chart 10 — the residual ACF at lag 12, with and without.**

*Reading:* chapter 10's diagnostic. If the festival effect has been
absorbed, the smear around the seasonal lag should be reduced. That is
a more direct test than an accuracy comparison and it answers a
different question — whether the model is better specified, rather than
whether it forecasts better.

Both are worth reporting and they can disagree.

### Beat 6 — Where this leaves you (half a page)

Deterministic outside information — calendars, holidays, known events —
can be brought into a model as a regressor, and the effect is
measurable.

One assumption has survived every model in Part VII. The error term's
variance has been constant throughout, and chapter 24 established that
for many series it is not.

Chapter 37.

No recap.

---

## 3. The `india` box

This chapter *is* the India box, so a separate one would be redundant.
Instead, use the space for the reverse: a **note on generality**.

Placed in beat 4, after chart 8.

Nothing in this chapter is specific to India. Chinese New Year moves
against the Gregorian calendar for the same reason Diwali does, and
affects East Asian industrial series in the same way. Ramadan moves
through the entire year and affects retail and working patterns across
much of the world. Thai and Vietnamese calendars have the same
structure.

The techniques are general; the tables are local. Any market whose
calendar is not the Gregorian one needs its own maintained list, and
the reason there is more published work on Easter than on Diwali is
about where the literature was written, not about which effect is
larger.

One paragraph — and it is worth stating, because a reader outside India
should not conclude this chapter is a regional curiosity.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Ladiray & Quenneville** | Trading-day and moving-holiday regressors as official-statistics practice, worked out in detail. This is their home ground | X-11 mechanics |
| **X-13ARIMA-SEATS Reference Manual** | The `regression` specification, the built-in `td` and `easter` variables, and the user-defined regressor mechanism | The full spec grammar |
| **fpp3 §7.4** | Calendar effects as ordinary regressors in a forecasting model, and Fourier terms as an alternative | The `fable` syntax |
| **Montgomery, Jennings & Kulahci** | Trading-day effects in industrial and retail data, from the operations side | Control charts |
| **Shumway & Stoffer** | Regression with deterministic components | — |
| **Hamilton** | Little; calendar effects are an applied concern | — |

**On examples:** the Diwali verification used the airline series as a
carrier. An actual Indian series would be far better — **check whether
one is bundled**, and if not, say so rather than presenting a synthetic
example as real.

---

## 5. Voice

**This chapter carries the book's distinctive claim and should be
written with care rather than enthusiasm.** The Diwali result is real
and verified; the temptation is to oversell it. The numbers do the work.

**The generality note in section 3 matters.** Without it, a reader in
Europe or the US may read this chapter as a regional aside rather than
as a general technique demonstrated on a case the literature has
neglected.

**Be honest in beat 5.** If the calendar regressor does not improve
out-of-sample accuracy on the series chosen, report that. A real effect
that does not improve forecasts is a genuine and instructive outcome.

Avoid, beyond earlier lists:

- Explaining the Hindu lunisolar calendar. Two sentences on why the
  date is not computable is enough.
- Presenting the maintained table as a shortcoming. It is what every
  implementation does, including QuantLib's.

---

## 6. Checklist

- [ ] Ten charts through `@example ch36`
- [ ] Opens by collecting the forward references from chapters 5, 16
      and 20
- [ ] Weekday counts shown varying by month and year
- [ ] Easter used as the established precedent before Diwali
- [ ] **The verified seasonal-factor shift reported** — 0.915164 to
      0.753974
- [ ] **Both error messages quoted verbatim**
- [ ] `disagreement` box on source discrepancies, with both verified
      cases
- [ ] The `isweekend` source line quoted, and the design consequence
      explained
- [ ] Out-of-sample comparison run and reported honestly
- [ ] Residual ACF at the seasonal lag compared
- [ ] **Generality note** so the chapter does not read as regional
- [ ] Ends by opening chapter 37, no recap
- [ ] 12–13 pages, British-Indian spelling

---

## 7. What to do

1. **Check whether an Indian series is bundled.** The Diwali
   verification used the airline series as a carrier, which was fine for
   proving the mechanism and is weak for a chapter. If nothing suitable
   exists, say so and flag it for the dataset work.
2. Re-run the with-and-without comparison and report the actual
   seasonal factors from that run.
3. Reproduce both error messages rather than copying them from here.
4. Run beat 5's out-of-sample comparison once and use the result,
   whatever it is.
5. Render every chart; honest CI note if the environment cannot.
6. Check against section 6.
