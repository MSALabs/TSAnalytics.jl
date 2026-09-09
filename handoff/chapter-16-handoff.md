# Handoff: Chapter 16 — Multiple Seasonality

`docs/src/introduction/16-multiple-seasonality.md`. Target 11–12 pages.

The last chapter of Part III. Chapter 15 ended by failing on a series
with three seasonal periods at once. This chapter handles it — and
carries what may be the most alarming disagreement box in the book,
because it is a live bug rather than a difference of convention.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. A [U] claim, re-verified — and it is worse than recorded

`all-chapters-handoff.md` tagged this **[U]** and warned it may have
been fixed since. It has not been. Verified two independent ways this
session against current `statsmodels`.

**From the source of `MSTL`:**

```python
if self.lmbda == "auto":
    y, lmbda = boxcox(self._y, lmbda=None)
    self.est_lmbda = lmbda
elif self.lmbda:                       # <-- zero is falsy
    y = boxcox(self._y, lmbda=self.lmbda)
```

`elif self.lmbda:` is a truthiness test. When `lmbda = 0` it evaluates
false, the branch is skipped, and **no transform is applied at all**.

In the Box-Cox family, `λ = 0` means *take logs* — the single most
commonly wanted transform in the whole book, and the one chapter 7
spent nine pages on.

**Confirmed behaviourally**, on a multiplicative series:

```
trend[0] with lmbda = 0      :  11.285054
trend[0] with lmbda = None   :  11.285054     <- identical
trend[0] with lmbda = 1e-8   :   2.391686     <- completely different

lmbda=0 identical to lmbda=None (no transform)?   True
lmbda=0 identical to lmbda=1e-8 (log applied)?    False
```

So asking for a log transform gets you no transform. And `1e-8`, which
is mathematically indistinguishable from zero, works correctly.

**This is the most dangerous class of bug in the book.** It fails
silently, it returns plausible numbers, and the adjacent value behaves
correctly — so anyone testing with `λ = 0.5` or `λ = 1e-6` would never
find it. The chapter should say all of that plainly, and should also
note that this package checks `!== nothing` rather than truthiness,
which is why it does not have the bug.

---

## 2. The chapter, beat by beat

### Beat 1 — Three rhythms at once (about 1.5 pages)

**Chart 1 — `vic_elec`, three panels at three zoom levels: two days,
two months, two years.**

*Reading:* at two days a clear daily cycle with a morning ramp and an
evening peak. At two months that daily cycle becomes a dense band and a
weekly rhythm appears — weekends sit lower. At two years both vanish
into a thick ribbon and an annual cycle emerges, driven by heating and
cooling.

Three periods, all real, all simultaneous — 48 half-hours, 336
half-hours, and about 17,520. This is the same "frequency is a choice"
point from chapter 2, now with consequences.

**Chart 2 — chapter 15's single-period STL fit on this series,
repeated.**

*Reading:* whichever period is chosen, the other two land in the
remainder. Show the remainder's ACF: it has obvious structure at the
periods that were not modelled. Part II's tests flag it immediately.

### Beat 2 — One at a time (about 2 pages)

The obvious approach, and it is essentially the right one.

**Chart 3 — the iterative fit: extract the daily component, then the
weekly from what remains, then the annual.**

*Reading:* MSTL runs STL repeatedly, once per period, subtracting each
component before fitting the next, and cycles through the whole
sequence a few times so that early estimates get refined once later
ones exist. There is no new mathematics — it is chapter 15's algorithm
in a loop.

**Chart 4 — the full MSTL decomposition: observed, three seasonal
components, trend, remainder.**

*Reading:* six panels where chapter 15 had four. Each seasonal
component is separately interpretable — the daily one is the working
day, the weekly one is the weekend effect, the annual one is climate.
That separability is the payoff, because they are driven by different
things and a business would act on them differently.

**Chart 5 — the remainder from chart 4 with its ACF, beside chart 2's.**

*Reading:* the structure at the unmodelled periods is gone. This is the
direct comparison that shows the method worked, and it uses Part II's
diagnostics rather than an assertion.

### Beat 3 — Order matters, a little (about 1.5 pages)

**Chart 6 — the same series decomposed with the periods supplied in
two different orders.**

*Reading:* the results are close but not identical. The iteration
reduces the dependence on ordering without eliminating it. The
convention is shortest period first, which is what most implementations
do and what this package should be checked against.

Note that this is a milder version of chapter 13's point — the
components are constructed, and a construction choice leaves a
fingerprint.

### Beat 4 — The trap (about 2.5 pages)

**Chart 7 — the falsy-zero bug, drawn.**

Take a multiplicative series. Decompose it three ways — `λ = None`,
`λ = 0`, `λ = 1e-8` — and plot the three trend estimates on one set of
axes.

*Reading:* two of the three lines lie exactly on top of each other, and
they are `λ = None` and `λ = 0`. The line that is different is
`λ = 1e-8`. Use the verified numbers: `11.285054`, `11.285054`,
`2.391686`.

Then the `disagreement` box, using section 1's material in full — the
source line, the behavioural confirmation, and why this failure mode is
worse than a wrong answer. A wrong answer can be caught. A silently
skipped step returns a plausible answer to a question you did not ask.

State that this package checks `!== nothing`, and why that is the right
test in a language where `0` is not falsy but a `nothing` sentinel is
the idiomatic way to mean "not supplied".

**`julia` box here.** Julia does not have truthiness — `if 0` is an
error, not a silent false. A whole category of bug is unavailable by
construction. Half a page, no more, and resist making it a language
argument; the point is that the type system caught something a human
review did not.

### Beat 5 — What multiple seasonality does not fix (about 1.5 pages)

**Chart 8 — a series where the daily pattern differs between weekdays
and weekends.**

*Reading:* MSTL fits one daily component and one weekly component and
adds them. It cannot represent a daily shape that *changes* on
weekends, because that is an interaction between two periods rather
than a sum of two effects. The remainder will show it.

This is a real limitation and it is the honest end to Part III. Every
method in these four chapters decomposes by addition, and additive
decomposition cannot represent interaction.

### Beat 6 — Where this leaves you (half a page)

Part III is finished. A series can be split into components you can
see, name and act on, with a method matched to how complicated the
seasonality is.

Everything in this Part has described what a series *has done*. None of
it forecasts. Decomposition is not a model — there is no mechanism, no
parameters to interpret, nothing to extrapolate honestly.

Part IV builds models that are genuinely fitted, and it starts by
asking what the correlograms of chapter 5 were trying to tell us.

No recap. One sentence marking the end of a Part.

---

## 3. The `india` box

Placed in beat 1, after chart 1.

Indian electricity demand has the same three periods as Victorian
demand, plus a fourth that Australia does not have — festival days
produce a distinct load shape, and the dates move each year against the
Gregorian calendar.

MSTL can handle three fixed periods. It cannot handle a fourth whose
timing shifts, because it decomposes by fixed period and a moving
festival has no fixed period. That is a real limit of everything in
Part III, and the remedy is a regressor built from actual dates rather
than a decomposition.

Chapter 36.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Bandara, Hyndman & Bergmeir (MSTL paper)** | The algorithm; cite it properly | Benchmarking detail |
| **fpp3 §12.1** | Complex seasonality as a named, practical problem with real examples, and `vic_elec` itself | The `fable` syntax |
| **Shumway & Stoffer** | Little directly, but their spectral treatment (chapter 6) is how one *detects* multiple periods before deciding to model them — worth a sentence and a callback |  — |
| **Ladiray & Quenneville** | That official statistics largely does not do this — X-13 handles one seasonal period — which is worth knowing when comparing methods | X-11 mechanics |
| **Montgomery, Jennings & Kulahci** | Multiple operational rhythms as the normal state of industrial data | — |
| **Tsay** | Intraday financial seasonality — the U-shaped trading-day volume pattern is a genuine second period | Volatility |

**On examples:** `vic_elec` (52,608 observations, three genuine
periods) is the canonical case and is bundled. `nyc_bikes` also has
daily and weekly structure if a second example helps.

---

## 5. Voice

**The bug is the chapter's centre and should be treated seriously
rather than gleefully.** `statsmodels` is an excellent library
maintained by careful people, and this is a two-character mistake of a
kind every codebase contains. The lesson is about silent failure modes,
not about anyone's competence — and writing it any other way would be
both unkind and less useful.

**Do not let the `julia` box become advocacy.** One paragraph on
truthiness, then move on. The reader is already using Julia.

**Beat 5 must not be an afterthought.** Ending Part III on a genuine
limitation is better than ending it on a triumph, and the interaction
problem is real.

Avoid, beyond earlier lists:

- Calling MSTL "just STL in a loop" dismissively. It is STL in a loop,
  and that is a virtue.
- Reproducing the whole `statsmodels` function. The four relevant lines
  are enough.

---

## 6. Checklist

- [ ] Eight charts through `@example ch16`
- [ ] Chart 1 shows three zoom levels revealing three periods
- [ ] Chapter 15's single-period failure shown before MSTL is
      introduced
- [ ] Remainder ACFs compared before and after — the diagnostic proof
- [ ] Ordering effect shown
- [ ] **The falsy-zero bug drawn**, with all three trend lines and the
      verified numbers
- [ ] `disagreement` box includes both the source line and the
      behavioural confirmation
- [ ] Package's `!== nothing` check stated
- [ ] `julia` box on truthiness, half a page, not advocacy
- [ ] Interaction limitation shown, not just described
- [ ] `india` box on festival load shape
- [ ] Ends by opening Part IV, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 7. What to do

1. **Re-run the falsy-zero demonstration during drafting.** It is a
   live bug in a maintained library; confirm it is still live at the
   version you build against, and state that version in the text. If it
   has been fixed by then, the chapter should say so and treat it
   historically — which is a good outcome and a better story than
   pretending otherwise.
2. Verify what this package's MSTL actually does with `lambda = 0` —
   the `!== nothing` claim is recorded from earlier work and should be
   confirmed against current source.
3. Check the ordering convention this package uses for periods.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
