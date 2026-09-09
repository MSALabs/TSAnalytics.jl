# Handoff: Chapter 15 — STL

The reference chapter. Written first so that the remaining forty have
something concrete to match rather than a style guide to interpret.
Target: 6–8 pages, `docs/src/introduction/15-stl.md`.

Read `chapter-writing-guide.md` alongside this. That document sets the
standard; this one supplies the verified material for one chapter.

---

## 1. A correction that must be made before drafting

Throughout the development of this project I have repeatedly cited the
following as an established finding:

> *"Python fixed a median bug in the original STL Fortran that R still
> has; the numbers genuinely differ near outliers when `robust=TRUE`."*

**I tested this properly this session, and it does not hold up.** The
verification is below. Do not write the chapter around the median-bug
story. What is actually true is more interesting, and is documented in
section 2.

This matters beyond one chapter. The claim had been carried forward
unverified across many sessions and was about to become the centrepiece
of the book's reference chapter. The standard the package was built to —
run it, do not recall it — is exactly what caught it.

---

## 2. What is actually true, with the numbers

**Setup.** A monthly series, `n = 120`, linear trend `50 + 0.4t`,
seasonal amplitude 12, Gaussian noise `σ = 1.5`, and one large outlier
of `+45` at index 60 (0-based). Seed 11. R and Python read the identical
CSV. Parameters matched explicitly where the two packages name them
differently: `s.window = seasonal = 7`, `s.degree = seasonal_deg = 0`,
`t.degree = trend_deg = 1`.

**Result 1 — they disagree, and not by floating-point noise:**

| Configuration | max abs trend difference | max abs seasonal difference |
|---|---|---|
| `robust = TRUE`, package defaults | **0.0486** | **0.0339** |
| `robust = FALSE`, package defaults | **0.133** | **0.110** |

**Result 2 — the disagreement is *larger* without robustness.** This is
what kills the median-bug story. If the cause were a robustness-weight
bug, `robust = FALSE` should agree to machine precision. It does not;
it is roughly three times worse.

**Result 3 — the main culprit is a parameter most users have never
heard of.** R's `stl` computes loess at every *j*th point and
interpolates between, for speed. The step size is `jump`, and R's
defaults are derived from the window widths:

```
s.window = 7, period = 12  ->  t.window = 23
                               s.jump = ceiling(7/10)  = 1
                               t.jump = ceiling(23/10) = 3
                               l.jump = ceiling(13/10) = 2
```

Python's `STL` defaults every jump to **1** — no interpolation, exact
loess at every point. So R ships a coarser approximation by default and
Python ships the exact computation, and neither documents this as a
difference from the other.

Setting all jumps to 1 on both sides:

| Configuration | max abs trend difference | max abs seasonal difference |
|---|---|---|
| `robust = TRUE`, jumps matched | 0.0019 | 0.0080 |
| `robust = FALSE`, jumps matched | 0.0535 | 0.124 |

The robust case improves by a factor of about 25. So the jump defaults
explain most of the robust-mode gap.

**Result 4 — something remains, and it is honest to say we do not know
what.** With jumps matched *and* iterations forced far past convergence
(`inner = 50`, `outer = 10`):

```
max abs trend difference    0.0254
max abs seasonal difference 0.0232
```

So it is not a convergence artefact either. A residual difference of
around 0.02–0.12 persists between two respected implementations of the
same published algorithm, and this handoff does not claim to know its
origin. **Write that honestly.** "We tracked it this far and no
further" is a better sentence than a confident wrong explanation, and
it is the only sentence the evidence supports.

### Where TSAnalytics.jl sits

Checked directly. `stl_decompose` has **no jump parameters at all** —
it computes loess at every point, which is Python's behaviour, not R's.
It also defaults `seasonal_degree = 1`, again matching Python
(`seasonal_deg = 1`) rather than R (`s.degree = 0`).

**The chapter must state both of these plainly.** A reader migrating
from R will get different numbers from this package and deserves to
know why before they discover it themselves.

---

## 3. Chapter structure — the six beats, filled in

### Beat 1 — The question (about half a page)

Open on a series with a visible problem. Recommended: Australian
quarterly production or Victorian electricity demand from the bundle,
or better, an Indian series if one is to hand. The question is concrete:
*there is a spike in this series; what does it do to the seasonal
pattern we estimate, and should it?*

Do not open with a definition of loess. Do not open with a formula.

### Beat 2 — The obvious attempt (about one page)

Classical decomposition, from chapter 14, applied to the same series.
Show it working reasonably. Then show what the spike does to it — the
seasonal index for that month is contaminated for every year, because
classical decomposition averages across cycles with equal weight.

This is the honest motivation for STL, and it is much more convincing
than asserting that STL is more flexible.

### Beat 3 — The idea (about one and a half pages)

Loess, briefly and concretely — a local regression, a window, a degree.
Then the inner loop: detrend, smooth each cycle-subseries, low-pass
filter, deseasonalise, smooth for trend, repeat. Then the outer loop:
compute robustness weights from the residuals, and downweight the
points the model cannot explain.

Notation arrives here, after the reader wants it. Cite Cleveland et al.
(1990) as the source.

### Beat 4 — Making it work (about one and a half pages)

Real code on bundled data:

```julia
using TSAnalytics
y = dataset("aus_production")     # or vic_elec, or the Indian series
d = stl_decompose(y, 12; robust = true)
plot(d)
```

Show the seasonal, trend and remainder. Show the robustness weights —
`STLDecomposition` exposes a `weights` field, which is genuinely
instructive here: the reader can see the outlier being downweighted
towards zero. Most treatments never show this, and it makes the outer
loop concrete rather than abstract.

Cover the parameters that matter: `seasonal_window`, `trend_window`,
`robust`. Explain what a `seasonal_window` of 7 versus 21 actually does
to the estimate — a short window lets the seasonal pattern evolve, a
long one holds it nearly fixed.

### Beat 5 — The complication (about one and a half pages)

This is the `!!! disagreement` box, and it is the chapter's centrepiece.
Use the verified material from section 2, told as the investigation it
actually was:

1. Same data, same nominal parameters, two respected implementations.
   The numbers differ in the second decimal place.
2. The natural first hypothesis is that robustness is at fault. Test
   it. The disagreement is *worse* without robustness — hypothesis dead.
3. Read the defaults properly. R interpolates loess between every third
   point for the trend; Python computes it at every point. Neither
   documents this as a divergence from the other.
4. Match the jumps. The robust-mode gap falls by a factor of 25.
5. Something remains, past convergence. We do not know what.

Then state where TSAnalytics.jl sits, and why: exact loess, no jump
approximation, matching Python.

The moral is worth stating explicitly, because it generalises well
beyond STL: *two implementations of one published algorithm are not the
same function, and the difference usually lives in defaults nobody
reads.*

### Beat 6 — Where this leaves you (about half a page)

STL handles one seasonal period. Electricity demand has three — daily,
weekly, annual. That gap opens chapter 16, MSTL.

Also name what STL genuinely cannot do: no forecasting on its own, no
calendar effects, no trading-day adjustment. Those wait for Parts VII
and VIII.

---

## 4. Examples

**Canonical, and available in the bundle:**

- `aus_production` or `aus_retail` — quarterly and monthly seasonality,
  the fpp3 workhorses for exactly this chapter
- `vic_elec` — mention only, as the motivation for chapter 16
- `co2` or `cmort` from `astsa` — Shumway & Stoffer's own STL territory

**Indian example.** Include one if the data supports it. IIP is the
natural candidate. If no suitable Indian series is bundled yet, say so
in the drafting notes rather than inventing one — and flag it for the
dataset work rather than quietly skipping the requirement.

**Cannot be reproduced:** none of significance for this chapter. STL's
canonical examples are almost all in `astsa` and `tsibbledata`, both
GPL-3. This chapter is fortunate; later ones are not.

---

## 5. The three recurring boxes

**`!!! disagreement`** — the whole of section 2, as beat 5. This is the
strongest such box available anywhere in the book; it earns its length.

**`!!! julia`** — a natural fit here: `stl_decompose` takes
`parallel::Bool = true`. The cycle-subseries smoothing is embarrassingly
parallel across the `period` subseries. One short paragraph on why this
particular loop parallelises cleanly and the trend loop does not.

**`!!! india`** — if the IIP example lands, this is where the festival
timing problem gets its first mention, forward-referencing chapter 36.
Keep it brief; the full treatment belongs in Part VII.

---

## 6. Voice and mechanics

- **British-Indian spelling**, per the writing guide: *behaviour*,
  *modelling*, *analyse*, *centre*, *summarise*. Note that the existing
  source is currently inconsistent on this; the chapter follows the
  guide, and the source cleanup is separate work.
- **Dates** as 14 March 2024.
- Measured, direct prose. No exclamation marks, no rhetorical questions
  beyond the chapter opening.
- Cite Cleveland, Cleveland, McRae and Terpenning (1990) for STL
  itself, and cross-reference `C-further-reading.md`.
- Cross-reference the Manual (`manual/03-decomposition.md`) for full
  signatures rather than reproducing them in the chapter.

---

## 7. Checklist before this chapter is done

- [ ] Opens with a question about visible data, not a definition
- [ ] Classical decomposition shown failing first (beat 2)
- [ ] Every code block has been run; output pasted is real
- [ ] Robustness weights plotted or shown, not just described
- [ ] `disagreement` box uses the verified numbers from section 2, and
      states the unexplained residual honestly
- [ ] TSAnalytics.jl's jump and `seasonal_degree` conventions disclosed
- [ ] `julia` box on the parallel cycle-subseries loop
- [ ] Indian example present, or its absence flagged for dataset work
- [ ] Ends by naming what STL cannot do, and opening chapter 16
- [ ] 6–8 pages
- [ ] British-Indian spelling throughout

---

## 8. What to do

1. Draft `docs/src/introduction/15-stl.md` to the structure in
   section 3.
2. Re-run every code block before pasting output. The verification in
   section 2 was done in R and Python; the *Julia* numbers for beat 4
   have **not** been generated yet and must be produced fresh, not
   estimated.
3. Do not repeat the median-bug claim anywhere, in this chapter or
   elsewhere. If it appears in other drafts or notes, correct it.
4. Review against section 7, then against
   `chapter-writing-guide.md`'s own checklist.
5. Treat the finished chapter as the reference for chapters 1–14 and
   16–41. Chapter 1 comes next, now that there is something to match.
