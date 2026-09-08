# Handoff: Restructuring "Introduction to Time Series Analysis"

For a fresh Claude Code session picking this up with no prior context.
**This is a structural migration, not content writing.** Chapters stay
stubs at the end of this task, exactly as they are now — only their
number, name, grouping and navigation change.

## Why now, and why this is nearly free

Verified directly against the repository this session: all 20 existing
Introduction chapters are **3-line stubs**. Exactly one file has real
content — `B-further-reading.md`, 39 lines, the bibliography that the
original skeleton handoff explicitly allowed as an exception.

**Nothing is lost by restructuring today.** Every week that content
gets written into the current structure raises the cost of this change,
and the current structure has a real problem (section 2). Do this
before drafting begins.

## Where this fits

- **Depends on:** the docs skeleton (already implemented — all 47 files
  and the `make.jl` nav are in place).
- **Supersedes:** the Introduction section of
  `docs-restructure-skeleton-handoff.md`. The Getting Started, Manual
  and API Reference sections of that handoff are unaffected — **do not
  touch them.**
- **Source of the new structure:** the book table of contents
  (`book-table-of-contents.md`), which was developed after the skeleton
  and reasons about sequencing far more carefully.

---

## 1. The design decision to make before starting

The original 20-chapter Introduction and the ~300-page book being
planned are, on inspection, the same document at two different lengths.
Maintaining both separately means writing the same explanations twice
and letting them drift.

| Option | What it means | Assessment |
|---|---|---|
| **A. They are the same thing** | The Introduction *is* the book, published free online; a print edition is generated from the same source | **Recommended.** Exactly the model fpp3 uses — free online textbook, paid print version, one source. Content written once serves both |
| **B. Introduction is a condensed book** | Short conceptual version online, long version in print | Two documents to keep in sync; they will drift |
| **C. Separate documents** | Unrelated | Duplicated effort for no benefit |

**This handoff assumes Option A.** That is why the new structure below
mirrors the book's parts exactly rather than inventing a third
organisation. If Option A is rejected, stop and revise this handoff —
the chapter list would change substantially.

Under Option A the Introduction carries Parts I–IX of the book. The
book's Chapter 1 (Getting Started) is **not** included, because the
docs already have a five-chapter Getting Started section that covers it.
Likewise the book's "Coming from R/Python" appendices live in the
Manual, not here.

---

## 2. What is actually wrong with the current structure

Not cosmetic. Four real problems:

**Differencing and unit-root testing are fused into one chapter** (old
`03-differencing-and-unit-roots.md`). These are different kinds of
thing. Differencing is a computational primitive — you do it to a
series. Unit-root testing is a diagnostic — you ask a question of a
series. Fusing them forces the chapter to teach a test before the
reader has met the concept of a hypothesis test at all.

**Stationarity sits at chapter 2, before the reader can compute
anything.** It is presented as an axiom rather than as the question it
actually is. Moving it into the diagnostics part lets it arrive as
*"here is a property we need; here is how to check for it"*.

**Whole topics are missing.** MSTL, the diagnostic panel, time series
cross-validation, asymmetric GARCH, volatility forecasting, realized
measures, Kalman smoothing, time-varying systems, calendar effects, and
seasonal adjustment have no chapter at all. Several of these are
implemented, verified, and among the package's more distinctive
capabilities.

**No structural home for the material that makes this book worth
writing.** The ~35 documented cases where R, Python and SAS disagree;
the Indian calendar work; the Julia implementation notes. In the
current structure these have nowhere to go.

---

## 3. The new structure — 41 chapters in 9 parts, 3 appendices

Navigation groups by part; **file numbering stays sequential 01–41
across the whole Introduction** so that file order matches reading
order and no renumbering is needed when a part gains a chapter.

### Part I — The Series and the Machine
```
01-why-model-a-time-series.md
02-what-a-time-series-is.md
03-differencing-and-integration.md
04-filters-and-moving-averages.md
05-autocorrelation.md
06-the-frequency-domain.md
07-transformations.md
```

### Part II — Interrogating a Series
```
08-stationarity.md
09-testing-for-unit-roots.md
10-testing-the-residuals.md
11-testing-distribution-and-variance.md
12-the-diagnostic-panel.md
```

### Part III — Pulling a Series Apart
```
13-components.md
14-classical-decomposition.md
15-stl.md
16-multiple-seasonality.md
```

### Part IV — Models That Remember
```
17-ar-and-ma-processes.md
18-fitting-arma.md
19-arima.md
20-seasonal-arima.md
21-choosing-an-order.md
22-forecasting.md
23-evaluating-honestly.md
```

### Part V — Models That Get Turbulent
```
24-why-variance-changes.md
25-arch-and-garch.md
26-asymmetry.md
27-forecasting-volatility.md
28-realized-measures.md
```

### Part VI — The Machinery Underneath
```
29-the-state-space-form.md
30-the-kalman-filter.md
31-smoothing.md
32-time-varying-systems.md
33-diffuse-initialisation.md
```

### Part VII — Bringing in Outside Information
```
34-regression-with-arima-errors.md
35-coefficients-that-drift.md
36-calendar-effects.md
37-autoregressive-errors-with-changing-variance.md
```

### Part VIII — What Statistical Agencies Do
```
38-official-seasonal-adjustment.md
39-x13-from-julia.md
40-adjusting-an-indian-series.md
```

### Part IX — The Frontier
```
41-the-frontier.md
```

### Appendices
```
A-checklist.md
B-verification.md
C-further-reading.md
```

**Part VI's placement is deliberate and is the structure's one genuinely
unusual choice.** Durbin & Koopman open with state space; Hamilton
treats it as a late separate topic. Here the reader fits ARMA models
throughout Part IV, and Part VI then reveals that every one of those
fits was already running a Kalman filter. This is not a rhetorical
device — it reflects the package's real architecture, where
`statespace/gaussianssm.jl` loads before `arma.jl` because ARMA
estimation is built on it. Chapter 30 should make that explicit.

---

## 4. Migration map

All chapter stubs are identical 3-line files, so `git rm` + create is
simpler than `git mv` for them. **One exception, flagged hard:**

> `B-further-reading.md` contains the only real content in the section
> (39 lines of bibliography). It must be **`git mv`'d to
> `C-further-reading.md`**, not deleted and recreated. Do not overwrite it.

| Existing file | Becomes | Note |
|---|---|---|
| `01-why-model-time-series.md` | `01-why-model-a-time-series.md` | rename only |
| `02-stationarity.md` | `08-stationarity.md` | **moves Part I → Part II** |
| `03-differencing-and-unit-roots.md` | `03-differencing-and-integration.md` **and** `09-testing-for-unit-roots.md` | **split** |
| `04-acf-and-pacf.md` | `05-autocorrelation.md` | renumber + rename |
| `05-filters-and-moving-averages.md` | `04-filters-and-moving-averages.md` | renumber |
| `06-spectral-analysis.md` | `06-the-frequency-domain.md` | rename |
| `07-transformations.md` | `07-transformations.md` | unchanged |
| `08-classical-decomposition.md` | `14-classical-decomposition.md` | renumber |
| `09-stl-and-robust-decomposition.md` | `15-stl.md` | renumber + rename |
| `10-testing-the-residuals.md` | `10-testing-the-residuals.md` | unchanged |
| `11-testing-for-structure.md` | `11-testing-distribution-and-variance.md` | rename |
| `12-the-arma-model.md` | `17-ar-and-ma-processes.md` | renumber + rename |
| `13-from-arma-to-arima.md` | `19-arima.md` | renumber + rename |
| `14-seasonal-arima.md` | `20-seasonal-arima.md` | renumber |
| `15-choosing-an-order.md` | `21-choosing-an-order.md` | renumber |
| `16-volatility-and-garch.md` | `25-arch-and-garch.md` | renumber + rename |
| `17-state-space-and-kalman.md` | `29-the-state-space-form.md` **and** `30-the-kalman-filter.md` | **split** |
| `18-diffuse-initialization.md` | `33-diffuse-initialisation.md` | renumber; note `s` spelling for consistency |
| `19-regression-with-arima-errors.md` | `34-regression-with-arima-errors.md` | renumber |
| `20-forecasting.md` | `22-forecasting.md` | renumber |
| `A-checklist.md` | `A-checklist.md` | unchanged |
| `B-further-reading.md` | `C-further-reading.md` | **`git mv` — has content** |

**21 new files** with no predecessor:
`02`, `12`, `13`, `16`, `18`, `23`, `24`, `26`, `27`, `28`, `31`, `32`,
`35`, `36`, `37`, `38`, `39`, `40`, `41`, `B-verification.md` — plus the
second half of each split.

Every new and migrated chapter file gets exactly:
```markdown
# [Chapter Title]

*This chapter is planned but not yet written.*
```

---

## 5. The three recurring threads — reserve the mechanism now

These carry the material that distinguishes this book. They are not
being written in this task, but the **mechanism must exist before
drafting starts**, or the first fifty uses will be inconsistent.

Documenter renders `!!! category "Title"` as
`<div class="admonition is-category">`, so custom categories work with
a stylesheet. Define three:

```markdown
!!! disagreement "When Implementations Disagree"
    R's `stats::arima` reports `nobs` as `n − d`; Python's `SARIMAX`
    reports the full `n`. Every information criterion downstream
    inherits the choice.

!!! india "The Indian Series"
    Diwali has no closed-form Gregorian date...

!!! julia "Under the Hood"
    Multiple dispatch is why `tsvalues` works across four container
    types without conversion...
```

**This requires a stylesheet**, which the docs do not currently have —
`make.jl` presently has `assets=String[]`. Create
`docs/src/assets/custom.css` with rules for `.admonition.is-disagreement`,
`.is-india`, `.is-julia`, and register it:

```julia
format = Documenter.HTML(;
    canonical = "https://MSALabs.github.io/TSAnalytics.jl",
    edit_link = "main",
    assets = ["assets/custom.css"],
),
```

Pick three visually distinct accent colours. Keep them muted — these
appear dozens of times and must not shout.

---

## 6. `make.jl` — nested navigation

Documenter accepts nested `Pair` vectors, so parts become collapsible
groups rather than 41 flat entries:

```julia
"Introduction to Time Series Analysis" => [
    "Part I — The Series and the Machine" => [
        "introduction/01-why-model-a-time-series.md",
        "introduction/02-what-a-time-series-is.md",
        "introduction/03-differencing-and-integration.md",
        "introduction/04-filters-and-moving-averages.md",
        "introduction/05-autocorrelation.md",
        "introduction/06-the-frequency-domain.md",
        "introduction/07-transformations.md",
    ],
    "Part II — Interrogating a Series" => [
        "introduction/08-stationarity.md",
        "introduction/09-testing-for-unit-roots.md",
        "introduction/10-testing-the-residuals.md",
        "introduction/11-testing-distribution-and-variance.md",
        "introduction/12-the-diagnostic-panel.md",
    ],
    # ... Parts III through IX, same shape ...
    "Appendices" => [
        "introduction/A-checklist.md",
        "introduction/B-verification.md",
        "introduction/C-further-reading.md",
    ],
],
```

Leave the Home, Getting Started, Manual and API Reference entries
exactly as they are.

---

## 7. Verification

Not a `@testset` — a build check:

1. `makedocs` completes with **zero errors**.
2. **No new missing-docstring warnings.** `checkdocs=:exports` must not
   start failing because pages moved.
3. **Every one of the 44 files appears in `pages`.** Documenter warns
   about orphaned files; there should be none. Cross-check the count:
   41 chapters + 3 appendices.
4. **`C-further-reading.md` still has its 39 lines.** Check this
   explicitly — it is the one thing this task can actually destroy.
5. No `@ref` links broken. The stubs contain none today, but
   `index.md` and the Getting Started chapters may link into the
   Introduction; grep for `introduction/` across `docs/src` and fix any
   path that changed.

```bash
grep -rn "introduction/" docs/src --include=*.md | grep -v "^docs/src/introduction/"
```

---

## 8. What to do

1. **Confirm the Option A decision in section 1.** Everything else
   follows from it. If the Introduction is not the book, stop here.
2. `git mv docs/src/introduction/B-further-reading.md docs/src/introduction/C-further-reading.md` — **do this first**, before any bulk deletion, so the one file with content is never at risk.
3. Remove the remaining 20 chapter stubs; create all 41 chapters plus
   `B-verification.md` per section 3, with the standard stub body.
4. Create `docs/src/assets/custom.css` and register it in `make.jl`
   (section 5). Add one worked example of each admonition to
   `B-verification.md` so the styling is visibly exercised before
   drafting begins.
5. Rewrite the `pages` block per section 6.
6. Run the verification in section 7.
7. **Stop.** Do not write chapter content in this task — that is
   deliberately separate, and paced one part at a time.
8. Update wherever documentation progress is tracked: the Introduction
   is restructured, 41 chapters awaiting content, and Part VI's
   placement was a deliberate architectural choice rather than an
   accident of ordering.
