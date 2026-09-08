# Handoff: Documentation Restructuring — Skeleton Only

For a fresh Claude Code session picking this up with no prior context.
**This handoff produces structure and stub pages only — no real prose
content.** Content is deliberately out of scope here and belongs to
separate, subsequent handoffs, one section (or even one chapter) at a
time, matching this project's own "one task at a time" discipline.
Writing real content into this handoff's scope would violate that
discipline, not honor it.

## Where this fits

- **Depends on:** nothing — this is pure restructuring of
  `docs/make.jl` and `docs/src/`, no changes to `src/`.
- **Model directly**: `SeasonalAdjustment.jl`'s live docs site
  (confirmed directly, fetched this session) — Home, a 5-chapter
  Getting Started, a 10-page task-oriented Manual, a 20-chapter +
  2-appendix conceptual Introduction, and an organized API Reference.
  TSAnalytics.jl currently has three pages total: `index.md`, one flat
  `getting_started.md`, and a single monolithic `api.md` that's already
  needed its own documented workaround (`size_threshold=1_000_000`)
  just to keep building past 200 KiB. This handoff brings TSAnalytics.jl
  to the same structural standard, for the same reason: TSAnalytics.jl
  is the *larger* package (fourteen chapters vs. one focused wrapper)
  and currently has noticeably less documentation structure for it.

---

## 1. Confirmed file-naming convention to match exactly

From `SeasonalAdjustment.jl`'s actual navigation, fetched directly:
```
docs/src/getting-started/01-installation.md
docs/src/getting-started/02-first-adjustment.md
...
docs/src/manual/01-specifications.md
...
docs/src/introduction/01-why-adjust.md
...
docs/src/introduction/A-checklist.md
docs/src/introduction/B-further-reading.md
docs/src/api.md
```
Two-digit zero-padded numeric prefixes for ordinary chapters, letter
prefixes (`A-`, `B-`) for appendices. Match this exactly — it's a real,
working convention already proven at scale on the sibling package, not
a new one to invent.

---

## 2. Proposed structure

### Getting Started (5 chapters, mirroring the sibling package's own arc: install → first result → check it → customize → where next)

```
docs/src/getting-started/01-installation.md
docs/src/getting-started/02-first-model.md          # fit an ARIMA, forecast it -- the TSAnalytics.jl equivalent of "your first adjustment"
docs/src/getting-started/03-was-it-any-good.md      # residual diagnostics, the diagnostic_plot recipe
docs/src/getting-started/04-beyond-defaults.md      # order selection, method= choices, model=:mle vs :tvss, etc.
docs/src/getting-started/05-where-next.md
```

### Manual (task-oriented "how do I", one worked call per page — covers the actual breadth this package has that the sibling doesn't need to)

```
docs/src/manual/01-primitives.md              # differencing, filters, ACF/PACF, periodogram, Box-Cox
docs/src/manual/02-diagnostics.md             # unit-root tests, portmanteau tests, ARCH-LM, DK heteroskedasticity
docs/src/manual/03-decomposition.md           # classical, STL, MSTL
docs/src/manual/04-fitting-arma-models.md     # fit_arma/fit_arima/fit_sarima
docs/src/manual/05-automatic-order-selection.md
docs/src/manual/06-garch-and-volatility.md
docs/src/manual/07-state-space-and-kalman.md
docs/src/manual/08-arimax-and-regression.md   # model=:mle vs :tvss, the AutoReg-GARCH stage
docs/src/manual/09-forecasting-and-accuracy.md
docs/src/manual/10-plotting.md                # the RecipesBase.jl layer
docs/src/manual/11-coming-from-r-python.md    # matches the sibling package's own translation page directly
```

### Introduction to Time Series Analysis (conceptual, readable without Julia in front of you — the direct analog to the sibling's 20-chapter treatment, mapped onto this package's own actual subject matter and built stages)

```
docs/src/introduction/01-why-model-time-series.md
docs/src/introduction/02-stationarity.md
docs/src/introduction/03-differencing-and-unit-roots.md
docs/src/introduction/04-acf-and-pacf.md
docs/src/introduction/05-filters-and-moving-averages.md
docs/src/introduction/06-spectral-analysis.md
docs/src/introduction/07-transformations.md          # Box-Cox, Guerrero's method
docs/src/introduction/08-classical-decomposition.md
docs/src/introduction/09-stl-and-robust-decomposition.md
docs/src/introduction/10-testing-the-residuals.md     # portmanteau, normality
docs/src/introduction/11-testing-for-structure.md     # unit roots, ARCH-LM, heteroskedasticity
docs/src/introduction/12-the-arma-model.md
docs/src/introduction/13-from-arma-to-arima.md
docs/src/introduction/14-seasonal-arima.md
docs/src/introduction/15-choosing-an-order.md         # information criteria, auto-order search
docs/src/introduction/16-volatility-and-garch.md
docs/src/introduction/17-state-space-and-kalman.md
docs/src/introduction/18-diffuse-initialization.md
docs/src/introduction/19-regression-with-arima-errors.md   # ARIMAX/SARIMAX, model=:mle vs :tvss
docs/src/introduction/20-forecasting.md
docs/src/introduction/A-checklist.md                  # diagnostic checklist, mirroring the sibling's own appendix
docs/src/introduction/B-further-reading.md            # the six books -- Hamilton, fpp3, Durbin & Koopman,
                                                        # Tsay, Ladiray & Quenneville, Shumway & Stoffer --
                                                        # this is the natural home for that whole comparison
                                                        # exercise's bibliography, not scattered across handoffs
```

**Note on chapter 15**: given the 6-book comparison found real, verified
gaps in existing tests (ADF response-surface, KPSS `:auto`, etc.) and
two genuinely new tests (ARCH-LM, DK heteroskedasticity) that are now
implemented, chapters 10/11 should reflect the *complete*, current
diagnostic suite once written, not the original seven — a detail for
the content-writing handoff, not this skeleton, but worth flagging so
whoever writes that chapter knows to check current `src/` state first
rather than write from an earlier mental model.

### API Reference — split the monolithic file

```
docs/src/api/primitives.md
docs/src/api/diagnostics.md
docs/src/api/decomposition.md
docs/src/api/arma-models.md
docs/src/api/garch.md
docs/src/api/state-space.md
docs/src/api/arimax.md
docs/src/api/forecasting.md
docs/src/api/plotting.md
```
Split by the same grouping as the Manual, not alphabetically — a user
looking for GARCH functions shouldn't have to know the exact function
name to find the right section. Removes the need for the
`size_threshold=1_000_000` workaround entirely once split; that
comment in `make.jl` can be deleted, not just left in place unused.

---

## 3. `make.jl` restructuring

```julia
using Documenter
using TSAnalytics

DocMeta.setdocmeta!(TSAnalytics, :DocTestSetup, :(using TSAnalytics); recursive=true)

makedocs(;
    modules=[TSAnalytics],
    authors="Mousum Dutta",
    sitename="TSAnalytics.jl",
    format=Documenter.HTML(;
        canonical="https://MSALabs.github.io/TSAnalytics.jl",
        edit_link="main",
        assets=String[],
        # size_threshold workaround removed -- no longer needed once
        # api.md is split; if this comes back, something regressed
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => [
            "getting-started/01-installation.md",
            "getting-started/02-first-model.md",
            "getting-started/03-was-it-any-good.md",
            "getting-started/04-beyond-defaults.md",
            "getting-started/05-where-next.md",
        ],
        "Manual" => [
            "manual/01-primitives.md",
            # ... all 11, in order
        ],
        "Introduction to Time Series Analysis" => [
            "introduction/01-why-model-time-series.md",
            # ... all 20 + A + B, in order
        ],
        "API Reference" => [
            "api/primitives.md",
            # ... all 9, in order
        ],
    ],
    doctest=true,
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/MSALabs/TSAnalytics.jl",
    devbranch="main",
)
```

---

## 4. Stub page content — the only content this handoff actually writes

Every new page gets exactly this, nothing more:
```markdown
# [Chapter Title]

*This chapter is planned but not yet written.*
```
**Do not write real prose for any chapter in this handoff** — that
temptation is real given how much material already exists from the
6-book comparison work, but filling content here would mean several
chapters get written well (the ones with fresh material in mind) and
others get skipped or rushed, which is worse than an honest, uniform
"not yet written" across the board. Content is the next, separate
handoff (or several).

**Exception**: `introduction/B-further-reading.md` may include the
actual bibliography entries for the six books directly (title, author,
year, one line on what each covers) — this is reference data, not
explanatory prose, and having it in place makes the *next* handoff's
job easier by giving every chapter something concrete to cite back to.

---

## 5. Test / build verification

```julia
# Not a @testset -- a build check. Confirm before considering this done:
using Documenter, TSAnalytics
# makedocs should complete with zero errors and zero missing-docstring
# warnings beyond what already existed before this restructuring --
# checkdocs=:exports must not start failing just because pages moved
```
Also confirm: every internal cross-reference (`@ref` links) that
existed in the old flat `getting_started.md`/`api.md` still resolves
after the split — a real, easy way for this kind of restructuring to
silently break existing links.

---

## What to do with this

1. Create every file in section 2 with the stub content from section 4
   (exception: `B-further-reading.md` gets the real bibliography).
2. Rewrite `make.jl` per section 3.
3. Run the build verification in section 5 — zero errors, zero new
   missing-docstring warnings, all existing cross-references intact.
4. **Stop here.** Do not proceed to writing real chapter content as
   part of this same task — that's deliberately separate, so it can be
   reviewed and paced chapter by chapter the way every other piece of
   this project has been.
5. Update `development-sequence.md` (or wherever documentation work is
   tracked) to mark the skeleton complete and list all ~40 stub pages
   now waiting for content, so the next handoff has a clear, complete
   checklist rather than needing to rediscover what's missing.
