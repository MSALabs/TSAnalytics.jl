# Bundled textbook datasets — provenance and licensing

90 catalogued datasets, accessible via [`dataset`](@ref)/[`datasets`](@ref)/
[`dataset_info`](@ref) (see `src/dataset_catalogue.jl`). 83 are bundled
directly (71 from `astsa`, 12 from `tsibbledata`); 7 more are catalogued
but deliberately not bundled (`kind == :external` — see below).

This duplicates [`DatasetInfo`](@ref)'s own content in human-readable
form, matching `SeasonalAdjustment.jl`'s own convention exactly, "for
anyone who opens the repository and asks where the numbers came from."

## What's here

| Source package | Datasets | Book | Licence |
|---|---|---|---|
| `astsa` v2.5.1 | 71 | Shumway & Stoffer, *Time Series Analysis and Its Applications* (4th ed.) | GPL-3 |
| `tsibbledata` | 12 | Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.) | GPL-3 |

`data/inventory.tsv` is the registry `src/dataset_catalogue.jl` actually
loads: real titles (from the source packages' own `man/*.Rd`
documentation), row/column counts, column names, licence, and the
originating book, per dataset. All 83 CSVs live directly in `data/`
(one flat folder, no per-package subfolders — `inventory.tsv`'s own
`filename` column is what the loader uses to find each one, so there's
no naming ambiguity even though both source packages' files sit side
by side); `data/LICENSE-GPL3.txt` is the full GPL-3 text as shipped by
`astsa` itself, and `data/DESCRIPTION-tsibbledata.txt` is that
package's own `DESCRIPTION` file (which declares `License: GPL-3`
without bundling a licence file — standard CRAN practice for a
CRAN-recognised standard licence).

## Licensing — the decision that was made, and by whom

**Both source packages are GPL-3.** TSAnalytics.jl itself is
MIT-licensed. Whether bundling GPL-3 *data* into an MIT package
triggers copyleft on the surrounding code is a genuinely contested
question — opinions differ on whether *data* files under GPL-3 trigger
copyleft the way *source code* would. This was **not** decided by
inference: the project maintainer was presented with three options
(a separate GPL-3 companion package; bundling directly; a
download-on-demand fetcher with no redistribution) and explicitly chose
to **bundle the data directly into TSAnalytics.jl** — see
`handoff/ts-datasets/ts-datasets-handoff.md` for the full option
comparison this decision was made against.

Attribution is preserved per package (`astsa`: David Stoffer;
`tsibbledata`: the tidyverts team, via their own `DESCRIPTION` file).

## What is NOT bundled, and why (`kind == :external`)

Checked directly this session, not assumed:

- **Hamilton, *Time Series Analysis*** — has **no datasets at all**.
  Verified directly: every appendix in the book is mathematical proofs,
  and there are only four passing empirical data references across 814
  pages. Nothing to catalogue.
- **Tsay, *Analysis of Financial Time Series*** — real data files exist
  on the author's real, current Chicago Booth page
  (`faculty.chicagobooth.edu/ruey-s-tsay/research/analysis-of-financial-time-series`,
  fetched directly this session — the handoff's own originally-proposed
  URL, `faculty.chicagobooth.edu/ruey.tsay/teaching/fts3`, is stale and
  returns 404), but the book states the data is substantially **CRSP**
  (Center for Research in Security Prices), a commercial subscription
  database. Redistribution would be a real licence violation.
  Catalogued (`m-ibmspln`, `m-bnd`, `m-gs1n3`, `m-5cln`, `d-ibmln` — a
  representative sample from the current page, not the book's full
  set) with the real, current URL; never bundled. **The handoff's own
  proposed example dataset name, `"m-ibm3dx2608"`, does not actually
  appear anywhere on the current page** — confirmed by fetching it
  directly, not assumed correct; the catalogued names above were
  verified to exist instead.
- **Cowpertwait & Metcalfe, *Introductory Time Series with R*** — the
  book's own data URL (`massey.ac.nz/~pscowper/ts`) has no stated
  licence, and (confirmed this session, via web search) **the link is
  no longer live**. Two representative dataset names (`global`, `cbe`)
  are catalogued with that original URL for provenance, honestly
  labelled as stale; would need direct permission from the authors
  before ever bundling.
- **Montgomery, Jennings & Kulahci, *Introduction to Time Series*** —
  ~20 data tables in printed Appendix B plus a Wiley FTP site. Wiley,
  all rights reserved, and no individual online file listing was found
  to enumerate — **not individually catalogued at all** (not even as
  `:external` entries), since there is nothing concrete to point at.
  Recorded here explicitly so this isn't silently rediscovered as a gap
  later.

**Calling [`dataset`](@ref) on an `:external` name throws a clear,
actionable `ErrorException`** naming the real source URL and the reason
it isn't bundled — it never returns `nothing` silently, and the
external catalogue above is a representative sample, not exhaustive
(Tsay's real page alone has dozens more files across every chapter).

## A practical note on the source PDFs

Extracting data numerically from the book PDFs is **not a viable
route**, independent of licensing. Hamilton and Montgomery are both
scans with OCR text layers that mangle mathematical subscripts and
URLs badly enough that numeric tables would not survive OCR reliably.
The companion-package route used here is both the legally cleaner and
the *only accurate* one.

## Conversion notes — what was done to the data

Conversions were performed in R directly from the source `.rda` files,
nothing retyped or transcribed by hand:

- `ts` objects → `time,value` CSV, `time` as R's own decimal year
  (`1960.25` == 1960 Q2). Multi-column `mts` objects keep their
  original column names.
- `xts`/`zoo` objects (`djia`, `lap.xts`, `sp500w`) → `date,...` CSV
  with ISO-8601 dates.
- `data.frame`/`tsibble` objects → written directly, date-like columns
  converted to strings.
- Plain numeric vectors → `index,value` CSV.

**`src/dataset_catalogue.jl` converts `time`/`date` columns to real
`Date`s at load time** (matching `SeasonalAdjustment.jl`'s own
ISO-8601 convention) — but **only when the column is genuinely
calendar time**. A real, non-obvious finding from building this
catalogue: several `time`/`index` columns are a **plain sequential
position, not a calendar year at all** — a seismic trace's sample
index (`EQ5`, `EXP6`, both start at `time=1,2,3,...`), a nucleotide
sequence position (`bnrf1ebv`, `bnrf1hvs`), `GGBsuicide`'s `index`
column. Treating a value like `1` as "the year 1 AD" would silently
fabricate a calendar span/frequency where none exists — the loader
checks whether the first time value falls in a plausible calendar-year
range (1500–2100) before converting, and reports `frequency=0`/
`span=nothing` (not a guessed value) for the sequential-index case.
[`DatasetInfo.span`](@ref) is `Union{Nothing,Tuple{Date,Date}}` rather
than a plain `Tuple{Date,Date}` specifically for this reason — a
deliberate, documented divergence from `SeasonalAdjustment.jl`'s own
struct, made for the same "don't invent, record the gap" reason as
`units` below.

**`units` is honestly `"unknown (...)"`** for every entry — the
source packages' `man/*.Rd` files were not individually read for this
session (83 files' worth), so a per-dataset unit wasn't independently
confirmed; recording it as unknown rather than guessing, matching
`SeasonalAdjustment.jl`'s own stated honesty standard for exactly this
situation.

**`citation` is one shared string per source package** (`astsa` or
`tsibbledata`), not 83 individually hand-curated literature citations
— the source packages' own `CITATION`/`DESCRIPTION` metadata, not
invented per dataset.

## A real bug found and fixed while building this: the `GDP`/`GNP` case collision

**`astsa` has two genuinely distinct pairs of objects differing only in
case**: `GDP` (305 obs, no title in `man/*.Rd`) vs. `gdp` (287 obs,
"Quarterly U.S. GDP"), and `GNP` (304 obs, no title) vs. `gnp` (223
obs, "Quarterly U.S. GNP"). **On this project's case-insensitive
development filesystem (Windows), `GDP.csv` and `gdp.csv` collide to
the same file** — whichever was written last during the original R→CSV
conversion silently overwrote the other, *before* this catalogue's own
code was written. Verified directly: the original bundle had lost
`GDP`'s real 305-row data (only the 287-row `gdp` survived under the
lowercase filename) and `gnp`'s real 223-row data (only the 304-row
`GNP` survived under the uppercase filename). Both were regenerated
directly from a freshly-installed real `astsa` package (confirmed
`packageVersion("astsa") == v"2.5"`) with collision-safe filenames
(`GDP_caps.csv`, `gnp_lower.csv` — see `inventory.tsv`'s own `filename`
column, which the loader uses instead of assuming `<name>.csv`) —
verified against the registry's own row counts for all four objects
afterward. No other name in the 83-dataset catalogue collides this way
(checked systematically, not just for this one pair).

## Five `astsa` objects that are genuinely not in this catalogue

`EBV` and `Months` are character vectors (labels, not observations,
not tabular data at all); `fmri` (the full nested-list object, as
opposed to `fmri1`, the flat Chapter-1 subset, which *is* bundled),
`sleep1`, and `sleep2` are nested list structures needing bespoke
per-object handling this session didn't attempt. Recorded here
explicitly rather than silently missing from a catalogue that claims
to cover `astsa`. (The original handoff draft's own list of unconverted
objects incorrectly included `beamd`/`eqexp` here — both are actually
present in the bundle as ordinary multivariate datasets, confirmed by
checking the actual files, not the draft's own claim.)
