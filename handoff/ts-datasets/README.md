# Bundled textbook datasets — provenance and licensing

83 datasets extracted from two GPL-3 R companion packages, converted to
CSV. **Read the licensing section before shipping any of this** — the
GPL-3 obligations are real and carry over.

## What's here

| Source package | Datasets | Book | Licence |
|---|---|---|---|
| `astsa` v2.5.1 | 71 | Shumway & Stoffer, *Time Series Analysis and Its Applications* (4th ed.) | GPL-3 |
| `tsibbledata` | 12 | Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.) | GPL-3 |

`INVENTORY.csv` lists every dataset with its real title (taken from the
source packages' own `man/*.Rd` documentation, not invented), row/column
counts, column names, licence and originating book.

## Licensing — the part that actually matters

**Both source packages are GPL-3.** This is genuinely permissive for
redistribution, but it is *not* public domain and carries conditions:

- Attribution must be preserved (`astsa`: David Stoffer; `tsibbledata`:
  the tidyverts team).
- **GPL-3 is a copyleft licence.** Bundling GPL-3 data into an
  MIT-licensed package is a real legal question, not a formality —
  opinions differ on whether *data* files under GPL-3 trigger copyleft
  on the surrounding code the way *source code* would. **This needs a
  deliberate decision, and ideally a licence check, before shipping.**
  Two safe options if the answer isn't clear:
  1. Ship the data in a **separate companion package** (e.g.
     `TSAnalyticsDatasets.jl`) licensed GPL-3, keeping the main package
     MIT — this is the cleanest separation and a common pattern.
  2. Ship a **fetcher** rather than the data, downloading from the
     original source on demand — no redistribution, no licence question.

`astsa/LICENSE-GPL3.txt` is the full GPL-3 text as shipped by `astsa`
itself. `tsibbledata` declares `License: GPL-3` in its `DESCRIPTION`
(included as `tsibbledata/DESCRIPTION-tsibbledata.txt`) without bundling
a licence file — standard CRAN practice, since GPL-3 is a
CRAN-recognised standard licence.

## What is NOT here, and why

Checked and deliberately excluded:

- **Hamilton, *Time Series Analysis*** — has **no datasets at all**.
  Verified directly: every appendix in the book is mathematical proofs,
  and there are only four passing empirical data references across 814
  pages. Nothing to bundle.
- **Tsay, *Analysis of Financial Time Series*** — 77 data files exist on
  the author's Chicago Booth page, but the book states the data comes
  substantially from **CRSP** (Center for Research in Security Prices),
  a commercial subscription database. **Redistribution would be a real
  violation.** Catalogue and link; never bundle.
- **Montgomery, Jennings & Kulahci, *Introduction to Time Series*** —
  ~20 data tables in printed Appendix B plus a Wiley FTP site. Wiley,
  all rights reserved. Not redistributable.
- **Cowpertwait & Metcalfe, *Introductory Time Series with R*** — 45
  data files on a Massey University course site, licence not stated.
  Would need permission; not assumed.

## A practical note on the source PDFs

Extracting data numerically from the book PDFs is **not a viable
route**, independent of licensing. Hamilton and Montgomery are both
scans with OCR text layers — Hamilton mangles mathematical subscripts
(`Y,_1` where `Y_{t-1}` is meant, `</>` for `φ`), Montgomery mangles
URLs (`vvww. york. ac. uk`, `ftp: I I ftp. wiley. com`). Numeric tables
would not survive OCR reliably. The companion-package route used here
is both the legally cleaner and the *only accurate* one.

## Conversion notes — what was done to the data

Conversions were performed in R directly from the source `.rda` files.
Nothing was retyped or transcribed by hand.

- **`ts` objects** → `time,value` CSV, with `time` as R's own decimal
  year (e.g. `1960.25` for 1960 Q2). Multi-column `mts` objects keep
  their original column names.
- **`xts`/`zoo` objects** (`djia`, `lap.xts`, `sp500w`) → `date,...`
  CSV with ISO-8601 dates, converted from the raw numeric index since
  the `xts` package itself wasn't available.
- **`data.frame` objects** → written directly.
- **Plain numeric vectors** → `index,value` CSV.
- **`tsibbledata`** → written as data frames, with date-like columns
  (`Date`, `yearmonth`, `yearquarter`, `POSIXct`) converted to strings.

**Five objects in `astsa` were deliberately not converted**, because
they are not tabular data: `EBV` and `Months` are character vectors
(labels, not observations), and `beamd`, `eqexp`, `fmri`, `sleep1`,
`sleep2` are nested list structures needing bespoke handling. These
should be either handled individually later or documented as
unavailable — not silently dropped from the catalogue.

**Spot-checks performed**: `jj` (Johnson & Johnson quarterly earnings)
has 84 observations, matching the known 21-years-by-4-quarters series;
`djia` retained its full OHLCV column structure through the `xts`
conversion; `soi` has correct monthly time steps from 1950.
