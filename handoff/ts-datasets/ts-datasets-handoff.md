# Handoff: Datasets for TSAnalytics.jl — Matching SeasonalAdjustment.jl's Pattern

For a fresh Claude Code session picking this up with no prior context.
This implements a `dataset()` catalogue for TSAnalytics.jl, structurally
matching `SeasonalAdjustment.jl`'s own — verified directly from that
package's real source this session, not from memory.

**Read section 1 before writing any code.** There is a genuine
licensing decision to make first, and it changes where the data
actually lives.

---

## 1. The licensing decision that must be made first

**83 datasets are ready** (see the accompanying bundle): 71 from
`astsa` (Shumway & Stoffer's own companion package) and 12 from
`tsibbledata` (fpp3's). **Both are GPL-3.**

TSAnalytics.jl is MIT-licensed. **GPL-3 is copyleft**, and whether
bundling GPL-3 *data* into an MIT package triggers copyleft on the
surrounding code is a real, genuinely contested question — not a
formality to wave through. Three honest options:

| Option | What it means | Trade-off |
|---|---|---|
| **A. Separate companion package** — `TSAnalyticsDatasets.jl`, GPL-3 | Main package stays MIT; data ships separately | Cleanest legal separation, a well-established pattern; costs one extra install step |
| **B. Bundle directly**, note GPL-3 per dataset | Simplest for users | Requires being comfortable with the copyleft question; get a real licence opinion first |
| **C. Fetcher only** — download on demand | No redistribution at all, no licence question | Needs network at runtime; breaks offline/CI use |

**Recommendation: Option A.** It gives users everything, keeps
TSAnalytics.jl's own licence unambiguous, and mirrors how R itself
separates `astsa`/`tsibbledata` from the packages that use them. This
handoff is written assuming A, but the API below is identical under B
— only the package boundary moves.

---

## 2. The API to match, verified from SeasonalAdjustment.jl's real source

```julia
dataset(name::AbstractString)                 # returns the data
dataset(name::AbstractString, sink)           # sink pattern: dataset("soi", DataFrame)
dataset(name::Symbol, args...)                # Symbol convenience
datasets()                                     # sorted Vector{String} of all names
datasets(sink)                                 # metadata as a table
dataset_info(name)                             # full provenance for one dataset
DatasetInfo                                    # the metadata struct
```
Exported exactly as: `export dataset, datasets, dataset_info, DatasetInfo`

### `DatasetInfo` — 13 fields, copy this structure exactly

```julia
struct DatasetInfo
    name::String
    title::String
    source::String
    url::String
    licence::String        # a first-class field, NOT a README afterthought
    citation::String
    frequency::Int
    n::Int
    span::Tuple{Date,Date}
    units::String
    retrieved::Union{Date,Nothing}
    notes::String
    kind::Symbol
end
```

### The `kind` taxonomy — extend it

SeasonalAdjustment.jl uses `:published` / `:derived` / `:synthetic`.
**Add a fourth for this catalogue**: `:external` — datasets that are
catalogued with real source URLs but deliberately *not* bundled for
licensing reasons (Tsay's CRSP-derived files, Montgomery's Wiley
tables, ITSR's Massey files). `dataset("m-ibm3dx2608")` on an
`:external` entry should throw a clear, actionable error naming the
source URL and the reason — never return nothing silently, and never
pretend the data is missing by accident when it's excluded on purpose.

---

## 3. Data format

SeasonalAdjustment.jl uses a single `date,value` shape for every
dataset, with one reader (`_read_dataset_csv`). **That won't work
here** — this catalogue includes genuinely multivariate data (`djia`
has OHLCV; `global_economy` has 9 columns; `vic_elec` has 5). Two
readers are needed, dispatched on the file's own header:

- `time,value` or `index,value` → univariate series (most `astsa` files)
- anything else → multivariate table

The bundle's `INVENTORY.csv` gives `n_cols` and `columns` per dataset,
so the registry can carry this rather than sniffing at read time.

**Time representation, worth deciding deliberately**: `astsa`'s `ts`
objects convert to R's decimal-year convention (`1960.25` = 1960 Q2),
which is *not* a `Date`. Either convert to proper `Date`s at package
build time using each series' known frequency (cleaner for users,
matches SeasonalAdjustment.jl's ISO-8601 convention), or keep decimal
years and document it. **Recommend converting** — but note that
`DatasetInfo.span::Tuple{Date,Date}` already forces this decision, so
converting is the more consistent choice.

---

## 4. Test matrix

```julia
using Test, Dates

@testset "every registered dataset loads" begin
    for name in datasets()
        info = dataset_info(name)
        if info.kind == :external
            @test_throws ErrorException dataset(name)   # must fail loudly, with a real message
        else
            d = dataset(name)
            @test size(d, 1) == info.n                   # registry n matches actual rows
        end
    end
end

@testset "every dataset has real provenance -- no empty licence fields" begin
    for name in datasets()
        info = dataset_info(name)
        @test !isempty(info.licence)
        @test !isempty(info.source)
        @test !isempty(info.citation)
        @test info.kind in (:published, :derived, :synthetic, :external)
    end
end

@testset "known-value spot checks -- real, verified numbers" begin
    # jj: Johnson & Johnson quarterly earnings, 84 obs (21 years x 4 quarters).
    # Verified directly during bundle creation, not assumed.
    jj = dataset("jj")
    @test size(jj, 1) == 84

    # djia retains full OHLCV structure through the xts conversion --
    # also verified directly during bundle creation
    djia = dataset("djia")
    @test size(djia, 2) == 6   # date + Open/High/Low/Close/Volume
end

@testset "sink pattern works, matching SeasonalAdjustment.jl" begin
    using DataFrames
    df = dataset("soi", DataFrame)
    @test df isa DataFrame
end

@testset "external datasets carry a usable source URL in the error" begin
    # a Tsay dataset -- catalogued, not bundled
    err = try dataset("m-ibm3dx2608"); nothing catch e; e end
    @test err !== nothing
    @test occursin("chicagobooth", sprint(showerror, err)) ||
          occursin("not bundled", sprint(showerror, err))
end
```

---

## 5. Documentation — `data/DATASETS.md`

SeasonalAdjustment.jl carries a human-readable provenance file
duplicating `DatasetInfo`'s content "for anyone who opens the
repository and asks where the numbers came from." Match this.

**Its honesty standard is the part worth copying most**, and it is
genuinely high — two real examples from that file:
- On a licence it couldn't confirm: *"exact redistribution terms not
  independently verified this session... Confirm terms before any wider
  publication or redistribution."*
- On missing units: *"unknown (not stated in the source; recorded
  honestly rather than guessed)."*

Apply the same standard here. Specifically, do not invent `units` or
`span` values for datasets where the source packages don't state them —
record the gap.

---

## 6. What's deliberately excluded, and must be catalogued as `:external`

All verified this session:

- **Hamilton** — **no datasets exist**. Every appendix is proofs; four
  passing empirical references in 814 pages. Nothing to catalogue.
- **Tsay** — 77 files, but the book states the data is substantially
  **CRSP** (commercial subscription). `:external`, with
  `faculty.chicagobooth.edu/ruey.tsay/teaching/fts3` as the URL.
- **Montgomery/Jennings/Kulahci** — ~20 printed tables + Wiley FTP.
  Wiley, all rights reserved. `:external`.
- **Cowpertwait & Metcalfe (ITSR)** — 45 files at
  `massey.ac.nz/~pscowper/ts`, licence not stated. `:external` pending
  a permission check.

---

## 7. What to do with this

1. **Resolve section 1's licensing question first.** Everything else
   depends on where the data lives.
2. Unpack the bundle's CSVs into `data/` (or the companion package's).
3. Implement `DatasetInfo`, `dataset`, `datasets`, `dataset_info` per
   section 2, with the `:external` extension.
4. Build the registry from the bundle's `INVENTORY.csv` — it already
   carries real titles, row/column counts, and licence per dataset,
   taken from the source packages' own documentation.
5. Handle the multivariate/univariate reader split (section 3) and
   decide the time-representation question deliberately.
6. Write `data/DATASETS.md` to SeasonalAdjustment.jl's honesty standard
   (section 5).
7. Run the test matrix (section 4).
8. **Five `astsa` objects were not converted** and are not in the
   bundle: `EBV`, `Months` (character vectors — labels, not data), and
   `beamd`, `eqexp`, `fmri`, `sleep1`, `sleep2` (nested lists needing
   bespoke handling). Either handle these individually or document them
   as unavailable — do not let them silently vanish from a catalogue
   that claims to cover `astsa`.
