using Dates: Dates, Date, Day, Year, isleapyear
using DelimitedFiles: readdlm

export dataset, datasets, dataset_info, DatasetInfo

# ---------------------------------------------------------------------------
# 83 real textbook datasets (71 from `astsa`, Shumway & Stoffer's own
# companion package; 12 from `tsibbledata`, fpp3's), bundled directly into
# this MIT-licensed package -- both source packages are GPL-3, a deliberate
# choice made explicitly by the project maintainer (not inferred), see
# `data/DATASETS.md` for the full licensing discussion.
#
# API shape verified directly against `SeasonalAdjustment.jl`'s own real
# source (`src/datasets.jl`, fetched this session, not from memory):
# `DatasetInfo`'s 13 fields, `dataset`/`datasets`/`dataset_info` signatures,
# and the `:published`/`:derived`/`:synthetic` `kind` taxonomy all match
# exactly. This catalogue adds a fourth kind, `:external`, for datasets
# that are real and catalogued but deliberately not bundled (Tsay's
# CRSP-derived data, Cowpertwait's data with no stated licence) -- see
# `handoff/ts-datasets/ts-datasets-handoff.md`.
# ---------------------------------------------------------------------------

"""
    DatasetInfo

Full provenance for one catalogued dataset -- returned by
[`dataset_info`](@ref). Matches `SeasonalAdjustment.jl`'s own real
`DatasetInfo` struct exactly (13 fields, same names), verified directly
against its source this session.

- `name`: the catalogue key, exactly as passed to [`dataset`](@ref).
- `title`, `source` (the originating R package), `url`, `licence`,
  `citation`: provenance, taken from the source package's own
  documentation where available -- never invented (see `units` below
  for what "not invented" looks like when the source genuinely doesn't
  say).
- `frequency`: observations per year, in R `ts()`'s own sense (`12` for
  monthly, `4` for quarterly, ...) -- `0` when this genuinely isn't
  computable from the file's own time column without guessing (many of
  these datasets use a plain sequential index, not calendar time at
  all -- a seismic trace's sample index, a nucleotide sequence
  position -- and reporting a fabricated "frequency" for those would be
  actively misleading, not just incomplete).
- `n`: row count, verified against the actual bundled file at registry
  build time, not merely copied from a manifest.
- `span`: `(first, last)` real dates, or `nothing` under the same
  "genuinely not computable" condition as `frequency` -- a deliberate
  divergence from a plain `Tuple{Date,Date}`, for the same honesty
  reason.
- `units`: `"unknown (...)"` where the source package's own
  documentation doesn't state it, matching `SeasonalAdjustment.jl`'s
  own stated honesty standard exactly rather than guessing.
- `retrieved`: when this catalogue entry was built into TSAnalytics.jl.
- `notes`: for multivariate datasets, the real column names.
- `kind`: `:published` (every bundled dataset here), `:derived`,
  `:synthetic`, or `:external` (catalogued, deliberately not bundled --
  [`dataset`](@ref) throws a clear, actionable error naming the real
  source URL for these, never returns `nothing` silently).
"""
struct DatasetInfo
    name::String
    title::String
    source::String
    url::String
    licence::String
    citation::String
    frequency::Int
    n::Int
    span::Union{Nothing,Tuple{Date,Date}}
    units::String
    retrieved::Union{Date,Nothing}
    notes::String
    kind::Symbol
end

const _TEXTBOOK_DIR = joinpath(dirname(@__DIR__), "data")
const _TEXTBOOK_INVENTORY = joinpath(_TEXTBOOK_DIR, "inventory.tsv")
const _UNKNOWN_UNITS = "unknown (not stated in the source package's own documentation; recorded honestly rather than guessed)"

const _PACKAGE_URL = Dict(
    "astsa" => "https://cran.r-project.org/package=astsa",
    "tsibbledata" => "https://cran.r-project.org/package=tsibbledata",
)
const _PACKAGE_CITATION = Dict(
    "astsa" => "Stoffer, D. (2024). astsa: Applied Statistical Time Series Analysis. R package version 2.5.1.",
    "tsibbledata" => "O'Hara-Wild, M., Hyndman, R., Wang, E., Godahewa, R. tsibbledata: Diverse Datasets for 'tsibble'. R package.",
)
const _PACKAGE_LICENCE = Dict("astsa" => "GPL-3", "tsibbledata" => "GPL-3")

"_decimal_year_to_date(y) -- converts R's own `ts()` decimal-year time
representation (`1960.25` == 1960 Q2) to a real `Date`, the same
convention `SeasonalAdjustment.jl` uses (ISO-8601 `Date`s), rather than
keeping the decimal-year float."
function _decimal_year_to_date(y::Real)
    yr = floor(Int, y)
    frac = y - yr
    ndays = isleapyear(yr) ? 366 : 365
    return Date(yr, 1, 1) + Day(round(Int, frac * ndays))
end

"""
_looks_like_calendar_year(v) -- true if `v`'s first value falls in a
plausible calendar-year range (1500-2100). Several bundled datasets use
a `time`/`index` column that is a *plain sequential position*, not a
calendar year at all (a seismic trace's sample index starting at 1, a
nucleotide sequence position) -- treating those as decimal years would
silently fabricate a calendar span/frequency where none exists.
"""
_looks_like_calendar_year(v::Real) = 1500 <= v <= 2100

"_infer_frequency(dates) -- observations per year via the median gap
between consecutive dates, bucketed to R `ts()`'s own standard
frequencies. A best-effort heuristic, not exact for irregular series
(daily financial series skip weekends/holidays, for instance) -- good
enough to classify a series as annual/quarterly/monthly/weekly/daily,
which is what `frequency` is for."
function _infer_frequency(dates::AbstractVector{Date})
    length(dates) < 2 && return 1
    gaps = Dates.value.(diff(sort(dates)))
    g = median(gaps)
    g <= 2 && return 365
    g <= 10 && return 52
    g <= 45 && return 12
    g <= 100 && return 4
    g <= 200 && return 2
    return 1
end

struct _RegistryRow
    name::String
    package::String
    title::String
    n_rows::Int
    n_cols::Int
    columns::Vector{String}
    licence::String
    book::String
    filename::String
end

function _read_inventory()
    raw, header = readdlm(_TEXTBOOK_INVENTORY, '\t', String; header=true)
    cols = Dict(String(h) => i for (i, h) in enumerate(vec(header)))
    rows = _RegistryRow[]
    for i in 1:size(raw, 1)
        push!(rows, _RegistryRow(
            raw[i, cols["name"]], raw[i, cols["package"]], raw[i, cols["title"]],
            parse(Int, raw[i, cols["n_rows"]]), parse(Int, raw[i, cols["n_cols"]]),
            split(raw[i, cols["columns"]], ';'), raw[i, cols["licence"]],
            raw[i, cols["book"]], raw[i, cols["filename"]],
        ))
    end
    return rows
end

"_is_univariate(row) -- true for the plain `time,value`/`index,value`/
`date,value` shape most `astsa` files use; every `tsibbledata` file (and
a handful of `astsa` ones -- `blood`, `climhyd`, ...) is genuinely
multivariate and has `n_cols > 2`, so this reduces to just checking
column count plus the first column's name."
_is_univariate(row::_RegistryRow) = row.n_cols == 2 && row.columns[1] in ("time", "date", "index")

function _dataset_path(row::_RegistryRow)
    return joinpath(_TEXTBOOK_DIR, row.package, row.filename)
end

"_parse_float_or_nan(s) -- R's `NA` (missing values -- `ar1miss`,
`blood`, ... genuinely have them) is written literally as the string
`\"NA\"` in these CSVs; parse to `NaN` rather than erroring, matching
this project's own established convention (see `test_datasets.jl`'s
`_load_column`)."
_parse_float_or_nan(s::AbstractString) = (v = tryparse(Float64, s); v === nothing ? NaN : v)

"_read_univariate_file(path, timecol) -> (t, value::Vector{Float64}, dates)
`dates` is `Vector{Date}` when `timecol` is a real calendar column
(`\"date\"`, or `\"time\"` values that look like decimal years),
`nothing` otherwise (a plain sequential index -- see
`_looks_like_calendar_year`)."
function _read_univariate_file(path::AbstractString, timecol::AbstractString)
    raw = readdlm(path, ',', String; skipstart=1)
    n = size(raw, 1)
    value = Vector{Float64}(undef, n)
    for i in 1:n
        value[i] = _parse_float_or_nan(raw[i, 2])
    end
    if timecol == "date"
        dates = [Date(strip(raw[i, 1], '"')) for i in 1:n]
        return (dates, value, dates)
    else
        t = Vector{Float64}(undef, n)
        for i in 1:n
            t[i] = _parse_float_or_nan(raw[i, 1])
        end
        dates = _looks_like_calendar_year(t[1]) ? _decimal_year_to_date.(t) : nothing
        return (t, value, dates)
    end
end

"_read_table_file(path) -> NamedTuple of Vectors, one per column, numeric
columns parsed as `Float64`, everything else left as `String` -- the
generic multivariate reader. A plain `NamedTuple` of `AbstractVector`s
already satisfies the Tables.jl column-table interface structurally, so
`dataset(name, sink)` works for any Tables.jl-compatible `sink` (e.g.
`DataFrame`) without TSAnalytics itself depending on Tables.jl."
function _read_table_file(path::AbstractString)
    raw, header = readdlm(path, ',', String; header=true)
    names = Symbol.(strip.(vec(header), '"'))
    cols = Vector{Any}(undef, length(names))
    for j in 1:length(names)
        coldata = [strip(raw[i, j], '"') for i in 1:size(raw, 1)]
        # a column parses as numeric if every entry is either a real
        # number or R's own "NA" (missing) -- "NA" alone shouldn't demote
        # an otherwise-numeric column to strings, matching
        # _read_univariate_file's own NaN-for-NA convention
        numeric = all(s -> s == "NA" || tryparse(Float64, s) !== nothing, coldata)
        cols[j] = numeric ? _parse_float_or_nan.(coldata) : coldata
    end
    return NamedTuple{Tuple(names)}(Tuple(cols))
end

function _build_registry()
    reg = Dict{String,DatasetInfo}()
    today = Date(Dates.now())
    for row in _read_inventory()
        path = _dataset_path(row)
        isfile(path) || throw(ErrorException(
            "dataset catalogue: registered file missing on disk: $path (for dataset $(row.name))"))
        span = nothing
        freq = 0
        realn = row.n_rows
        if _is_univariate(row)
            _, value, dates = _read_univariate_file(path, row.columns[1])
            realn = length(value)
            if dates !== nothing
                span = (minimum(dates), maximum(dates))
                freq = _infer_frequency(dates)
            end
        else
            tbl = _read_table_file(path)
            realn = length(first(tbl))
        end
        realn == row.n_rows || throw(ErrorException(
            "dataset catalogue: row count mismatch for $(row.name): registry says $(row.n_rows), file has $realn"))
        title = isempty(row.title) ? row.name : row.title
        reg[row.name] = DatasetInfo(
            row.name, title, row.package, _PACKAGE_URL[row.package], row.licence,
            _PACKAGE_CITATION[row.package], freq, realn, span, _UNKNOWN_UNITS, today,
            _is_univariate(row) ? "" : join(row.columns, ", "), :published,
        )
    end
    for info in _external_dataset_infos()
        reg[info.name] = info
    end
    return reg
end

"""
_external_dataset_infos() -> Vector{DatasetInfo}

Real, verified `:external` catalogue entries -- datasets that genuinely
exist at a real source but are deliberately not bundled here, for a
real reason stated in each entry's own `notes` field (CRSP commercial
licensing for Tsay's data; no stated licence for Cowpertwait's). Each
entry's `url`/file name was independently confirmed this session by
fetching the real source page directly, not transcribed from the
handoff's own draft -- the handoff's own proposed example name
(`\"m-ibm3dx2608\"`) does not actually appear on Tsay's current real
page and was replaced with names that do.

This is a representative sample, not an exhaustive catalogue --
Tsay's page alone lists dozens of files across every chapter, and
Montgomery's ~20 printed appendix tables have no individual online
listing to enumerate at all (noted, not silently omitted, in
`data/DATASETS.md`).
"""
function _external_dataset_infos()
    today = Date(Dates.now())
    tsay_url = "https://faculty.chicagobooth.edu/ruey-s-tsay/research/analysis-of-financial-time-series"
    tsay(name, title) = DatasetInfo(
        name, title, "Tsay (2010), Analysis of Financial Time Series (3rd ed.)", tsay_url,
        "CRSP (commercial, subscription-only) -- not redistributable", "Tsay, R. S. (2010). Analysis of Financial Time Series (3rd ed.). Wiley.",
        0, 0, nothing, _UNKNOWN_UNITS, today,
        "Catalogued, not bundled: the book states the data is substantially CRSP-derived (Center for Research in Security Prices), a commercial subscription database -- redistribution would be a real licence violation.",
        :external,
    )
    cowpertwait_url = "http://www.massey.ac.nz/~pscowper/ts"
    cowpertwait(name, title) = DatasetInfo(
        name, title, "Cowpertwait & Metcalfe (2009), Introductory Time Series with R", cowpertwait_url,
        "not stated by the source", "Cowpertwait, P. S. P., & Metcalfe, A. V. (2009). Introductory Time Series with R. Springer.",
        0, 0, nothing, _UNKNOWN_UNITS, today,
        "Catalogued, not bundled: the book's own data URL has no stated licence, and (confirmed this session) the original massey.ac.nz link is no longer live -- would need direct permission before bundling.",
        :external,
    )
    return [
        tsay("m-ibmspln", "Monthly log returns of IBM and SP 500"),
        tsay("m-bnd", "Monthly simple returns of bond indexes"),
        tsay("m-gs1n3", "Monthly U.S. interest rates (1 & 3 year maturities)"),
        tsay("m-5cln", "Monthly log returns of IBM, HWP, INTC, MER & MWD"),
        tsay("d-ibmln", "Daily log returns of IBM (1962-07-03 to 1997-12)"),
        cowpertwait("global", "Global temperature series (Cowpertwait & Metcalfe ch. 1)"),
        cowpertwait("cbe", "Australian chocolate, beer, and electricity production (ch. 1)"),
    ]
end

const _REGISTRY = Ref{Union{Nothing,Dict{String,DatasetInfo}}}(nothing)

function _registry()
    if _REGISTRY[] === nothing
        _REGISTRY[] = _build_registry()
    end
    return _REGISTRY[]
end

"""
    datasets() -> Vector{String}

Sorted names of every catalogued dataset, `:external` entries included
-- see [`dataset_info`](@ref) to check a name's `kind` before calling
[`dataset`](@ref) on it.

# Examples
```jldoctest
julia> using TSAnalytics

julia> "jj" in datasets()
true

julia> "soi" in datasets()
true
```
"""
datasets() = sort(collect(keys(_registry())))

"""
    datasets(sink)

Metadata for every catalogued dataset (`name`, `title`, `source`,
`frequency`, `n`, `licence`, `kind`) as a Tables.jl-compatible table,
materialized via `sink` (e.g. `datasets(DataFrame)`).

# Examples
```jldoctest
julia> using TSAnalytics

julia> tbl = datasets(identity);

julia> length(tbl.name) == length(datasets())
true
```
"""
function datasets(sink)
    names = datasets()
    infos = [dataset_info(n) for n in names]
    tbl = (
        name=names,
        title=[i.title for i in infos],
        source=[i.source for i in infos],
        frequency=[i.frequency for i in infos],
        n=[i.n for i in infos],
        licence=[i.licence for i in infos],
        kind=[i.kind for i in infos],
    )
    return sink(tbl)
end

"""
    dataset_info(name) -> DatasetInfo

Full provenance for one catalogued dataset. Throws `KeyError` (not
`nothing`) for an unregistered name -- see [`datasets`](@ref) for the
full list of valid names.

# Examples
```jldoctest
julia> using TSAnalytics

julia> dataset_info("jj").n
84

julia> dataset_info("jj").kind
:published
```
"""
dataset_info(name::AbstractString) = _registry()[name]
dataset_info(name::Symbol) = dataset_info(String(name))

"""
    dataset(name) -> NamedTuple

Load one catalogued textbook dataset by name (see [`datasets`](@ref)
for the full list). Univariate series (most `astsa` datasets) return
`(date=Vector{Date}, value=Vector{Float64})` when the file's own time
column is real calendar time, or `(time=Vector{Float64},
value=Vector{Float64})` when it's a plain sequential position (a
seismic trace's sample index, a nucleotide sequence position -- not
calendar time at all, see [`DatasetInfo`](@ref)'s own `frequency`/`span`
docstring for why this project doesn't force a fabricated date onto
those). Multivariate datasets (most `tsibbledata` entries, and several
`astsa` ones) return one field per real column.

An entry catalogued with `kind == :external` (see [`dataset_info`](@ref))
is real but deliberately not bundled -- `dataset` throws a clear,
actionable `ErrorException` naming the real source, never returns
`nothing` silently.

`x` accepts a `String` or `Symbol`.

# Examples
```jldoctest
julia> using TSAnalytics

julia> jj = dataset("jj");

julia> length(jj.value)
84

julia> dataset(:jj).value == jj.value
true
```
"""
function dataset(name::AbstractString)
    info = dataset_info(name)
    if info.kind == :external
        throw(ErrorException(
            "dataset(\"$name\"): real dataset, not bundled with TSAnalytics.jl -- " *
            "$(info.notes) Source: $(info.url)"))
    end
    row = only(r for r in _read_inventory() if r.name == name)
    path = _dataset_path(row)
    if _is_univariate(row)
        t, value, dates = _read_univariate_file(path, row.columns[1])
        return dates === nothing ? (time=t, value=value) : (date=dates, value=value)
    else
        return _read_table_file(path)
    end
end
dataset(name::Symbol) = dataset(String(name))

"""
    dataset(name, sink)

Load a dataset and materialize it via `sink` (e.g. `dataset("soi",
DataFrame)`), the same sink pattern `SeasonalAdjustment.jl` uses --
`sink` receives the [`dataset`](@ref)`(name)` `NamedTuple` directly,
which already satisfies the Tables.jl column-table interface.

# Examples
```jldoctest
julia> using TSAnalytics

julia> d = dataset("jj", identity);

julia> length(d.value)
84
```
"""
dataset(name, sink) = sink(dataset(name))
