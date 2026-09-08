# Where to Go Next

A few design notes worth knowing before you dig further into the
[Manual](../manual/01-primitives.md) or the conceptual
[Introduction to Time Series Analysis](../introduction/01-why-model-a-time-series.md):

- **No container lock-in.** See [`tsvalues`](@ref) — every function in
  this package accepts a `Vector`, `TSFrame`, `TimeArray`, or
  `DataFrame` column directly, not just a plain array.
- **Response-surface p-values by default.** `adf_test`/`pp_test` use the
  same finite-sample MacKinnon response-surface method R/`statsmodels`
  do; `kpss_test`'s `nlags=:auto` (Hobijn et al. 1998) is available
  alongside its `:short` default. See [`ADFTest`](@ref)/[`KPSSTest`](@ref)/
  [`PPTest`](@ref) and `development-sequence.md` for what's still
  approximate.
- **Reference, never port.** Every algorithm here is implemented natively
  from its primary paper or textbook, validated against R/Python output
  numbers on standard series (Nile, AirPassengers, sunspots) — never by
  translating another package's source. See the project's
  `development-sequence.md` for the full policy and roadmap.

For a specific function, jump straight to the
[API Reference](../api/primitives.md).
