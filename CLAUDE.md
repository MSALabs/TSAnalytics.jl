# TSAnalytics.jl — Project Context

This file is read automatically at the start of every Claude Code session
in this repo. It captures policy and convention that applies across all
work here — task-specific detail belongs in `handoff/`, not here.

## What this project is

A native Julia time series analysis package: ARIMA/SARIMAX, seasonal
decomposition, exponential smoothing, GARCH, and the diagnostic tests
that go with them. See `development-sequence.md` for the full staged
roadmap, dependency graph, and what's built vs. planned.

## Non-negotiable policies

1. **Reference, never port.** Every algorithm is implemented natively in
   Julia from its primary source — the original paper or textbook —
   never by translating another package's source code, in any language.
   R (`stats`, `forecast`, `tseries`, `urca`, `vars`, `rugarch`) and
   Python (`statsmodels`, `pmdarima`, `arch`) references are used only to
   (a) resolve ambiguity when a paper underspecifies an implementation
   detail, by checking what a mature reference implementation actually
   does, and (b) validate *output numbers* on standard series — never to
   copy code from.
2. **Container-agnostic by construction.** No dependency on, or
   integration code for, any specific time series container (TSFrames.jl,
   TimeSeries.jl, DataFrames.jl). Every public function accepts anything
   the `tsvalues`/`tsindex` interface (`src/interface.jl`) can be called
   on — by default, any iterable via `collect`.
3. **StatsAPI-first**, matching GLM.jl's conventions — `fit`, `coef`,
   `vcov`, `residuals`, `predict`, `loglikelihood`, `aic`, `bic`.
4. **One state-space engine, many models.** ARIMA/SARIMAX, Unobserved
   Components, and ETS all reduce to a shared `GaussianSSM` rather than
   each reimplementing filtering.
5. **`LinearAlgebra`/BLAS over hand-rolled loops** wherever it's the
   faster, more numerically stable choice.
6. **No unused dependencies.** Every entry in `Project.toml`'s `[deps]`
   earns its place. FFT/wavelet-style numerical primitives (in downstream
   packages like TSFeatures.jl) are a deliberate exception — those are
   infrastructure, not the package's intellectual differentiator.
7. **A function isn't done without a docstring and a verified example.**
   `docs/make.jl` enforces this in CI (`checkdocs=:exports`,
   `doctest=true`) — treat a docs build failure like a test failure.

## Naming conventions already established

- Separate functions over a `method=` string switch — e.g. `adf_test`/
  `kpss_test`, `convolution_filter`/`recursive_filter`, not one function
  with a mode keyword.
- Where a function's argument mirrors R or Python's naming (`init`,
  `sides`, `centre`), keep that exact name and spelling deliberately —
  it keeps validation against the reference direct, without a translation
  step. `centre`, not `center`, is intentional where noted.
- Coefficient/filter arguments stay in whatever order convention the
  reference implementation uses (usually reverse-time-order), even where
  a different order might read more intuitively, for the same
  validate-directly-against-reference reason.

## Working style

- Every function gets a test validated against a real reference number
  (R/Python computed once, hardcoded as the expected value) or a
  hand-verified computation — not just "it runs without erroring."
  Tolerance-based (`atol`/`rtol`), not exact equality — different
  implementations converge to slightly different numbers even when both
  are correct.
- `data/` holds bundled benchmark series (Nile, AirPassengers, sunspots,
  etc. — see `data/README.md`) for exactly this purpose.
- Check `handoff/` for a doc matching the stage you're working on before
  implementing from scratch — many stages already have verified reference
  signatures and hand-checked expected test values written up there.
- Cross off completed rows in `development-sequence.md` directly as you
  finish them — it doubles as the changelog.
