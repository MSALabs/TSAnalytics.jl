# TSAnalytics.jl

A native Julia time series analysis package: ARIMA/SARIMAX, seasonal
decomposition, exponential smoothing, GARCH, and the diagnostic tests
that go with them -- built incrementally, in the spirit of GLM.jl
(separation of model/predictor concerns, `StatsAPI` conventions,
`LinearAlgebra`-backed numerics).

**Status:** pre-v1.0, active development. See
[`development-sequence.md`](https://github.com/MSALabs/TSAnalytics.jl/blob/main/development-sequence.md)
in the repository root for the full staged roadmap and what's currently
built vs. planned.

- New to the package? Start with [Getting Started](getting_started.md).
- Looking for a specific function? See the [API Reference](api.md).

## Design principles

0. **Container-agnostic by construction.** No dependency on, or
   integration code for, any specific time series container -- see
   [`tsvalues`](@ref).
1. **StatsAPI-first**, matching GLM.jl's conventions.
2. **One state-space engine, many models.**
3. **Validated against reference implementations**, never ported from
   them -- see the "reference, never port" policy in
   `development-sequence.md`.
4. **`LinearAlgebra` over hand-rolled loops** wherever it's the faster,
   more numerically stable choice.

Full detail on all of these lives in the repository README.
