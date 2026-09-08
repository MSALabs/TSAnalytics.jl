# TSAnalytics.jl

TSAnalytics.jl is a native Julia package for time series analysis --
ARMA and ARIMA models, the GARCH family for volatility, state-space
methods and the Kalman filter, regression with ARIMA errors, and the
diagnostic and decomposition tools that sit around all of them.

Fitting a seasonal ARIMA model, checking whether a series has a unit
root, or estimating a GARCH(1,1) are all common, well-understood tasks
-- and until now, Julia has had no single package bringing them
together with the rigor a production statistical workflow needs. Each
piece of this package was built against a primary source (Hamilton;
Durbin & Koopman; Tsay; Shumway & Stoffer) and checked, wherever
possible, against real R and Python output -- not assumed correct
because the formula looked right on paper.

```julia
julia> using TSAnalytics, DelimitedFiles

julia> y = Float64.(readdlm(TSAnalytics.AIR_PASSENGERS, ','; skipstart=1)[:, 2]);

julia> m = auto_arima(log.(y); seasonal=true, m=12, D=1)
ARIMA(1,0,1)(0,1,1)[12], n=132 (ml, se: hessian)
───────────────────────────────────────────────────
          Coef.  Std. Error          z     Pr(>|z|)
───────────────────────────────────────────────────
ar1    0.995985  0.00495827  200.874    0.0
ma1   -0.399663  0.0898267    -4.44927  8.6161e-6
sma1  -0.552859  0.0740461    -7.46642  8.24088e-14
───────────────────────────────────────────────────
Log-likelihood: 245.61   AIC: -483.22   BIC: -471.69
```

This is real, directly-run output (`test_data/airpassengers.csv`, the
classic Box-Jenkins airline series, fit on the log scale -- the
standard treatment for its multiplicative seasonality). This package
also ships 90 real textbook datasets you can explore directly via
[`dataset`](@ref) -- see [Primitives](manual/01-primitives.md).
Diagnostics and
forecasting for models fit this way are covered in the
[Manual](manual/01-primitives.md) and [Getting Started](getting-started/03-was-it-any-good.md);
`residuals`/`forecast` are implemented for [`ARXModel`](@ref)/[`GarchModel`](@ref)
today, with the same for the ARMA/ARIMA/SARIMA family tracked as
follow-up work in `development-sequence.md`.

**This package does reimplement these methods natively in Julia.**
That is a deliberate choice, not a shortcut avoided: every stage was
built from a primary source first, then checked against real R and
Python execution wherever those tools could be reached, with every
disagreement between references investigated and resolved rather than
picked arbitrarily. Where verification depth differs -- and it
honestly does, stage to stage -- that is stated directly in this
documentation, not smoothed over.

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

!!! note "Status"
    Stages covering primitives (differencing, filters, ACF/PACF,
    periodogram, Box-Cox), diagnostics, decomposition (classical/STL/
    MSTL), AR-X and classical exponential smoothing, ARMA/ARIMA/SARIMA,
    the GARCH family, state-space methods and the Kalman filter, and
    regression with ARIMA errors (ARIMAX/SARIMAX, both the `:mle` and
    `:tvss` estimation paths) are complete. Full ETS, vector
    autoregression, cointegration, structural time series, and the
    companion TSFeatures.jl package are planned but not yet started.
    See [`development-sequence.md`](https://github.com/MSALabs/TSAnalytics.jl/blob/main/development-sequence.md)
    in the repository root for the complete, current, staged roadmap.

!!! warning "Alpha release"
    TSAnalytics.jl is not yet on Julia's General registry.

    ```julia
    using Pkg
    Pkg.add(PackageSpec(url = "https://github.com/MSALabs/TSAnalytics.jl",
                         rev = "v0.1.0-alpha.1"))
    ```

## Getting started

- Read [Installation](getting-started/01-installation.md), then
  [Your First Model](getting-started/02-first-model.md).
- New to time series analysis? Start with
  [Why Model Time Series At All?](introduction/01-why-model-time-series.md)
  -- the Introduction is written to be readable without Julia in front
  of you.
- Coming from R or Python? The Manual's
  [translation page](manual/11-coming-from-r-python.md) maps common
  workflows directly.
- Already know the task? The [Manual](manual/01-primitives.md) is
  organized around "how do I ..." questions, not a walkthrough.

## Help us improve

Found a bug, a discrepancy against R/Python, or a gap in the
documentation? Please open an issue at
[MSALabs/TSAnalytics.jl](https://github.com/MSALabs/TSAnalytics.jl/issues).

## How the documentation is structured

- **[Getting Started](getting-started/01-installation.md)** is a short,
  linear path from installing the package to fitting, checking, and
  customizing your first model -- read it in order, once, the first
  time you use the package.
- **[Manual](manual/01-primitives.md)** is task-oriented reference:
  eleven "how do I ..." pages, one per area (primitives, diagnostics,
  decomposition, fitting ARMA models, automatic order selection, GARCH
  and volatility, state space and the Kalman filter, ARIMAX and
  regression, forecasting and accuracy, plotting, and coming from
  R/Python) -- come back to whichever page matches the task in front of
  you.
- **[Introduction to Time Series Analysis](introduction/01-why-model-time-series.md)**
  is a conceptual, textbook-style treatment -- twenty chapters plus a
  diagnostic checklist and a further-reading appendix, readable without
  a Julia session open, for understanding *why* a method works, not
  just how to call it.
- **[API Reference](api/primitives.md)** is the complete function-level
  documentation, split by topic (primitives, diagnostics, decomposition,
  ARMA models, GARCH, state space, ARIMAX, forecasting, plotting) rather
  than one long alphabetical list.

## Citing

```bibtex
@software{TSAnalyticsJL,
    author  = {{XKDR Forum}},
    title   = {{TSAnalytics.jl}: Native time series analysis in Julia},
    year    = {2026},
    url     = {https://github.com/MSALabs/TSAnalytics.jl}
}
```

Work leaning on a specific method should also cite its primary source
directly -- Hamilton (1994) for ARMA/state-space theory, Durbin &
Koopman (2012) for the Kalman filter and diffuse initialization, Tsay
(2010) for the GARCH family. The
[Further Reading](introduction/B-further-reading.md) appendix carries
the complete list.

The repository is to move to the [xKDR](https://xkdr.org) organisation
in due course. The URL above is current as of this release.

## About XKDR Forum

TSAnalytics.jl is developed at [XKDR Forum](https://xkdr.org), a
non-profit research organisation based in Mumbai, India. It is the
foundation the rest of this project family is built on --
[SeasonalAdjustment.jl](https://github.com/MSALabs/SeasonalAdjustment.jl)
depends on it directly for exactly this reason, rather than
duplicating primitives that belong here.

## License

TSAnalytics.jl is licensed under the [MIT licence](https://github.com/MSALabs/TSAnalytics.jl/blob/main/LICENSE).
