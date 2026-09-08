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
        # api.md was split into docs/src/api/*.md (see
        # handoff/docs-restructure-skeleton-handoff.md); if this comes
        # back, something regressed.
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
            "manual/02-diagnostics.md",
            "manual/03-decomposition.md",
            "manual/04-fitting-arma-models.md",
            "manual/05-automatic-order-selection.md",
            "manual/06-garch-and-volatility.md",
            "manual/07-state-space-and-kalman.md",
            "manual/08-arimax-and-regression.md",
            "manual/09-forecasting-and-accuracy.md",
            "manual/10-plotting.md",
            "manual/11-coming-from-r-python.md",
        ],
        "Introduction to Time Series Analysis" => [
            "introduction/01-why-model-time-series.md",
            "introduction/02-stationarity.md",
            "introduction/03-differencing-and-unit-roots.md",
            "introduction/04-acf-and-pacf.md",
            "introduction/05-filters-and-moving-averages.md",
            "introduction/06-spectral-analysis.md",
            "introduction/07-transformations.md",
            "introduction/08-classical-decomposition.md",
            "introduction/09-stl-and-robust-decomposition.md",
            "introduction/10-testing-the-residuals.md",
            "introduction/11-testing-for-structure.md",
            "introduction/12-the-arma-model.md",
            "introduction/13-from-arma-to-arima.md",
            "introduction/14-seasonal-arima.md",
            "introduction/15-choosing-an-order.md",
            "introduction/16-volatility-and-garch.md",
            "introduction/17-state-space-and-kalman.md",
            "introduction/18-diffuse-initialization.md",
            "introduction/19-regression-with-arima-errors.md",
            "introduction/20-forecasting.md",
            "introduction/A-checklist.md",
            "introduction/B-further-reading.md",
        ],
        "API Reference" => [
            "api/primitives.md",
            "api/diagnostics.md",
            "api/decomposition.md",
            "api/arma-models.md",
            "api/garch.md",
            "api/state-space.md",
            "api/arimax.md",
            "api/forecasting.md",
            "api/plotting.md",
        ],
    ],
    # doctest=:fix locally regenerates expected doctest output when you
    # deliberately change behaviour; leave as default (true) in CI so a
    # docstring example silently drifting out of date fails the build.
    doctest=true,
    checkdocs=:exports,  # build fails if an exported name has no docstring
)

deploydocs(;
    repo="github.com/MSALabs/TSAnalytics.jl",
    devbranch="main",
)
