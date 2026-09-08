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
        assets=["assets/custom.css"],
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
            "Part I -- The Series and the Machine" => [
                "introduction/01-why-model-a-time-series.md",
                "introduction/02-what-a-time-series-is.md",
                "introduction/03-differencing-and-integration.md",
                "introduction/04-filters-and-moving-averages.md",
                "introduction/05-autocorrelation.md",
                "introduction/06-the-frequency-domain.md",
                "introduction/07-transformations.md",
            ],
            "Part II -- Interrogating a Series" => [
                "introduction/08-stationarity.md",
                "introduction/09-testing-for-unit-roots.md",
                "introduction/10-testing-the-residuals.md",
                "introduction/11-testing-distribution-and-variance.md",
                "introduction/12-the-diagnostic-panel.md",
            ],
            "Part III -- Pulling a Series Apart" => [
                "introduction/13-components.md",
                "introduction/14-classical-decomposition.md",
                "introduction/15-stl.md",
                "introduction/16-multiple-seasonality.md",
            ],
            "Part IV -- Models That Remember" => [
                "introduction/17-ar-and-ma-processes.md",
                "introduction/18-fitting-arma.md",
                "introduction/19-arima.md",
                "introduction/20-seasonal-arima.md",
                "introduction/21-choosing-an-order.md",
                "introduction/22-forecasting.md",
                "introduction/23-evaluating-honestly.md",
            ],
            "Part V -- Models That Get Turbulent" => [
                "introduction/24-why-variance-changes.md",
                "introduction/25-arch-and-garch.md",
                "introduction/26-asymmetry.md",
                "introduction/27-forecasting-volatility.md",
                "introduction/28-realized-measures.md",
            ],
            "Part VI -- The Machinery Underneath" => [
                "introduction/29-the-state-space-form.md",
                "introduction/30-the-kalman-filter.md",
                "introduction/31-smoothing.md",
                "introduction/32-time-varying-systems.md",
                "introduction/33-diffuse-initialisation.md",
            ],
            "Part VII -- Bringing in Outside Information" => [
                "introduction/34-regression-with-arima-errors.md",
                "introduction/35-coefficients-that-drift.md",
                "introduction/36-calendar-effects.md",
                "introduction/37-autoregressive-errors-with-changing-variance.md",
            ],
            "Part VIII -- What Statistical Agencies Do" => [
                "introduction/38-official-seasonal-adjustment.md",
                "introduction/39-x13-from-julia.md",
                "introduction/40-adjusting-an-indian-series.md",
            ],
            "Part IX -- The Frontier" => [
                "introduction/41-the-frontier.md",
            ],
            "Appendices" => [
                "introduction/A-checklist.md",
                "introduction/B-verification.md",
                "introduction/C-further-reading.md",
            ],
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
