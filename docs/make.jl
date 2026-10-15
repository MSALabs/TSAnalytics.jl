using Documenter
using TSAnalytics

DocMeta.setdocmeta!(TSAnalytics, :DocTestSetup, :(using TSAnalytics); recursive=true)

# A block that prints and then displays a plot or value loses its printed
# output in the rendered page; fail the build rather than ship it.
include(joinpath(@__DIR__, "check_example_blocks.jl"))
let bad = check_example_blocks(joinpath(@__DIR__, "src"))
    isempty(bad) || error("@example blocks that print and then display a value -- " *
                          "split them, or end with `nothing # hide`:\n  " * join(bad, "\n  "))
end

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
            "manual/10-exponential-smoothing.md",
            "manual/11-vector-autoregression.md",
            "manual/12-causality-and-impulse-responses.md",
            "manual/13-cointegration-and-vecm.md",
            "manual/14-structural-var.md",
            # Stage 11: numbered 17-20 (appended, no renames), placed here so
            # plotting and the R/Python comparison stay last
            "manual/17-unobserved-components.md",
            "manual/18-dynamic-factor-models.md",
            "manual/19-regime-switching.md",
            "manual/20-threshold-models.md",
            # Stage 12: 21-23, appended the same way
            "manual/21-intermittent-demand.md",
            "manual/22-change-points-and-breaks.md",
            "manual/23-lag-feature-regression.md",
            "manual/15-plotting.md",
            "manual/16-coming-from-r-python.md",
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
            "Part V -- Models That Forget" => [
                "introduction/24-weighting-the-recent-past.md",
                "introduction/25-a-taxonomy-of-thirty.md",
                "introduction/26-when-the-error-is-multiplicative.md",
                "introduction/27-theta-and-seasonality-ets-cannot-represent.md",
            ],
            "Part VI -- Models That Get Turbulent" => [
                "introduction/28-why-variance-changes.md",
                "introduction/29-arch-and-garch.md",
                "introduction/30-asymmetry.md",
                "introduction/31-forecasting-volatility.md",
                "introduction/32-realized-measures.md",
            ],
            "Part VII -- The Machinery Underneath" => [
                "introduction/33-the-state-space-form.md",
                "introduction/34-the-kalman-filter.md",
                "introduction/35-smoothing.md",
                "introduction/36-time-varying-systems.md",
                "introduction/37-diffuse-initialisation.md",
            ],
            "Part VIII -- Bringing in Outside Information" => [
                "introduction/38-regression-with-arima-errors.md",
                "introduction/39-coefficients-that-drift.md",
                "introduction/40-calendar-effects.md",
                "introduction/41-autoregressive-errors-with-changing-variance.md",
            ],
            "Part IX -- What Statistical Agencies Do" => [
                "introduction/42-official-seasonal-adjustment.md",
                "introduction/43-x13-from-julia.md",
                "introduction/44-adjusting-an-indian-series.md",
            ],
            "Part X -- Series That Move Together" => [
                "introduction/45-two-series-at-once.md",
                "introduction/46-fitting-a-vector-autoregression.md",
                "introduction/47-does-one-series-help-predict-another.md",
                "introduction/48-tracing-a-shock.md",
                "introduction/49-series-that-cannot-drift-apart.md",
                "introduction/50-how-many-relations.md",
                "introduction/51-shocks-with-names.md",
            ],
            "Part XI -- Series That Change Their Mind" => [
                "introduction/52-the-components-you-cannot-see.md",
                "introduction/53-when-a-component-stops-moving.md",
                "introduction/54-many-series-few-forces.md",
                "introduction/55-the-economy-has-two-moods.md",
                "introduction/56-a-different-model-on-each-side-of-a-line.md",
                "introduction/57-a-change-that-does-not-come-back.md",
                "introduction/58-watching-a-model-drift.md",
            ],
            "Part XII -- Beyond the Textbooks" => [
                "introduction/59-mostly-nothing-then-something.md",
                "introduction/60-any-regressor-you-like.md",
            ],
            "Part XIII -- The Frontier" => [
                "introduction/61-the-frontier.md",
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
            "api/exponential-smoothing.md",
            "api/garch.md",
            "api/state-space.md",
            "api/unobserved-components.md",
            "api/dynamic-factors.md",
            "api/regime-switching.md",
            "api/threshold-models.md",
            "api/intermittent-demand.md",
            "api/change-points.md",
            "api/arimax.md",
            "api/var.md",
            "api/cointegration.md",
            "api/structural.md",
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
