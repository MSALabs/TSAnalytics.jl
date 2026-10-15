using Test
using TSAnalytics
using Random
using LinearAlgebra: norm, eigvals, I, diag, diagm, dot, issymmetric, cholesky, Hermitian, tr, rank, triu, logdet
using Statistics: mean, std, var
# StatsAPI does not export these; importing them once here keeps a test from
# passing in isolation (where the author imported them) and erroring under the
# full suite, which is exactly how the missing `std` import slipped through.
using StatsAPI: coef, coefnames, vcov, residuals, fitted, predict, loglikelihood, aic, bic, nobs, stderror, dof
using Dates: Date

@testset "TSAnalytics.jl" begin
    include("test_interface.jl")
    include("test_dataset_catalogue.jl")
    include("test_gaussianssm.jl")
    include("test_gaussianssm_smoother.jl")
    include("test_gaussianssm_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_timevaryingssm.jl")
    include("test_timevaryingssm_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally; reuses test_gaussianssm_bulk.jl's 364 cases
    include("test_diffuseinit.jl")
    include("test_var_primitives.jl")   # Stage 10.0: tsmatrix, multi-response _ols, companion
    include("test_var.jl")              # Stage 10.1: after test_var_primitives.jl
    include("test_granger.jl")          # Stage 10.2: after test_var.jl
    include("test_irf.jl")              # Stage 10.3: after test_var.jl
    include("test_vardiag.jl")          # Stage 10.8: after test_var.jl
    include("test_coint.jl")            # Stage 10.4a: after test_unitroot.jl (extends its MacKinnon tables)
    include("test_phillips_ouliaris.jl") # Stage 10.4b
    include("test_johansen.jl")         # Stage 10.5
    include("test_vecm.jl")             # Stage 10.6
    include("test_svar.jl")             # Stage 10.7
    include("test_plotdata.jl")         # plot data for Stages 9-10 (counts in test_recipes.jl)
    include("test_ssm_multivariate.jl") # Stage 11.0a/b/d
    include("test_uc.jl")               # Stage 11.1 (and 11.0c/11.0e)
    include("test_dfm.jl")              # Stage 11.2
    include("test_markovswitching.jl")  # Stage 11.3
    include("test_tar.jl")              # Stage 11.4
    include("test_plot_stage11.jl")     # Stage 11 plot recipes
    include("test_stattools.jl")
    include("test_ccf.jl")
    include("test_unitroot.jl")
    include("test_seasonalunitroot.jl")
    include("test_diagnostics.jl")
    include("test_datasets.jl")
    include("test_differencing.jl")  # after test_datasets.jl: reuses its _load_column helper
    include("test_filters.jl")       # after test_datasets.jl: reuses its _load_column helper
    include("test_decompose.jl")
    include("test_transforms.jl")    # after test_datasets.jl: reuses its _load_column helper
    include("test_fourier.jl")
    include("test_spectral.jl")
    include("test_arspectral.jl")
    include("test_loess.jl")
    include("test_stl.jl")
    include("test_mstl.jl")
    include("test_optim.jl")
    include("test_monahan.jl")
    include("test_arx.jl")
    include("test_autoreg.jl")
    include("test_arma.jl")
    include("test_arima.jl")
    include("test_arima_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_sarima.jl")
    include("test_sarima_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_autoarima.jl")
    include("test_autoarima_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_arimax.jl")
    include("test_arimaxforecast.jl")
    include("test_vcov.jl")
    include("test_autoarimax.jl")
    include("test_forecast.jl")
    include("test_accuracy.jl")
    include("test_scoring.jl")
    include("test_combine.jl")
    include("test_tscv.jl")
    include("test_lagregression.jl")    # Stage 12.3
    include("test_intermittent.jl")     # Stage 12.1
    include("test_changepoint.jl")      # Stage 12.2
    include("test_plot_stage12.jl")     # Stage 12 plot recipes
    include("test_holtwinters.jl")
    include("test_ets.jl")         # after test_holtwinters.jl: the ETS(A,A,A) reduction reuses it
    include("test_ets_mult.jl")   # after test_ets.jl: multiplicative error, classes 2 and 3
    include("test_theta.jl")
    include("test_tbats.jl")      # after test_ets.jl and test_transforms.jl (boxcox)       # after test_ets.jl and test_datasets.jl: SES reuse plus _load_column
    include("test_garch.jl")
    include("test_garcht.jl")      # after test_garch.jl: dist=:t
    include("test_garch_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_garch_gjr_egarch_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_autoregarch.jl")   # after test_garch.jl: reuses GarchModel
    include("test_garchforecast.jl")
    include("test_garchforecast_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_realizedvol.jl")
    include("test_diagnosticplot.jl")  # after test_arma/test_arima/test_sarima: reuses fit_arma/fit_sarima
    include("test_recipes.jl")
end
