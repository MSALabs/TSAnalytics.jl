using Test
using TSAnalytics
using Random
using LinearAlgebra: norm
using Statistics: mean

@testset "TSAnalytics.jl" begin
    include("test_interface.jl")
    include("test_gaussianssm.jl")
    include("test_gaussianssm_smoother.jl")
    include("test_gaussianssm_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_timevaryingssm.jl")
    include("test_timevaryingssm_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally; reuses test_gaussianssm_bulk.jl's 364 cases
    include("test_stattools.jl")
    include("test_unitroot.jl")
    include("test_diagnostics.jl")
    include("test_datasets.jl")
    include("test_differencing.jl")  # after test_datasets.jl: reuses its _load_column helper
    include("test_filters.jl")       # after test_datasets.jl: reuses its _load_column helper
    include("test_decompose.jl")
    include("test_loess.jl")
    include("test_stl.jl")
    include("test_mstl.jl")
    include("test_optim.jl")
    include("test_monahan.jl")
    include("test_arx.jl")
    include("test_arma.jl")
    include("test_arima.jl")
    include("test_arima_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_sarima.jl")
    include("test_sarima_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_autoarima.jl")
    include("test_autoarima_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_forecast.jl")
    include("test_accuracy.jl")
    include("test_tscv.jl")
    include("test_holtwinters.jl")
    include("test_garch.jl")
    include("test_garch_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_garch_gjr_egarch_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_garchforecast.jl")
    include("test_garchforecast_bulk.jl")  # gated behind TSANALYTICS_FULL_TESTS internally
    include("test_realizedvol.jl")
end
