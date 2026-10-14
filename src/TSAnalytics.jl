module TSAnalytics

using LinearAlgebra
using Random
using Statistics
using StatsAPI
using StatsBase: StatsBase, CoefTable
using Optim: Optim, optimize, LBFGS, BFGS, NelderMead
using ForwardDiff: ForwardDiff
using FFTW: fft
using RecipesBase: RecipesBase, @recipe, @series, @layout

include("interface.jl")
include("dataset_catalogue.jl")
include("statespace/gaussianssm.jl")
include("statespace/timevaryingssm.jl")
include("statespace/diffuseinit.jl")
include("statespace/ssminit.jl")     # Stage 11.0c: partitioned initialisation
include("statespace/multivariate.jl") # Stage 11.0a/b: sequential multivariate filter and smoother
include("statespace/stationaryvar.jl") # Stage 11.0d: stationary VAR reparameterisation
include("abstract.jl")
include("datasets.jl")
include("differencing.jl")
include("filters.jl")
include("stattools.jl")
include("unitroot.jl")
include("loess.jl")
include("diagnostics.jl")
include("decompose.jl")
include("transforms.jl")
include("spectral.jl")
include("stl.jl")
include("mstl.jl")
include("seasonalunitroot.jl")   # after stl/mstl: seasonal_strength decomposes
include("optim.jl")
include("monahan.jl")
include("arma.jl")
include("arima.jl")
include("sarima.jl")
include("autoarima.jl")
include("arx.jl")
include("arimax.jl")
include("autoarimax.jl")
include("forecast.jl")
include("accuracy.jl")
include("scoring.jl")       # after accuracy.jl and forecast.jl: uses Forecast
include("combine.jl")      # after forecast.jl: returns a Forecast
include("fourier.jl")
include("arspectral.jl")   # after spectral.jl (PeriodogramResult) and stattools.jl
include("autoreg.jl")      # after arspectral.jl (ar_yw) and arx.jl (_ols)
include("tscv.jl")
include("holtwinters.jl")
include("ets.jl")          # after holtwinters.jl: reuses _hw_heuristic_init
include("theta.jl")        # after ets.jl (SES) and decompose.jl (classical_decompose)
include("tbats.jl")        # after transforms.jl (boxcox) and autoreg.jl (_ar_stationary)
include("var.jl")          # Stage 10.1: after unitroot.jl (_ols, _companion) and forecast.jl
include("granger.jl")      # Stage 10.2: after var.jl and diagnostics.jl (_f_ccdf)
include("irf.jl")          # Stage 10.3: after granger.jl (_var_indices, _vech_indices, _dupl_pinv)
include("vardiagnostics.jl") # Stage 10.8: after irf.jl and diagnostics.jl (JarqueBeraTest, ARCHLMTest, _f_ccdf)
include("cointegration.jl")  # Stage 10.4-10.5: after unitroot.jl (adf_test, MacKinnon tables, _bartlett_lrv) and var.jl
include("vecm.jl")           # Stage 10.6: after cointegration.jl (_johansen_rrr) and irf.jl
include("svar.jl")           # Stage 10.7: after irf.jl (the MA recursion, _impact_ma_rep) and optim.jl
include("uc.jl")             # Stage 11.1: after svar.jl; uses ssminit.jl, timevaryingssm.jl, monahan.jl (partrans), forecast.jl
include("dfm.jl")            # Stage 11.2: after uc.jl and var.jl (VARForecast, _companion)
include("markovswitching.jl") # Stage 11.3: after unitroot.jl (_ols), monahan.jl (partrans), optim.jl
include("tar.jl")              # Stage 11.4: after unitroot.jl (_companion), optim.jl
include("garch.jl")
include("autoregarch.jl")
include("garchforecast.jl")
include("realizedvol.jl")
include("diagnosticplot.jl")
include("plotdata.jl")       # plot data for Stages 9-10: after diagnosticplot.jl (diagnostic_plot) and svar.jl
include("recipes.jl")

export TimeSeriesModel, StateSpaceModel, UnivariateModel, HypothesisTest
export statistic, pvalue, critical_values

end # module TSAnalytics
