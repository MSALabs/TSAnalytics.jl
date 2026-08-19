module TSAnalytics

using LinearAlgebra
using Statistics
using StatsAPI
using StatsBase: StatsBase, CoefTable
using Optim: Optim, optimize, LBFGS, BFGS, NelderMead

include("interface.jl")
include("statespace/gaussianssm.jl")
include("abstract.jl")
include("datasets.jl")
include("differencing.jl")
include("filters.jl")
include("stattools.jl")
include("unitroot.jl")
include("loess.jl")
include("diagnostics.jl")
include("decompose.jl")
include("stl.jl")
include("mstl.jl")
include("optim.jl")
include("monahan.jl")
include("arx.jl")
include("forecast.jl")
include("accuracy.jl")
include("tscv.jl")
include("holtwinters.jl")

export TimeSeriesModel, StateSpaceModel, UnivariateModel, HypothesisTest
export statistic, pvalue

end # module TSAnalytics
