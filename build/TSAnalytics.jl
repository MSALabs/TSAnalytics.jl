module TSAnalytics

using LinearAlgebra
using Statistics
using StatsAPI  # imported now, unused until model fitting (fit/coef/etc.) lands in v0.1.0

include("interface.jl")
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

export TimeSeriesModel, StateSpaceModel, UnivariateModel, HypothesisTest
export statistic, pvalue

end # module TSAnalytics
