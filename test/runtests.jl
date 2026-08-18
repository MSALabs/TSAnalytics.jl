using Test
using TSAnalytics
using Random
using LinearAlgebra: norm
using Statistics: mean

@testset "TSAnalytics.jl" begin
    include("test_interface.jl")
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
end
