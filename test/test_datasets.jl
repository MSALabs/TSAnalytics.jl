using DelimitedFiles

"""Load a single named numeric column from one of the bundled benchmark
CSVs (test-only helper -- not part of the package's public API). Skips
the header row; non-numeric entries (e.g. `internet.csv`'s leading `NaN`)
parse to `NaN` rather than erroring, so callers can `filter(!isnan, ...)`
if needed."""
function _load_column(path::AbstractString, colname::AbstractString)
    raw, header = readdlm(path, ',', String; header=true)
    header = vec(header)
    idx = findfirst(==(colname), header)
    idx === nothing && throw(ArgumentError("column $colname not found in $(header)"))
    col = raw[:, idx]
    return map(col) do s
        v = tryparse(Float64, s)
        v === nothing ? NaN : v
    end
end

@testset "bundled datasets" begin
    # Nile: 100 annual observations, 1871-1970, values in a plausible range
    nile = _load_column(TSAnalytics.NILE, "flow")
    @test length(nile) == 100
    @test all(400 .< nile .< 1400)

    # AirPassengers: 144 monthly observations, 1949-1960, strictly positive,
    # generally increasing (classic multiplicative trend+seasonality series)
    air = _load_column(TSAnalytics.AIR_PASSENGERS, "passengers")
    @test length(air) == 144
    @test all(air .> 0)
    @test mean(air[end-11:end]) > mean(air[1:12])  # last year's average > first year's

    # Sunspots: 1700-1988, non-negative counts
    sunspots = _load_column(TSAnalytics.SUNSPOTS_YEAR, "value")
    @test length(sunspots) == 289
    @test all(sunspots .>= 0)

    # US macro change series: 4 numeric columns beyond Unemployment, all present
    cons = _load_column(TSAnalytics.US_CHANGE, "Consumption")
    unemp = _load_column(TSAnalytics.US_CHANGE, "Unemployment")
    @test length(cons) == length(unemp) == 187

    # Vehicle fatalities: two countries' series, same length
    nf = _load_column(TSAnalytics.VEHICLE_FATALITIES, "nf")
    ff = _load_column(TSAnalytics.VEHICLE_FATALITIES, "ff")
    @test length(nf) == length(ff) == 34

    # Internet: 100 observations, first differenced value has a leading NaN
    net = _load_column(TSAnalytics.INTERNET, "internet")
    dnet = _load_column(TSAnalytics.INTERNET, "dinternet")
    @test length(net) == 100
    @test isnan(dnet[1])
    @test all(!isnan, dnet[2:end])
end

@testset "Montgomery's fixtures reproduce the book's own numbers" begin
    # Table 2.3 (printed p. 54): every control limit of Figures 2.33-2.35,
    # recomputed from the transcribed errors
    e = _load_column(TSAnalytics.FORECAST_ERRORS, "error")
    @test length(e) == 50
    mrbar = mean(abs.(diff(e)))
    σ = 0.8865 * mrbar                                     # Eq. 2.48
    @test round(mean(e), digits = 2) == 0.28               # Fig. 2.33 X̄
    @test round(mean(e) + 3σ, digits = 2) == 8.91          # Fig. 2.33 UCL
    @test round(mean(e) - 3σ, digits = 2) == -8.34         # Fig. 2.33 LCL
    @test round(mrbar, digits = 2) == 3.24                 # Fig. 2.33 MR̄
    @test round(3.267 * mrbar, digits = 2) == 10.60        # Fig. 2.33 MR UCL
    @test round(5σ, digits = 2) == 14.38                   # Fig. 2.34 H (the text's 14.36 rounds MR̄ first)
    @test round(5 * 0.8865 * 3.24, digits = 2) == 14.36
    w = 3σ * sqrt(0.1 / 1.9)
    @test round(mean(e) + w, digits = 3) == 2.261          # Fig. 2.35, λ = 0.1
    @test round(mean(e) - w, digits = 3) == -1.697
    # Table 6.12 (printed p. 335): 104 weeks, spot values from the page
    c = _load_column(TSAnalytics.CEREAL, "sales")
    @test length(c) == 104
    @test (c[1], c[88], c[104]) == (102450.0, 201100.0, 163560.0)
    @test sum(c) == 16384250.0
end
