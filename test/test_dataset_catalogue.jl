@testset "dataset catalogue -- every registered dataset loads" begin
    for name in datasets()
        info = dataset_info(name)
        if info.kind == :external
            @test_throws ErrorException dataset(name)
        else
            d = dataset(name)
            @test length(first(d)) == info.n   # registry n matches actual rows
        end
    end
end

@testset "dataset catalogue -- every dataset has real provenance, no empty fields" begin
    for name in datasets()
        info = dataset_info(name)
        @test !isempty(info.licence)
        @test !isempty(info.source)
        @test !isempty(info.citation)
        @test info.kind in (:published, :derived, :synthetic, :external)
    end
end

@testset "dataset catalogue -- known-value spot checks" begin
    # jj: Johnson & Johnson quarterly earnings, 84 obs (21 years x 4 quarters)
    jj = dataset("jj")
    @test length(jj.value) == 84
    @test jj.date[1] == Date(1960, 1, 1)

    # djia retains full OHLCV structure through the xts conversion
    djia = dataset("djia")
    @test length(djia) == 6   # date + Open/High/Low/Close/Volume
    @test length(djia.date) == 2518

    # the GDP/gdp/GNP/gnp case-insensitive-filesystem collision fix:
    # 4 genuinely distinct astsa objects, verified not to have silently
    # overwritten each other during bundle creation on Windows
    @test length(dataset("GDP").value) == 305
    @test length(dataset("gdp").value) == 287
    @test length(dataset("GNP").value) == 304
    @test length(dataset("gnp").value) == 223
end

@testset "dataset catalogue -- univariate reader distinguishes real calendar time from a plain sequential index" begin
    # soi: monthly decimal-year time -> real Date column, frequency=12
    soi_info = dataset_info("soi")
    @test soi_info.frequency == 12
    @test soi_info.span !== nothing

    # sunspotz: biannual decimal-year -> frequency=2
    @test dataset_info("sunspotz").frequency == 2

    # EQ5: a seismic trace's sample index (1, 2, 3, ...), not calendar
    # time at all -- must NOT be misinterpreted as a decimal year
    eq5 = dataset("EQ5")
    @test haskey(eq5, :time)
    @test !haskey(eq5, :date)
    eq5_info = dataset_info("EQ5")
    @test eq5_info.span === nothing
    @test eq5_info.frequency == 0
end

@testset "dataset catalogue -- multivariate reader, including the pure-grid and NA cases" begin
    # soiltemp: a pure spatial grid, no time/date/index column at all
    st = dataset("soiltemp")
    @test length(st) == 36
    @test length(st[1]) == 64

    # ar1miss / blood: R's own "NA" parses to NaN, not a crash
    @test any(isnan, dataset("ar1miss").value)
    bl = dataset("blood")
    @test any(isnan, bl.WBC) || any(isnan, bl.PLT) || any(isnan, bl.HCT)

    # tsibbledata entries load through the same multivariate reader
    pelt = dataset("pelt")
    @test length(pelt.Year) == 91
end

@testset "dataset catalogue -- sink pattern" begin
    d = dataset("soi", identity)
    @test d isa NamedTuple
    @test d == dataset("soi")

    tbl = datasets(identity)
    @test length(tbl.name) == length(datasets())
    @test all(!isempty, tbl.name)
end

@testset "dataset catalogue -- external datasets carry a usable source URL in the error" begin
    err = try
        dataset("m-ibmspln")
        nothing
    catch e
        e
    end
    @test err !== nothing
    msg = sprint(showerror, err)
    @test occursin("chicagobooth", msg) || occursin("not bundled", msg)

    err2 = try
        dataset("global")
        nothing
    catch e
        e
    end
    @test err2 !== nothing
    @test occursin("massey.ac.nz", sprint(showerror, err2))
end

@testset "dataset catalogue -- argument validation and container-agnostic name lookup" begin
    @test dataset("jj") == dataset(:jj)
    @test dataset_info("jj") == dataset_info(:jj)
    @test_throws KeyError dataset_info("not-a-real-dataset-name")
    @test_throws KeyError dataset("not-a-real-dataset-name")
end
