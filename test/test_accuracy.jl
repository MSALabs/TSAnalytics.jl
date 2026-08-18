@testset "accuracy metrics (exact sktime+R validation)" begin
    # handoff/stage-5.3-accuracy-handoff.md, section 6. Verified this
    # session against real `sktime` (mean_absolute_error,
    # mean_squared_error(square_root=true),
    # mean_absolute_percentage_error(symmetric=false/true),
    # mean_absolute_scaled_error) via inspect.getsource + direct
    # execution, AND against real R (forecast::accuracy(), Rscript.exe
    # via full path) confirming R's MAPE is on the percentage scale.
    y_train = [5.0, 0.5, 4, 6, 3, 5, 2]
    y_true = [3.0, -0.5, 2, 7, 2]
    y_pred = [2.5, 0.0, 2, 8, 1.25]

    @test isapprox(mae(y_true, y_pred), 0.55; atol=1e-10)
    @test isapprox(rmse(y_true, y_pred), 0.6422616289332564; atol=1e-10)
    @test isapprox(mape(y_true, y_pred; as_percentage=false), 0.33690476190476193; atol=1e-10)
    @test isapprox(smape(y_true, y_pred; as_percentage=false), 0.5553379953379953; atol=1e-10)
    @test isapprox(mase(y_true, y_pred, y_train), 0.18333333333333335; atol=1e-10)

    # as_percentage=true (Julia's default) is exactly 100x the fraction,
    # and matches R's forecast::accuracy() MAPE column exactly (33.69048,
    # confirmed via direct Rscript.exe execution)
    @test isapprox(mape(y_true, y_pred), 100 * 0.33690476190476193; atol=1e-8)
    @test isapprox(mape(y_true, y_pred), 33.69047619047619; atol=1e-6)
    @test isapprox(smape(y_true, y_pred), 100 * 0.5553379953379953; atol=1e-8)

    # seasonal MASE (sp=4, independent second case from the handoff, also
    # confirmed via direct sktime execution)
    y_train2 = collect(1.0:24.0)
    y_true2 = [25.0, 27, 24, 30]
    y_pred2 = [24.0, 26, 25, 29]
    @test isapprox(mase(y_true2, y_pred2, y_train2; sp=4), 0.25; atol=1e-10)

    # accuracy() convenience wrapper
    acc = accuracy(y_true, y_pred, y_train)
    @test isapprox(acc.mae, 0.55; atol=1e-10)
    @test isapprox(acc.rmse, 0.6422616289332564; atol=1e-10)
    @test isapprox(acc.mape, 33.69047619047619; atol=1e-6)
    @test isapprox(acc.mase, 0.18333333333333335; atol=1e-10)
    @test haskey(acc, :mase)
    acc_no_train = accuracy(y_true, y_pred)
    @test !haskey(acc_no_train, :mase)
    @test keys(acc_no_train) == (:mae, :rmse, :mape)
end

@testset "accuracy metrics parameter coverage" begin
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [12.0, 18.0, 33.0, 37.0]

    # as_percentage true/false for both mape and smape
    @test mape(y_true, y_pred) == 100 * mape(y_true, y_pred; as_percentage=false)
    @test smape(y_true, y_pred) == 100 * smape(y_true, y_pred; as_percentage=false)
    @test mape(y_true, y_pred; as_percentage=true) == mape(y_true, y_pred)

    # mae/rmse identical (zero error) case
    @test mae(y_true, y_true) == 0.0
    @test rmse(y_true, y_true) == 0.0
    @test mape(y_true, y_true) == 0.0
    @test smape(y_true, y_true) == 0.0

    # rmse >= mae always (Jensen's inequality / AM-QM)
    @test rmse(y_true, y_pred) >= mae(y_true, y_pred)

    # every sp value for mase, on a longer series
    train = collect(1.0:50.0)
    for sp in (1, 2, 5, 10, 24)
        m = mase(y_true, y_pred, train; sp=sp)
        @test isfinite(m)
        @test m >= 0.0
    end

    # container-agnostic: ranges, tuples, Float32
    @test isapprox(mae(1:5, [1.0, 2, 3, 4, 6]), mae([1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 6]); atol=1e-10)
    @test isapprox(mae(Float32.(y_true), Float32.(y_pred)), mae(y_true, y_pred); atol=1e-5)
    @test isapprox(mase(y_true, y_pred, Float32.(train)), mase(y_true, y_pred, train); atol=1e-5)

    # accuracy() with sp forwarded to mase
    train_seas = collect(1.0:24.0)
    acc = accuracy([25.0, 27, 24, 30], [24.0, 26, 25, 29], train_seas; sp=4)
    @test isapprox(acc.mase, 0.25; atol=1e-10)

    # smape docstring warning is present and prominent
    doc = string(Base.Docs.doc(smape))
    @test occursin("Hyndman", doc)
    @test occursin("caution", doc) || occursin("Caution", doc)
end

@testset "accuracy metrics error paths" begin
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0]
    train = [1.0, 2.0, 3.0]

    @test_throws ArgumentError mae(y_true, y_pred)
    @test_throws ArgumentError rmse(y_true, y_pred)
    @test_throws ArgumentError mape(y_true, y_pred)
    @test_throws ArgumentError smape(y_true, y_pred)
    @test_throws ArgumentError mase(y_true, y_pred, train)

    @test_throws ArgumentError mase([1.0, 2, 3], [1.0, 2, 3], [1.0, 2, 3]; sp=0)
    @test_throws ArgumentError mase([1.0, 2, 3], [1.0, 2, 3], [1.0, 2.0]; sp=5)  # train too short
    @test_throws ArgumentError mase([1.0, 2, 3], [1.0, 2, 3], [1.0, 2.0]; sp=2)  # length(train)=2, not > sp=2

    # boundary: train must have MORE than sp observations, exactly sp+1 is the minimum allowed
    @test mase([1.0], [1.0], [1.0, 3.0]; sp=1) isa Float64  # length(train)=2 > sp=1: allowed
    @test_throws ArgumentError mase([1.0], [1.0], [1.0]; sp=1)  # length(train)=1, not > sp=1
end
