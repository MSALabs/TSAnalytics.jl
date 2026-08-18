@testset "cross-validation splitters (exact sktime index validation)" begin
    # handoff/stage-5.4-cv-handoff.md, section 6. Fold boundaries
    # verified this session against real sktime.split.ExpandingWindowSplitter/
    # SlidingWindowSplitter index output on y=arange(20) (0-indexed),
    # translated +1 for Julia's 1-indexing.

    # A: expanding, matches sktime's confirmed fold 0/1/last exactly
    foldsA = expanding_window_split(20; initial_window=10)
    @test foldsA[1] == (collect(1:10), [11])
    @test foldsA[2] == (collect(1:11), [12])
    @test length(foldsA) == 10
    @test foldsA[end] == (collect(1:19), [20])

    # B: sliding, window stays fixed size, confirmed fold 0/1 exactly
    foldsB = sliding_window_split(20; window_length=10)
    @test foldsB[1] == (collect(1:10), [11])
    @test foldsB[2] == (collect(2:11), [12])
    @test all(f -> length(f[1]) == 10, foldsB)

    # C: multi-horizon fh
    foldsC = expanding_window_split(20; initial_window=10, fh=[1, 2, 3])
    @test foldsC[1] == (collect(1:10), [11, 12, 13])

    # D: step_length
    foldsD = expanding_window_split(20; initial_window=10, step_length=2)
    @test length(foldsD) == 5
    @test [f[2][1] for f in foldsD] == [11, 13, 15, 17, 19]

    # tscv end-to-end with a trivial "naive" forecaster (forecast = last training value, repeated)
    y = collect(1.0:20.0)
    naive_forecast(train, h) = fill(train[end], h)
    errs = tscv(y, naive_forecast; h=1, window=nothing, initial=10)
    @test size(errs) == (10, 1)
    @test all(isapprox.(errs, 1.0; atol=1e-10))

    # sliding-window tscv
    errs_sliding = tscv(y, naive_forecast; h=1, window=10)
    @test all(isapprox.(errs_sliding, 1.0; atol=1e-10))

    # multi-horizon tscv
    errs_multi = tscv(y, (train, h) -> fill(train[end], h); h=[1, 2, 3], initial=10)
    @test size(errs_multi, 2) == 3
end

@testset "cross-validation R forecast::tsCV cross-check" begin
    # handoff's own stated "R could not be executed" boundary was
    # corrected this session (same R-reachable finding as Stage 5.2/5.3):
    # real Rscript.exe forecast::tsCV(ts(1:20), far, h=1) with a
    # mean-only Arima gives valid (non-NA) errors exactly at origins
    # 1:19 with values 2:20 -- confirming tscv's own (train_idx, test_idx)
    # semantics and error values agree with R fold-by-fold, module R's
    # extra always-NA trailing/leading padding rows which tscv omits.
    y = collect(1.0:20.0)
    zero_forecast(train, h) = fill(0.0, h)  # matches Arima(order=c(0,0,0), include.mean=FALSE)
    errs = tscv(y, zero_forecast; h=1, initial=1)
    @test size(errs) == (19, 1)
    @test vec(errs) == collect(2.0:20.0)

    errs_sliding = tscv(y, zero_forecast; h=1, window=10)
    @test size(errs_sliding) == (10, 1)
    @test vec(errs_sliding) == collect(11.0:20.0)
end

@testset "cross-validation parameter coverage and error paths" begin
    y = collect(1.0:30.0)
    naive_forecast(train, h) = fill(train[end], h)

    # every initial_window
    for iw in (1, 5, 10, 20)
        folds = expanding_window_split(30; initial_window=iw)
        @test length(folds) == 29 - iw + 1
        for (i, f) in enumerate(folds)
            @test f[1] == collect(1:(iw + i - 1))
        end
    end

    # every window_length
    for wl in (1, 5, 10, 20)
        folds = sliding_window_split(30; window_length=wl)
        @test all(f -> length(f[1]) == wl, folds)
    end

    # every step_length
    for sl in (1, 2, 3, 5)
        folds = expanding_window_split(30; initial_window=10, step_length=sl)
        origins = [f[2][1] for f in folds]
        @test origins == collect(10:sl:29) .+ 1
    end

    # fh as Integer vs equivalent 1-element Vector give identical folds
    @test expanding_window_split(20; initial_window=10, fh=2) ==
          expanding_window_split(20; initial_window=10, fh=[2])

    # container-agnostic tscv input (range, Float32)
    errs_range = tscv(1:20, naive_forecast; h=1, initial=10)
    @test size(errs_range) == (10, 1)
    errs_f32 = tscv(Float32.(collect(1.0:20.0)), naive_forecast; h=1, initial=10)
    @test all(isfinite, errs_f32)

    # error paths: splitters
    @test_throws ArgumentError expanding_window_split(20; initial_window=0)
    @test_throws ArgumentError expanding_window_split(20; initial_window=10, step_length=0)
    @test_throws ArgumentError expanding_window_split(20; initial_window=10, fh=0)
    @test_throws ArgumentError sliding_window_split(20; window_length=0)
    @test_throws ArgumentError sliding_window_split(20; window_length=10, step_length=0)
    @test_throws ArgumentError sliding_window_split(20; window_length=10, fh=[1, 0])

    # window too large / initial_window too large -> zero folds, not an error
    @test isempty(expanding_window_split(5; initial_window=10))
    @test isempty(sliding_window_split(5; window_length=10))
    @test size(tscv(collect(1.0:5.0), naive_forecast; initial=10)) == (0, 1)
end
