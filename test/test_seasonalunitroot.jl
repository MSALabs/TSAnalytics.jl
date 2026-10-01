using DelimitedFiles

# Stage 9B Tier 2.1. Dual-verified, which the handoff thought impossible:
# R's `forecast` 9.0.2 turned out to BE installed here, so `ocsb.test`,
# `seas.heuristic` and `nsdiffs` were all read from source and executed
# directly (verification/nsdiffs/nsdiffs.R). pmdarima 2.0.4 was executed too,
# and genuinely disagrees with R -- see the divergence testset.
#
# Two findings corrected the handoff's own premise:
#   * R's nsdiffs defaults to test="seas" (an STL seasonal-strength
#     heuristic), NOT Canova-Hansen. CH needs the separate `uroot` package
#     and is not reachable by default at all.
#   * seasonal_strength matches R EXACTLY once R's Loess jump shortcut is
#     disabled; against R's defaults it differs by 1e-4..8e-3, entirely
#     interpolation, and never enough to move the 0.64 threshold.

@testset "seasonal unit root tests" begin
    V = joinpath(@__DIR__, "verification", "nsdiffs")
    cardox = vec(readdlm(joinpath(V, "cardox240.csv"), ',', skipstart=1))
    etsy   = vec(readdlm(joinpath(V, "burg_y.csv")))
    airp   = vec(readdlm(joinpath(V, "airp.csv")))
    nile   = vec(readdlm(joinpath(V, "nile.csv")))

    @testset "seasonal_strength matches R's seas.heuristic" begin
        # R: forecast:::seas.heuristic(ts(y, frequency=m)) with
        #    stl(..., s.jump=1, t.jump=1, l.jump=1), i.e. R's own jump
        #    shortcut disabled. Agreement is to all ten printed digits.
        @test isapprox(seasonal_strength(cardox, 12), 0.9853673117; atol=1e-9)
        @test isapprox(seasonal_strength(etsy,    4), 0.9741637866; atol=1e-9)
        @test isapprox(seasonal_strength(airp,   12), 0.9413733570; atol=1e-9)
        @test isapprox(seasonal_strength(nile,    4), 0.2653732677; atol=1e-9)
    end

    @testset "R's default jump shortcut is the only difference" begin
        # R's own defaults (s.jump=2 etc. for s.window=11) give these; the
        # gap is interpolation, is small, and never crosses 0.64
        for (y, m, r_default) in ((cardox, 12, 0.9855093981), (etsy, 4, 0.9740632609),
                                   (airp, 12, 0.9406724903), (nile, 4, 0.2728710597))
            fs = seasonal_strength(y, m)
            @test isapprox(fs, r_default; atol=1e-2)       # same ballpark
            @test (fs > 0.64) == (r_default > 0.64)        # same decision
        end
    end

    @testset "seasonal_strength is bounded and behaves at the extremes" begin
        t = 1:240
        strong = [100 + 20sin(2pi*k/12) + 0.01k for k in t]
        none   = collect(1.0:240.0)
        @test 0.0 <= seasonal_strength(strong, 12) <= 1.0
        @test 0.0 <= seasonal_strength(none, 12) <= 1.0
        @test seasonal_strength(strong, 12) > 0.9
        @test seasonal_strength(none, 12) < 0.64
        @test seasonal_strength(fill(5.0, 100), 4) == 0.0   # constant: no structure
    end

    @testset "ocsb_test matches R's forecast::ocsb.test" begin
        # R: forecast:::ocsb.test(ts(y, frequency=m), maxlag=3, lag.method="AIC")
        for (y, m, stat, crit, lag) in
                ((cardox, 12, -1.532646448,  -1.8029627912, 2),
                 (etsy,    4, -1.1143280252, -1.8926999247, 1),
                 (airp,   12,  1.5187623803, -1.8029627912, 0),
                 (nile,    4, -9.3378058762, -1.8926999247, 2))
            t = ocsb_test(y, m)
            @test isapprox(t.statistic, stat; atol=1e-7)
            @test isapprox(t.critical, crit; atol=1e-9)
            @test t.lag_order == lag
            @test t.lag_method == :aic
            @test t.n > 0
        end
    end

    @testset "lag_method=:fixed with maxlag=0 matches R" begin
        # R: ocsb.test(ty, maxlag=0, lag.method="fixed")
        for (y, m, stat) in ((cardox, 12, -2.3135850291), (etsy, 4, -1.2101586072),
                              (airp, 12, 1.5187623803),   (nile, 4, -8.1204091826))
            t = ocsb_test(y, m; maxlag=0, lag_method=:fixed)
            @test isapprox(t.statistic, stat; atol=1e-7)
            @test t.lag_order == 0
        end
    end

    @testset "critical values match calcOCSBCritVal at every period" begin
        REF = Dict(2 => -1.9519830458, 4 => -1.8926999247, 6 => -1.8580558017,
                   7 => -1.8452364865, 12 => -1.8029627912, 24 => -1.7564450738,
                   52 => -1.7167435323)
        for (m, c) in REF
            @test isapprox(TSAnalytics._ocsb_critical(m), c; atol=1e-9)
        end
        # monotone in the period, as the smoothing implies
        cs = [TSAnalytics._ocsb_critical(m) for m in 2:60]
        @test issorted(cs)
    end

    @testset "pmdarima genuinely disagrees with R here" begin
        # Executed directly, pmdarima 2.0.4 OCSBTest(m=m)._compute_test_statistic:
        #   cardox -1.52134327  ets_y -1.13726582  airp 2.77013411  nile -8.77490357
        # It adds a constant to the auxiliary regression, does not lag Z4/Z5,
        # and indexes lag selection differently. This package follows R.
        PMD = Dict("cardox" => -1.52134327, "etsy" => -1.13726582,
                   "airp" => 2.77013411, "nile" => -8.77490357)
        @test !isapprox(ocsb_test(cardox, 12).statistic, PMD["cardox"]; atol=1e-4)
        @test !isapprox(ocsb_test(airp, 12).statistic,   PMD["airp"];   atol=1e-2)
        # on airp the two differ by more than 1.25, far beyond rounding
        @test abs(ocsb_test(airp, 12).statistic - PMD["airp"]) > 1.0
    end

    @testset "nsdiffs reaches R's answer under both tests" begin
        for (y, m, d) in ((cardox, 12, 1), (etsy, 4, 1), (airp, 12, 1), (nile, 4, 0))
            @test nsdiffs(y, m) == d                  # test=:seas, R's default
            @test nsdiffs(y, m; test=:ocsb) == d      # and R's test="ocsb"
        end
    end

    @testset "the two tests can disagree, and R agrees that they do" begin
        # log(AirPassengers) -- the field's most famous seasonal series --
        # is a case where R's own two tests reach different answers.
        # R: seas.heuristic 0.96445397 -> nsdiffs(seas) = 1
        #    ocsb.test stat -1.9508297529 vs critical -1.8029627912
        #                               -> nsdiffs(ocsb) = 0
        ly = log.(airp)
        t = ocsb_test(ly, 12)
        @test isapprox(t.statistic, -1.9508297529; atol=1e-8)
        @test t.lag_order == 0
        @test t.statistic < t.critical           # OCSB: no differencing
        @test seasonal_strength(ly, 12) > 0.64   # strength: difference it
        @test nsdiffs(ly, 12) == 1
        @test nsdiffs(ly, 12; test=:ocsb) == 0
        # which is exactly why the default has to be a documented choice
        @test nsdiffs(ly, 12) != nsdiffs(ly, 12; test=:ocsb)
    end

    @testset "nsdiffs on constructed extremes" begin
        t = 1:144
        strong = [100 + 15sin(2pi*k/12) + 0.05k for k in t]
        @test nsdiffs(strong, 12) == 1
        @test nsdiffs(collect(1.0:144.0), 12) == 0          # pure trend
        @test nsdiffs(fill(7.0, 144), 12) == 0              # constant
        @test nsdiffs(strong, 12; max_D=0) == 0             # max_D respected
        @test nsdiffs(strong, 12) <= 1                      # default max_D
    end

    @testset "nsdiffs degenerate inputs return 0 rather than erroring" begin
        @test nsdiffs(randn(MersenneTwister(3), 10), 12) == 0   # shorter than 2 periods
        @test nsdiffs(fill(1.0, 60), 4) == 0
    end

    @testset "unimplemented tests are refused by name" begin
        for bad in (:hegy, :ch)
            err = try; nsdiffs(cardox, 12; test=bad); nothing; catch e; e; end
            @test err isa ArgumentError
            @test occursin(string(bad), err.msg)
        end
        @test_throws ArgumentError nsdiffs(cardox, 12; test=:bogus)
    end

    @testset "argument validation" begin
        @test_throws ArgumentError ocsb_test(cardox, 1)              # not seasonal
        @test_throws ArgumentError ocsb_test(cardox, 12; lag_method=:bogus)
        @test_throws ArgumentError ocsb_test(cardox, 12; maxlag=-1)
        @test_throws ArgumentError ocsb_test([1.0, NaN, 3.0, 4.0, 5.0], 2)
        @test_throws ArgumentError nsdiffs(cardox, 1)
        @test_throws ArgumentError nsdiffs(cardox, 12; max_D=-1)
        @test_throws ArgumentError seasonal_strength(cardox, 1)
        @test_throws ArgumentError seasonal_strength(randn(MersenneTwister(4), 10), 12)
        # too short for the OCSB regression: refuses rather than returning noise
        @test_throws ArgumentError ocsb_test(randn(MersenneTwister(5), 8), 4; maxlag=3)
    end

    @testset "show prints the statistic against its critical value" begin
        out = sprint(show, ocsb_test(cardox, 12))
        @test occursin("OCSB", out)
        @test occursin("critical", out)
        @test occursin("lag order", out)
        @test !occursin("p-value", out)     # there is none, and none is invented
    end

    @testset "container-agnostic" begin
        @test nsdiffs(Tuple(cardox), 12) == nsdiffs(cardox, 12)
        @test isapprox(seasonal_strength(Tuple(cardox), 12),
                        seasonal_strength(cardox, 12); atol=1e-12)
    end
end

@testset "auto_arima detects D by itself (Stage 9B Tier 2.1)" begin
    V = joinpath(@__DIR__, "verification", "nsdiffs")
    y = vec(readdlm(joinpath(V, "cardox240.csv"), ',', skipstart=1))

    @testset "D is no longer required for seasonal=true" begin
        m = auto_arima(y; seasonal=true, m=12, max_p=2, max_q=2)
        @test m isa TSAnalytics.SarimaModel
        @test m.seasonal_order[2] == 1               # found D=1 unaided
        @test m.order == (1, 1, 1)
        @test m.seasonal_order == (0, 1, 1, 12)      # the airline model
    end

    @testset "seasonal_test selects the convention" begin
        a = auto_arima(y; seasonal=true, m=12, max_p=2, max_q=2, seasonal_test=:seas)
        b = auto_arima(y; seasonal=true, m=12, max_p=2, max_q=2, seasonal_test=:ocsb)
        @test a.seasonal_order == b.seasonal_order   # agree on this series
        @test a.seasonal_order[2] == nsdiffs(y, 12)
    end

    @testset "an explicit D still overrides detection" begin
        @test auto_arima(y; seasonal=true, m=12, D=0, max_p=1, max_q=1).seasonal_order[2] == 0
        @test auto_arima(y; seasonal=true, m=12, D=1, max_p=1, max_q=1).seasonal_order[2] == 1
    end

    @testset "validation still applies to an explicit D" begin
        @test_throws ArgumentError auto_arima(y; seasonal=true, m=12, D=-1)
        @test_throws ArgumentError auto_arima(y; seasonal=true, m=12, D=2, max_D=1)
    end
end
