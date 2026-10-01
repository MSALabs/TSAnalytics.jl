using DelimitedFiles

# Stage 9B Tier 1.1/1.2. Ground truth re-executed this session against real
# R 4.6.0 `arima(y, order=c(1,0,0), xreg=x)` + `predict(m, n.ahead=10,
# newxreg=xf)` (see verification/arimaxforecast/ax.R), independently
# reproducing every digit the handoff quoted. Python `statsmodels`
# ARIMA(y, exog=x, order=(1,0,0), trend='c') agrees with R to 5-6 decimals,
# so this target is genuinely dual-verified.

@testset "ARIMAX/SARIMAX forecast" begin
    vdir = joinpath(@__DIR__, "verification", "arimaxforecast")
    y  = vec(readdlm(joinpath(vdir, "ax_y.csv")))
    x  = vec(readdlm(joinpath(vdir, "ax_x.csv")))
    xf = vec(readdlm(joinpath(vdir, "ax_xf.csv")))

    m = fit_arimax(y, (1, 0, 0), reshape(x, :, 1))

    @testset "the fit itself still matches R" begin
        @test isapprox(m.beta[1], 2.1364359236; atol=1e-3)     # x coefficient
        @test isapprox(m.beta[2], -0.3070410373; atol=1e-3)    # intercept -- see below
        @test isapprox(m.arma.ar[1], 0.5847256127; atol=1e-3)
        @test isapprox(m.loglik, -197.959062542; atol=1e-4)
        @test isapprox(m.aic, 403.918125084; atol=1e-3)

        # `beta` carries the intercept as its LAST entry: fit_arimax appends a
        # column of ones to the design matrix when include_mean is in force, so
        # `arma.mean` stays `nothing` and the constant lives in `beta`.
        @test m.arma.mean === nothing
        @test length(m.beta) == 2
        @test all(==(1.0), m.exog[:, end])

        # `se` is ordered [exog..., arma...], NOT R's [ar1, intercept, x].
        # Same three numbers, different order -- worth asserting so a future
        # reordering cannot pass silently.
        @test isapprox(m.se, [0.1110204142, 0.3269969227, 0.0686772108]; atol=1e-3)
    end

    @testset "forecast matches R and Python" begin
        f = forecast(m, reshape(xf, :, 1), 10)
        @test f.horizon == 10
        @test length(f.point) == 10
        @test f.model_name == "ARIMAX(1,0,0)"

        @test isapprox(f.point[1:5],
            [-4.4532060919, -3.6415736308, -3.5780284626, -3.1939196254, -2.8211025062];
            atol=1e-3)
        @test isapprox(f.se[1:5],
            [0.9935811581, 1.1509701955, 1.2000561975, 1.2163845829, 1.2219172707];
            atol=1e-3)

        # agreement is far tighter than the stated tolerance -- hold it there,
        # so a regression that stays inside 1e-3 still fails
        @test isapprox(f.point[1], -4.4532060919; atol=1e-5)
        @test isapprox(f.se[1],     0.9935811581; atol=1e-6)

        @test predict(m, reshape(xf, :, 1), 10).point == f.point   # exact alias
    end

    @testset "se grows with horizon and decelerates" begin
        f = forecast(m, reshape(xf, :, 1), 10)
        @test issorted(f.se)
        @test f.se[10] - f.se[9] < f.se[2] - f.se[1]   # stationary AR
    end

    @testset "intervals nest by level" begin
        f = forecast(m, reshape(xf, :, 1), 10; level=[80.0, 95.0])
        @test size(f.lower) == (10, 2)
        @test all(f.lower[:, 2] .<= f.lower[:, 1])
        @test all(f.upper[:, 2] .>= f.upper[:, 1])
    end

    @testset "newexog validation names the mismatch" begin
        @test_throws ArgumentError forecast(m, reshape(xf[1:5], :, 1), 10)  # too few rows
        @test_throws ArgumentError forecast(m, hcat(xf, xf), 10)            # wrong columns
        @test_throws ArgumentError forecast(m, reshape(xf, :, 1), 0)        # zero horizon
        @test_throws ArgumentError forecast(m, reshape(xf, :, 1), 10; level=Float64[])
        @test_throws ArgumentError forecast(m, reshape(xf, :, 1), 10; level=[150.0])

        # the intercept column is reconstructed, never supplied
        @test_throws ArgumentError forecast(m, hcat(xf, ones(10)), 10)
    end

    @testset "zero exog reduces exactly to the plain model" begin
        # The reduction is the strongest available check: it exercises the
        # regression path, the differencing of the design matrix, and the psi
        # weights, and must land bit-identically on the non-exog answer.
        a = fit_arimax(y, (1, 0, 0), zeros(length(y), 1))
        b = fit_arima(y, (1, 0, 0))
        @test isapprox(a.loglik, b.arma.loglik; atol=1e-9)

        fa = forecast(a, zeros(5, 1), 5)
        fb = forecast(b, 5)
        @test isapprox(fa.point, fb.point; atol=1e-10)
        @test isapprox(fa.se, fb.se; atol=1e-12)
    end

    @testset "seasonal path reduces exactly to fit_sarima" begin
        cdx = vec(readdlm(joinpath(@__DIR__, "verification", "robustse", "cardox240.csv"),
                           ',', skipstart=1))
        a = fit_sarimax(cdx, (1, 1, 1), (0, 1, 1, 12), zeros(length(cdx), 1))
        b = fit_sarima(cdx, (1, 1, 1), (0, 1, 1, 12))
        @test isapprox(a.loglik, b.loglik; atol=1e-9)

        fa = forecast(a, zeros(12, 1), 12)
        fb = forecast(b, 12)
        # bit-identical: confirms the SARIMAX psi-weight polynomial multiplies
        # BOTH differencing operators back in, exactly as the SARIMA path does
        @test fa.point == fb.point
        @test fa.se == fb.se
        @test fa.model_name == "SARIMAX(1,1,1)(0,1,1)[12]"
    end

    @testset "seasonal forecast continues from the end of the sample" begin
        cdx = vec(readdlm(joinpath(@__DIR__, "verification", "robustse", "cardox240.csv"),
                           ',', skipstart=1))
        X  = reshape(Float64.(1:length(cdx)), :, 1) ./ length(cdx)
        Xf = reshape(Float64.((length(cdx)+1):(length(cdx)+12)), :, 1) ./ length(cdx)
        ms = fit_sarimax(cdx, (1, 1, 1), (0, 1, 1, 12), X; include_mean=false)
        f  = forecast(ms, Xf, 12)
        @test length(f.point) == 12
        @test issorted(f.se)
        @test abs(f.point[1] - cdx[end]) < 5.0      # a monthly CO2 step, not a jump
        @test all(isfinite, f.point)
    end

    @testset "model=:tvss is refused, naming why" begin
        mt = fit_arimax(y, (1, 0, 0), reshape(x, :, 1); model=:tvss)
        err = try
            forecast(mt, reshape(xf, :, 1), 5)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("tvss", err.msg)
        @test occursin("latent", err.msg)      # says what is actually missing
    end
end

@testset "uniform predict/forecast signature across every type (Stage 9B 1.2)" begin
    cdx = vec(readdlm(joinpath(@__DIR__, "verification", "robustse", "cardox240.csv"),
                       ',', skipstart=1))
    ma = fit_arma(cdx, (1, 1))
    mi = fit_arima(cdx, (1, 1, 1))
    ms = fit_sarima(cdx, (1, 1, 1), (0, 1, 1, 12))
    mx = arx(cdx, 2)

    @testset "every forecastable type answers predict(model, horizon)" begin
        for m in (ma, mi, ms, mx)
            @test applicable(predict, m, 6)
        end
    end

    @testset "predict and forecast are exact aliases" begin
        for m in (ma, mi, ms, mx)
            a = predict(m, 6)
            b = forecast(m, 6)
            @test a.point == b.point
            @test a.se == b.se
            @test a.lower == b.lower && a.upper == b.upper
        end
    end

    @testset "SarimaModel two-argument form equals the three-argument one" begin
        @test predict(ms, 6).point == predict(ms, cdx, 6).point
        @test predict(ms, 6).se == predict(ms, cdx, 6).se
    end

    @testset "ArmaModel forecast is the ARIMA d=0 reduction" begin
        a = predict(ma, 5)
        b = predict(fit_arima(cdx, (1, 0, 1)), 5)
        @test a.point == b.point
        @test a.se == b.se
    end

    @testset "residuals need no series argument" begin
        for m in (ma, ms)
            @test residuals(m) == residuals(m, cdx)
        end
        @test length(residuals(ms)) == ms.nobs
        @test length(residuals(ma)) == ma.nobs
    end

    @testset "original_y round-trips" begin
        @test ma.original_y == cdx
        @test ms.original_y == cdx
    end

    @testset "diagnostic_plot takes the model alone" begin
        for m in (ma, mi, ms)
            d = diagnostic_plot(m)
            @test d isa TSAnalytics.DiagnosticPlotResult
        end
        # the period is read off the model, so a seasonal fit reaches its own lag
        @test diagnostic_plot(ms).nlag == 36
        @test diagnostic_plot(mi).nlag == 20
    end
end
