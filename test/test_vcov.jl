using DelimitedFiles
using LinearAlgebra: diag, issymmetric

# Stage 9B Tier 2.2. The handoff verified the shape of this correctly: `vcov`
# is a FIT-PATH change, not an accessor. Only ARXModel retained a full matrix;
# everything else kept `se::Vector{Float64}`, the diagonal only, and the
# off-diagonal terms were gone by the time any accessor could run.
#
# Two things came out of doing it:
#   * `coef` omitted the estimated mean while `se` included it, so
#     `length(coef(m)) != length(stderror(m))` whenever a mean was fitted --
#     and ArmaModel's own docstring claimed they matched. `coef` now includes
#     it, agreeing with GarchModel, which always included its `mu`.
#   * `_vcov_to_se` clamped a negative variance to 0.0, which printed
#     `ma1 1.0 0.0 Inf NaN` -- a zero standard error claiming infinite
#     precision at a boundary optimum. It reports NaN now, which is what
#     fit_arma's docstring already promised.

@testset "vcov coverage (Stage 9B Tier 2.2)" begin
    V  = joinpath(@__DIR__, "verification")
    y  = vec(readdlm(joinpath(V, "nsdiffs", "cardox240.csv"), ',', skipstart=1))
    ax = vec(readdlm(joinpath(V, "arimaxforecast", "ax_x.csv")))
    ay = vec(readdlm(joinpath(V, "arimaxforecast", "ax_y.csv")))
    e  = dataset("sp500.gr").value[1:800]

    models = Dict(
        "arma"    => fit_arma(y, (2, 0)),
        "arima"   => fit_arima(y, (1, 1, 1)),
        "sarima"  => fit_sarima(y, (1, 1, 1), (0, 1, 1, 12)),
        "arimax"  => fit_arimax(ay, (1, 0, 0), reshape(ax, :, 1)),
        "sarimax" => fit_sarimax(ay, (1, 0, 0), (0, 0, 0, 1), reshape(ax, :, 1)),
        "garch"   => fit_garch(e, 1, 1),
        "arx"     => arx(y, 2),
    )

    @testset "every type answers vcov and stderror" begin
        for (nm, m) in models
            @test applicable(vcov, m)
            @test applicable(stderror, m)
            @test applicable(coef, m)
        end
    end

    @testset "vcov is k x k, matching coef" begin
        for (nm, m) in models
            k = length(coef(m))
            @test size(vcov(m)) == (k, k)
            @test length(stderror(m)) == k
        end
    end

    @testset "stderror is exactly the sqrt of the diagonal" begin
        for (nm, m) in models
            V2 = vcov(m)
            se = stderror(m)
            for i in eachindex(se)
                d = V2[i, i]
                if d > 0
                    @test isapprox(se[i], sqrt(d); rtol=1e-12)
                else
                    @test isnan(se[i])     # non-positive variance -> undefined
                end
            end
        end
    end

    @testset "vcov is symmetric" begin
        for (nm, m) in models
            V2 = vcov(m)
            any(isnan, V2) && continue
            @test isapprox(V2, transpose(V2); atol=1e-10)
        end
    end

    @testset "off-diagonal terms are real, not padding" begin
        # the whole reason to retain the matrix: `se` cannot reconstruct this
        m = fit_arma(y, (2, 0))
        V2 = vcov(m)
        @test size(V2) == (3, 3)
        @test abs(V2[1, 2]) > 1e-8          # AR(1) and AR(2) are correlated
        @test !isapprox(V2[1, 2], 0.0; atol=1e-8)
        # correlation implied by the matrix stays inside [-1, 1]
        r = V2[1, 2] / sqrt(V2[1, 1] * V2[2, 2])
        @test -1.0 <= r <= 1.0
    end

    @testset "ARIMAX vcov diagonal matches the verified R standard errors" begin
        # R: arima(y, order=c(1,0,0), xreg=x) -> sqrt(diag(var.coef))
        #    ar1 0.0686772108  intercept 0.3269969227  x 0.1110204142
        # Julia orders [exog..., arma...], so the x and intercept terms come
        # first -- asserted in Julia's order, not R's printed order.
        m = fit_arimax(ay, (1, 0, 0), reshape(ax, :, 1))
        se = sqrt.(diag(vcov(m)))
        @test isapprox(se, [0.1110204142, 0.3269969227, 0.0686772108]; atol=1e-3)
        @test isapprox(se, stderror(m); rtol=1e-12)
    end

    @testset "se_type selects the covariance, not just its diagonal" begin
        vs = Dict(st => vcov(fit_arma(y, (2, 0); se_type=st))
                   for st in (:hessian, :opg, :robust))
        # all three are genuinely different matrices, off-diagonal included
        @test !isapprox(vs[:hessian], vs[:opg]; rtol=1e-6)
        @test !isapprox(vs[:hessian], vs[:robust]; rtol=1e-6)
        @test !isapprox(vs[:hessian][1, 2], vs[:robust][1, 2]; rtol=1e-6)
        # and the coefficients are untouched by the choice
        a = fit_arma(y, (2, 0); se_type=:hessian)
        b = fit_arma(y, (2, 0); se_type=:robust)
        @test a.ar == b.ar
    end

    @testset "coef now includes the estimated mean" begin
        m = fit_sarima(y, (1, 0, 1), (0, 0, 0, 1))      # include_mean defaults true
        @test m.mean !== nothing
        @test length(coef(m)) == 3                       # phi, theta, mean
        @test coef(m)[end] == m.mean
        @test length(coef(m)) == length(stderror(m))     # the point of the change

        # and omits it when it was not estimated
        m0 = fit_sarima(y, (1, 0, 1), (0, 0, 0, 1); include_mean=false)
        @test m0.mean === nothing
        @test length(coef(m0)) == 2
        @test length(coef(m0)) == length(stderror(m0))

        # differencing forces the mean off, so coef shortens accordingly
        md = fit_arima(y, (1, 1, 1); include_mean=true)
        @test md.arma.mean === nothing
        @test length(coef(md)) == length(stderror(md))
    end

    @testset "a boundary optimum reports NaN, not a zero standard error" begin
        # ARMA(1,1) on undifferenced CO2 drives the MA term to exactly 1.0 --
        # the invertibility boundary -- so the information matrix is not
        # positive-definite and that parameter's standard error is undefined.
        m = fit_arma(y, (1, 1))
        @test isapprox(m.ma[1], 1.0; atol=1e-6)
        @test m.converged                        # it did converge, honestly
        @test any(d -> d <= 0, diag(vcov(m)))    # not positive-definite
        @test any(isnan, stderror(m))            # and said so
        @test !any(iszero, stderror(m))          # never 0.0, which would read as certainty

        # the printed table must not claim infinite significance
        out = sprint(show, m)
        @test !occursin("Inf", out)
        @test occursin("NaN", out)
    end

    @testset "_vcov_to_se on constructed matrices" begin
        f = TSAnalytics._vcov_to_se
        @test f([4.0 0.0; 0.0 9.0]) == [2.0, 3.0]
        @test isnan(f([-1.0 0.0; 0.0 9.0])[1])          # negative -> NaN
        @test f([-1.0 0.0; 0.0 9.0])[2] == 3.0          # the other is unaffected
        @test isnan(f([0.0 0.0; 0.0 9.0])[1])           # exactly zero is also undefined
        @test all(isnan, f(fill(NaN, 2, 2)))            # singular propagates
    end
end
