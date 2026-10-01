using DelimitedFiles

# Stage 9B Tier 2.4. Ground truth from real forecast::fourier (R 4.6.0),
# executed this session -- see verification/fourier/fourier.R. Four cases:
# K=2/m=12, K=6/m=12 (the Nyquist case, where R drops the sine column and
# returns 11 columns rather than 12), K=1/m=4, and the h form.

@testset "fourier_terms (Stage 9B Tier 2.4)" begin
    S3 = 0.8660254038      # sin(60 deg) = sqrt(3)/2, as R prints it

    @testset "K=2, m=12 matches R column for column" begin
        f = fourier_terms(48, 12, 2)
        @test f.names == ["S1-12", "C1-12", "S2-12", "C2-12"]
        @test size(f.terms) == (48, 4)
        @test isapprox(f.terms[1, :], [0.5, S3, S3, 0.5]; atol=1e-9)
        @test isapprox(f.terms[2, :], [S3, 0.5, S3, -0.5]; atol=1e-9)
        @test isapprox(f.terms[3, :], [1.0, 0.0, 0.0, -1.0]; atol=1e-9)
        @test isapprox(f.terms[4, :], [S3, -0.5, -S3, -0.5]; atol=1e-9)
    end

    @testset "K = period/2 drops the Nyquist sine column" begin
        # sin(2*pi*(6/12)*t) = sin(pi*t) = 0 for every integer t, so the
        # column is all zeros: no information, and a singular design matrix.
        # R returns 11 columns here, not 12.
        f = fourier_terms(48, 12, 6)
        @test size(f.terms, 2) == 11
        @test f.names[end] == "C6-12"
        @test !("S6-12" in f.names)
        @test isapprox(f.terms[1, :],
            [0.5, S3, S3, 0.5, 1.0, 0.0, S3, -0.5, 0.5, -S3, -1.0]; atol=1e-9)
        # and nothing that survived is a zero column
        @test all(j -> any(!≈(0.0), f.terms[:, j]), 1:size(f.terms, 2))
    end

    @testset "K=1, m=4 matches R" begin
        f = fourier_terms(20, 4, 1)
        @test f.names == ["S1-4", "C1-4"]
        for (i, row) in enumerate(([1.0, 0.0], [0.0, -1.0], [-1.0, 0.0],
                                    [0.0, 1.0], [1.0, 0.0]))
            @test isapprox(f.terms[i, :], row; atol=1e-12)
        end
    end

    @testset "the h form extrapolates exactly, because the terms are deterministic" begin
        f = fourier_terms(48, 12, 2)
        fh = fourier_terms(48, 12, 2; h=3)
        @test size(fh.terms) == (3, 4)
        @test fh.names == f.names
        # 49 = 1 mod 12, so the future rows reproduce rows 1:3 exactly. That
        # the two agree to 1e-12 is the point: there is no uncertainty in what
        # next January's regressor is, which is what makes these usable with
        # fit_arimax's newexog.
        @test isapprox(fh.terms, f.terms[1:3, :]; atol=1e-12)
    end

    @testset "columns are periodic with the period" begin
        f = fourier_terms(60, 12, 3)
        @test isapprox(f.terms[1:24, :], f.terms[25:48, :]; atol=1e-12)
    end

    @testset "columns are orthogonal over a whole number of cycles" begin
        # 48 = 4 full cycles of 12, so the harmonics are exactly orthogonal
        f = fourier_terms(48, 12, 4)
        G = f.terms' * f.terms
        k = size(G, 1)
        for i in 1:k, j in 1:k
            i == j ? (@test G[i, j] > 0) : (@test isapprox(G[i, j], 0.0; atol=1e-9))
        end
    end

    @testset "K = period/2 spans the same space as seasonal dummies" begin
        # both parameterise an arbitrary periodic pattern, and cost the same:
        # 11 Fourier columns vs 11 dummy columns for period 12
        f = fourier_terms(48, 12, 6)
        @test size(f.terms, 2) == 12 - 1
        # a seasonal dummy pattern is reproducible from them
        target = Float64[(t - 1) % 12 == 0 ? 1.0 : 0.0 for t in 1:48]
        X = hcat(ones(48), f.terms)
        resid = target .- X * (X \ target)
        @test maximum(abs.(resid)) < 1e-9
    end

    @testset "several periods at once, with duplicate frequencies kept once" begin
        f = fourier_terms(336, [24, 168], [2, 2])
        # 1/24 and 7/168 are the same frequency; 168's k=1,2 are 1/168, 2/168
        @test size(f.terms, 2) == 8
        @test f.names == ["S1-24", "C1-24", "S2-24", "C2-24",
                           "S1-168", "C1-168", "S2-168", "C2-168"]

        # a period repeated with itself contributes nothing the second time
        g = fourier_terms(48, [12, 12], [2, 2])
        @test size(g.terms, 2) == 4
        @test g.names == ["S1-12", "C1-12", "S2-12", "C2-12"]
    end

    @testset "K=0 yields no columns rather than an error" begin
        f = fourier_terms(48, 12, 0)
        @test size(f.terms) == (48, 0)
        @test isempty(f.names)
    end

    @testset "error paths" begin
        @test_throws ArgumentError fourier_terms(48, 12, 7)    # K > period/2
        @test_throws ArgumentError fourier_terms(48, 12, -1)
        @test_throws ArgumentError fourier_terms(0, 12, 2)
        @test_throws ArgumentError fourier_terms(48, 1, 1)     # period must be > 1
        @test_throws ArgumentError fourier_terms(48, 12, 2; h=0)
        @test_throws ArgumentError fourier_terms(48, [12, 4], [2])   # length mismatch
    end

    @testset "end to end: harmonic regression on a real seasonal series" begin
        # the case these exist for -- a seasonal pattern carried by a handful
        # of parameters instead of period-1 dummies, and extrapolable
        y = dataset("cardox").value[1:240]
        f = fourier_terms(length(y), 12, 2)
        m = fit_arimax(y, (1, 1, 0), f.terms; include_mean=false)
        @test m.converged
        @test length(m.beta) == 4
        @test all(isfinite, m.se)

        # and it forecasts, because the future regressors are known exactly
        fh = fourier_terms(length(y), 12, 2; h=12)
        fc = forecast(m, fh.terms, 12)
        @test length(fc.point) == 12
        @test all(isfinite, fc.point)
        @test issorted(fc.se)

        # more harmonics fit better in-sample, as they must with more parameters
        f4 = fourier_terms(length(y), 12, 4)
        m4 = fit_arimax(y, (1, 1, 0), f4.terms; include_mean=false)
        @test m4.loglik >= m.loglik - 1e-6
        @test length(m4.beta) == 8
    end
end
