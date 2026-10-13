using DelimitedFiles

# Stage 10.4b: the Phillips-Ouliaris Pu and Pz cointegration tests.
#
# statsmodels has no Phillips-Ouliaris test, so urca::ca.po is the only
# software reference. The critical values are transcribed from Phillips &
# Ouliaris (1990), Appendix B, and checked against ca.po. Pu, and Pz without
# deterministic terms, match ca.po to 1e-10. Pz with a constant or trend
# follows the paper's demeaned M_zz (§3 remark (f)) where ca.po uses raw z'z;
# that divergence is pinned below three ways: ca.po's number is reproduced
# by the raw-M_zz variant, the paper's statistic is invariant to a level
# shift where ca.po's is not, and the paper's has close to nominal size where
# ca.po's almost never rejects.

if !isdefined(@__MODULE__, :REF_R)
    include(joinpath(@__DIR__, "fixtures", "var", "refs.jl"))
end

_pfix(name) = readdlm(joinpath(@__DIR__, "fixtures", "var", name), ','; skipstart=1)
const _PO_FIX = [("po_yes", "coint2_yes.csv"), ("po_no", "coint2_no.csv"),
                 ("po_r1", "coint3_r1.csv"), ("po_trend", "coint2_trend.csv")]
const _PO_DN = Dict(:none => "none", :const => "constant", :trend => "trend")

# ca.po's Pz: the same Ω, but M_zz from the raw z'z whatever the deterministic case
function _urca_pz(z, det)
    n, m = size(z); T = n - 1
    l = trunc(Int, 4 * (T / 100)^0.25)
    D = det === :none ? zeros(T, 0) : det === :const ? ones(T, 1) : hcat(ones(T), 1.0:T)
    X = hcat(z[1:T, :], D)
    xi = z[2:n, :] - X * (X \ z[2:n, :])
    O = xi' * xi ./ T
    for s in 1:l
        G = xi[(s+1):T, :]' * xi[1:(T-s), :]
        O .+= (1 - s / (l + 1)) .* (G + G') ./ T
    end
    zl = z[2:n, :]
    return T * tr(O / (zl' * zl ./ T))
end

@testset "Pu and Pz(none) vs urca::ca.po" begin
    for (key, fx) in _PO_FIX
        R = REF_R_COINT[key]
        Y = _pfix(fx)
        for kind in (:Pu, :Pz), det in (:none, :const, :trend)
            ref = R["capo_$(kind)_$(_PO_DN[det])"]
            t = phillips_ouliaris_test(Y; kind=kind, deterministic=det)
            # the critical values, transcribed from the paper, equal urca's
            @test collect(t.critical_values.values) == Float64.(ref["cval"])
            @test t.critical_values.levels == (10.0, 5.0, 1.0)      # ASCENDING, unlike MacKinnon
            @test t.m == size(Y, 2) == R["m"]
            @test t.kind === kind && t.deterministic === det
            @test t.bandwidth == 5                                  # urca's "short" rule at n = 250
            if kind === :Pu || det === :none
                @test t.statistic ≈ ref["stat"] rtol=1e-10
            else
                # the documented divergence: urca's number is the raw-M_zz variant
                @test _urca_pz(Float64.(Y), det) ≈ ref["stat"] rtol=1e-10
                @test !isapprox(t.statistic, ref["stat"]; rtol=1e-2)
            end
        end
    end
end

@testset "Pz with a deterministic term: the paper's definition, by hand" begin
    Y = Float64.(_pfix("coint3_r1.csv"))
    n, m = size(Y); T = n - 1
    for det in (:const, :trend)
        D = det === :const ? ones(T, 1) : hcat(ones(T), 1.0:T)
        zl = Y[2:n, :]
        X = hcat(Y[1:T, :], D)
        xi = zl - X * (X \ zl)
        l = 5
        O = xi' * xi ./ T
        for s in 1:l
            G = xi[(s+1):T, :]' * xi[1:(T-s), :]
            O .+= (1 - s / (l + 1)) .* (G + G') ./ T
        end
        Zm = zl - D * (D \ zl)                         # demeaned / detrended
        @test phillips_ouliaris_test(Y; kind=:Pz, deterministic=det).statistic ≈
              T * tr((Zm' * Zm ./ T) \ O) rtol=1e-12
    end
end

@testset "invariance: the paper's statistic, not urca's" begin
    Y = Float64.(_pfix("coint2_yes.csv"))
    for det in (:const, :trend)
        a = phillips_ouliaris_test(Y; kind=:Pz, deterministic=det).statistic
        b = phillips_ouliaris_test(Y .+ 1000; kind=:Pz, deterministic=det).statistic
        c = phillips_ouliaris_test(Y .+ [5.0 -300.0]; kind=:Pz, deterministic=det).statistic
        @test a ≈ b rtol=1e-8
        @test a ≈ c rtol=1e-8
        # urca's changes by nearly half under the same shift
        @test _urca_pz(Y .+ 1000, det) < 0.6 * _urca_pz(Y, det)
    end
    # a linear trend added to every series leaves the detrended statistic alone
    tr_ = 0.3 .* (1:250)
    for kind in (:Pu, :Pz)
        a = phillips_ouliaris_test(Y; kind=kind, deterministic=:trend).statistic
        b = phillips_ouliaris_test(Y .+ tr_ .* [1.0 2.0]; kind=kind, deterministic=:trend).statistic
        @test a ≈ b rtol=1e-6
    end
    # Pz is normalization-invariant: reordering the series changes nothing.
    # Pu is not: it singles out the first series.
    for det in (:none, :const, :trend)
        @test phillips_ouliaris_test(Y; deterministic=det).statistic ≈
              phillips_ouliaris_test(Y[:, [2, 1]]; deterministic=det).statistic rtol=1e-10
        @test !isapprox(phillips_ouliaris_test(Y; kind=:Pu, deterministic=det).statistic,
                        phillips_ouliaris_test(Y[:, [2, 1]]; kind=:Pu, deterministic=det).statistic;
                        rtol=1e-3)
    end
    # and scale-invariant, each series in its own units
    @test phillips_ouliaris_test(Y .* [100.0 0.01]).statistic ≈ phillips_ouliaris_test(Y).statistic rtol=1e-9
end

if get(ENV, "TSANALYTICS_FULL_TESTS", "1") == "1"
    @testset "size under the null: close to nominal (urca's Pz is not)" begin
        # 4000 replications at n = 500 gave 4.3% (const) and 4.2% (trend) for
        # the paper's Pz against 0.25% and 0% for urca's. 1000 here.
        rng = MersenneTwister(1)
        R = 1000
        for (det, cv) in ((:const, 55.2202), (:trend, 81.3812))
            mine = 0; theirs = 0
            for _ in 1:R
                x = cumsum(randn(rng, 500, 2); dims=1) .+ 10 .* randn(rng, 1, 2)
                mine += phillips_ouliaris_test(x; deterministic=det).statistic > cv
                theirs += _urca_pz(x, det) > cv
            end
            @test 0.025 <= mine / R <= 0.075
            @test theirs / R < 0.01
        end
        # Pu: the paper's Table IIIb/IIIc, likewise
        for (det, cv) in ((:const, 33.7130), (:trend, 48.8439))
            hits = 0
            for _ in 1:R
                x = cumsum(randn(rng, 500, 2); dims=1)
                hits += phillips_ouliaris_test(x; kind=:Pu, deterministic=det).statistic > cv
            end
            @test 0.025 <= hits / R <= 0.075
        end
    end
end

@testset "size and power on the fixtures" begin
    for kind in (:Pu, :Pz), det in (:none, :const, :trend)
        @test reject(phillips_ouliaris_test(_pfix("coint2_yes.csv"); kind=kind, deterministic=det); level=1.0)
        @test reject(phillips_ouliaris_test(_pfix("coint3_r1.csv"); kind=kind, deterministic=det); level=1.0)
        @test !reject(phillips_ouliaris_test(_pfix("coint2_no.csv"); kind=kind, deterministic=det); level=10.0)
    end
    # coint3_r2 has two relations
    @test reject(phillips_ouliaris_test(_pfix("coint3_r2.csv")); level=1.0)
    # coint2_trend's relation contains a trend: the paper's Pz finds it with a
    # constant alone (76.68 > 71.93, the 1% point); urca's (21.16) misses it
    t = phillips_ouliaris_test(_pfix("coint2_trend.csv"); kind=:Pz, deterministic=:const)
    @test t.statistic ≈ 76.678328 atol=1e-5
    @test reject(t; level=1.0)
    @test REF_R_COINT["po_trend"]["capo_Pz_constant"]["stat"] < t.critical_values.values[1]
end

@testset "the deterministic choice is not cosmetic" begin
    Y = _pfix("coint2_trend.csv")
    none  = phillips_ouliaris_test(Y; kind=:Pu, deterministic=:none)
    const_ = phillips_ouliaris_test(Y; kind=:Pu, deterministic=:const)
    trend = phillips_ouliaris_test(Y; kind=:Pu, deterministic=:trend)
    @test !reject(none; level=10.0)            #   3.19 vs 20.39
    @test reject(const_; level=5.0)            #  50.24 vs 33.71
    @test reject(const_; level=1.0)            #  50.24 vs 48.00 -- also at 1%
    @test reject(trend; level=1.0)             # 170.02 vs 65.17
    @test trend.statistic > 50 * none.statistic
end

@testset "bandwidth: :short, :long and explicit" begin
    Y = _pfix("coint2_yes.csv")
    s = phillips_ouliaris_test(Y; bandwidth=:short)
    l = phillips_ouliaris_test(Y; bandwidth=:long)
    @test s.bandwidth == trunc(Int, 4 * (249 / 100)^0.25) == 5
    @test l.bandwidth == trunc(Int, 12 * (249 / 100)^0.25) == 15
    @test s.statistic != l.statistic
    @test phillips_ouliaris_test(Y; bandwidth=5).statistic == s.statistic
    z = phillips_ouliaris_test(Y; bandwidth=0)
    @test z.bandwidth == 0 && isfinite(z.statistic)
    # bandwidth 0 is the plain covariance of the VAR residuals
    zz = Float64.(Y); T = 249
    xi = zz[2:end, :] - hcat(zz[1:T, :], ones(T)) * (hcat(zz[1:T, :], ones(T)) \ zz[2:end, :])
    Zm = zz[2:end, :] .- sum(zz[2:end, :]; dims=1) ./ T
    @test z.statistic ≈ T * tr((Zm' * Zm) \ (xi' * xi)) rtol=1e-10
    @test_throws ArgumentError phillips_ouliaris_test(Y; bandwidth=-1)
    @test_throws ArgumentError phillips_ouliaris_test(Y; bandwidth=250)
    @test_throws ArgumentError phillips_ouliaris_test(Y; bandwidth=:bogus)
end

@testset "reuses the Bartlett long-run variance idea, consistently" begin
    # for one residual series the Ω here is _bartlett_lrv's estimator
    u = randn(MersenneTwister(3), 200)
    @test TSAnalytics._bartlett_lrv(u, 0) ≈ sum(abs2, u) / 200 atol=1e-12
    @test TSAnalytics._bartlett_lrv(u, 4) != TSAnalytics._bartlett_lrv(u, 0)
end

@testset "coverage: m = 2..6 only, and no p-value" begin
    Y7 = cumsum(randn(MersenneTwister(7), 250, 7); dims=1)
    @test_throws ArgumentError phillips_ouliaris_test(Y7)
    e = try phillips_ouliaris_test(Y7) catch err; err end
    @test occursin("six", lowercase(sprint(showerror, e)))
    for m in 2:6
        t = phillips_ouliaris_test(Y7[:, 1:m]; kind=:Pz, deterministic=:const)
        @test t.m == m
        @test t.critical_values.values == (TSAnalytics._PO_TABLES[(:Pz, :const)][m - 1][3],
                                           TSAnalytics._PO_TABLES[(:Pz, :const)][m - 1][5],
                                           TSAnalytics._PO_TABLES[(:Pz, :const)][m - 1][7])
    end
    @test_throws ArgumentError phillips_ouliaris_test(Y7[:, 1:1])
    @test_throws ArgumentError phillips_ouliaris_test(Y7[:, 1])

    t = phillips_ouliaris_test(_pfix("coint2_yes.csv"))
    @test !t.pvalue_available
    @test !hasfield(PhillipsOuliarisTest, :pvalue)            # nothing to fake
    @test_throws ArgumentError pvalue(t)
    e = try pvalue(t) catch err; err end
    @test occursin("critical_values", sprint(showerror, e))
    # tseries::po.test reports its table bounds as p-values
    @test REF_R_COINT["po_yes"]["po_demean1_lshort"]["p"] == 0.01
    @test REF_R_COINT["po_no"]["po_demean1_lshort"]["p"] == 0.15
end

@testset "transcribed tables: shape and monotonicity" begin
    for kind in (:Pu, :Pz), det in (:none, :const, :trend)
        tab = TSAnalytics._PO_TABLES[(kind, det)]
        @test length(tab) == 5
        for row in tab
            @test length(row) == 7 && issorted(collect(row))     # larger quantile at smaller size
        end
        for c in 1:7
            @test issorted([tab[i][c] for i in 1:5])            # grows with the number of regressors
        end
    end
    # more deterministic terms push every quantile up
    for kind in (:Pu, :Pz), i in 1:5, c in 1:7
        @test TSAnalytics._PO_TABLES[(kind, :none)][i][c] < TSAnalytics._PO_TABLES[(kind, :const)][i][c] <
              TSAnalytics._PO_TABLES[(kind, :trend)][i][c]
    end
end

@testset "refusals" begin
    Y = _pfix("coint2_yes.csv")
    @test_throws ArgumentError phillips_ouliaris_test(Y; kind=:Za)
    @test_throws ArgumentError phillips_ouliaris_test(Y; deterministic=:quadratic)
    @test_throws ArgumentError phillips_ouliaris_test(Y; deterministic=:bogus)
    Yn = copy(Y); Yn[3, 2] = NaN
    @test_throws ArgumentError phillips_ouliaris_test(Yn)
    @test_throws ArgumentError phillips_ouliaris_test(Y[1:6, :])
end

@testset "reject, contract, show, type stability" begin
    t = phillips_ouliaris_test(_pfix("coint3_r1.csv"); kind=:Pu, deterministic=:const)
    @test t isa HypothesisTest
    @test statistic(t) === t.statistic
    @test critical_values(t) === t.critical_values
    @test isconcretetype(PhillipsOuliarisTest)
    @test (@inferred phillips_ouliaris_test(_pfix("coint2_yes.csv"))) isa PhillipsOuliarisTest
    s = sprint(show, t)
    @test occursin("Phillips", s) && occursin("Pu", s)
    @test occursin("constant", lowercase(s))
    @test occursin("no p-value", s)
    @test reject(t; level=5.0) isa Bool
    for lv in (10.0, 5.0, 1.0, 10, 5, 1)
        @test reject(t; level=lv) == (t.statistic > t.critical_values.values[findfirst(==(Float64(lv)), (10.0, 5.0, 1.0))])
    end
    @test_throws ArgumentError reject(t; level=2.5)
    # reject reads levels by value: for the left-tailed Engle-Granger test the
    # same call compares the other way
    eg = coint_test(_pfix("coint2_yes.csv"); lags=4, autolag=nothing)
    @test reject(eg; level=5.0) == (eg.statistic < eg.critical_values.values[2])
    @test reject(eg; level=1.0) && reject(t; level=1.0)
    @test_throws ArgumentError reject(eg; level=2.5)
end
