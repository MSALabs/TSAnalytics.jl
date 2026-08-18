using DelimitedFiles

@testset "mstl_decompose matched-parameter agreement (exact statsmodels validation)" begin
    # handoff/stage-3.3-mstl-handoff.md, verified directly against
    # statsmodels.tsa.seasonal.MSTL's actual pure-Python source (not just
    # its docs) in this session. hourly_mstl.csv: n=500, daily (24) +
    # weekly (168) seasonality -- a scaled-down version of MSTL's own
    # docstring example, deliberately including a period (168) whose
    # cycle-subseries lengths (2-3) are NOT constant across phases, which
    # is what originally exposed a real indexing bug in stl_decompose
    # itself (see the "non-exact-multiple period" testset in test_stl.jl).
    y = vec(readdlm(TSAnalytics.HOURLY_MSTL, ','; skipstart=1, header=false))
    @test length(y) == 500

    r = mstl_decompose(y, [24, 168])
    @test r.periods == [24, 168]
    @test size(r.seasonal) == (500, 2)

    @test isapprox(r.trend[1:5],
                    [98.66100690955821, 98.68706573416274, 98.71312862554839,
                     98.73919562451476, 98.76526676572887]; atol=1e-6)
    @test isapprox(r.seasonal[1:5, 1],
                    [1.8387985105580222, 1.7198070473614895, 3.124145643755651,
                     4.966921234345688, 5.510167770617937]; atol=1e-6)
    @test isapprox(r.seasonal[1:5, 2],
                    [3.1553515072678917, 3.0226755759508843, 3.6907589890849475,
                     3.8009132092981326, 4.24600979878128]; atol=1e-6)
    @test isapprox(r.resid[1:5],
                    [-0.22299741314058963, 0.21830978675635038, 0.10678339268255854,
                     0.5560128117268022, 0.03735885809636841]; atol=1e-6)
    @test r.observed == y
    @test isapprox(r.observed, r.trend .+ vec(sum(r.seasonal; dims=2)) .+ r.resid; atol=1e-6)
    @test all(==(1.0), r.weights)  # no robust/outer requested anywhere in the chain
end

@testset "mstl_decompose single-period equivalence to stl_decompose (exact)" begin
    # num_seasons == 1 forces iterate=1 internally and is otherwise just
    # stl_decompose called once with the (possibly auto) window.
    y = vec(readdlm(TSAnalytics.HOURLY_MSTL, ','; skipstart=1, header=false))
    r = mstl_decompose(y, 24)
    r_direct = stl_decompose(y, 24; seasonal_window=11)  # default window = 7+4*1
    @test r.trend == r_direct.trend
    @test size(r.seasonal) == (500, 1)
    @test vec(r.seasonal[:, 1]) == r_direct.seasonal
    @test r.periods == [24]

    # iterate is silently ignored for a single period (matches Python's own
    # `iterate = 1 if num_seasons == 1 else self.iterate`, applied
    # unconditionally regardless of what the caller passed)
    r_it0 = mstl_decompose(y, 24; iterate=0)
    @test r_it0.trend == r_direct.trend
end

@testset "mstl_decompose explicit windows / iterate (exact statsmodels validation)" begin
    y = vec(readdlm(TSAnalytics.HOURLY_MSTL, ','; skipstart=1, header=false))

    r_w = mstl_decompose(y, [24, 168]; windows=[13, 21])
    @test isapprox(r_w.trend[1:3],
                    [98.66681344451823, 98.69279833979158, 98.71878691473187]; atol=1e-6)

    r_it3 = mstl_decompose(y, [24, 168]; iterate=3)
    @test isapprox(r_it3.trend[1:3],
                    [98.65835799990705, 98.68443982497571, 98.71052565255091]; atol=1e-6)

    # out-of-order periods with explicit paired windows: the window must
    # follow ITS OWN period through the ascending sort, not just be
    # positionally reassigned
    r_oo = mstl_decompose(y, [168, 24]; windows=[21, 13])
    @test r_oo.periods == [24, 168]
    @test isapprox(r_oo.trend, r_w.trend; atol=1e-10)

    # scalar windows: deliberate broadcast convenience, NOT matching Python
    # (which errors on a scalar `windows` with more than one period, per
    # `_process_windows`'s length check) -- self-consistency check instead
    # of an exact-value one, since there's no real-reference target for it
    r_scalar_w = mstl_decompose(y, [24, 168]; windows=15)
    r_explicit_w = mstl_decompose(y, [24, 168]; windows=[15, 15])
    @test r_scalar_w.trend == r_explicit_w.trend

    # stl_kwargs pass-through
    r_kw = mstl_decompose(y, [24, 168]; stl_kwargs=(seasonal_degree=0,))
    @test isapprox(r_kw.trend[1:3],
                    [99.50597394413556, 99.52415227543656, 99.54235694472843]; atol=1e-6)
end

@testset "mstl_decompose period-too-large dropped with warning (exact statsmodels validation)" begin
    y = vec(readdlm(TSAnalytics.HOURLY_MSTL, ','; skipstart=1, header=false))
    # period=400 >= 500/2=250 must be dropped (with a warning), not error;
    # matches real statsmodels exactly once dropped down to just period=24
    r = @test_logs (:warn,) mstl_decompose(y, [24, 400])
    @test r.periods == [24]
    @test size(r.seasonal) == (500, 1)
    @test isapprox(r.trend[1:3],
                    [101.81421767250018, 102.07735468579871, 102.3402104844111]; atol=1e-6)

    # boundary: period exactly n/2 is also dropped (>= comparison, not >)
    r_boundary = @test_logs (:warn,) mstl_decompose(y, [24, 250])
    @test r_boundary.periods == [24]

    # all periods too large -> nothing survives -> clear error, not a crash
    @test_throws ArgumentError mstl_decompose(y, [300, 400])
end

@testset "mstl_decompose lambda / Box-Cox (exact statsmodels validation)" begin
    y = vec(readdlm(TSAnalytics.HOURLY_MSTL, ','; skipstart=1, header=false))

    r_lam = mstl_decompose(y, [24, 168]; lambda=0.5)
    @test isapprox(r_lam.trend[1:3],
                    [17.848541486666853, 17.85114977514586, 17.85375848195093]; atol=1e-6)
    @test isapprox(r_lam.observed[1:3],
                    [18.34032050035038, 18.36151842513043, 18.5557599374065]; atol=1e-6)

    # lambda=0: genuinely applies log(x) here, unlike Python's own source
    # (elif self.lmbda: treats 0.0 as falsy, silently skipping the
    # transform -- confirmed by executing real statsmodels, not just
    # reading its source; this package deliberately does not reproduce
    # that as a considered choice, not an oversight)
    r_lam0 = mstl_decompose(y, [24, 168]; lambda=0)
    r_nolam = mstl_decompose(y, [24, 168])
    @test isapprox(r_lam0.observed, log.(y); atol=1e-10)
    @test r_lam0.trend != r_nolam.trend

    # error paths
    @test_throws ArgumentError mstl_decompose(y, [24, 168]; lambda=:auto)
    @test_throws ArgumentError mstl_decompose([1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2; lambda=1.0)
    @test_throws ArgumentError mstl_decompose(y, [24, 168]; lambda=:not_auto)  # only :auto is a valid Symbol
end

@testset "mstl_decompose parallel keyword" begin
    y = vec(readdlm(TSAnalytics.HOURLY_MSTL, ','; skipstart=1, header=false))
    r_par = mstl_decompose(y, [24, 168])
    r_seq = mstl_decompose(y, [24, 168]; parallel=false)
    @test r_par.trend == r_seq.trend
    @test r_par.seasonal == r_seq.seasonal
end

@testset "mstl_decompose parameter coverage and error paths" begin
    y = vec(readdlm(TSAnalytics.HOURLY_MSTL, ','; skipstart=1, header=false))

    # periods as a Vector vs a Tuple vs a UnitRange (any integer iterable)
    r_vec = mstl_decompose(y, [24, 168])
    r_range = mstl_decompose(y, 24:144:168)  # 24, 168
    @test r_vec.trend == r_range.trend

    # every scalar-period call must equal the standalone STL call
    for p in (12, 24, 48)
        r = mstl_decompose(y, p)
        r_direct = stl_decompose(y, p; seasonal_window=11)
        @test r.trend == r_direct.trend
    end

    # every iterate value on a genuine multi-period case
    for it in (1, 2, 3, 5)
        r = mstl_decompose(y, [24, 168]; iterate=it)
        @test all(isfinite, r.trend)
    end

    # container-agnostic
    r1 = mstl_decompose(y, [24, 168])
    r2 = mstl_decompose(Float32.(y), [24, 168])
    @test isapprox(r1.trend, r2.trend; atol=1e-2)

    # error paths
    @test_throws ArgumentError mstl_decompose(y, Int[])                    # empty periods
    @test_throws ArgumentError mstl_decompose(y, 1)                        # period < 2
    @test_throws ArgumentError mstl_decompose(y, [24, 1])                  # one period < 2
    @test_throws ArgumentError mstl_decompose(y, [24, 168]; windows=[13])  # length mismatch
    @test_throws ArgumentError mstl_decompose(y, [24, 168]; iterate=0)     # iterate < 1, multi-period
    y_nan = copy(y); y_nan[5] = NaN
    @test_throws ArgumentError mstl_decompose(y_nan, [24, 168])

    # window validation errors surface naturally from the underlying
    # stl_decompose call (even window, too-small window, etc.)
    @test_throws ArgumentError mstl_decompose(y, [24, 168]; windows=[12, 21])  # even
end
