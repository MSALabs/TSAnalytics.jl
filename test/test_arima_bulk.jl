# test/test_arima_bulk.jl
#
# GENERATED FILE -- do not hand-edit. Produced by
# test/verification/arima/bulk/gen_arima_cases.py + fit_r.R + fit_python.py
# (see test/verification/arima/bulk/gen_julia_test.py for the merge/emit step).
#
# 20 synthetic ARIMA(p,d,q) series, each dual-verified: R's
# arima(order=c(p,d,q), include.mean=FALSE, method="ML") and Python's
# ARIMA(order=(p,d,q), trend="n").fit() both fit the exact same data.
# Sweeps d in {0,1,2} across (p,q) in {(1,0),(0,1),(1,1),(2,0),(0,2),(2,1),(0,0)},
# each on its own independently-generated synthetic series (known true
# AR/MA structure, integrated d times for d>0), per
# handoff/stage-6.6-arima-handoff.md section 8's request for a systematic
# sweep beyond the handoff's own two spot-check cases.
#
# Tolerances are loose (1e-2) relative to Stage 6.5's single-case tests --
# these are 20 independent ML fits via two different optimizers (Julia's
# LBFGS vs R's BFGS / Python's L-BFGS-B), not fixed-coefficient likelihood
# evaluation, so small convergence-path differences are expected and normal,
# not a bug -- same reasoning as the single ARMA(1,1) case in
# test_arma.jl. loglik/aic are checked tighter (1e-1) since all three
# implementations are climbing the same likelihood surface regardless of
# path, so should land close to the same peak.
#
# Regenerate with (from test/verification/arima/bulk/):
#   python gen_arima_cases.py
#   python fit_python.py
#   Rscript fit_r.R   # or the full path to Rscript.exe
#   python gen_julia_test.py

using DelimitedFiles

if get(ENV, "TSANALYTICS_FULL_TESTS", "1") == "1"
@testset "fit_arima bulk -- dual-verified R+Python synthetic sweep" begin
    let name = "pq10_d0"  # case 1: order=(1,0,0), n=120
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq10_d0.csv"), ',', skipstart=1))
        m = fit_arima(y, (1, 0, 0); include_mean=false)
        @test m.arma.converged
        @test m.d == 0
        @test isapprox(m.arma.ar, [0.4868550967097316]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [0.48685795910410007]; atol=1e-2)  # Python
        @test isempty(m.arma.ma)
        @test isapprox(m.arma.loglik, -177.4565025402132; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -177.45650254195118; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 358.9130050804264; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 358.91300508390236; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 120  # R's n-d convention (Python reports 120 = full n instead)
        @test isapprox(m.arma.se, [0.07997530115924623]; atol=5e-2)  # R hessian se
    end

    let name = "pq10_d0_b"  # case 2: order=(1,0,0), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq10_d0_b.csv"), ',', skipstart=1))
        m = fit_arima(y, (1, 0, 0); include_mean=false)
        @test m.arma.converged
        @test m.d == 0
        @test isapprox(m.arma.ar, [-0.3274077783300125]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [-0.32743531652318897]; atol=1e-2)  # Python
        @test isempty(m.arma.ma)
        @test isapprox(m.arma.loglik, -213.84613088387454; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -213.84613094544352; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 431.6922617677491; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 431.69226189088704; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.07716512211422183]; atol=5e-2)  # R hessian se
    end

    let name = "pq01_d0"  # case 3: order=(0,0,1), n=120
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq01_d0.csv"), ',', skipstart=1))
        m = fit_arima(y, (0, 0, 1); include_mean=false)
        @test m.arma.converged
        @test m.d == 0
        @test isempty(m.arma.ar)
        @test isapprox(m.arma.ma, [0.5680084806926103]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.5680061142493832]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -163.8593341608207; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -163.85933416593747; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 331.7186683216414; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 331.71866833187494; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 120  # R's n-d convention (Python reports 120 = full n instead)
        @test isapprox(m.arma.se, [0.07202515814434464]; atol=5e-2)  # R hessian se
    end

    let name = "pq01_d0_b"  # case 4: order=(0,0,1), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq01_d0_b.csv"), ',', skipstart=1))
        m = fit_arima(y, (0, 0, 1); include_mean=false)
        @test m.arma.converged
        @test m.d == 0
        @test isempty(m.arma.ar)
        @test isapprox(m.arma.ma, [-0.2062797639868189]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [-0.2062845478361979]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -200.98124953219985; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -200.9812495376914; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 405.9624990643997; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 405.9624990753828; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.08121191194458827]; atol=5e-2)  # R hessian se
    end

    let name = "pq11_d0"  # case 5: order=(1,0,1), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq11_d0.csv"), ',', skipstart=1))
        m = fit_arima(y, (1, 0, 1); include_mean=false)
        @test m.arma.converged
        @test m.d == 0
        @test isapprox(m.arma.ar, [0.4891492494840707]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [0.4891676057513263]; atol=1e-2)  # Python
        @test isapprox(m.arma.ma, [0.3387035586159827]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.33869411772573155]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -223.65490722893685; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -223.65490726515117; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 453.3098144578737; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 453.30981453030233; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.100432229424865, 0.108553945310646]; atol=5e-2)  # R hessian se
    end

    let name = "pq11_d0_b"  # case 6: order=(1,0,1), n=180
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq11_d0_b.csv"), ',', skipstart=1))
        m = fit_arima(y, (1, 0, 1); include_mean=false)
        @test m.arma.converged
        @test m.d == 0
        @test isapprox(m.arma.ar, [-0.0347587286080371]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [-0.03745205830961339]; atol=1e-2)  # Python
        @test isapprox(m.arma.ma, [0.1492220912628889]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.151878931306233]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -255.94624356506498; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -255.9462540796966; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 517.8924871301299; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 517.8925081593932; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 180  # R's n-d convention (Python reports 180 = full n instead)
        @test isapprox(m.arma.se, [0.5495181705646776, 0.5421305607154411]; atol=5e-2)  # R hessian se
    end

    let name = "pq20_d0"  # case 7: order=(2,0,0), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq20_d0.csv"), ',', skipstart=1))
        m = fit_arima(y, (2, 0, 0); include_mean=false)
        @test m.arma.converged
        @test m.d == 0
        @test isapprox(m.arma.ar, [0.4560251519284749, -0.272494809414053]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [0.45603290275225705, -0.27249054179773163]; atol=1e-2)  # Python
        @test isempty(m.arma.ma)
        @test isapprox(m.arma.loglik, -214.5743393921874; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -214.5743394052865; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 435.1486787843748; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 435.148678810573; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.07905238706393329, 0.0790162688943541]; atol=5e-2)  # R hessian se
    end

    let name = "pq02_d0"  # case 8: order=(0,0,2), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq02_d0.csv"), ',', skipstart=1))
        m = fit_arima(y, (0, 0, 2); include_mean=false)
        @test m.arma.converged
        @test m.d == 0
        @test isempty(m.arma.ar)
        @test isapprox(m.arma.ma, [0.4627030618337736, 0.1644588585430053]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.4626979203080751, 0.1644549102008541]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -219.07731193128473; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -219.07731193781615; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 444.15462386256945; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 444.1546238756323; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.07967125311345745, 0.08098644229367788]; atol=5e-2)  # R hessian se
    end

    let name = "pq21_d0"  # case 9: order=(2,0,1), n=180
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq21_d0.csv"), ',', skipstart=1))
        m = fit_arima(y, (2, 0, 1); include_mean=false)
        @test m.arma.converged
        @test m.d == 0
        @test isapprox(m.arma.ar, [0.2694886506960071, -0.02739158100976471]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [0.2695152407507982, -0.027401045250696268]; atol=1e-2)  # Python
        @test isapprox(m.arma.ma, [0.4433734350816832]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.4433507967950242]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -244.29386316636047; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -244.2938631883614; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 496.58772633272093; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 496.5877263767228; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 180  # R's n-d convention (Python reports 180 = full n instead)
        @test isapprox(m.arma.se, [0.1789854538554814, 0.1276724947046609, 0.1638734512957256]; atol=5e-2)  # R hessian se
    end

    let name = "pq10_d1"  # case 10: order=(1,1,0), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq10_d1.csv"), ',', skipstart=1))
        m = fit_arima(y, (1, 1, 0); include_mean=false)
        @test m.arma.converged
        @test m.d == 1
        @test isapprox(m.arma.ar, [0.5489765133331641]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [0.5489801565088491]; atol=1e-2)  # Python
        @test isempty(m.arma.ma)
        @test isapprox(m.arma.loglik, -212.29412479737522; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -212.2941248035819; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 428.58824959475044; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 428.5882496071638; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 149  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.06833048758065492]; atol=5e-2)  # R hessian se
    end

    let name = "pq01_d1"  # case 11: order=(0,1,1), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq01_d1.csv"), ',', skipstart=1))
        m = fit_arima(y, (0, 1, 1); include_mean=false)
        @test m.arma.converged
        @test m.d == 1
        @test isempty(m.arma.ar)
        @test isapprox(m.arma.ma, [0.4538591558549186]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.4538553955630907]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -218.00789246230238; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -218.0078924392701; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 440.01578492460476; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 440.0157848785402; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 149  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.07513691009322018]; atol=5e-2)  # R hessian se
    end

    let name = "pq11_d1"  # case 12: order=(1,1,1), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq11_d1.csv"), ',', skipstart=1))
        m = fit_arima(y, (1, 1, 1); include_mean=false)
        @test m.arma.converged
        @test m.d == 1
        @test isapprox(m.arma.ar, [0.4451937033896559]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [0.44519349338594905]; atol=1e-2)  # Python
        @test isapprox(m.arma.ma, [0.26231628014348]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.2623151803008611]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -198.94805201761318; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -198.94805207170003; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 403.89610403522636; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 403.89610414340007; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 149  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.1075149288270925, 0.1086407238226716]; atol=5e-2)  # R hessian se
    end

    let name = "pq11_d1_b"  # case 13: order=(1,1,1), n=180
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq11_d1_b.csv"), ',', skipstart=1))
        m = fit_arima(y, (1, 1, 1); include_mean=false)
        @test m.arma.converged
        @test m.d == 1
        @test isapprox(m.arma.ar, [-0.3455839100958125]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [-0.34576598673073805]; atol=1e-2)  # Python
        @test isapprox(m.arma.ma, [0.1471810399252858]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.14736365188556436]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -258.83303078195894; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -258.83303024223864; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 523.6660615639179; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 523.6660604844773; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 179  # R's n-d convention (Python reports 180 = full n instead)
        @test isapprox(m.arma.se, [0.2433391699038189, 0.2508689450937936]; atol=5e-2)  # R hessian se
    end

    let name = "pq20_d1"  # case 14: order=(2,1,0), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq20_d1.csv"), ',', skipstart=1))
        m = fit_arima(y, (2, 1, 0); include_mean=false)
        @test m.arma.converged
        @test m.d == 1
        @test isapprox(m.arma.ar, [0.4081451082195128, -0.2610883401530829]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [0.40815515188427803, -0.2610811452667186]; atol=1e-2)  # Python
        @test isempty(m.arma.ma)
        @test isapprox(m.arma.loglik, -201.804062966329; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -201.80406302631513; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 409.608125932658; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 409.60812605263027; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 149  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.07920122006921605, 0.07947387125522486]; atol=5e-2)  # R hessian se
    end

    let name = "pq02_d1"  # case 15: order=(0,1,2), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq02_d1.csv"), ',', skipstart=1))
        m = fit_arima(y, (0, 1, 2); include_mean=false)
        @test m.arma.converged
        @test m.d == 1
        @test isempty(m.arma.ar)
        @test isapprox(m.arma.ma, [0.4905196753580148, 0.2840926791260475]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.4905116758270108, 0.2840881325143392]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -205.55085350264562; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -205.55085352307634; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 417.10170700529125; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 417.1017070461527; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 149  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.0833256510309645, 0.07652428200745898]; atol=5e-2)  # R hessian se
    end

    let name = "pq00_d1"  # case 16: order=(0,1,0), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq00_d1.csv"), ',', skipstart=1))
        m = fit_arima(y, (0, 1, 0); include_mean=false)
        @test m.arma.converged
        @test m.d == 1
        @test isempty(m.arma.ar)
        @test isempty(m.arma.ma)
        @test isapprox(m.arma.loglik, -215.88714952910732; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -215.88714953282067; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 433.77429905821464; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 433.77429906564134; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 149  # R's n-d convention (Python reports 150 = full n instead)
    end

    let name = "pq10_d2"  # case 17: order=(1,2,0), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq10_d2.csv"), ',', skipstart=1))
        m = fit_arima(y, (1, 2, 0); include_mean=false)
        @test m.arma.converged
        @test m.d == 2
        @test isapprox(m.arma.ar, [0.3684065926169515]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [0.36841028465282627]; atol=1e-2)  # Python
        @test isempty(m.arma.ma)
        @test isapprox(m.arma.loglik, -210.26209785747392; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -210.26209696323832; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 424.52419571494784; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 424.52419392647664; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 148  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.07638705856057258]; atol=5e-2)  # R hessian se
    end

    let name = "pq11_d2"  # case 18: order=(1,2,1), n=180
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq11_d2.csv"), ',', skipstart=1))
        m = fit_arima(y, (1, 2, 1); include_mean=false)
        @test m.arma.converged
        @test m.d == 2
        @test isapprox(m.arma.ar, [0.3919078058710712]; atol=1e-2)  # R
        @test isapprox(m.arma.ar, [0.39192949123013987]; atol=1e-2)  # Python
        @test isapprox(m.arma.ma, [0.3482192449138091]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.34821831138428394]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -222.89073036433908; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -222.89073238918118; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 451.78146072867816; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 451.78146477836236; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 178  # R's n-d convention (Python reports 180 = full n instead)
        @test isapprox(m.arma.se, [0.109025212710742, 0.1136932921289018]; atol=5e-2)  # R hessian se
    end

    let name = "pq01_d2"  # case 19: order=(0,2,1), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq01_d2.csv"), ',', skipstart=1))
        m = fit_arima(y, (0, 2, 1); include_mean=false)
        @test m.arma.converged
        @test m.d == 2
        @test isempty(m.arma.ar)
        @test isapprox(m.arma.ma, [0.3266275564620834]; atol=1e-2)  # R
        @test isapprox(m.arma.ma, [0.3266231301873741]; atol=1e-2)  # Python
        @test isapprox(m.arma.loglik, -207.23695744377363; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -207.2369556988358; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 418.47391488754727; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 418.4739113976716; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 148  # R's n-d convention (Python reports 150 = full n instead)
        @test isapprox(m.arma.se, [0.09016078071256972]; atol=5e-2)  # R hessian se
    end

    let name = "pq00_d2"  # case 20: order=(0,2,0), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "arima", "bulk", "data", "pq00_d2.csv"), ',', skipstart=1))
        m = fit_arima(y, (0, 2, 0); include_mean=false)
        @test m.arma.converged
        @test m.d == 2
        @test isempty(m.arma.ar)
        @test isempty(m.arma.ma)
        @test isapprox(m.arma.loglik, -214.29641136144062; atol=1e-1)  # R
        @test isapprox(m.arma.loglik, -214.2964113657012; atol=1e-1)  # Python
        @test isapprox(m.arma.aic, 430.59282272288124; atol=1e-1)  # R
        @test isapprox(m.arma.aic, 430.5928227314024; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 148  # R's n-d convention (Python reports 150 = full n instead)
    end

end
end # if TSANALYTICS_FULL_TESTS
