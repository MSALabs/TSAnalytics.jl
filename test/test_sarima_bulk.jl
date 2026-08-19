# test/test_sarima_bulk.jl
#
# GENERATED FILE -- do not hand-edit. Produced by
# test/verification/sarima/bulk/gen_sarima_cases.py + fit_r.R + fit_python.py
# (see test/verification/sarima/bulk/gen_julia_test.py for the merge/emit step).
#
# 12 synthetic SARIMA(p,d,q)(P,D,Q)_s series, each dual-verified: R's
# arima(order=c(p,d,q), seasonal=list(order=c(P,D,Q),period=s),
# include.mean=FALSE, method="ML") and Python's
# SARIMAX(order=(p,d,q), seasonal_order=(P,D,Q,s), trend="n").fit() both fit
# the exact same data. Sweeps every polynomial block in isolation and in
# combination (including the handoff's own explicitly-flagged highest-risk
# gap: a case with p,q,P,Q all > 0, exercising the theta/Theta
# parameter-unpacking indices no single-seasonal-term case can reach),
# d/D=0 and d/D>0, and two seasonal periods (s=4, s=12).
#
# Tolerances loose (1e-2) for coefficients / (1e-1) for loglik/aic, same
# reasoning as test_arima_bulk.jl: independent ML fits via different
# optimizers, not fixed-coefficient evaluation.
#
# Regenerate with (from test/verification/sarima/bulk/):
#   python gen_sarima_cases.py
#   python fit_python.py
#   Rscript fit_r.R   # or the full path to Rscript.exe
#   python gen_julia_test.py

using DelimitedFiles, StatsAPI

if get(ENV, "TSANALYTICS_FULL_TESTS", "1") == "1"
@testset "fit_sarima bulk -- dual-verified R+Python synthetic sweep" begin
    let name = "p_only"  # case 1: order=(1,0,0), seasonal_order=(0,0,0,4), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "p_only.csv"), ',', skipstart=1))
        m = fit_sarima(y, (1, 0, 0), (0, 0, 0, 4); include_mean=false)
        @test m.converged
        @test isapprox(m.phi, [0.4725088206792408]; atol=1e-2)  # R
        @test isapprox(m.phi, [0.4724865669807342]; atol=1e-2)  # Python
        @test isempty(m.theta)
        @test isempty(m.Phi)
        @test isempty(m.Theta)
        @test isapprox(m.loglik, -205.54572458227545; atol=1e-1)  # R
        @test isapprox(m.loglik, -205.54572463162822; atol=1e-1)  # Python
        @test isapprox(m.aic, 415.0914491645509; atol=1e-1)  # R
        @test isapprox(m.aic, 415.09144926325644; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d-D*s convention (Python reports 150)
        @test isapprox(m.se, [0.07291409670873836]; atol=5e-2)  # R hessian se
    end

    let name = "q_only"  # case 2: order=(0,0,1), seasonal_order=(0,0,0,4), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "q_only.csv"), ',', skipstart=1))
        m = fit_sarima(y, (0, 0, 1), (0, 0, 0, 4); include_mean=false)
        @test m.converged
        @test isempty(m.phi)
        @test isapprox(m.theta, [0.3611686406901319]; atol=1e-2)  # R
        @test isapprox(m.theta, [0.36116080228528236]; atol=1e-2)  # Python
        @test isempty(m.Phi)
        @test isempty(m.Theta)
        @test isapprox(m.loglik, -192.2867878926308; atol=1e-1)  # R
        @test isapprox(m.loglik, -192.28678789170203; atol=1e-1)  # Python
        @test isapprox(m.aic, 388.5735757852616; atol=1e-1)  # R
        @test isapprox(m.aic, 388.57357578340407; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d-D*s convention (Python reports 150)
        @test isapprox(m.se, [0.07061060121525115]; atol=5e-2)  # R hessian se
    end

    let name = "seasar_only"  # case 3: order=(0,0,0), seasonal_order=(1,0,0,4), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "seasar_only.csv"), ',', skipstart=1))
        m = fit_sarima(y, (0, 0, 0), (1, 0, 0, 4); include_mean=false)
        @test m.converged
        @test isempty(m.phi)
        @test isempty(m.theta)
        @test isapprox(m.Phi, [0.5064935170172242]; atol=1e-2)  # R
        @test isapprox(m.Phi, [0.5064951398162743]; atol=1e-2)  # Python
        @test isempty(m.Theta)
        @test isapprox(m.loglik, -206.79123985102186; atol=1e-1)  # R
        @test isapprox(m.loglik, -206.79123985261947; atol=1e-1)  # Python
        @test isapprox(m.aic, 417.5824797020437; atol=1e-1)  # R
        @test isapprox(m.aic, 417.58247970523894; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d-D*s convention (Python reports 150)
        @test isapprox(m.se, [0.07037950741066856]; atol=5e-2)  # R hessian se
    end

    let name = "seasma_only"  # case 4: order=(0,0,0), seasonal_order=(0,0,1,4), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "seasma_only.csv"), ',', skipstart=1))
        m = fit_sarima(y, (0, 0, 0), (0, 0, 1, 4); include_mean=false)
        @test m.converged
        @test isempty(m.phi)
        @test isempty(m.theta)
        @test isempty(m.Phi)
        @test isapprox(m.Theta, [-0.4493534735172254]; atol=1e-2)  # R
        @test isapprox(m.Theta, [-0.4493532245391296]; atol=1e-2)  # Python
        @test isapprox(m.loglik, -220.51463186403018; atol=1e-1)  # R
        @test isapprox(m.loglik, -220.51463186673718; atol=1e-1)  # Python
        @test isapprox(m.aic, 445.02926372806036; atol=1e-1)  # R
        @test isapprox(m.aic, 445.02926373347435; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d-D*s convention (Python reports 150)
        @test isapprox(m.se, [0.06719708013264596]; atol=5e-2)  # R hessian se
    end

    let name = "ar_seasar"  # case 5: order=(1,0,0), seasonal_order=(1,0,0,4), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "ar_seasar.csv"), ',', skipstart=1))
        m = fit_sarima(y, (1, 0, 0), (1, 0, 0, 4); include_mean=false)
        @test m.converged
        @test isapprox(m.phi, [0.3061380972650594]; atol=1e-2)  # R
        @test isapprox(m.phi, [0.3061410560774969]; atol=1e-2)  # Python
        @test isempty(m.theta)
        @test isapprox(m.Phi, [0.4358008251364592]; atol=1e-2)  # R
        @test isapprox(m.Phi, [0.43580304372980666]; atol=1e-2)  # Python
        @test isempty(m.Theta)
        @test isapprox(m.loglik, -226.15662280745258; atol=1e-1)  # R
        @test isapprox(m.loglik, -226.1566228118919; atol=1e-1)  # Python
        @test isapprox(m.aic, 458.31324561490516; atol=1e-1)  # R
        @test isapprox(m.aic, 458.3132456237838; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d-D*s convention (Python reports 150)
        @test isapprox(m.se, [0.08491696903699779, 0.08118025821965133]; atol=5e-2)  # R hessian se
    end

    let name = "ma_seasma"  # case 6: order=(0,0,1), seasonal_order=(0,0,1,4), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "ma_seasma.csv"), ',', skipstart=1))
        m = fit_sarima(y, (0, 0, 1), (0, 0, 1, 4); include_mean=false)
        @test m.converged
        @test isempty(m.phi)
        @test isapprox(m.theta, [0.4688276600311235]; atol=1e-2)  # R
        @test isapprox(m.theta, [0.4688212397875021]; atol=1e-2)  # Python
        @test isempty(m.Phi)
        @test isapprox(m.Theta, [0.2752183335903611]; atol=1e-2)  # R
        @test isapprox(m.Theta, [0.2752128835476034]; atol=1e-2)  # Python
        @test isapprox(m.loglik, -221.13214207250144; atol=1e-1)  # R
        @test isapprox(m.loglik, -221.13214207999312; atol=1e-1)  # Python
        @test isapprox(m.aic, 448.2642841450029; atol=1e-1)  # R
        @test isapprox(m.aic, 448.26428415998623; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d-D*s convention (Python reports 150)
        @test isapprox(m.se, [0.06804020360931017, 0.06940728079122054]; atol=5e-2)  # R hessian se
    end

    let name = "full_block"  # case 7: order=(1,0,1), seasonal_order=(1,0,1,4), n=200
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "full_block.csv"), ',', skipstart=1))
        m = fit_sarima(y, (1, 0, 1), (1, 0, 1, 4); include_mean=false)
        @test m.converged
        @test isapprox(m.phi, [0.4723205139287688]; atol=1e-2)  # R
        @test isapprox(m.phi, [0.47233932530759043]; atol=1e-2)  # Python
        @test isapprox(m.theta, [0.3204220510124915]; atol=1e-2)  # R
        @test isapprox(m.theta, [0.32040179020561393]; atol=1e-2)  # Python
        @test isapprox(m.Phi, [0.6727939423911642]; atol=1e-2)  # R
        @test isapprox(m.Phi, [0.6728528083574199]; atol=1e-2)  # Python
        @test isapprox(m.Theta, [-0.5069326933082372]; atol=1e-2)  # R
        @test isapprox(m.Theta, [-0.5070033797130105]; atol=1e-2)  # Python
        @test isapprox(m.loglik, -284.65049917529416; atol=1e-1)  # R
        @test isapprox(m.loglik, -284.65049924351865; atol=1e-1)  # Python
        @test isapprox(m.aic, 579.3009983505883; atol=1e-1)  # R
        @test isapprox(m.aic, 579.3009984870373; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 200  # R's n-d-D*s convention (Python reports 200)
        @test isapprox(m.se, [0.09502889250124412, 0.1042391039826536, 0.1928394887852641, 0.2207953495792437]; atol=5e-2)  # R hessian se
    end

    let name = "ar_seasar_d1D1"  # case 8: order=(1,1,0), seasonal_order=(1,1,0,4), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "ar_seasar_d1D1.csv"), ',', skipstart=1))
        m = fit_sarima(y, (1, 1, 0), (1, 1, 0, 4); include_mean=false)
        @test m.converged
        @test isapprox(m.phi, [0.1072837941965288]; atol=1e-2)  # R
        @test isapprox(m.phi, [0.10729061857582914]; atol=1e-2)  # Python
        @test isempty(m.theta)
        @test isapprox(m.Phi, [0.4415796638474301]; atol=1e-2)  # R
        @test isapprox(m.Phi, [0.44157925804645237]; atol=1e-2)  # Python
        @test isempty(m.Theta)
        @test isapprox(m.loglik, -212.73109248677346; atol=1e-1)  # R
        @test isapprox(m.loglik, -212.73113790177945; atol=1e-1)  # Python
        @test isapprox(m.aic, 431.4621849735469; atol=1e-1)  # R
        @test isapprox(m.aic, 431.4622758035589; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 145  # R's n-d-D*s convention (Python reports 150)
        @test isapprox(m.se, [0.09538394556443035, 0.08774279705858917]; atol=5e-2)  # R hessian se
    end

    let name = "ma_seasma_d1"  # case 9: order=(0,1,1), seasonal_order=(0,0,1,4), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "ma_seasma_d1.csv"), ',', skipstart=1))
        m = fit_sarima(y, (0, 1, 1), (0, 0, 1, 4); include_mean=false)
        @test m.converged
        @test isempty(m.phi)
        @test isapprox(m.theta, [0.4305188843715053]; atol=1e-2)  # R
        @test isapprox(m.theta, [0.4304893915103333]; atol=1e-2)  # Python
        @test isempty(m.Phi)
        @test isapprox(m.Theta, [0.2973980957312329]; atol=1e-2)  # R
        @test isapprox(m.Theta, [0.29740126190610905]; atol=1e-2)  # Python
        @test isapprox(m.loglik, -220.84936123717515; atol=1e-1)  # R
        @test isapprox(m.loglik, -220.84936130180833; atol=1e-1)  # Python
        @test isapprox(m.aic, 447.6987224743503; atol=1e-1)  # R
        @test isapprox(m.aic, 447.69872260361666; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 149  # R's n-d-D*s convention (Python reports 150)
        @test isapprox(m.se, [0.07915852811492159, 0.07271930200978019]; atol=5e-2)  # R hessian se
    end

    let name = "s12_ar_seasar"  # case 10: order=(1,0,0), seasonal_order=(1,0,0,12), n=120
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "s12_ar_seasar.csv"), ',', skipstart=1))
        m = fit_sarima(y, (1, 0, 0), (1, 0, 0, 12); include_mean=false)
        @test m.converged
        @test isapprox(m.phi, [0.2933020216855934]; atol=1e-2)  # R
        @test isapprox(m.phi, [0.2933069953820176]; atol=1e-2)  # Python
        @test isempty(m.theta)
        @test isapprox(m.Phi, [0.1969035627774162]; atol=1e-2)  # R
        @test isapprox(m.Phi, [0.19690779127952446]; atol=1e-2)  # Python
        @test isempty(m.Theta)
        @test isapprox(m.loglik, -167.697230915222; atol=1e-1)  # R
        @test isapprox(m.loglik, -167.6972309195126; atol=1e-1)  # Python
        @test isapprox(m.aic, 341.394461830444; atol=1e-1)  # R
        @test isapprox(m.aic, 341.3944618390252; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 120  # R's n-d-D*s convention (Python reports 120)
        @test isapprox(m.se, [0.08755368394714295, 0.09469931582120747]; atol=5e-2)  # R hessian se
    end

    let name = "s12_full"  # case 11: order=(1,0,1), seasonal_order=(1,1,0,12), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "s12_full.csv"), ',', skipstart=1))
        m = fit_sarima(y, (1, 0, 1), (1, 1, 0, 12); include_mean=false)
        @test m.converged
        @test isapprox(m.phi, [0.3033494174390762]; atol=1e-2)  # R
        @test isapprox(m.phi, [0.3034524726668296]; atol=1e-2)  # Python
        @test isapprox(m.theta, [0.2699293879344048]; atol=1e-2)  # R
        @test isapprox(m.theta, [0.2698366580078816]; atol=1e-2)  # Python
        @test isapprox(m.Phi, [0.3352009602804338]; atol=1e-2)  # R
        @test isapprox(m.Phi, [0.33520793672356747]; atol=1e-2)  # Python
        @test isempty(m.Theta)
        @test isapprox(m.loglik, -204.97017839933616; atol=1e-1)  # R
        @test isapprox(m.loglik, -204.97017789668624; atol=1e-1)  # Python
        @test isapprox(m.aic, 417.9403567986723; atol=1e-1)  # R
        @test isapprox(m.aic, 417.9403557933725; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 138  # R's n-d-D*s convention (Python reports 150)
        @test isapprox(m.se, [0.1445047495102479, 0.1426916874898858, 0.08590388064477271]; atol=5e-2)  # R hessian se
    end

    let name = "p2_seasar1"  # case 12: order=(2,0,0), seasonal_order=(1,0,0,4), n=150
        y = vec(readdlm(joinpath(@__DIR__, "verification", "sarima", "bulk", "data", "p2_seasar1.csv"), ',', skipstart=1))
        m = fit_sarima(y, (2, 0, 0), (1, 0, 0, 4); include_mean=false)
        @test m.converged
        @test isapprox(m.phi, [0.3605267101890404, -0.02794204361058585]; atol=1e-2)  # R
        @test isapprox(m.phi, [0.36053322558481676, -0.027943162023521095]; atol=1e-2)  # Python
        @test isempty(m.theta)
        @test isapprox(m.Phi, [0.5157873824349779]; atol=1e-2)  # R
        @test isapprox(m.Phi, [0.5157885547627858]; atol=1e-2)  # Python
        @test isempty(m.Theta)
        @test isapprox(m.loglik, -211.0422980653217; atol=1e-1)  # R
        @test isapprox(m.loglik, -211.0422980745819; atol=1e-1)  # Python
        @test isapprox(m.aic, 430.0845961306434; atol=1e-1)  # R
        @test isapprox(m.aic, 430.0845961491638; atol=1e-1)  # Python
        @test StatsAPI.nobs(m) == 150  # R's n-d-D*s convention (Python reports 150)
        @test isapprox(m.se, [0.08307704784246903, 0.0821240458068791, 0.07043369794134235]; atol=5e-2)  # R hessian se
    end

end
end # if TSANALYTICS_FULL_TESTS
