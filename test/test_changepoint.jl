using DelimitedFiles
using Statistics: mean
using Random: MersenneTwister

# Stage 12.2 -- change points and structural breaks. handoff/stage 12/docs/
# stage-12.2-changepoint-breaks-handoff.md. References: changepoint 2.3 and
# strucchange 1.6.0 (GPL; independent checks, figures only), ruptures 1.1.10
# (BSD-2). Expected values below were generated once from their output.

# generated from strucchange 1.6.0 / changepoint 2.3 output (scratch cp_ref.R), full precision
const _CP_PELT = [ # (fixture, penalty, min_segment_length, breakpoints)
    ("fx0_regen", 5, 1, [100,103,105,108,172,176,200]),
    ("fx0_regen", 5, 2, [100,103,105,108,172,176,200]),
    ("fx0_regen", 10, 1, [100,200]),
    ("fx0_regen", 10, 2, [100,200]),
    ("fx0_regen", 20, 1, [100,200]),
    ("fx0_regen", 20, 2, [100,200]),
    ("fx0_regen", 50, 1, [100,200]),
    ("fx0_regen", 50, 2, [100,200]),
    ("fx0_regen", 11.40756495, 1, [100,200]),
    ("fx0_regen", 11.40756495, 2, [100,200]),
    ("fx0_regen", 17.11134742, 1, [100,200]),
    ("fx0_regen", 17.11134742, 2, [100,200]),
    ("fxA_var", 5, 1, [100,102,107,109,110,111,112,113,116,119,120,121,122,123,124,127,129,130,132,133,134,135,136,137,139,141,142,143,144,145,146,151,155,156,157,158,159,160,161,163,169,170,171,172,174,175,179,180,184,186,189,190,192,193,195,197,198,199]),
    ("fxA_var", 5, 2, [100,102,107,109,113,116,119,121,123,125,127,130,132,135,137,139,141,143,149,158,161,163,169,174,179,184,186,189,193,195,197,199]),
    ("fxA_var", 10, 1, [100,102,109,113,116,119,120,121,123,124,127,129,130,132,133,137,139,141,142,143,144,145,146,151,160,161,163,169,174,179,180,184,186,189,190,193,195,198,199]),
    ("fxA_var", 10, 2, [100,102,109,113,121,127,130,132,135,137,139,141,143,149,158,161,163,169,174,179,193,195,197,199]),
    ("fxA_var", 20, 1, [109,113,121,130,132,133,141,142,151,169,174,179,198,199]),
    ("fxA_var", 20, 2, [109,113,121,130,132,143,151,169]),
    ("fxA_var", 50, 1, Int[]),
    ("fxA_var", 50, 2, Int[]),
    ("fxA_var", 11.40756495, 1, [100,102,109,113,122,123,124,127,129,130,132,133,137,139,141,142,143,144,145,146,151,169,174,179,193,195,198,199]),
    ("fxA_var", 11.40756495, 2, [100,102,109,113,121,127,130,132,135,137,142,151,169,174,179,193,195,197,199]),
    ("fxA_var", 17.11134742, 1, [109,113,121,130,132,133,141,142,151,169,174,179,198,199]),
    ("fxA_var", 17.11134742, 2, [109,113,121,130,132,143,151,169,174,179,197,199]),
    ("fxB_weak", 5, 1, [75,85,139,229,244,255]),
    ("fxB_weak", 5, 2, [75,85,139,229,244,255]),
    ("fxB_weak", 10, 1, [139]),
    ("fxB_weak", 10, 2, [139]),
    ("fxB_weak", 20, 1, [139]),
    ("fxB_weak", 20, 2, [139]),
    ("fxB_weak", 50, 1, Int[]),
    ("fxB_weak", 50, 2, Int[]),
    ("fxB_weak", 11.40756495, 1, [139]),
    ("fxB_weak", 11.40756495, 2, [139]),
    ("fxB_weak", 17.11134742, 1, [139]),
    ("fxB_weak", 17.11134742, 2, [139]),
    ("fxC_null", 5, 1, [185,197]),
    ("fxC_null", 5, 2, [185,197]),
    ("fxC_null", 10, 1, Int[]),
    ("fxC_null", 10, 2, Int[]),
    ("fxC_null", 20, 1, Int[]),
    ("fxC_null", 20, 2, Int[]),
    ("fxC_null", 50, 1, Int[]),
    ("fxC_null", 50, 2, Int[]),
    ("fxC_null", 11.40756495, 1, Int[]),
    ("fxC_null", 11.40756495, 2, Int[]),
    ("fxC_null", 17.11134742, 1, Int[]),
    ("fxC_null", 17.11134742, 2, Int[]),
    ("fxD_edge", 5, 1, [3,106,131,134,231,297]),
    ("fxD_edge", 5, 2, [3,106,131,134,231,297]),
    ("fxD_edge", 10, 1, [3,297]),
    ("fxD_edge", 10, 2, [3,297]),
    ("fxD_edge", 20, 1, [3,297]),
    ("fxD_edge", 20, 2, [3,297]),
    ("fxD_edge", 50, 1, Int[]),
    ("fxD_edge", 50, 2, Int[]),
    ("fxD_edge", 11.40756495, 1, [3,297]),
    ("fxD_edge", 11.40756495, 2, [3,297]),
    ("fxD_edge", 17.11134742, 1, [3,297]),
    ("fxD_edge", 17.11134742, 2, [3,297]),
    ("fxE_n2000", 5, 1, [75,76,312,319,374,400,406,686,687,728,731,756,761,763,800,854,857,1056,1059,1071,1090,1200,1550,1552,1600,1760,1878,1884,1908,1909,1910,1912]),
    ("fxE_n2000", 5, 2, [312,319,374,400,406,728,731,756,761,763,800,854,857,1056,1059,1071,1090,1200,1550,1552,1600,1760,1878,1884]),
    ("fxE_n2000", 10, 1, [400,800,1200,1600]),
    ("fxE_n2000", 10, 2, [400,800,1200,1600]),
    ("fxE_n2000", 20, 1, [400,800,1200,1600]),
    ("fxE_n2000", 20, 2, [400,800,1200,1600]),
    ("fxE_n2000", 50, 1, [400,800,1200,1600]),
    ("fxE_n2000", 50, 2, [400,800,1200,1600]),
    ("fxE_n2000", 15.20180492, 1, [400,800,1200,1600]),
    ("fxE_n2000", 15.20180492, 2, [400,800,1200,1600]),
    ("fxE_n2000", 22.80270738, 1, [400,800,1200,1600]),
    ("fxE_n2000", 22.80270738, 2, [400,800,1200,1600]),
    ("fxF_offgrid", 5, 1, [84,97,134,138,149,167,168,173,182,184,203,269,270,279]),
    ("fxF_offgrid", 5, 2, [84,97,134,138,149,168,173,182,184,203,269,271,279]),
    ("fxF_offgrid", 10, 1, [84,97,203]),
    ("fxF_offgrid", 10, 2, [84,97,203]),
    ("fxF_offgrid", 20, 1, [97,203]),
    ("fxF_offgrid", 20, 2, [97,203]),
    ("fxF_offgrid", 50, 1, [97,203]),
    ("fxF_offgrid", 50, 2, [97,203]),
    ("fxF_offgrid", 11.40756495, 1, [97,203]),
    ("fxF_offgrid", 11.40756495, 2, [97,203]),
    ("fxF_offgrid", 17.11134742, 1, [97,203]),
    ("fxF_offgrid", 17.11134742, 2, [97,203]),
    ("fxG_many", 5, 1, [11,17,34,39,41,60,90,100,109,121,148,180,210,240,270,276,290]),
    ("fxG_many", 5, 2, [11,17,34,39,41,60,90,100,109,121,148,180,210,240,270,276,290]),
    ("fxG_many", 10, 1, [19,60,90,121,148,180,210,240,270]),
    ("fxG_many", 10, 2, [19,60,90,121,148,180,210,240,270]),
    ("fxG_many", 20, 1, [60,90,121,148,180,210,240,270]),
    ("fxG_many", 20, 2, [60,90,121,148,180,210,240,270]),
    ("fxG_many", 50, 1, [60,90,121,148,180,210,240,270]),
    ("fxG_many", 50, 2, [60,90,121,148,180,210,240,270]),
    ("fxG_many", 11.40756495, 1, [19,60,90,121,148,180,210,240,270]),
    ("fxG_many", 11.40756495, 2, [19,60,90,121,148,180,210,240,270]),
    ("fxG_many", 17.11134742, 1, [60,90,121,148,180,210,240,270]),
    ("fxG_many", 17.11134742, 2, [60,90,121,148,180,210,240,270]),
    ("fxH_spike", 5, 1, [2,4,50,51,84,120,122,126,127,128]),
    ("fxH_spike", 5, 2, [2,4,50,52,84,120,122]),
    ("fxH_spike", 10, 1, [50,51,120,122]),
    ("fxH_spike", 10, 2, [50,52,120,122]),
    ("fxH_spike", 20, 1, [50,51,120,122]),
    ("fxH_spike", 20, 2, [50,52,120,122]),
    ("fxH_spike", 50, 1, [50,51,120,122]),
    ("fxH_spike", 50, 2, [120,122]),
    ("fxH_spike", 10.59663473, 1, [50,51,120,122]),
    ("fxH_spike", 10.59663473, 2, [50,52,120,122]),
    ("fxH_spike", 15.8949521, 1, [50,51,120,122]),
    ("fxH_spike", 15.8949521, 2, [50,52,120,122]),
    ("fxI_step", 5, 1, [5]),
    ("fxI_step", 5, 2, [5]),
    ("fxI_step", 10, 1, [5]),
    ("fxI_step", 10, 2, [5]),
    ("fxI_step", 20, 1, [5]),
    ("fxI_step", 20, 2, [5]),
    ("fxI_step", 50, 1, [5]),
    ("fxI_step", 50, 2, [5]),
    ("fxI_step", 4.605170186, 1, [5]),
    ("fxI_step", 4.605170186, 2, [5]),
    ("fxI_step", 6.907755279, 1, [5]),
    ("fxI_step", 6.907755279, 2, [5])]
const _CP_BINSEG = Dict( # penalty 2 log n, min segment 1, up to 20 splits
    "fx0_regen" => [100,200],
    "fxA_var" => Int[],
    "fxB_weak" => [139],
    "fxC_null" => Int[],
    "fxD_edge" => [3,297],
    "fxE_n2000" => [400,800,1200,1600],
    "fxF_offgrid" => [97,203],
    "fxG_many" => [19,60,90,121,151,180,210,216,240,270],
    "fxH_spike" => [86,120,122,129])
const _CP_VARMV = [ # (fixture, penalty, cost=:var, cost=:meanvar)
    ("fx0_regen", 11.40756495, [101,200], [100,200,233,235]),
    ("fx0_regen", 17.11134742, [101,200], [100,200]),
    ("fxA_var", 11.40756495, [100,199], [46,48,67,70,100,199]),
    ("fxA_var", 17.11134742, [100,199], [100,199]),
    ("fxB_weak", 11.40756495, Int[], [20,22,72,75,78,83,85,115,139,219,222,224,242,244]),
    ("fxB_weak", 17.11134742, Int[], [115,139]),
    ("fxC_null", 11.40756495, Int[], [5,7,9,29,31]),
    ("fxC_null", 17.11134742, Int[], [5,7,9]),
    ("fxD_edge", 11.40756495, [3,297], [3,54,56,217,222,297]),
    ("fxD_edge", 17.11134742, [3], [3,297]),
    ("fxE_n2000", 15.20180492, [406,1200,1599], [400,800,1041,1043,1200,1600,1893,1895]),
    ("fxE_n2000", 22.80270738, [406,1200,1599], [400,800,1200,1600]),
    ("fxF_offgrid", 11.40756495, [86,202], [9,11,81,83,97,203]),
    ("fxF_offgrid", 17.11134742, [86,202], [97,203]),
    ("fxG_many", 11.40756495, [20,22], [20,22,60,86,90,102,105,107,121,148,150,180,183,185,210,230,232,240,268,270,298]),
    ("fxG_many", 17.11134742, Int[], [60,90,104,107,121,148,150,180,210,240,270]),
    ("fxH_spike", 10.59663473, [49,51,120,122], [45,47,49,51,76,79,81,83,120,122]),
    ("fxH_spike", 15.8949521, [49,51,120,122], [47,49,51,120,122])]
const _CP_BP = Dict(
    "fx0_regen" => (rss = [1888.7695271101138, 1450.4536892669075, 256.47787632196042, 253.82975440259258, 252.92320891583032, 252.6819409754076], bic = [1414.7401981279133, 1346.9330101119513, 838.56379499431864, 846.85777253445508, 857.19197844757241, 868.31323148872377], sel = [100,200]),
    "fxA_var" => (rss = [1084.857504299634, 1074.3255063110619, 1066.3103515974558, 1060.1674954734203, 1056.5161321022906, 1055.8470849737948], bic = [1248.3971199473456, 1256.8779997200836, 1266.0389820549019, 1275.7132940893994, 1286.0858341690248, 1297.3033615872557], sel = Int[]),
    "fxB_weak" => (rss = [284.23155832976113, 255.8134131663345, 250.01190639065263, 248.62781645598847, 247.35781011051165, 247.52360173141261], bic = [846.57271812883221, 826.37800459959305, 830.9036350611193, 840.64575689478352, 850.51697596662984, 862.12554861977139], sel = [139]),
    "fxC_null" => (rss = [280.70784318999671, 277.04025297035912, 275.68508005703904, 272.90264533037248, 270.97978718822492, 272.39318829977645], bic = [842.83026983019874, 850.29235138825368, 860.22883236748089, 868.59317364504909, 877.87947189551994, 890.84773753813363], sel = Int[]),
    "fxD_edge" => (rss = [388.83477101993878, 384.26404639177224, 381.56460619576347, 379.32304161871008, 377.21083026924987, 377.69135021516973], bic = [940.58229254220316, 948.4424885535966, 957.73512763783015, 967.37509570086218, 977.10748105843936, 988.89696569740704], sel = Int[]),
    "fxE_n2000" => (rss = [8712.6304446898539, 7906.4238933172855, 5175.7140792639957, 3239.0297363862715, 1950.6213570221191, 3740.3353044448199], bic = [8634.2090728286676, 8455.2145417321517, 7623.019818382857, 6700.8146331303496, 5701.7647291861531, 7019.0211232186448], sel = [400,800,1200,1600]),
    "fxF_offgrid" => (rss = [1981.2918243118479, 1524.0983414635882, 304.10924239922394, 302.63652619740151, 301.4020376234854, 300.88768863066849], bic = [1429.087246714917, 1361.7909862298077, 889.66716842975723, 899.61838790106185, 909.79971628211274, 920.69488751580889], sel = [97,203]),
    "fxG_many" => (rss = [720.02819834836987, 701.430434906458, 675.77740094731769, 669.19880422174811, 658.232214374298, 661.9626290709233], bic = [1125.4230551600358, 1128.9800259503206, 1129.2101923567159, 1137.6829919933414, 1144.133535807769, 1157.2364980585364], sel = Int[]),
    "fxH_spike" => (rss = [665.07522463806458, 651.14584324527209, 621.76334829104644, 617.88809205805512, 616.66168694415512, 617.4976887889402], bic = [818.48860550831489, 824.85193877451889, 826.213754442957, 835.5599509543207, 845.75922446693448, 856.62681349948014], sel = Int[]))
const _CP_CI = Dict(
    "fx0_regen" => [99 100 101; 199 200 201],
    "fxB_weak" => [112 139 162],
    "fxE_n2000" => [398 400 402; 799 800 801; 1199 1200 1201; 1598 1600 1602],
    "fxF_offgrid" => [96 97 98; 202 203 204])
const _CP_FS = Dict( # (supF, aveF, expF, argmax)
    "fx0_regen" => (90.0532851505882, 38.3437813159051, 40.2842610166842, 100),
    "fxA_var" => (2.92140080651288, 0.474468045300866, 0.303543729665753, 113),
    "fxB_weak" => (33.1046255701795, 20.5799629176983, 13.968407115963, 139),
    "fxC_null" => (3.94506528828846, 1.36055114803427, 0.827368402833151, 197),
    "fxD_edge" => (3.54463539324986, 1.0255479053462, 0.627693335975256, 222),
    "fxE_n2000" => (203.733155643714, 46.1385485263139, 95.8310577799964, 800),
    "fxF_offgrid" => (89.3929572536289, 35.9870075739244, 39.772170590786, 97),
    "fxG_many" => (7.90118767291418, 1.3146343395008, 1.1762557001304, 60),
    "fxH_spike" => (4.23563713779872, 1.81746032974834, 1.121588809823, 86))
const _CP_EFP = [ # (fixture, type, statistic, strucchange p-value, process[2])
    ("fx0_regen", :ols_cusum, 3.92675165402279, 8.09352584951739e-14, -0.0262115906316583),
    ("fx0_regen", :rec_cusum, 3.59662045428425, 0, -0.0269626367612083),
    ("fx0_regen", :score_cusum, 3.4960648612827, 9.67739222090813e-11, -0.00414909995614747),
    ("fx0_regen", :re, 3.92675165402279, 8.09352584951739e-14, -0.026211590631658),
    ("fxA_var", :ols_cusum, 0.825550033681772, 0.503186480711198, 0.0261039792717543),
    ("fxA_var", :rec_cusum, 0.386662737375508, 0.868294220851385, -0.0228787416569893),
    ("fxA_var", :score_cusum, 2.15158336636319, 0.000381109853581862, 0.0282909493502927),
    ("fxA_var", :re, 0.825550033681772, 0.503186480711198, 0.0261039792717542),
    ("fxB_weak", :ols_cusum, 2.73236681441735, 6.55079236744349e-07, -0.0045252415565006),
    ("fxB_weak", :rec_cusum, 1.51967680683021, 0.000187234507945222, -0.0752478144230053),
    ("fxB_weak", :score_cusum, 2.73156950835054, 1.32162313748463e-06, -0.000470926424550872),
    ("fxB_weak", :re, 2.73236681441734, 6.55079236744349e-07, -0.00452524155650054),
    ("fxC_null", :ols_cusum, 0.938489016300368, 0.341824637510966, -0.0579015634770499),
    ("fxC_null", :rec_cusum, 0.390829838203737, 0.860940760841993, 0.0380571032216899),
    ("fxC_null", :score_cusum, 0.953239074406399, 0.542381618558154, -0.0582136158393272),
    ("fxC_null", :re, 0.93848901630037, 0.341824637510964, -0.0579015634770501),
    ("fxD_edge", :ols_cusum, 0.822333739959716, 0.508269582498201, -0.198596682834683),
    ("fxD_edge", :rec_cusum, 1.3436812774973, 0.00138555145531472, -0.00275759143548406),
    ("fxD_edge", :score_cusum, 1.06378173179487, 0.372398553315465, -0.134351269235245),
    ("fxD_edge", :re, 0.822333739959705, 0.508269582498219, -0.198596682834684),
    ("fxE_n2000", :ols_cusum, 6.66285761820637, 0, -0.0154017243427669),
    ("fxE_n2000", :rec_cusum, 5.24056593419629, 0, 0.00401550393869232),
    ("fxE_n2000", :score_cusum, 6.66072314703408, 0, -0.0154448991385205),
    ("fxE_n2000", :re, 6.66285761820638, 0, -0.0154017243427665),
    ("fxF_offgrid", :ols_cusum, 3.88529516214076, 1.54654067330284e-13, -0.0118525977370613),
    ("fxF_offgrid", :rec_cusum, 3.33710212539764, 0, -0.0511269858189189),
    ("fxF_offgrid", :score_cusum, 3.51639699443967, 7.27637949893278e-11, 0.0161077520104351),
    ("fxF_offgrid", :re, 3.88529516214076, 1.54654067330284e-13, -0.0118525977370608),
    ("fxG_many", :ols_cusum, 1.22953965891116, 0.0972467838046096, -0.0368550279804118),
    ("fxG_many", :rec_cusum, 0.723653585179303, 0.218233009099983, -0.00520539653804182),
    ("fxG_many", :score_cusum, 1.22833504724164, 0.186078724943625, -0.0348686029492458),
    ("fxG_many", :re, 1.22953965891116, 0.0972467838046098, -0.0368550279804118),
    ("fxH_spike", :ols_cusum, 1.01071502952389, 0.258690469832074, -0.0187990833028493),
    ("fxH_spike", :rec_cusum, 0.683045687258477, 0.272969981623872, -0.0154993592778492),
    ("fxH_spike", :score_cusum, 1.39993122431301, 0.0778184650513105, -0.0169016335401143),
    ("fxH_spike", :re, 1.01071502952388, 0.258690469832074, -0.0187990833028493)]

const _CP_FXD = joinpath(@__DIR__, "..", "handoff", "stage 12", "fixtures")
_cpfx(f) = Float64.(vec(readdlm(joinpath(_CP_FXD, f * ".csv"), ','; skipstart=1)))
# printed to ten digits: restore the exact penalty when it is one of the named ones
_cp_pen(p, n) = isapprox(p, 2log(n); rtol=1e-8) ? 2log(n) : isapprox(p, 3log(n); rtol=1e-8) ? 3log(n) : Float64(p)

@testset "PELT on fx0, the happy path" begin
    y = _cpfx("fx0_regen")
    @test length(y) == 300
    @test sum(y) ≈ 482.7709235930279 rtol = 1e-13                     # the handoff's regeneration check
    for pen in (10, 20, 50, 2log(300), 3log(300))
        @test changepoints(y; penalty=pen).breakpoints == [100, 200]
    end
    # the named penalties, lightest to heaviest: the light ones over-segment
    counts = Dict(nm => length(changepoints(y; penalty=nm).breakpoints) for nm in (:aic0, :hq0, :aic, :bic0, :hq, :bic))
    @test counts[:bic] == 2 && counts[:hq] == 2
    @test counts[:aic0] > counts[:aic] > counts[:bic]
    @test counts[:hq0] > counts[:hq]
    r = changepoints(y; penalty=:bic)
    @test r.breakpoints == [100, 200]
    @test r.segment_means ≈ [mean(y[1:100]), mean(y[101:200]), mean(y[201:300])]
    @test r.penalty ≈ 2log(300)
    @test r.cost === :mean && r.search === :pelt
    @test occursin("100, 200", sprint(show, r))
end

@testset "PELT = changepoint, 120 configurations" begin
    # ten fixtures x six penalties x minimum segment lengths 1 and 2 (changepoint
    # 2.3, penalty = "Manual"; ruptures agrees with jump = 1, charter §5.6)
    ok = 0
    for (f, pen, ms, want) in _CP_PELT
        y = _cpfx(f)
        got = changepoints(y; penalty=_cp_pen(pen, length(y)), min_segment_length=ms).breakpoints
        ok += got == want
        @test got == want
    end
    @test ok == 120
end

@testset "the penalty formulas" begin
    n = 300
    @test cp_penalty(:bic0, n) ≈ 5.7037824747 atol = 1e-10
    @test cp_penalty(:bic, n) ≈ 11.4075649493 atol = 1e-10
    @test cp_penalty(:sic, n) == cp_penalty(:bic, n)
    @test cp_penalty(:mbic, n) ≈ 17.1113474240 atol = 1e-10
    @test cp_penalty(:aic0, n) == 2.0
    @test cp_penalty(:aic, n) == 4.0
    @test cp_penalty(:hq0, n) ≈ 3.4822590936 atol = 1e-10
    @test cp_penalty(:hq, n) ≈ 6.9645181872 atol = 1e-10
    @test cp_penalty(:bic, n; cost=:meanvar) ≈ 3log(n)                # two parameters per change
    @test cp_penalty(:mbic, n; cost=:meanvar) ≈ 4log(n)
    @test_throws ArgumentError cp_penalty(:asymptotic, n)
    @test_throws ArgumentError cp_penalty(:bic, 2)
end

@testset "PELT = exact optimal partitioning" begin
    for f in ("fx0_regen", "fxG_many", "fxH_spike", "fxF_offgrid", "fxA_var")
        y = _cpfx(f)
        for pen in (5.0, 2log(length(y))), ms in (1, 2, 5)
            p = changepoints(y; penalty=pen, min_segment_length=ms)
            e = changepoints(y; penalty=pen, min_segment_length=ms, search=:exact)
            @test p.breakpoints == e.breakpoints
            @test p.criterion ≈ e.criterion rtol = 1e-12
        end
    end
    y = _cpfx("fxA_var")
    for cost in (:var, :meanvar)
        @test changepoints(y; cost=cost, penalty=:bic).breakpoints ==
              changepoints(y; cost=cost, penalty=:bic, search=:exact).breakpoints
    end
end

@testset "PELT refuses MBIC, naming the pruning violation" begin
    y = _cpfx("fxB_weak")
    e = try changepoints(y; penalty=:mbic); catch err; err end
    @test e isa ArgumentError
    @test occursin("pruning", e.msg) && occursin(":exact", e.msg) && occursin("+3.22", e.msg)
    # D19: changepoint's own default (MBIC + PELT) returns 187, 2.06672005 worse
    r = changepoints(y; penalty=:mbic, search=:exact, min_segment_length=1)
    @test r.breakpoints == [139]
    @test r.criterion ≈ 282.94063889 atol = 1e-8
    @test r.criterion < 285.00735894 - 2.06
    @test changepoints(y; penalty=:mbic, search=:binseg, min_segment_length=1).breakpoints == [139]
    # the arithmetic of the violation: halving a 100-point segment adds log 50 + log 50 - log 100
    @test 2log(50) - log(100) ≈ 3.218876 atol = 1e-6
end

@testset "the pruning condition holds for every shipped cost" begin
    # PELT is exact iff splitting a segment never raises the cost (K = 0):
    # C(a..b) >= C(a..t) + C(t+1..b) for every split, checked on random splits
    rng = MersenneTwister(12)
    data = Dict(:mean => _cpfx("fx0_regen"), :var => _cpfx("fxA_var"), :meanvar => _cpfx("fxA_var"),
                :poisson => Float64.(rand(rng, 0:6, 300)), :exponential => -log.(rand(rng, 300)))
    for (kind, y) in data
        c = TSAnalytics._CPCost(y, kind, nothing, false)
        viol = 0
        for _ in 1:190
            a = rand(rng, 1:200); b = a + rand(rng, 6:99); t = rand(rng, (a + 2):(b - 3))
            viol += TSAnalytics._cp_cost(c, a, b) < TSAnalytics._cp_cost(c, a, t) + TSAnalytics._cp_cost(c, t + 1, b) - 1e-9
        end
        @test viol == 0
    end
    # and the MBIC-modified cost does violate it (the handoff measured 143 of 190)
    c = TSAnalytics._CPCost(_cpfx("fxB_weak"), :mean, nothing, true)
    viol = 0
    for _ in 1:190
        a = rand(rng, 1:150); b = a + rand(rng, 20:140); t = rand(rng, (a + 5):(b - 5))
        viol += TSAnalytics._cp_cost(c, a, b) < TSAnalytics._cp_cost(c, a, t) + TSAnalytics._cp_cost(c, t + 1, b)
    end
    @test viol > 100
end

@testset "BinSeg is approximate" begin
    for (f, want) in _CP_BINSEG
        y = _cpfx(f)
        @test changepoints(y; penalty=2log(length(y)), search=:binseg, min_segment_length=1).breakpoints == want
    end
    # fxG_many: PELT finds 9 well placed, BinSeg 10 with two misplaced
    y = _cpfx("fxG_many")
    pelt = changepoints(y; penalty=2log(300), min_segment_length=1).breakpoints
    bs = changepoints(y; penalty=2log(300), search=:binseg, min_segment_length=1).breakpoints
    @test pelt == [19, 60, 90, 121, 148, 180, 210, 240, 270]
    @test bs == [19, 60, 90, 121, 151, 180, 210, 216, 240, 270]
    # fxA_var: BinSeg stops at once -- no single split clears the penalty -- where PELT splits 14 times
    @test isempty(changepoints(_cpfx("fxA_var"); penalty=2log(300), search=:binseg, min_segment_length=1).breakpoints)
    @test length(changepoints(_cpfx("fxA_var"); penalty=20.0, min_segment_length=1).breakpoints) == 14
end

@testset "fxF_offgrid: breaks at 97 and 203 exactly" begin
    y = _cpfx("fxF_offgrid")
    @test sum(y) ≈ 509.2309163920 atol = 1e-9
    for pen in (20, 50, 2log(300), 3log(300))
        @test changepoints(y; penalty=pen).breakpoints == [97, 203]
    end
    @test changepoints(y; penalty=10, min_segment_length=1).breakpoints == [84, 97, 203]
    # D20: ruptures at its defaults (jump = 5) gives 95, 100, 200, 205 -- there is no jump here
    @test !(95 in changepoints(y; penalty=10).breakpoints)
    @test breakpoints(y).breaks == [97, 203]
    @test break_confint(breakpoints(y)) == [96 97 98; 202 203 204]
    @test fstats(y).argmax == 97
end

@testset "fxA_var: a variance change needs a variance cost" begin
    y = _cpfx("fxA_var")
    @test length(changepoints(y; penalty=:mbic, search=:exact, min_segment_length=1).breakpoints) == 14
    @test length(changepoints(y; penalty=:bic, min_segment_length=1).breakpoints) == 28
    for (f, pen, v, mv) in _CP_VARMV
        f == "fxA_var" || continue
        @test changepoints(y; cost=:var, penalty=_cp_pen(pen, 300)).breakpoints == v
        @test changepoints(y; cost=:meanvar, penalty=_cp_pen(pen, 300)).breakpoints == mv
    end
    @test changepoints(y; cost=:var, penalty=:bic).breakpoints == [100, 199]
    @test changepoints(y; cost=:meanvar, penalty=3log(300)).breakpoints == [100, 199]
    # strucchange: correctly silent on the mean
    @test isempty(breakpoints(y).breaks)
    @test fstats(y).supF ≈ 2.92140080651288 rtol = 1e-12
    @test efp(y; type=:score_cusum).pvalue ≈ 0.000381109853581862 rtol = 1e-9
    @test efp(y; type=:ols_cusum).pvalue ≈ 0.503186480711198 rtol = 1e-9
    @test efp(y; type=:score_cusum).pvalue < 0.001 < 0.5 < efp(y; type=:ols_cusum).pvalue
end

@testset "fxB_weak: 139, not 187" begin
    y = _cpfx("fxB_weak")
    @test changepoints(y; penalty=:bic, min_segment_length=1).breakpoints == [139]
    @test changepoints(y; penalty=:bic, search=:binseg, min_segment_length=1).breakpoints == [139]
    @test breakpoints(y).breaks == [139]
    @test break_confint(breakpoints(y)) == [112 139 162]
    @test fstats(y).supF ≈ 33.1046255701795 rtol = 1e-12
    @test fstats(y).argmax == 139
    @test changepoints(y; penalty=:mbic, search=:exact, min_segment_length=1).breakpoints != [187]
    @test changepoints(y; penalty=50.0).breakpoints == Int[]
end

@testset "fxC_null: no false positives at any default" begin
    y = _cpfx("fxC_null")
    for pen in (:bic, :hq, 10.0, 20.0), ms in (1, 2)
        @test isempty(changepoints(y; penalty=pen, min_segment_length=ms).breakpoints)
    end
    @test isempty(changepoints(y; penalty=:mbic, search=:exact).breakpoints)
    @test isempty(changepoints(y; search=:binseg).breakpoints)
    @test isempty(breakpoints(y).breaks)
    for t in (:ols_cusum, :rec_cusum, :score_cusum, :re)
        @test efp(y; type=t).pvalue > 0.25
    end
    @test_throws ArgumentError break_confint(breakpoints(y))
end

@testset "fxD_edge and the minimum segment length" begin
    y = _cpfx("fxD_edge")
    @test changepoints(y; penalty=:bic, min_segment_length=1).breakpoints == [3, 297]
    @test changepoints(y; penalty=:bic).breakpoints == [3, 297]            # our default, 2
    @test changepoints(y; penalty=:bic, search=:binseg).breakpoints == [3, 297]
    # strucchange's h = 0.15 -> 45 makes a 3-point edge segment inadmissible
    b = breakpoints(y)
    @test b.h == 45
    @test b.max_breaks == 5                                           # ceil(300/45) - 2
    @test isempty(b.breaks)
    @test isempty(changepoints(y; penalty=:bic, min_segment_length=45).breakpoints) ||
          all(t -> 45 <= t <= 255, changepoints(y; penalty=:bic, min_segment_length=45).breakpoints)
    @test fstats(y).supF ≈ 3.54463539324986 rtol = 1e-12
    @test changepoints(y; penalty=:bic, min_segment_length=4).breakpoints != [3, 297]
end

@testset "fxH_spike" begin
    y = _cpfx("fxH_spike")
    @test changepoints(y; penalty=:bic, min_segment_length=1).breakpoints == [50, 51, 120, 122]
    @test changepoints(y; penalty=50.0, min_segment_length=1).breakpoints == [50, 51, 120, 122]
    # a minimum length of 2 cannot isolate the one-point spike at 51
    @test !(51 in changepoints(y; penalty=:bic, min_segment_length=2).breakpoints) ||
          50 ∉ changepoints(y; penalty=:bic, min_segment_length=2).breakpoints
    @test changepoints(y; penalty=2log(200), search=:binseg, min_segment_length=1).breakpoints == [86, 120, 122, 129]
    @test isempty(breakpoints(y).breaks)
    @test breakpoints(y).h == 30
    @test fstats(y).argmax == 86
    @test efp(y; type=:rec_cusum).pvalue ≈ 0.272969981623872 rtol = 1e-9
end

@testset "the index convention on fxI_step" begin
    y = _cpfx("fxI_step")
    @test y == vcat(zeros(5), fill(10.0, 5))
    r = changepoints(y; penalty=:bic, min_segment_length=1)
    @test r.breakpoints == [5]                    # last index of the first segment, as cpts()
    @test r.segment_means == [0.0, 10.0]
    @test 5 ∉ (r.breakpoints .+ 1)                # not the start of the next (ruptures' reading)
    @test 10 ∉ r.breakpoints                      # n is not appended (ruptures: [5, 10])
    @test changepoints(y; penalty=:bic, search=:exact, min_segment_length=1).breakpoints == [5]
    @test changepoints(y; penalty=:bic, search=:binseg, min_segment_length=1).breakpoints == [5]
    for cp in (5,)
        @test all(y[1:cp] .== 0) && all(y[(cp + 1):end] .== 10)
    end
    @test fstats(y; from=2).argmax == 5
    @test length(r.segment_vars) == 2
end

@testset "cost inventory: mean, var, meanvar, Poisson, exponential" begin
    # changepoint cpt.meanvar(test.stat = "Poisson"/"Exponential"), run here
    rng = MersenneTwister(1)
    pz = Float64.(vcat(rand(rng, 0:3, 100), rand(rng, 4:10, 100), rand(rng, 1:4, 100)))
    r = changepoints(pz; cost=:poisson, penalty=:bic, min_segment_length=1)
    @test length(r.breakpoints) >= 2
    @test any(t -> 95 <= t <= 105, r.breakpoints) && any(t -> 195 <= t <= 205, r.breakpoints)
    ex = vcat(-log.(rand(rng, 150)), -5 .* log.(rand(rng, 150)))
    re = changepoints(ex; cost=:exponential, penalty=:bic, min_segment_length=1)
    @test any(t -> 140 <= t <= 160, re.breakpoints)
    @test_throws ArgumentError changepoints([1.0, -1.0, 2.0]; cost=:poisson)
    @test_throws ArgumentError changepoints([1.5, 2.0, 3.0]; cost=:poisson)
    @test_throws ArgumentError changepoints([1.0, 0.0, 2.0]; cost=:exponential)
    @test_throws ArgumentError changepoints(pz; cost=:median)
    @test changepoints(_cpfx("fx0_regen"); cost=:var, penalty=2log(300)).breakpoints == [101, 200]
    @test changepoints(_cpfx("fx0_regen"); cost=:meanvar, penalty=3log(300)).breakpoints == [100, 200]
    @test changepoints(_cpfx("fxA_var"); cost=:var, penalty=:bic).min_segment_length == 2
    @test changepoints(_cpfx("fxA_var"); cost=:var, penalty=:bic, min_segment_length=1).min_segment_length == 2
    # the Poisson/exponential references from changepoint (cp_ref2.R): one change at 100,201 / 150
    for (f, pen, v, mv) in _CP_VARMV
        y = _cpfx(f)
        @test changepoints(y; cost=:var, penalty=_cp_pen(pen, length(y))).breakpoints == v
    end
end

@testset "scale non-invariance, and standardize" begin
    y = _cpfx("fx0_regen")
    k1 = length(changepoints(y; penalty=:bic).breakpoints)
    k3 = length(changepoints(3 .* y; penalty=:bic).breakpoints)
    @test k1 == 2
    @test k3 > k1                                    # same series, larger units, more changes
    s1 = changepoints(y; penalty=:bic, standardize=true).breakpoints
    s3 = changepoints(3 .* y; penalty=:bic, standardize=true).breakpoints
    @test s1 == s3                                   # standardised: unit-free
    @test s1 == [100, 200]
    @test changepoints(y; penalty=:bic, standardize=true).standardized
    @test occursin("standardised", sprint(show, changepoints(y; penalty=:bic, standardize=true)))
    @test_throws ArgumentError changepoints(y; cost=:var, standardize=true)
    @test_throws ArgumentError changepoints(ones(10); standardize=true)
end

@testset "Bai-Perron RSS and BIC tables, every m" begin
    for (f, t) in _CP_BP
        b = breakpoints(_cpfx(f))
        @test b.rss[1:length(t.rss)] ≈ t.rss rtol = 1e-12
        @test b.bic[1:length(t.bic)] ≈ t.bic rtol = 1e-12
        @test b.breaks == t.sel
    end
    b = breakpoints(_cpfx("fx0_regen"))
    @test b.rss[1] ≈ sum(abs2, _cpfx("fx0_regen") .- mean(_cpfx("fx0_regen"))) rtol = 1e-12
    @test issorted(b.rss[1:3]; rev=true)
end

@testset "Bai-Perron dates on fx0, fxF, fxE" begin
    b = breakpoints(_cpfx("fx0_regen"))
    @test b.dates[2] == [100] && b.dates[3] == [100, 200] && b.dates[4] == [100, 147, 200]
    @test b.breaks == [100, 200]
    @test b.bic[3] ≈ 838.56379499431864 rtol = 1e-13
    e = breakpoints(_cpfx("fxE_n2000"))
    @test e.breaks == [400, 800, 1200, 1600]
    @test e.h == 300
    @test breakpoints(_cpfx("fxF_offgrid")).breaks == [97, 203]
    @test breakpoints(_cpfx("fxB_weak")).dates[3] == [139, 228]
    # a regression with a trend: X = [1 t]
    y = _cpfx("fx0_regen")
    bx = breakpoints(y; X=hcat(ones(300), 1:300))
    @test bx.breaks == [100, 200]
    @test_throws ArgumentError breakpoints(y; h=200)
    @test_throws ArgumentError breakpoints(y; X=ones(10))
end

@testset "break_confint on four fixtures" begin
    for (f, want) in _CP_CI
        b = breakpoints(_cpfx(f))
        ci = break_confint(b)
        @test ci == want
        @test all(ci[:, 1] .<= ci[:, 2] .<= ci[:, 3])
    end
    b = breakpoints(_cpfx("fx0_regen"))
    @test break_confint(b; breaks=1)[2] == 100
    @test size(break_confint(b; level=0.99)) == (2, 3)
    w99 = break_confint(b; level=0.99); w90 = break_confint(b; level=0.90)
    @test all(w99[:, 3] .- w99[:, 1] .>= w90[:, 3] .- w90[:, 1])
    @test_throws ArgumentError break_confint(b; level=1.5)
    # the distribution function is a distribution function
    @test TSAnalytics._bai_cdf(-1e3, 1.0, 1.0, 1.0) < 1e-6
    @test TSAnalytics._bai_cdf(1e3, 1.0, 1.0, 1.0) > 1 - 1e-6
    @test TSAnalytics._bai_cdf(0.0, 1.0, 1.0, 1.0) ≈ 0.5 atol = 1e-12
end

@testset "fstats path and sup/ave/exp F" begin
    for (f, (sup, ave, ex, am)) in _CP_FS
        r = fstats(_cpfx(f))
        @test r.supF ≈ sup rtol = 1e-11
        @test r.aveF ≈ ave rtol = 1e-11
        @test r.expF ≈ ex rtol = 1e-11
        @test r.argmax == am
    end
    r = fstats(_cpfx("fx0_regen"))
    @test first(r.dates) == 45 && last(r.dates) == 255
    @test r.stats[1:3] ≈ [26.7031464695306, 27.8251528747893, 27.9199292210864] rtol = 1e-12
    @test occursin("no p-values", sprint(show, r))
end

@testset "the k-times-Chow convention" begin
    y = _cpfx("fxB_weak")
    n = 300
    τ = 139
    rss(v) = sum(abs2, v .- mean(v))
    ru = rss(y[1:τ]) + rss(y[(τ + 1):end]); rr = rss(y)
    r = fstats(y)
    @test r.stats[findfirst(==(τ), r.dates)] ≈ (rr - ru) / (ru / (n - 2)) rtol = 1e-12
    # with k = 2 regressors the statistic is twice the classical Chow F
    X = hcat(ones(n), collect(1.0:n))
    f2 = fstats(y; X=X)
    b(A, v) = (A \ v)
    res(A, v) = sum(abs2, v .- A * b(A, v))
    j = findfirst(==(τ), f2.dates)
    ru2 = res(X[1:τ, :], y[1:τ]) + res(X[(τ + 1):end, :], y[(τ + 1):end]); rr2 = res(X, y)
    chow = ((rr2 - ru2) / 2) / (ru2 / (n - 4))
    @test f2.stats[j] ≈ 2 * chow rtol = 1e-10
    @test f2.k == 2
    @test f2.stats[j] != chow
end

@testset "efp: four types on every fixture" begin
    for (f, t, stat, pR, first) in _CP_EFP
        e = efp(_cpfx(f); type=t)
        @test e.statistic ≈ stat rtol = 1e-12
        @test e.process[2, 1] ≈ first atol = 1e-12
        if pR > 1e-6
            @test e.pvalue ≈ pR rtol = 1e-6
        elseif pR > 1e-12
            # strucchange forms 1 - (1 + 2p)^k, which loses digits this far out;
            # the tail computed directly differs from it in the fifth digit
            @test e.pvalue ≈ pR rtol = 1e-4
        else
            @test e.pvalue < 1e-12                       # strucchange printed 0 or ~1e-13
        end
    end
end

@testset "Score-CUSUM sees the variance change; OLS-CUSUM does not" begin
    y = _cpfx("fxA_var")
    s = efp(y; type=:score_cusum)
    o = efp(y; type=:ols_cusum)
    @test s.pvalue < 0.001
    @test o.pvalue > 0.5
    @test size(s.process) == (301, 2)                     # mean and variance components
    @test s.statistic ≈ 2.15158336636319 rtol = 1e-12
    @test argmax(abs.(s.process[:, 2])) in 95:205         # the variance component moves
    @test maximum(abs, s.process[:, 2]) > maximum(abs, s.process[:, 1])
    @test s.limit === :bridge
    @test occursin("Score-based", sprint(show, s))
end

@testset "CUSUM p-values, closed form, and no false zeros" begin
    y = _cpfx("fx0_regen")
    o = efp(y; type=:ols_cusum)
    @test o.pvalue ≈ 8.09352584951739e-14 atol = 1e-12
    @test o.pvalue > 0
    r = efp(y; type=:rec_cusum)
    @test r.pvalue > 0 && r.pvalue < 1e-20                # strucchange prints 0 (D27)
    @test TSAnalytics._p_motion_max(3.59662045428425, 1)[1] ≈ r.pvalue
    e = efp(_cpfx("fxE_n2000"); type=:ols_cusum)
    @test 0 < e.pvalue < 1e-30
    # the bridge tail at a moderate value, against the series by hand
    x = 1.0
    @test TSAnalytics._p_bridge_max(x, 1)[1] ≈ 2 * sum((-1)^(i + 1) * exp(-2i^2) for i in 1:100) rtol = 1e-14
    @test TSAnalytics._p_bridge_max(0.05, 1)[1] == 1.0
    @test 0 <= TSAnalytics._p_motion_max(0.5, 1)[1] <= 1
    @test pvalue(o) == o.pvalue && statistic(o) == o.statistic
    @test_throws ArgumentError efp(y; type=:ols_mosum)
end

@testset "refusals" begin
    y = _cpfx("fx0_regen")
    @test_throws ArgumentError changepoints(y; min_segment_length=400)
    @test_throws ArgumentError changepoints([1.0])
    @test_throws ArgumentError changepoints(vcat(y[1:10], NaN))
    @test_throws ArgumentError changepoints(y; search=:segneigh)
    @test_throws ArgumentError changepoints(y; penalty=-1.0)
    @test_throws ArgumentError changepoints(y; penalty=:sic1)
    @test_throws ArgumentError breakpoints(y; h=1)
    @test_throws ArgumentError breakpoints(vcat(y, NaN))
    @test_throws ArgumentError fstats(y; from=0.6)
    @test_throws ArgumentError efp(y[1:3])
    @test_throws ArgumentError break_confint(breakpoints(_cpfx("fxC_null")))
    @test_throws ArgumentError breakpoints(y; max_breaks=-1)
end

@testset "determinism across repeated threaded runs" begin
    y = _cpfx("fxE_n2000")
    b = [breakpoints(y) for _ in 1:3]
    @test all(x -> x.rss == b[1].rss, b)
    @test all(x -> x.dates == b[1].dates, b)
    p = [changepoints(y; penalty=:bic).breakpoints for _ in 1:3]
    @test all(==(p[1]), p)
    @test b[1].breaks == [400, 800, 1200, 1600]
    @test p[1] == [400, 800, 1200, 1600]
end

@testset "performance" begin
    y = _cpfx("fxE_n2000")
    c = TSAnalytics._CPCost(y, :mean, nothing, false)
    TSAnalytics._cp_optimal(c, 2000, 2log(2000), 2, true)
    t = minimum(@elapsed(TSAnalytics._cp_optimal(c, 2000, 2log(2000), 2, true)) for _ in 1:5)
    @test t < 0.02                                   # changepoint's C: 0.002 s; measured here ~0.0015 s
    costalloc(c) = @allocated TSAnalytics._cp_cost(c, 1, 500)
    costalloc(c)
    @test costalloc(c) == 0
    tb = @elapsed breakpoints(y)
    @test tb < 30                                    # strucchange: 10.4 s
    @test changepoints(y; penalty=:bic).breakpoints == [400, 800, 1200, 1600]
    @inferred TSAnalytics._cp_cost(c, 1, 100)
end
