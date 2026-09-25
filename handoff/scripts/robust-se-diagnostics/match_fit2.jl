using DelimitedFiles, Printf, Statistics, Optim
e = vec(readdlm("garch_bench_data.csv", ','))
meansq = mean(e.^2)
unpack(p) = begin
    wgt = exp.(p[2:3]); d = 1 + sum(wgt)
    (exp(p[1]), wgt[1]/d, wgt[2]/d)
end
function nll(p)
    omega, alpha, beta = unpack(p)
    n = length(e); s2 = Vector{Float64}(undef, n); s2[1] = meansq
    for t in 2:n; s2[t] = omega + alpha*e[t-1]^2 + beta*s2[t-1]; end
    any(<=(0), s2) && return 1e10
    return 0.5*sum(log(2pi) .+ log.(s2) .+ e.^2 ./ s2)
end
best = nothing
for st in ([log(0.03), 1.83, 4.16], [log(0.05), 1.0, 3.0], [log(0.02), 2.0, 4.5])
    r = optimize(nll, st, NelderMead(), Optim.Options(iterations=50000, g_tol=1e-14))
    if best === nothing || r.minimum < best.minimum; global best = r; end
end
om, al, be = unpack(best.minimizer)
@printf("Julia refit, rugarch init : omega=%.8f alpha=%.8f beta=%.8f  LL=%.6f\n", om, al, be, -best.minimum)
@printf("rugarch's own fit         : omega=%.8f alpha=%.8f beta=%.8f  LL=%.6f\n", 0.02910576, 0.08743932, 0.89859614, -2079.6025)
@printf("abs diff                  : omega=%.2e  alpha=%.2e  beta=%.2e  LL=%.2e\n",
        abs(om-0.02910576), abs(al-0.08743932), abs(be-0.89859614), abs(-best.minimum-(-2079.6025)))
