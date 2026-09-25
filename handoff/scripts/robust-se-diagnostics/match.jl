using TSAnalytics, DelimitedFiles, LinearAlgebra, Printf, Statistics
e = vec(readdlm("garch_bench_data.csv", ','))
n = length(e)
w = 0.94 .^ (0:74); w ./= sum(w)
backcast = dot(w, e[1:75].^2)
meansq = mean(e.^2)

# GARCH(1,1) loglik with an explicit choice of how sigma2[1] is set
function gll(omega, alpha, beta, e; init_mode, init_val)
    n = length(e); s2 = Vector{Float64}(undef, n)
    s2[1] = init_mode === :direct ? init_val : omega + alpha*init_val + beta*init_val
    for t in 2:n
        s2[t] = omega + alpha*e[t-1]^2 + beta*s2[t-1]
    end
    return -0.5*sum(log(2pi) .+ log.(s2) .+ e.^2 ./ s2), s2
end

J = (0.0301204209, 0.0870910474, 0.8984497732)   # Julia's fitted params
R = (0.02910576, 0.08743932, 0.89859614)          # rugarch's fitted params

ll_j, s2_j = gll(J..., e; init_mode=:recursed, init_val=backcast)
@printf("Julia params + Julia backcast-through-recursion : LL=%.8f  sigma2[1]=%.10f\n", ll_j, s2_j[1])
@printf("   fit_garch's own reported loglik              : %.8f\n", fit_garch(e,1,1;mean_spec=:zero).loglik)

ll_r, s2_r = gll(J..., e; init_mode=:direct, init_val=meansq)
@printf("\nJulia params + rugarch init (sigma2[1]=mean e^2): LL=%.8f  sigma2[1]=%.10f\n", ll_r, s2_r[1])
@printf("   rugarch's LL at these same params            : -2079.60906763\n")
@printf("   difference                                   : %.2e\n", abs(ll_r - (-2079.60906763)))
