using Random, DelimitedFiles, Statistics
Random.seed!(20260925)
n = 1260                      # 5 years of daily trading data (252/yr)
omega, alpha, beta = 0.02, 0.09, 0.90
e = zeros(n); s2 = zeros(n)
s2[1] = omega/(1-alpha-beta)
e[1] = sqrt(s2[1])*randn()
for t in 2:n
    s2[t] = omega + alpha*e[t-1]^2 + beta*s2[t-1]
    e[t] = sqrt(s2[t])*randn()
end
writedlm("garch_bench_data.csv", e, ',')
println("n=", n, "  mean=", round(mean(e),digits=5), "  sd=", round(std(e),digits=5))
println("true params: omega=", omega, " alpha=", alpha, " beta=", beta)
