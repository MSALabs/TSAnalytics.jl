using Random, DelimitedFiles

Random.seed!(42)
n = 1000
true_beta = [1.5]
true_phi = [0.5]
true_omega = 0.05
true_alpha = 0.15
true_garch_beta = 0.75

x = randn(n)
h = zeros(n)
e = zeros(n)
nu = zeros(n)
h[1] = true_omega / (1 - true_alpha - true_garch_beta)
e[1] = sqrt(h[1]) * randn()
nu[1] = e[1]
for t in 2:n
    global_h = true_omega + true_alpha * e[t-1]^2 + true_garch_beta * h[t-1]
    h[t] = global_h
    e[t] = sqrt(h[t]) * randn()
    nu[t] = true_phi[1] * nu[t-1] + e[t]
end
y = x .* true_beta[1] .+ nu

open("test/verification/autoregarch/ar_garch_recovery.csv", "w") do io
    println(io, "y,x")
    for i in 1:n
        println(io, y[i], ",", x[i])
    end
end

println("true_beta=", true_beta, " true_phi=", true_phi, " true_omega=", true_omega,
        " true_alpha=", true_alpha, " true_garch_beta=", true_garch_beta)
println("var(nu)=", sum(nu.^2)/n, "  unconditional h=", true_omega/(1-true_alpha-true_garch_beta))
println("saved n=", n)
