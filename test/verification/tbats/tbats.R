suppressMessages(library(forecast))
y <- ts(scan("test/verification/ets/ets_y.csv", quiet=TRUE), frequency=4)
m <- tbats(y, use.box.cox=FALSE, use.trend=TRUE, use.damped.trend=FALSE,
           use.arma.errors=FALSE, use.parallel=FALSE)
cat("class:", class(m), "\n")
cat("names:", paste(names(m), collapse=" "), "\n")
cat("k.vector:", m$k.vector, " seasonal.periods:", m$seasonal.periods, "\n")
cat("lambda:", m$lambda, " damping:", m$damping.parameter, "\n")
cat("alpha:", m$alpha, " beta:", m$beta, "\n")
cat("gamma.one.v:", m$gamma.one.values, " gamma.two.v:", m$gamma.two.values, "\n")
cat("ar.coef:", m$ar.coefficients, " ma.coef:", m$ma.coefficients, "\n")
cat("likelihood:", m$likelihood, " AIC:", m$AIC, " variance:", m$variance, "\n")
cat("seed.states:", paste(sprintf("%.8f", m$seed.states), collapse=" "), "\n")
cat("fitted head:", paste(sprintf("%.8f", head(m$fitted.values,3)), collapse=" "), "\n")
cat("sse:", sum(m$errors^2), "\n")
f <- forecast(m, h=8, level=95)
cat("mean:", paste(sprintf("%.8f", as.numeric(f$mean)), collapse=" "), "\n")
cat("lo95:", paste(sprintf("%.8f", f$lower), collapse=" "), "\n")
suppressMessages(library(forecast))
y <- ts(scan("test/verification/ets/ets_y.csv", quiet=TRUE), frequency=1)
# BATS with no seasonality IS ETS(A,A,N): level + trend, additive error.
b <- bats(y, use.box.cox=FALSE, use.trend=TRUE, use.damped.trend=FALSE,
          use.arma.errors=FALSE, use.parallel=FALSE)
e <- ets(y, model="AAN", damped=FALSE)
cat(sprintf("bats: alpha=%.8f beta=%.8f sigma2=%.8f\n", b$alpha, b$beta, b$variance))
cat(sprintf("ets : alpha=%.8f beta=%.8f sigma2=%.8f\n",
            e$par["alpha"], e$par["beta"], e$sigma2))
fb <- forecast(b, h=8, level=95); fe <- forecast(e, h=8, level=95)
seb <- (as.numeric(fb$mean) - fb$lower) / qnorm(0.975)
see <- (as.numeric(fe$mean) - fe$lower) / qnorm(0.975)
cat("bats se:", paste(sprintf("%.5f", seb), collapse=" "), "\n")
cat("ets  se:", paste(sprintf("%.5f", see), collapse=" "), "\n")
cat("bats se / sqrt(variance):", paste(sprintf("%.5f", seb/sqrt(b$variance)), collapse=" "), "\n")
al <- b$alpha; be <- b$beta
cat("correct ETS(A,A,N) factor sqrt(1+sum((al+j*be)^2)):",
    paste(sprintf("%.5f", sapply(1:8, function(h) sqrt(1 + sum((al + (1:(h-1))*be)^2)))), collapse=" "), "\n")
cat("alpha-only factor sqrt(1+(h-1)*al^2):",
    paste(sprintf("%.5f", sapply(1:8, function(h) sqrt(1 + (h-1)*al^2))), collapse=" "), "\n")
