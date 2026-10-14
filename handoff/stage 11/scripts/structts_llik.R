f <- StructTS(Nile, type="level")
n <- length(Nile)
# reconstruct res$value from reported loglik
val <- -(f$loglik + 0.5*n*log(2*pi))/n
cat(sprintf("n = %d\nres$value (reconstructed) = %.10f\n", n, val))
cat(sprintf("loglik  reported = %.6f\n", f$loglik))
cat(sprintf("loglik0 reported = %.6f\n", f$loglik0))
cat(sprintf("loglik0 - loglik = %.6f   ; 1.5*n*log(2pi) = %.6f\n",
            f$loglik0 - f$loglik, 1.5*n*log(2*pi)))
# What the C routine returns at the optimum
Z <- f$model0
vx <- var(Nile)/100
cf <- f$coef/vx
Z$V[1,1] <- cf[1]*vx; Z$h <- cf[2]*vx
z <- .Call(stats:::C_KalmanLike, as.numeric(Nile), Z, -1L, FALSE, FALSE)
cat("C_KalmanLike returns:", sprintf("%.10f", z), "\n")
cat(sprintf("0.5*sum(z) = %.10f\n", 0.5*sum(z)))
# true gaussian loglik from the same filter run, computed by hand from KalmanRun
r <- KalmanRun(as.numeric(Nile), Z, -1L, update=FALSE)
cat("KalmanRun$Lik =", sprintf("%.10f", r$Lik), " s2 =", sprintf("%.10f", r$s2), "\n")
# scale check: nu
cat(sprintf("\nIf true llf = -nu*val - 0.5*n*log(2pi) for nu = n-1: %.6f\n",
            -(n-1)*val - 0.5*n*log(2*pi)))
cat(sprintf("statsmodels true llf at these params        : %.6f\n", -632.537695))
