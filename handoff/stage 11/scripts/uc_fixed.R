# Decisive check: R's filter at FIXED variances, innovations + filtered states exported
y <- as.numeric(Nile)
# build the local-level model by hand in R's Kalman representation
mod <- list(phi = matrix(1,1,1), A = matrix(1,1,1), Pn = matrix(1e6,1,1),
            D = matrix(0,0,0), h = 15098.577, V = matrix(1469.147,1,1),
            a = 0, P = matrix(0,1,1), T = matrix(1,1,1), R = matrix(1,1,1))
Z <- list(phi=matrix(1,1,1), A=matrix(1,1,1), Pn=matrix(1e6,1,1), h=15098.577,
          V=matrix(1469.147,1,1), a=0, P=matrix(0,1,1))
r <- KalmanRun(y, Z, nit=-1L, update=FALSE)
cat(sprintf("R KalmanRun: Lik = %.12f  s2 = %.12f\n", r$Lik, r$s2))
cat("R resid[1:5] =", sprintf("%.10f", head(r$resid,5)), "\n")
cat("R states[1:3] =", sprintf("%.10f", head(as.numeric(r$states),3)), "\n")
cat("R states[100] =", sprintf("%.10f", as.numeric(r$states)[100]), "\n")
write.csv(data.frame(resid=r$resid, state=as.numeric(r$states)), "uc_R_filter.csv", row.names=FALSE)
