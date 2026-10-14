y <- as.numeric(Nile)
f <- StructTS(ts(y, start=1871), type="level")
Z <- f$model0; str(Z, max.level=1)
vx <- var(y)/100
Z$V[1,1] <- f$coef[["level"]]; Z$h <- f$coef[["epsilon"]]
cat(sprintf("\nZ$P = %.6g   Z$Pn = %.6g   Z$a = %.6g\n", Z$P[1,1], Z$Pn[1,1], Z$a[1]))
r <- KalmanRun(y, Z, nit=-1L, update=FALSE)
cat(sprintf("Lik = %.12f  s2 = %.12f\n", r$Lik, r$s2))
cat("resid[1:4] =", sprintf("%.10f ", head(r$resid,4)), "\n")
cat("states[1:3] =", sprintf("%.10f ", head(as.numeric(r$states),3)), "\n")
sm <- tsSmooth(f)
write.csv(data.frame(resid=as.numeric(r$resid), filt=as.numeric(r$states),
                     smooth=as.numeric(sm)), "uc_R_filter.csv", row.names=FALSE)
cat(sprintf("\nvariances used: V=%.6f h=%.6f\n", Z$V[1,1], Z$h))
