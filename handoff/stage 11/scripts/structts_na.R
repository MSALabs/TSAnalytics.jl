y <- as.numeric(Nile); y[c(20,40,60)] <- NA
f <- StructTS(ts(y, start=1871), type="level")
n <- length(y); nobs <- sum(!is.na(y))
val <- -(f$loglik + 0.5*nobs*log(2*pi))/n
cat(sprintf("with 3 NAs: n=%d nobs=%d\n", n, nobs))
cat(sprintf("loglik  = %.6f\nloglik0 = %.6f\n", f$loglik, f$loglik0))
cat(sprintf("loglik0 - loglik = %.6f\n", f$loglik0 - f$loglik))
cat(sprintf("predicted if loglik0-loglik = n*log(2pi)+0.5*nobs*log(2pi): %.6f\n",
            n*log(2*pi) + 0.5*nobs*log(2*pi)))
cat(sprintf("coef: %s\n", paste(sprintf("%s=%.4f", names(f$coef), f$coef), collapse=" ")))
