options(digits=12)
y <- scan("burg_y.csv", quiet=TRUE)

cat("R pacf() IGNORES its own `method` argument -- all four are identical:\n")
for (mth in c("yule-walker","burg","ols","mle")) {
  p <- tryCatch(as.numeric(pacf(y, lag.max=6, plot=FALSE, method=mth)$acf), error=function(e) NA)
  cat(sprintf("  %-12s %s\n", mth, paste(format(round(p,10)), collapse=" ")))
}

cat("\nR's real Burg estimator is ar.burg(), not pacf(). The LAST AR\n")
cat("coefficient of an order-k Burg fit IS the partial autocorrelation at\n")
cat("lag k, so this gives R reference values for pacf(method=:burg):\n")
pk <- sapply(1:6, function(k)
  as.numeric(tail(ar.burg(y, order.max=k, aic=FALSE)$ar, 1)))
cat("  ", format(round(pk,10)), "\n")
