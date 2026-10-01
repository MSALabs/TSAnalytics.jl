options(digits=12)
set.seed(99)
y <- scan("ets_y.csv", quiet=TRUE)
for (mth in c("yule-walker","burg","ols","mle")) {
  p <- tryCatch(as.numeric(pacf(y, lag.max=6, plot=FALSE, method=mth)$acf), error=function(e) NA)
  cat(sprintf("  %-12s %s\n", mth, paste(format(round(p,10)), collapse=" ")))
}
cat("\nar.burg coefficients (order 3):\n")
cat(" ", format(round(as.numeric(ar.burg(y, order.max=3, aic=FALSE)$ar),10)), "\n")
