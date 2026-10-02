options(digits=12)
suppressWarnings(suppressMessages(library(forecast)))
y <- scan("ets_y.csv", quiet=TRUE); ty <- ts(y, frequency=4)
for (sp in list(c("ANN","FALSE"), c("AAN","FALSE"), c("AAA","FALSE"), c("AAA","TRUE"))) {
  f <- ets(ty, model=sp[1], damped=as.logical(sp[2]))
  fc <- forecast(f, h=8, level=c(80,95))
  cat(sprintf("%s damped=%s\n", sp[1], sp[2]))
  cat("  point:", format(round(as.numeric(fc$mean),8)), "\n")
  cat("  lo95 :", format(round(as.numeric(fc$lower[,2]),8)), "\n")
  cat("  sigma2:", format(round(f$sigma2,10)), "\n")
}
