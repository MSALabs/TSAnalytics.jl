options(digits=12)
suppressWarnings(suppressMessages(library(forecast)))

series <- list(
  cardox240 = list(f=12, file="cardox240.csv", skip=1),
  ets_y     = list(f=4,  file="burg_y.csv",   skip=0),
  airp      = list(f=12, file="airp.csv",     skip=0),
  nile      = list(f=4,  file="nile.csv",     skip=0)
)

for (nm in names(series)) {
  s <- series[[nm]]
  y <- if (s$skip > 0) read.csv(s$file)[[1]] else scan(s$file, quiet=TRUE)
  ty <- ts(y, frequency=s$f)
  cat("=== ", nm, "  n=", length(y), " m=", s$f, "\n", sep="")
  sh <- forecast:::seas.heuristic(ty)
  cat("  seas.heuristic   :", format(round(sh, 10)), "\n")
  cat("  nsdiffs(seas)    :", nsdiffs(ty, test="seas"), "\n")
  o <- forecast:::ocsb.test(ty, maxlag=3, lag.method="AIC")
  cat("  ocsb stat        :", format(round(as.numeric(o$statistics),10)), "\n")
  cat("  ocsb critical    :", format(round(as.numeric(o$critical),10)), "\n")
  cat("  ocsb lag.order   :", o$lag.order, "\n")
  cat("  nsdiffs(ocsb)    :", nsdiffs(ty, test="ocsb"), "\n")
  o0 <- forecast:::ocsb.test(ty, maxlag=0, lag.method="fixed")
  cat("  ocsb stat (fixed0):", format(round(as.numeric(o0$statistics),10)), "\n")
}
cat("\ncritical values by period:\n")
for (m in c(2,4,6,7,12,24,52)) cat("  m=", m, " -> ", format(round(forecast:::calcOCSBCritVal(m),10)), "\n", sep="")
