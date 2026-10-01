options(digits=12)
suppressWarnings(suppressMessages(library(forecast)))
fs <- function(y, f, jump1) {
  ty <- ts(y, frequency=f)
  fit <- if (jump1) stl(ty, s.window=11, s.jump=1, t.jump=1, l.jump=1) else stl(ty, s.window=11)
  seas <- as.numeric(fit$time.series[,"seasonal"])
  rem  <- as.numeric(y) - seas - as.numeric(fit$time.series[,"trend"])
  max(0, min(1, 1 - var(rem)/var(rem + seas)))
}
d <- list(cardox240=list(12,read.csv("cardox240.csv")[[1]]), ets_y=list(4,scan("burg_y.csv",quiet=TRUE)),
          airp=list(12,scan("airp.csv",quiet=TRUE)), nile=list(4,scan("nile.csv",quiet=TRUE)))
cat("series       default-jump   jump=1\n")
for (nm in names(d)) cat(sprintf("%-12s %.10f  %.10f\n", nm, fs(d[[nm]][[2]], d[[nm]][[1]], FALSE), fs(d[[nm]][[2]], d[[nm]][[1]], TRUE)))
