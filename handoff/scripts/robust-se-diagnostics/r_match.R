suppressMessages(library(rugarch))
e <- as.numeric(read.csv("garch_bench_data.csv", header=FALSE)[,1])
spec <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                   mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model="norm")
runfit <- function(lbl, ctl) {
  f <- tryCatch(ugarchfit(spec, e, solver="solnp", fit.control=ctl), error=function(err) NULL)
  if (is.null(f)) { cat(sprintf("%-28s FAILED\n", lbl)); return(invisible()) }
  cf <- coef(f)
  cat(sprintf("%-28s omega=%.8f alpha=%.8f beta=%.8f  LL=%.6f\n",
              lbl, cf["omega"], cf["alpha1"], cf["beta1"], likelihood(f)))
}
runfit("default (rec.init='all')", list())
runfit("rec.init='var'",           list(rec.init="var"))
runfit("rec.init=0.9504096717553657", list(rec.init=0.9504096717553657))
runfit("rec.init=1.9016916512549",  list(rec.init=1.9016916512549003))
