options(digits=12)
suppressMessages(library(rugarch))
e <- scan("garch_t.csv", quiet=TRUE)
cat("n =", length(e), "\n")
spec <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                   mean.model=list(armaOrder=c(0,0), include.mean=FALSE),
                   distribution.model="std")
for (ri in list(list(l="default 'all'", v=NULL), list(l="rec.init=0.94", v=0.94))) {
  ctl <- if (is.null(ri$v)) list() else list(rec.init=ri$v)
  f <- ugarchfit(spec, e, solver="hybrid", fit.control=ctl)
  cf <- coef(f)
  cat(sprintf("%-15s omega=%.10f alpha=%.10f beta=%.10f shape=%.8f LL=%.8f\n",
              ri$l, cf["omega"], cf["alpha1"], cf["beta1"], cf["shape"], likelihood(f)))
}
cat("\nnormal-dist fit for comparison:\n")
spn <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                  mean.model=list(armaOrder=c(0,0), include.mean=FALSE),
                  distribution.model="norm")
fn <- ugarchfit(spn, e, solver="hybrid", fit.control=list(rec.init=0.94))
cat(sprintf("  rec.init=0.94  omega=%.10f alpha=%.10f beta=%.10f LL=%.8f\n",
            coef(fn)["omega"], coef(fn)["alpha1"], coef(fn)["beta1"], likelihood(fn)))
