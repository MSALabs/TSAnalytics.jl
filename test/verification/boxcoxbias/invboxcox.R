options(digits=12)
suppressWarnings(suppressMessages(library(forecast)))
cases <- list(list(fc=c(1.0,1.5,2.0), se=c(0.2,0.3,0.4), lam=0.3),
              list(fc=c(2.0,2.5),     se=c(0.5,0.5),     lam=0.0),
              list(fc=c(3.0,4.0),     se=c(0.1,0.2),     lam=1.0),
              list(fc=c(1.2,1.8),     se=c(0.3,0.3),     lam=-0.5),
              list(fc=c(5.0,6.0),     se=c(0.4,0.6),     lam=0.75))
for (c in cases) {
  n <- InvBoxCox(c$fc, lambda=c$lam)
  a <- InvBoxCox(c$fc, lambda=c$lam, biasadj=TRUE, fvar=c$se^2)
  cat(sprintf("lambda=%5.2f\n  naive  : %s\n  biasadj: %s\n",
      c$lam, paste(format(round(n,10)), collapse=" "),
             paste(format(round(a,10)), collapse=" ")))
}
cat("\nbiasadj=TRUE without fvar:\n")
cat("  ", tryCatch({InvBoxCox(c(1.0), lambda=0.3, biasadj=TRUE); "no error"},
                   error=function(e) conditionMessage(e)), "\n")
