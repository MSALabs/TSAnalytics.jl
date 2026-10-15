suppressMessages(library(changepoint))
y <- read.csv("fx0_regen.csv")$y; n <- length(y)
cat("### CROPS on fx0, pen.value = c(log(n), 100)\n")
a <- cpt.mean(y, penalty="CROPS", pen.value=c(log(n),100), method="PELT")
cat("  class:", class(a), "\n")
cat("  pen.value.full:", paste(sprintf("%.4f",pen.value.full(a)),collapse=" "), "\n")
cat("  ncpts for each penalty:", paste(sapply(cpts.full(a),function(r) sum(!is.na(r))),collapse=" "), "\n")
print(cpts.full(a))
cat("\n### SegNeigh (exact DP) vs PELT at BIC, fxB_weak + fxG_many\n")
for(fx in c("fxB_weak","fxG_many")){
  z <- read.csv(paste0(fx,".csv"))$y
  p <- cpts(cpt.mean(z, penalty="BIC", method="PELT"))
  s <- cpts(cpt.mean(z, penalty="BIC", method="SegNeigh", Q=10))
  b <- cpts(cpt.mean(z, penalty="BIC", method="BinSeg", Q=10))
  cat(sprintf("  %-10s PELT=%-42s\n             SegNeigh=%-42s\n             BinSeg=%s\n",
      fx, paste(p,collapse=","), paste(s,collapse=","), paste(b,collapse=",")))
}
