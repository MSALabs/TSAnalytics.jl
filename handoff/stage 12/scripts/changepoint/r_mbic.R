suppressMessages(library(changepoint))
rss <- function(y) sum((y-mean(y))^2)
crit <- function(y, cp, pen, mbic){           # reconstruct the criterion by hand
  b <- c(0, cp, length(y)); tot <- 0
  for(i in 1:(length(b)-1)){
    seg <- y[(b[i]+1):b[i+1]]
    tot <- tot + rss(seg) + if(mbic) log(length(seg)) else 0
  }
  tot + length(cp)*pen
}
for(fx in c("fxG_many","fx0_regen","fxB_weak")){
  y <- read.csv(paste0(fx,".csv"))$y; n <- length(y)
  cat("=====",fx," n=",n,"  3log(n)=",sprintf("%.6f",3*log(n)),"\n",sep="")
  a <- cpt.mean(y, penalty="MBIC",   method="PELT")
  b <- cpt.mean(y, penalty="BIC",    method="PELT")
  m <- cpt.mean(y, penalty="Manual", pen.value=3*log(n), method="PELT")
  cat(sprintf("  MBIC            (%2d cpts) %s\n", length(cpts(a)), paste(cpts(a),collapse=",")))
  cat(sprintf("  BIC  =2log n    (%2d cpts) %s\n", length(cpts(b)), paste(cpts(b),collapse=",")))
  cat(sprintf("  Manual=3log n   (%2d cpts) %s\n", length(cpts(m)), paste(cpts(m),collapse=",")))
  cat("  MBIC == Manual(3log n)? ", identical(cpts(a),cpts(m)), "\n")
  # hand-reconstructed criterion check
  cat(sprintf("  hand MBIC-crit at MBIC cpts  = %.8f\n", crit(y,cpts(a),3*log(n),TRUE)))
  cat(sprintf("  hand MBIC-crit at Manual cpts= %.8f\n", crit(y,cpts(m),3*log(n),TRUE)))
  cat(sprintf("  hand RSS-crit  at Manual cpts= %.8f\n", crit(y,cpts(m),3*log(n),FALSE)))
  cat(sprintf("  hand RSS-crit  at MBIC cpts  = %.8f\n", crit(y,cpts(a),3*log(n),FALSE)))
  cat(sprintf("  changepoint logLik(MBIC obj) = %s\n", paste(sprintf("%.8f",logLik(a)),collapse=" / ")))
}
