suppressMessages(library(changepoint))
y <- read.csv("fx0_regen.csv")$y; n <- length(y)
cat("n=",n," log(n)=",sprintf("%.10f",log(n)),"\n\n",sep="")

cat("### penalty_decision(n=300, diffparam=1, method=PELT)\n")
for(p in c("SIC0","BIC0","SIC","BIC","MBIC","AIC0","AIC","Hannan-Quinn0","Hannan-Quinn","None")){
  v <- tryCatch(penalty_decision(p, 0, n, 1, asymcheck="mean.norm", method="PELT"),
                error=function(e) NA)
  cat(sprintf("%-16s %18.10f\n", p, v))
}
cat("\n### deprecated / error cases\n")
for(p in c("SIC1","BIC1","AIC1","Nonsense")){
  cat(sprintf("%-10s -> %s\n", p, tryCatch({penalty_decision(p,0,n,1,"mean.norm","PELT");"OK"},
        error=function(e) paste("ERROR:",conditionMessage(e)))))
}

cat("\n### cpt.mean PELT, named penalties, fx0 (minseglen default=1)\n")
for(p in c("MBIC","BIC","SIC","SIC0","AIC","AIC0","Hannan-Quinn","None")){
  a <- cpt.mean(y, penalty=p, method="PELT")
  cat(sprintf("%-14s pen.value=%12.6f cpts=%s\n", p, pen.value(a), paste(cpts(a),collapse=",")))
}

cat("\n### Manual at the equivalent numeric values (does Manual reproduce the named penalty?)\n")
eq <- list(BIC=2*log(n), SIC0=1*log(n), MBIC=3*log(n), AIC=4, "Hannan-Quinn"=4*log(log(n)))
for(nm in names(eq)){
  v <- eq[[nm]]
  a <- cpt.mean(y, penalty="Manual", pen.value=v, method="PELT")
  b <- cpt.mean(y, penalty=nm, method="PELT")
  cat(sprintf("%-14s value=%12.6f  Manual cpts=%-20s named cpts=%-20s MATCH=%s\n",
      nm, v, paste(cpts(a),collapse=","), paste(cpts(b),collapse=","),
      identical(cpts(a),cpts(b))))
}

cat("\n### pen.value=10/20/50 Manual, PELT and BinSeg\n")
for(v in c(10,20,50)){
  a <- cpt.mean(y, penalty="Manual", pen.value=v, method="PELT")
  cat(sprintf("Manual pen=%-4g PELT cpts=%s\n", v, paste(cpts(a),collapse=",")))
}
b2 <- cpt.mean(y, method="BinSeg", Q=2); cat("BinSeg Q=2 (MBIC default) cpts=", paste(cpts(b2),collapse=","), "\n")
for(v in c(10,20,50)){
  a <- cpt.mean(y, penalty="Manual", pen.value=v, method="BinSeg", Q=5)
  cat(sprintf("Manual pen=%-4g BinSeg Q=5 cpts=%s\n", v, paste(cpts(a),collapse=",")))
}

cat("\n### minseglen defaults (from formals)\n")
for(f in c("cpt.mean","cpt.var","cpt.meanvar")){
  cat(sprintf("%-12s minseglen default = %s\n", f, deparse(formals(get(f))$minseglen)))
}
cat("cpt.reg      minseglen default =", deparse(formals(cpt.reg)$minseglen), "\n")
