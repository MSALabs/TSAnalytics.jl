suppressMessages({library(changepoint);library(strucchange)})
y <- read.csv("fx0_regen.csv")$y; n <- length(y)

cat("### D1: is pen.value SILENTLY IGNORED when penalty is a named type?\n")
for(pv in c(0, 1, 1e6)){
  a <- cpt.mean(y, penalty="BIC", pen.value=pv, method="PELT")
  cat(sprintf("  penalty=BIC pen.value=%-8g -> effective pen=%.6f cpts=%s\n",
      pv, pen.value(a), paste(cpts(a),collapse=",")))
}
cat("  => pen.value is accepted without warning and discarded.\n")

cat("\n### D2: MOSUM p-value floor at exactly 0.01 (rule=2 extrapolation)\n")
big <- c(rep(0,150), rep(50,150))          # enormous break
for(tp in c("OLS-MOSUM","Rec-MOSUM","ME","OLS-CUSUM","Rec-CUSUM","RE")){
  e <- efp(big ~ 1, type=tp); s <- sctest(e)
  cat(sprintf("  %-11s stat=%12.4f p=%.10g\n", tp, s$statistic, s$p.value))
}
cat("  => MOSUM/ME saturate at p=0.01 exactly; CUSUM/RE give real tail values.\n")

cat("\n### D3: Asymptotic penalty warns for multiple changes but proceeds\n")
r <- withCallingHandlers(
  cpt.mean(y, penalty="Asymptotic", pen.value=0.05, method="PELT"),
  warning=function(w){cat("  WARNING:",conditionMessage(w),"\n"); invokeRestart("muffleWarning")})
cat("  cpts=",paste(cpts(r),collapse=","),"  pen=",pen.value(r),"\n")

cat("\n### D4: BinSeg hitting Q is a WARNING, not an error\n")
r <- withCallingHandlers(cpt.mean(y, method="BinSeg", Q=2),
  warning=function(w){cat("  WARNING:",conditionMessage(w),"\n"); invokeRestart("muffleWarning")})
cat("  cpts=",paste(cpts(r),collapse=","),"\n")

cat("\n### D5: strucchange breaks= silently truncated to ceiling(n/h)-2\n")
r <- withCallingHandlers(breakpoints(y ~ 1, breaks=20),
  warning=function(w){cat("  WARNING:",conditionMessage(w),"\n"); invokeRestart("muffleWarning")})
cat("  breakpoints=",paste(r$breakpoints,collapse=","),"\n")

cat("\n### D6: reproducibility - 5 identical reruns?\n")
s1 <- replicate(5, paste(cpts(cpt.mean(y,penalty="MBIC",method="PELT")),collapse=","))
s2 <- replicate(5, paste(breakpoints(y~1)$breakpoints,collapse=","))
cat("  cpt.mean    unique results:", length(unique(s1)), "->", unique(s1), "\n")
cat("  breakpoints unique results:", length(unique(s2)), "->", unique(s2), "\n")

cat("\n### D7: k>40 regressors in supF p-value => WARNING, capped\n")
set.seed(9); X <- matrix(rnorm(300*45),300); yy <- rnorm(300)
r <- withCallingHandlers(tryCatch(sctest(Fstats(yy ~ X), type="supF"),error=function(e)e),
  warning=function(w){cat("  WARNING:",conditionMessage(w),"\n"); invokeRestart("muffleWarning")})
cat("  result:", if(inherits(r,"error")) paste("ERR",conditionMessage(r)) else sprintf("stat=%.4f p=%.6g",r$statistic,r$p.value),"\n")

cat("\n### D8: Fstats trimming defaults (n=300)\n")
f <- Fstats(y ~ 1)
cat(sprintf("  length(Fstats)=%d  from=%d to=%d  lambda=%.6f  tau=1/(1+sqrt(lambda))=%.6f\n",
   length(f$Fstats), 300-length(f$Fstats)-44, 255, f$par, 1/(1+sqrt(f$par))))
cat("  sup attained at breakpoint =", f$breakpoint, "\n")
cat("  efp h default =", deparse(formals(efp)$h), "; Fstats from default =", deparse(formals(Fstats)$from),
    "; breakpoints h default =", deparse(formals(breakpoints.formula)$h), "\n")
