suppressMessages(library(strucchange))
# Can the supF p-value be reproduced from the Hansen table + documented formula alone?
e <- new.env(); sys.source("../src-strucchange-full/R/pvalue.Fstats.R", e)
beta <- get("sc.beta.sup", e)
myp <- function(x, k, lambda){
  m <- ncol(beta)-1
  tau <- if(lambda<1) lambda else 1/(1+sqrt(lambda))
  b <- beta[(((k-1)*25+1):(k*25)),]
  dummy <- b[,(1:m)] %*% x^(0:(m-1)); dummy <- dummy*(dummy>0)
  pp <- 1 - pchisq(dummy, b[,(m+1)])
  if(tau==0.5) 1-pchisq(x,k)
  else if(tau<=0.01) pp[25]
  else if(tau>=0.49) ((0.5-tau)*pp[1]+(tau-0.49)*(1-pchisq(x,k)))*100
  else { taua<-(0.51-tau)*50; t1<-floor(taua); (t1+1-taua)*pp[t1]+(taua-t1)*pp[t1+1] }
}
cat("### reproducing strucchange supF p-values from the embedded Hansen(1997) table\n")
for(fx in c("fx0_regen","fxB_weak","fxC_null","fxG_many","fxA_var")){
  y <- read.csv(paste0(fx,".csv"))$y
  f <- Fstats(y~1); st <- sctest(f,"supF")
  mine <- myp(as.numeric(st$statistic), f$nreg, f$par)
  cat(sprintf("  %-12s supF=%11.6f  strucchange p=%.10g  reimpl p=%.10g  diff=%.3g\n",
      fx, st$statistic, st$p.value, mine, abs(st$p.value-mine)))
}
cat("\n### supF table shape: ncol(sc.beta.sup)=", ncol(beta), "=> polynomial degree",
    ncol(beta)-2, "+ df column\n")
cat("    rows =", nrow(beta), "= 40 regressors x 25 tau grid points\n")
cat("    (aveF/expF tables have 5 cols => cubic polynomial + df)\n")

cat("\n### strucchange BIC: effective penalty per break, intercept-only\n")
y <- read.csv("fx0_regen.csv")$y; n <- length(y)
bpf <- breakpoints(y~1); s <- summary(bpf); B <- s$RSS["BIC",]; R <- s$RSS["RSS",]
for(m in 0:5){
  lhs <- n*(log(R[m+1])+1-log(n)+log(2*pi)) + log(n)*2*(m+1)
  cat(sprintf("  m=%d RSS=%12.6f  BIC(reported)=%12.6f  BIC(formula)=%12.6f  diff=%.3g\n",
      m, R[m+1], B[m+1], lhs, abs(B[m+1]-lhs)))
}
cat("  => BIC = n*(log(RSS)+1-log n+log 2pi) + log(n)*(nreg+1)*(m+1);",
    "per-break increment = (nreg+1)*log n =", sprintf("%.6f",2*log(n)), "\n")
