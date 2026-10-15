suppressMessages({library(changepoint);library(strucchange)})
y <- read.csv("fxE_n2000.csv")$y; n <- length(y)
cat("n =", n, "  truth (1-based segment ends): 400, 800, 1200, 1600\n\n")
tm <- function(lab, ex){ t<-system.time(r<-try(ex,silent=TRUE))[["elapsed"]]
  cat(sprintf("%-42s %8.3f s   %s\n", lab, t,
      if(inherits(r,"try-error")) "ERR" else paste(r,collapse=","))) }
tm("changepoint cpt.mean MBIC PELT",      cpts(cpt.mean(y,penalty="MBIC",method="PELT")))
tm("changepoint cpt.mean BIC PELT",       cpts(cpt.mean(y,penalty="BIC",method="PELT")))
tm("changepoint cpt.mean Manual=20 PELT", cpts(cpt.mean(y,penalty="Manual",pen.value=20,method="PELT")))
tm("changepoint cpt.mean MBIC BinSeg Q=10",cpts(cpt.mean(y,penalty="MBIC",method="BinSeg",Q=10)))
tm("changepoint cpt.meanvar MBIC PELT",   cpts(cpt.meanvar(y,penalty="MBIC",method="PELT")))
tm("changepoint cpt.mean MBIC SegNeigh Q=6", cpts(cpt.mean(y,penalty="BIC",method="SegNeigh",Q=6)))
tm("strucchange breakpoints(y~1) h=0.15", breakpoints(y~1)$breakpoints)
tm("strucchange Fstats+sctest supF",      {f<-Fstats(y~1); sprintf("%.6f",sctest(f,"supF")$statistic)})
bpf <- breakpoints(y~1)
tm("strucchange confint(bpf)",            as.vector(t(confint(bpf)$confint)))
s<-summary(bpf); cat("strucchange BIC m=0..:", sprintf("%.3f",s$RSS["BIC",]),"\n")
