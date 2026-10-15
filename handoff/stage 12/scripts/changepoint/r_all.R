suppressMessages({library(changepoint); library(strucchange)})
P <- function(v) if(length(v)==0) "(none)" else paste(v, collapse=",")
FX <- c("fx0_regen","fxA_var","fxB_weak","fxC_null","fxD_edge","fxF_offgrid","fxG_many","fxH_spike")
for(fx in FX){
  y <- read.csv(paste0(fx,".csv"))$y; n <- length(y)
  cat("\n################ ",fx,"  n=",n,"\n",sep="")
  # ---- changepoint: cpt.mean
  for(cfg in list(c("MBIC","PELT"),c("BIC","PELT"),c("MBIC","BinSeg"),c("BIC","BinSeg"))){
    a <- tryCatch(cpt.mean(y,penalty=cfg[1],method=cfg[2],Q=min(10,floor(n/4))),error=function(e)e)
    cat(sprintf("  cpt.mean %-7s %-7s : %s\n",cfg[1],cfg[2],
        if(inherits(a,"error")) paste("ERR",conditionMessage(a)) else P(cpts(a))))
  }
  for(v in c(10,20,50)){
    a <- cpt.mean(y,penalty="Manual",pen.value=v,method="PELT")
    cat(sprintf("  cpt.mean Manual pen=%-3g PELT : %s\n",v,P(cpts(a))))
  }
  # ---- changepoint: cpt.var / cpt.meanvar
  for(f in c("cpt.var","cpt.meanvar")){
    a <- tryCatch(get(f)(y,penalty="MBIC",method="PELT"),error=function(e)e)
    cat(sprintf("  %-12s MBIC PELT      : %s\n",f,
        if(inherits(a,"error")) paste("ERR",conditionMessage(a)) else P(cpts(a))))
  }
  # ---- changepoint.np
  a <- tryCatch({suppressMessages(library(changepoint.np)); cpt.np(y,penalty="MBIC",method="PELT")},
                error=function(e)e)
  cat(sprintf("  cpt.np       MBIC PELT      : %s\n",
      if(inherits(a,"error")) paste("ERR",conditionMessage(a)) else P(cpts(a))))
  # ---- strucchange
  bpf <- tryCatch(breakpoints(y ~ 1), error=function(e)e)
  if(inherits(bpf,"error")) cat("  strucchange breakpoints     : ERR",conditionMessage(bpf),"\n")
  else {
    cat(sprintf("  sc breakpoints (BIC sel)    : %s   [h=%d, max breaks=%d]\n",
        P(bpf$breakpoints), floor(n*0.15), ncol(bpf$RSS.table)/2))
    s <- summary(bpf)
    cat("  sc BIC table m=0..:", paste(sprintf("%.4f",s$RSS["BIC",]),collapse=" "),"\n")
    cat("  sc RSS table m=0..:", paste(sprintf("%.4f",s$RSS["RSS",]),collapse=" "),"\n")
    ci <- tryCatch(confint(bpf), error=function(e)e, warning=function(w)w)
    if(inherits(ci,"confint.breakpoints")) {
      cat("  sc confint 95%:\n"); print(ci$confint)
    } else cat("  sc confint: ",class(ci)[1],":",conditionMessage(ci),"\n")
  }
  fs <- tryCatch(Fstats(y ~ 1), error=function(e)e)
  if(!inherits(fs,"error")){
    for(tp in c("supF","aveF","expF")){
      st <- sctest(fs, type=tp)
      cat(sprintf("  sc %-5s stat=%14.6f p=%.6g\n",tp,st$statistic,st$p.value))
    }
  }
  for(tp in c("OLS-CUSUM","Rec-CUSUM","Score-CUSUM","OLS-MOSUM","Rec-MOSUM","RE","ME")){
    e <- tryCatch(efp(y ~ 1, type=tp), error=function(er)er)
    if(inherits(e,"error")){ cat(sprintf("  efp %-12s ERR %s\n",tp,conditionMessage(e))); next }
    st <- tryCatch(sctest(e), error=function(er)er)
    cat(sprintf("  efp %-12s stat=%12.6f p=%-12.6g\n",tp,st$statistic,st$p.value))
  }
}
