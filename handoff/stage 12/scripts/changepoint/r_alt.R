suppressMessages({library(ecp);library(wbs);library(fpop);library(Segmentor3IsBack);library(changepoint.np)})
for(fx in c("fx0_regen","fxF_offgrid","fxB_weak","fxA_var")){
  y <- read.csv(paste0(fx,".csv"))$y; n <- length(y)
  cat("\n####",fx,"n=",n,"\n")
  r <- try(e.divisive(as.matrix(y), min.size=2, R=199, alpha=1)$estimates, silent=TRUE)
  cat("  ecp::e.divisive  estimates   :", if(inherits(r,"try-error")) "ERR" else paste(r,collapse=","),
      " (incl. 1 and n+1)\n")
  r <- try({w<-wbs(y); cp<-changepoints(w); cp$cpt.ic$mbic.penalty}, silent=TRUE)
  cat("  wbs::changepoints mbic       :", if(inherits(r,"try-error")) "ERR" else paste(sort(r),collapse=","),"\n")
  r <- try({w<-wbs(y); sort(changepoints(w)$cpt.ic$ssic.penalty)}, silent=TRUE)
  cat("  wbs::changepoints ssic       :", if(inherits(r,"try-error")) "ERR" else paste(r,collapse=","),"\n")
  r <- try(Fpop(y, lambda=2*log(n))$t.est, silent=TRUE)
  cat("  fpop::Fpop lambda=2log n     :", if(inherits(r,"try-error")) "ERR" else paste(r,collapse=","),
      " (incl. n)\n")
  r <- try(Fpop(y, lambda=3*log(n))$t.est, silent=TRUE)
  cat("  fpop::Fpop lambda=3log n     :", if(inherits(r,"try-error")) "ERR" else paste(r,collapse=","),"\n")
  r <- try({s<-Segmentor(y, model=2, Kmax=6); getBreaks(s)[3,]}, silent=TRUE)
  cat("  Segmentor3IsBack K=3 breaks  :", if(inherits(r,"try-error")) "ERR" else paste(r,collapse=","),"\n")
  r <- try(cpts(cpt.np(y, penalty="MBIC", method="PELT")), silent=TRUE)
  cat("  changepoint.np cpt.np MBIC   :", if(inherits(r,"try-error")) "ERR" else paste(r,collapse=","),"\n")
}
