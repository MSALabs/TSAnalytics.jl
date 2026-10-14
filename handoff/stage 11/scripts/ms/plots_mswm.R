suppressMessages(library(MSwM))
d<-read.csv("hamilton_rgnp.csv"); df<-data.frame(y=d$rgnp); m0<-lm(y~1,df)
set.seed(101)
fit<-msmFit(m0,k=2,p=4,sw=c(TRUE,rep(FALSE,4),FALSE),
  control=list(parallelization=FALSE,maxiter=500,tol=1e-10,maxiterOuter=5,maxiterInner=10))
for (nm in c("plotProb","plotReg","plotDiag","plot")) {
  pdf(paste0("mswm_",nm,".pdf"))
  r<-try(switch(nm, plotProb=plotProb(fit), plotReg=plotReg(fit), plotDiag=plotDiag(fit,which=1), plot=plot(fit)), silent=TRUE)
  dev.off()
  cat(sprintf("  %-9s %s\n", nm, if(inherits(r,"try-error")) paste("ERROR:",trimws(r)) else "OK"))
}
cat("\nsummary(fit):\n"); print(summary(fit))
