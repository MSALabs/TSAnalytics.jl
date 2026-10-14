suppressMessages(library(MSwM))
d<-read.csv("hamilton_rgnp.csv"); df<-data.frame(y=d$rgnp); m0<-lm(y~1,df)
sw<-c(TRUE,rep(FALSE,4),FALSE)
cat("parallelization=TRUE, set.seed(101) each time:\n")
for (i in 1:3) {
  set.seed(101)
  f<-try(msmFit(m0,k=2,p=4,sw=sw,control=list(parallelization=TRUE,maxiter=500,tol=1e-10,maxiterOuter=5,maxiterInner=10)),silent=TRUE)
  if (inherits(f,"try-error")) cat("  run",i,"FAILED\n") else
   cat(sprintf("  run %d loglik=%.8f int=[%+.6f,%+.6f] P11=%.6f\n",i,-f@Fit@logLikel,f@Coef[1,1],f@Coef[2,1],f@transMat[1,1]))
}
cat("parallelization=FALSE, set.seed(101) each time (reproducible?):\n")
for (i in 1:2) {
  set.seed(101)
  f<-msmFit(m0,k=2,p=4,sw=sw,control=list(parallelization=FALSE,maxiter=500,tol=1e-10,maxiterOuter=5,maxiterInner=10))
  cat(sprintf("  run %d loglik=%.8f int=[%+.6f,%+.6f] P11=%.6f\n",i,-f@Fit@logLikel,f@Coef[1,1],f@Coef[2,1],f@transMat[1,1]))
}
cat("no set.seed, parallelization=FALSE (5 runs):\n")
for (i in 1:5) {
  f<-try(msmFit(m0,k=2,p=4,sw=sw,control=list(parallelization=FALSE,maxiter=500,tol=1e-10,maxiterOuter=5,maxiterInner=10)),silent=TRUE)
  if (inherits(f,"try-error")) cat("  run",i,"FAILED\n") else
  cat(sprintf("  run %d loglik=%.8f int=[%+.6f,%+.6f]\n",i,-f@Fit@logLikel,f@Coef[1,1],f@Coef[2,1]))
}
