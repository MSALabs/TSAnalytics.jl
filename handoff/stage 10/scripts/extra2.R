suppressMessages({library(vars); library(jsonlite)})
## Near-unit-root (stable but close to boundary): two independent AR(1) with phi = 0.98
set.seed(20261016)
n <- 220; Yn <- matrix(0,n,2); e <- matrix(rnorm(n*2),n,2)
for (t in 2:n) Yn[t,] <- c(0.98,0.98)*Yn[t-1,] + e[t,]
Yn <- Yn[21:220,]; colnames(Yn) <- c("w1","w2")
write.csv(round(Yn,12),"fixtures/var2_nearunit.csv",row.names=FALSE)
## Explosive: spectral radius > 1, short so it doesn't overflow
set.seed(20261017)
m <- 60; Ye <- matrix(0,m,2); ee <- matrix(rnorm(m*2),m,2)
Aex <- matrix(c(1.05,0.02,0.00,1.03),2,2,byrow=TRUE)
for (t in 2:m) Ye[t,] <- Aex %*% Ye[t-1,] + ee[t,]
colnames(Ye) <- c("v1","v2")
write.csv(round(Ye,12),"fixtures/var2_explosive.csv",row.names=FALSE)

mk <- function(Y,tag){ v <- VAR(Y,p=1,type="const")
  list(roots=roots(v), roots_max=max(roots(v)), stable=max(roots(v))<1,
       Bcoef=Bcoef(v), logLik=as.numeric(logLik(v)),
       Sigma_df=crossprod(resid(v))/(v$obs-ncol(v$datamat)+ncol(Y))) }
out <- list(nearunit=mk(Yn,"nu"), explosive=mk(Ye,"ex"))
A <- as.matrix(read.csv("fixtures/var2_p1.csv"))
out$err_p_too_large <- tryCatch({VAR(A[1:8,],p=5,type="const");"no error"},error=function(e) conditionMessage(e))
out$err_p_zero      <- tryCatch({VAR(A,p=0,type="const");"no error"},error=function(e) conditionMessage(e))
out$err_nan         <- tryCatch({B<-A;B[5,1]<-NA;VAR(B,p=1,type="const");"no error"},error=function(e) conditionMessage(e))
out$err_bad_type    <- tryCatch({VAR(A,p=1,type="bogus");"no error"},error=function(e) conditionMessage(e))
write(toJSON(out,digits=17,auto_unbox=TRUE,matrix="rowmajor"),"fixtures/ref_R_extra.json")
cat("ok\n")
