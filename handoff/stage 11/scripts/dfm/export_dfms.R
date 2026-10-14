library(dfms); library(collapse)
options(digits=16)
for (tag in c("A","B")) {
  path <- paste0("fixture", tag, ".csv"); r <- if (tag=="A") 1L else 2L
  y <- as.matrix(read.csv(path))
  for (meth in c("DGR","BM")) {
    m <- DFM(y, r=r, p=1L, em.method=meth, max.iter=5000L, tol=1e-10, min.iter=0L)
    pre <- paste0("exp_", tag, "_", meth, "_")
    write.table(unname(m$C), paste0(pre,"C.csv"), sep=",", row.names=FALSE, col.names=FALSE)
    write.table(unname(m$A), paste0(pre,"A.csv"), sep=",", row.names=FALSE, col.names=FALSE)
    write.table(unname(m$Q), paste0(pre,"Q.csv"), sep=",", row.names=FALSE, col.names=FALSE)
    write.table(unname(m$R), paste0(pre,"R.csv"), sep=",", row.names=FALSE, col.names=FALSE)
    write.table(unname(m$P_0),paste0(pre,"P0.csv"), sep=",", row.names=FALSE, col.names=FALSE)
    cat(tag, meth, "logLik=", format(as.numeric(logLik(m)), digits=16), "\n")
  }
}
# Also: evaluate dfms SKF loglik at STATSMODELS' fitted parameters (fixture A)
smC <- matrix(c(0.70090682,0.60429108,-0.57537393,0.52373497,0.66507265), ncol=1)
smA <- matrix(0.64824775,1,1); smQ <- matrix(1,1,1)
smR <- diag(c(0.14812824,0.36549749,0.42432314,0.52214416,0.23249862))
y <- as.matrix(read.csv("fixtureA.csv")); X <- fscale(y)
P0_stat <- matrix(as.numeric(smQ)/(1-as.numeric(smA)^2),1,1)
r1 <- SKF(X, smA, smC, smQ, smR, matrix(0,1,1), P0_stat, TRUE)
cat("\n[dfms SKF at statsmodels params, STATIONARY P0]  loglik =", format(r1$loglik, digits=16), "\n")
cat("   P0_stationary =", P0_stat, "\n")
# and with dfms's own fixed P0 from BM run
P0_bm <- as.matrix(read.csv("exp_A_BM_P0.csv", header=FALSE))
r2 <- SKF(X, smA, smC, smQ, smR, matrix(0,1,1), P0_bm, TRUE)
cat("[dfms SKF at statsmodels params, dfms BM P0=", P0_bm, "] loglik =", format(r2$loglik, digits=16), "\n")
