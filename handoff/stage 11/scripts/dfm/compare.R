library(dfms); library(collapse); options(digits=16)
for (tag in c("A","B")) {
 r <- if (tag=="A") 1L else 2L
 y <- as.matrix(read.csv(paste0("fixture",tag,".csv"))); X <- fscale(y); n <- ncol(X)
 for (meth in c("DGR","BM")) {
  pre <- paste0("exp_",tag,"_",meth,"_")
  rd <- function(f) as.matrix(read.csv(paste0(pre,f), header=FALSE))
  C<-rd("C.csv"); A<-rd("A.csv"); Q<-rd("Q.csv"); R<-rd("R.csv"); P0<-rd("P0.csv")
  F0 <- matrix(if(meth=="BM") 0 else NA, r, 1)
  if (meth=="DGR") { # DGR uses first row of PCs as F_0; recompute
    ed <- eigen(cov(X), symmetric=TRUE); PCS <- X %*% ed$vectors
    sgn <- c(-1,1)[(colSums(PCS * rowMeans(X)) > 0) + 1L]
    v <- (ed$vectors * rep(sgn, each=nrow(ed$vectors)))[, 1:r, drop=FALSE]
    F0 <- matrix((X %*% v)[1L, ], r, 1)
  }
  # (a) dfms own fixed P0
  la <- SKF(X, A, C, Q, R, F0, P0, TRUE)$loglik
  # (b) stationary P0 implied by FINAL A,Q
  P0s <- matrix(solve(diag(r*r) - kronecker(A,A), as.vector(Q)), r, r)
  lb <- SKF(X, A, C, Q, R, F0, P0s, TRUE)$loglik
  cat(sprintf("%s %-4s  llf[dfms fixed P0]=%.10f   llf[stationary P0]=%.10f   gap=%.3e\n",
              tag, meth, la, lb, la-lb))
  cat("   P0_fixed=", P0, " | P0_stationary(final)=", P0s, "\n")
  write.table(P0s, paste0(pre,"P0stat.csv"), sep=",", row.names=FALSE, col.names=FALSE)
  write.table(F0,  paste0(pre,"F0.csv"),     sep=",", row.names=FALSE, col.names=FALSE)
 }
}
