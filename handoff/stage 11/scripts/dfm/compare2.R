library(dfms); library(collapse); options(digits=16)
for (tag in c("A","B")) {
 r <- if (tag=="A") 1L else 2L
 y <- as.matrix(read.csv(paste0("fixture",tag,".csv"))); X <- fscale(y)
 for (meth in c("DGR","BM")) {
  m <- DFM(y, r=r, p=1L, em.method=meth, max.iter=5000L, tol=1e-10, min.iter=0L)
  s <- m$ss_full
  rep_ll <- as.numeric(logLik(m))
  # llf at FINAL params with EM-estimated initial condition (ss_full F_0,P_0)
  l_em  <- SKF(X, s$A, s$C, s$Q, s$R, matrix(s$F_0,ncol=1), s$P_0, TRUE)$loglik
  # llf at FINAL params with STATIONARY init implied by final A,Q
  rp <- nrow(s$A)
  P0s <- matrix(solve(diag(rp*rp) - kronecker(s$A,s$A), as.vector(s$Q)), rp, rp)
  l_st  <- SKF(X, s$A, s$C, s$Q, s$R, matrix(0,rp,1), P0s, TRUE)$loglik
  cat(sprintf("%s %-4s  reported=%.10f  refit[EM init]=%.10f  refit[stationary init]=%.10f\n",
              tag, meth, rep_ll, l_em, l_st))
  cat(sprintf("        ss_full F_0=%s  P_0=%s\n", paste(round(s$F_0,8),collapse=","), paste(round(as.vector(s$P_0),8),collapse=",")))
  cat(sprintf("        n_em_iter=%d   loglik path tail: %s\n", length(m$loglik),
              paste(sprintf("%.8f", tail(m$loglik,3)), collapse=" -> ")))
  # export final state-space for python cross-check
  pre <- paste0("ss_",tag,"_",meth,"_")
  for (nmv in c("A","C","Q","R")) write.table(unname(s[[nmv]]), paste0(pre,nmv,".csv"), sep=",", row.names=FALSE, col.names=FALSE)
  write.table(matrix(s$F_0,ncol=1), paste0(pre,"F0.csv"), sep=",", row.names=FALSE, col.names=FALSE)
  write.table(s$P_0, paste0(pre,"P0.csv"), sep=",", row.names=FALSE, col.names=FALSE)
  write.csv(m$F_qml, paste0(pre,"Fqml.csv"), row.names=FALSE)
 }
}
