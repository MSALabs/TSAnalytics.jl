library(dfms); library(collapse)
options(digits=12)
for (tag in c("A","B")) {
  path <- paste0("fixture", tag, ".csv"); r <- if (tag=="A") 1L else 2L
  y <- as.matrix(read.csv(path))
  cat(strrep("#",90), "\n### FIXTURE ", tag, " T=", nrow(y), " N=", ncol(y), " r=", r, "\n", sep="")
  # verify fscale == (y-mean)/sd(ddof=1)
  zs <- fscale(y); zmine <- scale(y)
  cat("  max|fscale - base::scale| =", max(abs(zs - zmine)), "  (base scale uses n-1)\n")
  for (meth in c("none","DGR","BM")) {
    m <- DFM(y, r=r, p=1L, em.method=meth, max.iter=5000L, tol=1e-10, min.iter=0L)
    cat("\n  --- em.method=", meth, " ---\n", sep="")
    cat("  logLik =", format(as.numeric(logLik(m)), digits=14), "\n")
    cat("  converged =", m$converged, " num_iter =", m$num_iter, "\n")
    cat("  C (loadings, n x r):\n"); print(unname(m$C))
    cat("  A (factor VAR):\n");      print(unname(m$A))
    cat("  Q:\n");                   print(unname(m$Q))
    cat("  diag(R) =", diag(m$R), "\n")
    Fs <- if (meth=="none") m$F_2s else m$F_kal
    if (is.null(Fs)) Fs <- m$F_2s
    cat("  cov(F_smooth):\n"); print(cov(Fs))
    cat("  t(C)%*%C:\n"); print(crossprod(unname(m$C)))
    cat("  F_smooth first 3:\n"); print(head(Fs,3))
    write.csv(Fs, paste0("dfms_", tag, "_", meth, "_Fsmooth.csv"), row.names=FALSE)
    cat("  names(m) =", paste(names(m), collapse=", "), "\n")
  }
}
