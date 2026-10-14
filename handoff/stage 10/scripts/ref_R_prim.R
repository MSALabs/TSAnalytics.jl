suppressMessages({library(vars); library(jsonlite)})
out <- list()
A <- as.matrix(read.csv("fixtures/var2_p1.csv"))
B3 <- as.matrix(read.csv("fixtures/var3_p2.csv"))
U  <- as.matrix(read.csv("fixtures/uschange.csv"))
K1 <- as.numeric(read.csv("fixtures/var1_k1.csv")[,1])

## multi-response OLS: lm with cbind() on the left
mr <- function(Y, p, tag) {
  n <- nrow(Y); K <- ncol(Y); T <- n - p
  X <- Y[(p+1):n, , drop=FALSE]
  Z <- cbind(1, do.call(cbind, lapply(1:p, function(j) Y[(p+1-j):(n-j), , drop=FALSE])))
  colnames(Z) <- c("const", unlist(lapply(1:p, function(j) paste0(colnames(Y), ".l", j))))
  fit <- lm(X ~ Z - 1)
  E <- residuals(fit); B <- coef(fit)
  out[[tag]] <<- list(K=K, p=p, T=T,
    Z_head = head(Z, 3), Z_tail = tail(Z, 2), Z_dim = dim(Z),
    B = B, B_rownames = rownames(B), B_colnames = colnames(B),
    sigma_df = crossprod(E) / (T - ncol(Z)),
    sigma_mle = crossprod(E) / T,
    resid_head = head(E, 2),
    # per-equation SEs, which a multi-response fit must reproduce
    se = sapply(summary(fit), function(s) s$coefficients[, "Std. Error"]))
}
mr(A, 1, "mols_A_p1"); mr(A, 2, "mols_A_p2"); mr(B3, 2, "mols_B_p2"); mr(U, 2, "mols_US_p2")

## k = 1: the primitive must handle it even though fit_var refuses
a1 <- ar.ols(K1, order.max=1, aic=FALSE, demean=FALSE, intercept=TRUE)
a2 <- ar.ols(K1, order.max=2, aic=FALSE, demean=FALSE, intercept=TRUE)
a3 <- ar.ols(K1, order.max=3, aic=FALSE, demean=FALSE, intercept=TRUE)
out$k1_ar_ols <- list(
  p1=list(ar=as.numeric(a1$ar), intercept=as.numeric(a1$x.intercept),
          var.pred=a1$var.pred, asy_se_ar=as.numeric(a1$asy.se.coef$ar),
          asy_se_int=as.numeric(a1$asy.se.coef$x.mean)),
  p2=list(ar=as.numeric(a2$ar), intercept=as.numeric(a2$x.intercept), var.pred=a2$var.pred),
  p3=list(ar=as.numeric(a3$ar), intercept=as.numeric(a3$x.intercept), var.pred=a3$var.pred))

## companion matrix and its eigenvalues, via vars' own roots()
comp <- function(Y, p, tag) {
  v <- VAR(Y, p=p, type="const"); Bc <- Bcoef(v); K <- ncol(Y)
  Amats <- lapply(1:p, function(j) Bc[, ((j-1)*K+1):(j*K), drop=FALSE])
  C <- matrix(0, K*p, K*p)
  C[1:K, ] <- do.call(cbind, Amats)
  if (p > 1) C[(K+1):(K*p), 1:(K*(p-1))] <- diag(K*(p-1))
  ev <- eigen(C, only.values=TRUE)$values
  out[[tag]] <<- list(K=K, p=p, companion=C, eig_mod=sort(Mod(ev), decreasing=TRUE),
                      vars_roots=roots(v), eig_re=Re(ev), eig_im=Im(ev))
}
comp(A, 1, "comp_A_p1"); comp(A, 2, "comp_A_p2"); comp(B3, 2, "comp_B_p2")
comp(A, 4, "comp_A_p4"); comp(U, 2, "comp_US_p2")

## collinearity: a duplicated column
out$err_collinear <- tryCatch({
  Ac <- cbind(A, A[,1]); colnames(Ac) <- c("y1","y2","y1dup")
  v <- VAR(Ac, p=1, type="const")
  paste("no error; Bcoef has NA:", any(is.na(Bcoef(v))))
}, error=function(e) conditionMessage(e))
out$meta <- list(R=R.version.string, vars=as.character(packageVersion("vars")))
write(toJSON(out, digits=17, auto_unbox=TRUE, matrix="rowmajor"), "fixtures/ref_R_prim.json")
cat("primitive R refs written\n")
