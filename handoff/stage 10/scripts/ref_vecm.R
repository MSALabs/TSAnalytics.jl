suppressMessages({library(urca); library(vars); library(tsDyn); library(jsonlite)})
out <- list()
gr <- function(fx, K, r, tag) {
  Y <- as.matrix(read.csv(paste0("fixtures/", fx)))
  j <- ca.jo(Y, type="trace", ecdet="none", K=K, spec="longrun")
  rl <- cajorls(j, r=r)
  cf <- coef(rl$rlm)
  v2 <- vec2var(j, r=r)
  res <- list(K=K, r=r, ncol=ncol(Y), names=colnames(Y),
    cajorls_beta = rl$beta,
    cajorls_coef = cf, cajorls_coef_rownames = rownames(cf), cajorls_coef_colnames = colnames(cf),
    cajorls_resid_head = head(residuals(rl$rlm), 2),
    cajorls_sigma = crossprod(residuals(rl$rlm)) / (nrow(residuals(rl$rlm)) - nrow(cf)),
    vec2var_A = v2$A, vec2var_deterministic = v2$deterministic,
    vec2var_resid_head = head(resid(v2), 2),
    vec2var_fcst = predict(v2, n.ahead=4)$fcst)
  ## tsDyn, both estimators
  for (es in c("2OLS","ML")) {
    m <- tryCatch(tsDyn::VECM(Y, lag=K-1, r=r, estim=es, include="const"),
                  error=function(e) NULL)
    if (!is.null(m)) {
      res[[paste0("tsdyn_", es)]] <- list(coef=coef(m), beta=m$model.specific$beta,
                                          coef_names=colnames(coef(m)))
    } else res[[paste0("tsdyn_", es)]] <- "error"
  }
  out[[tag]] <<- res
}
gr("coint2_yes.csv", 2, 1, "yes_K2_r1")
gr("coint3_r1.csv",  2, 1, "r1_K2_r1")
gr("coint3_r1.csv",  2, 2, "r1_K2_r2")
gr("coint3_r2.csv",  2, 2, "r2_K2_r2")
gr("coint2_yes.csv", 3, 1, "yes_K3_r1")

## reduction: r = K is a VAR in differences (no cointegration restriction)
Y <- as.matrix(read.csv("fixtures/coint2_yes.csv"))
out$reduction_rK <- tryCatch({
  j <- ca.jo(Y, type="trace", ecdet="none", K=2, spec="longrun")
  rl <- cajorls(j, r=2); list(beta=rl$beta, coef=coef(rl$rlm))
}, error=function(e) conditionMessage(e))
out$err_r0 <- tryCatch({j <- ca.jo(Y, type="trace", ecdet="none", K=2); cajorls(j, r=0); "no error"},
                       error=function(e) conditionMessage(e))
out$err_rK1 <- tryCatch({j <- ca.jo(Y, type="trace", ecdet="none", K=2); cajorls(j, r=3); "no error"},
                        error=function(e) conditionMessage(e))
out$meta <- list(R=R.version.string, urca=as.character(packageVersion("urca")),
                 vars=as.character(packageVersion("vars")), tsDyn=as.character(packageVersion("tsDyn")))
write(toJSON(out, digits=17, auto_unbox=TRUE, matrix="rowmajor"), "fixtures/ref_R_vecm.json")
cat("vecm R refs written\n")
