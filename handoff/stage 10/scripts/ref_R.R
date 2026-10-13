suppressMessages({library(vars); library(jsonlite)})
out <- list()

grab <- function(Y, p, type, tag, h=5) {
  v <- VAR(Y, p=p, type=type)
  K <- ncol(Y)
  B <- Bcoef(v)                          # K x (K*p + n.det)
  co <- lapply(v$varresult, function(e) coef(summary(e)))
  pr <- predict(v, n.ahead=h, ci=0.95)
  res <- list(
    K = K, p = p, type = type,
    obs = v$obs, totobs = v$totobs,
    Bcoef = B, Bcoef_colnames = colnames(B), Bcoef_rownames = rownames(B),
    Sigma_df = crossprod(resid(v)) / (v$obs - ncol(v$datamat) + K),   # vars' own summary() Sigma
    Sigma_ols = crossprod(resid(v)) / (v$obs - (K*p + ifelse(type=="const",1,ifelse(type=="both",2,ifelse(type=="trend",1,0))))),
    Sigma_mle = crossprod(resid(v)) / v$obs,
    logLik = as.numeric(logLik(v)),
    coef_est = lapply(co, function(m) m[, "Estimate"]),
    coef_se  = lapply(co, function(m) m[, "Std. Error"]),
    coef_t   = lapply(co, function(m) m[, "t value"]),
    resid_head = head(resid(v), 3), resid_tail = tail(resid(v), 3),
    fitted_head = head(fitted(v), 3),
    roots = roots(v), roots_mod_max = max(roots(v)),
    fcst_point = sapply(pr$fcst, function(f) f[, "fcst"]),
    fcst_lower = sapply(pr$fcst, function(f) f[, "lower"]),
    fcst_upper = sapply(pr$fcst, function(f) f[, "upper"]),
    fcst_CI    = sapply(pr$fcst, function(f) f[, "CI"])
  )
  out[[tag]] <<- res
}

sel <- function(Y, lag.max, type, tag) {
  s <- VARselect(Y, lag.max=lag.max, type=type)
  out[[tag]] <<- list(selection = s$selection, criteria = s$criteria,
                      rownames = rownames(s$criteria), lag.max = lag.max, type = type)
}

A <- as.matrix(read.csv("fixtures/var2_p1.csv"))
B3 <- as.matrix(read.csv("fixtures/var3_p2.csv"))
K1 <- as.matrix(read.csv("fixtures/var1_k1.csv"))
SH <- as.matrix(read.csv("fixtures/var2_short.csv"))
US <- as.matrix(read.csv("fixtures/uschange.csv"))
E5 <- as.matrix(read.csv("fixtures/econ5.csv"))

grab(A,  1, "const", "A_p1_const")
grab(A,  2, "const", "A_p2_const")
grab(A,  1, "none",  "A_p1_none")
grab(A,  1, "trend", "A_p1_trend")
grab(A,  1, "both",  "A_p1_both")
grab(B3, 2, "const", "B_p2_const")
grab(B3, 3, "const", "B_p3_const")
grab(SH, 2, "const", "SH_p2_const")
grab(US, 2, "const", "US_p2_const", h=8)
grab(E5, 2, "const", "E5_p2_const", h=4)

sel(A,  8, "const", "sel_A_const")
sel(A,  8, "none",  "sel_A_none")
sel(A,  8, "trend", "sel_A_trend")
sel(A,  8, "both",  "sel_A_both")
sel(B3, 8, "const", "sel_B_const")
sel(US, 8, "const", "sel_US_const")
sel(E5, 6, "const", "sel_E5_const")


## k = 1: both references REFUSE a univariate VAR -- capture the messages and an
## ar.ols reference, which is exactly what a k=1 VAR(p) with a constant is.
out[["k1_refusal_R"]] <- tryCatch({VAR(K1, p=1, type="const"); "no error"},
                                  error=function(e) conditionMessage(e))
a1 <- ar.ols(as.numeric(K1), order.max=1, aic=FALSE, demean=FALSE, intercept=TRUE)
a2 <- ar.ols(as.numeric(K1), order.max=2, aic=FALSE, demean=FALSE, intercept=TRUE)
out[["k1_ar_ols"]] <- list(
  p1 = list(ar = as.numeric(a1$ar), intercept = as.numeric(a1$x.intercept),
            var.pred = a1$var.pred, asy.se = as.numeric(a1$asy.se.coef$ar),
            se_intercept = as.numeric(a1$asy.se.coef$x.mean)),
  p2 = list(ar = as.numeric(a2$ar), intercept = as.numeric(a2$x.intercept),
            var.pred = a2$var.pred))
out[["meta"]] <- list(R = R.version.string, vars = as.character(packageVersion("vars")))
write(toJSON(out, digits = 17, auto_unbox = TRUE, matrix = "rowmajor"), "fixtures/ref_R.json")
cat("R refs written\n")
