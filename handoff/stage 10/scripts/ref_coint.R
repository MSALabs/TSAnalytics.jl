suppressMessages({library(urca); library(tseries); library(jsonlite)})
out <- list(); urcval <- urca:::.urcval

## --- MacKinnon cointegration tables, N = 1..12, all four trend cases ---
tabs <- list()
itvmap <- c(n=1, c=2, ct=3, ctt=4)
for (tn in names(itvmap)) {
  itv <- itvmap[[tn]]
  for (niv in 1:12) {
    key <- sprintf("%s_N%d", tn, niv)
    crit <- sapply(c(0.01,0.05,0.10), function(a) urcval(a, nobs=250, niv=niv, itt=1, itv=itv, nc=1))
    crit_asym <- sapply(c(0.01,0.05,0.10), function(a) urcval(a, nobs=0, niv=niv, itt=1, itv=itv, nc=1))
    pv <- sapply(c(-5.0,-4.0,-3.5,-3.3670,-3.0,-2.0,-1.0,0.0), function(s)
                 urcval(s, nobs=250, niv=niv, itt=1, itv=itv, nc=2))
    tabs[[key]] <- list(crit_n250=crit, crit_asym=crit_asym, pvals=pv)
  }
}
out$mackinnon_tables <- tabs
out$mackinnon_stats_grid <- c(-5.0,-4.0,-3.5,-3.3670,-3.0,-2.0,-1.0,0.0)

## --- Phillips-Ouliaris: tseries::po.test and urca::ca.po ---
po <- function(fx, tag) {
  Y <- as.matrix(read.csv(paste0("fixtures/", fx)))
  r <- list(m=ncol(Y))
  for (dem in c(FALSE, TRUE)) {
    p <- suppressWarnings(po.test(Y, demean=dem, lshort=TRUE))
    q <- suppressWarnings(po.test(Y, demean=dem, lshort=FALSE))
    r[[paste0("po_demean", as.integer(dem), "_lshort")]] <-
      list(stat=as.numeric(p$statistic), p=as.numeric(p$p.value), par=as.numeric(p$parameter))
    r[[paste0("po_demean", as.integer(dem), "_llong")]] <-
      list(stat=as.numeric(q$statistic), p=as.numeric(q$p.value), par=as.numeric(q$parameter))
  }
  for (ty in c("Pu","Pz")) for (dm in c("none","constant","trend")) {
    cp <- ca.po(Y, demean=dm, type=ty)
    r[[paste0("capo_", ty, "_", dm)]] <- list(stat=as.numeric(cp@teststat), cval=as.numeric(cp@cval))
  }
  out[[tag]] <<- r
}
po("coint2_yes.csv","po_yes"); po("coint2_no.csv","po_no")
po("coint3_r1.csv","po_r1");  po("coint2_trend.csv","po_trend")

## --- a hand-rolled Engle-Granger in R, to cross-check statsmodels' aeg ---
eg <- function(fx, trend, tag) {
  Y <- as.matrix(read.csv(paste0("fixtures/", fx)))
  y <- Y[,1]; X <- Y[,-1,drop=FALSE]
  Z <- switch(trend, "n"=X, "c"=cbind(1,X), "ct"=cbind(1,seq_along(y),X))
  u <- residuals(lm(y ~ Z - 1))
  a <- ur.df(u, type="none", lags=12, selectlag="AIC")   # EG step 2: no constant
  out[[tag]] <<- list(trend=trend, N=ncol(Y), lags=a@lags,
                      adf_stat=as.numeric(a@teststat)[1],
                      resid_head=head(u,3), nobs=length(y))
}
eg("coint2_yes.csv","c","eg_yes_c"); eg("coint2_no.csv","c","eg_no_c")
eg("coint3_r1.csv","c","eg_r1_c");  eg("coint2_trend.csv","ct","eg_trend_ct")
eg("coint2_yes.csv","n","eg_yes_n")

out$err_po_one_col <- tryCatch({po.test(as.matrix(read.csv("fixtures/var1_k1.csv"))); "no error"},
                               error=function(e) conditionMessage(e))
out$err_capo_7col  <- tryCatch({ca.po(as.matrix(read.csv("fixtures/climhyd.csv")), type="Pz"); "no error"},
                               error=function(e) conditionMessage(e))
out$meta <- list(R=R.version.string, urca=as.character(packageVersion("urca")),
                 tseries=as.character(packageVersion("tseries")))
write(toJSON(out, digits=17, auto_unbox=TRUE, matrix="rowmajor"), "fixtures/ref_R_coint.json")
cat("coint R refs written\n")
