suppressMessages({library(urca); library(tseries)})
egfix <- function(fx, trend, lag) {
  Y <- as.matrix(read.csv(paste0("fixtures/", fx)))
  y <- Y[,1]; X <- Y[,-1,drop=FALSE]
  Z <- switch(trend, "n"=X, "c"=cbind(1,X), "ct"=cbind(1,seq_along(y),X))
  u <- residuals(lm(y ~ Z - 1))
  a <- ur.df(u, type="none", lags=lag, selectlag="Fixed")
  cat(sprintf("%-18s trend=%-3s lag=%d  ADF stat = %.12f\n", fx, trend, lag, as.numeric(a@teststat)[1]))
}
for (fx in c("coint2_yes.csv","coint2_no.csv")) for (lg in c(4)) egfix(fx,"c",lg)
egfix("coint3_r1.csv","c",4); egfix("coint2_trend.csv","ct",4); egfix("coint2_yes.csv","n",4)
cat("\n-- ca.po beyond its table (m = 7) --\n")
set.seed(7); M7 <- matrix(cumsum(rnorm(250*7)), 250, 7); colnames(M7) <- paste0("v",1:7)
r <- tryCatch({x <- ca.po(M7, type="Pz"); paste("no error; teststat =", as.numeric(x@teststat),
      "cval =", paste(as.numeric(x@cval), collapse=","))}, error=function(e) conditionMessage(e))
cat(r, "\n")
r6 <- tryCatch({x <- ca.po(M7[,1:6], type="Pz"); paste("m=6 ok; teststat =", round(as.numeric(x@teststat),6))},
               error=function(e) conditionMessage(e))
cat(r6, "\n")
