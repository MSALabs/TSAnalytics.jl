options(digits=12)
y <- read.csv("cardox240.csv")[[1]]

cat("=== arima(y, order=c(2,0,0)) -- full var.coef ===\n")
m <- arima(y, order=c(2,0,0))
cat("coef names:", names(m$coef), "\n")
V <- m$var.coef
for (i in 1:nrow(V)) cat("  ", format(round(V[i,],12)), "\n")
cat("se        :", format(round(sqrt(diag(V)),10)), "\n")
cat("corr(ar1,ar2):", format(round(V[1,2]/sqrt(V[1,1]*V[2,2]),10)), "\n")

cat("\n=== arima(y, order=c(1,1,1), seasonal=list(order=c(0,1,1), period=12)) ===\n")
ms <- arima(y, order=c(1,1,1), seasonal=list(order=c(0,1,1), period=12))
cat("coef names:", names(ms$coef), "\n")
Vs <- ms$var.coef
for (i in 1:nrow(Vs)) cat("  ", format(round(Vs[i,],12)), "\n")
cat("se        :", format(round(sqrt(diag(Vs)),10)), "\n")
cat("corr(ar1,ma1):", format(round(Vs[1,2]/sqrt(Vs[1,1]*Vs[2,2]),10)), "\n")
