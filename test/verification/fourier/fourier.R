options(digits=12)
suppressWarnings(suppressMessages(library(forecast)))
ty <- ts(rnorm(48), frequency=12)

cat("=== K=2, m=12, rows 1:4 ===\n")
X <- fourier(ty, K=2); cat("cols:", colnames(X), "\n")
for (i in 1:4) cat("  ", format(round(X[i,],10)), "\n")

cat("\n=== K=6, m=12 -- Nyquist sin column dropped ===\n")
X6 <- fourier(ty, K=6)
cat("ncol:", ncol(X6), " (2*6 = 12 would be the naive count)\n")
cat("cols:", colnames(X6), "\n")
cat("row1:", format(round(X6[1,],10)), "\n")

cat("\n=== K=1, m=4, rows 1:5 ===\n")
t4 <- ts(rnorm(20), frequency=4); X4 <- fourier(t4, K=1)
cat("cols:", colnames(X4), "\n")
for (i in 1:5) cat("  ", format(round(X4[i,],10)), "\n")

cat("\n=== future rows: fourier(ty, K=2, h=3) ===\n")
Xh <- fourier(ty, K=2, h=3)
for (i in 1:3) cat("  ", format(round(Xh[i,],10)), "\n")

cat("\n=== K too large ===\n")
cat("  ", tryCatch({fourier(ty, K=7); "no error"}, error=function(e) conditionMessage(e)), "\n")
