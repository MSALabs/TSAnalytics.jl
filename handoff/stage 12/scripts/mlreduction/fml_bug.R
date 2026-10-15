suppressMessages({library(forecastML); library(dplyr)}); options(digits=17)
FIX <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
C <- read.csv(file.path(FIX,"fixture_c_exog.csv")); NY <- sum(!is.na(C$y))
hist <- C[1:NY,]; dat <- data.frame(y=hist$y, x_known=hist$x_known, x_past=hist$x_past)
dts <- as.Date(hist$date)
cat("### TRAIN frame with dynamic_features='x_known', lookback=1:6, horizons=1:6, h=1\n")
l <- create_lagged_df(dat, type="train", method="direct", outcome_col=1, horizons=1:6,
                      lookback=1:6, dates=dts, frequency="1 month", dynamic_features="x_known")
A <- l$horizon_1
cat("dim:", paste(dim(A), collapse="x"), "\n")
cat("column summary (n distinct / n NA / sd):\n")
for (cc in names(A)) cat(sprintf("  %-14s distinct=%-5d NA=%-5d sd=%s\n", cc,
    length(unique(A[[cc]])), sum(is.na(A[[cc]])),
    if (is.numeric(A[[cc]])) sprintf("%.6g", sd(A[[cc]], na.rm=TRUE)) else "non-numeric"))
cat("\nhead(4) of the x_past_lag_* block:\n")
print(head(A[, grep("x_past", names(A))], 4), digits=12)
cat("\nraw hist$x_past[1:8]:"); cat(sprintf(" %.8g", hist$x_past[1:8]), "\n")
cat("\n### SAME but WITHOUT dynamic_features (x_known lagged too)\n")
l0 <- create_lagged_df(dat, type="train", method="direct", outcome_col=1, horizons=1:6,
                       lookback=1:6, dates=dts, frequency="1 month")
B <- l0$horizon_1
cat("dim:", paste(dim(B),collapse="x"), " cols:", paste(names(B), collapse=", "), "\n")
cat("x_past_lag_1 sd:", sprintf("%.6g", sd(B$x_past_lag_1, na.rm=TRUE)),
    " distinct:", length(unique(B$x_past_lag_1)), "\n")
cat("head(3):\n"); print(head(B[,c("y","y_lag_1","x_known_lag_1","x_past_lag_1")],3), digits=12)
cat("\nOLS on the no-dynamic design:\n")
cfb <- coef(lm(y ~ ., data=B)); for (nm in names(cfb)) cat(sprintf("   %-16s %.17g\n", nm, cfb[[nm]]))
cat("\n### lookback=1:2 WITH dynamic_features (the case that WORKED earlier)\n")
l2 <- create_lagged_df(dat, type="train", method="direct", outcome_col=1, horizons=1:2,
                       lookback=1:2, dates=dts, frequency="1 month", dynamic_features="x_known")
D <- l2$horizon_1
cat("cols:", paste(names(D),collapse=", "), "\n")
for (cc in grep("x_past", names(D), value=TRUE))
  cat(sprintf("  %-14s sd=%.6g distinct=%d\n", cc, sd(D[[cc]], na.rm=TRUE), length(unique(D[[cc]]))))
