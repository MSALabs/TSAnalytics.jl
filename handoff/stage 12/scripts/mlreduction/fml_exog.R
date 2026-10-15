suppressMessages({library(forecastML); library(dplyr)}); options(digits=17)
FIX <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
C <- read.csv(file.path(FIX,"fixture_c_exog.csv")); NY <- sum(!is.na(C$y)); H <- 6
cat("fixture C:", NY, "observed y rows,", nrow(C)-NY, "future-only rows\n\n")
hist <- C[1:NY, ]; fut <- C[(NY+1):nrow(C), ]
dat  <- data.frame(y=hist$y, x_known=hist$x_known, x_past=hist$x_past)

cat("### (A) BOTH exog LAGGED (past-only semantics): the default\n")
l1 <- create_lagged_df(dat, type="train", method="direct", outcome_col=1, horizons=1:2, lookback=1:2)
cat("h=1 cols:", paste(names(l1$horizon_1), collapse=", "), "\n")
cat("h=2 cols:", paste(names(l1$horizon_2), collapse=", "), "\n\n")

cat("### (B) x_known as a DYNAMIC feature (known-future, CONTEMPORANEOUS); x_past lagged\n")
l2 <- create_lagged_df(dat, type="train", method="direct", outcome_col=1, horizons=1:2,
                       lookback=1:2, dynamic_features="x_known")
cat("h=1 cols:", paste(names(l2$horizon_1), collapse=", "), "\n")
cat("h=1 dim:", paste(dim(l2$horizon_1), collapse="x"), "\n")
print(head(l2$horizon_1, 3), digits=12)

ols_fit  <- function(data, ...) lm(y ~ ., data=data)
ols_pred <- function(model, data) data.frame(y_pred=as.numeric(predict(model, newdata=data)))
w <- create_windows(l2, window_length=0)
m <- train_model(l2, w, model_name="ols", model_function=ols_fit)
cat("\nh=1 OLS coefficients (true DGP: y = 0.4 y_{t-1} + 2.0 x_known_t + 1.5 x_past_t):\n")
cf <- coef(m$horizon_1$window_1$model)
for (nm in names(cf)) cat(sprintf("   %-14s %.17g\n", nm, cf[[nm]]))

cat("\n### (C) FORECAST TIME: dynamic features must be SUPPLIED via predict_future\n")
pf <- data.frame(x_known = fut$x_known)
lf <- create_lagged_df(dat, type="forecast", method="direct", outcome_col=1, horizons=1:2,
                       lookback=1:2, dynamic_features="x_known", predict_future=pf)
cat("forecast rows h=1:\n"); print(lf$horizon_1, digits=12)
fc <- predict(m, prediction_function=list(ols_pred), data=lf)
cat("\nforecasts:\n"); print(as.data.frame(fc), digits=17)

cat("\n### (D) OMIT predict_future while dynamic_features are declared -> ?\n")
cat(tryCatch({
  lfx <- create_lagged_df(dat, type="forecast", method="direct", outcome_col=1, horizons=1:2,
                          lookback=1:2, dynamic_features="x_known")
  paste("ACCEPTED. cols:", paste(names(lfx$horizon_1),collapse=","),
        "| x_known values:", paste(sprintf("%.6g", lfx$horizon_1$x_known), collapse=","))
}, error=function(e) paste("REJECTED:", conditionMessage(e))), "\n")

cat("\n### (E) predict_future with TOO FEW rows (3 of 6) -> ?\n")
cat(tryCatch({
  lfy <- create_lagged_df(dat, type="forecast", method="direct", outcome_col=1, horizons=6,
                          lookback=1:6, dynamic_features="x_known",
                          predict_future=data.frame(x_known=fut$x_known[1:3]))
  paste("ACCEPTED. nrow:", nrow(lfy$horizon_6), "x_known:", paste(sprintf("%.6g", lfy$horizon_6$x_known), collapse=","))
}, error=function(e) paste("REJECTED:", conditionMessage(e))), "\n")

cat("\n### (F) static_features: a time-invariant column\n")
dat2 <- data.frame(y=hist$y, grp=rep("a", NY), x_known=hist$x_known)
cat(tryCatch({
  ls2 <- create_lagged_df(dat2, type="train", method="direct", outcome_col=1, horizons=1,
                          lookback=1:2, static_features="grp")
  paste("cols:", paste(names(ls2$horizon_1), collapse=", "))
}, error=function(e) paste("ERR:", conditionMessage(e))), "\n")
