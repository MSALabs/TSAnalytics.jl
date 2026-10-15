suppressMessages({library(forecastML); library(dplyr)}); options(digits=17)
FIX <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
C <- read.csv(file.path(FIX,"fixture_c_exog.csv")); NY <- sum(!is.na(C$y)); H <- 6
hist <- C[1:NY,]; fut <- C[(NY+1):nrow(C),]
dat   <- data.frame(y=hist$y, x_known=hist$x_known, x_past=hist$x_past)
dts   <- as.Date(hist$date)
FUTX  <- data.frame(index=as.Date(fut$date), x_known=fut$x_known)   # closure data

cat("### predict_future is a FUNCTION f(data, dates) -> df with an 'index' date col\n")
pf_fun <- function(data, dates) FUTX

l2 <- create_lagged_df(dat, type="train", method="direct", outcome_col=1, horizons=1:H,
                       lookback=1:H, dates=dts, frequency="1 month", dynamic_features="x_known")
cat("h=1 cols:", paste(names(l2$horizon_1), collapse=", "), "\n")
cat("h=6 cols:", paste(names(l2$horizon_6), collapse=", "), "\n")
cat("-> NOTE: x_known (dynamic) appears UNLAGGED in EVERY horizon's design;\n")
cat("   x_past is lagged and loses lags < h as the horizon grows.\n\n")

ols_fit  <- function(data, ...) lm(y ~ ., data=data)
ols_pred <- function(model, data) data.frame(y_pred=as.numeric(predict(model, newdata=data)))
w <- create_windows(l2, window_length=0)
m <- train_model(l2, w, model_name="ols", model_function=ols_fit)
cat("h=1 OLS coefs (DGP: y_t = 0.4 y_{t-1} + 2.0 x_known_t + 1.5 x_past_t + e):\n")
cf <- coef(m$horizon_1$window_1$model); for (nm in names(cf)) cat(sprintf("   %-15s %.17g\n", nm, cf[[nm]]))

lf <- create_lagged_df(dat, type="forecast", method="direct", outcome_col=1, horizons=1:H,
                       lookback=1:H, dates=dts, frequency="1 month",
                       dynamic_features="x_known", predict_future=pf_fun)
cat("\nforecast-type h=1 row:\n"); print(lf$horizon_1, digits=12)
cat("forecast-type h=6 rows:\n"); print(lf$horizon_6, digits=12)
fc <- predict(m, prediction_function=list(ols_pred), data=lf)
fcd <- as.data.frame(fc)
diagn <- fcd[fcd$model_forecast_horizon == fcd$horizon, ]
cat("\n### forecastML DIRECT path (diagonal: model h used for period n+h), WITH exog\n")
for (i in seq_len(nrow(diagn))) cat(sprintf("  h=%d  %.17g\n", diagn$horizon[i], diagn$y_pred[i]))

cat("\n### (D) dynamic_features declared but predict_future OMITTED\n")
cat(tryCatch({
  lfx <- create_lagged_df(dat, type="forecast", method="direct", outcome_col=1, horizons=1:2,
                          lookback=1:2, dates=dts, frequency="1 month", dynamic_features="x_known")
  paste("ACCEPTED (no error). x_known in forecast rows:",
        paste(lfx$horizon_1$x_known, collapse=","), "| class:", class(lfx$horizon_1$x_known))
}, error=function(e) paste("REJECTED:", conditionMessage(e))), "\n")
cat("   -> then predict() fails with a CASCADE, not a clear message:\n")
cat(tryCatch({
  lfx <- create_lagged_df(dat, type="forecast", method="direct", outcome_col=1, horizons=1:2,
                          lookback=1:2, dates=dts, frequency="1 month", dynamic_features="x_known")
  l2b <- create_lagged_df(dat, type="train", method="direct", outcome_col=1, horizons=1:2,
                          lookback=1:2, dates=dts, frequency="1 month", dynamic_features="x_known")
  mb <- train_model(l2b, create_windows(l2b, window_length=0), model_name="ols", model_function=ols_fit)
  pp <- predict(mb, prediction_function=list(ols_pred), data=lfx); "ACCEPTED"
}, error=function(e) paste("ERROR:", conditionMessage(e))), "\n")

cat("\n### (E) predict_future returning TOO FEW future rows (3 of 6)\n")
cat(tryCatch({
  pf3 <- function(data, dates) FUTX[1:3,]
  lfy <- create_lagged_df(dat, type="forecast", method="direct", outcome_col=1, horizons=1:H,
                          lookback=1:H, dates=dts, frequency="1 month",
                          dynamic_features="x_known", predict_future=pf3)
  paste("ACCEPTED. h=6 nrow:", nrow(lfy$horizon_6), " x_known:",
        paste(sprintf("%.6g", lfy$horizon_6$x_known), collapse=", "))
}, error=function(e) paste("REJECTED:", conditionMessage(e))), "\n")

cat("\n### (F) predict_future WITHOUT the required 'index' column\n")
cat(tryCatch({
  pfb <- function(data, dates) data.frame(x_known=fut$x_known)
  create_lagged_df(dat, type="forecast", method="direct", outcome_col=1, horizons=1:2, lookback=1:2,
                   dates=dts, frequency="1 month", dynamic_features="x_known", predict_future=pfb); "ACCEPTED"
}, error=function(e) paste("REJECTED (clear msg):", conditionMessage(e))), "\n")
