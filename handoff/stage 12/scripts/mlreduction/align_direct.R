suppressMessages({library(forecastML); library(dplyr)}); options(digits=17)
FIX <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d <- read.csv(file.path(FIX,"fixture_a_ar2.csv")); dat <- data.frame(y=d$y); n <- nrow(dat)
H <- 6; W <- 2
cat("### CLAIM: sktime direct(window_length=2) at horizon h uses origin-anchored\n")
cat("### lags {1,2} and target y_{t+h}; forecastML direct uses TARGET-anchored lags\n")
cat("### {h..max}. The two coincide when forecastML's lookback_control for horizon h\n")
cat("### is set to {h, h+1} -- i.e. lag h and h+1 relative to the target == lag 1,2\n")
cat("### relative to the origin. Testing that equivalence.\n\n")
lbc <- lapply(1:H, function(h) list(h + 0:(W-1)))   # one list per horizon, one vector per column of `data`
ldf <- create_lagged_df(dat, type="train", method="direct", outcome_col=1,
                        horizons=1:H, lookback_control=lbc)
for (h in 1:H) cat(sprintf("h=%d dim=%s cols: %s\n", h,
  paste(dim(ldf[[paste0("horizon_",h)]]),collapse="x"),
  paste(names(ldf[[paste0("horizon_",h)]]), collapse=", ")))
ols_fit  <- function(data, ...) lm(y ~ ., data=data)
ols_pred <- function(model, data) data.frame(y_pred=as.numeric(predict(model, newdata=data)))
m <- train_model(ldf, create_windows(ldf, window_length=0), model_name="ols", model_function=ols_fit)
cat("\n### forecastML coefficients, lookback_control = {h, h+1} per horizon\n")
for (h in 1:H) {
  mdl <- m[[paste0("horizon_",h)]][["window_1"]]$model; cf <- coef(mdl)
  cat(sprintf("h=%d  n=%d  ", h, nrow(model.matrix(mdl))))
  cat(paste(sprintf("%s=%.17g", names(cf), cf), collapse="  "), "\n")
}
ldf_f <- create_lagged_df(dat, type="forecast", method="direct", outcome_col=1,
                          horizons=1:H, lookback_control=lbc)
fc <- as.data.frame(predict(m, prediction_function=list(ols_pred), data=ldf_f))
diagn <- fc[fc$model_forecast_horizon == fc$horizon, ]
cat("\n### forecastML DIRECT path (diagonal), lookback_control={h,h+1}\n")
cat("FML_DIRECT_PATH")
for (i in seq_len(nrow(diagn))) cat(sprintf(" %.17g", diagn$y_pred[i])); cat("\n")
cat("\n### forecastML RECURSIVE path hand-rolled from an h=1 model with lags 1:2\n")
l1 <- create_lagged_df(dat, type="train", method="direct", outcome_col=1, horizons=1, lookback=1:2)
m1 <- lm(y ~ ., data=l1$horizon_1)
cat("h=1 model coefs: "); cat(paste(sprintf("%s=%.17g", names(coef(m1)), coef(m1)), collapse="  "), "\n")
hist <- d$y; preds <- numeric(H)
for (h in 1:H) {
  nd <- data.frame(y_lag_1=hist[length(hist)], y_lag_2=hist[length(hist)-1])
  preds[h] <- as.numeric(predict(m1, newdata=nd)); hist <- c(hist, preds[h])
}
cat("FML_RECURSIVE_PATH"); cat(sprintf(" %.17g", preds)); cat("\n")
