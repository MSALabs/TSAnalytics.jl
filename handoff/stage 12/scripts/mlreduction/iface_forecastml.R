suppressMessages({library(forecastML); library(dplyr)})
options(digits = 17)
FIX <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d <- read.csv(file.path(FIX, "fixture_a_ar2.csv"))
dat <- data.frame(y = d$y)          # outcome_col = 1

cat("### forecastML version:", as.character(packageVersion("forecastML")), "\n\n")

# ---- 1. the reduction: create_lagged_df ----
ldf <- create_lagged_df(dat, type = "train", method = "direct",
                        outcome_col = 1, horizons = 1:6, lookback = 1:2)
cat("class(ldf):", class(ldf), "\n")
cat("length(ldf) (one data.frame PER HORIZON):", length(ldf), "\n")
cat("names(ldf):", names(ldf), "\n\n")
for (h in c(1, 2, 6)) {
  nm <- paste0("horizon_", h)
  cat(sprintf("--- %s: dim = %s ; cols = %s\n", nm,
      paste(dim(ldf[[nm]]), collapse=" x "), paste(names(ldf[[nm]]), collapse=", ")))
  print(head(ldf[[nm]], 3))
}
cat("\nattributes(ldf) names:", paste(names(attributes(ldf)), collapse=", "), "\n")
cat("attr horizons:", attributes(ldf)$horizons, "\n")
cat("attr lookback:", attributes(ldf)$lookback, "\n\n")

# ---- 2. the INJECTED MODEL FUNCTION contract ----
cat("### INJECTED MODEL FUNCTION: signature is f(data, ...) -> any object\n")
ols_fit <- function(data, ...) {
  # forecastML passes a data.frame whose FIRST column is the outcome
  lm(y ~ ., data = data)
}
# and a predict function: f(model, data) -> data.frame of 1 column
ols_pred <- function(model, data) {
  data.frame(y_pred = as.numeric(predict(model, newdata = data)))
}

w <- create_windows(ldf, window_length = 0)   # window_length=0 -> train on all data, no CV
cat("create_windows(window_length=0):\n"); print(w)

m <- train_model(ldf, w, model_name = "ols", model_function = ols_fit)
cat("\nclass(m):", class(m), " length:", length(m), "\n")
cat("names(m):", names(m), "\n")
cat("\n### FITTED COEFFICIENTS per horizon (15+ sig digits)\n")
for (h in 1:6) {
  mdl <- m[[paste0("horizon_", h)]][["window_1"]]$model
  cf <- coef(mdl)
  cat(sprintf("h=%d  n=%d  ", h, length(mdl$residuals)))
  cat(paste(sprintf("%s=%.17g", names(cf), cf), collapse="  "), "\n")
}

# ---- 3. FORECAST ----
ldf_f <- create_lagged_df(dat, type = "forecast", method = "direct",
                          outcome_col = 1, horizons = 1:6, lookback = 1:2)
cat("\n### forecast-type lagged df, horizon_1:\n"); print(ldf_f$horizon_1)
cat("### forecast-type lagged df, horizon_6:\n"); print(ldf_f$horizon_6)

fc <- predict(m, prediction_function = list(ols_pred), data = ldf_f)
cat("\n### FORECASTS h=1..6\n")
print(as.data.frame(fc), digits = 17)

# ---- 4. does it accept a NON-function model object? ----
cat("\n### PROBE: pass a fitted model OBJECT instead of a function\n")
res <- tryCatch({ train_model(ldf, w, model_name="bad", model_function = lm(y~., data=dat)); "accepted" },
                error = function(e) paste("REJECTED:", conditionMessage(e)))
cat(res, "\n")
cat("\n### PROBE: method='multi_output'\n")
res2 <- tryCatch({
  lmo <- create_lagged_df(dat, type="train", method="multi_output", outcome_col=1, horizons=6, lookback=1:2)
  cat("length:", length(lmo), " dim:", paste(dim(lmo[[1]]), collapse=" x "), "\n")
  cat("cols:", paste(names(lmo[[1]]), collapse=", "), "\n"); "ok"}, error=function(e) paste("ERR:", conditionMessage(e)))
cat(res2, "\n")
cat("\n### PROBE: is there a 'recursive' method at all?\n")
res3 <- tryCatch({create_lagged_df(dat, type="train", method="recursive", outcome_col=1, horizons=1, lookback=1:2); "accepted"},
                 error=function(e) paste("REJECTED:", conditionMessage(e)))
cat(res3, "\n")
