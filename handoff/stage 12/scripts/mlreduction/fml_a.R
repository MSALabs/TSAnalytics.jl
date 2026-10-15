suppressMessages({library(forecastML); library(dplyr)})
options(digits = 17)
FIX <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d <- read.csv(file.path(FIX, "fixture_a_ar2.csv"))
dat <- data.frame(y = d$y)

cat("### A. DIRECT reduction, horizons 1:6, lookback 1:6 (must cover horizon)\n")
ldf <- create_lagged_df(dat, type="train", method="direct", outcome_col=1,
                        horizons=1:6, lookback=1:6)
for (h in 1:6) {
  dfh <- ldf[[paste0("horizon_", h)]]
  cat(sprintf("h=%d  dim=%-10s cols: %s\n", h, paste(dim(dfh), collapse="x"),
      paste(names(dfh), collapse=", ")))
}
cat("\n--- horizon_1 head(4) ---\n"); print(head(ldf$horizon_1, 4), digits=17)
cat("\n--- horizon_3 head(4) ---\n"); print(head(ldf$horizon_3, 4), digits=17)
cat("\nNOTE: attr(ldf$horizon_h,'lookback') =",
    paste(sapply(1:6, function(h) paste(attributes(ldf[[paste0("horizon_",h)]])$lookback, collapse="/")), collapse=" | "), "\n")

ols_fit  <- function(data, ...) lm(y ~ ., data = data)
ols_pred <- function(model, data) data.frame(y_pred = as.numeric(predict(model, newdata = data)))

w <- create_windows(ldf, window_length = 0)
m <- train_model(ldf, w, model_name="ols", model_function=ols_fit)
cat("\n### FITTED OLS COEFFICIENTS per horizon (17 sig digits)\n")
for (h in 1:6) {
  mdl <- m[[paste0("horizon_", h)]][["window_1"]]$model
  cf <- coef(mdl); X <- model.matrix(mdl)
  cat(sprintf("h=%d  designX=%dx%d\n", h, nrow(X), ncol(X)))
  for (nmm in names(cf)) cat(sprintf("      %-14s %.17g\n", nmm, cf[[nmm]]))
}
ldf_f <- create_lagged_df(dat, type="forecast", method="direct", outcome_col=1,
                          horizons=1:6, lookback=1:6)
cat("\n--- forecast-type rows ---\n")
for (h in c(1,3,6)) { cat(sprintf("h=%d: ", h)); print(ldf_f[[paste0("horizon_",h)]], digits=17) }
fc <- predict(m, prediction_function=list(ols_pred), data=ldf_f)
cat("\n### forecastML DIRECT forecasts h=1..6\n")
print(as.data.frame(fc), digits=17)
cat("\nPOINT FORECAST VECTOR:\n")
cat(sprintf("%.17g\n", as.data.frame(fc)$y_pred))

cat("\n\n### B. MULTI_OUTPUT reduction, horizons=6, lookback=1:2\n")
lmo <- create_lagged_df(dat, type="train", method="multi_output", outcome_col=1,
                        horizons=6, lookback=1:2)
cat("length:", length(lmo), " dim:", paste(dim(lmo[[1]]), collapse="x"), "\n")
cat("cols:", paste(names(lmo[[1]]), collapse=", "), "\n")
cat("--- head(5) ---\n"); print(head(lmo[[1]], 5), digits=17)
cat("\n*** ALIGNMENT CHECK: raw y[1:8] =\n"); cat(sprintf("%.17g ", d$y[1:8]), "\n")

cat("\n\n### C. PROBE: non-function model_function\n")
cat(tryCatch({train_model(ldf,w,model_name="bad",model_function=lm(y~.,data=dat)); "ACCEPTED"},
    error=function(e) paste("REJECTED:", conditionMessage(e))), "\n")
cat("### PROBE: method='recursive'\n")
cat(tryCatch({create_lagged_df(dat,type="train",method="recursive",outcome_col=1,horizons=1,lookback=1:2); "ACCEPTED"},
    error=function(e) paste("REJECTED:", conditionMessage(e))), "\n")
cat("### PROBE: direct with lookback 1:2 but horizons 1:6 (the earlier crash)\n")
cat(tryCatch({create_lagged_df(dat,type="train",method="direct",outcome_col=1,horizons=1:6,lookback=1:2); "ACCEPTED"},
    error=function(e) paste("CRASH:", conditionMessage(e))), "\n")
