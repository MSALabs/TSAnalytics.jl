suppressMessages({library(forecastML); library(dplyr)}); options(digits=17)
FIX <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d <- read.csv(file.path(FIX,"fixture_a_ar2.csv")); dat <- data.frame(y=d$y)
ldf <- create_lagged_df(dat, type="train", method="direct", outcome_col=1, horizons=1:3, lookback=1:3)
cat("### create_windows signature:\n"); print(args(create_windows))
cat("\n### create_windows(window_length=20) -- CONTIGUOUS VALIDATION BLOCKS\n")
w <- create_windows(ldf, window_length=20); print(w)
cat("\n### create_windows(window_length=20, skip=10)\n"); print(create_windows(ldf, window_length=20, skip=10))
cat("\n### create_windows(window_length=20, include_partial_window=FALSE)\n")
print(create_windows(ldf, window_length=20, include_partial_window=FALSE))
cat("\n### create_windows(window_length=0) -- no CV, train on everything\n"); print(create_windows(ldf, window_length=0))
cat("\n!! NOTE: forecastML's 'windows' are VALIDATION blocks carved OUT of the series;\n")
cat("   the model trains on EVERYTHING OUTSIDE the block (both before AND after it).\n")
cat("   That is NOT rolling-origin CV -- it is blocked/holdout CV in the Bergmeir-\n")
cat("   Hyndman-Koo sense cited in the DESCRIPTION. Verifying by row counts:\n")
ols_fit  <- function(data, ...) lm(y ~ ., data=data)
m <- train_model(ldf, w, model_name="ols", model_function=ols_fit)
for (wi in seq_along(w$window_start)) {
  mdl <- m$horizon_1[[paste0("window_",wi)]]$model
  cat(sprintf("   window %d: valid rows %d..%d ; TRAIN n = %d (of %d usable)\n",
      wi, w$window_start[wi], w$window_stop[wi], nrow(model.matrix(mdl)), nrow(ldf$horizon_1)))
}
cat("\n### return_error(): what metrics?\n")
ols_pred <- function(model, data) data.frame(y_pred=as.numeric(predict(model, newdata=data)))
pr <- predict(m, prediction_function=list(ols_pred), data=ldf)
er <- return_error(pr)
cat("names(return_error):", paste(names(er), collapse=", "), "\n")
for (nm in names(er)) { cat("--", nm, "cols:", paste(names(er[[nm]]), collapse=", "), "\n") }
cat("\nerror_global:\n"); print(as.data.frame(er$error_global), digits=10)
cat("\n### PLOT METHODS exposed by forecastML\n")
print(grep("^plot\\.", ls(asNamespace("forecastML"), all.names=TRUE), value=TRUE))
cat("\n### man pages (public API surface)\n")
print(sort(sub("\\.Rd$","", list.files("/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/rsrc/forecastML/man"))))
