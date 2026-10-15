suppressMessages({library(forecastML); library(dplyr)}); options(digits=17)
FIX <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d <- read.csv(file.path(FIX,"fixture_a_ar2.csv")); dat <- data.frame(y=d$y)
ldf <- create_lagged_df(dat,type="train",method="direct",outcome_col=1,horizons=1:3,lookback=1:3)
w <- create_windows(ldf, window_length=20)
cat("windows object columns:", paste(names(w), collapse=", "), "\n\n")
ols_fit <- function(data, ...) lm(y ~ ., data=data)
m <- train_model(ldf, w, model_name="ols", model_function=ols_fit)
ri <- as.integer(attributes(ldf)$row_indices)
cat("### CLAIM: forecastML trains on everything OUTSIDE the validation block,\n")
cat("### i.e. BOTH sides -- not an expanding/rolling ORIGIN. Verifying:\n")
cat(sprintf("usable design rows for h=1: %d (original indices %d..%d)\n\n",
    nrow(ldf$horizon_1), min(ri), max(ri)))
for (i in seq_len(nrow(w))) {
  mdl <- m$horizon_1[[paste0("window_",i)]]$model
  ntr <- nrow(model.matrix(mdl))
  nval <- sum(ri >= w$start[i] & ri <= w$stop[i])
  cat(sprintf("window %d: valid idx %3d..%3d (%2d rows) -> TRAIN n = %3d ; usable-minus-valid = %3d  %s\n",
      i, w$start[i], w$stop[i], nval, ntr, nrow(ldf$horizon_1)-nval,
      if (ntr == nrow(ldf$horizon_1)-nval) "MATCH => trains on BOTH sides" else "differs"))
}
cat("\n### plot.windows / plot.lagged_df / plot.training_results etc. all require ggplot2\n")
cat("### and render to a device. Confirming they run (no device, just object class):\n")
for (f in c("plot.windows","plot.lagged_df")) {
  obj <- if (f=="plot.windows") w else ldf
  cat(sprintf("  %-22s -> %s\n", f, tryCatch(paste(class(plot(obj)), collapse="/"),
      error=function(e) paste("ERR:", substr(conditionMessage(e),1,70)))))
}
pr <- predict(m, prediction_function=list(function(model,data) data.frame(y_pred=as.numeric(predict(model,newdata=data)))), data=ldf)
er <- return_error(pr)
for (f in c("plot.training_results","plot.validation_error")) {
  obj <- if (f=="plot.training_results") pr else er
  cat(sprintf("  %-22s -> %s\n", f, tryCatch(paste(class(plot(obj)), collapse="/"),
      error=function(e) paste("ERR:", substr(conditionMessage(e),1,70)))))
}
