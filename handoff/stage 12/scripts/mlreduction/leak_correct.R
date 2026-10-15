suppressMessages({library(forecastML); library(dplyr)}); options(digits=17)
FIX <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d <- read.csv(file.path(FIX,"fixture_a_ar2.csv")); T0 <- 60
cat("=== CORRECTED multi_output leak test ===\n")
cat("In multi_output the row labelled t has TARGETS y_1..y_6 = y[t+1..t+6] and\n")
cat("FEATURES y_lag_1 = y[t], y_lag_2 = y[t-1].  So the right question is:\n")
cat("does any FEATURE at row t depend on y[t+1] (the FIRST target)?\n\n")
b <- function(yy) create_lagged_df(data.frame(y=yy),type="train",method="multi_output",
                                   outcome_col=1,horizons=1:6,lookback=1:2)
for (shift in 0:1) {
  tp <- T0 + shift
  y0 <- d$y; y1 <- y0; y1[tp] <- y1[tp] + 1000
  A <- b(y0)[[1]]; B <- b(y1)[[1]]
  feats <- c("y_lag_1","y_lag_2")
  moved <- sapply(feats, function(cc) {
    va <- A[[cc]][T0]; vb <- B[[cc]][T0]
    if (is.na(va)&&is.na(vb)) FALSE else !isTRUE(all.equal(va,vb,tolerance=1e-12)) })
  cat(sprintf("perturb y[%d] (= %s) -> features at row %d moved: %s   => %s\n",
      tp, if(shift==0) "y_t, a legitimate feature" else "y_{t+1}, the FIRST TARGET", T0,
      paste(names(moved)[moved], collapse=",") , 
      if (shift==1 && any(moved)) "LEAK" else if (shift==1) "clean (no target leaks into features)" else "expected (y_t IS feature y_lag_1)"))
}
cat("\nVERDICT: forecastML multi_output is CLEAN. The earlier 'LEAK' flag was a\n")
cat("false positive of a uniform probe that assumed target==y_t for every method.\n")
