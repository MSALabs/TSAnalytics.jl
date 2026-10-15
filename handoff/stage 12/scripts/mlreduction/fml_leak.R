suppressMessages({library(forecastML); library(dplyr)}); options(digits=17)
FIX <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/fixtures"
d <- read.csv(file.path(FIX,"fixture_a_ar2.csv")); T0 <- 60
leak <- function(label, builder) {
  y0 <- d$y; y1 <- y0; y1[T0] <- y1[T0] + 1000
  a <- builder(y0); b <- builder(y1)
  # match rows by the row_indices attribute so we compare the SAME time point
  ria <- attributes(a)$row_indices; rib <- attributes(b)$row_indices
  ia <- which(ria == T0); ib <- which(rib == T0)
  if (length(ia)==0) { cat(sprintf("  %-46s row t0 not in design\n", label)); return(invisible()) }
  A <- a[[1]]; B <- b[[1]]
  fe <- setdiff(names(A), c())
  bad <- character(0)
  for (cc in fe) {
    va <- A[[cc]][ia]; vb <- B[[cc]][ib]
    if (!(is.na(va) && is.na(vb)) && !isTRUE(all.equal(va, vb, tolerance=1e-12)))
      bad <- c(bad, sprintf("%s: %.10g -> %.10g", cc, va, vb))
  }
  # the OUTCOME column y is EXPECTED to change (it IS the target); separate it
  outcome <- attributes(a)$outcome_name
  bad_feat <- bad[!grepl(paste0("^", outcome, ":"), bad)]
  bad_feat <- bad_feat[!grepl("^y_[0-9]+:", bad_feat)]
  cat(sprintf("  %-46s [%s]  cols: %s\n", label, if(length(bad_feat)) "LEAK" else "clean",
              paste(names(A), collapse=", ")))
  for (s in bad_feat) cat("      !!", s, "\n")
}
cat("=== forecastML create_lagged_df: does any FEATURE in row t see y[t]? ===\n")
leak("direct, lookback=1:2, h=1", function(yy)
  create_lagged_df(data.frame(y=yy),type="train",method="direct",outcome_col=1,horizons=1,lookback=1:2,keep_rows=TRUE))
leak("direct, lookback=1:6, h=3", function(yy)
  create_lagged_df(data.frame(y=yy),type="train",method="direct",outcome_col=1,horizons=3,lookback=1:6,keep_rows=TRUE))
leak("multi_output, lookback=1:2, h=1:6", function(yy)
  create_lagged_df(data.frame(y=yy),type="train",method="multi_output",outcome_col=1,horizons=1:6,lookback=1:2))
cat("\n=== ADVERSARIAL: can lookback contain 0? ===\n")
cat(tryCatch({create_lagged_df(data.frame(y=d$y),type="train",method="direct",outcome_col=1,horizons=1,lookback=0:2); "ACCEPTED (lookback=0:2)"},
  error=function(e) paste("REJECTED:", conditionMessage(e))), "\n")
cat("\n=== dynamic_features: a column used UNLAGGED (contemporaneous) ===\n")
dd <- data.frame(y=d$y, xx=d$y)   # xx is a COPY of y, declared 'dynamic' -> used unlagged
ld <- create_lagged_df(dd,type="train",method="direct",outcome_col=1,horizons=1,lookback=1:2,
                       dynamic_features="xx", keep_rows=TRUE)
cat("cols:", paste(names(ld[[1]]),collapse=", "), "\n")
cat("row 10: y =", sprintf("%.12g", ld[[1]]$y[10]), " xx =", sprintf("%.12g", ld[[1]]$xx[10]), "\n")
cat("-> dynamic_features are taken CONTEMPORANEOUSLY (no lag). If such a column is\n")
cat("   not genuinely known at forecast time, that is leakage BY USER CONSTRUCTION,\n")
cat("   and forecastML issues no warning.\n")
