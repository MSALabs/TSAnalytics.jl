suppressMessages(library(MSwM))
d <- read.csv("hamilton_rgnp.csv")
y <- d$rgnp
cat("n =", length(y), " mean =", sprintf("%.10f", mean(y)), "\n")
df <- data.frame(y = y)
m0 <- lm(y ~ 1, df)

runfit <- function(sw, p, seed, label, maxiterOuter=5) {
  set.seed(seed)
  fit <- try(msmFit(m0, k = 2, p = p, sw = sw,
                    control = list(parallelization = FALSE, maxiter = 500,
                                   tol = 1e-10, maxiterOuter = maxiterOuter,
                                   maxiterInner = 10)), silent = TRUE)
  if (inherits(fit, "try-error")) { cat(label, "FAILED:", fit, "\n"); return(NULL) }
  cat("\n#####", label, " seed=", seed, "\n")
  cat("  -logLikel (MSwM stores NEGATIVE ll) =", sprintf("%.8f", fit@Fit@logLikel),
      "  => loglik =", sprintf("%.8f", -fit@Fit@logLikel), "\n")
  cat("  Coef:\n"); print(round(fit@Coef, 8))
  cat("  std (per regime):", sprintf("%.8f", fit@std), "\n")
  cat("  transMat (MSwM):\n"); print(round(fit@transMat, 8))
  cat("    colsums:", round(colSums(fit@transMat),8), " rowsums:", round(rowSums(fit@transMat),8), "\n")
  cat("  iniProb:", sprintf("%.8f", fit@iniProb), "\n")
  cat("  AIC:", sprintf("%.6f", AIC(fit)), " nobs(filtProb):", nrow(fit@Fit@filtProb), "\n")
  fit
}

sw_sm   <- c(TRUE, rep(FALSE,4), FALSE)   # switch intercept only, common variance
sw_all  <- c(TRUE, rep(TRUE,4),  TRUE )   # switch everything incl variance
sw_intv <- c(TRUE, rep(FALSE,4), TRUE )   # switch intercept + variance

f1 <- runfit(sw_sm,   4, 101, "A: p=4 sw=intercept only, COMMON variance (= statsmodels default)")
f2 <- runfit(sw_intv, 4, 101, "B: p=4 sw=intercept + variance")
f3 <- runfit(sw_all,  4, 101, "C: p=4 sw=everything")

if (!is.null(f1)) {
  sp <- f1@Fit@smoProb       # nobs+1 x k
  fp <- f1@Fit@filtProb
  cat("\n  smoProb dim:", dim(sp), "  filtProb dim:", dim(fp), "\n")
  write.csv(sp, "mswm_A_smoProb.csv", row.names = FALSE)
  write.csv(fp, "mswm_A_filtProb.csv", row.names = FALSE)
  write.csv(f1@Coef, "mswm_A_coef.csv", row.names = FALSE)
  writeLines(c(sprintf("loglik=%.10f", -f1@Fit@logLikel),
               sprintf("std=%s", paste(sprintf("%.10f", f1@std), collapse=",")),
               sprintf("transMat=%s", paste(sprintf("%.10f", as.vector(f1@transMat)), collapse=",")),
               sprintf("iniProb=%s", paste(sprintf("%.10f", f1@iniProb), collapse=","))),
             "mswm_A_summary.txt")
}
# seed sensitivity for A
cat("\n##### seed sensitivity, spec A\n")
for (s in c(1,2,7,42,101,2024)) {
  set.seed(s)
  ff <- try(msmFit(m0,k=2,p=4,sw=sw_sm,control=list(parallelization=FALSE,maxiter=500,tol=1e-10,maxiterOuter=5,maxiterInner=10)),silent=TRUE)
  if (inherits(ff,"try-error")) cat(sprintf("  seed %5d  FAILED\n", s))
  else cat(sprintf("  seed %5d  loglik=%.8f  int=[%+.6f,%+.6f]  std=[%.6f,%.6f]  P11=%.6f P22=%.6f\n",
      s, -ff@Fit@logLikel, ff@Coef[1,1], ff@Coef[2,1], ff@std[1], ff@std[2],
      ff@transMat[1,1], ff@transMat[2,2]))
}
