suppressMessages({library(changepoint);library(strucchange)})
y <- read.csv("fxI_step.csv")$y
cat("series:", y, "  n =", length(y), "\n")
cat("TRUTH: segment1 = 1-based 1..5 (value 0); segment2 = 1-based 6..10 (value 10)\n\n")
a <- cpt.mean(y, penalty="Manual", pen.value=1, method="PELT", minseglen=1)
cat("changepoint cpt.mean cpts()        =", cpts(a), "\n")
cat("changepoint seg.len()              =", seg.len(a), "\n")
cat("changepoint param.est()$mean       =", param.est(a)$mean, "\n")
b <- breakpoints(y ~ 1, h = 2, breaks = 1)
cat("strucchange breakpoints (h=2,br=1) =", b$breakpoints, "\n")
cat("strucchange breakfactor levels     =", paste(table(breakfactor(b)),collapse=" / "),
    " (sizes of each segment)\n")
cat("strucchange breakdates             =", breakdates(b), "\n")
