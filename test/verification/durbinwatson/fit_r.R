# Cross-checks the DW statistic AND runs R's real (exact) lmtest::dwtest
# p-value on the same x,y data gen_and_verify.py generated -- documents
# how different the :approx (large-sample normal) method this project
# implements is from R's actual exact method, honestly, rather than
# just asserting :approx works.
suppressMessages(library(lmtest))

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
DIR <- dirname(normalizePath(script_path))

for (label in c("ar1", "wn")) {
    d <- read.csv(file.path(DIR, paste0("durbinwatson_", label, ".csv")))
    m <- lm(y ~ x, data = d)
    t <- dwtest(m)  # default alternative = "greater"
    cat(label, ": DW =", unname(t$statistic), " exact p-value (greater) =", t$p.value, "\n")
}
