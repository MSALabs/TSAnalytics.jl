f <- "src/MSwM/R/2MSM.r"
L <- readLines(f)
# find lines that start a function assignment  name=function(  or name <- function(
starts <- grep("^[.A-Za-z][A-Za-z0-9._]*\\s*(=|<-)\\s*function", L)
nm <- sub("\\s*(=|<-)\\s*function.*$","",L[starts])
# end = line before next start (or next top-level setMethod/setClass)
tops <- sort(unique(c(starts, grep("^setMethod|^setClass|^setGeneric", L), length(L)+1)))
cat(sprintf("TOTAL LINES %d\n", length(L)))
for (i in seq_along(starts)) {
  s <- starts[i]
  e <- min(tops[tops > s]) - 1
  cat(sprintf("%-34s %5d-%5d  %5d\n", nm[i], s, e, e-s+1))
}
cat("\n--- 1global.r ---\n")
L2 <- readLines("src/MSwM/R/1global.r"); cat(L2, sep="\n")
