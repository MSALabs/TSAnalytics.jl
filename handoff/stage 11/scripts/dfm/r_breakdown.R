files <- Sys.glob("dfms-src/R/*.R")
tot <- 0
for (f in files) {
  lines <- readLines(f, warn=FALSE)
  ex <- parse(f, keep.source=TRUE)
  sr <- attr(ex, "srcref")
  cat(sprintf("\n##### %s  total_lines=%d  n_toplevel=%d\n", f, length(lines), length(ex)))
  for (i in seq_along(ex)) {
    r <- sr[[i]]
    span <- r[3] - r[1] + 1
    e <- ex[[i]]
    nm <- "<expr>"
    if (is.call(e) && length(e) >= 2 && as.character(e[[1]])[1] %in% c("<-","=","assign")) {
      nm <- paste(deparse(e[[2]]), collapse="")
    }
    isfun <- is.call(e) && length(e)>=3 && is.call(e[[3]]) && as.character(e[[3]][[1]])[1]=="function"
    cat(sprintf("  %-38s %5d-%-5d span=%-5d %s\n", substr(nm,1,38), r[1], r[3], span, if(isfun) "FUN" else ""))
  }
  tot <- tot + length(lines)
}
cat(sprintf("\nTOTAL R lines = %d\n", tot))
