options(digits=17)
ROOT <- "/tmp/claude-0/-home-claude/eb040610-9b28-5c37-9f56-976a4d7fa171/scratchpad/s12/mlreduction/rsrc/forecastML/R"
fs <- sort(list.files(ROOT, pattern="\\.R$", full.names=TRUE))
cat(sprintf("%-26s %7s %7s  %s\n","file","lines","ndefs","top-level functions (lines)"))
tot <- 0
for (f in fs) {
  e <- parse(f, keep.source=TRUE)
  src <- readLines(f, warn=FALSE); tot <- tot + length(src)
  rows <- character(0)
  for (i in seq_along(e)) {
    ex <- e[[i]]
    if (is.call(ex) && as.character(ex[[1]]) %in% c("<-","=") && is.call(ex[[3]]) &&
        identical(as.character(ex[[3]][[1]]), "function")) {
      sr <- attr(e, "srcref")[[i]]
      rows <- c(rows, sprintf("%s(%d)", as.character(ex[[2]]), sr[3]-sr[1]+1))
    }
  }
  cat(sprintf("%-26s %7d %7d  %s\n", basename(f), length(src), length(rows),
      paste(rows, collapse=" ")))
}
cat(sprintf("\nforecastML R/ TOTAL: %d lines across %d files\n", tot, length(fs)))
cat("\n### LICENSE\n"); cat(readLines(file.path(dirname(ROOT),"LICENSE"), warn=FALSE), sep="\n")
cat("\n\n### DESCRIPTION License field:\n")
cat(grep("^License", readLines(file.path(dirname(ROOT),"DESCRIPTION")), value=TRUE), "\n")
