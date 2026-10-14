options(digits=12)
suppressMessages(library(tsDyn))
mk <- tsDyn:::makeTh
cat("=== ties test: sort(z) vs sort(unique(z)) grids can differ ===\n")
set.seed(3); zt <- round(rnorm(200),1)   # many ties
a <- mk(sort(zt), trim=0.15, th=list(exact=NULL,int=c("from","to"),around="val",ngrid="All"), nthresh=1)
b <- mk(sort(unique(zt)), trim=0.15, th=list(exact=NULL,int=c("from","to"),around="val",ngrid="All"), nthresh=1)
cat("n(z)=",length(zt)," n unique=",length(unique(zt)),"\n")
cat("grid from sort(z): n=",length(a)," values:",paste(a,collapse=","),"\n")
cat("grid from unique : n=",length(b)," values:",paste(b,collapse=","),"\n")
cat("SAME:", isTRUE(all.equal(a,b)),"  => authoritative tsDyn convention is allTh = sort(z) WITH ties\n")

cat("\n=== selectSETAR grid cost (trace reports combination counts) ===\n")
y <- log10(lynx)
for (mm in c(2,3,5)) {
  t0<-Sys.time()
  out <- capture.output(s <- selectSETAR(y, m=mm, thDelay=0:(mm-1), trace=TRUE, plot=FALSE, nthresh=1))
  el <- as.numeric(difftime(Sys.time(),t0,units="secs"))
  cmb <- grep("combinations", out, value=TRUE)
  cat(sprintf("m=%2d  nthresh=1  rows=%d  %.3fs  | %s\n", mm, nrow(s$allTh), el, trimws(gsub("\\s+"," ",cmb))))
}
cat("\n=== nthresh=2 cost ===\n")
for (mm in c(2,3,5)) {
  t0<-Sys.time()
  out <- capture.output(s2 <- selectSETAR(y, m=mm, thDelay=0:(mm-1), trace=TRUE, plot=FALSE, nthresh=2))
  el <- as.numeric(difftime(Sys.time(),t0,units="secs"))
  cat(sprintf("m=%2d  nthresh=2  %.3fs  bests: %s\n", mm, el, paste(names(s2$bests),round(s2$bests,6),collapse=" ")))
  cat("   trace lines:\n"); cat(paste("    ", grep("Searching|combinations|iteration|Second|conditional", out, value=TRUE), collapse="\n"), "\n")
}
cat("\n=== default criterion and args of selectSETAR ===\n")
f <- args(selectSETAR); print(f)
