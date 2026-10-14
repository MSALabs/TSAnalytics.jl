options(digits=12)
suppressMessages(library(tsDyn))
y <- as.numeric(log10(lynx)); n<-length(y)
z <- y[1:112]   # thDelay=1 -> y_{t-2}
mk <- tsDyn:::makeTh
cand_pkg <- mk(allTh=sort(z), trim=0.15, th=list(exact=NULL,int=c("from","to"),around="val",ngrid="All"), trace=TRUE, nthresh=1)
cat("selectSETAR-style (allTh = sort(z), NOT unique): n cand =", length(cand_pkg), "\n")
cand_uni <- mk(allTh=sort(unique(z)), trim=0.15, th=list(exact=NULL,int=c("from","to"),around="val",ngrid="All"), trace=FALSE, nthresh=1)
cat("my handroll style (allTh = sort(unique(z)))     : n cand =", length(cand_uni), "\n")
cat("identical sets:", isTRUE(all.equal(sort(cand_pkg), sort(cand_uni))), "\n")
cat("setdiff pkg-not-uni:", length(setdiff(cand_pkg,cand_uni)), " uni-not-pkg:", length(setdiff(cand_uni,cand_pkg)), "\n")
cat("range pkg:", range(cand_pkg), " range uni:", range(cand_uni), "\n")
cat("\n### does lynx z have duplicates? n=",length(z)," n unique=",length(unique(z)),"\n")
