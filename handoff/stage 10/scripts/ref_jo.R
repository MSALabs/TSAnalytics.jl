suppressMessages({library(urca); library(jsonlite)})
out <- list()
jo <- function(fx, K, ecdet, spec, tag) {
  Y <- as.matrix(read.csv(paste0("fixtures/", fx)))
  res <- list()
  for (ty in c("trace","eigen")) {
    j <- ca.jo(Y, type=ty, ecdet=ecdet, K=K, spec=spec)
    res[[ty]] <- list(teststat=as.numeric(j@teststat), cval=j@cval,
                      lambda=as.numeric(j@lambda), P=j@P, K=j@lag, ecdet=j@ecdet, spec=j@spec)
  }
  j <- ca.jo(Y, type="trace", ecdet=ecdet, K=K, spec=spec)
  res$V <- j@V; res$W <- j@W; res$Vorg <- j@Vorg
  res$SVD_dims <- dim(j@V)
  out[[tag]] <<- res
}
jo("coint2_yes.csv", 2, "none",  "longrun",  "yes_K2_none_LR")
jo("coint2_yes.csv", 2, "const", "longrun",  "yes_K2_const_LR")
jo("coint2_yes.csv", 2, "trend", "longrun",  "yes_K2_trend_LR")
jo("coint2_yes.csv", 2, "none",  "transitory","yes_K2_none_TR")
jo("coint2_yes.csv", 3, "none",  "longrun",  "yes_K3_none_LR")
jo("coint2_no.csv",  2, "none",  "longrun",  "no_K2_none_LR")
jo("coint3_r1.csv",  2, "none",  "longrun",  "r1_K2_none_LR")
jo("coint3_r1.csv",  2, "const", "longrun",  "r1_K2_const_LR")
jo("coint3_r2.csv",  2, "none",  "longrun",  "r2_K2_none_LR")
jo("uschange.csv",   2, "const", "longrun",  "US_K2_const_LR")
jo("econ5.csv",      2, "const", "longrun",  "E5_K2_const_LR")
## cajorls: the VECM given a rank
Y <- as.matrix(read.csv("fixtures/coint3_r1.csv"))
j <- ca.jo(Y, type="trace", ecdet="none", K=2, spec="longrun")
for (r in 1:2) {
  rl <- cajorls(j, r=r)
  out[[paste0("cajorls_r1fix_r", r)]] <- list(
    beta=rl$beta, coefs=coef(rl$rlm), rank=r,
    coef_names=rownames(coef(rl$rlm)))
}
## refusals
out$err_K1 <- tryCatch({ca.jo(Y, type="trace", ecdet="none", K=1); "no error"},
                       error=function(e) conditionMessage(e))
out$err_onecol <- tryCatch({ca.jo(Y[,1,drop=FALSE], type="trace", K=2); "no error"},
                           error=function(e) conditionMessage(e))
out$err_badtype <- tryCatch({ca.jo(Y, type="bogus", K=2); "no error"},
                            error=function(e) conditionMessage(e))
out$err_13col <- tryCatch({M<-matrix(cumsum(rnorm(250*13)),250,13); x<-ca.jo(M,type="trace",K=2)
                           paste("no error; cval dim:", paste(dim(x@cval),collapse="x"),
                                 "first row:", paste(x@cval[1,],collapse=","))},
                          error=function(e) conditionMessage(e))
out$meta <- list(R=R.version.string, urca=as.character(packageVersion("urca")))
write(toJSON(out, digits=17, auto_unbox=TRUE, matrix="rowmajor"), "fixtures/ref_R_johansen.json")
cat("johansen R refs written\n")
