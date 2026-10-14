suppressMessages({library(vars); library(jsonlite)})
out <- list()
Y <- as.matrix(read.csv("fixtures/var2_p1.csv"))
Y3 <- as.matrix(read.csv("fixtures/var3_p2.csv"))
v  <- VAR(Y,  p=2, type="const")
v3 <- VAR(Y3, p=2, type="const")

## A-model: lower-triangular A with NA to estimate
am <- diag(2); am[2,1] <- NA
for (em in c("scoring","direct")) {
  s <- tryCatch(SVAR(v, Amat=am, estmethod=em, max.iter=200), error=function(e) NULL)
  if (!is.null(s)) out[[paste0("A_k2_", em)]] <- list(
    A=s$A, Ase=s$Ase, B=s$B, Bse=s$Bse, LRIM=s$LRIM, Sigma.U=s$Sigma.U,
    LR=if(!is.null(s$LR)) list(stat=as.numeric(s$LR$statistic), p=as.numeric(s$LR$p.value),
                               par=as.numeric(s$LR$parameter)) else NULL,
    opt_conv=if(!is.null(s$opt)) s$opt$convergence else NA, estmethod=em)
}
## B-model
bm <- diag(2); bm[2,1] <- NA
s <- tryCatch(SVAR(v, Bmat=bm, estmethod="scoring", max.iter=200), error=function(e) NULL)
if (!is.null(s)) out$B_k2_scoring <- list(A=s$A, B=s$B, Bse=s$Bse, Sigma.U=s$Sigma.U)
## just-identified Cholesky-equivalent A-model (A lower triangular, B = I)
am2 <- matrix(c(1,NA,0,1),2,2)   # column-major: A[1,1]=1, A[2,1]=NA, A[1,2]=0, A[2,2]=1
s <- tryCatch(SVAR(v, Amat=am2, estmethod="scoring", max.iter=200), error=function(e) NULL)
if (!is.null(s)) out$A_k2_justid <- list(A=s$A, B=s$B, Sigma.U=s$Sigma.U)
## Blanchard-Quah
bq <- tryCatch(BQ(v), error=function(e) NULL)
if (!is.null(bq)) out$BQ_k2 <- list(A=bq$A, B=bq$B, LRIM=bq$LRIM, Sigma.U=bq$Sigma.U)
bq3 <- tryCatch(BQ(v3), error=function(e) NULL)
if (!is.null(bq3)) out$BQ_k3 <- list(A=bq3$A, B=bq3$B, LRIM=bq3$LRIM)
## trivariate A-model
am3 <- diag(3); am3[2,1] <- NA; am3[3,1] <- NA; am3[3,2] <- NA
s3 <- tryCatch(SVAR(v3, Amat=am3, estmethod="scoring", max.iter=300), error=function(e) NULL)
if (!is.null(s3)) out$A_k3_scoring <- list(A=s3$A, Ase=s3$Ase, B=s3$B, Sigma.U=s3$Sigma.U)
## SVAR IRF and FEVD
if (!is.null(bq)) {
  out$BQ_irf  <- irf(bq, n.ahead=4, boot=FALSE)$irf
  out$BQ_fevd <- lapply(fevd(bq, n.ahead=4), as.matrix)
}
## over-identified: all off-diagonals NA plus a zero -> LR test
amo <- diag(2); amo[1,2] <- 0; amo[2,1] <- NA
out$err_underid <- tryCatch({SVAR(v, Amat=matrix(NA,2,2), estmethod="scoring"); "no error"},
                            error=function(e) conditionMessage(e))
out$err_noNA    <- tryCatch({SVAR(v, Amat=diag(2), estmethod="scoring"); "no error"},
                            error=function(e) conditionMessage(e))
out$err_both    <- tryCatch({SVAR(v, Amat=am, Bmat=bm, estmethod="scoring"); "no error (AB model)"},
                            error=function(e) conditionMessage(e))
out$meta <- list(R=R.version.string, vars=as.character(packageVersion("vars")))
write(toJSON(out, digits=17, auto_unbox=TRUE, matrix="rowmajor"), "fixtures/ref_R_svar.json")
cat("svar R refs written; keys:", paste(names(out), collapse=", "), "\n")
