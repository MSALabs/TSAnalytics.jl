suppressMessages({library(vars); library(jsonlite)})
out <- list()

grab <- function(fx, p, h, tag) {
  Y <- as.matrix(read.csv(paste0("fixtures/", fx)))
  v <- VAR(Y, p=p, type="const")
  io <- irf(v, n.ahead=h, boot=FALSE, ortho=TRUE)
  inn <- irf(v, n.ahead=h, boot=FALSE, ortho=FALSE)
  ioc <- irf(v, n.ahead=h, boot=FALSE, ortho=TRUE,  cumulative=TRUE)
  inc <- irf(v, n.ahead=h, boot=FALSE, ortho=FALSE, cumulative=TRUE)
  fe <- fevd(v, n.ahead=h)
  S  <- crossprod(resid(v)) / (v$obs - ncol(v$datamat) + ncol(Y))
  out[[tag]] <<- list(
    K=ncol(Y), p=p, h=h, names=colnames(Y), obs=v$obs,
    # irf[[impulse]] is (h+1) x K, rows = horizon 0..h, cols = response
    irf_ortho      = io$irf,
    irf_nonortho   = inn$irf,
    irf_ortho_cum  = ioc$irf,
    irf_nonortho_cum = inc$irf,
    Psi = Psi(v, nstep=h),      # orthogonalised MA, (K,K,h+1)
    Phi = Phi(v, nstep=h),      # non-orthogonalised MA, (K,K,h+1)
    fevd = lapply(fe, function(m) as.matrix(m)),   # fevd[[variable]] is h x K, rows = horizon 1..h
    Sigma_df = S,
    chol_lower = t(chol(S))
  )
}

grab("var2_p1.csv",  1, 6, "A_p1_h6")
grab("var2_p1.csv",  2, 8, "A_p2_h8")
grab("var3_p2.csv",  2, 6, "B_p2_h6")
grab("uschange.csv", 2, 6, "US_p2_h6")
grab("econ5.csv",    2, 4, "E5_p2_h4")
grab("var2_short.csv", 2, 4, "SH_p2_h4")

## Cholesky-ordering dependence: permute the columns and re-run
Y <- as.matrix(read.csv("fixtures/var2_p1.csv"))
vP <- VAR(Y[, c("y2","y1")], p=2, type="const")
out[["A_p2_permuted"]] <- list(
  names = c("y2","y1"),
  irf_ortho = irf(vP, n.ahead=4, boot=FALSE, ortho=TRUE)$irf,
  irf_nonortho = irf(vP, n.ahead=4, boot=FALSE, ortho=FALSE)$irf,
  fevd = lapply(fevd(vP, n.ahead=4), function(m) as.matrix(m)))

## Bootstrap bands -- recorded for reproducibility, NOT for cross-verification
v <- VAR(Y, p=2, type="const")
set.seed(4321)
ib <- irf(v, n.ahead=4, boot=TRUE, runs=100, ci=0.95, seed=4321)
out[["A_p2_boot"]] <- list(runs=100, ci=0.95, seed=4321,
                           irf=ib$irf, Lower=ib$Lower, Upper=ib$Upper)

## refusals
out$err_bad_impulse <- tryCatch({irf(v, impulse="nope", n.ahead=2, boot=FALSE); "no error"},
                                error=function(e) conditionMessage(e))
out$err_neg_ahead   <- tryCatch({irf(v, n.ahead=-1, boot=FALSE); "no error"},
                                error=function(e) conditionMessage(e))
out$err_zero_ahead  <- tryCatch({x<-irf(v, n.ahead=0, boot=FALSE); paste("no error, dim:", paste(dim(x$irf$y1),collapse="x"))},
                                error=function(e) conditionMessage(e))
out$err_fevd_zero   <- tryCatch({fevd(v, n.ahead=0); "no error"},
                                error=function(e) conditionMessage(e))
out$meta <- list(R=R.version.string, vars=as.character(packageVersion("vars")))
write(toJSON(out, digits=17, auto_unbox=TRUE, matrix="rowmajor"), "fixtures/ref_R_irf.json")
cat("irf R refs written\n")
