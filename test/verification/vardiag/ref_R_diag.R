suppressMessages({library(vars); library(jsonlite)})
out <- list()
ht <- function(h) list(statistic=as.numeric(h$statistic), parameter=as.numeric(h$parameter),
                       p.value=as.numeric(h$p.value), method=h$method)
grab <- function(fx, p, tag, lags.pt=8, lags.bg=5, lags.multi=5, lags.single=4) {
  Y <- as.matrix(read.csv(paste0("fixtures/", fx)))
  v <- VAR(Y, p=p, type="const")
  n <- normality.test(v, multivariate.only=FALSE)
  a <- arch.test(v, lags.multi=lags.multi, lags.single=lags.single, multivariate.only=FALSE)
  out[[tag]] <<- list(K=ncol(Y), p=p, obs=v$obs, names=colnames(Y),
    lags.pt=lags.pt, lags.bg=lags.bg, lags.multi=lags.multi, lags.single=lags.single,
    pt_asymptotic = ht(serial.test(v, lags.pt=lags.pt, type="PT.asymptotic")$serial),
    pt_adjusted   = ht(serial.test(v, lags.pt=lags.pt, type="PT.adjusted")$serial),
    bg            = ht(serial.test(v, lags.bg=lags.bg, type="BG")$serial),
    es            = ht(serial.test(v, lags.bg=lags.bg, type="ES")$serial),
    jb_mul        = ht(n$jb.mul$JB), jb_skew = ht(n$jb.mul$Skewness),
    jb_kurt       = ht(n$jb.mul$Kurtosis),
    jb_uni        = lapply(n$jb.uni, ht),
    arch_mul      = ht(a$arch.mul),
    arch_uni      = lapply(a$arch.uni, ht))
}
grab("var2_p1.csv",  2, "A_p2")
grab("var2_p1.csv",  1, "A_p1")
grab("var3_p2.csv",  2, "B_p2")
grab("uschange.csv", 2, "US_p2", lags.pt=10, lags.bg=4, lags.multi=4)
grab("econ5.csv",    2, "E5_p2", lags.pt=6, lags.bg=3, lags.multi=3)
grab("var2_short.csv", 2, "SH_p2", lags.pt=4, lags.bg=2, lags.multi=2)
grab("climhyd.csv",  2, "C_p2", lags.pt=8, lags.bg=3, lags.multi=2)

## refusals
Y <- as.matrix(read.csv("fixtures/var2_p1.csv")); v <- VAR(Y, p=2, type="const")
out$err_pt_le_p   <- tryCatch({serial.test(v, lags.pt=2); "no error"}, error=function(e) conditionMessage(e))
out$err_pt_zero   <- tryCatch({serial.test(v, lags.pt=0); "no error"}, error=function(e) conditionMessage(e))
out$err_pt_huge   <- tryCatch({x<-serial.test(v, lags.pt=300); paste("no error, df =", as.numeric(x$serial$parameter))},
                              error=function(e) conditionMessage(e))
out$err_arch_zero <- tryCatch({arch.test(v, lags.multi=0); "no error"}, error=function(e) conditionMessage(e))
out$meta <- list(R=R.version.string, vars=as.character(packageVersion("vars")))
write(toJSON(out, digits=17, auto_unbox=TRUE, matrix="rowmajor"), "fixtures/ref_R_diag.json")
cat("diag R refs written\n")
