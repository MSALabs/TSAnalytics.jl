suppressMessages({library(vars); library(lmtest); library(jsonlite)})
out <- list()

gr <- function(Y, p, cause, tag) {
  v <- VAR(Y, p=p, type="const")
  g <- causality(v, cause=cause)
  out[[tag]] <<- list(
    K=ncol(Y), p=p, cause=cause, obs=v$obs,
    granger_stat = as.numeric(g$Granger$statistic),
    granger_df1  = as.numeric(g$Granger$parameter[["df1"]]),
    granger_df2  = as.numeric(g$Granger$parameter[["df2"]]),
    granger_p    = as.numeric(g$Granger$p.value),
    granger_method = g$Granger$method,
    inst_stat = as.numeric(g$Instant$statistic),
    inst_df   = as.numeric(g$Instant$parameter),
    inst_p    = as.numeric(g$Instant$p.value),
    inst_method = g$Instant$method
  )
}

A  <- as.matrix(read.csv("fixtures/var2_p1.csv"))
B3 <- as.matrix(read.csv("fixtures/var3_p2.csv"))
SH <- as.matrix(read.csv("fixtures/var2_short.csv"))
US <- as.matrix(read.csv("fixtures/uschange.csv"))
E5 <- as.matrix(read.csv("fixtures/econ5.csv"))

gr(A, 1, "y2", "A_p1_cause_y2");  gr(A, 1, "y1", "A_p1_cause_y1")
gr(A, 2, "y2", "A_p2_cause_y2");  gr(A, 2, "y1", "A_p2_cause_y1")
gr(A, 4, "y2", "A_p4_cause_y2")
gr(B3, 2, "x1", "B_p2_cause_x1"); gr(B3, 2, "x2", "B_p2_cause_x2")
gr(B3, 2, "x3", "B_p2_cause_x3")
gr(B3, 2, c("x1","x2"), "B_p2_cause_x1x2")
gr(B3, 2, c("x2","x3"), "B_p2_cause_x2x3")
gr(SH, 2, "y2", "SH_p2_cause_y2")
gr(US, 2, "Income", "US_p2_cause_Income")
gr(US, 2, c("Income","Production"), "US_p2_cause_IncProd")
gr(US, 2, "Savings", "US_p2_cause_Savings")
gr(E5, 2, "gnp", "E5_p2_cause_gnp")

## lmtest::grangertest -- SINGLE-EQUATION F test, bivariate only
gt <- function(y, x, order, tag) {
  g <- grangertest(y ~ x, order=order)
  out[[tag]] <<- list(order=order, res_df_unrestricted=g$Res.Df[1], res_df_restricted=g$Res.Df[2],
                      df=g$Df[2], F=g$F[2], p=g$`Pr(>F)`[2])
}
gt(A[,"y1"], A[,"y2"], 1, "lm_A_p1_y2_to_y1")
gt(A[,"y1"], A[,"y2"], 2, "lm_A_p2_y2_to_y1")
gt(A[,"y2"], A[,"y1"], 2, "lm_A_p2_y1_to_y2")
gt(A[,"y1"], A[,"y2"], 4, "lm_A_p4_y2_to_y1")
gt(US[,"Consumption"], US[,"Income"], 2, "lm_US_p2_Inc_to_Cons")
gt(B3[,"x3"], B3[,"x1"], 2, "lm_B_p2_x1_to_x3")

## refusals
out$err_cause_all   <- tryCatch({causality(VAR(A,p=2,type="const"), cause=c("y1","y2")); "no error"},
                                error=function(e) conditionMessage(e))
out$err_cause_bad   <- tryCatch({causality(VAR(A,p=2,type="const"), cause="nope"); "no error"},
                                error=function(e) conditionMessage(e))
out$err_cause_none  <- tryCatch({causality(VAR(A,p=2,type="const")); "no error"},
                                error=function(e) conditionMessage(e))
out$meta <- list(R=R.version.string, vars=as.character(packageVersion("vars")),
                 lmtest=as.character(packageVersion("lmtest")))
write(toJSON(out, digits=17, auto_unbox=TRUE, matrix="rowmajor"), "fixtures/ref_R_granger.json")
cat("granger R refs written\n")
