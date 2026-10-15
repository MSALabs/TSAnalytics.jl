suppressMessages(library(forecast)); options(digits=17)
cases <- list(
 all_zeros_n10=rep(0,10), all_zeros_n1=0,
 one_nonzero_n10_mid=c(0,0,0,5,0,0,0,0,0,0),
 one_nonzero_n10_last=c(0,0,0,0,0,0,0,0,0,5),
 one_nonzero_n10_first=c(5,0,0,0,0,0,0,0,0,0),
 len1_nonzero=3, len2_0_3=c(0,3), len2_3_0=c(3,0), len2_3_4=c(3,4),
 two_nonzero_end_nz=c(0,2,0,0,3), two_nonzero_end_zero=c(0,2,0,3,0),
 no_zeros_n6=c(1,2,3,2,1,2), negative=c(1,-1,0,2))
for (nm in names(cases)) {
  y <- cases[[nm]]
  r <- tryCatch({ f<-croston(y,h=3,alpha=0.1)
                  list(m=paste(sprintf("%.12g",as.numeric(f$mean)),collapse=" "),
                       fitnull=is.null(f$fitted), modnull=is.null(f$model), e=NA) },
                error=function(e) list(m="ERROR",fitnull=NA,modnull=NA,e=conditionMessage(e)))
  cat(sprintf("%-23s h1..h3= %-30s fitted_NULL=%-5s model_NULL=%-5s %s\n",
      nm, r$m, r$fitnull, r$modnull, if(is.na(r$e)) "" else paste("ERR:",r$e)))
}
