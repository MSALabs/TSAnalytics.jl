suppressMessages(library(changepoint))
FX <- c("fx0_regen","fxA_var","fxB_weak","fxC_null","fxD_edge","fxF_offgrid","fxG_many","fxH_spike")
out <- c()
for(fx in FX){ y <- read.csv(paste0(fx,".csv"))$y; n<-length(y)
  for(p in c(5,10,11.4075649493,17.1113474240,20,50)){
    for(ms in c(1,2)){
      cp <- cpts(cpt.mean(y,penalty="Manual",pen.value=p,method="PELT",minseglen=ms))
      out <- c(out, sprintf("%s|%.10f|%d|%s", fx, p, ms, paste(cp,collapse=",")))
    }}}
writeLines(out,"map_R.txt"); cat("wrote",length(out),"rows\n")
