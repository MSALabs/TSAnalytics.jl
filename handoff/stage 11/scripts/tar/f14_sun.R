options(digits=12)
suppressMessages(library(tsDyn))
## Hansen (1999) transformation, the tsDyn help-page example
sun <- (sqrt(sunspot.year + 1) - 1) * 2
cat("sunspot.year: n=",length(sun)," range ",start(sunspot.year)[1],"-",end(sunspot.year)[1],"\n")
cat("sun head6:", paste(sprintf("%.12f",head(as.numeric(sun),6)),collapse=" "),"\n")
cat("sun sum:",sum(sun)," mean:",mean(sun),"\n\n")
cat("=== setar(sun, m=11, thDelay=1) [Hansen spec] ===\n")
m <- suppressWarnings(setar(sun, m=11, thDelay=1))
cat("th=",sprintf("%.12f",as.numeric(tsDyn:::getTh(coef(m))))," SSR=",sprintf("%.12f",deviance(m)),
    " AIC=",AIC(m)," k=",m$k," RegProp=",m$model.specific$RegProp,"\n")
print(round(coef(m),10))
cat("\n=== selectSETAR(sun, m=11, thDelay=0:1, criterion='SSR') ===\n")
t0<-Sys.time()
s <- selectSETAR(sun, m=11, thDelay=0:1, criterion="SSR", trace=FALSE, plot=FALSE, nthresh=1)
cat("elapsed",as.numeric(difftime(Sys.time(),t0,units="secs")),"s; bests:\n"); print(s$bests)
cat("\n=== also sunspotarea ===\n")
sa <- log(sunspot.month[1:600])
cat("skip\n")
