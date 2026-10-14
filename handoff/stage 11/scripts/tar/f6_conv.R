options(digits=12)
suppressMessages(library(tsDyn))
y <- log10(lynx)
cat("=== A. boundary: <= vs < at threshold ===\n")
m1 <- setar(y, m=2, thDelay=1); th <- as.numeric(tsDyn:::getTh(coef(m1)))
z  <- as.numeric(y)[1:112]           # y_{t-2} for t=3..114  (thDelay=1 -> lag 2)
cat("th =", sprintf("%.12f",th), "; # z exactly == th:", sum(z==th), "\n")
reg <- na.omit(m1$model.specific$regime)
cat("sum(z<=th)=", sum(z<=th), " sum(z<th)=", sum(z<th), " regime==1 count=", sum(reg==1), "\n")
i <- which(z==th); cat("index with z==th:", i, " -> assigned regime:", reg[i], " (1=Low)\n")
cat("CONCLUSION: low regime is z <=", "th\n")

cat("\n=== B. threshold variable identity: thDelay d -> y_{t-(d+1)} ? ===\n")
for (dl in 0:1) {
  mm <- setar(y, m=2, thDelay=dl)
  tv <- as.numeric(mm$model.specific$thVar)
  cat("thDelay=",dl," thVar length=",length(tv)," matches y_{t-1}:",
      isTRUE(all.equal(tv, as.numeric(y)[2:113])), " matches y_{t-2}:",
      isTRUE(all.equal(tv, as.numeric(y)[1:112])), "\n")
  cat("   mTh =", mm$model.specific$mTh, "\n")
}

cat("\n=== C. include= (intercept per regime) ===\n")
for (inc in c("const","none","trend","both")) {
  mm <- try(setar(y, m=2, thDelay=1, include=inc), silent=TRUE)
  if (inherits(mm,"try-error")) { cat(inc,": ERROR\n"); next }
  cat(sprintf("%-6s k=%d coefnames: %s\n", inc, mm$k, paste(names(coef(mm)),collapse=",")))
}
cat("\n=== C2. common= ===\n")
for (cm in c("none","include")) {
  mm <- try(setar(y, m=2, thDelay=1, common=cm), silent=TRUE)
  if(inherits(mm,"try-error")) {cat(cm,": ERROR:",conditionMessage(attr(mm,"condition")),"\n"); next}
  cat(sprintf("%-8s k=%d coefnames: %s\n", cm, mm$k, paste(names(coef(mm)),collapse=",")))
}

cat("\n=== D. nthresh=2 ===\n")
m2 <- setar(y, m=2, thDelay=1, nthresh=2)
print(summary(m2))
cat("coef:", paste(sprintf("%.12f",coef(m2)),collapse=" "),"\n")
cat("names:", paste(names(coef(m2)),collapse=" "),"\n")
cat("AIC=",AIC(m2)," SSR=",deviance(m2)," k=",m2$k," RegProp=",m2$model.specific$RegProp,"\n")
cat("regime table:\n"); print(table(m2$model.specific$regime, useNA="ifany"))

cat("\n=== E. mL != mH ===\n")
m3 <- setar(y, m=3, mL=1, mH=3, thDelay=1)
cat("coefnames:", paste(names(coef(m3)),collapse=","),"\n")
cat("coef:", paste(sprintf("%.12f",coef(m3)),collapse=" "),"\n")
cat("k=",m3$k," SSR=",deviance(m3)," AIC=",AIC(m3)," mL=",m3$model.specific$mL," mH=",m3$model.specific$mH,"\n")

cat("\n=== F. ML / MH non-contiguous lag subsets ===\n")
m4 <- setar(y, m=3, ML=c(1,3), MH=c(1,2), thDelay=1)
cat("coefnames:", paste(names(coef(m4)),collapse=","),"  k=",m4$k," SSR=",deviance(m4),"\n")

cat("\n=== G. fixed th given by user (fixedTh) ===\n")
m5 <- setar(y, m=2, thDelay=1, th=3.31005573775)
cat("k=",m5$k," (vs",m1$k,"when searched) fixedTh=",m5$model.specific$fixedTh," AIC=",AIC(m5)," AIC searched=",AIC(m1),"\n")

cat("\n=== H. model='MTAR' ===\n")
m6 <- setar(y, m=2, thDelay=1, model="MTAR")
cat("th=",sprintf("%.12f",tsDyn:::getTh(coef(m6)))," SSR=",deviance(m6)," thVar head:",head(as.numeric(m6$model.specific$thVar),3),"\n")
cat("   (MTAR uses differenced threshold variable)\n")
