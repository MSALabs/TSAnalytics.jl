options(digits=12)
suppressMessages(library(tsDyn))
y <- as.numeric(log10(lynx)); n<-length(y); m<-2
Y<-y[3:n]; X1<-y[2:(n-1)]; X2<-y[1:(n-2)]
ssr_at <- function(z,th){isL<-z<=th; X<-cbind(cbind(1,X1,X2)*isL, cbind(1,X1,X2)*(!isL))
  if(min(sum(isL),sum(!isL))<4) return(NA); sum(lm.fit(X,Y)$residuals^2)}
z<-X2; allTh<-sort(unique(z)); ng<-length(allTh)
down<-ceiling(0.15*ng); up<-floor(ng*0.85)
idx_tsdyn <- seq(from=down,to=up,length.out=up-down)
cat("ng=",ng," down=",down," up=",up," length.out=",up-down,"\n")
cat("first 6 raw seq indices:", paste(sprintf("%.4f",head(idx_tsdyn,6)),collapse=" "),"\n")
cand_tsdyn <- unique(allTh[idx_tsdyn])
cand_full  <- allTh[down:up]
cat("n cand tsDyn-style =",length(cand_tsdyn)," n cand full range =",length(cand_full),"\n")
cat("skipped values (in full but not tsDyn grid):", length(setdiff(cand_full,cand_tsdyn)),"\n")
s1<-sapply(cand_tsdyn,function(t)ssr_at(z,t)); s2<-sapply(cand_full,function(t)ssr_at(z,t))
cat(sprintf("argmin tsDyn-grid th=%.12f SSR=%.12f\n", cand_tsdyn[which.min(s1)], min(s1,na.rm=TRUE)))
cat(sprintf("argmin FULL-grid  th=%.12f SSR=%.12f\n", cand_full[which.min(s2)], min(s2,na.rm=TRUE)))
cat("SAME ARGMIN:", isTRUE(all.equal(cand_tsdyn[which.min(s1)], cand_full[which.min(s2)])),"\n")
# also thDelay=0
z<-X1; allTh<-sort(unique(z)); ng<-length(allTh); down<-ceiling(0.15*ng); up<-floor(ng*0.85)
c1<-unique(allTh[seq(down,up,length.out=up-down)]); c2<-allTh[down:up]
s1<-sapply(c1,function(t)ssr_at(z,t)); s2<-sapply(c2,function(t)ssr_at(z,t))
cat(sprintf("delay0: tsDyn-grid th=%.12f SSR=%.12f | FULL th=%.12f SSR=%.12f | same=%s\n",
  c1[which.min(s1)],min(s1,na.rm=TRUE), c2[which.min(s2)],min(s2,na.rm=TRUE),
  isTRUE(all.equal(c1[which.min(s1)],c2[which.min(s2)]))))
