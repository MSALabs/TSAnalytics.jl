suppressMessages(library(changepoint))
y <- read.csv("fxB_weak.csv")$y; n <- length(y); pen <- 3*log(n)
cs <- cumsum(c(0,y)); cs2 <- cumsum(c(0,y^2))
RSS <- function(a,b){ L<-b-a+1; s<-cs[b+1]-cs[a]; s2<-cs2[b+1]-cs2[a]; s2-s*s/L }
mbcost <- function(a,b) RSS(a,b)+log(b-a+1)

cat("### exact MBIC criterion, m=0 and m=1, brute force over all tau\n")
c0 <- mbcost(1,n)
cat(sprintf("m=0: crit=%.8f\n", c0))
best <- Inf; bt <- NA
v <- numeric(n-1)
for(t in 1:(n-1)){ v[t] <- mbcost(1,t)+mbcost(t+1,n)+pen; if(v[t]<best){best<-v[t]; bt<-t} }
cat(sprintf("m=1: exact argmin tau=%d  crit=%.8f\n", bt, best))
cat(sprintf("     crit at tau=139 (Manual answer) = %.8f\n", v[139]))
cat(sprintf("     crit at tau=187 (MBIC answer)   = %.8f\n", v[187]))
cat(sprintf("PELT-MBIC returned tau=%s -> suboptimal by %.8f\n",
    paste(cpts(cpt.mean(y,penalty='MBIC',method='PELT')),collapse=","),
    v[187]-best))

cat("\n### is PELT's K=0 pruning condition valid for the MBIC cost?\n")
cat("need: cost(a,s)+cost(s+1,b) <= cost(a,b)  i.e.  log L1 + log L2 <= log L\n")
viol <- 0; tot <- 0
for(k in 1:200){ a<-sample(1:(n-20),1); b<-a+sample(10:50,1); if(b>n) next
  s<-a+sample(1:(b-a-1),1); tot<-tot+1
  if(mbcost(a,s)+mbcost(s+1,b) > mbcost(a,b)) viol<-viol+1 }
cat(sprintf("random segment splits: %d/%d VIOLATE the K=0 pruning inequality\n", viol, tot))
cat(sprintf("example: L=100 split 50/50 -> logL1+logL2-logL = %.6f (>0 means violated)\n",
    log(50)+log(50)-log(100)))

cat("\n### same check for the plain RSS (mll_mean) cost\n")
viol<-0; tot<-0
for(k in 1:200){ a<-sample(1:(n-20),1); b<-a+sample(10:50,1); if(b>n) next
  s<-a+sample(1:(b-a-1),1); tot<-tot+1
  if(RSS(a,s)+RSS(s+1,b) > RSS(a,b)+1e-9) viol<-viol+1 }
cat(sprintf("random segment splits: %d/%d violate (expect 0)\n", viol, tot))
