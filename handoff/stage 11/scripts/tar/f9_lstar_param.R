options(digits=12)
suppressMessages(library(tsDyn))
y <- log10(lynx); n<-length(y)
lf <- lstar(y, m=2, thDelay=1, trace=FALSE, control=list(maxit=5000))
co <- coef(lf); g<-co["gamma"]; th<-co["th"]
cat("coef:\n"); print(co)
xx <- cbind(as.numeric(y)[2:113], as.numeric(y)[1:112])  # y_{t-1}, y_{t-2}
z  <- xx[,2]  # thDelay=1
Gv <- plogis(z, th, 1/g)
cat("G range:", range(Gv), " mean:", mean(Gv), "\n")
phi1 <- co[c("const.L","phiL.1","phiL.2")]; phi2 <- co[c("const.H","phiH.1","phiH.2")]
XL <- cbind(1, xx); XH <- cbind(1, xx)
fit_add <- XL%*%phi1 + (XH%*%phi2)*Gv                    # type=1 additive
fit_cvx <- (XL%*%phi1)*(1-Gv) + (XH%*%phi2)*Gv           # convex combination
Y <- as.numeric(y)[3:114]
cat("SSR additive  :", sprintf("%.12f", sum((Y-fit_add)^2)), "\n")
cat("SSR convex    :", sprintf("%.12f", sum((Y-fit_cvx)^2)), "\n")
cat("tsDyn deviance:", sprintf("%.12f", deviance(lf)), "\n")
cat("MATCH additive:", isTRUE(all.equal(sum((Y-fit_add)^2), as.numeric(deviance(lf)))), "\n")
cat("=> PARAMETERIZATION: additive. High-regime LEVEL coefs = phiL + phiH:\n")
print(phi1 + unname(phi2))
cat("\n=== vcov.lstar formula check: 2*SSR/n * H^-1 ===\n")
H <- lf$model.specific$hessian
vc <- 2*lf$model.specific$value/lf$str$n.used * solve(H)
cat("n.used=",lf$str$n.used," value(SSR)=",lf$model.specific$value,"\n")
cat("max|vcov(lf)-formula| =", max(abs(vcov(lf)-vc)), "\n")
cat("SE th =", sqrt(diag(vcov(lf)))["th"], " SE gamma =", sqrt(diag(vcov(lf)))["gamma"], "\n")
cat("\n=== summary.lstar's built-in nonlinearity test ===\n")
s <- summary(lf); cat("names(s):", paste(names(s),collapse=" "),"\n")
cat("s$nlTest:\n"); print(s[grepl("test|Test|F", names(s))])
