options(digits=12)
suppressMessages(library(tsDyn))
y <- as.numeric(log10(lynx)); n <- length(y); m <- 2; d <- 1

## ---- independent base-R SETAR: conditional LS grid search -------------
# regressand y_t, t = m+1..n ; regressors 1, y_{t-1}, y_{t-2}
Y  <- y[(m+1):n]
X1 <- y[(m+1):n - 1]
X2 <- y[(m+1):n - 2]
Tn <- length(Y)
cat("T (effective obs) =", Tn, "\n")

ssr_at <- function(z, th) {
  isL <- z <= th
  Xl <- cbind(1, X1, X2) * isL
  Xh <- cbind(1, X1, X2) * (!isL)
  X  <- cbind(Xl, Xh)
  fit <- lm.fit(X, Y)
  list(ssr = sum(fit$residuals^2), coef = fit$coefficients, nL = sum(isL))
}

grid_search <- function(thDelay, trim = 0.15) {
  z <- if (thDelay == 0) X1 else X2
  allTh <- sort(unique(z))
  ng <- length(allTh)
  down <- ceiling(trim * ng); up <- floor(ng * (1 - trim))
  cand <- unique(allTh[seq(from = down, to = up, length.out = up - down)])
  cat(sprintf("thDelay=%d: n unique z = %d, down=%d up=%d, n candidates = %d\n",
              thDelay, ng, down, up, length(cand)))
  ssrs <- sapply(cand, function(t) ssr_at(z, t)$ssr)
  best <- cand[which.min(ssrs)]
  list(th = best, ssr = min(ssrs), cand = cand, ssrs = ssrs,
       fit = ssr_at(z, best))
}

for (dl in 0:1) {
  g <- grid_search(dl)
  cat(sprintf("  BEST th = %.12f  SSR = %.12f  nL = %d\n", g$th, g$ssr, g$fit$nL))
  cat("  coef:", paste(sprintf("%.12f", g$fit$coef), collapse=" "), "\n")
  assign(paste0("g", dl), g)
}

cat("\n### Compare with tsDyn::setar(y, m=2, thDelay=1) ###\n")
mt <- setar(log10(lynx), m=2, thDelay=1)
ct <- coef(mt)
cat("tsDyn th      :", sprintf("%.12f", tsDyn:::getTh(ct)), "\n")
cat("handroll th   :", sprintf("%.12f", g1$th), "\n")
cat("th identical  :", identical(as.numeric(tsDyn:::getTh(ct)), g1$th), " diff=", as.numeric(tsDyn:::getTh(ct))-g1$th, "\n")
cat("tsDyn SSR     :", sprintf("%.12f", deviance(mt)), "\n")
cat("handroll SSR  :", sprintf("%.12f", g1$ssr), "\n")
cat("SSR max abs diff:", abs(deviance(mt)-g1$ssr), "\n")
cat("tsDyn coef    :", paste(sprintf("%.12f", head(ct,-1)), collapse=" "), "\n")
cat("handroll coef :", paste(sprintf("%.12f", g1$fit$coef), collapse=" "), "\n")
cat("coef max abs diff:", max(abs(as.numeric(head(ct,-1)) - as.numeric(g1$fit$coef))), "\n")
cat("tsDyn nL      :", round(mt$model.specific$RegProp[1]*Tn), " handroll nL:", g1$fit$nL, "\n")

## ---- hand-rolled SEs, replicate tsDyn convention ----------------------
z <- X2; isL <- z <= g1$th
Xd <- cbind(cbind(1,X1,X2)*isL, cbind(1,X1,X2)*(!isL))
fit <- lm.fit(Xd, Y); p <- ncol(Xd); SSR <- sum(fit$residuals^2)
XtXi <- solve(crossprod(Xd))
cat("\n--- SE variants ---\n")
for (nn in c(n, Tn)) for (pp in c(p, p+1)) {
  rv <- SSR/(nn-pp)
  cat(sprintf("df=%d-%d=%d : SE = %s\n", nn, pp, nn-pp,
    paste(sprintf("%.11f", sqrt(diag(XtXi)*rv)), collapse=" ")))
}
cat("tsDyn SE      :", paste(sprintf("%.11f", summary(mt)$coefficients[,2]), collapse=" "), "\n")
cat("\nmse(tsDyn)=", mse(mt), " SSR/n(114)=", SSR/114, " SSR/T(112)=", SSR/112, "\n")
cat("AIC(tsDyn)=", AIC(mt), " n*log(mse)+2k with n=114,k=7:", 114*log(SSR/114)+2*7,
    " with n=112:", 112*log(SSR/112)+2*7, "\n")
cat("BIC(tsDyn)=", BIC(mt), " n*log(mse)+log(n)*k n=114:", 114*log(SSR/114)+log(114)*7, "\n")
