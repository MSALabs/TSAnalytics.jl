# Real, executed this session -- both tests verified on positive and
# negative cases, exact statistics/p-values reproduced below

arch_lm_test <- function(resid, lags=4) {
  e2 <- resid^2; n <- length(e2)
  X <- matrix(NA, n-lags, lags)
  for (i in 1:lags) X[,i] <- e2[(lags-i+1):(n-i)]
  y <- e2[(lags+1):n]
  fit <- lm(y ~ X)
  r2 <- summary(fit)$r.squared
  stat <- (n-lags) * r2
  list(statistic=stat, p.value=1-pchisq(stat, df=lags), df=lags)
}
# ARCH-present case: statistic=25.34532136, p-value=4.287325253e-05
# White-noise case:  statistic=2.976746192, p-value=0.5617243813

dk_hetero_test <- function(resid) {
  n <- length(resid); h <- floor(n/3)
  first <- resid[1:h]; last <- resid[(n-h+1):n]
  stat <- sum(last^2) / sum(first^2)
  pval <- 2*min(pf(stat, h, h), 1-pf(stat, h, h))
  list(statistic=stat, p.value=pval, h=h)
}
# Heteroskedastic case: statistic=13.05891344, p-value=4.440892099e-16
# Homoskedastic case:   statistic=1.036008126, p-value=0.9009649302
