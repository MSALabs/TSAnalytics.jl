# Stage 9.3 reference: R forecast::thetaf, the standard Theta method.
suppressMessages(library(forecast))
f <- function(lbl, y, m, h) {
  yt <- ts(y, frequency = m)
  fc <- thetaf(yt, h = h, level = c(80, 95))
  # the pieces thetaf computes internally, recomputed here so each can be
  # checked separately rather than only the final forecast
  xx <- yt
  seas <- FALSE
  if (m > 1) {
    r <- as.numeric(acf(yt, lag.max = m, plot = FALSE)$acf)[-1]
    stat <- sqrt((1 + 2 * sum(r[-m]^2)) / length(yt))
    seas <- (abs(r[m]) / stat > qnorm(0.95))
    cat(sprintf("%s  acf_m=%.10f  stat=%.10f  ratio=%.10f  seasonal=%s\n",
                lbl, r[m], stat, abs(r[m])/stat, seas))
    if (seas) { d <- decompose(yt, type = "multiplicative"); xx <- seasadj(d)
                cat(sprintf("%s  figure=%s\n", lbl,
                    paste(sprintf("%.10f", d$figure), collapse=" "))) }
  }
  s <- ses(xx, h = h)
  b2 <- lsfit(0:(length(xx) - 1), xx)$coefficients[2] / 2
  cat(sprintf("%s  n=%d  alpha=%.10f  sigma2=%.10f  drift_half=%.10f\n",
              lbl, length(yt), s$model$par["alpha"], s$model$sigma2, b2))
  cat(sprintf("%s  ses_level=%.10f\n", lbl, as.numeric(s$mean)[1]))
  cat(sprintf("%s  mean=%s\n", lbl, paste(sprintf("%.10f", as.numeric(fc$mean)), collapse=" ")))
  cat(sprintf("%s  lo95=%s\n", lbl, paste(sprintf("%.10f", fc$lower[,2]), collapse=" ")))
  cat(sprintf("%s  hi95=%s\n", lbl, paste(sprintf("%.10f", fc$upper[,2]), collapse=" ")))
  cat(sprintf("%s  lo80=%s\n", lbl, paste(sprintf("%.10f", fc$lower[,1]), collapse=" ")))
  cat(sprintf("%s  fitted_head=%s\n", lbl, paste(sprintf("%.10f", head(as.numeric(fc$fitted),4)), collapse=" ")))
  cat(sprintf("%s  fitted_tail=%s\n", lbl, paste(sprintf("%.10f", tail(as.numeric(fc$fitted),3)), collapse=" ")))
  cat(sprintf("%s  resid_sse=%.10f\n", lbl, sum(as.numeric(fc$residuals)^2, na.rm=TRUE)))
  cat("\n")
}
y <- scan("test/verification/theta/theta_y.csv", quiet = TRUE)
f("SIM4", y, 4, 8)
f("SIM1", y, 1, 8)            # same data, declared non-seasonal
ap <- scan("test_data/airpassengers.csv", quiet = TRUE, skip = 1, sep = ",",
           what = list(NULL, numeric()))[[2]]
f("AIRP", ap, 12, 12)
f("LAIRP", log(ap), 12, 12)
n <- scan("test_data/nile.csv", quiet = TRUE, skip = 1, sep = ",",
          what = list(NULL, numeric()))[[2]]
f("NILE", n, 1, 10)
