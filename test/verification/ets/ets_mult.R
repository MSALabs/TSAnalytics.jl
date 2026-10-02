# Stage 9.1 (second half): the multiplicative-error and multiplicative-seasonal
# ETS forms. R's forecast::ets searches 15 models by default -- it excludes the
# multiplicative-trend forms (allow.multiplicative.trend=FALSE) and treats the
# additive-error/multiplicative-seasonal trio as forbidden.
suppressMessages(library(forecast))
y <- ts(scan("test/verification/ets/ets_y.csv", quiet = TRUE), frequency = 4)
cat("series positive:", all(y > 0), "  n =", length(y), "\n\n")

specs <- list(c("MNN", "FALSE"), c("MAN", "FALSE"), c("MAN", "TRUE"),
              c("MNA", "FALSE"), c("MAA", "FALSE"), c("MAA", "TRUE"),
              c("MNM", "FALSE"), c("MAM", "FALSE"), c("MAM", "TRUE"))
for (sp in specs) {
  m <- try(ets(y, model = sp[1], damped = as.logical(sp[2])), silent = TRUE)
  if (inherits(m, "try-error")) { cat(sp[1], sp[2], "FAILED\n"); next }
  lbl <- paste0(sp[1], if (sp[2] == "TRUE") "d" else "")
  cat(sprintf("%-5s method=%-12s sse=%.8f np=%d loglik=%.8f aic=%.6f aicc=%.6f sigma2=%.10f\n",
      lbl, m$method, sum(residuals(m)^2), length(m$par), m$loglik, m$aic, m$aicc, m$sigma2))
  cat(sprintf("%-5s par: %s\n", lbl,
      paste(sprintf("%s=%.8f", names(m$par), m$par), collapse = " ")))
  f <- forecast(m, h = 8, level = 95)
  cat(sprintf("%-5s mean=%s\n", lbl, paste(sprintf("%.8f", as.numeric(f$mean)), collapse = " ")))
  cat(sprintf("%-5s lo95=%s\n", lbl, paste(sprintf("%.8f", f$lower), collapse = " ")))
  cat(sprintf("%-5s fitted_head=%s\n", lbl,
      paste(sprintf("%.8f", head(as.numeric(fitted(m)), 3)), collapse = " ")))
  cat("\n")
}
cat("--- what ets() picks over its whole default space ---\n")
a <- ets(y)
cat("auto:", a$method, " aicc=", sprintf("%.6f", a$aicc), "\n")
cat("--- are the A-error/M-seasonal forms refused? ---\n")
r <- try(ets(y, model = "ANM"), silent = TRUE)
cat("ANM:", if (inherits(r, "try-error")) "refused" else r$method, "\n")

cat("\n--- the multiplicative-error likelihood, decomposed ---\n")
# -2logL for an M-error model carries a Jacobian: dy/deps = yhat, so
# logL = -(n/2)log(sum eps^2) - sum(log|yhat|)   (Hyndman et al. 2008 eq 5.3)
n <- length(y)
for (sp in specs) {
  m <- try(ets(y, model = sp[1], damped = as.logical(sp[2])), silent = TRUE)
  if (inherits(m, "try-error")) next
  lbl <- paste0(sp[1], if (sp[2] == "TRUE") "d" else "")
  jac <- sum(log(abs(as.numeric(fitted(m)))))
  rebuilt <- -n/2 * log(sum(residuals(m)^2)) - jac
  cat(sprintf("%-5s jac=%.8f  rebuilt=%.8f  reported=%.8f  diff=%.2e\n",
      lbl, jac, rebuilt, m$loglik, rebuilt - m$loglik))
}
