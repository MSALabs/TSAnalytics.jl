options(digits=12)
suppressWarnings(suppressMessages(library(forecast)))
y <- scan("ets_y.csv", quiet=TRUE)
ty <- ts(y, frequency=4)
specs <- list(c("ANN",FALSE), c("AAN",FALSE), c("ANA",FALSE),
              c("AAA",FALSE), c("AAN",TRUE),  c("AAA",TRUE))
labs  <- c("ETS(A,N,N)","ETS(A,A,N)","ETS(A,N,A)","ETS(A,A,A)","ETS(A,Ad,N)","ETS(A,Ad,A)")
for (i in seq_along(specs)) {
  mdl <- specs[[i]][1]; dmp <- as.logical(specs[[i]][2])
  f <- ets(ty, model=mdl, damped=dmp)
  sse <- sum(residuals(f)^2)
  cat(sprintf("%-12s loglik=%14.8f aic=%13.6f aicc=%13.6f sse=%12.6f sigma2=%10.7f np=%d\n",
      labs[i], f$loglik, f$aic, f$aicc, sse, f$sigma2, length(f$par)))
  cat("             par:", paste(names(f$par), format(round(f$par,8)), sep="=", collapse=" "), "\n")
}
