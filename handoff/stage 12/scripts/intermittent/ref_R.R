suppressMessages(library(forecast))
options(digits=17)
g <- function(x) sprintf("%.17g", x)

e<-new.env(); load("extpkg/expsmooth/data/partx.rda", envir=e)
write.csv(data.frame(y=as.numeric(get("partx",envir=e))), "fx_partx_n51.csv", row.names=FALSE, quote=FALSE)

fx <- list(
  F1_poisson040_n60 = scan("../intermittent.csv", skip=1, quiet=TRUE),
  F2_sparse010_n120 = scan("fx_sparse_p010_n120.csv", skip=1, quiet=TRUE),
  F3_dense150_n120  = scan("fx_dense_p150_n120.csv",  skip=1, quiet=TRUE),
  F4_partx_n51      = scan("fx_partx_n51.csv", skip=1, quiet=TRUE)
)

for (nm in names(fx)) {
  y <- fx[[nm]]
  cat("\n################ ", nm, " n=", length(y), " nonzero=", sum(y>0), " sum=", sum(y), "\n", sep="")
  f <- croston(y, h=6, alpha=0.1)
  cat("method      : ", f$method, "\n")
  cat("class       : ", paste(class(f),collapse=","), "\n")
  cat("names(f)    : ", paste(names(f),collapse=","), "\n")
  cat("INTERVALS   : lower is.null=", is.null(f$lower), " upper is.null=", is.null(f$upper),
      " level is.null=", is.null(f$level), "\n", sep="")
  cat("h1..h6      : ", paste(g(as.numeric(f$mean)), collapse="  "), "\n")
  # demand / interval SES components
  d <- f$model$demand; p <- f$model$period
  cat("demand SES  : alpha=", g(d$model$par["alpha"]), " l0=", g(d$model$par["l"]),
      " fcst=", g(as.numeric(d$mean)[1]), " n_nonzero=", length(d$x), "\n", sep="")
  cat("interval SES: alpha=", g(p$model$par["alpha"]), " l0=", g(p$model$par["l"]),
      " fcst=", g(as.numeric(p$mean)[1]), " n_intervals=", length(p$x), "\n", sep="")
  cat("demand vec  : ", paste(d$x, collapse=","), "\n")
  cat("interval vec: ", paste(p$x, collapse=","), "\n")
  cat("ratio check : ", g(as.numeric(d$mean)[1]/as.numeric(p$mean)[1]), "\n")
  # fitted + accuracy
  fit <- as.numeric(f$fitted)
  cat("fitted[1:8] : ", paste(g(fit[1:8]), collapse="  "), "\n")
  cat("fitted[n-3:n]:", paste(g(tail(fit,4)), collapse="  "), "\n")
  cat("n NA fitted : ", sum(is.na(fit)), "\n")
  acc <- accuracy(f)
  cat("accuracy cols:", paste(colnames(acc), collapse=","), "\n")
  cat("accuracy vals:", paste(g(as.numeric(acc)), collapse="  "), "\n")
  # manual in-sample MASE, naive-1 denominator (sp=1)
  denom <- mean(abs(diff(y)))
  cat("MASE(sp=1) manual: ", g(mean(abs(y-fit), na.rm=TRUE)/denom),
      "  denom=", g(denom), "\n", sep="")
  # ADI / CV2 (SBC)
  nzd <- which(y!=0); k<-length(nzd); z<-y[nzd]
  iv <- c(nzd[1], if(k>1) nzd[2:k]-nzd[1:(k-1)] else numeric(0))
  cat("ADI p=", g(mean(iv)), " CV2 v=", g((sd(z)/mean(z))^2),
      "  SBC: p>1.32=", mean(iv)>1.32, " v>0.49=", (sd(z)/mean(z))^2>0.49, "\n", sep="")
}
