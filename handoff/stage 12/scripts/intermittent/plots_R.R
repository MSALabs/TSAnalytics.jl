suppressMessages(library(forecast))
y <- ts(scan("../intermittent.csv", skip=1, quiet=TRUE), frequency=1)
f <- croston(y, h=6, alpha=0.1)
cat("--- plot.forecast (base graphics) ---\n")
r <- tryCatch({ pdf("p_base.pdf"); plot(f); dev.off(); "OK" },
              error=function(e){ try(dev.off(),silent=TRUE); paste("ERROR:",conditionMessage(e)) },
              warning=function(w){ try(dev.off(),silent=TRUE); paste("WARNING:",conditionMessage(w)) })
cat(r,"\n")
cat("--- autoplot (ggplot2) ---\n")
r2 <- tryCatch({ suppressMessages(library(ggplot2)); p<-autoplot(f)
                 ggsave("p_ggplot.png", p, width=6, height=3, dpi=80); "OK" },
               error=function(e) paste("ERROR:",conditionMessage(e)))
cat(r2,"\n")
cat("--- is there any croston-specific plot/autoplot/decomposition method? ---\n")
cat("plot methods      :", paste(as.character(methods("plot"))[grep("forecast|croston", as.character(methods("plot")))], collapse=", "), "\n")
cat("autoplot methods  :", paste(grep("croston|forecast", as.character(methods("autoplot")), value=TRUE), collapse=", "), "\n")
cat("any *croston* S3  :", paste(grep("croston", ls(asNamespace("forecast"), all.names=TRUE), value=TRUE), collapse=", "), "\n")
# does plot show the demand/interval components?
cat("--- components accessible for a manual decomposition plot? ---\n")
cat("f$model$demand$x  len:", length(f$model$demand$x), "\n")
cat("f$model$period$x  len:", length(f$model$period$x), "\n")
cat("f$model$demand$fitted len:", length(f$model$demand$fitted), "\n")
