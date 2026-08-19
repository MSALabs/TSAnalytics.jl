library(jsonlite)

args_env <- commandArgs(trailingOnly = FALSE)
script_arg <- sub("^--file=", "", args_env[grep("^--file=", args_env)])
DIR <- dirname(normalizePath(script_arg))
manifest <- fromJSON(file.path(DIR, "manifest.json"))

results <- list()
for (i in seq_len(nrow(manifest))) {
  name <- manifest$name[i]
  p <- manifest$p[i]; d <- manifest$d[i]; q <- manifest$q[i]
  y <- read.csv(file.path(DIR, "data", paste0(name, ".csv")))$x
  res <- tryCatch({
    m <- arima(y, order = c(p, d, q), include.mean = FALSE, method = "ML")
    ar <- if (p > 0) as.numeric(coef(m)[paste0("ar", 1:p)]) else numeric(0)
    ma <- if (q > 0) as.numeric(coef(m)[paste0("ma", 1:q)]) else numeric(0)
    se <- sqrt(diag(m$var.coef))
    list(ar = ar, ma = ma, loglik = m$loglik, aic = m$aic, bic = BIC(m),
         nobs = m$nobs, se = as.numeric(se))
  }, warning = function(w) {
    m <- suppressWarnings(arima(y, order = c(p, d, q), include.mean = FALSE, method = "ML"))
    ar <- if (p > 0) as.numeric(coef(m)[paste0("ar", 1:p)]) else numeric(0)
    ma <- if (q > 0) as.numeric(coef(m)[paste0("ma", 1:q)]) else numeric(0)
    se <- sqrt(diag(m$var.coef))
    list(ar = ar, ma = ma, loglik = m$loglik, aic = m$aic, bic = BIC(m),
         nobs = m$nobs, se = as.numeric(se), warning = as.character(w))
  }, error = function(e) list(error = as.character(e)))
  results[[name]] <- res
  cat(name, ": ", toJSON(res, auto_unbox = TRUE), "\n")
}

write(toJSON(results, auto_unbox = TRUE, digits = 15), file.path(DIR, "r_results.json"))
cat("done\n")
