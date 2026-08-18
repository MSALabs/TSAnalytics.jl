
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
here <- dirname(normalizePath(file_arg))
cases <- fromJSON(file.path(here, "cases.json"), simplifyDataFrame = FALSE)

data_cache <- new.env()
get_data <- function(name) {
  if (!exists(name, envir = data_cache, inherits = FALSE)) {
    y <- read.csv(file.path(here, "data", paste0(name, ".csv")))$y
    assign(name, y, envir = data_cache)
  }
  get(name, envir = data_cache, inherits = FALSE)
}

results <- data.frame(id = integer(0), r_loglik = double(0), r_error = character(0))

for (c in cases) {
  y <- get_data(c$dataset)
  ar <- unlist(c$ar); ma <- unlist(c$ma)
  sar <- unlist(c$sar); sma <- unlist(c$sma)
  fixed <- c(ar, ma, sar, sma)

  seasonal <- if (c$P > 0 || c$Q > 0) {
    list(order = c(c$P, 0, c$Q), period = c$s)
  } else {
    list(order = c(0, 0, 0), period = 0)
  }

  ll <- tryCatch({
    fit <- arima(y, order = c(c$p, 0, c$q), seasonal = seasonal,
                 include.mean = FALSE, fixed = fixed,
                 transform.pars = FALSE, optim.control = list(maxit = 0))
    as.numeric(fit$loglik)
  }, error = function(e) NA_real_)

  results <- rbind(results, data.frame(id = c$id, r_loglik = ll, r_error = ifelse(is.na(ll), "error", "")))
}

write.csv(results, file.path(here, "r_results.csv"), row.names = FALSE)
cat("Wrote r_results.csv with", nrow(results), "rows\n")
