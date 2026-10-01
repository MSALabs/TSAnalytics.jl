options(digits=12)
y <- scan("ets_y.csv", quiet=TRUE)
d <- stl(ts(y, frequency=4), s.window="periodic")
s <- as.numeric(d$time.series[,"seasonal"])
cat("first 12 seasonal:\n"); cat(" ", format(round(s[1:12],10)), "\n")
cat("cycle-identical (1:4 vs 5:8 vs 9:12): ",
    isTRUE(all.equal(s[1:4], s[5:8])) && isTRUE(all.equal(s[5:8], s[9:12])), "\n")
cat("exact equality: ", identical(s[1:4], s[5:8]), "\n")
cat("max |s[t] - s[t+4]| over series: ", max(abs(s[1:(length(s)-4)] - s[5:length(s)])), "\n")
