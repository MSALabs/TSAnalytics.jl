options(digits=12)
y <- scan("ets_y.csv", quiet=TRUE)
ty <- ts(y, frequency=4)
a <- stl(ty, s.window="periodic")
b <- stl(ty, s.window=1201, s.degree=0, s.jump=1, t.window=7, t.jump=1, l.jump=1)
sb <- as.numeric(b$time.series[,"seasonal"])
# force exact periodicity the same way R's periodic branch does
wc <- cycle(ty); sb <- tapply(sb, wc, mean)[wc]
cat("periodic  seasonal[1:4]:", format(round(as.numeric(a$time.series[1:4,'seasonal']),10)), "\n")
cat("s.jump=1  seasonal[1:4]:", format(round(as.numeric(sb[1:4]),10)), "\n")
cat("periodic  trend[1:4]   :", format(round(as.numeric(a$time.series[1:4,'trend']),10)), "\n")
cat("s.jump=1  trend[1:4]   :", format(round(as.numeric(b$time.series[1:4,'trend']),10)), "\n")
