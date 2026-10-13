set.seed(20261018)
n <- 250
## Fixture A: bivariate cointegrated.  x ~ I(1), z = 2 + 1.5x + stationary error
x <- cumsum(rnorm(n))
z <- 2 + 1.5*x + arima.sim(list(ar=0.4), n, sd=0.8)
write.csv(data.frame(x=round(x,12), z=round(as.numeric(z),12)), "fixtures/coint2_yes.csv", row.names=FALSE)

## Fixture B: bivariate NOT cointegrated -- two independent random walks
set.seed(20261019)
w1 <- cumsum(rnorm(n)); w2 <- cumsum(rnorm(n))
write.csv(data.frame(w1=round(w1,12), w2=round(w2,12)), "fixtures/coint2_no.csv", row.names=FALSE)

## Fixture C: trivariate, ONE cointegrating relation
set.seed(20261020)
f <- cumsum(rnorm(n)); g <- cumsum(rnorm(n))
h <- 1.0*f - 0.5*g + arima.sim(list(ar=0.3), n, sd=0.6)
write.csv(data.frame(f=round(f,12), g=round(g,12), h=round(as.numeric(h),12)),
          "fixtures/coint3_r1.csv", row.names=FALSE)

## Fixture D: trivariate, TWO cointegrating relations (one common trend)
set.seed(20261021)
tr <- cumsum(rnorm(n))
d1 <- tr + rnorm(n)*0.5; d2 <- 2*tr + rnorm(n)*0.5; d3 <- -tr + rnorm(n)*0.5
write.csv(data.frame(d1=round(d1,12), d2=round(d2,12), d3=round(d3,12)),
          "fixtures/coint3_r2.csv", row.names=FALSE)

## Fixture E: cointegrated with a deterministic trend in the relation
set.seed(20261022)
q <- cumsum(rnorm(n))
s <- 1 + 0.02*(1:n) + 0.8*q + rnorm(n)*0.7
write.csv(data.frame(q=round(q,12), s=round(s,12)), "fixtures/coint2_trend.csv", row.names=FALSE)
cat("coint fixtures written\n")
for (f in c("coint2_yes","coint2_no","coint3_r1","coint3_r2","coint2_trend"))
  cat(f, dim(read.csv(sprintf("fixtures/%s.csv",f))), "\n")
