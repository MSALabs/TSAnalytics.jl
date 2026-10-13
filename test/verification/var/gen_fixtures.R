# Stage 10.1 fixture generation -- deterministic, seeded.
set.seed(20261013)
n <- 300

## Fixture A: bivariate VAR(1), stationary, intercept
A1 <- matrix(c(0.5, -0.1, 0.2, 0.3), 2, 2, byrow=TRUE)   # rows = equations
nu <- c(0.4, -0.2)
Sig <- matrix(c(1.0, 0.3, 0.3, 0.75), 2, 2)
L <- chol(Sig)
E <- matrix(rnorm(n*2), n, 2) %*% L
Ya <- matrix(0, n, 2)
for (t in 2:n) Ya[t,] <- nu + A1 %*% Ya[t-1,] + E[t,]
Ya <- Ya[101:300,]                      # burn-in 100 -> n = 200
colnames(Ya) <- c("y1","y2")
write.csv(round(Ya, 12), "fixtures/var2_p1.csv", row.names=FALSE)

## Fixture B: trivariate VAR(2), stationary -- order selection should find 2
set.seed(20261014)
B1 <- matrix(c( 0.40, 0.10, 0.00,
                0.05, 0.50, 0.10,
                0.00, 0.05, 0.45), 3, 3, byrow=TRUE)
B2 <- matrix(c( 0.20, 0.00, 0.05,
                0.00, 0.15, 0.00,
                0.10, 0.00, 0.20), 3, 3, byrow=TRUE)
nu3 <- c(0.1, 0.2, -0.1)
Sig3 <- diag(c(1.0, 0.8, 1.2)); Sig3[1,2] <- Sig3[2,1] <- 0.25; Sig3[1,3] <- Sig3[3,1] <- -0.15
L3 <- chol(Sig3)
m <- 400
E3 <- matrix(rnorm(m*3), m, 3) %*% L3
Yb <- matrix(0, m, 3)
for (t in 3:m) Yb[t,] <- nu3 + B1 %*% Yb[t-1,] + B2 %*% Yb[t-2,] + E3[t,]
Yb <- Yb[151:400,]                      # n = 250
colnames(Yb) <- c("x1","x2","x3")
write.csv(round(Yb, 12), "fixtures/var3_p2.csv", row.names=FALSE)

## Fixture C: univariate-as-multivariate (k = 1) -- VAR must reduce to AR exactly
set.seed(20261015)
z <- numeric(260); for (t in 2:260) z[t] <- 0.3 + 0.6*z[t-1] + rnorm(1)
z <- z[61:260]
write.csv(data.frame(z=round(z,12)), "fixtures/var1_k1.csv", row.names=FALSE)

## Fixture D: short sample -- df edge cases
write.csv(round(Ya[1:30,],12), "fixtures/var2_short.csv", row.names=FALSE)

cat("written\n")
for (f in c("var2_p1","var3_p2","var1_k1","var2_short")) {
  d <- read.csv(sprintf("fixtures/%s.csv", f)); cat(f, dim(d), "\n")
}
