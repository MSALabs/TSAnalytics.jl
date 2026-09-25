hetero_ar2.csv
--------------
400 observations of a deliberately heteroskedastic AR(2):
    y[t] = 0.5*y[t-1] - 0.2*y[t-2] + randn()*(0.5 + 0.8*abs(y[t-1]))
generated once in Julia (seed 31) and frozen here as a fixture rather
than regenerated in the test -- Julia's RNG stream is not guaranteed
identical across Julia versions, and the expected values below come
from R, so the data must be fixed.

Ground truth, R 4.6.0 + sandwich 3.x, via:
    d <- data.frame(y=y[3:n], l1=y[2:(n-1)], l2=y[1:(n-2)])
    m <- lm(y ~ l1 + l2, data=d)
    sqrt(diag(vcovHC(m, type="HC0")))

    coef       : 0.10738514  0.38856381  -0.12684148
    classical  : 0.12227303  0.05000284   0.05001202   (R's n-k convention)
    HC0        : 0.10502832  0.17410491   0.16766742
    HC1        : 0.10542641  0.17476482   0.16830292

Note R's classical differs from arx's by the documented sqrt(n/(n-k))
factor (arx uses the conditional-MLE n-denominator, matching
statsmodels AutoReg). The HC0 figures carry no such convention
difference and match arx's se_type=:robust exactly.
