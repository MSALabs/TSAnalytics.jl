rugarch_garch11_z.csv
---------------------
Standardized residuals (z) and raw residuals from a real
rugarch 1.5.6 sGARCH(1,1) fit, zero mean, normal errors, on the
1260-observation simulated series in
handoff/scripts/robust-se-diagnostics/garch_bench_data.csv.

Exported directly from the fit object:
    f@fit$z  and  residuals(f)

Ground truth from rugarch::signbias(f), matched by sign_bias_test to
every printed digit:

                       t-value        prob
    Sign Bias          0.76039831   0.4471594
    Negative Sign Bias 0.03979805   0.9682605
    Positive Sign Bias 0.23210355   0.8164954
    Joint Effect       2.08252126   0.5554569   (Wald chi-square, 3 df)

Note rugarch fits ONE joint regression
    z^2_t ~ c + S-_{t-1} + S-_{t-1}*e_{t-1} + S+_{t-1}*e_{t-1}
and reads three |t| values off it -- NOT three separate regressions as
the Engle & Ng paper's exposition suggests. Read from rugarch's own
source (getMethod("signbias","uGARCHfit")), not reconstructed. p-values
use the exact t distribution, matching lm's summary().
