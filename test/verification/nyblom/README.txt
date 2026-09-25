rugarch_garch11_scores.csv
--------------------------
Per-observation score contributions (n x 3: omega, alpha1, beta1)
exported straight out of a real rugarch 1.5.6 sGARCH(1,1) fit --
f@fit$scores -- on the 1260-observation series in
handoff/scripts/robust-se-diagnostics/garch_bench_data.csv.

Using rugarch's OWN scores rather than recomputing them in Julia is
deliberate: it isolates the statistic under test. A mismatch then means
the Nyblom formula differs, not that two implementations arrived at
slightly different score matrices.

Ground truth from rugarch::nyblom(f):

    joint      : 0.6638156567
    individual : 0.2633213626  0.3825469283  0.2736797979

matched by nyblom_test to all 10 printed digits.

Critical values, from rugarch:::.nyblomCritical (transcribed into
_NYBLOM_CRITICAL, k = 1..20):
    k=1 : 0.353 0.470 0.748
    k=3 : 0.846 1.010 1.350
