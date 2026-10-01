suppressMessages(library(rugarch))
e <- as.numeric(read.csv("garch_bench_data.csv", header=FALSE)[,1])
spec <- ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                   mean.model=list(armaOrder=c(0,0), include.mean=FALSE), distribution.model="norm")
runfit <- function(lbl, ctl) {
  f <- tryCatch(ugarchfit(spec, e, solver="solnp", fit.control=ctl), error=function(err) NULL)
  if (is.null(f)) { cat(sprintf("%-28s FAILED\n", lbl)); return(invisible()) }
  cf <- coef(f)
  cat(sprintf("%-28s omega=%.8f alpha=%.8f beta=%.8f  LL=%.6f\n",
              lbl, cf["omega"], cf["alpha1"], cf["beta1"], likelihood(f)))
}
# rec.init semantics, from rugarch's own ugarchfit-methods.Rd:
#   'all'        -- use every observation for the unconditional variance
#   integer >= 1 -- the NUMBER OF DATA POINTS to use for that calculation
#   value < 1    -- the EWMA weighting (decay)
# The first version of this script passed variance VALUES (0.9504, 1.9017)
# where rugarch expects a count or a decay, so two of its four runs were
# answering a different question than their labels claimed. 0.9504 happened
# to land near the right answer only because it is close to the 0.94 decay.
runfit("default (rec.init='all')",  list())
runfit("rec.init=0.94  (EWMA decay, matches this package)", list(rec.init=0.94))
runfit("rec.init=75    (first 75 obs)", list(rec.init=75))

# ---------------------------------------------------------------------------
# Re-executed 2026-10-01 against rugarch 1.5.6 (which IS installed, contrary
# to Stage 7.1's note). Results:
#
#   default ('all')                 omega=0.0291057576 alpha=0.0874393226
#                                   beta=0.8985961381  LL=-2079.60250317
#   rec.init=0.94  (EWMA decay)     omega=0.0302788081 alpha=0.0872381062
#                                   beta=0.8982327206  LL=-2078.54246871
#   rec.init=75    (first 75 obs)   omega=0.0290786294 alpha=0.0867474563
#                                   beta=0.8993299489  LL=-2079.07054652
#
#   this package (fit_garch)        omega=0.0301204209 alpha=0.0870910474
#                                   beta=0.8984497732  LL=-2078.55638045
#
# So rec.init=0.94 closes the log-likelihood gap from 1.046 to 0.014 and the
# coefficients to ~1.5e-04. The residual is that rugarch's EWMA runs over the
# whole sample where this package and Python's `arch` use the first 75
# observations.
#
# The handoff's own seed diagnosis is CONFIRMED exactly. Via ugarchfilter with
# fixed.pars set to this package's fitted parameters:
#   rugarch LL at those params = -2079.60906763,  sigma2[1] = 1.9016916513
# which is mean(e^2) to all printed digits, and matches the figure the handoff
# quoted. Note ugarchfilter ignores rec.init, so alignment only works at fit
# time, not filter time.
# ---------------------------------------------------------------------------
