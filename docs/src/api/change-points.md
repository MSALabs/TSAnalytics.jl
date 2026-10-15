# API: Change Points and Structural Breaks

Two traditions of one problem. **Detection** searches for the segmentation
that minimises a cost plus a penalty per change: PELT, exact optimal
partitioning and binary segmentation, with mean, variance, mean-and-variance,
Poisson and exponential costs — `changepoint`'s and `ruptures`' problem,
reproduced exactly. **Testing and dating** follows Bai and Perron: the best
segmentation for every number of breaks, BIC selection, confidence intervals
for the break dates, the F-statistic path, and the CUSUM family of
fluctuation tests — `strucchange`'s problem, reproduced to rounding.

Breakpoints are the last index of the segment before each change. The
supF/aveF/expF p-values and the MOSUM tests are not offered: their null
distributions need tables that are not available under a compatible licence.

## Detection

```@docs
changepoints
ChangePointResult
cp_penalty
```

## Breaks: dating and testing

```@docs
breakpoints
BreakpointsResult
break_confint
fstats
FStatsResult
efp
EFPResult
```
