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
penalty_path
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
efp_boundary
```

## Plotting

The Nile's annual flow, 1871–1970, has one break: the Aswan dam, begun in
1898. The segmented series shows the segment means as a step, the break,
and Bai's confidence interval for its date as a bar at the foot. Here
that is observations 25 to 32, 1895 to 1902, `strucchange`'s interval
exactly.

```@example cpapi
using TSAnalytics, Plots, DelimitedFiles
nile = Float64.(readdlm(TSAnalytics.NILE, ',', skipstart = 1)[:, end])
cp = changepoints(nile; standardize = true)
bp = breakpoints(nile)
plot(cp; confint = break_confint(bp), size = (700, 320))
```

`penalty_path = true` draws the number of change points against the
penalty instead (data from [`penalty_path`](@ref)):

```@example cpapi
plot(cp; penalty_path = true, size = (600, 320))
```

The Bai–Perron selection curve: BIC and RSS against the number of breaks.

```@example cpapi
plot(bp; size = (700, 300))
```

The F statistic for one break at each date. No critical boundary is
drawn: the supF null distribution has no closed form, and its tables are
not available.

```@example cpapi
plot(fstats(nile); size = (600, 320))
```

The fluctuation processes do have closed-form boundaries
([`efp_boundary`](@ref)), so the test can be read off the plot: the
process leaving the band is the test rejecting.

```@example cpapi
plot(plot(efp(nile)), plot(efp(nile; type = :rec_cusum)); layout = (1, 2), size = (800, 320))
```

`diagnostic_plot(bp)` gives the residual panels of the selected
segmentation.
