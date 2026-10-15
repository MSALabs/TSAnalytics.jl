# API: Intermittent Demand

Series that are mostly zero, with occasional demands: Croston's method and
its bias-corrected (SBA) and optimised forms, TSB, the temporal-aggregation
methods ADIDA and IMAPA, and the ADI/CV² classification that chooses
between them. Every point forecast agrees with `statsforecast` exactly, and
classical Croston with R's `forecast::croston` to all 17 digits. No method
here has prediction intervals, because no reference computes any.

## Methods

```@docs
croston
tsb
adida
imapa
IntermittentModel
IntermittentForecast
```

## The decomposition and the classification

```@docs
demand_intervals
DemandIntervals
adi_cv2
ADICV2Result
```

## Plotting

A mostly-zero series is drawn as stems: a line through the zeros would
show fractional demand on days with none. Built from a fitted model,
`demand_intervals` carries the smoothed levels, and the plot shows what
Croston's method divides: the demand sizes and the intervals, each with
its level.

```@example imtapi
using TSAnalytics, Plots
y = [2, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 1, 2, 0, 0, 0, 2, 0,
     1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 2, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]
m = croston(y)
plot(demand_intervals(m); size = (700, 600))
```

The final levels, 1.2979 and 1.9692, give the forecast 0.6591 per period.
`histogram = true` shows the intervals' distribution instead, the final
interval level marked:

```@example imtapi
plot(demand_intervals(m); histogram = true, size = (600, 320))
```

The classification as a point on the ADI–CV² plane. The quadrant names are
Syntetos, Boylan and Croston's categories; `tsintermittent::idclass` draws
only the SBA-or-Croston split.

```@example imtapi
plot(adi_cv2(y); size = (600, 360))
```

`plot(forecast(m, h))` draws the flat forecast as stems as well, and
`diagnostic_plot(m)` gives the residual panels for the one-step errors.
