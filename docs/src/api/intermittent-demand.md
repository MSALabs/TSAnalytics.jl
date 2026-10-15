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
