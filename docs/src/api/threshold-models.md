# API: Threshold Models

Autoregressions whose coefficients depend on which side of a threshold a
lagged value falls (Tong 1990): the self-exciting threshold AR (SETAR),
with an abrupt switch, and the logistic smooth-transition AR (LSTAR), with
a gradual one. `thDelay = d` means the threshold variable is
`y_{t−(d+1)}`, and the low regime is `z ≤ th`, inclusive -- both as in R's
`tsDyn`, so its numbers can be checked directly.

## SETAR

```@docs
fit_setar
SETARModel
select_setar
tsdyn_ic
```

## LSTAR

```@docs
fit_lstar
LSTARModel
```

## Testing for a threshold

```@docs
setar_test
ThresholdTest
```
