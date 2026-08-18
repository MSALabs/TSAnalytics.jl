# API Reference

## Container-agnostic interface

```@docs
tsvalues
tsindex
```

## Differencing

```@docs
diff
diffinv
tsdiff
tsundiff
```

## Filters & smoothing

```@docs
convolution_filter
recursive_filter
moving_average
```

## Decomposition

```@docs
classical_decompose
ClassicalDecomposition
stl_decompose
STLDecomposition
mstl_decompose
MSTLDecomposition
```

## Descriptive statistics

```@docs
acf
pacf
ACFResult
```

## Unit root tests

```@docs
adf_test
kpss_test
pp_test
ADFTest
KPSSTest
PPTest
```

## Diagnostics

```@docs
ljungbox_test
qs_test
jarque_bera_test
LjungBoxTest
QSTest
JarqueBeraTest
```

## Univariate models

```@docs
arx
ARXModel
forecast
Forecast
```

## Model-fitting infrastructure

```@docs
OptimResult
partrans
invpartrans
```

## Abstract type hierarchy

```@docs
TimeSeriesModel
StateSpaceModel
UnivariateModel
HypothesisTest
statistic
pvalue
```
