# API: Decomposition

An STL decomposition of the Indian Index of Industrial Production:
observed, trend, seasonal and remainder. The worked explanation, and when
to prefer each method, is in
[Manual: Decomposition](../manual/03-decomposition.md).

```@example apidecomp
using TSAnalytics, Plots
plot(stl_decompose(dataset("iip_india").value, 12); size = (700, 600))
```

## Classical (moving-average) decomposition

```@docs
classical_decompose
ClassicalDecomposition
```

## STL — one seasonal period

```@docs
stl_decompose
STLDecomposition
```

## MSTL — several seasonal periods

```@docs
mstl_decompose
MSTLDecomposition
```
