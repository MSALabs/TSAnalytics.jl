# Verification

*This chapter is planned but not yet written.*

This appendix collects the recurring material that distinguishes this
book from a standard time-series textbook: documented disagreements
between reference implementations, and Julia-specific implementation
notes. Each gets its own admonition category so a reader can recognize
the kind of aside at a glance. One worked example of each, to exercise
the styling:

Indian data is deliberately *not* one of these categories. It appears
throughout the book as ordinary worked examples on the bundled
`iip_india` series — Chapters 1, 5, 8, 14, 15, 20 and 36 among
others — rather than as a recurring aside, because a series is a
series and the material stands on the same footing as `cmort` or
`nyse`.

!!! disagreement "When Implementations Disagree"
    R's `stats::arima()` reports `nobs`/`n.used` as `n − d` (verified
    directly against real R output); Python's `statsmodels ARIMA`
    reports the full `n`. This isn't a display quirk -- `statsmodels`
    uses diffuse state augmentation internally and genuinely keeps all
    `n` observations in its likelihood, while [`ArimaModel`](@ref)
    differences the series first and genuinely only has `n − d`
    effective observations to compute a stationary likelihood on.
    Every information criterion (`aic`/`bic`) inherits whichever
    convention its `nobs` used, so comparing `AIC` across R, Python,
    and this package for the same series and order is only meaningful
    once you know which convention each one is following.

!!! julia "Under the Hood"
    [`tsvalues`](@ref) is only two methods: identity on
    `AbstractVector{<:Real}`, and `collect(Float64, x)` on everything
    else. It doesn't need a method per container type, because
    `TSFrames.TSFrame`, `TimeSeries.TimeArray`, and
    `DataFrames.DataFrame` never actually reach it as containers --
    `tsf[:, :Close]`, `values(ta)`, and `df.Close` each already return
    a plain `Vector` using that package's *own* accessor, before
    `TSAnalytics` code runs at all. There is no
    TSAnalytics-side integration layer for any of them.
