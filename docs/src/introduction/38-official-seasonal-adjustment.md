# Official Seasonal Adjustment

Part III gave you three ways to split a series into trend, seasonal and
remainder. Statistical offices do not use any of them.

That is worth sitting with. The Bureau of Labor Statistics, Eurostat,
the ONS, India's MoSPI — every one of them publishes seasonally
adjusted series, and none of them reaches for STL. They run a program
called X-13ARIMA-SEATS, whose lineage runs back to 1954, and the
reasons are as much institutional as statistical.

This chapter is a bridge. It explains why a whole other program exists,
and hands off to the package that wraps it.

## What you are doing is not what they are doing

When you call [`stl_decompose`](@ref), you are answering a question for
yourself. You will look at the components, form a view, and move on.
Nothing depends on the exact numbers, and nobody will ask you next
quarter why they changed.

A statistical office publishing an adjusted series faces a different
set of constraints:

**It will be revised, and the revisions will be scrutinised.** Next
month's data changes this month's seasonal factor. That is unavoidable
— any method that uses the whole series to estimate a component will
revise when the series grows. What an agency needs is a *revision
policy*: a documented, defensible rule for when factors are
re-estimated and when they are frozen. STL has no such concept.

**The method must be the same next year.** Someone will compare the
adjusted series across a decade. If the procedure changed in the middle
— a different smoother span, a different outlier rule — the comparison
is contaminated in a way that is nearly impossible to communicate.
Agencies therefore freeze their specifications, version them, and
publish them.

**The diagnostics are published too.** Not just the adjusted series but
a set of standard quality measures, so that a user can tell a
well-adjusted series from a badly adjusted one without re-running
anything. The `M` and `Q` statistics that come out of X-11 exist
precisely for this, and they are quoted in agency documentation.

**Calendar effects are modelled, not smoothed away.** February has 28
days or 29. Some months have five Mondays. Easter moves. A moving
holiday lands in different months in different years. None of these is
seasonal in the fixed-period sense, and a smoother that treats them as
noise will leave them in the "adjusted" series where they will be
mistaken for real movement. This is
[Chapter 36](36-calendar-effects.md)'s problem, at national scale.

**Extreme values are identified and explained.** Not downweighted
quietly, as `robust=true` does, but detected, classified as additive
outliers, level shifts or temporary changes, and *reported*. Someone
will ask which observations were treated as outliers, and "the bisquare
weight fell below 0.5" is not an answer an agency can publish.

## The lineage

The method behind all of this is **X-11**, developed at the US Census
Bureau and released in 1965, building on Julius Shiskin's 1954 work.
At its core it is an iterated application of moving averages — trend by
a Henderson filter, seasonal by a moving average of each calendar
position's values, both refined over several passes with outlier
adjustment between them.

X-11 is, in other words, classical decomposition taken extremely
seriously. The filters are chosen rather than assumed, the iteration
handles the interaction between trend and seasonal that the one-pass
version ignores, and the outlier treatment is explicit.

Its weakness was the endpoints. A centred moving average cannot reach
the most recent observation — exactly the observation everyone cares
about — so X-11 had to use asymmetric filters near the ends, and those
filters produced large revisions as new data arrived.

**X-11-ARIMA** (Statistics Canada, 1980) fixed this by fitting an ARIMA
model first and using it to *forecast the series forward*, so that the
symmetric filter has data to work with at what used to be the end. This
is the single most important idea in the lineage, and it is why an
ARIMA model sits inside a seasonal adjustment program at all.

**X-12-ARIMA** (Census, 1998) generalised the pre-adjustment step into
**RegARIMA**: a regression with ARIMA errors — precisely the model of
[Chapter 34](34-regression-with-arima-errors.md) — where the regressors
are calendar effects, outliers, and anything else you want removed
before the seasonal filters run.

**X-13ARIMA-SEATS** (Census, 2012) is the current version. It keeps
everything above and adds **SEATS**, an alternative adjustment engine
developed at the Bank of Spain that derives the filters from the fitted
ARIMA model itself rather than applying fixed moving averages. One
program, two adjustment philosophies, the same pre-adjustment machinery
feeding both.

## Why it is not in this package

Two reasons, one practical and one principled.

The practical one: X-13ARIMA-SEATS is the binary that national
statistical offices actually run in production. Reimplementing it means
reimplementing SEATS's spectral factorisation, X-11's exact filter
cascade, and the automatic model selection, and then convincing
somebody that your version produces the same numbers. That is a large
amount of work whose best possible outcome is "identical to the thing
that already exists".

The principled one: this is a different activity. Everything in this
book is about *understanding* a series — fitting a model, checking it,
forecasting from it. Seasonal adjustment as an agency practises it is
about *publishing* one, with all the reproducibility and
documentation requirements that implies. Mixing the two would make both
harder to explain.

So it lives in a sibling package,
[SeasonalAdjustment.jl](https://github.com/MSALabs/SeasonalAdjustment.jl),
which wraps the real Census Bureau binary. The boundary is clean:

> [`stl_decompose`](@ref) is how **you** decompose a series.
> X-13 is how a statistical office **publishes** one.

## When you need which

| Situation | Reach for |
|---|---|
| Exploring — what does this series look like underneath? | [`stl_decompose`](@ref) |
| Several seasonal periods at once (hourly, daily, weekly) | [`mstl_decompose`](@ref) |
| You need the seasonal factors to be interpretable and stable | X-13 |
| A moving holiday is in play | X-13, with a holiday regressor |
| Someone else will audit the numbers | X-13 |
| You are publishing an official statistic | X-13, and read the agency guidance |

The next two chapters show what using it looks like, and work through
the case that motivates it most sharply from inside this book's own
narrative: an Indian series with Diwali in it.

## See also

- [Chapters 13–16](13-components.md) — the decomposition methods this chapter is contrasting against
- [Chapter 34](34-regression-with-arima-errors.md) — RegARIMA, which is the same model
- [Chapter 39](39-x13-from-julia.md) — running X-13 from Julia
