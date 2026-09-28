# Adjusting an Indian Series

[Chapter 36](36-calendar-effects.md) established the problem and
modelled it: Diwali moves between October and November, so no
fixed-period seasonal component can represent it, and a regressor
built from a table of real festival dates fixes that — worth about
four index points on India's industrial production index, decisively
non-zero, and earning its place by AIC.

That chapter *modelled* the effect. This one is about what it takes to
**publish an adjusted series** with the effect removed, which is a
different and harder job.

## What Chapter 36 left undone

Recall where it got to. [`fit_sarimax`](@ref) with a Diwali indicator
and a lockdown dummy gave a good fit, a coefficient with a plausible
sign and a defensible magnitude, and a clear improvement in AIC.

Everything about that is correct, and none of it is a seasonally
adjusted series.

**There is no adjusted series to publish.** The fit gives coefficients
and residuals. Producing an adjusted series means subtracting the
estimated seasonal *and* calendar effects from the original and
reporting the remainder as a series in its own right — with its own
revisions, its own history, and its own users.

**The window was a guess.** A one-month indicator says the effect lands
entirely in the festival month. Plants shut for some number of days
around Diwali, and that number does not respect month boundaries. A
weighted window spread across the weeks either side is at least as
defensible, and Chapter 36 said so. **Choosing between them by eye is
not a method**, and an agency cannot publish "we picked the one that
looked better".

**The seasonal factors are still fixed-period.** The `(0,1,1)[12]`
term assumes whatever seasonal shape remains after the Diwali
regressor is a clean twelve-month pattern. Over fifteen years that is a
strong assumption, and nothing in the fit tested it.

**There are no published diagnostics.** A user of the adjusted series
has no way to tell a good adjustment from a bad one without refitting
everything themselves.

**Nothing was said about revisions.** Next month's data changes the
seasonal factors. What is the rule?

## What X-13 does instead

The RegARIMA step of X-13 is the same model as Chapter 36's —
regression with ARIMA errors — but the surrounding machinery is what
turns a fit into a publication.

**The calendar regressors are tested, not assumed.** `aictest` decides
by AIC whether trading-day and Easter effects belong, and reports the
decision. For India the relevant effects are different but the
mechanism is the same.

**A user-defined holiday regressor is a first-class input.** This is
the part that matters for a lunisolar calendar. X-13's
`regression { user = (...) usertype = (holiday) }` block accepts a
regressor you supply, and SeasonalAdjustment.jl builds it from a real
NSE calendar:

```julia
using SeasonalAdjustment, Dates

Xdiwali = custom_holiday_regressor(
    Date(2011, 4, 1), Date(2026, 3, 1), INDIA_NSE,
    yr -> diwali_date(yr);          # your date table, one entry per year
    freq = :month,
)

res = x13(iip_series;
          regression_user = Xdiwali,
          automdl = true, outlier = true, transform = :auto)
```

!!! note "Not executed here"
    As in [Chapter 39](39-x13-from-julia.md), these blocks are written
    against the real API but are not run at build time —
    SeasonalAdjustment.jl is not a dependency of these docs.

Two details in `custom_holiday_regressor` are worth pulling out,
because both are the kind of thing that is obvious only after someone
has been caught by it.

It **skips a holiday that falls on a weekend**, because a market
closure that coincides with a day the market was already closed has no
incremental effect to explain. Real NSE holiday listings annotate
exactly this case as "no extra closure". A naive indicator would put a
`1` there and ask the model to explain a difference that does not
exist.

It **treats an untabulated year as zero, silently** — which is a
deliberate choice, not an oversight, and the reason `holidaylist`
errors loudly on a year it does not have. A calendar that quietly falls
back to only its fixed holidays for next year is worse than one that
refuses.

**Outliers are named.** The 2020 lockdown does not need a hand-built
dummy over a window you chose; `outlier = true` finds it, classifies it
as a level shift or temporary change, and reports the dates. Chapter 36
picked April–June 2020 by looking at the series, and said so. X-13
picks it by a documented rule, which is what you need when somebody
asks why those three months and not four.

**The seasonal is not fixed-period.** X-11's filters let the seasonal
shape evolve, in the same spirit as STL but with a published filter
specification rather than a span you chose.

**The diagnostics come out with the result.** `mstats(res).q` is one
number a downstream user can check. `qs(res)` tests explicitly for
seasonality *remaining* in the adjusted series, which is the question
that matters and the one Chapter 36 never asked.

## The part that does not get easier

None of this solves the underlying data problem, and it is worth being
blunt about that.

Chapter 36's disagreement box found a **nineteen-day spread** between
four equally authoritative-looking sources on the date of Guru Nanak
Jayanti 2026. The reason is structural: Diwali, Guru Nanak Jayanti and
most of India's actual festival calendar follow a lunisolar system
whose observed dates are set by regional astronomical convention and
announcement, not by arithmetic anyone can run in advance. Easter is
computable. Diwali is not.

So the holiday regressor is only ever as good as the table behind it,
the table has to be maintained by hand every year, and it cannot be
extrapolated forward with any confidence. X-13 will happily consume a
wrong regressor and produce a confident, well-diagnosed, wrong
adjustment.

The only fully reliable source is the exchange's or the government's
own published circular for the year in question. That is a
data-governance problem, not a statistical one, and no amount of
modelling machinery makes it go away.

## Why this is the motivating case

Most of the seasonal adjustment literature was written about American
and European series, where the awkward moving holiday is Easter and
Easter is computable in six lines. The machinery reflects that: Easter
has a dedicated built-in regressor in X-13 and always has.

An Indian series makes the general problem visible. There is no
built-in regressor for Diwali, there cannot be a closed-form one, and
the effect is large enough to matter — a few index points in a series
whose month-to-month variation is not much bigger. The
`regression_user` mechanism is the general answer, and India is where
you find out that you need it.

This is the argument for SeasonalAdjustment.jl's India-aware calendar
support existing at all. Wrapping X-13 is the easy part; knowing that
`INDIA_NSE` needs to be there, and that a weekend-coinciding holiday
must be skipped, comes from the series.

## Where to go next

SeasonalAdjustment.jl's own documentation has the real treatment —
its Moving Holidays chapter works this end to end, including the window
choice this chapter flagged and the diagnostics for checking whether
the adjustment worked.

This book ends at the boundary. What you have from Part III is enough
to decompose a series for your own understanding, and what
[Chapter 36](36-calendar-effects.md) gives you is enough to model a
moving holiday inside an ARIMA. Publishing an official adjusted series
is a different job, and it has its own package and its own book.

## See also

- [Chapter 36](36-calendar-effects.md) — the effect, modelled, with real data and the date-source problem in full
- [Chapter 38](38-official-seasonal-adjustment.md) — why agencies work this way
- [Chapter 39](39-x13-from-julia.md) — the interface
- [Chapter 41](41-the-frontier.md) — what else this package does not do
