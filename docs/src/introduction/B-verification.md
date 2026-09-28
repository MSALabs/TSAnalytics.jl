# Appendix B: Verification

Every number in this book is a real number. Not a plausible number, not
one that came out of the package and looked about right — one that was
produced by running something, and in almost every case by running
something else as well and comparing.

This appendix sets out what that means in practice, because it is the
claim the rest of the book rests on.

## The standard

**Reference, never port.** Every algorithm here is implemented natively
in Julia from its primary source — the original paper or the textbook.
Never by translating another package's source code, in any language.

R (`stats`, `forecast`, `tseries`, `urca`, `rugarch`) and Python
(`statsmodels`, `pmdarima`, `arch`) are used for exactly two things:

1. **Resolving ambiguity** when a paper underspecifies an
   implementation detail, by checking what a mature implementation
   actually does; and
2. **Validating output numbers** on standard series.

Never to copy code from. That distinction is why the disagreements
documented throughout this book are *decisions* rather than accidents —
each one was reached by finding out that two references differ, working
out which is right or whether both are, and choosing deliberately.

**Every function gets a test against a real reference number.** R or
Python computed once, hardcoded as the expected value, or a
hand-verified computation. Not "it runs without erroring".

**Tolerance-based, never exact equality.** Different implementations
converge to slightly different numbers even when both are correct —
different optimisers, different stopping rules, different orders of
floating-point operations. A test that demands exact equality across
languages is testing the wrong thing.

## What that looks like in the repository

The package carries 173 fixture files across 20 verification
directories, and 22 ground-truth transcripts recording what was
executed to produce them. A transcript is not a summary — it is the
actual session, so a future reader can tell the difference between "R
returned this" and "the handoff said R returns this".

The full test suite runs 7,428 assertions. Nine test files carry
additional bulk suites, gated behind `TSANALYTICS_FULL_TESTS`, that
sweep hundreds or thousands of parameter combinations rather than spot
checks — the seasonal state-space bulk suite alone covers 364 cases.

`test_data/` holds the internal benchmark series — Nile,
AirPassengers, sunspots — used as validation fixtures. That is a
separate thing from `data/`, which is the public catalogue of 90
textbook datasets reached through [`dataset`](@ref). Internal fixtures
are not public API, and the two are deliberately not conflated.

## The two recurring asides

Two kinds of aside appear throughout this book, each with its own
styling so you can recognise the kind at a glance.

The first records a case where two reference implementations returned
different numbers for the same question, and what was done about it.
Eighteen of these appear across the Introduction:

!!! disagreement "When Implementations Disagree"
    R's `stats::arima()` reports `nobs`/`n.used` as `n − d` (verified
    directly against real R output); Python's `statsmodels ARIMA`
    reports the full `n`. This is not a display quirk — `statsmodels`
    uses diffuse state augmentation internally and genuinely keeps all
    `n` observations in its likelihood, while [`ArimaModel`](@ref)
    differences the series first and genuinely only has `n − d`
    effective observations to compute a stationary likelihood on.

    Every information criterion (`aic`/`bic`) inherits whichever
    convention its `nobs` used, so comparing AIC across R, Python and
    this package for the same series and order is only meaningful once
    you know which convention each one is following.

The second records something about the Julia implementation that a
reader coming from R or Python would not expect. Twelve of these
appear:

!!! julia "Under the Hood"
    [`tsvalues`](@ref) is only two methods: identity on
    `AbstractVector{<:Real}`, and `collect(Float64, x)` on everything
    else. It does not need a method per container type, because
    `TSFrames.TSFrame`, `TimeSeries.TimeArray` and
    `DataFrames.DataFrame` never actually reach it as containers —
    `tsf[:, :Close]`, `values(ta)` and `df.Close` each already return a
    plain `Vector` using that package's *own* accessor, before any
    TSAnalytics code runs.

    There is no TSAnalytics-side integration layer for any of them, and
    that is the whole design: the package is container-agnostic by
    construction rather than by adaptation.

Indian data is deliberately **not** one of these categories. It appears
throughout as ordinary worked examples on the bundled `iip_india`
series — Chapters 1, 5, 8, 14, 15, 20, 36 and 37 among others — rather
than as a recurring aside, because a series is a series and the
material stands on the same footing as `cmort` or `nyse`.

## Verification caught real errors

This is the part worth stating plainly, because "we verified it" is
cheap to say.

Verifying against a live reference rather than transcribing a handoff's
own proposed values caught, among others: a missing `+1` for `sigma2`
in the ARMA AIC/BIC calculation; `partrans` not being ForwardDiff-safe;
a standard-error convention in `arx` that differed from `AutoReg`'s
actual reported `bse` by `sqrt(n/(n-k))`; a seasonal-dummy reference
category off by one against `AutoReg`'s own; and a residual definition
that was double-scaling by `sqrt(sigma2)` and producing residuals with
a third of the correct spread.

None of those would have failed a test written against the
implementation's own output. All of them failed a test written against
somebody else's.

Writing this book caught three more: [`stl_decompose`](@ref) crashing
with a `DimensionMismatch` on a large outlier under `robust=true`;
`fit_arma(y, (1,0); method=:css_ml)` silently returning a unit root
with `loglik = -Inf` while reporting `converged = true`; and a test
that passed in isolation but errored under the full suite because
`std` was not imported. Documentation that executes is a test suite
with a different shape.

## What the documentation build enforces

Both gates are live in CI and a failure is treated like a test failure:

- **`checkdocs = :exports`** — every exported symbol must have a
  docstring, and a broken `@ref` fails the build.
- **`doctest = true`** — a stale `jldoctest` output fails the build.

Every `@example` block in this book executes at build time on Ubuntu
with `JULIA_NUM_THREADS=4`, so the threaded paths are genuinely
exercised. The numbers you read in the output blocks were produced by
running the code immediately above them.

## The honest limits

Verification establishes agreement with a reference. It does not
establish correctness, and the two are not the same thing.

Where the references themselves disagree — and eighteen chapters record
a case where they do — agreement with one is disagreement with the
other, and the choice was made on the merits rather than by majority.
Where all the references share a limitation, as they do on prediction
intervals treating estimated parameters as known, this package shares
it too and says so rather than quietly doing something different.

And a validated number on a badly specified model is still a bad
answer. That is what [Appendix A](A-checklist.md) is for.

## See also

- [Appendix A](A-checklist.md) — the checklist
- [Appendix C](C-further-reading.md) — the six reference books the policy is checked against
- [Coming from R or Python](../manual/11-coming-from-r-python.md) — every documented divergence in one place
