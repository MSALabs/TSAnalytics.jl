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

R (`stats`, `forecast`, `tseries`, `urca`, `vars`, `rugarch`, `tsDyn`,
`MSwM`, `dfms`) and Python (`statsmodels`, `pmdarima`, `arch`, `aeon`)
are used for exactly two things:

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

The package carries 303 fixture files across 41 verification
directories, and 22 ground-truth transcripts recording what was
executed to produce them. A transcript is not a summary — it is the
actual session, so a future reader can tell the difference between "R
returned this" and "the handoff said R returns this".

The full test suite runs more than 17,000 assertions (17,140 at the end of Stage 12's implementation). Nine test files carry
additional bulk suites, gated behind `TSANALYTICS_FULL_TESTS`, that
sweep hundreds or thousands of parameter combinations rather than spot
checks — the seasonal state-space bulk suite alone covers 364 cases.

`test_data/` holds the internal benchmark series — Nile,
AirPassengers, sunspots — used as validation fixtures. That is a
separate thing from `data/`, which is the public catalogue of 84
bundled textbook datasets reached through [`dataset`](@ref). Internal fixtures
are not public API, and the two are deliberately not conflated.

## The two recurring asides

Two kinds of aside appear throughout this book, each with its own
styling so you can recognise the kind at a glance.

The first records a case where two reference implementations returned
different numbers for the same question, and what was done about it.
Thirty-four of these appear across the Introduction:

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
reader coming from R or Python would not expect. Twenty-four of these
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
series — Chapters 1, 5, 8, 14, 15, 20, 40 and 41 among others — rather
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

The models of components, factors and regimes added two of the same
kind. The worked example for [`fit_dfm`](@ref) failed on the Julia
version the documentation is built with and passed on the one the tests
run on: there, matrix multiplication skips an entry whose value is zero,
and the automatic-differentiation numbers it was multiplying count as
zero when their *value* is zero even if their derivative is not — so a
derivative silently vanished. And drawing every new chart by hand, once,
found titles and axis labels that disappeared because a later series in
the same panel overwrote them, which no test of the chart's data could
see.

## What the documentation build enforces

Both gates are live in CI and a failure is treated like a test failure:

- **`checkdocs = :exports`** — every exported symbol must have a
  docstring, and a broken `@ref` fails the build.
- **`doctest = true`** — a stale `jldoctest` output fails the build.

Every `@example` block in this book executes at build time on Ubuntu
with `JULIA_NUM_THREADS=4`, so the threaded paths are genuinely
exercised. The numbers you read in the output blocks were produced by
running the code immediately above them.

## Where the standard is weaker, and where it is different

One place departs from the dual-reference norm, and it says so at the
point of use rather than only here.

!!! note "This section used to list ETS as single-verified"
    Its stage notes recorded that `forecast::ets()` "could not be
    obtained" and scoped the work against `statsmodels` alone. R's
    `forecast` 9.0.2 turned out to be installed after all, so
    [`fit_ets`](@ref) is dual-verified like everything else — on two
    series, the simulated fixture and `log(dataset("jj").value)`, across
    all six models. That re-check also found that `statsmodels`' damped
    fits stop at its `phi` bound with worse SSE than R's, which a
    single-reference validation would have adopted as correct.

**Verified against the definition instead of a package.** The
distributional scoring rules — [`crps_normal`](@ref),
[`pinball_loss`](@ref), [`winkler_score`](@ref) — have no reachable
reference implementation at all: neither R's `scoringRules` nor Python's
`properscoring` is installed here, and R's `forecast` carries none of
them. So `crps_normal`'s closed form is checked against **numerical
integration of the CRPS integral** (agreeing to `5e-13`),
`pinball_loss` against the identity that it is half the MAE at the
median *and* the defining property that it is minimised at the true
quantile, and `winkler_score` against hand computation.

That is arguably a *stronger* check than reproducing another
implementation — a package can be wrong, a definition cannot — but it is
a different kind of check, and a reader comparing numbers across
languages should know which one they are relying on.

**Cited, but not to a page in hand.** The multivariate Kalman filter
behind [`fit_dfm`](@ref) processes each period's observations one series
at a time — the univariate treatment of multivariate series in Durbin &
Koopman (2001). Of the six textbooks this project checks its citations
against, none describes it: Hamilton (1994, eqs. 13.2.15–13.2.22,
printed pp. 379–381, PDF 394–396) and Shumway & Stoffer (2017, eq. 6.22, printed p. 293,
PDF 303) both form and invert the full innovation covariance, and Hamilton's index
entry for "Durbin" is the Durbin–Watson statistic. So the algorithm is
cited to Durbin & Koopman directly, from a copy not in hand when it was
written, and its correctness rests on the numbers instead: it agrees with
`statsmodels` to `1.6e-12` and with `dfms`' C++ filter to `1.4e-11` on a
five-series fixture. The section number should be checked against a
physical copy; until it is, this is the one citation in the book that is
not to a page someone has read.

**Cited, with no table anywhere in the corpus.** The supremum F statistic
for a break at an unknown date ([`fstats`](@ref)) has a non-standard
distribution. None of the six textbooks tabulates it: Hamilton's
statistical tables (Appendix B, printed pp. 751–768) end with Johansen's,
and Montgomery's (Appendix A, printed pp. 387–406) end with Durbin–Watson;
the other four have no table appendix. Hamilton (§14.3, printed p. 425,
PDF 440) states the test and the trimming and leaves the distribution to
Andrews (1993, *Econometrica* 61, 821–856). So the citation is to Andrews
directly, and because his tables, and Hansen's (1997) approximation to
them, are not available under a compatible licence, the package reports
the statistics with no p-value at all rather than an unverified one.
[Chapter 57](57-a-change-that-does-not-come-back.md) shows by simulation
what the missing distribution does.

## The textbooks have errata too

Checking numbers against printed pages finds mistakes in the books as
well as in the software. Six are on record, each read from a rasterised
page (Hamilton's and Montgomery's PDFs are OCR scans whose text layers
drop and mangle characters):

| Where | What | How the documentation handles it |
|---|---|---|
| Hamilton §14.3 against §15.5 (printed pp. 425, 450) | The break date `T₀` is the last observation of the first regime in one section and the first of the second in the other | [Chapter 57](57-a-change-that-does-not-come-back.md) names the section whenever it cites a convention |
| Montgomery, Example 2.13 (printed p. 63) | The text computes the CUSUM interval as `H = 14.36`; Figure 2.34 shows 14.38 | The figure is right: Table 2.3's data give `5σ̂ = 14.378`, and the text's figure comes from rounding the mean moving range to 3.24 first ([Chapter 58](58-watching-a-model-drift.md)) |
| Montgomery, Table 6.13 (printed p. 337) | The intervention estimates (`θ = 0.75886`, `ω₀ = −2071.5`, se 637.4) are not reproduced from Table 6.12's data by the stated model, nor by the step in growth, with or without drift, by CSS or ML; R's `arima` agrees with this package | Reported as found, with the reproducible estimates beside it ([Chapter 57](57-a-change-that-does-not-come-back.md)) |
| Tsay, Eq. (4.9) | Minus signs before the SETAR lag terms, contradicting Eqs. (4.8) and (4.11) | [Chapter 56](56-a-different-model-on-each-side-of-a-line.md) uses the form of (4.8) |
| Tsay, Eq. (4.53) (printed p. 219, PDF 246) | The regime-1 residual variance is printed as 0.76 (regime 2: 0.165), while the Markov-switching fit to the same data on the next page gives 0.031 for its calm state; 0.076 is the likely intended value | Not reproduced |
| `fpp` notes, p. 101 | The code draws the smoothed level in blue and the filtered in red; the legend labels blue "Filtered" and red "Smoothed" | Not reproduced |

The Montgomery CUSUM entry corrects an earlier reading in this project's
own handoff notes, which took the text's 14.36 as right and the figure as
wrong. Recomputing from the data settled it.

## When a reference is simply wrong

The two asides above cover two situations: the references disagree with
each other, or they share a limitation this package shares too. The
exponential-smoothing and multivariate work needed a third — **one
reference is simply wrong**, and no second reference exists to outvote
it. Those cases were settled by a different method: simulate the
model's own recursion millions of times and see which number the
simulation agrees with. A simulation cannot be wrong in the way an
implementation can, because it uses nothing but the model's definition.

Three such cases were found in R's `forecast` package:

| What | R gives | Simulation gives | This package |
|---|---|---|---|
| ETS(A,N,A) interval variance at `h = 4`, as a multiple of `σ²` | `3.0536` | `2.5808` (4,000,000 paths) | `2.5808` |
| ETS(M,N,A) intervals at `h = 4`, from R's own fitted state | 25.6% too wide | within 0.09% of this package at every horizon (6,000,000 paths) | — |
| ETS(M,Ad,M) point forecast | damping summed as `1 + φ + … + φ^(h−1)` | `φ + … + φ^h`, which R's own one-step recursion implies (8,000,000 paths) | `φ + … + φ^h` |
| Seasonal TBATS interval half-width at `h = 8` | `2.640` | `7.31` (3,000,000 paths; Python's `tbats` agrees) | `7.31` |

The first two are one fault: a seasonal shock counted one period early,
`floor(h/m)` elapsed seasonal innovations where the correct count is
`floor((h−1)/m)`. The last builds its shock weights `w′F^(j−1)g` without the trend's `β`:
R's `forecast.tbats` reads a field, `beta.v`, that a fitted `tbats`
object does not carry.

In each case R's **neighbouring** branch is correct: trended ETS,
non-seasonal damped ETS, non-seasonal `bats`. That is how each fault was
localised to a few lines rather than written off as "R is wrong", and it
is what makes the claim credible: a package that is right next door is
not being accused of general carelessness, only of one specific slip.

The multivariate work found two more, of a different kind — not a wrong
model, but a wrong number:

- **`vars::causality` reports p = 0** for any moderately strong
  rejection, because it computes the p-value as `1 − CDF` and the
  subtraction underflows. `statsmodels` uses the survival function and
  agrees with this package.
- **`statsmodels`' `test_inst_causality` raises an error** whenever both
  the causing and the caused block have more than one variable — an
  indexing fault, verified at three block sizes. It never returns a
  wrong answer, but for a general block `vars` is the only reference.

Change points, intermittent demand and lag regression added six:

- **`changepoint`'s default** (MBIC with PELT) returns a segmentation 2.07
  worse on its own criterion than the optimum, with no warning, because
  MBIC breaks PELT's pruning rule.
- **`ruptures`' default `jump = 5`** turns true changes at 97 and 203 into
  95, 100, 200 and 205.
- **`sktime`** accepts a rolling window feature at lag 0, which is the
  target itself, and reports an in-sample R² of 1.
- **`statsforecast`'s SBA** multiplies by a literal 0.95 instead of
  `1 − α/2`.
- **`strucchange`** prints p-values of exactly 0 where the closed-form
  tail is `6.8e-23`, and loses the sixth digit of p-values near `1e-10`,
  by computing `1 − (1 − p)^k` rather than the tail directly.
- **`utilsforecast`'s `cfe`** computes the opposite sign to its own
  docstring.

The models of components, factors and regimes found the most, in
references that are otherwise careful:

- **R's `StructTS` reports a `loglik` that is not the log-likelihood** —
  it scales by the wrong count and adds a normalising term with the
  wrong sign — and clips negative variances to zero, so a reported zero
  is ambiguous. `statsmodels` agrees with this package to `1e-10`.
- **`statsmodels`' one-series-at-a-time filter silently drops nearly
  degenerate updates**: with idiosyncratic variances of `1e-12` it
  reports a log-likelihood of `−696.17` where the true value, which the
  conventional filter and this package both compute, is about `−1.4e14`.
- **`MSwM` stores the negative log-likelihood** in its `logLikel` slot,
  and under its default parallel restarts gives three different optima
  from one seed.
- **`statsmodels` returns the starting values** for the transition
  probabilities of a switching model in which nothing switches, with no
  warning; this package refuses the model.
- **`tsDyn`'s LSTAR optimiser stops short** on a likelihood that is
  nearly flat in the slope; this package reaches a lower sum of squares
  and reproduces `tsDyn`'s exactly at `tsDyn`'s parameters.

And one that is not an error on either side but would read like one: for
the Johansen test with an unrestricted constant, `urca` uses
Osterwald-Lenum's Table 1.1\*, which assumes the data have no drift,
where the paper assigns Table 1. `statsmodels`' own simulations agree
with Table 1, and so does this package. The trace statistics themselves
agree with both references to `1e-10`; only the critical values differ.

All of these are written up at their point of use — the exponential
smoothing cases in [Exponential Smoothing](../manual/10-exponential-smoothing.md),
the multivariate ones in [Causality and Impulse Responses](../manual/12-causality-and-impulse-responses.md)
and [Cointegration and VECM](../manual/13-cointegration-and-vecm.md) — and collected in
[Coming from R or Python](../manual/16-coming-from-r-python.md).

## The honest limits

Verification establishes agreement with a reference. It does not
establish correctness, and the two are not the same thing.

Where the references themselves disagree — and thirty-five chapters record
a case where they do — agreement with one is disagreement with the
other, and the choice was made on the merits rather than by majority.
Where all the references share a limitation, as they do on prediction
intervals treating estimated parameters as known, this package shares
it too and says so rather than quietly doing something different.

And a validated number on a badly specified model is still a bad
answer. That is what [Appendix A](A-checklist.md) is for.

## See also

- [Appendix A](A-checklist.md) — the checklist
- [Appendix C](C-further-reading.md) — the six reference books the policy is checked against, and the primary papers
- [Coming from R or Python](../manual/16-coming-from-r-python.md) — every documented divergence in one place
