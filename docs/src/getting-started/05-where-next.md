# Where to Go Next

You have fitted a model, checked it, and changed the settings that
matter. The rest of the documentation is organised around the *kind*
of question you are asking, not around the function list.

## Four sections, four kinds of question

| You are asking | Go to | Shape |
|---|---|---|
| "How do I *do* X?" | **[Manual](../manual/01-primitives.md)** | Task-oriented. Short, entered from search, one `@example` per task. Nothing is explained twice. |
| "*Why* does X work that way, and when does it fail?" | **[Introduction to Time Series Analysis](../introduction/01-why-model-a-time-series.md)** | A book. 37 chapters, read in order, building from autocorrelation to combined AR-GARCH models. |
| "What are the arguments to X?" | **[API Reference](../api/primitives.md)** | Every exported function and type, with its docstring. |
| "I know how to do this in R/Python" | **[Coming from R or Python](../manual/11-coming-from-r-python.md)** | Translation tables, and — more importantly — where the numbers deliberately differ. |

The division is real rather than decorative: the Manual will not
explain what a unit root *is*, and the Introduction will not be a good
place to look up an argument name. If you find yourself reading a
chapter to answer a how-do-I question, the Manual page probably exists.

## If you are new to time series

Read the Introduction in order, at least through Part II. It is written
to be read rather than consulted, and each chapter opens by connecting
to the one before. Chapters 1–12 cover everything you need before
fitting anything, which is more than most treatments admit.

## If you know the material and want the package

Skim the Manual, then use the API reference. The one thing worth
reading properly is
[Coming from R or Python](../manual/11-coming-from-r-python.md),
because it lists the places where this package returns a *different
number* than the one you are used to — and says which of those are
deliberate improvements rather than incompatibilities.

## Three design decisions worth knowing

- **No container lock-in.** Every function accepts anything
  [`tsvalues`](@ref) can be called on — a `Vector`, a `TSFrame`
  column, `values(ta)` from TimeSeries, a DataFrame column. There is
  no adapter layer, because none is needed. See
  [Installation](01-installation.md).

- **Response-surface p-values by default.** `adf_test`/`pp_test` use
  the same finite-sample MacKinnon response-surface method R and
  `statsmodels` do, rather than interpolating a printed table;
  `kpss_test` offers `nlags=:auto` (Hobijn et al. 1998) alongside its
  `:short` default. See [`ADFTest`](@ref), [`KPSSTest`](@ref),
  [`PPTest`](@ref).

- **Reference, never port.** Every algorithm is implemented natively
  from its primary paper or textbook, then validated against real R
  and Python *output numbers*. No routine here is a translation of
  another package's source. Where the reference implementations
  disagree with each other — which happens more than you would
  expect — the Introduction documents the disagreement rather than
  silently picking one.

## What the package does not do yet

Stated plainly, because finding out by hitting it is worse:

- **No `dist=:t`** for GARCH models — normal innovations only.
- **No VAR, VECM, or multivariate models.**
- **No `forecast`/`predict` for the regression-with-ARIMA-errors
  models** (`ArimaxModel`, `SarimaxModel`), though they fit fine.

`development-sequence.md` in the repository is the full roadmap, and
the last chapter of the Introduction,
[The Frontier](../introduction/41-the-frontier.md), discusses where
the field goes from here.
