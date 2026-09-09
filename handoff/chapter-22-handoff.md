# Handoff: Chapter 22 — Forecasting

`docs/src/introduction/22-forecasting.md`. Target 12–13 pages.

Twenty-one chapters in, the book finally forecasts something. The delay
was deliberate — a forecast from an unchecked model is a guess with a
confidence interval — but it has been a long wait and the chapter
should acknowledge that.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. A hard dependency

**This chapter cannot be written until the benchmark methods land.**
`benchmark-forecasts-handoff.md` specifies `mean_forecast`, `naive`,
`seasonal_naive` and `drift`; none exists yet, and beats 2 and 5 depend
on all four.

Confirm they are implemented and exported before starting.

## 2. Verified material

All formulas fetched from fpp3 §5.2 and §5.5 directly rather than
recalled.

**There are four benchmark methods, not three.** The mean method is the
fourth and is the right benchmark for a series with neither trend nor
seasonality.

**Point forecasts:**

| method | ŷ(T+h) |
|---|---|
| Mean | mean of `y` |
| Naive | `y[T]` |
| Seasonal naive | the value from the same season one or more periods back |
| Drift | `y[T] + h·(y[T] − y[1])/(T − 1)` |

**Residual standard deviation** — `σ̂ = sqrt( Σeₜ² / (T − K − M) )`,
where `K` is estimated parameters and `M` is uncomputable residuals.
**The divisor differs by method and is easy to get wrong:**

| method | K | M | divisor |
|---|---|---|---|
| Mean | 1 | 0 | `T − 1` |
| Naive | 0 | 1 | `T − 1` |
| Seasonal naive | 0 | m | `T − m` |
| Drift | 1 | 1 | `T − 2` |

**Multi-step forecast standard deviation**, with `k` the integer part
of `(h−1)/m`:

| method | σ̂(h) |
|---|---|
| Mean | `σ̂·sqrt(1 + 1/T)` |
| Naive | `σ̂·sqrt(h)` |
| Seasonal naive | `σ̂·sqrt(k+1)` |
| Drift | `σ̂·sqrt(h·(1 + h/(T−1)))` |

Two details worth care. Drift carries `T − 1`, matching its point
forecast. And seasonal naive's standard deviation is a **step function**
in `h` — flat within each block of `m`, then jumping — which is a
distinctive shape and makes a good chart.

---

## 3. The chapter, beat by beat

### Beat 1 — What a forecast is (about 1 page)

**Chart 1 — a fitted model, extended forward, with no interval.**

*Reading:* a line continuing past the data. It looks authoritative and
it is nearly useless on its own, because it says nothing about how
wrong it might be. The rest of the chapter is mostly about the missing
part.

### Beat 2 — Four forecasts that require no model (about 2.5 pages)

**Chart 2 — all four benchmark methods on one series. Four panels.**

*Reading:* the mean forecast is a flat line at the average. Naive is a
flat line at the last value. Seasonal naive repeats the last full
period. Drift extends the line from first observation to last.

Each is trivial and each is the right answer for some series. Naive is
optimal for a random walk — chapter 8 showed why, and it means that for
many financial series no model beats it. Seasonal naive is
embarrassingly hard to beat on strongly seasonal data.

**Chart 3 — a sophisticated model losing to seasonal naive on a real
series.**

Find one. They are not rare.

*Reading:* this is the chapter's most useful chart and the reason
benchmarks come before models rather than after. A forecast that cannot
beat repeating last year's values has not earned its complexity, and
without the benchmark on the same axes nobody would know.

### Beat 3 — How wrong might it be (about 2.5 pages)

**Chart 4 — the naive forecast with intervals at 80% and 95%.**

*Reading:* the intervals widen as `sqrt(h)`. Report the actual formula
and show the widths. This is the same fanning as chapter 19's
integrated ARIMA forecast, for the same reason — under a random walk
uncertainty accumulates without bound.

**Chart 5 — seasonal naive's step-function intervals.**

*Reading:* the width is constant for `m` steps, then jumps, then is
constant again. The shape is unusual enough that it looks like a bug
the first time you see it. It is not — it follows directly from
`sqrt(k+1)`, because within one seasonal cycle you are always
forecasting from the same observation.

**Chart 6 — the same forecast with the residual standard deviation
computed using the wrong divisor.**

*Reading:* use the table from section 2. Computing seasonal naive's
`σ̂` with `T − 1` instead of `T − m` gives intervals that are
systematically too narrow. On monthly data with a short series the
error is not small. This is the kind of detail that separates an
implementation that is right from one that looks right.

### Beat 4 — Forecasting through a transformation (about 2 pages)

**Chart 7 — a forecast made on the log scale, back-transformed naively
and with a bias correction. Three lines.**

*Reading:* chapter 7 introduced this and here it has consequences.
Exponentiating the mean of the logs gives approximately the *median*,
which is systematically below the mean for a right-skewed series. If
someone asked for an expected value and received a median, they have a
forecast that is consistently too low and no indication of it.

State which this package's back-transformation performs. Check rather
than assume.

### Beat 5 — Measuring how wrong it was (about 2.5 pages)

**Chart 8 — forecast errors from three methods on a held-out period.**

*Reading:* introduce MAE and RMSE. RMSE punishes large errors more,
which matters when one big miss is worse than several small ones, and
that is a judgement about the application rather than about statistics.

MAPE next, with its failure stated plainly: it is undefined at zero,
asymmetric between over- and under-prediction, and meaningless for a
series that can go negative. It remains the most-used metric in
business forecasting, which is worth saying without moralising.

**Chart 9 — MASE across the same three methods.**

*Reading:* MASE divides by the naive forecast's in-sample mean absolute
error, so a value below 1 beats the naive benchmark and above 1 loses
to it. That makes it comparable across series with different units,
which none of the others are.

**The identity worth showing:** the MASE of the naive forecast is
exactly 1, by construction. It is a clean fact, it makes the metric
concrete, and it is a genuine test — this package's `mase` and its
`naive` should satisfy it to machine precision.

### Beat 6 — Where this leaves you (half a page)

You can forecast, put an honest interval around it, and check whether
it beat a benchmark that took no effort.

Everything in this chapter has been measured on one held-out period.
One split is one draw, and a method can win it by luck.

Chapter 23.

No recap.

---

## 4. The `india` box

Placed in beat 5, after the MAPE discussion.

MAPE is the default reported metric across most Indian corporate and
government forecasting practice, and a great deal of Indian data is
exactly what MAPE handles worst — series with genuine zeros, and series
with values small enough that a small absolute error becomes an
enormous percentage.

District-level agricultural output and sub-category industrial
production both do this routinely. A MAPE of 400% on such a series
usually means one month had a value near zero, not that the forecast
was catastrophic. MASE has neither problem and is barely used.

One paragraph.

---

## 5. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **fpp3 §5.2, §5.5, §5.8** | Nearly everything structural here — the four benchmarks, the interval formulas, MASE, and the insistence that benchmarks come first | The `fable` syntax |
| **Hyndman & Koehler (2006)** | MASE itself; cite it properly | The full metric survey |
| **Hamilton ch. 4** | Why the optimal forecast is a conditional expectation, which justifies everything else | The projection theory |
| **Montgomery, Jennings & Kulahci** | Forecast error as an operational cost, and why RMSE versus MAE is a business decision | Control charts |
| **Shumway & Stoffer** | Forecast intervals from the state-space view | The state-space machinery; Part VI |
| **Tsay** | That for many financial series the naive forecast is the best available, and this is a finding rather than a failure | Volatility forecasting |

**On examples:** `aus_production`, `aus_retail`, `jj`, `gafa_stock`,
`GNP23` bundled.

---

## 6. Voice

**Benchmarks before models, and mean it.** The temptation is to present
them as a warm-up before the real forecasting. Chart 3 — a real model
losing to seasonal naive — is what makes the chapter useful, and it
should not be softened.

**Do not moralise about MAPE.** It is genuinely bad for many series and
genuinely entrenched, and readers who use it daily will stop listening
if the tone is scolding.

Avoid, beyond earlier lists:

- Deriving the interval formulas. State them, use them, cite fpp3.
- Presenting MASE as the correct metric. It is better-behaved, not
  universally right.
- Calling the benchmarks "naive methods" as a group — one of them is
  literally named Naive and the collision is confusing.

---

## 7. Checklist

- [ ] Nine charts through `@example ch22`
- [ ] **All four** benchmark methods, including mean
- [ ] A real model shown losing to seasonal naive
- [ ] `sqrt(h)` widening shown for naive
- [ ] Seasonal naive's step-function intervals shown
- [ ] Wrong-divisor demonstration included
- [ ] Back-transformation bias shown; package behaviour checked and
      stated
- [ ] MAPE's failure modes stated without moralising
- [ ] MASE-equals-one identity demonstrated
- [ ] `india` box on MAPE and near-zero series
- [ ] Ends by opening chapter 23, no recap
- [ ] 12–13 pages, British-Indian spelling

---

## 8. What to do

1. **Confirm the benchmark methods exist before starting.** If they do
   not, this chapter cannot be written honestly.
2. Find a real series where a fitted model loses to seasonal naive.
   Try several from the bundle; this is the chapter's key chart and it
   must use real data.
3. Check what `boxcox_inv` does about bias correction and state it.
4. Verify the MASE-equals-one identity in this package as part of
   drafting.
5. Render every chart; honest CI note if the environment cannot.
6. Check against section 7.
