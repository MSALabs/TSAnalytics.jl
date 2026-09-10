# Handoff: Chapter 28 — Realized Measures

`docs/src/introduction/28-realized-measures.md`. Target 10–11 pages.

The last chapter of Part V, and the one where the approach changes.
Everything in chapters 25 to 27 *inferred* volatility from daily
returns. This chapter *measures* it from intraday data, and the
difference is more than technical.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. A [U] claim, re-verified

Recorded claim: *"no mature Python realized-volatility library exists;
R's `highfrequency` is the reference."*

**The Python half is confirmed.** Checked directly against `arch`
8.0.0, the natural place for it:

```
arch version: 8.0.0
arch.realized exists?  False
```

There is no `realized` submodule. The package that owns GARCH in Python
does not cover realized measures.

**The R half remains unverified** — CRAN has been unreachable
throughout this project, so `highfrequency` could not be inspected.
Write the Python half as verified and the R half as reported.

### What this means for the chapter's verification standard

This is the one chapter in Part V with **no cross-language reference to
check against**. That is a real constraint and the chapter should say
so. Verification here has to come from simulation with known
properties — generate a price path with a known integrated variance,
confirm the estimator recovers it; generate one with a known jump,
confirm the jump test finds it at the right rate.

That is a weaker standard than the rest of the book uses and a stronger
one than nothing. Say which it is.

---

## 2. The chapter, beat by beat

### Beat 1 — Volatility you can see (about 1.5 pages)

**Chart 1 — one day of intraday prices at five-minute intervals, above
the same day reduced to a single daily return.**

*Reading:* the top panel contains seventy-eight observations of how the
price moved. The bottom contains one number. Chapters 25 to 27 used
only the bottom panel and inferred the day's volatility from a model.
The top panel makes it almost directly observable.

That is the shift. Volatility stops being a latent parameter and starts
being a quantity you can compute — which changes what can be checked,
because chapter 27's forecasts were being evaluated against something
nobody could see.

### Beat 2 — Add up the squares (about 2 pages)

**Chart 2 — realized variance for a series of days, beside a fitted
GARCH conditional variance for the same days.**

*Reading:* realized variance is the sum of squared intraday returns
over a day. It tracks the GARCH estimate broadly and moves more sharply
— because it is measuring rather than smoothing, and a model that
smooths cannot react to a single day the way a direct measurement can.

Report the correlation between the two. It should be high and not
close to one, and the gap is informative.

**Chart 3 — realized variance computed at several sampling
frequencies: 1-minute, 5-minute, 30-minute.**

*Reading, and this is the beat's real content:* in theory, sampling more
finely gives a better estimate. In practice the 1-minute version is
noticeably larger, and systematically so.

The reason is microstructure noise — bid-ask bounce, discrete tick
sizes, order arrival. At fine frequencies you are measuring the trading
mechanism as much as the price process, and the noise inflates the sum
of squares. Five minutes is the conventional compromise and it is a
compromise, chosen empirically rather than derived.

This is a genuinely different kind of problem from anything else in the
book — the estimator improves with more data until it starts measuring
the wrong thing.

### Beat 3 — Separating jumps from volatility (about 2.5 pages)

**Chart 4 — two simulated days: one with steady volatility, one with a
single large jump. Same realized variance, different character.**

*Reading:* realized variance cannot tell them apart. It adds up squared
returns and a jump contributes a large square just as sustained
volatility does. But they mean different things — one is a change in
the level, the other is a property of the process — and a forecast
should treat them differently, because jumps do not persist and
volatility does.

**Chart 5 — bipower variation on the same two days.**

*Reading:* bipower variation multiplies *adjacent absolute* returns
instead of squaring each one. A single large return is multiplied by
its small neighbours, so its influence is limited. The estimator is
therefore robust to jumps while remaining a valid measure of continuous
volatility.

Give the formula and note the constant that makes it comparable to
realized variance. The difference between the two estimators is an
estimate of the jump contribution, which is the basis for the test.

**Chart 6 — the Barndorff-Nielsen–Shephard jump test statistic across
many simulated days, half with jumps and half without.**

*Reading:* report the false positive rate on the no-jump days and the
detection rate on the jump days. Earlier work in this project recorded
roughly 4.2% and 100% respectively against a nominal 5% — **re-run this
rather than quoting it**, and report whatever comes out.

This is the chapter's verification, and it is worth being explicit
about the method: with no reference implementation to compare against,
correctness is established by checking that a test calibrated for 5%
false positives delivers about 5% on data with no jumps, and finds
jumps that are genuinely there. That is a statistical validation rather
than a numerical one.

### Beat 4 — Good and bad volatility (about 1.5 pages)

**Chart 7 — realized semivariance: the sum of squared returns split
into those from negative moves and those from positive ones.**

*Reading:* the two halves are not equal, and the downside half is
usually larger. This is chapter 26's asymmetry appearing again, now as
a direct measurement rather than a model parameter.

There is something satisfying about it. GJR and EGARCH inferred
asymmetry from daily data by fitting a parameter; semivariance simply
counts it. Where both are available they should broadly agree, and
checking that they do is a reasonable sanity test on the model.

### Beat 5 — Using it (about 1.5 pages)

**Chart 8 — a GARCH volatility forecast evaluated against realized
variance rather than squared daily returns.**

*Reading:* chapter 27 forecast a quantity nobody could observe, so
evaluating those forecasts meant comparing them against squared daily
returns — a noisy, unbiased but extremely imprecise proxy. Realized
variance is a far better target, and forecast evaluation improves
enormously as a result.

Report the difference in evaluation precision if it can be computed
cleanly. This is the practical payoff of the whole chapter and it is
worth landing.

Note the honest limitation: all of this needs intraday data, which is
expensive, frequently unavailable, and does not exist at all for most
macroeconomic series. The techniques in this chapter apply to a narrow
class of data.

### Beat 6 — Where this leaves you (half a page)

Part V is finished. Volatility can be modelled from daily data,
forecast forward, and — where intraday data exists — measured directly
and used to check the models.

Everything in Parts IV and V has treated the series as arriving on its
own, with no outside information. Part VI opens the machinery
underneath all of it, and Part VII brings the outside world in.

No recap. One sentence marking the end of a Part.

---

## 3. The `india` box

Placed in beat 5, after chart 8.

NSE provides intraday data, so realized measures are computable for
Indian equities — but the trading day is shorter than in the US and
there is a substantial overnight gap, particularly given how much
relevant news arrives from other time zones while the market is shut.

Realized variance computed over the trading day alone systematically
understates total daily variance, because it misses the overnight
move. The standard remedies — scaling up, or adding the squared
overnight return — are both approximations and neither is obviously
right. Anyone computing realized measures for Indian data should decide
this explicitly rather than inherit whatever the software defaults to.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Tsay ch. 5** | The intraday framing and the microstructure noise problem; this is Tsay's chapter on high-frequency data | Market microstructure theory |
| **Andersen, Bollerslev, Diebold & Labys via Tsay** | Realized variance itself; cite properly | The asymptotic theory |
| **Barndorff-Nielsen & Shephard (2004, 2006)** | Bipower variation and the jump test; cite both | The full development |
| **Shumway & Stoffer** | Little; high-frequency data is outside their scope | — |
| **Hamilton** | Nothing; the field postdates the book | — |
| **fpp3** | Nothing | — |

**On examples:** the bundle has no intraday financial data. **This
chapter must use simulated intraday paths**, and should say so
explicitly and repeatedly — a reader who thinks these are real prices
will draw wrong conclusions about magnitudes.

Check the bundle again before writing; if something suitable exists,
use it.

---

## 5. Voice

**Be honest about the verification standard.** This is the only chapter
in Part V with no reference implementation to check against, and the
simulation-based validation in beat 3 is what stands in for it. Say
that plainly — it is more credible than glossing over it.

**Beat 2's frequency finding is the most surprising thing here.** An
estimator that gets worse as you feed it more data contradicts
everything else in the book, and the reason is genuinely interesting.

**Do not oversell.** These techniques need data most readers do not
have. A chapter that reads as though realized measures supersede GARCH
would be misleading — they complement it, on a narrow class of series.

Avoid, beyond earlier lists:

- Market microstructure theory. Two sentences on why noise exists is
  enough.
- The full realized-measure zoo. Realized variance, bipower variation,
  semivariance and the jump test are plenty.
- Presenting simulated data as though it were real.

---

## 6. Checklist

- [ ] Eight charts through `@example ch28`
- [ ] Intraday and daily views contrasted at the outset
- [ ] Realized variance compared against a GARCH fit, correlation
      reported
- [ ] **Sampling frequency shown to matter**, with the microstructure
      explanation
- [ ] Bipower variation shown handling a jump that realized variance
      cannot
- [ ] **Jump test calibration re-run**, false positive and detection
      rates reported from that run
- [ ] The verification standard stated explicitly as
      simulation-based
- [ ] `arch.realized`'s absence stated as verified; the R half as
      reported
- [ ] Semivariance connected back to chapter 26's asymmetry
- [ ] Forecast evaluation against realized variance shown as the payoff
- [ ] Data limitation stated honestly
- [ ] All simulated data labelled as simulated
- [ ] `india` box on the overnight gap
- [ ] Ends by opening Part VI, no recap
- [ ] 10–11 pages, British-Indian spelling

---

## 7. What to do

1. **Re-run the jump test calibration.** The 4.2%/100% figures are from
   earlier work and have not been re-verified. Use at least 1,000
   simulated days per arm and report what comes out.
2. **Check the bundle for any intraday series** before committing to
   simulated data throughout.
3. Confirm which realized measures this package actually exports —
   `realized_variance`, `realized_semivariance`, `bipower_variation`,
   `jump_test` and `realized_measures` were all seen in the export
   list, and the chapter should use their real names.
4. Retry CRAN for `highfrequency`; if it installs, upgrade the R half
   of the claim.
5. Render every chart; honest CI note if the environment cannot.
6. Check against section 6.
