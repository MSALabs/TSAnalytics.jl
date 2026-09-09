# Handoff: Chapter 23 — Evaluating Honestly

`docs/src/introduction/23-evaluating-honestly.md`. Target 11–12 pages.

The last chapter of Part IV. Chapter 22 measured a forecast against one
held-out period. This chapter asks whether that measurement can be
trusted, and mostly the answer is that one split is one draw.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Framing

This chapter has no external disagreement box, and manufacturing one
would be wrong. What it has instead is a **demonstration** that carries
the same weight — the variability of a single-split evaluation, measured
directly. That is the chapter's centrepiece and it should be run rather
than described.

The `julia` box has a natural home: `tscv` takes a *function*, which
makes comparing arbitrary methods a one-liner, and this package's
callback signature `(train, h) -> forecast` is worth showing.

---

## 2. The chapter, beat by beat

### Beat 1 — One split is one draw (about 2 pages)

**Chart 1 — the same two methods compared on five different train/test
splits of the same series, with the winner marked on each.**

*Reading, and this is the chapter's argument in one picture:* the
winner changes. Method A wins on three splits and loses on two, or some
similar mix. Nothing about either method changed; only where the line
was drawn changed.

A single held-out period is a sample of size one from the distribution
of possible evaluations, and chapter 22 quietly treated it as though it
were the truth. Everyone does. It is the most common evaluation error
in applied forecasting and it is entirely invisible when you only ever
run one split.

**Chart 2 — the distribution of the RMSE difference between the two
methods across many splits.**

*Reading:* a spread that probably straddles zero. If the distribution
includes zero comfortably, the honest conclusion is that the two
methods are not distinguishable on this series with this much data —
which is a real finding and a more useful one than a spurious winner.

### Beat 2 — Splitting a series is not splitting a dataset (about 1.5 pages)

**Chart 3 — random k-fold splitting applied to a time series, drawn
with the folds shaded.**

*Reading:* the ordinary machine-learning approach shuffles observations
into folds. On a time series that means training on observations from
after the test period — using the future to predict the past. The
resulting error estimate is optimistic and sometimes wildly so, because
neighbouring observations are correlated and a model can effectively
memorise a test point from its neighbours.

State plainly that this is a common and serious error, and that it does
not announce itself — the numbers just come out better.

### Beat 3 — Rolling forward (about 2.5 pages)

**Chart 4 — the expanding-window scheme, drawn: successive training
sets growing, each followed by a forecast.**

*Reading:* fit on everything up to time `t`, forecast, move forward,
repeat. Each forecast uses only information that existed when it was
made, which is the only honest arrangement. The result is many forecast
errors instead of one, and their distribution is what chart 2 wanted.

**Chart 5 — expanding window against rolling window.**

*Reading:* the expanding window keeps all history; the rolling window
keeps a fixed length and discards the oldest. Expanding uses more data.
Rolling adapts if the process changes. Which is right depends on
whether you believe the past remains relevant — an empirical question,
and one this scheme can actually answer by trying both.

**`julia` box here.** `tscv` takes a function with signature
`(train, h) -> forecast`, so comparing four methods is four lines and
no special-casing. Show it. Half a page.

### Beat 4 — Horizons behave differently (about 2 pages)

**Chart 6 — RMSE against forecast horizon for three methods.**

*Reading:* the lines are not parallel and they cross. A method that
wins at `h = 1` can lose at `h = 12`, because short-horizon accuracy is
mostly about capturing short-run dynamics and long-horizon accuracy is
mostly about getting the trend and seasonality right.

**"Which method is better" is therefore not a well-posed question
without a horizon attached**, and a great many published comparisons
omit it.

**Chart 7 — the same comparison at a single horizon, showing how the
ranking would look if only `h = 1` had been reported.**

*Reading:* a clean, confident, incomplete answer. This is what most
evaluations report.

### Beat 5 — The comparison that matters (about 2 pages)

**Chart 8 — four methods compared under proper cross-validation: the
four benchmarks from chapter 22 and one fitted model.**

*Reading:* run it on a real bundled series and report what actually
happens, whatever it is. If the fitted model wins comfortably, say so.
If it loses to seasonal naive, say that — chapter 22 already
established this is common and not shameful.

The point is the method of comparison, not the outcome. **Do not
select a series where the model wins.** If the first honest attempt
shows the benchmark winning, that is the more instructive chapter.

**Chart 9 — a residual diagnostic panel for the winning method.**

*Reading:* close the loop. Cross-validation says which method forecasts
better; chapter 12's panel says whether the winner is well specified.
They can disagree — a misspecified model can forecast adequately — and
knowing which question you asked is the whole discipline of this
chapter.

### Beat 6 — Where this leaves you (half a page)

Part IV is finished. You can identify a model, fit it, choose its order,
forecast with honest intervals, and evaluate it in a way that does not
flatter it.

Every model in this Part has assumed constant variance. Chapter 11
showed a series where that was plainly false and nothing since has
addressed it.

Part V does.

No recap. One sentence marking the end of a Part.

---

## 3. The `india` box

Placed in beat 3, after chart 5.

The expanding-versus-rolling choice is not academic for Indian
macroeconomic series. Structural breaks are frequent and datable — the
1991 liberalisation, the 2016 demonetisation, the 2017 GST transition —
and an expanding window trains on a regime that may no longer apply.

A rolling window handles this by forgetting, which is crude but honest.
The alternative is to model the break explicitly, which is chapter 41's
territory and not built. In the meantime, comparing both windows on the
same series is cheap and tells you whether the older data is helping or
hurting.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **fpp3 §5.10** | Time series cross-validation as standard practice rather than an advanced technique, and the rolling-origin framing | The `fable` syntax |
| **Hyndman & Athanasopoulos more broadly** | The insistence that evaluation is where forecasting is won or lost | — |
| **Montgomery, Jennings & Kulahci** | The operational view — a forecast is evaluated by what it costs when wrong, and different horizons have different costs | Control charts |
| **Tsay** | Backtesting as it is practised in finance, where the discipline is strongest because money is at stake | Trading strategy detail |
| **Hamilton** | Little; out-of-sample evaluation is not his focus, which is worth noticing about the econometric tradition | — |
| **Shumway & Stoffer** | Prediction error as a model-comparison tool | — |

**On examples:** `aus_production`, `aus_retail`, `GNP23`, `vic_elec`
bundled. The benchmark comparison needs the chapter 22 methods.

---

## 5. Voice

**Beat 1 must be run, not asserted.** The claim that a single split can
be misleading is easy to state and unconvincing until the reader sees
the winner change. Generate the splits, count the flips, report the
number.

**Beat 5's honesty matters more than its outcome.** The instruction not
to select a favourable series is genuine — if the fitted model loses,
that is the better chapter, and it is consistent with everything
chapters 21 and 22 established.

Avoid, beyond earlier lists:

- Calling cross-validation "the gold standard". It is more honest than
  one split, and it is still an estimate.
- Presenting the expanding/rolling choice as having a right answer.
- Any sentence containing "best practice".

---

## 6. Checklist

- [ ] Nine charts through `@example ch23`
- [ ] **The winner-changes demonstration run**, with the flip count
      reported
- [ ] Distribution of the RMSE difference shown, and its overlap with
      zero interpreted honestly
- [ ] Random k-fold shown as the error it is, with why it flatters
- [ ] Expanding and rolling windows both drawn and compared
- [ ] `julia` box on `tscv`'s function argument, half a page
- [ ] RMSE-against-horizon lines shown crossing
- [ ] Benchmark comparison run on a series **not** selected for a
      favourable outcome
- [ ] Cross-validation and diagnostic panel shown answering different
      questions
- [ ] `india` box on structural breaks and window choice
- [ ] Ends by opening Part V, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 7. What to do

1. Run beat 1's experiment first. If the winner does not change on the
   series you chose, try others — but report the flip rate honestly
   rather than hunting for a dramatic one.
2. Confirm `tscv`'s exact callback signature and whether it accepts a
   `Forecast`-returning function directly. The benchmark-methods
   handoff proposed teaching it to; check whether that landed.
3. Run beat 5's comparison once and use the result, whatever it is.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
