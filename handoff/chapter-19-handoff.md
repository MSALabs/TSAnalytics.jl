# Handoff: Chapter 19 — ARIMA

`docs/src/introduction/19-arima.md`. Target 10–11 pages.

Chapter 18 fitted models to stationary series. Chapter 3 showed how to
make a series stationary. This chapter joins them, and the join turns
out to be less mechanical than it looks.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

The `nobs` finding was verified for chapter 3 and pays off here. On a
100-point random walk, ARIMA(1,1,0):

```
Python SARIMAX:  nobs = 100   loglik = -146.871833   aic = 297.743665
R arima:         nobs =  99   loglik = -146.8718     aic = 297.7437
series length =  100
```

`nobs` differs; loglik and AIC are identical. The consequence lands in
AICc, whose correction term carries `n`:

| n | correction, full n | correction, n−d | difference |
|---|---|---|---|
| 100 | 0.1237 | 0.1250 | 0.0013 |
| 30 | 0.4444 | 0.4615 | 0.0171 |
| 20 | 0.7059 | 0.7500 | 0.0441 |

Chapter 3 introduced this as a curiosity about counting. Here it
becomes a live problem, because AICc is what chapter 21 uses to choose
between models.

---

## 2. The chapter, beat by beat

### Beat 1 — The obvious composition (about 1 page)

**Chart 1 — a trending series, its first difference, and an ARMA fit to
the difference. Three panels.**

*Reading:* difference until stationary, fit ARMA, done. That is
genuinely what ARIMA is, and stating it plainly early is better than
building suspense. The rest of the chapter is about the details that
make this less clean than it sounds — what happens to forecasts, what
happens to the constant, and what happens to the bookkeeping.

### Beat 2 — Getting back to the level (about 2.5 pages)

**Chart 2 — a forecast made on the differenced scale, and the same
forecast integrated back to levels. Two panels.**

*Reading:* the differenced forecast converges to a constant, which
looks unremarkable. Integrated back, that constant becomes a straight
line with slope — the forecast of *changes* becoming a forecast of a
*trajectory*. The two panels show the same object and imply completely
different things about the future.

**Chart 3 — prediction intervals on both scales.**

*Reading, and this is the beat's payload:* on the differenced scale the
intervals are roughly constant width. Integrated back they fan out,
widening without bound. That is not a modelling artefact — it is the
honest consequence of a unit root. If shocks never die out, uncertainty
about the level accumulates forever, and any model that produced
constant-width level intervals for a differenced series would be lying.

Connect back to chapter 8: this fanning is the practical face of
non-stationarity, and it is why long-horizon forecasts of a random-walk
series are nearly useless however good the model is.

### Beat 3 — The constant that changes meaning (about 2 pages)

**Chart 4 — three forecasts: `d=0` with a mean, `d=1` with no
constant, `d=1` with a constant. Three panels.**

*Reading:* with `d = 0` the constant is the series mean and the
forecast reverts to it. With `d = 1` and no constant the forecast is
flat at the last level. With `d = 1` *and* a constant, the constant is
now the mean of the *differences*, so it becomes a slope and the
forecast trends indefinitely.

The same argument in the same slot means three different things
depending on `d`. This trips up a great many people, and the three
panels make it unmistakable in a way that a paragraph does not.

Note the software convention honestly: most packages suppress the
constant when `d ≥ 1` by default, precisely because an unintended
deterministic trend extrapolated over a long horizon is a common and
embarrassing failure. State what this package does.

### Beat 4 — Counting what you have (about 2 pages)

**Chart 5 — the same series fitted at `d = 0`, `1` and `2`, with the
number of usable observations marked on each.**

*Reading:* each difference costs an observation off the front. On 84
quarterly observations, `d = 2` leaves 82 — negligible. On a 30-point
series it is a tenth of the data, and chapter 3's variance rule of
thumb is worth re-running before spending it.

**Then the `disagreement` box**, using section 1's material.

Frame it as bookkeeping with consequences. Two implementations count
the same fitted model's observations differently — 99 against 100. The
log-likelihood and AIC come out identical, so nobody notices. AICc
does not, because it carries `n` explicitly, and AICc is the default
criterion for automatic model selection in chapter 21.

The numbers say the rest: 0.0013 at `n = 100`, 0.0441 at `n = 20`. On a
short series two implementations can rank two candidate models
differently for no reason other than a counting convention.

### Beat 5 — Fitting one properly (about 2 pages)

**Chart 6 — a full ARIMA fit on a real bundled series, with the
diagnostic panel from chapter 12.**

Use `GNP23` or a `global_economy` series.

*Reading:* the workflow is now complete — test for a unit root
(chapter 9), difference (chapter 3), fit (chapter 18), check
(chapter 12). Walk it once, end to end, without introducing anything
new. This is the first time in the book that all the pieces have been
used together, and the reader should see that the parts they learned
separately compose.

**Chart 7 — the same series with one difference too many, and its
diagnostic panel.**

*Reading:* the over-differenced fit shows chapter 3's negative lag-1
signature in the residual ACF, and the diagnostic panel flags it. The
tools catch the mistake, which is the point of having them.

### Beat 6 — Where this leaves you (half a page)

ARIMA handles trend. It does nothing about seasonality — and every
seasonal series in Part III would defeat it.

Chapter 20.

No recap.

---

## 3. The `india` box

Placed in beat 4, after chart 5.

Indian quarterly GDP in its current base runs to a few dozen
observations. At `n = 50`, `d = 1` and a seasonal difference as well,
the usable sample drops sharply — and the AICc gap in the table above
is at its widest exactly there.

This compounds chapter 9's `india` box. Short series make unit-root
tests uninformative, and the same shortness makes the model-selection
criterion sensitive to a counting convention. Neither problem is
fixable by better software.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Box & Jenkins via Montgomery** | The integration idea and the workflow; ARIMA is their construction | The full methodology apparatus |
| **fpp3 §9.5, §9.8** | The constant-term explanation, which fpp3 handles better than anyone, and the fanning intervals | The `fable` syntax |
| **Hamilton ch. 15** | Why the intervals must widen without bound under a unit root | The asymptotics |
| **Shumway & Stoffer** | Worked ARIMA fits on real series, end to end | — |
| **Tsay** | That log prices are I(1) and log returns are I(0), so financial ARIMA is usually ARMA on returns | Volatility |
| **Cowpertwait & Metcalfe** | Patience with the integration bookkeeping | R specifics |

**On examples:** `GNP23`, `global_economy`, `varve`, `gafa_stock`
bundled.

---

## 5. Voice

**Beat 3 is worth the space.** The changing meaning of the constant is
one of the most common real confusions in applied ARIMA work, and three
panels settle it permanently.

**Do not oversell the fanning intervals as a flaw.** They are correct.
A reader who finds them uncomfortable has understood something true
about forecasting a random walk.

Avoid, beyond earlier lists:

- Backshift-operator algebra for its own sake. Chapter 3 introduced it;
  use it, do not re-teach it.
- Presenting ARIMA as a single unified model. It is ARMA plus
  bookkeeping, and saying so makes it easier, not less impressive.

---

## 6. Checklist

- [ ] Seven charts through `@example ch19`
- [ ] Forecast shown on both differenced and level scales
- [ ] Fanning intervals shown and explained as correct, not a defect
- [ ] The constant's three meanings shown in three panels
- [ ] The package's constant-suppression default stated
- [ ] `disagreement` box with the verified `nobs` numbers and the AICc
      correction table
- [ ] Full end-to-end workflow walked once, using chapters 9, 3, 18, 12
- [ ] Over-differenced case shown being caught by the diagnostics
- [ ] `india` box compounding chapter 9's
- [ ] Ends by opening chapter 20, no recap
- [ ] 10–11 pages, British-Indian spelling

---

## 7. What to do

1. Confirm this package's behaviour on the constant when `d ≥ 1` — is
   it suppressed by default, and is there an override? Beat 3 needs the
   real answer.
2. Confirm which `nobs` convention this package reports. If it differs
   from both R and Python, that is worth saying.
3. Re-run the AICc correction table live rather than copying it.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
