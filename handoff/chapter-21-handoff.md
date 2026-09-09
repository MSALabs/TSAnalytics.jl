# Handoff: Chapter 21 — Choosing an Order Automatically

`docs/src/introduction/21-choosing-an-order.md`. Target 12–13 pages.

Chapter 20 left the reader making seven decisions by eye on every
series. This chapter automates that, and then spends most of its length
being honest about how well the automation actually works.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

### Order recovery, measured

An exhaustive search over `p, q ∈ {0,1,2,3}` on data generated from a
**known AR(2)**, 40 replications per cell, selecting by AIC and by
AICc:

```
true AR(2), n =  100:   AIC 40.0%   AICc 27.5%
true AR(2), n =  250:   AIC 40.0%   AICc 45.0%
true AR(2), n = 1000:   AIC 70.0%   AICc 67.5%
```

**At `n = 250` — a comfortable sample by most standards — automatic
selection finds the true order under half the time.** Even at `n = 1000`
it fails roughly three times in ten, on data generated from a model
inside the search space with no contamination, no seasonality and no
misspecification of any kind.

This confirms the substance of a claim recorded earlier in this project
as "about a third of the time", though the exact figure differs. **Use
your own run's numbers, not these**, and state the replication count —
at 40 replications these percentages carry roughly ±8 percentage points
of uncertainty, which is worth saying rather than presenting them as
precise.

### The criterion defaults differ

R's `auto.arima` defaults to **AICc**. Python's `pmdarima` defaults to
**AIC**. This package defaults to AICc, following R.

The recovery table shows this is not a neutral choice: at `n = 100`
AIC and AICc disagree by twelve percentage points, in AIC's favour on
this particular experiment. At `n = 250` the ordering reverses. Neither
dominates.

---

## 2. The chapter, beat by beat

### Beat 1 — Seven decisions, one series (about 1 page)

**Chart 1 — a grid of nine candidate fits to the same series, each
with its residual ACF.**

*Reading:* several of these look acceptable. Chapter 12 established
that a clean diagnostic panel means "not obviously wrong" rather than
"right", and here that limitation bites — the panel cannot rank the
acceptable candidates against each other. Something else is needed, and
it has to be a number, because nine panels is already too many to
compare by eye and a real search visits far more.

### Beat 2 — Trading fit against complexity (about 2.5 pages)

**Chart 2 — log-likelihood against number of parameters for a nested
sequence of models.**

*Reading:* it rises monotonically. It must — a larger model contains
the smaller one, so it can always match it and usually beats it
slightly. Maximum likelihood alone will therefore always choose the
largest model offered, which is useless as a selection rule.

**Chart 3 — the same models scored by AIC, AICc and BIC. Three lines.**

*Reading:* each criterion adds a penalty and each penalises
differently. BIC's penalty grows with sample size and so prefers
smaller models. AICc's correction matters when parameters are numerous
relative to observations. The three lines have minima in different
places, which is the first sign that "the best model" depends on who is
asking.

Give the AICc correction explicitly — `2k(k+1)/(n−k−1)` — and connect
to chapter 19: this is the term the `nobs` convention feeds into.

### Beat 3 — Searching without fitting everything (about 2 pages)

**Chart 4 — the Hyndman-Khandakar stepwise path across the (p,q)
grid, drawn as a route.**

*Reading:* start from a small set of candidates, fit them, move towards
whichever neighbour improves the criterion, repeat until nothing
improves. The route visits perhaps a dozen models where an exhaustive
search would fit over a hundred.

State the trade-off plainly: stepwise can stop at a local optimum. It
usually does not, and it is enormously faster, which is why it is the
default in R and here.

**Chart 5 — stepwise and exhaustive results compared across a set of
series.**

*Reading:* they usually agree. Report how often they did in your run.
Where they disagree, the exhaustive search finds a marginally better
criterion value, and whether that translates to a better forecast is a
separate question chapter 23 is equipped to answer and this one is not.

### Beat 4 — How often is it right? (about 3 pages)

The chapter's centre.

**Chart 6 — the recovery experiment, drawn as a bar chart across
sample sizes and criteria.**

*Reading, and take the space:* use the verified table. Data generated
from a known AR(2). The true model inside the search space. No
contamination, no seasonality, no misspecification. And at `n = 250`
the procedure identifies the true order under half the time.

**Chart 7 — what it selects when it is wrong.**

Tabulate the selected orders across replications.

*Reading:* the errors are not random. It mostly picks neighbours —
AR(1), AR(3), ARMA(1,1) — models that are close in structure and nearly
as good in likelihood. That is reassuring for forecasting and
discouraging for interpretation. If you want a forecast, a neighbouring
model usually forecasts similarly. **If you want to claim the data was
generated by an AR(2), automatic selection is not evidence for that.**

That distinction — selection for prediction versus selection for
inference — is the most useful thing in the chapter and most treatments
skip it.

**Then the `disagreement` box.** R defaults to AICc, Python to AIC, and
the recovery table shows the choice changes the answer: twelve
percentage points apart at `n = 100`, reversed at `n = 250`. Neither is
better. This package follows R, and that is a convention rather than a
finding.

### Beat 5 — With regressors, the differencing test moves (about 2 pages)

**Chart 8 — a trending regressor and a response that is stationary once
the regressor is accounted for.**

*Reading:* testing the raw response for a unit root gives the wrong
answer, because the response inherits the regressor's trend. **[V]** —
`pmdarima` handles this by regressing the response on the regressors
first and running the differencing tests on the *residuals*, which was
read from its source and confirmed by running it.

**Chart 9 — the same series, differencing order chosen on raw data
versus on residuals.**

*Reading:* the residual-based approach protects against the worst
failure. It does not solve the problem — verified directly, a
near-cointegrated case still selected `d = 1` where `d = 0` was correct.
Report that honestly; it connects to chapter 9's power problem and to
the cointegration work that Part IX lists as unbuilt.

### Beat 6 — Where this leaves you (half a page)

You can search a model space automatically, and you know the search
finds the true order well under half the time on data where the true
order exists.

That is fine if you want a forecast and a problem if you want a claim.
Which of those you want determines how much of this chapter's output to
believe — and chapter 22 finally produces the forecast.

No recap.

---

## 3. The `india` box

Placed in beat 4, after chart 6.

The recovery table is a direct statement about Indian quarterly
macroeconomic data. At `n = 50` — roughly what the current GDP base
provides — automatic order selection is operating well below the
`n = 100` row, and that row already shows under half the true orders
being found.

The practical consequence is that a published claim of the form "Indian
GDP follows an ARIMA(1,1,2)" is a statement about a selection procedure
rather than about the economy. Using the selected model to forecast is
reasonable. Interpreting its order is not.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Hyndman & Khandakar (2008)** | The stepwise algorithm; cite it properly | Implementation detail |
| **fpp3 §9.7** | The practical framing and the warning that automatic selection is a starting point | The `fable` syntax |
| **Hamilton ch. 4** | Why information criteria take the form they do, and the AIC/BIC consistency distinction | The derivations |
| **Shumway & Stoffer** | AICc's motivation for small samples | — |
| **Montgomery, Jennings & Kulahci** | Model selection inside an operational workflow where somebody has to ship something | — |
| **Tsay** | That in finance the selected order is rarely interpreted, only used — which is the healthy attitude beat 4 argues for | Volatility |

**On examples:** simulated data for the recovery experiment, labelled
as such. `GNP23`, `aus_production`, `global_economy` for the real fits.

---

## 5. Voice

**Beat 4 must not read as an attack on automatic selection.** It is
extremely useful and the alternative — seven decisions by eye — is
worse and less reproducible. The point is calibration: know what the
tool does well, which is finding a model that forecasts adequately, and
what it does badly, which is identifying truth.

**Report the recovery numbers with their uncertainty.** Forty
replications is not many. Saying "roughly 40%, ±8 points" is more
honest and more persuasive than a bare percentage.

Avoid, beyond earlier lists:

- Deriving AIC. State what it penalises and why.
- Presenting one criterion as correct.
- The word "optimal" for a selected model.

---

## 6. Checklist

- [ ] Nine charts through `@example ch21`
- [ ] Log-likelihood shown rising monotonically with parameters
- [ ] Three criteria compared, with minima in different places
- [ ] Stepwise path drawn as a route
- [ ] **Recovery experiment run fresh**, with replication count and
      uncertainty stated
- [ ] What it selects when wrong — the neighbour finding
- [ ] Prediction-versus-inference distinction made explicitly
- [ ] `disagreement` box on AIC/AICc defaults, using the recovery table
- [ ] Residual-based differencing with exogenous regressors shown, and
      its honest limit stated
- [ ] `india` box on short samples and interpretability
- [ ] Ends by opening chapter 22, no recap
- [ ] 12–13 pages, British-Indian spelling

---

## 7. What to do

1. **Re-run the recovery experiment with more replications** — 200 per
   cell if the time allows — and use those numbers. Forty was enough to
   establish the finding, not enough to put on a page as a headline.
2. Tabulate what gets selected when the true order is missed, for
   chart 7. That table does not exist yet.
3. Confirm this package's default criterion and whether stepwise is the
   default search.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
