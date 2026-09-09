# Handoff: Chapter 20 — Seasonal ARIMA

`docs/src/introduction/20-seasonal-arima.md`. Target 11–12 pages.

The model most people mean when they say "ARIMA", and the one named
after the dataset the reader met on page seven.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. An honest limitation to disclose

This package's automatic seasonal differencing has a documented gap:
there is **no Canova-Hansen or OCSB test implemented**, so `D` must be
supplied explicitly rather than chosen automatically. R's `auto.arima`
selects it; this package does not.

That is a real difference from the obvious reference implementation and
it must be stated in the chapter rather than left for a user to
discover. It also gives beat 4 something honest to say about how `D` is
chosen in practice, which is mostly by looking.

---

## 2. The chapter, beat by beat

### Beat 1 — The model with a dataset named after it (about 1.5 pages)

**Chart 1 — the airline series, once more.**

Deliberate callback. The reader met this on page seven, differenced it
in chapter 3, logged it in chapter 7, decomposed it in chapters 14 and
15, and has never modelled it.

*Reading:* twelve years of monthly international airline passengers,
1949 to 1960. An upward trend, a strong annual pattern, and seasonal
swings that grow with the level — the last of which chapter 7 fixed
with a log. The model this chapter builds is called the airline model
because it was built for this series, and it has been the default
starting point for monthly data ever since.

Note the caution earned in chapter 1: this project's own copy of this
series was corrupted for a while. The version used here is from the
Census Bureau's `Testairline.spc`.

### Beat 2 — Two kinds of memory (about 2 pages)

**Chart 2 — the ACF of the twice-differenced log airline series, with
the seasonal lags highlighted.**

*Reading:* structure at lag 1 and structure at lag 12, and they are
doing different jobs. Lag 1 is this month against last month. Lag 12 is
this January against last January. A model needs both, and treating lag
12 as just another lag would require eleven intervening parameters
nobody wants to estimate.

**Chart 3 — the multiplicative structure, drawn.**

Show which lags the `(p,d,q)(P,D,Q)ₘ` form actually generates — the
non-seasonal lags, the seasonal lags, and the interaction lags at 11
and 13 that the multiplication produces for free.

*Reading:* the interaction terms are the elegant part and the part
nobody expects. Multiplying the polynomials generates terms at
`m ± 1` without anyone asking for them, and those terms are real —
December's behaviour relative to the previous January is not
independent of December's relative to November. Parsimony and structure
arrive together.

### Beat 3 — The airline model (about 2 pages)

**Chart 4 — the fitted ARIMA(0,1,1)(0,1,1)₁₂ on log airline data, with
its forecast.**

*Reading:* two moving-average parameters describe twelve years of
monthly data with trend and seasonality. Report the actual coefficients
and log-likelihood from the run.

Explain why this particular specification is so durable: one difference
handles trend, one seasonal difference handles the annual pattern, and
one MA term at each scale absorbs what remains. It is the smallest
model that can plausibly describe a trending seasonal series, and it is
right often enough to be the default first attempt.

**Chart 5 — the diagnostic panel for that fit.**

*Reading:* run chapter 12's panel on it. And here is the fact worth
reporting honestly, recorded from earlier work in this project — **the
airline model on the airline series passes the headline diagnostics
while simultaneously showing a flagged spectral peak and a Ljung-Box
failure at lags 3 and 4.** The most studied series in the field, fitted
with the model named after it, produces a split verdict.

**Verify this before writing it.** It has not been re-run this session.
If it reproduces, it is the best possible illustration of chapter 12's
point that diagnostics do not speak with one voice. If it does not,
drop it and say nothing.

### Beat 4 — Choosing D (about 2 pages)

**Chart 6 — the series with `D = 0` and `D = 1`, and the ACF of each.**

*Reading:* with `D = 0` the seasonal lags dominate the correlogram and
nothing else is visible. With `D = 1` they are gone. That contrast is
how `D` is chosen in practice — by looking, mostly, because the formal
tests are less standard and less trusted than the unit-root tests of
chapter 9.

Then the honest disclosure from section 1: R's `auto.arima` selects `D`
using a seasonal-strength test; this package requires it explicitly.
Say why that is a real limitation and not a design preference.

**Chart 7 — an over-seasonally-differenced fit.**

*Reading:* the same negative-spike signature as chapter 3, now at the
seasonal lag. `D = 2` is almost never right and the correlogram says so
plainly.

### Beat 5 — When one seasonal period is not the problem (about 2 pages)

**Chart 8 — a SARIMA fit to a series whose seasonal pattern has
changed shape over time.**

Use `aus_retail` or a constructed case.

*Reading:* SARIMA assumes a stable seasonal structure in the same way
chapter 14's classical decomposition did, just expressed through
parameters rather than indices. When the pattern evolves, the residuals
show it. STL handled this by letting the seasonal component drift;
SARIMA cannot, because its seasonal behaviour is fixed by coefficients
estimated over the whole sample.

This is worth stating because a reader who has just met STL might
reasonably assume SARIMA inherits its flexibility. It does not.

**Chart 9 — `vic_elec` or another multiple-seasonality series, with a
SARIMA fit.**

*Reading:* one seasonal period, three in the data, and the same failure
chapter 16 diagnosed. SARIMA is a single-period model.

### Beat 6 — Where this leaves you (half a page)

You can fit a seasonal model, and you have chosen `p`, `d`, `q`, `P`,
`D`, `Q` and `m` by hand — seven decisions, most of them by eye, on
every series.

That does not scale, and it is not reproducible between two analysts.

Chapter 21.

No recap.

---

## 3. The `india` box

Placed in beat 4, after chart 6.

A SARIMA model with `m = 12` assumes the seasonal effect recurs every
twelve months. For Indian monthly data the largest single seasonal
event does not — Diwali moves between October and November, so the
"annual" pattern is not annual at a fixed lag.

SARIMA cannot represent that at all. It will estimate a compromise
seasonal structure that is wrong in both months, and the residuals will
show a smear around lag 12 rather than a clean failure. The fix is not
a better seasonal order; it is a regressor built from the actual
festival dates, which is chapter 36.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Box & Jenkins via Montgomery** | The multiplicative seasonal construction and the airline model itself | The full methodology |
| **fpp3 §9.9** | The clearest explanation of what the seasonal terms actually do, and the practical guidance on choosing `D` | The `fable` syntax |
| **Shumway & Stoffer** | Worked SARIMA identification from correlograms | — |
| **Hamilton** | The polynomial-multiplication view that makes the interaction lags obvious | The algebra beyond that |
| **Ladiray & Quenneville** | That official seasonal adjustment uses a RegARIMA model of exactly this family as its preadjustment step — forward reference to chapter 39 | X-11 |
| **Cowpertwait & Metcalfe** | Patience with the notation, which is genuinely dense | R specifics |

**On examples:** the airline series (from `Testairline.spc`),
`aus_production`, `aus_retail`, `jj`, `vic_elec` all available.

---

## 5. Voice

**The notation is the chapter's main obstacle.** `(p,d,q)(P,D,Q)ₘ` is
seven symbols and readers glaze over. Chart 3 — drawing which lags the
model actually touches — does more than any amount of explanation.

**Report the split-verdict finding carefully or not at all.** If it
reproduces it is one of the book's best moments; if it is written from
memory it is exactly the failure the STL episode warned about.

Avoid, beyond earlier lists:

- Expanding the polynomial product algebraically. Chart 3 shows the
  result; the derivation belongs in the Manual if anywhere.
- Calling the airline model "simple". Two parameters is parsimonious;
  the structure they imply is not simple.

---

## 6. Checklist

- [ ] Nine charts through `@example ch20`
- [ ] Airline series reintroduced with the chapter 1 provenance caution
- [ ] Chart 3 draws which lags the multiplicative form generates,
      including the `m ± 1` interactions
- [ ] Airline model fitted with real coefficients reported
- [ ] **Split-verdict diagnostic finding verified before inclusion**,
      or omitted
- [ ] `D` selection shown by correlogram comparison
- [ ] The missing Canova-Hansen/OCSB test disclosed as a real limitation
- [ ] Over-seasonal-differencing shown
- [ ] SARIMA's fixed seasonality contrasted with STL's drift
- [ ] Multiple-seasonality failure shown
- [ ] `india` box on the moving festival defeating fixed `m`
- [ ] Ends by opening chapter 21, no recap
- [ ] 11–12 pages, British-Indian spelling

---

## 7. What to do

1. **Verify the airline split-verdict finding.** Fit
   ARIMA(0,1,1)(0,1,1)₁₂ to the log airline series, run the full
   diagnostic panel, and check whether Ljung-Box fails at lags 3–4 and
   whether a spectral peak is flagged. Report what actually happens.
2. Confirm the `D`-selection limitation is still accurate against
   current source before disclosing it.
3. Build chart 3 carefully — it is the chapter's main teaching device
   and a sloppy version wastes it.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
