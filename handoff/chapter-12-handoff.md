# Handoff: Chapter 12 — The Diagnostic Panel

`docs/src/introduction/12-the-diagnostic-panel.md`. Target 10–11 pages.

The last chapter of Part II, and the shortest. Chapters 8 to 11 built a
collection of tests. This one assembles them into the single picture
practitioners actually use, and — more usefully — teaches the reader to
read a verdict that is not unanimous.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

### The four-panel display is a genuine convention, not one package's habit

Confirmed across three independent sources:

- R base's `tsdiag()`
- `astsa::sarima()` — read from its actual R source
- Python's `SARIMAXResults.plot_diagnostics()`

All produce the same four panels: standardised residuals over time, the
ACF of residuals, a normal Q-Q plot, and **Ljung-Box p-values plotted
across a range of lags** rather than reported as a single number.

That fourth panel is the interesting one and the one most people have
never thought about. A single Ljung-Box p-value is a choice of lag
dressed up as a result.

### The lag count is not a fixed default **[S]**

Read from `astsa::sarima`'s R source:

```r
nlag = ifelse(S < 7, 20, 3*S)
nlag = min(nlag, 52)
ppq  = p + q + P + Q - (fixed params) + abs(fitdf)
if (nlag < ppq + 8) nlag = ppq + 8
```

So the number of lags shown scales with the seasonal period — 20 for
non-seasonal or short-period data, `3m` for longer, capped at 52 — and
is then padded to sit at least eight beyond the number of estimated
parameters. Nobody's default is `lags = 10`, and a package that uses one
is making a decision it has not thought about.

### The package's own implementation

`diagnostic_plot` exists with three call forms: on raw residuals, on an
`ArmaModel`, and on a `SarimaModel`. The model forms presumably infer
`ppq` and `fitdf` rather than requiring them — **confirm this before
writing**, since beat 3 depends on it.

---

## 2. The chapter, beat by beat

### Beat 1 — Four tests, four separate looks (about 1 page)

**Chart 1 — the four diagnostics from chapters 10 and 11 as four
separate figures, presented awkwardly.**

*Reading:* running each test in turn works, and nobody does it. In
practice a model gets fitted, someone glances at something, and the
model ships. The remedy is not discipline; it is making the complete
check cheap enough that skipping it takes more effort than doing it.

### Beat 2 — One picture (about 2 pages)

**Chart 2 — the full four-panel diagnostic display on a well-specified
model.**

*Reading, panel by panel:* the standardised residuals should look like
noise around zero with no drift and no changing spread — this panel
catches the variance problems of chapter 11 by eye. The ACF should sit
inside its band. The Q-Q plot should follow the line, with tail
departures being the common and usually tolerable failure. And the
Ljung-Box panel should sit above the 0.05 line at every lag shown.

Note that panel four contains chapters 10 and 11's material
simultaneously, and panel one silently duplicates part of chapter 11's
work. The panel is not four unrelated tests; it is a designed set with
deliberate overlap.

### Beat 3 — How many lags, and why it matters (about 2 pages)

**Chart 3 — the same model's Ljung-Box panel drawn with 10, 20 and 40
lags. Three panels.**

*Reading:* the p-values are not the same, and on a borderline model the
verdict can change with the lag count. That is why the choice cannot be
left to a default nobody examined, and why `astsa`'s formula scales with
the seasonal period — testing a monthly model out to only ten lags never
looks at the seasonal lag at all.

Give the formula from section 1 and walk through it for a monthly
seasonal model: `S = 12`, so `nlag = 36`, capped at 52, padded past the
parameter count. Then state what this package does.

**Chart 4 — the effect of `fitdf` on the same panel.**

*Reading:* chapter 10 introduced this and the panel is where it bites
visibly. Without the correction the p-value curve sits too high across
the board and the model looks better than it is.

### Beat 4 — Reading a split verdict (about 3 pages)

The chapter's real subject. Three cases, each a chart.

**Chart 5 — a model that fails only panel two.**

*Reading:* structure left in the ACF at an identifiable lag. This is
the easy case — the lag tells you what to add, and the model is
under-specified in a specific, fixable way.

**Chart 6 — a model that passes panels two and four and fails panel
one.**

*Reading:* correlation is gone, variance is not stable. This is
chapter 11's trapdoor appearing in the panel, and the remedy is not
another AR term — no amount of mean modelling fixes a variance problem.
This is the panel telling you to go to Part V.

**Chart 7 — a model that passes everything except the Q-Q plot's
tails.**

*Reading, and this is the most practically useful case:* heavy tails
with everything else clean. The honest answer is that the point
forecasts are fine and the prediction intervals are too narrow. Whether
that matters depends entirely on what the forecast is for. A reader
who treats every failed panel as fatal will discard usable models;
one who treats every failure as ignorable will ship bad intervals.

**Then the case worth remembering.** The airline series, fitted with
the airline model, passes the headline diagnostics and simultaneously
shows a flagged spectral peak and a Ljung-Box failure at lags 3 and 4 —
a real, documented disagreement between diagnostics on the most studied
series in the field. If that example can be reproduced here it is the
chapter's best illustration that "the diagnostics" do not speak with
one voice. **Verify it before using it**; it is recorded from earlier
work in this project and has not been re-run this session.

### Beat 5 — What the panel cannot tell you (about 1.5 pages)

**Chart 8 — two models on the same series, one with three parameters
and one with fifteen, both with clean panels.**

*Reading:* chapter 10 made this point about the ACF and the panel
inherits it. Diagnostics test whether a model has captured enough. They
are entirely silent on whether it used too much. Both panels here are
clean and one model is badly over-fitted.

That is what information criteria are for, and chapter 21 handles them.
Until then, a clean diagnostic panel means *not obviously wrong*, which
is a genuinely useful thing to know and a much weaker claim than most
people read into it.

### Beat 6 — Where this leaves you (half a page)

Part II is finished. The reader can test a series before modelling it
and test a model after fitting it, in one picture.

What they cannot do is fit anything. Every model in this Part has been
assumed into existence.

Part III separates a series into pieces you can see. Part IV builds the
first models that are genuinely fitted.

No recap. One sentence marking the end of a Part is enough.

---

## 3. The `india` box

Placed in beat 4, after chart 6.

For Indian monthly data the fourth panel needs enough lags to reach 12,
and preferably 24 — the seasonal lag is where a model of Indian
industrial or retail data most often fails, because the festival
calendar moves and a fixed twelve-month structure cannot absorb it.

A panel drawn with a default of ten lags will show a clean bill of
health on a model with obvious residual seasonality, simply because it
never looked far enough. On this data the lag count is not a cosmetic
choice.

One paragraph, forward-referencing chapter 36.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Shumway & Stoffer / `astsa`** | The panel itself, and the lag-count formula, which is theirs and is better reasoned than any alternative | — |
| **fpp3 §5.4** | The insistence that diagnostics are mandatory, and the practice of showing the residual plot beside the tests | The `fable` syntax |
| **Box & Jenkins via Montgomery** | Diagnostic checking as a named stage of a defined workflow — identify, estimate, check, repeat | The full workflow diagram |
| **Tsay** | That panel one catches volatility clustering, which the other three panels do not | GARCH |
| **Hamilton** | Little; he does not treat diagnostics graphically | — |
| **Cowpertwait & Metcalfe** | Reading a panel patiently, one component at a time | R specifics |

**On examples:** any fitted model on a bundled series. The airline
model on the airline series for beat 4's split verdict.

---

## 5. Voice

**This chapter is about judgement, not procedure.** Its value is
entirely in beat 4 — teaching a reader what to do when three panels
pass and one fails. A version that describes the four panels and stops
has wasted the chapter.

**Do not present the panel as a pass/fail gate.** It is a set of
readings, and the useful skill is interpreting a mixed result. Say so
explicitly, because the visual format invites a binary reading.

Avoid, beyond earlier lists:

- Repeating chapters 10 and 11's explanations. Assume them; this
  chapter assembles rather than re-teaches.
- Any suggestion that a clean panel validates a model.

---

## 6. Checklist

- [ ] Eight charts through `@example ch12`
- [ ] All four panels explained once, briefly, without re-teaching
      chapters 10–11
- [ ] Lag-count formula from `astsa` given and applied to a monthly
      example; this package's behaviour stated
- [ ] `fitdf` effect shown on the panel
- [ ] **Three split-verdict cases**, each with a different remedy — this
      is the chapter's core
- [ ] The over-fitting blind spot restated for the panel
- [ ] Airline split-verdict example verified before inclusion, or
      omitted
- [ ] `india` box on lag count reaching the seasonal lag
- [ ] Ends by opening Part III, no recap
- [ ] 10–11 pages, British-Indian spelling

---

## 7. What to do

1. **Confirm `diagnostic_plot`'s three call forms** and whether the
   model forms infer `ppq` and `fitdf` automatically. Beat 3 depends on
   it.
2. Check what lag count this package actually uses by default and
   whether it follows the `astsa` formula. If it does not, say so
   rather than describing the formula as though it were implemented.
3. **Verify the airline split-verdict example before writing it.** It
   is recorded from earlier work and has not been re-run. If it does
   not reproduce, drop it — the chapter has three constructed cases
   already.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
