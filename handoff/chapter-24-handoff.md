# Handoff: Chapter 24 — Why Variance Changes

`docs/src/introduction/24-why-variance-changes.md`. Target 10–11 pages.

Part V opens. Every model in Part IV assumed the variance was a fixed
number. Chapter 11 showed a series where that was plainly false and
nothing since has done anything about it. This chapter picks up that
thread.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Framing

This is a motivation chapter and it should not pretend otherwise. Its
job is to make the reader want a variance model, using data rather than
assertion, and to hand over to chapter 25 with the problem clearly
stated.

It carries an `india` box and a `julia` box. **No disagreement box** —
the material here is descriptive and uncontested, and manufacturing a
controversy would be worse than not having one.

---

## 2. The chapter, beat by beat

### Beat 1 — The residuals that passed (about 1.5 pages)

**Chart 1 — chapter 11's residuals again, with their Ljung-Box p-value
printed on the plot.**

Deliberate continuity. Same data, same comfortable p-value, same
obvious structure.

*Reading:* thirteen chapters of tooling have been applied to this
series and every one of them says it is fine. The correlogram is clean,
the portmanteau test passes, the model is not obviously wrong by any
standard from Part II except the one panel that looked at the residuals
directly.

And the plot shows quiet stretches followed by violent ones. Something
is predictable here and nothing built so far can predict it.

### Beat 2 — Prices and returns (about 2 pages)

**Chart 2 — `nyse` as a price series and as returns. Two panels.**

*Reading:* chapter 3 established that prices are I(1) and returns are
I(0) — differencing logs *is* the return transformation. So the return
series is the stationary object, and by Part IV's standards it should
now be tractable.

The return panel has no visible trend, no seasonality, and a mean
indistinguishable from zero. Its ACF is essentially empty. By every
criterion from Parts I to IV this series has no structure at all.

**Chart 3 — the ACF of returns beside the ACF of squared returns.**

*Reading, and this is the beat that carries the chapter:* the first
panel is empty. The second is not — squared returns are strongly
autocorrelated, often out to twenty or thirty lags.

The *sign* of tomorrow's return is unpredictable, which is what an
efficient market implies and what the empty first panel shows. The
*size* is entirely predictable. Those are different questions and only
one of them has been asked so far.

### Beat 3 — What clustering looks like (about 2 pages)

**Chart 4 — rolling standard deviation of returns over a long window.**

*Reading:* it moves, and it moves persistently. Calm periods last
months; turbulent periods last months. This is not noise in the
estimate — the window is wide enough that sampling variation would be
small — it is the thing itself changing.

**Chart 5 — the return distribution against a fitted normal, and the
matching Q-Q plot. Two panels.**

*Reading:* too much mass in the centre and too much in the tails.
Financial returns are famously not normal, and chapter 11 explained why
that matters — prediction intervals computed from normal quantiles will
be too narrow and their coverage worse than advertised.

Worth noting that a mixture of normals with different variances is
itself heavy-tailed. If the variance changes over time and you pool
everything together, you get fat tails automatically. **The
non-normality and the clustering may be the same phenomenon seen two
ways**, which is a genuinely useful thing to realise before chapter 25
starts modelling.

### Beat 4 — Not every variance problem is clustering (about 2 pages)

**Chart 6 — three series with three different variance behaviours.**

A series with clustering; a series with a one-off level shift in
variance; a series whose variance grows steadily with the level.

*Reading:* three problems, three remedies. The third was solved in
chapter 7 with a transformation and needs nothing further. The second
is a structural break and wants an intervention term, not a variance
model. Only the first is what Part V is about.

Chapter 11 gave two tests — ARCH-LM for clustering, the variance-ratio
test for shifts — and this chart is where knowing which one fired
becomes actionable.

**Chart 7 — ARCH-LM applied to all three.**

*Reading:* report the actual statistics. The clustering series should
reject decisively; the others may or may not, and whatever happens
should be reported rather than tidied.

### Beat 5 — Why anyone cares (about 1.5 pages)

**Chart 8 — the same forecast with constant-variance intervals and with
time-varying intervals, over a period that includes a turbulent stretch.**

*Reading:* the constant-variance intervals are too wide in the calm
period and too narrow in the turbulent one. They are wrong in both
directions and right only on average, which is the least useful place
for an interval to be right.

State the stakes concretely. A risk limit set from a constant-variance
model is too loose exactly when markets are turbulent — which is
exactly when it matters. That is not an academic concern; it is why
this entire literature exists and why regulators require these models.

### Beat 6 — Where this leaves you (half a page)

The mean is unpredictable and the variance is not. Nothing in the book
so far models a variance that moves.

Chapter 25 does, and the idea is simpler than the twenty years of
literature it launched would suggest.

No recap.

---

## 3. The `india` box

Placed in beat 4, after chart 6.

Indian equity and currency series show both patterns, and
distinguishing them matters. There is genuine clustering of the kind
chapter 25 models. There are also datable variance shifts at policy
moments — the 1991 liberalisation, the 2016 demonetisation
announcement, the introduction of currency futures — where the
variability changed level and stayed changed.

Fitting a clustering model to a structural break produces a model that
persistently over-predicts volatility in the calm regime and
under-predicts in the turbulent one. The tests in chapter 11
distinguish them; the temptation is to skip the tests and reach
straight for GARCH.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Tsay chs. 1, 3** | The entire framing. Returns, their stylised facts, and the squared-return ACF as the diagnostic that opens the subject. This is Tsay's home ground and the chapter should lean on it | The distributional theory |
| **Engle (1982) via Hamilton** | Why this was a genuinely surprising idea in 1982 — that variance could be modelled at all | The derivation |
| **fpp3** | Almost nothing; fpp3 barely treats volatility, which is worth noticing as a real difference between the forecasting and financial-econometrics traditions | — |
| **Shumway & Stoffer ch. 5** | GARCH introduced alongside `nyse`, and their treatment of returns | — |
| **Montgomery, Jennings & Kulahci** | Variance shifts as an industrial phenomenon — process variability changes when equipment ages, and the remedy differs from the financial case | Control charts |
| **Hamilton ch. 21** | The formal statement of conditional heteroskedasticity | The asymptotics |

**On examples:** `nyse` (2,000), `sp500w`, `djia`, `gafa_stock` all
bundled and all suitable.

---

## 5. Voice

**Chart 3 is the chapter and deserves room.** Two panels — one empty,
one full — and the entire justification for Part V. Do not rush it.

**The fat-tails-and-clustering connection in beat 3 is worth making
explicitly.** Most treatments present them as two separate stylised
facts. That they may be one fact seen twice is more interesting and
more useful.

**Do not introduce GARCH.** The temptation is strong and the whole
chapter is a setup for chapter 25. Naming the model here would waste
the setup.

Avoid, beyond earlier lists:

- Efficient-market-hypothesis discussion. One sentence about why the
  mean is unpredictable is enough; the debate is not this book's.
- Calling volatility "risk". They are related and not identical, and
  the conflation causes real confusion.

---

## 6. Checklist

- [ ] Eight charts through `@example ch24`
- [ ] Opens on chapter 11's residuals, same data, deliberate continuity
- [ ] Returns shown to have an empty ACF and a full squared-return ACF
- [ ] Rolling standard deviation shown moving persistently
- [ ] Fat tails connected to time-varying variance as possibly one
      phenomenon
- [ ] Three variance problems distinguished, with different remedies
- [ ] ARCH-LM run on all three, results reported as they come
- [ ] The stakes made concrete — intervals wrong in both directions
- [ ] `india` box distinguishing clustering from policy breaks
- [ ] GARCH not named
- [ ] Ends by opening chapter 25, no recap
- [ ] 10–11 pages, British-Indian spelling

---

## 7. What to do

1. Choose the beat 1 residual series so chart 3's contrast is
   unmistakable. `nyse` returns will work; try `gafa_stock` and
   `sp500w` too and use the clearest.
2. Run ARCH-LM on all three beat 4 series and report the real numbers,
   including any that do not behave as expected.
3. Render every chart; honest CI note if the environment cannot.
4. Check against section 6.
