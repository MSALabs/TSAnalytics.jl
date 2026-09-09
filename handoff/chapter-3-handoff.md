# Handoff: Chapter 3 — Differencing and Integration

`docs/src/introduction/03-differencing-and-integration.md`. Target 11–13 pages.

Chapter 2 settled what a time series is. This chapter deals with the
first thing you almost always have to do to one before anything else
works: remove the part that will not sit still.

Same requirements as chapters 1 and 2. The reading template from
chapter 1's handoff applies unchanged.

---

## 1. A [U] claim, re-verified

`all-chapters-handoff.md` tagged this chapter's disagreement box **[U]**
— carried forward across sessions, never re-tested — and said to
re-verify before writing. I have done that, and it holds. Closing the
loop explicitly, because the STL episode showed what happens when a
[U] claim goes into a chapter unchecked.

**The claim:** R reports `nobs` after differencing as `n − d`; Python
reports the full `n`.

**Verified**, on a 100-point random walk, ARIMA(1,1,0):

```
Python SARIMAX(1,1,0):  nobs = 100   loglik = -146.871833   aic = 297.743665
R arima(1,1,0):         nobs =  99   loglik = -146.8718     aic = 297.7437
series length        =  100
```

**But the finding is more interesting than the claim was.** The `nobs`
values differ, and the log-likelihood and AIC are *identical to every
digit shown*. So the disagreement does not propagate where you would
expect.

Where it does propagate is **AICc**, which carries `n` explicitly in
its correction term `2k(k+1)/(n−k−1)`:

| n | correction, full n | correction, n−d | difference |
|---|---|---|---|
| 100 | 0.1237 | 0.1250 | 0.0013 |
| 50 | 0.2553 | 0.2609 | 0.0056 |
| 30 | 0.4444 | 0.4615 | 0.0171 |
| 20 | 0.7059 | 0.7500 | 0.0441 |

**So the disagreement is invisible on long series and material on short
ones** — which is precisely backwards from where people look for it.
And AICc is the default criterion for automatic order selection in both
R's `auto.arima` and this package. On a twenty-observation series the
two conventions can rank two candidate models differently.

That is the box. It is a better story than "the numbers differ",
because the honest version is "the numbers agree until they suddenly
matter".

---

## 2. The package API

Confirmed exported: **`tsdiff`**, **`tsundiff`**, **`diffinv`**.

The `tsdiff`/`tsundiff` pair is the round trip, and the round-trip
property is a chart in its own right (chart 7). Check what `diffinv`
does separately before writing about it — the name suggests it is the
lower-level integration primitive, but confirm rather than assume.

---

## 3. The chapter, beat by beat

### Beat 1 — A series that will not sit still (about 1 page)

**Chart 1 — a real trending series.** `global_economy` (pick one
country's GDP) or `GNP23`.

*Reading:* the mean of the first twenty years and the mean of the last
twenty years are not remotely the same number. So "the mean of this
series" is not a meaningful quantity, which is awkward, because
chapter 1's demonstration was entirely about the mean and its standard
error. If the mean itself is not a fixed thing, the problem is worse
than a wrong standard error. Nothing in the toolkit assumes a moving
target.

### Beat 2 — Two series that look alike and need opposite treatment (about 3 pages)

This is the chapter's most important beat and the one most books
under-serve. Hamilton is the exception and this is where to borrow from
him.

**Chart 2 — a random walk and a deterministic linear trend plus noise,
side by side. Two panels, no labels saying which is which.**

Ask the reader to say which is which. Most cannot, reliably. That is
the point.

*Reading:* one of these has a fixed underlying line and wanders around
it; the other has no underlying line at all and simply accumulates its
own shocks. The first is *trend-stationary*: subtract the line and what
remains is well-behaved. The second is *difference-stationary*: there
is no line to subtract, and the only thing that behaves is the
sequence of changes. They look similar because both drift, and drift is
what the eye sees.

**Chart 3 — the two-by-two. Four panels: each series detrended, each
series differenced.**

*Reading, and take real space here:* detrending the trend-stationary
series produces exactly what you want — noise around zero. Detrending
the random walk produces residuals that are still heavily
autocorrelated, still wandering, and now carry a spurious line that was
never there. Differencing the random walk produces clean noise.
Differencing the trend-stationary series produces something
over-differenced, with a distinctive negative correlation at lag one
that beat 5 will return to.

**Each treatment fixes one series and damages the other.** And the two
originals were nearly indistinguishable by eye. That is the whole
argument for chapter 9's unit-root tests — you cannot settle this by
looking, so you need a test.

State plainly that this distinction has real economic stakes. Whether
GDP is trend-stationary or difference-stationary determines whether a
recession is a temporary deviation from a path, or a permanent
reduction in the level. Economists have argued about it for decades.

### Beat 3 — Differencing, properly (about 1.5 pages)

Now the mechanics, arriving as the answer to a question the reader is
holding.

First differences. The backshift operator, introduced because it makes
seasonal differencing expressible rather than because notation is fun.
`(1 − B)y_t` and `(1 − B^m)y_t`.

**Chart 4 — the beat 1 series and its first difference, two panels.**

*Reading:* the level series has no stable mean; the difference series
does. What has been lost is the level itself — the difference series
cannot tell you whether GDP is large or small, only how fast it is
changing. That loss is real and is the reason `tsundiff` exists.

### Beat 4 — Seasonal differences and doing both (about 2.5 pages)

**Chart 5 — `jj` and its seasonal difference at lag 4. Two panels.**

*Reading:* the quarterly pattern is gone. The trend is not. A seasonal
difference removes seasonality and leaves everything else, which is
why it is almost never used alone.

**Chart 6 — three panels: `jj` raw, after `d=1`, after both `d=1` and
`D=1`.**

*Reading:* this progression is the airline model's differencing, and
the reader will meet it again by name in chapter 20. Each step removes
one structure and leaves the others. Worth noting the order does not
matter mathematically — `(1−B)(1−B⁴)` and `(1−B⁴)(1−B)` are the same
operator — but the intermediate pictures differ, and looking at both is
how you decide whether you needed the second one at all.

Also note the cost in observations: `d=1` and `D=1` on quarterly data
costs five observations off the front. On a short series that is a
significant fraction, and it connects directly to beat 5's box.

**Chart 7 — round trip. `tsundiff(tsdiff(y))` plotted over the
original.**

*Reading:* they lie exactly on top of each other, and that is not
trivial. Differencing throws away the level; integrating it back
requires knowing the starting value, and any function that does not ask
for one is guessing. Show the maximum absolute difference between
original and reconstructed — it should be at machine precision, and
saying so is more convincing than saying "they match".

### Beat 5 — Too much of a good thing (about 2 pages)

**Chart 8 — the ACF of a correctly differenced series beside the ACF of
an over-differenced one.**

*Reading:* the over-differenced series shows a large negative spike at
lag 1 that was not there before. Differencing a series that did not
need it does not merely waste an observation — it *introduces*
structure, a moving-average component that is an artefact of the
operation rather than a property of the data. Recognising that
signature is a practical skill and this chart is where the reader
acquires it.

**Chart 9 — variance against number of differences. Compute the
variance of the series after 0, 1, 2 and 3 differences, and plot it.**

*Reading:* the variance falls, reaches a minimum, and then climbs
again. The minimum identifies how many differences the series wanted.
This is a genuinely useful rule of thumb, it takes four lines of code,
and it appears in almost no introductory treatment. It also fails in
edge cases, so present it as a diagnostic rather than a decision
procedure — chapter 9 provides the actual tests.

**Then the `disagreement` box**, using section 1's material. The
structure to tell it in: the two implementations report different
`nobs`; the AIC values agree anyway, so most people never notice; and
the difference surfaces in AICc, which is the default for automatic
model selection, and only bites on short series where nobody is
looking.

### Beat 6 — Where this leaves you (half a page)

You can remove a trend two different ways, you know they are not
interchangeable, and you have no reliable way to decide which a given
series needs. Looking at the picture does not work — chart 2 proved
that.

That is chapter 9. But before the tests come the tools they are built
from, and the next of those is the moving average.

No recap.

---

## 4. The `india` box

Placed in beat 4, after chart 6.

India's Index of Industrial Production has been rebased several times —
the base year moved to 2011-12, and before that to 2004-05. When a
rebased series is spliced onto its predecessor, the join is a level
shift. Difference across it and you get a single enormous spike that is
purely an artefact of the accounting, not of Indian industry.

The spike is easy to see once you look for it and easy to miss if you
do not. Anyone differencing a long Indian macroeconomic series should
check where the base changed before trusting anything downstream.

One paragraph, and it makes the chapter's abstract material concrete
for a reader working with Indian data.

---

## 5. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Hamilton** | The trend-stationary versus difference-stationary distinction, and why it is a substantive economic question rather than a technicality. This is the one place Hamilton is clearly the best available source | The full asymptotic machinery around unit roots; that belongs near chapter 9, and even there only in outline |
| **fpp3 §9.1** | The practical framing — how many differences, and the warning against over-differencing | The `fable` syntax |
| **Shumway & Stoffer** | Their explicit detrending-versus-differencing comparison, and `gtemp`/`varve` as worked cases | Their fairly rapid move into ARIMA |
| **Tsay** | That prices are I(1) and returns are I(0) — differencing logs *is* the return transformation, which is the single most-used application of this chapter in practice, and it sets up Part V | Distributional properties of returns |
| **Cowpertwait & Metcalfe** | The patient introduction to the backshift operator | R-specific scaffolding |
| **Montgomery, Jennings & Kulahci** | Differencing's place in the Box–Jenkins workflow as a step with a decision attached | The regression chapters |

**On examples:** `global_economy`, `GNP23`, `jj`, `gtemp_land` and
`varve` are all bundled and GPL-3. Tsay's price-to-return point can be
made with `gafa_stock` or `nyse`, both bundled, rather than his CRSP
series. No external data needed.

---

## 6. Voice

Chapter 1 and 2 guidance applies. Two things specific to this chapter:

**Do not make the backshift operator the point.** It is notation, it
takes half a page, and books that linger on it lose readers who would
have been fine with "difference it again". Introduce it where it earns
its keep — seasonal differencing — and move on.

**Beat 2 deserves genuine care.** The trend-stationary versus
difference-stationary question has occupied serious economists for
forty years and remains unsettled for real series. Write it as a live
question, not as a taxonomy to memorise. If the chapter has one
passage worth rereading, it should be that one.

Additional things to avoid, beyond the earlier lists:

- Calling differencing "simple". It is mechanically simple and
  conceptually loaded, and saying otherwise misleads.
- Presenting the variance rule of thumb as a decision procedure.
- Any sentence of the form "As we saw in the previous section".

---

## 7. Checklist

- [ ] Nine charts through `@example ch3`
- [ ] Nine readings, five-part template, varied openings
- [ ] Chart 2 genuinely withholds which series is which
- [ ] Chart 3's two-by-two shows each treatment failing on one series
- [ ] Round-trip chart reports the actual maximum absolute error
- [ ] Over-differencing's negative lag-1 ACF spike shown, not described
- [ ] Variance-versus-differences chart present, framed as a diagnostic
      not a rule
- [ ] `disagreement` box uses the verified `nobs` numbers **and** the
      AICc correction table — the point is that AIC agrees and AICc
      does not
- [ ] `india` box on IIP rebasing
- [ ] Ends by opening chapter 4, no recap
- [ ] 11–13 pages
- [ ] British-Indian spelling

---

## 8. What to do

1. **Check what `diffinv` actually does** before writing about it. Three
   differencing-related functions are exported — `tsdiff`, `tsundiff`,
   `diffinv` — and I have only inferred the third's role from its name.
2. Generate charts 2 and 3 first. If the random walk and the trend
   series are too easy to tell apart at your chosen parameters, tune
   them until they genuinely are not — the beat depends on the reader
   being unable to call it.
3. Re-run the AICc correction table rather than copying it, so the
   numbers on the page came from a live calculation.
4. Render every chart; use the honest CI note from `10-plotting.md` if
   the local environment cannot.
5. Check against section 7.
