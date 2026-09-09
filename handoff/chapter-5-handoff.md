# Handoff: Chapter 5 — Autocorrelation

`docs/src/introduction/05-autocorrelation.md`. Target 12–13 pages.

Chapter 4 smoothed a series by assuming nearby observations are
related. This chapter measures that relationship, and the measurement
turns out to be the single most useful diagnostic in the book.

Same requirements as chapters 1–4. Reading template from chapter 1's
handoff.

---

## 1. A disagreement this chapter was not allocated

`all-chapters-handoff.md` gave chapter 5 a `julia` box and no
disagreement. Checking directly, there is one, and it changes which
model a reader would choose.

**The point estimates agree.** Verified on a 60-point series:

```
R acf():                          0.520010  0.403513  0.375020  0.316979  0.239605
statsmodels adjusted=False:       0.520010  0.403513  0.375020  0.316979  0.239605
statsmodels adjusted=True:        0.528824  0.417428  0.394758  0.339621  0.261387
```

R matches statsmodels' default exactly. The `adjusted=True` variant —
dividing by `n − k` instead of `n` — is available in Python and not in
R, and the gap grows with lag: 1.7% at lag 1, 9% at lag 5.

**Worth saying why the "biased" estimator is the default everywhere.**
The `n − k` version is unbiased and is *not* guaranteed positive
semi-definite, so it can produce an autocovariance matrix that no real
process could have generated. The `n` version is biased towards zero
and always valid. Statistics normally prefers unbiased; here it does
not, and the reason is worth a paragraph.

**The confidence bands do not agree, and this is the real finding.**

```
R's acf plot:              constant band, ±1.96/√n at every lag
statsmodels plot_acf:      bartlett_confint=True by default — widens with lag
```

Verified on the same series, `n = 60`:

| lag | acf | R band | Bartlett band | verdict |
|---|---|---|---|---|
| 1 | +0.5200 | ±0.2530 | ±0.2530 | significant under both |
| 2 | +0.4035 | ±0.2530 | ±0.3141 | significant under both |
| 4 | **+0.3170** | ±0.2530 | ±0.3708 | **significant under R only** |
| 8 | +0.1380 | ±0.2530 | ±0.4069 | significant under neither |

The two conventions agree exactly at lag 1 — both reduce to `1/√n` —
and diverge from lag 2 onwards. At lag 4 they give opposite answers on
the same data.

Since the ACF is the primary tool for choosing a model order, this is
not cosmetic. Two analysts using two respected packages will read the
same picture differently.

**Where this package sits**, confirmed from source: `acf` takes
`bartlett=true` as its default, matching statsmodels, with
`bartlett=false` available for the constant band. The docstring already
documents this. The chapter should say so plainly and show both.

---

## 2. The package API

Confirmed: `acf(x, lags; alpha=0.05, demean=true, adjusted=false,
bartlett=true, qstat=false)` and `pacf(x, lags; alpha=0.05,
method=:yw)`.

Both return `ACFResult`, tagged `:acf` or `:pacf` in a `kind` field,
and a single plot recipe branches on it. `pacf` supports `:yw`, `:ywm`
and `:ols`; `:burg` raises a clear error saying it is not implemented.

---

## 3. The chapter, beat by beat

### Beat 1 — How related is a series to its own past? (about 1 page)

**Chart 1 — `rec`, the recruitment series, 453 observations.**

*Reading:* the series oscillates with a period of roughly a year, and
the oscillation is not regular enough to be a clean seasonal pattern.
Chapter 4 smoothed something like this by averaging neighbours, on the
assumption that neighbours are informative. That assumption was never
checked. This chapter checks it.

### Beat 2 — Correlation with a shifted copy (about 2 pages)

The obvious attempt: take the series, shift it by one, and correlate.

**Chart 2 — lag plots. `rec` against itself at lags 1, 4, 8 and 12.
Four scatter panels.**

*Reading, and take space:* at lag 1 the points fall close to a line —
this month's recruitment tells you a great deal about next month's. At
lag 4 the cloud is rounder. At lag 12 there is structure again, because
the series has an annual rhythm. Autocorrelation is nothing more
mysterious than the correlation coefficient of each of these
scatterplots, collected into a sequence.

Starting here rather than with the formula matters. A reader who has
seen the scatterplots will never think of the ACF as an abstraction.

**Chart 3 — the ACF of `rec`, all lags at once.**

*Reading:* each bar is one of the scatterplots from chart 2, reduced to
a single number. The decay from lag 1, the trough, and the bump near
lag 12 are all visible at a glance, which is why nobody draws forty
scatterplots.

### Beat 3 — Reading the shapes (about 2.5 pages)

**Chart 4 — three ACFs side by side: white noise, an AR(1), and `jj`.**

*Reading:* white noise gives bars that are all small and inside the
band — the picture of nothing happening, and worth knowing by sight
because it is what a well-fitted model's residuals should look like.
The AR(1) decays geometrically. `jj` has spikes at 4, 8 and 12,
because quarterly data repeats every four observations.

**Chart 5 — the ACF of a trending series.**

*Reading:* the bars decay slowly and stay outside the band for a very
long time. This is the signature of a series that has not been
differenced, and it is the most common thing a beginner's ACF plot is
trying to tell them. Chapter 3 gave the fix; this is how the need for
it announces itself.

### Beat 4 — The partial autocorrelation (about 2 pages)

Motivate it properly. If `y_t` correlates with `y_{t-1}`, and `y_{t-1}`
with `y_{t-2}`, then `y_t` will correlate with `y_{t-2}` even if there
is no direct link. The ACF cannot separate the two. The PACF can.

**Chart 6 — ACF and PACF of a simulated AR(2). Two panels.**

*Reading:* the ACF decays gradually; the PACF cuts off sharply after
lag 2. That cut-off is the order of the process, read directly off the
picture.

**Chart 7 — ACF and PACF of a simulated MA(1). Two panels.**

*Reading:* exactly the mirror image — the ACF cuts off after lag 1 and
the PACF decays. This pair of pictures is the classical Box-Jenkins
identification method, and chapter 17 will use it in earnest. A reader
who internalises these two charts can identify simple models by eye.

Note honestly that real series rarely look this clean, and that chapter
21's automatic selection exists partly because of that.

### Beat 5 — Whose confidence band? (about 2.5 pages)

**Chart 8 — one ACF, two band conventions overlaid.**

Draw the `rec` or the 60-point ACF once, with the constant R-style band
and the widening Bartlett band both shown, and mark the lag where they
disagree.

*Reading:* the two bands answer different questions. The constant band
tests each lag against the hypothesis that the entire series is white
noise. Bartlett's band tests lag `k` against the hypothesis that the
series is a moving average of order `k − 1` — that is, that everything
up to `k − 1` is real and lag `k` is the first spurious one. The second
is usually the question you actually mean when reading an ACF for model
identification.

Then the `disagreement` box using section 1's table. The lag-4 case is
the whole point: `+0.3170`, significant under R's band, not significant
under Bartlett's, same data.

Disclose that this package defaults to Bartlett with `bartlett=false`
available, and that R users will see narrower bands than they expect.

**`julia` box** — one `ACFResult` type serves both functions, tagged by
a `kind` field, and one plot recipe branches on it for the axis label.
A small design decision, half a page, no more.

### Beat 6 — Where this leaves you (half a page)

You can now measure how a series relates to its own past, at every lag
at once, and read a model order off the picture.

What you cannot do is see a cycle whose period you do not already
suspect. The ACF of the sunspot series will show you that something
repeats; it will not tell you it is eleven years. For that you need to
stop asking about lags and start asking about frequencies.

That is chapter 6.

No recap.

---

## 4. The `india` box

Placed in beat 3, after chart 4.

Indian monthly industrial data carries an annual rhythm, so its ACF
shows the expected spike at lag 12 — but often a weaker and less
regular one than a comparable Western series. Part of the reason is
that the festival calendar moves: the Diwali production surge lands in
October some years and November in others, so the "annual" pattern is
not repeating at a fixed lag. The ACF, which only knows about fixed
lags, blurs it.

This is a genuine limitation of the tool rather than a fault in the
data, and it is one reason the calendar regressors of chapter 36 exist.

One paragraph.

---

## 5. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Shumway & Stoffer** | `rec` and `soi`, and their treatment of the ACF as a *picture* first | The move into asymptotic distribution theory |
| **fpp3 §2.8** | Lag plots before the ACF — the best pedagogical order anyone uses | The `feasts` syntax |
| **Hamilton** | Why the biased estimator is preferred, and the positive semi-definiteness argument | The full spectral connection; chapter 6 |
| **Cowpertwait & Metcalfe** | Patience with the correlation-of-a-shifted-copy idea | R specifics |
| **Montgomery, Jennings & Kulahci** | The ACF as a routine diagnostic step, not a special occasion | Regression material |
| **Tsay** | That returns typically show almost no autocorrelation while their *squares* show plenty — a forward reference to Part V that costs one sentence here | Volatility modelling itself |

**On examples:** `rec`, `soi`, `jj`, `sunspotz`, `gtemp_land` all
bundled and GPL-3.

---

## 6. Voice

Earlier guidance applies. Specific here:

**Lead with pictures, not formulae.** The ACF has a two-line definition
and a genuinely visual meaning, and chapter after chapter in other
books gets the order wrong. Chart 2's scatterplots come before any
formula.

**Do not oversell identification.** The AR(2)/MA(1) charts are clean
because they are simulated. Say so. A reader who expects real data to
look like chart 6 will be disappointed by their first real ACF and
conclude the method is broken.

Avoid, beyond earlier lists:

- The word "simply" attached to anything involving a formula.
- Introducing the Yule-Walker equations. The PACF's *meaning* is what
  matters here; the computation is the Manual's business.

---

## 7. Checklist

- [ ] Eight charts through `@example ch5`
- [ ] Lag-plot scatterplots come before the ACF formula
- [ ] ACF and PACF signature pair shown for both AR and MA
- [ ] Trending-series ACF shown as the "difference me" signal
- [ ] `disagreement` box with the verified band table; lag 4 flipping is
      the centrepiece
- [ ] The package's `bartlett=true` default disclosed
- [ ] The biased-versus-unbiased estimator explained, including why
      "biased" wins
- [ ] `julia` box on `ACFResult`, half a page
- [ ] `india` box on the moving festival blurring lag 12
- [ ] Simulated charts labelled as simulated
- [ ] Ends by opening chapter 6, no recap
- [ ] 12–13 pages, British-Indian spelling

---

## 8. What to do

1. Re-run the band comparison in section 1 during drafting so the table
   on the page is live output, and check whether the lag-4 flip
   survives your choice of series and seed. **If it does not, find a
   case where it does** — a disagreement box needs an actual
   disagreement, not a description of one.
2. Confirm what `pacf`'s `:ywm` method is before mentioning it; I have
   only seen the argument name.
3. Render every chart; use the honest CI note if the local environment
   cannot.
4. Check against section 7.
