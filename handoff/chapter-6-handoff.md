# Handoff: Chapter 6 — The Frequency Domain

`docs/src/introduction/06-the-frequency-domain.md`. Target 12–13 pages.

Chapter 5 asked how a series relates to its own past, lag by lag. This
chapter asks a different question about the same data — what rhythms is
it made of? — and the answer often reveals structure the ACF cannot.

Same requirements as chapters 1–5. Reading template from chapter 1.

---

## 1. The finding that makes this chapter necessary

**The periodogram is an inconsistent estimator.** More data does not
make it less noisy. Verified this session on white noise:

```
n =  128:  mean 1.6359   sd 1.6166   sd/mean 0.988
n = 1024:  mean 1.9347   sd 1.9526   sd/mean 1.009
n = 8192:  mean 2.0014   sd 2.0005   sd/mean 1.000
```

A sixty-four-fold increase in sample size leaves the relative scatter
exactly where it started, at essentially 1.0. The mean converges
nicely to the true spectral density of 2.0; the *variability around it*
does not shrink at all.

This is genuinely counterintuitive and it is the single most important
thing in the chapter. Every other estimator in this book behaves better
with more data. This one does not, because each extra observation adds
a new frequency to estimate rather than improving the estimates of the
existing ones. **That is why smoothing a periodogram is mandatory
rather than a refinement**, and it deserves a chart of its own.

## 2. The disagreement, already verified

Two authoritative sources, opposite defaults, same computation:

```
base R  spec.pgram(x, ...):   taper = 0.1   (10% split-cosine-bell)
astsa   mvspec(x, ...):       taper = 0     -- deliberately
```

`astsa` is Shumway & Stoffer's own package, and its documentation says
the zero default exists to force what it calls "conscious tapering".
Base R applies a taper without being asked.

Confirmed real defaults for `spec.pgram`: `taper=0.1`, `detrend=TRUE`,
`demean=FALSE`, `pad=0`, `fast=TRUE`. Note that `detrend=TRUE` is doing
more work than most users realise — it removes a linear trend before
computing anything.

**This package follows `astsa`** — `periodogram(x; taper=0.0,
detrend=true, demean=false, pad=0)`. Disclose it.

## 3. Verified worked example

Available from earlier work, reproducible:

```
series: 5·sin(2πt/12) + 3·cos(2πt/12) + noise, n=96, seed 42
spec.pgram(x, taper=0.1):
  peak frequency = 0.08333333333   (exactly 1/12)
  df = 1.791590494   bandwidth = 0.003007032652
  spec[1:5] = 5.787370716, 1.375818839, 2.646642418, 2.651190832, 9.823573797
```

---

## 4. The chapter, beat by beat

### Beat 1 — A cycle you cannot see (about 1 page)

**Chart 1 — `sunspotz`, plotted raw.**

*Reading:* something clearly repeats. Counting peaks by eye gives
something like a decade, but the peaks are uneven in height and not
evenly spaced, so any number you write down is a guess. The ACF from
chapter 5 will confirm that *something* repeats without telling you
its period with any precision. The question this chapter answers is how
to get the number.

### Beat 2 — Every series is a sum of waves (about 2 pages)

The idea, arriving before the machinery.

**Chart 2 — a constructed series and its three component sinusoids.
Four panels.**

Build a series from two or three known sine waves plus noise, and show
the parts and the whole.

*Reading:* the sum looks nothing like any of its parts. Given only the
bottom panel, recovering the top three is not obviously possible — and
yet it is, because sinusoids of different frequencies are orthogonal
and each carries its own share of the variance. That decomposition is
what the periodogram computes.

**Chart 3 — the periodogram of that constructed series.**

*Reading:* three spikes, at exactly the frequencies used to build it.
The vertical axis is variance, so the height of each spike is the
share of the series' variance that wave accounts for. Nothing has been
assumed about the data; this is an exact re-expression of it.

### Beat 3 — Why more data does not help (about 2.5 pages)

**Chart 4 — three periodograms of white noise at n = 128, 1024 and
8192. Three panels, same vertical scale.**

*Reading, and this is the chapter's centrepiece:* all three look
equally ragged. The theoretical spectral density of white noise is
flat, and none of these three pictures looks flat. The mean of the
periodogram values does converge — 1.64, 1.93, 2.00 against a true
value of 2.0 — but the scatter around it does not shrink at all. The
ratio of standard deviation to mean stays at 1.0 across a sixty-fourfold
increase in data.

The reason is structural rather than a defect. Doubling the sample
length doubles the number of frequencies estimated. Each estimate is
still based on effectively the same amount of information, so each
stays equally noisy. You end up with twice as many equally bad
estimates rather than the same number of better ones.

**Chart 5 — the same series smoothed with three different spans.**

*Reading:* averaging neighbouring periodogram values trades frequency
resolution for stability, which is exactly the trade chapter 4 made in
the time domain. The connection is worth naming explicitly — smoothing
a spectrum and smoothing a series are the same operation performed in
two different places, and a reader who sees that has understood
something structural about both chapters.

Report `df` and `bandwidth` alongside, and explain them in one sentence
each. They are how a spectral estimate declares how much smoothing it
did.

### Beat 4 — Real spectra (about 2.5 pages)

**Chart 6 — the spectrum of `sunspotz`.**

*Reading:* a clear peak, and reading the frequency off it gives a
period of roughly eleven years. Chapter 5's ACF could not have given
that number. This is the payoff for the whole chapter and it should be
allowed to land.

**Chart 7 — the spectrum of `soi`.**

*Reading:* two features rather than one — an annual peak, and a
broader, lower-frequency band corresponding to the El Niño cycle at
roughly four years. The second is not a sharp spike, and that is
informative in itself: El Niño is quasi-periodic, recurring on an
irregular schedule, which a broad band represents honestly and a single
frequency would not.

**Chart 8 — spectral leakage. A sinusoid at a frequency that does not
land exactly on a Fourier frequency.**

*Reading:* instead of one clean spike there is a peak with substantial
skirts spreading either side. The energy has leaked. This is not
measurement error; it is what happens when a finite record is
implicitly treated as one period of an infinitely repeating signal that
does not actually repeat. Tapering is the standard remedy, which brings
the chapter to its disagreement.

### Beat 5 — Whose taper? (about 1.5 pages)

**Chart 9 — the same series with taper 0 and taper 0.1, overlaid.**

*Reading:* the tapered version has smaller skirts and slightly reduced
peak height. It has traded a little bias for less leakage.

Then the `disagreement` box from section 2. The framing worth using:
base R applies a 10% taper silently, `astsa` refuses to taper at all so
that the user has to choose, and both are defensible positions held by
people who know exactly what they are doing. Disclose that this package
follows `astsa`, and that a reader porting code from base R will get
different numbers unless they set `taper=0.1` explicitly.

Mention `detrend=TRUE` here as well. It is on by default in both, it
removes a linear trend before anything else happens, and users who have
already differenced their series are often unaware they are detrending
twice.

### Beat 6 — Where this leaves you (half a page)

You can now find a period without knowing it in advance, and you know
why the raw estimate must be smoothed.

What none of Part I has addressed is the assumption running under all
of it. Every technique so far has assumed the series behaves the same
way at the start as at the end — same mean, same variance, same
rhythms. That assumption has a name and it is usually false.

Chapter 7 handles one of the ways it fails.

No recap.

---

## 5. The `india` box

Placed in beat 4, after chart 7.

Indian monsoon rainfall has a dominant annual frequency, which the
spectrum shows immediately. It also has quasi-periodic variation on a
longer scale, related to El Niño — which is why `soi` is a genuinely
relevant series for Indian agriculture and not merely a Pacific
curiosity. The relationship is well established in the meteorological
literature and it shows up as the same broad low-frequency band that
chart 7 displays.

One paragraph, and it makes `soi` feel less like a borrowed American
example.

---

## 6. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Shumway & Stoffer ch. 4** | The whole approach — spectral analysis as a *basic* tool, not an advanced topic. `soi`, `sunspotz`, and the `mvspec` taper philosophy | The full multivariate spectral development |
| **Hamilton ch. 6** | Why the periodogram is inconsistent, stated properly | The measure-theoretic framing |
| **fpp3** | Almost nothing — fpp3 barely treats the frequency domain, which is worth noticing as a real difference in emphasis between the practical and classical traditions | — |
| **Cowpertwait & Metcalfe** | A gentle entry to the idea that a series is a sum of waves | R specifics |
| **Ladiray & Quenneville** | That spectral peaks are used as a *diagnostic* by statistical agencies — residual seasonality shows up as a peak — forward-referenced to chapter 39 | X-11 mechanics |
| **Tsay** | Little; frequency methods are not central to his book | — |

**On examples:** `sunspotz`, `soi`, `speech`, `EQ5`/`EXP6` all bundled.
`speech` is worth a mention if space allows — it is the one series here
where the frequency content is the entire point.

---

## 7. Voice

**Do not derive the Fourier transform.** The chapter needs the reader to
believe that a series can be decomposed into waves and to read the
resulting picture. Neither requires the algebra, and including it will
lose people who would otherwise have been fine.

**The inconsistency result deserves emphasis, not a footnote.** It is
the most surprising thing in Part I and the reason the rest of the
chapter is shaped as it is.

Avoid, beyond earlier lists:

- "Frequency domain" and "time domain" used before either is explained.
- Complex numbers. They are not needed for anything in this chapter.
- Calling the periodogram "the spectrum". It is an estimate of it, and
  a bad one, which is the whole point.

---

## 8. Checklist

- [ ] Nine charts through `@example ch6`
- [ ] The inconsistency chart present, with the sd/mean ratios stated
- [ ] Constructed-series chart precedes any real data
- [ ] `sunspotz` period read off the spectrum as an actual number
- [ ] `soi`'s broad band explained as quasi-periodicity, not noise
- [ ] Leakage demonstrated before tapering is introduced
- [ ] `disagreement` box on taper defaults; package's `astsa` alignment
      disclosed
- [ ] `detrend=TRUE` default mentioned
- [ ] `india` box on monsoon and El Niño
- [ ] Ends by opening chapter 7, no recap
- [ ] 12–13 pages, British-Indian spelling

---

## 9. What to do

1. Re-run the inconsistency experiment during drafting. The numbers in
   section 1 are real but the chart needs its own run.
2. Confirm `spectral_density`'s `spans` argument behaviour before
   writing chart 5 — I have the signature but have not run it.
3. Check whether `periodogram` returns `df` and `bandwidth` or only
   `spectral_density` does. Beat 3 assumes the latter.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 8.
