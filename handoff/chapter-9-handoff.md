# Handoff: Chapter 9 — Testing for Unit Roots

`docs/src/introduction/09-testing-for-unit-roots.md`. Target 12–13 pages.

Chapter 3 showed two series needing opposite treatment and said a test
would decide. Chapter 8 showed the eye cannot. This chapter delivers
the tests, and is honest about how far they get you.

Same requirements as earlier chapters. Reading template from chapter 1.

---

## 1. Verified material

### The four-quadrant experiment, run this session

ADF and KPSS on four series of known construction, `n = 200`, both with
a constant and no trend term:

| series | ADF p | KPSS p | verdict given |
|---|---|---|---|
| white noise (stationary) | 0.0000 | 0.1000 | stationary ✓ |
| random walk (unit root) | 0.0870 | 0.0152 | unit root ✓ |
| **trend + noise (trend-stationary)** | 0.9635 | 0.0100 | **unit root ✗** |
| **AR(0.95) (stationary)** | 0.0653 | 0.0100 | **unit root ✗** |

**Two of the four are wrong, and they are wrong in different ways.**

The trend-stationary series is misdiagnosed because neither test was
told a trend might be present. Specify a trend term and both give the
right answer. The tests are only as good as the model you hand them,
and the default specification silently mis-diagnoses exactly the case
chapter 3 spent three pages on.

The AR(0.95) series is misdiagnosed for a different reason entirely.
It is genuinely stationary, correctly specified, and still called a
unit root — because at `n = 200` the tests cannot separate `φ = 0.95`
from `φ = 1`. That is a power problem, not a specification error, and
no amount of care fixes it.

### KPSS p-values are clipped, and statsmodels says so

Running the above raised, three times:

```
InterpolationWarning: The test statistic is outside of the range of
p-values available in the look-up table.
```

So the `0.0100` and `0.1000` figures above are **not p-values**. They
are the edges of a lookup table. KPSS critical values are tabulated,
not computed from a closed form, and outside the tabulated range the
implementation returns the nearest bound and warns. A reader who
reports "p = 0.01" from KPSS is reporting a table boundary.

This is a genuinely practical gotcha and belongs in the chapter.

### MacKinnon's response surface **[S]**

Read from `statsmodels.tsa.adfvalues` source rather than executed. ADF
p-values in both R and Python come from MacKinnon's response-surface
regression — tables of polynomial coefficients per regression type,
with a regime cutoff (`tau_star`) selecting between small-p and large-p
coefficient sets, then `norm.cdf(polyval(coeffs, teststat))`. Simple
interpolation among tabulated critical values is meaningfully cruder.

### Phillips-Perron's two statistics **[S]**

The `τ` and `ρ` forms have different null distributions. Several
implementations, including this package, return no p-value for `ρ`
rather than a wrong one — the same honest-refusal pattern that appears
again in chapter 10 with Durbin-Watson.

---

## 2. The chapter, beat by beat

### Beat 1 — The question chapter 3 left open (about 1 page)

**Chart 1 — the two series from chapter 3, side by side again.**

Deliberate callback. The random walk and the trend-stationary series
that could not be told apart.

*Reading:* six chapters ago these were indistinguishable and the
promise was that a test would settle it. Here is the test. It settles
it less completely than the promise implied, and the shape of the
shortfall is worth knowing before trusting any of it.

### Beat 2 — Testing for a unit root (about 2.5 pages)

The Dickey-Fuller idea, stated plainly: regress the change on the
level, and ask whether the coefficient on the level is zero. If it is,
there is nothing pulling the series back and it wanders freely.

Note the awkwardness that makes this subject hard — **the null
hypothesis is non-stationarity**. Failing to reject does not establish
a unit root; it establishes that you could not rule one out. Most
people read it backwards.

**Chart 2 — ADF applied to the four test series, statistics and
critical values shown.**

*Reading:* white noise rejects decisively. The random walk does not.
So far the test does what it should.

**Chart 3 — ADF's rejection rate against φ, from 0.5 to 1.0.**

Simulate at each φ, run ADF, record the rejection rate.

*Reading, and this is the chapter's most important figure:* at `φ = 0.5`
the test rejects almost always, correctly. At `φ = 0.95` it rejects
rarely, incorrectly — the series is stationary and the test says it may
not be. The curve does not fall off a cliff at 1.0; it sags for a long
way before it. Everything in that sagging region is a stationary
process the test will call a unit root.

State the consequence plainly. Many economic series sit at φ around
0.9 to 0.99. A large share of published unit-root findings are
statements about test power rather than about the world.

### Beat 3 — Testing the other way round (about 2 pages)

KPSS reverses the null: stationarity is the hypothesis being tested.

**Chart 4 — KPSS on the same four series.**

*Reading:* reversing the null does not double your information, but it
does let you cross-check. Two tests with opposite nulls give four
possible combinations, and only two of them are conclusive.

**Chart 5 — the four-quadrant table as a figure.**

| | KPSS does not reject | KPSS rejects |
|---|---|---|
| **ADF rejects** | stationary | contradictory |
| **ADF does not reject** | inconclusive | unit root |

*Reading:* the diagonal is what you hope for. The off-diagonal
happens often enough to matter, and each cell means something
different. Contradictory usually means the specification is wrong —
often a missing trend term. Inconclusive usually means the series is
too short or too persistent to tell, which is the beat 2 power problem
wearing a different hat.

Then the verified table from section 1, including both failures.

### Beat 4 — Getting the specification right (about 2 pages)

**Chart 6 — the trend-stationary series tested with and without a
trend term. Four results.**

*Reading:* with a constant only, both tests say unit root. Add a trend
term and both give the right answer. Nothing about the data changed;
the question changed. This is the single most common way unit-root
testing goes wrong in practice, and it goes wrong silently — the
default specification is the one most people use and it is the wrong
one for any series with a visible trend.

**Chart 7 — the effect of lag selection on the ADF statistic.**

*Reading:* too few lags leaves autocorrelation in the residuals and the
test is invalid. Too many and power drains away. Automatic selection by
information criterion is the usual compromise, and the honest summary
is that the answer moves when the lag count moves, which is worth
knowing before treating any single p-value as decisive.

### Beat 5 — What the numbers actually are (about 2 pages)

Two `disagreement` boxes, or one with two parts.

**KPSS p-value clipping**, from section 1. Show the warning. Explain
that KPSS critical values are tabulated and the reported 0.01 or 0.10
is a table edge rather than a computed probability.

**MacKinnon's response surface** for ADF, from section 1. Both R and
Python use it; simple interpolation is cruder. Note that this package
implements the response surface following the same coefficients.

**Phillips-Perron's ρ**, briefly: a second test statistic with a
different null distribution, for which this package returns no p-value
rather than a wrong one. One paragraph, and note that the same
principle recurs in chapter 10.

### Beat 6 — Where this leaves you (half a page)

You can test for a unit root, you know the tests have a specification
trap and a power problem, and you know to run two of them with opposite
nulls.

What you cannot yet do is check whether a fitted model has actually
captured the structure it was supposed to. That requires looking at
what the model left behind.

Chapter 10.

No recap.

---

## 3. The `india` box

Placed in beat 2, after chart 3.

Indian quarterly GDP has been published in its current form since the
2011-12 base revision, which gives a series of a few dozen
observations. Chart 3's power curve is a direct statement about what
can be learned from that: at `n = 50` a unit-root test on a persistent
series is close to uninformative.

The practical response used by most Indian macro work is to lean on
economic reasoning rather than the test — output is treated as
difference-stationary because that is what growth theory implies, not
because a test said so. That is a defensible position and worth stating
as such, rather than pretending the test settled it.

One paragraph.

---

## 4. What each book contributes

| Book | Take | Leave |
|---|---|---|
| **Hamilton chs. 15–17** | The definitive treatment. The null-is-non-stationarity awkwardness, and why the distributions are non-standard | The functional central limit theorem |
| **fpp3 §9.1** | The practical use — run it, difference, run it again | The theory, which fpp3 largely skips |
| **Shumway & Stoffer** | ADF as a routine step with worked examples | — |
| **Tsay** | Unit roots in financial series, and why testing them matters for pairs trading and cointegration | Advanced cointegration; that is a later stage |
| **Montgomery, Jennings & Kulahci** | The Box-Jenkins workflow context | — |
| **Cowpertwait & Metcalfe** | The gentle version of a hard idea | R specifics |

**On examples:** `GNP23`, `varve`, `gtemp_land`, `global_economy`
bundled. The four constructed test series are simulated and should be
labelled.

---

## 5. Voice

**Do not present these tests as reliable.** They are the best available
and they fail on two of four constructed cases where the answer is
known. A chapter that oversells them produces readers who trust a
p-value they should not.

**The null-hypothesis direction needs saying more than once.** Failing
to reject a unit root is not evidence of one. Readers get this wrong
constantly and one careful sentence will not fix it.

Avoid, beyond earlier lists:

- Deriving the Dickey-Fuller distribution. State that it is
  non-standard and why that matters practically.
- Presenting the four-quadrant table as a decision procedure. Two of
  its cells mean "find out more".

---

## 6. Checklist

- [ ] Seven charts through `@example ch9`
- [ ] Power curve (chart 3) present — the chapter's key figure
- [ ] The four-quadrant experiment shown with both failures explicit
- [ ] The trend-specification trap demonstrated, not described
- [ ] KPSS p-value clipping shown with the actual warning text
- [ ] MacKinnon response surface explained; package's use stated
- [ ] PP `ρ` no-p-value policy noted
- [ ] Null-hypothesis direction stated at least twice
- [ ] `india` box on short quarterly GDP series
- [ ] Ends by opening chapter 10, no recap
- [ ] 12–13 pages, British-Indian spelling

---

## 7. What to do

1. Re-run the four-quadrant experiment during drafting. Reproduce the
   `InterpolationWarning` text exactly as it appears rather than
   paraphrasing it.
2. Generate the power curve properly — at least 500 replications per φ
   value, and state the number in the caption.
3. Confirm this package's ADF actually uses the response surface rather
   than interpolation before claiming it does. The Stage 2 fill-gap
   handoff specified the change; check it landed.
4. Render every chart; honest CI note if the environment cannot.
5. Check against section 6.
