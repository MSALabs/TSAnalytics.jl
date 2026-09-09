# Chapter Writing Guide

The standard for all 41 chapters of *Introduction to Time Series
Analysis*. Written once, referenced for every chapter, so that the
fortieth reads like the first.

---

## 1. Four requirements, and one that needs adjusting

**Take the best from each book** — adopted, and made concrete in
section 5.

**Read like a story** — adopted, and made concrete in section 4. This
is the requirement most likely to be quietly abandoned under deadline,
so it gets a structural template rather than an exhortation.

**Sound like it was written by an Indian author** — adopted, and made
concrete in section 3. It is also the requirement most easily done
badly, so section 3 says plainly what it does and does not mean.

**Cover all relevant examples from all books** — **this one needs
adjusting, and here is why.**

Three obstacles, all real:

*Licensing.* Only 83 of the textbook datasets can be redistributed —
the GPL-3 ones from `astsa` and `tsibbledata`. Tsay's series come
substantially from CRSP, a commercial subscription database.
Montgomery's are Wiley, all rights reserved. Cowpertwait's licence is
unstated. A chapter cannot work through an example whose data the
reader is unable to load.

*Copyright.* A worked example in a textbook is not just data — it is
the author's chosen framing, sequence and commentary. Reproducing that
is not permitted regardless of whether the data is free.

*Volume.* Eight books across forty-one chapters would run far past
three hundred pages, and a chapter that covers everything covers
nothing well.

**The adjusted requirement**: every chapter must work through the
**canonical examples** — the ones that recur across two or more books
precisely because they teach the concept better than anything else does
— plus at least one Indian example wherever the data supports it.
Where a canonical example cannot be shipped, substitute from the
83-dataset bundle and say so in the text rather than silently omitting
it. Section 6 maps this out.

---

## 2. A decision needed first: spelling convention

The codebase is currently **inconsistent**, and this must be settled
before forty-one chapters entrench the confusion. Counted directly:

| Word | Occurrences |
|---|---|
| `licence` | 13 |
| `license` | 2 |
| `behavior` | 25 |
| `behaviour` | 6 |
| `modeling` | 2 |

So the package spells *licence* the British-Indian way and *behavior*
the American way, in the same codebase.

**Recommendation: British-Indian spelling throughout**, which matches
the dominant `licence` usage, matches Indian academic and official
convention, and matches the author's own variety of English. That means
*behaviour*, *modelling*, *analyse*, *centre*, *summarise*,
*generalise*, *licence* (noun) — and correcting the 25 instances of
`behavior` and 2 of `modeling` in the existing source as a small
separate cleanup, not silently diverging from them.

The alternative — American spelling throughout — is equally defensible
as a convention, but conflicts with the author's voice and with the
majority `licence` usage. Either way, **pick one and enforce it.**
Julia's own function names (`transform`, `normalize`) stay as they are;
this governs prose only.

---

## 3. Voice

The author is an Indian statistician writing from XKDR Forum in Mumbai.
The book should sound like that person, because that is who is writing
it.

**What this means concretely:**

- **British-Indian spelling**, per section 2.
- **Indian institutions cited alongside the usual ones.** The Reserve
  Bank of India, MoSPI, the NSE and the CSO belong in the same
  sentences as the Federal Reserve, the ABS and the US Census Bureau.
  Most books in this field cite only the latter set; there is no reason
  for that.
- **Indian data throughout**, not confined to a single token chapter.
  The IIP series, NSE trading calendars and the Diwali effect are
  genuinely good teaching examples, not decoration.
- **Indian numbering where the context is Indian.** A figure quoted
  from an RBI release may sensibly be given in crore; a statistical
  quantity should not be.
- **Dates as 14 March 2024**, not 3/14/2024.
- **Measured, direct, unhurried prose.** Formal without being stiff.
  This is the register of good Indian technical and academic writing,
  and it happens to be the right register for the book anyway.

**What this does not mean.** It does not mean introducing
non-standard constructions, deliberate errors, or any performance of
Indianness. Indian English in professional writing is a full standard
variety, and in formal technical prose it differs from British English
mainly in vocabulary and reference points rather than grammar.
Manufacturing quirks would be both condescending and worse writing.
The aim is an authentic voice, not an accent.

---

## 4. Structure: what "reads like a story" actually means

A story has a question, an attempt, a complication and a resolution. A
taxonomy has a list of definitions. Most textbook chapters are
taxonomies. Every chapter here follows six beats:

**1. The question.** Open with something the reader wants to know,
about data they can see. Not a definition. *"Indian industrial
production rose 4.2% last October. Did it really, or was that Diwali?"*
is a chapter opening. *"Let* X_t *be a stochastic process"* is not.

**2. The obvious attempt.** What a sensible person tries first, and why
it is not quite enough. This earns the concept before it arrives.

**3. The idea.** Now the actual method, arriving as the answer to a
question the reader is already holding. Definitions and notation come
here, after the motivation, not before it.

**4. Making it work.** The Julia code. Real data from the bundle, real
output, nothing elided. The reader should be able to run every line.

**5. The complication.** Where it breaks, what it assumes, or — best of
all — a `!!! disagreement` box. These are the chapter's natural moment
of tension: two respected implementations, the same input, different
answers. That is a genuine mystery with a genuine resolution, and it is
material no other book on this subject has.

**6. Where this leaves you.** What the reader can now do, what they
still cannot, and the question that opens the next chapter. A chapter
should end somewhere other than where it began.

Not every chapter needs all six at equal weight, but a chapter missing
beats 1, 5 and 6 has become a taxonomy and should be sent back.

**Length**: roughly 6–8 pages. A chapter running past ten either
contains two chapters or has stopped being selective.

**One deliberate exception: Chapter 1.** It carries thirteen worked
examples, each with a full reading, because it is the gallery chapter —
the one place a reader meets the whole range of what a time series can
be, decides whether to trust the author, and is introduced to series
referenced by name throughout the rest of the book. That runs to
roughly 14–16 pages. No other chapter gets this licence; if a second
chapter seems to need it, that is a sign that chapter is trying to do
too much, not that the exception should widen.

---

## 5. What each book is best at

Take the strength, leave the structure. No chapter should be
recognisably modelled on any single one of these.

| Book | Its genuine strength | Use it for |
|---|---|---|
| **Hamilton** | Mathematical rigour; derivations that explain *why* a result holds | Beat 3, when a claim needs proper justification rather than assertion |
| **Hyndman & Athanasopoulos (fpp3)** | Practical judgement; honest about what does not work | Beats 1, 2 and 6 — framing and limitations |
| **Shumway & Stoffer** | Worked examples on real series; spectral methods treated as basic, not exotic | Beat 4, and the canonical datasets |
| **Durbin & Koopman** | One unifying framework beneath apparently separate models | Part VI, where the Kalman filter is revealed underneath Part IV |
| **Tsay** | Financial intuition; why volatility behaves as it does | Part V, beats 1 and 2 |
| **Ladiray & Quenneville** | The mechanics of X-11, filter by filter | Part VIII |
| **Montgomery, Jennings & Kulahci** | Industrial and engineering framing; monitoring | Beat 1, for non-financial, non-macro examples |
| **Cowpertwait & Metcalfe** | A gentle on-ramp; assumes less | Early chapters, calibrating the entry level |

---

## 6. Examples: canonical series, and what to do about the ones we cannot ship

**Confirmed available in the bundle** (all GPL-3, all loadable by the
reader):

| Series | Source | Canonical for |
|---|---|---|
| `jj` | astsa | Trend with multiplicative seasonality; log transforms |
| `soi`, `rec` | astsa | Cross-correlation; spectral analysis; lagged regression |
| `varve` | astsa | Variance stabilisation; long memory |
| `cmort`, `tempr`, `part` | astsa | Regression with autocorrelated errors |
| `sunspotz` | astsa | Cycles; spectral peaks |
| `nyse`, `sp500w`, `djia` | astsa | Volatility clustering; GARCH |
| `GNP23`, `GDP23` | astsa | ARIMA on macroeconomic data |
| `pelt` | tsibbledata | The lynx–hare cycle |
| `aus_production`, `aus_retail` | tsibbledata | Seasonal ARIMA; decomposition |
| `vic_elec` | tsibbledata | Multiple seasonality (MSTL) |
| `PBS` | tsibbledata | Hierarchical structure; many related series |
| `gafa_stock` | tsibbledata | Financial returns |
| `global_economy` | tsibbledata | Cross-country panels |

**Not available, and the honest substitutions:**

- *Tsay's financial series* (CRSP) → use `nyse`, `sp500w`, `djia`,
  `gafa_stock`. These support every volatility concept in Part V.
  Where a specific Tsay result is worth citing, cite it as a finding
  rather than reproducing the example.
- *Montgomery's engineering series* (Wiley) → substitute from the
  bundle, or construct a simulated series and label it as simulated.
- *Cowpertwait's series* (licence unstated) → substitute.

**The Indian examples** are ours to construct and are the book's
distinguishing thread. IIP with a Diwali regressor, NSE trading-day
effects, and the rebasing problem all appear in Parts VII and VIII, and
smaller Indian illustrations belong earlier wherever they fit
naturally.

**When a canonical example cannot be reproduced, say so in the text.**
*"Tsay works this through on IBM daily returns from CRSP, which is
licensed data; the same behaviour is visible in `nyse`, which ships
with this package."* That sentence costs one line and is more useful to
the reader than a silent omission.

---

## 7. Per-chapter checklist

Before a chapter is considered done:

- [ ] Opens with a question about visible data, not a definition
- [ ] The obvious naive attempt appears and is shown to be insufficient
- [ ] Every code block runs, against bundled data, with real output
- [ ] At least one canonical example from the reference books
- [ ] An Indian example, where the subject admits one
- [ ] At least one of the three recurring boxes (`disagreement`,
      `india`, `julia`) — most chapters have material for more
- [ ] Ends by naming what the reader still cannot do
- [ ] Spelling convention followed consistently (section 2)
- [ ] 6–8 pages
- [ ] Every claim about R or Python behaviour has been *run*, not
      recalled. This is the standard the package was built to, and the
      book must not fall below it
- [ ] Cross-references to the Manual and API Reference where a reader
      may want the full signature

---

## 8. Where to start

**Not chapter 1.** Chapter 1 sets the tone for the whole book and is
the hardest thing to write before the voice is established.

**Start with chapter 15, STL.** It has everything the template needs:
a clear motivating question, a genuinely interesting method, code that
runs on `aus_production` or `vic_elec`, and one of the strongest
`disagreement` boxes available — a median bug in the original Fortran
that Python fixed and R did not, where the numbers genuinely differ
near outliers and neither implementation is wrong.

Write it, review it against section 7, and treat it as the reference
chapter. Then chapter 1, once the voice is settled and there is
something to point at.
