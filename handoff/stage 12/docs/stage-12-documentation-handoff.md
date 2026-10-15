# Stage 12 — Documentation handoff
## API reference, manual, and the book

**Written:** 2026-10-15
**Repository HEAD at writing:** `2eccb82`
**Prerequisite reading:** `stage-12-overall-handoff.md` and the three sub-stage handoffs.

---

## 0. What exists now [V]

| Surface | Current | Stage 12 adds |
|---|---|---|
| API pages | **17** | **2 new, 2 updated** |
| Manual chapters | **20** | **3 new, 2 updated** |
| Book chapters | **57** in **12 Parts** (+3 appendices) | **4 new across 2 Parts**, plus a Frontier rewrite |

`arx` is documented in **`api/arimax.md`** and in manual chapters 01, 08, 10, 11 and 16 [V] — which matters, because 12.3 extends `arx` rather than adding a type (§5.2).

---

## 1. The six sources — and one topic they simply do not cover

Offsets re-verified; **one correction to the Stage 11 table**: Cowpertwait's Chapter 3 offset is **+12, not +13** (PDF 64 carries printed folio 52) [V]. Everything else held.

### 1.1 Coverage, by topic

| | 12.1 intermittent | 12.2 change-point / breaks | 12.3 lag-feature / ML |
|---|---|---|---|
| **Hamilton** | **none** | **§14.3 (424–426), §15.5 (449–451) — primary for theory** | **none** |
| **Montgomery** | **none** | **§2.6.3 (60–66) monitoring, §6.7 (330–338) intervention — co-primary** | §7.7 (372–375), cautionary, no numbers |
| **Tsay** | **none** | §5.8 (274–276), §4.2.2 (211–213) | **§4.1.9 (199–205) — primary, fully worked** |
| **fpp** | **none** | printed 103, the recursive-residual primitive only | **Ch. 13 (131–133) — primary for specification** |
| **Cowpertwait** | **none** | Exercise 3.8 (printed 66) — a one-formula exercise | **none** |
| **Shumway & Stoffer** | **none** | **none** outside Markov-switching | **none** |

### 1.2 Intermittent demand is absent from all six books

Zero hits for `intermittent|croston|sporadic|slow-moving|zero[- ]inflat` across all six full texts, ~271k lines, with follow-up sweeps for `spare part|lumpy|inventory control|demand forecast` and `count data|Poisson` returning only false positives. [V]

**Including fpp** — Croston does not appear in Hyndman's 138-page lecture notes, despite his maintaining `forecast::croston`.

**Consequence, and it must be stated plainly in the chapter:** there is no citable source for Croston, SBA or TSB in this corpus. The chapter cites **Croston (1972)**, **Syntetos & Boylan**, and the *published* **Hyndman & Athanasopoulos *Forecasting: Principles and Practice*** (the book, §12.x — not these notes). **Citing any of the six for this topic would be a fabrication.** The nearest adjacent material in the whole corpus is Shumway §6.9's Poisson-HMM for counts, which is a different model class.

This is a useful negative: it tells the reader the topic sits outside the classical curriculum, which is true and worth saying.

### 1.3 No Chow test, and no critical-value tables, in any of the six

Checked by name sweep (every `Chow` hit is **Chow, W. M. (1965)** on adaptive smoothing, or "Chowdhury") and by phrasing sweep. **Six books, zero Chow tests.** [V]

Hamilton comes closest and gives it as a **Wald** test (§14.3), with the known-break case as the base and the unknown-break case as a sup over `T₀ ∈ [0.15T, 0.85T]` — **the trimming fraction is stated explicitly in his text**, which usefully pins the default a port should use — and the distribution outsourced to Andrews (1993).

**No table of critical values for any change-point statistic exists in any of the six.** Verified by inventorying the corpus's only two statistical-table appendices: Hamilton's Appendix B (printed 751–768) stops at Johansen; Montgomery's Appendix A (printed 387–406) stops at Durbin–Watson. The other four have no table appendix at all. [V]

**So the supF null distribution genuinely has to come from the papers** — Andrews (1993), *Econometrica* 61:821–56, whose full citation is printed at Hamilton p. 433, with Chan (1991) and Andrews & Ploberger (1994) as the threshold-analogue sources from Tsay p. 211. This independently corroborates the charter's §6 decision to scope supF p-values out: the corpus supplies only the citations.

---

## 2. The notation reconciliation — and a book that contradicts itself

### 2.1 Hamilton indexes the break two different ways in the same book

| | convention |
|---|---|
| **§14.3**, printed 425 | `d₁ₜ = 1 for t ≤ T₀` → **`T₀` is the LAST observation of the FIRST regime** |
| **§15.5 [15.5.7]**, printed 450 | `t < T₀` is regime 1, `t ≥ T₀` is regime 2 → **`T₀` is the FIRST observation of the SECOND regime** |

Same symbol, same book, opposite inclusion. Both read from rasterized pages (PDF 440 and 465), since Hamilton's text layer is OCR. [V]

**A chapter that cites "Hamilton's convention" without naming the section will be off by one half the time.**

### 2.2 The full collision table

| Symbol | Hamilton §14.3 | Hamilton §15.5 | Tsay §5.8 | Montgomery §6.7 | Montgomery §2.6.3 | Cowpertwait Ex. 3.8 |
|---|---|---|---|---|---|---|
| break date | `T₀` | `T₀` | `t_o` | **`T`** | — | — |
| indexing | last of first | **first of second** | last of first (stated in words) | **first of second** | n/a | n/a |
| `T` means | sample size | sample size | — | **the break date** | **the CUSUM target** | — |
| `τ` means | — | — | — | — | — | **the CUSUM target** |
| test form | **Wald**, χ²(a) | none (model only) | sup-LR; F (TAR-F) | t on `ω̂₀` | **none — a signal, not a test** | none |
| trimming | **explicit: 0.15T–0.85T** | — | `[v, u]` | — | — | — |

**The indexing splits 2–2.** There is no safe default. **Our convention is the 1-based inclusive end of the preceding segment** (12.2 handoff §4.1), matching `changepoint` and `strucchange`; the chapter states it once, in the first paragraph, and maps it to each cited book.

**And `T` means three different things across two books.** Montgomery's §2.6.3 `T` is the CUSUM *target value*, his §6.7 `T` is the *break date*, and Hamilton's `T` is the *sample size*. Any chapter citing both Montgomery sections must rename.

---

## 3. The numeric cross-checks the corpus does supply

Two are worth building a chapter around.

### 3.1 Montgomery Table 6.13 — the best known-date cross-check in the corpus

Printed 337 (PDF 350), read from a rasterized page [V]. Weekly sales of a cereal brand, 104 weeks, data printed in full in Table 6.12; a rival launched at **Week 88**; pre-intervention ARIMA(0,1,1) fitted to weeks 1–87, model `yₜ = ω₀Sₜ^(88) + εₜ/(1−θB)`:

| parameter | estimate | SE | t | p |
|---|---|---|---|---|
| `MA1,1` | **0.75886** | 0.06555 | 11.58 | < .0001 |
| `NUM1` (ω₀) | **−2071.5** | 637.43397 | −3.25 | 0.0016 |

Variance estimate 82507397; AIC 2171.807; SBC 2177.076; 103 residuals.

**Real data printed in the book, exact estimates, exact standard errors, exact p-values** — reproducible, and the obvious worked example for the known-break-date section.

### 3.2 Tsay §4.1.9 — the only ML-reduction cross-check anywhere

Printed 203–204 [V]. IBM monthly log returns including dividends, Jan 1926 – Dec 1997 training (**864 observations**), a **3–2–1 skip-layer** network on lags `r_{t−1}, r_{t−2}, r_{t−3}`:

- fitted network printed in full, with both logistic hidden units written out
- **residual SE 6.56** against the AR(1) benchmark's **6.61**
- **out-of-sample MSFE (1998–1999): random walk with drift 91.85, AR(1) 91.70, network 91.74**
- and, unusually honestly, **an instability band**: across random starting weights the network's out-of-sample MSE ranges **89.46 to 93.65**

That last figure is the most useful thing in the corpus for this topic — a published statement that the method's answer moves more than the difference it is trying to detect. It is also exactly the point the 12.3 handoff makes by measuring DirRec ≡ Direct under OLS.

### 3.3 Montgomery's own internal inconsistency — do not use 14.38

Example 2.13 computes `H = 5σ̂ = 5(0.8865)(3.24) = **14.36**`, while Figure 2.34 on the same page is labelled **UCL = 14.38**. Both read from the raster — **the discrepancy is in the book**, and the text is arithmetically right (5 × 0.8865 × 3.24 = 14.3613). [V]

Do not use 14.38 as a target. And note that Montgomery's published CUSUM example concludes "no obvious forecasting model inadequacies" — it is an **in-control** case, so it tests nothing about detection.

---

## 4. Montgomery as monitoring, not testing — a distinct citation

Confirmed and extended: §2.6.3 is unambiguously a **statistical-process-control** device applied to one-step-ahead forecast errors. No null hypothesis, no p-value, no break-date estimate anywhere in the section. [V]

**Decision limits, all given explicitly** [V]:

| device | printed | limits |
|---|---|---|
| moving-range σ̂ | 61 | `σ̂ = 0.8865·MR̄` |
| CUSUM | 62 | `K = 0.5σ_e`, `H = 5σ_e` |
| EWMA | 64 | `UCL/LCL = T ± 3σ_e√(λ/(2−λ))·√(1−(1−λ)^{2t})`, typical `0.05 < λ < 0.2` |
| tracking signals | 65–66 | `CETS`, `SETS`; thresholds left unspecified |

Published chart limits: Fig. 2.33 individuals **UCL 8.91 / X̄ 0.28 / LCL −8.34**, MR **UCL 10.60 / MR̄ 3.24**; Fig. 2.34 CUSUM **±14.38**; Fig. 2.35 EWMA **UCL 2.261 / 0.282 / −1.697** at λ = 0.1. (`pdftotext` silently dropped `UCL=8.91` — rasterization was necessary.) [V]

**This is a genuinely different object from `strucchange`'s hypothesis test and deserves its own section and its own citation.** The manual is where it belongs: "is this model still right?" is a practitioner question, and it is the one thing no other book in the corpus covers.

---

## 5. API reference — two new pages, two updated

### 5.1 New

| Page | Covers | Charts |
|---|---|---|
| `api/intermittent-demand.md` | `croston`, `tsb`, `adida`, `imapa`, `IntermittentModel`, `demand_intervals`, `adi_cv2`, `cfe`, `spis` | **3** — stem decomposition, ADI/CV² quadrant, inter-arrival histogram |
| `api/change-points.md` | `changepoints`, `breakpoints`, `break_confint`, `fstats`, `efp`, and their result types | **4** — segmented series with bands, selection curve, F-statistic path, fluctuation process |

### 5.2 Updated, not new

**`api/arimax.md`** documents `arx` [V], so 12.3's additions belong there: the injected regressor, the `direct` strategy, past-only exog, and — the one users will notice most — **that `forecast` now accepts future exog, so `arx` + `fourier_terms` finally works**. Its current docstring calls that "a real, deliberate gap, not an oversight" [S]; the update must say the gap is closed.

**`api/plotting.md`** gains the eight new recipes.

**No new page for 12.3.** Adding one would imply a new entry point, which §1 of the 12.3 handoff rules out.

---

## 6. Manual — three new chapters, 21–23

Append, as with Stage 11: 21–23, with `15-plotting` and `16-coming-from-r-python` updated. Those two are deliberately terminal and the cost of appending is updating them.

| Chapter | Title | Charts | Sources |
|---|---|---|---|
| `21-intermittent-demand.md` | Series That Are Mostly Zero | **5** | **no book** — Croston (1972), Syntetos & Boylan, published FPP §12.x |
| `22-change-points-and-breaks.md` | Finding When Something Changed | **6** | **Montgomery §2.6.3 + §6.7** (practitioner), Hamilton §14.3 (theory) |
| `23-lag-feature-regression.md` | Any Regressor, Any Lags | **4** | Tsay §4.1.9, fpp Ch. 13, Montgomery §7.7 |

**15 new manual charts.**

### 6.1 Montgomery is the register for chapter 22

Charter aside, the survey's judgement is worth following: Montgomery's treatment is *"practitioner / SPC-engineering. Formulas given as recipes with default constants, every method illustrated on real data with software output pasted in. Zero asymptotic theory."* That is the manual's voice. Chapter 22 should carry a "has this model drifted?" section built on §2.6.3's limits, because `nyblom_test` already says the parameters moved and nothing yet says *when* or *watch for it*.

### 6.2 `16-coming-from-r-python.md` is where the defects belong

It is where a reader arriving from `changepoint`, `ruptures`, `forecast::croston` or `sktime` will look, and Stage 12 contributes four defects that change behaviour: **D19** (changepoint's default is suboptimal), **D20** (ruptures' `jump=5`), **D21** (sktime's lag-0 leak), **D24** (SBA's hardcoded 0.95). Each gets a line with the measured evidence.

### 6.3 The Indian worked example

Each chapter ends on Indian data, as Parts X and XI established. `iip_india.csv`, `cpi.csv`, `imports.csv`, `exports.csv` are in the uploads. [U — not re-read this session; confirm the columns before writing.]

| Chapter | Example |
|---|---|
| 21 | — *(no intermittent Indian series is to hand; use a generated one and say so, or source one)* |
| 22 | IIP around 2020: a level break with a confidence interval, and the CUSUM that would have flagged it |
| 23 | IIP with CPI as a past-only regressor and Fourier terms as known-future |

**Chapter 21 is the honest gap.** Do not manufacture an "Indian intermittent series" from aggregate data that is not intermittent.

---

## 7. The book — four chapters across two Parts

### 7.1 The structural recommendation

Stage 12's three topics are not one theme, and forcing them into one Part would be artificial. Two moves:

**Move 1 — change-point and breaks join Part XI.** Part XI is "Series That Change Their Mind", with chapters on unobserved components, dynamic factors, Markov switching and thresholds. **A structural break is exactly a series changing its mind**, and the pairing is substantive: Markov switching has the regime unobserved and recurring, a threshold has it observed and recurring, a structural break has it unobserved and *permanent*. Appending two chapters there puts the three side by side, which is the comparison the reader needs.

**Move 2 — intermittent demand and the ML reduction form a new Part XII**, with the Frontier moving to XIII.

| Part | # | Chapter | Charts |
|---|---|---|---|
| **XI** (append) | 57 | **A change that does not come back** — breaks, dating, confidence intervals, the known-versus-unknown distinction | 9 |
| **XI** (append) | 58 | **Watching a model drift** — CUSUM and friends as monitoring rather than testing | 7 |
| **XII** (new) | 59 | **Mostly nothing, then something** — intermittent demand | 9 |
| **XII** (new) | 60 | **Any regressor you like** — the lag-feature reduction, and why it is already here | 7 |
| **XIII** | 61 | **The Frontier** (was 57) | 2 |

**32 new book charts.**

The alternative — one new Part XII for all three topics — is cheaper by one renumbering and worse by one wrong adjacency. Reading order won in Stage 11 and should win again, but this is a judgement call and the implementation may reasonably disagree; the reasons are above.

### 7.2 Chapter 58 earns its place

A whole chapter on monitoring may look indulgent next to one on breaks. It is not, for a reason the corpus makes concrete: **Montgomery's published CUSUM example is an in-control case** that detects nothing, and his §2.6.3 never frames the question as a hypothesis test. The distinction between *"did the parameters change?"* (a test, with a null and a p-value) and *"has my model stopped working?"* (a signal, with decision limits and an ARL) is real, and conflating them is the commonest mistake in applied forecasting. The chapter exists to separate them.

It is also where `nyblom_test` — already built, Stage 7 — finally gets a companion.

### 7.3 Source map

| Ch | Hamilton | Montgomery | Tsay | fpp | Cowpertwait | Shumway |
|---|---|---|---|---|---|---|
| 57 breaks | **§14.3 (424–426), §15.5 (449–451), Fig. 15.4** | **§6.7, Table 6.13 (337)** | §5.8 (274–276), §4.2.2 (211–213) | p. 103 | — | *(cited as not covering it)* |
| 58 monitoring | — | **§2.6.3 (60–66), Figs. 2.33–2.35** | — | — | Ex. 3.8 (66) | — |
| 59 intermittent | — | — | — | — | — | §6.9 *(as the adjacent count model)* |
| 60 reduction | — | §7.7 (372–375) | **§4.1.9 (199–205), Figs. 4.8–4.10** | **Ch. 13 (131–133)** | — | §4.8, §5.5 *(classical distributed lags, as the contrast)* |
| 61 Frontier | Ch. 13 | §2.6.3 | Ch. 11 | Ch. 9 | Ch. 12 | Ch. 6 |

**Chapter 59 cites no book for its own subject, and says so in its first paragraph.** That is the honest version, and §1.2 is why.

### 7.4 Figures worth redrawing — all from our own fits, never reproduced

| Source | What it shows | Our version |
|---|---|---|
| **Hamilton Fig. 15.4** (printed 451) | log real GNP 1952–84 with a piecewise-shifted trend overlaid, from Lam (1990) | Ch. 57 — a series with break-adjusted trend |
| **Montgomery Fig. 6.17** (printed 335) | weekly cereal sales with "Week 88" annotated | Ch. 57 — the known-date intervention |
| **Montgomery Figs. 2.33–2.35** (printed 62–65) | individuals, CUSUM and EWMA charts with limits | Ch. 58 — all three, on forecast errors |
| **Tsay Fig. 4.9** (printed 205) | one-step-ahead probability forecasts with actual directions overlaid | Ch. 60 |
| **Tsay Fig. 4.8** (printed 200) | the 2–3–1 architecture diagram | Ch. 60 — **as inline SVG**, a mechanism diagram not a data chart |
| **Montgomery Fig. 2.16(a)/(b)** (printed 35) | sunspots raw versus log | Ch. 58's lead-in on variance stabilisation |

### 7.5 Rewriting the Frontier

Four entries come out of its "Smaller, specific gaps" table: intermittent demand, change-point detection, and — from the regimes table — the structural-break and CUSUM tests. `forecast` for `model=:tvss` and the missing-data policy stay.

**Add what Stage 12 deliberately did not build**, with the measured reasons: the supF/aveF/expF p-values (charter §6 — a licensing problem, not a mathematical one), MOSUM, DirRec and multi-output (provably identical to Direct under OLS), conformal intervals, and prediction intervals for Croston (no reference implementation exists in either package).

---

## 8. Verification obligations

1. **Every prose number comes from an executed `@example` block on the same page**, or is a cited published figure with printed and PDF page numbers.
2. **Any number from Hamilton or Montgomery must be read from a rasterized page.** Both are OCR scans; `pdftotext` silently dropped `UCL=8.91` from Montgomery's Figure 2.33, and Hamilton's mangles subscripts and Greek. Montgomery additionally has a Portuguese annotation layer that **overwrites page text in the extracted layer** on at least PDF 75 and 78. [V]
3. **Five errata are now on record across the corpus. Do not reproduce them:**
   - Hamilton's self-contradictory break indexing between §14.3 and §15.5 (§2.1)
   - Montgomery's H = 14.36 in the text against 14.38 in Figure 2.34 (§3.3)
   - Tsay Eq. (4.9)'s minus signs on the lag terms *(from the Stage 11 survey)*
   - Tsay Eq. (4.53)'s regime-1 variance printed as 0.76 where 0.076 is meant *(Stage 11)*
   - fpp p. 101's swapped legend labels *(Stage 11)*
4. **The Cowpertwait offset table is corrected**: Chapter 3 is **+12**, not +13.
5. **The supF citation debt is now documented rather than assumed.** §1.3 — no table exists in the corpus, so Andrews (1993) is cited directly. Record it in `docs/src/introduction/B-verification.md` alongside the Durbin & Koopman entry Stage 11 added.

---

## 9. Budget

| Surface | Items | Charts | Lines |
|---|---|---|---|
| API | 2 new + 2 updated | 7 | 500–700 |
| Manual | 3 new + 2 updated | 15 | 1,900–2,400 |
| Book | 4 new + 1 rewritten | 32 | 4,500–5,800 |
| **total** | | **54** | **6,900–8,900** |

Smaller than Stage 11's 66 charts and 9,100–11,400 lines, which is right: three narrower topics, and one of them mostly documents capability that already exists.

---

## 10. What not to do

1. **Do not cite any of the six books for intermittent demand.** §1.2 — none covers it, including fpp.
2. **Do not cite "Hamilton's break convention" without naming the section.** §2.1.
3. **Do not use Montgomery's 14.38.** §3.3.
4. **Do not present Montgomery's CUSUM example as a detection success.** It is in-control and detects nothing.
5. **Do not claim any of the six supplies change-point critical values.** §1.3.
6. **Do not add an API page for 12.3.** §5.2 — it extends `arx`.
7. **Do not manufacture an Indian intermittent series.** §6.3.
8. **Do not take a number from Hamilton or Montgomery without rasterizing.** §8.2.
9. **Do not reproduce any figure from the six books.** Redraw from our own fits.
10. **Do not conflate monitoring with testing.** §7.2 — it is why chapter 58 exists.

---

## 11. Corrections found while implementing (appended 2026-10-15)

1. **§3.3 is reversed: Montgomery's Figure 2.34 (14.38) is right and the text's 14.36 is the rounding artefact.** Table 2.3 (printed p. 54, PDF 67) prints the 50 errors in full; transcribed, they reproduce every limit the book publishes for them -- X̄ 0.282, individuals UCL 8.909 / LCL −8.345, MR UCL 10.597 about MR̄ 3.2437, EWMA steady state 2.2611 / −1.6971 -- and give `H = 5σ̂ = 14.3776`. The text's `5(0.8865)(3.24) = 14.36` rounds MR̄ to 3.24 before multiplying. Chapter 58 uses 14.38 and says why. [V] fixture `test_data/montgomery_forecast_errors.csv`.
2. **§3.1's Table 6.13 cannot be reproduced from Table 6.12.** The data (transcribed and checked twice against a 250-dpi raster; the text layer misaligns the last column by two rows) under the stated model `y = ω₀S + (1 − θB)/(1 − B)ε` give, in R's `arima`, ML `θ = 0.5211, ω₀ = −4205.5` (se 8472) and CSS `0.5257, −4277.3`; the package agrees. Neither the step in growth (`0.5621, −2367.4`, se 1094) nor either form with drift (closest: growth + drift, `θ = 0.7307`, `ω₀ = −3547.6`, se 759, variance 84.4M) gives SAS's `0.75886, −2071.5`, se 637.4, variance 82,507,397. A standard error near 637 is only consistent with the step acting on the differences. Documented in chapter 57 and Appendix B as a sixth erratum, not used as a target. [V] fixture `test_data/montgomery_cereal.csv`.
3. **Polio is a real intermittent series in the catalogue** (`dataset("polio")`, 168 months, 64 zeros, ADI 1.615, CV² 0.840 -- lumpy). §6.3's "no intermittent series is to hand" holds for Indian data only; chapters 21 and 59 use polio and say the Indian gap remains.
4. **Shumway §6.9 page range**: printed pp. 334–341, PDF 344–351 (offset +10). **Tsay §4.1.9**: printed 203–204 is PDF 230–231 (offset +27).

