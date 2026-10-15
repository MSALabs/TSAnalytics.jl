# Stage 11 — Documentation handoff
## API reference, manual, and a new book Part

**Written:** 2026-10-15
**Repository HEAD at writing:** `ad0010e`
**Prerequisite reading:** `stage-11-overall-handoff.md` and the five sub-stage
handoffs. This document assumes the API surface they define.

**Scope:** the three documentation surfaces for Stage 11 —
API reference pages, manual chapters, and a new Part of the book — all with
charts, and all sourced against the **six textbooks** in `/mnt/user-data/uploads/`.

---

## 0. What exists now

Measured at HEAD. [V]

| Surface | Current | Stage 11 adds |
|---|---|---|
| API pages (`docs/src/api/`) | **13** | **4** |
| Manual chapters (`docs/src/manual/`) | **16** | **4** |
| Book chapters (`docs/src/introduction/`) | **52** in **11 Parts** (+3 appendices) | **6** in **1 new Part**, plus a rewrite of chapter 52 |

Existing API pages: `arimax`, `arma-models`, `cointegration`, `decomposition`,
`diagnostics`, `exponential-smoothing`, `forecasting`, `garch`, `plotting`,
`primitives`, `state-space`, `structural`, `var`.

**Note `structural.md` is already taken — it is Structural VAR** (its first line
is "# API: Structural VAR") [V]. §5.1 deals with the collision this creates.

---

## 1. The six sources — what each can and cannot be cited for

All six were read in this session. Page offsets verified; **record them, because
every one is different and three of the six are non-obvious.**

| Book | PDF pages | Printed → PDF offset | Text layer |
|---|---|---|---|
| Hamilton, *Time Series Analysis* (1994) | 814 | **+15** (constant) | **OCR scan — equations unreliable.** Rasterize before trusting any number |
| Shumway & Stoffer, 4th ed. (2017) | 568 | **+10** (constant) | clean, native pdfTeX |
| Tsay, *Analysis of Financial Time Series*, 3rd ed. (2010) | 714 | **+27** (constant) | clean; display math loses some glyphs |
| Montgomery, Jennings & Kulahci (2008) | 469 | **+13** (constant) | **OCR scan with a stray Portuguese annotation layer — equations mangled** |
| Cowpertwait & Metcalfe, *Introductory Time Series with R* (2009) | 262 | **drifts +14 → +8** | clean |
| Hyndman & Athanasopoulos, *fpp* notes (2014) | 138 | **0** — printed = PDF | clean |

**Cowpertwait's offset is not constant.** Springer blank verso pages are omitted
from the PDF, so the book loses roughly one PDF page per chapter boundary: +14
for Ch. 1–2, +13 Ch. 3, +12 Ch. 4–5, +11 Ch. 6–7, +10 Ch. 8–9, +9 Ch. 10, +8
Ch. 11–12. For Chapter 12, printed = PDF − 8. [V]

### 1.1 Coverage matrix — which book covers which Stage 11 topic

| | 11.1 UC | 11.2 DFM | 11.3 Markov-switching | 11.4 TAR/SETAR | 11.4 STAR/LSTAR | Kalman filter |
|---|---|---|---|---|---|---|
| **Hamilton** | **none** | formulation only (§13.1 p.376) | **Ch. 22, canonical** | **none** | **none** | Ch. 13 |
| **Shumway & Stoffer** | §6.5 + Ex. 6.4 | Ex. 6.2 / 6.7 | §6.9–6.10 | **§5.4** | **none** | **Ch. 6, primary** |
| **Tsay** | **§11.3.5, best named treatment** | **none** (Ch. 9 is *static*) | §4.1.4, §12.9 | **§4.1.2, primary** | **§4.1.3, primary** | Ch. 11 |
| **Montgomery** | none | none | none | none | none | **none** — the word never appears |
| **Cowpertwait** | partial, unnamed (§12.3.1) | none | none | none | none | Ch. 12 (Bayesian DLM) |
| **fpp** | **Ch. 9, best pedagogy** | none | none | none | none | Ch. 9 |

**Four hard gaps across all six books**, each of which the chapter must either
source elsewhere or state as unsourced:

1. **No dynamic factor model with estimation anywhere.** Hamilton §13.1 gives the
   Stock & Watson (1991) *formulation* with no data and no numbers; Shumway &
   Stoffer Example 6.2/6.7 is a two-series one-factor instance with printed
   estimates but is not presented as a DFM; **Tsay has zero occurrences of
   "dynamic factor"** and his Chapter 9 is explicitly the *static,
   cross-sectional* kind — he states on printed p. 469 that the factors are
   assumed serially uncorrelated. **Chapter 11.2's book chapter must say this in
   one line**, because "Tsay has a factor-models chapter" is the obvious
   misreading and it is wrong.
2. **No STAR/LSTAR outside Tsay.** Shumway & Stoffer cover SETAR but not smooth
   transition; Hamilton covers neither.
3. **No exact diffuse initialisation and no concentrated state-space likelihood**
   in Hamilton or Shumway & Stoffer.
4. **No sequential (univariate) treatment of a multivariate observation vector in
   any of the six.** §1.3.

### 1.2 Per-book citation licence — what each may be cited *for*

| Book | Cite it for | Citing it here would be a stretch |
|---|---|---|
| **Hamilton** | Markov-switching: Ch. 22 §22.4 for the filter, the likelihood, the Kim (1993) smoother in closed form, the ergodic initialisation, the EM transition update; **Table 22.1 (printed 698) for published estimates**; the stationary initialisation `vec(P_{1\|0}) = [I_{r²} − (F⊗F)]⁻¹vec(Q)` (printed 378) | UC, threshold, or anything with code — he predates R, and Ch. 13 has **no tables, no figures and no empirical application** |
| **Shumway & Stoffer** | The Kalman filter machinery (Properties 6.1–6.3, printed 293/297/300); EM; missing data §6.4; **SETAR §5.4 printed 261–265** with a fully specified `flu` fit; the reproducible `set.seed(999)` fixture | Smooth transition (absent); local *linear* trend (their §6.5 trend is multiplicative with φ>1, not a stochastic slope) |
| **Tsay** | **Threshold and smooth transition, §4.1.2–4.1.3** — and decisively, see §4.2; the TAR-F test §4.2.2; **UC under its own name, §11.3.5 + Table 11.4**; the `(T,Z,R,Q,H)` system-matrix notation | Dynamic factor models — his Ch. 9 is static and he says so |
| **Montgomery** | **The practitioner register only**: §4.8 (printed 217–220) for the exponential-smoother ≡ IMA(1,1) identity `θ = 1−λ`; §2.4.2 + Example 2.6 for classical decomposition with a four-panel residual workflow; **§2.6.3 (printed 60–66) for forecast monitoring and tracking signals**, which no other book covers | **Any Stage 11 topic.** It has none of them, and no Kalman filter |
| **Cowpertwait** | The Bayesian DLM derivation of the filter (printed 230–233); **Figs. 12.1 and 12.2** — the best filter-versus-smoother-across-a-level-shift picture available; mid-series intervention by inflating W; the point that the *ratio* of the variances drives the filter | **`StructTS` — the book never uses it.** Also no stochastic seasonal, no cycle, no MLE of variances, and its Ch. 12 is built entirely on `sspir`, archived from CRAN |
| **fpp** | **The best pedagogy for UC**: Ch. 9 §9.1 definitions of local level / local linear trend / BSM / trigonometric (printed 94–95); the ETS↔structural correspondence; **the Gaussian state-space log-likelihood formula (printed 99)**; **the diffuse-initialisation convention (printed 98)**; `StructTS`/`tsSmooth` syntax | DFM, Markov-switching, threshold — covers none; and any claim about `StructTS` *numerics*, since it prints no `StructTS` output at all |

### 1.3 Two corrections to assumptions this project was working from

**Correction 1 — Cowpertwait & Metcalfe do not use `StructTS`.** The string does
not appear anywhere in the 262 pages. Their Chapter 12 is built on the `sspir`
package (`SS`, `kfilter`, `smoother`) with user-specified V and W matrices, and
they **never estimate variances by maximum likelihood** — they say so twice
(printed 238 and 243). **`StructTS` is in fpp, Chapter 9.** [V]

Consequence: Cowpertwait contributes nothing to the three `StructTS` defects
(charter D1–D3), because their framework cannot pose any of the three questions.

**Correction 2 — the sequential univariate filter cannot be cited to any of the
six.** 11.0 §6.2 chose the Durbin & Koopman §6.4 sequential treatment, and the
charter's Appendix A flagged that this project's D&K citations rest on training
memory. **That flag is now worse, not better:** Hamilton forms and inverts the
n×n matrix (eqs [13.2.15], [13.2.19], [13.2.22], printed 379–381); Shumway &
Stoffer form and invert the q×q matrix (eq 6.22, Property 6.1, printed 293);
Hamilton's author index lists "Durbin, James, 226" which is the **Durbin–Watson
statistic**, and "Koopmans, T. C., 638" which is **Tjalling Koopmans**, a
different person. Shumway & Stoffer mention Durbin & Koopman (2001) once, as a
reading pointer (printed 288), and take no method from it. [V]

The nearest adjacent material is Shumway & Stoffer **§6.4 Missing Data
Modifications** (printed 310–315), which partitions the observation vector and
runs the filter at reduced dimension — but still **one joint inverse per
period**. It is a dimension-reduction trick for missingness, not sequential
processing.

**So 11.0's algorithm must cite Durbin & Koopman (2001) directly**, and that
edition is still not in hand. The 11.0 handoff already writes the recursion out
in full so the code does not depend on the citation; this confirms that was the
right call, and the obligation to verify the section number against a physical
copy stands.

---

## 2. Two published cross-checks found by reading, both verified here

These are the substantive gain from the six books, and both are new.

### 2.1 Hamilton's own Table 22.1 corroborates nine of ten Markov-switching targets

**Printed p. 698 = PDF p. 713. Verified by rasterizing the page and reading the
image** — Hamilton's text layer is OCR and cannot be trusted for numbers. [V]

| Project target (statsmodels) | Hamilton Table 22.1 | agreement |
|---|---|---|
| `const[1]` = +1.1635220287 | **μ̂₁ = 1.16** (0.07) | ✓ |
| `const[0]` = −0.3588026189 | **μ̂₂ = −0.36** (0.26) | ✓ |
| `p[1->1]` = 0.90408495 | **p̂*₁₁ = 0.90** (0.04) | ✓ |
| `p[0->0]` = 0.7546643177 | **p̂*₂₂ = 0.75** (0.10) | ✓ |
| `sigma2` = 0.5913638324 | **σ̂² = 0.59** (0.10) | ✓ |
| `ar.L1` = 0.0134798930 | **φ̂₁ = 0.01** (0.12) | ✓ |
| `ar.L2` = −0.0575297622 | **φ̂₂ = −0.06** (0.14) | ✓ |
| `ar.L3` = −0.2469918252 | **φ̂₃ = −0.25** (0.11) | ✓ |
| `ar.L4` = −0.2129275902 | **φ̂₄ = −0.21** (0.11) | ✓ |

All nine match to the two decimals Hamilton prints. Combined with statsmodels'
own E-views expectation (`test_markov_autoregression.py:502`), **the Stage 11.3
target now has three independent corroborations, one of them from the model's
originator in print.** That is the strongest provenance anything in this project
has.

**Three caveats to record with it:**

- **No log-likelihood is printed.** `−181.2633942812` gets no corroboration here.
- **The regime labels are flipped.** The project's regime 0 is Hamilton's state 2
  (contraction). Chapter prose must be explicit about which is which.
- **The sample period is not stated in the text** — only "1952–84" in the caption
  of Figure 22.4(b). Cite Hamilton (1989) *Econometrica* 57:357–84 for the
  1951:II–1984:IV sample, not the book.

Hamilton also describes the **32 = 2⁵ composite-state expansion** for the AR(4)
switching model at printed 698 and the summing of even-numbered rows to recover
`P{s*_t = 2}` [V] — independent confirmation of the `k^(p+1)` striding that
11.3 §4.1 specifies.

### 2.2 Tsay's published unobserved-components fit reproduces — a new fixture

Tsay Example 11.3 (printed 606) fits `y_t = μ_t + γ_t + e_t` to **log quarterly
J&J earnings per share**, with a local level (no slope) and a dummy seasonal at
s = 4, in **S-PLUS with SsfPack**. The dataset is R's built-in
`JohnsonJohnson`, so it is freely reproducible.

Tested in this session against statsmodels [V] `s11/tsay_jj.py`:

| parameter | statsmodels | Tsay (printed 606) | relative difference |
|---|---|---|---|
| σ̂_η (level) | 7.269789203e-02 | **7.269655e-02** | **1.85e-05** |
| σ̂_ω (seasonal) | 2.931638067e-02 | **2.931691e-02** | **1.81e-05** |
| σ̂_e (irregular) | 1.279361226e-05 | 2.044516e-06 | **6.3×** |

**Two of three agree to 1.8e-05 across two entirely different software stacks
sixteen years apart. The third is a boundary case** — both are effectively zero,
and they differ by a factor of six, which is exactly the behaviour the charter's
§5.5 predicted for a variance at its boundary.

**This is worth more than the agreement alone**: it is an independent, published
confirmation of the charter's tiered test policy. Non-boundary parameters
reproduce tightly across implementations; boundary parameters do not, and
asserting them would be asserting an optimiser's path.

**Add `jj_log.csv` to the fixture set** (n = 84, `sum = 92.7729171041`, from
`log(datasets::JohnsonJohnson)`). The 11.1 handoff's §7.2 tier-1 table gains a
row, and the item gains a third reference it did not have.

Tsay's **Example 11.1** gives a second, purely local-level check: Alcoa
intradaily realised volatility, 340 observations, **σ̂_η = 0.0735, σ̂_e = 0.4803**,
with the ARIMA(0,1,1) identity (θ̂ = 0.858, σ̂_a = 0.5184) agreeing to three
decimal places (printed 559–560). The data is not bundled with R, so it is a
citation rather than a fixture.

---

## 3. Notation — the conflict is real and the chapter must declare a convention

Four of the six books use mutually incompatible notation for the same objects,
and two of the collisions are silent.

| Object | Hamilton | Shumway & Stoffer | Tsay | fpp | Cowpertwait |
|---|---|---|---|---|---|
| State vector | **ξ_t** | **x_t** | **s_t** | **x_t** | **θ_t** |
| Transition | **F** | **Φ** | **T_t** | **G** | **G_t** |
| Design | **H′** | **A_t** | **Z_t** | **f′** | **F′_t** |
| State noise cov | Q | Q | Q_t | W | W_t |
| Obs noise cov | **R** | **R** | **H_t** | σ² | V_t |
| Exogenous regressors | **x_t** | u_t | — | — | — |
| State equation timing | `ξ_{t+1} = Fξ_t + …` | `x_t = Φx_{t−1} + …` | `s_{t+1} = T_t s_t + …` | `x_t = Gx_{t−1} + …` | `θ_t = G_tθ_{t−1} + …` |

**The two silent traps:**

1. **`x` is the state in Shumway & Stoffer and fpp, but the exogenous regressors
   in Hamilton.** `A` is the design matrix in Shumway & Stoffer and the
   exogenous-coefficient matrix in Hamilton.
2. **`v` and `w` are swapped** between Hamilton and Shumway & Stoffer — Hamilton's
   `w` is the observation noise, theirs is the state noise. **And `H` is the
   design matrix in Hamilton but the observation covariance in Tsay.**

Plus a timing offset: Hamilton and Tsay index the state equation forward
(`s_{t+1} = …`), Shumway & Stoffer, fpp and Cowpertwait contemporaneously.

**And the transition matrices are transposes of each other.** Hamilton's `P` is
**column-stochastic** — he writes at printed 678 that the row-j column-i element
is `p_ij`, so columns sum to one — while Shumway & Stoffer's is **row-stochastic**
(Property 6.7). `P_Hamilton = P_S&S′`. [V]

### Decision

**Adopt Tsay's `(T, Z, R, Q, H)` system-matrix notation throughout**, because:

- it is already what `TimeVaryingSSM` uses (`T`, `Z`, `R`, `Q`, `H` are the
  literal field names at `src/statespace/timevaryingssm.jl:49`) [S], so the docs
  match the code with no translation;
- it is Durbin & Koopman's notation, which is what `KFAS` and the wider
  state-space literature use;
- Tsay states it explicitly at Eqs. (11.26)–(11.27), printed 576–577, so it is
  citable.

**For the transition matrix, keep the package's left-stochastic convention**
(`P[i,j] = Pr[S_t = i | S_{t−1} = j]`, columns summing to one — which is
statsmodels' and Hamilton's), and **say so in the first paragraph of the
Markov-switching chapter**, with a one-line note that Shumway & Stoffer
transpose it. A transposed `P` gives plausible-looking nonsense, which is the
worst kind of error.

**Every chapter that cites more than one of the six must open with a notation
note.** This is not optional politeness; silently mixing Hamilton and Shumway &
Stoffer produces code that is wrong in ways that still run.

---

## 4. Two findings that change what the chapters say

### 4.1 fpp supplies the citation that makes `StructTS` defect D1 provable

The charter's defect D1 says R's `StructTS$loglik` is not a log-likelihood,
established by arithmetic: `loglik0 − loglik = n·log(2π) + 0.5·n_obs·log(2π)`
exactly, and a **positive** `n·log(2π)` cannot be a Gaussian normalising constant.

**fpp printed p. 99 prints the correct formula:**

```
log L = −(T/2)·log(2π) − (1/2)·Σ_t log v̂_{t|t−1} − (1/2)·Σ_t e_t²/v̂_{t|t−1}
```

[V] The `log(2π)` coefficient is **−T/2**, negative. So the chapter can state the
defect against a published formula — from Hyndman, who maintains the `forecast`
package — rather than against a derivation of our own.

Likewise **fpp printed p. 98 states the diffuse-initialisation convention**
(`x_{1|0} = 0`, `P_{1|0} = kI` for large k), which makes `StructTS`'s
initialisation-at-the-first-observation (defect D2) a documented *deviation* from
standard practice rather than an unsupported claim. And **fpp printed 94–95**
treats σ² = 0 as a *substantive, interpretable* boundary — deterministic
seasonality, deterministic trend — which is exactly why clipping at zero (defect
D3) is severe: the one value the parameterisation lets you read off as a
structural statement is the one a clipped negative is indistinguishable from.

**Three defects, three published citations. Use them.**

### 4.2 Tsay states the additive STAR form *and* its translation — the factor-of-67 trap

The 11.4 handoff's §1.2 records that LSTAR is **additive**, not a convex
combination, and that using the convex form gives SSR 290.77 against the correct
4.337643232152 — wrong by 67×.

**Tsay Eq. (4.15), printed 184, states the additive form** —
`x_t = c₀ + Σφ_{0,i}x_{t−i} + F((x_{t−d} − Δ)/s)·(c₁ + Σφ_{1,i}x_{t−i})` — and
then, **uniquely among the six, prints the translation on the next page**: the
high-regime conditional mean is `μ_{2t} = (c₀+c₁) + Σ(φ_{0,i}+φ_{1,i})x_{t−i}`,
i.e. the second bracket is an **increment**. [V]

That is the exact bridge the bug turned on, in print, in a standard text. **The
book chapter should quote the structure of both forms and cite Tsay printed
184–185 for the translation.**

Three caveats to carry with the citation:

- **Tsay's scale `s` is the reciprocal of the usual slope γ** — he writes
  `(x_{t−d} − Δ)/s` where `tsDyn` and Teräsvirta write `γ(z − th)`. So `s ≈ 1/γ`.
  A second reparameterisation hazard, independent of the additive/convex one.
- **His threshold symbol changes three times**: `γ_j` in Eq. (4.9), `r₁` in
  Eq. (4.46), `γ₁,γ₂` in Eq. (8.44).
- **His regime boundary convention flips between chapters**: Eq. (4.9) is
  `γ_{j−1} ≤ x_{t−d} < γ_j` (lower-inclusive); Eq. (8.44) is upper-inclusive.
  The package uses `z ≤ γ` for the low regime (11.4 §4.2), matching Eq. (4.9).

**And one erratum**: Tsay Eq. (4.9) as printed shows **minus signs** before the
lag terms, which contradicts every other SETAR equation in the book (4.8, 4.11,
4.46, 4.53). Confirmed on the rendered page, so it is a typesetting error, not an
extraction artefact. **Do not reproduce Eq. (4.9)'s signs.**

---

## 5. API reference pages — four new

Each follows the established pattern: a short framing paragraph, a link to the
corresponding manual chapter, one `@example` block producing a chart, then
`@docs` blocks. `checkdocs=:exports` means **the build fails if an exported name
has no docstring** [S] `docs/make.jl`, so every new export must appear.

### 5.1 The naming collision, and the resolution

`api/structural.md` is **Structural VAR**. "Structural time series" is the other
standard name for unobserved components, and Tsay uses exactly that phrase
(§11.3.5, "structural time series model (STSM)").

**Resolution: name the page `unobserved-components.md`, and never use the word
"structural" for it in a page title, heading, or cross-reference.** Mention the
synonym once, in the body, with an explicit disambiguation:

> These are also called *structural time series models* in the literature (Tsay,
> §11.3.5; Harvey 1989). That is unrelated to the *structural VAR* of
> [API: Structural VAR](structural.md), which identifies shocks in a
> multivariate system. The clash of names is unfortunate and predates both.

The same disambiguation belongs in the manual chapter and in the book chapter,
because the collision will mislead readers coming from either direction.

### 5.2 The four pages

| Page | Covers | Charts |
|---|---|---|
| `api/unobserved-components.md` | `fit_uc`, `UCModel`, `uc_decomposition`, `UCDecomposition`, the five convenience constructors, `state_decomposition(::UCModel)` | **1** — component panels with bands, from `jj_log` |
| `api/dynamic-factor.md` | `fit_dfm`, `DynamicFactorModel`, `factor_paths`, `rotate_factors`, `factor_r2`, `series_r2`, `factor_selection` | **2** — factor paths with bands; loadings heatmap |
| `api/regime-switching.md` | `fit_ms_regression`, `fit_ms_autoregression`, `MarkovSwitchingModel`, `regime_probabilities`, `canonicalise`, `expected_durations` | **1** — regime-shaded series, `hamilton_rgnp` |
| `api/threshold.md` | `fit_setar`, `fit_lstar`, `select_setar`, `setar_test`, `SETARModel`, `LSTARModel`, `SETARSelection`, `ThresholdTest`, `transition_function` | **2** — regime-coloured series; SSR profile |

**Six new charts in the API reference**, against the current six across all
thirteen pages [V] — so this doubles the API's chart count, which is appropriate
since Stage 11's outputs are mostly things you look at.

### 5.3 The `state-space.md` update

`api/state-space.md` documents the `GaussianSSM`/`TimeVaryingSSM` engine and
says it is "exported as public API" [S]. Stage 11.0 adds multivariate-observation
methods and `SSMInit` to that engine.

**Update it rather than adding a page.** Add the multivariate `kalman_filter` and
`kalman_smoother` methods, `SSMInit`, and a short paragraph on the sequential
treatment — including the honest note that it is cited to Durbin & Koopman
(2001) directly, since none of the six books state it (§1.3).

---

## 6. Manual chapters — four new, numbered 17–20

The manual runs `01-primitives` through `16-coming-from-r-python`. **Append as
17–20 rather than inserting**, for the opposite reason to the Stage 10 decision:
there, inserting at 11–14 bought correct reading order within a coherent
multivariate block. Here the four new chapters are *independent tracks*, not a
block that belongs before plotting, and `15-plotting` / `16-coming-from-r-python`
are deliberately terminal — a plotting chapter that precedes four chapters with
new plots, and an R/Python comparison that precedes four new comparisons, would
both be wrong.

**But both terminal chapters need updating**, and that is the cost of appending:

- `15-plotting.md` gains the thirteen new recipes
- `16-coming-from-r-python.md` gains the Stage 11 equivalences — and this is the
  chapter where the reference defects belong, because it is where a reader
  arriving from `StructTS` or `tsDyn` will look

| Chapter | Title | Charts | Primary sources |
|---|---|---|---|
| `17-unobserved-components.md` | Unobserved Components | **5** | fpp Ch. 9 (pedagogy), Tsay §11.3.5 + Table 11.4 (naming), Montgomery §4.8 (the smoothing bridge) |
| `18-dynamic-factor-models.md` | Dynamic Factor Models | **4** | Shumway & Stoffer Ex. 6.2/6.7, Hamilton §13.1 (formulation) |
| `19-regime-switching.md` | Regime Switching | **4** | Hamilton Ch. 22, Shumway & Stoffer §6.9–6.10 |
| `20-threshold-models.md` | Threshold and Smooth-Transition Models | **5** | Tsay §4.1.2–4.1.3, §4.2.2; Shumway & Stoffer §5.4 |

**18 new manual charts**, against the current 231 `@example` blocks across the
manual [V].

### 6.1 The register — fpp is the model, and Montgomery supplies what it lacks

fpp Chapter 9's format is exactly right for a manual page: the model in three to
five equations, a bullet list of what the parameters mean **and what the
degenerate cases mean** (`σ_ζ² = σ_ξ² = 0 ⟹ y_t = ℓ_0 + t·b_0 + ε_t`), the exact
call, the plot. Compress to that shape.

**Montgomery supplies the one thing no other book does**: §2.6.3 (printed 60–66)
on monitoring a fitted model — tracking signals, when a model has drifted,
what to do about it. Nothing in Hamilton, Tsay or Shumway & Stoffer covers it.
`17-unobserved-components.md` should carry a short "is this model still right?"
section citing it, because a UC model fitted once and left running is the exact
case it addresses.

### 6.2 The manual's own cross-check obligation

Each manual chapter ends with a worked example on **Indian data** — the Part X
chapters established this, and `global_economy` plus the uploaded `iip_india.csv`,
`cpi.csv`, `imports.csv`, `exports.csv` cover it. [U — the uploads were not
re-read this session; confirm the columns before writing.]

The natural fits:

| Chapter | Indian example |
|---|---|
| 17 UC | IIP decomposed into level, stochastic seasonal and irregular — the direct contrast with `stl_decompose` |
| 18 DFM | CPI, IIP, imports, exports driven by one common factor |
| 19 Regime switching | IIP growth, two regimes |
| 20 Threshold | IIP growth with a threshold on its own lag |

---

## 7. The book — a new Part, and the Frontier moves

### 7.1 The structural decision

The book runs Part I through **Part XI "The Frontier"**, which is a single
chapter (52) [V]. Chapter 52 currently *names* unobserved components, dynamic
factor models, regimes and threshold models as what the package lacks — four of
its own gap tables are about to become wrong.

**Two options, and the recommendation is the more expensive one.**

- **Append**: new Part XII after the Frontier. Cheap, no renumbering — and
  **wrong**, because a book that says "here is the frontier" and then crosses it
  for six more chapters reads as an error.
- **Insert**: the new Part becomes **XI**, the Frontier becomes **XII**, and
  chapter 52 becomes chapter 58. Costs the renumbering of one chapter plus every
  cross-reference into it, and a rewrite of its gap tables.

**Recommend inserting.** This is the same call as the Stage 10 manual
numbering, where appending was recommended to avoid churn and the implementation
correctly overrode it in favour of reading order. Reading order wins again, and
the cost here is smaller — one chapter moves, not fourteen cross-references.

### 7.2 Part XI — "Series That Change Their Mind"

Six chapters. The title parallels the existing Part naming ("Models That
Remember", "Models That Forget", "Models That Get Turbulent", "Series That Move
Together") and names the thing all four Stage 11 models have in common: the
series' behaviour is not fixed.

| # | Chapter | Covers | Charts |
|---|---|---|---|
| 53 | **The components you cannot see** | UC: level, slope, seasonal, cycle; the contrast with STL; why a fitted decomposition carries intervals and a filtered one does not | 8 |
| 54 | **When a component stops moving** | Variances at the boundary; deterministic versus stochastic components; what σ² = 0 means and why clipping it destroys the reading | 6 |
| 55 | **Many series, few forces** | DFM: the common factor, identification, why the rotation is arbitrary and what we do about it | 8 |
| 56 | **The economy has two moods** | Markov switching: the Hamilton filter, the Kim smoother, regime probabilities, expected durations | 9 |
| 57 | **A different model on each side of a line** | SETAR and LSTAR: the threshold, the grid search, smooth versus abrupt transition | 9 |
| 58 | **The Frontier** (was 52) | Rewritten: four gaps closed, the remaining ones restated | 2 |

**42 new book charts**, against the current 595 `@example` blocks in the
introduction [V].

### 7.3 Chapter-by-chapter source map

This is the "cover all six books" requirement made concrete. Each chapter cites
what the book actually contains — §1.1's coverage matrix is what makes this
honest rather than decorative.

| Ch | Hamilton | Shumway & Stoffer | Tsay | Montgomery | Cowpertwait | fpp |
|---|---|---|---|---|---|---|
| 53 | — | §6.5, Ex. 6.4 (p. 296), Ex. 6.10 J&J | **§11.3.5, Table 11.4, Ex. 11.3** | §4.8 (ES ≡ IMA(1,1)) | Figs. 12.1–12.2 | **Ch. 9 §9.1–9.2, p. 94–96** |
| 54 | — | — | Ex. 11.1 Alcoa | §2.4.2 trend/seasonal | Fig. 12.6 (frozen component) | **p. 94–95 (σ²=0 as a claim)**, p. 98–99 |
| 55 | **§13.1, Stock & Watson (1991)** | **Ex. 6.2/6.7 globtemp** | §9 — *cited as the static contrast* | — | — | — |
| 56 | **Ch. 22, Table 22.1, Fig. 22.4** | §6.9–6.10, Ex. 6.18 flu | §4.1.4, Table 4.1; §12.9 | — | — | — |
| 57 | — | **§5.4, Ex. 5.7 flu SETAR** | **§4.1.2–4.1.3, §4.2.2, §4.5, Table 4.3** | — | — | — |
| 58 | Ch. 13 (stationary init) | Ch. 6 (the filter) | Ch. 11 | §2.6.3 (monitoring) | Ch. 12 | Ch. 9 |

**All six books are cited, and none is stretched.** Montgomery appears in two
chapters and in the manual, for the three things it genuinely offers;
Cowpertwait appears for its two excellent figures and its DLM derivation. The
alternative — forcing all six into every chapter — would mean citing Montgomery
for Markov switching, which it does not contain.

### 7.4 The figures worth redrawing

Identified by reading. All must be **redrawn from our own fits**, never
reproduced — these are copyrighted books.

| Source figure | What it shows | Our version |
|---|---|---|
| fpp printed 96 | 4-panel BSM decomposition of `austourists` | Ch. 53, same shape on `jj_log` and Indian IIP |
| fpp printed 97 | The "Kalman recursions" schematic | Ch. 53 or the manual — **a mechanism diagram, drawn as inline SVG**, not a data chart |
| fpp printed 102 | Seasonally adjusted series via the Kalman smoother | Ch. 53 — the direct contrast with Part IX's X-13 |
| Cowpertwait Fig. 12.1 | Filter versus smoother across a level shift | Ch. 53 — the single best teaching picture for the filter/smoother distinction |
| Cowpertwait Fig. 12.4 | Same data, two variance ratios, visibly different behaviour | Ch. 54 — the signal-to-noise argument |
| Cowpertwait Fig. 12.6 | A component with W = 0.0001 barely moves (axis range 1e-3) | Ch. 54 — what a frozen component looks like |
| **Hamilton Fig. 22.4** | Smoothed recession probability with NBER dates marked | **Ch. 56 — the canonical Markov-switching visual** |
| Shumway & Stoffer Fig. 5.8 | Lag plot `x_t` vs `x_{t−1}` with lowess and the threshold line | **Ch. 57 — the classic SETAR diagnostic** |
| Tsay Fig. 4.1 | Simulated 2-regime TAR(1) with the threshold line | Ch. 57 — trivially reproducible from his Eq. (4.8) |
| Tsay Fig. 11.6 | Smoothed trend and seasonal of log J&J EPS | Ch. 53 — **and we can now match its numbers** (§2.2) |
| Montgomery Fig. 2.16(a)/(b) | Sunspots raw versus log — variance stabilisation | Ch. 57's lead-in, since the sunspot fixture is Hansen's transform of the same series |

### 7.5 The pedagogical device to steal

Cowpertwait's chapter template, stated in their preface and followed rigidly:
**motivate → define → simulate from the model → fit back and recover the known
parameters → fit to real data → exercises.**

The simulate-recover step is the most valuable part and transplants directly.
It is also how you *demonstrate* a reference defect to a reader rather than
asserting it: simulate a local level with a known σ²_level, fit with `StructTS`,
and watch the recovery fail or return a clipped zero. Chapter 54 should do
exactly that.

### 7.6 Rewriting chapter 58 (was 52)

Its "Series that move together: what is still missing" and "Smaller, specific
gaps" tables currently list, among others, dynamic factor models and
intermittent-demand methods. **Four entries come out** (UC, DFM, Markov
switching, TAR/STAR) and the remaining ones stay. **Add what Stage 11
deliberately did not build**, with the measured reasons from the sub-stage
handoffs — EM for dynamic factors, mixed frequency, the `star`/`addRegime`
family, Hansen's threshold confidence set — so the Frontier chapter stays an
honest inventory rather than a wish list.

---

## 8. The `@example` trap — a build-failure check, not a later sweep

The Stage 9/10 documentation work found that Documenter **silently discards
printed output from an `@example` block whose last expression is a plot**: the
page renders the code and the chart, and the prose quotes a number the reader
never saw. 41 blocks were split across 18 introduction chapters and 2 manual
pages to fix it, and a scan of all 838 blocks then found 236 ending in a plot
with **none** containing a print statement. [V, from that audit]

**Stage 11 adds roughly 66 charts across three surfaces.** Every block that
prints a number and then draws a chart must be split **at authoring time**.

**Add the check to CI**, as a grep that fails the build:

> any `@example` block containing a `print`/`println`/`@show` **and** ending in a
> plot call

That is a ten-line script and it makes the trap unrepeatable. Put it in
`docs/make.jl`'s surrounding tooling, not in a reviewer's memory.

---

## 9. Verification obligations for the documentation itself

The project's discipline applies to the docs, not just the code.

1. **Every number quoted in prose must come from an executed `@example` block on
   the same page**, or be a cited published figure with its source. A prose
   figure with no visible provenance is the exact failure the `@example` trap
   produced.
2. **Every citation must carry both printed and PDF page numbers**, because the
   offsets differ per book and one of them drifts (§1).
3. **Any number taken from Hamilton or Montgomery must be read from a rasterized
   page**, not from `pdftotext`. Both are OCR scans; Hamilton's mangles
   subscripts and Greek letters, Montgomery's has a Portuguese annotation layer
   bleeding into extraction. [V]
4. **Four errata were found in the six books.** Do not reproduce them:
   - Tsay Eq. (4.9) prints minus signs before the lag terms, contradicting
     Eqs. (4.8), (4.11), (4.46), (4.53)
   - Tsay Eq. (4.53)'s regime-1 variance is printed as 0.76 where 0.076 is almost
     certainly intended — the linear benchmark is 0.090, so a "quiet" regime with
     eight times the variance is implausible. **Flagged as unverified; do not use
     it as a cross-check.**
   - fpp printed 101 swaps the legend labels: blue is `tsSmooth` (smoothed) but
     labelled "Filtered level", red is `fitted` (filtered) but labelled
     "Smoothed level"
   - Cowpertwait Fig. 12.4's caption says V = 2.2 where the text (printed 239)
     says 2.1; and their §12.8 command summary lists `smooth` where the code
     calls `smoother`
5. **The citation-provenance debt is now explicit.** The charter's Appendix A
   flagged that the Durbin & Koopman and Ladiray & Quenneville citations rest on
   training memory. §1.3 confirms **none of the six books can substitute for the
   D&K citation** that 11.0's filter depends on. Record it in
   `docs/src/introduction/B-verification.md` as a known gap rather than letting
   it pass silently.

---

## 10. Budget

| Surface | Items | Charts | Estimated lines |
|---|---|---|---|
| API pages | 4 new + 1 updated | 6 | 700–900 |
| Manual chapters | 4 new + 2 updated | 18 | 2,400–3,000 |
| Book Part XI | 5 new + 1 rewritten | 42 | 6,000–7,500 |
| CI check for the `@example` trap | 1 script | — | 10–20 |
| **total** | | **66** | **9,100–11,400** |

For scale: the Stage 9 and 10 documentation work took the book from 41 to 52
chapters and the manual from 12 to 16. [V, from that audit]

---

## 11. What not to do

1. **Do not call the UC page or chapter "structural".** §5.1 — `api/structural.md`
   is Structural VAR and the book's chapter 51 is SVAR.
2. **Do not cite Cowpertwait for `StructTS`.** They never use it. §1.3.
3. **Do not cite Tsay Chapter 9 for dynamic factor models.** It is static and
   cross-sectional, and he says so at printed 469. Cite it as the explicit
   contrast instead.
4. **Do not cite Montgomery for any Stage 11 topic.** It contains none of them,
   and the word "Kalman" does not appear in its 469 pages. Cite it for the
   exponential-smoothing bridge, classical decomposition, and model monitoring.
5. **Do not cite any of the six for the sequential univariate filter.** §1.3.
6. **Do not mix notations without a declared convention.** §3.
7. **Do not reproduce the four errata.** §9.4.
8. **Do not quote Hamilton's log-likelihood** — Table 22.1 prints point estimates
   and standard errors only, and no log-likelihood appears anywhere in Chapter 22.
9. **Do not append the new Part after the Frontier.** §7.1.
10. **Do not defer the `@example` split to a later sweep.** §8.
11. **Do not reproduce any figure from the six books.** Redraw from our own fits.
    §7.4.
