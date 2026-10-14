# Stage 11.4 — Threshold autoregression (SETAR) and smooth-transition autoregression (LSTAR)

**Prerequisite reading:** `stage-11-overall-handoff.md`. **This item needs
nothing from 11.0** — it is pure regression and grid search over `_ols`, and is
genuinely independent of the state-space track.

**Difficulty: Moderate–High.** Roughly 45% of it is a thin layer over `_ols` and
`arx.jl`'s existing lag construction; the other 55% is the exact conventions of
the candidate-threshold grid, the `nthresh=2` conditional search, LSTAR's start
grid, and a bootstrap test. The hand-rolled verification (§3) makes the core
unusually safe.

**Roadmap correction.** `development-sequence.md` says "`tsDyn::setar` (R) —
**weak Python support**". That is wrong but nearly right: `aeon 1.6.0` ships
`aeon.forecasting.machine_learning.SETAR` (201 lines) which **agrees with
`tsDyn` to 12 printed digits**, but fixes the delay at 1, selects no AR order by
criterion, and has no inference, no LSTAR and no `nthresh=2`. statsmodels has
**zero** threshold-AR content — a full-tree grep for
`setar|lstar|smooth.transition|threshold.autoregress` returns no hits. [V]

So: **dual-referenced for the SETAR point estimator, single-referenced (`tsDyn`
only) for the search criteria, inference, LSTAR and the bootstrap test.**

---

## 1. The models

### 1.1 SETAR — self-exciting threshold AR

```
y_t = c_L + Σ_{j∈M_L} φ_{L,j} y_{t−j} + ε_t     if z_t ≤ γ
      c_H + Σ_{j∈M_H} φ_{H,j} y_{t−j} + ε_t     if z_t >  γ
```

with `z_t = y_{t−(d+1)}` the threshold variable (note the indexing — §4.2).

Estimated by **conditional least squares**: for each candidate `γ`, fit one
dummy-expanded OLS on `[X·1{z≤γ}, X·1{z>γ}]` and take the minimum SSR. That is
literally all it is, and §3's bit-exact hand-roll is the proof.

### 1.2 LSTAR — logistic smooth-transition AR

```
ŷ_t = X_t' φ_L + (X_t' φ_H) · G(z_t; γ, γ_th)
```

with `G = 1/(1 + exp(−γ(z_t − γ_th)))`.

**The parameterisation is ADDITIVE, not a convex combination.** This is the
single largest trap in the item. [V] `s11/tar/f9_lstar_param.R`: SSR with the
additive form is **4.337643232152**, exactly `deviance(lstar)`; with the convex
form `(1−G)·φ_L + G·φ_H` it is **290.77** — wrong by a factor of 67.

**The printed "High regime" coefficients are increments**, so the high-regime
*levels* are `φ_L + φ_H`. On the lynx fixture those are
`−0.534974408877, 1.669806789842, −0.621041599748`. [V]

Also: **γ multiplies the raw `(z − γ_th)` with no standardisation by `sd(z)`**,
unlike the common Teräsvirta convention `γ/σ_z`. [V] That changes what γ means
numerically and must be documented, because a user comparing to a textbook will
get a different number.

### 1.3 Scope

| In | Out and why |
|---|---|
| `setar` (1 and 2 thresholds) | **`star` / `addRegime` / `testRegime`** — another 420+ reference lines, and its γ pins at the start-grid bound even on the canonical example [V] |
| `selectSETAR` (grid over threshold × delay × orders) | **`KapShinTest`, `BBCTest`** — the package author's own disclaimer: *"should be considered as experimental. No consistency checks with original paper could be made"* (charter defect D15). They cannot serve as references at all |
| `lstar` | `delta.test`, `llar` — side diagnostics, seed-dependent, low value |
| `setarTest` (Hansen 1999 bootstrap) | `MTAR` model variant — cheap to add later, one line in the threshold-variable construction |
| plots | |

---

## 2. Fixtures and exact targets

### 2.1 Fixture A — `log10(datasets::lynx)`, the textbook case

n = 114, `sum = 331.017667873`, `mean = 2.90366375327`.
First 6: 2.429752280002, 2.506505032405, 2.767155866082, 2.940018155008,
3.168792020314, 3.450403086155. sha256 `c8e279fbc88f18437797a26fa85602410453a87975975593c8666a6b1417bd27`.

**A1. `setar(y, m=2, thDelay=1)`** — defaults `trim=0.15`, `include="const"`,
`common="none"`, `nthresh=1`. [V] `s11/tar/f1_setar_lynx.R`, re-verified
independently in this session.

```
th = 3.3100557377508912        (thDelay=1 → z_t = y_{t−2})
rows used = 112,  n_low = 78,  n_high = 34,  k = 7 (6 slopes + 1 threshold)
```

| coefficient | estimate | std. error | t |
|---|---|---|---|
| `const.L` | 0.58843692925212931 | 0.14330661378560067 | 4.10614 |
| `phiL.1` | 1.2642792839182719 | 0.065256287325703141 | 19.37406 |
| `phiL.2` | −0.42842921164134479 | 0.077486946473903501 | −5.52905 |
| `const.H` | 1.1656919479037287 | 0.87660576968756576 | 1.32978 |
| `phiH.1` | 1.5992540700908249 | 0.10896581100312897 | 14.67666 |
| `phiH.2` | −1.0115754904952012 | 0.26501113872650156 | −3.81711 |

```
SSR  = 4.3481912792005826
mse  = 0.038142028764917388        = SSR/114, NOT SSR/112        ← §4.1
AIC  = -358.373987591728           = 114·log(mse) + 2·7
BIC  = -339.22059845296656
MAPE = 5.649%
residual quantiles: −0.57112127220, −0.10943090605, 0.01764138087, 0.11646802185, 0.51626988231
diag(vcov) = 0.02053678555470, 0.00425838303553, 0.00600422687385,
             0.76843767544953, 0.01187354796757, 0.07023090364912, 0.00000000000000
characteristic roots: L = 1.52777874135 (×2), H = 0.994262021909 (×2)
ar_mean: L = 3.58475289886, H = 2.82714380146
predict(n.ahead=5) = 3.3485758177, 2.9490750890, 2.4946750617, 2.4789330144, 2.6537089156
```

**Note the final `diag(vcov)` entry is exactly 0** — that is the threshold, which
`tsDyn` zero-pads (charter defect D13). §5.2.

The high-regime root of 0.994262 triggers a `Possible unit root in the high
regime` warning, which reproduced in this session [V]. Worth emitting an
equivalent.

**A2. `setar(y, m=2, thDelay=0)`**: th = 2.557507201906, coefficients
0.405942732118, 1.245677428926, −0.333928504230, 1.180869464937,
1.547698349232, −0.956274108919; SSR = 4.565530806743.

**A3. Pooled linear AR(2)** — the comparison that motivates the model:
`const = 1.057600456442 (se 0.12083699374)`, `phi.1 = 1.384237711639 (se
0.06333183622)`, `phi.2 = −0.747775720384 (se 0.06338507067)`,
SSR = 5.78258084172, AIC = −333.873712237, k = 3.
SETAR improves SSR by **24.8%** and AIC from −333.87 to −358.37.

**A4. `selectSETAR(y, m=2, thDelay=0:1)`** — default criterion `pooled-AIC`,
**592 fits** (74 thresholds × 2 delays × 2 `mL` × 2 `mH`):

```
bests: thDelay=1, th=3.31005573775, pooled-AIC=-32.07652679559, mL=2, mH=2
top 5: (1,2,2, 3.31005573775, -32.0765267956)
       (1,2,2, 3.32613095671, -30.8942818150)
       (1,2,2, 3.26387267687, -30.3664513768)
       (1,2,2, 3.21431389742, -28.8191039294)
       (1,2,2, 3.21031851983, -28.6478076597)
allTh matrix: 592 × 5
```

**A5. `setar(y, m=2, thDelay=1, nthresh=2)`** — sequential search:

```
th1 = 2.611723308007   th2 = 3.310055737751
const.L 0.572916292592 (0.23852440947)  phiL.1 1.398050235753 (0.10699944188)  phiL.2 -0.572948456361 (0.14534970297)
const.M 1.561316851385 (0.48419192919)  phiM.1 1.214974024205 (0.08129938952)  phiM.2 -0.699594817932 (0.18588175008)
const.H 1.165691947904 (0.86158769351)  phiH.1 1.599254070091 (0.10709900051)  phiH.2 -1.011575490495 (0.26047094790)
SSR = 4.08380041426   AIC = -357.525427817   k = 11
regime counts 40 / 38 / 34
```

**The high regime is identical to the 1-threshold fit's high regime** (`th2`
equals the 1-threshold `th`), which is a consequence of the conditional search
and is worth asserting — it confirms the sequential structure.

**A6. `lstar(y, m=2, thDelay=1)`**, `control=list(maxit=5000)`:

```
start grid picked gamma = 11.1538461538, th = 3.33748551848, SSE = 4.33766354337
optimum           gamma = 11.153834351444, th = 3.339198526369, SSE = 4.33764323215 (converged)
```

| coefficient | estimate | std. error | z |
|---|---|---|---|
| `const.L` | 0.489101416212 | 0.20491524525 | 2.38685 |
| `phiL.1` | 1.246539926533 | 0.06787121417 | 18.36625 |
| `phiL.2` | −0.366432799971 | 0.10430101686 | −3.51322 |
| `const.H` | −1.024075825089 | 2.43009870400 | −0.42141 |
| `phiH.1` | 0.423266863310 | 0.17214631591 | 2.45876 |
| `phiH.2` | −0.254608799777 | 0.58542260103 | −0.43491 |
| `gamma` | 11.153834351444 | 10.00479258390 | 1.11485 |
| `th` | 3.339198526369 | 0.09274925250 | 36.00243 |

```
SSR = 4.33764323215   mse = 0.0380495020364   AIC = -356.65087013   BIC = -334.761282543   k = 8
diag(vcov) = 0.04199025774, 0.004606501713, 0.01087870212, 5.905379711,
             0.02963435408, 0.3427196218, 100.0958746, 0.008602423839
built-in nonlinearity F = 12.4459817154, p = 1.38150119187e-05
G(z) range = [3.40372245713e-09, 0.996447296207], mean = 0.273310796644
```

**A7. `selectLSTAR(y, m=2, thDelay=0:1, mL=1:2, mH=1:2)`** — the full 8-row table
is in `fixture_lynx_setar.json`; the best is `thDelay=1, mL=2, mH=1,
AIC=−358.519566542, th=3.37974275871, gamma=8.74030335404`. Note rows 5–8 all
have **γ pinned at ≈100**, the start-grid upper bound (§6.3).

### 2.2 Fixture B — Hansen's sunspot transform

`sun = (sqrt(sunspot.year + 1) − 1) · 2`, n = 289 (1700–1988),
`sum = 3151.20015794`, `mean = 10.9038067749`.

`setar(sun, m=11, thDelay=1)`: **th = 7.423375191512**, SSR = 907.442712049961,
AIC = 380.674883408, k = 25, regime proportions 0.309352517986 / 0.690647482014.
Full 24 coefficients in the JSON. [V]

Corroboration worth noting: `tsDyn`'s own development comment at
`selectSETAR.R:520` records `around=7.444575` for this case, consistent with the
measured 7.42337519151. [S]

### 2.3 Fixture C — simulated SETAR with a known DGP

`fixture_setar_sim.csv`, 1000 rows, sha256
`96f87cee4a20b1bf40acbdae02a881afe69adc00ef4ffb196fa73bb9d7ba75e8`, round-trips
bit-exactly at 17 significant digits. **The series is the fixture, not the
seed** — R's RNG stream is not portable (charter §6.6).

True DGP: `const.L = 0.3, phiL.1 = 0.5 | const.H = −0.2, phiH.1 = 0.9`,
threshold 0, delay 0. Summary: mean 0.435436080662, sd 1.28768121011, range
[−3.5286965202, 4.7267744043].

Recovery by `setar(s, m=1, thDelay=0)` [V]:

| | estimate (se) | true |
|---|---|---|
| th | 0.0127793045 | 0 |
| `const.L` | 0.23751762454458322 (0.079993633065496755) | 0.3 |
| `phiL.1` | 0.44953311723945383 (0.078034597295509919) | 0.5 |
| `const.H` | −0.27843809089464733 (0.065272771619280689) | −0.2 |
| `phiH.1` | 0.88205356007858393 (0.042680558818975908) | 0.9 |

SSR = 1001.7766997942632, AIC = 11.775123330175814, n_low = 385, n_high = 614.
With `th` fixed at the true 0: 0.230389423843, 0.444148537350, −0.263473337001,
0.874406039907, SSR = 1004.11661592.

All four parameters recover within **1.3 standard errors**, and `const.L` within
0.78 — a good recovery test, and the kind of statement that belongs in the book
chapter.

---

## 3. Verification — three independent implementations agree, two bit-exactly

This is the strongest verification position of any item in Stage 11, and it
should shape the test suite.

### 3.1 The hand-rolled base-R grid search is bit-exact

A from-scratch implementation — build `Y, y_{t−1}, y_{t−2}`; candidate set
`unique(allTh[seq(ceiling(0.15·ng), floor(0.85·ng), length.out = up−down)])`; for
each candidate `lm.fit([X·1{z≤γ}, X·1{z>γ}], Y)`; take min SSR — gives [V]
`s11/tar/f2_handroll.R`:

```
tsDyn th  : 3.310055737751     handroll th  : 3.310055737751     diff = 0
tsDyn SSR : 4.348191279201     handroll SSR : 4.348191279201     diff = 0
coefficient max abs diff : 0
n_low : 78 both
```

**Differences of exactly zero**, not "agreement to 1e-12". Repeated on fixture C
(699 candidates): th = 0.012779304500 and SSR = 1001.776699794263, identical.

**This is ~35 lines of R.** It is the proof that the SETAR core is small, and it
is the template for the Julia implementation: if the Julia version does not also
agree bit-exactly, the candidate grid or the dummy expansion is wrong, and
nothing else can be.

### 3.2 `aeon` agrees to 12 digits

`log10(lynx)`, `aeon SETAR(lag=2)` against `tsDyn::setar(y[-1], m=2, thDelay=0)`
on the matched 111-row sample [V] `s11/tar/f10_aeon_cmp.R`:

| quantity | aeon 1.6.0 | tsDyn 11.0.5.2 |
|---|---|---|
| threshold | 2.557507201906 | 2.557507201906 |
| intercept_low | 0.416800274876 | 0.416800274876 |
| coefs_low | 1.237891018066, −0.332005825545 | 1.237891018066, −0.332005825545 |
| intercept_high | 1.180869464937 | 1.180869464937 |
| coefs_high | 1.547698349232, −0.956274108919 | 1.547698349232, −0.956274108919 |
| SSE | 4.562802243613 | 4.562802243613 |
| n_low / n_high | 30 / 81 | 30 / 81 |

Note `aeon` uses `maxlag = lag_order + delay`, so it drops **one more
observation** than `tsDyn` for the same AR order — hence the matched-sample
construction. Its grid is over `np.unique(...)` where `tsDyn`'s includes ties
(§4.3), which happens not to matter here but does in general.

`aeon` is BSD-3 **by project policy; the LICENSE file was not read** [U]. Treat
it with the no-transcription discipline until someone checks.

### 3.3 What is single-referenced

Only `tsDyn` provides: the candidate-grid construction, `selectSETAR`'s criteria,
the standard errors, `nthresh=2`, LSTAR, and `setarTest`. For each of those,
**repeat the §3.1 trick**: write an independent implementation of the same
quantity in plain R (or Python) and check the two agree. It worked perfectly for
SETAR and is the only second opinion available. In particular, a 20-line script
that evaluates LSTAR's SSR on a `(γ, th)` mesh and reports the minimum is a real
check on the profile objective.

---

## 4. The conventions that must be reproduced exactly

### 4.1 `tsDyn` divides by the full series length (charter §5.8)

For `log10(lynx)`: n = 114, regression rows = 112, and

```
mse    = SSR / 114
AIC    = 114·log(mse) + 2k
BIC    = 114·log(mse) + log(114)·k
resvar = SSR / (114 − p)        p = number of slope coefficients = 6
```

[V] Only `SSR/(114−6)` reproduces `const.L`'s reported SE of **0.14330661379**:

| df | `const.L` SE |
|---|---|
| **114 − 6 = 108 (tsDyn)** | **0.14330661379** ✓ |
| 114 − 7 = 107 | 0.14397471351 |
| 112 − 6 = 106 | 0.14465224531 |
| 112 − 7 = 105 | 0.14533943325 |

**`_ols` uses `SSR/(n−k)` on the actual 112×6 design and will not match.** Two
options, and this is a design decision:

- **Reproduce `tsDyn`'s denominator** for comparability, with a docstring saying
  it uses the full series length and why that is `tsDyn`'s choice.
- **Use the correct denominator** (`SSR/(rows − k)`) and record the reference's
  figures as "computed by `tsDyn`'s convention" in the test.

**Recommendation: compute the statistically defensible version and expose the
reference's as an option.** `mse` and `AIC` scaled by a length that includes rows
the regression never saw is not defensible, and silently inheriting it to make a
test pass is the kind of thing the charter's §2.3 exists to prevent. But the test
must then assert *both* — ours to its own definition, and `tsDyn`'s formula
reproduced exactly from the same SSR — so the discrepancy is pinned rather than
papered over.

### 4.2 Indexing and boundaries

| convention | measured [V] |
|---|---|
| Low regime is **`z ≤ γ`**, inclusive | exactly one lynx observation has `z == th`; `sum(z ≤ th) = 78 = n_low`, `sum(z < th) = 77`, `regime[63] == 1` |
| **`thDelay = d` means `z_t = y_{t−(d+1)}`** — zero-indexed on the lag | `thDelay=0 → y_{t−1}` (`mTh = (1,0)`), `thDelay=1 → y_{t−2}` (`mTh = (0,1)`); constraint `thDelay < m` |
| `include="const"` default gives **one intercept per regime** (`common="none"`) | `"none"` → 5 params, `"trend"` → per-regime trend, `"both"` → 9, `common="include"` → one shared intercept (k=6) |
| `mL`/`mM`/`mH` are **per-regime AR orders and may differ** | `m=3, mL=1, mH=3` → `const.L, phiL.1, const.H, phiH.1..3, th`, k=7, SSR = 4.97708484151, th = 2.406540180434 |
| `ML`/`MH` allow **non-contiguous lag subsets** | `ML=c(1,3), MH=c(1,2)` → `const.L, phiL.1, phiL.3, const.H, phiH.1, phiH.2, th`, SSR = 4.11329623954 |
| **The threshold must be passed bit-exact** | passing the 12-digit print `th=3.31005573775` (below the true 3.3100557377508912) moves the boundary observation to the high regime: SSR 4.3481912792 → 4.4217191153, n_low 78 → 77 |
| `k` counts the threshold **only when it was searched** | searched k=7, AIC −358.374; supplied `th` k=6, AIC −360.374, **same SSR** (charter defect D13) |
| `model="MTAR"` differences the threshold variable | lynx: th = −0.315224649687, SSR = 4.86310363051 |

**The bit-exact threshold trap deserves an API response:** our `SETARModel` must
carry the threshold at full precision, and `show` should print enough digits to
round-trip (or print a note). A user who copies a printed threshold back into a
fixed-threshold fit and gets a different SSR will reasonably think something is
broken.

### 4.3 The candidate grid — fiddly, and a plausible simplification is wrong

[V] `s11/tar/f12_grid_exact.R`, calling `tsDyn:::makeTh` directly:

```
allTh  = sort(z)                              # FULL sorted vector, INCLUDING TIES
ng     = length(allTh)
down   = ceiling(trim · ng)
up     = floor(ng · (1 − trim))
ninGrid = (up − down) · {All: 1, Half: 1/2, Third: 1/3, Quarter: 1/4}
ths    = allTh[seq(down, up, length.out = ninGrid)]      # FRACTIONAL indices, silently truncated
th     = unique(ths)
```

Two subtleties:

1. **`sort(z)` includes ties, not `sort(unique(z))`.** Demonstrated to differ: 200
   draws of `round(rnorm, 1)` (45 unique values) give a **21-value grid from
   `sort(z)` against 31 from `sort(unique(z))`**, with different ranges. They
   coincide for lynx by luck. **Use `sort(z)` with ties.**
2. **`seq(..., length.out = up − down)` produces fractional indices** (for lynx:
   17, 18.0137, 19.0274, …) which R truncates on indexing, skipping exactly 1 of
   the 75 in-range values. Verified the skip does not change the argmin here
   (both grids give th = 3.310055737751, SSR = 4.348191279201) — but it is a real
   behavioural difference.

**Decision: reproduce (1), do not reproduce (2).** The ties behaviour is
substantive and changes which thresholds are considered; the fractional-index
truncation is a bug that happens not to bite. Document the departure and assert
that our grid is a superset of `tsDyn`'s for the lynx case, with the argmin
identical.

For `log10(lynx)`, `m=2`: **108 unique `z` values out of 112 → 74 candidate
thresholds.** [V]

### 4.4 `nthresh=2` is sequential, not a joint grid

Find γ₁ over the 1-D grid; search γ₂ conditional on γ₁; re-search γ₁ conditional
on γ₂; `max.iter = 2` alternations. Confirmed by the trace's two "Second best: …
(conditionnal on th= …)" lines [V].

And `criterion="pooled-AIC"` **is not implemented for `nthresh=2`** — it warns
and silently falls back to SSR (charter defect D14) [V].

**A joint 2-D grid is the obvious reading and is wrong.** It would also be
`O(ncand²)` where the sequential search is `O(ncand)`. Reproduce the sequential
structure, expose `max_iter`, and **refuse** a criterion that is not implemented
for two thresholds rather than silently substituting one.

### 4.5 Three different trim defaults in one package

`setar`/`selectSETAR` **0.15**, `setarTest` **0.1**, `lstar`'s starting grid
**0.1**. [S] signatures

**Ours uses one default (0.15) everywhere**, with `trim` exposed on each
function. Record `tsDyn`'s three values in the docstring so a comparison can be
set up correctly.

---

## 5. Inference

### 5.1 Regime coefficients

Conditional-OLS standard errors with `resvar = SSR/(n − p)` per §4.1, t-tests on
`t(n − p)`. `_ols` already returns `se`; only the denominator differs.

### 5.2 The threshold — the reference gives nothing

`vcov.setar` **zero-pads the threshold row and column**. Verified:
`diag(vcov(m1))[7] == 0` exactly (charter defect D13) [V]. The non-standard
threshold asymptotics are simply not implemented.

So matching `tsDyn` here is free, and that is also the problem: **if
TSAnalytics.jl wants honest threshold inference — Hansen's (1997, 2000)
likelihood-ratio profile confidence set, whose limit distribution is
non-Gaussian — there is no reference implementation in either R or Python to
check against.**

**This is a scope decision to make explicitly.** Two defensible answers:

- **Do what the reference does**: report no threshold standard error, and say in
  the docstring *why* — that the threshold's asymptotic distribution is
  non-standard, so a Gaussian standard error would be misleading. This is the
  honest-refusal pattern and it costs nothing.
- **Build the LR-profile confidence set**, verified against Hansen's published
  critical values rather than against a running program. This is the single
  hardest sub-item in 11.4 and it would be a genuine capability neither reference
  has.

**Recommendation: ship the first, and add the second as a separate, clearly
scoped item** — with the `threshold_ci` function returning the profile region.
The groundwork is nearly free: the SSR-versus-threshold profile is already
computed by the grid search, so storing it costs one vector and immediately gives
the §8 plot.

### 5.3 LSTAR

`tsDyn` reports standard errors for **all 8 parameters including γ and `th`**,
from `vcov = 2·SSR/n·H⁻¹` with `H = optimHess(θ, SSR)` over all parameters and
`n = 114`. Verified `max|vcov − formula| = 0` [V].

This is statistically dubious — γ's profile is nearly flat, which is why its
reported SE is **10.005** against an estimate of **11.154** — but it is what the
reference does, and reproducing it makes the comparison possible.

**ForwardDiff can compute that Hessian better than `optimHess` can**, which means
ours will **not** match to machine precision. Expect **~4–6 digits on the
standard errors and exact agreement on the coefficients.** Say so in the test's
comment rather than loosening the tolerance silently.

### 5.4 The nonlinearity tests

| test | statistic | null | defaults |
|---|---|---|---|
| `setarTest` (Hansen 1999) | `F_ij = n·(S_i − S_j)/S_j`, `n = N − m` | **bootstrap only**, no asymptotic p-value | **`nboot = 10`**, `trim = 0.1`, `test = "1vs"`, `boot.scheme = "resample"` |
| `summary.lstar`'s built-in | `anova` F: `yy ~ xx` against `yy ~ xx + xx*z` | F(m, ·) asymptotic | always printed |
| `isLinear.lstar` | Luukkonen-type | — | **broken in 11.0.5.2** (charter defect D12) |

Measured, `setarTest(log10(lynx), m=2, thDelay=0:1, nboot=200, trim=0.15,
seed=1234)`, 8.59 s [V] `s11/tar/f5_test.R`:

```
SSRs:   AR = 5.78258084172   SETAR(2) = 4.34819127920   SETAR(3) = 4.08380041426
Ftests: 1vs2 = 36.94677181537   1vs3 = 46.58979102185   2vs3 = 7.25103429878
PvalBoot (1vs): 1vs2 = 0,  1vs3 = 0;   (2vs3) = 0.555
Critical values (nboot=200, seed=1234):
            0.90          0.95         0.975         0.99
1vs2  15.0806675937  16.7602990397  18.4110807325  18.8319866334
1vs3  25.8448685612  27.7967192572  29.7707148283  31.9323982526
2vs3  13.9850629374  15.7814251214  16.8608508352  19.2817103392
```

The F statistic uses **`n = N − m = 112`**, verified exactly:
`36.94677181537 / ((5.78258 − 4.34819)/4.34819) = 112.0`.

**The SSRs and F statistics are portable expectations. The bootstrap critical
values and p-values are NOT** — they are seed-dependent, and `nboot = 10` is far
too small to be stable (charter §9.10). Our default should be **1000**, with the
reference's 10 noted as a trap.

`ThresholdTest <: HypothesisTest` is the "critical values only" shape Stage 10
§6.4 already had to accommodate [U] — reuse that design.

Note `plot.Hansen99Test` overlays a χ²(1+m) curve **only for y-axis scaling** — it
is not used for p-values [S]. Do not mistake it for an asymptotic null.

---

## 6. Performance — the clearest win in Stage 11

### 6.1 The measured reference cost

[V] `s11/tar/f13_cost.R`, on only 114 observations:

| spec | OLS fits | wall clock (R) |
|---|---|---|
| m=2, thDelay 0:1, pooled-AIC | 592 | **1.505 s** |
| m=3, thDelay 0:2, pooled-AIC | 1,971 | **6.871 s** |
| m=5, thDelay 0:4, pooled-AIC | 8,875 | **34.888 s** |
| m=11, thDelay 0:1, **SSR** (sunspot, n=289) | 142 | 0.175 s |
| m=2, thDelay 0:1, nthresh=2 | 148 + conditional | 0.029 s |

Thirty-five seconds for a 114-point series is the kind of number that makes the
case for the package.

### 6.2 Incremental cross-products — the design

Every one of those fits is a full factorisation of a design that differs from the
previous candidate's by a handful of rows. So:

1. **Sort once** by the threshold variable `z`.
2. Walk the split point upward through the candidate grid.
3. Maintain `X_L'X_L`, `X_L'y`, `X_H'X_H`, `X_H'y` by **adding the crossing rows
   to the low block and subtracting them from the high block** — a rank-1 update
   per crossed observation.
4. Solve the two `k×k` systems by Cholesky.

Cost per candidate falls from `O(n k²)` plus a factorisation to **`O(k²)` plus
two `k×k` Cholesky solves**. For the m=5 case above that is a 10–100× reduction
in work.

**Numerical caution, and it is real:** running cross-product updates accumulate
error over 74–700 candidates. Two mitigations, both cheap:

- Keep the cumulative sums in `Float64` but **recompute `X'X` from scratch at the
  argmin** before reporting coefficients. The search can tolerate 1e-10 drift;
  the reported fit cannot.
- Assert, in a test, that the search's chosen threshold and the recomputed SSR
  agree with a naive full-refit search **bit-exactly on the lynx fixture** (§3.1
  makes this checkable). If they differ, the update is wrong or the tolerance is
  too loose.

### 6.3 LSTAR's start grid is not optional

`tsDyn` searches **nTh = 200 × nGamma = 40 = 8,000 starting points** before
running BFGS over just `(γ, th)` with the regime coefficients profiled out
analytically [S] `lstar.R`, 210 code lines.

That is not excess: **γ pins at the start-grid bound of 100 on two of the three
fixtures** [V] — `star()` on lynx (γ = 100.000108716038) and `lstar()` on fixture
C (γ = 100.0000020810) — because the objective in γ is nearly flat and BFGS
barely moves it. The grid is what finds the basin.

Build the grid, make it configurable, and **`Threads.@threads` over it** (charter
§7.3) — 8,000 profiled OLS fits is embarrassingly parallel and each is tiny.

**Report when γ hits the bound.** A fitted γ at the grid maximum means "the
transition is effectively abrupt, consider SETAR", and that is information a user
needs. `tsDyn` warns; ours should too, and should say what it implies.

### 6.4 Other requirements

Per charter §7.2: `@views`, `mul!`, no allocation in the candidate loop,
`@code_warntype` clean, `@allocated` zero in the inner loop asserted.

---

## 7. Verification plan

### 7.1 Tier 1 — bit-exact, which is unusual and should be exploited

| test | target | tolerance |
|---|---|---|
| SETAR on lynx, `m=2, thDelay=1` | th = 3.3100557377508912, SSR = 4.3481912792005826 | **exact** on th; 1e-12 on SSR |
| the six coefficients | §2.1 A1 table | 1e-10 |
| incremental search vs naive full-refit search | identical argmin | **exact** |
| SETAR on fixture C, `m=1, thDelay=0` | th = 0.012779304500, SSR = 1001.776699794263 | **exact** / 1e-10 |
| SETAR on sunspot, `m=11, thDelay=1` | th = 7.423375191512, SSR = 907.442712049961 | exact / 1e-10 |
| `aeon`-matched sample | th = 2.557507201906, SSE = 4.562802243613 | exact / 1e-12 |

The threshold is selected from a finite candidate set, so **exact** is the right
assertion — any difference means a different candidate was chosen, which is a bug
and not a rounding difference.

### 7.2 Standard errors

Both conventions (§4.1): ours by `SSR/(rows − k)`, and `tsDyn`'s by
`SSR/(114 − 6)` reproduced from the same SSR, matching **0.14330661379** to
1e-10. Plus the four-way denominator table as an explicit test, so the choice
cannot silently drift.

### 7.3 `selectSETAR`

The A4 table: `bests` row exact, the top-5 `pooled-AIC` values to 1e-9, and
`allTh` having exactly **592 rows**. The row count is the test that the grid
dimensions are right.

### 7.4 `nthresh=2`

A5's `th1`/`th2` exact, the nine coefficients to 1e-9, regime counts 40/38/34
exact, **and** the structural assertion that `th2` equals the 1-threshold fit's
threshold.

### 7.5 LSTAR

A6: coefficients to 1e-8, `SSR = 4.33764323215` to 1e-10, `G` range
`[3.40372245713e-09, 0.996447296207]` to 1e-9, and the **additive-form
assertion** — compute SSR both ways and assert the additive one is 4.3376 and the
convex one is ≈290.77, so the trap is pinned permanently.

Standard errors to **1e-4 relative only**, with §5.3's reason in the comment.

### 7.6 `setarTest`

SSRs and F statistics to 1e-9 (§5.4), including the `n = N − m = 112` scaling
identity. **No assertion on bootstrap critical values or p-values.** Instead:
assert that with a fixed seed our own bootstrap is reproducible, and that the
critical values increase with the quantile level.

### 7.7 Reductions

1. **LSTAR as γ → ∞ approaches SETAR** at the same threshold. Fit LSTAR with γ
   fixed at 1e6 and `th` fixed at the SETAR threshold; the SSR must approach
   SETAR's 4.3481912792 to ~1e-6. The measured `G` range already reaches
   `[3.4e-09, 0.9964]`, so the limit is genuinely approachable.
2. **A single regime reduces to `arx`.** `nthresh=0`, or a threshold outside the
   data range, must give exactly the pooled AR(2) of §2.1 A3: `const =
   1.057600456442`, SSR = 5.78258084172.
3. **A supplied threshold equal to the searched one gives the same SSR** but
   `k = 6` rather than 7 — the charter D13 distinction, asserted both ways.
4. **Non-contiguous lag subsets** reproduce `ML=c(1,3), MH=c(1,2)` → SSR =
   4.11329623954.

### 7.8 Testset inventory

| # | Testset | Assertions (min) |
|---|---|---|
| 1 | SETAR lynx: threshold exact, SSR, 6 coefficients | 16 |
| 2 | SETAR lynx: standard errors, both denominators | 14 |
| 3 | the four-way denominator table | 8 |
| 4 | SETAR lynx `thDelay=0` | 10 |
| 5 | SETAR sunspot `m=11` | 14 |
| 6 | SETAR fixture C, DGP recovery within 1.3 se | 14 |
| 7 | `aeon`-matched sample, bit-exact | 10 |
| 8 | incremental search = naive search, exact | 8 |
| 9 | candidate grid: 74 candidates from 108 unique values; ties included | 12 |
| 10 | candidate grid: ties vs unique differ on the tied series (21 vs 31) | 8 |
| 11 | boundary: `z ≤ γ` inclusive, `n_low = 78`, `regime[63] == 1` | 10 |
| 12 | `thDelay` indexing: `d=0 → y_{t−1}` | 8 |
| 13 | bit-exact threshold trap: truncated th gives SSR 4.4217191153 | 8 |
| 14 | `include=` variants: param counts 5 / 7 / 9 / 6 | 12 |
| 15 | per-regime orders `mL ≠ mH`: SSR 4.97708484151 | 10 |
| 16 | non-contiguous `ML`/`MH`: SSR 4.11329623954 | 10 |
| 17 | `selectSETAR`: bests row, top-5, 592 rows | 14 |
| 18 | `nthresh=2`: thresholds, 9 coefficients, counts, `th2` identity | 18 |
| 19 | `nthresh=2`: unimplemented criterion refuses | 6 |
| 20 | LSTAR lynx: coefficients, SSR, G range | 16 |
| 21 | LSTAR: additive vs convex form (4.3376 vs 290.77) | 8 |
| 22 | LSTAR standard errors, 1e-4 relative | 10 |
| 23 | LSTAR: γ at the bound is reported | 6 |
| 24 | `selectLSTAR`: the 8-row table | 16 |
| 25 | `setarTest`: SSRs, F statistics, `n = N − m` identity | 12 |
| 26 | `setarTest`: own-seed reproducibility, monotone critical values | 8 |
| 27 | reduction: LSTAR γ→∞ approaches SETAR | 8 |
| 28 | reduction: single regime = pooled `arx` | 10 |
| 29 | supplied vs searched threshold: same SSR, k=6 vs 7 | 8 |
| 30 | refusals: `thDelay ≥ m`, `trim ≥ 0.5`, too few per regime | 12 |
| 31 | unit-root warning in the high regime (root 0.994262) | 6 |
| 32 | performance: allocations, incremental-update accuracy, scaling | 10 |

**32 testsets, ≥ 340 assertions.** The highest count in the stage, which is
appropriate: it is the item with the best reference agreement, so the tests can
be the most demanding.

---

## 8. Charts

Detail in the plot-recipes handoff. `tsDyn`'s methods, identified by reading
their source — **none were rendered** [S], so appearance is unverified:

| method | lines | draws |
|---|---|---|
| `plot.setar` | 75 | phase plot (`y_t` against `y_{t−j}`, points coloured by regime, fitted curve dashed, "low"/"high" legend); regime-switching plot (series over time coloured by regime, `abline(h=th)`); transition variable over time with the threshold line; **sorted** transition variable with threshold and trim bounds |
| `plot.lstar` | 60 | phase plots coloured by **transition-function quantile** rather than hard regime; regime-switching plot |
| `plot.nlar` | 30 | 8 panels: series, residuals, ACF/PACF of each, average mutual information of each |
| `plot.Hansen99Test` | 73 | **bootstrap density of each F statistic**, observed statistic as a dashed red vertical, χ² reference curve for scale |
| `plot.selectSETAR` | 15 | **criterion against candidate threshold** — the SSR profile. `stop`s unless `criterion="SSR"` |
| `autopairs` / `autotriples` | 18 / 17 | bivariate and trivariate phase plots, including persp/image/contour |

**The phase plot is not implemented for `nthresh=2`** — it prints a title saying
so [S].

**Two plots neither reference has, and both are nearly free:**

1. **The transition function directly** — `G` against `z`, or `G` against `t`.
   `plot.lstar` only colours points by `G` quantile; plotting `G` itself is the
   obvious way to see how abrupt the transition is, and it is one line given the
   fit.
2. **The threshold confidence set** — the LR profile with the critical-value
   cutoff, shading the region. The SSR profile is already stored by the grid
   search (§6.2), so this costs nothing beyond §5.2's decision.

Also worth having: `plot.selectSETAR`'s SSR profile **without** the
`criterion="SSR"` restriction, since the profile is meaningful for any criterion.

---

## 9. Budget

| Piece | Reference (code lines) | Julia estimate |
|---|---|---|
| `setar` core | `setar_low` **280** | 200–260 |
| design builders + objectives | 100 across 8 + 74 across 9 | 100–140 |
| candidate grid (`makeTh`) | **81** | 70–100 |
| `selectSETAR` + criteria | 160 + 176 | 220–280 |
| `nthresh=2` conditional search | (inside the above) | 90–120 |
| incremental cross-product search | — (reference has none) | 120–160 |
| `lstar` | **210** | 180–240 |
| `setarTest` + simulator + bootstrap | 69 + 26 + 90 + 33 | 180–240 |
| result types, `show`, `predict` | ~200 across methods | 150–200 |
| plots (6 recipes) | 240 across methods | 180–240 |
| **total** | ≈1050 core + 460 tests/plots | **1490–1980** |

Larger than the line-count intuition suggests, and the reason is §4 — the
conventions, not the algorithm.

---

## 10. Suggested build order

1. **SETAR at a supplied threshold** — one `_ols` call on the dummy-expanded
   design. Test against A1 with `th` fixed bit-exactly.
2. **The candidate grid**, tested against the 74-candidate count and the
   ties-versus-unique demonstration, before any search uses it.
3. **The naive search**, tested bit-exactly against A1, C and sunspot.
4. **The incremental search**, tested bit-exactly against the naive one.
5. **Standard errors**, both denominators.
6. `selectSETAR`, then `nthresh=2`.
7. **LSTAR**, start grid first (it is what makes the optimiser work), then the
   profile BFGS, then the standard errors.
8. `setarTest`.
9. Plots.

Steps 1–4 are where the bit-exact agreement lives; nothing after step 4 can be
trusted if any of them is loose.

---

## 11. What not to do

1. **Do not transcribe `tsDyn`'s grid or design-matrix code.** GPL (charter §4).
   §4.2 and §4.3 record the behaviour precisely enough to re-derive.
2. **Do not use `sort(unique(z))` for the candidate grid.** §4.3 — it changes the
   candidate set, demonstrated.
3. **Do not use the convex-combination form for LSTAR.** §1.2 — wrong by 67×.
4. **Do not implement a joint 2-D grid for `nthresh=2`.** §4.4.
5. **Do not skip LSTAR's start grid.** §6.3 — γ pins at the bound and BFGS alone
   will not find the basin.
6. **Do not assert bootstrap critical values.** §5.4.
7. **Do not implement `star`/`addRegime`/`testRegime`, `KapShinTest` or
   `BBCTest`.** Charter §9.7 — the latter two carry the author's own disclaimer
   that they were never checked against the papers.
8. **Do not report a Gaussian standard error for the threshold.** §5.2 — its
   asymptotics are non-standard, and `tsDyn`'s zero-padding is at least honest
   about having nothing to say.
9. **Do not inherit `tsDyn`'s full-length denominator silently.** §4.1 — compute
   the defensible version, expose theirs, and test both.
10. **Do not print a threshold that cannot round-trip.** §4.2.
